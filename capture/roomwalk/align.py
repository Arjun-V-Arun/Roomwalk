"""Turning an arbitrary reconstruction into a room you can stand in.

Structure from motion has no idea which way is up or how big anything is. Its
coordinate frame is whatever the first image pair happened to be, and its unit
is arbitrary. A viewer dropped into that frame starts sideways, at the wrong
height, walking at the wrong speed.

We fix all three from the camera poses alone:

* Up: people don't roll their phones sideways while filming, so every
  camera's right edge is roughly horizontal. Gravity is the direction
  perpendicular to all of them.
* Scale: people hold phones at roughly eye level. Where the floor was filmed,
  camera height above it in model units, divided by a typical filming height
  (1.45 m), gives metres per unit. If the floor was never in view, the
  ceiling or a typical room size is used instead and flagged as less certain.
* Heading and origin: face the direction the video started in, and put the
  middle of the walk at x = z = 0 with the floor at y = 0.

The result is a similarity transform (rotation, uniform scale, translation)
that the web viewer applies to the splat, plus a start pose and a box the
camera is kept inside.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

EYE_HEIGHT = 1.45


@dataclass
class Poses:
    names: list[str]
    centers: np.ndarray  # (N, 3) camera centres in model coordinates
    ups: np.ndarray  # (N, 3) each camera's up direction in model coordinates
    forwards: np.ndarray  # (N, 3) each camera's viewing direction
    points: np.ndarray | None = None  # (M, 3) sparse points, optional
    rights: np.ndarray | None = None  # (N, 3) each camera's right direction


@dataclass
class Alignment:
    rotation: np.ndarray  # 3x3
    scale: float
    translation: np.ndarray  # (3,)
    up_spread_deg: float
    roll_deg: float  # mean sideways tilt of the phone; large values mean "up" is unreliable
    scale_source: str
    scale_confidence: str  # high (floor seen) | medium (ceiling) | low (room-size guess) | none
    start_position: np.ndarray
    start_target: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray

    def apply(self, p: np.ndarray) -> np.ndarray:
        return self.scale * (np.asarray(p) @ self.rotation.T) + self.translation

    def manifest_transform(self) -> dict:
        return {
            "position": [round(float(x), 5) for x in self.translation],
            "quaternion": [round(float(x), 6) for x in quat_xyzw(self.rotation)],
            "scale": round(float(self.scale), 6),
        }

    def manifest_view(self) -> dict:
        r = lambda v: [round(float(x), 3) for x in v]  # noqa: E731
        return {
            "start": {"position": r(self.start_position), "target": r(self.start_target)},
            "bounds": {"min": r(self.bounds_min), "max": r(self.bounds_max)},
            "eyeHeight": EYE_HEIGHT,
        }


def poses_from_world_to_camera(names, rotations, translations, points=None) -> Poses:
    """Build Poses from COLMAP-style world-to-camera rotations R and translations t.

    COLMAP cameras look down +z with y pointing down in the image (OpenCV
    convention), so in world coordinates: centre = -R^T t, forward = R^T z,
    up = -R^T y.
    """
    R = np.asarray(rotations, float)
    t = np.asarray(translations, float)
    Rt = np.transpose(R, (0, 2, 1))
    centers = -np.einsum("nij,nj->ni", Rt, t)
    forwards = Rt[:, :, 2]
    ups = -Rt[:, :, 1]
    rights = Rt[:, :, 0]
    return Poses(list(names), centers, ups, forwards, None if points is None else np.asarray(points, float), rights)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1 + 1e-9:
        # Opposite vectors: rotate 180 degrees about any axis perpendicular to a.
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def yaw(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def quat_xyzw(R: np.ndarray) -> np.ndarray:
    from .demo_room import matrix_to_quat

    w, x, y, z = matrix_to_quat(R)
    return np.array([x, y, z, w])


# Typical Indian PG room: about 2.9 m floor to ceiling, about 3.2 m across.
CEILING_HEIGHT = 2.9
TYPICAL_HALF_WIDTH = 1.6


def find_surface(pts: np.ndarray, cam_h: float, lowest: bool, n_bins: int = 60) -> tuple[float, int, float] | None:
    """Height, support and width of the floor (lowest=True) or ceiling.

    Walls put points at every height, so their height histogram is smooth. A
    floor or ceiling is a horizontal surface: many points at one height, a
    distinct peak. Scanning the histogram from the bottom up, the floor is the
    first bin that is well populated and at least 3x denser than the bins just
    above it. A bed top or desk is also a peak, so the surface must also span
    at least 60% of the room in both directions; a bed covers about a third.

    Note what is not required: being under the camera path. From the middle of
    a room the floor you can see is near the walls, not under your feet.
    """
    side = pts[pts[:, 1] < cam_h] if lowest else pts[pts[:, 1] > cam_h]
    if len(side) < 30:
        return None
    h = side[:, 1]
    a, b = (float(np.percentile(h, 0.2)), cam_h) if lowest else (cam_h, float(np.percentile(h, 99.8)))
    if b - a <= 1e-9:
        return None
    edges = np.linspace(a, b, n_bins + 1)
    counts, _ = np.histogram(h, edges)
    room_ext = np.percentile(pts[:, [0, 2]], 95, axis=0) - np.percentile(pts[:, [0, 2]], 5, axis=0)
    min_count = max(15, int(0.001 * len(pts)))
    for i in (range(n_bins) if lowest else range(n_bins - 1, -1, -1)):
        if counts[i] < min_count:
            continue
        beyond = counts[i + 2:i + 7] if lowest else counts[max(0, i - 6):max(0, i - 1)]
        if len(beyond) == 0 or counts[i] < 3 * max(float(np.median(beyond)), 1.0):
            continue
        layer = side[(h >= edges[max(i - 1, 0)]) & (h <= edges[min(i + 2, n_bins)])]
        ext = np.percentile(layer[:, [0, 2]], 95, axis=0) - np.percentile(layer[:, [0, 2]], 5, axis=0)
        width = float(np.min(ext / np.maximum(room_ext, 1e-9)))
        if width >= 0.6:
            return float(np.median(layer[:, 1])), int(len(layer)), width
    return None


def estimate_scale(c: np.ndarray, p: np.ndarray | None, eye_height: float) -> tuple[float, float, str, str]:
    """Metres per model unit, floor height (model units), how it was found, confidence.

    Fallbacks, in order: floor (camera height above it), ceiling (assumed
    room height), typical room width. The last two are flagged so a wrong
    guess is visible rather than silent.
    """
    cam_h = float(np.median(c[:, 1]))
    if p is None or len(p) < 50:
        return 1.0, cam_h - eye_height, "none (no sparse points)", "none"

    floor = find_surface(p, cam_h, lowest=True)
    if floor is not None:
        return (eye_height / (cam_h - floor[0]), floor[0],
                f"camera height above floor ({floor[1]} points spanning {floor[2]:.0%} of the room)", "high")

    ceiling = find_surface(p, cam_h, lowest=False)
    if ceiling is not None:
        scale = (CEILING_HEIGHT - eye_height) / (ceiling[0] - cam_h)
        return (scale, cam_h - eye_height / scale,
                f"camera distance to ceiling ({ceiling[1]} points spanning {ceiling[2]:.0%} of the room)", "medium")

    # Last resort: walls. Points beyond the walk are mostly walls; assume a typical room size.
    lo, hi = c[:, [0, 2]].min(0), c[:, [0, 2]].max(0)
    pad = 0.15 * (hi - lo)
    outside = p[~np.all((p[:, [0, 2]] >= lo - pad) & (p[:, [0, 2]] <= hi + pad), axis=1)]
    if len(outside) >= 50:
        centre = np.median(c[:, [0, 2]], axis=0)
        r = float(np.median(np.linalg.norm(outside[:, [0, 2]] - centre, axis=1)))
        if r > 1e-9:
            scale = TYPICAL_HALF_WIDTH / r
            return scale, cam_h - eye_height / scale, "assumed typical room width (floor not seen)", "low"
    return 1.0, cam_h - eye_height, "none (floor not seen)", "none"


def estimate_up(poses: Poses) -> np.ndarray:
    """Gravity direction in model coordinates.

    Tilting a phone up or down (pitch) moves its up vector but not its right
    edge; only rolling it sideways tilts the right edge, and people rarely roll
    while filming. So gravity is the direction most perpendicular to every
    camera's right vector: the smallest-eigenvalue eigenvector of the sum of
    their outer products. Averaging up vectors instead is biased whenever the
    phone spends more time pointing down at some walls than others.

    If the phone barely turned, all right vectors are parallel and that
    direction is undefined, so we fall back to the average up vector.
    """
    ups = poses.ups / np.linalg.norm(poses.ups, axis=1, keepdims=True)
    mean_up = ups.mean(axis=0)
    mean_up /= np.linalg.norm(mean_up)
    if poses.rights is None or len(poses.rights) < 3:
        return mean_up
    r = poses.rights / np.linalg.norm(poses.rights, axis=1, keepdims=True)
    vals, vecs = np.linalg.eigh(r.T @ r)
    if vals[1] < 0.05 * vals[2]:
        return mean_up  # right vectors all point the same way: the walk did not turn
    up = vecs[:, 0]
    return up if up @ mean_up > 0 else -up


def compute_alignment(poses: Poses, eye_height: float = EYE_HEIGHT, margin: float = 0.5) -> Alignment:
    if len(poses.centers) < 3:
        raise ValueError("need at least 3 registered cameras to align the room")
    up = estimate_up(poses)
    ups = poses.ups / np.linalg.norm(poses.ups, axis=1, keepdims=True)
    spread = float(np.degrees(np.mean(np.arccos(np.clip(ups @ up, -1, 1)))))
    roll = 0.0
    if poses.rights is not None:
        rights = poses.rights / np.linalg.norm(poses.rights, axis=1, keepdims=True)
        roll = float(np.degrees(np.mean(np.abs(np.arcsin(np.clip(rights @ up, -1, 1))))))

    R_up = rotation_between(up, np.array([0.0, 1.0, 0.0]))
    # Face the direction the walk started in. Three.js cameras look down -z.
    f0 = R_up @ poses.forwards[0]
    heading = np.arctan2(f0[0], f0[2])
    R = yaw(np.pi - heading) @ R_up

    c = poses.centers @ R.T
    pts = None if poses.points is None else poses.points @ R.T
    scale, floor_y, source, confidence = estimate_scale(c, pts, eye_height)

    center = np.array([np.median(c[:, 0]), floor_y, np.median(c[:, 2])])
    translation = -scale * center
    cw = scale * c + translation
    start = np.array([float(np.median(cw[:, 0])), eye_height, float(np.median(cw[:, 2]))])
    lo = cw.min(axis=0) - margin
    hi = cw.max(axis=0) + margin
    lo[1], hi[1] = 0.3, max(2.4, float(hi[1]))
    return Alignment(
        rotation=R, scale=float(scale), translation=translation, up_spread_deg=spread, roll_deg=roll,
        scale_source=source,
        scale_confidence=confidence,
        start_position=start, start_target=start + np.array([0.0, -0.1, -1.0]),
        bounds_min=lo, bounds_max=hi,
    )


def load_poses_pycolmap(model_dir: str | Path, max_points: int = 60000) -> Poses:
    import pycolmap

    rec = pycolmap.Reconstruction(str(model_dir))
    names, Rs, ts = [], [], []
    for image_id in sorted(rec.reg_image_ids(), key=lambda i: rec.image(i).name):
        img = rec.image(image_id)
        pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        names.append(img.name)
        Rs.append(np.asarray(pose.rotation.matrix()))
        ts.append(np.asarray(pose.translation))
    pts = np.array([p.xyz for p in rec.points3D.values()]) if rec.num_points3D() else None
    if pts is not None and len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
    return poses_from_world_to_camera(names, Rs, ts, pts)


def load_poses_text(model_dir: str | Path, max_points: int = 60000) -> Poses:
    """Read COLMAP's text export (images.txt, points3D.txt) without pycolmap."""
    model_dir = Path(model_dir)
    names, Rs, ts = [], [], []
    lines = [ln for ln in (model_dir / "images.txt").read_text().splitlines() if ln and not ln.startswith("#")]
    for ln in lines[::2]:  # every image takes two lines: pose, then 2D points
        parts = ln.split()
        qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
        names.append(parts[9])
        Rs.append(quat_to_matrix(qw, qx, qy, qz))
        ts.append([tx, ty, tz])
    order = np.argsort(names)
    pts = None
    pfile = model_dir / "points3D.txt"
    if pfile.exists():
        rows = [ln.split()[1:4] for ln in pfile.read_text().splitlines() if ln and not ln.startswith("#")]
        if rows:
            pts = np.array(rows, float)
            if len(pts) > max_points:
                pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
    return poses_from_world_to_camera([names[i] for i in order], np.array(Rs)[order], np.array(ts)[order], pts)


def quat_to_matrix(w, x, y, z) -> np.ndarray:
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
