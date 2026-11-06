"""A textured box room rendered with exact plane homographies.

Used to test structure from motion and alignment against known camera poses.
A convex room seen from inside needs no depth buffer: every pixel sees exactly
one wall, floor or ceiling, so each surface is drawn with one perspective warp
of its texture and masked by its (near-plane clipped) outline.

World frame matches the viewer: metres, Y up, floor at y = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

W, D, H = 3.0, 3.6, 2.9
X0, X1, Z0, Z1 = -W / 2, W / 2, -D / 2, D / 2
PX_PER_M = 300


def _speckle(img, rng, amount=10):
    noise = rng.normal(0, amount, img.shape[:2])[..., None]
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)


def _blocks(img, rng, n, x0, y0, x1, y1, smin=10, smax=60):
    for _ in range(n):
        x, y = int(rng.integers(x0, x1)), int(rng.integers(y0, y1))
        w, h = int(rng.integers(smin, smax)), int(rng.integers(smin, smax))
        c = tuple(int(v) for v in rng.integers(30, 230, 3))
        cv2.rectangle(img, (x, y), (x + w, y + h), c, -1)


def textures(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = PX_PER_M
    tex = {}
    # Floor: 60 cm vitrified tiles with grout and per-tile tone.
    fw, fd = int(W * p), int(D * p)
    floor = np.zeros((fd, fw, 3), np.uint8)
    t = int(0.6 * p)
    for i in range(0, fd, t):
        for j in range(0, fw, t):
            tone = rng.integers(-14, 14)
            floor[i:i + t, j:j + t] = np.clip(np.array([175, 200, 215]) + tone, 0, 255)
    floor[:: t, :] = (120, 135, 145)
    floor[:, :: t] = (120, 135, 145)
    for k in range(0, fd, t):
        floor[k:k + 2, :] = (120, 135, 145)
    for k in range(0, fw, t):
        floor[:, k:k + 2] = (120, 135, 145)
    cv2.ellipse(floor, (int(0.4 * fw), int(0.7 * fd)), (130, 90), 0, 0, 360, (60, 70, 160), -1)  # rug
    cv2.ellipse(floor, (int(0.4 * fw), int(0.7 * fd)), (100, 65), 0, 0, 360, (150, 200, 230), 6)
    # What a lived-in floor has and a bare tile pattern lacks: marble veins,
    # stains, slippers, a bucket, a bag, a newspaper. Without these the floor
    # gives SIFT almost nothing it can match between views.
    for _ in range(60):
        pts = np.cumsum(rng.normal(0, 9, (12, 2)), axis=0) + rng.uniform([0, 0], [fw, fd])
        cv2.polylines(floor, [pts.astype(np.int32)], False, (150, 168, 180), 2)
    for _ in range(25):
        c = (int(rng.uniform(0, fw)), int(rng.uniform(0, fd)))
        cv2.ellipse(floor, c, (int(rng.uniform(8, 30)), int(rng.uniform(5, 18))), float(rng.uniform(0, 180)), 0, 360,
                    (150, 175, 190), -1)
    for _ in range(6):
        c = np.array([rng.uniform(80, fw - 80), rng.uniform(80, fd - 80)])
        col = tuple(int(v) for v in rng.integers(20, 200, 3))
        for dx in (-22, 22):
            cv2.ellipse(floor, (int(c[0] + dx), int(c[1])), (16, 40), float(rng.uniform(-20, 20)), 0, 360, col, -1)
    cv2.circle(floor, (int(0.85 * fw), int(0.25 * fd)), 55, (180, 90, 30), -1)  # bucket
    cv2.circle(floor, (int(0.85 * fw), int(0.25 * fd)), 42, (120, 60, 20), 4)
    cv2.rectangle(floor, (int(0.15 * fw), int(0.2 * fd)), (int(0.15 * fw) + 110, int(0.2 * fd) + 70), (40, 40, 60), -1)  # bag
    paper = (int(0.6 * fw), int(0.45 * fd))
    cv2.rectangle(floor, paper, (paper[0] + 120, paper[1] + 160), (225, 228, 230), -1)  # newspaper
    for k in range(10):
        cv2.putText(floor, "NEWS" if k == 0 else "lorem ipsum", (paper[0] + 6, paper[1] + 18 + 14 * k),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 if k == 0 else 0.3, (40, 40, 40), 1)
    tex["floor"] = _speckle(floor, rng, 6)

    ceil = np.full((fd, fw, 3), 236, np.uint8)
    cv2.circle(ceil, (fw // 2, fd // 2), 40, (120, 140, 160), -1)
    cv2.rectangle(ceil, (fw // 2 - 300, fd // 2 - 8), (fw // 2 + 300, fd // 2 + 8), (60, 90, 120), -1)
    tex["ceiling"] = _speckle(ceil, rng, 3)

    def wall(width_m):
        img = np.full((int(H * p), int(width_m * p), 3), (190, 215, 200), np.uint8)
        img[-int(0.1 * p):] = (170, 190, 205)  # skirting tiles
        return img

    back = wall(W)  # window wall
    ww = back.shape[1]
    wy0, wy1 = int((H - 2.1) * p), int((H - 0.9) * p)
    sky = np.linspace([250, 225, 200], [240, 190, 140], wy1 - wy0).astype(np.uint8)[:, None, :]
    back[wy0:wy1, int(0.9 * p):int(2.1 * p)] = sky
    _blocks(back, rng, 25, int(1.0 * p), int(wy0 + 0.4 * p), int(2.0 * p), wy1, 8, 40)  # buildings outside
    for x in range(int(0.95 * p), int(2.1 * p), int(0.12 * p)):
        back[wy0:wy1, x:x + 4] = (50, 50, 55)
    cv2.rectangle(back, (int(0.5 * p), wy0 - 40), (int(0.9 * p), wy1 + 60), (40, 150, 200), -1)  # curtains
    cv2.rectangle(back, (int(2.1 * p), wy0 - 40), (int(2.5 * p), wy1 + 60), (40, 150, 200), -1)
    for x in range(int(0.5 * p), int(2.5 * p), 14):
        if not int(0.9 * p) < x < int(2.1 * p):
            back[wy0 - 40:wy1 + 60, x:x + 3] = (30, 120, 170)
    tex["back"] = _speckle(back, rng)

    front = wall(W)  # door wall
    cv2.rectangle(front, (int(0.2 * p), int((H - 2.1) * p)), (int(1.1 * p), front.shape[0]), (40, 75, 120), -1)
    for gy in (0.4, 1.1, 1.8):
        cv2.rectangle(front, (int(0.3 * p), int((H - 2.1 + gy) * p)), (int(1.0 * p), int((H - 2.1 + gy + 0.55) * p)), (30, 60, 100), 3)
    cv2.rectangle(front, (int(1.3 * p), int((H - 1.3) * p)), (int(1.55 * p), int((H - 1.15) * p)), (240, 240, 240), -1)
    for k in range(4):
        cv2.rectangle(front, (int((1.32 + 0.055 * k) * p), int((H - 1.27) * p)), (int((1.36 + 0.055 * k) * p), int((H - 1.2) * p)), (20, 20, 20), -1)
    cv2.circle(front, (int(2.2 * p), int((H - 2.4) * p)), 45, (30, 30, 30), -1)
    cv2.circle(front, (int(2.2 * p), int((H - 2.4) * p)), 40, (245, 245, 245), -1)
    _blocks(front, rng, 18, int(1.7 * p), int((H - 1.9) * p), int(2.8 * p), int((H - 1.0) * p))  # calendar and notices
    tex["front"] = _speckle(front, rng)

    left = wall(D)  # bed wall
    cv2.rectangle(left, (int(0.4 * p), int((H - 0.6) * p)), (int(2.3 * p), left.shape[0]), (30, 50, 110), -1)  # bed side
    for x in range(int(0.4 * p), int(2.3 * p), 22):
        cv2.line(left, (x, int((H - 0.6) * p)), (x, int((H - 0.45) * p)), (60, 60, 200), 6)  # blanket stripes
    sun = (int(1.3 * p), int((H - 1.65) * p))
    cv2.rectangle(left, (int(0.8 * p), int((H - 1.9) * p)), (int(1.8 * p), int((H - 1.3) * p)), (120, 160, 240), -1)
    cv2.circle(left, sun, 25, (210, 240, 255), -1)
    _blocks(left, rng, 30, int(2.5 * p), int((H - 1.8) * p), int(3.4 * p), int((H - 0.8) * p), 6, 30)  # shelf of things
    tex["left"] = _speckle(left, rng)

    right = wall(D)  # almirah and desk wall
    cv2.rectangle(right, (int(0.4 * p), int((H - 1.85) * p)), (int(1.3 * p), right.shape[0]), (150, 152, 155), -1)
    cv2.line(right, (int(0.85 * p), int((H - 1.85) * p)), (int(0.85 * p), right.shape[0]), (60, 60, 60), 3)
    cv2.rectangle(right, (int(0.8 * p), int((H - 1.0) * p)), (int(0.83 * p), int((H - 0.8) * p)), (40, 40, 40), -1)
    cv2.rectangle(right, (int(2.4 * p), int((H - 0.78) * p)), (int(3.4 * p), int((H - 0.72) * p)), (40, 90, 140), -1)  # desk top
    _blocks(right, rng, 20, int(2.5 * p), int((H - 1.1) * p), int(3.3 * p), int((H - 0.8) * p), 8, 30)
    cv2.rectangle(right, (int(1.6 * p), int((H - 2.35) * p)), (int(2.8 * p), int((H - 2.3) * p)), (250, 250, 250), -1)  # tube light
    tex["right"] = _speckle(right, rng)
    return tex


@dataclass
class Plane:
    name: str
    origin: np.ndarray  # world position of texture pixel (0, 0): top-left as seen from inside
    u: np.ndarray  # world vector across the full texture width
    v: np.ndarray  # world vector down the full texture height


def planes() -> list[Plane]:
    a = np.array
    return [
        Plane("back", a([X0, H, Z0]), a([W, 0, 0]), a([0, -H, 0])),
        Plane("front", a([X1, H, Z1]), a([-W, 0, 0]), a([0, -H, 0])),
        Plane("left", a([X0, H, Z1]), a([0, 0, -D]), a([0, -H, 0])),
        Plane("right", a([X1, H, Z0]), a([0, 0, D]), a([0, -H, 0])),
        Plane("floor", a([X0, 0, Z0]), a([W, 0, 0]), a([0, 0, D])),
        Plane("ceiling", a([X0, H, Z1]), a([W, 0, 0]), a([0, 0, -D])),
    ]


def look_rotation(position, target, up=(0, 1, 0)) -> np.ndarray:
    """World-to-camera rotation for an OpenCV camera (x right, y down, z forward)."""
    z = np.asarray(target, float) - np.asarray(position, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def _clip_near(pts_cam: np.ndarray, near=0.05) -> np.ndarray:
    out = []
    n = len(pts_cam)
    for i in range(n):
        a, b = pts_cam[i], pts_cam[(i + 1) % n]
        ina, inb = a[2] >= near, b[2] >= near
        if ina:
            out.append(a)
        if ina != inb:
            t = (near - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.array(out)


def render(R, t, K, size, tex, pls=None) -> np.ndarray:
    w, h = size
    frame = np.zeros((h, w, 3), np.uint8)
    for pl in pls or planes():
        img = tex[pl.name]
        th, tw = img.shape[:2]
        corners = np.array([pl.origin, pl.origin + pl.u, pl.origin + pl.u + pl.v, pl.origin + pl.v])
        cam = corners @ R.T + t
        poly = _clip_near(cam)
        if len(poly) < 3:
            continue
        proj = poly @ K.T
        pix = proj[:, :2] / proj[:, 2:3]
        Hm = K @ np.column_stack([R @ (pl.u / tw), R @ (pl.v / th), R @ pl.origin + t])
        warped = cv2.warpPerspective(img, Hm, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, np.round(pix).astype(np.int32), 255)
        frame[mask > 0] = warped[mask > 0]
    return frame


def walk_poses(n: int, eye: float = 1.45, seed: int = 0, floor_pass: bool = True) -> list[tuple[np.ndarray, np.ndarray]]:
    """Two loops like the scan guide: along the walls looking across, then the middle looking out.

    With floor_pass, the middle loop keeps the floor in the bottom of the frame
    and glances up at the ceiling once, as the guide asks. Without it the phone
    stays nearly level, which is how people film when nobody tells them otherwise.
    """
    rng = np.random.default_rng(seed)
    poses = []
    half = n // 2
    for i in range(n):
        outer = i < half
        k = i / half if outer else (i - half) / (n - half)
        a = 2 * np.pi * k + (0 if outer else 0.6)
        r = 0.62 if outer else 0.22
        pos = np.array([W / 2 * r * np.cos(a), eye + 0.03 * np.sin(7 * a) + rng.normal(0, 0.01), D / 2 * r * np.sin(a)])
        look = a + np.pi + 0.5 if outer else a
        pitch = -0.12 + 0.12 * np.sin(3 * a)
        if floor_pass and not outer:
            pitch = -0.45 + 0.9 * np.exp(-(((k - 0.7) / 0.07) ** 2))
        target = pos + np.array([np.cos(look), pitch, np.sin(look)])
        poses.append((pos, target))
    return poses


def make_scan_video(path: Path, n_frames=150, fps=5, size=(960, 540), hfov_deg=66.0, intro=None, seed=0):
    """Write a video of a walk through the box room. Returns the true camera poses."""
    w, h = size
    f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    tex = textures(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    n_intro = 0
    if intro is not None:
        intro = cv2.resize(intro, size)
        n_intro = int(2 * fps)
        for _ in range(n_intro):
            writer.write(intro)
    truth = []
    for pos, target in walk_poses(n_frames, seed=seed):
        R = look_rotation(pos, target)
        t = -R @ pos
        writer.write(render(R, t, K, size, tex))
        truth.append(pos)
    writer.release()
    return np.array(truth), n_intro, K
