"""Alignment is checked against a scene where the right answer is known.

We place cameras the way a person films a room (eye height, walking a loop,
looking around), add floor and wall points, then hide all of it behind a
random rotation, scale and offset, the way structure from motion would.
compute_alignment has to undo that.
"""

import numpy as np
import pytest

from roomwalk.align import compute_alignment, poses_from_world_to_camera, quat_to_matrix, rotation_between
from roomwalk.demo_room import matrix_to_quat


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return quat_to_matrix(*q)


def look_rotation(forward, up=(0, 1, 0)):
    """World-to-camera rotation for an OpenCV camera looking along `forward`."""
    z = np.asarray(forward, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))  # image x points right
    x /= np.linalg.norm(x)
    y = np.cross(z, x)  # image y points down
    R = np.stack([x, y, z])  # rows: camera axes in world coordinates
    assert np.isclose(np.linalg.det(R), 1.0)
    return R


def room_scene(rng, n=60, eye=1.5, tilt_noise=0.05):
    centers, Rs, ts = [], [], []
    for k in range(n):
        a = 2 * np.pi * k / n
        c = np.array([1.1 * np.cos(a), eye + rng.normal(0, 0.05), 0.9 * np.sin(a)])
        fwd = np.array([np.cos(a + 1.2), rng.normal(0, tilt_noise), np.sin(a + 1.2)])
        R = look_rotation(fwd)
        centers.append(c)
        Rs.append(R)
        ts.append(-R @ c)
    floor = np.c_[rng.uniform(-1.5, 1.5, 3000), rng.normal(0, 0.01, 3000), rng.uniform(-1.8, 1.8, 3000)]
    wall = np.c_[rng.uniform(-1.5, 1.5, 3000), rng.uniform(0, 2.8, 3000), np.full(3000, -1.8)]
    return np.array(centers), np.array(Rs), np.array(ts), np.vstack([floor, wall])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_recovers_up_scale_and_floor(seed):
    rng = np.random.default_rng(seed)
    centers, Rs, ts, pts = room_scene(rng)
    # What SfM would hand us: same scene in a random similarity frame.
    Q, s, off = random_rotation(rng), float(rng.uniform(0.2, 5)), rng.normal(0, 3, 3)
    to_model = lambda p: (s * (p @ Q.T)) + off  # noqa: E731
    model_centers = to_model(centers)
    model_R = np.array([R @ Q.T for R in Rs])  # world-to-camera in the model frame
    model_t = np.array([-R @ c for R, c in zip(model_R, model_centers)])
    poses = poses_from_world_to_camera([f"{i:03d}" for i in range(len(Rs))], model_R, model_t, to_model(pts))

    al = compute_alignment(poses, eye_height=1.5)
    back = al.apply(model_centers)
    # Height and floor come out right by construction; the real test of the
    # scale estimate is a horizontal distance. Cameras 0 and n/2 sit at opposite
    # ends of the 1.1 m semi-axis, so they should be 2.2 m apart.
    assert np.allclose(back[:, 1].mean(), 1.5, atol=0.08)
    assert abs(np.percentile(al.apply(to_model(pts))[:, 1], 2)) < 0.08
    ring = np.linalg.norm(back[0, [0, 2]] - back[len(back) // 2, [0, 2]])
    assert ring == pytest.approx(2.2, rel=0.05)
    # Gravity is recovered: true world up maps onto +y.
    assert np.allclose(al.rotation @ (Q @ [0, 1, 0]), [0, 1, 0], atol=0.03)
    assert al.up_spread_deg < 6


def test_start_view_faces_first_camera_heading():
    rng = np.random.default_rng(5)
    centers, Rs, ts, pts = room_scene(rng)
    poses = poses_from_world_to_camera(range(len(Rs)), Rs, ts, pts)
    al = compute_alignment(poses)
    first_fwd = al.rotation @ poses.forwards[0]
    first_fwd[1] = 0
    first_fwd /= np.linalg.norm(first_fwd)
    assert np.allclose(first_fwd, [0, 0, -1], atol=1e-6)


def test_manifest_quaternion_matches_rotation():
    rng = np.random.default_rng(9)
    R = random_rotation(rng)
    w, x, y, z = matrix_to_quat(R)
    assert np.allclose(quat_to_matrix(w, x, y, z), R, atol=1e-9)


def test_rotation_between_handles_opposites():
    R = rotation_between(np.array([0, -1.0, 0]), np.array([0, 1.0, 0]))
    assert np.allclose(R @ [0, -1, 0], [0, 1, 0])
    assert np.allclose(R @ R.T, np.eye(3))


def _walls_only(rng, with_patch=False, with_ceiling=False):
    """Room seen with the phone held level: walls between 0.5 and 1.9 m, no floor."""
    centers, Rs, ts, _ = room_scene(rng)
    n = 4000
    side = rng.integers(0, 4, n)
    u = rng.uniform(-1, 1, n)
    y = rng.uniform(0.5, 1.9, n)
    x = np.where(side == 0, -1.5, np.where(side == 1, 1.5, u * 1.5))
    z = np.where(side == 2, -1.8, np.where(side == 3, 1.8, u * 1.8))
    pts = [np.c_[x, y, z]]
    if with_patch:  # a bed top: low, flat and small
        pts.append(np.c_[rng.uniform(-1.0, -0.3, 800), rng.normal(0.58, 0.005, 800), rng.uniform(-0.9, 0.0, 800)])
    if with_ceiling:
        pts.append(np.c_[rng.uniform(-1.4, 1.4, 1500), rng.normal(2.9, 0.01, 1500), rng.uniform(-1.7, 1.7, 1500)])
    return centers, Rs, ts, np.vstack(pts)


def _scale_error(al, centers):
    back = al.apply(centers)
    ring = np.linalg.norm(back[0, [0, 2]] - back[len(back) // 2, [0, 2]])
    return abs(ring / 2.2 - 1)


def test_no_floor_in_view_is_flagged_not_trusted():
    rng = np.random.default_rng(11)
    centers, Rs, ts, pts = _walls_only(rng)
    # Hide the true size: SfM output is in arbitrary units, here 3.7 per metre.
    al = compute_alignment(poses_from_world_to_camera(range(len(Rs)), Rs, ts * 3.7, pts * 3.7), eye_height=1.5)
    assert al.scale_confidence == "low"
    assert _scale_error(al, centers * 3.7) < 0.4


def test_bed_top_is_not_mistaken_for_floor():
    rng = np.random.default_rng(12)
    centers, Rs, ts, pts = _walls_only(rng, with_patch=True)
    al = compute_alignment(poses_from_world_to_camera(range(len(Rs)), Rs, ts, pts), eye_height=1.5)
    assert al.scale_confidence != "high"


def test_ceiling_is_used_when_floor_is_missing():
    rng = np.random.default_rng(13)
    centers, Rs, ts, pts = _walls_only(rng, with_ceiling=True)
    al = compute_alignment(poses_from_world_to_camera(range(len(Rs)), Rs, ts, pts), eye_height=1.45)
    assert al.scale_confidence == "medium"
    assert _scale_error(al, centers) < 0.1


def test_gravity_is_not_biased_by_looking_down():
    """Half the walk looks 35 degrees down, but only while facing some walls.
    Averaging up vectors tilts gravity; right vectors do not care about pitch."""
    rng = np.random.default_rng(21)
    Rs, ts = [], []
    for k in range(80):
        a = 2 * np.pi * k / 80
        c = np.array([np.cos(a), 1.45, np.sin(a)])
        pitch = -0.6 if np.cos(a) > 0 else 0.0
        fwd = np.array([np.cos(a + 2.5), np.tan(pitch), np.sin(a + 2.5)])
        R = look_rotation(fwd)
        Rs.append(R)
        ts.append(-R @ c)
    Q = random_rotation(rng)
    model_R = np.array([R @ Q.T for R in Rs])
    model_t = np.array(ts)  # rotation about the origin keeps centres consistent with t
    al = compute_alignment(poses_from_world_to_camera(range(80), model_R, model_t))
    true_up_in_model = Q @ [0, 1, 0]
    assert np.degrees(np.arccos(np.clip(al.rotation @ true_up_in_model @ [0, 1, 0], -1, 1))) < 0.5
    assert al.roll_deg < 0.5
