"""End-to-end: video with a freshness code -> check -> frames -> SfM -> alignment.

The video is a walk through the synthetic box room along a known path, so the
recovered camera positions can be compared with the truth. Training is
skipped (it needs a GPU); everything before it runs for real, including
pycolmap. Takes a few minutes on a laptop CPU.

    pytest -m slow tests/test_e2e_sfm.py -s
"""

import json
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from roomwalk import freshness as fr
from roomwalk.cli import main as roomwalk
from roomwalk.sfm import load_poses
from roomwalk.align import quat_to_matrix
from synthetic_room import make_scan_video

pytest.importorskip("pycolmap")


def umeyama(src, dst):
    """Least-squares similarity (rotation, scale, translation) taking src onto dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(d0.T @ s0 / len(src))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    scale = (S * np.diag(E)).sum() / s0.var(0).sum()
    return R, scale, mu_d - scale * R @ mu_s


@pytest.mark.slow
def test_video_to_aligned_room(tmp_path):
    room = "e2e-room"
    tok = fr.issue(room, fr.load_secret(), now=datetime.now(timezone.utc) - timedelta(minutes=2))
    code = cv2.imread(str(fr.render_qr(tok, tmp_path / "code.png")))
    card = np.full((540, 960, 3), 235, np.uint8)
    side = 500
    code = cv2.resize(code, (int(side * code.shape[1] / code.shape[0]), side))
    x0 = (960 - code.shape[1]) // 2
    card[20:20 + side, x0:x0 + code.shape[1]] = code
    video = tmp_path / "walk.mp4"
    fps = 5
    truth, n_intro, _K = make_scan_video(video, n_frames=160, fps=fps, intro=card)

    work = tmp_path / "work"
    rc = roomwalk(["build", str(video), "--room", room, "--work", str(work), "--to", "align",
                   "--frames", "80", "--max-side", "960"])
    assert rc == 0
    state = json.loads((work / "state.json").read_text())
    assert state["report"]["freshness"]["status"] == "verified"
    assert state["report"]["verdict"] in ("PASS", "WARN")

    poses = load_poses(work / "dataset")
    sfm = state["sfm"]
    assert sfm["registered"] / sfm["images"] > 0.9, sfm

    tf = state["align"]["transform"]
    qx, qy, qz, qw = tf["quaternion"]
    aligned = tf["scale"] * poses.centers @ quat_to_matrix(qw, qx, qy, qz).T + np.array(tf["position"])
    picked = json.loads((work / "selected.json").read_text())
    true_idx = [picked[int(name.split("_")[1].split(".")[0])] - n_intro for name in poses.names]
    T = truth[true_idx]

    R_fit, s_fit, _t = umeyama(aligned, T)
    gravity_err = np.degrees(np.arccos(np.clip((R_fit @ [0, 1, 0])[1], -1, 1)))
    scale_err = abs(1 / s_fit - 1)
    result = {
        "registered": f"{sfm['registered']}/{sfm['images']}",
        "gravity_error_deg": round(float(gravity_err), 2),
        "scale_error_pct": round(100 * float(scale_err), 1),
        "height_error_cm": round(100 * float(np.abs(aligned[:, 1] - T[:, 1]).mean()), 1),
        "up_spread_deg": round(state["align"]["up_spread_deg"], 1),
        "scale_source": state["align"]["scale_source"],
    }
    print("\nE2E result:", json.dumps(result))
    (tmp_path / "e2e.json").write_text(json.dumps(result))
    assert state["align"]["scale_confidence"] == "high"
    assert gravity_err < 3.0
    assert scale_err < 0.08
