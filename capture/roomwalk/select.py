"""Choosing which video frames to reconstruct from.

Structure from motion wants frames that are sharp, overlap their neighbours and
are spread evenly along the walk. Taking every Nth frame gives even spacing but
keeps whatever motion blur landed on that frame. Instead we cut the video into
equal time windows and keep the sharpest frame in each window. A window whose
best frame is still blurry is dropped rather than polluting the reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .video import iter_frames, resize_long_side, resize_to_width

WORK_WIDTH = 640
MAX_SCORED_FRAMES = 3000


@dataclass
class Scored:
    index: int
    t: float
    sharpness: float
    brightness: float


@dataclass
class Selection:
    frames: list[Scored]
    windows: int
    dropped_blurry: int
    dropped_excluded: int
    blur_threshold: float
    poster: Scored | None


def score_frames(path: str, frame_count_hint: int = 0) -> list[Scored]:
    every = max(1, int(np.ceil(frame_count_hint / MAX_SCORED_FRAMES))) if frame_count_hint else 1
    out: list[Scored] = []
    for index, t, frame in iter_frames(path, every=every):
        gray = cv2.cvtColor(resize_to_width(frame, WORK_WIDTH), cv2.COLOR_BGR2GRAY)
        out.append(Scored(index, t, float(cv2.Laplacian(gray, cv2.CV_64F).var()), float(gray.mean())))
    return out


def choose(
    scored: list[Scored],
    target: int = 150,
    exclude: list[tuple[float, float]] | None = None,
    blur_abs: float = 25.0,
    blur_rel: float = 0.4,
) -> Selection:
    if not scored:
        raise ValueError("no frames to choose from")
    exclude = exclude or []

    def excluded(s: Scored) -> bool:
        return any(a <= s.t <= b for a, b in exclude)

    usable = [s for s in scored if not excluded(s)]
    dropped_excluded = len(scored) - len(usable)
    if not usable:
        raise ValueError("every frame is inside an excluded range")
    sharp = np.array([s.sharpness for s in usable])
    threshold = max(blur_abs, blur_rel * float(np.median(sharp)))
    t0, t1 = usable[0].t, usable[-1].t
    windows = max(1, min(target, len(usable)))
    edges = np.linspace(t0, t1 + 1e-6, windows + 1)
    picked: list[Scored] = []
    dropped_blurry = 0
    j = 0
    for w in range(windows):
        best: Scored | None = None
        while j < len(usable) and usable[j].t < edges[w + 1]:
            s = usable[j]
            if best is None or s.sharpness > best.sharpness:
                best = s
            j += 1
        if best is None:
            continue
        if best.sharpness < threshold:
            dropped_blurry += 1
            continue
        picked.append(best)

    # Poster for link previews: sharp and well exposed. Brightness near mid-grey
    # beats a frame staring into a window.
    def poster_score(s: Scored) -> float:
        exposure = 1.0 - min(1.0, abs(s.brightness - 128.0) / 128.0)
        return np.log1p(s.sharpness) * (0.3 + exposure)

    poster = max(picked, key=poster_score) if picked else None
    return Selection(picked, windows, dropped_blurry, dropped_excluded, threshold, poster)


def export(
    path: str,
    selection: Selection,
    images_dir: str | Path,
    max_side: int = 1600,
    poster_path: str | Path | None = None,
    jpeg_quality: int = 95,
) -> list[Path]:
    """Decode the video once more and write the chosen frames as JPEGs."""
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    for old in images_dir.glob("*.jpg"):
        old.unlink()
    wanted = {s.index for s in selection.frames}
    poster_index = selection.poster.index if selection.poster else None
    written: list[Path] = []
    last = max(wanted | ({poster_index} if poster_index is not None else set()))
    for index, _t, frame in iter_frames(path):
        if index > last:
            break
        if index in wanted:
            # Sequential names keep COLMAP's sequential matcher in time order.
            out = images_dir / f"frame_{len(written):05d}.jpg"
            cv2.imwrite(str(out), resize_long_side(frame, max_side), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            written.append(out)
        if poster_path is not None and index == poster_index:
            Path(poster_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(poster_path), resize_long_side(frame, 1200), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return written
