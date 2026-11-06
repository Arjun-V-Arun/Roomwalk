"""Reading phone videos: basic metadata and frame iteration."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    # When the phone says it started recording. None if the container has no tag.
    recorded_at: str | None
    # Where recorded_at came from: "metadata", "file-mtime" or "unknown".
    recorded_at_source: str

    @property
    def portrait(self) -> bool:
        return self.height > self.width

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)


def _ffprobe_creation_time(path: Path) -> str | None:
    """Return the recording time as an ISO 8601 UTC string, if ffprobe can find one."""
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
        data = json.loads(out or "{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    tags: list[dict] = [data.get("format", {}).get("tags", {})]
    tags += [s.get("tags", {}) for s in data.get("streams", [])]
    # iPhones write a local-time tag with an offset; most Android phones write UTC creation_time.
    for key in ("com.apple.quicktime.creationdate", "creation_time"):
        for t in tags:
            value = t.get(key)
            if value:
                parsed = parse_time(value)
                if parsed is not None:
                    return parsed.astimezone(timezone.utc).isoformat()
    return None


def parse_time(value: str) -> datetime | None:
    text = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        # Some phones write "+0530" without a colon.
        try:
            dt = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open {path}. Is it a video file?")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"{path} has no readable frames")
    # Read the size from a decoded frame, so phone rotation metadata is already applied.
    h, w = frame.shape[:2]
    recorded = _ffprobe_creation_time(path)
    source = "metadata"
    if recorded is None:
        recorded = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        source = "file-mtime"
    return VideoInfo(
        path=str(path), width=w, height=h, fps=float(fps), frame_count=count,
        duration_s=count / fps if fps > 0 else 0.0,
        recorded_at=recorded, recorded_at_source=source,
    )


def resize_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    scale = width / w
    return cv2.resize(img, (width, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def resize_long_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return img
    scale = max_side / long_side
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def iter_frames(path: str | Path, every: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, time_seconds, bgr_frame) for every `every`-th frame.

    Frames are decoded in order. Seeking is avoided because phone videos often
    use variable frame rates, where seeking by index is slow or inexact.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    index = 0
    try:
        while True:
            if index % every == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                t = msec / 1000.0 if msec and msec > 0 else index / fps
                yield index, t, frame
            else:
                # grab() skips the pixel conversion, which is most of the decode cost.
                if not cap.grab():
                    break
            index += 1
    finally:
        cap.release()
