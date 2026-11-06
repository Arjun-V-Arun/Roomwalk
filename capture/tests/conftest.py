"""Synthetic test videos.

A real room is replaced by a long, detailed panorama strip. Panning a window
across it imitates turning on the spot with a phone, so we know exactly how far
the "camera" turned and which seconds we made blurry or dark.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

HFOV_LANDSCAPE = 66.0


def panorama(width: int = 6000, height: int = 720, seed: int = 0, plain: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), (200, 205, 210), np.uint8)
    if plain:
        noise = rng.normal(0, 1.5, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)
    for _ in range(width // 6):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        w, h = int(rng.integers(10, 140)), int(rng.integers(10, 140))
        color = tuple(int(c) for c in rng.integers(20, 235, 3))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1 if rng.random() < 0.6 else 3)
    for _ in range(width // 40):
        x, y = int(rng.integers(0, width)), int(rng.integers(30, height))
        cv2.putText(img, "".join(rng.choice(list("ABCDEFGH0123456789"), 6)), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.6, 1.6)), (30, 30, 30), 2)
    return img


def qr_frame(text: str, size: tuple[int, int]) -> np.ndarray:
    import qrcode

    w, h = size
    frame = np.full((h, w, 3), 245, np.uint8)
    q = qrcode.make(text, border=4).convert("RGB")
    side = int(h * 0.75)
    q = cv2.resize(np.array(q)[:, :, ::-1], (side, side), interpolation=cv2.INTER_NEAREST)
    y, x = (h - side) // 2, (w - side) // 2
    frame[y:y + side, x:x + side] = q
    return frame


def make_video(
    path: Path,
    seconds: float = 32.0,
    fps: int = 8,
    size: tuple[int, int] = (1280, 720),
    turn_deg: float = 360.0,
    blur: list[tuple[float, float]] = (),
    dark: list[tuple[float, float]] = (),
    qr_text: str | None = None,
    qr_seconds: float = 2.0,
    plain: bool = False,
) -> Path:
    w, h = size
    pano = panorama(height=h, plain=plain)
    pano = np.concatenate([pano, pano[:, : w + 10]], axis=1)  # wrap around for a full loop
    px_per_deg = w / HFOV_LANDSCAPE
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = int(seconds * fps)
    room_start = qr_seconds if qr_text else 0.0
    for i in range(n):
        t = i / fps
        if qr_text and t < qr_seconds:
            frame = qr_frame(qr_text, size)
        else:
            progress = (t - room_start) / max(1e-6, seconds - room_start)
            x = int(progress * turn_deg * px_per_deg) % (pano.shape[1] - w - 5)
            frame = pano[:, x : x + w].copy()
            if any(a <= t <= b for a, b in blur):
                frame = cv2.GaussianBlur(frame, (0, 0), 6)
            if any(a <= t <= b for a, b in dark):
                frame = (frame * 0.12).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def tmp_video(tmp_path):
    def _make(name: str = "room.mp4", **kw):
        return make_video(tmp_path / name, **kw)

    return _make


@pytest.fixture(autouse=True)
def isolated_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("ROOMWALK_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ROOMWALK_SECRET", raising=False)
