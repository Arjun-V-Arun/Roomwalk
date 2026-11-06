"""Capture checks: is this video good enough to rebuild the room in 3D?

Reconstruction fails quietly. A blurry or half-covered video still produces a
3D scene, just a bad one, and you only find out after 20 minutes of training.
These checks run in seconds and say what to fix before anyone trains.

Every metric is computed on a downscaled grayscale copy (640 px wide), so
thresholds do not depend on whether the phone recorded 1080p or 4K.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

import cv2
import numpy as np

from . import freshness as fr
from .video import VideoInfo, iter_frames, parse_time, probe, resize_to_width

WORK_WIDTH = 640


@dataclass
class Thresholds:
    min_duration_fail_s: float = 10.0
    min_duration_warn_s: float = 30.0
    max_duration_warn_s: float = 240.0
    min_short_side_warn: int = 720
    # A frame is blurry if its Laplacian variance is below max(abs, rel * median).
    blur_abs: float = 25.0
    blur_rel: float = 0.4
    blurry_warn: float = 0.25
    blurry_fail: float = 0.5
    dark_luma: float = 45.0
    dark_warn: float = 0.2
    dark_fail: float = 0.6
    # Fraction of pixels at 250+ that makes a frame "blown out" (usually a window).
    blown_pixels: float = 0.15
    blown_warn: float = 0.25
    # Fewer trackable corners than this and feature matching struggles (plain walls).
    low_texture_corners: int = 60
    low_texture_warn: float = 0.3
    low_texture_fail: float = 0.7
    # Camera speed in frame-widths per second.
    fast_speed: float = 0.9
    fast_warn: float = 0.15
    min_turn_warn_deg: float = 300.0
    qr_window_s: float = 20.0


@dataclass
class FrameSample:
    index: int
    t: float
    sharpness: float
    brightness: float
    blown: float
    corners: int
    dx: float = 0.0  # median horizontal feature motion since the previous sample, px at WORK_WIDTH
    dy: float = 0.0
    speed: float = 0.0  # frame widths per second
    has_qr: bool = False


@dataclass
class Finding:
    level: str  # fail | warn | info
    code: str
    message: str
    hint: str = ""


@dataclass
class CaptureReport:
    video: dict
    verdict: str
    findings: list[dict]
    summary: dict
    freshness: dict
    qr_segment: list[float] | None
    samples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# FAST with an absolute contrast threshold. goodFeaturesToTrack's quality level
# is relative to the strongest corner in the frame, so on a blank wall it
# happily returns hundreds of "corners" made of sensor noise.
_FAST = cv2.FastFeatureDetector_create(threshold=16, nonmaxSuppression=True)
MAX_CORNERS = 400


def frame_metrics(gray: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Sharpness, mean brightness, blown-out fraction and trackable corners for one frame."""
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    blown = float((gray >= 250).mean())
    kps = sorted(_FAST.detect(gray, None), key=lambda k: -k.response)[:MAX_CORNERS]
    pts = np.array([k.pt for k in kps], np.float32).reshape(-1, 1, 2)
    return sharp, bright, blown, pts


def _median_flow(prev_gray, gray, prev_pts) -> tuple[float, float] | None:
    if prev_pts is None or len(prev_pts) < 8:
        return None
    nxt, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, winSize=(21, 21), maxLevel=3)
    if nxt is None:
        return None
    good = status.reshape(-1) == 1
    if good.sum() < 8:
        return None
    d = (nxt[good] - prev_pts[good]).reshape(-1, 2)
    return float(np.median(d[:, 0])), float(np.median(d[:, 1]))


def analyze(path: str, sample_fps: float = 6.0, thresholds: Thresholds | None = None):
    """Single decoding pass. Returns (VideoInfo, samples, first QR text found or None)."""
    th = thresholds or Thresholds()
    info = probe(path)
    every = max(1, round(info.fps / sample_fps))
    detector = fr.make_detector()
    samples: list[FrameSample] = []
    prev_gray = prev_pts = None
    prev_t = 0.0
    qr_text: str | None = None
    misses_after_qr = 0
    for index, t, frame in iter_frames(path, every=every):
        small = resize_to_width(frame, WORK_WIDTH)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharp, bright, blown, pts = frame_metrics(gray)
        s = FrameSample(index, t, sharp, bright, blown, int(len(pts)))
        flow = _median_flow(prev_gray, gray, prev_pts) if prev_gray is not None else None
        if flow is not None and t > prev_t:
            s.dx, s.dy = flow
            s.speed = float(np.hypot(*flow)) / WORK_WIDTH / (t - prev_t)
        # Look for the freshness code near the start. Decode at full resolution,
        # because the code can be small in frame even when filmed up close.
        if t <= th.qr_window_s and misses_after_qr < 3:
            text = fr.decode_qr(detector, frame)
            if text and text.startswith(fr.PREFIX + ":"):
                s.has_qr = True
                qr_text = qr_text or text
                misses_after_qr = 0
            elif qr_text:
                misses_after_qr += 1
        samples.append(s)
        prev_gray, prev_pts, prev_t = gray, pts, t
    if not samples:
        raise ValueError(f"{path}: no frames decoded")
    # Containers sometimes report a wrong frame count; trust the decoded timeline.
    last = samples[-1].t + 1.0 / max(sample_fps, 1e-6)
    if info.duration_s <= 0 or abs(info.duration_s - last) > 2.0:
        info.duration_s = round(last, 2)
    return info, samples, qr_text


def qr_segment(samples: list[FrameSample]) -> list[float] | None:
    """Time span in which the freshness code is on screen, padded by half a second."""
    hits = [s.t for s in samples if s.has_qr]
    if not hits:
        return None
    return [max(0.0, min(hits) - 0.5), max(hits) + 0.5]


def blur_threshold(sharpness: np.ndarray, th: Thresholds) -> float:
    return max(th.blur_abs, th.blur_rel * float(np.median(sharpness)))


def evaluate(info: VideoInfo, samples: list[FrameSample], th: Thresholds) -> tuple[list[Finding], dict]:
    seg = qr_segment(samples)
    # Judge the room footage only, not the seconds spent filming the code.
    room = [s for s in samples if not (seg and seg[0] <= s.t <= seg[1])] or samples
    sharp = np.array([s.sharpness for s in room])
    bright = np.array([s.brightness for s in room])
    blown = np.array([s.blown for s in room])
    corners = np.array([s.corners for s in room])
    speed = np.array([s.speed for s in room[1:]] or [0.0])
    bt = blur_threshold(sharp, th)
    blurry = float((sharp < bt).mean())
    dark = float((bright < th.dark_luma).mean())
    blown_frac = float((blown > th.blown_pixels).mean())
    low_tex = float((corners < th.low_texture_corners).mean())
    fast = float((speed > th.fast_speed).mean())
    hfov = 50.0 if info.portrait else 66.0
    net_dx = sum(s.dx for s in room)
    est_turn = abs(net_dx) / WORK_WIDTH * hfov
    summary = {
        "duration_s": round(info.duration_s, 1),
        "resolution": f"{info.width}x{info.height}",
        "orientation": "portrait" if info.portrait else "landscape",
        "median_sharpness": round(float(np.median(sharp)), 1),
        "blur_threshold": round(bt, 1),
        "blurry_fraction": round(blurry, 3),
        "dark_fraction": round(dark, 3),
        "blown_fraction": round(blown_frac, 3),
        "low_texture_fraction": round(low_tex, 3),
        "median_corners": int(np.median(corners)),
        "fast_fraction": round(fast, 3),
        "p90_speed": round(float(np.percentile(speed, 90)), 3),
        "estimated_turn_deg": round(est_turn),
        "samples": len(samples),
    }

    f: list[Finding] = []
    d = info.duration_s
    if d < th.min_duration_fail_s:
        f.append(Finding("fail", "too_short", f"Video is only {d:.0f} s long",
                         "Film for about a minute so every wall is seen from a few angles."))
    elif d < th.min_duration_warn_s:
        f.append(Finding("warn", "short", f"Video is {d:.0f} s long",
                         "Aim for 45 to 90 seconds. Short videos leave gaps behind furniture."))
    elif d > th.max_duration_warn_s:
        f.append(Finding("warn", "long", f"Video is {d:.0f} s long",
                         "Over 4 minutes adds processing time without adding much detail."))
    if info.short_side < th.min_short_side_warn:
        f.append(Finding("warn", "low_resolution", f"Resolution is {info.width}x{info.height}",
                         "Record at 1080p or higher."))
    if blurry > th.blurry_fail:
        f.append(Finding("fail", "blurry", f"{blurry:.0%} of frames are blurry",
                         "Move slower, keep the phone steady, and turn on more lights."))
    elif blurry > th.blurry_warn:
        f.append(Finding("warn", "blurry", f"{blurry:.0%} of frames are blurry",
                         "Blurry frames are skipped, but large blurry stretches leave holes."))
    if dark > th.dark_fail:
        f.append(Finding("fail", "dark", f"{dark:.0%} of frames are too dark",
                         "Switch on all the lights and open the curtains, then film again."))
    elif dark > th.dark_warn:
        f.append(Finding("warn", "dark", f"{dark:.0%} of frames are dark",
                         "Dark areas come out noisy and smeared. Turn on more lights."))
    if blown_frac > th.blown_warn:
        f.append(Finding("warn", "blown_out", f"{blown_frac:.0%} of frames have large blown-out areas",
                         "Bright windows wash out. Tap the screen on a wall to set exposure."))
    if low_tex > th.low_texture_fail:
        f.append(Finding("fail", "low_texture", f"{low_tex:.0%} of frames show almost no detail",
                         "Plain walls cannot be matched. Keep furniture, doors or the floor in every shot."))
    elif low_tex > th.low_texture_warn:
        f.append(Finding("warn", "low_texture", f"{low_tex:.0%} of frames show little detail",
                         "Avoid pointing straight at blank walls; include an edge or object in each shot."))
    if fast > th.fast_warn:
        f.append(Finding("warn", "too_fast", f"Camera moves too fast in {fast:.0%} of the video",
                         "Turn slowly, about a quarter turn every 5 seconds."))
    if est_turn < th.min_turn_warn_deg:
        f.append(Finding("warn", "coverage", f"Estimated turn is only about {est_turn:.0f} degrees",
                         "Walk a full loop of the room so every wall is filmed."))
    return f, summary


def verdict_of(findings: list[Finding]) -> str:
    levels = {x.level for x in findings}
    return "FAIL" if "fail" in levels else "WARN" if "warn" in levels else "PASS"


def check_video(
    path: str,
    room_id: str | None = None,
    secret: bytes | None = None,
    max_age_h: float = fr.DEFAULT_MAX_AGE_H,
    thresholds: Thresholds | None = None,
    sample_fps: float = 6.0,
) -> CaptureReport:
    th = thresholds or Thresholds()
    info, samples, qr_text = analyze(path, sample_fps=sample_fps, thresholds=th)
    findings, summary = evaluate(info, samples, th)
    seg = qr_segment(samples)

    recorded = parse_time(info.recorded_at) if info.recorded_at else None
    if room_id is None or secret is None:
        fresh = fr.FreshnessResult("not_checked", "No room id given, freshness not checked")
    elif qr_text is None:
        fresh = fr.FreshnessResult("not_found", "No RoomWalk code found in the first 20 seconds")
    else:
        fresh = fr.verify(qr_text, room_id, secret, recorded or datetime.now().astimezone(),
                          max_age_h=max_age_h, seen_at_s=seg[0] + 0.5 if seg else None)
    if room_id is not None and not fresh.verified:
        level = "warn" if fresh.status == "not_found" else "fail"
        findings.append(Finding(level, "freshness", f"Freshness code: {fresh.detail}",
                                "Film the code up close for two seconds before filming the room."))
    if fresh.verified and info.recorded_at_source != "metadata":
        findings.append(Finding("info", "weak_timestamp",
                                "Recording time came from the file date, not the video metadata",
                                "Send the original file, not a forwarded copy, so the phone's timestamp survives."))

    return CaptureReport(
        video=asdict(info),
        verdict=verdict_of(findings),
        findings=[asdict(x) for x in findings],
        summary=summary,
        freshness=fresh.to_dict(),
        qr_segment=seg,
        samples=[asdict(s) for s in samples],
    )


def format_report(r: CaptureReport) -> str:
    s = r.summary
    marks = {"fail": "FAIL", "warn": "WARN", "info": "INFO"}
    lines = [
        f"Verdict: {r.verdict}",
        f"  {s['duration_s']} s, {s['resolution']} {s['orientation']}, "
        f"turned about {s['estimated_turn_deg']} degrees",
        f"  blurry {s['blurry_fraction']:.0%}, dark {s['dark_fraction']:.0%}, "
        f"low detail {s['low_texture_fraction']:.0%}, too fast {s['fast_fraction']:.0%}",
        f"  freshness: {r.freshness['status']} ({r.freshness['detail']})",
    ]
    for x in r.findings:
        lines.append(f"  [{marks[x['level']]}] {x['message']}")
        if x["hint"]:
            lines.append(f"         {x['hint']}")
    return "\n".join(lines)


def plot_timeline(r: CaptureReport, out_png: str) -> str | None:
    """Sharpness, brightness and detail over time, with problem frames marked."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    t = [x["t"] for x in r.samples]
    th = r.summary["blur_threshold"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    series = [
        ("sharpness", "Sharpness", th, "#2a6fdb"),
        ("brightness", "Brightness", Thresholds().dark_luma, "#d98c1f"),
        ("corners", "Trackable corners", Thresholds().low_texture_corners, "#2f9e6a"),
    ]
    for ax, (key, label, line, color) in zip(axes, series):
        ax.plot(t, [x[key] for x in r.samples], color=color, linewidth=1.4)
        ax.axhline(line, color="#c0392b", linewidth=1, linestyle="--")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        if r.qr_segment:
            ax.axvspan(*r.qr_segment, color="#999999", alpha=0.15)
    axes[-1].set_xlabel("Seconds")
    axes[0].set_title(f"Capture check: {r.verdict}  (dashed line = problem threshold, grey = code on screen)",
                      fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png
