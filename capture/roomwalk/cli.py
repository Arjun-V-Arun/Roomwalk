"""Command line: roomwalk code | check | build | demo | unpublish.

`build` runs the whole pipeline and can resume from any stage, because the
slow stages (structure from motion, training) should not rerun when only the
publishing step changed:

    check -> frames -> sfm -> align -> train -> convert -> publish
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import freshness as fr
from .tools import ToolMissing, log

STAGES = ["check", "frames", "sfm", "align", "train", "convert", "publish"]
REPO_ROOT = Path(__file__).resolve().parents[2]

DEMO_META = {
    "title": "Furnished single room with study table",
    "area": "Synthetic demo, not a real listing",
    "rent": 9500,
    "deposit": 19000,
    "notes": "Furnished: bed, study table, chair, steel almirah, ceiling fan. Shared bathroom on the same floor.",
}
DEMO_VIEW = {
    "start": {"position": [0.55, 1.5, 1.25], "target": [-0.2, 1.25, -1.0]},
    "bounds": {"min": [-1.3, 0.35, -1.6], "max": [1.3, 2.5, 1.6]},
    "eyeHeight": 1.45,
}
DEMO_SPOTS = [
    {"id": "window", "label": "Window", "position": [0.0, 1.5, 0.3], "target": [0.0, 1.45, -1.8]},
    {"id": "bed", "label": "Bed", "position": [0.45, 1.55, -0.2], "target": [-1.0, 0.55, 0.9]},
    {"id": "desk", "label": "Study table", "position": [0.1, 1.5, 0.0], "target": [1.15, 0.8, -1.3]},
    {"id": "almirah", "label": "Almirah", "position": [-0.5, 1.5, 0.4], "target": [1.25, 1.0, 0.7]},
    {"id": "door", "label": "Door", "position": [-0.4, 1.5, -0.9], "target": [0.6, 1.1, 1.8]},
]


def default_web_dir() -> Path:
    return REPO_ROOT / "web"


# ---------------------------------------------------------------- commands


def cmd_code(a) -> int:
    secret = fr.load_secret()
    tok = fr.issue(a.room, secret)
    out = Path(a.out or f"{a.room}-code.png")
    fr.render_qr(tok, out, a.max_age_h)
    print(f"Code for {a.room}: {tok.text}")
    print(f"QR image: {out.resolve()}")
    print(f"Send the image to the tenant. Their video must start with it and be recorded within {a.max_age_h:g} h.")
    print(f"Signing key: {fr.secret_path()} (keep it; videos are checked against it)")
    return 0


def cmd_check(a) -> int:
    from .quality import check_video, format_report, plot_timeline

    secret = fr.load_secret() if a.room else None
    report = check_video(a.video, room_id=a.room, secret=secret, max_age_h=a.max_age_h)
    print(format_report(report))
    if a.json:
        Path(a.json).write_text(json.dumps(report.to_dict(), indent=2))
    if a.plot and plot_timeline(report, a.plot):
        print(f"Timeline chart: {a.plot}")
    return 2 if report.verdict == "FAIL" else 0


class State:
    """Results of each stage, saved so a later run can pick up where it stopped."""

    def __init__(self, work: Path):
        self.path = work / "state.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def __getitem__(self, k):
        return self.data[k]

    def get(self, k, default=None):
        return self.data.get(k, default)

    def set(self, k, v):
        self.data[k] = v
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, default=str))


def cmd_build(a) -> int:
    from .align import compute_alignment
    from .convert import clean, ply_vertex_count, to_rad, to_sog
    from .publish import publish, read_json
    from .quality import check_video, format_report, plot_timeline
    from .select import choose, export, score_frames
    from .sfm import load_poses, run_sfm
    from .train import train

    fr.check_room_id(a.room)
    work = Path(a.work or Path("work") / a.room).resolve()
    dataset = work / "dataset"
    state = State(work)
    start = STAGES.index(a.from_stage)
    if a.ply:
        start = max(start, STAGES.index("convert"))
    stop = STAGES.index(a.to_stage)
    wanted = lambda stage: start <= STAGES.index(stage) <= stop  # noqa: E731

    if wanted("check") or wanted("frames"):
        if not a.video or not Path(a.video).exists():
            print("A video is needed for the check and frames stages.", file=sys.stderr)
            return 1
    if wanted("check"):
        report = check_video(a.video, room_id=a.room, secret=fr.load_secret(), max_age_h=a.max_age_h)
        print(format_report(report))
        work.mkdir(parents=True, exist_ok=True)
        (work / "capture-report.json").write_text(json.dumps(report.to_dict(), indent=2))
        plot_timeline(report, str(work / "capture-timeline.png"))
        state.set("report", {k: v for k, v in report.to_dict().items() if k != "samples"})
        if report.verdict == "FAIL" and not a.force:
            print("\nThe video failed the capture check. Film again, or pass --force to continue anyway.")
            return 2

    if wanted("frames"):
        report = state["report"]
        seg = report.get("qr_segment")
        scored = score_frames(a.video, report["video"].get("frame_count", 0))
        sel = choose(scored, target=a.frames, exclude=[tuple(seg)] if seg else None)
        paths = export(a.video, sel, dataset / "images", max_side=a.max_side, poster_path=work / "poster.jpg")
        # frame_00000.jpg ... map back to these video frame numbers, in order.
        (work / "selected.json").write_text(json.dumps([s.index for s in sel.frames]))
        log(f"kept {len(paths)} frames ({sel.dropped_blurry} windows too blurry, "
            f"{sel.dropped_excluded} frames showing the code skipped)")
        state.set("frames", {"count": len(paths), "dropped_blurry": sel.dropped_blurry,
                             "blur_threshold": sel.blur_threshold})

    if wanted("sfm"):
        res = run_sfm(dataset, matcher=a.matcher, colmap_bin=a.colmap)
        log(f"registered {res.registered} of {res.images} frames, {res.points} sparse points"
            + (f", {res.other_models} disconnected pieces set aside" if res.other_models else ""))
        state.set("sfm", asdict(res))
        if res.registered_fraction < 0.3 and not a.force:
            print("Under 30% of frames could be placed. The 3D result would be mostly holes. "
                  "Film again more slowly with more lights, or pass --force.")
            return 2
        if res.registered_fraction < 0.6:
            log("warning: under 60% of frames were placed; expect gaps")

    if wanted("align"):
        al = compute_alignment(load_poses(dataset))
        log(f"aligned: scale {al.scale:.3f} from {al.scale_source}; average phone roll {al.roll_deg:.1f} degrees")
        if al.scale_confidence in ("low", "none"):
            log("warning: the floor was never clearly in view, so room size is a guess. Next time, keep the "
                "floor in the bottom of the frame during the middle loop.")
        if al.roll_deg > 10:
            log("warning: the phone was often tilted sideways, so 'up' is uncertain; the room may start tilted")
        state.set("align", {"transform": al.manifest_transform(), "view": al.manifest_view(),
                            "up_spread_deg": al.up_spread_deg, "roll_deg": al.roll_deg,
                            "scale_source": al.scale_source,
                            "scale_confidence": al.scale_confidence})

    raw_ply = Path(a.ply) if a.ply else None
    if wanted("train") and raw_ply is None:
        raw_ply = train(dataset, work / "train", iters=a.iters, sh_degree=a.sh, max_splats=a.max_splats)
        state.set("train", {"ply": str(raw_ply)})
    if raw_ply is None and state.get("train"):
        raw_ply = Path(state["train"]["ply"])

    if wanted("convert"):
        if raw_ply is None or not raw_ply.exists():
            print("No trained .ply found. Run the train stage, or pass --ply path/to/export.ply", file=sys.stderr)
            return 1
        out = work / "out"
        cleaned = clean(raw_ply, out / "room.ply", sh_bands=a.sh)
        sog = to_sog(cleaned, out / "room.sog")
        rad = None if a.no_rad else to_rad(cleaned, out / "rad")
        state.set("convert", {"ply": str(cleaned), "sog": str(sog), "rad": str(rad) if rad else None,
                              "count": ply_vertex_count(cleaned)})

    if wanted("publish"):
        conv = state["convert"]
        align = state.get("align") or {}
        report = state.get("report")
        if not align:
            log("warning: no alignment saved; the room is published as trained and may start sideways")
        web = Path(a.web or default_web_dir())
        old = read_json(web / "public" / "rooms" / a.room / "manifest.json", {})
        meta = {k: v for k, v in {"title": a.title, "area": a.area, "rent": a.rent, "deposit": a.deposit,
                                  "notes": a.notes}.items() if v is not None}
        manifest = publish(
            web, a.room, conv["sog"], conv["count"], rad_header=conv.get("rad"),
            poster=work / "poster.jpg", meta=meta, transform=align.get("transform"), view=align.get("view"),
            capture_report=report, freshness=report.get("freshness") if report else None,
            scanned_at=report["video"]["recorded_at"] if report else None,
            hotspots=old.get("hotspots", []) if a.keep_spots else None,
        )
        print(f"\nPublished {a.room} to {manifest.parent}")
        print("Next: cd web && npm run build, then open /r/" + a.room + "/?edit=1 to add spots.")
    if stop < STAGES.index("publish"):
        print(f"Stopped after the {a.to_stage} stage. Resume with --from {STAGES[stop + 1]}.")
    return 0


def cmd_demo(a) -> int:
    from .convert import to_rad, to_sog
    from .demo_room import make_demo_room
    from .publish import publish

    work = Path(a.work or Path("work") / "demo").resolve()
    ply, n = make_demo_room(work / "room.ply")
    log(f"generated {n:,} splats")
    sog = to_sog(ply, work / "room.sog")
    rad = None if a.no_rad else to_rad(ply, work / "rad")
    manifest = publish(a.web or default_web_dir(), "demo", sog, n, rad_header=rad, meta=DEMO_META,
                       view=DEMO_VIEW, hotspots=DEMO_SPOTS, synthetic=True,
                       poster=Path(a.poster) if a.poster else None)
    print(f"Published the demo room to {manifest.parent}")
    return 0


def cmd_unpublish(a) -> int:
    from .publish import unpublish

    unpublish(a.web or default_web_dir(), a.room)
    print(f"Removed {a.room}")
    return 0


# ---------------------------------------------------------------- parser


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="roomwalk", description="Check, rebuild and publish phone scans of rooms.")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("code", help="issue a one-time freshness code (QR image) for a room")
    c.add_argument("room")
    c.add_argument("--out", help="PNG path (default <room>-code.png)")
    c.add_argument("--max-age-h", type=float, default=fr.DEFAULT_MAX_AGE_H)
    c.set_defaults(fn=cmd_code)

    k = sub.add_parser("check", help="check a video before reconstructing it")
    k.add_argument("video")
    k.add_argument("--room", help="room id, to verify the freshness code")
    k.add_argument("--json", help="write the full report here")
    k.add_argument("--plot", help="write a timeline chart (PNG) here")
    k.add_argument("--max-age-h", type=float, default=fr.DEFAULT_MAX_AGE_H)
    k.set_defaults(fn=cmd_check)

    b = sub.add_parser("build", help="video to published room, resumable by stage")
    b.add_argument("video", nargs="?")
    b.add_argument("--room", required=True)
    b.add_argument("--title")
    b.add_argument("--area")
    b.add_argument("--rent", type=int)
    b.add_argument("--deposit", type=int)
    b.add_argument("--notes")
    b.add_argument("--work", help="working folder (default work/<room>)")
    b.add_argument("--web", help="web app folder (default: the repo's web/)")
    b.add_argument("--from", dest="from_stage", choices=STAGES, default="check")
    b.add_argument("--to", dest="to_stage", choices=STAGES, default="publish", help="stop after this stage")
    b.add_argument("--frames", type=int, default=150, help="frames to reconstruct from")
    b.add_argument("--max-side", type=int, default=1600)
    b.add_argument("--matcher", choices=["auto", "exhaustive", "sequential"], default="auto")
    b.add_argument("--colmap", help="path to the COLMAP program (otherwise pycolmap is used)")
    b.add_argument("--iters", type=int, default=30000)
    b.add_argument("--sh", type=int, default=1, choices=[0, 1, 2, 3], help="spherical harmonic degree kept")
    b.add_argument("--max-splats", type=int, default=1_500_000)
    b.add_argument("--ply", help="skip training and use this trained .ply")
    b.add_argument("--no-rad", action="store_true", help="skip the streaming RAD format")
    b.add_argument("--keep-spots", action="store_true", help="keep hotspots from the previous publish")
    b.add_argument("--force", action="store_true", help="continue past failed checks")
    b.add_argument("--max-age-h", type=float, default=fr.DEFAULT_MAX_AGE_H)
    b.set_defaults(fn=cmd_build)

    d = sub.add_parser("demo", help="publish the synthetic demo room")
    d.add_argument("--web")
    d.add_argument("--work")
    d.add_argument("--poster", help="poster image for link previews")
    d.add_argument("--no-rad", action="store_true")
    d.set_defaults(fn=cmd_demo)

    u = sub.add_parser("unpublish", help="remove a room from the web app")
    u.add_argument("room")
    u.add_argument("--web")
    u.set_defaults(fn=cmd_unpublish)
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    try:
        return a.fn(a)
    except ToolMissing as exc:
        print(str(exc), file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
