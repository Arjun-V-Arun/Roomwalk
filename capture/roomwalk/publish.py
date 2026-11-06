"""Putting a finished room into the web app.

Each room gets a folder under web/public/rooms/<id>/ holding the splat files,
a poster photo for link previews and manifest.json. rooms/index.json lists
every room for the home page. The site is static, so publishing is just
writing files; run `npm run build` in web/ afterwards to regenerate pages.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .convert import dir_bytes
from .freshness import check_room_id

SCHEMA = 1
DEFAULT_VIEW = {
    "start": {"position": [0.0, 1.45, 0.0], "target": [0.0, 1.4, -1.0]},
    "bounds": {"min": [-3.0, 0.3, -3.0], "max": [3.0, 2.6, 3.0]},
    "eyeHeight": 1.45,
}
IDENTITY = {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0], "scale": 1.0}
INDEX_FIELDS = ("id", "title", "area", "rent", "scannedAt", "poster", "synthetic")


def rooms_dir(web_dir: str | Path) -> Path:
    return Path(web_dir) / "public" / "rooms"


def read_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def publish(
    web_dir: str | Path,
    room_id: str,
    sog: str | Path,
    splat_count: int,
    rad_header: str | Path | None = None,
    poster: str | Path | None = None,
    meta: dict | None = None,
    transform: dict | None = None,
    view: dict | None = None,
    capture_report: dict | None = None,
    freshness: dict | None = None,
    scanned_at: str | None = None,
    hotspots: list[dict] | None = None,
    synthetic: bool = False,
) -> Path:
    check_room_id(room_id)
    dest = rooms_dir(web_dir) / room_id
    old = read_json(dest / "manifest.json", {})
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(sog, dest / "room.sog")
    rad = None
    if rad_header is not None:
        src_dir = Path(rad_header).parent
        rad_dir = dest / "rad"
        if rad_dir.exists():
            shutil.rmtree(rad_dir)
        shutil.copytree(src_dir, rad_dir, ignore=shutil.ignore_patterns("*.ply"))
        rad = {"file": f"rad/{Path(rad_header).name}", "bytes": dir_bytes(rad_dir)}
    elif (dest / "rad").exists():
        shutil.rmtree(dest / "rad")  # a stale tree from an older scan would not match
    poster_name = None
    if poster is not None and Path(poster).exists():
        shutil.copyfile(poster, dest / "poster.jpg")
        poster_name = "poster.jpg"
    elif (dest / "poster.jpg").exists():
        poster_name = "poster.jpg"

    meta = {**{k: old.get(k) for k in ("title", "area", "rent", "deposit", "notes")}, **(meta or {})}
    capture = None
    if capture_report:
        capture = {
            "verdict": capture_report["verdict"],
            "findings": [{k: f[k] for k in ("level", "code", "message")} for f in capture_report["findings"]],
            "summary": {k: capture_report["summary"].get(k) for k in
                        ("duration_s", "resolution", "blurry_fraction", "estimated_turn_deg")},
        }
    manifest = {
        "schema": SCHEMA,
        "id": room_id,
        "title": meta.get("title") or room_id.replace("-", " ").title(),
        "area": meta.get("area"),
        "rent": meta.get("rent"),
        "deposit": meta.get("deposit"),
        "notes": meta.get("notes"),
        "synthetic": synthetic,
        "scannedAt": scanned_at or old.get("scannedAt"),
        "publishedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "freshness": freshness,
        "capture": capture,
        "splats": {
            "count": splat_count,
            "sog": {"file": "room.sog", "bytes": (dest / "room.sog").stat().st_size},
            "rad": rad,
        },
        "poster": poster_name,
        "transform": transform or IDENTITY,
        "view": view or DEFAULT_VIEW,
        # A new scan moves everything, so old hotspots only survive if passed back in.
        "hotspots": hotspots if hotspots is not None else [],
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    update_index(web_dir, manifest)
    return dest / "manifest.json"


def update_index(web_dir: str | Path, manifest: dict) -> Path:
    path = rooms_dir(web_dir) / "index.json"
    index = read_json(path, {"rooms": []})
    entry = {k: manifest.get(k) for k in INDEX_FIELDS}
    entry["verified"] = bool(manifest.get("freshness") and manifest["freshness"].get("status") == "verified")
    entry["splatCount"] = manifest["splats"]["count"]
    rooms = [r for r in index["rooms"] if r["id"] != manifest["id"]] + [entry]
    # Real rooms first, newest scan first; the synthetic demo goes last.
    real = sorted([r for r in rooms if not r.get("synthetic")], key=lambda r: r.get("scannedAt") or "", reverse=True)
    demo = [r for r in rooms if r.get("synthetic")]
    path.write_text(json.dumps({"rooms": real + demo}, indent=2), encoding="utf-8")
    return path


def unpublish(web_dir: str | Path, room_id: str) -> None:
    check_room_id(room_id)
    dest = rooms_dir(web_dir) / room_id
    if dest.exists():
        shutil.rmtree(dest)
    path = rooms_dir(web_dir) / "index.json"
    index = read_json(path, {"rooms": []})
    index["rooms"] = [r for r in index["rooms"] if r["id"] != room_id]
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")
