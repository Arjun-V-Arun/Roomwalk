"""Cleaning a trained splat and packing it for phones.

Two delivery formats come out of one cleaned PLY:

* SOG (PlayCanvas): one small file, about 10x smaller than PLY. The browser
  downloads all of it, then Spark builds a level-of-detail tree on the phone.
* RAD (Spark, optional): a precomputed level-of-detail tree split into chunks.
  The browser fetches the header and the coarse chunks first, so the room
  appears before the whole file arrives. Bigger in total, faster to first view.

The benchmark script compares the two on a throttled connection.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .tools import build_lod, log, run, splat_transform


def ply_vertex_count(path: str | Path) -> int:
    with open(path, "rb") as fh:
        for _ in range(200):
            line = fh.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"{path}: no vertex count in PLY header")


def clean(raw_ply: str | Path, out_ply: str | Path, sh_bands: int = 1, floaters: bool = True) -> Path:
    """Drop broken splats, floaters and unused spherical harmonic bands."""
    out = Path(out_ply)
    out.parent.mkdir(parents=True, exist_ok=True)
    base = splat_transform() + ["-w", str(raw_ply), "-N", "-H", str(sh_bands)]
    if floaters:
        # Floater removal voxelises the scene on the GPU. If no GPU adapter is
        # available it fails, and we fall back to the cheaper filters only.
        proc = run(base + ["-F", str(out)])
        if proc.returncode == 0:
            return out
        log("floater filter failed (needs a GPU adapter); continuing without it")
    proc = run(base + [str(out)])
    if proc.returncode != 0:
        raise RuntimeError("splat-transform could not clean the PLY")
    return out


def to_sog(ply: str | Path, out_sog: str | Path, cpu: bool = False) -> Path:
    out = Path(out_sog)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = splat_transform() + ["-w"] + (["-g", "cpu"] if cpu else []) + [str(ply), str(out)]
    if run(cmd).returncode != 0:
        if not cpu:
            log("SOG compression on the GPU failed, retrying on the CPU")
            return to_sog(ply, out_sog, cpu=True)
        raise RuntimeError("splat-transform could not write SOG")
    return out


def to_rad(ply: str | Path, out_dir: str | Path, name: str = "room") -> Path | None:
    """Chunked RAD for streaming. Returns the header path, or None if build-lod is not installed."""
    exe = build_lod()
    if exe is None:
        log("build-lod not found, skipping the streaming format (SOG still works)")
        return None
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    # build-lod names its output after the input file and writes beside it.
    staged = out_dir / f"{name}.ply"
    shutil.copyfile(ply, staged)
    proc = run([exe, "--quality", "--rad-chunked", staged.name], cwd=out_dir)
    staged.unlink(missing_ok=True)
    header = out_dir / f"{name}-lod.rad"
    if proc.returncode != 0 or not header.exists():
        raise RuntimeError("build-lod failed")
    return header


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
