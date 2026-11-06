"""Training the Gaussian splat with Brush.

Brush is an open-source trainer that runs on any modern GPU through WebGPU, so
it needs no CUDA toolkit. It reads the COLMAP dataset directly, including
OPENCV lens distortion. Released versions name a few flags differently from
the development branch, so we read `--help` and pick whichever exists.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .tools import ToolMissing, brush, log, run

INSTALL_HINT = (
    "Brush not found. Download it from https://github.com/ArthurBrussee/brush/releases, "
    "then put brush_app.exe on PATH or set ROOMWALK_BRUSH to its full path."
)


def brush_flags(exe: str) -> dict[str, str]:
    try:
        text = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        text = ""
    return {"iters": "--total-train-iters" if "--total-train-iters" in text else "--total-steps"}


def train(
    dataset: str | Path,
    out_dir: str | Path,
    iters: int = 30000,
    max_resolution: int = 1600,
    sh_degree: int = 1,
    max_splats: int = 1_500_000,
    extra: list[str] | None = None,
) -> Path:
    exe = brush()
    if exe is None:
        raise ToolMissing(INSTALL_HINT)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    flags = brush_flags(exe)
    cmd = [
        exe, str(Path(dataset).resolve()),
        flags["iters"], str(iters),
        "--max-resolution", str(max_resolution),
        "--sh-degree", str(sh_degree),
        "--max-splats", str(max_splats),
        "--export-every", str(iters),
        "--export-path", str(out_dir),
        "--export-name", "raw.ply",
        *(extra or []),
    ]
    log(f"training {iters} steps with Brush; on an RTX 4070 laptop expect roughly 15 to 30 minutes")
    if run(cmd).returncode != 0:
        raise RuntimeError("Brush training failed. Run the printed command by hand to see its output.")
    ply = out_dir / "raw.ply"
    if not ply.exists():
        candidates = sorted(out_dir.glob("*.ply"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise RuntimeError(f"Brush finished but wrote no .ply into {out_dir}")
        ply = candidates[-1]
    return ply
