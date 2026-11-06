"""Finding and running the external programs the pipeline depends on."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SPLAT_TRANSFORM_VERSION = "3.10.3"


class ToolMissing(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[roomwalk] {msg}", flush=True)


def run(cmd: list[str], cwd: str | Path | None = None, quiet: bool = False) -> subprocess.CompletedProcess:
    log("$ " + " ".join(str(c) for c in cmd))
    start = time.perf_counter()
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd, text=True,
                          capture_output=quiet, encoding="utf-8", errors="replace")
    log(f"  finished in {time.perf_counter() - start:.1f} s (exit {proc.returncode})")
    return proc


def splat_transform() -> list[str]:
    """Command prefix for PlayCanvas splat-transform: env override, global install, then npx."""
    env = os.environ.get("ROOMWALK_SPLAT_TRANSFORM")
    if env:
        return env.split()
    exe = shutil.which("splat-transform")
    if exe:
        return [exe]
    npx = shutil.which("npx")
    if npx:
        return [npx, "--yes", f"@playcanvas/splat-transform@{SPLAT_TRANSFORM_VERSION}"]
    raise ToolMissing("splat-transform needs Node.js. Install Node 20+ and run: npm i -g @playcanvas/splat-transform")


def build_lod() -> str | None:
    """Spark's LoD builder (optional). Set ROOMWALK_BUILD_LOD or put build-lod on PATH."""
    env = os.environ.get("ROOMWALK_BUILD_LOD")
    if env and Path(env).exists():
        return env
    return shutil.which("build-lod")


def brush() -> str | None:
    """Brush trainer: the v0.3 release ships brush_app, newer builds ship brush."""
    env = os.environ.get("ROOMWALK_BRUSH")
    if env and Path(env).exists():
        return env
    for name in ("brush", "brush_app"):
        exe = shutil.which(name)
        if exe:
            return exe
    return None


def python_exe() -> str:
    return sys.executable
