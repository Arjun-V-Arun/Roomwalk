"""Structure from motion: where was the phone for each frame?

Gaussian splat training needs a camera pose for every image plus a sparse
point cloud to start from. COLMAP finds both by matching SIFT features
between frames and solving for geometry.

Matching strategy matters for rooms. A room scan is a loop: the last frames
look at the same wall as the first. Sequential matching only compares nearby
frames, so it never sees that the loop closes and the two ends drift apart.
For up to ~160 frames we compare every pair (exhaustive). Above that,
sequential matching with quadratic overlap (frame i against i+1, i+2, i+4,
i+8 ...) keeps the cost down while still linking distant parts of the walk.

Backends: pycolmap (pip install, CPU, default) or the COLMAP command-line
program (faster with a CUDA build). Both write the standard dataset layout:

    dataset/images/frame_00000.jpg ...
    dataset/sparse/0/{cameras,images,points3D}.bin
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .tools import log, run

EXHAUSTIVE_MAX = 160


@dataclass
class SfmResult:
    images: int
    registered: int
    points: int
    mean_reprojection_error: float | None
    model_dir: str
    other_models: int

    @property
    def registered_fraction(self) -> float:
        return self.registered / max(1, self.images)


def _pick_largest(sparse: Path, counts: dict[str, int]) -> tuple[Path, int]:
    """Keep the biggest sub-model as sparse/0. COLMAP splits the scene when it
    cannot connect some frames to the rest; the smaller pieces are set aside."""
    if not counts:
        raise RuntimeError("Structure from motion registered no images. The video may be too blurry or too plain.")
    best = max(counts, key=counts.get)
    others = [k for k in counts if k != best]
    aside = sparse.parent / "sparse_rejected"
    if others:
        aside.mkdir(exist_ok=True)
        for k in others:
            shutil.move(str(sparse / k), str(aside / k))
    if best != "0":
        tmp = sparse / "_best"
        shutil.move(str(sparse / best), str(tmp))
        shutil.move(str(tmp), str(sparse / "0"))
    return sparse / "0", len(others)


def run_pycolmap(dataset: Path, matcher: str, num_images: int) -> SfmResult:
    import pycolmap

    images, db, sparse = dataset / "images", dataset / "database.db", dataset / "sparse"
    db.unlink(missing_ok=True)
    for stale in (sparse, dataset / "sparse_txt", dataset / "sparse_rejected"):
        if stale.exists():
            shutil.rmtree(stale)
    sparse.mkdir(parents=True)

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"  # phone lenses have some distortion; Brush handles OPENCV directly
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = 1600
    extraction.sift.max_num_features = 8192
    log(f"extracting features from {num_images} frames (pycolmap)")
    pycolmap.extract_features(db, images, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=extraction)
    if matcher == "exhaustive":
        log("matching every pair of frames")
        pycolmap.match_exhaustive(db)
    else:
        log("matching nearby frames with quadratic overlap")
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = 15
        pairing.quadratic_overlap = True
        pycolmap.match_sequential(db, pairing_options=pairing)
    log("solving camera poses (incremental mapping)")
    recs = pycolmap.incremental_mapping(db, images, sparse)
    counts = {str(k): r.num_reg_images() for k, r in recs.items()}
    model, others = _pick_largest(sparse, counts)
    rec = pycolmap.Reconstruction(str(model))
    return SfmResult(num_images, rec.num_reg_images(), rec.num_points3D(),
                     float(rec.compute_mean_reprojection_error()), str(model), others)


def run_colmap_cli(dataset: Path, matcher: str, num_images: int, colmap: str) -> SfmResult:
    images, db, sparse = dataset / "images", dataset / "database.db", dataset / "sparse"
    db.unlink(missing_ok=True)
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    steps = [
        [colmap, "feature_extractor", "--database_path", db, "--image_path", images,
         "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "OPENCV"],
        [colmap, "exhaustive_matcher", "--database_path", db] if matcher == "exhaustive" else
        [colmap, "sequential_matcher", "--database_path", db, "--SequentialMatching.overlap", "15",
         "--SequentialMatching.quadratic_overlap", "1"],
        [colmap, "mapper", "--database_path", db, "--image_path", images, "--output_path", sparse],
    ]
    for cmd in steps:
        if run(cmd).returncode != 0:
            raise RuntimeError(f"COLMAP step failed: {cmd[1]}")
    # Convert every sub-model to text in a scratch folder, so we can count
    # images and later align without needing pycolmap.
    scratch = dataset / "_txt"
    if scratch.exists():
        shutil.rmtree(scratch)
    counts: dict[str, int] = {}
    for sub in sorted(p for p in sparse.iterdir() if p.is_dir()):
        txt = scratch / sub.name
        txt.mkdir(parents=True)
        run([colmap, "model_converter", "--input_path", sub, "--output_path", txt, "--output_type", "TXT"], quiet=True)
        lines = [ln for ln in (txt / "images.txt").read_text().splitlines() if ln and not ln.startswith("#")]
        counts[sub.name] = len(lines) // 2
    best = max(counts, key=counts.get) if counts else None
    model, others = _pick_largest(sparse, counts)
    text_model = dataset / "sparse_txt"
    if text_model.exists():
        shutil.rmtree(text_model)
    shutil.move(str(scratch / best), str(text_model))
    shutil.rmtree(scratch)
    pts = sum(1 for ln in (text_model / "points3D.txt").read_text().splitlines() if ln and not ln.startswith("#"))
    return SfmResult(num_images, counts[best], pts, None, str(model), others)


def run_sfm(dataset: str | Path, matcher: str = "auto", colmap_bin: str | None = None) -> SfmResult:
    dataset = Path(dataset)
    num_images = len(list((dataset / "images").glob("*.jpg")))
    if num_images < 20:
        raise RuntimeError(f"Only {num_images} frames in {dataset / 'images'}; need at least 20")
    if matcher == "auto":
        matcher = "exhaustive" if num_images <= EXHAUSTIVE_MAX else "sequential"
    if colmap_bin:
        return run_colmap_cli(dataset, matcher, num_images, colmap_bin)
    try:
        import pycolmap  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install pycolmap (pip install pycolmap) or pass --colmap path/to/colmap") from exc
    return run_pycolmap(dataset, matcher, num_images)


def load_poses(dataset: str | Path):
    """Camera poses and sparse points for alignment, from whichever backend ran."""
    from .align import load_poses_pycolmap, load_poses_text

    dataset = Path(dataset)
    if (dataset / "sparse_txt" / "images.txt").exists():
        return load_poses_text(dataset / "sparse_txt")
    return load_poses_pycolmap(dataset / "sparse" / "0")
