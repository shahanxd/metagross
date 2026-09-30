"""Download the GAIA-URJC 5-label off-road segmentation mirrors from Hugging Face.

Datasets (both CC BY-NC-SA 3.0, research use only; see docs/MODEL_CARD.md):

* ``rugd5``    -> ``GAIA-URJC/RUGD-5Labels-resized`` (~2.06 GB; train/val/test zips,
                  400x320 images, single-channel masks with ids 0..4).
* ``offroad5`` -> ``GAIA-URJC/OFFROAD5`` (~8.5 GB; RUGD + RELLIS-3D + GOOSE merged into
                  the same 5 labels). Intended for the AWS GPU box only.

Downloads are resumable (``huggingface_hub`` keeps partial blobs in its cache and
resumes on re-run) and each split is unzipped exactly once: a ``.unzipped`` marker
file with the zip's size is written after a successful extraction, so re-running the
script is cheap and idempotent.

Usage::

    python scripts/download_seg_data.py --dataset rugd5 --splits val test train
    python scripts/download_seg_data.py --dataset offroad5 --root /data/seg
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
import zipfile
from pathlib import Path

LOG = logging.getLogger("download_seg_data")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data"

DATASETS: dict[str, dict[str, str]] = {
    "rugd5": {"repo_id": "GAIA-URJC/RUGD-5Labels-resized", "subdir": "rugd5"},
    "offroad5": {"repo_id": "GAIA-URJC/OFFROAD5", "subdir": "offroad5"},
}
ALL_SPLITS = ("val", "test", "train")
MAX_RETRIES = 5  # network retries per zip (download resumes from the partial blob)
RETRY_SLEEP_S = 10.0

# Zero-shot research baseline: SegFormer-B0 fine-tuned on ADE20K (NVIDIA checkpoint,
# NVIDIA Source Code License - non-commercial), ONNX export by Xenova on the HF hub.
ZERO_SHOT_REPO = "Xenova/segformer-b0-finetuned-ade-512-512"
ZERO_SHOT_FILE = "onnx/model.onnx"
ZERO_SHOT_NAME = "segformer_b0_ade"
ZERO_SHOT_INPUT_HW = (320, 416)  # same input as the deploy LR-ASPP; dynamic axes allow any /32 size
ZERO_SHOT_SIDECAR = {
    "name": "SegFormer-B0 ADE20K (zero-shot, mapped to 5 classes)",
    "kind": "ade150",
    "input_hw": list(ZERO_SHOT_INPUT_HW),
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "output": "logits (1, 150, H/4, W/4), ADE20K SceneParse150 ids",
    "classes": {"0": "background/sky", "1": "obstacle", "2": "water/mud", "3": "unstable (grass/dirt)", "4": "stable (asphalt/concrete/gravel path)"},
    "class_mapping": "metagross.autonomy.perception.semantics.ADE20K_TO_SEM5 (unlisted ADE ids -> obstacle)",
    "source": f"https://huggingface.co/{ZERO_SHOT_REPO} ({ZERO_SHOT_FILE}); weights nvidia/segformer-b0-finetuned-ade-512-512",
    "licence": "NVIDIA Source Code License for SegFormer (non-commercial, research/evaluation only). NOT for deployment.",
    "training_data": "ADE20K (no off-road training); used zero-shot as a research baseline",
}


def download_zero_shot_model(models_dir: Path) -> Path:
    """Fetch the SegFormer-B0-ADE ONNX into ``models/segformer_b0_ade.onnx`` + sidecar JSON."""
    import json

    from huggingface_hub import hf_hub_download

    models_dir.mkdir(parents=True, exist_ok=True)
    dst = models_dir / f"{ZERO_SHOT_NAME}.onnx"
    if not dst.exists():
        src = Path(hf_hub_download(repo_id=ZERO_SHOT_REPO, filename=ZERO_SHOT_FILE))
        shutil.copyfile(src, dst)
    (models_dir / f"{ZERO_SHOT_NAME}.json").write_text(json.dumps(ZERO_SHOT_SIDECAR, indent=1), encoding="utf-8")
    LOG.info("zero-shot model at %s (%.1f MB)", dst, dst.stat().st_size / 1e6)
    return dst


def _download_zip(repo_id: str, split: str, zip_dir: Path) -> Path:
    """Fetch ``<split>.zip`` into ``zip_dir`` with retries; returns the local path."""
    from huggingface_hub import hf_hub_download  # local import: optional dependency at train time

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{split}.zip",
                repo_type="dataset",
                local_dir=str(zip_dir),
            )
            return Path(path)
        except Exception as exc:  # network errors are retried; hf_hub resumes partial files
            last_exc = exc
            LOG.warning("download %s/%s.zip attempt %d/%d failed: %s", repo_id, split, attempt, MAX_RETRIES, exc)
            time.sleep(RETRY_SLEEP_S)
    raise RuntimeError(f"giving up on {repo_id}/{split}.zip") from last_exc


def _unzip_once(zip_path: Path, out_dir: Path) -> None:
    """Extract ``zip_path`` into ``out_dir`` unless a matching marker says it is done."""
    marker = out_dir / f".{zip_path.stem}.unzipped"
    size = str(zip_path.stat().st_size)
    if marker.exists() and marker.read_text().strip() == size:
        LOG.info("%s already extracted (marker present)", zip_path.name)
        return
    LOG.info("extracting %s -> %s", zip_path.name, out_dir)
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    marker.write_text(size)
    LOG.info("extracted %s in %.1f s", zip_path.name, time.time() - t0)


def download(dataset: str, splits: list[str], root: Path, keep_zip: bool = True) -> Path:
    """Download + extract the requested splits. Returns the dataset directory."""
    spec = DATASETS[dataset]
    out_dir = root / spec["subdir"]
    zip_dir = out_dir / "_zips"
    zip_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        t0 = time.time()
        LOG.info("fetching %s %s.zip", spec["repo_id"], split)
        zpath = _download_zip(spec["repo_id"], split, zip_dir)
        LOG.info("have %s (%.1f MB) after %.0f s", zpath.name, zpath.stat().st_size / 1e6, time.time() - t0)
        _unzip_once(zpath, out_dir)
        if not keep_zip:
            zpath.unlink(missing_ok=True)
    cache = zip_dir / ".cache"
    if not keep_zip and cache.exists():
        shutil.rmtree(cache, ignore_errors=True)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="rugd5")
    ap.add_argument("--splits", nargs="+", choices=ALL_SPLITS, default=list(ALL_SPLITS))
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="parent directory (default: <repo>/data)")
    ap.add_argument("--delete-zips", action="store_true", help="remove zips after extraction to save disk")
    ap.add_argument("--zero-shot-model", action="store_true", help="also fetch the SegFormer-B0-ADE ONNX baseline into models/")
    ap.add_argument("--models-only", action="store_true", help="only fetch the zero-shot model, no dataset")
    ap.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.zero_shot_model or args.models_only:
        download_zero_shot_model(args.models_dir)
    if args.models_only:
        return 0
    out = download(args.dataset, list(args.splits), args.root, keep_zip=not args.delete_zips)
    LOG.info("done: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
