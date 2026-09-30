"""Download KITTI odometry sequences (grayscale stereo + calib + times) and GT poses.

Sources
-------
* Images / calib / times: Hugging Face dataset mirrors ``yujie2696/kitti_odometry_XX``
  (layout: ``image_0/*.png`` = left gray, ``image_1/*.png`` = right gray,
  ``calib.txt``, ``times.txt``).
* Ground-truth poses: ``Huangying-Zhan/kitti-odom-eval`` (GitHub raw), one 3x4
  row-major camera-0 pose per line.

Layout written (git-ignored)::

    data/kitti/sequences/XX/{image_0,image_1,calib.txt,times.txt}
    data/kitti/poses/XX.txt

The download is resumable: ``huggingface_hub.snapshot_download`` skips files already
present in ``local_dir`` and a sequence is marked complete with a ``.complete``
stamp only after every file is present. Sequences are fetched in the order given
(default 07, 05, 00 - 07 first because it is the smallest and is what the VO
evaluation runs on first).

Known mirror defect (found 2026-09-30): in ``yujie2696/kitti_odometry_00`` frames 000000-001100
(image_0 and image_1) are a highway drive, not KITTI 00, and ``times.txt`` has 1101 lines; frames
001101-004540 match the KITTI 00 ground truth. Evaluate 00 with ``--first-frame 1101`` (see
``metagross/eval/kitti_vo.py``) or replace the prefix from the official KITTI archive.

Usage::

    python scripts/download_kitti.py            # 07, 05, 00
    python scripts/download_kitti.py --seqs 07  # one sequence
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import list_repo_files, snapshot_download

LOG = logging.getLogger("download_kitti")

REPO_ROOT = Path(__file__).resolve().parents[1]
KITTI_ROOT = REPO_ROOT / "data" / "kitti"
HF_REPO_FMT = "yujie2696/kitti_odometry_{seq}"
GT_URL_FMT = (
    "https://raw.githubusercontent.com/Huangying-Zhan/kitti-odom-eval/master/"
    "dataset/kitti_odom/gt_poses/{seq}.txt"
)
ALLOW_PATTERNS = ["image_0/*", "image_1/*", "calib.txt", "times.txt"]
DEFAULT_SEQS = ("07", "05", "00")
MAX_WORKERS = 8  # parallel HTTP downloads (I/O bound; does not load the CPU)
RETRIES = 5
HTTP_TIMEOUT_S = 60.0


def download_gt_poses(seq: str, out_dir: Path) -> Path:
    """Fetch the ground-truth pose file for ``seq`` unless it is already present."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{seq}.txt"
    if dst.exists() and dst.stat().st_size > 0:
        LOG.info("GT poses %s already present (%s)", seq, dst)
        return dst
    url = GT_URL_FMT.format(seq=seq)
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, timeout=HTTP_TIMEOUT_S)
            r.raise_for_status()
            tmp = dst.with_suffix(".tmp")
            tmp.write_bytes(r.content)
            tmp.replace(dst)
            LOG.info("GT poses %s -> %s (%d lines)", seq, dst, r.text.count("\n"))
            return dst
        except requests.RequestException as exc:
            LOG.warning("GT %s attempt %d failed: %s", seq, attempt, exc)
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"could not download GT poses for sequence {seq}")


def _sequence_complete(seq_dir: Path, expected: list[str]) -> bool:
    return all((seq_dir / f).exists() for f in expected)


def download_sequence(seq: str, root: Path = KITTI_ROOT) -> Path:
    """Download one sequence (images + calib + times). Resumable and idempotent."""
    seq_dir = root / "sequences" / seq
    stamp = seq_dir / ".complete"
    if stamp.exists():
        LOG.info("sequence %s already complete (%s)", seq, seq_dir)
        return seq_dir
    repo = HF_REPO_FMT.format(seq=seq)
    expected = [f for f in list_repo_files(repo, repo_type="dataset") if f != ".gitattributes"]
    LOG.info("sequence %s: %d files from %s", seq, len(expected), repo)
    for attempt in range(1, RETRIES + 1):
        try:
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                local_dir=str(seq_dir),
                allow_patterns=ALLOW_PATTERNS,
                max_workers=MAX_WORKERS,
            )
            if _sequence_complete(seq_dir, expected):
                stamp.write_text(f"{len(expected)} files\n", encoding="utf-8")
                LOG.info("sequence %s complete", seq)
                return seq_dir
            LOG.warning("sequence %s incomplete after attempt %d; retrying", seq, attempt)
        except Exception as exc:  # network errors surface as several exception types
            LOG.warning("sequence %s attempt %d failed: %s", seq, attempt, exc)
            time.sleep(5.0 * attempt)
    raise RuntimeError(f"could not complete download of sequence {seq}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", nargs="+", default=list(DEFAULT_SEQS), help="sequence ids, in download order")
    ap.add_argument("--root", type=Path, default=KITTI_ROOT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)  # one line per file otherwise
    for seq in args.seqs:
        seq = f"{int(seq):02d}"
        download_gt_poses(seq, args.root / "poses")
        download_sequence(seq, args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
