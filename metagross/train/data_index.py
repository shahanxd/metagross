"""File indexing for the GAIA-URJC five-label off-road datasets (no torch dependency).

Two on-disk layouts are recognised (both extracted by ``scripts/download_seg_data.py``):

* **RUGD-5Labels-resized** (``data/rugd5``)::

      <split>/image_res/<seq>_<frame>.png           RGB, 400x320 (W x H)
      <split>/labelids_res/<seq>_<frame>_group5.png  uint8 ids 0..4

* **OFFROAD5** (``data/offroad5``; RUGD + RELLIS-3D + GOOSE merged)::

      <split>/images/<stem>.png
      <split>/masks_id5/<stem>_group5.png   (RUGD, RELLIS-3D)
      <split>/masks_id5/<stem>_group5b.png  (GOOSE / GOOSE-Ex)

Label ids (both layouts): 0 background/sky, 1 obstacle, 2 water/mud,
3 unstable (grass/dirt), 4 stable (concrete/asphalt). 255 is treated as ignore.

Splits are the dataset-provided, scene-disjoint train/val/test splits; file lists
are sorted so indexing is deterministic, and :func:`subset` draws seeded subsets.
Each sample carries a ``source`` tag (``rugd`` | ``rellis`` | ``goose``) so metrics
can be broken down per source dataset, and a ``sequence`` tag (scene name).
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

LOG = logging.getLogger(__name__)

IGNORE_INDEX = 255
IMAGE_EXTS = (".png", ".jpg", ".jpeg")

LAYOUTS: dict[str, tuple[str, str]] = {
    # layout name -> (image dir, mask dir)
    "rugd5": ("image_res", "labelids_res"),
    "offroad5": ("images", "masks_id5"),
}
MASK_SUFFIXES = ("_group5b", "_group5")  # longest first
_TRAILING_FRAME = re.compile(r"[_-]?\d+$")


@dataclass(frozen=True)
class SampleRef:
    """One (image, mask) pair on disk plus provenance tags."""

    image: Path
    mask: Path
    source: str  # "rugd" | "rellis" | "goose"
    sequence: str  # scene / recording name, for grouping

    @property
    def stem(self) -> str:
        return self.image.stem


def detect_layout(split_dir: Path) -> str:
    """Return the layout name whose image/mask folders exist under ``split_dir``."""
    for name, (img_dir, mask_dir) in LAYOUTS.items():
        if (split_dir / img_dir).is_dir() and (split_dir / mask_dir).is_dir():
            return name
    raise FileNotFoundError(f"no known GAIA layout under {split_dir} (expected one of {LAYOUTS})")


def infer_source(image_stem: str, mask_stem: str) -> str:
    """Source dataset of a sample from its file names.

    GOOSE masks carry the ``_group5b`` suffix; RELLIS-3D frames are named
    ``frameNNNNNN-<timestamp>``; everything else is RUGD (``creek_00001``, ``trail-7_...``).
    """
    if mask_stem.endswith("_group5b"):
        return "goose"
    if image_stem.startswith("frame"):
        return "rellis"
    return "rugd"


def infer_sequence(image_stem: str, source: str) -> str:
    """Scene / recording name used for grouping (e.g. ``creek``, ``2022-11-11_aying``)."""
    if source == "goose":
        return image_stem.split("__")[0]
    if source == "rellis":
        return "rellis"
    return _TRAILING_FRAME.sub("", image_stem) or image_stem


def resolve_split_dir(root: Path | str, split: str) -> Path:
    """``root/<split>``, or the same one wrapper folder deeper if the zip carried one.

    Accepts ``root/<split>/<layout dirs>`` (expected), ``root/<wrapper>/<split>/...`` and
    ``root/<split>/<wrapper>/...`` where ``<wrapper>`` is the only sub-folder at that level.
    """
    root = Path(root)
    direct = root / split
    candidates = [direct]
    subdirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))] if root.is_dir() else []
    if len(subdirs) == 1:
        candidates.append(subdirs[0] / split)
    if direct.is_dir():
        inner = [d for d in direct.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]
        if len(inner) == 1:
            candidates.append(inner[0])
    for cand in candidates:
        try:
            detect_layout(cand)
        except FileNotFoundError:
            continue
        if cand != direct:
            LOG.warning("using nested split folder %s", cand)
        return cand
    return direct


def list_samples(root: Path | str, split: str) -> list[SampleRef]:
    """Index ``root/<split>`` (either layout). Missing masks are skipped with a warning."""
    split_dir = resolve_split_dir(root, split)
    layout = detect_layout(split_dir)
    img_dir, mask_dir = (split_dir / d for d in LAYOUTS[layout])
    # One directory listing instead of a stat() per candidate mask name.
    mask_names = {p.name for p in mask_dir.iterdir()}
    out: list[SampleRef] = []
    missing = 0
    for image in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        mask = None
        for suf in MASK_SUFFIXES:
            name = f"{image.stem}{suf}.png"
            if name in mask_names:
                mask = mask_dir / name
                break
        if mask is None:
            missing += 1
            continue
        src = infer_source(image.stem, mask.stem)
        out.append(SampleRef(image=image, mask=mask, source=src, sequence=infer_sequence(image.stem, src)))
    if missing:
        LOG.warning("%s/%s: %d images without masks skipped", root, split, missing)
    LOG.info("indexed %d samples in %s (%s layout)", len(out), split_dir, layout)
    return out


def subset(samples: Sequence[SampleRef], n: int | None, seed: int = 0) -> list[SampleRef]:
    """Deterministic seeded subset of size ``n`` (order preserved); ``None``/``0`` = all."""
    if not n or n >= len(samples):
        return list(samples)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(samples), size=n, replace=False))
    return [samples[i] for i in idx]


def list_fingerprint(samples: Iterable[SampleRef]) -> str:
    """Short SHA-256 of the sorted relative file names (records exactly which files were used)."""
    h = hashlib.sha256()
    for s in samples:
        h.update(f"{s.image.parent.parent.name}/{s.image.name}\n".encode())
    return h.hexdigest()[:16]


def read_rgb(path: Path) -> np.ndarray:
    """Read an image file as (H, W, 3) uint8 RGB."""
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    """Read a single-channel label png as (H, W) uint8 ids."""
    import cv2

    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise IOError(f"cannot read {path}")
    if m.ndim == 3:  # defensive: some exports save labels as 3 identical channels
        m = m[..., 0]
    return m.astype(np.uint8, copy=False)
