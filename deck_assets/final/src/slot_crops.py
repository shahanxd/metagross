"""Deck slot crops of figures whose full version carries prose the slide caption already holds.

Run from the repo root after the source figures are regenerated:
    python deck_assets/final/src/slot_crops.py

run_map_pair_slot.png  the two map panels of run_map_pair.png (titles, subtitles, maps), without the
                       under-panel text, legend and footnotes; the slide's Fig. 3 caption carries the
                       symbol key, the numbers and the selection disclosure. Same pixels, no resampling.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

LOG = logging.getLogger("slot_crops")
OUT = Path(__file__).resolve().parents[1]
FRAME_LUMA_MAX = 215  # map-panel border rows are darker than this across the panel width
FRAME_MIN_FRAC = 0.9  # fraction of the left panel's width a border row must cover
BORDER_PAD_PX = 4  # keep a few white pixels under the border


def panel_bottom(gray: np.ndarray, x0: int, x1: int, y_from: int) -> int:
    """Last row at or after ``y_from`` that is a panel border across columns x0..x1 (bottom frame line)."""
    rows = [y for y in range(y_from, gray.shape[0]) if (gray[y, x0:x1] < FRAME_LUMA_MAX).mean() > FRAME_MIN_FRAC]
    if not rows:
        raise ValueError("no panel border found")
    first = rows[0]
    run = [y for y in rows if y - first < 6]  # the border is a line a few px thick
    return run[-1]


def run_map_pair_slot() -> Path:
    src = OUT / "run_map_pair.png"
    im = Image.open(src).convert("RGB")
    gray = np.asarray(im.convert("L")).astype(int)
    w = im.size[0]
    # the left panel spans about 1% .. 49% of the width; search below the top third (titles are above)
    y = panel_bottom(gray, int(0.014 * w), int(0.488 * w), gray.shape[0] // 3)
    out = OUT / "run_map_pair_slot.png"
    im.crop((0, 0, w, min(gray.shape[0], y + 1 + BORDER_PAD_PX))).save(out)
    LOG.info("wrote %s %s", out, Image.open(out).size)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_map_pair_slot()
