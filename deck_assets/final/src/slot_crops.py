"""Deck slot crops of figures whose full version carries prose the slide caption already holds.

Run from the repo root after the source figures are regenerated:
    python deck_assets/final/src/slot_crops.py

run_map_pair_slot.png  the two map panels of run_map_pair.png (titles, subtitles, maps), without the
                       under-panel text, legend and footnotes; the slide's Fig. 3 caption carries the
                       symbol key, the numbers and the selection disclosure. Same pixels, no resampling.
console_costmap_slot.png  the costmap raster of console_1.png with its callouts and the whole scale bar.
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


SAT_MIN = 60  # max-min channel spread of a costmap cell colour (the page chrome is near-grey)
SAT_ROW_FRAC = 0.5  # a costmap row is mostly coloured cells
CROP_PAD_PX = 6
BAR_LUMA_MAX = 80  # the scale bar is drawn in near-black ink
BAR_SEARCH_PX = 80  # the scale-bar box sits in the raster's bottom rows and may overhang its right edge
BAR_BOX_PAD_PX = 12  # the box's white margin right of the bar's end cap


def scale_bar_right(gray: np.ndarray, x1: int, y1: int) -> int:
    """Rightmost dark pixel of the scale bar in the raster's bottom rows, searching just past column x1."""
    band = gray[max(0, y1 - BAR_SEARCH_PX):y1 + 1, x1:min(gray.shape[1], x1 + BAR_SEARCH_PX)]
    cols = np.where((band < BAR_LUMA_MAX).any(0))[0]
    if not cols.size or cols[0] > 2:
        return x1
    end = int(cols[0])
    for c in cols[1:]:  # the bar runs on from x1; stop at the first gap (the next panel's ink is further right)
        if c - end > 2:
            break
        end = int(c)
    return x1 + end + BAR_BOX_PAD_PX


def console_costmap_slot() -> Path:
    """console_costmap_slot.png: the costmap raster of console_1.png with its numbered callouts.

    The raster is the longest run of rows whose left-half pixels are mostly saturated colour
    (the green mode banner above it is a shorter run); columns likewise. The right edge is widened to
    take in the scale bar, whose box overhangs the raster.
    """
    src = OUT / "console_1.png"
    im = Image.open(src).convert("RGB")
    a = np.asarray(im).astype(int)
    sat = (a.max(2) - a.min(2)) > SAT_MIN
    half = a.shape[1] // 2
    rows = sat[:, :half].mean(1) > SAT_ROW_FRAC
    runs, start = [], None
    for y, on in enumerate(list(rows) + [False]):
        if on and start is None:
            start = y
        elif not on and start is not None:
            runs.append((start, y - 1))
            start = None
    y0, y1 = max(runs, key=lambda r: r[1] - r[0])
    cols = np.where(sat[y0:y1 + 1, :half].mean(0) > SAT_ROW_FRAC)[0]
    x0, x1 = int(cols.min()), int(cols.max())
    x1 = max(x1, scale_bar_right(np.asarray(im.convert("L")).astype(int), x1, y1))
    out = OUT / "console_costmap_slot.png"
    im.crop((max(0, x0 - CROP_PAD_PX), max(0, y0 - CROP_PAD_PX), x1 + 1 + CROP_PAD_PX, y1 + 1 + CROP_PAD_PX)).save(out)
    LOG.info("wrote %s %s (raster x %d..%d, y %d..%d)", out, Image.open(out).size, x0, x1, y0, y1)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_map_pair_slot()
    console_costmap_slot()
