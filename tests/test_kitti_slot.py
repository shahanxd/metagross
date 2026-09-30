"""Slot-sized KITTI drift figure (deck_assets/kitti_figures.py::fig_drift_slot).

Synthetic runs only (no KITTI data needed): the PNG must be exactly 1128 x 652 px (2x a 564 x 326
slide slot) and every visible text, including ticks, legend and the honesty chip, >= 26 px tall
(em size) at that resolution.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.text
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def _kf():
    spec = importlib.util.spec_from_file_location("kitti_figures_under_test", REPO / "deck_assets" / "kitti_figures.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _synthetic():
    runs, curves = {}, {}
    for k, (seq, ff, n) in enumerate((("00", 1101, 3440), ("05", 0, 2761), ("07", 0, 1101))):
        L = list(range(100, 2001 if seq != "07" else 601, 100))
        runs[seq] = {"res": {"first_frame": ff, "frames": n, "path_length_m": 700.0 + 1000.0 * (2 - k)}}
        curves[seq] = {"lengths_m": L, "mean_pct": list(1.0 + 0.5 * np.sin(np.array(L) / 400.0) + 0.3 * k)}
    return runs, curves


def test_slot_png_size_and_min_text(tmp_path):
    kf = _kf()
    from metagross.eval import plot_style as ps

    ps.apply_style()
    runs, curves = _synthetic()
    fig = kf.fig_drift_slot(runs, curves)
    fig.canvas.draw()  # creates tick labels
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    assert len(texts) >= 10
    too_small = [(t.get_text(), t.get_fontsize() * ps.DPI / 72.0) for t in texts
                 if t.get_fontsize() * ps.DPI / 72.0 < kf.SLOT_MIN_TEXT_PX - 1e-6]
    assert not too_small, too_small
    labels = {t.get_text() for t in texts}
    assert any(s.startswith("Tested") for s in labels)  # honesty chip present
    assert any("00*" in s for s in labels)  # legend present, sub-sequence marked
    out = ps.save_fig(fig, "s3_kitti_test", out_dir=tmp_path, formats=("png",))
    with Image.open(out[0]) as im:
        assert im.size == (2 * kf.SLOT_SIZE_SLIDE_PX[0], 2 * kf.SLOT_SIZE_SLIDE_PX[1]) == (1128, 652)


def test_px_to_pt_roundtrip():
    kf = _kf()
    from metagross.eval import plot_style as ps

    assert abs(kf.px_to_pt(26.0) * ps.DPI / 72.0 - 26.0) < 1e-9
