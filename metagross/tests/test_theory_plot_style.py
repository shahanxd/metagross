"""Tests for the deck chart style (metagross.eval.plot_style) and theory figure export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from metagross.contracts.messages import CELL_COLORS, CellState
from metagross.eval import plot_style as ps
from metagross.eval import theory as T


def test_cell_hex_matches_contract() -> None:
    assert set(ps.CELL_HEX) == set(CELL_COLORS)
    assert ps.CELL_HEX[CellState.GROUND] == "#22A05A"  # (34, 160, 90)
    assert ps.CELL_HEX[CellState.DITCH_CANDIDATE] == "#C828A0"  # (200, 40, 160)
    assert set(ps.LEGEND_ORDER) <= set(ps.CELL_LABELS)


def test_rgb_to_hex_validates() -> None:
    assert ps.rgb_to_hex((0, 0, 0)) == "#000000"
    with pytest.raises(ValueError):
        ps.rgb_to_hex((256, 0, 0))


def test_honesty_labels_complete() -> None:
    assert set(ps.HONESTY_COLORS) == {"Tested", "Simulated", "Estimated", "Proposed", "Literature"}
    with pytest.raises(ValueError):
        ps.honesty_text("Guessed")  # type: ignore[arg-type]


def test_size_inches_gives_2x_png() -> None:
    w_in, h_in = ps.size_inches("card")
    assert (round(w_in * ps.DPI), round(h_in * ps.DPI)) == (900 * ps.EXPORT_SCALE, 520 * ps.EXPORT_SCALE)


def test_style_rc_is_white_and_spineless() -> None:
    rc = ps.style_rc()
    assert rc["figure.facecolor"] == "#FFFFFF" and rc["axes.facecolor"] == "#FFFFFF"
    assert rc["axes.spines.top"] is False and rc["axes.spines.right"] is False
    assert rc["svg.fonttype"] == "none"


def test_save_fig_writes_png_and_svg_at_deck_size(tmp_path: Path) -> None:
    with ps.deck_style():
        fig, ax = ps.new_figure("card")
        ax.plot([0, 1], [0, 1])
        ps.add_honesty_chip(fig, "Simulated", "unit test")
        paths = ps.save_fig(fig, "sub/test_fig", out_dir=tmp_path)
    assert [pth.suffix for pth in paths] == [".png", ".svg"]
    with Image.open(paths[0]) as im:
        assert im.size == (1800, 1040)
        assert im.getpixel((2, 2))[:3] == (255, 255, 255)  # white background
    assert "<svg" in paths[1].read_text(encoding="utf-8")[:500]


def test_cell_state_cmap_indexes_contract_colours() -> None:
    cmap = ps.cell_state_cmap()
    r, g, b, _ = cmap(int(CellState.WATER))
    assert (round(r * 255), round(g * 255), round(b * 255)) == CELL_COLORS[CellState.WATER]


def test_build_all_writes_figures_and_json(tmp_path: Path) -> None:
    out_json = tmp_path / "theory.json"
    s = T.build_all(out_dir=tmp_path, results_json=out_json)
    for name in ("ditch_detectability", "safe_speed_envelope", "stereo_depth_error"):
        assert (tmp_path / "theory" / f"{name}.png").is_file()
        assert (tmp_path / "theory" / f"{name}.svg").is_file()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["label"] == "Estimated" and data["detail"] == "analytic model"
    assert len(s.figures) == 6
