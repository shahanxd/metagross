"""Compositor + cards: 1920x1080 frames, colour key, BEV projection geometry."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from metagross.autonomy.link import codec
from metagross.contracts.messages import CELL_COLORS, CellState
from tests.test_video_replay import _make_run
from video import cards, layout, style
from video.demo_fake import DemoChaseRenderer, build_demo_run
from video.replay import RunReplay

cv2.setNumThreads(2)


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_demo_run(tmp_path_factory.mktemp("demo") / "run", max_t=1.2)


def _is_frame(img: np.ndarray) -> None:
    assert img.shape == (layout.H, layout.W, 3) and img.dtype == np.uint8


def test_dashboard_frame_from_minimal_run(tmp_path: Path) -> None:
    rp = RunReplay(_make_run(tmp_path))
    dash = layout.Dashboard(gt_at=lambda t: rp.gt_pose_at(t)[:3])
    img = dash.compose(rp.frame_at(1.5), caption=layout.Caption("Title", "Sub"), subtitle="A subtitle.")
    _is_frame(img)
    # top bar is the panel colour, the page background elsewhere
    assert tuple(img[5, 960]) == style.PANEL
    assert tuple(img[layout.FOOTER_Y0 - 10, 5]) == style.BG


def _has_colour(region: np.ndarray, rgb: tuple[int, int, int], tol: int = 12) -> bool:
    return bool((np.abs(region.astype(int) - np.array(rgb)).max(axis=-1) <= tol).any())


def test_simulated_chip_and_operator_link_are_always_drawn(tmp_path: Path) -> None:
    rp = RunReplay(_make_run(tmp_path))
    dash = layout.Dashboard(gt_at=lambda t: rp.gt_pose_at(t)[:3])
    for t in (0.0, 5.0, 11.9):  # before, during and after the debug ticks (telemetry only at the end)
        img = dash.compose(rp.frame_at(t))
        assert _has_colour(img[:layout.TOPBAR_H, layout.W - 420:], style.SIMULATED_RGB)  # top-bar chip
        assert _has_colour(img[layout.FOOTER_Y0:, :500], style.SIMULATED_RGB)  # footer chip
        x0, y0, x1, y1 = layout.LINK
        assert _has_colour(img[y0:y1, x0:x1], CELL_COLORS[CellState.GROUND], tol=2)  # the costmap packet is shown
    split = layout.compose_split(dash, rp.frame_at(2.0), dash, rp.frame_at(2.0), "A", "B")
    assert _has_colour(split[:layout.TOPBAR_H, layout.W - 420:], style.SIMULATED_RGB)


def test_dashboard_frame_with_images_and_chase(demo_run: Path) -> None:
    rp = RunReplay(demo_run, chase_renderer=DemoChaseRenderer(), chase_size=layout.CHASE_SIZE)
    dash = layout.Dashboard(gt_at=lambda t: rp.gt_pose_at(t)[:3])
    f = rp.frame_at(1.0)
    assert f.left_rgb is not None and f.chase_rgb is not None and f.mission.demo_fake
    img = dash.compose(f)
    _is_frame(img)
    x0, y0, x1, y1 = layout.CHASE
    assert img[y0 + 200:y1 - 200, x0 + 200:x1 - 200].std() > 5.0  # chase panel carries an image


def test_split_screen_frame(tmp_path: Path) -> None:
    rp = RunReplay(_make_run(tmp_path))
    da, db = layout.Dashboard("TYPICAL"), layout.Dashboard("FULL")
    img = layout.compose_split(da, rp.frame_at(1.0), db, rp.frame_at(1.0), "TYPICAL STACK", "METAGROSS")
    _is_frame(img)


def test_bev_inverse_affine_matches_forward_projection() -> None:
    view = layout.BevView((3.0, -2.0, 0.7), 300.0, 200.0, 26.0)
    origin, res = (-10.0, -12.0), 0.2
    M = view.inverse_affine_to_grid(origin, res)
    for col, row in ((10, 20), (55, 3), (70, 64)):
        centre = np.array([[origin[0] + (col + 0.5) * res, origin[1] + (row + 0.5) * res]])
        u, v = view.a_to_px(centre)[0]
        back = M @ np.array([u, v, 1.0])
        np.testing.assert_allclose(back, [col, row], atol=1e-6)


def test_track_up_convention() -> None:
    view = layout.BevView((0.0, 0.0, math.pi / 2), 100.0, 100.0, 10.0)  # heading along +y_A
    ahead = view.a_to_px(np.array([[0.0, 1.0]]))[0]
    left = view.a_to_px(np.array([[-1.0, 0.0]]))[0]
    np.testing.assert_allclose(ahead, [100.0, 90.0], atol=1e-9)  # forward is up
    np.testing.assert_allclose(left, [90.0, 100.0], atol=1e-9)  # left is left


def test_palettes_follow_the_contract_colour_key() -> None:
    lut = style.cell_palette()
    for st, rgb in CELL_COLORS.items():
        assert tuple(lut[int(st)]) == rgb
    u4 = style.u4_palette()
    assert tuple(u4[codec.U4_UNSEEN]) == CELL_COLORS[CellState.UNSEEN]
    assert tuple(u4[codec.U4_GROUND_MIN]) == CELL_COLORS[CellState.GROUND]
    assert tuple(u4[codec.U4_DITCH_CANDIDATE]) == CELL_COLORS[CellState.DITCH_CANDIDATE]
    assert tuple(u4[codec.U4_LETHAL]) == CELL_COLORS[CellState.POSITIVE]
    assert tuple(u4[codec.U4_WATER]) == CELL_COLORS[CellState.WATER]


def test_cards_are_deck_style_frames(tmp_path: Path) -> None:
    chart = tmp_path / "chart.png"
    cv2.imwrite(str(chart), np.full((300, 400, 3), 128, np.uint8))
    for img in (cards.title_card("Title", "Subtitle"),
                cards.evidence_card("1.8", "%", "drift per km", "Tested", chart, bullets=["a", "b"]),
                cards.evidence_card("{placeholder}", "", "missing chart", "Simulated", tmp_path / "nope.png"),
                cards.end_card(lines=["x"])):
        _is_frame(img)
        assert tuple(img[5, 5]) == style.DECK_BG


def test_turbo_marks_invalid_disparity() -> None:
    d = np.array([[0.0, np.nan, 10.0, 48.0]], np.float32)
    rgb = layout.turbo(d, 48.0)
    assert tuple(rgb[0, 0]) == tuple(rgb[0, 1]) != tuple(rgb[0, 2])
