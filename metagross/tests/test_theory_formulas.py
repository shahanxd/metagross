"""Unit tests for the analytic deck models in metagross.eval.theory (synthetic, fast)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metagross.config import defaults as D
from metagross.eval import theory as T


@pytest.fixture(scope="module")
def p() -> T.TheoryParams:
    return T.TheoryParams()


def test_params_come_from_defaults(p: T.TheoryParams) -> None:
    assert p.fx_px == D.FX
    assert p.n_px == D.MIN_PIXELS_ON_TARGET
    assert p.cam_height_m == D.CAM_HEIGHT_M
    assert p.brake_decel_mps2 == D.BRAKE_DECEL_MPS2
    assert p.margin_m == D.GOVERNOR_MARGIN_M
    assert p.v_platform_mps == D.VEHICLE.max_speed_mps
    assert p.theta_min_rad == pytest.approx(D.MIN_PIXELS_ON_TARGET / D.FX)


@pytest.mark.parametrize("w", [0.1, 0.3, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("h", [0.4, 0.9, 1.6])
def test_ditch_detection_range_solves_threshold(w: float, h: float, p: T.TheoryParams) -> None:
    r = float(T.ditch_detection_range_m(w, h, p.fx_px, p.n_px))
    assert r > 0
    # At R_det the visible angle equals n * IFOV exactly; closer is above, farther below.
    assert float(T.ditch_angle_rad(r, w, h)) == pytest.approx(p.theta_min_rad, rel=1e-12)
    assert float(T.ditch_angle_rad(0.9 * r, w, h)) > p.theta_min_rad
    assert float(T.ditch_angle_rad(1.1 * r, w, h)) < p.theta_min_rad


def test_ditch_detection_known_value(p: T.TheoryParams) -> None:
    # Hand calculation: k = 0.9*0.3*440.39/6 = 19.818; R = (-0.3 + sqrt(0.09 + 79.27)) / 2 = 4.304 m
    r = float(T.ditch_detection_range_m(0.3, 0.9, p.fx_px, p.n_px))
    assert r == pytest.approx(4.304, abs=2e-3)


def test_ditch_angle_falls_as_inverse_square_rock_as_inverse() -> None:
    r = np.array([10.0, 20.0, 40.0])
    ditch = T.ditch_angle_rad(r, 0.3, 0.9)
    rock = T.positive_angle_rad(r, 0.3)
    # Doubling range: ditch angle drops ~4x (R >> w), rock angle exactly 2x.
    assert ditch[0] / ditch[1] == pytest.approx(4.0, rel=0.05)
    assert ditch[1] / ditch[2] == pytest.approx(4.0, rel=0.02)
    assert rock[0] / rock[1] == pytest.approx(2.0)


def test_exact_angle_matches_approximation_far_away() -> None:
    r = np.array([5.0, 10.0, 20.0])
    approx = T.ditch_angle_rad(r, 0.5, 0.9)
    exact = T.ditch_angle_exact_rad(r, 0.5, 0.9)
    rel = np.abs(exact - approx) / exact
    assert np.all(rel < 0.04)  # small-angle error < 4 % beyond 5 m for a 0.9 m mast
    assert rel[2] < rel[0]  # and shrinks with range


def test_positive_detection_range(p: T.TheoryParams) -> None:
    r = T.positive_detection_range_m(0.3, p.fx_px, p.n_px)
    assert r == pytest.approx(0.3 * p.fx_px / p.n_px)
    assert float(T.positive_angle_rad(r, 0.3)) == pytest.approx(p.theta_min_rad)


@pytest.mark.parametrize("v", [0.0, 0.5, 1.0, 2.0, 3.3])
def test_max_safe_speed_inverts_stopping_distance(v: float, p: T.TheoryParams) -> None:
    d = float(T.stopping_distance_m(v, p.brake_decel_mps2, p.t_reaction_s, p.margin_m))
    v_back = float(T.max_safe_speed_mps(d, p.brake_decel_mps2, p.t_reaction_s, p.margin_m))
    assert v_back == pytest.approx(v, abs=1e-9)


def test_stopping_distance_values(p: T.TheoryParams) -> None:
    # a = 1.5, T_r = 0.6, B = 0.5: 1 m/s -> 1/3 + 0.6 + 0.5; 2 m/s -> 4/3 + 1.2 + 0.5
    d = T.stopping_distance_m(np.array([1.0, 2.0]), 1.5, 0.6, 0.5)
    np.testing.assert_allclose(d, [1.0 / 3.0 + 1.1, 4.0 / 3.0 + 1.7])


def test_max_safe_speed_zero_inside_margin() -> None:
    v = T.max_safe_speed_mps(np.array([0.0, 0.3, 0.5]), 1.5, 0.6, 0.5)
    np.testing.assert_array_equal(v, [0.0, 0.0, 0.0])


def test_envelope_monotone_and_shape(p: T.TheoryParams) -> None:
    h = np.linspace(0.4, 1.6, 7)
    w = np.linspace(0.2, 1.2, 5)
    env = T.safe_speed_envelope(h, w, p)
    assert env.shape == (7, 5)
    assert np.all(np.diff(env, axis=0) > 0)  # taller mast -> faster
    assert np.all(np.diff(env, axis=1) > 0)  # wider design ditch -> seen earlier -> faster


def test_min_mast_height_inverts_envelope(p: T.TheoryParams) -> None:
    for v in (1.0, 2.0):
        h = T.min_mast_height_m(0.3, v, p)
        v_at = float(T.safe_speed_envelope(np.array([h]), np.array([0.3]), p)[0, 0])
        assert v_at == pytest.approx(v, rel=1e-9)


def test_stereo_sigma(p: T.TheoryParams) -> None:
    s = T.stereo_depth_sigma_m(np.array([1.0, 2.0, 4.0]), p.fx_px, p.baseline_m, 0.25)
    assert s[0] == pytest.approx(0.25 / (p.fx_px * p.baseline_m))
    np.testing.assert_allclose(s[1:] / s[:-1], [4.0, 4.0])  # quadratic in Z
    # Halving the baseline doubles the error.
    s_half = T.stereo_depth_sigma_m(4.0, p.fx_px, p.baseline_m / 2.0, 0.25)
    assert float(s_half) == pytest.approx(2.0 * s[2])


def test_agrees_with_live_governor(p: T.TheoryParams) -> None:
    """The deck model and the onboard governor must implement the same closed forms."""
    gov = pytest.importorskip("metagross.autonomy.planning.governor")
    for w in (0.3, 0.8):
        assert float(T.ditch_detection_range_m(w, p.cam_height_m, p.fx_px, p.n_px)) == pytest.approx(
            gov.ditch_detection_range(p.cam_height_m, w, p.fx_px, p.n_px))
    for r in (0.2, 1.5, 4.3, 9.0):
        assert float(T.max_safe_speed_mps(r, 1.5, 0.6, 0.5)) == pytest.approx(gov.v_from_range(r, 1.5, 0.6, 0.5))


def test_summary_contents(p: T.TheoryParams) -> None:
    s = T.compute_summary(p)
    assert s.label == "Estimated"
    widths = [d["width_m"] for d in s.ditch_detection]
    assert p.design_ditch_w_m in widths
    design = s.ditch_detection[widths.index(p.design_ditch_w_m)]
    assert design["r_det_m"] == pytest.approx(4.304, abs=2e-3)
    assert s.envelope["v_ours_capped_mps"] == pytest.approx(min(s.envelope["v_ours_mps"], p.v_platform_mps))
    assert s.envelope["frac_grid_below_bel_rsp"] == 0.0  # 1.0 m/s is inside the whole plotted envelope
    assert all(c["label"] == "Estimated" for c in s.claims)
    assert math.isfinite(s.stereo[-1]["sigma_z_m"])
