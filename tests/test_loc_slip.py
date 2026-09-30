"""Slip ratio, IMMOBILISED latch and online chi estimation."""

from __future__ import annotations

import pytest

from metagross.autonomy.localization.slip import SlipConfig, SlipEstimator

R_WHEEL, TRACK = 0.13, 0.50
DT = 0.1


def tick(est: SlipEstimator, t: float, v_wheel: float, v_true: float, w_true: float = 0.0, chi: float = 1.4,
         vo: bool = True, commanded=None):
    d_wheel = v_wheel * DT
    dyaw_wd = w_true * DT * chi  # wheel-differential yaw without chi = chi * true yaw
    return est.update(t, d_wheel, dyaw_wd, DT, v_true * DT if vo else None, w_true * DT if vo else None, commanded)


def test_no_slip_rolling():
    est = SlipEstimator()
    for k in range(40):
        out = tick(est, k * DT, 1.0, 1.0)
    assert out["slip"] == pytest.approx(0.0, abs=1e-9) and not out["immobilised"]


def test_partial_slip_ratio():
    est = SlipEstimator()
    for k in range(40):
        out = tick(est, k * DT, 1.0, 0.7)
    assert out["slip"] == pytest.approx(0.3, abs=1e-6)


def test_immobilised_after_hold_time():
    est = SlipEstimator(SlipConfig(window_s=2.0, immobilised_hold_s=3.0))
    flags = []
    for k in range(60):  # wheels spin at 1 m/s, vehicle does not move
        flags.append(bool(tick(est, k * DT, 1.0, 0.0)["immobilised"]))
    first = flags.index(True)
    assert 29 <= first <= 31  # 3.0 s after slip first exceeded 0.6
    # recovers as soon as the vehicle moves again and the window refills
    for k in range(60, 100):
        out = tick(est, k * DT, 1.0, 1.0)
    assert not out["immobilised"]


def test_not_immobilised_when_not_commanded():
    est = SlipEstimator()
    for k in range(60):
        out = tick(est, k * DT, 1.0, 0.0, commanded=False)
    assert not out["immobilised"]


@pytest.mark.parametrize("chi_true", [1.2, 1.7])
def test_chi_converges(chi_true):
    est = SlipEstimator(chi0=1.4)
    for k in range(300):
        out = tick(est, k * DT, 0.5, 0.5, w_true=0.4, chi=chi_true)
    assert out["chi_hat"] == pytest.approx(chi_true, abs=0.01)


def test_chi_clipped_and_ignored_without_turning():
    est = SlipEstimator(chi0=1.4)
    for k in range(50):  # straight line: chi unobservable, stays at prior
        out = tick(est, k * DT, 1.0, 1.0, w_true=0.0)
    assert out["chi_hat"] == pytest.approx(1.4)
    for k in range(50, 400):  # ratio 2.6 is inside the outlier clip but above chi_max
        out = tick(est, k * DT, 0.5, 0.5, w_true=0.4, chi=2.6)
    assert out["chi_hat"] == pytest.approx(2.0)
