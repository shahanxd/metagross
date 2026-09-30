"""KITTI-protocol trajectory metrics against trajectories with known error."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metagross.eval.traj_metrics import (
    ate_rmse,
    evaluate,
    kitti_errors,
    load_kitti_poses,
    save_kitti_poses,
    segment_drift,
    trajectory_distances,
)


def straight(n: int, step_m: float) -> np.ndarray:
    P = np.tile(np.eye(4), (n, 1, 1))
    P[:, 2, 3] = np.arange(n) * step_m  # forward along camera z
    return P


def yaw_drift(n: int, step_m: float, rad_per_frame: float) -> np.ndarray:
    """Integrate forward steps with a constant per-frame heading change about camera y."""
    P = [np.eye(4)]
    c, s = math.cos(rad_per_frame), math.sin(rad_per_frame)
    inc = np.eye(4)
    inc[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    inc[2, 3] = step_m
    for _ in range(n - 1):
        P.append(P[-1] @ inc)
    return np.stack(P)


def test_perfect_estimate_has_zero_error():
    gt = straight(1200, 1.0)
    res = evaluate(gt, gt.copy())
    assert res["t_err_pct"] == pytest.approx(0.0, abs=1e-9)
    assert res["r_err_deg_per_100m"] == pytest.approx(0.0, abs=1e-6)
    assert res["ate_rmse_m"] == pytest.approx(0.0, abs=1e-9)


def test_scale_error_gives_exact_t_err():
    gt = straight(1200, 1.0)
    est = gt.copy()
    est[:, :3, 3] *= 1.02  # 2 % too long
    e = kitti_errors(gt, est)
    # KITTI protocol: a segment ends at the first frame *beyond* L (L + 1 m here with 1 m
    # steps) but is normalised by L, so each segment reads 2 % * (L + 1) / L.
    seg_lengths = [L for L in range(100, 900, 100) for first in range(0, 1200, 10) if first + L + 1 < 1200]
    expected = 2.0 * np.mean([(L + 1) / L for L in seg_lengths])
    assert e["t_err_pct"] == pytest.approx(expected, rel=1e-9)
    assert e["r_err_deg_per_100m"] == pytest.approx(0.0, abs=1e-6)
    d = segment_drift(gt, est)
    assert d["drift_100m_m"] == pytest.approx(0.02 * 101, rel=1e-9)
    assert d["drift_1000m_m"] == pytest.approx(0.02 * 1001, rel=1e-9)
    a = ate_rmse(gt, est)
    assert a["final_err_m"] == pytest.approx(0.02 * 1199, rel=1e-6)


def test_rotation_drift_gives_expected_r_err():
    rate = math.radians(0.01)  # 0.01 deg per 1 m frame -> 1 deg / 100 m
    gt = straight(900, 1.0)
    est = yaw_drift(900, 1.0, rate)
    e = kitti_errors(gt, est)
    assert e["r_err_deg_per_100m"] == pytest.approx(1.0, rel=0.02)


def test_short_trajectory_reports_none():
    gt = straight(50, 1.0)
    d = segment_drift(gt, gt)
    assert d["drift_100m_m"] is None and d["drift_100m_n"] == 0
    assert math.isnan(kitti_errors(gt, gt)["t_err_pct"])


def test_io_roundtrip(tmp_path):
    gt = yaw_drift(20, 0.5, 0.01)
    save_kitti_poses(tmp_path / "p.txt", gt)
    back = load_kitti_poses(tmp_path / "p.txt")
    assert np.allclose(back, gt, atol=1e-8)
    assert trajectory_distances(gt)[-1] == pytest.approx(19 * 0.5, rel=1e-6)
