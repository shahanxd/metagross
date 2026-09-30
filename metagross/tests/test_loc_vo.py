"""Stereo VO on exact synthetic geometry: recovered motion must match the truth."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metagross.autonomy.localization.vo import (
    StereoVO,
    VOConfig,
    coverage_fraction,
    pose_hessian_min_eig,
    rotation_angle,
)
from tests.test_loc_synth import BASELINE_M, K, disparity_from_depth, make_texture, pose, render


@pytest.fixture(scope="module")
def tex() -> np.ndarray:
    return make_texture(3)


def _run(poses: list[np.ndarray], tex: np.ndarray, cfg: VOConfig | None = None):
    vo = StereoVO(K(), BASELINE_M, cfg or VOConfig())
    out = []
    for k, T in enumerate(poses):
        img, depth = render(T, tex)
        out.append(vo.process(img, disparity=disparity_from_depth(depth), t=0.2 * k))
    return out


def test_first_frame_is_init(tex):
    res = _run([np.eye(4)], tex)[0]
    assert not res.ok and res.reason == "init"


def test_forward_motion_recovered(tex):
    poses = [pose(0, 0, 0.3 * k) for k in range(4)]
    results = _run(poses, tex)
    for k in range(1, 4):
        r = results[k]
        assert r.ok, r.reason
        T_true = np.linalg.inv(poses[k - 1]) @ poses[k]
        assert np.linalg.norm(r.T_prev_cur[:3, 3] - T_true[:3, 3]) < 0.02
        assert rotation_angle(r.T_prev_cur[:3, :3].T @ T_true[:3, :3]) < math.radians(0.3)
        s = r.stats
        assert s["inliers"] >= 100 and s["inlier_ratio"] > 0.8
        assert s["reproj_rmse_px"] < 0.8
        assert 0.0 < s["coverage"] <= 1.0 and s["hess_min_eig"] > 0


def test_turning_motion_recovered(tex):
    poses = [pose(0.05 * k, 0, 0.25 * k, math.radians(3.0 * k)) for k in range(4)]
    results = _run(poses, tex)
    for k in range(1, 4):
        r = results[k]
        assert r.ok, r.reason
        T_true = np.linalg.inv(poses[k - 1]) @ poses[k]
        assert np.linalg.norm(r.T_prev_cur[:3, 3] - T_true[:3, 3]) < 0.03
        assert rotation_angle(r.T_prev_cur[:3, :3].T @ T_true[:3, :3]) < math.radians(0.5)


def test_motion_gate_rejects_teleport(tex):
    cfg = VOConfig(max_speed_mps=0.5, gate_slack_m=0.0)  # 0.1 m allowed per 0.2 s frame
    results = _run([pose(0, 0, 0), pose(0, 0, 0.4)], tex, cfg)
    assert not results[1].ok and results[1].reason.startswith("motion_gate")


def test_blank_image_fails_cleanly(tex):
    vo = StereoVO(K(), BASELINE_M)
    img, depth = render(np.eye(4), tex)
    disp = disparity_from_depth(depth)
    vo.process(img, disparity=disp, t=0.0)
    blank = np.full_like(img, 128)
    r = vo.process(blank, disparity=disp, t=0.2)
    assert not r.ok and r.stats["inliers"] == 0


def test_sgbm_path_runs(tex):
    """Without a provided disparity the VO computes SGBM from a real right image."""
    vo = StereoVO(K(), BASELINE_M, VOConfig(num_disparities=64))
    for k in range(3):
        T = pose(0, 0, 0.3 * k)
        left, _ = render(T, tex)
        T_r = T.copy()
        T_r[:3, 3] += T[:3, :3] @ np.array([BASELINE_M, 0, 0])
        right, _ = render(T_r, tex)
        r = vo.process(left, right, t=0.2 * k)
    assert r.ok, r.reason
    assert abs(r.T_prev_cur[2, 3] - 0.3) < 0.05


def test_helpers():
    pts = np.array([[1, 1], [2, 2], [300, 200], [301, 201]], np.float32)
    assert coverage_fraction(pts, 320, 240) == pytest.approx(2 / 16)
    P = np.array([[x, y, z] for x in (-2, 0, 2) for y in (-1, 1) for z in (4.0, 8.0)])
    assert pose_hessian_min_eig(P, 200.0, 200.0) > 0
    assert pose_hessian_min_eig(P[:2], 200.0, 200.0) == 0.0
