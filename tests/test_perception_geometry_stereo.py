"""Perception: camera geometry, stereo matcher and tier-0 disparity handling."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from metagross.autonomy.perception import synthetic as syn
from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.autonomy.perception.stereo import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    SGBMParams,
    StereoMatcher,
    disparity_for_frame,
    make_clahe,
    resample_disparity,
)
from metagross.config import defaults
from metagross.contracts.messages import SensorFrame

cv2.setNumThreads(2)


@pytest.fixture(scope="module")
def geom() -> CameraGeometry:
    return CameraGeometry(defaults.stereo_calibration())


@pytest.fixture(scope="module")
def flat_pair(geom):
    return syn.render_stereo_pair(geom, syn.make_scene("flat"), seed=5)


def test_flat_ground_backprojects_to_zero_height(geom):
    d = syn.render_disparity(geom, syn.make_scene("flat"))
    pts = geom.points_from_disparity(d, 2, 2)
    assert len(pts) > 10000
    assert np.max(np.abs(pts.z)) < 0.01  # m
    assert np.max(pts.zc) <= geom.max_range_m + 1e-6


def test_projection_roundtrip(geom):
    rng = np.random.default_rng(0)
    p = np.stack([rng.uniform(2, 10, 50), rng.uniform(-2, 2, 50), rng.uniform(-0.5, 0.5, 50)], axis=1)
    u, v, zc = geom.project_body(p)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok = (ui >= 0) & (ui < geom.width) & (vi >= 0) & (vi < geom.height)
    back = geom.t_bc + zc[ok, None] * (geom.R_bc @ np.stack([(u[ok] - geom.cx) / geom.fx, (v[ok] - geom.cy) / geom.fy, np.ones(ok.sum())])).T
    assert np.allclose(back, p[ok], atol=1e-6)


def test_flat_horizon_row_matches_pitch(geom):
    zc = geom.flat_ground_depth()
    first = np.nonzero(np.isfinite(zc).any(axis=1))[0][0]
    expected = geom.cy - geom.fy * np.tan(np.radians(-defaults.CAM_PITCH_DEG))
    assert abs(first - expected) <= 1.0


def test_sgbm_accuracy_on_synthetic_pair(geom, flat_pair):
    left, right, truth = flat_pair
    m = StereoMatcher()
    d = m.compute(left, right)
    assert d.dtype == np.float32 and d.shape == truth.shape
    roi = (truth > geom.min_disparity_px)
    roi[:, : m.params.num_disparities] = False  # SGBM cannot match the left margin
    valid = roi & (d > 0)
    assert valid.sum() / roi.sum() > 0.9
    assert np.median(np.abs(d[valid] - truth[valid])) < 0.3  # px
    assert m.last_ms > 0


def test_lr_check_only_removes_pixels(flat_pair):
    left, right, _ = flat_pair
    d0 = StereoMatcher(SGBMParams(lr_check=False)).compute(left, right)
    d1 = StereoMatcher(SGBMParams(lr_check=True)).compute(left, right)
    assert (d1 > 0).sum() <= (d0 > 0).sum()
    both = (d1 > 0) & (d0 > 0)
    assert np.allclose(d1[both], d0[both])


def test_shared_clahe_parameters():
    c = make_clahe()
    assert c.getClipLimit() == CLAHE_CLIP_LIMIT
    assert tuple(c.getTilesGridSize()) == CLAHE_TILE_GRID


def test_tier0_half_resolution_is_rescaled(geom):
    full = syn.render_disparity(geom, syn.make_scene("flat"))
    half = cv2.resize(full, (geom.width // 2, geom.height // 2), interpolation=cv2.INTER_NEAREST)
    half = np.where(half > 0, half / 2.0, -1.0).astype(np.float32)  # disparity in half-res pixels
    up = resample_disparity(half, geom.width, geom.height)
    both = (up > 0) & (full > 0)
    assert np.median(np.abs(up[both] - full[both])) < 0.3
    assert np.all(up[half.repeat(2, 0).repeat(2, 1) <= 0] <= 0)
    frame = SensorFrame(0.0, 0, None, None, 0.0, 0.0, 0.0, sensor_mode="tier0_disparity", disparity=half)
    d, ms = disparity_for_frame(frame, defaults.stereo_calibration(), StereoMatcher())
    assert d.shape == (geom.height, geom.width) and ms >= 0.0


def test_frame_without_data_raises():
    frame = SensorFrame(0.0, 0, None, None, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        disparity_for_frame(frame, defaults.stereo_calibration(), StereoMatcher())
