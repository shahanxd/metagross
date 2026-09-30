"""Perception: ground model (v-disparity + continuity-checked RANSAC bands) and BEV grid."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metagross.autonomy.perception import synthetic as syn
from metagross.autonomy.perception.bev import BevSpec, accumulate, expected_coverage, visible_range
from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.autonomy.perception.ground import fit_ground
from metagross.config import defaults
from metagross.contracts.messages import CellState, StereoCalibration


def _geom(pitch_deg: float = defaults.CAM_PITCH_DEG) -> CameraGeometry:
    c = defaults.stereo_calibration()
    T = c.T_body_cam.copy()
    p = math.radians(-pitch_deg)
    R_level = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    R_pitch = np.array([[math.cos(p), 0.0, math.sin(p)], [0.0, 1.0, 0.0], [-math.sin(p), 0.0, math.cos(p)]])
    T[:3, :3] = R_pitch @ R_level
    return CameraGeometry(StereoCalibration(c.width, c.height, c.fx, c.fy, c.cx, c.cy, c.baseline_m, T))


def _fit(geom: CameraGeometry, scene: syn.Scene, noise: float = 0.1):
    d = syn.render_disparity(geom, scene, noise_px=noise, seed=2)
    pts = geom.points_from_disparity(d, 2, 4)
    return fit_ground(pts, d, geom, seed=0), d


@pytest.mark.parametrize(
    "scene,pitch,tol",
    [
        (syn.Scene(name="flat"), defaults.CAM_PITCH_DEG, 0.01),
        (syn.Scene(base=syn.Plane(pitch_deg=4.0, roll_deg=3.0)), defaults.CAM_PITCH_DEG, 0.02),
        (syn.Scene(base=syn.Plane(pitch_deg=-5.0)), defaults.CAM_PITCH_DEG, 0.02),
        (syn.Scene(name="flat"), -16.0, 0.01),  # camera pitched further down
        (syn.Scene(name="flat"), -8.0, 0.02),  # camera pitched up
        (syn.Scene(base=syn.Rolling(0.15, 8.0)), defaults.CAM_PITCH_DEG, 0.04),
    ],
)
def test_ground_height_on_tilted_pitched_and_rolling_terrain(scene, pitch, tol):
    geom = _geom(pitch)
    model, _ = _fit(geom, scene)
    xs, ys = np.meshgrid(np.linspace(2.0, 11.0, 19), np.linspace(-1.5, 1.5, 7))
    err = model.height(xs, ys) - scene(xs, ys)
    assert np.max(np.abs(err)) < tol, np.round(err, 3)
    assert model.n_fitted >= 5


def test_v_disparity_horizon_follows_pitch():
    for pitch in (-8.0, -12.0, -16.0):
        geom = _geom(pitch)
        model, _ = _fit(geom, syn.Scene(name="flat"), noise=0.05)
        expected = geom.cy - geom.fy * math.tan(math.radians(-pitch))
        assert model.vdisp is not None
        assert abs(model.vdisp.horizon_row - expected) < 2.0


def test_expected_disparity_matches_rendered_ground():
    geom = _geom()
    scene = syn.Scene(base=syn.Plane(pitch_deg=3.0, roll_deg=-2.0))
    model, d_clean = _fit(geom, scene, noise=0.1)
    d_true = syn.render_disparity(geom, scene)
    v, u = np.meshgrid(np.arange(150, 400, 7), np.arange(0, 640, 11), indexing="ij")
    de = model.expected_disparity(geom, v, u)
    ok = d_true[v, u] > geom.min_disparity_px
    assert np.percentile(np.abs(de[ok] - d_true[v, u][ok]), 95) < 0.1


def test_crest_far_bands_are_not_accepted_as_ground():
    geom = _geom()
    model, _ = _fit(geom, syn.make_scene("crest"))
    # ground beyond the drop-off is not connected: model extrapolates the lip plane
    assert abs(float(model.height(np.array(9.0), np.array(0.0)))) < 0.05
    assert not model.bands[-1].fitted


def test_bev_accumulate_statistics():
    spec = BevSpec()
    x = np.array([1.01, 1.02, 1.03, 5.55])
    y = np.array([0.01, 0.02, 0.03, -0.45])
    z = np.array([0.0, 0.1, 0.2, -0.3])
    h = z.copy()
    st = accumulate(spec, x, y, z, h)
    i = int((1.02 - spec.x_min) / spec.res)
    j = int((0.02 - spec.y_min) / spec.res)
    assert st.count[i, j] == 3
    assert st.h_min[i, j] == pytest.approx(0.0) and st.h_max[i, j] == pytest.approx(0.2)
    assert st.z_mean[i, j] == pytest.approx(0.1)
    assert st.count.sum() == 4
    assert np.isnan(st.h_min[0, 0])


def test_expected_coverage_lut():
    geom = _geom()
    spec = BevSpec()
    px, rows = expected_coverage(geom, spec)
    X, Y = spec.centres()
    assert np.all(px[X < 1.0] == 0)  # behind / under the camera
    centre = np.abs(Y) < 0.05
    near = px[centre & (np.abs(X - 3.0) < 0.05)].mean()
    far = px[centre & (np.abs(X - 9.0) < 0.05)].mean()
    assert near > far > 0
    # flat-ground rows per cell ~ fy * H * res / R^2 (small-angle), within 25 %
    r = 6.0
    approx = geom.fy * defaults.CAM_HEIGHT_M * spec.res / (r - defaults.CAM_FORWARD_M) ** 2
    got = rows[centre & (np.abs(X - r) < 0.05)].mean()
    assert abs(got - approx) / approx < 0.25
    assert np.all(px[X > defaults.CAM_MAX_RANGE_M + 1.0] == 0)


def test_visible_range_rule():
    geom = _geom()
    spec = BevSpec()
    px, _ = expected_coverage(geom, spec)
    X, _ = spec.centres()
    states = np.full(spec.shape, CellState.UNSEEN, dtype=np.uint8)
    assert visible_range(spec, states, px) == 0.0
    states[(X < 7.0)] = CellState.GROUND
    assert 6.4 <= visible_range(spec, states, px) <= 7.0
    states[(X > 3.0) & (X < 3.5)] = CellState.OCCLUDED  # a hidden ring cuts certification
    assert visible_range(spec, states, px) <= 3.0
