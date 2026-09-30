"""Tier-0 synthetic depth sensor + proprioceptive sensors."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from metagross.config import defaults
from metagross.contracts.scenario import MATERIAL_TO_SEM, OBJECT_SEM, SKY_SEM
from metagross.sim.geometry import GridSpec, pose_matrix
from metagross.sim.hazards import ditch_drop
from metagross.sim.objects import Primitive
from metagross.sim.sensors import Gyro, GyroParams, Tier0DepthSensor, Tier0Params, WheelEncoders
from metagross.sim.terrain import Terrain

CAL = defaults.stereo_calibration()
T_BC = defaults.camera_extrinsics()


def _flat(shape=(600, 600), origin=(-5.0, -15.0), mat_id: int = 0) -> Terrain:
    g = GridSpec(origin[0], origin[1], 0.05, *shape)
    return Terrain(g, np.zeros(shape, np.float32), np.full(shape, mat_id, np.uint8))


def _pixel_rays(T_wc: np.ndarray) -> np.ndarray:
    uu, vv = np.meshgrid(np.arange(CAL.width), np.arange(CAL.height))
    d = np.stack([(uu - CAL.cx) / CAL.fx, (vv - CAL.cy) / CAL.fy, np.ones_like(uu, float)], -1)
    return d @ T_wc[:3, :3].T  # world direction with camera-z = 1 (ray parameter = depth)


def _project(T_wc: np.ndarray, p_world: np.ndarray) -> tuple[int, int, float]:
    pc = T_wc[:3, :3].T @ (p_world - T_wc[:3, 3])
    return int(round(CAL.fx * pc[0] / pc[2] + CAL.cx)), int(round(CAL.fy * pc[1] / pc[2] + CAL.cy)), float(pc[2])


def test_flat_ground_disparity_matches_analytic():
    sensor = Tier0DepthSensor(_flat(), [], Tier0Params(lr_check=False))
    T_wc = pose_matrix(0, 0, 0, 0, 0, 0) @ T_BC
    res = sensor.render(T_wc, noise=False, want_gt=True)
    d = _pixel_rays(T_wc)
    with np.errstate(divide="ignore"):
        Z = -T_wc[2, 3] / d[..., 2]
    ok = (d[..., 2] < 0) & (Z <= defaults.CAM_MAX_RANGE_M)
    d_true = np.where(ok, CAL.fx * CAL.baseline_m / np.where(ok, Z, 1.0), 0.0)
    got = res.disparity
    both = ok & (got > 0)
    assert both.sum() / ok.sum() > 0.97  # almost every ground pixel within range is measured
    err = np.abs(got[both] - d_true[both])
    assert np.percentile(err, 95) < 0.25 and err.max() < 0.5  # px at 640 x 400
    # Above the horizon: sky -> invalid disparity, semantic 'sky'.
    assert np.all(got[~(d[..., 2] < 0)] == 0)
    assert np.all(res.semantic_gt[:20] == SKY_SEM)
    assert np.all(res.semantic_gt[-20:] == MATERIAL_TO_SEM[0])


def test_output_resolution_and_calibration_modes():
    t = _flat()
    full = Tier0DepthSensor(t, [])
    small = Tier0DepthSensor(t, [], Tier0Params(output="internal"))
    T_wc = pose_matrix(0, 0, 0, 0, 0, 0) @ T_BC
    a = full.render(T_wc, noise=False).disparity
    b = small.render(T_wc, noise=False).disparity
    assert a.shape == (CAL.height, CAL.width) and a.dtype == np.float32
    assert b.shape == (CAL.height // 2, CAL.width // 2)
    cf = full.calibration()
    assert (cf.width, cf.height, cf.fx, cf.cx, cf.cy) == (CAL.width, CAL.height, CAL.fx, CAL.cx, CAL.cy)
    cs = small.calibration()
    assert cs.fx == pytest.approx(CAL.fx / 2) and cs.cx == pytest.approx((CAL.cx + 0.5) / 2 - 0.5)
    np.testing.assert_allclose(a[::2, ::2], 2.0 * b, rtol=1e-6)  # full-res disparity in full-res px


def _trench_terrain(width: float, depth: float, x_c: float) -> Terrain:
    t = _flat()
    ditch = {"type": "ditch", "polyline": [[x_c, -20.0], [x_c, 20.0]], "width": width, "depth": depth, "gaps": []}
    return Terrain(t.grid, t.height - ditch_drop(t.grid, ditch), t.material)


def test_ditch_interior_hidden_behind_lip():
    """A 1 m wide, 1 m deep trench 5 m ahead: its floor is invisible (occluded by the near lip),
    the upper part of the far wall is visible, exactly as with a real camera at 0.9 m height."""
    x_c, w, depth = 5.0, 1.0, 1.0
    sensor = Tier0DepthSensor(_trench_terrain(w, depth, x_c), [], Tier0Params(lr_check=False))
    T_wc = pose_matrix(0, 0, 0, 0, 0, 0) @ T_BC
    res = sensor.render(T_wc, noise=False, want_gt=True)
    depth_img = res.depth_gt
    terrain = sensor.terrain
    hidden, visible = 0, 0
    for y in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for x in np.linspace(x_c - 0.2, x_c + 0.2, 5):  # trench floor / lower walls (true surface points)
            u, v, z = _project(T_wc, np.array([x, y, float(terrain.height_at(x, y))]))
            hidden += depth_img[v, u] < z - 0.05
        xf = x_c + 0.45  # upper far wall, just below the far lip
        u, v, z = _project(T_wc, np.array([xf, y, float(terrain.height_at(xf, y))]))
        visible += abs(depth_img[v, u] - z) < 0.02 * z
    assert hidden == 25  # every floor point is occluded
    assert visible == 5  # the far-lip face is seen
    # Ground-truth fraction check: no rendered pixel lies deeper than the near-lip ray allows.
    # Rays grazing the near lip descend 0.9 / 4.5 m per m, so nothing below ~0.2 m depth is seen.
    d = _pixel_rays(T_wc)
    pts_z = T_wc[2, 3] + d[..., 2] * np.where(depth_img > 0, depth_img, 0.0)
    assert pts_z[depth_img > 0].min() > -0.3


def test_objects_are_raycast_with_correct_depth_and_label():
    trunk = Primitive("vcyl", np.array([4.0, 0.0, 1.0]), (0.2, -0.2, 2.5), math.hypot(0.2, 1.35))
    sensor = Tier0DepthSensor(_flat(), [trunk], Tier0Params(lr_check=False))
    T_wc = pose_matrix(0, 0, 0, 0, 0, 0) @ T_BC
    res = sensor.render(T_wc, noise=False, want_gt=True)
    cam = T_wc[:3, 3]
    u, v, _ = _project(T_wc, np.array([4.0, cam[1], cam[2]]))  # trunk centre at camera height
    ray = _pixel_rays(T_wc)[v, u]
    # Analytic depth: intersect the ray with the cylinder x^2+y^2 = r^2 around (4, 0).
    o = cam[:2] - np.array([4.0, 0.0])
    a, b, c = ray[:2] @ ray[:2], 2 * o @ ray[:2], o @ o - 0.2 ** 2
    t_hit = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
    assert res.depth_gt[v, u] == pytest.approx(t_hit, rel=2e-3)
    assert res.semantic_gt[v, u] == OBJECT_SEM


def test_left_right_occlusion_band_left_of_foreground_object():
    pole = Primitive("vcyl", np.array([2.0, 0.0, 1.0]), (0.1, -0.2, 3.0), 1.6)
    t = _flat()
    with_lr = Tier0DepthSensor(t, [pole]).render(pose_matrix(0, 0, 0, 0, 0, 0) @ T_BC, noise=False).disparity
    without = Tier0DepthSensor(t, [pole], Tier0Params(lr_check=False)).render(pose_matrix(0, 0, 0, 0, 0, 0) @ T_BC, noise=False).disparity
    newly_invalid = (without > 0) & (with_lr == 0)
    row = 220  # ground behind the pole is ~3.4 m away (15 px) vs ~31 px on the pole
    cols = np.nonzero(newly_invalid[row])[0]
    cols = cols[cols > 40]  # ignore the left-border band (no right-image correspondence)
    pole_cols = np.nonzero(without[row] > 25)[0]
    assert len(cols) > 8
    assert cols.max() < pole_cols.min()  # the band sits immediately LEFT of the pole
    assert pole_cols.min() - cols.min() < 40


def test_noise_level_and_dropout():
    sensor = Tier0DepthSensor(_flat(mat_id=2), [])
    T_wc = pose_matrix(0, 0, 0, 0, 0, 0) @ T_BC
    clean = sensor.render(T_wc, noise=False).disparity
    noisy = sensor.render(T_wc, rng=np.random.default_rng(3)).disparity
    m = (clean > 0) & (noisy > 0)
    e = (noisy - clean)[m]
    # Robust std (MAD) at full resolution ~ sigma_d = 0.25 px.
    sig = 1.4826 * np.median(np.abs(e - np.median(e)))
    assert 0.2 < sig < 0.3
    drop = 1 - m.sum() / (clean > 0).sum()
    assert 0.0 < drop < 0.1  # gravel is well textured


def test_water_is_mostly_dropped_and_lighting_events_hurt():
    t = _flat(mat_id=5)
    sensor = Tier0DepthSensor(t, [])
    T_wc = pose_matrix(0, 0, 0, 0, 0, 0) @ T_BC
    clean = sensor.render(T_wc, noise=False).disparity > 0
    water = sensor.render(T_wc, rng=np.random.default_rng(0)).disparity > 0
    assert water.sum() / clean.sum() < 0.6
    g = _flat(mat_id=2)
    s2 = Tier0DepthSensor(g, [])
    base = (s2.render(T_wc, rng=np.random.default_rng(0)).disparity > 0).sum()
    dust = {"fog_density": 0.0, "active_events": [{"type": "dust", "gain": 0.2, "t0": 0, "t1": 1}]}
    glare = {"fog_density": 0.0, "sun_dir_world": [1.0, 0.0, 0.05], "active_events": [{"type": "glare", "gain": 2.0, "t0": 0, "t1": 1}]}
    n_dust = (s2.render(T_wc, lighting=dust, rng=np.random.default_rng(0)).disparity > 0).sum()
    n_glare = (s2.render(T_wc, lighting=glare, rng=np.random.default_rng(0)).disparity > 0).sum()
    assert n_dust < 0.7 * base and n_glare < 0.95 * base


def test_render_speed_budget():
    """Regression guard only (CPU is shared while tests run); measured numbers go in the report."""
    rng = np.random.default_rng(0)
    g = GridSpec(0.0, 0.0, 0.05, 800, 1280)
    t = Terrain(g, (rng.standard_normal(g.shape) * 0.02).astype(np.float32), np.zeros(g.shape, np.uint8))
    sensor = Tier0DepthSensor(t, [])
    T_wc = pose_matrix(20, 20, 0, 0.05, -0.05, 0.3) @ T_BC
    sensor.render(T_wc, rng=rng)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        sensor.render(T_wc, rng=rng)
        ts.append(time.perf_counter() - t0)
    assert np.median(ts) < 0.4


def test_encoders_quantise_wheel_angle():
    enc = WheelEncoders(4096)
    q = 2 * math.pi / 4096
    l, r = enc.read(10.0 * q + 0.3 * q, -3.2 * q)
    assert l == pytest.approx(10 * q) and r == pytest.approx(-4 * q)


def test_gyro_bias_and_noise_statistics():
    p = GyroParams(bias0_sigma=0.0, bias_rw=0.0, noise_density=2e-3)
    g = Gyro(np.random.default_rng(0), p)
    x = np.array([g.sample(0.5, 0.2) for _ in range(4000)])
    assert x.mean() == pytest.approx(0.5, abs=3 * 2e-3 / math.sqrt(0.2) / math.sqrt(4000) + 1e-4)
    assert x.std() == pytest.approx(2e-3 / math.sqrt(0.2), rel=0.05)
    biased = Gyro(np.random.default_rng(1), GyroParams(bias0_sigma=0.01, bias_rw=0.0, noise_density=0.0))
    assert biased.sample(0.0, 0.2) == pytest.approx(biased.bias)
