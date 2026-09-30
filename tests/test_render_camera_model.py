"""Fast, browser-free tests for the renderer's camera model, colour transport and test scenes."""

from __future__ import annotations

import base64
import math

import numpy as np
import pytest

from metagross.config import defaults
from metagross.contracts.scenario import SCHEMA
from metagross.sim.render.bridge import _jsonable, yuv420_to_rgb
from metagross.sim.render.camera_model import (
    CV_TO_GL,
    T_world_cam,
    calib_to_js,
    pose_to_T_world_body,
    project_world_points,
    projection_matrix_gl,
    rot_zyx,
)
from metagross.sim.render.testscene import make_test_scenario, route_poses, terrain_height


def _gl_pixel(P: np.ndarray, Xc: np.ndarray, W: int, H: int) -> np.ndarray:
    """OpenCV camera points -> GL clip -> NDC -> OpenCV pixel (centres at integers)."""
    Xgl = Xc * np.array([1.0, -1.0, -1.0])
    clip = np.c_[Xgl, np.ones(len(Xgl))] @ P.T
    ndc = clip[:, :3] / clip[:, 3:4]
    return np.c_[(ndc[:, 0] + 1) * W / 2 - 0.5, (1 - ndc[:, 1]) * H / 2 - 0.5]


def test_gl_projection_reproduces_opencv_pinhole() -> None:
    cal = defaults.stereo_calibration()
    P = projection_matrix_gl(cal.fx, cal.fy, cal.cx, cal.cy, cal.width, cal.height)
    rng = np.random.default_rng(0)
    Xc = np.c_[rng.uniform(-3, 3, 200), rng.uniform(-2, 2, 200), rng.uniform(0.5, 60, 200)]
    uv = _gl_pixel(P, Xc, cal.width, cal.height)
    ref = np.c_[cal.fx * Xc[:, 0] / Xc[:, 2] + cal.cx, cal.fy * Xc[:, 1] / Xc[:, 2] + cal.cy]
    assert np.max(np.abs(uv - ref)) < 1e-6


def test_gl_projection_off_centre_principal_point() -> None:
    P = projection_matrix_gl(500.0, 480.0, 300.25, 210.75, 640, 400)
    Xc = np.array([[0.3, -0.2, 4.0], [0.0, 0.0, 10.0]])
    uv = _gl_pixel(P, Xc, 640, 400)
    assert np.allclose(uv[1], [300.25, 210.75], atol=1e-9)
    assert np.allclose(uv[0], [500 * 0.075 + 300.25, 480 * -0.05 + 210.75], atol=1e-9)


def test_positive_pitch_is_nose_down_and_yaw_ccw() -> None:
    fwd = rot_zyx(0.0, math.radians(10), 0.0) @ np.array([1.0, 0, 0])
    assert fwd[2] < 0
    left = rot_zyx(0.0, 0.0, math.radians(90)) @ np.array([1.0, 0, 0])
    assert np.allclose(left, [0, 1, 0], atol=1e-12)


def test_camera_extrinsics_and_baseline_direction() -> None:
    T_l = T_world_cam([0, 0, 0, 0, 0, 0], "left")
    T_r = T_world_cam([0, 0, 0, 0, 0, 0], "right")
    assert np.allclose(T_l, defaults.camera_extrinsics())
    # right camera sits baseline metres to the RIGHT (-y body) of the left one
    assert np.allclose(T_r[:3, 3] - T_l[:3, 3], [0.0, -defaults.BASELINE_M, 0.0], atol=1e-12)
    # optical axis looks forward and CAM_PITCH_DEG down
    z_axis = T_l[:3, 2]
    assert z_axis[0] > 0.9 and math.isclose(math.degrees(math.asin(-z_axis[2])), -defaults.CAM_PITCH_DEG, abs_tol=1e-9)


def test_project_world_points_matches_stereo_disparity() -> None:
    cal = defaults.stereo_calibration()
    pose = [3.0, -2.0, 0.4, 0.02, -0.03, 0.7]
    T = T_world_cam(pose, "left")
    Pc = np.array([[0.4, 0.1, 7.0], [-1.0, 0.3, 12.0]])
    Pw = Pc @ T[:3, :3].T + T[:3, 3]
    uv_l, z_l = project_world_points(Pw, pose, cal, "left")
    uv_r, z_r = project_world_points(Pw, pose, cal, "right")
    assert np.allclose(z_l, Pc[:, 2]) and np.allclose(z_r, Pc[:, 2])
    assert np.allclose(uv_l[:, 0] - uv_r[:, 0], cal.fx * cal.baseline_m / Pc[:, 2], atol=1e-9)
    assert np.allclose(uv_l[:, 1], uv_r[:, 1], atol=1e-9)  # rectified: same row


def test_pose_matrix_is_rigid() -> None:
    T = pose_to_T_world_body([1, 2, 3, 0.1, -0.2, 2.5])
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-12)
    assert np.allclose(CV_TO_GL @ CV_TO_GL, np.eye(4))


def test_calib_to_js_round_trips_extrinsics() -> None:
    d = calib_to_js()
    assert d["width"] == defaults.IMG_W and d["height"] == defaults.IMG_H
    assert np.allclose(np.array(d["T_body_cam"]).reshape(4, 4), defaults.camera_extrinsics())


def _rgb_to_yuv420(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Python mirror of the post shader (mode 2) + the JS 2x2 chroma mean."""
    c = rgb.astype(np.float64)
    Y = np.clip(np.floor(c @ [0.299, 0.587, 0.114] + 0.5), 0, 255)
    U = np.clip(128 - 0.168736 * c[..., 0] - 0.331264 * c[..., 1] + 0.5 * c[..., 2], 0, 255)
    V = np.clip(128 + 0.5 * c[..., 0] - 0.418688 * c[..., 1] - 0.081312 * c[..., 2], 0, 255)
    U8, V8 = np.floor(U + 0.5), np.floor(V + 0.5)
    H, W = Y.shape
    blk = lambda a: ((a.reshape(H // 2, 2, W // 2, 2).sum(axis=(1, 3)) + 2) // 4)  # noqa: E731
    return Y.astype(np.uint8), blk(U8).astype(np.uint8), blk(V8).astype(np.uint8)


def test_yuv420_round_trip_is_close_on_smooth_colour() -> None:
    yy, xx = np.mgrid[0:40, 0:64].astype(np.float64)
    rgb = np.stack([120 + 60 * np.sin(xx / 9), 100 + 50 * np.cos(yy / 7), 90 + 40 * np.sin((xx + yy) / 11)], -1).astype(np.uint8)
    out = yuv420_to_rgb(*_rgb_to_yuv420(rgb))
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - rgb.astype(int)).mean() < 2.0
    # luma is preserved almost exactly (it is what stereo matching uses)
    import cv2

    assert np.abs(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(int) - cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(int)).max() <= 2


def test_jsonable_converts_numpy() -> None:
    d = _jsonable({"pose": np.arange(6, dtype=np.float32), "t": np.float64(1.5), "k": (np.int64(3),), "bad": float("nan")})
    assert d == {"pose": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "t": 1.5, "k": [3], "bad": None}


@pytest.mark.parametrize("flat", [False, True])
def test_test_scenario_follows_schema(flat: bool) -> None:
    sc = make_test_scenario(101, flat=flat, extent_m=(20.0, 10.0))
    assert sc["schema"] == SCHEMA
    ny, nx = sc["terrain"]["shape"]
    h = np.frombuffer(base64.b64decode(sc["terrain"]["height_b64"]), "<f4")
    m = np.frombuffer(base64.b64decode(sc["terrain"]["material_b64"]), np.uint8)
    assert h.size == nx * ny and m.size == nx * ny and m.max() <= 5
    if flat:
        assert np.all(h == 0)
    else:
        assert any(hz["type"] == "ditch" for hz in sc["hazards"])
        p = route_poses(sc, 3, 0.0, 10.0)
        assert len(p) == 3 and all(len(q) == 6 for q in p)
        assert math.isfinite(terrain_height(sc, 1.0, 0.5))
    assert make_test_scenario(101, flat=flat, extent_m=(20.0, 10.0)) == sc  # deterministic
