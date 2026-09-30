"""End-to-end Localizer on the synthetic ray-cast scene (VO + health + EKF + slip)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metagross.autonomy.localization.localizer import Localizer, LocalizerConfig, camera_to_body_increment
from metagross.config.defaults import CHI_NOMINAL, VEHICLE, camera_extrinsics
from metagross.contracts.messages import SensorFrame, StereoCalibration
from tests.test_loc_synth import BASELINE_M, CX, CY, FX, FY, H, W, disparity_from_depth, make_texture, pose, render

DT = 0.2
R_LEVEL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # camera -> body, level camera


def calib() -> StereoCalibration:
    T = np.eye(4)
    T[:3, :3] = R_LEVEL
    T[:3, 3] = [0.0, 0.0, 0.9]
    return StereoCalibration(W, H, FX, FY, CX, CY, BASELINE_M, T)


@pytest.fixture(scope="module")
def tex() -> np.ndarray:
    return make_texture(7)


def drive(loc: Localizer, tex: np.ndarray, n: int, v: float, w: float = 0.0, wheel_gain: float = 1.0,
          blank: range = range(0), images: bool = True, gyro_bias: float = 0.0):
    """Drive a body-frame (v, w) arc; returns outputs and the true final body pose."""
    x = y = yaw = 0.0
    phi_l = phi_r = 0.0
    outs = []
    for k in range(n):
        if k > 0:
            ym = yaw + 0.5 * w * DT
            x += v * DT * math.cos(ym)
            y += v * DT * math.sin(ym)
            yaw += w * DT
            vl = (v - w * CHI_NOMINAL * VEHICLE.track_width_m / 2) * wheel_gain
            vr = (v + w * CHI_NOMINAL * VEHICLE.track_width_m / 2) * wheel_gain
            phi_l += vl * DT / VEHICLE.wheel_radius_m
            phi_r += vr * DT / VEHICLE.wheel_radius_m
        img = disp = None
        if images:
            img, depth = render(pose(-y, 0.0, x, yaw), tex)  # body (x fwd, y left) -> camera (x right, z fwd)
            disp = disparity_from_depth(depth)
            if k in blank:
                img = np.full_like(img, 128)
                disp = np.zeros_like(disp)
        frame = SensorFrame(t=k * DT, seq=k, left_rgb=None if img is None else np.dstack([img] * 3),
                            right_gray=None, wheel_angle_l_rad=phi_l, wheel_angle_r_rad=phi_r,
                            gyro_z_rps=w + gyro_bias, sensor_mode="stereo" if images else "tier0_disparity")
        outs.append(loc.update(frame, disp))
    return outs, (x, y, yaw)


def test_camera_to_body_uses_extrinsics():
    T_cam = np.eye(4)
    T_cam[2, 3] = 1.0  # camera moved 1 m along its optical axis
    dx, dy, dyaw = camera_to_body_increment(T_cam, camera_extrinsics())
    # the UGV camera is pitched 12 deg down, so the optical axis is not horizontal
    assert dx == pytest.approx(math.cos(math.radians(12.0)), abs=1e-9) and abs(dy) < 1e-9 and dyaw == 0.0


def test_straight_with_consistent_wheels(tex):
    loc = Localizer(calib(), VEHICLE, model_path=None)
    outs, (x, y, _) = drive(loc, tex, 12, v=1.0)
    o = outs[-1]
    assert o["pose_xy_yaw"][0] == pytest.approx(x, abs=0.05) and abs(o["pose_xy_yaw"][1]) < 0.05
    assert all(r["vo_ok"] for r in outs[1:])
    assert o["health"]["q"] > 0.5 and set(o["health"]) >= {"p_fail", "q", "vo_inliers", "img_lapvar"}
    assert o["pos_sigma_m"] < 0.1 and o["slip"] == pytest.approx(0.0, abs=0.05)
    assert {"vo", "health", "ekf", "total"} <= set(o["timings_ms"])


def test_wheel_slip_is_detected_and_vo_wins(tex):
    loc = Localizer(calib(), VEHICLE, model_path=None)
    outs, (x, _, _) = drive(loc, tex, 15, v=1.0, wheel_gain=2.0)  # wheels read double
    o = outs[-1]
    assert o["slip"] == pytest.approx(0.5, abs=0.08)
    assert o["pose_xy_yaw"][0] == pytest.approx(x, abs=0.25)  # wheels alone would say 2x


def test_turning_arc_and_gyro(tex):
    loc = Localizer(calib(), VEHICLE, model_path=None)
    outs, (x, y, yaw) = drive(loc, tex, 10, v=0.8, w=0.25)
    px, py, pyaw = outs[-1]["pose_xy_yaw"]
    assert px == pytest.approx(x, abs=0.08) and py == pytest.approx(y, abs=0.08)
    assert pyaw == pytest.approx(yaw, abs=math.radians(1.5))


def test_camera_only_mode(tex):
    cfg = LocalizerConfig(use_wheel_odom=False)
    loc = Localizer(calib(), VEHICLE, cfg, model_path=None)
    outs, (x, y, _) = drive(loc, tex, 12, v=1.0, wheel_gain=0.0)  # encoders dead: must not matter
    assert outs[-1]["pose_xy_yaw"][0] == pytest.approx(x, abs=0.08)


def test_blank_frames_rejected_and_bridged(tex):
    loc = Localizer(calib(), VEHICLE, model_path=None)
    outs, (x, _, _) = drive(loc, tex, 14, v=1.0, blank=range(6, 9))
    for k in range(6, 9):
        assert not outs[k]["vo_accepted"]
        assert outs[k]["health"]["q_gate"] < 0.4 or not outs[k]["vo_ok"]
    sig_before, sig_during = outs[5]["pos_sigma_m"], outs[8]["pos_sigma_m"]
    assert sig_during > sig_before  # uncertainty grows while bridging on wheels
    assert outs[-1]["pose_xy_yaw"][0] == pytest.approx(x, abs=0.1)


def test_tier0_without_images_dead_reckons():
    loc = Localizer(calib(), VEHICLE, model_path=None)
    outs, (x, _, _) = drive(loc, None, 10, v=1.0, images=False)
    o = outs[-1]
    assert not o["vo_ok"] and o["health"]["vo_available"] == 0.0
    assert o["pose_xy_yaw"][0] == pytest.approx(x, abs=1e-6)


def test_gyro_bias_learned_while_stationary(tex):
    loc = Localizer(calib(), VEHICLE, model_path=None)
    # a 10-sigma bias (prior 1e-3 rad/s) needs ~20 s of standing still to be learned to 20 %
    drive(loc, tex, 100, v=0.0, gyro_bias=0.01, images=False)
    assert loc.gyro_bias == pytest.approx(0.01, rel=0.2)
