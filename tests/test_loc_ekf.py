"""SE(2) EKF: wheel odometry prediction, gyro, relative VO updates, camera-only mode."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metagross.autonomy.localization.ekf import EKFConfig, PlanarEKF, skid_steer_increment, wrap_angle

R_WHEEL, TRACK, CHI = 0.13, 0.50, 1.4


def wheel_angles_for(v: float, w: float, dt: float, chi: float = CHI) -> tuple[float, float]:
    """Inverse skid-steer model: wheel angle increments producing (v, w) over dt."""
    vl = v - w * chi * TRACK / 2
    vr = v + w * chi * TRACK / 2
    return vl * dt / R_WHEEL, vr * dt / R_WHEEL


def test_skid_steer_increment_inverse():
    dl, dr = wheel_angles_for(1.0, 0.5, 0.1)
    d, dyaw = skid_steer_increment(dl, dr, R_WHEEL, TRACK, CHI)
    assert d == pytest.approx(0.1) and dyaw == pytest.approx(0.05)


def test_straight_line_wheels():
    ekf = PlanarEKF()
    for _ in range(100):  # 10 m at 1 m/s, dt = 0.1
        ekf.predict_wheels(*wheel_angles_for(1.0, 0.0, 0.1), R_WHEEL, TRACK, CHI)
    x, y, yaw = ekf.pose
    assert x == pytest.approx(10.0, abs=1e-9) and y == pytest.approx(0.0, abs=1e-9) and yaw == pytest.approx(0.0)
    # uncertainty grows with distance, lateral more than longitudinal (yaw error lever arm)
    assert ekf.pos_sigma_m() > 0.1
    assert ekf.P[1, 1] > ekf.P[0, 0]


def test_turn_quarter_circle():
    ekf = PlanarEKF()
    v, w, dt = 1.0, math.pi / 2 / 5.0, 0.01  # quarter circle in 5 s, radius 2/pi*5 m
    for _ in range(500):
        ekf.predict_wheels(*wheel_angles_for(v, w, dt), R_WHEEL, TRACK, CHI)
    radius = v / w
    x, y, yaw = ekf.pose
    assert yaw == pytest.approx(math.pi / 2, abs=1e-6)
    assert x == pytest.approx(radius, abs=1e-3) and y == pytest.approx(radius, abs=1e-3)


def test_gyro_replaces_wheel_yaw():
    ekf = PlanarEKF()
    dl, dr = wheel_angles_for(1.0, 0.0, 0.1)  # wheels say straight
    ekf.predict_wheels(dl, dr, R_WHEEL, TRACK, CHI, gyro_dyaw=0.02, dt=0.1)
    assert ekf.pose[2] == pytest.approx(0.02)


@pytest.mark.parametrize("noise_scale,tol", [(1.0, 0.15), (3.0, 0.05)])
def test_vo_update_corrects_scale_error(noise_scale, tol):
    """Wheels over-read by 20 % (slip); healthy VO pulls the estimate to the truth, and
    more so when the odometry noise is inflated by the slip monitor."""
    ekf = PlanarEKF()
    for _ in range(50):
        dl, dr = wheel_angles_for(1.2, 0.0, 0.1)
        ekf.predict_wheels(dl, dr, R_WHEEL, TRACK, CHI, noise_scale=noise_scale)
        R = ekf.vo_noise(0.1, 0.0, 0.0, inliers=300, q=1.0)
        assert ekf.update_relative(0.1, 0.0, 0.0, R, dt=0.1)
        ekf.clone()
    x, y, _ = ekf.pose
    assert x == pytest.approx(5.0, abs=tol)  # truth 5.0, wheels alone 6.0
    assert abs(y) < 1e-6


def test_vo_noise_scales_with_inliers_and_q():
    ekf = PlanarEKF()
    base = ekf.vo_noise(0.1, 0, 0, inliers=150, q=1.0)
    assert np.allclose(ekf.vo_noise(0.1, 0, 0, inliers=75, q=1.0), 2 * base)
    assert np.allclose(ekf.vo_noise(0.1, 0, 0, inliers=150, q=0.5), 2 * base)


def test_relative_update_in_rotated_frame():
    """VO increments are in the clone's body frame: after a 90 deg turn, forward = +y."""
    ekf = PlanarEKF()
    ekf.reset((0.0, 0.0, math.pi / 2))
    ekf.predict_increment(0.0, 0.0, 0.05, 0.01)  # no motion, some uncertainty
    R = np.diag([1e-6, 1e-6, 1e-8])
    ekf.update_relative(1.0, 0.0, 0.0, R)
    x, y, yaw = ekf.pose
    assert x == pytest.approx(0.0, abs=1e-3) and y == pytest.approx(1.0, abs=1e-2)


def test_camera_only_constant_velocity():
    ekf = PlanarEKF(EKFConfig())
    for _ in range(10):
        ekf.predict_constant_velocity(0.1)
        R = ekf.vo_noise(0.2, 0, 0, 300, 1.0)
        ekf.update_relative(0.2, 0.0, 0.0, R, dt=0.1)
        ekf.clone()
    assert ekf.v_body[0] == pytest.approx(2.0, rel=0.05)
    x0 = ekf.pose[0]
    s0 = ekf.pos_sigma_m()
    for _ in range(5):  # VO outage: bridge on constant velocity
        ekf.predict_constant_velocity(0.1)
        ekf.clone()
    assert ekf.pose[0] == pytest.approx(x0 + 1.0, rel=0.05)
    assert ekf.pos_sigma_m() > s0


def test_wrap_angle():
    assert wrap_angle(3 * math.pi) == pytest.approx(math.pi)
    assert wrap_angle(-3 * math.pi / 2) == pytest.approx(math.pi / 2)
