"""Planar (SE(2)) EKF fusing skid-steer wheel odometry, gyro and stereo VO.

State ``x = [x, y, yaw]`` of the body frame in the A-frame (m, m, rad; x forward
at launch, y left, yaw counter-clockwise from +x). The filter starts at
(0, 0, 0) with zero covariance: the A-frame is *defined* by the launch pose.

VO measures the *relative* motion between two camera frames, not an absolute
pose, so the update uses stochastic cloning (Roumeliotis & Burdick, ICRA 2002):
the filter carries a clone of the pose at the previous VO frame; the
measurement is the SE(2) difference between the current pose and the clone,
expressed in the clone's body frame, ``h = [R(yaw_c)^T (p - p_c), yaw - yaw_c]``.
After every VO frame the clone is reset to the current pose. This keeps the
cross-correlation between consecutive VO increments and the odometry
prediction consistent.

Prediction sources (per tick):
* wheels: ``d = r (dphi_L + dphi_R) / 2``, ``dyaw = r (dphi_R - dphi_L) / (chi B)``
  with an effective-track factor chi (skid-steer), midpoint integration;
* gyro (optional): replaces the wheel-differential yaw increment,
  ``dyaw = (omega_z - bias) dt``;
* constant velocity (camera-only mode): the last accepted VO body velocity,
  with process noise growing with dt.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

LOG = logging.getLogger(__name__)


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float(math.atan2(math.sin(a), math.cos(a)))


def skid_steer_increment(dphi_l: float, dphi_r: float, wheel_radius_m: float, track_width_m: float,
                         chi: float) -> tuple[float, float]:
    """Body-frame increment (distance m, yaw rad) from wheel angle increments (rad)."""
    d = wheel_radius_m * (dphi_l + dphi_r) * 0.5
    dyaw = wheel_radius_m * (dphi_r - dphi_l) / (chi * track_width_m)
    return d, dyaw


@dataclass(slots=True)
class EKFConfig:
    """Noise model. Standard deviations; per-metre terms scale with |distance| travelled."""

    # wheel odometry: sigma_d = odo_d_frac*|d| + odo_d_floor (m); skid-steer on loose ground
    odo_d_frac: float = 0.10
    odo_d_floor_m: float = 0.002
    # wheel-differential yaw: sigma = odo_yaw_frac*|dyaw| + odo_yaw_per_m*|d| + floor (rad)
    odo_yaw_frac: float = 0.20  # skid-steer chi uncertainty dominates
    odo_yaw_per_m: float = 0.02
    odo_yaw_floor_rad: float = math.radians(0.05)
    # lateral skid (unmodelled sideways slip) per metre travelled
    odo_lat_per_m: float = 0.02
    # gyro: white noise density (rad/s/sqrt(Hz)) integrated over dt, plus residual bias
    gyro_noise_rps: float = 0.01
    gyro_bias_sigma_rps: float = 0.002
    # constant-velocity model (camera-only): accel / yaw-accel random walk
    cv_accel_mps2: float = 1.0
    cv_yaw_accel_rps2: float = 1.0
    # VO measurement: sigma = vo_floor + vo_frac*|increment| at the reference inlier count
    vo_xy_floor_m: float = 0.005
    vo_xy_frac: float = 0.02
    vo_yaw_floor_rad: float = math.radians(0.1)
    vo_yaw_frac: float = 0.02
    vo_ref_inliers: float = 150.0  # measurement noise is nominal at this inlier count
    q_floor: float = 0.05  # health score floor in the 1/q covariance inflation
    # Optional innovation gate (chi-square 3 dof; 16.27 = p 0.999). Disabled by default:
    # when the wheels slip, VO and odometry disagree and VO is the one to trust, so VO
    # integrity is judged by the health monitor + motion-sanity gate instead.
    mahalanobis_gate: Optional[float] = None


class PlanarEKF:
    """SE(2) EKF with one stochastic clone for relative (VO) measurements.

    Internal state is 6-D: ``[x, y, yaw, x_c, y_c, yaw_c]`` (current pose, clone).
    """

    def __init__(self, config: Optional[EKFConfig] = None) -> None:
        self.cfg = config or EKFConfig()
        self.reset()

    def reset(self, x0: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.s = np.zeros(6)
        self.s[:3] = x0
        self.s[3:] = x0
        self.P = np.zeros((6, 6))
        self.v_body = np.zeros(2)  # (forward m/s, yaw rate rad/s) for the constant-velocity model
        self.n_updates = 0
        self.n_rejected = 0

    # ------------------------------------------------------------------ accessors
    @property
    def pose(self) -> tuple[float, float, float]:
        return float(self.s[0]), float(self.s[1]), float(self.s[2])

    @property
    def cov(self) -> np.ndarray:
        return self.P[:3, :3].copy()

    def pos_sigma_m(self) -> float:
        """1-sigma position uncertainty along the worst axis (m)."""
        ev = np.linalg.eigvalsh(self.P[:2, :2])
        return float(math.sqrt(max(ev[-1], 0.0)))

    # ------------------------------------------------------------------ prediction
    def predict_increment(self, d: float, dyaw: float, sigma_d: float, sigma_yaw: float,
                          sigma_lat: float = 0.0) -> None:
        """Propagate the current pose by a body-frame increment (forward d m, yaw dyaw rad).

        Midpoint integration; ``sigma_*`` are 1-sigma noises on (d, lateral, dyaw).
        """
        x, y, yaw = self.s[:3]
        ym = yaw + 0.5 * dyaw
        c, s = math.cos(ym), math.sin(ym)
        self.s[0] = x + d * c
        self.s[1] = y + d * s
        self.s[2] = wrap_angle(yaw + dyaw)
        F = np.eye(6)
        F[0, 2] = -d * s
        F[1, 2] = d * c
        # noise inputs: [d, lateral, dyaw]
        G = np.zeros((6, 3))
        G[0, 0], G[1, 0] = c, s
        G[0, 1], G[1, 1] = -s, c
        G[0, 2], G[1, 2], G[2, 2] = -0.5 * d * s, 0.5 * d * c, 1.0
        Q = np.diag([sigma_d ** 2, sigma_lat ** 2, sigma_yaw ** 2])
        self.P = F @ self.P @ F.T + G @ Q @ G.T

    def predict_wheels(self, dphi_l: float, dphi_r: float, wheel_radius_m: float, track_width_m: float,
                       chi: float, gyro_dyaw: Optional[float] = None, dt: float = 0.0,
                       noise_scale: float = 1.0) -> tuple[float, float]:
        """Wheel-odometry prediction; if ``gyro_dyaw`` is given it replaces the wheel yaw.

        ``noise_scale`` (>= 1) inflates the translational odometry noise, e.g. when
        the slip monitor reports slip. Returns the (d, dyaw) increment applied.
        """
        c = self.cfg
        d, dyaw_w = skid_steer_increment(dphi_l, dphi_r, wheel_radius_m, track_width_m, chi)
        sig_d = (c.odo_d_frac * abs(d) + c.odo_d_floor_m) * noise_scale
        sig_lat = c.odo_lat_per_m * abs(d) * noise_scale
        if gyro_dyaw is not None:
            dyaw = gyro_dyaw
            sig_yaw = c.gyro_noise_rps * math.sqrt(max(dt, 1e-6)) + c.gyro_bias_sigma_rps * dt
        else:
            dyaw = dyaw_w
            sig_yaw = c.odo_yaw_frac * abs(dyaw_w) + c.odo_yaw_per_m * abs(d) + c.odo_yaw_floor_rad
        self.predict_increment(d, dyaw, sig_d, sig_yaw, sig_lat)
        return d, dyaw

    def predict_constant_velocity(self, dt: float) -> None:
        """Camera-only prediction from the last VO body velocity; noise grows with dt."""
        c = self.cfg
        v, w = self.v_body
        sig_d = 0.5 * c.cv_accel_mps2 * dt * dt + c.odo_d_floor_m
        sig_yaw = 0.5 * c.cv_yaw_accel_rps2 * dt * dt + c.odo_yaw_floor_rad
        sig_lat = 0.5 * c.cv_accel_mps2 * dt * dt
        self.predict_increment(v * dt, w * dt, sig_d, sig_yaw, sig_lat)

    # ------------------------------------------------------------------ VO update
    def vo_noise(self, dx: float, dy: float, dyaw: float, inliers: float, q: float) -> np.ndarray:
        """Measurement covariance of a planar VO increment, inflated by 1/inliers and 1/q."""
        c = self.cfg
        dist = math.hypot(dx, dy)
        sxy = c.vo_xy_floor_m + c.vo_xy_frac * dist
        syaw = c.vo_yaw_floor_rad + c.vo_yaw_frac * abs(dyaw)
        scale = (c.vo_ref_inliers / max(inliers, 1.0)) / max(q, c.q_floor)
        return np.diag([sxy ** 2, sxy ** 2, syaw ** 2]) * scale

    def update_relative(self, dx: float, dy: float, dyaw: float, R: np.ndarray, dt: Optional[float] = None) -> bool:
        """Fuse a relative planar motion (clone body frame -> current body frame).

        ``dx, dy`` in metres in the clone's body frame, ``dyaw`` in rad, ``R`` 3x3
        covariance. Returns False (and leaves the state untouched) if the
        innovation fails the Mahalanobis gate.
        """
        x, y, yaw, xc, yc, yawc = self.s
        cc, sc = math.cos(yawc), math.sin(yawc)
        ddx, ddy = x - xc, y - yc
        h = np.array([cc * ddx + sc * ddy, -sc * ddx + cc * ddy, wrap_angle(yaw - yawc)])
        H = np.zeros((3, 6))
        H[0, 0], H[0, 1] = cc, sc
        H[1, 0], H[1, 1] = -sc, cc
        H[2, 2] = 1.0
        H[0, 3], H[0, 4] = -cc, -sc
        H[1, 3], H[1, 4] = sc, -cc
        H[2, 5] = -1.0
        H[0, 5] = -sc * ddx + cc * ddy
        H[1, 5] = -cc * ddx - sc * ddy
        z = np.array([dx, dy, dyaw])
        innov = z - h
        innov[2] = wrap_angle(innov[2])
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        m2 = float(innov @ S_inv @ innov)
        gate = self.cfg.mahalanobis_gate
        if gate is not None and m2 > gate and self.n_updates > 0:
            self.n_rejected += 1
            LOG.debug("VO update gated: mahalanobis^2=%.1f", m2)
            return False
        Kg = self.P @ H.T @ S_inv
        self.s = self.s + Kg @ innov
        self.s[2] = wrap_angle(self.s[2])
        I_KH = np.eye(6) - Kg @ H
        self.P = I_KH @ self.P @ I_KH.T + Kg @ R @ Kg.T  # Joseph form
        self.n_updates += 1
        if dt is not None and dt > 0:
            self.v_body = np.array([dx / dt, dyaw / dt])
        return True

    def clone(self) -> None:
        """Reset the clone to the current pose (call once per VO frame, after any update)."""
        self.s[3:] = self.s[:3]
        P = self.P
        P[3:, :] = P[:3, :]
        P[:, 3:] = P[:, :3]
        self.P = 0.5 * (P + P.T)
