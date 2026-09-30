"""Skid-steer mixer: body command (v, omega) -> wheel angular rates.

Skid-steer kinematics with an effective track factor chi (>1 because lateral tyre
slip widens the effective track, Mandow et al. 2007):

    v_L = v - omega * chi * B / 2,    v_R = v + omega * chi * B / 2
    omega_{L,R} [rad/s] = v_{L,R} / r

B = ``VehicleSpec.track_width_m``, r = ``wheel_radius_m``, chi = the localiser's
estimate ``chi_hat`` (``CHI_NOMINAL`` until one is available).

Limits applied, in order, to the body command:
1. yaw-rate limit |omega| <= max_yaw_rate;
2. acceleration limits (``max_accel`` speeding up, ``max_decel`` slowing down) and a
   jerk limit on v, plus an angular-acceleration limit on omega; an emergency stop
   bypasses jerk and uses the full brake deceleration;
3. curvature-preserving wheel saturation: if either wheel exceeds max_wheel_rad_s
   both are scaled by the same factor (omega/v, hence path curvature, is kept).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metagross.config.defaults import BRAKE_DECEL_MPS2, CHI_NOMINAL, VEHICLE
from metagross.contracts.messages import DriveMode, VehicleSpec, WheelCmd

MAX_JERK_MPS3 = 4.0  # comfort / traction jerk limit on v
MAX_YAW_ACCEL_RPS2 = 4.0  # angular acceleration limit on omega


def body_to_wheels(v: float, omega: float, chi: float, track_m: float, radius_m: float) -> tuple[float, float]:
    """(v [m/s], omega [rad/s]) -> (omega_L, omega_R) [rad/s]."""
    half = omega * chi * track_m / 2.0
    return (v - half) / radius_m, (v + half) / radius_m


def wheels_to_body(omega_l: float, omega_r: float, chi: float, track_m: float, radius_m: float) -> tuple[float, float]:
    """Inverse of :func:`body_to_wheels`."""
    v = radius_m * (omega_r + omega_l) / 2.0
    omega = radius_m * (omega_r - omega_l) / (chi * track_m)
    return v, omega


def saturate_wheels(omega_l: float, omega_r: float, limit: float) -> tuple[float, float, float]:
    """Scale both wheel rates by one factor so neither exceeds ``limit``. Returns (l, r, scale)."""
    m = max(abs(omega_l), abs(omega_r))
    if m <= limit or m == 0.0:
        return omega_l, omega_r, 1.0
    k = limit / m
    return omega_l * k, omega_r * k, k


@dataclass(frozen=True, slots=True)
class MixerParams:
    track_width_m: float = VEHICLE.track_width_m
    wheel_radius_m: float = VEHICLE.wheel_radius_m
    max_wheel_rad_s: float = VEHICLE.max_wheel_rad_s
    max_yaw_rate: float = VEHICLE.max_yaw_rate_rps
    max_accel: float = VEHICLE.max_accel_mps2
    max_decel: float = BRAKE_DECEL_MPS2
    max_jerk: float = MAX_JERK_MPS3
    max_yaw_accel: float = MAX_YAW_ACCEL_RPS2

    @classmethod
    def from_vehicle(cls, vs: VehicleSpec) -> "MixerParams":
        return cls(vs.track_width_m, vs.wheel_radius_m, vs.max_wheel_rad_s, vs.max_yaw_rate_rps, vs.max_accel_mps2)


class SkidSteerMixer:
    """Stateful (rate-limited) mixer. Call :meth:`reset` at mission start."""

    def __init__(self, params: MixerParams = MixerParams()) -> None:
        self.p = params
        self.reset()

    def reset(self) -> None:
        self.v = 0.0  # last applied body speed [m/s]
        self.w = 0.0  # last applied yaw rate [rad/s]
        self.a = 0.0  # last applied longitudinal accel [m/s^2]
        self._t: float | None = None

    def mix(self, t: float, v_des: float, w_des: float, chi_hat: float = CHI_NOMINAL, emergency: bool = False) -> tuple[float, float]:
        """Rate-limit (v_des, w_des) and convert to wheel rates. Returns (omega_L, omega_R) [rad/s]."""
        p = self.p
        dt = 0.0 if self._t is None else max(t - self._t, 0.0)
        self._t = t
        chi = chi_hat if (chi_hat is not None and np.isfinite(chi_hat) and chi_hat > 0.5) else CHI_NOMINAL
        w_des = float(np.clip(w_des, -p.max_yaw_rate, p.max_yaw_rate))
        if dt > 0.0:
            a_des = (v_des - self.v) / dt
            if emergency:
                a = float(np.clip(a_des, -p.max_decel, p.max_accel))
            else:
                a = float(np.clip(a_des, self.a - p.max_jerk * dt, self.a + p.max_jerk * dt))
                a = float(np.clip(a, -p.max_decel, p.max_accel))
            v = self.v + a * dt
            # do not overshoot the target because of the jerk-limited accel
            if (v_des - self.v) * (v_des - v) < 0.0:
                v, a = v_des, (v_des - self.v) / dt
            w = float(np.clip(w_des, self.w - p.max_yaw_accel * dt, self.w + p.max_yaw_accel * dt))
            if emergency and abs(w_des) < abs(self.w):
                w = w_des
        else:
            v, w, a = 0.0, 0.0, 0.0  # first tick: start from rest
        wl, wr = body_to_wheels(v, w, chi, p.track_width_m, p.wheel_radius_m)
        wl, wr, k = saturate_wheels(wl, wr, p.max_wheel_rad_s)
        if k < 1.0:
            v, w = v * k, w * k
        self.v, self.w, self.a = v, w, a
        return wl, wr

    def command(self, t: float, seq: int, v_des: float, w_des: float, mode: DriveMode, compute_ms: float,
                chi_hat: float = CHI_NOMINAL, emergency: bool = False) -> WheelCmd:
        """Convenience wrapper producing a :class:`WheelCmd`."""
        wl, wr = self.mix(t, v_des, w_des, chi_hat, emergency)
        return WheelCmd(t=t, seq=seq, omega_l_rad_s=float(wl), omega_r_rad_s=float(wr), mode=mode, compute_ms=float(compute_ms))
