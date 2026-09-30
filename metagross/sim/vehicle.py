"""Kinematic skid-steer vehicle with actuator lag, slip, terrain following and latency injection.

Model (per physics step ``dt = 1 / PHYSICS_HZ``)
-----------------------------------------------
* Wheel speeds (rad/s, per side) follow the commanded speeds through a first-order lag with time
  constant ``defaults.ACTUATOR_LAG_S`` (exact discretisation), then an angular-acceleration limit
  ``max_accel_mps2 / wheel_radius``; commands are clamped to ``max_wheel_rad_s``.
* Skid-steer kinematics with the effective-track factor chi (ICR model, Mandow et al. 2007):

      v     = (1 - s) * r * (w_l + w_r) / 2                  [m/s, along body x]
      omega = (1 - s) * r * (w_r - w_l) / (chi * B)          [rad/s, yaw rate]

  where ``s`` is the longitudinal slip (scenario ``vehicle.slip_long``, x ``SLIP_FACTOR`` on
  mud / water) and ``chi`` the true per-material value from ``vehicle.chi_by_material`` sampled
  under the body origin. The autonomy only knows ``defaults.CHI_NOMINAL``.
* Terrain following: heights of the four wheel-contact points (bilinear on the heightmap) give
  ``z`` (mean), ``pitch = atan2(h_rear - h_front, wheelbase)`` (positive = nose down) and
  ``roll = atan2(h_left - h_right, track)`` (positive = left side up), i.e. the REP-103 FLU
  convention of :mod:`metagross.sim.geometry`.
* Encoders integrate the *actual wheel rotation* (including the part lost to slip), not ground
  travel, exactly like real shaft encoders.

The model is kinematic: no gravity along slopes, no sinkage; hazards are judged by the referee.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from metagross.config import defaults
from metagross.contracts.messages import VehicleSpec, WheelCmd
from metagross.sim.terrain import Terrain

WHEELBASE_M = 0.45  # front-rear wheel-contact distance (Scout-Mini class); autonomy does not need it
SLIP_FACTOR = {4: 2.0, 5: 2.5}  # slip multiplier on mud (4) and water (5) cells


@dataclass
class VehicleState:
    """True vehicle state (world frame, SI units)."""

    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0  # wrapped to [-pi, pi)
    yaw_unwrapped: float = 0.0
    v: float = 0.0  # ground speed along body x (m/s)
    omega: float = 0.0  # yaw rate (rad/s)
    wl: float = 0.0  # actual left wheel speed (rad/s)
    wr: float = 0.0
    cmd_l: float = 0.0  # command currently applied (after latency), rad/s
    cmd_r: float = 0.0
    enc_l: float = 0.0  # cumulative true wheel angle (rad)
    enc_r: float = 0.0
    odo_m: float = 0.0  # cumulative ground path length (m)
    material: int = 0
    chi: float = defaults.CHI_NOMINAL
    slip: float = 0.0

    def pose6(self) -> list[float]:
        return [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]


class SkidSteerVehicle:
    """Integrates :class:`VehicleState` on a :class:`Terrain` at ``dt`` seconds per step."""

    def __init__(self, terrain: Terrain, params: dict, spec: VehicleSpec = defaults.VEHICLE,
                 dt: float = 1.0 / defaults.PHYSICS_HZ, lag_s: float = defaults.ACTUATOR_LAG_S):
        self.terrain = terrain
        self.spec = spec
        self.dt = dt
        self.alpha = 1.0 - math.exp(-dt / lag_s) if lag_s > 0 else 1.0
        self.dw_max = spec.max_accel_mps2 / spec.wheel_radius_m * dt
        chi = params.get("chi_by_material", {})
        self.chi_by_material = {int(k): float(v) for k, v in chi.items()}
        self.slip_long = float(params.get("slip_long", 0.0))
        self.state = VehicleState()

    # ------------------------------------------------------------------ geometry
    def contact_points(self, x: float | None = None, y: float | None = None, yaw: float | None = None) -> np.ndarray:
        """(4, 2) world xy of the wheel contacts in order FL, FR, RL, RR."""
        s = self.state
        x = s.x if x is None else x
        y = s.y if y is None else y
        yaw = s.yaw if yaw is None else yaw
        hb, ht = 0.5 * WHEELBASE_M, 0.5 * self.spec.track_width_m
        local = np.array([[hb, ht], [hb, -ht], [-hb, ht], [-hb, -ht]])
        c, sn = math.cos(yaw), math.sin(yaw)
        return local @ np.array([[c, sn], [-sn, c]]) + np.array([x, y])

    def _settle(self) -> None:
        """Set z / roll / pitch from the four contact heights."""
        s = self.state
        h = self.terrain.height_at(*self.contact_points().T)
        fl, fr, rl, rr = (float(v) for v in h)
        s.z = 0.25 * (fl + fr + rl + rr)
        s.pitch = math.atan2(0.5 * (rl + rr) - 0.5 * (fl + fr), WHEELBASE_M)
        s.roll = math.atan2(0.5 * (fl + rl) - 0.5 * (fr + rr), self.spec.track_width_m)

    # ------------------------------------------------------------------ API
    def reset(self, x: float, y: float, yaw: float, t: float = 0.0) -> VehicleState:
        self.state = VehicleState(t=t, x=x, y=y, yaw=yaw, yaw_unwrapped=yaw)
        self._settle()
        self._update_material()
        return self.state

    def set_command(self, omega_l: float, omega_r: float) -> None:
        lim = self.spec.max_wheel_rad_s
        self.state.cmd_l = min(max(float(omega_l), -lim), lim)
        self.state.cmd_r = min(max(float(omega_r), -lim), lim)

    def _update_material(self) -> None:
        s = self.state
        m = int(self.terrain.material_at(s.x, s.y))
        s.material = m
        s.chi = self.chi_by_material.get(m, defaults.CHI_NOMINAL)
        s.slip = min(self.slip_long * SLIP_FACTOR.get(m, 1.0), 0.9)

    def step(self) -> VehicleState:
        """Advance one physics step (``dt`` s)."""
        s, dt = self.state, self.dt
        a, lim = self.alpha, self.dw_max
        s.wl += min(max(a * (s.cmd_l - s.wl), -lim), lim)
        s.wr += min(max(a * (s.cmd_r - s.wr), -lim), lim)
        self._update_material()
        r, B = self.spec.wheel_radius_m, self.spec.track_width_m
        k = 1.0 - s.slip
        s.v = k * r * 0.5 * (s.wl + s.wr)
        s.omega = k * r * (s.wr - s.wl) / (s.chi * B)
        yaw_mid = s.yaw + 0.5 * s.omega * dt
        v_h = s.v * math.cos(s.pitch)  # horizontal component of the along-body speed
        s.x += v_h * math.cos(yaw_mid) * dt
        s.y += v_h * math.sin(yaw_mid) * dt
        s.yaw_unwrapped += s.omega * dt
        s.yaw = (s.yaw_unwrapped + math.pi) % (2 * math.pi) - math.pi
        s.enc_l += s.wl * dt
        s.enc_r += s.wr * dt
        s.odo_m += abs(s.v) * dt
        s.t += dt
        self._settle()
        return s

    def snapshot(self) -> VehicleState:
        return replace(self.state)


@dataclass
class _Queued:
    apply_step: int
    cmd: WheelCmd


class CommandQueue:
    """Latency injection: a command computed for the frame taken at physics step ``k`` becomes
    active at step ``k + ceil(compute_ms / dt_ms)``.

    Application steps are forced to be non-decreasing (a later command can never overtake an
    earlier one, as on a serial onboard pipeline). The previous command stays active meanwhile.
    """

    def __init__(self, dt: float = 1.0 / defaults.PHYSICS_HZ):
        self.dt_ms = dt * 1000.0
        self._q: deque[_Queued] = deque()
        self._last_apply = -1

    def delay_steps(self, compute_ms: float) -> int:
        return max(0, int(math.ceil(max(float(compute_ms), 0.0) / self.dt_ms - 1e-9)))

    def push(self, cmd: WheelCmd, issue_step: int) -> int:
        apply = max(issue_step + self.delay_steps(cmd.compute_ms), self._last_apply)
        self._last_apply = apply
        self._q.append(_Queued(apply, cmd))
        return apply

    def pop_due(self, step: int) -> WheelCmd | None:
        """Most recent command whose application step is <= ``step`` (older ones are dropped)."""
        out = None
        while self._q and self._q[0].apply_step <= step:
            out = self._q.popleft().cmd
        return out

    def __len__(self) -> int:
        return len(self._q)
