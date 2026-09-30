"""Skid-steer kinematics, actuator lag, slip, terrain following and latency injection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metagross.config import defaults
from metagross.contracts.messages import DriveMode, WheelCmd
from metagross.sim.geometry import GridSpec
from metagross.sim.miniworld import mini_scenario
from metagross.sim.terrain import Terrain
from metagross.sim.vehicle import WHEELBASE_M, CommandQueue, SkidSteerVehicle
from metagross.sim.world import World

R = defaults.VEHICLE.wheel_radius_m
B = defaults.VEHICLE.track_width_m
DT = 1.0 / defaults.PHYSICS_HZ


def _terrain(h: np.ndarray | None = None, mat: np.ndarray | None = None, shape=(400, 400)) -> Terrain:
    g = GridSpec(0.0, 0.0, 0.05, *shape)
    h = np.zeros(shape, np.float32) if h is None else h
    m = np.zeros(shape, np.uint8) if mat is None else mat
    return Terrain(g, h, m)


def _vehicle(terrain: Terrain, chi: dict | None = None, slip: float = 0.05) -> SkidSteerVehicle:
    chi = chi or {str(k): 1.5 for k in range(6)}
    v = SkidSteerVehicle(terrain, {"chi_by_material": chi, "slip_long": slip})
    v.reset(5.0, 10.0, 0.0)
    return v


def _run(v: SkidSteerVehicle, wl: float, wr: float, seconds: float) -> None:
    v.set_command(wl, wr)
    for _ in range(int(round(seconds / DT))):
        v.step()


def test_straight_line_speed_includes_slip():
    v = _vehicle(_terrain(), slip=0.05)
    w = 5.0  # rad/s -> 0.65 m/s wheel surface speed
    _run(v, w, w, 5.0)
    s = v.state
    assert s.v == pytest.approx((1 - 0.05) * R * w, rel=1e-3)
    assert s.yaw == pytest.approx(0.0, abs=1e-12)
    assert s.y == pytest.approx(10.0, abs=1e-9)
    # Encoders count wheel rotation, not ground travel: ground = (1 - s) * r * angle.
    assert s.x - 5.0 == pytest.approx((1 - 0.05) * R * s.enc_l, rel=1e-3)


@pytest.mark.parametrize("chi", [1.3, 1.7])
def test_turn_in_place_yaw_rate_uses_true_chi(chi):
    v = _vehicle(_terrain(), chi={str(k): chi for k in range(6)}, slip=0.04)
    w = 4.0
    _run(v, -w, w, 3.0)
    expected = (1 - 0.04) * R * (2 * w) / (chi * B)
    assert v.state.omega == pytest.approx(expected, rel=1e-3)
    assert abs(v.state.v) < 1e-9


def test_chi_is_sampled_per_material():
    mat = np.zeros((400, 400), np.uint8)
    mat[:, 200:] = 2  # gravel east of x = 10 m
    t = _terrain(mat=mat)
    v = _vehicle(t, chi={"0": 1.6, "1": 1.5, "2": 1.3, "3": 1.4, "4": 1.7, "5": 1.7})
    _run(v, -3.0, 3.0, 2.0)
    om_grass = v.state.omega
    v.reset(15.0, 10.0, 0.0)
    _run(v, -3.0, 3.0, 2.0)
    assert v.state.material == 2
    assert v.state.omega / om_grass == pytest.approx(1.6 / 1.3, rel=1e-3)


def test_first_order_lag_time_constant():
    v = _vehicle(_terrain())
    cmd = 0.5  # small enough that the acceleration limit never binds
    v.set_command(cmd, cmd)
    n = int(round(defaults.ACTUATOR_LAG_S / DT))
    for _ in range(n):
        v.step()
    assert v.state.wl == pytest.approx(cmd * (1 - math.exp(-1.0)), rel=1e-6)


def test_acceleration_limit():
    v = _vehicle(_terrain())
    v.set_command(20.0, 20.0)
    prev = 0.0
    for _ in range(20):
        v.step()
        assert v.state.wl - prev <= defaults.VEHICLE.max_accel_mps2 / R * DT + 1e-9
        prev = v.state.wl


def test_command_clamped_to_max_wheel_speed():
    v = _vehicle(_terrain())
    v.set_command(1e3, -1e3)
    assert v.state.cmd_l == pytest.approx(defaults.VEHICLE.max_wheel_rad_s)
    assert v.state.cmd_r == pytest.approx(-defaults.VEHICLE.max_wheel_rad_s)


def test_terrain_following_pitch_and_roll():
    g = GridSpec(0.0, 0.0, 0.05, 400, 400)
    xs, ys = g.cell_centres()
    slope = math.tan(math.radians(10.0))
    v = _vehicle(_terrain(h=(xs * slope).astype(np.float32)))
    v.reset(10.0, 10.0, 0.0)  # facing uphill (+x)
    assert v.state.pitch == pytest.approx(-math.radians(10.0), abs=1e-3)  # nose up = negative pitch
    assert v.state.roll == pytest.approx(0.0, abs=1e-6)
    assert v.state.z == pytest.approx(10.0 * slope, abs=1e-3)
    v.reset(10.0, 10.0, math.pi / 2)  # facing north: the slope rises to the right -> left side down
    assert v.state.roll == pytest.approx(-math.radians(10.0), abs=1e-3)
    assert WHEELBASE_M > 0


def _cmd(seq: int, compute_ms: float, w: float = 1.0) -> WheelCmd:
    return WheelCmd(t=0.0, seq=seq, omega_l_rad_s=w, omega_r_rad_s=w, mode=DriveMode.NOMINAL, compute_ms=compute_ms)


def test_command_queue_rounds_up_to_physics_steps():
    q = CommandQueue(DT)
    assert q.delay_steps(0.0) == 0
    assert q.delay_steps(20.0) == 1
    assert q.delay_steps(20.1) == 2
    assert q.delay_steps(45.0) == 3
    assert q.push(_cmd(0, 45.0), issue_step=10) == 13
    assert q.pop_due(12) is None
    assert q.pop_due(13).seq == 0


def test_command_queue_never_overtakes():
    q = CommandQueue(DT)
    a = q.push(_cmd(0, 150.0), issue_step=0)  # slow tick
    b = q.push(_cmd(1, 5.0), issue_step=5)  # fast tick right after
    assert (a, b) == (8, 8)
    got = q.pop_due(8)
    assert got.seq == 1 and len(q) == 0  # both due: only the latest is applied


def test_world_applies_command_after_injected_latency():
    h = np.zeros((200, 400), np.float32)
    w = World(mini_scenario(h, start=(2.0, 5.0, 0.0), goal=(18.0, 5.0)))
    w.queue_command(_cmd(0, 45.0, w=3.0))  # 45 ms -> 3 physics steps
    seen = []
    for _ in range(5):
        w.step()
        seen.append(w.state.cmd_l)
    # Frame taken at step 0; 45 ms -> 3 steps -> active for the integration starting at t = 60 ms.
    assert seen == [0.0, 0.0, 0.0, 3.0, 3.0]
