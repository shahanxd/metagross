"""Closed-loop toy tests of the full AutonomyStack (planning + control + safety wiring).

A tiny kinematic world (unicycle with first-order actuator lag) is driven by the real
node. Perception is replaced by a synthetic oracle BEV (camera FOV + range limits over a
truth grid of CellStates) and localisation by perfect odometry. These stand-ins live
here, in tests, so the autonomy package never touches anything resembling ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from metagross.autonomy.control.mixer import wheels_to_body
from metagross.autonomy.node import AutonomyStack
from metagross.autonomy.planning.rolling_map import BevGeometry
from metagross.config.defaults import ACTUATOR_LAG_S, CAM_FORWARD_M, CHI_NOMINAL, HFOV_DEG, VEHICLE, stereo_calibration
from metagross.contracts.messages import CellState, DriveMode, MissionSpec, SensorFrame

TRUTH_RES = 0.1


@dataclass
class ToyState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    w: float = 0.0
    wl: float = 0.0  # cumulative wheel angles [rad]
    wr: float = 0.0


class ToyWorld:
    """Truth grid of CellStates (A-frame, 0.1 m, origin at (x0, y0)) + the kinematic vehicle."""

    def __init__(self, size_xy=(40.0, 30.0), origin=(-10.0, -15.0)) -> None:
        nx, ny = int(size_xy[0] / TRUTH_RES), int(size_xy[1] / TRUTH_RES)
        self.origin = origin
        self.truth = np.full((ny, nx), int(CellState.GROUND), np.uint8)  # [iy, ix]
        self.s = ToyState()
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
        self.X = origin[0] + (ix + 0.5) * TRUTH_RES
        self.Y = origin[1] + (iy + 0.5) * TRUTH_RES
        self.bands: list[tuple[str, float, float, int, tuple[float, float] | None]] = []  # infinite bands

    def add_disc(self, cx, cy, r, state=CellState.POSITIVE) -> None:
        self.truth[(self.X - cx) ** 2 + (self.Y - cy) ** 2 <= r * r] = int(state)

    def add_band_x(self, x0, x1, state=CellState.DITCH_CANDIDATE, gap_y=None) -> None:
        """Band x0 <= x <= x1 of ``state``, unbounded in y (also outside the truth grid),
        optionally with a traversable gap gap_y = (y0, y1)."""
        self.bands.append(("x", x0, x1, int(state), gap_y))

    def add_band_y(self, y0, y1, state=CellState.POSITIVE) -> None:
        """Band y0 <= y <= y1 of ``state``, unbounded in x (a wall along the x axis)."""
        self.bands.append(("y", y0, y1, int(state), None))

    def lookup(self, x, y) -> np.ndarray:
        ix = np.floor((x - self.origin[0]) / TRUTH_RES).astype(int)
        iy = np.floor((y - self.origin[1]) / TRUTH_RES).astype(int)
        ok = (ix >= 0) & (ix < self.truth.shape[1]) & (iy >= 0) & (iy < self.truth.shape[0])
        out = np.full(np.shape(x), int(CellState.GROUND), np.uint8)
        out[ok] = self.truth[iy[ok], ix[ok]]
        for axis, a0, a1, st, gap in self.bands:
            c = x if axis == "x" else y
            m = (c >= a0) & (c <= a1)
            if gap is not None:
                m &= ~((y >= gap[0]) & (y <= gap[1]))
            out[m] = st
        return out

    def step(self, omega_l, omega_r, dt, substeps=10) -> None:
        v_cmd, w_cmd = wheels_to_body(omega_l, omega_r, CHI_NOMINAL, VEHICLE.track_width_m, VEHICLE.wheel_radius_m)
        h = dt / substeps
        s = self.s
        for _ in range(substeps):
            a = h / ACTUATOR_LAG_S
            s.v += a * (v_cmd - s.v)
            s.w += a * (w_cmd - s.w)
            s.x += s.v * math.cos(s.yaw + 0.5 * s.w * h) * h
            s.y += s.v * math.sin(s.yaw + 0.5 * s.w * h) * h
            s.yaw += s.w * h
            half = s.w * CHI_NOMINAL * VEHICLE.track_width_m / 2.0
            s.wl += (s.v - half) / VEHICLE.wheel_radius_m * h
            s.wr += (s.v + half) / VEHICLE.wheel_radius_m * h

    def footprint_states(self) -> np.ndarray:
        """Truth states under the vehicle rectangle (0.8 x 0.6 m) sampled at 5 cm."""
        s = self.s
        u, v = np.meshgrid(np.linspace(-VEHICLE.length_m / 2, VEHICLE.length_m / 2, 17), np.linspace(-VEHICLE.width_m / 2, VEHICLE.width_m / 2, 13))
        c, sn = math.cos(s.yaw), math.sin(s.yaw)
        return self.lookup(s.x + c * u - sn * v, s.y + sn * u + c * v)


class OracleBevPerception:
    """Synthetic perception: truth states inside the camera footprint (72 deg HFOV,
    1.5..10 m from the camera), UNSEEN elsewhere. Uses perception's BEV convention."""

    def __init__(self, world: ToyWorld, r_min=1.5, r_max=10.0) -> None:
        self.world = world
        self.geo = BevGeometry()
        xb, yb = self.geo.cell_centres_body()
        rng = np.hypot(xb - CAM_FORWARD_M, yb)
        bearing = np.arctan2(yb, xb - CAM_FORWARD_M)
        self.visible = (rng >= r_min) & (rng <= r_max) & (np.abs(bearing) <= math.radians(HFOV_DEG / 2))
        self.xb, self.yb = xb, yb
        self.r_max = r_max

    def process(self, frame, pose):
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        xa = x + c * self.xb - s * self.yb
        ya = y + s * self.xb + c * self.yb
        states = np.where(self.visible, self.world.lookup(xa, ya), int(CellState.UNSEEN)).astype(np.uint8)
        cost = np.where(states == CellState.GROUND, 0.0, 1.0).astype(np.float32)
        return {"disparity": None, "cell_state_local": states, "cost_local": cost, "height_local": None, "semantic_mask": None,
                "missing_ground_mask": None, "r_vis_m": self.r_max, "timings_ms": {}}


class PerfectOdomLocalizer:
    """Perfect odometry (reads the toy vehicle state). Test-only."""

    def __init__(self, world: ToyWorld) -> None:
        self.world = world

    def update(self, frame, disparity):
        s = self.world.s
        return {"pose_xy_yaw": (s.x, s.y, s.yaw), "pos_sigma_m": 0.0, "health": {"p_fail": 0.0}, "vo_ok": True,
                "slip": 0.0, "chi_hat": CHI_NOMINAL, "timings_ms": {}}


def run_loop(world: ToyWorld, goal, config=None, t_max=40.0, dt=0.2):
    """Drive the stack in the toy world. Returns (stack, log dict)."""
    stack = AutonomyStack(perception=OracleBevPerception(world), localizer=PerfectOdomLocalizer(world))
    cfg = {"fixed_latency_s": 0.05, **(config or {})}  # deterministic governor reaction time
    stack.reset(MissionSpec("toy", goal, 2.0, t_max), stereo_calibration(), VEHICLE, cfg)
    log = {"xy": [], "mode": [], "v": [], "compute_ms": [], "timings": [], "foot": [], "tel": 0, "reason": []}
    for k in range(int(t_max / dt)):
        s = world.s
        frame = SensorFrame(t=k * dt, seq=k, left_rgb=None, right_gray=None, wheel_angle_l_rad=s.wl, wheel_angle_r_rad=s.wr,
                            gyro_z_rps=s.w, sensor_mode="tier0_disparity", disparity=None)
        cmd, tel, dbg = stack.step(frame)
        world.step(cmd.omega_l_rad_s, cmd.omega_r_rad_s, dt)
        log["xy"].append((world.s.x, world.s.y))
        log["mode"].append(cmd.mode)
        log["v"].append(world.s.v)
        log["compute_ms"].append(cmd.compute_ms)
        log["timings"].append(dbg.timings_ms)
        log["foot"].append(world.footprint_states())
        log["reason"].append(dbg.extras["reason"])
        log["tel"] += tel is not None
        if cmd.mode in (DriveMode.ARRIVED, DriveMode.SAFE_STOP) and abs(world.s.v) < 1e-3:
            break
    return stack, log


def test_drives_to_goal_around_obstacle():
    world = ToyWorld()
    world.add_disc(7.0, 0.0, 0.8)  # rock straight on the line A->B
    goal = (14.0, 0.0)
    _, log = run_loop(world, goal, t_max=40.0)
    xy = np.array(log["xy"])
    assert log["mode"][-1] == DriveMode.ARRIVED, f"final mode {log['mode'][-1]}, reasons {log['reason'][-5:]}"
    assert math.hypot(xy[-1, 0] - goal[0], xy[-1, 1] - goal[1]) <= 2.0
    # never any footprint overlap with the rock, and it really went around it
    assert not any(np.any(f == CellState.POSITIVE) for f in log["foot"])
    near = xy[np.abs(xy[:, 0] - 7.0) < 0.5]
    assert len(near) and np.min(np.abs(near[:, 1])) > 0.8 + VEHICLE.width_m / 2 - 0.05
    assert max(log["v"]) > 0.8  # it actually drove (not a crawl)
    assert log["tel"] >= 2


def test_stops_before_ditch_candidate_band():
    """Enclosed corridor (walls at |y| = 3 m and behind the start) whose only exit toward the
    goal crosses a missing-ground band: the vehicle must stop short of it, STOP_AND_LOOK, and
    finally SAFE_STOP, without the footprint ever touching the band."""
    world = ToyWorld()
    world.add_band_x(6.0, 6.6, CellState.DITCH_CANDIDATE)  # full-width missing-ground band
    world.add_band_y(3.0, 3.4)
    world.add_band_y(-3.4, -3.0)
    world.add_band_x(-2.4, -2.0, CellState.POSITIVE)
    goal = (14.0, 0.0)
    stack, log = run_loop(world, goal, config={"dead_end_memory": False}, t_max=60.0)
    assert not any(np.any(f == CellState.DITCH_CANDIDATE) for f in log["foot"]), "vehicle drove into the ditch"
    xy = np.array(log["xy"])
    front = xy[:, 0] + VEHICLE.length_m / 2
    assert front.max() < 6.0
    assert front.max() > 3.0, "never approached the ditch"
    modes = set(log["mode"])
    assert DriveMode.STOP_AND_LOOK in modes
    assert log["mode"][-1] == DriveMode.SAFE_STOP and abs(log["v"][-1]) < 0.05


def test_dead_end_memory_in_enclosed_corridor():
    """Same enclosed corridor with the dead-end memory on (default): the stalls are recorded as
    blocked discs, the footprint never touches the band, and with no way out the supervisor
    finally latches SAFE_STOP."""
    world = ToyWorld()
    world.add_band_x(6.0, 6.6, CellState.DITCH_CANDIDATE)
    world.add_band_y(3.0, 3.4)
    world.add_band_y(-3.4, -3.0)
    world.add_band_x(-2.4, -2.0, CellState.POSITIVE)
    goal = (14.0, 0.0)
    stack, log = run_loop(world, goal, t_max=200.0)
    assert not any(np.any(f == CellState.DITCH_CANDIDATE) for f in log["foot"]), "vehicle drove into the ditch"
    xy = np.array(log["xy"])
    assert (xy[:, 0] + VEHICLE.length_m / 2).max() < 6.0
    assert stack.dead_ends is not None and stack.dead_ends.n_marked >= 1
    assert any(r.startswith("DEAD_END") for r in log["reason"])
    assert log["mode"][-1] == DriveMode.SAFE_STOP and abs(log["v"][-1]) < 0.05


def test_finds_gap_in_ditch():
    """F2-like: a full-width ditch with a 2.5 m gap off the direct line; the vehicle must go
    through the gap without the footprint ever touching the ditch."""
    world = ToyWorld()
    world.add_band_x(6.0, 6.6, CellState.DITCH_CANDIDATE, gap_y=(2.5, 5.0))
    goal = (14.0, 0.0)
    _, log = run_loop(world, goal, t_max=45.0)
    assert not any(np.any(f == CellState.DITCH_CANDIDATE) for f in log["foot"]), "vehicle drove into the ditch"
    assert log["mode"][-1] == DriveMode.ARRIVED, f"final mode {log['mode'][-1]}, reasons {log['reason'][-5:]}"


def test_blind_vehicle_never_leaves_launch_apron():
    """Unknown is never free: with a perception that sees nothing, the vehicle may only use
    the certified launch apron and must stop inside it."""
    from metagross.autonomy.node_stubs import StubBlindPerception
    from metagross.autonomy.planning.rolling_map import LAUNCH_APRON_M

    world = ToyWorld()
    stack = AutonomyStack(perception=StubBlindPerception(), localizer=PerfectOdomLocalizer(world))
    stack.reset(MissionSpec("blind", (14.0, 0.0), 2.0, 30.0), stereo_calibration(), VEHICLE, {})
    reach = []
    for k in range(100):
        s = world.s
        frame = SensorFrame(t=0.2 * k, seq=k, left_rgb=None, right_gray=None, wheel_angle_l_rad=s.wl, wheel_angle_r_rad=s.wr,
                            gyro_z_rps=s.w)
        cmd, _, _ = stack.step(frame)
        world.step(cmd.omega_l_rad_s, cmd.omega_r_rad_s, 0.2)
        c, sn = math.cos(world.s.yaw), math.sin(world.s.yaw)
        reach.append(max(math.hypot(world.s.x + d * c, world.s.y + d * sn) for d in (-0.4, 0.4)))
    assert max(reach) <= LAUNCH_APRON_M
    assert abs(world.s.v) < 0.05


def test_typical_stack_ablation_enters_ditch():
    """Sanity check of the ablation switches: unknown-is-free + no negative-obstacle
    detector + no governor + fixed cruise speed drives straight into the same ditch."""
    world = ToyWorld()
    world.add_band_x(6.0, 6.6, CellState.DITCH_CANDIDATE)
    cfg = {"unknown_is_free": True, "use_negobs": False, "use_governor": False, "fixed_speed_mps": 1.5}
    _, log = run_loop(world, (14.0, 0.0), config=cfg, t_max=15.0)
    assert any(np.any(f == CellState.DITCH_CANDIDATE) for f in log["foot"])


@pytest.mark.parametrize("hz", [5.0])
def test_node_loop_rate(hz):
    """Planning+control stack (excluding real perception/localisation) must leave most of
    the 200 ms budget of the 5 Hz loop to perception."""
    world = ToyWorld()
    world.add_disc(7.0, 0.5, 0.6)
    _, log = run_loop(world, (14.0, 0.0), t_max=12.0, dt=1.0 / hz)
    ms = np.array(log["compute_ms"][3:])
    assert np.median(ms) < 80.0, f"median compute {np.median(ms):.1f} ms"
