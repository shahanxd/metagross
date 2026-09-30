"""Integration fixes of build phase II (see docs/BUILD_LOG.md):

* Tier-0 frames (no images): VO unavailable is not a VO fault -> no HEALTH SAFE_STOP;
* world-side wheel-command watchdog;
* launch apron is assumed ground, not evidence for DYNAMIC detection;
* local terrain level: rolling terrain is not a field of POSITIVE / DEPRESSION cells, rocks still are;
* segmenter at 1/3 camera rate; debug image + measured speed in the DebugBundle;
* contract constant additions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from metagross.autonomy.localization.localizer import Localizer
from metagross.autonomy.node import DEBUG_IMAGE_WH, EveryNthSegmenter, AutonomyStack
from metagross.autonomy.perception.bev import BevSpec, accumulate
from metagross.autonomy.perception.positive import count_outside, terrain_costs, terrain_level
from metagross.autonomy.planning.rolling_map import RollingMap
from metagross.config import defaults
from metagross.contracts import ipc
from metagross.contracts.messages import COSTMAP_U4_CODES, CellState, DriveMode, MissionSpec, SensorFrame, WheelCmd
from metagross.sim.miniworld import mini_scenario
from metagross.sim.world import World
from test_node_closed_loop import OracleBevPerception, ToyWorld  # noqa: E402 - pytest puts tests/ on sys.path

CAL = defaults.stereo_calibration()


# ----------------------------------------------------------------------------- health / tier0
def test_tier0_without_images_is_not_a_vo_fault():
    """Real Localizer on image-less Tier-0 frames: vo_available = 0, the stack keeps q = 1,
    never SAFE_STOPs on health and actually drives."""
    world = ToyWorld()
    stack = AutonomyStack(perception=OracleBevPerception(world), localizer=Localizer(CAL, defaults.VEHICLE))
    stack.reset(MissionSpec("t0", (14.0, 0.0), 2.0, 30.0), CAL, defaults.VEHICLE, {"fixed_latency_s": 0.05})
    modes, reasons = [], []
    for k in range(75):
        s = world.s
        fr = SensorFrame(t=0.2 * k, seq=k, left_rgb=None, right_gray=None, wheel_angle_l_rad=s.wl, wheel_angle_r_rad=s.wr,
                         gyro_z_rps=s.w, sensor_mode="tier0_disparity", disparity=None)
        cmd, _, dbg = stack.step(fr)
        world.step(cmd.omega_l_rad_s, cmd.omega_r_rad_s, 0.2)
        modes.append(cmd.mode)
        reasons.append(dbg.extras["reason"])
        assert dbg.health["vo_available"] == 0.0 and dbg.health["q"] == 1.0
    assert DriveMode.SAFE_STOP not in modes, reasons
    assert world.s.x > 5.0


def test_stereo_frame_keeps_integrity_gating():
    """With images present the localiser's q is used as is (here: black images -> low q)."""
    loc = Localizer(CAL, defaults.VEHICLE)
    blank = np.zeros((CAL.height, CAL.width, 3), np.uint8)
    for k in range(3):
        out = loc.update(SensorFrame(0.2 * k, k, blank, blank[..., 0], 0.0, 0.0, 0.0, "stereo"), None)
    assert out["health"]["vo_available"] == 1.0
    assert out["health"]["q"] < 0.7


# ----------------------------------------------------------------------------- world watchdog
def test_world_wheel_command_timeout():
    w = World(mini_scenario(np.zeros((200, 400), np.float32), start=(2.0, 5.0, 0.0), goal=(18.0, 5.0)))
    w.queue_command(WheelCmd(0.0, 0, 5.0, 5.0, DriveMode.NOMINAL, 0.0))
    cmd_l = []
    for _ in range(40):
        w.step()
        cmd_l.append(w.state.cmd_l)
    # applied from the first step; stopped once 0.5 s passed without a new command
    assert cmd_l[5] == 5.0
    n_ok = int(round(defaults.WHEEL_CMD_TIMEOUT_S * defaults.PHYSICS_HZ))
    assert cmd_l[n_ok + 3] == 0.0 and w.n_cmd_timeouts == 1


# ----------------------------------------------------------------------------- rolling map apron
def test_obstacle_inside_launch_apron_is_static_not_dynamic():
    rm = RollingMap()
    rm.seed_apron((0.0, 0.0, 0.0), 0.0, 2.0)
    spec = BevSpec()
    st = np.zeros(spec.shape, np.uint8)
    X, Y = spec.centres()
    st[(np.abs(X - 1.5) < 0.15) & (np.abs(Y) < 0.15)] = CellState.POSITIVE
    rm.integrate(0.2, (0.0, 0.0, 0.0), st)
    rm.integrate(0.4, (0.0, 0.0, 0.0), st)  # lethal needs support from 2 frames
    ix, iy = rm.geometry.to_index(np.array([1.5]), np.array([0.0]))
    assert rm.state[iy[0], ix[0]] == CellState.POSITIVE
    assert not np.any(rm.state == CellState.DYNAMIC)


# ----------------------------------------------------------------------------- terrain level
def _synthetic_points(rock: bool, seed: int = 0):
    """Rolling terrain (0.3 m amplitude, 8 m wavelength along x and y) sampled densely, heights
    relative to a flat 'band model' z = 0, optionally with a 0.4 m tall, 0.6 m wide rock at (5, 1)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(1.5, 10.0, 60000)
    y = rng.uniform(-4.0, 4.0, 60000)
    z = 0.3 * np.sin(2 * math.pi * x / 8.0) * np.cos(2 * math.pi * y / 8.0) + rng.normal(0, 0.01, x.size)
    if rock:
        m = np.hypot(x - 5.0, y - 1.0) < 0.3
        z[m] += 0.4
    return x, y, z


@pytest.mark.parametrize("rock", [False, True])
def test_rolling_terrain_not_lethal_but_rock_is(rock):
    spec = BevSpec()
    x, y, z = _synthetic_points(rock)
    stats = accumulate(spec, x, y, z, z)
    gz = np.zeros(spec.shape, np.float32)
    lvl = terrain_level(stats, spec, gz)
    n_up, n_dn = count_outside(spec, x, y, z, lvl, defaults.STEP_LETHAL_M, defaults.DEPRESSION_LETHAL_M)
    tc = terrain_costs(stats, spec, gz, stats.count > 0, lvl, n_up)
    X, Y = spec.centres()
    rock_cells = np.hypot(X - 5.0, Y - 1.0) < 0.2
    far_from_rock = np.hypot(X - 5.0, Y - 1.0) > 1.0
    assert int((tc.positive & far_from_rock).sum()) == 0
    assert int((n_dn >= 2).sum()) == 0
    if rock:
        assert tc.positive[rock_cells].mean() > 0.8


# ----------------------------------------------------------------------------- semantics rate / debug
def test_segmenter_runs_every_third_frame():
    calls = []

    def seg(rgb):
        calls.append(1)
        return np.full(rgb.shape[:2], len(calls), np.uint8), np.zeros(rgb.shape[:2], np.float32)

    s = EveryNthSegmenter(seg, 3)
    img = np.zeros((4, 6, 3), np.uint8)
    outs = [int(s(img)[0][0, 0]) for _ in range(7)]
    assert outs == [1, 1, 1, 2, 2, 2, 3] and s.runs == 3 and s.calls == 7


def test_debug_bundle_left_image_and_speed():
    world = ToyWorld()
    stack = AutonomyStack(perception=OracleBevPerception(world), localizer=Localizer(CAL, defaults.VEHICLE))
    stack.reset(MissionSpec("dbg", (14.0, 0.0), 2.0, 30.0), CAL, defaults.VEHICLE, {"fixed_latency_s": 0.05, "debug_image_every_n": 2})
    img = np.full((CAL.height, CAL.width, 3), 90, np.uint8)
    have = []
    for k in range(4):
        fr = SensorFrame(0.2 * k, k, img, img[..., 0], 0.0, 0.0, 0.0, "stereo")
        _, _, dbg = stack.step(fr)
        have.append("left_rgb" in dbg.extras)
        assert "speed_meas" in dbg.extras
    assert have == [True, False, True, False]
    assert dbg.extras["tick"] == 3
    _, _, dbg = stack.step(SensorFrame(0.8, 4, img, img[..., 0], 0.0, 0.0, 0.0, "stereo"))
    assert dbg.extras["left_rgb"].shape == (DEBUG_IMAGE_WH[1], DEBUG_IMAGE_WH[0], 3)


def test_localizer_receives_perceptions_disparity_before_pose_dependent_perception():
    world = ToyWorld()
    order = []
    marker = np.full((4, 4), 7.0, np.float32)

    class SpyPerception(OracleBevPerception):
        def compute_disparity(self, frame):
            order.append("stereo")
            return marker

        def process(self, frame, pose):
            order.append(("process", tuple(round(v, 6) for v in pose)))
            return super().process(frame, pose)

    class SpyLocalizer:
        def update(self, frame, disparity):
            order.append(("loc", disparity is marker))
            s = world.s
            return {"pose_xy_yaw": (s.x, s.y, s.yaw), "pos_sigma_m": 0.0, "health": {"p_fail": 0.0}, "vo_ok": True,
                    "slip": 0.0, "chi_hat": 1.4, "timings_ms": {}}

    stack = AutonomyStack(perception=SpyPerception(world), localizer=SpyLocalizer())
    stack.reset(MissionSpec("ord", (14.0, 0.0), 2.0, 30.0), CAL, defaults.VEHICLE, {"fixed_latency_s": 0.05})
    world.s.x = 1.25
    stack.step(SensorFrame(0.0, 0, None, None, 0.0, 0.0, 0.0, "tier0_disparity"))
    assert order == ["stereo", ("loc", True), ("process", (1.25, 0.0, 0.0))]


# ----------------------------------------------------------------------------- contract constants
def test_contract_constant_additions():
    for k, v in {"seg_model_path": None, "seg_threads": 2, "launch_apron_m": 2.0, "seed": 0, "fixed_latency_s": None,
                 "stub_perception": False, "stub_localizer": False, "debug_image_every_n": 2}.items():
        assert ipc.DEFAULT_AUTONOMY_CONFIG[k] == v
    assert (defaults.BEV_NX, defaults.BEV_NY) == BevSpec().shape
    assert defaults.BEV_LOCAL_X_MIN_M == BevSpec().x_min and defaults.BEV_LOCAL_Y_MIN_M == BevSpec().y_min
    assert set(COSTMAP_U4_CODES) == set(range(16))
