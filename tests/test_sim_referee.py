"""Referee on hand-built mini-worlds: collision, ditch / depression entry, water, tip-over,
stuck, timeout, success, dynamic obstacles, false-stop classification."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metagross.contracts.messages import DriveMode, WheelCmd
from metagross.sim.geometry import GridSpec
from metagross.sim.hazards import gt_hazard_raster
from metagross.sim.miniworld import mini_scenario
from metagross.sim.referee import count_false_stops
from metagross.sim.world import World

NY, NX = 200, 600  # 10 m x 30 m at 0.05 m
FLAT = np.zeros((NY, NX), np.float32)


def _drive(world: World, w: float, seconds: float, wr: float | None = None) -> None:
    """Open-loop drive, re-issuing the command at 5 Hz like the autonomy (the world's actuator
    watchdog stops the wheels after WHEEL_CMD_TIMEOUT_S without a new command)."""
    for k in range(int(seconds * 50)):
        if world.done:
            break
        if k % 10 == 0:
            world.queue_command(WheelCmd(world.t, k, w, w if wr is None else wr, DriveMode.NOMINAL, 0.0))
        world.step()


def test_collision_with_rock_ends_episode():
    rock = {"type": "rock", "xyz": [8.0, 5.0, -0.05], "radius": 0.4, "squash": 0.8, "seed": 1}
    w = World(mini_scenario(FLAT, start=(2.0, 5.0, 0.0), goal=(28.0, 5.0), objects=[rock]))
    _drive(w, 6.0, 20.0)
    assert w.done and w.referee.failure_type == "collision"
    assert w.referee.collisions == 1
    assert w.referee.min_clearance <= 0.0
    # Contact happens when the front bumper (x + 0.4) reaches the rock surface (8 - r_ground).
    assert w.state.x + 0.4 == pytest.approx(8.0 - 0.4 * math.sqrt(1 - (0.05 / 0.32) ** 2), abs=0.05)


def test_passing_beside_rock_reports_positive_clearance():
    rock = {"type": "rock", "xyz": [8.0, 6.0, -0.05], "radius": 0.4, "squash": 0.8, "seed": 1}
    w = World(mini_scenario(FLAT, start=(2.0, 5.0, 0.0), goal=(14.0, 5.0), objects=[rock], success_radius_m=0.5))
    _drive(w, 6.0, 30.0)
    assert w.referee.success, w.referee.events
    # Lateral gap = 1.0 - 0.3 (half width) - ground radius of the rock.
    r_ground = 0.4 * math.sqrt(1 - (0.05 / 0.32) ** 2)
    assert w.referee.min_clearance == pytest.approx(1.0 - 0.3 - r_ground, abs=0.01)


def test_small_rock_below_clearance_is_not_a_collision():
    pebble = {"type": "rock", "xyz": [8.0, 5.0, -0.05], "radius": 0.15, "squash": 0.8, "seed": 1}  # top 0.07 m
    w = World(mini_scenario(FLAT, start=(2.0, 5.0, 0.0), goal=(14.0, 5.0), objects=[pebble], success_radius_m=0.5))
    _drive(w, 6.0, 30.0)
    assert w.referee.success and w.referee.collisions == 0


def test_ditch_entry_detected():
    ditch = {"type": "ditch", "polyline": [[10.0, -1.0], [10.0, 11.0]], "width": 0.6, "depth": 0.5, "gaps": []}
    w = World(mini_scenario(FLAT, start=(2.0, 5.0, 0.0), goal=(28.0, 5.0), hazards=[ditch]))
    _drive(w, 6.0, 30.0)
    assert w.referee.failure_type == "ditch_entry"
    ev = [e for e in w.referee.events if e.type == "ditch_entry"][0]
    assert ev.detail["wheel"] in ("FL", "FR")
    # Front contacts are 0.225 m ahead of the body origin; the lip is at x = 9.7 m.
    assert w.state.x + 0.225 == pytest.approx(9.7, abs=0.08)


def test_depression_rule_without_ditch_polygon():
    h = FLAT.copy()
    g = GridSpec(0.0, 0.0, 0.05, NY, NX)
    xs, ys = g.cell_centres()
    # Unlisted pit 0.4 m long, 0.3 m deep. (A pit wider than half the 1 m median window would pull
    # the median into itself; listed ditches are caught by the ditch mask instead.)
    h[(np.abs(xs - 10.0) < 0.2) & (np.abs(ys - 5.0) < 1.0)] = -0.3
    w = World(mini_scenario(h, start=(2.0, 5.0, 0.0), goal=(28.0, 5.0)))
    assert not w.hazards.ditch.any()
    _drive(w, 6.0, 30.0)
    assert w.referee.failure_type == "ditch_entry"
    ev = [e for e in w.referee.events if e.type == "ditch_entry"][0]
    assert not ev.detail["in_ditch_mask"] and ev.detail["depth_below_median_m"] > 0.15


def test_bypass_gap_is_safe():
    ditch = {"type": "ditch", "polyline": [[10.0, -1.0], [10.0, 11.0]], "width": 0.8, "depth": 0.6, "gaps": [[4.5, 7.5]]}
    w = World(mini_scenario(FLAT, start=(2.0, 5.0, 0.0), goal=(14.0, 5.0), hazards=[ditch], success_radius_m=0.5))
    _drive(w, 6.0, 30.0)  # the straight line y = 5 m passes through the gap (arc 4.5..7.5 m => y 3.5..6.5 m)
    assert w.referee.success and w.referee.ditch_entries == 0


def test_water_entry():
    mat = np.zeros((NY, NX), np.uint8)
    mat[:, 220:260] = 5  # water strip x in [11, 13] m
    w = World(mini_scenario(FLAT, mat, start=(2.0, 5.0, 0.0), goal=(28.0, 5.0)))
    _drive(w, 6.0, 30.0)
    assert w.referee.failure_type == "water_entry" and w.referee.water_entries == 1


def test_tip_over_on_side_slope():
    g = GridSpec(0.0, 0.0, 0.05, NY, NX)
    xs, ys = g.cell_centres()
    ramp = np.clip((xs - 10.0) / 2.0, 0.0, 1.0)
    h = ((ys - 5.0) * math.tan(math.radians(30.0)) * ramp * ramp * (3 - 2 * ramp)).astype(np.float32)
    w = World(mini_scenario(h, start=(2.0, 5.0, 0.0), goal=(28.0, 5.0)))
    _drive(w, 6.0, 30.0)
    assert w.referee.failure_type == "tip_over"


def test_stuck_and_timeout():
    w = World(mini_scenario(FLAT, start=(2.0, 5.0, 0.0), goal=(28.0, 5.0)))
    _drive(w, 0.0, 25.0)
    assert w.referee.failure_type == "stuck" and w.referee.t_end == pytest.approx(20.0, abs=1.01)
    w2 = World(mini_scenario(FLAT, start=(2.0, 5.0, 0.0), goal=(28.0, 5.0), timeout_s=3.0))
    _drive(w2, 1.0, 10.0)
    assert w2.referee.failure_type == "timeout" and w2.t == pytest.approx(3.0, abs=0.021)


def test_arrived_short_after_declared_arrival_outside_radius():
    """ARRIVED declared (mode forwarded via note_command) while GT is 20 m from B -> arrived_short once at rest."""
    w = World(mini_scenario(FLAT, start=(2.0, 5.0, 0.0), goal=(28.0, 5.0)))
    _drive(w, 3.0, 2.0)
    w.referee.note_command(w.t, DriveMode.ARRIVED)
    _drive(w, 0.0, 5.0)
    assert w.referee.failure_type == "arrived_short"
    ev = [e for e in w.referee.events if e.type == "arrived_short"][0]
    assert ev.detail["final_error_m"] > 2.0 and w.referee.t_end < 7.0
    # without a declaration nothing changes (the 'stuck' rule still applies later)
    w2 = World(mini_scenario(FLAT, start=(2.0, 5.0, 0.0), goal=(28.0, 5.0)))
    w2.referee.note_command(0.0, "NOMINAL")
    _drive(w2, 0.0, 5.0)
    assert not w2.done


def test_success_within_radius():
    w = World(mini_scenario(FLAT, start=(2.0, 5.0, 0.0), goal=(6.0, 5.0), success_radius_m=1.0))
    _drive(w, 6.0, 20.0)
    assert w.referee.success
    assert math.hypot(w.state.x - 6.0, w.state.y - 5.0) <= 1.0 + 0.02


def test_dynamic_walker_triggers_and_collides():
    walker = {"type": "walker", "size": [0.5, 0.5, 1.7], "trigger_dist_m": 5.0, "path": [[12.0, 8.0], [12.0, 5.0]], "speed": 1.2}
    w = World(mini_scenario(FLAT, start=(2.0, 5.0, 0.0), goal=(28.0, 5.0), dynamic=[walker]))
    _drive(w, 5.0, 30.0)
    types = [e.type for e in w.referee.events]
    assert "dynamic_trigger" in types
    trig = [e for e in w.referee.events if e.type == "dynamic_trigger"][0]
    assert 12.0 - (w.gt_log()["x"][int(trig.t * 30)]) == pytest.approx(5.0, abs=0.3)
    assert w.referee.failure_type == "collision"
    assert [e for e in w.referee.events if e.type == "collision"][0].detail["object"].startswith("dyn_walker")


def test_false_stop_classification():
    ditch = {"type": "ditch", "polyline": [[10.0, -1.0], [10.0, 11.0]], "width": 0.6, "depth": 0.5, "gaps": []}
    scn = mini_scenario(FLAT, start=(2.0, 5.0, 0.0), goal=(28.0, 5.0), hazards=[ditch])
    hz = gt_hazard_raster(scn)
    t = np.arange(0, 20, 1 / 30)
    x = np.full_like(t, 3.0)
    v = np.full_like(t, 0.5)
    v[(t > 2) & (t < 4)] = 0.0  # stop at x = 3 m: nothing within 5 m ahead -> false stop
    x[t >= 10] = 7.0
    v[(t > 12) & (t < 14)] = 0.0  # stop at x = 7 m: ditch 2.7 m ahead -> justified
    n, stops = count_false_stops(t, x, np.full_like(t, 5.0), np.zeros_like(t), v, hz, (28.0, 5.0), 2.0)
    assert n == 1 and len(stops) == 2
    assert stops[0]["justified"] is False and stops[1]["reason"] == "hazard_ahead"
