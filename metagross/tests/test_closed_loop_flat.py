"""Regression: the real AutonomyStack (real perception + localiser, tier-0 depth sensor, in-process
lockstep with the World incl. latency injection) crosses a flat, empty world at >= 80 % of its
speed cap, for both the FULL stack and the TYPICAL baseline.

Before the build-phase-II closed-loop fixes both crawled (DEV F1: 0.01-0.19 m/s mean) because the
MPPI nominal got stuck turning the wrong way after a route swing and the supervisor's 3 s
no-progress clock fired STOP_AND_LOOK mid-turn.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from metagross.autonomy.node import AutonomyStack
from metagross.config import defaults
from metagross.contracts import ipc
from metagross.contracts.messages import OperatorAction, OperatorCmd
from metagross.sim.miniworld import mini_scenario
from metagross.sim.world import World

FLAT_LEN_M = 34.0  # world length along x [m]
FLAT_WID_M = 10.0
START = (2.0, 5.0, 0.0)
GOAL = (30.0, 5.0)
CRUISE_X = (12.0, 26.0)  # GT x window [m] over which cruise speed is measured (after the launch ramp)
MIN_SPEED_FRACTION = 0.8
FIXED_LATENCY_MS = 100.0  # compute latency injected as actuation delay and used by the governor [ms]

CONFIGS = {
    "FULL": {},
    "TYPICAL": {"name": "TYPICAL", "unknown_is_free": True, "use_negobs": False, "use_governor": False, "fixed_speed_mps": 1.5},
}


def _run(config: dict, t_max: float = 45.0) -> tuple[World, list[tuple[float, float, float]]]:
    """Lockstep loop; returns the world and per-tick (t, gt_x, v_cap_eff). Compute latency is
    fixed (governor and actuation) so the test is deterministic on a loaded machine."""
    h = np.zeros((int(FLAT_WID_M / 0.05), int(FLAT_LEN_M / 0.05)), np.float32)
    scn = mini_scenario(h, start=START, goal=GOAL, timeout_s=t_max, slip=0.0, chi=defaults.CHI_NOMINAL, seed=9101)
    w = World(scn, sensor_mode="tier0")
    st = AutonomyStack()
    cfg = {**ipc.DEFAULT_AUTONOMY_CONFIG, **config, "use_semantics": False, "fixed_latency_s": FIXED_LATENCY_MS / 1e3}
    st.reset(w.mission_spec(), w.calibration(), defaults.VEHICLE, cfg)
    st.operator(OperatorCmd(t=0.0, action=OperatorAction.GO))
    spf = int(round(defaults.PHYSICS_HZ / defaults.CAMERA_HZ_BATCH))
    log, seq = [], 0
    while not w.done:
        if w.step_index % spf == 0:
            cmd, _, dbg = st.step(w.make_sensor_frame(w.t, seq))
            w.queue_command(replace(cmd, compute_ms=FIXED_LATENCY_MS))
            log.append((w.t, w.state.x, float(dbg.v_cap_mps)))
            seq += 1
        w.step()
    return w, log


@pytest.mark.parametrize("name", list(CONFIGS))
def test_crosses_flat_world_at_speed_cap(name):
    w, log = _run(CONFIGS[name])
    assert w.referee.success, f"{name}: {w.referee.failure_type}"
    assert w.referee.collisions == 0
    gt = w.gt_log()
    t, x = gt["t"], gt["x"]
    assert x.max() >= CRUISE_X[1], "never crossed the cruise window"
    k0, k1 = int(np.argmax(x >= CRUISE_X[0])), int(np.argmax(x >= CRUISE_X[1]))
    v_mean = (x[k1] - x[k0]) / (t[k1] - t[k0])
    caps = [c for (tt, xx, c) in log if CRUISE_X[0] <= xx <= CRUISE_X[1]]
    cap = float(np.median(caps))
    print(f"{name}: cruise {v_mean:.3f} m/s, median cap {cap:.3f} m/s, t_end {w.t:.1f} s")
    assert cap > 1.0, f"{name}: speed cap itself collapsed ({cap:.2f} m/s)"
    assert v_mean >= MIN_SPEED_FRACTION * cap, f"{name}: {v_mean:.2f} m/s < {MIN_SPEED_FRACTION} x cap {cap:.2f} m/s"
