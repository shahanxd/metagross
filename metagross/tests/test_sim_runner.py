"""End-to-end runner tests with a tiny dummy autonomy defined here (not metagross.autonomy).

The dummy uses ONLY what a real onboard stack gets: MissionSpec + SensorFrame (wheel encoders
and gyro). It dead-reckons in the A-frame and steers proportionally towards the goal.
Run through a normal Python (``python -m pytest``): the runner spawns a separate OS process.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from metagross.config import defaults
from metagross.contracts import ipc
from metagross.contracts.messages import DriveMode, OperatorCmd, SensorFrame, WheelCmd
from metagross.sim.batch import Job, run_batch
from metagross.sim.runner import run_episode
from metagross.sim.scenario import make_scenario, write_scenario

DEV_SEED = 102  # F1_trail


def dummy_autonomy_main(conn, run_dir: str) -> None:
    """Proportional go-to-goal controller on wheel odometry + gyro (A-frame).
    Same entry convention as metagross.autonomy.process.autonomy_main(conn, run_dir)."""
    mission = cfg = None
    x = y = yaw = 0.0
    last = None
    r = defaults.VEHICLE.wheel_radius_m
    b_eff = defaults.CHI_NOMINAL * defaults.VEHICLE.track_width_m
    while True:
        msg = conn.recv()
        kind = msg[0]
        if kind == ipc.MSG_RESET:
            _, mission, calib, vehicle, cfg = msg
            conn.send((ipc.MSG_READY,))
        elif kind == ipc.MSG_OPERATOR:
            assert isinstance(msg[1], OperatorCmd)
        elif kind == ipc.MSG_FRAME:
            t0 = time.perf_counter()
            f: SensorFrame = msg[1]
            assert f.sensor_mode == "tier0_disparity" and f.disparity is not None and f.left_rgb is None
            if cfg.get("dummy_error_at_seq") == f.seq:
                conn.send((ipc.MSG_ERROR, "dummy failure"))
                continue
            if last is not None:
                dt = f.t - last.t
                ds = 0.5 * r * ((f.wheel_angle_l_rad - last.wheel_angle_l_rad) + (f.wheel_angle_r_rad - last.wheel_angle_r_rad))
                yaw_mid = yaw + 0.5 * f.gyro_z_rps * dt
                x += ds * math.cos(yaw_mid)
                y += ds * math.sin(yaw_mid)
                yaw += f.gyro_z_rps * dt
            last = f
            gx, gy = mission.goal_xy_a
            err = math.atan2(gy - y, gx - x) - yaw
            err = (err + math.pi) % (2 * math.pi) - math.pi
            dist = math.hypot(gx - x, gy - y)
            v = min(float(cfg.get("dummy_speed", 1.0)), 0.5 * dist)
            om = max(-1.0, min(1.0, 1.5 * err))
            wl = (v - 0.5 * om * b_eff) / r
            wr = (v + 0.5 * om * b_eff) / r
            compute = cfg.get("dummy_compute_ms")
            ms = (time.perf_counter() - t0) * 1e3 if compute is None else float(compute)
            conn.send((ipc.MSG_CMD, WheelCmd(f.t, f.seq, wl, wr, DriveMode.NOMINAL, ms), None))
        elif kind == ipc.MSG_CLOSE:
            break
    conn.close()


@pytest.fixture(scope="module")
def dev_scenario(tmp_path_factory):
    d = tmp_path_factory.mktemp("scn")
    scn = make_scenario(DEV_SEED)
    assert scn["split"] == "dev"
    return write_scenario(scn, d / "dev" / f"{DEV_SEED}.json")


def test_run_episode_with_dummy_autonomy(dev_scenario, tmp_path):
    run_dir = tmp_path / "run"
    res = run_episode(dev_scenario, {"name": "DUMMY", "dummy_compute_ms": 70.0}, run_dir, sensor_mode="tier0",
                      camera_hz=5.0, autonomy_target=dummy_autonomy_main, max_sim_s=12.0, max_wall_s=240.0)
    assert res["error"] is None, res["error"]
    for f in ("result.json", "gt/states.npz", "gt/events.json", "gt/cmds.npz"):
        assert (run_dir / f).exists()
    assert json.loads((run_dir / "result.json").read_text())["seed"] == DEV_SEED
    required = {"success", "failure_type", "time", "path_length", "spl", "min_clearance", "final_error", "ditch_entries",
                "collisions", "mean_speed", "false_stops", "config", "seed", "family", "sha256", "machine"}
    assert required <= set(res)
    assert res["failure_type"] in {None, "timeout", "collision", "ditch_entry", "water_entry", "stuck", "tip_over"}
    assert res["config"]["name"] == "DUMMY" and res["config"]["use_negobs"] is True  # merged over defaults
    st = np.load(run_dir / "gt" / "states.npz")
    assert {"t", "x", "y", "z", "roll", "pitch", "yaw", "v", "omega", "wl", "wr"} <= set(st.files)
    assert np.median(np.diff(st["t"])) == pytest.approx(1 / 30, abs=0.011)
    assert res["path_length"] > 3.0  # the dummy actually drove
    # Latency injection: 70 ms -> 4 physics steps (80 ms) after each frame.
    cm = np.load(run_dir / "gt" / "cmds.npz")
    np.testing.assert_allclose(cm["apply_t"] - cm["t"], 0.08, atol=1e-9)
    assert res["n_frames"] == len(cm["t"]) and abs(res["n_frames"] - res["time"] * 5.0) <= 2
    # Operator-side mission record.
    ms = json.loads((run_dir / "mission.json").read_text())
    assert {"mission_id", "goal_xy_a", "success_radius_m", "timeout_s", "goal_sigma_m", "goal_entry"} <= set(ms)
    rng = float(np.hypot(*ms["goal_xy_a"]))
    assert ms["goal_entry"]["range_m"] == pytest.approx(rng, abs=1e-3)
    assert ms["goal_entry"]["bearing_deg"] == pytest.approx(np.degrees(np.arctan2(ms["goal_xy_a"][1], ms["goal_xy_a"][0])), abs=1e-3)
    assert ms["goal_sigma_m"] >= 0.5
    assert res["n_cmd_timeouts"] == 0  # a 5 Hz autonomy never trips the 0.5 s actuator watchdog


def test_autonomy_error_is_reported(dev_scenario, tmp_path):
    res = run_episode(dev_scenario, {"dummy_error_at_seq": 3}, tmp_path / "err", autonomy_target=dummy_autonomy_main,
                      max_sim_s=10.0, max_wall_s=120.0)
    assert res["failure_type"] == "autonomy_error" and "dummy failure" in res["error"]
    assert res["n_frames"] == 3


def test_batch_is_resumable(dev_scenario, tmp_path):
    root = dev_scenario.parent.parent
    jobs = [Job(DEV_SEED, {**ipc.DEFAULT_AUTONOMY_CONFIG, "name": "DUMMY"})]
    kw = dict(autonomy_target=dummy_autonomy_main, max_sim_s=3.0, max_wall_s=120.0)
    csv1 = run_batch(jobs, root, tmp_path / "runs", workers=1, **kw)
    rp = tmp_path / "runs" / "DUMMY" / f"{DEV_SEED:03d}" / "result.json"
    m1 = rp.stat().st_mtime_ns
    csv2 = run_batch(jobs, root, tmp_path / "runs", workers=1, **kw)
    assert rp.stat().st_mtime_ns == m1  # skipped on resume
    lines = csv2.read_text().strip().splitlines()
    assert csv1 == csv2 and len(lines) == 2 and lines[0].startswith("config,seed,family")
