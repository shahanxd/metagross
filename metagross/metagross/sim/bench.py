"""Reproducible performance measurements for the simulated world.

Measures, on one scenario:

* Tier-0 synthetic depth sensor: wall-clock ms per frame (full ``make_sensor_frame`` call:
  render + noise + upsample) along an open-loop drive at 5 Hz;
* physics: ms per ``World.step`` (vehicle + dynamics + referee + GT logging);
* optionally a closed-loop episode with a given autonomy target (``module:function``), reporting
  its wall time.

    python -m metagross.sim.bench --seed 102 --frames 60 --out results/sim_benchmark.json
    python -m metagross.sim.bench --seed 102 --episode-target test_sim_runner:dummy_autonomy_main \
        --extra-sys-path tests --episode-sim-s 60

Numbers depend on machine load; the output JSON records machine info and thread settings.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from metagross.config import defaults
from metagross.contracts.messages import DriveMode, WheelCmd

log = logging.getLogger(__name__)


def _stats(ms: list[float]) -> dict[str, float]:
    a = np.asarray(ms, float)
    return {"n": int(a.size), "mean": round(float(a.mean()), 3), "median": round(float(np.median(a)), 3),
            "p95": round(float(np.percentile(a, 95)), 3), "max": round(float(a.max()), 3)}


def bench_open_loop(seed: int, frames: int, camera_hz: float = defaults.CAMERA_HZ_BATCH) -> dict:
    from metagross.sim.scenario import make_scenario
    from metagross.sim.world import World

    scn = make_scenario(seed)
    w = World(scn, sensor_mode="tier0", timeout_s=1e6)
    steps = int(round(defaults.PHYSICS_HZ / camera_hz))
    r = defaults.VEHICLE.wheel_radius_m
    w.queue_command(WheelCmd(0.0, 0, 0.9 / r, 1.0 / r, DriveMode.NOMINAL, 0.0))  # gentle left arc, ~1 m/s
    for k in range(frames):
        w.make_sensor_frame(w.t, k)
        for _ in range(steps):
            if w.done:
                break
            w.step()
        if w.done:
            break
    return {"seed": seed, "family": scn["family"], "frames_rendered": len(w.sensor_ms), "tier0_frame_ms": _stats(w.sensor_ms),
            "physics_step_ms": _stats(w.physics_ms), "stopped_by": w.referee.failure_type or ("success" if w.referee.success else None)}


def bench_episode(seed: int, target: str, sim_s: float) -> dict:
    from metagross.sim.runner import run_episode
    from metagross.sim.scenario import make_scenario, write_scenario

    with tempfile.TemporaryDirectory() as td:
        p = write_scenario(make_scenario(seed), Path(td) / f"{seed}.json")
        t0 = time.perf_counter()
        res = run_episode(p, {"name": "BENCH"}, Path(td) / "run", autonomy_target=target, max_sim_s=sim_s)
        wall = time.perf_counter() - t0
    keys = ("success", "failure_type", "time", "path_length", "n_frames", "sensor_ms_mean", "sensor_breakdown_ms",
            "physics_ms_per_step", "compute_ms_mean")
    return {"target": target, "max_sim_s": sim_s, "wall_s": round(wall, 3), **{k: res[k] for k in keys}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=102)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--episode-target", default=None)
    ap.add_argument("--episode-sim-s", type=float, default=60.0)
    ap.add_argument("--extra-sys-path", default=None, help="directory added to sys.path (e.g. tests)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import cv2

    cv2.setNumThreads(2)
    if a.extra_sys_path:
        sys.path.insert(0, str(Path(a.extra_sys_path).resolve()))
    from metagross.sim.runner import machine_info

    out = {"machine": machine_info(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "cv2_threads": cv2.getNumThreads(),
           "open_loop": bench_open_loop(a.seed, a.frames)}
    if a.episode_target:
        out["episode"] = bench_episode(a.seed, a.episode_target, a.episode_sim_s)
    txt = json.dumps(out, indent=1)
    log.info("benchmark:\n%s", txt)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
