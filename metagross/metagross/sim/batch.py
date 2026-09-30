"""Batch runner: many (seed, config) episodes over N worker processes, resumable, summary CSV.

Layout: ``<out_root>/<config name>/<seed:03d>/`` holds each run (see :mod:`metagross.sim.runner`).
A job whose ``result.json`` already exists is skipped (resume). The summary CSV is rebuilt from all
``result.json`` files of the requested jobs at the end.

CLI::

    python -m metagross.sim.batch --split dev --workers 2 --out results/runs \
        [--configs configs.json] [--seeds 100 101] [--camera-hz 5] [--max-sim-s 120]

``configs.json`` is a list of autonomy config dicts (each with a ``name``) merged over
``ipc.DEFAULT_AUTONOMY_CONFIG``; the default is the single FULL config.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from metagross.config import defaults
from metagross.contracts import ipc
from metagross.sim.scenario import scenario_seeds, split_of

log = logging.getLogger(__name__)

SUMMARY_FIELDS = ("config", "seed", "family", "split", "success", "failure_type", "time", "path_length", "spl",
                  "optimal_path_length", "min_clearance", "final_error", "ditch_entries", "collisions", "water_entries",
                  "mean_speed", "false_stops", "n_frames", "compute_ms_mean", "sensor_ms_mean", "wall_time_s", "error")


@dataclass
class Job:
    seed: int
    config: dict = field(default_factory=lambda: dict(ipc.DEFAULT_AUTONOMY_CONFIG))

    @property
    def name(self) -> str:
        return str(self.config.get("name", "FULL"))


def job_dir(out_root: Path, job: Job) -> Path:
    return Path(out_root) / job.name / f"{job.seed:03d}"


def scenario_path(scenario_root: Path, seed: int) -> Path:
    return Path(scenario_root) / split_of(seed) / f"{seed}.json"


def install_referee_mode_hook() -> bool:
    """Forward every autonomy command's drive mode to the referee (``Referee.note_command``) so that
    an estimated arrival outside the success radius ends the episode as ``arrived_short`` instead of
    20 s later as 'stuck'. Wraps ``World.queue_command`` once per process (idempotent; the referee
    keeps only the first ARRIVED, so a world that already forwards the mode is unaffected).
    Returns True if the hook was installed by this call. Sim-side only: the autonomy never sees it."""
    from metagross.sim.world import World

    if getattr(World.queue_command, "_forwards_mode", False):
        return False
    orig = World.queue_command

    def queue_command(self: World, cmd: Any) -> int:
        self.referee.note_command(self.t, cmd.mode)
        return orig(self, cmd)

    queue_command._forwards_mode = True  # type: ignore[attr-defined]
    World.queue_command = queue_command  # type: ignore[method-assign]
    return True


def _run_job(job: Job, scenario_root: str, out_root: str, kwargs: dict[str, Any]) -> dict:
    """Worker entry (top-level so it pickles under spawn)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2

    cv2.setNumThreads(1)
    from metagross.sim.runner import run_episode

    kwargs = dict(kwargs)
    if kwargs.pop("arrival_ends_episode", True):
        install_referee_mode_hook()
    return run_episode(scenario_path(Path(scenario_root), job.seed), job.config, job_dir(Path(out_root), job), **kwargs)


def write_summary(jobs: list[Job], out_root: Path, csv_path: Path) -> int:
    rows = 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for job in jobs:
            rp = job_dir(out_root, job) / "result.json"
            if not rp.exists():
                continue
            res = json.loads(rp.read_text(encoding="utf-8"))
            w.writerow({**res, "config": job.name})
            rows += 1
    return rows


def run_batch(jobs: list[Job], scenario_root: str | Path, out_root: str | Path, workers: int = 2,
              csv_path: str | Path | None = None, **run_kwargs: Any) -> Path:
    """Run all jobs not yet finished; returns the summary CSV path."""
    out_root = Path(out_root)
    todo = [j for j in jobs if not (job_dir(out_root, j) / "result.json").exists()]
    log.info("%d jobs, %d already done, %d to run on %d workers", len(jobs), len(jobs) - len(todo), len(todo), workers)
    if todo:
        with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=get_context("spawn")) as ex:
            futs = {ex.submit(_run_job, j, str(scenario_root), str(out_root), run_kwargs): j for j in todo}
            for fut in as_completed(futs):
                j = futs[fut]
                try:
                    r = fut.result()
                    log.info("done %s seed %d: success=%s failure=%s", j.name, j.seed, r["success"], r["failure_type"])
                except Exception:  # keep the batch going; the job stays unfinished and is retried on resume
                    log.exception("job %s seed %d crashed", j.name, j.seed)
    csv_path = Path(csv_path) if csv_path else out_root / "summary.csv"
    n = write_summary(jobs, out_root, csv_path)
    log.info("summary: %d rows -> %s", n, csv_path)
    return csv_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("dev", "eval"), default="dev")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--configs", type=Path, help="JSON list of autonomy config dicts")
    ap.add_argument("--scenarios", type=Path, default=Path("data/scenarios"))
    ap.add_argument("--out", type=Path, default=Path("results/runs"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--camera-hz", type=float, default=defaults.CAMERA_HZ_BATCH)
    ap.add_argument("--sensor-mode", choices=("tier0", "stereo"), default="tier0")
    ap.add_argument("--max-sim-s", type=float)
    ap.add_argument("--max-wall-s", type=float)
    ap.add_argument("--autonomy-target", default=None, help="module:function (default metagross.autonomy.process:autonomy_main)")
    ap.add_argument("--no-arrival-end", action="store_true",
                    help="do not end episodes on an estimated arrival outside the goal radius ('arrived_short')")
    ap.add_argument("--config-names", nargs="*", help="run only these config names from --configs")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seeds = a.seeds or list(scenario_seeds(a.split))
    configs = json.loads(a.configs.read_text(encoding="utf-8")) if a.configs else [dict(ipc.DEFAULT_AUTONOMY_CONFIG)]
    if isinstance(configs, dict):  # results/configs/eval_configs.json style: {"configs": [...], ...}
        configs = configs["configs"]
    if a.config_names:
        configs = [c for c in configs if c.get("name") in set(a.config_names)]
    jobs = [Job(s, {**ipc.DEFAULT_AUTONOMY_CONFIG, **{k: v for k, v in c.items() if not k.startswith("_")}})
            for c in configs for s in seeds]
    run_batch(jobs, a.scenarios, a.out, a.workers, sensor_mode=a.sensor_mode, camera_hz=a.camera_hz, max_sim_s=a.max_sim_s,
              max_wall_s=a.max_wall_s, autonomy_target=a.autonomy_target, arrival_ends_episode=not a.no_arrival_end)


if __name__ == "__main__":
    main()
