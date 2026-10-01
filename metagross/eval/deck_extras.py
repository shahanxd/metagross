"""Deck numbers that are derived from existing result files rather than produced by an evaluation.

Each is recomputed here from its source and written to ``results/deck_extras.json`` with a
``claims`` list that :mod:`metagross.eval.claims` collects:

* KITTI distances driven (sum of ``path_length_m`` over the three sequences)       Tested
* the analytic safe-speed envelope minimum (``theory.json#envelope.v_min_mps``)     Estimated
* EVAL tier-0 FULL: median MPPI time per tick and median telemetry packet period   Simulated
* the operator-console frame on the deck: distance to B at EVAL seed 44, seq 38    Simulated

Run::

    python -m metagross.eval.deck_extras && python -m metagross.eval.claims
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
from pathlib import Path
from typing import Iterable

import numpy as np

LOG = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT_FILE = RESULTS / "deck_extras.json"
KITTI_SEQS = ("00", "05", "07")
EVAL_FULL = RESULTS / "runs_eval_tier0" / "FULL"
CONSOLE_SEED, CONSOLE_SEQ = 44, 38  # the replayed frame shown on the deck (deck_assets/final/console.json)


def median_period(times_s: Iterable[float]) -> float:
    """Median gap (s) between consecutive timestamps."""
    t = np.asarray(sorted(times_s), float)
    if t.size < 2:
        raise ValueError("need at least two timestamps")
    return float(np.median(np.diff(t)))


def dist_to_goal(pose_xy: Iterable[float], goal_xy: Iterable[float]) -> float:
    """Euclidean distance (m) in the A-frame from the stack's pose estimate to B."""
    (x, y), (gx, gy) = list(pose_xy)[:2], list(goal_xy)[:2]
    return math.hypot(gx - x, gy - y)


def _telemetry(run_dir: Path) -> list[dict]:
    with (run_dir / "autonomy" / "telemetry.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def kitti_rows(results: Path = RESULTS) -> list[dict]:
    lengths = {s: float(json.loads((results / f"kitti_vo_{s}.json").read_text(encoding="utf-8"))["path_length_m"])
               for s in KITTI_SEQS}
    rows = [{"id": f"kitti{s}_path_length", "value": f"{m / 1000:.2f}" if m >= 1000 else f"{m:.0f}",
             "unit": "km" if m >= 1000 else "m", "label": "Tested", "source": f"results/kitti_vo_{s}.json#path_length_m",
             "note": f"KITTI {s} ground-truth path length driven" + (" (frames 1101-4540)" if s == "00" else "")}
            for s, m in lengths.items()]
    total_km = sum(lengths.values()) / 1000
    rows.append({"id": "kitti_pooled_path_km", "value": f"{total_km:.2f}", "unit": "km", "label": "Tested",
                 "source": " + ".join(f"results/kitti_vo_{s}.json#path_length_m" for s in KITTI_SEQS),
                 "note": "total KITTI distance over which the pooled 1.87 % segment drift was measured"})
    return rows


def envelope_rows(results: Path = RESULTS) -> list[dict]:
    env = json.loads((results / "theory.json").read_text(encoding="utf-8"))["envelope"]
    return [{"id": "theory_envelope_v_min_mps", "value": f"{float(env['v_min_mps']):.2f}", "unit": "m/s", "label": "Estimated",
             "source": "results/theory.json#envelope.v_min_mps",
             "note": f"lowest ditch-visibility speed limit over masts {env['h_range_m']} m and ditches {env['w_range_m']} m; "
                     "analytic, level ground, small-angle model"}]


def eval_runtime_rows(full_dir: Path = EVAL_FULL) -> list[dict]:
    mppi, periods, n_runs = [], [], 0
    for run in sorted(p for p in full_dir.iterdir() if p.is_dir()):
        n_runs += 1
        with (run / "autonomy" / "timings.csv").open(newline="", encoding="utf-8") as fh:
            mppi += [float(r["mppi"]) for r in csv.DictReader(fh)]
        periods.append(median_period(p["t"] for p in _telemetry(run)))
    base = f"EVAL closed loop run 2, tier0, FULL, {n_runs} runs (seeds 0-59)"
    return [
        {"id": "closed_loop_eval_tier0_FULL_mppi_p50_ms", "value": f"{float(np.median(mppi)):.1f}", "unit": "ms",
         "label": "Simulated", "source": "results/runs_eval_tier0/FULL/*/autonomy/timings.csv#mppi",
         "note": f"{base}; median MPPI time per 5 Hz tick over {len(mppi)} ticks, 512 rollouts; 4 vCPU container"},
        {"id": "closed_loop_eval_tier0_FULL_telemetry_period_s", "value": f"{float(np.median(periods)):.1f}", "unit": "s",
         "label": "Simulated", "source": "results/runs_eval_tier0/FULL/*/autonomy/telemetry.jsonl#t",
         "note": f"{base}; median gap between telemetry packets (the 2 Hz timer runs on the 5 Hz tick)"},
    ]


def console_rows(full_dir: Path = EVAL_FULL, seed: int = CONSOLE_SEED, seq: int = CONSOLE_SEQ) -> list[dict]:
    run = full_dir / f"{seed:03d}"
    pkt = next(p for p in _telemetry(run) if int(p["seq"]) == seq)
    goal = json.loads((run / "mission.json").read_text(encoding="utf-8"))["goal_xy_a"]
    d = dist_to_goal(pkt["pose"], goal)
    return [{"id": f"console_eval_s{seed:03d}_seq{seq}_dist_to_b_m", "value": f"{d:.1f}", "unit": "m", "label": "Simulated",
             "source": f"results/runs_eval_tier0/FULL/{seed:03d}/autonomy/telemetry.jsonl#seq={seq}.pose + mission.json#goal_xy_a",
             "note": f"operator-console replay frame on the deck (t = {pkt['t']} s): distance from the stack's pose estimate to B"}]


def build(results: Path = RESULTS) -> dict:
    claims = kitti_rows(results) + envelope_rows(results) + eval_runtime_rows(results / "runs_eval_tier0" / "FULL") \
        + console_rows(results / "runs_eval_tier0" / "FULL")
    return {"what": __doc__.strip().splitlines()[0], "generated_by": "metagross/eval/deck_extras.py", "claims": claims}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    args = ap.parse_args()
    doc = build()
    args.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    LOG.info("wrote %s: %d claims", args.out, len(doc["claims"]))


if __name__ == "__main__":
    main()
