"""Aggregate closed-loop batch runs into ``results/closed_loop_dev.json`` (+ a combined summary CSV).

Reads, per run directory ``<root>/<config>/<seed:03d>/`` written by :mod:`metagross.sim.runner`:

* ``result.json`` (referee verdict, SPL, mean speed, ditch entries, collisions, false stops, ...);
* ``autonomy/timings.csv`` (per-tick ``compute_ms``: autonomy wall time per 5 Hz tick);
* ``gt/cmds.npz`` (per-command drive ``mode``) or, for older runs, ``autonomy/telemetry.jsonl``:
  ``arrived_est`` = the vehicle declared ARRIVED in its *own* pose estimate, which separates
  planning competence from odometry drift (the referee judges the true goal within
  ``success_radius_m``, read from ``mission.json``).

Failure type (``failure_type``): the referee's verdict, except that a run which declared ARRIVED
but ended outside the success radius with a non-hazard verdict ('stuck' / 'timeout', from runs
made before the referee learnt ``arrived_short``) is re-labelled **'arrived_short'** (odometry
error, not a planning stall). ``failure_type_referee`` keeps the raw verdict.

Units: speeds m/s, times s, compute ms, errors m. Per (sensor mode, config, family) and overall:
n, success rate, SPL mean, mean speed mean (referee: path / episode time), drive speed mean (GT
path / time until the estimated arrival or the episode end), ditch entries / collisions / water
entries / out-of-bounds / false stops (sums), arrived_est rate, arrived_short count, reach rate
(success or arrived_est), final-error p50 / p90, compute p50 / p95 over all ticks.

CLI::

    python -m metagross.sim.closed_loop_summary --tier0 results/runs_dev_tier0 \
        --stereo results/runs_dev_stereo --out results/closed_loop_dev.json

This is evaluation code on the simulator side (it may read GT-derived results); the autonomy never
imports it.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

log = logging.getLogger(__name__)

DEV_SEEDS = range(100, 130)  # tuning split; EVAL seeds 0-59 are never aggregated here
BEFORE_WINDOW_S = 20.0  # the integration-pass 'before' runs were capped at 20 s of sim time
FAMILY_ORDER = ("F1_trail", "F2_ditch_field", "F3_crest_ditch", "F4_sudden_obstacle", "F5_lighting", "F6_water_mud")
ARRIVED_MODE = "ARRIVED"  # DriveMode.ARRIVED.value
DEFAULT_SUCCESS_RADIUS_M = 2.0  # used only when a run has no mission.json
RELABEL_AS_ARRIVED_SHORT = ("stuck", "timeout")  # non-hazard verdicts that hide an estimated arrival


def _t_arrived(run_dir: Path) -> Optional[float]:
    """First sim time [s] the autonomy commanded mode ARRIVED (gt/cmds.npz, else telemetry); None if never."""
    cp = run_dir / "gt" / "cmds.npz"
    if cp.exists():
        c = np.load(cp)
        if "mode" in c.files and len(c["mode"]):
            k = np.flatnonzero(c["mode"].astype(str) == ARRIVED_MODE)
            return float(c["t"][k[0]]) if len(k) else None
    tel = run_dir / "autonomy" / "telemetry.jsonl"
    if tel.exists():
        with open(tel, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("mode") == ARRIVED_MODE:
                        return float(rec["t"])
    return None


def effective_failure_type(failure_type: Optional[str], success: bool, arrived_est: bool, final_error: Optional[float],
                           success_radius_m: float) -> Optional[str]:
    """Referee verdict, with 'stuck' / 'timeout' after an estimated arrival outside the success radius
    re-labelled 'arrived_short' (see module docstring)."""
    if success:
        return failure_type
    if (arrived_est and failure_type in RELABEL_AS_ARRIVED_SHORT and final_error is not None
            and final_error > success_radius_m):
        return "arrived_short"
    return failure_type


def load_run(run_dir: Path) -> Optional[dict[str, Any]]:
    """One run's result.json plus derived fields (compute_ms list, arrived_est, effective failure type);
    None if unfinished."""
    rp = run_dir / "result.json"
    if not rp.exists():
        return None
    r = json.loads(rp.read_text(encoding="utf-8"))
    comp: list[float] = []
    tp = run_dir / "autonomy" / "timings.csv"
    if tp.exists():
        with open(tp, newline="", encoding="utf-8") as fh:
            comp = [float(row["compute_ms"]) for row in csv.DictReader(fh) if row.get("compute_ms")]
    t_arr = _t_arrived(run_dir)
    radius = DEFAULT_SUCCESS_RADIUS_M
    mp = run_dir / "mission.json"
    if mp.exists():
        radius = float(json.loads(mp.read_text(encoding="utf-8")).get("success_radius_m", radius))
    r["_compute_ms"] = comp
    r["arrived_est"] = t_arr is not None
    r["t_arrived_est"] = t_arr
    r["failure_type_referee"] = r.get("failure_type")
    r["failure_type"] = effective_failure_type(r.get("failure_type"), bool(r["success"]), t_arr is not None,
                                               r.get("final_error"), radius)
    # speed while driving: GT path length / time up to the end of the episode or the (estimated) arrival,
    # whichever is first (the referee's mean_speed also counts the 20 s 'stuck' wait after arrival)
    t_end = min(float(r["time"]), t_arr) if t_arr is not None else float(r["time"])
    r["drive_speed"] = r["mean_speed"]
    gp = run_dir / "gt" / "states.npz"
    if gp.exists() and t_end > 0:
        gt = np.load(gp)
        k = gt["t"] <= t_end + 1e-9
        seg = np.hypot(np.diff(gt["x"][k]), np.diff(gt["y"][k]))
        r["drive_speed"] = round(float(seg.sum()) / t_end, 4)
    return r


def collect(root: Path, configs: Optional[Iterable[str]] = None) -> list[dict[str, Any]]:
    """All finished DEV runs under ``root`` (optionally only these config names)."""
    rows = []
    if not root.exists():
        return rows
    for cdir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if configs is not None and cdir.name not in configs:
            continue
        for sdir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            r = load_run(sdir)
            if r is None or int(r["seed"]) not in DEV_SEEDS:
                continue
            r["config_name"] = cdir.name
            rows.append(r)
    return rows


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comp = np.concatenate([np.asarray(r["_compute_ms"], float) for r in rows]) if rows else np.zeros(0)
    fin = [r["final_error"] for r in rows if r.get("final_error") is not None]
    reach = [bool(r["success"]) or bool(r["arrived_est"]) for r in rows]
    return {
        "n": len(rows),
        "seeds": [int(r["seed"]) for r in rows],
        "success_rate": round(float(np.mean([r["success"] for r in rows])), 4) if rows else None,
        "n_success": int(sum(bool(r["success"]) for r in rows)),
        "spl_mean": round(float(np.mean([r["spl"] for r in rows])), 4) if rows else None,
        "mean_speed_mps": round(float(np.mean([r["mean_speed"] for r in rows])), 4) if rows else None,
        "drive_speed_mps": round(float(np.mean([r["drive_speed"] for r in rows])), 4) if rows else None,
        "ditch_entries": int(sum(r["ditch_entries"] for r in rows)),
        "collisions": int(sum(r["collisions"] for r in rows)),
        "water_entries": int(sum(r.get("water_entries", 0) for r in rows)),
        "false_stops": int(sum(r["false_stops"] for r in rows)),
        "out_of_bounds": int(sum(r["failure_type"] == "out_of_bounds" for r in rows)),
        "arrived_est_rate": round(float(np.mean([r["arrived_est"] for r in rows])), 4) if rows else None,
        "arrived_short": int(sum(r["failure_type"] == "arrived_short" for r in rows)),
        "reach_rate": round(float(np.mean(reach)), 4) if rows else None,
        "n_reach": int(sum(reach)),
        "final_error_median_m": round(float(np.median(fin)), 3) if fin else None,
        "final_error_p90_m": round(float(np.percentile(fin, 90)), 3) if fin else None,
        "failure_types": {str(k): int(v) for k, v in zip(*np.unique([str(r["failure_type"]) for r in rows], return_counts=True))} if rows else {},
        "compute_ms_p50": round(float(np.percentile(comp, 50)), 1) if comp.size else None,
        "compute_ms_p95": round(float(np.percentile(comp, 95)), 1) if comp.size else None,
        "n_ticks": int(comp.size),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """{config: {'all': agg, 'by_family': {family: agg}}}."""
    out: dict[str, Any] = {}
    for cfg in sorted({r["config_name"] for r in rows}):
        rc = [r for r in rows if r["config_name"] == cfg]
        fams = [f for f in FAMILY_ORDER if any(r["family"] == f for r in rc)]
        out[cfg] = {"all": _agg(rc), "by_family": {f: _agg([r for r in rc if r["family"] == f]) for f in fams}}
    return out


def per_run_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("config_name", "seed", "family", "success", "failure_type", "failure_type_referee", "time", "path_length", "spl", "mean_speed",
            "drive_speed", "final_error", "arrived_est", "t_arrived_est", "ditch_entries", "collisions", "water_entries",
            "false_stops", "compute_ms_mean", "compute_ms_p95")
    return [{k: r.get(k) for k in keys} for r in rows]


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Combined per-run CSV (all configs) at ``path``."""
    table = per_run_table(rows)
    if not table:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


def before_after(before_csv: Path, root: Path, window_s: float = BEFORE_WINDOW_S) -> list[dict[str, Any]]:
    """Same (config, seed) before vs after: GT path length in the first ``window_s`` of sim time
    (the earlier integration runs were capped at 20 s), plus the after-run verdict."""
    out: list[dict[str, Any]] = []
    if not before_csv.exists():
        return out
    with open(before_csv, newline="", encoding="utf-8") as fh:
        before = list(csv.DictReader(fh))
    for b in before:
        d = root / b["config"] / f"{int(b['seed']):03d}"
        if int(b["seed"]) not in DEV_SEEDS or not (d / "result.json").exists() or not (d / "gt" / "states.npz").exists():
            continue
        r = json.loads((d / "result.json").read_text(encoding="utf-8"))
        gt = np.load(d / "gt" / "states.npz")
        k = gt["t"] <= window_s + 1e-9
        path = float(np.hypot(np.diff(gt["x"][k]), np.diff(gt["y"][k])).sum())
        out.append({"config": b["config"], "seed": int(b["seed"]), "family": b["family"],
                    "before_path_m": float(b["path_length"]), "before_speed_mps": float(b["mean_speed"]),
                    "before_failure": b["failure_type"], "after_path_m": round(path, 3),
                    "after_speed_mps": round(path / window_s, 4), "after_success": bool(r["success"]),
                    "after_failure": r["failure_type"], "after_final_error_m": r["final_error"]})
    return out


def make_claims(doc: dict[str, Any], out_name: str) -> list[dict[str, str]]:
    """Claims-ledger rows (label Simulated) for the headline aggregates in ``doc``."""
    claims: list[dict[str, str]] = []
    for mode in ("tier0", "stereo"):
        agg = doc.get(mode, {}).get("aggregate", {})
        for cfg, a in agg.items():
            al = a["all"]
            base = f"results/{out_name}#{mode}.aggregate.{cfg}.all"
            note = f"DEV seeds {al['seeds'][0]}-{al['seeds'][-1]} (n={al['n']}), sensor mode {mode}, referee on GT"
            claims += [
                {"id": f"closed_loop_{mode}_{cfg}_success", "value": f"{al['n_success']}/{al['n']}", "label": "Simulated",
                 "source": f"{base}.success_rate", "note": note + "; success = true goal within the mission success radius"},
                {"id": f"closed_loop_{mode}_{cfg}_arrived_est", "value": f"{al['arrived_est_rate']:.2f}", "label": "Simulated",
                 "source": f"{base}.arrived_est_rate", "note": note + "; share of runs that reached the goal in their own pose estimate"},
                {"id": f"closed_loop_{mode}_{cfg}_drive_speed", "value": f"{al['drive_speed_mps']:.2f} m/s", "label": "Simulated",
                 "source": f"{base}.drive_speed_mps", "note": note + "; GT path / time until estimated arrival or episode end"},
                {"id": f"closed_loop_{mode}_{cfg}_ditch_collisions", "value": f"{al['ditch_entries']} ditch / {al['collisions']} coll",
                 "label": "Simulated", "source": f"{base}.ditch_entries", "note": note},
                {"id": f"closed_loop_{mode}_{cfg}_arrived_short", "value": f"{al['arrived_short']}/{al['n']}", "label": "Simulated",
                 "source": f"{base}.arrived_short",
                 "note": note + "; declared ARRIVED in its own estimate but ended outside the success radius (odometry error)"},
                {"id": f"closed_loop_{mode}_{cfg}_final_error_p50_p90",
                 "value": f"{al['final_error_median_m']} / {al['final_error_p90_m']} m", "label": "Simulated",
                 "source": f"{base}.final_error_median_m", "note": note + "; GT distance to B at episode end"},
                {"id": f"closed_loop_{mode}_{cfg}_compute_p50_p95", "value": f"{al['compute_ms_p50']} / {al['compute_ms_p95']} ms",
                 "label": "Simulated", "source": f"{base}.compute_ms_p50",
                 "note": note + "; autonomy wall time per 5 Hz tick on the shared 4-core laptop (other jobs running)"},
            ]
    return claims


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier0", type=Path, default=Path("results/runs_dev_tier0"))
    ap.add_argument("--stereo", type=Path, default=Path("results/runs_dev_stereo"))
    ap.add_argument("--out", type=Path, default=Path("results/closed_loop_dev.json"))
    ap.add_argument("--extra", type=Path, help="JSON merged into the output (notes)")
    ap.add_argument("--before", type=Path, default=Path("results/runs_integration/tier0/summary.csv"),
                    help="earlier tier0 summary CSV for the same-seed before/after table")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    doc: dict[str, Any] = {"split": "dev", "seeds": [DEV_SEEDS.start, DEV_SEEDS.stop - 1]}
    for mode, root in (("tier0", a.tier0), ("stereo", a.stereo)):
        rows = collect(root)
        write_summary_csv(rows, root / "summary.csv")
        doc[mode] = {"root": root.as_posix(), "aggregate": aggregate(rows), "runs": per_run_table(rows)}
        log.info("%s: %d runs", mode, len(rows))
    doc["before_after_same_seeds"] = {
        "before_source": a.before.as_posix(), "after_source": a.tier0.as_posix(),
        "metric": f"GT path length in the first {BEFORE_WINDOW_S:.0f} s of sim time (before runs were capped there)",
        "rows": before_after(a.before, a.tier0)}
    if a.extra is not None and a.extra.exists():
        doc.update(json.loads(a.extra.read_text(encoding="utf-8")))
    doc["claims"] = make_claims(doc, a.out.name)
    a.out.write_text(json.dumps(doc, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    log.info("wrote %s", a.out)


if __name__ == "__main__":
    main()
