"""Round-2 verifier aggregation of DEV closed-loop runs (sim side: reads GT results; never imported by autonomy).

Walks results/verify_dev2/<group>/<config>/<seed:03d>/ and writes
results/verify_dev2/verify_dev2.json (+ per_seed.csv) and prints markdown tables per (group, config, family).
Round-1 baseline for the same (config, seed) is read from results/verify_dev (FULL; TYPICAL is the
round-1 typical stack at 1.5 m/s, not T1 at 2.0 m/s, so only FULL is compared seed-by-seed).

Per-run metrics (units: m, s, m/s, ms, bytes):
* success / failure_type / final_error / ditch / collisions / water / mean_speed: referee result.json (GT).
* arrived_est: the autonomy commanded mode ARRIVED on any tick (gt/cmds.npz 'mode').
* arrived_short: referee failure_type == 'arrived_short' (estimate says ARRIVED, GT goal > 2 m away).
* oob: failure_type == 'out_of_bounds'.
* fwd_check_pk / fwd_check_ep: telemetry packets (~2 Hz) whose reason starts with FWD_CHECK / rising edges.
* look_n: autonomy.log transitions '-> STOP_AND_LOOK' (exact); look_ticks: ticks with mode STOP_AND_LOOK.
* safe_stop_n: transitions to SAFE_STOP; health transitions (CAUTION/DEGRADED) with reason.
* compute p50/p95: autonomy/timings.csv compute_ms.
* odo_ratio: estimated path length (telemetry pose) / GT path length at the same timestamps.
* packet bytes: telemetry packet_bytes median / p95 / fraction > 600 B.
"""
from __future__ import annotations

import csv
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Downloads\sih again\metagross")
VER = ROOT / "results" / "verify_dev2"
R1 = ROOT / "results" / "verify_dev"
PACKET_BUDGET_B = 600
HOLDOUT = {120, 126, 121, 127, 122, 128}  # suggested untuned seeds (F1 120/126, F2 121/127, F3 122/128)
LOG = logging.getLogger("agg2")
TRANS_RE = re.compile(r"t=([\d.]+) mode (\w+) -> (\w+) \((.*)\)$")


def fam(s: str) -> str:
    return s.split("_")[0]


def read_tel(run: Path) -> list[dict]:
    p = run / "autonomy" / "telemetry.jsonl"
    out = []
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def transitions(run: Path) -> list[tuple[float, str, str, str]]:
    p = run / "autonomy" / "autonomy.log"
    out = []
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = TRANS_RE.search(line)
            if m:
                out.append((float(m.group(1)), m.group(2), m.group(3), m.group(4)))
    return out


def compute_ms(run: Path) -> np.ndarray:
    p = run / "autonomy" / "timings.csv"
    vals = []
    if p.exists():
        with p.open() as f:
            for r in csv.DictReader(f):
                try:
                    vals.append(float(r["compute_ms"]))
                except (KeyError, ValueError):
                    pass
    return np.asarray(vals)


def odo_ratio(run: Path, tel: list[dict]) -> float | None:
    """Estimated / GT path length over the telemetry timestamps (GT interpolated at those times)."""
    p = run / "gt" / "states.npz"
    if not p.exists() or len(tel) < 3:
        return None
    z = np.load(p)
    t = np.array([d["t"] for d in tel])
    pe = np.array([d["pose"][:2] for d in tel])
    gx, gy = np.interp(t, z["t"], z["x"]), np.interp(t, z["t"], z["y"])
    de = np.linalg.norm(np.diff(pe, axis=0), axis=1).sum()
    dg = np.hypot(np.diff(gx), np.diff(gy)).sum()
    return float(de / dg) if dg > 1.0 else None


def run_row(group: str, run: Path) -> dict:
    r = json.loads((run / "result.json").read_text())
    tel = read_tel(run)
    tr = transitions(run)
    cms = compute_ms(run)
    cm = np.load(run / "gt" / "cmds.npz") if (run / "gt" / "cmds.npz").exists() else None
    modes = cm["mode"] if cm is not None else np.array([])
    arrived_ticks = np.flatnonzero(modes == "ARRIVED")
    reasons = [str(d.get("reason", "")) for d in tel]
    fwd = np.array([x.startswith("FWD_CHECK") for x in reasons], dtype=bool)
    pk = np.array([d.get("packet_bytes", 0) for d in tel], dtype=float)
    health = [(t, a, b, why) for (t, a, b, why) in tr if b in ("CAUTION", "DEGRADED") or (a in ("CAUTION", "DEGRADED") and b == "NOMINAL")]
    row = {
        "group": group, "config": r["config"]["name"], "seed": int(r["seed"]), "family": r["family"], "split": r["split"],
        "sensor_mode": r.get("sensor_mode"), "success": bool(r["success"]), "failure": r["failure_type"], "time_s": r["time"],
        "path_m": r["path_length"], "opt_m": r["optimal_path_length"], "spl": r["spl"], "mean_speed": r["mean_speed"],
        "final_error": r["final_error"], "ditch": r["ditch_entries"], "coll": r["collisions"], "water": r.get("water_entries", 0),
        "oob": r["failure_type"] == "out_of_bounds", "arrived_short": r["failure_type"] == "arrived_short",
        "arrived_est": bool(arrived_ticks.size), "t_arrived_est": float(cm["t"][arrived_ticks[0]]) if arrived_ticks.size else None,
        "false_stops": r["false_stops"], "min_clearance": r.get("min_clearance"),
        "fwd_check_pk": int(fwd.sum()), "fwd_check_ep": int(np.sum(fwd[1:] & ~fwd[:-1]) + (1 if fwd.size and fwd[0] else 0)),
        "look_n": sum(1 for x in tr if x[2] == "STOP_AND_LOOK"), "look_ticks": int(np.sum(modes == "STOP_AN")),
        "safe_stop_n": sum(1 for x in tr if x[2] == "SAFE_STOP"), "safe_stop_reasons": [x[3] for x in tr if x[2] == "SAFE_STOP"],
        "health_transitions": health, "dead_end_pk": sum(1 for x in reasons if x.startswith("DEAD_END")),
        "compute_p50": float(np.percentile(cms, 50)) if cms.size else None,
        "compute_p95": float(np.percentile(cms, 95)) if cms.size else None, "n_ticks": int(cms.size),
        "odo_ratio": odo_ratio(run, tel),
        "pkt_med": float(np.median(pk)) if pk.size else None, "pkt_p95": float(np.percentile(pk, 95)) if pk.size else None,
        "pkt_over_frac": float(np.mean(pk > PACKET_BUDGET_B)) if pk.size else None, "n_pkt": int(pk.size),
        "wall_s": r.get("wall_time_s"), "error": r.get("error"), "run_dir": str(run.relative_to(ROOT)),
        "_cms": cms, "_pk": pk,
    }
    r1 = R1 / group / row["config"] / f"{row['seed']:03d}" / "result.json"
    if group in ("tier0", "stereo") and row["config"] == "FULL" and r1.exists():
        p = json.loads(r1.read_text())
        cm1 = r1.parent / "gt" / "cmds.npz"
        a1 = bool(np.any(np.load(cm1)["mode"] == "ARRIVED")) if cm1.exists() else None
        row["r1"] = {"success": p["success"], "failure": p["failure_type"], "final_error": p["final_error"], "path_m": p["path_length"],
                     "mean_speed": p["mean_speed"], "ditch": p["ditch_entries"], "coll": p["collisions"],
                     "water": p.get("water_entries", 0), "arrived_est": a1}
    return row


def pct(a: list[float], q: float) -> float | None:
    return float(np.percentile(a, q)) if a else None


def agg(rows: list[dict]) -> dict:
    cms = np.concatenate([r["_cms"] for r in rows]) if rows else np.zeros(0)
    pk = np.concatenate([r["_pk"] for r in rows]) if rows else np.zeros(0)
    fails: dict[str, int] = defaultdict(int)
    for r in rows:
        fails[r["failure"] or "success"] += 1
    fe = [r["final_error"] for r in rows]
    odo = [r["odo_ratio"] for r in rows if r["odo_ratio"] is not None]
    return {
        "n": len(rows), "success": sum(r["success"] for r in rows), "arrived_est": sum(r["arrived_est"] for r in rows),
        "arrived_short": sum(r["arrived_short"] for r in rows), "final_error_p50": pct(fe, 50), "final_error_p90": pct(fe, 90),
        "ditch": sum(r["ditch"] for r in rows), "coll": sum(r["coll"] for r in rows), "water": sum(r["water"] for r in rows),
        "oob": sum(r["oob"] for r in rows), "mean_speed": float(np.mean([r["mean_speed"] for r in rows])) if rows else None,
        "fwd_check_pk": sum(r["fwd_check_pk"] for r in rows), "fwd_check_ep": sum(r["fwd_check_ep"] for r in rows),
        "look_n": sum(r["look_n"] for r in rows), "safe_stop_n": sum(r["safe_stop_n"] for r in rows),
        "compute_p50": float(np.percentile(cms, 50)) if cms.size else None, "compute_p95": float(np.percentile(cms, 95)) if cms.size else None,
        "n_ticks": int(cms.size), "odo_ratio_p50": pct(odo, 50), "odo_ratio_min": min(odo) if odo else None,
        "odo_ratio_max": max(odo) if odo else None, "pkt_med": float(np.median(pk)) if pk.size else None,
        "pkt_p95": float(np.percentile(pk, 95)) if pk.size else None, "pkt_over_frac": float(np.mean(pk > PACKET_BUDGET_B)) if pk.size else None,
        "outcomes": dict(fails), "seeds": sorted(r["seed"] for r in rows),
    }


def f(x: float | None, nd: int = 2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    groups = [g for g in sys.argv[1:]] or ["tier0", "stereo"]
    out: dict = {"what": "Round-2 independent verifier re-run of DEV closed loop (metagross.sim.batch), DEV seeds 100-129 only",
                 "code_state": "uncommitted working tree on top of fee1da9 (see code_hashes_start.txt / code_hashes_end.txt)",
                 "groups": {}, "per_seed": []}
    all_rows: list[dict] = []
    for group in groups:
        base = VER / group
        rows = [run_row(group, rj.parent) for rj in sorted(base.glob("*/*/result.json"))]
        if not rows:
            continue
        all_rows += rows
        by: dict = defaultdict(list)
        for r in rows:
            by[(r["config"], fam(r["family"]))].append(r)
            by[(r["config"], "ALL")].append(r)
            by[(r["config"], "HOLDOUT" if r["seed"] in HOLDOUT else "NON_HOLDOUT")].append(r)
        g: dict = {}
        for (cfg, fm), rr in sorted(by.items()):
            g.setdefault(cfg, {})[fm] = agg(rr)
        # round-1 comparison (FULL, same seeds)
        cmp = [r for r in rows if "r1" in r]
        if cmp:
            g["_round1_same_seeds"] = {
                "n": len(cmp), "r2_success": sum(r["success"] for r in cmp), "r1_success": sum(r["r1"]["success"] for r in cmp),
                "r2_arrived_est": sum(r["arrived_est"] for r in cmp), "r1_arrived_est": sum(bool(r["r1"]["arrived_est"]) for r in cmp),
                "r2_ditch_coll_water": [sum(r[k] for r in cmp) for k in ("ditch", "coll", "water")],
                "r1_ditch_coll_water": [sum(r["r1"][k] for r in cmp) for k in ("ditch", "coll", "water")],
                "r2_final_error_p50": pct([r["final_error"] for r in cmp], 50), "r1_final_error_p50": pct([r["r1"]["final_error"] for r in cmp], 50),
                "flips": [(r["seed"], r["r1"]["failure"] or "success", r["failure"] or "success") for r in cmp
                          if (r["r1"]["failure"] or "success") != (r["failure"] or "success")],
            }
        out["groups"][group] = g
    for r in all_rows:
        out["per_seed"].append({k: v for k, v in r.items() if not k.startswith("_")})
    out["eval_seed_rows"] = [r["seed"] for r in all_rows if r["split"] != "dev" or r["seed"] < 100]
    (VER / "verify_dev2_agg.json").write_text(json.dumps(out, indent=1, default=str))
    keys = ["group", "config", "seed", "family", "success", "failure", "time_s", "path_m", "opt_m", "spl", "mean_speed", "final_error",
            "ditch", "coll", "water", "oob", "arrived_est", "t_arrived_est", "arrived_short", "fwd_check_pk", "fwd_check_ep", "look_n",
            "look_ticks", "safe_stop_n", "compute_p50", "compute_p95", "odo_ratio", "pkt_med", "pkt_p95", "pkt_over_frac", "wall_s", "run_dir"]
    with (VER / "per_seed.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys + ["r1_success", "r1_failure", "r1_final_error"])
        for r in all_rows:
            p = r.get("r1", {})
            w.writerow([r.get(k) for k in keys] + [p.get("success"), p.get("failure"), p.get("final_error")])
    for group, g in out["groups"].items():
        print(f"\n### {group}")
        print("| config | family | n | success | arrived_est | arrived_short | final err p50/p90 (m) | ditch | coll | water | OOB | mean speed (m/s) | FWD_CHECK pk (ep) | STOP_AND_LOOK | SAFE_STOP | compute p50/p95 (ms) | odo ratio p50 [min-max] | pkt med/p95 B (>600 B) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for cfg, fams in g.items():
            if cfg.startswith("_"):
                continue
            for fm, a in fams.items():
                print(f"| {cfg} | {fm} | {a['n']} | {a['success']} | {a['arrived_est']} | {a['arrived_short']} | {f(a['final_error_p50'])}/{f(a['final_error_p90'])} | "
                      f"{a['ditch']} | {a['coll']} | {a['water']} | {a['oob']} | {f(a['mean_speed'])} | {a['fwd_check_pk']} ({a['fwd_check_ep']}) | {a['look_n']} | "
                      f"{a['safe_stop_n']} | {f(a['compute_p50'], 0)}/{f(a['compute_p95'], 0)} | {f(a['odo_ratio_p50'], 3)} [{f(a['odo_ratio_min'], 3)}-{f(a['odo_ratio_max'], 3)}] | "
                      f"{f(a['pkt_med'], 0)}/{f(a['pkt_p95'], 0)} ({f(a['pkt_over_frac'], 2)}) |")
        if "_round1_same_seeds" in g:
            print("round-1 (results/verify_dev) same seeds, FULL:", json.dumps(g["_round1_same_seeds"], default=str))
    print("\nper seed:")
    for r in all_rows:
        p = r.get("r1", {})
        print(f"{r['group']:6s} {r['config']:4s} {r['seed']} {fam(r['family'])} {'OK ' if r['success'] else '-- '}{r['failure']!s:13s} "
              f"t={r['time_s']:.0f} path={r['path_m']:.1f}/{r['opt_m']:.1f} ferr={r['final_error']:.2f} v={r['mean_speed']:.2f} "
              f"arr={int(r['arrived_est'])} d/c/w={r['ditch']}/{r['coll']}/{r['water']} fwd={r['fwd_check_pk']} look={r['look_n']} "
              f"ss={r['safe_stop_n']} odo={f(r['odo_ratio'], 3)} cmp={f(r['compute_p50'], 0)}/{f(r['compute_p95'], 0)} "
              f"hl={[(round(t, 1), b, why) for (t, a, b, why) in r['health_transitions']][:4]} | r1: {(p.get('failure') or 'success') if p else ''} {p.get('final_error', '')}")


if __name__ == "__main__":
    sys.exit(main())
