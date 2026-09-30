"""Independent aggregation of the verifier's DEV closed-loop runs (sim side, reads GT results).

Walks <root>/<group>/<config>/<seed:03d>/result.json under results/verify_dev and writes
results/verify_dev/verify_dev.json + per_seed.csv, and prints markdown tables. Also compares, per
(config, seed), with the planner's runs in results/runs_dev_tier0 and results/runs_dev_stereo.
Units: m, s, m/s, ms.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Downloads\sih again\metagross")
VER = ROOT / "results" / "verify_dev"
PLANNER = {"tier0": ROOT / "results" / "runs_dev_tier0", "tier0_noneg": None, "stereo": ROOT / "results" / "runs_dev_stereo"}
LOG = logging.getLogger("verify_agg")


def t_arrived(run: Path) -> float | None:
    """First telemetry time (s) at which the autonomy declared ARRIVED in its own estimate."""
    tel = run / "autonomy" / "telemetry.jsonl"
    if not tel.exists():
        return None
    with tel.open() as f:
        for line in f:
            if '"ARRIVED"' in line:
                d = json.loads(line)
                if d.get("mode") == "ARRIVED":
                    return float(d["t"])
    return None


def arrived_est(run: Path) -> bool:
    return t_arrived(run) is not None


def modes(run: Path) -> dict[str, int]:
    tel = run / "autonomy" / "telemetry.jsonl"
    out: dict[str, int] = defaultdict(int)
    if tel.exists():
        with tel.open() as f:
            for line in f:
                try:
                    out[json.loads(line)["mode"]] += 1
                except Exception:  # noqa: BLE001
                    pass
    return dict(out)


def compute_ms(run: Path) -> np.ndarray:
    p = run / "autonomy" / "timings.csv"
    if not p.exists():
        return np.zeros(0)
    vals = []
    with p.open() as f:
        for r in csv.DictReader(f):
            try:
                vals.append(float(r["compute_ms"]))
            except Exception:  # noqa: BLE001
                pass
    return np.asarray(vals)


def gt_speed_first(run: Path, t_max: float = 20.0) -> float | None:
    p = run / "gt" / "states.npz"
    if not p.exists():
        return None
    z = np.load(p)
    keys = list(z.keys())
    t = z["t"] if "t" in keys else None
    if t is None or "x" not in keys or "y" not in keys:
        return None
    xy = np.stack([z["x"], z["y"]], axis=1)
    m = t <= t_max
    if m.sum() < 2:
        return None
    d = np.linalg.norm(np.diff(xy[m, :2], axis=0), axis=1).sum()
    return float(d / max(t[m][-1] - t[m][0], 1e-6))


def load_rows(group: str) -> list[dict]:
    rows = []
    base = VER / group
    if not base.exists():
        return rows
    for rj in sorted(base.glob("*/*/result.json")):
        run = rj.parent
        r = json.loads(rj.read_text())
        cms = compute_ms(run)
        ta = t_arrived(run)
        fs_terminal = sum(1 for s in r.get("stops", []) if not s["justified"] and ta is not None and s["t"] >= ta - 1.0)
        row = {
            "group": group, "config": r["config"]["name"], "seed": int(r["seed"]), "family": r["family"], "split": r["split"],
            "success": bool(r["success"]), "failure": r["failure_type"], "time_s": r["time"], "path_m": r["path_length"],
            "opt_m": r["optimal_path_length"], "spl": r["spl"], "mean_speed": r["mean_speed"], "final_error": r["final_error"],
            "ditch": r["ditch_entries"], "coll": r["collisions"], "water": r.get("water_entries", 0), "false_stops": r["false_stops"],
            "min_clearance": r.get("min_clearance"), "arrived_est": ta is not None, "t_arrived_est": ta,
            "false_stops_terminal": fs_terminal, "n_ticks": int(cms.size),
            "compute_p50": float(np.percentile(cms, 50)) if cms.size else None,
            "compute_p95": float(np.percentile(cms, 95)) if cms.size else None,
            "gt_speed_first20": gt_speed_first(run), "error": r.get("error"), "modes": modes(run),
            "_cms": cms,
        }
        pl = PLANNER.get(group)
        if pl is not None:
            prj = pl / row["config"] / f"{row['seed']:03d}" / "result.json"
            if prj.exists():
                p = json.loads(prj.read_text())
                row["planner"] = {"success": p["success"], "failure": p["failure_type"], "path_m": p["path_length"],
                                  "final_error": p["final_error"], "spl": p["spl"], "mean_speed": p["mean_speed"],
                                  "ditch": p["ditch_entries"], "coll": p["collisions"], "water": p.get("water_entries", 0),
                                  "false_stops": p["false_stops"], "arrived_est": arrived_est(prj.parent)}
        rows.append(row)
    return rows


def agg(rows: list[dict]) -> dict:
    cms = np.concatenate([r["_cms"] for r in rows]) if rows else np.zeros(0)
    fails = defaultdict(int)
    for r in rows:
        fails[r["failure"] or "success"] += 1
    return {
        "n": len(rows), "success": sum(r["success"] for r in rows),
        "spl_mean": float(np.mean([r["spl"] for r in rows])) if rows else None,
        "mean_speed_mean": float(np.mean([r["mean_speed"] for r in rows])) if rows else None,
        "ditch": sum(r["ditch"] for r in rows), "coll": sum(r["coll"] for r in rows), "water": sum(r["water"] for r in rows),
        "false_stops": sum(r["false_stops"] for r in rows), "arrived_est": sum(r["arrived_est"] for r in rows),
        "false_stops_after_est_arrival": sum(r["false_stops_terminal"] for r in rows),
        "timeouts": sum(1 for r in rows if r["failure"] == "timeout"), "stuck": sum(1 for r in rows if r["failure"] == "stuck"),
        "final_error_median": float(np.median([r["final_error"] for r in rows])) if rows else None,
        "outcomes": dict(fails),
        "compute_p50": float(np.percentile(cms, 50)) if cms.size else None,
        "compute_p95": float(np.percentile(cms, 95)) if cms.size else None,
        "n_ticks": int(cms.size),
    }


def fam(s: str) -> str:
    return s.split("_")[0]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out: dict = {"what": "Independent verifier re-run of DEV closed loop (metagross.sim.batch), DEV seeds only",
                 "groups": {}, "per_seed": []}
    all_rows = []
    for group in ("tier0", "tier0_noneg", "stereo"):
        rows = load_rows(group)
        if not rows:
            continue
        all_rows += rows
        g: dict = {}
        by = defaultdict(list)
        for r in rows:
            by[(r["config"], fam(r["family"]))].append(r)
            by[(r["config"], "ALL")].append(r)
        for (cfg, f), rr in sorted(by.items()):
            g.setdefault(cfg, {})[f] = agg(rr)
        # planner comparison on identical (config, seed)
        cmp = defaultdict(list)
        for r in rows:
            if "planner" in r:
                cmp[r["config"]].append(r)
        g["_planner_same_seeds"] = {
            c: {"n": len(rr), "verifier_success": sum(r["success"] for r in rr), "planner_success": sum(r["planner"]["success"] for r in rr),
                "outcome_mismatch": [(r["seed"], r["failure"], r["planner"]["failure"]) for r in rr if r["failure"] != r["planner"]["failure"]],
                "verifier_arrived_est": sum(r["arrived_est"] for r in rr), "planner_arrived_est": sum(r["planner"]["arrived_est"] for r in rr)}
            for c, rr in cmp.items()}
        out["groups"][group] = g
    for r in all_rows:
        rr = {k: v for k, v in r.items() if k != "_cms"}
        out["per_seed"].append(rr)
    out["eval_seed_rows"] = [r["seed"] for r in all_rows if r["split"] != "dev" or r["seed"] < 100]
    src = "results/verify_dev/verify_dev.json"
    claims = []
    for group, cfg, cid in (("tier0", "FULL", "full"), ("tier0", "TYPICAL", "typical"), ("tier0_noneg", "NO-NEG", "noneg"), ("stereo", "FULL", "stereo_full")):
        a = out["groups"].get(group, {}).get(cfg, {}).get("ALL")
        if not a:
            continue
        seeds = sorted({r["seed"] for r in all_rows if r["group"] == group and r["config"] == cfg})
        note = (f"verifier re-run, metagross.sim.batch {group} config {cfg}, DEV seeds {seeds[0]}-{seeds[-1]} (n={a['n']}), "
                f"2 workers on a shared laptop; success = referee GT goal within 2 m")
        claims += [
            {"id": f"verify_dev_{cid}_success", "value": f"{a['success']}/{a['n']}", "label": "Simulated", "source": src, "note": note},
            {"id": f"verify_dev_{cid}_arrived_est", "value": f"{a['arrived_est']}/{a['n']}", "label": "Simulated", "source": src,
             "note": note + "; arrived_est = ARRIVED declared in the vehicle's own pose estimate"},
            {"id": f"verify_dev_{cid}_ditch_entries", "value": str(a["ditch"]), "label": "Simulated", "source": src, "note": note},
            {"id": f"verify_dev_{cid}_collisions", "value": str(a["coll"]), "label": "Simulated", "source": src, "note": note},
        ]
    out["claims"] = claims
    (VER / "verify_dev.json").write_text(json.dumps(out, indent=1, default=str))
    keys = ["group", "config", "seed", "family", "success", "failure", "time_s", "path_m", "opt_m", "spl", "mean_speed", "final_error",
            "ditch", "coll", "water", "false_stops", "min_clearance", "arrived_est", "compute_p50", "compute_p95", "gt_speed_first20", "error"]
    with (VER / "per_seed.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + ["planner_success", "planner_failure", "planner_final_error"])
        for r in all_rows:
            p = r.get("planner", {})
            w.writerow([r.get(k) for k in keys] + [p.get("success"), p.get("failure"), p.get("final_error")])
    # print tables
    for group, g in out["groups"].items():
        print(f"\n### {group}")
        print("| config | family | n | success | arrived_est | SPL | mean speed | ditch | coll | water | false stops (after est. arrival) | outcomes | final err med | compute p50/p95 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for cfg, fams in g.items():
            if cfg.startswith("_"):
                continue
            for f, a in fams.items():
                print(f"| {cfg} | {f} | {a['n']} | {a['success']} | {a['arrived_est']} | {a['spl_mean']:.2f} | {a['mean_speed_mean']:.2f} | "
                      f"{a['ditch']} | {a['coll']} | {a['water']} | {a['false_stops']} ({a['false_stops_after_est_arrival']}) | {a['outcomes']} | {a['final_error_median']:.2f} | "
                      f"{a['compute_p50']:.0f}/{a['compute_p95']:.0f} |")
        print("planner same seeds:", json.dumps(g.get("_planner_same_seeds"), default=str))
    print("\nper seed:")
    for r in all_rows:
        p = r.get("planner", {})
        print(f"{r['group']:11s} {r['config']:7s} {r['seed']} {fam(r['family'])} succ={int(r['success'])} {r['failure']!s:14s} "
              f"path={r['path_m']:.1f}/{r['opt_m']:.1f} ferr={r['final_error']:.2f} v={r['mean_speed']:.2f} arr={int(r['arrived_est'])} "
              f"d/c/w={r['ditch']}/{r['coll']}/{r['water']} fs={r['false_stops']} | planner: {p.get('success')} {p.get('failure')} ferr={p.get('final_error')}")


if __name__ == "__main__":
    sys.exit(main())
