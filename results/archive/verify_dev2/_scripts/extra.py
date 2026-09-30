"""Build results/verify_dev2/verify_extra.json: pytest summary, back-to-back baseline (fee1da9) comparison,
near-field false-lethal corridor counts, true-ditch labelling, perception A/B (if run), golden runs, findings,
and the extra claims. Every number is read from files written by the verifier's own runs / scripts.
Units: m, s, m/s, ms, counts.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Downloads\sih again\metagross")
VER = ROOT / "results" / "verify_dev2"
SRC = "results/verify_dev2/verify_dev2.json"


def read_text_any(p: Path) -> str:
    b = p.read_bytes()
    return b.replace(b"\x00", b"").decode("utf-8", errors="replace")


def pytest_summary(name: str = "pytest_full.log", when: str = "07:24-07:28, stereo batch running") -> dict:
    p = VER / name
    if not p.exists():
        return {}
    txt = read_text_any(p)
    m = re.findall(r"(\d+) failed, (\d+) passed", txt)
    failed = sorted(set(re.findall(r"FAILED (tests/\S+)", txt)))
    out = {"command": f"$env:OMP_NUM_THREADS=2; .venv\\Scripts\\python.exe -m pytest -q -p no:cacheprovider ({when})",
           "log": f"results/verify_dev2/{name}", "failed_tests": failed}
    if m:
        out["failed"], out["passed"] = int(m[-1][0]), int(m[-1][1])
    else:
        mp = re.findall(r"(\d+) passed", txt)
        out["failed"], out["passed"] = 0, int(mp[-1]) if mp else None
    t = re.findall(r"in ([\d.]+)s", txt)
    out["duration_s"] = float(t[-1]) if t else None
    return out


def results(group_dir: Path) -> dict[int, dict]:
    out = {}
    for rj in sorted(group_dir.glob("FULL/*/result.json")):
        r = json.loads(rj.read_text())
        cm = rj.parent / "gt" / "cmds.npz"
        arr = bool(np.any(np.load(cm)["mode"] == "ARRIVED")) if cm.exists() else None
        out[int(r["seed"])] = {"success": r["success"], "failure": r["failure_type"] or "success", "final_error": r["final_error"],
                               "ditch": r["ditch_entries"], "coll": r["collisions"], "water": r.get("water_entries", 0),
                               "mean_speed": r["mean_speed"], "arrived_est": arr, "compute_ms_p95": r.get("compute_ms_p95")}
    return out


def baseline_cmp() -> dict:
    new, old = results(VER / "tier0"), results(VER / "tier0_fee1da9")
    seeds = sorted(set(new) & set(old))
    if not seeds:
        return {}
    s = lambda d, k: sum(1 for x in seeds if d[x][k])  # noqa: E731
    hold = [x for x in seeds if x in (120, 121, 122, 126, 127, 128)]
    return {
        "what": "tier0 FULL, DEV seeds 100-129: fee1da9 (git archive snapshot run from scratch) vs working tree, "
                "each at 2 workers with one stereo worker in parallel (fee1da9 07:29-, working tree 07:02-07:24)",
        "n": len(seeds),
        "fee1da9": {"success": s(old, "success"), "arrived_est": s(old, "arrived_est"), "ditch": sum(old[x]["ditch"] for x in seeds),
                    "coll": sum(old[x]["coll"] for x in seeds), "water": sum(old[x]["water"] for x in seeds),
                    "final_error_p50": float(np.median([old[x]["final_error"] for x in seeds])),
                    "mean_speed": float(np.mean([old[x]["mean_speed"] for x in seeds])),
                    "holdout_success": f"{sum(1 for x in hold if old[x]['success'])}/{len(hold)}"},
        "tree": {"success": s(new, "success"), "arrived_est": s(new, "arrived_est"), "ditch": sum(new[x]["ditch"] for x in seeds),
                 "coll": sum(new[x]["coll"] for x in seeds), "water": sum(new[x]["water"] for x in seeds),
                 "final_error_p50": float(np.median([new[x]["final_error"] for x in seeds])),
                 "mean_speed": float(np.mean([new[x]["mean_speed"] for x in seeds])),
                 "holdout_success": f"{sum(1 for x in hold if new[x]['success'])}/{len(hold)}"},
        "flips": [(x, old[x]["failure"], new[x]["failure"]) for x in seeds if old[x]["failure"] != new[x]["failure"]],
    }


def main() -> None:
    extra: dict = {"pytest": pytest_summary(), "pytest_end": pytest_summary("pytest_full_end.log", "08:05, after all batches, no other verifier load"),
                   "baseline_fee1da9_vs_tree_tier0_full": baseline_cmp()}
    for name in ("fp_corridor", "ditch_cert", "perception_ab", "perception_122_ablation", "ditch_margin"):
        p = VER / f"{name}.json"
        if p.exists():
            extra[name] = json.loads(p.read_text())
    notes = VER / "_scripts" / "notes.json"
    if notes.exists():
        extra.update(json.loads(notes.read_text()))
    claims: dict = {}
    b = extra["baseline_fee1da9_vs_tree_tier0_full"]
    if b:
        note = b["what"]
        claims["tier0_full_baseline_fee1da9_success"] = {"value": f"{b['fee1da9']['success']}/{b['n']}", "unit": "runs", "note": note}
        claims["tier0_full_tree_success_same_session"] = {"value": f"{b['tree']['success']}/{b['n']}", "unit": "runs", "note": note}
    ab = extra.get("perception_ab")
    if ab:
        n_ab = "results/verify_dev2/_scripts/pd_ab.py: metagross.eval.perception_dev stereo metrics on identical cached DEV frames (drive 102,106,120,125,101; approach 103,121,122), fee1da9 perception copy vs working tree, run back to back 07:42-07:50"
        r1, r2 = ab["r1_fee1da9"]["stereo"]["pooled"], ab["r2_tree"]["stereo"]["pooled"]
        for key, unit in (("positive_rate_off", "fraction"), ("cand_rate_off", "fraction"), ("lethal_rate_off", "fraction")):
            claims[f"perception_ab_stereo_{key}"] = {"value": f"{r1[key]:.4f} -> {r2[key]:.4f}", "unit": unit,
                                                     "note": n_ab + "; share of observed off-hazard cells (> buffer from any GT hazard)"}
        ce1, ce2 = r1["certified_hazard_in_envelope"], r2["certified_hazard_in_envelope"]
        claims["perception_ab_stereo_certified_hazard_in_envelope"] = {
            "value": f"{ce1['cells']}/{ce1['of']} -> {ce2['cells']}/{ce2['of']}", "unit": "cells",
            "note": n_ab + "; GT lethal cells in the path-corridor stopping envelope that perception certifies as ground (all on 122 approach)"}
    abl = extra.get("perception_122_ablation")
    if abl:
        v = abl["no_min_obstacle_size"]["certified_hazard_in_envelope"]
        d = abl["tree_default"]["certified_hazard_in_envelope"]
        claims["perception_122_min_obstacle_size_ablation"] = {
            "value": f"{d['cells']}/{d['of']} -> {v['cells']}/{v['of']}", "unit": "cells",
            "note": "results/verify_dev2/_scripts/pd_122.py: DEV 122 stereo approach (cached), tree perception default vs min_obstacle_size=False; "
                    "certified hazard cells in the stopping envelope (other round-2 switches: no effect)"}
    agg = json.loads((VER / "verify_dev2_agg.json").read_text())
    st = [r for r in agg["per_seed"] if r["group"] == "stereo"]
    t1 = {r["seed"]: r for r in st if r["config"] == "T1"}
    fu = {r["seed"]: r for r in st if r["config"] == "FULL" and r["seed"] in t1}
    if t1:
        pair = {}
        for name, g in (("FULL", fu), ("T1", t1)):
            pair[name] = {k: sum(int(r[k]) for r in g.values()) for k in ("success", "ditch", "coll", "water", "oob")}
            pair[name]["mean_speed"] = float(np.mean([r["mean_speed"] for r in g.values()]))
            pair[name]["outcomes"] = {s: (g[s]["failure"] or "success") for s in sorted(g)}
        pair["seeds"] = sorted(t1)
        extra["stereo_full_vs_t1_same_seeds"] = pair
        note_p = (f"stereo FULL vs T1 (unknown_is_free, no negobs, no governor, 2.0 m/s) on the same {len(t1)} DEV seeds "
                  f"{sorted(t1)}; referee counts")
        for name in ("FULL", "T1"):
            p = pair[name]
            claims[f"stereo_{name.lower()}_same16_safety_events"] = {
                "value": f"success {p['success']}/{len(t1)}, ditch {p['ditch']}, collisions {p['coll']}, out-of-bounds {p['oob']}",
                "unit": "runs / counts", "note": note_p}
    fp = {r["run"].replace("\\", "/"): r for r in extra.get("fp_corridor", [])}
    extra["extra_claims"] = claims
    (VER / "verify_extra.json").write_text(json.dumps(extra, indent=1, default=str))
    print(json.dumps({k: (v if k != "fp_corridor" else len(v)) for k, v in extra.items() if k in ("pytest", "baseline_fee1da9_vs_tree_tier0_full")}, indent=1, default=str))
    print("fp runs:", list(fp)[:3])


if __name__ == "__main__":
    sys.exit(main())
