"""Build the 'extra' block (before/after on the same DEV seeds, notes, claims) for closed_loop_dev.json."""
import csv, json, sys
from pathlib import Path
import numpy as np

REPO = Path(r"D:\Downloads\sih again\metagross")
before_csv = REPO / "results/runs_integration/tier0/summary.csv"
after_root = REPO / "results/runs_dev_tier0"
WINDOW_S = 20.0  # the 'before' runs were capped at 20 s of sim time (max_sim_s=20)

before = list(csv.DictReader(open(before_csv, encoding="utf-8")))
rows = []
for b in before:
    cfg, seed = b["config"], int(b["seed"])
    d = after_root / cfg / f"{seed:03d}"
    if not (d / "result.json").exists():
        continue
    r = json.loads((d / "result.json").read_text())
    gt = np.load(d / "gt" / "states.npz")
    k = gt["t"] <= WINDOW_S + 1e-9
    path20 = float(np.hypot(np.diff(gt["x"][k]), np.diff(gt["y"][k])).sum())
    rows.append({"config": cfg, "seed": seed, "family": b["family"],
                 "before_path_m_first20s": float(b["path_length"]), "before_mean_speed_mps": float(b["mean_speed"]),
                 "before_failure": b["failure_type"],
                 "after_path_m_first20s": round(path20, 3), "after_mean_speed_first20s_mps": round(path20 / WINDOW_S, 4),
                 "after_success": r["success"], "after_failure": r["failure_type"], "after_final_error_m": r["final_error"]})
extra = {"before_after_same_seeds": {
    "before_source": "results/runs_integration/tier0/summary.csv (commit e2cbefd, integration pass, 20 s sim cap)",
    "after_source": "results/runs_dev_tier0/<config>/<seed>/ (this pass, full mission timeout)",
    "metric": "GT path length in the first 20 s of sim time / 20 s (same window both sides)",
    "rows": rows}}
json.dump(extra, open(sys.argv[1], "w"), indent=1)
for r in rows:
    print(r["config"], r["seed"], r["before_mean_speed_mps"], "->", r["after_mean_speed_first20s_mps"], r["after_failure"])
