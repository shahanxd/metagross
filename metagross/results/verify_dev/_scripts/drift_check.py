"""Red-team: check the 'tier0 wheel-odometry over-counts distance' root-cause claim.

For every tier0 FULL DEV run: compare the autonomy's own travelled distance (telemetry pose, 2 Hz)
with the GT travelled distance (gt/states.npz) over the same time span. Offline eval only.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Downloads\sih again\metagross\results\runs_dev_tier0") / sys.argv[1]
first = True
rows = []
for run in sorted(ROOT.iterdir()):
    if not (run / "result.json").exists():
        continue
    st = np.load(run / "gt" / "states.npz")
    if first:
        print("states keys:", {k: st[k].shape for k in st.files})
        first = False
    tel = [json.loads(l) for l in (run / "autonomy" / "telemetry.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    tp = np.array([p["t"] for p in tel])
    pose = np.array([p["pose"][:2] for p in tel])
    est_len = float(np.sum(np.linalg.norm(np.diff(pose, axis=0), axis=1)))
    t = st["t"]
    xy = np.stack([st["x"], st["y"]], axis=1)
    # GT path sampled at telemetry times (same 2 Hz sampling so polyline shortcutting is comparable)
    gx = np.interp(tp, t, xy[:, 0]); gy = np.interp(tp, t, xy[:, 1])
    gt_len = float(np.sum(np.hypot(np.diff(gx), np.diff(gy))))
    res = json.loads((run / "result.json").read_text(encoding="utf-8"))
    rows.append((run.name, res["family"], res["success"], res["failure_type"], est_len, gt_len, est_len / max(gt_len, 1e-6)))
for r in rows:
    print("%s %-20s succ=%-5s %-12s est=%6.2f gt=%6.2f ratio=%.3f" % r)
ratios = np.array([r[6] for r in rows if r[5] > 5])
print("n=%d ratio est/gt median %.3f min %.3f max %.3f" % (len(ratios), np.median(ratios), ratios.min(), ratios.max()))
