"""Recompute the planner's claimed per-family tables from the planner's own run files (verifier)."""
import collections
import csv
import datetime
import glob
import json
import os

import numpy as np

BASE = r"D:\Downloads\sih again\metagross\results"
for sub, cfgs in (("runs_dev_tier0", ("FULL", "TYPICAL")), ("runs_dev_stereo", ("FULL",))):
    for cfg in cfgs:
        paths = sorted(glob.glob(os.path.join(BASE, sub, cfg, "*", "result.json")))
        rows = [(p, json.load(open(p))) for p in paths]
        fams = collections.defaultdict(list)
        for p, r in rows:
            fams[r["family"][:2]].append((p, r))
            fams["ALL"].append((p, r))
        for f, rr in sorted(fams.items()):
            cms = []
            for p, r in rr:
                tp = os.path.join(os.path.dirname(p), "autonomy", "timings.csv")
                if os.path.exists(tp):
                    cms += [float(d["compute_ms"]) for d in csv.DictReader(open(tp))]
            cms = np.array(cms)
            print(sub, cfg, f, len(rr), "succ", sum(r["success"] for _, r in rr),
                  "spl", round(float(np.mean([r["spl"] for _, r in rr])), 2),
                  "v", round(float(np.mean([r["mean_speed"] for _, r in rr])), 2),
                  "ditch", sum(r["ditch_entries"] for _, r in rr), "coll", sum(r["collisions"] for _, r in rr),
                  "water", sum(r.get("water_entries", 0) for _, r in rr), "fs", sum(r["false_stops"] for _, r in rr),
                  "p50/p95", (round(float(np.percentile(cms, 50))), round(float(np.percentile(cms, 95)))) if cms.size else None,
                  dict(collections.Counter(r["failure_type"] for _, r in rr)))
        mt = [os.path.getmtime(p) for p in paths]
        if mt:
            print("  result.json mtime", datetime.datetime.fromtimestamp(min(mt)).strftime("%H:%M"), "-",
                  datetime.datetime.fromtimestamp(max(mt)).strftime("%H:%M"), "seeds", sorted(r["seed"] for _, r in rows))
