"""Verifier (sim side): closest GT approach to a ditch edge per F2/F3 run, and the GT speed there.

For every run under results/verify_dev2/{tier0,stereo}/{FULL,T1}/<seed> of families F2/F3: GT body-origin
distance (m) to the nearest ditch edge (polyline distance - width/2, gap stretches excluded) sampled every
5 physics steps; prints the minimum, the GT speed (m/s) at that moment, and the max GT speed while within
1 m of a ditch edge. Writes results/verify_dev2/ditch_margin.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from golden import hazard_dist  # noqa: E402

ROOT = Path(r"D:\Downloads\sih again\metagross")
VER = ROOT / "results" / "verify_dev2"
NEAR_M = 1.0
STRIDE = 5


def main() -> None:
    rows = []
    for grp in ("tier0/FULL", "tier0/T1", "stereo/FULL", "stereo/T1"):
        for rj in sorted((VER / grp).glob("*/result.json")):
            r = json.loads(rj.read_text())
            if r["family"][:2] not in ("F2", "F3"):
                continue
            scn = json.loads((ROOT / "data" / "scenarios" / r["split"] / f"{r['seed']}.json").read_text())
            z = np.load(rj.parent / "gt" / "states.npz")
            xy = np.stack([z["x"][::STRIDE], z["y"][::STRIDE]], 1)
            v = z["v"][::STRIDE]
            d = hazard_dist(scn, xy)["ditch"]
            if not np.isfinite(d).any():
                continue
            i = int(np.argmin(d))
            near = d < NEAR_M
            rows.append({"group": grp, "seed": int(r["seed"]), "family": r["family"], "outcome": r["failure_type"] or "success",
                         "min_edge_dist_m": round(float(d[i]), 2), "speed_at_min_mps": round(float(v[i]), 2),
                         "max_speed_within_1m_mps": round(float(v[near].max()), 2) if near.any() else None})
    for x in rows:
        print(x)
    (VER / "ditch_margin.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
