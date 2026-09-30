"""Verifier (sim side): how per-frame perception labels TRUE ditch cells in stereo debug bundles.

For each debug bundle, BEV cell centres (egocentric, 0.1 m, x fwd in [-2, 14), y left in [-6, 6)) are mapped to the
world with the GT pose; a cell is a GT ditch cell when its centre lies inside a ditch (distance to the ditch
polyline, gap stretches excluded, < width/2). Per forward-range bin (body x, m) and |y| <= 3 m we count the
labels perception gave those cells: GROUND (1, i.e. certified drivable = the dangerous error), DITCH_CANDIDATE (4),
DEPRESSION (3), UNSEEN (0), other. Also the same for the rolling-map crop (extra_map_crop_state) is not attempted.
Usage: python ditch_cert.py <run_dir> [...]
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from golden import densify  # noqa: E402

ROOT = Path(r"D:\Downloads\sih again\metagross")
RES, X0, Y0 = 0.10, -2.0, -6.0
BINS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 14)]
LAT_M = 3.0
NAMES = {0: "unseen", 1: "GROUND", 3: "depr", 4: "DITCH", 2: "pos", 5: "crest_sh", 6: "occl", 7: "water"}


def main() -> None:
    for arg in sys.argv[1:]:
        run = Path(arg) if Path(arg).is_absolute() else ROOT / arg
        res = json.loads((run / "result.json").read_text())
        scn = json.loads((ROOT / "data" / "scenarios" / res["split"] / f"{res['seed']}.json").read_text())
        pts, halfw = [], []
        for h in scn["hazards"]:
            if h["type"] != "ditch":
                continue
            p, s = densify(np.asarray(h["polyline"], float))
            keep = np.ones(len(p), bool)
            for g0, g1 in h.get("gaps", []):
                keep &= ~((s >= g0) & (s <= g1))
            pts.append(p[keep])
            halfw.append(np.full(keep.sum(), 0.5 * float(h["width"])))
        if not pts:
            print(run.relative_to(ROOT), "no ditch in scenario")
            continue
        P, HW = np.vstack(pts), np.concatenate(halfw)
        tree = cKDTree(P)
        z = np.load(run / "gt" / "states.npz")
        xs = X0 + (np.arange(160) + 0.5) * RES
        ys = Y0 + (np.arange(120) + 0.5) * RES
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        lat = np.abs(YY) <= LAT_M
        counts = {b: {} for b in BINS}
        for f in sorted(glob.glob(str(run / "autonomy" / "debug" / "tick_*.npz"))):
            b = np.load(f)
            st = b["cell_state_local"]
            t = float(b["t"])
            i = min(np.searchsorted(z["t"], t), len(z["t"]) - 1)
            x, y, yaw = z["x"][i], z["y"][i], z["yaw"][i]
            wx = x + XX * math.cos(yaw) - YY * math.sin(yaw)
            wy = y + XX * math.sin(yaw) + YY * math.cos(yaw)
            d, k = tree.query(np.stack([wx.ravel(), wy.ravel()], 1), distance_upper_bound=2.0)
            inside = np.zeros(d.shape, bool)
            ok = np.isfinite(d)
            inside[ok] = d[ok] < HW[k[ok]]
            inside = inside.reshape(XX.shape) & lat
            for (a0, a1) in BINS:
                m = inside & (XX >= a0) & (XX < a1)
                if not m.any():
                    continue
                v, c = np.unique(st[m], return_counts=True)
                for vv, cc in zip(v, c):
                    counts[(a0, a1)][int(vv)] = counts[(a0, a1)].get(int(vv), 0) + int(cc)
        RESULTS.append({"run": str(run.relative_to(ROOT)), "family": res["family"], "config": res["config"]["name"],
                        "bins": {f"{a0}-{a1}": {"n": sum(c.values()), "ground": c.get(1, 0),
                                                "ground_frac": (c.get(1, 0) / sum(c.values())) if sum(c.values()) else None,
                                                "labels": {NAMES.get(k, str(k)): v for k, v in c.items()}}
                                 for (a0, a1), c in counts.items()}})
        print(f"{run.relative_to(ROOT)} ({res['family']} {res['config']['name']}): GT-ditch cells by perception label, |y|<={LAT_M} m")
        for bb, c in counts.items():
            tot = sum(c.values())
            if not tot:
                continue
            g = c.get(1, 0)
            print(f"   x {bb[0]:>2}-{bb[1]:<2} m: n={tot:5d}  GROUND={g} ({g / tot:.0%})  "
                  + "  ".join(f"{NAMES.get(kk, kk)}={vv}" for kk, vv in sorted(c.items()) if kk != 1))


RESULTS: list[dict] = []

if __name__ == "__main__":
    import os

    main()
    if os.environ.get("DC_OUT"):
        Path(os.environ["DC_OUT"]).write_text(json.dumps(RESULTS, indent=1))
