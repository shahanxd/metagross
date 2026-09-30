"""Verifier (sim side): near-field POSITIVE / DITCH_CANDIDATE cells in the drive corridor of stereo debug bundles.

For each run dir, over debug bundles (autonomy/debug/tick_*.npz, cell_state_local = egocentric BEV, 0.1 m cells,
x in [-2, 14) m forward, y in [-6, 6) m left), counts cells with state POSITIVE (2) and DITCH_CANDIDATE (4) in the
corridor x in [0, 6] m, |y| <= 2 m. Each bundle is tagged 'clear' when GT has no ditch / crest / water within 10 m
of the vehicle and no static object / tree centre inside the corridor (+0.5 m margin) -- then every such cell is false.
Reports, for the start bundle (t = 0) and over all 'clear' bundles: max / median counts and the fraction of clear
bundles with >= 1 such cell. Units: counts of 0.1 m cells, m, s.
Usage: python fp_corridor.py <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from golden import hazard_dist  # noqa: E402

ROOT = Path(r"D:\Downloads\sih again\metagross")
RES, X0, Y0 = 0.10, -2.0, -6.0
CORR_X, CORR_Y = (0.0, 6.0), 2.0
HAZ_CLEAR_M = 10.0
OBJ_MARGIN_M = 0.5
POSITIVE, DITCH = 2, 4


def corridor_mask(shape: tuple[int, int]) -> np.ndarray:
    xs = X0 + (np.arange(shape[0]) + 0.5) * RES
    ys = Y0 + (np.arange(shape[1]) + 0.5) * RES
    return ((xs >= CORR_X[0]) & (xs <= CORR_X[1]))[:, None] & (np.abs(ys) <= CORR_Y)[None, :]


def main() -> None:
    for arg in sys.argv[1:]:
        run = Path(arg) if Path(arg).is_absolute() else ROOT / arg
        res = json.loads((run / "result.json").read_text())
        scn = json.loads((ROOT / "data" / "scenarios" / res["split"] / f"{res['seed']}.json").read_text())
        z = np.load(run / "gt" / "states.npz")
        objs = []
        for o in scn["objects"]:
            p = o.get("xyz") or o.get("xy")
            r = float(o.get("radius", o.get("trunk_r", 0.3)))
            objs.append((p[0], p[1], r))
        objs = np.asarray(objs) if objs else np.zeros((0, 3))
        rows = []
        for f in sorted(glob.glob(str(run / "autonomy" / "debug" / "tick_*.npz"))):
            b = np.load(f)
            st = b["cell_state_local"]
            m = corridor_mask(st.shape)
            t = float(b["t"])
            i = min(np.searchsorted(z["t"], t), len(z["t"]) - 1)
            x, y, yaw = z["x"][i], z["y"][i], z["yaw"][i]
            hd = hazard_dist(scn, np.array([[x, y]]))
            dh = min(hd["ditch"][0], hd["crest"][0], abs(hd["water"][0]))
            obj_in = False
            if len(objs):
                dx, dy = objs[:, 0] - x, objs[:, 1] - y
                fx, fy = dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
                obj_in = bool(np.any((fx >= CORR_X[0] - OBJ_MARGIN_M - objs[:, 2]) & (fx <= CORR_X[1] + OBJ_MARGIN_M + objs[:, 2])
                                     & (np.abs(fy) <= CORR_Y + OBJ_MARGIN_M + objs[:, 2])))
            rows.append((t, int(np.sum(st[m] == POSITIVE)), int(np.sum(st[m] == DITCH)), dh, obj_in))
        if not rows:
            print(run, "no debug bundles")
            continue
        a = np.array([(r[0], r[1], r[2], r[3], r[4]) for r in rows], dtype=float)
        clear = (a[:, 3] > HAZ_CLEAR_M) & (a[:, 4] < 0.5)
        c = a[clear]
        s0 = a[0]
        RESULTS.append({"run": str(run.relative_to(ROOT)), "n_bundles": len(a), "n_clear": int(clear.sum()),
                        "start_pos": int(s0[1]), "start_ditch": int(s0[2]),
                        "clear_pos_max": int(c[:, 1].max()) if len(c) else None, "clear_pos_median": float(np.median(c[:, 1])) if len(c) else None,
                        "clear_pos_frac_ge1": float(np.mean(c[:, 1] >= 1)) if len(c) else None,
                        "clear_ditch_max": int(c[:, 2].max()) if len(c) else None, "clear_ditch_median": float(np.median(c[:, 2])) if len(c) else None,
                        "clear_ditch_frac_ge1": float(np.mean(c[:, 2] >= 1)) if len(c) else None})
        print(f"{run.relative_to(ROOT)}: n_bundles={len(a)} clear={int(clear.sum())} | start t={s0[0]:.1f}: POS={int(s0[1])} DITCH={int(s0[2])} "
              f"(GT hazard {s0[3]:.1f} m, object in corridor={bool(s0[4])}) | clear bundles: POS max/median {int(c[:, 1].max()) if len(c) else '-'}/"
              f"{np.median(c[:, 1]) if len(c) else '-'}, frac>=1 {np.mean(c[:, 1] >= 1) if len(c) else float('nan'):.2f}; "
              f"DITCH max/median {int(c[:, 2].max()) if len(c) else '-'}/{np.median(c[:, 2]) if len(c) else '-'}, frac>=1 {np.mean(c[:, 2] >= 1) if len(c) else float('nan'):.2f}")


RESULTS: list[dict] = []

if __name__ == "__main__":
    import os

    main()
    if os.environ.get("FP_OUT"):
        Path(os.environ["FP_OUT"]).write_text(json.dumps(RESULTS, indent=1))
