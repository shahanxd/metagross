"""What happened near the ditches in one run (verifier, sim/eval side: reads GT + scenario).

Usage: python ditch_story.py <run_dir> <scenario_json>
World frame: x east, y north (m). Debug bundles are A-frame / egocentric (autonomy side).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\Downloads\sih again\metagross")
from metagross.contracts.messages import CellState  # noqa: E402


def seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    q = a + t[:, None] * ab
    return np.linalg.norm(p - q, axis=1), t


def main(run: Path, scen: Path) -> None:
    s = json.loads(scen.read_text())
    gt = np.load(run / "gt" / "states.npz")
    t, P = gt["t"], np.stack([gt["x"], gt["y"]], 1)
    v = gt["v"]
    res = json.loads((run / "result.json").read_text())
    print(f"run {run} cfg={res['config']['name']} outcome={res['failure_type']} success={res['success']} path={res['path_length']:.1f} "
          f"ferr={res['final_error']:.2f} ditch={res['ditch_entries']} t_end={res['time']}")
    print("events:", (run / "gt" / "events.json").read_text().replace("\n", " "))
    for hi, h in enumerate(s["hazards"]):
        if h["type"] != "ditch":
            continue
        L = np.array(h["polyline"])
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(L, axis=0), axis=1))]
        best = np.full(len(P), np.inf)
        best_arc = np.zeros(len(P))
        for k in range(len(L) - 1):
            d, tt = seg_dist(P, L[k], L[k + 1])
            m = d < best
            best[m] = d[m]
            best_arc[m] = arc[k] + tt[m] * (arc[k + 1] - arc[k])
        in_gap = np.zeros(len(P), bool)
        for g0, g1 in h["gaps"]:
            in_gap |= (best_arc >= g0) & (best_arc <= g1)
        gmid = [(g0 + g1) / 2 for g0, g1 in h["gaps"]]
        gxy = [np.array([np.interp(a, arc, L[:, 0]), np.interp(a, arc, L[:, 1])]) for a in gmid]
        solid = best.copy()
        solid[in_gap] = np.inf
        i = int(np.argmin(solid))
        # edge distance = centre distance - half width
        print(f"  ditch {hi}: width {h['width']:.2f} depth {h['depth']:.2f}; gap centre(s) {[g.round(1).tolist() for g in gxy]}")
        print(f"    closest approach to solid ditch centreline: {solid[i]:.2f} m (edge {solid[i]-h['width']/2:.2f} m) at t={t[i]:.1f}s "
              f"pos=({P[i,0]:.1f},{P[i,1]:.1f}) v={v[i]:.2f} m/s")
        crossed = np.flatnonzero(best < h["width"] / 2 + 0.3)
        if crossed.size:
            print(f"    within half-width+0.3 m of centreline: t={t[crossed[0]]:.1f}-{t[crossed[-1]]:.1f}s, in gap: {bool(in_gap[crossed].all())}")
        dg = min(np.linalg.norm(P - g, axis=1).min() for g in gxy)
        print(f"    closest approach to a gap centre: {dg:.2f} m")
        # debug bundles near the closest approach
        dbg = sorted((run / "autonomy" / "debug").glob("tick_*.npz"))
        ts = []
        for f in dbg:
            z = np.load(f, allow_pickle=True)
            ts.append((float(z["t"]), f))
        near = [f for tt_, f in ts if abs(tt_ - t[i]) <= 6.0]
        for f in near:
            z = np.load(f, allow_pickle=True)
            ex = json.loads(str(z["extras_json"]))
            cs = z["cell_state_local"]
            n_d = int((cs == CellState.DITCH_CANDIDATE).sum())
            n_dep = int((cs == CellState.DEPRESSION).sum())
            n_pos = int((cs == CellState.POSITIVE).sum())
            n_cs = int((cs == CellState.CREST_SHADOW).sum())
            crop = z["extra_map_crop_state"]
            conf = z["extra_map_crop_confirmed"]
            n_conf = int(((crop == CellState.DITCH_CANDIDATE) & conf).sum())
            gi = int(np.argmin(np.abs(t - float(z["t"]))))
            print(f"    t={float(z['t']):5.1f} GT({P[gi,0]:.1f},{P[gi,1]:.1f}) v_gt={v[gi]:.2f} dist_ditch={best[gi]:.2f} | mode={ex['mode']} reason={ex['reason']} "
                  f"v_cap={float(z['v_cap_mps']):.2f} R_cert={float(z['r_cert_m']):.2f} cmd_v={ex['cmd_v']:.2f} route_ok={ex['route_ok']} "
                  f"bind={ex['gov_binding']} gov={ {k: round(val, 2) for k, val in ex['gov_terms'].items()} } | BEV ditch={n_d} dep={n_dep} pos={n_pos} crest={n_cs} "
                  f"map_conf_ditch={n_conf} new_lethal={ex['map_new_lethal']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
