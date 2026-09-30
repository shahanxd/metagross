"""Verifier helper (sim side, reads GT): per-run timeline of hazard proximity vs governor / mode, to pick golden runs.

For a run dir <run>/ (result.json, gt/states.npz, autonomy/telemetry.jsonl) prints, every telemetry packet (~2 Hz):
t [s], GT speed [m/s], GT distance [m] to the nearest ditch edge (polyline distance - width/2, gap stretches
excluded) / crest line / water polygon edge (negative = inside), telemetry mode, v_cap [m/s], R_cert [m], q, reason.
Also prints mode transitions from autonomy.log and lighting events of the scenario.
Usage: python golden.py <run_dir> [--near 8]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Downloads\sih again\metagross")
DENSIFY_M = 0.1  # polyline densification step [m]


def densify(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(points (N,2), arc length (N,)) of a polyline resampled every DENSIFY_M."""
    pts, s = [poly[0]], [0.0]
    acc = 0.0
    for a, b in zip(poly[:-1], poly[1:]):
        L = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(L / DENSIFY_M)))
        for k in range(1, n + 1):
            pts.append(a + (b - a) * k / n)
            s.append(acc + L * k / n)
        acc += L
    return np.asarray(pts), np.asarray(s)


def hazard_dist(scn: dict, xy: np.ndarray) -> dict[str, np.ndarray]:
    """GT distance [m] from each xy (N,2) to the nearest hazard of each kind (inf if none)."""
    out = {"ditch": np.full(len(xy), np.inf), "crest": np.full(len(xy), np.inf), "water": np.full(len(xy), np.inf)}
    for h in scn["hazards"]:
        typ = h["type"]
        if typ in ("ditch", "crest"):
            p, s = densify(np.asarray(h["polyline"], float))
            keep = np.ones(len(p), bool)
            for g0, g1 in h.get("gaps", []):
                keep &= ~((s >= g0) & (s <= g1))
            p = p[keep]
            if len(p) == 0:
                continue
            d = np.min(np.linalg.norm(xy[:, None, :] - p[None, :, :], axis=2), axis=1)
            if typ == "ditch":
                d = d - 0.5 * float(h.get("width", 0.0))
            out[typ] = np.minimum(out[typ], d)
        elif typ == "water":
            poly = np.asarray(h["polygon"], float)
            p, _ = densify(np.vstack([poly, poly[:1]]))
            d = np.min(np.linalg.norm(xy[:, None, :] - p[None, :, :], axis=2), axis=1)
            inside = np.array([point_in_poly(q, poly) for q in xy])
            out["water"] = np.minimum(out["water"], np.where(inside, -d, d))
    return out


def point_in_poly(q: np.ndarray, poly: np.ndarray) -> bool:
    x, y = q
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--near", type=float, default=1e9, help="print only rows with a hazard closer than this [m]")
    a = ap.parse_args()
    run = a.run if a.run.is_absolute() else ROOT / a.run
    res = json.loads((run / "result.json").read_text())
    scn = json.loads((ROOT / "data" / "scenarios" / res["split"] / f"{res['seed']}.json").read_text())
    z = np.load(run / "gt" / "states.npz")
    tel = [json.loads(l) for l in (run / "autonomy" / "telemetry.jsonl").read_text().splitlines() if l.strip()]
    t = np.array([d["t"] for d in tel])
    xy = np.stack([np.interp(t, z["t"], z["x"]), np.interp(t, z["t"], z["y"])], axis=1)
    v = np.interp(t, z["t"], z["v"])
    hd = hazard_dist(scn, xy)
    print(f"{run.relative_to(ROOT)}: seed {res['seed']} {res['family']} {res['config']['name']} {res['sensor_mode']} "
          f"success={res['success']} failure={res['failure_type']} time={res['time']} ferr={res['final_error']} "
          f"ditch={res['ditch_entries']} coll={res['collisions']} water={res.get('water_entries')}")
    print("lighting events:", scn["lighting"]["events"], "dynamic:", [(d["type"], d["trigger_dist_m"]) for d in scn["dynamic"]])
    log = (run / "autonomy" / "autonomy.log").read_text(errors="replace")
    for m in re.finditer(r"t=([\d.]+) mode (\w+) -> (\w+) \((.*)\)", log):
        print("  transition", m.group(0))
    print("     t   v_gt  d_ditch d_crest d_water mode           v_cap  Rcert     q  reason")
    for i, d in enumerate(tel):
        dmin = min(hd["ditch"][i], hd["crest"][i], abs(hd["water"][i]))
        if dmin > a.near:
            continue
        f = lambda x: "   -  " if not np.isfinite(x) else f"{x:6.2f}"  # noqa: E731
        print(f"{d['t']:6.1f} {v[i]:6.2f} {f(hd['ditch'][i])} {f(hd['crest'][i])} {f(hd['water'][i])}  {d['mode']:13s} "
              f"{d.get('v_cap_mps', float('nan')):5.2f} {d.get('r_cert_m', float('nan')):6.2f} {d['health']['q']:5.2f}  {d.get('reason', '')}")


if __name__ == "__main__":
    main()
