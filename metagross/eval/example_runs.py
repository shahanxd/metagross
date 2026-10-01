"""Worked example from the EVAL closed loop: FULL vs TYPICAL on one held-out world.

The deck shows one seed as a bird's-eye pair (deck_assets/final/run_map_pair.png). This module
recomputes every number that figure and its caption quote, from the logged runs, and writes them
to ``results/example_runs_eval.json`` with a ``claims`` list that :mod:`metagross.eval.claims`
picks up. It also lists every trench entry of each config over all 60 EVAL runs, so a caption can
say how typical the example is.

Inputs (all read-only):
    results/runs_eval_tier0/summary.csv                         per-run outcome table
    results/runs_eval_tier0/<CFG>/<seed:03d>/result.json        referee outcome and time
    results/runs_eval_tier0/<CFG>/<seed:03d>/gt/states.npz      GT pose and speed per 0.04 s step
    results/runs_eval_tier0/FULL/<seed:03d>/autonomy/debug/tick_*.npz   the stack's per-tick log
    data/scenarios/eval/<seed>.json                             world (GT trench mask via the referee)

Label: Simulated (tier-0 synthetic depth sensor, no images). Run::

    python -m metagross.eval.example_runs && python -m metagross.eval.claims
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

LOG = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "results" / "runs_eval_tier0"
SCEN = REPO / "data" / "scenarios" / "eval"
OUT_FILE = REPO / "results" / "example_runs_eval.json"

HEADLINE_SEED = 38  # F3 crest with a hidden trench: FULL reached B, TYPICAL drove into the trench
MATCH_TOL_M = 1.0  # a confirmed map cell counts as "on the trench" within this GT distance
MIN_MATCHED_CELLS = 3  # first tick with at least this many matched cells = confirmation
AFTER_WINDOW_S = 3.0  # window after confirmation for the speed and turn-rate readout


@dataclass(frozen=True)
class Tick:
    """One logged autonomy tick (subset of ``autonomy/debug/tick_*.npz``)."""

    t: float  # s, scene time
    confirmed: np.ndarray  # bool (rows=y, cols=x), cumulative confirmed-ditch layer of the rolling map
    crop_origin: tuple[float, float]  # A-frame (m) of cell (0, 0)'s corner
    res_m: float  # m per cell
    gov_binding: str  # which speed-governor term was the lowest ('platform', 'r_cert', ...)
    cmd_w: float  # rad/s, commanded yaw rate


def ditch_entry_seeds(rows: Iterable[dict]) -> dict[str, list[int]]:
    """Seeds whose referee failure type is ``ditch_entry``, per config (sorted)."""
    out: dict[str, list[int]] = {}
    for r in rows:
        if r.get("failure_type") == "ditch_entry":
            out.setdefault(r["config_name"], []).append(int(r["seed"]))
    return {k: sorted(v) for k, v in sorted(out.items())}


def ditch_entries_by_family(rows: Iterable[dict]) -> dict[str, dict[str, int]]:
    """Runs whose referee failure type is ``ditch_entry``, counted per config and scenario family."""
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        if r.get("failure_type") == "ditch_entry":
            fam = out.setdefault(r["config_name"], {})
            fam[r["family"]] = fam.get(r["family"], 0) + 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def first_confirmation(ticks: Iterable[Tick], start_xy: tuple[float, float], start_yaw: float,
                       dist_to_trench: Callable[[np.ndarray, np.ndarray], np.ndarray],
                       tol_m: float = MATCH_TOL_M, min_cells: int = MIN_MATCHED_CELLS) -> Optional[dict]:
    """First tick whose confirmed-ditch layer holds >= ``min_cells`` cells within ``tol_m`` of the GT trench.

    Map cells are in the A-frame (origin = start pose, x along the launch heading) and are mapped
    to the world with ``start_xy`` / ``start_yaw``. ``dist_to_trench(xw, yw)`` returns the GT
    distance (m) from world points to the trench. Returns the tick time, the matched cells' world
    coordinates and the counts, or None if the trench is never confirmed.
    """
    c, s = math.cos(start_yaw), math.sin(start_yaw)
    for tk in ticks:
        if not tk.confirmed.any():
            continue
        ii, jj = np.nonzero(tk.confirmed)
        xa = tk.crop_origin[0] + (jj + 0.5) * tk.res_m
        ya = tk.crop_origin[1] + (ii + 0.5) * tk.res_m
        xw, yw = start_xy[0] + c * xa - s * ya, start_xy[1] + s * xa + c * ya
        on = dist_to_trench(xw, yw) <= tol_m
        if int(on.sum()) < min_cells:
            continue
        return {"t": tk.t, "xw": xw[on], "yw": yw[on], "n_cells": int(len(on)), "n_on_trench": int(on.sum())}
    return None


def after_window(ticks: Iterable[Tick], t0: float, span_s: float = AFTER_WINDOW_S) -> dict:
    """Governor binding counts and peak |commanded yaw rate| over ticks in [t0, t0 + span_s]."""
    sel = [tk for tk in ticks if t0 <= tk.t <= t0 + span_s]
    binding: dict[str, int] = {}
    for tk in sel:
        binding[tk.gov_binding] = binding.get(tk.gov_binding, 0) + 1
    return {"n_ticks": len(sel), "gov_binding_counts": dict(sorted(binding.items())),
            "max_abs_cmd_w_rad_s": max((abs(tk.cmd_w) for tk in sel), default=float("nan"))}


# ---- I/O -----------------------------------------------------------------------------------
def _summary(runs: Path) -> list[dict]:
    with (runs / "summary.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_ticks(run_dir: Path) -> list[Tick]:
    ticks = []
    for f in sorted(glob.glob(str(run_dir / "autonomy" / "debug" / "tick_*.npz"))):
        b = np.load(f)
        ex = json.loads(str(b["extras_json"]))
        ticks.append(Tick(t=float(b["t"]), confirmed=np.asarray(b["extra_map_crop_confirmed"], bool),
                          crop_origin=tuple(ex["map_crop_origin"]), res_m=float(ex["map_res_m"]),
                          gov_binding=str(ex.get("gov_binding")), cmd_w=float(ex.get("cmd_w", float("nan")))))
    return ticks


def _trench_distance(scn: dict) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """GT distance (m) to the referee's ditch mask, as a function of world x, y."""
    from scipy.ndimage import distance_transform_edt

    from metagross.sim.hazards import gt_hazard_raster
    from metagross.sim.terrain import Terrain

    terrain = Terrain.from_scenario(scn)
    g = terrain.grid
    dist = distance_transform_edt(~gt_hazard_raster(scn, terrain).ditch) * g.res

    def f(xw: np.ndarray, yw: np.ndarray) -> np.ndarray:
        gi = np.clip(np.round((np.asarray(yw) - g.origin_y) / g.res).astype(int), 0, g.ny - 1)
        gj = np.clip(np.round((np.asarray(xw) - g.origin_x) / g.res).astype(int), 0, g.nx - 1)
        return dist[gi, gj]
    return f


def headline_pair(seed: int = HEADLINE_SEED, runs: Path = RUNS, scen_dir: Path = SCEN) -> dict:
    """All numbers of the FULL-vs-TYPICAL example on one EVAL seed."""
    scn = json.loads((scen_dir / f"{seed}.json").read_text(encoding="utf-8"))
    res = {cfg: json.loads((runs / cfg / f"{seed:03d}" / "result.json").read_text(encoding="utf-8"))
           for cfg in ("FULL", "TYPICAL")}
    st = np.load(runs / "FULL" / f"{seed:03d}" / "gt" / "states.npz")
    ticks = _load_ticks(runs / "FULL" / f"{seed:03d}")
    det = first_confirmation(ticks, tuple(scn["start"]["xy"]), float(scn["start"]["yaw"]), _trench_distance(scn))
    out: dict = {"seed": seed, "family": scn["family"], "split": scn.get("split"),
                 "hazards": [{k: h[k] for k in ("type", "drop", "depth", "width") if k in h} for h in scn.get("hazards", [])],
                 "outcome": {cfg: ("success" if r["success"] else r["failure_type"]) for cfg, r in res.items()},
                 "end_time_s": {cfg: float(r["time"]) for cfg, r in res.items()}}
    if det is None:
        LOG.warning("seed %d: FULL never confirmed the trench", seed)
        return out
    t = det["t"]
    vx, vy = float(np.interp(t, st["t"], st["x"])), float(np.interp(t, st["t"], st["y"]))
    win = (st["t"] >= t) & (st["t"] <= t + AFTER_WINDOW_S)
    out["full_confirmation"] = {
        "t_s": t, "n_confirmed_cells": det["n_cells"], "n_on_gt_trench": det["n_on_trench"],
        "range_to_nearest_cell_m": float(np.hypot(det["xw"] - vx, det["yw"] - vy).min()),
        "gt_speed_at_mps": float(np.interp(t, st["t"], st["v"])),
        "gt_speed_min_next3s_mps": float(st["v"][win].min()),
        "gt_yaw_rate_max_abs_next3s_rad_s": float(np.abs(st["omega"][win]).max()),
        **after_window(ticks, t),
    }
    return out


def claims(doc: dict) -> list[dict]:
    """Ledger rows (Simulated) for the example pair and the trench-entry lists."""
    s = doc["pair"]["seed"]
    src = f"results/{OUT_FILE.name}"
    base = (f"EVAL closed loop run 2, sensor mode tier0, held-out seed {s} ({doc['pair']['family']}); "
            "one selected example, see docs/EVAL_PREREGISTRATION.md for the aggregate")
    rows = [
        {"id": f"example_eval_s{s:03d}_full_reached_b_s", "value": f"{doc['pair']['end_time_s']['FULL']:.1f}", "unit": "s",
         "source": f"{src}#pair.end_time_s.FULL", "note": f"{base}; FULL reached B at this time"},
        {"id": f"example_eval_s{s:03d}_typical_end_s", "value": f"{doc['pair']['end_time_s']['TYPICAL']:.1f}", "unit": "s",
         "source": f"{src}#pair.end_time_s.TYPICAL",
         "note": f"{base}; TYPICAL run ended ({doc['pair']['outcome']['TYPICAL']}) at this time"},
    ]
    for k, h in enumerate(doc["pair"].get("hazards", [])):
        for key, fmt, what in (("drop", "{:.1f}", "height lost past the crest"), ("depth", "{:.2f}", "trench depth")):
            if key in h:
                rows.append({"id": f"example_eval_s{s:03d}_{h['type']}_{key}_m", "value": fmt.format(h[key]), "unit": "m",
                             "source": f"{src}#pair.hazards[{k}].{key}",
                             "note": f"{base}; scenario input (data/scenarios/eval/{s}.json): {what}"})
    conf = doc["pair"].get("full_confirmation")
    if conf:
        rows += [
            {"id": f"example_eval_s{s:03d}_full_trench_confirmed_s", "value": f"{conf['t_s']:.1f}", "unit": "s",
             "source": f"{src}#pair.full_confirmation.t_s",
             "note": f"{base}; first tick whose rolling map held >= {MIN_MATCHED_CELLS} confirmed ditch cells on the GT trench"},
            {"id": f"example_eval_s{s:03d}_full_trench_confirmed_range_m", "value": f"{conf['range_to_nearest_cell_m']:.1f}",
             "unit": "m", "source": f"{src}#pair.full_confirmation.range_to_nearest_cell_m",
             "note": f"{base}; GT distance from the vehicle to the nearest confirmed trench cell at confirmation"},
            {"id": f"example_eval_s{s:03d}_full_trench_cells_on_gt", "value": f"{conf['n_on_gt_trench']}/{conf['n_confirmed_cells']}",
             "unit": "cells", "source": f"{src}#pair.full_confirmation.n_on_gt_trench",
             "note": f"{base}; confirmed ditch cells within {MATCH_TOL_M} m of the GT trench / all confirmed cells in the map crop"},
            {"id": f"example_eval_s{s:03d}_full_speed_min_after_confirm_mps", "value": f"{conf['gt_speed_min_next3s_mps']:.1f}",
             "unit": "m/s", "source": f"{src}#pair.full_confirmation.gt_speed_min_next3s_mps",
             "note": f"{base}; lowest GT speed in the {AFTER_WINDOW_S:.0f} s after confirmation; "
                     f"governor binding in that window: {conf['gov_binding_counts']} (ticks), so the slowdown is the turn, "
                     f"not the seen-ground speed cap"},
            {"id": f"example_eval_s{s:03d}_full_cmd_w_max_after_confirm", "value": f"{conf['max_abs_cmd_w_rad_s']:.2f}",
             "unit": "rad/s", "source": f"{src}#pair.full_confirmation.max_abs_cmd_w_rad_s",
             "note": f"{base}; peak commanded |yaw rate| in the {AFTER_WINDOW_S:.0f} s after confirmation (turn toward the gap)"},
        ]
    for cfg, seeds in doc["ditch_entry_seeds"].items():
        rows.append({"id": f"example_eval_ditch_entry_seeds_{cfg}", "value": ", ".join(str(x) for x in seeds),
                     "unit": "seeds", "source": f"{src}#ditch_entry_seeds.{cfg}",
                     "note": "EVAL closed loop run 2, tier0, all 60 seeds; seeds whose referee failure type is ditch_entry"})
    for cfg, fams in doc.get("ditch_entries_by_family", {}).items():
        for fam, n in fams.items():
            rows.append({"id": f"example_eval_ditch_entries_{cfg}_{fam}", "value": str(n), "unit": "runs",
                         "source": f"{src}#ditch_entries_by_family.{cfg}.{fam}",
                         "note": f"EVAL closed loop run 2, tier0; runs of 10 in family {fam} whose referee failure "
                                 "type is ditch_entry"})
    return [{**r, "label": "Simulated"} for r in rows]


def build(runs: Path = RUNS, scen_dir: Path = SCEN, seed: int = HEADLINE_SEED) -> dict:
    doc = {"what": __doc__.strip().splitlines()[0], "label": "Simulated (tier-0 synthetic depth sensor, no images)",
           "inputs": str(runs.relative_to(REPO)) if runs.is_relative_to(REPO) else str(runs),
           "pair": headline_pair(seed, runs, scen_dir),
           "ditch_entry_seeds": ditch_entry_seeds(_summary(runs)),
           "ditch_entries_by_family": ditch_entries_by_family(_summary(runs))}
    doc["claims"] = claims(doc)
    return doc


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=HEADLINE_SEED)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    args = ap.parse_args()
    doc = build(seed=args.seed)
    args.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    LOG.info("wrote %s: %d claims", args.out, len(doc["claims"]))


if __name__ == "__main__":
    main()
