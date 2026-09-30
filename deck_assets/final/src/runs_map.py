"""Bird's-eye run maps from real closed-loop EVAL runs, for the final deck.

Run from the repo root:
    OMP_NUM_THREADS=2 python deck_assets/final/src/runs_map.py

Outputs (deck_assets/final/):
    run_map_pair.png / .svg / .json         FULL vs TYPICAL on one held-out EVAL world (seed 38, F3 crest +
                                            hidden trench). Every number on it is registered in
                                            results/claims.csv (results/example_runs_eval.json + closed_loop_eval.json).
    run_map_pair_s007.*, run_map_pair_s037.*  backup pairs (F2 ditch field). Their per-seed numbers are NOT yet in
                                            results/claims.csv; the sidecar lists the rows to register.
    run_map_gallery.png / .svg / .json      four hand-picked FULL successes (F1, F2, F3, F5) with TYPICAL's path on
                                            the same world and the per-family success rates.

Every drawn element comes from files in the repo:
    world        data/scenarios/eval/<seed>.json  (heightmap, material raster, hazards, objects, lighting),
                 decoded with metagross.sim.terrain.Terrain.from_scenario and the referee's ground-truth hazard
                 raster metagross.sim.hazards.gt_hazard_raster
    paths        results/runs_eval_tier0/<CONFIG>/<seed:03d>/gt/states.npz  (GT x, y, v per 0.04 s step)
    outcomes     results/runs_eval_tier0/<CONFIG>/<seed:03d>/result.json + gt/events.json, cross-checked against
                 results/runs_eval_tier0/summary.csv
    rates        results/runs_eval_tier0/summary.csv, asserted equal to the registered rows of results/claims.csv
    detection    results/runs_eval_tier0/FULL/<seed:03d>/autonomy/debug/tick_*.npz (the stack's own per-tick log);
                 for seed 38 asserted equal to the registered results/example_runs_eval.json
Label: Simulated (tier-0 synthetic depth sensor, no images; held-out EVAL seeds).

Labels are placed by a small collision checker: each candidate position is scored against an occupancy raster
(objects, paths, trench, crest, markers, earlier labels) and the map frame; the cheapest position wins, and optional
labels (10 s ticks) are dropped when every candidate overlaps something.
"""
from __future__ import annotations

import csv
import glob
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import RendererAgg  # noqa: E402
from matplotlib.colors import LightSource, to_rgb  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.legend_handler import HandlerBase  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Polygon  # noqa: E402
from scipy.ndimage import distance_transform_edt, gaussian_filter  # noqa: E402

from metagross.sim.hazards import gt_hazard_raster  # noqa: E402
from metagross.sim.objects import parse_static_objects, rect_corners  # noqa: E402
from metagross.sim.terrain import Terrain  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "deck_assets" / "final"
RUNS = "results/runs_eval_tier0"
SUMMARY = f"{RUNS}/summary.csv"
SCEN = "data/scenarios/eval"
CLAIMS_CSV = "results/claims.csv"
EXAMPLE_JSON = "results/example_runs_eval.json"

# ---- palette (deck brief) ------------------------------------------------------------------
NAVY = "#1F497D"      # ours / FULL
GREY = "#9A9A9A"      # baseline / TYPICAL
GREY_TXT = "#6B6B6B"  # TYPICAL header text (the path grey is too light for text)
GREEN = "#2E7D32"     # success
RED = "#C62828"       # hazard / failure
AMBER = "#D98E04"     # crest (only meaning of amber in these figures)
BLUE = "#1E88E5"      # water
INK = "#1A1A1A"
MUTED = "#555555"     # secondary text, >= 7:1 on white
BG = "#FFFFFF"
OBJ_LETHAL = "#2F2F2F"
OBJ_SMALL = "#A3A3A3"
TRAIL_LEGEND = "#DCD1BA"  # the gravel-trail tint as it renders after hill-shading
# material tints for the relief background (kept very light so the data carries the colour)
MAT_RGB = {
    0: "#F3F3EF",  # grass
    1: "#F0ECE3",  # dirt
    2: "#E6DCC6",  # gravel trail
    3: "#ECECE9",  # rocky ground
    4: "#D9CCB4",  # mud
    5: "#CFE3F7",  # water (also outlined in BLUE)
}
CONTOUR = "#CFCFC9"
FRAME = "#BDBDBD"

# ---- type sizes (pt). Saved at 200 dpi, ~1830 px wide; shown ~900 px wide (x0.49) ------------
DPI = 200
FS_TITLE = 14.0    # panel title                        (39 px saved, ~19 px on the slide)
FS_SUB = 11.0      # panel subtitle / below-panel text  (31 px, ~15 px)
FS_MAPLAB = 10.5   # in-map labels                      (29 px, ~14 px)
FS_END = 11.0      # in-map end-of-run callouts, bold
FS_LETTER = 12.5   # A / B
FS_TICK = 10.0     # "10 s" tick labels                 (28 px, ~14 px)
FS_LEGEND = 10.5
FS_NOTE = 10.5     # selection disclosure line
FS_PROV = 10.0     # provenance line, MUTED (#555555, 7.5:1 on white)
LINE = 1.28        # line height in em

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
    "svg.fonttype": "path",
    "figure.facecolor": BG,
    "axes.facecolor": BG,
})
HALO = [pe.withStroke(linewidth=3.6, foreground="white")]
BG_RES_M = 0.10  # relief raster resolution in the figure (terrain is 0.05 m; halved to keep the SVG small)

# ---- layout constants (inches; figure saved at DPI=200 -> 9 in = 1800 px) --------------------
FIG_W_IN = 9.0
PANEL_GAP_IN = 0.16
PANEL_W_IN = (FIG_W_IN - PANEL_GAP_IN) / 2 - 0.02
PANEL_H_IN = PANEL_W_IN * 40.0 / 64.0     # world is 64 m x 40 m, drawn at equal scale
RELIEF_SMOOTH_M = 0.15
RELIEF_VERT_EXAG = 3.0
OBJ_MIN_R_LETHAL = 0.30                   # m, drawing floor so tree trunks stay visible
OBJ_MIN_R_SMALL = 0.18
M_PER_PT = 64.0 / (PANEL_W_IN * 72.0)     # map metres per typographic point inside a panel

# ---- analysis constants ----------------------------------------------------------------------
MATCH_TOL_M = 1.0          # confirmed map cell counts as "on the trench" within this GT distance (= example_runs.py)
MIN_MATCHED_CELLS = 3      # first tick with >= this many matched cells = confirmation (= example_runs.py)
AFTER_WINDOW_S = 3.0       # speed / turn-rate readout window after confirmation (= example_runs.py)
HOLD_SPEED_TOL = 0.10      # m/s: "holds v" only when the speed dipped by less than this
TURN_CMD_W = 0.8           # rad/s: "turns hard" only when the peak commanded |yaw rate| exceeds this
AT_CREST_M = 2.0           # m: "at the crest" only when the vehicle is this close to the crest polyline

CONFIG_TXT = {
    "FULL": ("FULL stack", "unseen ground is not free, ditch detector"),
    "TYPICAL": ("TYPICAL baseline", "unseen ground = free, no ditch detector"),
}
FAMILY_TXT = {
    "F1_trail": "F1 Trail",
    "F2_ditch_field": "F2 Ditch field",
    "F3_crest_ditch": "F3 Crest + hidden trench",
    "F4_sudden_obstacle": "F4 Sudden obstacle",
    "F5_lighting": "F5 Lighting",
    "F6_water_mud": "F6 Water / mud",
}
END_TXT = {
    "success": "Reached B",
    "ditch_entry": "Into the trench",
    "out_of_bounds": "Off the map",
    "stuck": "Stuck",
    "collision": "Hit an obstacle",
    "water_entry": "Into water",
    "arrived_short": "Stopped short of B",
    "timeout": "Timed out",
}
WHEEL_TXT = {"FL": "front-left", "FR": "front-right", "RL": "rear-left", "RR": "rear-right"}


# =============================================================================================
# data
# =============================================================================================
def _read_csv(rel: str) -> list[dict]:
    with open(ROOT / rel, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


SUM_ROWS = _read_csv(SUMMARY)
SUM = {(r["config_name"], int(r["seed"])): r for r in SUM_ROWS}
CLAIMS = {r["id"]: r for r in _read_csv(CLAIMS_CSV)}


def claim(cid: str) -> str:
    assert cid in CLAIMS, f"claim {cid} not registered in {CLAIMS_CSV}"
    return CLAIMS[cid]["value"]


def family_success(cfg: str, fam: str) -> dict:
    """Runs that reached B in one family, counted from summary.csv and asserted equal to the registered claim."""
    rows = [r for r in SUM_ROWS if r["config_name"] == cfg and r["family"] == fam]
    k = sum(r["success"] == "True" for r in rows)
    cid_k, cid_n = f"closed_loop_eval_tier0_{cfg}_{fam}_success", f"closed_loop_eval_tier0_{cfg}_{fam}_n"
    assert int(claim(cid_k)) == k and int(claim(cid_n)) == len(rows), (cfg, fam, k, len(rows))
    return {"k": k, "n": len(rows), "claim_ids": [cid_k, cid_n]}


def f2f3_success(cfg: str) -> dict:
    k = sum(r["success"] == "True" for r in SUM_ROWS
            if r["config_name"] == cfg and r["family"] in ("F2_ditch_field", "F3_crest_ditch"))
    cid = f"closed_loop_eval_tier0_{cfg}_F2F3_ditch_crest_success"
    assert int(claim(cid)) == k and int(claim(f"closed_loop_eval_tier0_{cfg}_F2F3_ditch_crest_n")) == 20
    return {"k": k, "n": 20, "claim_id": cid}


def ditch_entry_seeds(cfg: str) -> dict:
    seeds = sorted(int(r["seed"]) for r in SUM_ROWS if r["config_name"] == cfg and r["failure_type"] == "ditch_entry")
    cid = f"example_eval_ditch_entry_seeds_{cfg}"
    assert claim(cid) == ", ".join(str(s) for s in seeds), (cfg, seeds, claim(cid))
    fams = [SUM[(cfg, s)]["family"] for s in seeds]
    return {"seeds": seeds, "families": fams, "claim_id": cid}


def f2f3_full_wins() -> list[dict]:
    """F2/F3 EVAL seeds where FULL reached B and TYPICAL did not (summary.csv)."""
    out = []
    for (cfg, seed), r in sorted(SUM.items(), key=lambda kv: kv[0][1]):
        if cfg != "FULL" or r["family"] not in ("F2_ditch_field", "F3_crest_ditch") or r["success"] != "True":
            continue
        t = SUM[("TYPICAL", seed)]
        if t["success"] != "True":
            out.append({"seed": seed, "family": r["family"], "typical_failure": t["failure_type"],
                        "typical_time_s": float(t["time"])})
    return out


def load_world(seed: int) -> dict:
    scn = json.loads((ROOT / SCEN / f"{seed}.json").read_text())
    terrain = Terrain.from_scenario(scn)
    hz = gt_hazard_raster(scn, terrain)
    _, feet = parse_static_objects(scn, terrain)
    g = terrain.grid
    dist = distance_transform_edt(~hz.ditch) * g.res if hz.ditch.any() else None
    return {"seed": seed, "scn": scn, "terrain": terrain, "hz": hz, "feet": feet, "trench_dist": dist}


def trench_dist_at(world: dict, xw, yw) -> np.ndarray:
    """GT distance (m) from world points to the referee's trench mask (inf when the world has no trench)."""
    g, d = world["terrain"].grid, world["trench_dist"]
    xw, yw = np.asarray(xw, float), np.asarray(yw, float)
    if d is None:
        return np.full(xw.shape, np.inf)
    gi = np.clip(np.round((yw - g.origin_y) / g.res).astype(int), 0, g.ny - 1)
    gj = np.clip(np.round((xw - g.origin_x) / g.res).astype(int), 0, g.nx - 1)
    return d[gi, gj]


def load_run(cfg: str, seed: int) -> dict:
    d = ROOT / RUNS / cfg / f"{seed:03d}"
    st = np.load(d / "gt" / "states.npz")
    res = json.loads((d / "result.json").read_text())
    events = json.loads((d / "gt" / "events.json").read_text())
    row = SUM[(cfg, seed)]
    outcome = "success" if res["success"] else res["failure_type"]
    # cross-check result.json against the summary table the deck's numbers come from
    assert (row["success"] == "True") == bool(res["success"]), (cfg, seed)
    assert (row["failure_type"] or None) == res["failure_type"], (cfg, seed)
    assert abs(float(row["time"]) - res["time"]) < 1e-6, (cfg, seed)
    term = [e for e in events if e["type"] == outcome]
    assert term, (cfg, seed, outcome, events)
    return {
        "cfg": cfg, "seed": seed, "dir": str(d.relative_to(ROOT)),
        "t": st["t"], "x": st["x"], "y": st["y"], "v": st["v"], "omega": st["omega"],
        "outcome": outcome, "time": float(res["time"]), "event": term[0],
        "path_length": float(res["path_length"]), "mean_speed": float(res["mean_speed"]),
        "final_error": float(res["final_error"]), "spl": res.get("spl"),
        "ditch_entries": int(res["ditch_entries"]), "collisions": int(res["collisions"]),
        "water_entries": int(res["water_entries"]),
    }


def at_time(run: dict, t: float) -> tuple[float, float]:
    return float(np.interp(t, run["t"], run["x"])), float(np.interp(t, run["t"], run["y"]))


def polyline_dist(q: np.ndarray, px: float, py: float) -> float:
    """Distance (m) from a point to a polyline."""
    a, b = q[:-1], q[1:]
    ab = b - a
    tt = np.clip(((px - a[:, 0]) * ab[:, 0] + (py - a[:, 1]) * ab[:, 1]) / np.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
    cx, cy = a[:, 0] + tt * ab[:, 0], a[:, 1] + tt * ab[:, 1]
    return float(np.hypot(cx - px, cy - py).min())


def trench_confirmation(run: dict, world: dict) -> dict | None:
    """First tick at which FULL's own rolling map held >= MIN_MATCHED_CELLS confirmed-ditch cells on the GT trench.

    Same definition as metagross/eval/example_runs.py (which registers seed 38). Source: autonomy/debug/tick_*.npz.
    ``extra_map_crop_confirmed`` is the cumulative confirmed-ditch layer of the rolling map cropped around the
    vehicle, in the A-frame (origin = start pose, x along the launch heading); cells are mapped to the world with
    the scenario start pose and matched to the referee's GT trench mask within MATCH_TOL_M.
    """
    x0, y0 = world["scn"]["start"]["xy"]
    yaw0 = float(world["scn"]["start"]["yaw"])
    c, s_ = math.cos(yaw0), math.sin(yaw0)
    files = sorted(glob.glob(str(ROOT / run["dir"] / "autonomy" / "debug" / "tick_*.npz")))
    first_any = None
    ticks = []
    for f in files:
        b = np.load(f)
        ex = json.loads(str(b["extras_json"]))
        ticks.append((float(b["t"]), ex))
    for f, (t_tick, ex) in zip(files, ticks):
        conf = np.load(f)["extra_map_crop_confirmed"]
        if not conf.any():
            continue
        ox, oy = ex["map_crop_origin"]
        r = float(ex["map_res_m"])
        ii, jj = np.nonzero(conf)
        xa, ya = ox + (jj + 0.5) * r, oy + (ii + 0.5) * r
        xw, yw = x0 + c * xa - s_ * ya, y0 + s_ * xa + c * ya
        dcell = trench_dist_at(world, xw, yw)
        on = dcell <= MATCH_TOL_M
        if first_any is None:
            first_any = {"t_s": t_tick, "n_cells": int(len(on)), "n_on_gt_trench": int(on.sum())}
        if on.sum() < MIN_MATCHED_CELLS:
            continue
        vx, vy = at_time(run, t_tick)
        win = (run["t"] >= t_tick) & (run["t"] <= t_tick + AFTER_WINDOW_S)
        tw = [(tt, e) for tt, e in ticks if t_tick <= tt <= t_tick + AFTER_WINDOW_S]
        binding: dict[str, int] = {}
        for _, e in tw:
            binding[str(e.get("gov_binding"))] = binding.get(str(e.get("gov_binding")), 0) + 1
        crest = [np.asarray(h["polyline"], float) for h in world["scn"]["hazards"] if h["type"] == "crest"]
        return {
            "t_s": t_tick, "file": str(Path(f).relative_to(ROOT)), "vehicle_xy_m": [vx, vy],
            "n_confirmed_cells": int(len(on)), "n_on_gt_trench": int(on.sum()),
            "n_not_on_trench": int((~on).sum()),
            "median_dist_off_trench_cells_to_trench_m": (float(np.median(dcell[~on])) if (~on).any() else None),
            "match_tolerance_m": MATCH_TOL_M, "min_cells": MIN_MATCHED_CELLS, "map_res_m": r,
            "range_to_nearest_cell_m": float(np.hypot(xw[on] - vx, yw[on] - vy).min()),
            "vehicle_dist_to_gt_trench_m": float(trench_dist_at(world, vx, vy)),
            "vehicle_dist_to_crest_m": (min(polyline_dist(q, vx, vy) for q in crest) if crest else None),
            "gt_speed_at_mps": float(np.interp(t_tick, run["t"], run["v"])),
            "gt_speed_min_next3s_mps": float(run["v"][win].min()),
            "gt_yaw_rate_max_abs_next3s_rad_s": float(np.abs(run["omega"][win]).max()),
            "n_ticks_next3s": len(tw), "gov_binding_counts_next3s": dict(sorted(binding.items())),
            "max_abs_cmd_w_next3s_rad_s": max(abs(float(e.get("cmd_w", 0.0))) for _, e in tw),
            "cmd_v_range_next3s_mps": [min(float(e["cmd_v"]) for _, e in tw), max(float(e["cmd_v"]) for _, e in tw)],
            "first_tick_with_any_confirmed_cells": first_any,
        }
    return None


def trench_crossings(run: dict, world: dict) -> list[dict]:
    """Where the GT path crosses each trench polyline, and whether the crossing is inside a scenario gap."""
    x, y, t = run["x"], run["y"], run["t"]
    p0 = np.stack([x[:-1], y[:-1]], 1)
    rr = np.stack([np.diff(x), np.diff(y)], 1)
    out = []
    k = 0
    for h in world["scn"].get("hazards", []):
        if h["type"] != "ditch":
            continue
        q = np.asarray(h["polyline"], float)
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
        for j in range(len(q) - 1):
            a, s = q[j], q[j + 1] - q[j]
            den = rr[:, 0] * s[1] - rr[:, 1] * s[0]
            ok = np.abs(den) > 1e-12
            ap = a - p0
            tt = np.where(ok, (ap[:, 0] * s[1] - ap[:, 1] * s[0]) / np.where(ok, den, 1), -1)
            uu = np.where(ok, (ap[:, 0] * rr[:, 1] - ap[:, 1] * rr[:, 0]) / np.where(ok, den, 1), -1)
            for i in np.nonzero(ok & (tt >= 0) & (tt <= 1) & (uu >= 0) & (uu <= 1))[0]:
                sa = float(arc[j] + uu[i] * np.hypot(*s))
                out.append({"trench": k, "t_s": round(float(t[i] + tt[i] * (t[i + 1] - t[i])), 2),
                            "arc_m": round(sa, 2), "gaps_arc_m": h.get("gaps", []),
                            "in_gap": any(g0 <= sa <= g1 for g0, g1 in h.get("gaps", []))})
        k += 1
    return sorted(out, key=lambda d: d["t_s"])


def n_trenches(world: dict) -> int:
    return sum(h["type"] == "ditch" for h in world["scn"].get("hazards", []))


# =============================================================================================
# text measurement and wrapping
# =============================================================================================
_R = RendererAgg(100, 100, DPI)


def text_w_in(s: str, fs: float, weight: str = "normal") -> float:
    w, _, _ = _R.get_text_width_height_descent(s, FontProperties(family="DejaVu Sans", size=fs, weight=weight),
                                               ismath=False)
    return w / DPI


def wrap(text: str, fs: float, width_in: float, weight: str = "normal") -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split(" "):
            trial = word if not cur else cur + " " + word
            if cur and text_w_in(trial, fs, weight) > width_in:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
    return lines


def block_h_in(n_lines: int, fs: float) -> float:
    return n_lines * fs * LINE / 72.0


# =============================================================================================
# occupancy raster for label placement
# =============================================================================================
OCC_RES = 0.2  # m


class Occ:
    """Boolean layers over one map panel (world metres) that labels must not cover."""

    LAYERS = ("obj", "path", "haz", "mark", "label")

    def __init__(self, ext: tuple[float, float, float, float]):
        self.ext = ext
        self.nx = int(math.ceil((ext[1] - ext[0]) / OCC_RES))
        self.ny = int(math.ceil((ext[3] - ext[2]) / OCC_RES))
        self.xc = ext[0] + (np.arange(self.nx) + 0.5) * OCC_RES
        self.yc = ext[2] + (np.arange(self.ny) + 0.5) * OCC_RES
        self.X, self.Y = np.meshgrid(self.xc, self.yc)
        self.layers = {k: np.zeros((self.ny, self.nx), bool) for k in self.LAYERS}

    def copy(self) -> "Occ":
        o = Occ.__new__(Occ)
        o.ext, o.nx, o.ny, o.xc, o.yc, o.X, o.Y = self.ext, self.nx, self.ny, self.xc, self.yc, self.X, self.Y
        o.layers = {k: v.copy() for k, v in self.layers.items()}
        return o

    def _points_mask(self, xs, ys) -> np.ndarray:
        m = np.zeros((self.ny, self.nx), bool)
        j = np.floor((np.asarray(xs) - self.ext[0]) / OCC_RES).astype(int)
        i = np.floor((np.asarray(ys) - self.ext[2]) / OCC_RES).astype(int)
        ok = (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx)
        m[i[ok], j[ok]] = True
        return m

    def add_points(self, layer: str, xs, ys, r: float) -> None:
        m = self._points_mask(np.atleast_1d(xs), np.atleast_1d(ys))
        if not m.any():
            return
        self.layers[layer] |= (distance_transform_edt(~m) * OCC_RES <= r) if r > 0 else m

    def add_polyline(self, layer: str, xs, ys, r: float) -> None:
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        seg = np.hypot(np.diff(xs), np.diff(ys))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        ss = np.arange(0.0, s[-1] + 1e-9, OCC_RES / 2) if s[-1] > 0 else np.array([0.0])
        self.add_points(layer, np.interp(ss, s, xs), np.interp(ss, s, ys), r)

    def add_disc(self, layer: str, x: float, y: float, r: float) -> None:
        self.layers[layer] |= (self.X - x) ** 2 + (self.Y - y) ** 2 <= r * r

    def add_orect(self, layer: str, cx: float, cy: float, hl: float, hw: float, yaw: float) -> None:
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = self.X - cx, self.Y - cy
        self.layers[layer] |= (np.abs(c * dx + s * dy) <= hl) & (np.abs(-s * dx + c * dy) <= hw)

    def add_rect(self, layer: str, rect) -> None:
        x0, y0, x1, y1 = rect
        self.layers[layer] |= (self.X >= x0) & (self.X <= x1) & (self.Y >= y0) & (self.Y <= y1)

    def add_raster(self, layer: str, mask: np.ndarray, factor: int, r: float) -> None:
        """World raster (0.05 m) -> occupancy grid by block-any, then dilate by r."""
        ny, nx = self.ny * factor, self.nx * factor
        mm = np.zeros((ny, nx), bool)
        mm[:min(ny, mask.shape[0]), :min(nx, mask.shape[1])] = mask[:ny, :nx]
        m = mm.reshape(self.ny, factor, self.nx, factor).any(axis=(1, 3))
        if m.any():
            self.layers[layer] |= distance_transform_edt(~m) * OCC_RES <= r

    def count_rect(self, rect, layers) -> int:
        x0, y0, x1, y1 = rect
        j0, j1 = int(max(0, math.floor((x0 - self.ext[0]) / OCC_RES))), int(min(self.nx, math.ceil((x1 - self.ext[0]) / OCC_RES)))
        i0, i1 = int(max(0, math.floor((y0 - self.ext[2]) / OCC_RES))), int(min(self.ny, math.ceil((y1 - self.ext[2]) / OCC_RES)))
        if j1 <= j0 or i1 <= i0:
            return 0
        return int(sum(self.layers[k][i0:i1, j0:j1].sum() for k in layers))

    def count_points(self, xs, ys, layers) -> int:
        j = np.floor((np.asarray(xs) - self.ext[0]) / OCC_RES).astype(int)
        i = np.floor((np.asarray(ys) - self.ext[2]) / OCC_RES).astype(int)
        ok = (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx)
        return int(sum(self.layers[k][i[ok], j[ok]].sum() for k in layers))

    def inside(self, rect, margin: float = 0.3) -> bool:
        x0, y0, x1, y1 = rect
        return (x0 >= self.ext[0] + margin and x1 <= self.ext[1] - margin
                and y0 >= self.ext[2] + margin and y1 <= self.ext[3] - margin)


def data_rect(ax, artist, pad: float = 0.0) -> tuple[float, float, float, float]:
    bb = artist.get_window_extent(ax.figure.canvas.get_renderer())
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return (min(x0, x1) - pad, min(y0, y1) - pad, max(x0, x1) + pad, max(y0, y1) + pad)


def ring(dists, angles_deg=range(0, 360, 30)) -> list[tuple]:
    """Candidate offsets around an anchor, with the text aligned away from the anchor."""
    out = []
    for d in dists:
        for a in angles_deg:
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            ha = "left" if ca > 0.3 else ("right" if ca < -0.3 else "center")
            va = "bottom" if sa > 0.3 else ("top" if sa < -0.3 else "center")
            out.append((d * ca, d * sa, ha, va))
    return out


TEXT_LAYERS = ("obj", "path", "haz", "mark", "label")
PLACEMENT_LOG: list[dict] = []


def place_text(ax, occ: Occ, texts, anchor, cands, *, fs: float, colour: str = INK, weight: str = "normal",
               leader: bool = True, required: bool = True, name: str = "", pad: float = 0.35,
               shrink: float = 0.8, text_layers=TEXT_LAYERS, ha_multi: str | None = None) -> dict | None:
    """Place one label at the cheapest collision-free candidate; draw it (and a leader) and mark it occupied.

    ``texts`` is a string or a list of variants (later variants are slightly penalised). ``cands`` are
    (dx, dy, ha, va) offsets from ``anchor``. Cost = 60 x covered cells + 4 x leader cells over objects/labels
    + 1.5 x leader cells over paths/hazards + 0.25 x distance + small order penalties. Candidates that leave the
    map frame are skipped. Optional labels are dropped when every candidate covers something.
    """
    if isinstance(texts, str):
        texts = [texts]
    best = None
    for vi, txt in enumerate(texts):
        for ci, (dx, dy, ha, va) in enumerate(cands):
            x, y = anchor[0] + dx, anchor[1] + dy
            t = ax.text(x, y, txt, fontsize=fs, color=colour, fontweight=weight, ha=ha, va=va, zorder=12,
                        path_effects=HALO, linespacing=1.15, multialignment=ha_multi or ha)
            rect_raw = data_rect(ax, t)
            t.remove()
            rect = (rect_raw[0] - pad, rect_raw[1] - pad, rect_raw[2] + pad, rect_raw[3] + pad)
            if not occ.inside(rect_raw, margin=0.4):
                continue
            c_text = occ.count_rect(rect, text_layers)
            seg, c_lead = None, 0.0
            if leader:
                seg = leader_segment(rect_raw, anchor, shrink)
                if seg is not None:
                    xs, ys = sample_segment(seg)
                    far = np.hypot(xs - anchor[0], ys - anchor[1]) > 1.3
                    c_lead = (4.0 * occ.count_points(xs[far], ys[far], ("obj", "label", "mark"))
                              + 1.5 * occ.count_points(xs[far], ys[far], ("path", "haz")))
            cost = 60.0 * c_text + c_lead + 0.25 * math.hypot(dx, dy) + 0.15 * ci + 3.0 * vi
            if best is None or cost < best["cost"]:
                best = {"cost": cost, "c_text": c_text, "c_lead": c_lead, "txt": txt, "x": x, "y": y, "ha": ha,
                        "va": va, "rect": rect, "rect_raw": rect_raw, "seg": seg}
    if best is None or (not required and best["c_text"] > 0):
        PLACEMENT_LOG.append({"name": name, "placed": False, "reason": "no clear candidate"})
        return None
    if best["c_text"] > 0 or best["c_lead"] > 0:
        print(f"  [place] {name!r}: covered cells {best['c_text']}, leader cost {best['c_lead']:.1f}")
    ax.text(best["x"], best["y"], best["txt"], fontsize=fs, color=colour, fontweight=weight, ha=best["ha"],
            va=best["va"], zorder=12, path_effects=HALO, linespacing=1.15, multialignment=ha_multi or best["ha"])
    occ.add_rect("label", best["rect"])
    if best["seg"] is not None:
        (x0, y0), (x1, y1) = best["seg"]
        ax.plot([x0, x1], [y0, y1], color=MUTED, lw=1.0, zorder=11, solid_capstyle="butt")
        occ.add_polyline("label", [x0, x1], [y0, y1], 0.15)
    PLACEMENT_LOG.append({"name": name, "placed": True, "covered_cells": best["c_text"],
                          "leader_cost": round(best["c_lead"], 1)})
    return best


def leader_segment(rect, anchor, shrink: float):
    """Leader from the text box edge (0.25 m gap) to ``shrink`` m short of the anchor; None when too short."""
    x0, y0, x1, y1 = rect
    ax_, ay_ = anchor
    px, py = min(max(ax_, x0), x1), min(max(ay_, y0), y1)
    if x0 < ax_ < x1 and y0 < ay_ < y1:
        return (px, py), (ax_, ay_)  # anchor inside the text: heavily penalised through the cells
    L = math.hypot(ax_ - px, ay_ - py)
    if L < shrink + 1.0:
        return None
    ux, uy = (ax_ - px) / L, (ay_ - py) / L
    return (px + 0.25 * ux, py + 0.25 * uy), (ax_ - shrink * ux, ay_ - shrink * uy)


def sample_segment(seg, step: float = 0.1):
    (x0, y0), (x1, y1) = seg
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step))
    return np.linspace(x0, x1, n), np.linspace(y0, y1, n)


# =============================================================================================
# drawing
# =============================================================================================
def extent(world: dict) -> tuple[float, float, float, float]:
    g = world["terrain"].grid
    return (g.origin_x - g.res / 2, g.origin_x + (g.nx - 0.5) * g.res,
            g.origin_y - g.res / 2, g.origin_y + (g.ny - 0.5) * g.res)


def relief_rgb(world: dict) -> np.ndarray:
    """Light shaded relief tinted by material. Rows along +y (drawn with origin='lower')."""
    t = world["terrain"]
    h = gaussian_filter(t.height.astype(np.float64), RELIEF_SMOOTH_M / t.grid.res)
    ls = LightSource(azdeg=315, altdeg=40)
    # LightSource assumes row 0 at the top (north); our row 0 is south, so flip in and out.
    shade = np.flipud(ls.hillshade(np.flipud(h), vert_exag=RELIEF_VERT_EXAG, dx=t.grid.res, dy=t.grid.res))
    lut = np.array([to_rgb(MAT_RGB.get(k, MAT_RGB[0])) for k in range(256)])
    rgb = np.clip(lut[t.material] * (0.80 + 0.25 * shade[..., None]), 0, 1)
    k = int(round(BG_RES_M / t.grid.res))
    ny, nx = (rgb.shape[0] // k) * k, (rgb.shape[1] // k) * k
    return rgb[:ny, :nx].reshape(ny // k, k, nx // k, k, 3).mean(axis=(1, 3))


def draw_world(ax, world: dict) -> Occ:
    """Basemap (relief, contours, water, trench, crest, objects). Returns the occupancy of the static world."""
    t, hz, scn = world["terrain"], world["hz"], world["scn"]
    g = t.grid
    ext = extent(world)
    ax.imshow(relief_rgb(world), origin="lower", extent=ext, interpolation="bilinear", zorder=0)
    xs = g.origin_x + np.arange(g.nx) * g.res
    ys = g.origin_y + np.arange(g.ny) * g.res
    # 0.25 m contours of the landform (trench excavation added back so the trench is not contoured twice)
    land = gaussian_filter((t.height + hz.ditch_drop)[::2, ::2], 2.0)
    lv = np.arange(math.floor(land.min() * 4) / 4, land.max() + 0.25, 0.25)
    ax.contour(xs[::2], ys[::2], land, levels=lv, colors=CONTOUR, linewidths=0.45, zorder=1,
               negative_linestyles="solid")
    occ = Occ(ext)
    factor = int(round(OCC_RES / g.res))
    if hz.water.any():  # water / mud: flat, semantically hazardous
        ax.contourf(xs, ys, hz.water.astype(np.float32), levels=[0.5, 1.5], colors=[BLUE], alpha=0.55, zorder=2)
        ax.contour(xs, ys, hz.water.astype(np.float32), levels=[0.5], colors=[BLUE], linewidths=0.8, zorder=2)
        occ.add_raster("haz", hz.water, factor, 0.2)
    if hz.ditch.any():  # trenches: the referee's GT ditch mask (cells excavated > 5 cm)
        ax.contourf(xs, ys, hz.ditch.astype(np.float32), levels=[0.5, 1.5], colors=[RED], zorder=3)
        occ.add_raster("haz", hz.ditch, factor, 0.35)
    for h in scn.get("hazards", []):
        if h["type"] == "crest":
            q = np.asarray(h["polyline"])
            ax.plot(q[:, 0], q[:, 1], color=AMBER, lw=1.8, ls=(0, (4, 2.5)), zorder=3)
            occ.add_polyline("haz", q[:, 0], q[:, 1], 0.4)
    # static objects: dark if taller than ground clearance (lethal), light grey if drive-over
    for fp in world["feet"]:
        col = OBJ_LETHAL if fp.lethal else OBJ_SMALL
        if fp.kind == "circle":
            r = max(fp.r, OBJ_MIN_R_LETHAL if fp.lethal else OBJ_MIN_R_SMALL)
            ax.add_patch(Circle((fp.cx, fp.cy), r, facecolor=col, edgecolor="none", zorder=4))
            occ.add_disc("obj", fp.cx, fp.cy, r + 0.3)
        else:
            ax.add_patch(Polygon(rect_corners(fp), closed=True, facecolor=col, edgecolor="none", zorder=4))
            occ.add_orect("obj", fp.cx, fp.cy, fp.hl + 0.3, fp.hw + 0.3, fp.yaw)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(FRAME)
        sp.set_linewidth(0.8)
    return occ


def draw_path(ax, occ: Occ, run: dict, colour: str, lw: float, zorder: float = 7.0) -> None:
    """GT path with a white casing (drawn over the frame edge so exits stay visible)."""
    x, y = run["x"], run["y"]
    ax.plot(x, y, color="white", lw=lw + 2.4, solid_capstyle="round", solid_joinstyle="round", zorder=zorder - 1)
    ax.plot(x, y, color=colour, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=zorder)
    occ.add_polyline("path", x, y, (lw + 2.4) / 2 * M_PER_PT + 0.3)


def draw_tick_dots(ax, occ: Occ, run: dict, colour: str, every: float = 10.0) -> list[dict]:
    ticks = []
    tt = every
    while tt < run["time"] - 2.0:
        px, py = at_time(run, tt)
        ax.plot(px, py, "o", ms=5.2, mfc="white", mec=colour, mew=1.5, zorder=8)
        occ.add_disc("mark", px, py, 0.6)
        ticks.append({"t_s": tt, "x_m": round(px, 2), "y_m": round(py, 2)})
        tt += every
    return ticks


def label_ticks(ax, occ: Occ, run: dict, ticks: list[dict]) -> None:
    """'10 s' labels beside the tick dots; each label is dropped (dot kept) when it would cover anything."""
    x, y = run["x"], run["y"]
    for tk in ticks:
        i = int(np.searchsorted(run["t"], tk["t_s"]))
        j0, j1 = max(i - 12, 0), min(i + 12, len(x) - 1)
        hx, hy = x[j1] - x[j0], y[j1] - y[j0]
        n = math.hypot(hx, hy) or 1.0
        base = math.degrees(math.atan2(hx / n, -hy / n))  # left normal of travel
        angles = [base, base + 180, base + 40, base - 40, base + 140, base + 220]
        cands = ring([2.0, 2.6], angles)
        res = place_text(ax, occ, f"{tk['t_s']:.0f} s", (tk["x_m"], tk["y_m"]), cands, fs=FS_TICK, colour=MUTED,
                         leader=False, required=False, name=f"{run['cfg']} tick {tk['t_s']:.0f}", pad=0.25)
        tk["labelled"] = res is not None


def draw_start_goal(ax, occ: Occ, world: dict) -> None:
    scn = world["scn"]
    sx, sy = scn["start"]["xy"]
    bx, by = scn["goal"]["xy"]
    r = float(scn["mission"]["success_radius_m"])
    ax.add_patch(Circle((bx, by), r, facecolor="none", edgecolor=INK, lw=1.1, ls=(0, (3, 2)), zorder=5))
    ax.plot(sx, sy, "o", ms=7.5, mfc="white", mec=INK, mew=1.6, zorder=10)
    ax.plot(bx, by, "o", ms=3.4, mfc=INK, mec=INK, zorder=10)
    occ.add_disc("mark", sx, sy, 0.9)
    occ.add_disc("mark", bx, by, r + 0.35)


def label_start_goal(ax, occ: Occ, world: dict) -> None:
    scn = world["scn"]
    r = float(scn["mission"]["success_radius_m"])
    for (px, py), lab, d in ((scn["start"]["xy"], "A", 1.6), (scn["goal"]["xy"], "B", r + 1.0)):
        cands = [(-d, 0, "right", "center"), (d, 0, "left", "center"), (0, d, "center", "bottom"),
                 (0, -d, "center", "top"), (-d * .75, d * .75, "right", "bottom"), (d * .75, d * .75, "left", "bottom"),
                 (-d * .75, -d * .75, "right", "top"), (d * .75, -d * .75, "left", "top")]
        place_text(ax, occ, lab, (px, py), cands, fs=FS_LETTER, weight="bold", leader=False, name=lab, pad=0.2)


def end_marker(ax, occ: Occ, run: dict, small: bool = False) -> tuple[float, float]:
    ex, ey = float(run["x"][-1]), float(run["y"][-1])
    if run["outcome"] == "success":
        ax.plot(ex, ey, "o", ms=(6.5 if small else 8.5), mfc=GREEN, mec="white", mew=1.4, zorder=10, clip_on=False)
    else:
        ax.plot(ex, ey, marker="X", ms=(10 if small else 12), mfc=RED, mec="white", mew=1.3, zorder=10, clip_on=False)
    occ.add_disc("mark", ex, ey, 1.1)
    return ex, ey


def draw_scale_bar(ax, x0: float, y0: float, length: float = 10.0) -> list:
    """10 m bar (half black, half white) with a north arrow to its right. Returns the artists."""
    h = 0.6
    arts = []
    for k in range(2):
        arts.append(ax.add_patch(plt.Rectangle((x0 + k * length / 2, y0), length / 2, h,
                                               facecolor=(INK if k == 0 else "white"), edgecolor=INK, lw=0.8,
                                               zorder=11)))
    for xx, lab in ((x0, "0"), (x0 + length, f"{length:.0f} m")):
        arts.append(ax.text(xx, y0 + h + 0.7, lab, fontsize=FS_TICK, color=INK, ha="center", va="bottom", zorder=11,
                            path_effects=HALO))
    nx_ = x0 + length + 4.6
    arts.append(ax.add_patch(FancyArrowPatch((nx_, y0 - 0.1), (nx_, y0 + 3.6),
                                             arrowstyle="-|>,head_length=4,head_width=2.4", color=INK, lw=1.2,
                                             zorder=11)))
    arts.append(ax.text(nx_, y0 + 4.2, "N", fontsize=FS_TICK, color=INK, ha="center", va="bottom", fontweight="bold",
                        zorder=11, path_effects=HALO))
    return arts


def place_scale_bar(ax, occ: Occ) -> dict:
    """Scale bar + north arrow in the emptiest corner region of the panel."""
    e = occ.ext
    xs = [e[0] + 2.0, e[0] + 8.0, e[1] - 20.5, e[1] - 26.5]
    ys = [e[2] + 1.6, e[2] + 5.0, e[3] - 7.0, e[3] - 10.0]
    best = None
    for yi, y0 in enumerate(ys):
        for xi, x0 in enumerate(xs):
            arts = draw_scale_bar(ax, x0, y0)
            rects = [data_rect(ax, a) for a in arts]
            for a in arts:
                a.remove()
            rect = (min(r[0] for r in rects) - 0.5, min(r[1] for r in rects) - 0.5,
                    max(r[2] for r in rects) + 0.5, max(r[3] for r in rects) + 0.5)
            if not occ.inside(rect, margin=0.2):
                continue
            c = occ.count_rect(rect, TEXT_LAYERS) + 0.5 * (xi % 2 + yi % 2)
            if best is None or c < best[0]:
                best = (c, x0, y0, rect)
    c, x0, y0, rect = best
    if c >= 1:
        print(f"  [place] scale bar: covered cells {c}")
    draw_scale_bar(ax, x0, y0)
    occ.add_rect("label", rect)
    return {"x0_m": round(x0, 2), "y0_m": round(y0, 2), "covered_cells": c}


def gap_midpoints(world: dict) -> list[tuple[float, float]]:
    out = []
    for h in world["scn"].get("hazards", []):
        if h["type"] != "ditch":
            continue
        q = np.asarray(h["polyline"], float)
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
        for s0, s1 in h.get("gaps", []):
            sm = 0.5 * (s0 + s1)
            out.append((float(np.interp(sm, arc, q[:, 0])), float(np.interp(sm, arc, q[:, 1]))))
    return out


def point_on_polyline(q: np.ndarray, frac: float) -> tuple[float, float]:
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
    s = frac * arc[-1]
    return float(np.interp(s, arc, q[:, 0])), float(np.interp(s, arc, q[:, 1]))


def hazard_summary(world: dict) -> list[dict]:
    out = []
    for h in world["scn"].get("hazards", []):
        d = {"type": h["type"]}
        for k in ("depth", "width", "drop"):
            if k in h:
                d[k + "_m"] = h[k]
        if "gaps" in h:
            d["gaps_arc_m"] = h["gaps"]
        out.append(d)
    return out


class BracketHandler(HandlerBase):
    """Legend handle: a short line with end ticks (lighting-event window)."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        x0, x1 = -xdescent + 1, width - xdescent - 1
        yc = height / 2 - ydescent
        hh = height * 0.45
        arts = [Line2D([x0, x1], [yc, yc], color=INK, lw=1.8, transform=trans),
                Line2D([x0, x0], [yc - hh, yc + hh], color=INK, lw=1.8, transform=trans),
                Line2D([x1, x1], [yc - hh, yc + hh], color=INK, lw=1.8, transform=trans)]
        return arts


class _Bracket:
    pass


def legend_items(keys: list[str]) -> tuple[list, list]:
    lab = {
        "full_path": (Line2D([], [], color=NAVY, lw=2.8), "FULL path"),
        "typ_path": (Line2D([], [], color=GREY, lw=2.2), "TYPICAL path"),
        "reached": (Line2D([], [], marker="o", ls="none", ms=8, mfc=GREEN, mec="white"), "reached B"),
        "failed_oob": (Line2D([], [], marker="X", ls="none", ms=10, mfc=RED, mec="white"), "left the map"),
        "trench": (Patch(facecolor=RED, edgecolor="none"), "trench (ground truth)"),
        "crest": (Line2D([], [], color=AMBER, lw=1.8, ls=(0, (4, 2.5))), "crest line"),
        "trail": (Patch(facecolor=TRAIL_LEGEND, edgecolor="none"), "gravel trail"),
        "object": (Line2D([], [], marker="o", ls="none", ms=7, mfc=OBJ_LETHAL, mec="none"), "rock / tree / log"),
        "small": (Line2D([], [], marker="o", ls="none", ms=5, mfc=OBJ_SMALL, mec="none"), "drive-over rock"),
        "water": (Patch(facecolor=BLUE, alpha=0.55, edgecolor=BLUE), "water / mud"),
        "radius": (Line2D([], [], color=INK, lw=1.1, ls=(0, (3, 2))), "2 m goal radius"),
        "ticks": (Line2D([], [], marker="o", ls="none", ms=5.2, mfc="white", mec=MUTED, mew=1.5), "every 10 s"),
        "confirm": (Line2D([], [], marker="D", ls="none", ms=7.5, mfc=NAVY, mec="white", mew=1.2),
                    "FULL confirms the trench"),
        "light": (_Bracket(), "lighting event (on FULL's path)"),
    }
    hs = [lab[k] for k in keys]
    return [h for h, _ in hs], [t for _, t in hs]


def put_legend(fig, x: float, y_top: float, keys: list[str], ncol: int) -> None:
    handles, labels = legend_items(keys)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x, y_top), ncol=ncol, frameon=False,
               fontsize=FS_LEGEND, handlelength=1.7, columnspacing=1.5, handletextpad=0.5, labelspacing=0.45,
               borderaxespad=0.0, borderpad=0.0, handler_map={_Bracket: BracketHandler()})


LEG_ROW_IN = FS_LEGEND * 1.45 / 72.0


def put_lines(fig, x_in: float, y_top_in: float, lines: list[str], fs: float, W: float, H: float,
              colour: str = INK, weight: str = "normal") -> None:
    for k, ln in enumerate(lines):
        fig.text(x_in / W, (y_top_in - k * fs * LINE / 72.0) / H, ln, fontsize=fs, color=colour, fontweight=weight,
                 ha="left", va="top")


def save(fig, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.10, facecolor=BG)
    plt.close(fig)


def run_record(run: dict) -> dict:
    return {
        "config": run["cfg"], "run_dir": run["dir"],
        "outcome": run["outcome"], "end_time_s": run["time"], "end_event": dict(run["event"]),
        "path_length_m": run["path_length"], "mean_speed_mps": run["mean_speed"],
        "gt_speed_max_mps": float(run["v"].max()),
        "final_error_m": run["final_error"], "spl": run["spl"],
        "ditch_entries": run["ditch_entries"], "collisions": run["collisions"], "water_entries": run["water_entries"],
        "end_xy_m": [round(float(run["x"][-1]), 2), round(float(run["y"][-1]), 2)],
        "sources": [f"{run['dir']}/result.json", f"{run['dir']}/gt/events.json", f"{run['dir']}/gt/states.npz",
                    f"{SUMMARY} (row {run['cfg']},{run['seed']})"],
    }


# =============================================================================================
# figure: FULL vs TYPICAL pair
# =============================================================================================
PAIR_ROLE = {38: "headline", 7: "backup", 37: "backup"}


def pair_texts(seed: int, world: dict, full: dict, typ: dict, det: dict | None, cross: dict) -> dict:
    """In-map labels and below-panel sentences, generated from the data (every clause checked here)."""
    out: dict = {"full_below": "", "typ_below": "", "confirm_label": None, "speed_clause": None}
    if det is not None:
        t, rng = det["t_s"], det["range_to_nearest_cell_m"]
        v0, vmin = det["gt_speed_at_mps"], det["gt_speed_min_next3s_mps"]
        n_on, n_all = det["n_on_gt_trench"], det["n_confirmed_cells"]
        gov = det["gov_binding_counts_next3s"]
        platform_dominant = gov.get("platform", 0) >= 0.8 * det["n_ticks_next3s"]
        where = ("at the crest, " if det["vehicle_dist_to_crest_m"] is not None
                 and det["vehicle_dist_to_crest_m"] <= AT_CREST_M else "")
        one = n_trenches(world) == 1
        if n_on == n_all:
            first = f"{t:.1f} s: {where}FULL confirms the trench {rng:.1f} m ahead"
        else:
            first = (f"{t:.1f} s: first confirmed ditch cells on {'the' if one else 'a'} trench, {rng:.1f} m ahead "
                     f"({n_on} of {n_all} confirmed cells lie on {'it' if one else 'a trench'}, "
                     f"{n_all - n_on} {'are' if n_all - n_on != 1 else 'is'} not on any trench)")
        if det["max_abs_cmd_w_next3s_rad_s"] >= TURN_CMD_W and v0 - vmin > HOLD_SPEED_TOL and platform_dominant:
            # the slowdown coincides with a hard commanded turn while the seen-distance governor was not binding
            speed = f"turns hard toward the gap (slowing to {vmin:.1f} m/s)"
        elif v0 - vmin <= HOLD_SPEED_TOL:
            speed = f"holds {v0:.1f} m/s"
        else:
            speed = None  # a dip that the logs do not tie to the trench: say nothing about speed
        out["speed_clause"] = speed
        out["confirm_label"] = f"{t:.1f} s"
        in_gap = [c for c in cross["FULL"]]
        all_gaps = bool(in_gap) and all(c["in_gap"] for c in in_gap)
        ntr = len({c["trench"] for c in in_gap})
        thru = ("goes through the gap" if ntr == 1 else f"crosses all {ntr} trenches at their gaps") if all_gaps else None
        parts = [first]
        if speed:
            parts.append(speed)
        if thru:
            parts.append(thru)
        s = parts[0]
        if len(parts) == 3:
            s += f", {parts[1]} and {parts[2]}"
        elif len(parts) == 2:
            s += f"; {parts[1]}"
        out["full_below"] = s + f". Reached B at {full['time']:.1f} s."
    ev = typ["event"]
    wheel = WHEEL_TXT.get(ev.get("wheel", ""), "")
    n_typ_cross = len(cross["TYPICAL"])
    lead = "Treats unseen ground as free, 1.5 m/s set-point"
    if typ["outcome"] == "ditch_entry":
        out["typ_below"] = f"{lead}: drove into the trench at {typ['time']:.1f} s ({wheel} wheel first)."
    elif typ["outcome"] == "out_of_bounds":
        tail = "; crossed no trench" if n_typ_cross == 0 else ""
        out["typ_below"] = f"{lead}: drove off the map at {typ['time']:.1f} s{tail}."
    else:
        out["typ_below"] = f"{lead}: {END_TXT[typ['outcome']].lower()} at {typ['time']:.1f} s."
    return out


def fig_pair(seed: int, name: str) -> dict:
    world = load_world(seed)
    full, typ = load_run("FULL", seed), load_run("TYPICAL", seed)
    assert full["outcome"] == "success", (seed, full["outcome"])
    fam = world["scn"]["family"]
    role = PAIR_ROLE[seed]
    det = trench_confirmation(full, world)
    cross = {"FULL": trench_crossings(full, world), "TYPICAL": trench_crossings(typ, world)}
    registered = None
    if seed == 38:  # the headline pair: every number must equal the registered results/example_runs_eval.json
        ex = json.loads((ROOT / EXAMPLE_JSON).read_text())["pair"]
        fc = ex["full_confirmation"]
        assert ex["seed"] == 38 and abs(ex["end_time_s"]["FULL"] - full["time"]) < 1e-9
        assert abs(ex["end_time_s"]["TYPICAL"] - typ["time"]) < 1e-9
        assert abs(fc["t_s"] - det["t_s"]) < 1e-9 and fc["n_on_gt_trench"] == det["n_on_gt_trench"]
        assert fc["n_confirmed_cells"] == det["n_confirmed_cells"]
        assert abs(fc["range_to_nearest_cell_m"] - det["range_to_nearest_cell_m"]) < 1e-6
        assert abs(fc["gt_speed_min_next3s_mps"] - det["gt_speed_min_next3s_mps"]) < 1e-9
        assert fc["gov_binding_counts"] == det["gov_binding_counts_next3s"]
        assert abs(fc["max_abs_cmd_w_rad_s"] - det["max_abs_cmd_w_next3s_rad_s"]) < 1e-9
        registered = {k: claim(k) for k in (
            "example_eval_s038_full_reached_b_s", "example_eval_s038_typical_end_s",
            "example_eval_s038_full_trench_confirmed_s", "example_eval_s038_full_trench_confirmed_range_m",
            "example_eval_s038_full_speed_min_after_confirm_mps", "example_eval_s038_full_cmd_w_max_after_confirm",
            "example_eval_s038_full_trench_cells_on_gt")}
        assert registered["example_eval_s038_full_reached_b_s"] == f"{full['time']:.1f}"
        assert registered["example_eval_s038_typical_end_s"] == f"{typ['time']:.1f}"
        assert registered["example_eval_s038_full_trench_confirmed_s"] == f"{det['t_s']:.1f}"
        assert registered["example_eval_s038_full_trench_confirmed_range_m"] == f"{det['range_to_nearest_cell_m']:.1f}"
        assert registered["example_eval_s038_full_speed_min_after_confirm_mps"] == f"{det['gt_speed_min_next3s_mps']:.1f}"
    txt = pair_texts(seed, world, full, typ, det, cross)
    de = {c: ditch_entry_seeds(c) for c in ("FULL", "TYPICAL")}
    f23 = {c: f2f3_success(c) for c in ("FULL", "TYPICAL")}

    def seeds_txt(d):
        fams = {f.split("_")[0] for f in d["families"]}
        pre = (f"{next(iter(fams))} seeds " if len(fams) == 1 else "seeds ")
        return pre + " and ".join(str(s) for s in d["seeds"])
    note = (f"{'One selected seed' if role == 'headline' else 'Backup example, one selected seed'}. "
            f"Over all 60 EVAL runs each stack drove into a trench twice (FULL: {seeds_txt(de['FULL'])}; "
            f"TYPICAL: {seeds_txt(de['TYPICAL'])}). Ditch-field and crest worlds reached B: "
            f"FULL {f23['FULL']['k']}/20, TYPICAL {f23['TYPICAL']['k']}/20.")
    assert len(de["FULL"]["seeds"]) == 2 and len(de["TYPICAL"]["seeds"]) == 2  # "twice" above
    fam_txt = FAMILY_TXT[fam].split(" ", 1)[1].lower()
    fam_code = FAMILY_TXT[fam].split(" ", 1)[0]
    prov = (f"Simulated · tier-0 synthetic depth sensor, no images · held-out EVAL seed {seed} ({fam_code} {fam_txt}) · "
            f"{RUNS}/{{FULL,TYPICAL}}/{seed:03d}"
            + (f", {EXAMPLE_JSON}" if seed == 38 else "") + f", {CLAIMS_CSV}")

    # ---- geometry (inches, bottom-up) ----
    W = FIG_W_IN
    text_w = PANEL_W_IN - 0.05
    full_lines = wrap(txt["full_below"], FS_SUB, text_w - 0.22)
    typ_lines = wrap(txt["typ_below"], FS_SUB, text_w)
    n_below = max(len(full_lines), len(typ_lines))
    note_lines = wrap(note, FS_NOTE, W - 0.05)
    prov_lines = wrap(prov, FS_PROV, W - 0.05)
    keys = ["trench"] + (["crest"] if any(h["type"] == "crest" for h in world["scn"]["hazards"]) else []) + \
           ["trail", "object", "small"] + (["water"] if world["hz"].water.any() else []) + ["radius", "ticks"] + \
           (["confirm"] if det is not None else [])
    ncol = 5 if len(keys) > 8 else 4
    n_leg_rows = math.ceil(len(keys) / ncol)
    y_prov = 0.02 + block_h_in(len(prov_lines), FS_PROV)
    y_note = y_prov + 0.04 + block_h_in(len(note_lines), FS_NOTE)
    y_leg = y_note + 0.12 + n_leg_rows * LEG_ROW_IN
    y_below = y_leg + 0.16 + block_h_in(n_below, FS_SUB)
    y_panel = y_below + 0.08
    header = 0.70
    H = y_panel + PANEL_H_IN + header
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    xs_in = [0.0, PANEL_W_IN + PANEL_GAP_IN]
    axs = [fig.add_axes([x / W, y_panel / H, PANEL_W_IN / W, PANEL_H_IN / H]) for x in xs_in]
    records, placements = {}, {}
    crest = [hh for hh in world["scn"]["hazards"] if hh["type"] == "crest"]
    ditch = [hh for hh in world["scn"]["hazards"] if hh["type"] == "ditch"]
    for k, (ax, run, colour) in enumerate(((axs[0], full, NAVY), (axs[1], typ, GREY))):
        PLACEMENT_LOG.clear()
        occ = draw_world(ax, world)
        draw_path(ax, occ, run, colour, lw=3.0)
        ticks = draw_tick_dots(ax, occ, run, colour)
        draw_start_goal(ax, occ, world)
        ex, ey = end_marker(ax, occ, run)
        if run["cfg"] == "FULL" and det is not None:
            dx_, dy_ = det["vehicle_xy_m"]
            ax.plot(dx_, dy_, "D", ms=8.5, mfc=NAVY, mec="white", mew=1.4, zorder=10)
            occ.add_disc("mark", dx_, dy_, 0.9)
        sb = place_scale_bar(ax, occ) if k == 0 else None
        ok = run["outcome"] == "success"
        place_text(ax, occ, [f"{END_TXT[run['outcome']]}\nat {run['time']:.1f} s", f"{END_TXT[run['outcome']]} at {run['time']:.1f} s"],
                   (ex, ey), ring([3.5, 5.0, 7.0, 9.0]), fs=FS_END, weight="bold", colour=(INK if ok else RED),
                   name=f"{run['cfg']} end", shrink=1.1)
        if run["cfg"] == "FULL" and det is not None:
            place_text(ax, occ, txt["confirm_label"], tuple(det["vehicle_xy_m"]), ring([2.2, 3.2, 4.5, 6.0]),
                       fs=FS_MAPLAB, weight="bold", colour=NAVY, name="confirm label", shrink=0.9)
        if run["cfg"] == "FULL":
            gaps = gap_midpoints(world)
            if gaps and len(ditch) == 1:
                dmin = [float(np.min(np.hypot(full["x"] - gx, full["y"] - gy))) for gx, gy in gaps]
                gx, gy = gaps[int(np.argmin(dmin))]
                place_text(ax, occ, ["gap in the trench", "gap"], (gx, gy), ring([3.0, 4.5, 6.0, 8.0]), fs=FS_MAPLAB,
                           name="gap", shrink=0.6)
        else:
            if crest:
                q = np.asarray(crest[0]["polyline"], float)
                anc = point_on_polyline(q, 0.82)
                place_text(ax, occ, [f"crest, {crest[0]['drop']:.1f} m drop", f"crest,\n{crest[0]['drop']:.1f} m drop"],
                           anc, ring([2.5, 4.0, 6.0], [180, 150, 210, 120, 240, 0, 30, 330]), fs=FS_MAPLAB,
                           name="crest", shrink=0.4)
            if ditch and len(ditch) == 1:
                q = np.asarray(ditch[0]["polyline"], float)
                anc = point_on_polyline(q, 0.82)
                place_text(ax, occ, [f"trench, {ditch[0]['depth']:.2f} m deep", f"trench,\n{ditch[0]['depth']:.2f} m deep"],
                           anc, ring([2.5, 4.0, 6.0], [0, 30, 330, 60, 300, 180]), fs=FS_MAPLAB, name="trench",
                           shrink=0.4)
        label_start_goal(ax, occ, world)
        label_ticks(ax, occ, run, ticks)
        placements[run["cfg"]] = list(PLACEMENT_LOG) + ([{"name": "scale bar", **sb}] if sb else [])
        cfg_name, cfg_desc = CONFIG_TXT[run["cfg"]]
        x_in = xs_in[k]
        fig.text((x_in + 0.01) / W, (y_panel + PANEL_H_IN + 0.34) / H, cfg_name, fontsize=FS_TITLE, fontweight="bold",
                 color=(NAVY if run["cfg"] == "FULL" else GREY_TXT), ha="left", va="bottom")
        fig.text((x_in + 0.01) / W, (y_panel + PANEL_H_IN + 0.08) / H, cfg_desc, fontsize=FS_SUB, color=MUTED,
                 ha="left", va="bottom")
        rec = run_record(run)
        rec["time_ticks"] = ticks
        rec["trench_crossings"] = cross[run["cfg"]]
        records[run["cfg"]] = rec

    # below-panel sentences (the diamond before FULL's sentence is the in-map marker)
    y_top = y_below - 0.01
    if det is not None:
        fig.add_artist(Line2D([0.075 / W], [(y_top - FS_SUB * 0.62 / 72) / H], marker="D", ms=7.5, mfc=NAVY,
                              mec="white", mew=1.0, transform=fig.transFigure))
    put_lines(fig, 0.01 + (0.22 if det is not None else 0.0), y_top, full_lines, FS_SUB, W, H)
    put_lines(fig, xs_in[1] + 0.01, y_top, typ_lines, FS_SUB, W, H)
    put_legend(fig, 0.004, y_leg / H, keys, ncol)
    put_lines(fig, 0.01, y_note, note_lines, FS_NOTE, W, H, colour=INK)
    put_lines(fig, 0.01, y_prov, prov_lines, FS_PROV, W, H, colour=MUTED)
    save(fig, name)

    # ---- sidecar ----
    numbers = [
        {"what": "FULL outcome and time", "value": f"reached B at {full['time']:.1f} s ({full['time']})",
         "source": f"{full['dir']}/result.json; {SUMMARY} row FULL,{seed}",
         "claim_id": "example_eval_s038_full_reached_b_s" if seed == 38 else None},
        {"what": "TYPICAL outcome and time", "value": f"{typ['outcome']} at {typ['time']:.1f} s ({typ['time']})",
         "source": f"{typ['dir']}/result.json + gt/events.json; {SUMMARY} row TYPICAL,{seed}",
         "claim_id": "example_eval_s038_typical_end_s" if seed == 38 else None},
    ]
    if det is not None:
        numbers += [
            {"what": "FULL trench confirmation time (in-map label and below-panel sentence)", "value": f"{det['t_s']:.1f} s",
             "source": det["file"], "how": f"first tick whose rolling map held >= {MIN_MATCHED_CELLS} confirmed-ditch "
             f"cells within {MATCH_TOL_M} m of the GT trench (A-frame -> world via the scenario start pose)",
             "claim_id": "example_eval_s038_full_trench_confirmed_s" if seed == 38 else None},
            {"what": "range to the nearest confirmed trench cell", "value": f"{det['range_to_nearest_cell_m']:.1f} m",
             "source": f"{det['file']} + {full['dir']}/gt/states.npz", "how": "GT vehicle position at that tick to the "
             "nearest matched confirmed cell",
             "claim_id": "example_eval_s038_full_trench_confirmed_range_m" if seed == 38 else None},
            {"what": "confirmed cells on the GT trench / all confirmed cells in the map crop",
             "value": f"{det['n_on_gt_trench']}/{det['n_confirmed_cells']}", "source": det["file"],
             "claim_id": "example_eval_s038_full_trench_cells_on_gt" if seed == 38 else None,
             "drawn": det["n_on_gt_trench"] != det["n_confirmed_cells"]},
        ]
        if txt["speed_clause"]:
            numbers.append({"what": "speed clause", "value": txt["speed_clause"],
                            "source": f"{full['dir']}/gt/states.npz v; debug ticks gov_binding / cmd_w",
                            "how": f"GT speed at the tick {det['gt_speed_at_mps']:.3f} m/s, min over the next "
                                   f"{AFTER_WINDOW_S:.0f} s {det['gt_speed_min_next3s_mps']:.3f} m/s; governor binding "
                                   f"{det['gov_binding_counts_next3s']}; peak |cmd_w| "
                                   f"{det['max_abs_cmd_w_next3s_rad_s']:.2f} rad/s",
                            "claim_id": ("example_eval_s038_full_speed_min_after_confirm_mps + "
                                         "example_eval_s038_full_cmd_w_max_after_confirm") if seed == 38 else None})
    if crest:
        numbers.append({"what": "crest drop", "value": f"{crest[0]['drop']:.1f} m ({crest[0]['drop']})",
                        "source": f"{SCEN}/{seed}.json hazards[crest].drop (scenario input, not a result)",
                        "claim_id": None})
    if ditch and len(ditch) == 1:
        numbers.append({"what": "trench depth", "value": f"{ditch[0]['depth']:.2f} m ({ditch[0]['depth']})",
                        "source": f"{SCEN}/{seed}.json hazards[ditch].depth (scenario input, not a result)",
                        "claim_id": None})
    numbers += [
        {"what": "trench entries over all 60 EVAL runs",
         "value": f"FULL {de['FULL']['seeds']}, TYPICAL {de['TYPICAL']['seeds']}", "source": SUMMARY,
         "claim_id": f"{de['FULL']['claim_id']}, {de['TYPICAL']['claim_id']}"},
        {"what": "F2 + F3 reached B", "value": f"FULL {f23['FULL']['k']}/20, TYPICAL {f23['TYPICAL']['k']}/20",
         "source": "results/closed_loop_eval.json (= summary.csv count)",
         "claim_id": f"{f23['FULL']['claim_id']}, {f23['TYPICAL']['claim_id']}"},
        {"what": "success radius", "value": f"{world['scn']['mission']['success_radius_m']:.0f} m",
         "source": f"{SCEN}/{seed}.json mission.success_radius_m (scenario input)", "claim_id": None},
    ]
    unregistered = [n for n in numbers if n["claim_id"] is None and "scenario input" not in n["source"]]
    to_register = []
    if seed != 38:
        base = (f"EVAL closed loop run 2, sensor mode tier0, held-out seed {seed} ({fam}); one selected example, see "
                "docs/EVAL_PREREGISTRATION.md for the aggregate")
        to_register = [
            {"id": f"example_eval_s{seed:03d}_full_reached_b_s", "value": f"{full['time']:.1f}", "unit": "s",
             "label": "Simulated", "source": f"{full['dir']}/result.json#time", "note": f"{base}; FULL reached B"},
            {"id": f"example_eval_s{seed:03d}_typical_end_s", "value": f"{typ['time']:.1f}", "unit": "s",
             "label": "Simulated", "source": f"{typ['dir']}/result.json#time",
             "note": f"{base}; TYPICAL run ended ({typ['outcome']})"},
        ]
        if det is not None:
            to_register += [
                {"id": f"example_eval_s{seed:03d}_full_trench_confirmed_s", "value": f"{det['t_s']:.1f}", "unit": "s",
                 "label": "Simulated", "source": det["file"], "note": f"{base}; first tick with >= {MIN_MATCHED_CELLS} "
                 "confirmed ditch cells on the GT trench (same rule as metagross/eval/example_runs.py)"},
                {"id": f"example_eval_s{seed:03d}_full_trench_confirmed_range_m",
                 "value": f"{det['range_to_nearest_cell_m']:.1f}", "unit": "m", "label": "Simulated",
                 "source": det["file"], "note": f"{base}; GT distance to the nearest confirmed trench cell"},
                {"id": f"example_eval_s{seed:03d}_full_trench_cells_on_gt",
                 "value": f"{det['n_on_gt_trench']}/{det['n_confirmed_cells']}", "unit": "cells", "label": "Simulated",
                 "source": det["file"], "note": f"{base}; confirmed ditch cells within {MATCH_TOL_M} m of the GT trench / all"},
            ]
            if txt["speed_clause"] and txt["speed_clause"].startswith("holds"):
                to_register.append({"id": f"example_eval_s{seed:03d}_full_speed_at_confirm_mps",
                                    "value": f"{det['gt_speed_at_mps']:.1f}", "unit": "m/s", "label": "Simulated",
                                    "source": f"{full['dir']}/gt/states.npz",
                                    "note": f"{base}; GT speed at confirmation (min over the next 3 s "
                                            f"{det['gt_speed_min_next3s_mps']:.2f})"})
    side = {
        "figure": name, "role": role,
        "label": "Simulated (tier-0 synthetic depth sensor, no images; held-out EVAL seed)",
        "generator": "deck_assets/final/src/runs_map.py",
        "provenance_status": ("all measured numbers registered in results/claims.csv" if seed == 38 else
                              "NOT REGISTERED: the per-seed numbers below are read from results/runs_eval_tier0 but "
                              "have no row in results/claims.csv; register `claims_to_register` (e.g. by extending "
                              "metagross/eval/example_runs.py to more seeds) before this backup goes on a slide"),
        "seed": seed, "family": fam, "split": world["scn"]["split"],
        "scenario": f"{SCEN}/{seed}.json", "scenario_sha256": world["scn"]["sha256"],
        "world_m": {"x": [0.0, 64.0], "y": [0.0, 40.0], "res_m": world["terrain"].grid.res},
        "start_xy_m": world["scn"]["start"]["xy"], "goal_xy_m": world["scn"]["goal"]["xy"],
        "success_radius_m": world["scn"]["mission"]["success_radius_m"],
        "hazards": hazard_summary(world),
        "text_on_figure": {"full_below": txt["full_below"], "typical_below": txt["typ_below"], "note": note,
                           "provenance": prov, "headers": {c: CONFIG_TXT[c] for c in CONFIG_TXT}},
        "numbers": numbers,
        "numbers_without_claim_row": unregistered,
        "claims_to_register": to_register,
        "registered_values_checked": registered,
        "runs": records,
        "full_trench_confirmation": det,
        "selection": {
            "f2f3_seeds_full_reached_b_typical_failed": f2f3_full_wins(),
            "trench_entries_all_60_eval": {c: de[c] for c in de},
            "why_this_seed": ("headline: the trench is hidden behind a crest; TYPICAL's two trench entries are seeds 7 "
                              "and 38, FULL's are 14 and 26 (on F3 seed 26 FULL drove into the trench and TYPICAL "
                              "reached B), so this seed is one favourable example, not a rate"
                              if seed == 38 else "backup pair; same caveats as run_map_pair.json"),
        },
        "label_placement": placements,
        "how_drawn": {
            "background": f"hillshade (az 315, alt 40, vert. exag. {RELIEF_VERT_EXAG}) of the scenario heightmap "
                          f"smoothed by {RELIEF_SMOOTH_M} m, tinted by material (gravel trail = beige band); 0.25 m "
                          "contours of the landform (heightmap + trench excavation, smoothed)",
            "trench": "GT ditch mask from metagross.sim.hazards.gt_hazard_raster (cells excavated > 0.05 m)",
            "crest": "scenario hazards[type=crest].polyline",
            "objects": "static footprints from metagross.sim.objects.parse_static_objects; dark = protrudes above "
                       f"ground clearance (lethal), light = drive-over; true radius with a drawing floor of "
                       f"{OBJ_MIN_R_LETHAL}/{OBJ_MIN_R_SMALL} m",
            "paths": "GT body-origin x, y from gt/states.npz (0.04 s physics steps)",
            "end markers": "last GT state; green dot = reached B, red X = referee terminal event (drawn unclipped "
                           "so an exit at the frame stays whole)",
            "confirmation marker": "navy diamond = GT vehicle position at the confirmation tick",
            "labels": "placed by an occupancy check (objects, paths, trench, crest, markers, earlier labels); tick "
                      "labels are dropped when every candidate overlaps something",
        },
    }
    (OUT / f"{name}.json").write_text(json.dumps(side, indent=1))
    return side


# =============================================================================================
# figure: gallery of FULL successes (with TYPICAL on the same world)
# =============================================================================================
GALLERY = [(24, "F1_trail"), (37, "F2_ditch_field"), (2, "F3_crest_ditch"), (4, "F5_lighting")]
EVENT_WORD = {"glare": "glare", "dim": "dimming", "dust": "dust"}


def lighting_bracket(ax, occ: Occ, run: dict, t0: float, t1: float, off: float = 1.9, tick: float = 1.2):
    """Dimension-line bracket beside FULL's path over [t0, t1] (the side with fewer conflicts)."""
    m = (run["t"] >= t0) & (run["t"] <= t1)
    xs, ys = run["x"][m], run["y"][m]
    from scipy.ndimage import gaussian_filter1d
    gx, gy = gaussian_filter1d(np.gradient(xs), 8), gaussian_filter1d(np.gradient(ys), 8)
    n = np.hypot(gx, gy) + 1e-9
    nx, ny = -gy / n, gx / n
    best = None
    for side in (1.0, -1.0):
        bx, by = xs + side * off * nx, ys + side * off * ny
        c = occ.count_points(bx, by, ("obj", "mark", "haz", "label"))
        if best is None or c < best[0]:
            best = (c, side, bx, by)
    c, side, bx, by = best
    ax.plot(bx, by, color=INK, lw=1.8, zorder=9, solid_capstyle="butt")
    for k in (0, -1):
        ax.plot([xs[k] + side * (off - tick / 2) * nx[k], xs[k] + side * (off + tick / 2) * nx[k]],
                [ys[k] + side * (off - tick / 2) * ny[k], ys[k] + side * (off + tick / 2) * ny[k]],
                color=INK, lw=1.8, zorder=9, solid_capstyle="butt")
    occ.add_polyline("label", bx, by, 0.35)
    mid = len(bx) // 2
    return (float(bx[mid]), float(by[mid])), side, (float(side * nx[mid]), float(side * ny[mid]))


def fig_gallery(name: str) -> dict:
    W = FIG_W_IN
    header = 0.62
    row_gap = 0.14
    fams_shown = [f for _, f in GALLERY]
    rates = {f: {c: family_success(c, f) for c in ("FULL", "TYPICAL")} for f in FAMILY_TXT}
    not_shown = [f for f in FAMILY_TXT if f not in fams_shown]
    note = ("Hand-picked seeds: one FULL success per family, so this is not a success rate. Not shown: " +
            " and ".join(f"{FAMILY_TXT[f]} (reached B: FULL {rates[f]['FULL']['k']}/10, TYPICAL "
                         f"{rates[f]['TYPICAL']['k']}/10)" for f in not_shown) + ".")
    prov = ("Simulated · tier-0 synthetic depth sensor, no images: glare and dimming act only as depth dropout and "
            f"noise · held-out EVAL seeds {', '.join(str(s) for s, _ in GALLERY)} · {RUNS}, "
            "results/closed_loop_eval.json, " + CLAIMS_CSV)
    note_lines = wrap(note, FS_NOTE, W - 0.05)
    prov_lines = wrap(prov, FS_PROV, W - 0.05)
    # which end states occur, for the legend
    runs = {s: (load_run("FULL", s), load_run("TYPICAL", s)) for s, _ in GALLERY}
    typ_fail = sorted({t["outcome"] for _, t in runs.values() if t["outcome"] != "success"})
    assert typ_fail in ([], ["out_of_bounds"]), typ_fail  # legend says "left the map"
    keys = ["full_path", "typ_path", "reached"] + (["failed_oob"] if typ_fail else []) + \
           ["trench", "crest", "trail", "object", "small", "radius", "light"]
    ncol = 4
    n_leg_rows = math.ceil(len(keys) / ncol)
    y_prov = 0.02 + block_h_in(len(prov_lines), FS_PROV)
    y_note = y_prov + 0.04 + block_h_in(len(note_lines), FS_NOTE)
    y_leg = y_note + 0.12 + n_leg_rows * LEG_ROW_IN
    y_row1 = y_leg + 0.14
    y_row0 = y_row1 + PANEL_H_IN + header + row_gap
    H = y_row0 + PANEL_H_IN + header
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    panels = []
    for k, (seed, fam) in enumerate(GALLERY):
        world = load_world(seed)
        assert world["scn"]["family"] == fam, (seed, fam)
        full, typ = runs[seed]
        assert full["outcome"] == "success", seed
        x_in = (k % 2) * (PANEL_W_IN + PANEL_GAP_IN)
        y_in = y_row0 if k < 2 else y_row1
        ax = fig.add_axes([x_in / W, y_in / H, PANEL_W_IN / W, PANEL_H_IN / H])
        PLACEMENT_LOG.clear()
        occ = draw_world(ax, world)
        draw_path(ax, occ, typ, GREY, lw=2.0, zorder=6.5)
        draw_path(ax, occ, full, NAVY, lw=2.8, zorder=7.5)
        draw_start_goal(ax, occ, world)
        end_marker(ax, occ, typ, small=True)
        end_marker(ax, occ, full)
        sb = place_scale_bar(ax, occ) if k == 0 else None
        light_marks = []
        for ev in world["scn"]["lighting"]["events"]:
            t0, t1 = float(ev["t0"]), float(ev["t1"])
            if t0 >= full["time"]:
                continue  # event starts after FULL's arrival
            anc, side, nrm = lighting_bracket(ax, occ, full, t0, min(t1, full["time"]))
            ang = math.degrees(math.atan2(nrm[1], nrm[0]))
            place_text(ax, occ, [f"{EVENT_WORD[ev['type']]} {t0:.1f}-{t1:.1f} s",
                                 f"{EVENT_WORD[ev['type']]}\n{t0:.1f}-{t1:.1f} s"], anc,
                       ring([1.8, 3.0, 4.5, 6.0], [ang, ang + 30, ang - 30, ang + 60, ang - 60, ang + 90, ang - 90]),
                       fs=FS_MAPLAB, name=f"light {ev['type']}", shrink=0.3)
            light_marks.append({"type": ev["type"], "t0_s": t0, "t1_s": t1, "gain": ev["gain"],
                                "drawn_until_s": min(t1, full["time"]), "bracket_side": side})
        label_start_goal(ax, occ, world)
        r = rates[fam]
        t_title = FAMILY_TXT[fam]
        fig.text((x_in + 0.01) / W, (y_in + PANEL_H_IN + 0.31) / H, t_title, fontsize=13, fontweight="bold",
                 color=INK, ha="left", va="bottom")
        fig.text((x_in + 0.01 + text_w_in(t_title, 13, "bold")) / W, (y_in + PANEL_H_IN + 0.31) / H,
                 f"  ·  seed {seed}", fontsize=FS_SUB, color=MUTED, ha="left", va="bottom")
        sub = f"reached B in this family: FULL {r['FULL']['k']}/10, TYPICAL {r['TYPICAL']['k']}/10"
        assert text_w_in(sub, FS_SUB) < PANEL_W_IN - 0.05, sub
        fig.text((x_in + 0.01) / W, (y_in + PANEL_H_IN + 0.07) / H, sub, fontsize=FS_SUB, color=MUTED,
                 ha="left", va="bottom")
        rec = run_record(full)
        rec.update({"seed": seed, "family": fam, "scenario": f"{SCEN}/{seed}.json",
                    "scenario_sha256": world["scn"]["sha256"], "hazards": hazard_summary(world),
                    "typical_same_seed": run_record(typ),
                    "trench_crossings": {"FULL": trench_crossings(full, world), "TYPICAL": trench_crossings(typ, world)},
                    "family_rates": r,
                    "lighting": {k2: world["scn"]["lighting"][k2] for k2 in ("sun_elev_deg", "fog_density", "events")},
                    "lighting_events_drawn": light_marks,
                    "label_placement": list(PLACEMENT_LOG) + ([{"name": "scale bar", **sb}] if sb else [])})
        panels.append(rec)
    put_legend(fig, 0.004, y_leg / H, keys, ncol)
    put_lines(fig, 0.01, y_note, note_lines, FS_NOTE, W, H, colour=INK)
    put_lines(fig, 0.01, y_prov, prov_lines, FS_PROV, W, H, colour=MUTED)
    save(fig, name)
    side = {
        "figure": name, "label": "Simulated (tier-0 synthetic depth sensor, no images; held-out EVAL seeds)",
        "generator": "deck_assets/final/src/runs_map.py",
        "provenance_status": ("numbers on the figure: per-family reached-B rates (registered rows "
                              "closed_loop_eval_tier0_<CFG>_<FAMILY>_success) and lighting windows (scenario inputs). "
                              "Per-seed times are deliberately not drawn (not registered)."),
        "text_on_figure": {"note": note, "provenance": prov},
        "numbers": (
            [{"what": f"{FAMILY_TXT[f]} reached B, {c}", "value": f"{rates[f][c]['k']}/10",
              "source": "results/closed_loop_eval.json (= summary.csv count)", "claim_id": rates[f][c]["claim_ids"][0]}
             for f in FAMILY_TXT for c in ("FULL", "TYPICAL")]
            + [{"what": f"seed {p['seed']} lighting window {e['type']}", "value": f"{e['t0_s']:.1f}-{e['t1_s']:.1f} s",
                "source": f"{SCEN}/{p['seed']}.json lighting.events (scenario input, not a result)", "claim_id": None}
               for p in panels for e in p["lighting_events_drawn"]]),
        "per_seed_outcomes_drawn": [
            {"seed": p["seed"], "family": p["family"], "FULL": p["outcome"], "TYPICAL": p["typical_same_seed"]["outcome"],
             "source": f"{SUMMARY} rows FULL,{p['seed']} and TYPICAL,{p['seed']}"} for p in panels],
        "panels": panels,
        "selection": ("Hand-picked, one FULL success per family for F1, F2, F3, F5 (F4 sudden obstacle and F6 water / "
                      "mud are not shown; FULL reached B on 2/10 and 1/10 of them). FULL reached B on F1 seeds "
                      + ", ".join(str(int(r_["seed"])) for r_ in SUM_ROWS if r_["config_name"] == "FULL"
                                  and r_["family"] == "F1_trail" and r_["success"] == "True")
                      + "; F2 seeds " + ", ".join(str(int(r_["seed"])) for r_ in SUM_ROWS if r_["config_name"] == "FULL"
                                                  and r_["family"] == "F2_ditch_field" and r_["success"] == "True")
                      + "; F3 seeds " + ", ".join(str(int(r_["seed"])) for r_ in SUM_ROWS if r_["config_name"] == "FULL"
                                                  and r_["family"] == "F3_crest_ditch" and r_["success"] == "True")
                      + "; F5 all 10. Seed 24's FULL path detours north of the trail rather than following it. "
                        "TYPICAL also reached B on seeds 24 and 4; it left the map on seeds 37 and 2."),
        "note_lighting": ("In tier-0 there are no images: glare is a depth-dropout disc and halo around the sun "
                          "direction, dimming inflates depth noise and dropout (metagross/sim/sensors.py)."),
        "how_drawn": ("same basemap as run_map_pair; navy = FULL GT path, grey = TYPICAL GT path on the same world; "
                      "ink brackets beside FULL's path = time windows of scenario lighting events while FULL drove"),
    }
    # the selection text above must agree with the data it summarises
    for p in panels:
        if p["seed"] in (24, 4):
            assert p["typical_same_seed"]["outcome"] == "success"
        if p["seed"] in (37, 2):
            assert p["typical_same_seed"]["outcome"] == "out_of_bounds"
    (OUT / f"{name}.json").write_text(json.dumps(side, indent=1))
    return side


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for seed, nm in ((38, "run_map_pair"), (7, "run_map_pair_s007"), (37, "run_map_pair_s037")):
        print(nm)
        fig_pair(seed, nm)
    print("run_map_gallery")
    fig_gallery("run_map_gallery")
    print("wrote", ", ".join(f"{n}.png/.svg/.json" for n in
                              ("run_map_pair", "run_map_pair_s007", "run_map_pair_s037", "run_map_gallery")))
