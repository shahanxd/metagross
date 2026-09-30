"""Bird's-eye run maps from real closed-loop EVAL runs, for the final deck.

Run from the repo root:
    OMP_NUM_THREADS=2 python deck_assets/final/src/runs_map.py

Outputs (deck_assets/final/):
    run_map_pair.png / .svg / .json         FULL vs TYPICAL on the same held-out EVAL world (seed 38,
                                            F3 crest + hidden trench): FULL reached B, TYPICAL drove
                                            into the trench.
    run_map_pair_s007.*, run_map_pair_s037.*  two alternative pairs (F2 ditch field) for backup slides.
    run_map_gallery.png / .svg / .json      four FULL successes from four scenario families.

Every drawn element comes from files in the repo:
    world        data/scenarios/eval/<seed>.json  (heightmap, material raster, hazards, objects),
                 decoded with metagross.sim.terrain.Terrain.from_scenario and the ground-truth
                 hazard raster metagross.sim.hazards.gt_hazard_raster (the referee's own masks)
    paths        results/runs_eval_tier0/<CONFIG>/<seed:03d>/gt/states.npz  (GT x, y per 0.04 s step)
    outcomes     results/runs_eval_tier0/<CONFIG>/<seed:03d>/result.json + gt/events.json,
                 cross-checked against results/runs_eval_tier0/summary.csv
    detection    results/runs_eval_tier0/FULL/<seed:03d>/autonomy/debug/tick_*.npz  (the stack's own
                 per-tick log: first tick whose rolling map confirmed new trench cells)
Label: Simulated (tier-0 synthetic depth sensor, no images; held-out EVAL seeds).
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
from matplotlib.colors import LightSource, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Polygon  # noqa: E402

from metagross.sim.hazards import gt_hazard_raster  # noqa: E402
from metagross.sim.objects import parse_static_objects, rect_corners  # noqa: E402
from metagross.sim.terrain import Terrain  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "deck_assets" / "final"
RUNS = "results/runs_eval_tier0"
SUMMARY = f"{RUNS}/summary.csv"
SCEN = "data/scenarios/eval"

# ---- palette (deck brief) ------------------------------------------------------------------
NAVY = "#1F497D"      # ours / FULL
GREY = "#9A9A9A"      # baseline / TYPICAL
GREY_TXT = "#6B6B6B"  # TYPICAL header text (the path grey is too light for text)
GREEN = "#2E7D32"     # success
RED = "#C62828"       # hazard / failure
AMBER = "#D98E04"     # crest / warning
BLUE = "#1E88E5"      # water
INK = "#1A1A1A"
MUTED = "#5C5C5C"
FAINT = "#8A8A8A"
GRID = "#E6E6E6"
BG = "#FFFFFF"
OBJ_LETHAL = "#2F2F2F"
OBJ_SMALL = "#A3A3A3"
# material tints for the relief background (kept very light so the data carries the colour)
MAT_RGB = {
    0: "#F3F3EF",  # grass
    1: "#F0ECE3",  # dirt
    2: "#E6DCC6",  # gravel trail
    3: "#ECECE9",  # rocky ground
    4: "#D9CCB4",  # mud
    5: "#CFE3F7",  # water (also outlined in BLUE)
}

CONFIG_TXT = {
    "FULL": ("FULL stack", "drives only on ground it has seen, speed governor"),
    "TYPICAL": ("TYPICAL baseline", "unknown = free, no ditch detector, fixed 1.5 m/s"),
}
FAMILY_TXT = {
    "F1_trail": "F1  Trail",
    "F2_ditch_field": "F2  Ditch field",
    "F3_crest_ditch": "F3  Crest + hidden trench",
    "F4_sudden_obstacle": "F4  Sudden obstacle",
    "F5_lighting": "F5  Lighting",
    "F6_water_mud": "F6  Water / mud",
}
EVENT_TXT = {
    "success": "Reached B",
    "ditch_entry": "Drove into the trench",
    "out_of_bounds": "Drove off the map",
    "stuck": "Stuck",
    "collision": "Hit an obstacle",
    "water_entry": "Drove into water",
    "arrived_short": "Stopped short of B",
    "timeout": "Timed out",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
    "svg.fonttype": "path",
    "figure.facecolor": BG,
    "axes.facecolor": BG,
})
DPI = 200
HALO = [pe.withStroke(linewidth=3.2, foreground="white")]
BG_RES_M = 0.10  # relief raster resolution in the figure (terrain is 0.05 m; halved to keep the SVG small)


# ---- data ----------------------------------------------------------------------------------
def summary_rows() -> dict:
    with open(ROOT / SUMMARY, newline="") as fh:
        return {(r["config_name"], int(r["seed"])): r for r in csv.DictReader(fh)}


SUM = summary_rows()


def load_world(seed: int) -> dict:
    scn = json.loads((ROOT / SCEN / f"{seed}.json").read_text())
    terrain = Terrain.from_scenario(scn)
    hz = gt_hazard_raster(scn, terrain)
    _, feet = parse_static_objects(scn, terrain)
    return {"seed": seed, "scn": scn, "terrain": terrain, "hz": hz, "feet": feet}


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
        "t": st["t"], "x": st["x"], "y": st["y"], "v": st["v"],
        "outcome": outcome, "time": float(res["time"]), "event": term[0],
        "path_length": float(res["path_length"]), "mean_speed": float(res["mean_speed"]),
        "final_error": float(res["final_error"]), "spl": res.get("spl"),
        "ditch_entries": int(res["ditch_entries"]), "collisions": int(res["collisions"]),
        "water_entries": int(res["water_entries"]),
    }


def first_trench_confirmation(run: dict, world: dict, tol_m: float = 1.0, min_cells: int = 3) -> dict | None:
    """First tick at which FULL's own rolling map held confirmed trench cells that lie on the GT trench.

    Source: autonomy/debug/tick_*.npz (the stack's per-tick log). ``extra_map_crop_confirmed`` is the
    cumulative confirmed-ditch layer of the rolling map, cropped +-10 m around the vehicle, in the
    A-frame (origin = start pose, x along the launch heading). Cells are mapped to the world with the
    scenario start pose and matched against the referee's GT ditch mask (within ``tol_m``); the
    first tick with at least ``min_cells`` matched cells is the confirmation. Also reported: the GT
    distance from the vehicle to the nearest matched cell, GT speed at that tick and the lowest GT
    speed over the following 3 s.
    """
    from scipy.ndimage import distance_transform_edt

    g = world["terrain"].grid
    dist_gt = distance_transform_edt(~world["hz"].ditch) * g.res
    x0, y0 = world["scn"]["start"]["xy"]
    yaw0 = float(world["scn"]["start"]["yaw"])
    c, s_ = math.cos(yaw0), math.sin(yaw0)
    files = sorted(glob.glob(str(ROOT / run["dir"] / "autonomy" / "debug" / "tick_*.npz")))
    for f in files:
        b = np.load(f)
        conf = b["extra_map_crop_confirmed"]
        if not conf.any():
            continue
        ex = json.loads(str(b["extras_json"]))
        ox, oy = ex["map_crop_origin"]
        r = float(ex["map_res_m"])
        ii, jj = np.nonzero(conf)
        xa, ya = ox + (jj + 0.5) * r, oy + (ii + 0.5) * r
        xw, yw = x0 + c * xa - s_ * ya, y0 + s_ * xa + c * ya
        gi = np.clip(np.round((yw - g.origin_y) / g.res).astype(int), 0, g.ny - 1)
        gj = np.clip(np.round((xw - g.origin_x) / g.res).astype(int), 0, g.nx - 1)
        on_trench = dist_gt[gi, gj] <= tol_m
        if on_trench.sum() < min_cells:
            continue
        t = float(b["t"])
        vx, vy = at_time(run, t)
        win = (run["t"] >= t) & (run["t"] <= t + 3.0)
        return {"t": t, "x": vx, "y": vy,
                "n_confirmed_cells_in_crop": int(len(on_trench)), "n_on_gt_trench": int(on_trench.sum()),
                "match_tolerance_m": tol_m, "map_res_m": r,
                "dist_to_nearest_confirmed_cell_m": float(np.hypot(xw[on_trench] - vx, yw[on_trench] - vy).min()),
                "dist_to_gt_trench_m": float(dist_gt[int(round((vy - g.origin_y) / g.res)), int(round((vx - g.origin_x) / g.res))]),
                "v_at_mps": float(np.interp(t, run["t"], run["v"])), "v_min_next3s_mps": float(run["v"][win].min()),
                "file": str(Path(f).relative_to(ROOT))}
    return None


def at_time(run: dict, t: float) -> tuple[float, float]:
    return float(np.interp(t, run["t"], run["x"])), float(np.interp(t, run["t"], run["y"]))


# ---- drawing -------------------------------------------------------------------------------
def extent(world: dict) -> tuple[float, float, float, float]:
    g = world["terrain"].grid
    return (g.origin_x - g.res / 2, g.origin_x + (g.nx - 0.5) * g.res,
            g.origin_y - g.res / 2, g.origin_y + (g.ny - 0.5) * g.res)


def relief_rgb(world: dict) -> np.ndarray:
    """Light shaded relief tinted by material. Rows along +y (drawn with origin='lower')."""
    from scipy.ndimage import gaussian_filter

    t = world["terrain"]
    # light smoothing (0.15 m) so centimetre roughness does not read as noise at slide scale
    h = gaussian_filter(t.height.astype(np.float64), RELIEF_SMOOTH_M / t.grid.res)
    ls = LightSource(azdeg=315, altdeg=40)
    # LightSource assumes row 0 at the top (north); our row 0 is south, so flip in and out.
    shade = np.flipud(ls.hillshade(np.flipud(h), vert_exag=RELIEF_VERT_EXAG, dx=t.grid.res, dy=t.grid.res))
    lut = np.array([to_rgb(MAT_RGB.get(k, MAT_RGB[0])) for k in range(256)])
    rgb = np.clip(lut[t.material] * (0.80 + 0.25 * shade[..., None]), 0, 1)
    k = int(round(BG_RES_M / t.grid.res))
    ny, nx = (rgb.shape[0] // k) * k, (rgb.shape[1] // k) * k
    return rgb[:ny, :nx].reshape(ny // k, k, nx // k, k, 3).mean(axis=(1, 3))


def draw_world(ax, world: dict) -> None:
    from scipy.ndimage import gaussian_filter

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
    if hz.water.any():  # water / mud: flat, semantically hazardous
        ax.contourf(xs, ys, hz.water.astype(np.float32), levels=[0.5, 1.5], colors=[BLUE], alpha=0.55, zorder=2)
        ax.contour(xs, ys, hz.water.astype(np.float32), levels=[0.5], colors=[BLUE], linewidths=0.8, zorder=2)
    if hz.ditch.any():  # trenches: the referee's GT ditch mask (cells excavated > 5 cm)
        ax.contourf(xs, ys, hz.ditch.astype(np.float32), levels=[0.5, 1.5], colors=[RED], zorder=3)
    for h in scn.get("hazards", []):
        if h["type"] == "crest":
            q = np.asarray(h["polyline"])
            ax.plot(q[:, 0], q[:, 1], color=AMBER, lw=1.6, ls=(0, (4, 2.5)), zorder=3)
    # static objects: dark if taller than ground clearance (lethal), light grey if drive-over
    for fp in world["feet"]:
        col = OBJ_LETHAL if fp.lethal else OBJ_SMALL
        if fp.kind == "circle":
            r = max(fp.r, OBJ_MIN_R_LETHAL if fp.lethal else OBJ_MIN_R_SMALL)
            ax.add_patch(Circle((fp.cx, fp.cy), r, facecolor=col, edgecolor="none", zorder=4))
        else:
            ax.add_patch(Polygon(rect_corners(fp), closed=True, facecolor=col, edgecolor="none", zorder=4))
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(FRAME)
        sp.set_linewidth(0.8)


def draw_path(ax, run: dict, colour: str, lw: float = 3.0, ticks_every: float | None = 10.0,
              tick_side: float = 1.0, tick_fs: float = 8.0, avoid: list | None = None,
              trench_dist=None) -> list[dict]:
    """GT path with a white casing; optional hollow dots every ``ticks_every`` seconds.

    Each tick label goes on the ``tick_side`` of the path (+1 = left of travel); it flips to the other
    side, or is dropped (the dot stays), when it would sit on an ``avoid`` point, an earlier label or
    the trench (``trench_dist(x, y)`` = distance to the GT trench, m).
    """
    avoid = list(avoid or [])
    x, y = run["x"], run["y"]
    ax.plot(x, y, color="white", lw=lw + 2.4, solid_capstyle="round", solid_joinstyle="round", zorder=6)
    ax.plot(x, y, color=colour, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=7)
    ticks, placed = [], []
    if ticks_every:
        tt = ticks_every
        while tt < run["time"] - 2.0:
            px, py = at_time(run, tt)
            ax.plot(px, py, "o", ms=4.6, mfc="white", mec=colour, mew=1.4, zorder=8)
            i = int(np.searchsorted(run["t"], tt))
            j0, j1 = max(i - 10, 0), min(i + 10, len(x) - 1)
            hx, hy = x[j1] - x[j0], y[j1] - y[j0]
            n = math.hypot(hx, hy) or 1.0
            labelled = False
            for side in (tick_side, -tick_side):
                lx, ly = px - 2.1 * hy / n * side, py + 2.1 * hx / n * side
                clear = (all(math.hypot(lx - a, ly - b) > 4.0 for a, b in placed)
                         and all(math.hypot(lx - a, ly - b) > 2.6 for a, b in avoid)
                         and (trench_dist is None or trench_dist(lx, ly) > 1.6))
                if clear:
                    ax.text(lx, ly, f"{tt:.0f} s", fontsize=tick_fs, color=MUTED, ha="center", va="center",
                            zorder=9, path_effects=HALO)
                    placed.append((lx, ly))
                    labelled = True
                    break
            ticks.append({"t_s": tt, "x_m": round(px, 2), "y_m": round(py, 2), "labelled": labelled})
            tt += ticks_every
    return ticks


def draw_ab(ax, world: dict, fs: float = 11.0, a_off=(-2.6, 0.0), b_off=(3.4, 0.0)) -> None:
    scn = world["scn"]
    sx, sy = scn["start"]["xy"]
    bx, by = scn["goal"]["xy"]
    r = float(scn["mission"]["success_radius_m"])
    ax.add_patch(Circle((bx, by), r, facecolor="none", edgecolor=INK, lw=1.0, ls=(0, (3, 2)), zorder=5))
    ax.plot(sx, sy, "o", ms=6.5, mfc="white", mec=INK, mew=1.5, zorder=10)
    ax.plot(bx, by, "o", ms=3.0, mfc=INK, mec=INK, zorder=10)
    for (px, py), (ox, oy), lab in (((sx, sy), a_off, "A"), ((bx, by), b_off, "B")):
        ax.text(px + ox, py + oy, lab, fontsize=fs, fontweight="bold", color=INK, ha="center", va="center",
                zorder=11, path_effects=HALO)


def end_marker(ax, run: dict) -> tuple[float, float]:
    ex, ey = float(run["x"][-1]), float(run["y"][-1])
    if run["outcome"] == "success":
        ax.plot(ex, ey, "o", ms=7.5, mfc=GREEN, mec="white", mew=1.4, zorder=10)
    else:
        ax.plot(ex, ey, marker="X", ms=11, mfc=RED, mec="white", mew=1.3, zorder=10)
    return ex, ey


def callout(ax, text: str, xy, xytext, fs: float = 9.5, colour: str = INK, ha: str = "left",
            weight: str = "normal", va: str = "center") -> None:
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fs, color=colour, ha=ha, va=va, fontweight=weight,
                zorder=12, path_effects=HALO, linespacing=1.2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9, shrinkA=3, shrinkB=4))


def scale_bar(ax, x0: float, y0: float, length: float = 10.0, fs: float = 8.5) -> None:
    """10 m bar (half black, half white) and a north arrow to its right."""
    h = 0.55
    for k in range(2):
        ax.add_patch(plt.Rectangle((x0 + k * length / 2, y0), length / 2, h,
                                   facecolor=(INK if k == 0 else "white"), edgecolor=INK, lw=0.8, zorder=11))
    for xx, lab in ((x0, "0"), (x0 + length, f"{length:.0f} m")):
        ax.text(xx, y0 + h + 0.8, lab, fontsize=fs, color=INK, ha="center", va="bottom", zorder=11, path_effects=HALO)
    nx_ = x0 + length + 4.2
    ax.add_patch(FancyArrowPatch((nx_, y0 - 0.2), (nx_, y0 + 3.4), arrowstyle="-|>,head_length=4,head_width=2.4",
                                 color=INK, lw=1.1, zorder=11))
    ax.text(nx_, y0 + 4.3, "N", fontsize=fs, color=INK, ha="center", va="bottom", fontweight="bold",
            zorder=11, path_effects=HALO)


def trench_distance_fn(world: dict):
    """Callable (x, y) -> distance (m) to the nearest GT trench cell (inf when there is no trench)."""
    from scipy.ndimage import distance_transform_edt

    hz, g = world["hz"], world["terrain"].grid
    if not hz.ditch.any():
        return lambda x, y: float("inf")
    d = distance_transform_edt(~hz.ditch) * g.res

    def f(x: float, y: float) -> float:
        i = int(np.clip(round((y - g.origin_y) / g.res), 0, g.ny - 1))
        j = int(np.clip(round((x - g.origin_x) / g.res), 0, g.nx - 1))
        return float(d[i, j])
    return f


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


def legend_row(fig, x: float, y: float, items: list, fs: float = 8.5) -> None:
    lab = {
        "trench": (Patch(facecolor=RED, edgecolor="none"), "trench (ground truth)"),
        "crest": (Line2D([], [], color=AMBER, lw=1.6, ls=(0, (4, 2.5))), "crest line"),
        "object": (Line2D([], [], marker="o", ls="none", ms=6, mfc=OBJ_LETHAL, mec="none"), "rock / tree / log"),
        "small": (Line2D([], [], marker="o", ls="none", ms=4.5, mfc=OBJ_SMALL, mec="none"), "drive-over rock"),
        "water": (Patch(facecolor=BLUE, alpha=0.55, edgecolor=BLUE), "water / mud"),
        "radius": (Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 2))), "2 m goal radius"),
        "ticks": (Line2D([], [], marker="o", ls="none", ms=4.6, mfc="white", mec=MUTED, mew=1.4), "every 10 s"),
        "light": (Line2D([], [], color=AMBER, lw=6, alpha=0.45, solid_capstyle="butt"), "lighting event active"),
    }
    hs = [lab[k] for k in items]
    fig.legend([h for h, _ in hs], [t for _, t in hs], loc="lower left", bbox_to_anchor=(x, y),
               ncol=len(hs), frameon=False, fontsize=fs, handlelength=1.5, columnspacing=1.1,
               handletextpad=0.45, borderaxespad=0.0, borderpad=0.0)


def add_map_axes(fig, x_in: float, y_in: float, W: float, H: float):
    return fig.add_axes([x_in / W, y_in / H, PANEL_W_IN / W, PANEL_H_IN / H])


def save(fig, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.10, facecolor=BG)
    plt.close(fig)


def run_record(run: dict) -> dict:
    return {
        "config": run["cfg"], "run_dir": run["dir"],
        "outcome": run["outcome"], "outcome_text": EVENT_TXT.get(run["outcome"], run["outcome"]),
        "end_time_s": run["time"], "end_event": dict(run["event"]),
        "path_length_m": run["path_length"], "mean_speed_mps": run["mean_speed"],
        "final_error_m": run["final_error"], "spl": run["spl"],
        "ditch_entries": run["ditch_entries"], "collisions": run["collisions"], "water_entries": run["water_entries"],
        "end_xy_m": [round(float(run["x"][-1]), 2), round(float(run["y"][-1]), 2)],
        "sources": [f"{run['dir']}/result.json", f"{run['dir']}/gt/events.json", f"{run['dir']}/gt/states.npz",
                    f"{SUMMARY} (row {run['cfg']},{run['seed']})"],
    }


# ---- layout constants (inches; figure saved at DPI=200 -> 9 in = 1800 px) --------------------
FIG_W_IN = 9.0
PANEL_GAP_IN = 0.16
PANEL_W_IN = (FIG_W_IN - PANEL_GAP_IN) / 2 - 0.02
PANEL_H_IN = PANEL_W_IN * 40.0 / 64.0     # world is 64 m x 40 m, drawn at equal scale
HEADER_IN = 0.60                          # two header lines above each panel
FOOT_IN = 0.52                            # legend row + provenance line
RELIEF_SMOOTH_M = 0.15
RELIEF_VERT_EXAG = 3.0
OBJ_MIN_R_LETHAL = 0.30                   # m, drawing floor so tree trunks stay visible
OBJ_MIN_R_SMALL = 0.18
CONTOUR = "#CFCFC9"
FRAME = "#BDBDBD"
TREND_V_DROP = 0.2                        # m/s: "slows to" only when the speed really dropped


# ---- figure: FULL vs TYPICAL pair ----------------------------------------------------------
PAIR_LAYOUT = {
    # hand-placed label positions (map metres, relative to the anchor unless noted) so that no label
    # covers a path, the trench or the goal. det_text / crest_text are absolute positions.
    38: {"full_end": (4.2, 6.8, "left"), "typ_end": (4.5, -9.5, "left"), "det_text": (2.5, 31.0),
         "gap": (3.5, 8.5), "crest_text": (30.0, 37.0), "trench_text": (36.2, 35.2), "scale": (3.0, 2.2),
         "tick_side": 1.0, "b_off": (3.4, 0.0)},
    7: {"full_end": (-3.5, 8.5, "right"), "typ_end": (-4.0, 3.0, "right"), "det_text": (2.5, 5.5),
        "gap": (3.2, -2.2), "crest_text": None, "trench_text": None, "scale": (3.0, 34.0),
        "tick_side": 1.0, "b_off": (0.0, 3.5)},
    37: {"full_end": (3.2, 7.5, "left"), "typ_end": (-6.5, -2.5, "right"), "det_text": (2.5, 33.0),
         "gap": None, "crest_text": None, "trench_text": None, "scale": (44.5, 2.2),
         "tick_side": -1.0, "b_off": (3.4, 0.0)},
}


def fig_pair(seed: int, name: str) -> dict:
    world = load_world(seed)
    full, typ = load_run("FULL", seed), load_run("TYPICAL", seed)
    assert full["outcome"] == "success", (seed, full["outcome"])
    fam = world["scn"]["family"]
    lay = PAIR_LAYOUT[seed]
    det = first_trench_confirmation(full, world)

    W, H = FIG_W_IN, FOOT_IN + PANEL_H_IN + HEADER_IN
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    axs = [add_map_axes(fig, 0.0, FOOT_IN, W, H), add_map_axes(fig, PANEL_W_IN + PANEL_GAP_IN, FOOT_IN, W, H)]
    records = {}
    tdist = trench_distance_fn(world)
    for ax, run, colour in ((axs[0], full, NAVY), (axs[1], typ, GREY)):
        draw_world(ax, world)
        avoid = [(float(run["x"][-1]), float(run["y"][-1])), tuple(world["scn"]["start"]["xy"])]
        if run["cfg"] == "FULL" and det is not None:
            avoid.append((det["x"], det["y"]))
        ticks = draw_path(ax, run, colour, lw=3.0, tick_side=lay["tick_side"], avoid=avoid, trench_dist=tdist)
        draw_ab(ax, world, b_off=lay["b_off"])
        ex, ey = end_marker(ax, run)
        dx, dy, ha = lay["full_end" if run["cfg"] == "FULL" else "typ_end"]
        ok = run["outcome"] == "success"
        callout(ax, f"{EVENT_TXT[run['outcome']]}\nat {run['time']:.1f} s", (ex, ey), (ex + dx, ey + dy),
                fs=9.5, ha=ha, weight="bold", colour=(INK if ok else RED))
        scale_bar(ax, *lay["scale"])
        cfg_name, cfg_desc = CONFIG_TXT[run["cfg"]]
        bb = ax.get_position()
        fig.text(bb.x0 + 0.003, bb.y1 + 0.33 / H, cfg_name, fontsize=12, fontweight="bold",
                 color=(NAVY if run["cfg"] == "FULL" else GREY_TXT), ha="left", va="bottom")
        fig.text(bb.x0 + 0.003, bb.y1 + 0.07 / H, cfg_desc, fontsize=9.5, color=MUTED, ha="left", va="bottom")
        rec = run_record(run)
        rec["time_ticks"] = ticks
        records[run["cfg"]] = rec

    # the gap FULL drove through (nearest gap to FULL's path), labelled on the FULL panel
    gaps = gap_midpoints(world)
    gap_used = None
    if gaps:
        dmin = [float(np.min(np.hypot(full["x"] - gx, full["y"] - gy))) for gx, gy in gaps]
        gap_used = gaps[int(np.argmin(dmin))]
        if lay.get("gap"):
            gx, gy = gap_used
            callout(axs[0], "gap in the trench", (gx, gy), (gx + lay["gap"][0], gy + lay["gap"][1]), fs=9)
    crest = [hh for hh in world["scn"]["hazards"] if hh["type"] == "crest"]
    ditch = [hh for hh in world["scn"]["hazards"] if hh["type"] == "ditch"]
    if lay.get("crest_text") and crest:
        axs[1].text(*lay["crest_text"], f"crest, {crest[0]['drop']:.1f} m drop", fontsize=9, color=INK, ha="right",
                    va="center", zorder=12, path_effects=HALO)
    if lay.get("trench_text") and ditch:
        axs[1].text(*lay["trench_text"], f"trench behind it,\n{ditch[0]['depth']:.2f} m deep", fontsize=9,
                    color=INK, ha="left", va="center", zorder=12, path_effects=HALO, linespacing=1.2)
    if det is not None and lay.get("det_text"):
        axs[0].plot(det["x"], det["y"], "o", ms=6.5, mfc=AMBER, mec="white", mew=1.2, zorder=10)
        v0, v1 = det["v_at_mps"], det["v_min_next3s_mps"]
        speed = (f"slows to {v1:.1f} m/s" if v1 < v0 - TREND_V_DROP else f"holds {v0:.1f} m/s")
        callout(axs[0], f"{det['t']:.1f} s: trench confirmed\n{det['dist_to_nearest_confirmed_cell_m']:.1f} m ahead, "
                        f"{speed}", (det["x"], det["y"]), lay["det_text"], fs=9, ha="left")

    items = ["trench"] + (["crest"] if crest else []) + ["object", "small"] + \
            (["water"] if world["hz"].water.any() else []) + ["radius", "ticks"]
    legend_row(fig, 0.004, 0.21 / H, items)
    fam_txt = FAMILY_TXT[fam].replace("  ", " ")
    fig.text(0.004, 0.02 / H, f"Simulated · tier-0 synthetic depth sensor, no images · held-out EVAL seed {seed} "
                              f"({fam_txt}) · {RUNS}/{{FULL,TYPICAL}}/{seed:03d}",
             fontsize=8, color=FAINT, ha="left", va="bottom")
    save(fig, name)

    side = {
        "figure": name, "label": "Simulated (tier-0 synthetic depth sensor, no images; held-out EVAL seed)",
        "generator": "deck_assets/final/src/runs_map.py",
        "seed": seed, "family": fam, "split": world["scn"]["split"],
        "scenario": f"{SCEN}/{seed}.json", "scenario_sha256": world["scn"]["sha256"],
        "world_m": {"x": [0.0, 64.0], "y": [0.0, 40.0], "res_m": world["terrain"].grid.res},
        "start_xy_m": world["scn"]["start"]["xy"], "goal_xy_m": world["scn"]["goal"]["xy"],
        "success_radius_m": world["scn"]["mission"]["success_radius_m"],
        "hazards": hazard_summary(world),
        "gap_used_by_full_xy_m": gap_used,
        "runs": records,
        "full_trench_confirmation": det,
        "numbers_drawn": {
            "FULL end": f"{EVENT_TXT[full['outcome']]} at {full['time']:.1f} s (result.json time, summary.csv)",
            "TYPICAL end": f"{EVENT_TXT[typ['outcome']]} at {typ['time']:.1f} s (result.json time, gt/events.json)",
            "crest drop / trench depth": "scenario hazards[].drop / hazards[].depth" if crest else None,
            "trench confirmed": ("full_trench_confirmation.t, .dist_to_nearest_confirmed_cell_m, .v_at_mps, "
                                 ".v_min_next3s_mps" if det else None),
            "time ticks": "GT position at t = 10, 20, ... s (gt/states.npz)",
        },
        "selection": ("F2/F3 EVAL seeds where FULL reached B and TYPICAL failed (summary.csv): 2 (F3, TYPICAL "
                      "out_of_bounds), 7 (F2, ditch_entry), 37 (F2, out_of_bounds), 38 (F3, ditch_entry). Seed 38 is "
                      "the headline pair: the trench is hidden behind the crest, the 'unknown is never free' case; "
                      "7 and 37 are backups."),
        "how_drawn": {
            "background": f"hillshade (az 315, alt 40, vert. exag. {RELIEF_VERT_EXAG}) of the scenario heightmap "
                          f"smoothed by {RELIEF_SMOOTH_M} m, tinted by material; 0.25 m contours of the landform "
                          "(heightmap + trench excavation, smoothed)",
            "trench": "GT ditch mask from metagross.sim.hazards.gt_hazard_raster (cells excavated > 0.05 m)",
            "crest": "scenario hazards[type=crest].polyline",
            "objects": "static footprints from metagross.sim.objects.parse_static_objects; dark = protrudes above "
                       f"ground clearance (lethal), light = drive-over; true radius with a drawing floor of "
                       f"{OBJ_MIN_R_LETHAL}/{OBJ_MIN_R_SMALL} m",
            "paths": "GT body-origin x, y from gt/states.npz (0.04 s physics steps)",
            "end markers": "last GT state; green dot = reached B, red X = referee terminal event",
        },
    }
    (OUT / f"{name}.json").write_text(json.dumps(side, indent=1))
    return side


# ---- figure: gallery of FULL successes -----------------------------------------------------
GALLERY = [
    # (seed, family, scale bar position or None, lighting-label offsets by event type, B-letter offset)
    (24, "F1_trail", (3.0, 2.2), {}, (0.0, 3.5)),
    (37, "F2_ditch_field", None, {}, (0.0, 3.5)),
    (2, "F3_crest_ditch", None, {}, (0.0, 3.5)),
    (4, "F5_lighting", None, {"glare": (0.0, 6.5), "dim": (-9.0, -7.0)}, (3.4, 0.0)),
]
GALLERY_NOTE = {
    "F1_trail": "rocks on the trail, trees beside it",
    "F2_ditch_field": "three trenches, one gap each",
    "F3_crest_ditch": "trench hidden behind a crest",
    "F5_lighting": "low sun ({elev:.0f}°), glare, dimming",
}
EVENT_WORD = {"glare": "glare", "dim": "dimming", "dust": "dust"}


def fig_gallery(name: str) -> dict:
    row_gap = 0.14
    W = FIG_W_IN
    H = FOOT_IN + 2 * (PANEL_H_IN + HEADER_IN) + row_gap
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    rows_y = [FOOT_IN + PANEL_H_IN + HEADER_IN + row_gap, FOOT_IN]
    panels = []
    any_light = False
    for k, (seed, fam, sb, light_off, b_off) in enumerate(GALLERY):
        world = load_world(seed)
        assert world["scn"]["family"] == fam, (seed, fam)
        run = load_run("FULL", seed)
        assert run["outcome"] == "success", seed
        typ = load_run("TYPICAL", seed)
        ax = add_map_axes(fig, (k % 2) * (PANEL_W_IN + PANEL_GAP_IN), rows_y[k // 2], W, H)
        draw_world(ax, world)
        light_marks = []
        for ev in world["scn"]["lighting"]["events"]:
            t0, t1 = float(ev["t0"]), float(ev["t1"])
            if t0 >= run["time"]:
                continue  # event starts after arrival
            m = (run["t"] >= t0) & (run["t"] <= min(t1, run["time"]))
            ax.plot(run["x"][m], run["y"][m], color=AMBER, lw=9.0, alpha=0.45, solid_capstyle="round", zorder=5)
            any_light = True
            mx, my = at_time(run, 0.5 * (t0 + min(t1, run["time"])))
            ox, oy = light_off.get(ev["type"], (0.0, 6.5))
            callout(ax, f"{EVENT_WORD[ev['type']]} {t0:.1f}-{t1:.1f} s", (mx, my), (mx + ox, my + oy), fs=9,
                    ha="center")
            light_marks.append({"type": ev["type"], "t0_s": t0, "t1_s": t1, "gain": ev["gain"],
                                "drawn_until_s": min(t1, run["time"]), "label_anchor_xy_m": [round(mx, 2), round(my, 2)]})
        draw_path(ax, run, NAVY, lw=2.6, ticks_every=None)
        draw_ab(ax, world, fs=10.5, b_off=b_off)
        end_marker(ax, run)
        if sb:
            scale_bar(ax, *sb)
        bb = ax.get_position()
        note = GALLERY_NOTE[fam].format(elev=world["scn"]["lighting"]["sun_elev_deg"])
        fig.text(bb.x0 + 0.003, bb.y1 + 0.31 / H, FAMILY_TXT[fam], fontsize=11.5, fontweight="bold", color=INK,
                 ha="left", va="bottom")
        fig.text(bb.x0 + 0.003, bb.y1 + 0.07 / H, f"seed {seed} · {note} · reached B at {run['time']:.1f} s",
                 fontsize=9.2, color=MUTED, ha="left", va="bottom")
        rec = run_record(run)
        rec.update({"seed": seed, "family": fam, "scenario": f"{SCEN}/{seed}.json",
                    "scenario_sha256": world["scn"]["sha256"], "hazards": hazard_summary(world),
                    "typical_same_seed": {"outcome": typ["outcome"], "end_time_s": typ["time"]},
                    "lighting": {k2: world["scn"]["lighting"][k2] for k2 in ("sun_elev_deg", "fog_density", "events")},
                    "lighting_events_drawn": light_marks, "subtitle": note})
        panels.append(rec)
    items = ["trench", "crest", "object", "small", "radius"] + (["light"] if any_light else [])
    legend_row(fig, 0.004, 0.21 / H, items)
    fig.text(0.004, 0.02 / H, "Simulated · tier-0 synthetic depth, no images (lighting = depth dropout and noise) · "
                              f"FULL, held-out EVAL seeds {', '.join(str(g[0]) for g in GALLERY)} · {RUNS}/FULL",
             fontsize=8, color=FAINT, ha="left", va="bottom")
    save(fig, name)
    side = {
        "figure": name, "label": "Simulated (tier-0 synthetic depth sensor, no images; held-out EVAL seeds)",
        "generator": "deck_assets/final/src/runs_map.py",
        "panels": panels,
        "numbers_drawn": "per panel: seed, sun elevation (scenario lighting.sun_elev_deg), lighting event windows "
                         "(scenario lighting.events t0/t1), reached-B time (result.json time = summary.csv time)",
        "selection": ("One FULL success per family for F1, F2, F3, F5. F1 seed 24 weaves through rocks on the trail; "
                      "F2 seed 37 has three trenches; F3 seed 2 is the only F3 trench seed FULL completed besides 38 "
                      "(used in run_map_pair); F5 seed 4 has glare and dimming inside the run window."),
        "note_lighting": ("In tier-0 there are no images: lighting events act on the synthetic depth sensor as "
                          "dropout, noise and dust phantom returns (metagross/sim/sensors.py)."),
        "how_drawn": "same basemap as run_map_pair; amber bands = stretches of path while a lighting event was active",
    }
    (OUT / f"{name}.json").write_text(json.dumps(side, indent=1))
    return side


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_pair(38, "run_map_pair")
    fig_pair(7, "run_map_pair_s007")
    fig_pair(37, "run_map_pair_s037")
    fig_gallery("run_map_gallery")
    print("wrote", ", ".join(f"{n}.png/.svg/.json" for n in
                              ("run_map_pair", "run_map_pair_s007", "run_map_pair_s037", "run_map_gallery")))
