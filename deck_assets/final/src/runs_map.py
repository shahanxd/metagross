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
    1: "#EEE8DC",  # dirt
    2: "#E6DCC6",  # gravel trail
    3: "#E9E9E6",  # rocky ground
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


def first_trench_confirmation(run: dict, world: dict) -> dict | None:
    """First tick at which FULL's rolling map confirmed new trench cells (its own debug log).

    Also reports the GT distance from the vehicle to the nearest GT trench cell at that moment and
    the lowest GT speed over the following 3 s.
    """
    files = sorted(glob.glob(str(ROOT / run["dir"] / "autonomy" / "debug" / "tick_*.npz")))
    for f in files:
        b = np.load(f)
        ex = json.loads(str(b["extras_json"]))
        if ex.get("map_confirmed_ditch", 0) > 0:
            t = float(b["t"])
            x = float(np.interp(t, run["t"], run["x"]))
            y = float(np.interp(t, run["t"], run["y"]))
            g = world["terrain"].grid
            ii, jj = np.nonzero(world["hz"].ditch)
            dx = g.origin_x + jj * g.res - x
            dy = g.origin_y + ii * g.res - y
            dist = float(np.sqrt(dx * dx + dy * dy).min())
            win = (run["t"] >= t) & (run["t"] <= t + 3.0)
            v_min = float(run["v"][win].min())
            return {"t": t, "x": x, "y": y, "n_cells": int(ex["map_confirmed_ditch"]),
                    "dist_to_trench_m": dist, "v_min_next3s": v_min,
                    "v_at": float(np.interp(t, run["t"], run["v"])),
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
    t = world["terrain"]
    h = t.height.astype(np.float64)
    ls = LightSource(azdeg=315, altdeg=40)
    # LightSource assumes row 0 at the top (north); our row 0 is south, so flip in and out.
    shade = np.flipud(ls.hillshade(np.flipud(h), vert_exag=4.0, dx=t.grid.res, dy=t.grid.res))
    lut = np.array([to_rgb(MAT_RGB.get(k, MAT_RGB[0])) for k in range(256)])
    base = lut[t.material]
    rgb = base * (0.80 + 0.26 * shade[..., None])
    rgb = np.clip(rgb, 0, 1)
    k = int(round(BG_RES_M / t.grid.res))
    ny, nx = (rgb.shape[0] // k) * k, (rgb.shape[1] // k) * k
    return rgb[:ny, :nx].reshape(ny // k, k, nx // k, k, 3).mean(axis=(1, 3))


def draw_world(ax, world: dict, contours: bool = True, crest_label: bool = False, detail: float = 1.0) -> None:
    t, hz, scn = world["terrain"], world["hz"], world["scn"]
    g = t.grid
    ext = extent(world)
    ax.imshow(relief_rgb(world), origin="lower", extent=ext, interpolation="bilinear", zorder=0)
    xs = g.origin_x + np.arange(g.nx) * g.res
    ys = g.origin_y + np.arange(g.ny) * g.res
    if contours:
        # 0.25 m contours of the landform (trench excavation added back so the trench is not contoured twice)
        land = t.height + hz.ditch_drop
        from scipy.ndimage import gaussian_filter
        land = gaussian_filter(land[::2, ::2], 2.0)
        lv = np.arange(math.floor(land.min() * 4) / 4, land.max() + 0.25, 0.25)
        ax.contour(xs[::2], ys[::2], land, levels=lv, colors="#C9C9C4", linewidths=0.45 * detail, zorder=1)
    # water / mud (flat, semantically hazardous)
    if hz.water.any():
        ax.contourf(xs, ys, hz.water.astype(np.float32), levels=[0.5, 1.5], colors=[BLUE], alpha=0.55, zorder=2)
        ax.contour(xs, ys, hz.water.astype(np.float32), levels=[0.5], colors=[BLUE], linewidths=0.8, zorder=2)
    # trenches: the referee's GT ditch mask (cells excavated > 5 cm)
    if hz.ditch.any():
        ax.contourf(xs, ys, hz.ditch.astype(np.float32), levels=[0.5, 1.5], colors=[RED], zorder=3)
    # crest lines
    for h in scn.get("hazards", []):
        if h["type"] == "crest":
            p = np.asarray(h["polyline"])
            ax.plot(p[:, 0], p[:, 1], color=AMBER, lw=1.6 * detail, ls=(0, (4, 2.5)), zorder=3, solid_capstyle="butt")
    # static objects: dark if taller than ground clearance (lethal), light grey if drive-over
    for fp in world["feet"]:
        col = OBJ_LETHAL if fp.lethal else OBJ_SMALL
        if fp.kind == "circle":
            r = max(fp.r, 0.30 if fp.lethal else 0.18)
            ax.add_patch(Circle((fp.cx, fp.cy), r, facecolor=col, edgecolor="none", zorder=4))
        else:
            ax.add_patch(Polygon(rect_corners(fp), closed=True, facecolor=col, edgecolor="none", zorder=4))
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#BDBDBD")
        s.set_linewidth(0.8)


def draw_path(ax, run: dict, colour: str, lw: float = 3.0, ticks_every: float | None = 10.0,
              tick_side: float = 1.0, tick_fs: float = 8.0) -> list[dict]:
    x, y = run["x"], run["y"]
    ax.plot(x, y, color="white", lw=lw + 2.4, solid_capstyle="round", solid_joinstyle="round", zorder=6)
    ax.plot(x, y, color=colour, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=7)
    ticks = []
    if ticks_every:
        tt = ticks_every
        while tt < run["time"] - 2.0:
            px, py = at_time(run, tt)
            ax.plot(px, py, "o", ms=4.6, mfc="white", mec=colour, mew=1.4, zorder=8)
            # label offset perpendicular to the local heading
            i = int(np.searchsorted(run["t"], tt))
            j0, j1 = max(i - 10, 0), min(i + 10, len(x) - 1)
            hx, hy = x[j1] - x[j0], y[j1] - y[j0]
            n = math.hypot(hx, hy) or 1.0
            nx_, ny_ = -hy / n * tick_side, hx / n * tick_side
            ax.text(px + 1.9 * nx_, py + 1.9 * ny_, f"{tt:.0f} s", fontsize=tick_fs, color=MUTED,
                    ha="center", va="center", zorder=9, path_effects=HALO)
            ticks.append({"t_s": tt, "x_m": round(px, 2), "y_m": round(py, 2)})
            tt += ticks_every
    return ticks


def draw_ab(ax, world: dict, fs: float = 11.0, a_off=(-2.6, 0.0), b_off=(0.0, 3.3), show_radius: bool = True) -> None:
    scn = world["scn"]
    ax_, ay_ = scn["start"]["xy"]
    bx, by = scn["goal"]["xy"]
    r = float(scn["mission"]["success_radius_m"])
    if show_radius:
        ax.add_patch(Circle((bx, by), r, facecolor="none", edgecolor=INK, lw=1.0, ls=(0, (3, 2)), zorder=5))
    ax.plot(ax_, ay_, "o", ms=6.5, mfc="white", mec=INK, mew=1.5, zorder=10)
    ax.plot(bx, by, "o", ms=3.0, mfc=INK, mec=INK, zorder=10)
    for (px, py), (ox, oy), lab in (((ax_, ay_), a_off, "A"), ((bx, by), b_off, "B")):
        ax.text(px + ox, py + oy, lab, fontsize=fs, fontweight="bold", color=INK, ha="center", va="center",
                zorder=11, path_effects=HALO)


def end_marker(ax, run: dict, success: bool, colour: str) -> tuple[float, float]:
    ex, ey = float(run["x"][-1]), float(run["y"][-1])
    if success:
        ax.plot(ex, ey, "o", ms=7.5, mfc=GREEN, mec="white", mew=1.4, zorder=10)
    else:
        ax.plot(ex, ey, marker="X", ms=11, mfc=RED, mec="white", mew=1.3, zorder=10)
    return ex, ey


def callout(ax, text: str, xy: tuple[float, float], xytext: tuple[float, float], fs: float = 9.5,
            colour: str = INK, ha: str = "left", weight: str = "normal", arrow: bool = True) -> None:
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fs, color=colour, ha=ha, va="center", fontweight=weight,
                zorder=12, path_effects=HALO, linespacing=1.15,
                arrowprops=(dict(arrowstyle="-", color=MUTED, lw=0.9, shrinkA=2, shrinkB=4) if arrow else None))


def scale_bar(ax, x0: float, y0: float, length: float = 10.0, fs: float = 8.5, north: bool = True) -> None:
    h = 0.55
    for k in range(2):
        ax.add_patch(plt.Rectangle((x0 + k * length / 2, y0), length / 2, h,
                                   facecolor=(INK if k == 0 else "white"), edgecolor=INK, lw=0.8, zorder=11))
    ax.text(x0, y0 + h + 0.8, "0", fontsize=fs, color=INK, ha="center", va="bottom", zorder=11, path_effects=HALO)
    ax.text(x0 + length, y0 + h + 0.8, f"{length:.0f} m", fontsize=fs, color=INK, ha="center", va="bottom",
            zorder=11, path_effects=HALO)
    if north:
        nx_ = x0 + length + 4.0
        ax.add_patch(FancyArrowPatch((nx_, y0 - 0.2), (nx_, y0 + 3.4), arrowstyle="-|>,head_length=4,head_width=2.4",
                                     color=INK, lw=1.1, zorder=11))
        ax.text(nx_, y0 + 4.4, "N", fontsize=fs, color=INK, ha="center", va="bottom", fontweight="bold",
                zorder=11, path_effects=HALO)


def gap_midpoints(world: dict) -> list[tuple[float, float]]:
    out = []
    for h in world["scn"].get("hazards", []):
        if h["type"] != "ditch":
            continue
        p = np.asarray(h["polyline"], float)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
        for s0, s1 in h.get("gaps", []):
            sm = 0.5 * (s0 + s1)
            out.append((float(np.interp(sm, s, p[:, 0])), float(np.interp(sm, s, p[:, 1]))))
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


def legend_row(fig, x: float, y: float, fs: float = 8.5, items: list | None = None) -> None:
    items = items or ["trench", "crest", "object", "small", "water", "radius"]
    handles = []
    lab = {
        "trench": (Patch(facecolor=RED, edgecolor="none"), "trench (GT)"),
        "crest": (Line2D([], [], color=AMBER, lw=1.6, ls=(0, (4, 2.5))), "crest line"),
        "object": (Line2D([], [], marker="o", ls="none", ms=6, mfc=OBJ_LETHAL, mec="none"), "rock / tree / log"),
        "small": (Line2D([], [], marker="o", ls="none", ms=4.5, mfc=OBJ_SMALL, mec="none"), "drive-over rock"),
        "water": (Patch(facecolor=BLUE, alpha=0.55, edgecolor=BLUE), "water / mud"),
        "radius": (Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 2))), "2 m success radius"),
        "ticks": (Line2D([], [], marker="o", ls="none", ms=4.6, mfc="white", mec=MUTED, mew=1.4), "every 10 s"),
    }
    for k in items:
        handles.append(lab[k])
    fig.legend([h for h, _ in handles], [t for _, t in handles], loc="lower right", bbox_to_anchor=(x, y),
               ncol=len(handles), frameon=False, fontsize=fs, handlelength=1.6, columnspacing=1.2,
               handletextpad=0.5, borderaxespad=0.0)


def save(fig, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.10, facecolor=BG)
    plt.close(fig)


def run_record(run: dict) -> dict:
    ev = dict(run["event"])
    return {
        "config": run["cfg"], "run_dir": run["dir"],
        "outcome": run["outcome"], "outcome_text": EVENT_TXT.get(run["outcome"], run["outcome"]),
        "end_time_s": run["time"], "end_event": ev,
        "path_length_m": run["path_length"], "mean_speed_mps": run["mean_speed"],
        "final_error_m": run["final_error"], "spl": run["spl"],
        "ditch_entries": run["ditch_entries"], "collisions": run["collisions"], "water_entries": run["water_entries"],
        "end_xy_m": [round(float(run["x"][-1]), 2), round(float(run["y"][-1]), 2)],
        "sources": [f"{run['dir']}/result.json", f"{run['dir']}/gt/events.json", f"{run['dir']}/gt/states.npz",
                    f"{SUMMARY} (row {run['cfg']},{run['seed']})"],
    }


# ---- figure: FULL vs TYPICAL pair ----------------------------------------------------------
PAIR_LAYOUT = {
    # per-seed label placement (map metres). Hand-placed so nothing covers the paths or the trench.
    38: {"full_end": (-9.0, 7.5, "left"), "typ_end": (-2.0, -9.5, "left"), "det": (-2.0, -8.0),
         "gap": (4.0, 7.0), "crest": (14.2, 36.0), "scale": (3.0, 2.2), "tick_side": 1.0,
         "a_off": (-2.6, 0.0), "b_off": (1.2, 3.4)},
    7: {"full_end": (-4.0, 8.5, "right"), "typ_end": (6.0, -3.0, "left"), "det": (-11.0, -7.0),
        "gap": (-10.0, -2.5), "crest": None, "scale": (3.0, 2.2), "tick_side": 1.0,
        "a_off": (-2.6, 0.0), "b_off": (0.0, 3.4)},
    37: {"full_end": (-2.0, 7.0, "right"), "typ_end": (4.0, 1.5, "left"), "det": (-14.0, -4.5),
         "gap": None, "crest": None, "scale": (48.0, 2.2), "tick_side": -1.0,
         "a_off": (-2.6, 0.0), "b_off": (1.2, 3.4)},
}


def fig_pair(seed: int, name: str) -> dict:
    world = load_world(seed)
    full, typ = load_run("FULL", seed), load_run("TYPICAL", seed)
    fam = world["scn"]["family"]
    L = PAIR_LAYOUT[seed]
    det = first_trench_confirmation(full, world)

    fig = plt.figure(figsize=(9.0, 3.95), dpi=DPI)
    w, gap_x, left = 0.487, 0.018, 0.004
    h = w * 9.0 / 3.95 / 1.6
    axs = [fig.add_axes([left, 0.115, w, h]), fig.add_axes([left + w + gap_x, 0.115, w, h])]
    records = {}
    for ax, run, colour in ((axs[0], full, NAVY), (axs[1], typ, GREY)):
        draw_world(ax, world)
        ticks = draw_path(ax, run, colour, lw=3.0, tick_side=L["tick_side"])
        draw_ab(ax, world, a_off=L["a_off"], b_off=L["b_off"])
        ok = run["outcome"] == "success"
        ex, ey = end_marker(ax, run, ok, colour)
        key = "full_end" if run["cfg"] == "FULL" else "typ_end"
        dx, dy, ha = L[key]
        txt = f"{EVENT_TXT[run['outcome']]}\nat {run['time']:.1f} s"
        callout(ax, txt, (ex, ey), (ex + dx, ey + dy), fs=9.5, ha=ha, weight="bold",
                colour=(INK if ok else RED))
        sx, sy = L["scale"]
        scale_bar(ax, sx, sy)
        cfg_name, cfg_desc = CONFIG_TXT[run["cfg"]]
        bb = ax.get_position()
        fig.text(bb.x0 + 0.004, bb.y1 + 0.075, cfg_name, fontsize=12, fontweight="bold",
                 color=(NAVY if run["cfg"] == "FULL" else GREY_TXT), ha="left", va="bottom")
        fig.text(bb.x0 + 0.004, bb.y1 + 0.018, cfg_desc, fontsize=9.5, color=MUTED, ha="left", va="bottom")
        rec = run_record(run)
        rec["time_ticks"] = ticks
        records[run["cfg"]] = rec

    # the trench gap FULL used, and the crest, labelled once (left panel)
    gaps = gap_midpoints(world)
    if L.get("gap") and gaps:
        gx, gy = min(gaps, key=lambda p: math.hypot(p[0] - full["x"][np.argmin(np.abs(full["t"] - 25))],
                                                     p[1] - full["y"][np.argmin(np.abs(full["t"] - 25))]))
        callout(axs[0], "gap in the trench", (gx, gy), (gx + L["gap"][0], gy + L["gap"][1]), fs=9)
    crest = [hh for hh in world["scn"]["hazards"] if hh["type"] == "crest"]
    ditch = [hh for hh in world["scn"]["hazards"] if hh["type"] == "ditch"]
    if L.get("crest") and crest:
        cx, cy = L["crest"]
        axs[1].text(cx, cy, f"crest, {crest[0]['drop']:.1f} m drop", fontsize=9, color=INK, ha="right",
                    va="center", zorder=12, path_effects=HALO)
        axs[1].text(cx + 3.6, cy - 3.3, f"trench behind it\n{ditch[0]['depth']:.2f} m deep",
                    fontsize=9, color=RED, ha="left", va="center", zorder=12, path_effects=HALO)
    if det is not None and L.get("det"):
        axs[0].plot(det["x"], det["y"], "o", ms=6.5, mfc=AMBER, mec="white", mew=1.2, zorder=10)
        callout(axs[0], f"{det['t']:.1f} s: trench confirmed\n{det['dist_to_trench_m']:.1f} m ahead, slows to "
                        f"{det['v_min_next3s']:.1f} m/s",
                (det["x"], det["y"]), (det["x"] + L["det"][0], det["y"] + L["det"][1]), fs=9, ha="left")

    items = ["trench"] + (["crest"] if crest else []) + ["object", "small"] + \
            (["water"] if world["hz"].water.any() else []) + ["radius", "ticks"]
    legend_row(fig, 0.995, 0.012, items=items)
    fam_txt = FAMILY_TXT[fam].replace("  ", " ")
    fig.text(0.006, 0.012, f"Simulated · tier-0 synthetic depth sensor, no images · held-out EVAL seed {seed} "
                           f"({fam_txt}) · {RUNS}/{{FULL,TYPICAL}}/{seed:03d}",
             fontsize=8, color=FAINT, ha="left", va="bottom")
    save(fig, name)

    side = {
        "figure": name, "label": "Simulated (tier-0 synthetic depth sensor, no images; held-out EVAL seed)",
        "seed": seed, "family": fam, "split": world["scn"]["split"],
        "scenario": f"{SCEN}/{seed}.json", "scenario_sha256": world["scn"]["sha256"],
        "world_m": {"x": [0.0, 64.0], "y": [0.0, 40.0], "res_m": world["terrain"].grid.res},
        "start_xy_m": world["scn"]["start"]["xy"], "goal_xy_m": world["scn"]["goal"]["xy"],
        "success_radius_m": world["scn"]["mission"]["success_radius_m"],
        "hazards": hazard_summary(world),
        "runs": records,
        "full_first_trench_confirmation": det,
        "selection": ("F2/F3 EVAL seeds where FULL reached B and TYPICAL failed (from summary.csv): "
                      "2 (F3, TYPICAL out_of_bounds), 7 (F2, ditch_entry), 37 (F2, out_of_bounds), "
                      "38 (F3, ditch_entry). Seed 38 is the headline pair: the trench is hidden behind the "
                      "crest, which is exactly the 'unknown is never free' case."),
        "how_drawn": {
            "background": "hillshade of the scenario heightmap (az 315, alt 40, vert. exag. 4) tinted by material, "
                          "0.25 m contours of the landform (heightmap + trench excavation, lightly smoothed)",
            "trench": "GT ditch mask from metagross.sim.hazards.gt_hazard_raster (cells excavated > 0.05 m)",
            "objects": "static footprints from metagross.sim.objects.parse_static_objects; dark = protrudes above "
                       "ground clearance (lethal), light = drive-over; drawn at true radius, min 0.30/0.18 m for visibility",
            "paths": "GT body-origin x, y from gt/states.npz (0.04 s physics steps)",
            "time_ticks": "GT position every 10 s",
            "trench_confirmation": "first debug tick with extras_json.map_confirmed_ditch > 0 (newly confirmed trench "
                                   "cells in the rolling map); distance = GT vehicle position to nearest GT ditch cell; "
                                   "speed = min GT speed over the next 3 s",
        },
    }
    (OUT / f"{name}.json").write_text(json.dumps(side, indent=1))
    return side


# ---- figure: gallery of FULL successes -----------------------------------------------------
GALLERY = [
    # (seed, family, label placement for the end callout (dx, dy, ha), scale-bar corner)
    (24, "F1_trail", (-3.0, 7.0, "right"), (3.0, 2.2)),
    (37, "F2_ditch_field", (-2.0, 7.0, "right"), (48.0, 2.2)),
    (2, "F3_crest_ditch", (-3.0, 7.0, "right"), (3.0, 2.2)),
    (4, "F5_lighting", (-2.0, -8.5, "right"), (3.0, 29.5)),
]
GALLERY_NOTE = {
    "F1_trail": "rocks on the trail, trees beside it",
    "F2_ditch_field": "three trenches, one gap each",
    "F3_crest_ditch": "trench hidden behind a crest",
    "F5_lighting": "sun {elev:.0f}° above horizon, glare + dimming",
}


def fig_gallery(name: str) -> dict:
    fig = plt.figure(figsize=(9.0, 6.55), dpi=DPI)
    w, gx, left = 0.487, 0.018, 0.004
    h = w * 9.0 / 6.55 / 1.6
    rows_y = [0.555, 0.085]
    panels = []
    for k, (seed, fam, (dx, dy, ha), (sx, sy)) in enumerate(GALLERY):
        world = load_world(seed)
        assert world["scn"]["family"] == fam, (seed, fam)
        run = load_run("FULL", seed)
        assert run["outcome"] == "success", seed
        ax = fig.add_axes([left + (k % 2) * (w + gx), rows_y[k // 2], w, h])
        draw_world(ax, world)
        light_marks = []
        if fam == "F5_lighting":
            # lighting events active during the run: highlight those stretches of path in amber
            for ev in world["scn"]["lighting"]["events"]:
                t0, t1 = float(ev["t0"]), min(float(ev["t1"]), run["time"])
                if t0 >= run["time"]:
                    continue
                m = (run["t"] >= t0) & (run["t"] <= t1)
                ax.plot(run["x"][m], run["y"][m], color=AMBER, lw=9.0, alpha=0.45, solid_capstyle="round", zorder=5)
                light_marks.append({"type": ev["type"], "t0_s": round(t0, 2), "t1_s": round(t1, 2),
                                    "gain": ev["gain"], "xy_mid_m": at_time(run, 0.5 * (t0 + t1))})
        ticks = draw_path(ax, run, NAVY, lw=2.6, ticks_every=None)
        draw_ab(ax, world, fs=10.5)
        ex, ey = end_marker(ax, run, True, NAVY)
        callout(ax, f"Reached B at {run['time']:.1f} s", (ex, ey), (ex + dx, ey + dy), fs=9.5, ha=ha, weight="bold")
        for lm in light_marks:
            mx, my = lm["xy_mid_m"]
            word = {"glare": "glare", "dim": "dimming", "dust": "dust"}[lm["type"]]
            callout(ax, f"{word} {lm['t0_s']:.0f}-{lm['t1_s']:.0f} s", (mx, my), (mx + 1.0, my + 6.5), fs=9, ha="center")
        if k == 0:
            scale_bar(ax, sx, sy)
        bb = ax.get_position()
        note = GALLERY_NOTE[fam].format(elev=world["scn"]["lighting"]["sun_elev_deg"])
        fig.text(bb.x0 + 0.004, bb.y1 + 0.045, f"{FAMILY_TXT[fam]}", fontsize=11.5, fontweight="bold", color=INK,
                 ha="left", va="bottom")
        fig.text(bb.x0 + 0.004, bb.y1 + 0.012, f"seed {seed} · {note}", fontsize=9.2, color=MUTED, ha="left",
                 va="bottom")
        rec = run_record(run)
        rec.update({"seed": seed, "family": fam, "scenario": f"{SCEN}/{seed}.json",
                    "scenario_sha256": world["scn"]["sha256"], "hazards": hazard_summary(world),
                    "typical_same_seed": run_record(load_run("TYPICAL", seed))["outcome"],
                    "typical_same_seed_time_s": load_run("TYPICAL", seed)["time"],
                    "lighting": {k2: world["scn"]["lighting"][k2] for k2 in ("sun_elev_deg", "fog_density", "events")},
                    "lighting_events_drawn": light_marks, "note": note})
        panels.append(rec)
    legend_row(fig, 0.995, 0.012, items=["trench", "crest", "object", "small", "radius"])
    fig.text(0.006, 0.012, f"Simulated · tier-0 synthetic depth sensor, no images · FULL stack, held-out EVAL "
                           f"seeds {', '.join(str(g[0]) for g in GALLERY)} · {RUNS}/FULL",
             fontsize=8, color=FAINT, ha="left", va="bottom")
    save(fig, name)
    side = {
        "figure": name, "label": "Simulated (tier-0 synthetic depth sensor, no images; held-out EVAL seeds)",
        "panels": panels,
        "selection": ("One FULL success per family for F1, F2, F3, F5 (F4 and F6 excluded by the brief). "
                      "F2 seed 37 has three trenches; F3 seed 2 is the only other F3 trench seed FULL completed "
                      "(seed 38 is used in run_map_pair); F5 seed 4 has glare and dimming inside the run window; "
                      "F1 seed 24 weaves through rocks on the trail."),
        "note_lighting": "In tier-0 there are no images: lighting events act on the synthetic depth sensor as "
                         "dropout, noise and dust phantom returns (metagross/sim/sensors.py).",
        "how_drawn": "same basemap as run_map_pair; amber bands = path stretches while a lighting event was active",
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
