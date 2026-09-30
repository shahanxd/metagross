"""Closed-loop EVAL figures for the final deck.

Run from the repo root:
    OMP_NUM_THREADS=2 python deck_assets/final/src/closed_loop.py

Outputs (next to this folder's parent, deck_assets/final/):
    eval_by_family.png / .svg / .json   reached-B per scenario family, FULL vs TYPICAL
    eval_outcomes.png  / .svg / .json   how every one of the 60 EVAL runs ended, per config

Every number is read from results/runs_eval_tier0/summary.csv (per-run rows) and cross-checked
against the aggregates in results/closed_loop_eval.json; the script stops if they disagree.
DEV numbers (seeds 100-129) from results/closed_loop_dev.json go into the json sidecars only.
Label: Simulated (tier-0 synthetic depth sensor, no images).
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import transforms  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "deck_assets" / "final"
EVAL_JSON = "results/closed_loop_eval.json"
EVAL_CSV = "results/runs_eval_tier0/summary.csv"
DEV_JSON = "results/closed_loop_dev.json"

# ---- palette (deck brief) ------------------------------------------------------------------
NAVY = "#1F497D"      # ours / FULL
GREY = "#9A9A9A"      # baseline / TYPICAL
GREEN = "#2E7D32"     # success
RED = "#C62828"       # hazard / collision
AMBER = "#D98E04"     # crest / ditch
BLUE = "#1E88E5"      # water
INK = "#1A1A1A"
MUTED = "#5C5C5C"
FAINT = "#8A8A8A"
GRID = "#E6E6E6"
BG = "#FFFFFF"
# outcome palette: validated with the dataviz validator (adjacent CVD dE >= 10.3, normal >= 15.9).
# "stuck" is deliberately a neutral taupe (a safe stop, not a hazard); every segment is count-labelled.
OUTCOME_COLOURS = {
    "success": GREEN,
    "arrived_short": "#7FC06F",
    "stuck": "#8C8577",
    "collision": RED,
    "water_entry": BLUE,
    "ditch_entry": AMBER,
    "out_of_bounds": "#6A3D9A",
}
OUTCOME_NAMES = {
    "success": "Reached B",
    "arrived_short": "Stopped short of B",
    "stuck": "Stuck",
    "collision": "Collision",
    "water_entry": "Water entry",
    "ditch_entry": "Ditch entry",
    "out_of_bounds": "Left the map",
}
OUTCOME_ORDER = list(OUTCOME_NAMES)

FAMILIES = [
    ("F1_trail", "F1  Trail", "rocks 0.2-1.0 m on the trail, trees along it"),
    ("F2_ditch_field", "F2  Ditch field", "1-3 trenches across the route, one gap each"),
    ("F3_crest_ditch", "F3  Crest + ditch", "0.4-0.8 m crest, hidden trench behind it in 5 of 10"),
    ("F4_sudden_obstacle", "F4  Sudden obstacle", "walker, box or boulder crosses 4-6 m ahead"),
    ("F5_lighting", "F5  Lighting", "low sun, glare, dimming, dust"),
    ("F6_water_mud", "F6  Water / mud", "flat water and mud on the route, dry way round"),
]
CONFIGS = [
    ("FULL", "FULL stack", "seen ground only, speed governor"),
    ("TYPICAL", "TYPICAL baseline", "unknown = free, fixed 1.5 m/s"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "svg.fonttype": "path",  # glyphs as outlines: renders identically without DejaVu Sans installed
    "figure.facecolor": BG,
    "axes.facecolor": BG,
})
DPI = 200


# ---- data ----------------------------------------------------------------------------------
def load_runs() -> list[dict]:
    with open(ROOT / EVAL_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["success"] = r["success"] == "True"
        r["outcome"] = "success" if r["success"] else (r["failure_type"] or "unknown")
    return rows


def tally(rows: list[dict]) -> dict:
    """Counts per config: overall outcomes and reached-B per family."""
    out = {}
    for cfg, _, _ in CONFIGS:
        mine = [r for r in rows if r["config_name"] == cfg]
        outcomes = Counter(r["outcome"] for r in mine)
        fam = {f: sum(r["success"] for r in mine if r["family"] == f) for f, _, _ in FAMILIES}
        fam_n = {f: sum(1 for r in mine if r["family"] == f) for f, _, _ in FAMILIES}
        seeds = sorted(int(r["seed"]) for r in mine)
        out[cfg] = {"n": len(mine), "seeds": [seeds[0], seeds[-1]], "outcomes": dict(outcomes),
                    "reached_by_family": fam, "n_by_family": fam_n}
    return out


def cross_check(t: dict, agg: dict) -> None:
    for cfg in t:
        a = agg[cfg]["all"]
        assert t[cfg]["n"] == a["n"], cfg
        assert t[cfg]["outcomes"].get("success", 0) == a["n_success"], cfg
        ft = {("success" if k == "None" else k): v for k, v in a["failure_types"].items()}
        assert ft == t[cfg]["outcomes"], (cfg, ft, t[cfg]["outcomes"])
        assert t[cfg]["outcomes"].get("out_of_bounds", 0) == a["out_of_bounds"], cfg
        for f, _, _ in FAMILIES:
            b = agg[cfg]["by_family"][f]
            assert t[cfg]["reached_by_family"][f] == b["n_success"], (cfg, f)
            assert t[cfg]["n_by_family"][f] == b["n"], (cfg, f)
    assert set(OUTCOME_ORDER) >= {k for c in t for k in t[c]["outcomes"]}, "unexpected outcome type"


# ---- drawing helpers -----------------------------------------------------------------------
def round_end_bar(ax, x0, x1, yc, h, colour, r_px=7, round_right=True, round_left=False):
    """Horizontal bar with rounded data-end(s), square at the baseline (radius in pixels)."""
    if x1 <= x0:
        return
    inv = ax.transData.inverted()
    p0 = inv.transform((0, 0))
    p1 = inv.transform((r_px, r_px))
    rx, ry = abs(p1[0] - p0[0]), abs(p1[1] - p0[1])
    rx = min(rx, (x1 - x0) / 2)
    ry = min(ry, h / 2)
    y0, y1 = yc - h / 2, yc + h / 2
    verts, codes = [], []

    def mv(p):
        verts.append(p); codes.append(MPath.MOVETO)

    def ln(p):
        verts.append(p); codes.append(MPath.LINETO)

    def cv(c, p):
        verts.extend([c, p]); codes.extend([MPath.CURVE3, MPath.CURVE3])

    if round_left:
        mv((x0 + rx, y0))
    else:
        mv((x0, y0))
    if round_right:
        ln((x1 - rx, y0)); cv((x1, y0), (x1, y0 + ry)); ln((x1, y1 - ry)); cv((x1, y1), (x1 - rx, y1))
    else:
        ln((x1, y0)); ln((x1, y1))
    if round_left:
        ln((x0 + rx, y1)); cv((x0, y1), (x0, y1 - ry)); ln((x0, y0 + ry)); cv((x0, y0), (x0 + rx, y0))
    else:
        ln((x0, y1)); ln((x0, y0))
    verts.append((0, 0)); codes.append(MPath.CLOSEPOLY)
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=colour, edgecolor="none", zorder=3))


def provenance(fig, text, y=0.018):
    fig.text(0.012, y, text, fontsize=10, color=FAINT, ha="left", va="bottom")


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.12, facecolor=BG)
    plt.close(fig)


def luminance(hex_colour: str) -> float:
    rgb = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


# ---- figure 1: reached B per family --------------------------------------------------------
def fig_by_family(t: dict) -> dict:
    fig = plt.figure(figsize=(9.0, 6.2), dpi=DPI)
    ax = fig.add_axes([0.335, 0.13, 0.64, 0.74])
    n_fam = len(FAMILIES)
    bar_h, gap = 0.34, 0.035
    ax.set_xlim(0, 10)
    ax.set_ylim(n_fam - 0.45, -0.62)
    fig.canvas.draw()

    drawn = []
    for i, (fam, name, desc) in enumerate(FAMILIES):
        for j, (cfg, _, _) in enumerate(CONFIGS):
            v = t[cfg]["reached_by_family"][fam]
            n = t[cfg]["n_by_family"][fam]
            yc = i + (-1 if j == 0 else 1) * (bar_h / 2 + gap / 2)
            colour = NAVY if cfg == "FULL" else GREY
            round_end_bar(ax, 0, v, yc, bar_h, colour, r_px=12)
            if v == 0:
                ax.plot([0, 0], [yc - bar_h / 2, yc + bar_h / 2], color=colour, lw=2.5, zorder=3,
                        solid_capstyle="butt")
            ax.text(v + 0.14, yc, f"{v}", va="center", ha="left", fontsize=13.5,
                    color=INK if cfg == "FULL" else MUTED, fontweight="bold" if cfg == "FULL" else "normal",
                    zorder=4)
            drawn.append({"what": f"{cfg} reached B, {fam}", "value": v, "of": n,
                          "source": EVAL_CSV,
                          "how": f"count of rows with config_name={cfg}, family={fam}, success=True "
                                 f"(matches {EVAL_JSON}#tier0.aggregate.{cfg}.by_family.{fam}.n_success)"})
        # family label: name + what it tests
        tr = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(-0.035, i, name, transform=transforms.offset_copy(tr, fig=fig, y=9, units="points"),
                ha="right", va="center", fontsize=13.5, color=INK)
        ax.text(-0.035, i, desc, transform=transforms.offset_copy(tr, fig=fig, y=-10, units="points"),
                ha="right", va="center", fontsize=10.5, color=MUTED)

    # grid and axes
    ax.set_xticks(range(0, 11, 2))
    ax.tick_params(axis="x", labelsize=12.5, length=0, pad=6)
    ax.set_yticks([])
    for x in range(0, 11, 2):
        ax.axvline(x, color=GRID, lw=1, zorder=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Runs that reached B (of 10 per family)", fontsize=12.5, color=INK, labelpad=8)

    # annotations (muted, factual)
    y_f2 = 1 - bar_h - gap / 2
    y_f3 = 2 + bar_h + gap / 2
    bx = 7.05
    ax.plot([bx, bx + 0.14, bx + 0.14, bx], [y_f2, y_f2, y_f3, y_f3], color=MUTED, lw=1.2, zorder=2,
            solid_joinstyle="miter")
    f23_full = t["FULL"]["reached_by_family"]["F2_ditch_field"] + t["FULL"]["reached_by_family"]["F3_crest_ditch"]
    f23_typ = t["TYPICAL"]["reached_by_family"]["F2_ditch_field"] + t["TYPICAL"]["reached_by_family"]["F3_crest_ditch"]
    knock = dict(boxstyle="square,pad=0.25", facecolor=BG, edgecolor="none")
    ax.text(bx + 0.35, 1.5, f"Ditch + crest\nfamilies: {f23_full} vs {f23_typ}", va="center", ha="left",
            fontsize=12.5, color=INK, linespacing=1.3, bbox=knock, zorder=4)
    coll_f4 = t["FULL"]["f4_collisions"]
    ax.text(6.0, 3, f"FULL: {coll_f4} collisions, obstacle\ncrossing in from the side",
            va="center", ha="left", fontsize=11.5, color=MUTED, linespacing=1.3, bbox=knock, zorder=4)
    ax.text(1.6, 5, "Tier-0 sensor has no images,\nso no water cue for either stack",
            va="center", ha="left", fontsize=11.5, color=MUTED, linespacing=1.3, bbox=knock, zorder=4)

    # key (legend) above the plot: swatch + name + total
    tot = {c: t[c]["outcomes"].get("success", 0) for c, _, _ in CONFIGS}
    kx = 0.335
    for cfg, label, _ in CONFIGS:
        colour = NAVY if cfg == "FULL" else GREY
        fig.patches.append(Rectangle((kx, 0.915), 0.022, 0.03, transform=fig.transFigure,
                                     facecolor=colour, edgecolor="none"))
        txt = fig.text(kx + 0.03, 0.93, f"{label}   {tot[cfg]}/{t[cfg]['n']} overall", fontsize=13,
                       va="center", ha="left", color=INK)
        fig.canvas.draw()
        bb = txt.get_window_extent().transformed(fig.transFigure.inverted())
        kx = bb.x1 + 0.04

    provenance(fig, "Simulated · tier-0 depth sensor, no images · 60 pre-registered EVAL scenarios (seeds 0-59) · "
                    "results/closed_loop_eval.json", y=0.0)
    save(fig, "eval_by_family")
    drawn.append({"what": "FULL reached B, all families", "value": tot["FULL"], "of": t["FULL"]["n"],
                  "source": EVAL_CSV, "how": "count of FULL rows with success=True"})
    drawn.append({"what": "TYPICAL reached B, all families", "value": tot["TYPICAL"], "of": t["TYPICAL"]["n"],
                  "source": EVAL_CSV, "how": "count of TYPICAL rows with success=True"})
    drawn.append({"what": "Ditch + crest families (F2+F3) reached B, FULL vs TYPICAL",
                  "value": f"{f23_full} vs {f23_typ}", "of": 20, "source": EVAL_CSV,
                  "how": "sum of F2_ditch_field and F3_crest_ditch success counts per config"})
    drawn.append({"what": "FULL collisions in F4 sudden obstacle", "value": coll_f4, "of": 10,
                  "source": EVAL_CSV,
                  "how": "count of FULL F4 rows with failure_type=collision; 'from the side' per docs/BUILD_LOG.md "
                         "(EVAL run 2: 'collision 7 (F4 side approaches)')"})
    return {"figure": "eval_by_family", "numbers": drawn}


# ---- figure 2: how every run ended ---------------------------------------------------------
def fig_outcomes(t: dict) -> dict:
    fig = plt.figure(figsize=(9.0, 4.9), dpi=DPI)
    ax = fig.add_axes([0.25, 0.245, 0.66, 0.525])
    n = t["FULL"]["n"]
    ax.set_xlim(0, n)
    ax.set_ylim(1.78, -0.62)
    fig.canvas.draw()
    bar_h = 0.62
    px_per_run = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
    drawn = []
    for i, (cfg, name, desc) in enumerate(CONFIGS):
        counts = t[cfg]["outcomes"]
        x = 0.0
        segs = [(k, counts.get(k, 0)) for k in OUTCOME_ORDER if counts.get(k, 0) > 0]
        for si, (k, c) in enumerate(segs):
            colour = OUTCOME_COLOURS[k]
            gap_runs = 4.5 / px_per_run  # 2 px surface gap at the ~0.5x scale the slide shows it
            x0 = x + (gap_runs / 2 if si > 0 else 0)
            x1 = x + c - (gap_runs / 2 if si < len(segs) - 1 else 0)
            round_end_bar(ax, x0, x1, i, bar_h, colour, r_px=12,
                          round_right=(si == len(segs) - 1), round_left=False)
            label = f"{c} reached B ({100 * c / n:.0f}%)" if k == "success" else f"{c}"
            txt_colour = "#FFFFFF" if luminance(colour) < 0.3 else INK
            width_px = (x1 - x0) * px_per_run
            inside = ax.text((x0 + x1) / 2, i, label, ha="center", va="center", fontsize=13.5, color=txt_colour,
                             fontweight="bold" if k == "success" else "normal", zorder=5)
            need_px = inside.get_window_extent(renderer=fig.canvas.get_renderer()).width + 16
            if width_px < need_px:  # does not fit: label outside the bar with a short leader
                inside.remove()
                side = -1 if i == 0 else 1
                ty = i + side * (bar_h / 2 + 0.30)
                ax.plot([(x0 + x1) / 2] * 2, [i + side * bar_h / 2, ty - side * 0.12], color=MUTED, lw=1,
                        zorder=2, clip_on=False)
                ax.text((x0 + x1) / 2, ty, label, ha="center", va="center", fontsize=13, color=INK,
                        clip_on=False)
            drawn.append({"what": f"{cfg} outcome '{OUTCOME_NAMES[k]}' ({k})", "value": c, "of": n,
                          "source": EVAL_CSV,
                          "how": ("count of rows with success=True" if k == "success"
                                  else f"count of rows with failure_type={k}") +
                                 f" for config_name={cfg} (matches {EVAL_JSON}#tier0.aggregate.{cfg}.all.failure_types)"})
            x += c
        # config label
        tr = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(-0.03, i, name, transform=transforms.offset_copy(tr, fig=fig, y=9, units="points"),
                ha="right", va="center", fontsize=14, color=INK, fontweight="bold" if cfg == "FULL" else "normal")
        ax.text(-0.03, i, desc, transform=transforms.offset_copy(tr, fig=fig, y=-10, units="points"),
                ha="right", va="center", fontsize=10.5, color=MUTED)

    # the zero that matters: FULL never left the map
    oob_full = t["FULL"]["outcomes"].get("out_of_bounds", 0)
    oob_typ = t["TYPICAL"]["outcomes"].get("out_of_bounds", 0)
    ax.text(n + 0.8, 0, f"{oob_full} left\nthe map", ha="left", va="center", fontsize=12.5, color=INK,
            fontweight="bold", linespacing=1.2, clip_on=False)
    ax.text(n + 0.8, 1, f"{oob_typ} left\nthe map", ha="left", va="center", fontsize=12.5, color=MUTED,
            linespacing=1.2, clip_on=False)
    for cfg, v in (("FULL", oob_full), ("TYPICAL", oob_typ)):
        drawn.append({"what": f"{cfg} runs that left the map (annotation right of bar)", "value": v, "of": n,
                      "source": EVAL_CSV,
                      "how": f"count of {cfg} rows with failure_type=out_of_bounds "
                             f"(matches {EVAL_JSON}#tier0.aggregate.{cfg}.all.out_of_bounds)"})
    for cfg, _, _ in CONFIGS:
        c = t[cfg]["outcomes"].get("success", 0)
        drawn.append({"what": f"{cfg} reached-B share printed in the green segment", "value": f"{100 * c / n:.0f}%",
                      "of": n, "source": EVAL_CSV, "how": f"{c}/{n}, rounded to whole percent"})

    ax.set_xticks([0, 15, 30, 45, 60])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="x", labelsize=12, length=0, pad=6)
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    for xv in (15, 30, 45):
        ax.axvline(xv, color=GRID, lw=1, zorder=0)
    ax.set_xlabel(f"Share of the {n} held-out EVAL runs", fontsize=12.5, color=INK, labelpad=6)

    # legend: two rows, same order as the segments
    items = OUTCOME_ORDER
    rows = [items[:4], items[4:]]
    y0 = 0.935
    col_x = []  # row 2 reuses row 1's column positions so the key reads as a grid
    for r, row in enumerate(rows):
        kx = 0.25
        for ci, k in enumerate(row):
            if r > 0:
                kx = col_x[ci]
            else:
                col_x.append(kx)
            yy = y0 - r * 0.068
            fig.patches.append(Rectangle((kx, yy - 0.017), 0.02, 0.034, transform=fig.transFigure,
                                         facecolor=OUTCOME_COLOURS[k], edgecolor="none"))
            txt = fig.text(kx + 0.027, yy, OUTCOME_NAMES[k], fontsize=12.5, va="center", ha="left", color=INK)
            fig.canvas.draw()
            bb = txt.get_window_extent().transformed(fig.transFigure.inverted())
            kx = bb.x1 + 0.04

    fig.text(0.012, 0.066,
             "Stopped short: declared arrival on its own pose estimate but ended outside the 2 m success radius.  "
             "Stuck: < 0.5 m progress in 20 s.",
             fontsize=10, color=MUTED, ha="left", va="bottom")
    provenance(fig, "Simulated · tier-0 depth sensor, no images · EVAL seeds 0-59, pre-registered · "
                    "results/runs_eval_tier0/summary.csv", y=0.0)
    save(fig, "eval_outcomes")
    return {"figure": "eval_outcomes", "numbers": drawn}


# ---- DEV numbers for the sidecar ------------------------------------------------------------
def dev_numbers() -> dict:
    d = json.loads((ROOT / DEV_JSON).read_text())
    out = {"source": DEV_JSON, "split": "dev", "seeds": d["seeds"],
           "note": "DEV seeds 100-129 were used for tuning; EVAL seeds 0-59 were not. Not drawn.", "configs": {}}
    for cfg, v in d["tier0"]["aggregate"].items():
        a = v["all"]
        out["configs"][cfg] = {
            "n": a["n"], "reached_B": a["n_success"],
            "out_of_bounds": a["out_of_bounds"],
            "outcomes": {("success" if k == "None" else k): c for k, c in a["failure_types"].items()},
            "reached_B_by_family": {f: b["n_success"] for f, b in v["by_family"].items()},
            "n_by_family": {f: b["n"] for f, b in v["by_family"].items()},
            "spl_mean": a["spl_mean"], "mean_speed_mps": a["mean_speed_mps"],
            "path": f"{DEV_JSON}#tier0.aggregate.{cfg}",
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_runs()
    t = tally(rows)
    t["FULL"]["f4_collisions"] = sum(1 for r in rows if r["config_name"] == "FULL"
                                     and r["family"] == "F4_sudden_obstacle" and r["outcome"] == "collision")
    # F3 description ("hidden trench in 5 of 10"): trench present for even seed // 6 (metagross/sim/scenario.py)
    f3_seeds = sorted({int(r["seed"]) for r in rows if r["family"] == "F3_crest_ditch"})
    f3_trench = sum(1 for sd in f3_seeds if (sd // 6) % 2 == 0)
    assert (f3_trench, len(f3_seeds)) == (5, 10), (f3_trench, f3_seeds)
    ev = json.loads((ROOT / EVAL_JSON).read_text())
    cross_check(t, ev["tier0"]["aggregate"])
    dev = dev_numbers()
    common = {
        "label": "Simulated",
        "split": "eval (pre-registered, docs/EVAL_PREREGISTRATION.md, run 2, commit 45ec399)",
        "sensor": "tier-0 synthetic depth sensor (no images: no VO, no segmentation)",
        "configs": {"FULL": "default stack: seen ground only, missing-ground detector, seen-distance speed governor",
                    "TYPICAL": "unknown_is_free, no missing-ground detector, no governor, fixed 1.5 m/s"},
        "sources": [EVAL_CSV, EVAL_JSON, DEV_JSON],
        "cross_check": "per-run CSV counts equal the JSON aggregates (asserted in this script)",
        "dev_split_not_drawn": dev,
        "generator": "deck_assets/final/src/closed_loop.py",
    }
    fam_spec = fig_by_family(t)
    fam_spec["numbers"].append({
        "what": "F3 EVAL seeds with a hidden trench behind the crest (family label text)",
        "value": f3_trench, "of": len(f3_seeds), "source": "metagross/sim/scenario.py + " + EVAL_CSV,
        "how": "F3 seeds in the CSV with (seed // 6) even; odd ones are crest-only safe controls"})
    for spec in (fam_spec, fig_outcomes(t)):
        (OUT / f"{spec['figure']}.json").write_text(json.dumps({**spec, **common}, indent=2) + "\n")
        print("wrote", OUT / f"{spec['figure']}.png")


if __name__ == "__main__":
    main()
