"""Closed-loop EVAL figures for the final deck.

Run from the repo root:
    OMP_NUM_THREADS=2 python deck_assets/final/src/closed_loop.py

Outputs (deck_assets/final/):
    eval_by_family.png / .svg / .json   reached-B per scenario family, FULL vs TYPICAL
    eval_outcomes.png  / .svg / .json   how every one of the 60 EVAL runs ended, per config

Every drawn number is counted from results/runs_eval_tier0/summary.csv (per-run rows), cross-checked
against the aggregates in results/closed_loop_eval.json AND against its registered row in
results/claims.csv; the script stops if any of them disagree. The json sidecars list each drawn number
with its claim id, source and how it was computed.
The EVAL numbers are run 2: a disclosed protocol deviation (docs/EVAL_PREREGISTRATION.md).
DEV numbers (seeds 100-129, results/closed_loop_dev.json) go into the sidecars only; they are not in
the claims ledger and are not drawn.
Label: Simulated (tier-0 synthetic depth sensor, no images).
"""
from __future__ import annotations

import csv
import json
import math
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
CLAIMS_CSV = "results/claims.csv"
PREREG = "docs/EVAL_PREREGISTRATION.md"
CLAIM_PREFIX = "closed_loop_eval_tier0_"

# ---- palette (deck brief) ------------------------------------------------------------------
NAVY = "#1F497D"      # ours / FULL
GREY = "#9A9A9A"      # baseline / TYPICAL
GREEN = "#2E7D32"     # success
RED = "#C62828"       # hazard / collision
AMBER = "#D98E04"     # crest / ditch
BLUE = "#1E88E5"      # water
INK = "#1A1A1A"
MUTED = "#5C5C5C"     # secondary text, 6.6:1 on white
FAINT = "#6E6E6E"     # provenance text, 5.1:1 on white
GRID = "#E6E6E6"
BG = "#FFFFFF"
WHITE = "#FFFFFF"
# Outcome palette. Green = reached B only. The two failures that did NOT enter a hazard (stopped short,
# stuck) are neutrals (sand, taupe); hazards keep the brief's saturated hues, plus plum for out of bounds.
# dataviz validator (light): lightness band PASS, adjacent CVD worst dE 8.0 (target 8), normal-vision
# worst dE 18.4; chroma floor fails only for the two neutrals, on purpose. Every segment carries its count
# label and a key entry, and segments are separated by white gaps.
OUTCOME_COLOURS = {
    "success": GREEN,
    "arrived_short": "#C4B08C",
    "stuck": "#7A7366",
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
    "out_of_bounds": "Out of bounds",
}
OUTCOME_ORDER = list(OUTCOME_NAMES)
# claims.csv id suffix per outcome (FULL has no fail_out_of_bounds row because it is 0; *_oob covers both)
OUTCOME_CLAIM = {
    "success": "success", "arrived_short": "fail_arrived_short", "stuck": "fail_stuck",
    "collision": "fail_collision", "water_entry": "fail_water_entry", "ditch_entry": "fail_ditch_entry",
    "out_of_bounds": "oob",
}

# Family descriptors: scenario *design* parameters from metagross/sim/scenario.py (docstring + F*_ constants,
# _f1_trail_objects, _f4_dynamic) and metagross/sim/sensors.py (how tier-0 models F5 lighting). Not results.
FAMILIES = [
    ("F1_trail", "F1  Trail",
     "rocks (radius 0.2-1 m) on and beside\nthe trail, trees along it"),
    ("F2_ditch_field", "F2  Ditch field",
     "1-3 trenches across the route,\none gap in each"),
    ("F3_crest_ditch", "F3  Crest + ditch",
     "0.4-0.8 m high crest; hidden trench\njust behind it in 5 of the 10 runs"),
    ("F4_sudden_obstacle", "F4  Sudden obstacle",
     "walker, box or boulder steps out from\nbehind a bush once the UGV is 4-6 m away"),
    ("F5_lighting", "F5  Lighting",
     "glare, dimming and dust, simulated as\ndepth dropout, noise and phantom returns"),
    ("F6_water_mud", "F6  Water / mud",
     "flat water and mud on the route,\nwith a dry way round"),
]
CONFIGS = [
    ("FULL", "FULL stack", "seen ground only, speed governor"),
    ("TYPICAL", "TYPICAL baseline", "unknown = free, fixed 1.5 m/s"),
]
PROV_SPLIT = f"EVAL seeds 0-59, run 2 (a disclosed re-run, see {PREREG})"

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
PAD_IN = 0.12


# ---- data ----------------------------------------------------------------------------------
def load_runs() -> list[dict]:
    with open(ROOT / EVAL_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["success"] = r["success"] == "True"
        r["outcome"] = "success" if r["success"] else (r["failure_type"] or "unknown")
        for k in ("ditch_entries", "collisions", "water_entries", "false_stops"):
            r[k] = int(r[k])
    return rows


def load_claims() -> dict[str, dict]:
    with open(ROOT / CLAIMS_CSV, newline="") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


CLAIMS: dict[str, dict] = {}


def claim(suffix: str, value: float) -> str:
    """Assert that ``value`` is registered in results/claims.csv under CLAIM_PREFIX+suffix; return the id."""
    cid = CLAIM_PREFIX + suffix
    assert cid in CLAIMS, f"not registered in {CLAIMS_CSV}: {cid}"
    reg = float(CLAIMS[cid]["value"])
    assert math.isclose(reg, float(value), abs_tol=1e-9), (cid, reg, value)
    assert CLAIMS[cid]["label"] == "Simulated", cid
    return cid


def tally(rows: list[dict]) -> dict:
    """Counts per config: overall outcomes, reached-B / oob / ditch entries / collisions per family."""
    out = {}
    for cfg, _, _ in CONFIGS:
        mine = [r for r in rows if r["config_name"] == cfg]
        fam = {}
        for f, _, _ in FAMILIES:
            fr = [r for r in mine if r["family"] == f]
            fam[f] = {"n": len(fr), "success": sum(r["success"] for r in fr),
                      "oob": sum(r["outcome"] == "out_of_bounds" for r in fr),
                      "ditch": sum(r["ditch_entries"] for r in fr),
                      "fail_collision": sum(r["outcome"] == "collision" for r in fr)}
        out[cfg] = {"n": len(mine), "outcomes": dict(Counter(r["outcome"] for r in mine)), "fam": fam,
                    "false_stops": sum(r["false_stops"] for r in mine)}
    return out


def cross_check(t: dict, agg: dict) -> None:
    for cfg in t:
        a = agg[cfg]["all"]
        assert t[cfg]["n"] == a["n"], cfg
        assert t[cfg]["outcomes"].get("success", 0) == a["n_success"], cfg
        ft = {("success" if k == "None" else k): v for k, v in a["failure_types"].items()}
        assert ft == t[cfg]["outcomes"], (cfg, ft, t[cfg]["outcomes"])
        assert t[cfg]["outcomes"].get("out_of_bounds", 0) == a["out_of_bounds"], cfg
        assert t[cfg]["false_stops"] == a["false_stops"], cfg
        for f, _, _ in FAMILIES:
            b, m = agg[cfg]["by_family"][f], t[cfg]["fam"][f]
            assert m["success"] == b["n_success"] and m["n"] == b["n"], (cfg, f)
            assert m["oob"] == b["out_of_bounds"] and m["ditch"] == b["ditch_entries"], (cfg, f)
    assert set(OUTCOME_ORDER) >= {k for c in t for k in t[c]["outcomes"]}, "unexpected outcome type"


def paired(rows: list[dict], fams: tuple[str, ...]) -> dict:
    """Seed-paired reached-B comparison (not drawn, not registered: context for slide writers)."""
    by = {(r["config_name"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows if r["family"] in fams})
    f_only = [s for s in seeds if by[("FULL", s)]["success"] and not by[("TYPICAL", s)]["success"]]
    t_only = [s for s in seeds if by[("TYPICAL", s)]["success"] and not by[("FULL", s)]["success"]]
    k, n = min(len(f_only), len(t_only)), len(f_only) + len(t_only)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"families": list(fams), "seeds_only_FULL_reached_B": f_only, "seeds_only_TYPICAL_reached_B": t_only,
            "exact_two_sided_sign_test_p": round(p, 3),
            "how": f"per-seed pairing of FULL and TYPICAL rows in {EVAL_CSV}; exact McNemar (binomial) test on "
                   "the discordant seeds. Computed here; NOT registered in results/claims.csv."}


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

    mv((x0 + rx, y0) if round_left else (x0, y0))
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


def bottom_notes(fig, lines: list[tuple[str, str]], y0=0.0, step=0.034):
    """Footnote/provenance lines stacked upward from the bottom-left corner: (text, colour)."""
    for k, (text, colour) in enumerate(reversed(lines)):
        fig.text(0.012, y0 + k * step, text, fontsize=11, color=colour, ha="left", va="bottom")


def save(fig, name):
    """Save PNG and SVG with ONE shared tight box, so both files have the same extent and aspect."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer()).padded(PAD_IN)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=DPI, bbox_inches=bb, facecolor=BG)
    plt.close(fig)
    from PIL import Image  # matplotlib dependency; read back the real pixel size

    w, h = float(bb.width), float(bb.height)
    with Image.open(OUT / f"{name}.png") as im:
        png_px = list(im.size)
    return {"width_in": round(w, 4), "height_in": round(h, 4),
            "png_px": png_px, "svg_pt": [round(w * 72, 1), round(h * 72, 1)]}


def luminance(hex_colour: str) -> float:
    rgb = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def label_colour(fill: str) -> str:
    """White or ink, whichever contrasts more with the fill; must reach WCAG 4.5:1."""
    best = max((WHITE, INK), key=lambda c: contrast(c, fill))
    assert contrast(best, fill) >= 4.5, (fill, contrast(best, fill))
    return best


# ---- figure 1: reached B per family --------------------------------------------------------
def fig_by_family(t: dict) -> dict:
    fig = plt.figure(figsize=(10.0, 7.4), dpi=DPI)
    ax = fig.add_axes([0.335, 0.19, 0.625, 0.70])
    n_fam = len(FAMILIES)
    bar_h, gap = 0.34, 0.035
    ax.set_xlim(0, 10)
    ax.set_ylim(n_fam - 0.45, -0.62)
    fig.canvas.draw()

    drawn = []
    for i, (fam, name, desc) in enumerate(FAMILIES):
        for j, (cfg, _, _) in enumerate(CONFIGS):
            v = t[cfg]["fam"][fam]["success"]
            n = t[cfg]["fam"][fam]["n"]
            yc = i + (-1 if j == 0 else 1) * (bar_h / 2 + gap / 2)
            colour = NAVY if cfg == "FULL" else GREY
            round_end_bar(ax, 0, v, yc, bar_h, colour, r_px=12)
            if v == 0:
                ax.plot([0, 0], [yc - bar_h / 2, yc + bar_h / 2], color=colour, lw=2.5, zorder=3,
                        solid_capstyle="butt")
            ax.text(v + 0.14, yc, f"{v}", va="center", ha="left", fontsize=14,
                    color=INK if cfg == "FULL" else MUTED, fontweight="bold" if cfg == "FULL" else "normal",
                    zorder=4)
            drawn.append({"what": f"{cfg} reached B, {fam}", "value": v, "of": n,
                          "claim_id": claim(f"{cfg}_{fam}_success", v), "source": EVAL_CSV,
                          "how": f"count of rows with config_name={cfg}, family={fam}, success=True "
                                 f"(= {EVAL_JSON}#tier0.aggregate.{cfg}.by_family.{fam}.n_success)"})
        # family label: name + what it tests (scenario design, two lines)
        tr = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(-0.03, i, name, transform=transforms.offset_copy(tr, fig=fig, y=14, units="points"),
                ha="right", va="center", fontsize=14, color=INK)
        ax.text(-0.03, i, desc, transform=transforms.offset_copy(tr, fig=fig, y=-9.5, units="points"),
                ha="right", va="center", fontsize=11, color=MUTED, linespacing=1.2)

    # grid and axes
    ax.set_xticks(range(0, 11, 2))
    ax.tick_params(axis="x", labelsize=13, length=0, pad=6)
    ax.set_yticks([])
    for x in range(0, 11, 2):
        ax.axvline(x, color=GRID, lw=1, zorder=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Runs that reached B (of 10 per family)", fontsize=13, color=INK, labelpad=8)

    knock = dict(boxstyle="square,pad=0.2", facecolor=BG, edgecolor="none")

    # F2 + F3 mini table: reached B, ditch entries, out of bounds (FULL vs TYPICAL)
    f23 = ("F2_ditch_field", "F3_crest_ditch")
    s23 = {c: sum(t[c]["fam"][f]["success"] for f in f23) for c in ("FULL", "TYPICAL")}
    d23 = {c: sum(t[c]["fam"][f]["ditch"] for f in f23) for c in ("FULL", "TYPICAL")}
    o23 = {c: sum(t[c]["fam"][f]["oob"] for f in f23) for c in ("FULL", "TYPICAL")}
    y_top, y_bot = 1 - bar_h - gap / 2, 2 + bar_h + gap / 2
    bx = 6.75
    ax.plot([bx, bx + 0.14, bx + 0.14, bx], [y_top, y_top, y_bot, y_bot], color=MUTED, lw=1.2, zorder=2,
            solid_joinstyle="miter")
    tx, vx = bx + 0.4, 10.35
    ax.text(tx, 1.5 - 0.62, "F2 + F3, FULL vs TYPICAL", ha="left", va="center", fontsize=11.5, color=MUTED,
            clip_on=False, zorder=4, bbox=knock)
    table = [("Reached B", s23, True), ("Ditch entries", d23, False), ("Out of bounds", o23, False)]
    for k, (lab, vals, strong) in enumerate(table):
        yy = 1.5 - 0.2 + k * 0.4
        ax.text(tx, yy, lab, ha="left", va="center", fontsize=13, color=INK, clip_on=False, zorder=4)
        ax.text(vx, yy, f"{vals['FULL']} vs {vals['TYPICAL']}", ha="right", va="center", fontsize=13,
                color=INK, fontweight="bold" if strong else "normal", clip_on=False, zorder=4)

    # F4: FULL is worse than the baseline
    c4 = {c: t[c]["fam"]["F4_sudden_obstacle"]["fail_collision"] for c in ("FULL", "TYPICAL")}
    hit = f4_collision_objects()  # asserts every F4 collision was with the moving obstacle ("hit the obstacle")
    ax.text(5.75, 3, f"FULL is worse here: {c4['FULL']} of 10 runs\n"
                     f"hit the obstacle (TYPICAL {c4['TYPICAL']}); it enters\n"
                     "from outside the sensor's view",
            va="center", ha="left", fontsize=12, color=INK, linespacing=1.22, bbox=knock, zorder=4)
    ax.text(1.6, 5, "Tier-0 sensor has no images,\nso neither stack has a water cue",
            va="center", ha="left", fontsize=12, color=MUTED, linespacing=1.25, bbox=knock, zorder=4)

    # key (legend) above the plot: swatch + name + total
    tot = {c: t[c]["outcomes"].get("success", 0) for c, _, _ in CONFIGS}
    kx = 0.335
    for cfg, label, _ in CONFIGS:
        colour = NAVY if cfg == "FULL" else GREY
        fig.patches.append(Rectangle((kx, 0.925), 0.02, 0.027, transform=fig.transFigure,
                                     facecolor=colour, edgecolor="none"))
        txt = fig.text(kx + 0.028, 0.9385, f"{label}   {tot[cfg]}/{t[cfg]['n']} overall", fontsize=13.5,
                       va="center", ha="left", color=INK)
        fig.canvas.draw()
        bb = txt.get_window_extent().transformed(fig.transFigure.inverted())
        kx = bb.x1 + 0.04

    bottom_notes(fig, [
        (f"Overall gap +{tot['FULL'] - tot['TYPICAL']}/60 is within batch noise ({PREREG}).", MUTED),
        ("Simulated · tier-0 depth sensor, no images · " + PROV_SPLIT, FAINT),
        (f"Counts: {EVAL_CSV} = {EVAL_JSON}, registered in {CLAIMS_CSV}", FAINT),
    ])
    size = save(fig, "eval_by_family")

    drawn += [
        {"what": f"{cfg} reached B, all families (key)", "value": tot[cfg], "of": t[cfg]["n"],
         "claim_id": claim(f"{cfg}_success", tot[cfg]), "source": EVAL_CSV,
         "how": f"count of {cfg} rows with success=True"} for cfg in ("FULL", "TYPICAL")]
    drawn.append({"what": "Overall gap FULL - TYPICAL (footnote)", "value": tot["FULL"] - tot["TYPICAL"], "of": 60,
                  "claim_id": [claim("FULL_success", tot["FULL"]), claim("TYPICAL_success", tot["TYPICAL"])],
                  "source": EVAL_CSV, "how": "difference of the two registered totals; 'within batch noise' "
                                             f"quoted from {PREREG} 'Reading'"})
    for cfg in ("FULL", "TYPICAL"):
        drawn += [
            {"what": f"{cfg} reached B in F2+F3 (mini table)", "value": s23[cfg], "of": 20,
             "claim_id": claim(f"{cfg}_F2F3_ditch_crest_success", s23[cfg]), "source": EVAL_CSV,
             "how": "sum of F2_ditch_field and F3_crest_ditch success counts"},
            {"what": f"{cfg} ditch entries in F2+F3 (mini table)", "value": d23[cfg],
             "claim_id": [claim(f"{cfg}_F2_ditch_field_ditch", t[cfg]["fam"]["F2_ditch_field"]["ditch"]),
                          claim(f"{cfg}_F3_crest_ditch_ditch", t[cfg]["fam"]["F3_crest_ditch"]["ditch"]),
                          claim(f"{cfg}_ditch", d23[cfg])],
             "source": EVAL_CSV,
             "how": "sum of ditch_entries over F2 and F3 rows; equals the all-family total (no ditch entries "
                    "elsewhere)"},
            {"what": f"{cfg} out-of-bounds runs in F2+F3 (mini table)", "value": o23[cfg], "of": 20,
             "claim_id": [claim(f"{cfg}_F2_ditch_field_oob", t[cfg]["fam"]["F2_ditch_field"]["oob"]),
                          claim(f"{cfg}_F3_crest_ditch_oob", t[cfg]["fam"]["F3_crest_ditch"]["oob"]),
                          claim(f"{cfg}_oob", o23[cfg])],
             "source": EVAL_CSV,
             "how": "count of F2+F3 rows with failure_type=out_of_bounds; equals the all-family total"},
            {"what": f"{cfg} F4 runs that ended in a collision (F4 note)", "value": c4[cfg], "of": 10,
             "claim_id": claim(f"{cfg}_F4_sudden_obstacle_fail_collision", c4[cfg]), "source": EVAL_CSV,
             "how": "count of F4 rows with failure_type=collision; every one is a GT collision with the moving "
                    f"object (dyn_*) per results/runs_eval_tier0/<cfg>/<seed>/gt/events.json: {hit[cfg]}; "
                    f"'enters from outside the sensor's view' per {PREREG} 'Known limits in tier-0'"},
        ]
    return {"figure": "eval_by_family", "size": size, "numbers": drawn}


# ---- figure 1b: the same counts sized for the deck's 652 x 400 px right-hand slot -----------
SLOT_W_PX, SLOT_H_PX = 652, 400  # slide pixels (1920 x 1080 slide)
SLOT_SCALE = 3  # output pixels per slide pixel
SLOT_DPI = 100 * SLOT_SCALE
SLOT_PT_PER_PX = SLOT_SCALE * 72 / SLOT_DPI  # font size in pt that renders 1 slide px tall
SLOT_NAMES = {"F1_trail": "F1  Trail", "F2_ditch_field": "F2  Ditch field", "F3_crest_ditch": "F3  Crest + ditch",
              "F4_sudden_obstacle": "F4  Sudden obstacle", "F5_lighting": "F5  Lighting",
              "F6_water_mud": "F6  Water, mud"}


def fig_by_family_slot(t: dict) -> dict:
    """Bars only, every label >= 15 slide px; the F2+F3 table, F4 note and provenance go in the slide caption."""
    px = SLOT_PT_PER_PX
    fig = plt.figure(figsize=(SLOT_W_PX / 100, SLOT_H_PX / 100), dpi=SLOT_DPI)
    left, right, bottom, top = 186, 606, 30, 360  # plot box in slide px (x from left, y from bottom)
    ax = fig.add_axes([left / SLOT_W_PX, bottom / SLOT_H_PX, (right - left) / SLOT_W_PX, (top - bottom) / SLOT_H_PX])
    n_fam = len(FAMILIES)
    bar_h, gap = 0.37, 0.05
    ax.set_xlim(0, 10)
    ax.set_ylim(n_fam - 0.5, -0.5)
    fig.canvas.draw()
    drawn = []
    for i, (fam, _, _) in enumerate(FAMILIES):
        for j, (cfg, _, _) in enumerate(CONFIGS):
            v, n = t[cfg]["fam"][fam]["success"], t[cfg]["fam"][fam]["n"]
            yc = i + (-1 if j == 0 else 1) * (bar_h / 2 + gap / 2)
            colour = NAVY if cfg == "FULL" else GREY
            round_end_bar(ax, 0, v, yc, bar_h, colour, r_px=4 * SLOT_SCALE)
            if v == 0:
                ax.plot([0, 0], [yc - bar_h / 2, yc + bar_h / 2], color=colour, lw=2.5, zorder=3, solid_capstyle="butt")
            ax.text(v + 0.15, yc, f"{v}", va="center", ha="left", fontsize=16 * px,
                    color=INK if cfg == "FULL" else MUTED, fontweight="bold" if cfg == "FULL" else "normal", zorder=4)
            drawn.append({"what": f"{cfg} reached B, {fam}", "value": v, "of": n,
                          "claim_id": claim(f"{cfg}_{fam}_success", v), "source": EVAL_CSV})
        ax.text(-0.25, i, SLOT_NAMES[fam], ha="right", va="center", fontsize=17 * px, color=INK)
    ax.set_xticks(range(0, 11, 2))
    ax.tick_params(axis="x", labelsize=15 * px, length=0, pad=3 * SLOT_SCALE / 2)
    ax.set_yticks([])
    for x in range(0, 11, 2):
        ax.axvline(x, color=GRID, lw=1, zorder=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    # key: swatch + name + overall total, then what the bars count
    tot = {c: t[c]["outcomes"].get("success", 0) for c, _, _ in CONFIGS}
    kx, ky = 8 / SLOT_W_PX, 1 - 16 / SLOT_H_PX
    for cfg, label, _ in CONFIGS:
        colour = NAVY if cfg == "FULL" else GREY
        fig.patches.append(Rectangle((kx, ky - 7 / SLOT_H_PX), 14 / SLOT_W_PX, 14 / SLOT_H_PX,
                                     transform=fig.transFigure, facecolor=colour, edgecolor="none"))
        txt = fig.text(kx + 20 / SLOT_W_PX, ky, f"{label} {tot[cfg]}/{t[cfg]['n']}", fontsize=16 * px,
                       va="center", ha="left", color=INK, fontweight="bold" if cfg == "FULL" else "normal")
        fig.canvas.draw()
        kx = txt.get_window_extent().transformed(fig.transFigure.inverted()).x1 + 22 / SLOT_W_PX
    fig.text(1 - 8 / SLOT_W_PX, ky, "reached B, of 10 per family", fontsize=15 * px, va="center", ha="right",
             color=MUTED)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"eval_by_family_slot.{ext}", dpi=SLOT_DPI, facecolor=BG)
    plt.close(fig)
    drawn += [{"what": f"{cfg} reached B, all families (key)", "value": tot[cfg], "of": t[cfg]["n"],
               "claim_id": claim(f"{cfg}_success", tot[cfg]), "source": EVAL_CSV} for cfg in ("FULL", "TYPICAL")]
    return {"figure": "eval_by_family_slot", "numbers": drawn,
            "size": {"slide_px": [SLOT_W_PX, SLOT_H_PX], "png_px": [SLOT_W_PX * SLOT_SCALE, SLOT_H_PX * SLOT_SCALE]},
            "note": "deck slot version of eval_by_family: same counts; min text 15 slide px; the F2+F3 table, the F4 "
                    "note and the provenance line are carried by the slide caption"}


# ---- figure 2: how every run ended ---------------------------------------------------------
def fig_outcomes(t: dict) -> dict:
    fig = plt.figure(figsize=(10.0, 5.3), dpi=DPI)
    ax = fig.add_axes([0.25, 0.33, 0.64, 0.44])
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
            gap_runs = 4.5 / px_per_run  # ~2 px surface gap at the ~0.5x scale the slide shows it
            x0 = x + (gap_runs / 2 if si > 0 else 0)
            x1 = x + c - (gap_runs / 2 if si < len(segs) - 1 else 0)
            round_end_bar(ax, x0, x1, i, bar_h, colour, r_px=12,
                          round_right=(si == len(segs) - 1), round_left=False)
            label = f"{c} reached B ({100 * c / n:.0f}%)" if k == "success" else f"{c}"
            width_px = (x1 - x0) * px_per_run
            inside = ax.text((x0 + x1) / 2, i, label, ha="center", va="center", fontsize=14,
                             color=label_colour(colour), fontweight="bold" if k == "success" else "normal",
                             zorder=5)
            need_px = inside.get_window_extent(renderer=fig.canvas.get_renderer()).width + 16
            if width_px < need_px:  # does not fit: label outside the bar with a short leader
                inside.remove()
                side = -1 if i == 0 else 1
                ty = i + side * (bar_h / 2 + 0.30)
                ax.plot([(x0 + x1) / 2] * 2, [i + side * bar_h / 2, ty - side * 0.12], color=MUTED, lw=1,
                        zorder=2, clip_on=False)
                ax.text((x0 + x1) / 2, ty, label, ha="center", va="center", fontsize=13.5, color=INK,
                        clip_on=False)
            drawn.append({"what": f"{cfg} outcome '{OUTCOME_NAMES[k]}' ({k})", "value": c, "of": n,
                          "claim_id": claim(f"{cfg}_{OUTCOME_CLAIM[k]}", c), "source": EVAL_CSV,
                          "how": ("count of rows with success=True" if k == "success"
                                  else f"count of rows with failure_type={k}") +
                                 f" for config_name={cfg} (= {EVAL_JSON}#tier0.aggregate.{cfg}.all.failure_types)"})
            x += c
        # config label
        tr = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(-0.03, i, name, transform=transforms.offset_copy(tr, fig=fig, y=10, units="points"),
                ha="right", va="center", fontsize=14.5, color=INK, fontweight="bold" if cfg == "FULL" else "normal")
        ax.text(-0.03, i, desc, transform=transforms.offset_copy(tr, fig=fig, y=-10, units="points"),
                ha="right", va="center", fontsize=11.5, color=MUTED)

    # the zero that matters: FULL never went out of bounds (on these 60 EVAL runs)
    oob = {c: t[c]["outcomes"].get("out_of_bounds", 0) for c in ("FULL", "TYPICAL")}
    ax.text(n + 0.8, 0, f"{oob['FULL']} out of\nbounds", ha="left", va="center", fontsize=13, color=INK,
            fontweight="bold", linespacing=1.2, clip_on=False)
    ax.text(n + 0.8, 1, f"{oob['TYPICAL']} out of\nbounds", ha="left", va="center", fontsize=13, color=MUTED,
            linespacing=1.2, clip_on=False)
    for cfg in ("FULL", "TYPICAL"):
        drawn.append({"what": f"{cfg} runs out of bounds (annotation right of bar)", "value": oob[cfg], "of": n,
                      "claim_id": claim(f"{cfg}_oob", oob[cfg]), "source": EVAL_CSV,
                      "how": f"count of {cfg} rows with failure_type=out_of_bounds "
                             f"(= {EVAL_JSON}#tier0.aggregate.{cfg}.all.out_of_bounds)"})
        c = t[cfg]["outcomes"].get("success", 0)
        rate = float(CLAIMS[claim(f"{cfg}_success_rate", round(c / n, 3))]["value"])
        drawn.append({"what": f"{cfg} reached-B share printed in the green segment", "value": f"{100 * c / n:.0f}%",
                      "of": n, "claim_id": CLAIM_PREFIX + f"{cfg}_success_rate", "source": EVAL_CSV,
                      "how": f"{c}/{n} = {rate:.3f}, rounded to whole percent"})

    ax.set_xticks([0, 15, 30, 45, 60])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="x", labelsize=13, length=0, pad=6)
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    for xv in (15, 30, 45):
        ax.axvline(xv, color=GRID, lw=1, zorder=0)
    ax.set_xlabel(f"Share of the {n} held-out EVAL runs", fontsize=13, color=INK, labelpad=6)

    # key: two rows, same order as the segments, row 2 reuses row 1's columns
    rows = [OUTCOME_ORDER[:4], OUTCOME_ORDER[4:]]
    y0 = 0.94
    col_x = []
    for r, row in enumerate(rows):
        kx = 0.25
        for ci, k in enumerate(row):
            if r > 0:
                kx = col_x[ci]
            else:
                col_x.append(kx)
            yy = y0 - r * 0.062
            fig.patches.append(Rectangle((kx, yy - 0.0155), 0.018, 0.031, transform=fig.transFigure,
                                         facecolor=OUTCOME_COLOURS[k], edgecolor="none"))
            txt = fig.text(kx + 0.025, yy, OUTCOME_NAMES[k], fontsize=13, va="center", ha="left", color=INK)
            fig.canvas.draw()
            bb = txt.get_window_extent().transformed(fig.transFigure.inverted())
            kx = bb.x1 + 0.04

    bottom_notes(fig, [
        ("Stopped short: declared arrival on its own pose estimate but ended outside the 2 m success radius.", MUTED),
        ("Stuck: < 0.5 m progress in 20 s.   Out of bounds: drove off the edge of the simulated world.", MUTED),
        ("Simulated · tier-0 depth sensor, no images · " + PROV_SPLIT, FAINT),
        (f"Counts: {EVAL_CSV} = {EVAL_JSON}, registered in {CLAIMS_CSV}", FAINT),
    ], step=0.046)
    size = save(fig, "eval_outcomes")
    return {"figure": "eval_outcomes", "size": size, "numbers": drawn}


# ---- DEV numbers for the sidecar ------------------------------------------------------------
def dev_numbers() -> dict:
    d = json.loads((ROOT / DEV_JSON).read_text())
    out = {"source": DEV_JSON, "split": "dev", "seeds": d["seeds"],
           "note": "DEV seeds 100-129 were used for tuning; EVAL seeds 0-59 were not. Not drawn and NOT registered "
                   f"in {CLAIMS_CSV}: register before quoting on a slide.", "configs": {}}
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


def f4_collision_objects() -> dict[str, dict[int, str]]:
    """Object hit in each F4 collision run (GT referee events); asserts all are the moving F4 object."""
    rows = [r for r in load_runs() if r["family"] == "F4_sudden_obstacle" and r["outcome"] == "collision"]
    out: dict[str, dict[int, str]] = {"FULL": {}, "TYPICAL": {}}
    for r in rows:
        ev = json.loads((ROOT / "results/runs_eval_tier0" / r["config_name"] / f"{r['seed']:03d}" / "gt" /
                         "events.json").read_text())
        ev = ev if isinstance(ev, list) else ev.get("events", [])
        objs = [e["object"] for e in ev if e.get("type") == "collision"]
        assert objs and all(o.startswith("dyn_") for o in objs), (r["config_name"], r["seed"], objs)
        out[r["config_name"]][r["seed"]] = objs[0]
    return out


def context(rows: list[dict], t: dict) -> dict:
    """Facts behind the headline/caption wording that are not drawn (for slide writers)."""
    by = {(r["config_name"], r["seed"]): r for r in rows}
    oob_seeds = sorted(r["seed"] for r in rows if r["config_name"] == "TYPICAL" and r["outcome"] == "out_of_bounds")
    full_ditch = sorted(r["seed"] for r in rows if r["config_name"] == "FULL" and r["outcome"] == "ditch_entry")
    assert all(by[("FULL", s)]["family"] == "F3_crest_ditch" and (s // 6) % 2 == 0 for s in full_ditch), full_ditch
    return {
        "TYPICAL_out_of_bounds_seeds": oob_seeds,
        "FULL_outcome_on_those_seeds": dict(Counter(by[("FULL", s)]["outcome"] for s in oob_seeds)),
        "FULL_ditch_entry_seeds": full_ditch,
        "FULL_ditch_entry_note": "both are F3 seeds with a hidden trench (seed // 6 even), i.e. the case the "
                                 "missing-ground detector targets",
        "false_stops": {c: t[c]["false_stops"] for c in ("FULL", "TYPICAL")},
        "false_stops_claim_ids": [claim(f"{c}_false_stops", t[c]["false_stops"]) for c in ("FULL", "TYPICAL")],
        "paired_F2_F3": paired(rows, ("F2_ditch_field", "F3_crest_ditch")),
        "paired_F4": paired(rows, ("F4_sudden_obstacle",)),
        "how": f"computed from {EVAL_CSV}; only false_stops is registered in {CLAIMS_CSV}",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CLAIMS.update(load_claims())
    rows = load_runs()
    t = tally(rows)
    # F3 descriptor ("hidden trench in 5 of the 10 runs"): trench present for even seed // 6 (scenario.py)
    f3_seeds = sorted({r["seed"] for r in rows if r["family"] == "F3_crest_ditch"})
    f3_trench = [sd for sd in f3_seeds if (sd // 6) % 2 == 0]
    assert (len(f3_trench), len(f3_seeds)) == (5, 10), (f3_trench, f3_seeds)
    ev = json.loads((ROOT / EVAL_JSON).read_text())
    cross_check(t, ev["tier0"]["aggregate"])
    common = {
        "label": "Simulated",
        "split": f"eval seeds 0-59, run 2 (commit 45ec399). Run 2 is a disclosed protocol deviation: run 1 "
                 f"(superseded) scored 17/60 vs 15/60; see {PREREG}",
        "sensor": "tier-0 synthetic depth sensor (no images: no VO, no segmentation)",
        "configs": {"FULL": "default stack: seen ground only, missing-ground detector, seen-distance speed governor",
                    "TYPICAL": "unknown_is_free, no missing-ground detector, no governor, fixed 1.5 m/s"},
        "sources": [EVAL_CSV, EVAL_JSON, CLAIMS_CSV, DEV_JSON],
        "cross_check": "per-run CSV counts equal the JSON aggregates and the registered claims (asserted here)",
        "scope": "EVAL only. Do not generalise 'never out of bounds' beyond these 60 runs (see dev_split_not_drawn).",
        "context_not_drawn": context(rows, t),
        "dev_split_not_drawn": dev_numbers(),
        "generator": "deck_assets/final/src/closed_loop.py",
    }
    fam_spec = fig_by_family(t)
    fam_spec["numbers"].append({
        "what": "F3 EVAL seeds with a hidden trench behind the crest (family descriptor)",
        "value": len(f3_trench), "of": len(f3_seeds), "seeds": f3_trench,
        "claim_id": None, "source": "metagross/sim/scenario.py (f3_has_trench) + " + EVAL_CSV,
        "how": "scenario design, not a result: F3 seeds with (seed // 6) even; odd ones are crest-only controls"})
    fam_spec["descriptors_source"] = ("scenario design parameters, not results: metagross/sim/scenario.py "
                                      "(docstring, F2_*/F3_*/F4_TRIGGER_M constants, _f1_trail_objects, _f4_dynamic) "
                                      "and metagross/sim/sensors.py (tier-0 lighting = depth dropout, noise, "
                                      "dust phantom returns)")
    out_spec = fig_outcomes(t)
    out_spec["term_definitions"] = {
        "Stopped short of B": "referee arrived_short: the stack declared ARRIVED on its own pose estimate but the "
                              "body ended outside the 2 m success radius",
        "Stuck": "referee stuck: < 0.5 m progress in 20 s",
        "Out of bounds": "referee out_of_bounds: body origin left the simulated terrain raster (0.5 m margin, "
                         "metagross/sim/referee.py); not the robot's own rolling map",
    }
    slot_spec = fig_by_family_slot(t)
    for spec in (fam_spec, out_spec, slot_spec):
        (OUT / f"{spec['figure']}.json").write_text(json.dumps({**spec, **common}, indent=2) + "\n")
        print("wrote", OUT / f"{spec['figure']}.png", spec["size"])


if __name__ == "__main__":
    main()
