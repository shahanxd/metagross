"""KITTI camera-only stereo VO: drift per sequence + top-view trajectories (final deck figure).

Run from the repo root:
    OMP_NUM_THREADS=2 python deck_assets/final/src/kitti_drift.py            # with literature reference lines
    OMP_NUM_THREADS=2 python deck_assets/final/src/kitti_drift.py --no-literature   # ours only

Outputs (deck_assets/final/):
    kitti_drift.png / .svg / .json            (or kitti_drift_plain.* with --no-literature)

Numbers (Tested, real KITTI stereo, camera only, no GPS, no loop closure):
    t_err, r_err, path length, frames, segment counts  <- results/kitti_vo_{00,05,07}.json
    pooled t_err over all three sequences              <- results/kitti_summary.json#pooled
    literature reference lines (optional, NOT ours)    <- results/kitti_summary.json#literature_context

Trajectories: the per-frame pose files (results/raw/kitti_vo_<seq>_poses.txt) and the KITTI GT
(data/kitti/poses/<seq>.txt) are not in this container (data/ and results/raw/ are git-ignored and
were only on the team laptop). The polylines are therefore recovered from the vector paths of
deck_assets/kitti_traj_panel.svg, which deck_assets/kitti_figures.py drew from exactly those files
(GT and VO, first pose aligned, no scale or rotation fit). SVG points are converted back to metres
with that panel's 100 m scale bar; the y axis is flipped back (SVG y grows downward). The recovery
is checked against the JSONs (GT polyline length vs path_length_m, end-point gap vs final_err_m) and
the script stops if the length check is off by more than 1 %. The polylines are display resolution
(matplotlib path simplification), so they are used for drawing only: no metric is computed from them.
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import transforms  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "deck_assets" / "final"
SEQS = ("00", "05", "07")
SEQ_JSON = "results/kitti_vo_{seq}.json"
SUMMARY_JSON = "results/kitti_summary.json"
TRAJ_SVG = "deck_assets/kitti_traj_panel.svg"
SCALE_BAR_M = 100.0  # length of the scale bar in kitti_traj_panel.svg (kitti_figures.SCALE_BAR_M)
LENGTH_TOL = 0.01  # recovered GT polyline length must match path_length_m within 1 %

# ---- palette (deck brief) ------------------------------------------------------------------
NAVY = "#1F497D"      # ours
GREY = "#9A9A9A"      # reference / baseline
INK = "#1A1A1A"
MUTED = "#5C5C5C"
FAINT = "#8A8A8A"
GRID = "#E6E6E6"
BG = "#FFFFFF"
GT_TRACK = "#B5B5B5"  # ground-truth track: a wide light-grey road under our navy line

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "svg.fonttype": "path",
    "figure.facecolor": BG,
    "axes.facecolor": BG,
})
DPI = 200


# ---- data ----------------------------------------------------------------------------------
def load_metrics() -> tuple[dict, dict]:
    runs = {s: json.loads((ROOT / SEQ_JSON.format(seq=s)).read_text()) for s in SEQS}
    summary = json.loads((ROOT / SUMMARY_JSON).read_text())
    for s in SEQS:  # the summary must agree with the per-run files
        assert abs(summary["sequences"][s]["t_err_pct"] - runs[s]["t_err_pct"]) < 1e-9, s
    return runs, summary


def _svg_points(d: str) -> np.ndarray:
    return np.array([[float(a), float(b)] for a, b in re.findall(r"[ML]\s*([-\d.eE]+)\s+([-\d.eE]+)", d)])


def load_trajectories(runs: dict) -> tuple[dict, list]:
    """GT and VO top-view polylines (metres, x right / z forward, start at the origin) per sequence."""
    ns = "{http://www.w3.org/2000/svg}"
    root = ET.parse(ROOT / TRAJ_SVG).getroot()
    axes = [g for g in root.iter(ns + "g") if g.get("id", "").startswith("axes_")]
    assert len(axes) == len(SEQS), "unexpected panel count in kitti_traj_panel.svg"
    out, checks = {}, []
    for seq, ax in zip(SEQS, axes):
        lines = []
        for g in ax:
            if g.get("id", "").startswith("line2d"):
                p = g.find(ns + "path")
                if p is not None:
                    lines.append((p.get("style", ""), _svg_points(p.get("d"))))
        # drawing order in kitti_figures.fig_traj_panel: GT, VO, then the scale bar (+ 2 end ticks)
        (gt_style, gt), (vo_style, vo), (_, bar) = lines[0], lines[1], lines[2]
        assert "#0f172a" in gt_style and "#2563eb" in vo_style, f"{seq}: line colours not as expected"
        pt_per_m = (bar[1, 0] - bar[0, 0]) / SCALE_BAR_M
        o = gt[0]

        def to_m(a: np.ndarray) -> np.ndarray:
            return np.c_[(a[:, 0] - o[0]) / pt_per_m, -(a[:, 1] - o[1]) / pt_per_m]

        G, V = to_m(gt), to_m(vo)
        L = float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum())
        L_ref = runs[seq]["path_length_m"]
        gap = float(np.linalg.norm(G[-1] - V[-1]))
        assert abs(L / L_ref - 1) < LENGTH_TOL, f"{seq}: recovered GT length {L:.1f} m vs {L_ref:.1f} m"
        checks.append({"sequence": seq, "points_gt": len(G), "points_vo": len(V), "svg_pt_per_m": pt_per_m,
                       "recovered_gt_length_m": round(L, 1), "json_path_length_m": round(L_ref, 1),
                       "recovered_end_gap_m": round(gap, 1), "json_final_err_m": round(runs[seq]["final_err_m"], 1)})
        out[seq] = (G, V)
    return out, checks


# ---- drawing helpers -----------------------------------------------------------------------
def round_end_bar(ax, x0, x1, yc, h, colour, r_px=12):
    """Horizontal bar, rounded at the data end, square at the baseline (radius in pixels)."""
    inv = ax.transData.inverted()
    p0, p1 = inv.transform((0, 0)), inv.transform((r_px, r_px))
    rx, ry = min(abs(p1[0] - p0[0]), (x1 - x0) / 2), min(abs(p1[1] - p0[1]), h / 2)
    y0, y1 = yc - h / 2, yc + h / 2
    verts = [(x0, y0), (x1 - rx, y0), (x1, y0), (x1, y0 + ry), (x1, y1 - ry), (x1, y1), (x1 - rx, y1),
             (x0, y1), (x0, y0), (0, 0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=colour, edgecolor="none", zorder=3))


def fit_equal(ax, xy: np.ndarray, pad=0.07, bottom_band=0.16):
    """Equal-aspect limits that fill the axes box, with a free band at the bottom for the scale bar."""
    fig = ax.figure
    bb = ax.get_position()
    box = (bb.width * fig.get_figwidth()) / (bb.height * fig.get_figheight())
    x0, y0 = xy.min(0)
    x1, y1 = xy.max(0)
    w, h = x1 - x0, y1 - y0
    x0, x1, y0, y1 = x0 - pad * w, x1 + pad * w, y0 - (pad + bottom_band) * h, y1 + pad * h
    w, h = x1 - x0, y1 - y0
    if w / h < box:
        e = (box * h - w) / 2
        x0, x1 = x0 - e, x1 + e
    else:
        e = (w / box - h) / 2
        y0, y1 = y0 - e, y1 + e
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")


def scale_bar(ax, length_m=100.0, fontsize=11.5):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr = x1 - 0.05 * (x1 - x0)
    xl = xr - length_m
    yb = y0 + 0.06 * (y1 - y0)
    ax.plot([xl, xr], [yb, yb], color=INK, lw=2.2, solid_capstyle="butt", zorder=6)
    for x in (xl, xr):
        ax.plot([x, x], [yb - 0.015 * (y1 - y0), yb + 0.015 * (y1 - y0)], color=INK, lw=1.2, zorder=6)
    ax.text((xl + xr) / 2, yb + 0.03 * (y1 - y0), f"{length_m:.0f} m", ha="center", va="bottom", fontsize=fontsize,
            color=INK, zorder=6)


def km(m: float) -> str:
    return f"{m / 1000:.2f} km" if m >= 1000 else f"{m:.0f} m"


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.12, facecolor=BG)
    plt.close(fig)


# ---- figure --------------------------------------------------------------------------------
# text sizes in points at DPI 200 (13 pt = 36 px in the PNG; ~18 px when the figure is shown 900 px wide)
FS_MIN = 13.0     # footnotes, sub-labels, literature labels, provenance, scale-bar labels
FS_TICK = 13.5
FS_LABEL = 14.0
FS_NAME = 14.5
FS_VALUE = 15.0
LIT_LINE = "#5C5C5C"  # literature reference lines: dashed dark grey (the wide light-grey GT track means something else)
LIT_DASH = (0, (5, 3))
VALUE_ON_BAR = "#FFFFFF"  # value labels sit inside the navy bars (contrast ~9:1), clear of the reference lines


def _fig_y0(fig, artist) -> float:
    fig.canvas.draw()
    return artist.get_window_extent().transformed(fig.transFigure.inverted()).y0


def build(literature: bool) -> dict:
    runs, summary = load_metrics()
    traj, checks = load_trajectories(runs)
    pooled = summary["pooled"]
    lit_rows = {r["method"]: r for r in summary["literature_context"]["rows"]}
    drawn = []

    # vertical layout in inches from the top, so both variants get the same bar thickness and no empty band
    top = -1.35 if literature else -0.55  # bar-axes top in data units (room for the literature labels)
    y_last = 3 + 0.2  # pooled row sits a little lower than the three sequences, under a thin separator
    y_bottom = y_last + 0.5
    KEY_IN, HEAD_IN, MAP_IN, GAP_IN, UNIT_IN, BOTTOM_IN = 0.42, 0.36, 2.6, 0.3, 0.58, 1.9
    bars_in = (y_bottom - top) * UNIT_IN
    H = KEY_IN + HEAD_IN + MAP_IN + GAP_IN + bars_in + BOTTOM_IN
    fig = plt.figure(figsize=(9.0, H), dpi=DPI)

    def fy(inches_from_top: float) -> float:
        return 1 - inches_from_top / H

    # key for the trajectory panels
    ky = fy(KEY_IN / 2)
    kx = 0.012
    for colour, lw, label in ((GT_TRACK, 5.0, "Ground truth (GPS/INS)"), (NAVY, 2.0, "Our stereo VO, camera only")):
        fig.add_artist(Line2D([kx, kx + 0.04], [ky, ky], transform=fig.transFigure, color=colour, lw=lw,
                              solid_capstyle="round"))
        t = fig.text(kx + 0.05, ky, label, fontsize=FS_LABEL, va="center", ha="left", color=INK)
        fig.canvas.draw()
        kx = t.get_window_extent().transformed(fig.transFigure.inverted()).x1 + 0.035
    fig.add_artist(Line2D([kx + 0.008], [ky], transform=fig.transFigure, ls="none", marker="o", ms=8,
                          color=INK, mec=BG, mew=1.5))
    fig.text(kx + 0.024, ky, "start", fontsize=FS_LABEL, va="center", ha="left", color=INK)

    # trajectory small multiples
    w, gap, left = 0.318, 0.023, 0.012
    for i, seq in enumerate(SEQS):
        G, V = traj[seq]
        res = runs[seq]
        ax = fig.add_axes([left + i * (w + gap), fy(KEY_IN + HEAD_IN + MAP_IN), w, MAP_IN / H])
        ax.plot(G[:, 0], G[:, 1], color=GT_TRACK, lw=5.0, solid_capstyle="round", solid_joinstyle="round", zorder=2)
        ax.plot(V[:, 0], V[:, 1], color=NAVY, lw=1.9, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax.plot([0], [0], "o", ms=8, color=INK, mec=BG, mew=1.6, zorder=5)
        fit_equal(ax, np.vstack([G, V]), bottom_band=0.19)
        scale_bar(ax, SCALE_BAR_M, fontsize=FS_MIN)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        name = f"KITTI {seq}" + ("*" if not res.get("full_sequence", True) else "")
        ax.text(0.0, 1.035, name, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_NAME,
                fontweight="bold", color=INK)
        ax.text(1.0, 1.035, km(res["path_length_m"]), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FS_LABEL, color=MUTED)
        drawn.append({"what": f"KITTI {seq} top-view trajectory, GT and VO (drawn, not measured here)",
                      "value": f"{len(G)} GT / {len(V)} VO polyline points",
                      "source": TRAJ_SVG,
                      "how": "vector paths of the panel drawn by deck_assets/kitti_figures.py from "
                             f"results/raw/kitti_vo_{seq}_poses.txt and data/kitti/poses/{seq}.txt (absent here); "
                             "converted to metres with its 100 m scale bar"})
        drawn.append({"what": f"KITTI {seq} distance driven (map header)", "value": km(res["path_length_m"]),
                      "source": SEQ_JSON.format(seq=seq) + "#path_length_m", "how": "GT path length"})

    # drift bars
    ax = fig.add_axes([0.30, fy(KEY_IN + HEAD_IN + MAP_IN + GAP_IN + bars_in), 0.64, bars_in / H])
    rows = [(f"KITTI {s}" + ("*" if not runs[s].get("full_sequence", True) else ""),
             f"{km(runs[s]['path_length_m'])} · {runs[s]['frames']:,} frames",
             runs[s]["t_err_pct"], runs[s]["r_err_deg_per_100m"], runs[s]["n_segments"], s) for s in SEQS]
    total_m = sum(runs[s]["path_length_m"] for s in SEQS)
    rows.append(("All three, pooled", f"{km(total_m)} · {pooled['n_segments']:,} segments",
                 pooled["t_err_pct"], pooled["r_err_deg_per_100m"], pooled["n_segments"], "pooled"))
    xmax = 2.6
    assert len(rows) == 4
    ax.set_xlim(0, xmax)
    ax.set_ylim(y_bottom, top)
    fig.canvas.draw()
    bar_h = 0.52
    tr = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for i, (name, sub, terr, rerr, nseg, key) in enumerate(rows):
        yc = y_last if key == "pooled" else i
        round_end_bar(ax, 0, terr, yc, bar_h, NAVY)
        bold = key == "pooled"
        ax.text(terr - 0.045, yc, f"{terr:.2f} %", va="center", ha="right", fontsize=FS_VALUE, color=VALUE_ON_BAR,
                fontweight="bold", zorder=5)
        ax.text(-0.03, yc, name, transform=transforms.offset_copy(tr, fig=fig, y=9.0, units="points"),
                ha="right", va="center", fontsize=FS_NAME, color=INK, fontweight="bold" if bold else "normal")
        ax.text(-0.03, yc, sub, transform=transforms.offset_copy(tr, fig=fig, y=-10.0, units="points"),
                ha="right", va="center", fontsize=FS_MIN, color=MUTED)
        src = SUMMARY_JSON + "#pooled" if key == "pooled" else SEQ_JSON.format(seq=key)
        drawn.append({"what": f"{name}: KITTI-protocol translational error t_err (bar and label)",
                      "value": round(terr, 2), "unit": "%", "source": src + ("" if key == "pooled" else "#t_err_pct"),
                      "how": ("segment-weighted mean over all 100-800 m segments of 00*, 05, 07" if key == "pooled"
                              else f"mean over {nseg} segments of 100-800 m, KITTI devkit protocol")})
        drawn.append({"what": f"{name}: label under the name", "value": sub,
                      "source": src, "how": "path_length_m (GT) and frames / n_segments; pooled km = sum of the three"})
        drawn.append({"what": f"{name}: rotational error r_err (json only, not drawn)", "value": round(rerr, 2),
                      "unit": "deg/100 m", "source": src, "how": "KITTI devkit protocol"})
    ax.plot([-0.06, xmax], [len(rows) - 1 - 0.4, len(rows) - 1 - 0.4], color=GRID, lw=1.0, zorder=1, clip_on=False)

    if literature:
        for method, lines in (("ORB-SLAM2 (stereo)", ("ORB-SLAM2 stereo: {v:.2f} %", "(full SLAM, loop closure)")),
                              ("VISO2-S", ("VISO2-S: {v:.2f} %", "(frame-to-frame stereo VO)"))):
            v = lit_rows[method]["t_err_pct"]
            ax.plot([v, v], [top + 0.05, y_bottom], color=LIT_LINE, lw=1.3, ls=LIT_DASH, zorder=1)
            ax.text(v - 0.035, top + 0.12, lines[0].format(v=v) + "\n" + lines[1], ha="right", va="top",
                    fontsize=FS_MIN, color=MUTED, linespacing=1.25)
            drawn.append({"what": f"Literature reference line: {method}", "value": v, "unit": "%",
                          "label": "Literature (NOT our measurement)",
                          "source": SUMMARY_JSON + "#literature_context (KITTI leaderboard, "
                                    "https://www.cvlibs.net/datasets/kitti/eval_odometry.php, retrieved 2026-09-30)",
                          "how": "leaderboard value on held-out test sequences 11-21; ours are training sequences, "
                                 "so the comparison is indicative only"})

    ax.set_xticks(np.arange(0, xmax + 1e-9, 0.5))
    ax.set_xticklabels([f"{x:g}" for x in np.arange(0, xmax + 1e-9, 0.5)])
    ax.tick_params(axis="x", labelsize=FS_TICK, length=0, pad=6)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for x in np.arange(0.5, xmax + 1e-9, 0.5):
        ax.plot([x, x], [top if not literature else -0.55, y_bottom], color=GRID, lw=1, zorder=0)
    ax.set_xlabel("Translational error, % of segment length\n(KITTI protocol: mean over all 100-800 m segments)",
                  fontsize=FS_LABEL, color=INK, labelpad=8, linespacing=1.3)

    # footnotes directly under the x label, provenance under them (no empty band in either variant)
    note = ["*00: frames 1101-4540 only; the downloaded mirror's first 1,101 frames are not sequence 00.",
            "Maps: aligned at the start pose only, no scale or rotation fit."]
    if literature:
        note.append("Dashed lines: KITTI leaderboard on test sequences 11-21 (literature, not our runs).")
    foot = fig.text(0.012, _fig_y0(fig, ax.xaxis.label) - 0.02, "\n".join(note), fontsize=FS_MIN, color=MUTED,
                    ha="left", va="top", linespacing=1.3)
    prov = "Tested on KITTI odometry · results/kitti_vo_{00,05,07}.json, results/kitti_summary.json"
    fig.text(0.012, _fig_y0(fig, foot) - 0.014, prov, fontsize=FS_MIN, color=MUTED, ha="left", va="top")
    name = "kitti_drift" if literature else "kitti_drift_plain"
    save(fig, name)
    with Image.open(OUT / f"{name}.png") as im:
        png_px = list(im.size)
    json_only = {
        "end_point_error_m_vs_distance": {
            s: {"final_err_m": round(runs[s]["final_err_m"], 1), "path_length_m": round(runs[s]["path_length_m"], 1),
                "ate_max_m": round(runs[s]["ate_max_m"], 1)} for s in SEQS},
        "note": "NOT drawn and NOT in results/claims.csv. t_err is a mean relative error over 100-800 m sub-segments, "
                "not the error at the end of the 5.8 km; end-point errors are listed so nobody reads 1.87 % as "
                "'109 m after 5.8 km'. Source: results/kitti_vo_0X.json#final_err_m, #ate_max_m.",
        "vo_config_caveat": "These runs used the VO config logged in results/kitti_vo_0X.json#vo_config (KLT 21 px / "
                            "30 iterations, no photometric normalisation). The current onboard default (KLT 15 px / "
                            "10 iterations, photometric normalisation on) was re-measured on 07 only: 2.037 % "
                            "(results/kitti_vo_07_check.json, claims.csv kitti07_t_err_pct_current_vo). 00* and 05 "
                            "were not re-run with it.",
    }
    info = {"figure": name, "label": "Tested", "literature_lines": literature, "png_px": png_px,
            "provenance": "Tested on KITTI odometry (real stereo, camera only, no GPS, no loop closure); "
                          "00 evaluated on frames 1101-4540",
            "numbers": drawn, "json_only_not_drawn": json_only, "trajectory_recovery_checks": checks,
            "trajectory_source_note": "results/raw/kitti_vo_<seq>_poses.txt and data/kitti/poses/<seq>.txt are not "
                                      "present in this container; polylines recovered from " + TRAJ_SVG +
                                      " (see this script's docstring). Metrics come only from the JSONs."}
    if literature:
        info["literature_registration_needed"] = (
            "ORB-SLAM2 1.15 % and VISO2-S 2.44 % are in results/kitti_summary.json#literature_context but NOT in "
            "results/claims.csv (label Literature), and ORB-SLAM2 (Mur-Artal & Tardos, IEEE T-RO 33(5) 2017) and "
            "VISO2/StereoScan (Geiger, Ziegler & Stiller, IEEE IV 2011) are not in docs/REFERENCES.md. Register and "
            "cite them before using this variant, or use kitti_drift_plain.")
    return info


SLOT_W_PX, SLOT_H_PX = 652, 470  # box on the 1920x1080 slide (slide 3, right column)
SLOT_PX_PER_IN = 100.0  # 1 figure inch = 100 slide px, so a font of p points shows as p * 100/72 slide px


def build_slot() -> dict:
    """Slide-3 version composed for a 652x470 slide box: labels >= 16 slide px, no footnotes (they go in the caption)."""
    runs, summary = load_metrics()
    traj, checks = load_trajectories(runs)
    pooled = summary["pooled"]
    drawn = []
    fig = plt.figure(figsize=(SLOT_W_PX / SLOT_PX_PER_IN, SLOT_H_PX / SLOT_PX_PER_IN), dpi=DPI)

    # key
    ky, kx = 0.965, 0.005
    for colour, lw, label in ((GT_TRACK, 6.0, "Ground truth"), (NAVY, 2.4, "Our stereo VO, camera only")):
        fig.add_artist(Line2D([kx, kx + 0.05], [ky, ky], transform=fig.transFigure, color=colour, lw=lw,
                              solid_capstyle="round"))
        t = fig.text(kx + 0.062, ky, label, fontsize=13, va="center", ha="left", color=INK)
        fig.canvas.draw()
        kx = t.get_window_extent().transformed(fig.transFigure.inverted()).x1 + 0.04
    fig.add_artist(Line2D([kx + 0.008], [ky], transform=fig.transFigure, ls="none", marker="o", ms=8,
                          color=INK, mec=BG, mew=1.5))
    fig.text(kx + 0.028, ky, "start", fontsize=13, va="center", ha="left", color=INK)

    # trajectory small multiples
    w, gap, left = 0.31, 0.02, 0.006
    for i, seq in enumerate(SEQS):
        G, V = traj[seq]
        res = runs[seq]
        ax = fig.add_axes([left + i * (w + gap), 0.515, w, 0.36])
        ax.plot(G[:, 0], G[:, 1], color=GT_TRACK, lw=5.0, solid_capstyle="round", solid_joinstyle="round", zorder=2)
        ax.plot(V[:, 0], V[:, 1], color=NAVY, lw=1.9, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax.plot([0], [0], "o", ms=7, color=INK, mec=BG, mew=1.4, zorder=5)
        fit_equal(ax, np.vstack([G, V]), bottom_band=0.05)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        star = "*" if not res.get("full_sequence", True) else ""
        ax.text(0.0, 1.03, f"{seq}{star}", transform=ax.transAxes, ha="left", va="bottom", fontsize=13.5,
                fontweight="bold", color=INK)
        ax.text(1.0, 1.03, km(res["path_length_m"]), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=12.5, color=MUTED)
        drawn.append({"what": f"KITTI {seq} top-view trajectory, GT and VO (drawn, not measured here)",
                      "value": f"{len(G)} GT / {len(V)} VO polyline points", "source": TRAJ_SVG,
                      "how": "same recovery as kitti_drift_plain (see docstring)"})

    # drift bars
    ax = fig.add_axes([0.20, 0.13, 0.70, 0.31])
    rows = [(f"KITTI {s}" + ("*" if not runs[s].get("full_sequence", True) else ""), runs[s]["t_err_pct"], s)
            for s in SEQS]
    rows.append(("Pooled", pooled["t_err_pct"], "pooled"))
    xmax = 2.6
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(rows) - 0.4, -0.5)
    fig.canvas.draw()
    for i, (name, terr, key) in enumerate(rows):
        yc = i + (0.15 if key == "pooled" else 0)
        round_end_bar(ax, 0, terr, yc, 0.56, NAVY, r_px=10)
        bold = key == "pooled"
        ax.text(terr + 0.04, yc, f"{terr:.2f} %", va="center", ha="left", fontsize=14, color=INK, fontweight="bold",
                zorder=5)
        ax.text(-0.05, yc, name, ha="right", va="center", fontsize=13, color=INK,
                fontweight="bold" if bold else "normal")
        src = SUMMARY_JSON + "#pooled" if key == "pooled" else SEQ_JSON.format(seq=key) + "#t_err_pct"
        drawn.append({"what": f"{name}: KITTI-protocol translational error t_err (bar)", "value": round(terr, 2),
                      "unit": "%", "source": src})
    ax.set_xticks(np.arange(0, xmax + 1e-9, 0.5))
    ax.set_xticklabels([f"{x:g}" for x in np.arange(0, xmax + 1e-9, 0.5)])
    ax.tick_params(axis="x", labelsize=11.5, length=0, pad=4)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for x in np.arange(0.5, xmax + 1e-9, 0.5):
        ax.plot([x, x], [-0.5, len(rows) - 0.4], color=GRID, lw=1, zorder=0)
    ax.set_xlabel("Drift, % of distance travelled (KITTI protocol)", fontsize=12, color=INK, labelpad=5)
    name = "kitti_slot"
    for ext in ("png", "svg"):  # fixed canvas (no tight bbox) so the PNG is exactly the slot aspect
        fig.savefig(OUT / f"{name}.{ext}", dpi=DPI * 1.5, facecolor=BG)
    plt.close(fig)
    return {"figure": name, "label": "Tested", "slot_px": [SLOT_W_PX, SLOT_H_PX],
            "provenance": "Tested on KITTI odometry (real stereo, camera only, no GPS, no loop closure); "
                          "00 evaluated on frames 1101-4540",
            "numbers": drawn, "trajectory_recovery_checks": checks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-literature", action="store_true", help="omit the KITTI leaderboard reference lines")
    ap.add_argument("--slot", action="store_true", help="slide-3 slot version (kitti_slot.*), no literature lines")
    args = ap.parse_args()
    info = build_slot() if args.slot else build(literature=not args.no_literature)
    (OUT / f"{info['figure']}.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {OUT / info['figure']}.png/.svg/.json")


if __name__ == "__main__":
    main()
