"""Deck figures and summary for the camera-only stereo VO evaluation on real KITTI sequences.

Reads only generated results (never re-runs VO):

* ``results/kitti_vo_<seq>.json`` - metrics written by :mod:`metagross.eval.kitti_vo`,
* ``results/raw/kitti_vo_<seq>_poses.txt`` - the estimated poses of that run,
* ``data/kitti/poses/<seq>.txt`` - KITTI ground truth (GPS/INS).

Writes (PNG at 2x slide size, 300 dpi, plus SVG, via :func:`plot_style.save_fig`):

* ``deck_assets/kitti_traj_panel`` (wide, 1600x600) - top-view trajectories, one small multiple per sequence,
* ``deck_assets/kitti_drift_table`` (wide, 1600x600) - segment drift bars at 100/500/1000 m + metrics table,
* ``deck_assets/kitti_drift_card`` (card, 900x520) - drift (% of distance) vs segment length,
* ``deck_assets/slots/s3_kitti.png`` (slide slot 564x326, PNG only, exactly 1128x652 px) - the same drift
  curves re-laid out for the slot with every text >= 26 px (``--slot-only`` renders just this file),
* ``results/kitti_summary.json`` - every number shown, how it was measured, a claims list (Tested) and
  a clearly separated literature block (Literature, KITTI leaderboard, not our measurement).

Frames: KITTI camera-0 poses (x right, y down, z forward); the top view is the x-z plane, metres.

Usage::

    python deck_assets/kitti_figures.py            # sequences 00 05 07
    python deck_assets/kitti_figures.py --slot-only   # only deck_assets/slots/s3_kitti.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python deck_assets/kitti_figures.py`
    sys.path.insert(0, str(REPO_ROOT))

from metagross.eval import plot_style as ps  # noqa: E402
from metagross.eval.traj_metrics import (  # noqa: E402
    KITTI_LENGTHS_M,
    STEP_SIZE,
    align_first,
    load_kitti_poses,
    segment_errors,
)

LOG = logging.getLogger("kitti_figures")

RESULTS = REPO_ROOT / "results"
KITTI_POSES = REPO_ROOT / "data" / "kitti" / "poses"
DEFAULT_SEQS = ("00", "05", "07")

CHIP_DETAIL = "KITTI odometry, camera-only stereo VO"
GT_COLOR = ps.TOKENS["text"]  # ground truth: near-black ink
OURS_COLOR = ps.TOKENS["vo_blue"]  # ours: VO blue
#: Per-sequence identity on the drift-vs-length card: blue family (all are "ours") + marker shape
#: + direct label, so identity never relies on colour alone.
SEQ_STYLE: dict[str, tuple[str, str]] = {"00": (ps.TOKENS["accent"], "o"), "05": ("#60A5FA", "s"),
                                         "07": (ps.TOKENS["navy"], "^")}
#: Sequential light -> dark blue for the three drift horizons (ordinal).
HORIZON_COLORS: dict[int, str] = {100: "#93B4F5", 500: ps.TOKENS["accent"], 1000: ps.TOKENS["navy"]}
HORIZONS_M: tuple[int, ...] = (100, 500, 1000)
CURVE_LENGTHS_M: tuple[int, ...] = tuple(range(100, 2001, 100))  # drift-vs-length card
MIN_SEGMENTS = 10  # a length is plotted only if at least this many segments of it exist
SCALE_BAR_M = 100.0  # same scale-bar length in every trajectory panel

#: Literature context - KITTI odometry leaderboard (test sequences 11-21, 100-800 m segments).
#: Verified against https://www.cvlibs.net/datasets/kitti/eval_odometry.php on 2026-09-30.
LITERATURE_ROWS: tuple[dict[str, Any], ...] = (
    {"method": "ORB-SLAM2 (stereo)", "setting": "stereo, full SLAM (BA + loop closure)",
     "t_err_pct": 1.15, "r_err_deg_per_m": 0.0027},
    {"method": "VISO2-S", "setting": "stereo VO (frame-to-frame)", "t_err_pct": 2.44, "r_err_deg_per_m": 0.0114},
    {"method": "VISO2-M", "setting": "monocular VO", "t_err_pct": 11.94, "r_err_deg_per_m": 0.0234},
)
#: Provenance of individual runs that differ from the plain CLI invocation (kept with the numbers).
RUN_NOTES: dict[str, str] = {
    "00": "Sub-sequence: frames 1101-4540 of KITTI 00 (python -m metagross.eval.kitti_vo --seqs 00 --threads 2 "
          "--first-frame 1101). Frames 000000-001100 of the Hugging Face mirror yujie2696/kitti_odometry_00 are "
          "not KITTI 00 (a highway drive; hard scene cut between 001100 and 001101; see data_issues). The mirror's "
          "times.txt has 1101 lines for 4541 frames, so VO used the nominal dt of VOConfig.kitti() (0.1 s); dt only "
          "enters VO's motion-plausibility gate.",
}
#: Frame index where the usable part of the mirrored KITTI 00 starts (first residential frame).
KITTI00_FIRST_VALID = 1101
#: Slide-slot variant of the drift card (deck slide 3): the slot is 564 x 326 slide px and the PNG is
#: rendered at plot_style.EXPORT_SCALE (2x) = 1128 x 652 px. Same data and style as the card, fewer
#: words, and every text >= SLOT_MIN_TEXT_PX at the exported size so it stays legible when projected.
SLOT_NAME = "slots/s3_kitti"
SLOT_SIZE_SLIDE_PX: tuple[int, int] = (564, 326)
SLOT_MIN_TEXT_PX = 26.0
#: Font sizes of the slot figure in exported PNG px (converted to pt by :func:`px_to_pt`).
SLOT_TEXT_PX: dict[str, float] = {"title": 34.0, "label": 28.0, "tick": 27.0, "legend": 27.0, "chip": 27.0,
                                  "direct": 28.0, "note": 26.0}
SLOT_X_TICKS_M: tuple[int, ...] = (0, 500, 1000, 1500, 2000)
SLOT_Y_TICKS_PCT: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)
SLOT_CHIP_DETAIL = "KITTI, camera only"
INVALID_00_POSES = RESULTS / "raw" / "kitti_vo_00_fullrun_invalid_poses.txt"
INVALID_00_JSON = RESULTS / "raw" / "kitti_vo_00_fullrun_invalid.json"
LITERATURE_SOURCE = "https://www.cvlibs.net/datasets/kitti/eval_odometry.php"
LITERATURE_RETRIEVED = "2026-09-30"


# --------------------------------------------------------------------------- data
def load_run(seq: str) -> dict[str, Any]:
    """Result JSON + GT + estimated poses for one sequence (GT sliced to the evaluated frames)."""
    res = json.loads((RESULTS / f"kitti_vo_{seq}.json").read_text(encoding="utf-8"))
    est = load_kitti_poses(RESULTS / "raw" / f"kitti_vo_{seq}_poses.txt")
    ff = int(res.get("first_frame", 0))
    gt = load_kitti_poses(KITTI_POSES / f"{seq}.txt")[ff: ff + est.shape[0]]
    if est.shape[0] != res["frames"]:
        raise ValueError(f"seq {seq}: {est.shape[0]} poses but JSON says {res['frames']} frames")
    return {"res": res, "gt": gt, "est": est}


def drift_curve(gt: np.ndarray, est: np.ndarray, lengths: tuple[int, ...] = CURVE_LENGTHS_M) -> dict[str, list]:
    """Mean end-point drift (% of L) over all KITTI-protocol segments of each length L (start every 10 frames)."""
    out: dict[str, list] = {"lengths_m": [], "mean_pct": [], "median_pct": [], "p90_pct": [], "n_segments": []}
    for L in lengths:
        seg = segment_errors(gt, est, lengths=(L,))
        if seg.shape[0] < MIN_SEGMENTS:
            continue
        pct = seg[:, 2] * 100.0
        out["lengths_m"].append(int(L))
        out["mean_pct"].append(float(pct.mean()))
        out["median_pct"].append(float(np.median(pct)))
        out["p90_pct"].append(float(np.percentile(pct, 90)))
        out["n_segments"].append(int(seg.shape[0]))
    return out


def mirror00_evidence() -> Optional[dict[str, Any]]:
    """Why frames 0-1100 of the mirrored 00 are excluded: per-frame VO-vs-GT relative-motion error of the
    archived full run (frame-to-frame, so each value depends only on frames k-1 and k)."""
    if not INVALID_00_POSES.exists():
        return None
    est = load_kitti_poses(INVALID_00_POSES)
    gt = load_kitti_poses(KITTI_POSES / "00.txt")[: est.shape[0]]
    rel_e = np.linalg.inv(est[:-1]) @ est[1:]
    rel_g = np.linalg.inv(gt[:-1]) @ gt[1:]
    err = np.linalg.norm((np.linalg.inv(rel_e) @ rel_g)[:, :3, 3], axis=1)  # err[k-1]: transition k-1 -> k
    b = KITTI00_FIRST_VALID
    full = json.loads(INVALID_00_JSON.read_text(encoding="utf-8")) if INVALID_00_JSON.exists() else {}
    return {
        "label": "Tested",
        "issue": "Hugging Face mirror yujie2696/kitti_odometry_00: image_0/image_1 frames 000000-001100 show a "
                 "highway drive, not KITTI 00 (residential); the scene cuts hard between 001100 and 001101, and "
                 "times.txt has 1101 lines. The 1101-frame count and content suggest KITTI 01 was written over the "
                 "start of 00 (inference, not verified against the official archive).",
        "per_frame_translation_error_m": {
            "how": "full 0-4540 VO run (archived in results/raw/kitti_vo_00_fullrun_invalid*.json/.txt); error of "
                   "each VO relative motion vs the KITTI 00 GT relative motion, |t(inv(dT_est) dT_gt)|",
            f"median_frames_1_to_{b - 1}": float(np.median(err[: b - 1])),
            f"p90_frames_1_to_{b - 1}": float(np.percentile(err[: b - 1], 90)),
            f"median_frames_{b + 1}_to_{est.shape[0] - 1}": float(np.median(err[b:])),
            f"max_frames_{b + 1}_to_{est.shape[0] - 1}": float(err[b:].max()),
        },
        "invalid_full_run_metrics_not_a_claim": {k: full.get(k) for k in ("t_err_pct", "r_err_deg_per_100m",
                                                                           "ate_rmse_m", "drift_1000m_pct")},
        "action": f"KITTI 00 is evaluated on frames {b}-4540 only (--first-frame {b}).",
    }


def pooled_kitti_errors(runs: dict[str, dict[str, Any]]) -> dict[str, float]:
    """KITTI t_err / r_err pooled over every 100..800 m segment of all given sequences (segment-weighted)."""
    segs = np.concatenate([segment_errors(r["gt"], r["est"], lengths=KITTI_LENGTHS_M) for r in runs.values()])
    return {"t_err_pct": float(segs[:, 2].mean() * 100.0),
            "r_err_deg_per_100m": float(np.degrees(segs[:, 1].mean()) * 100.0),
            "n_segments": int(segs.shape[0])}


# --------------------------------------------------------------------------- figures
def _fmt_km(m: float) -> str:
    return f"{m / 1000.0:.2f} km" if m >= 1000.0 else f"{m:.0f} m"


def _name(seq: str, res: dict[str, Any]) -> str:
    """'KITTI 00' or 'KITTI 00*' when only a sub-sequence was evaluated (explained by :func:`_subseq_note`)."""
    return f"KITTI {seq}*" if int(res.get("first_frame", 0)) else f"KITTI {seq}"


def _subseq_note(runs: dict[str, dict[str, Any]]) -> str:
    """Footnote text for sub-sequence runs ('' if every run is a full sequence)."""
    parts = [f"*{s}: frames {r['res']['first_frame']}-{r['res']['first_frame'] + r['res']['frames'] - 1} "
             f"(the downloaded mirror's first {r['res']['first_frame']} frames are not {s})"
             for s, r in runs.items() if int(r["res"].get("first_frame", 0))]
    return "; ".join(parts)


def _fit_limits(ax: Any, xs: np.ndarray, ys: np.ndarray, pad: float = 0.06, bottom_band: float = 0.14) -> None:
    """Equal-aspect limits that fill the axes box, with a free band at the bottom for the scale bar."""
    fig = ax.figure
    bb = ax.get_position()
    box_w, box_h = bb.width * fig.get_figwidth(), bb.height * fig.get_figheight()
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    x0, x1 = x0 - pad * w, x1 + pad * w
    y0, y1 = y0 - (pad + bottom_band) * h, y1 + pad * h
    w, h = x1 - x0, y1 - y0
    target = box_w / box_h  # data width / height that fills the box at equal aspect
    if w / h < target:
        extra = (target * h - w) / 2.0
        x0, x1 = x0 - extra, x1 + extra
    else:
        extra = w / target - h
        y0, y1 = y0 - extra * 0.5, y1 + extra * 0.5
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")


def _scale_bar(ax: Any, length_m: float) -> None:
    """Horizontal scale bar in the lower-right corner of an equal-aspect axes (data units = m)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr = x1 - 0.04 * (x1 - x0)
    xl = xr - length_m
    yb = y0 + 0.05 * (y1 - y0)
    ax.plot([xl, xr], [yb, yb], color=ps.TOKENS["text"], lw=2.0, solid_capstyle="butt", zorder=6)
    for x in (xl, xr):
        ax.plot([x, x], [yb - 0.012 * (y1 - y0), yb + 0.012 * (y1 - y0)], color=ps.TOKENS["text"], lw=1.0, zorder=6)
    ax.text((xl + xr) / 2.0, yb + 0.022 * (y1 - y0), f"{length_m:.0f} m", ha="center", va="bottom",
            fontsize=ps.FONT_PT["annot"], color=ps.TOKENS["text"], family=ps.mono_family(), zorder=6)


def fig_traj_panel(runs: dict[str, dict[str, Any]]) -> Any:
    """Three top-view small multiples: GT black, ours blue, first-pose aligned, 100 m scale bar."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=ps.size_inches("wide"))
    n = len(runs)
    axes = fig.subplots(1, n)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.74, bottom=0.10, wspace=0.06)
    mono = ps.mono_family()
    for ax, (seq, r) in zip(axes, runs.items()):
        gt, est, res = r["gt"], align_first(r["gt"], r["est"]), r["res"]
        gx, gz, ex, ez = gt[:, 0, 3], gt[:, 2, 3], est[:, 0, 3], est[:, 2, 3]
        ax.plot(gx, gz, color=GT_COLOR, lw=1.5, zorder=3)
        ax.plot(ex, ez, color=OURS_COLOR, lw=1.5, zorder=4)
        ax.plot([gx[0]], [gz[0]], "o", ms=6.5, color=GT_COLOR, mec=ps.TOKENS["bg"], mew=1.4, zorder=7)
        _fit_limits(ax, np.concatenate([gx, ex]), np.concatenate([gz, ez]))
        ax.annotate("start", (gx[0], gz[0]), xytext=(7, -3), textcoords="offset points", ha="left", va="top",
                    fontsize=ps.FONT_PT["annot"], color=ps.TOKENS["text"],
                    bbox=dict(boxstyle="square,pad=0.1", fc=ps.TOKENS["bg"], ec="none", alpha=0.85), zorder=8)
        _scale_bar(ax, SCALE_BAR_M)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ("left", "bottom", "top", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(ps.TOKENS["border"])
        ax.set_facecolor(ps.TOKENS["bg"])
        d1000 = res.get("drift_1000m_pct")
        d_txt = f"1 km drift {d1000:.2f} %" if d1000 is not None else f"500 m drift {res['drift_500m_pct']:.2f} %"
        ax.set_title(_name(seq, res), loc="left", pad=17, fontsize=ps.FONT_PT["title"])
        ax.text(0.0, 1.015, f"{_fmt_km(res['path_length_m'])} · t_err {res['t_err_pct']:.2f} % · {d_txt}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=ps.FONT_PT["small"], color=ps.TOKENS["text_secondary"], family=mono)
    fig.text(0.015, 0.975, "Camera-only stereo VO on real KITTI drives", ha="left", va="top",
             fontsize=ps.FONT_PT["title"] + 1.5, fontweight="bold", color=ps.TOKENS["text"])
    handles = [Line2D([], [], color=GT_COLOR, lw=1.8), Line2D([], [], color=OURS_COLOR, lw=1.8),
               Line2D([], [], ls="none", marker="o", ms=6, color=GT_COLOR, mec=ps.TOKENS["bg"])]
    fig.legend(handles, ["Ground truth (GPS/INS)", "METAGROSS stereo VO, camera only", "start"],
               loc="upper left", bbox_to_anchor=(0.012, 0.915), ncol=3, handlelength=2.2, columnspacing=1.8,
               fontsize=ps.FONT_PT["small"])
    ps.add_honesty_chip(fig, "Tested", CHIP_DETAIL, loc=(0.985, 0.975))
    note = _subseq_note(runs)
    fig.text(0.015, 0.02, "Top view (x-z), start pose aligned only: no scale or rotation fit. Frame-to-frame VO "
             "(SGBM + KLT + PnP-RANSAC), no loop closure, no bundle adjustment. 100 m scale bar in every panel; "
             "panels differ in scale." + (f"\n{note}." if note else ""), ha="left", va="bottom",
             fontsize=ps.FONT_PT["annot"], color=ps.TOKENS["text_muted"], linespacing=1.4)
    return fig


def fig_drift_table(runs: dict[str, dict[str, Any]]) -> Any:
    """Grouped bars of segment drift (%) at 100/500/1000 m per sequence, plus a metrics table."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=ps.size_inches("wide"))
    ax = fig.add_axes((0.055, 0.17, 0.40, 0.66))
    tab = fig.add_axes((0.50, 0.17, 0.485, 0.70))
    mono = ps.mono_family()
    seqs = list(runs)
    width, gap = 0.25, 0.02
    ymax = 0.0
    for i, seq in enumerate(seqs):
        res = runs[seq]["res"]
        for j, L in enumerate(HORIZONS_M):
            x = i + (j - 1) * (width + gap)
            v = res.get(f"drift_{L}m_pct")
            if v is None:
                ax.text(x, 0.05, f"drive\n< {L / 1000:.0f} km", ha="center", va="bottom", fontsize=ps.FONT_PT["annot"] - 1,
                        color=ps.TOKENS["text_muted"])
                continue
            ymax = max(ymax, v)
            ax.bar(x, v, width=width, color=HORIZON_COLORS[L], edgecolor=ps.TOKENS["bg"], lw=0.8, zorder=3)
            ax.text(x, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=ps.FONT_PT["annot"],
                    color=ps.TOKENS["text"], family=mono, zorder=4)
    ax.set_xticks(range(len(seqs)))
    ax.set_xticklabels([f"{_name(s, runs[s]['res'])}\n{_fmt_km(runs[s]['res']['path_length_m'])}" for s in seqs])
    ax.tick_params(axis="x", length=0, labelcolor=ps.TOKENS["text"])
    ax.set_ylim(0, max(ymax * 1.22, 1.0))
    ax.set_xlim(-0.55, len(seqs) - 0.45)
    ax.set_ylabel("Drift, % of segment length")
    ps.style_axes(ax, grid="y")
    ax.xaxis.grid(False)  # rcParams enable both; bars only need horizontal reference lines
    ax.legend([Patch(color=HORIZON_COLORS[L]) for L in HORIZONS_M], [f"{L} m segments" for L in HORIZONS_M],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handlelength=1.1, handleheight=1.1,
              columnspacing=1.4, borderaxespad=0.3)

    # Metrics table drawn with text (cleaner than matplotlib.table): numbers right-aligned in mono.
    tab.set_axis_off()
    cols = [("Sequence", 0.00, "left"), ("Drive", 0.30, "right"), ("Frames", 0.43, "right"),
            ("t_err %", 0.585, "right"), ("r_err °/100 m", 0.79, "right"), ("VO fails", 1.0, "right")]
    y_head, row_h = 0.95, 0.15
    for name, x, ha in cols:
        tab.text(x, y_head, name, ha=ha, va="center", fontsize=ps.FONT_PT["small"], fontweight="bold",
                 color=ps.TOKENS["text_secondary"], transform=tab.transAxes)
    tab.plot([0, 1], [y_head - 0.07] * 2, color=ps.TOKENS["text_secondary"], lw=0.8, transform=tab.transAxes,
             clip_on=False)
    rows = [[_name(s, runs[s]["res"]), _fmt_km(runs[s]["res"]["path_length_m"]), f"{runs[s]['res']['frames']}",
             f"{runs[s]['res']['t_err_pct']:.2f}", f"{runs[s]['res']['r_err_deg_per_100m']:.2f}",
             f"{runs[s]['res']['vo_failures']}"] for s in seqs]
    if len(seqs) > 1:  # pooled row: every 100-800 m segment of all sequences, segment-weighted
        pooled = pooled_kitti_errors(runs)
        rows.append(["All (pooled)", _fmt_km(sum(runs[s]["res"]["path_length_m"] for s in seqs)),
                     f"{sum(runs[s]['res']['frames'] for s in seqs)}", f"{pooled['t_err_pct']:.2f}",
                     f"{pooled['r_err_deg_per_100m']:.2f}", f"{sum(runs[s]['res']['vo_failures'] for s in seqs)}"])
    for i, vals in enumerate(rows):
        y = y_head - 0.07 - row_h * (i + 0.5)
        pooled_row = vals[0].startswith("All")
        if pooled_row:
            tab.plot([0, 1], [y + row_h / 2] * 2, color=ps.TOKENS["text_secondary"], lw=0.8,
                     transform=tab.transAxes, clip_on=False)
        for (name, x, ha), v in zip(cols, vals):
            tab.text(x, y, v, ha=ha, va="center", fontsize=ps.FONT_PT["body"], transform=tab.transAxes,
                     color=ps.TOKENS["text"], family=None if name == "Sequence" else mono,
                     fontweight="bold" if (name == "t_err %" or pooled_row) else "normal")
        if not pooled_row:
            tab.plot([0, 1], [y - row_h / 2] * 2, color=ps.TOKENS["border"], lw=0.6, transform=tab.transAxes,
                     clip_on=False)
    tab.text(0.0, y_head - 0.07 - row_h * len(rows) - 0.06,
             "t_err / r_err: KITTI protocol, mean over all 100-800 m segments (start every 10 frames);\n"
             "pooled = all segments of all sequences averaged together. VO fails: frames where VO\n"
             "rejected the estimate (bridged at constant velocity).",
             ha="left", va="top", fontsize=ps.FONT_PT["annot"], color=ps.TOKENS["text_secondary"],
             transform=tab.transAxes, linespacing=1.5)
    fig.text(0.015, 0.975, "Segment drift on real KITTI drives, camera-only stereo VO", ha="left", va="top",
             fontsize=ps.FONT_PT["title"] + 1.5, fontweight="bold", color=ps.TOKENS["text"])
    ps.add_honesty_chip(fig, "Tested", CHIP_DETAIL, loc=(0.985, 0.975))
    note = _subseq_note(runs)
    fig.text(0.015, 0.02, "Drift = mean end-point translation error of all segments of that length ÷ length. "
             "Frame-to-frame VO (SGBM + KLT + PnP-RANSAC), no loop closure, no bundle adjustment, no wheel/IMU input."
             + (f"\n{note}." if note else ""), ha="left", va="bottom", fontsize=ps.FONT_PT["annot"],
             color=ps.TOKENS["text_muted"], linespacing=1.4)
    return fig


def fig_drift_card(runs: dict[str, dict[str, Any]], curves: dict[str, dict[str, list]]) -> Any:
    """Card: mean drift (% of distance) vs segment length, one line per sequence, direct labels."""
    fig, ax = ps.new_figure("card")
    mono = ps.mono_family()
    ymax = 0.0
    for seq, c in curves.items():
        color, marker = SEQ_STYLE.get(seq, (OURS_COLOR, "o"))
        L, v = np.array(c["lengths_m"]), np.array(c["mean_pct"])
        if L.size == 0:
            continue
        ymax = max(ymax, float(v.max()))
        ax.plot(L, v, color=color, lw=1.8, marker=marker, ms=4.5, mec=ps.TOKENS["bg"], mew=0.9, zorder=4,
                label=f"{_name(seq, runs[seq]['res'])} ({_fmt_km(runs[seq]['res']['path_length_m'])})")
        # 07 ends under the 00 line, so its label sits below-right of the last point.
        dy, va = ((-5, "top") if seq == "07" else (0, "center"))
        ax.annotate(f"{seq}", (L[-1], v[-1]), xytext=(6, dy), textcoords="offset points", ha="left", va=va,
                    fontsize=ps.FONT_PT["small"], fontweight="bold", color=ps.TOKENS["text"], family=mono)
    ax.set_xlim(0, max(max(c["lengths_m"]) for c in curves.values() if c["lengths_m"]) + 150)
    ax.set_ylim(0, max(ymax * 1.25, 1.0))
    ax.set_xlabel("Distance travelled, segment length (m)")
    ax.set_ylabel("Drift, % of distance")
    ax.set_title("Drift vs distance travelled")
    ax.legend(loc="lower right", ncol=3, handlelength=1.8, columnspacing=1.2)
    ps.style_axes(ax)
    ps.add_honesty_chip(fig, "Tested", CHIP_DETAIL)
    note = _subseq_note(runs)
    ps.add_footnote(fig, f"Drift = mean end-point error of all segments of length L (start every {STEP_SIZE} frames) ÷ L;\n"
                         f"shown where ≥ {MIN_SEGMENTS} segments exist. Frame-to-frame VO: no loop closure, no BA."
                         + (f"\n{note}." if note else ""))
    return fig


def px_to_pt(px: float) -> float:
    """Font size in points that renders ``px`` pixels tall (em size) in a PNG saved at plot_style.DPI."""
    return px * 72.0 / ps.DPI


def fig_drift_slot(runs: dict[str, dict[str, Any]], curves: dict[str, dict[str, list]]) -> Any:
    """Slot-sized drift card (1128 x 652 px): same curves, markers and colours as :func:`fig_drift_card`,
    manual layout, every text >= SLOT_MIN_TEXT_PX (checked by tests/test_kitti_slot.py)."""
    import matplotlib.pyplot as plt

    fs = {k: px_to_pt(v) for k, v in SLOT_TEXT_PX.items()}
    fig = plt.figure(figsize=ps.size_inches(SLOT_SIZE_SLIDE_PX))
    ax = fig.add_axes((0.095, 0.255, 0.885, 0.615))  # figure fractions: room for title row + note row
    mono = ps.mono_family()
    ymax, xmax = 0.0, 0.0
    for seq, c in curves.items():
        color, marker = SEQ_STYLE.get(seq, (OURS_COLOR, "o"))
        L, v = np.array(c["lengths_m"]), np.array(c["mean_pct"])
        if L.size == 0:
            continue
        ymax, xmax = max(ymax, float(v.max())), max(xmax, float(L.max()))
        ax.plot(L, v, color=color, lw=2.0, marker=marker, ms=4.0, mec=ps.TOKENS["bg"], mew=0.8, zorder=4,
                label=f"{_name(seq, runs[seq]['res'])[len('KITTI '):]} · {_fmt_km(runs[seq]['res']['path_length_m'])}")
        dy, va = ((-4, "top") if seq == "07" else (0, "center"))  # 07 ends under the 00 line (as on the card)
        ax.annotate(seq, (L[-1], v[-1]), xytext=(5, dy), textcoords="offset points", ha="left", va=va,
                    fontsize=fs["direct"], fontweight="bold", color=ps.TOKENS["text"], family=mono)
    ax.set_xlim(0, xmax + 190)
    ax.set_ylim(0, max(ymax * 1.25, 1.0))
    ax.set_xticks(SLOT_X_TICKS_M)
    ax.set_yticks(SLOT_Y_TICKS_PCT)
    ax.tick_params(labelsize=fs["tick"], pad=3)
    ax.set_xlabel("Distance travelled, segment length (m)", fontsize=fs["label"], labelpad=4)
    ax.set_ylabel("Drift, % of distance", fontsize=fs["label"], labelpad=6)
    ax.legend(loc="lower right", ncol=3, fontsize=fs["legend"], handlelength=1.6, columnspacing=1.0,
              handletextpad=0.5, borderaxespad=0.2)
    ps.style_axes(ax)
    fig.text(0.012, 0.975, "Drift vs distance travelled", ha="left", va="top", fontsize=fs["title"],
             fontweight="bold", color=ps.TOKENS["text"])
    ps.add_honesty_chip(fig, "Tested", SLOT_CHIP_DETAIL, loc=(0.985, 0.972))
    fig.texts[-1].set_fontsize(fs["chip"])  # same chip style as every deck figure, slot-sized text
    sub = "".join(f"*{s}: frames {r['res']['first_frame']}-{r['res']['first_frame'] + r['res']['frames'] - 1}. "
                  for s, r in runs.items() if int(r["res"].get("first_frame", 0)))
    fig.text(0.012, 0.018, f"{sub}Frame-to-frame VO, no loop closure.", ha="left", va="bottom",
             fontsize=fs["note"], color=ps.TOKENS["text_secondary"])
    return fig


# --------------------------------------------------------------------------- summary
def build_summary(runs: dict[str, dict[str, Any]], curves: dict[str, dict[str, list]],
                  figures: list[str]) -> dict[str, Any]:
    """All numbers shown in the figures, how they were measured, claims (Tested) and literature (separate)."""
    keys = ("frames", "first_frame", "full_sequence", "path_length_m", "t_err_pct", "r_err_deg_per_100m", "n_segments",
            "drift_100m_m", "drift_100m_pct", "drift_100m_n", "drift_500m_m", "drift_500m_pct", "drift_500m_n",
            "drift_1000m_m", "drift_1000m_pct", "drift_1000m_n", "ate_rmse_m", "ate_max_m", "final_err_m",
            "vo_failures", "vo_failure_pct", "inliers_median", "vo_ms_mean", "vo_ms_median", "vo_ms_p95",
            "fps_vo", "fps_with_png_decode", "opencv_threads", "resolution", "timestamps", "timestamp")
    seqs = {s: {"label": "Tested", "source": f"results/kitti_vo_{s}.json",
                **{k: r["res"].get(k) for k in keys}} for s, r in runs.items()}
    for s, row in seqs.items():
        if row["timestamps"] is None:  # JSONs written before the field existed; their times.txt matched the frames
            row["timestamps"] = "times.txt (field absent in the run JSON; times.txt length matches the frame count)"
        if row["first_frame"] is None:  # field added after the 05/07 runs, which started at frame 0
            row["first_frame"] = 0
        if s in RUN_NOTES:
            row["run_note"] = RUN_NOTES[s]
    pooled = pooled_kitti_errors(runs)
    # Rows use the claims-ledger shape {id, value, unit, label, source, note} (metagross/eval/claims.py);
    # ids follow the ledger's kitti<seq>_<metric> naming so each number has exactly one ledger row.
    claims: list[dict[str, Any]] = []
    for s, r in runs.items():
        res = r["res"]
        ff = int(res.get("first_frame", 0))
        span = f"frames {ff}-{ff + res['frames'] - 1}" if ff else f"all {res['frames']} frames"
        ctx = f"KITTI {s} ({span}, {_fmt_km(res['path_length_m'])} drive), camera-only stereo VO"
        src = f"results/kitti_vo_{s}.json"
        claims.append({"id": f"kitti{s}_t_err_pct", "value": round(res["t_err_pct"], 2), "unit": "%",
                       "label": "Tested", "source": f"{src}#t_err_pct",
                       "note": f"{ctx}; KITTI protocol, mean over {res['n_segments']} segments of 100-800 m"})
        claims.append({"id": f"kitti{s}_r_err_deg100m", "value": round(res["r_err_deg_per_100m"], 2),
                       "unit": "deg/100 m", "label": "Tested", "source": f"{src}#r_err_deg_per_100m",
                       "note": f"{ctx}; KITTI protocol"})
        for L in HORIZONS_M:
            v = res.get(f"drift_{L}m_pct")
            if v is not None:
                claims.append({"id": f"kitti{s}_drift_{L}m_pct", "value": round(v, 2), "unit": "%",
                               "label": "Tested", "source": f"{src}#drift_{L}m_pct",
                               "note": f"{ctx}; mean end-point drift over {res[f'drift_{L}m_n']} segments of {L} m "
                                       f"(= {res[f'drift_{L}m_m']:.1f} m)"})
    pooled_span = " + ".join(
        s + (f" (frames {r['res']['first_frame']}-{r['res']['first_frame'] + r['res']['frames'] - 1})"
             if int(r["res"].get("first_frame", 0)) else "") for s, r in runs.items())
    for key, unit, pid in (("t_err_pct", "%", "kitti_pooled_t_err_pct"),
                           ("r_err_deg_per_100m", "deg/100 m", "kitti_pooled_r_err_deg100m")):
        claims.append({"id": pid, "value": round(pooled[key], 2), "unit": unit, "label": "Tested",
                       "source": f"results/kitti_summary.json#pooled.{key}",
                       "note": f"KITTI {pooled_span}, camera-only stereo VO; KITTI protocol pooled over "
                               f"{pooled['n_segments']} segments of 100-800 m"})
    return {
        "label": "Tested",
        "generated_by": "deck_assets/kitti_figures.py (reads results/kitti_vo_<seq>.json + results/raw poses)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "how_measured": {
            "data": "KITTI odometry training sequences (real car drives, Karlsruhe), full-resolution grayscale "
                    "rectified stereo (image_0/image_1), GT = KITTI GPS/INS poses (data/kitti/poses/<seq>.txt).",
            "estimator": "metagross.autonomy.localization.vo.StereoVO with VOConfig.kitti(), run by "
                         "metagross.eval.kitti_vo.run_vo: per-frame SGBM disparity, KLT tracking, PnP-RANSAC + LM, "
                         "frame-to-frame; camera only (no wheels/IMU), no loop closure, no bundle adjustment. "
                         "Rejected frames are bridged with the last accepted relative motion and counted as failures.",
            "t_err_r_err": "KITTI devkit protocol (port in metagross.eval.traj_metrics.kitti_errors): every 10th "
                           "start frame x segment lengths 100..800 m; t_err = mean |t(E)|/L (%), r_err = mean "
                           "angle(R(E))/L (deg/100 m).",
            "segment_drift": "metagross.eval.traj_metrics.segment_drift: same relative-pose error for a single "
                             "segment length L (100/500/1000 m); mean end-point error in m and % of L.",
            "drift_vs_length": f"segment_errors per L in {CURVE_LENGTHS_M[0]}..{CURVE_LENGTHS_M[-1]} m step 100, "
                               f"kept where >= {MIN_SEGMENTS} segments exist; mean / median / p90 of error % of L.",
            "pooled": "all 100..800 m segments of the listed sequences concatenated, then averaged (segment-weighted).",
            "ate": "absolute position error after aligning the first pose only (no scale / rotation fit).",
            "timing": "fps_vo = 1000 / mean VO wall time per frame incl. SGBM, single process, OpenCV threads as "
                      "listed, on a shared 4-core laptop CPU with other engineers' jobs running (not a clean benchmark).",
            "not_tuned_on": "VOConfig.kitti() was not changed for these runs; no parameters were fitted to 00/05/07 here.",
        },
        "sequences": seqs,
        "data_issues": ({"00": ev} if "00" in runs and (ev := mirror00_evidence()) is not None else {}),
        "pooled": {"label": "Tested", "sequences": list(runs), **pooled},
        "drift_vs_length": {"label": "Tested", **curves},
        "figures": figures,
        "claims": claims,
        "literature_context": {
            "label": "Literature",
            "note": "NOT our measurements. KITTI odometry leaderboard values, computed by the benchmark server on "
                    "the held-out test sequences 11-21 (100..800 m segments). Our numbers are on training "
                    "sequences 00/05/07, so the comparison is indicative only.",
            "source": LITERATURE_SOURCE,
            "retrieved": LITERATURE_RETRIEVED,
            "rows": [{**row, "label": "Literature", "r_err_deg_per_100m": round(row["r_err_deg_per_m"] * 100.0, 2)}
                     for row in LITERATURE_ROWS],
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="KITTI VO deck figures + results/kitti_summary.json")
    ap.add_argument("--seqs", nargs="+", default=list(DEFAULT_SEQS))
    ap.add_argument("--no-summary", action="store_true", help="figures only")
    ap.add_argument("--slot-only", action="store_true",
                    help=f"render only deck_assets/{SLOT_NAME}.png (no other figures, no summary)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    runs: dict[str, dict[str, Any]] = {}
    for s in args.seqs:
        s = f"{int(s):02d}"
        if not (RESULTS / f"kitti_vo_{s}.json").exists():
            LOG.warning("no results for sequence %s; skipped", s)
            continue
        runs[s] = load_run(s)
    if not runs:
        raise SystemExit("no KITTI results found")
    curves = {s: drift_curve(r["gt"], r["est"]) for s, r in runs.items()}
    ps.apply_style()
    slot = ps.save_fig(fig_drift_slot(runs, curves), SLOT_NAME, formats=("png",))
    if args.slot_only:
        return 0
    figures: list[str] = [p.relative_to(REPO_ROOT).as_posix() for p in slot]
    for name, fig in (("kitti_traj_panel", fig_traj_panel(runs)), ("kitti_drift_table", fig_drift_table(runs)),
                      ("kitti_drift_card", fig_drift_card(runs, curves))):
        figures += [p.relative_to(REPO_ROOT).as_posix() for p in ps.save_fig(fig, name)]
    if not args.no_summary:
        out = RESULTS / "kitti_summary.json"
        out.write_text(json.dumps(build_summary(runs, curves, figures), indent=2, ensure_ascii=False), encoding="utf-8")
        LOG.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
