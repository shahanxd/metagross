"""Tier-0 odometry on the 30 DEV drives: wheel + gyro vs wheel + gyro + depth odometry (final deck figure).

Run from the repo root:
    OMP_NUM_THREADS=2 python deck_assets/final/src/depth_odom.py           # plot the stored analysis
    OMP_NUM_THREADS=2 python deck_assets/final/src/depth_odom.py --rerun   # re-run the analysis first (~1 min)

Data (first one that exists):
    results/odometry_dev.json            the module's default output, once the orchestrator registers it
    deck_assets/final/odometry_dev.json  written by
        python -m metagross.eval.odometry_dev analyse --tune 100 ... 129 --out <json>
which replays the recorded tier-0 drives in results/raw/odometry_dev/<seed>.npz (DEV seeds only)
through the onboard Localizer twice: depth odometry off (wheel + gyro EKF) and on. Everything else
is identical. Scored against simulator ground truth in the launch (A) frame.

All 30 drives are shown as ONE set. The analysis file may still carry an old tune/holdout split
(seeds 100-115 / 116-129), but that split is not a clean holdout: localizer.py documents that the
depth-odometry yaw-sigma floor and the DO-yaw gyro-bias gating were chosen on DEV seeds 100-129
(tail of final error, citing seed 124), i.e. on the whole set and on the metric plotted here.
So the figure uses summary_all only and says the localiser was tuned on these drives.

Outputs (deck_assets/final/): depth_odom.png / .svg / .json
Label: Simulated (tier-0 synthetic depth sensor, no images, open-loop drives along the oracle path).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "deck_assets" / "final"
DATA_CANDIDATES = ("results/odometry_dev.json", "deck_assets/final/odometry_dev.json")
DEV = list(range(100, 130))
RERUN_TMP = Path("/tmp/claude-0/final/odometry_dev.json")

# ---- palette (deck brief) ------------------------------------------------------------------
NAVY = "#1F497D"      # ours: with depth odometry
GREY = "#9A9A9A"      # baseline: wheel + gyro
INK = "#1A1A1A"
MUTED = "#5C5C5C"     # secondary text and the provenance line (>= 4.5:1 on white)
GRID = "#E6E6E6"
LINK = "#C8C8C8"      # per-drive connector
BG = "#FFFFFF"

# text sizes in points at DPI 200 (13 pt = 36 px in the PNG; ~18 px when shown 900 px wide)
FS_MIN = 13.0
FS_TICK = 13.5
FS_LABEL = 14.0
FS_VALUE = 14.5

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FS_LABEL,
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": MUTED,
    "svg.fonttype": "path",
    "figure.facecolor": BG,
    "axes.facecolor": BG,
})
DPI = 200


def rerun() -> None:
    """Re-run the analysis over all 30 DEV drives as one set (no tune/holdout framing)."""
    RERUN_TMP.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "metagross.eval.odometry_dev", "analyse", "--tune", *map(str, DEV),
           "--out", str(RERUN_TMP)]
    subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
    shutil.copy(RERUN_TMP, ROOT / DATA_CANDIDATES[1])


def load() -> tuple[dict, list[dict], str]:
    rel = next(p for p in DATA_CANDIDATES if (ROOT / p).exists())
    d = json.loads((ROOT / rel).read_text())
    assert d["label"] == "Simulated"
    rows = list(d.get("drives_tune") or []) + list(d.get("drives_holdout") or [])
    seeds = sorted(r["seed"] for r in rows)
    assert seeds == DEV, f"expected exactly DEV seeds 100-129, got {seeds}"
    assert all(s >= 100 for s in seeds), "EVAL seed in the odometry set"
    return d, rows, rel


def build() -> dict:
    d, rows, data_rel = load()
    s_all = d["summary_all"]
    before = np.array([r["wheel_gyro"]["final_err_m"] for r in rows])
    after = np.array([r["wheel_gyro_depth"]["final_err_m"] for r in rows])
    seeds = [r["seed"] for r in rows]
    fams = [r["family"] for r in rows]
    paths = np.array([r["wheel_gyro"]["gt_path_m"] for r in rows])
    slips = np.array([r["scenario_slip_long"] for r in rows])
    med_b, med_a = float(np.median(before)), float(np.median(after))
    # the plotted medians must be the analysis' own all-30 medians
    assert abs(med_b - s_all["wheel_gyro"]["final_err_m_median"]) < 1e-9
    assert abs(med_a - s_all["wheel_gyro_depth"]["final_err_m_median"]) < 1e-9
    improved = int(np.sum(after < before))
    n = len(rows)
    sc_b = s_all["wheel_gyro"]["scale_err_pct_median"]
    sc_a = s_all["wheel_gyro_depth"]["scale_err_pct_median"]
    pct_b = s_all["wheel_gyro"]["final_err_pct_median"]
    pct_a = s_all["wheel_gyro_depth"]["final_err_pct_median"]

    order = np.argsort(-before, kind="stable")  # left to right: largest wheel + gyro error first
    xs = np.arange(n)
    ymax = 5.5
    assert max(before.max(), after.max()) < ymax

    fig = plt.figure(figsize=(9.0, 5.9), dpi=DPI)
    ax = fig.add_axes([0.085, 0.27, 0.70, 0.56])
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylim(0, ymax)
    for yv in range(1, 6):
        ax.axhline(yv, color=GRID, lw=1, zorder=0)
    ax.plot([-0.8, n - 0.2], [0, 0], color="#BDBDBD", lw=1, zorder=0, clip_on=False)
    # medians: thin dashed reference line per series, drawn behind the drives
    ax.axhline(med_b, color=GREY, lw=1.4, ls=(0, (5, 3)), zorder=1)
    ax.axhline(med_a, color=NAVY, lw=1.4, ls=(0, (5, 3)), zorder=1)
    for x, i in zip(xs, order):
        ax.plot([x, x], [before[i], after[i]], color=LINK, lw=2.0, zorder=2, solid_capstyle="round")
    ax.scatter(xs, before[order], s=78, color=GREY, edgecolor=BG, linewidth=1.5, zorder=3)
    ax.scatter(xs, after[order], s=78, color=NAVY, edgecolor=BG, linewidth=1.5, zorder=4)

    # direct labels for the medians, in the right margin (ink text; the dashed line carries identity)
    xr = n - 0.2 + 0.35
    ax.text(xr, med_b, f"median\n{med_b:.2f} m", ha="left", va="center", fontsize=FS_VALUE, color=INK,
            fontweight="bold", linespacing=1.1, clip_on=False)
    ax.text(xr, med_a, f"median\n{med_a:.2f} m", ha="left", va="center", fontsize=FS_VALUE, color=INK,
            fontweight="bold", linespacing=1.1, clip_on=False)
    for y, c in ((med_b, GREY), (med_a, NAVY)):
        ax.plot([n - 0.2, xr - 0.1], [y, y], color=c, lw=1.4, ls=(0, (5, 3)), clip_on=False, zorder=1)

    ax.set_xticks([])
    ax.set_yticks(range(0, 6))
    ax.tick_params(axis="y", length=0, labelsize=FS_TICK, pad=5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel("Position error at the end of the drive (m)", fontsize=FS_LABEL, labelpad=8)
    ax.set_xlabel(f"{n} DEV drives (seeds {DEV[0]}-{DEV[-1]}), sorted by wheel + gyro error", fontsize=FS_LABEL,
                  labelpad=10)

    # key (top) and the one-line result under it
    fig.canvas.draw()
    ky, kx = 0.955, 0.085
    for colour, label in ((GREY, "Wheel + gyro EKF"), (NAVY, "Wheel + gyro + depth odometry")):
        fig.add_artist(Line2D([kx + 0.008], [ky], transform=fig.transFigure, ls="none", marker="o", ms=10,
                              color=colour, mec=BG, mew=1.5))
        t = fig.text(kx + 0.026, ky, label, fontsize=FS_LABEL, va="center", ha="left", color=INK)
        fig.canvas.draw()
        kx = t.get_window_extent().transformed(fig.transFigure.inverted()).x1 + 0.035
    fig.add_artist(Line2D([kx + 0.006, kx + 0.006], [ky - 0.022, ky + 0.022], transform=fig.transFigure,
                          color=LINK, lw=2.0, solid_capstyle="round"))
    fig.text(kx + 0.022, ky, "one drive", fontsize=FS_LABEL, va="center", ha="left", color=INK)
    imp = f"Final error drops on all {n} drives" if improved == n else f"Final error drops on {improved} of {n} drives"
    fig.text(0.085, 0.885, f"{imp} · median distance over-count from wheel slip {sc_b:.1f} % → {sc_a:.1f} %",
             fontsize=FS_MIN, color=MUTED, ha="left", va="center")

    # footnote placed just under the x label, provenance under it
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    y_lab = ax.xaxis.label.get_window_extent().transformed(inv).y0
    foot = fig.text(0.012, y_lab - 0.022,
                    f"Each drive ({paths.min():.0f}-{paths.max():.0f} m, open loop along the scenario's ground-truth "
                    f"path, {100 * slips.min():.0f}-{100 * slips.max():.0f} % wheel slip)\n"
                    "is replayed through the onboard localiser twice; only depth odometry differs.\n"
                    "The localiser was tuned on these same DEV drives, so this is not a held-out test.",
                    fontsize=FS_MIN, color=MUTED, ha="left", va="top", linespacing=1.3)
    fig.canvas.draw()
    y_ft = foot.get_window_extent().transformed(inv).y0
    fig.text(0.012, y_ft - 0.018, f"Simulated, tier-0 synthetic depth (no images), DEV seeds 100-129 · {data_rel}",
             fontsize=FS_MIN, color=MUTED, ha="left", va="top")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"depth_odom.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.12, facecolor=BG)
    plt.close(fig)
    with Image.open(OUT / "depth_odom.png") as im:
        png_px = list(im.size)

    per_drive = [{"seed": seeds[i], "family": fams[i], "x_rank": int(x),
                  "wheel_gyro_final_err_m": round(float(before[i]), 3),
                  "wheel_gyro_depth_final_err_m": round(float(after[i]), 3)} for x, i in zip(xs, order)]
    src_d = f"{data_rel}#drives_tune[]+drives_holdout[]"
    drawn = [
        {"what": "per-drive final position error, wheel + gyro (grey dots) and + depth odometry (navy dots)",
         "value": per_drive, "unit": "m",
         "source": f"{src_d}.wheel_gyro.final_err_m / .wheel_gyro_depth.final_err_m",
         "how": "|estimated - GT| planar position at the last frame, launch (A) frame; drives ordered by "
                "wheel + gyro error, largest first"},
        {"what": "median final error, wheel + gyro (grey dashed line, label)", "value": round(med_b, 2), "unit": "m",
         "source": f"{data_rel}#summary_all.wheel_gyro.final_err_m_median",
         "how": "median over the 30 drives (asserted equal to np.median of the per-drive values)"},
        {"what": "median final error, + depth odometry (navy dashed line, label)", "value": round(med_a, 2),
         "unit": "m", "source": f"{data_rel}#summary_all.wheel_gyro_depth.final_err_m_median",
         "how": "median over the 30 drives (asserted equal to np.median of the per-drive values)"},
        {"what": "drives whose final error went down (header line)", "value": improved, "of": n,
         "source": src_d, "how": "count of drives with wheel_gyro_depth.final_err_m < wheel_gyro.final_err_m"},
        {"what": "median along-track distance over-count (scale error), before -> after (header line)",
         "value": f"{sc_b:.1f} -> {sc_a:.1f}", "unit": "%",
         "source": f"{data_rel}#summary_all.wheel_gyro(_depth).scale_err_pct_median",
         "how": "estimated planar path length / GT planar path length - 1, median over the 30 drives"},
        {"what": "drive length range and wheel-slip range (footnote)",
         "value": f"{paths.min():.1f}-{paths.max():.1f} m; {100 * slips.min():.1f}-{100 * slips.max():.1f} %",
         "source": f"{src_d}.wheel_gyro.gt_path_m, .scenario_slip_long", "how": "min / max over the 30 drives"},
    ]
    return {
        "figure": "depth_odom", "label": "Simulated", "png_px": png_px,
        "provenance": "tier-0 synthetic depth sensor (no images, no SGBM); 30 recorded DEV drives "
                      "(results/raw/odometry_dev/*.npz) replayed open loop along the GT oracle path; no EVAL seeds",
        "data_file": data_rel,
        "command": "python -m metagross.eval.odometry_dev analyse --tune " + " ".join(map(str, DEV)) +
                   " --out results/odometry_dev.json",
        "split_note": "All 30 DEV drives are one set. The stored analysis carries a tune (100-115) / holdout "
                      "(116-129) split, but it is not a clean holdout: localizer.py documents localiser settings "
                      "chosen on DEV seeds 100-129 using final error (incl. seed 124), and the stack was tuned on "
                      "DEV 100-129 (docs/BUILD_LOG.md). The split medians are NOT used.",
        "numbers": drawn,
        "json_only_not_drawn": {
            "median_final_err_pct_of_path": [round(pct_b, 1), round(pct_a, 1)],
            "max_final_err_m": [round(s_all["wheel_gyro"]["final_err_m_max"], 2),
                                round(s_all["wheel_gyro_depth"]["final_err_m_max"], 2)],
            "worst_drive": "seed 105 (F4_sudden_obstacle): 5.02 m -> 3.99 m",
            "depth_odom_accepted_frac_median": round(s_all["depth_odom_accepted_frac_median"], 3),
            "depth_odom_ms": "about 10 ms per frame (median of per-drive medians 9.4 ms, max per-drive p95 15.8 ms "
                             "in the stored run; a re-run in the same container gave 9.6 / 18.6 ms), 4-vCPU Linux "
                             "cloud container, shared, load-dependent. The analysis JSON's timing_note says "
                             "'laptop'; that is a fixed string and is wrong for this run.",
            "order": "[wheel + gyro, + depth odometry]",
            "source": f"{data_rel}#summary_all",
        },
        "registration_needed": {
            "status": "NOT yet in results/ nor results/claims.csv (this agent may not write there). Before the "
                      "figure goes on a slide: run the command above (writes results/odometry_dev.json), re-run this "
                      "script (it then cites results/odometry_dev.json), and append these rows.",
            "proposed_claims_rows": [
                f"depth_odom_final_err_median_m_wheel_gyro,{med_b:.2f},m,Simulated,"
                "results/odometry_dev.json#summary_all.wheel_gyro.final_err_m_median,"
                "\"30 DEV drives (seeds 100-129), tier-0 synthetic depth, open loop along GT oracle path, 40-60 m; "
                "localiser tuned on these drives\"",
                f"depth_odom_final_err_median_m_with_do,{med_a:.2f},m,Simulated,"
                "results/odometry_dev.json#summary_all.wheel_gyro_depth.final_err_m_median,"
                "\"same 30 DEV drives replayed with depth odometry on; everything else identical\"",
                f"depth_odom_drives_improved,{improved}/{n},,Simulated,"
                "results/odometry_dev.json#drives_tune[],\"drives whose final error went down with depth odometry\"",
                f"depth_odom_scale_err_median_pct,{sc_b:.1f} -> {sc_a:.1f},%,Simulated,"
                "results/odometry_dev.json#summary_all.*.scale_err_pct_median,"
                "\"median along-track distance over-count from wheel slip, wheel + gyro vs + depth odometry\"",
            ],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true", help="re-run the odometry analysis before plotting")
    args = ap.parse_args()
    if args.rerun or not any((ROOT / p).exists() for p in DATA_CANDIDATES):
        rerun()
    info = build()
    (OUT / "depth_odom.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {OUT / 'depth_odom'}.png/.svg/.json  png {info['png_px']}")


if __name__ == "__main__":
    main()
