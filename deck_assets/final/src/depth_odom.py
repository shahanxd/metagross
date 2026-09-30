"""Tier-0 odometry on 30 DEV drives: wheel + gyro vs wheel + gyro + depth odometry (final deck figure).

Run from the repo root:
    OMP_NUM_THREADS=2 python deck_assets/final/src/depth_odom.py           # plot the stored analysis
    OMP_NUM_THREADS=2 python deck_assets/final/src/depth_odom.py --rerun   # re-run the analysis first (~1 min)

Data: deck_assets/final/odometry_dev.json, written by
    python -m metagross.eval.odometry_dev analyse --tune 100 ... 115 --holdout 116 ... 129 --out <json>
which replays the recorded tier-0 drives in results/raw/odometry_dev/<seed>.npz (DEV seeds only)
through the onboard Localizer twice: depth odometry off (wheel + gyro EKF) and on. Everything else
is identical. Scored against simulator ground truth in the launch frame.

Outputs (deck_assets/final/): depth_odom.png / .svg / .json
Label: Simulated (tier-0 synthetic depth sensor, open-loop drives along the oracle path).
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

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "deck_assets" / "final"
DATA = OUT / "odometry_dev.json"
DATA_REL = "deck_assets/final/odometry_dev.json"
TUNE = list(range(100, 116))
HOLDOUT = list(range(116, 130))

# ---- palette (deck brief) ------------------------------------------------------------------
NAVY = "#1F497D"      # ours: with depth odometry
GREY = "#9A9A9A"      # baseline: wheel + gyro
INK = "#1A1A1A"
MUTED = "#5C5C5C"
FAINT = "#8A8A8A"
GRID = "#E6E6E6"
LINK = "#CFCFCF"      # per-drive connector
BG = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
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
    tmp = Path("/tmp/claude-0/final/odometry_dev.json")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "metagross.eval.odometry_dev", "analyse", "--tune", *map(str, TUNE),
           "--holdout", *map(str, HOLDOUT), "--out", str(tmp)]
    subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
    shutil.copy(tmp, DATA)


def load() -> dict:
    d = json.loads(DATA.read_text())
    assert d["label"] == "Simulated"
    assert d["tune_seeds"] == TUNE and d["holdout_seeds"] == HOLDOUT, "unexpected seed split"
    assert all(100 <= s <= 129 for s in d["tune_seeds"] + d["holdout_seeds"]), "EVAL seed in the odometry set"
    return d


def split_stats(rows: list[dict], summary: dict) -> dict:
    before = np.array([r["wheel_gyro"]["final_err_m"] for r in rows])
    after = np.array([r["wheel_gyro_depth"]["final_err_m"] for r in rows])
    st = {
        "n": len(rows),
        "seeds": [r["seed"] for r in rows],
        "before": before, "after": after,
        "median_before_m": float(np.median(before)), "median_after_m": float(np.median(after)),
        "median_before_pct": summary["wheel_gyro"]["final_err_pct_median"],
        "median_after_pct": summary["wheel_gyro_depth"]["final_err_pct_median"],
        "scale_before_pct": summary["wheel_gyro"]["scale_err_pct_median"],
        "scale_after_pct": summary["wheel_gyro_depth"]["scale_err_pct_median"],
        "improved": int(np.sum(after < before)),
        "path_m": [r["wheel_gyro"]["gt_path_m"] for r in rows],
    }
    # the plotted medians must be the analysis' own medians
    assert abs(st["median_before_m"] - summary["wheel_gyro"]["final_err_m_median"]) < 1e-9
    assert abs(st["median_after_m"] - summary["wheel_gyro_depth"]["final_err_m_median"]) < 1e-9
    return st


def build() -> dict:
    d = load()
    splits = [("Tuning drives", f"seeds {TUNE[0]}-{TUNE[-1]} · n = {len(TUNE)}", "tune",
               split_stats(d["drives_tune"], d["summary_tune"])),
              ("Holdout drives", f"seeds {HOLDOUT[0]}-{HOLDOUT[-1]} · n = {len(HOLDOUT)}",
               "holdout", split_stats(d["drives_holdout"], d["summary_holdout"]))]
    allrows = d["drives_tune"] + d["drives_holdout"]
    paths = [r["wheel_gyro"]["gt_path_m"] for r in allrows]
    slips = [r["scenario_slip_long"] for r in allrows]
    ymax = 5.6
    assert max(max(s["before"].max(), s["after"].max()) for *_, s in splits) < ymax

    fig = plt.figure(figsize=(9.0, 6.2), dpi=DPI)
    axes = [fig.add_axes([0.085, 0.265, 0.43, 0.57]), fig.add_axes([0.555, 0.265, 0.43, 0.57])]
    drawn = []
    for ax, (title, sub, key, st) in zip(axes, splits):
        ax.set_xlim(-0.95, 1.95)
        ax.set_ylim(0, ymax)
        for yv in range(1, 6):
            ax.axhline(yv, color=GRID, lw=1, zorder=0)
        for b, a in zip(st["before"], st["after"]):
            ax.plot([0, 1], [b, a], color=LINK, lw=1.2, zorder=1, solid_capstyle="round")
        ax.scatter(np.zeros(st["n"]), st["before"], s=64, color=GREY, edgecolor=BG, linewidth=1.6, zorder=3)
        ax.scatter(np.ones(st["n"]), st["after"], s=64, color=NAVY, edgecolor=BG, linewidth=1.6, zorder=4)
        # medians: short ink tick on the outer side of each column, value beside it
        for x, med, pct, side in ((0, st["median_before_m"], st["median_before_pct"], -1),
                                  (1, st["median_after_m"], st["median_after_pct"], 1)):
            x0, x1 = (x - 0.3, x - 0.1) if side < 0 else (x + 0.1, x + 0.3)
            ax.plot([x0, x1], [med, med], color=INK, lw=2.6, solid_capstyle="butt", zorder=5)
            ax.text(x1 + 0.05 if side > 0 else x0 - 0.05, med, f"median\n{med:.2f} m", ha="left" if side > 0 else "right",
                    va="center", fontsize=12, color=INK, fontweight="bold", linespacing=1.15, zorder=6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Wheel + gyro", "+ depth\nodometry"], fontsize=12.5, linespacing=1.15)
        ax.tick_params(axis="x", length=0, pad=7)
        ax.tick_params(axis="y", length=0, labelsize=12.5, pad=4)
        ax.set_yticks(range(0, 6))
        for s in ax.spines.values():
            s.set_visible(False)
        ax.plot([-0.95, 1.95], [0, 0], color="#BDBDBD", lw=1, zorder=0, clip_on=False)
        t = ax.text(0.0, 1.2, title, transform=ax.transAxes, ha="left", va="baseline", fontsize=14,
                    fontweight="bold", color=INK)
        if key == "holdout":
            fig.canvas.draw()
            bb = t.get_window_extent().transformed(ax.transAxes.inverted())
            ax.text(bb.x1 + 0.03, 1.2, "scored once", transform=ax.transAxes, ha="left",
                    va="baseline", fontsize=11, color=MUTED)
        imp = f"all {st['n']} improved" if st["improved"] == st["n"] else f"{st['improved']} of {st['n']} improved"
        ax.text(0.0, 1.115, f"{sub} · {imp}", transform=ax.transAxes, ha="left", va="baseline", fontsize=11.5,
                color=INK)
        ax.text(0.0, 1.04, f"median distance over-count {st['scale_before_pct']:.1f} % \u2192 "
                f"{st['scale_after_pct']:.1f} %", transform=ax.transAxes, ha="left", va="baseline", fontsize=11,
                color=MUTED)
        summ = f"summary_{key}"
        drawn += [
            {"what": f"{title}: per-drive final position error, wheel + gyro (grey dots)",
             "value": [round(v, 3) for v in st["before"]], "unit": "m", "seeds": st["seeds"],
             "source": f"{DATA_REL}#drives_{key}[].wheel_gyro.final_err_m",
             "how": "|estimated - GT| planar position at the last frame, launch (A) frame"},
            {"what": f"{title}: per-drive final position error, + depth odometry (navy dots)",
             "value": [round(v, 3) for v in st["after"]], "unit": "m", "seeds": st["seeds"],
             "source": f"{DATA_REL}#drives_{key}[].wheel_gyro_depth.final_err_m",
             "how": "same drive replayed with depth odometry on; everything else identical"},
            {"what": f"{title}: median final error, wheel + gyro", "value": round(st["median_before_m"], 2),
             "unit": "m", "source": f"{DATA_REL}#{summ}.wheel_gyro.final_err_m_median",
             "how": "median over the drives (checked against np.median of the per-drive values)"},
            {"what": f"{title}: median final error, + depth odometry", "value": round(st["median_after_m"], 2),
             "unit": "m", "source": f"{DATA_REL}#{summ}.wheel_gyro_depth.final_err_m_median",
             "how": "median over the drives (checked against np.median of the per-drive values)"},
            {"what": f"{title}: drives whose final error went down", "value": st["improved"], "of": st["n"],
             "source": DATA_REL, "how": "count of drives with wheel_gyro_depth.final_err_m < wheel_gyro.final_err_m"},
            {"what": f"{title}: median along-track distance over-count (scale error), before -> after",
             "value": f"{st['scale_before_pct']:.1f} -> {st['scale_after_pct']:.1f}", "unit": "%",
             "source": f"{DATA_REL}#{summ}.wheel_gyro(.._depth).scale_err_pct_median",
             "how": "estimated planar path length / GT planar path length - 1, median over drives"},
            {"what": f"{title}: median final error as % of path (json only, not drawn)",
             "value": f"{st['median_before_pct']:.1f} -> {st['median_after_pct']:.1f}", "unit": "%",
             "source": f"{DATA_REL}#{summ}.*.final_err_pct_median", "how": "median over drives"},
        ]
    axes[0].set_ylabel("Final position error at the end of the drive (m)", fontsize=12.5, labelpad=8)
    axes[1].set_yticklabels([])

    fig.text(0.012, 0.045,
             f"Each line is one drive ({min(paths):.0f}-{max(paths):.0f} m along the scenario's oracle path, "
             f"{100 * min(slips):.0f}-{100 * max(slips):.0f} % longitudinal wheel slip) replayed\n"
             "through the onboard localiser twice; only depth odometry differs. Holdout drives were recorded and\n"
             "scored only after the depth-odometry parameters were frozen.",
             fontsize=10.5, color=MUTED, ha="left", va="bottom", linespacing=1.35)
    fig.text(0.012, 0.0, "Simulated · tier-0 depth sensor, 30 DEV drives (seeds 100-129), open loop · "
             "deck_assets/final/odometry_dev.json", fontsize=10, color=FAINT, ha="left", va="bottom")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"depth_odom.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.12, facecolor=BG)
    plt.close(fig)

    s_all = d["summary_all"]
    drawn.append({"what": "drive length range and wheel-slip range (footnote)",
                  "value": f"{min(paths):.1f}-{max(paths):.1f} m; {100 * min(slips):.1f}-{100 * max(slips):.1f} %",
                  "source": f"{DATA_REL}#drives_*[].wheel_gyro.gt_path_m, scenario_slip_long",
                  "how": "min / max over the 30 drives"})
    return {
        "figure": "depth_odom", "label": "Simulated",
        "provenance": "tier-0 synthetic depth sensor; 30 recorded DEV drives (results/raw/odometry_dev/*.npz) "
                      "replayed open loop; no EVAL seeds",
        "command": "python -m metagross.eval.odometry_dev analyse --tune " + " ".join(map(str, TUNE)) +
                   " --holdout " + " ".join(map(str, HOLDOUT)) + " --out /tmp/claude-0/final/odometry_dev.json",
        "numbers": drawn,
        "all_30_not_drawn": {
            "median_final_err_m": [round(s_all["wheel_gyro"]["final_err_m_median"], 2),
                                   round(s_all["wheel_gyro_depth"]["final_err_m_median"], 2)],
            "median_final_err_pct": [round(s_all["wheel_gyro"]["final_err_pct_median"], 2),
                                     round(s_all["wheel_gyro_depth"]["final_err_pct_median"], 2)],
            "median_scale_err_pct": [round(s_all["wheel_gyro"]["scale_err_pct_median"], 2),
                                     round(s_all["wheel_gyro_depth"]["scale_err_pct_median"], 2)],
            "max_final_err_m": [round(s_all["wheel_gyro"]["final_err_m_max"], 2),
                                round(s_all["wheel_gyro_depth"]["final_err_m_max"], 2)],
            "depth_odom_ms_median_of_drive_medians": round(s_all["depth_odom_ms_median_of_drive_medians"], 1),
            "depth_odom_ms_max_of_drive_p95": round(s_all["depth_odom_ms_max_of_drive_p95"], 1),
            "depth_odom_accepted_frac_median": round(s_all["depth_odom_accepted_frac_median"], 3),
            "order": "[wheel + gyro, + depth odometry]",
            "source": f"{DATA_REL}#summary_all",
            "timing_caveat": "ms measured in this Linux cloud container (4 vCPU, shared), not on the team laptop; "
                             "the analysis JSON's timing_note text is a fixed string that says 'laptop'",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true", help="re-run the odometry analysis before plotting")
    args = ap.parse_args()
    if args.rerun or not DATA.exists():
        rerun()
    info = build()
    (OUT / "depth_odom.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {OUT / 'depth_odom'}.png/.svg/.json")


if __name__ == "__main__":
    main()
