"""Compute and operator-link budget figures for the final deck, from the held-out EVAL runs.

Run from the repo root:
    OMP_NUM_THREADS=2 python deck_assets/final/src/compute_link.py

Outputs (deck_assets/final/):
    compute_budget.png / .svg / .json   wall time per 5 Hz control tick, per module, vs the 200 ms tick
    link_budget.png    / .svg / .json   telemetry packet sizes vs the 600 B radio budget

Sources (nothing else is read):
    results/runs_eval_tier0/FULL/*/autonomy/timings.csv     one row per control tick (60 EVAL runs)
    results/runs_eval_tier0/FULL/*/autonomy/telemetry.jsonl one line per telemetry packet, with the
                                                            size the onboard encoder produced
    results/closed_loop_eval.json                           cross-check of tick count and compute p50/p95

Label: Simulated (tier-0 synthetic depth sensor, no images). Compute was measured while the
episodes ran: Linux container, 4 vCPU, no GPU, 4 episodes in parallel (BUILD_LOG 2026-09-30).
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "deck_assets" / "final"
RUNS = "results/runs_eval_tier0/FULL"
EVAL_JSON = "results/closed_loop_eval.json"

# ---- palette (deck brief) ------------------------------------------------------------------
NAVY = "#1F497D"      # ours / FULL
NAVY_LIGHT = "#94A9C4"  # same hue, lighter step (p95); ordinal pair validated with the dataviz validator
GREY = "#9A9A9A"
GREEN = "#2E7D32"
RED = "#C62828"
AMBER = "#D98E04"
INK = "#1A1A1A"
MUTED = "#5C5C5C"
FAINT = "#8A8A8A"
GRID = "#E6E6E6"
BG = "#FFFFFF"

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
TICK_BUDGET_MS = 200.0  # 5 Hz control tick

MODULES = [  # (timings.csv column, label) in pipeline order: sense -> act
    ("localizer", "Localisation (EKF + depth odometry)"),
    ("perception", "Perception (ground, hazards, ditches)"),
    ("map", "Rolling map"),
    ("costmap", "Costmap"),
    ("global", "Global planner"),
    ("supervisor", "Safety supervisor"),
    ("governor", "Speed governor"),
    ("mppi", "MPPI, 512 rollouts"),
    ("gate_mixer", "Safety gate + wheel mixer"),
    ("telemetry", "Telemetry packet *"),
]
TEL_ACTIVE_MS = 0.05  # a telemetry column above this means the tick built a packet


def provenance(fig, lines: list[str], y0: float = 0.012) -> None:
    for i, text in enumerate(reversed(lines)):
        fig.text(0.012, y0 + i * 0.030, text, fontsize=9.5, color=FAINT, ha="left", va="bottom")


def save(fig, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.12, facecolor=BG)
    plt.close(fig)


def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q))


# ---- compute budget ------------------------------------------------------------------------
def load_timings() -> tuple[dict[str, np.ndarray], int]:
    files = sorted((ROOT / RUNS).glob("*/autonomy/timings.csv"))
    cols: dict[str, list[float]] = {}
    for f in files:
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                for k, v in row.items():
                    cols.setdefault(k, []).append(float(v))
    return {k: np.asarray(v) for k, v in cols.items()}, len(files)


def compute_budget() -> dict:
    tm, n_runs = load_timings()
    n_ticks = int(tm["total"].size)
    ev = json.loads((ROOT / EVAL_JSON).read_text())["tier0"]["aggregate"]["FULL"]["all"]
    # cross-check against the aggregate the EVAL report was built from
    assert ev["n_ticks"] == n_ticks, (ev["n_ticks"], n_ticks)
    assert abs(ev["compute_ms_p50"] - np.median(tm["compute_ms"])) < 0.06, ev["compute_ms_p50"]
    assert abs(ev["compute_ms_p95"] - pct(tm["compute_ms"], 95)) < 0.06, ev["compute_ms_p95"]

    rows = []
    for key, label in MODULES:
        a = tm[key]
        note = "all ticks"
        if key == "telemetry":
            a = a[a > TEL_ACTIVE_MS]
            note = f"only the {a.size} ticks that built a packet (every 3rd tick)"
        rows.append({"key": key, "label": label, "n": int(a.size), "p50_ms": round(float(np.median(a)), 2),
                     "p95_ms": round(pct(a, 95), 2), "max_ms": round(float(a.max()), 2), "over": note})
    tot = tm["total"]
    total = {"key": "total", "label": "Whole tick", "n": n_ticks, "p50_ms": round(float(np.median(tot)), 2),
             "p95_ms": round(pct(tot, 95), 2), "p99_ms": round(pct(tot, 99), 2), "max_ms": round(float(tot.max()), 2),
             "ticks_over_budget": int((tot > TICK_BUDGET_MS).sum()), "over": "all ticks"}
    cmd = tm["compute_ms"]
    sense_to_cmd = {"p50_ms": round(float(np.median(cmd)), 2), "p95_ms": round(pct(cmd, 95), 2),
                    "max_ms": round(float(cmd.max()), 2), "ticks_over_budget": int((cmd > TICK_BUDGET_MS).sum())}

    # ---- figure ----
    labels = [r["label"] for r in rows] + [total["label"]]
    p50 = [r["p50_ms"] for r in rows] + [total["p50_ms"]]
    p95 = [r["p95_ms"] for r in rows] + [total["p95_ms"]]
    n = len(labels)
    y = np.arange(n, dtype=float)
    y[-1] += 0.55  # air between the modules and the whole-tick row
    h = 0.56

    fig = plt.figure(figsize=(9.0, 7.4), dpi=DPI)
    ax = fig.add_axes([0.385, 0.285, 0.585, 0.665])
    ax.barh(y, p95, height=h, color=NAVY_LIGHT, linewidth=0, zorder=2)
    ax.barh(y, p50, height=h, color=NAVY, linewidth=0, zorder=3)
    for yi, a, b in zip(y, p50, p95):
        fmt = (lambda v: f"{v:.2f}") if b < 1 else (lambda v: f"{v:.1f}")
        ax.text(b + 2.2, yi, f"{fmt(a)} / {fmt(b)}", va="center", ha="left", fontsize=11.5, color=INK, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.get_yticklabels()[-1].set_color(INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 214)
    ax.set_ylim(y[-1] + 1.35, -1.35)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.tick_params(axis="x", labelsize=12, length=0, pad=6)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("Wall time per 5 Hz control tick (ms)", fontsize=12.5, labelpad=8)
    # separator above the whole-tick row
    ysep = (y[-2] + y[-1]) / 2 + 0.05
    ax.plot([-0.62, 1.0], [ysep, ysep], transform=ax.get_yaxis_transform(), color=GRID, lw=1.0, clip_on=False)

    # the budget
    ax.axvline(TICK_BUDGET_MS, color=RED, lw=1.8, zorder=5)
    ax.text(TICK_BUDGET_MS - 2.5, -1.05, "200 ms tick budget (5 Hz)", ha="right", va="center", fontsize=12, color=INK)

    # headroom on the whole-tick row
    yt = y[-1]
    b95 = total["p95_ms"]
    x0 = b95 + 44  # clear of the value label
    ax.annotate("", xy=(TICK_BUDGET_MS - 1.5, yt), xytext=(x0, yt),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, shrinkA=0, shrinkB=0), zorder=4)
    ax.text(TICK_BUDGET_MS - 4, yt + 0.42, f"{TICK_BUDGET_MS - b95:.0f} ms headroom at p95",
            ha="right", va="top", fontsize=11.5, color=INK)

    # legend (two series): swatches, top-right empty area
    handles = [Patch(color=NAVY, label="median"), Patch(color=NAVY_LIGHT, label="95th percentile")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.43, 0.955), frameon=False, fontsize=11.5,
              handlelength=1.2, handleheight=0.9, borderaxespad=0, labelspacing=0.35)
    ax.text(0.445, 0.845, "labels: median / p95, ms", transform=ax.transAxes, fontsize=11, color=MUTED, va="top")

    # summary sentence under the axis
    over = total["ticks_over_budget"]
    fig.text(0.012, 0.162,
             f"Median tick {total['p50_ms']:.0f} ms ({100 * total['p50_ms'] / TICK_BUDGET_MS:.0f} % of budget); "
             f"{over} of {n_ticks:,} ticks went over (max {total['max_ms']:.0f} ms).",
             fontsize=12, color=INK, ha="left")
    provenance(fig, [
        "* Telemetry: over the ticks that built a packet (every 3rd tick). Tier-0 frames carry no images, so",
        "  SGBM stereo, visual odometry and the segmenter are not in this budget.",
        f"Simulated, tier-0, EVAL seeds 0-59, FULL, {n_ticks:,} ticks · Linux container, 4 vCPU, no GPU, 4 episodes in parallel",
        "results/runs_eval_tier0/FULL/*/autonomy/timings.csv",
    ])
    save(fig, "compute_budget")

    sidecar = {
        "figure": "compute_budget",
        "label": "Simulated (tier-0 synthetic depth sensor, no images); compute measured on this machine",
        "machine": "Linux cloud container, 4 vCPU, no GPU, Python 3.11, OMP_NUM_THREADS=1, 4 EVAL episodes in parallel "
                   "(each episode = simulator process + autonomy process), per docs/BUILD_LOG.md 2026-09-30",
        "source": f"{RUNS}/*/autonomy/timings.csv ({n_runs} runs, {n_ticks} ticks)",
        "method": "Per module: numpy median and 95th percentile (linear interpolation) over every tick of all 60 FULL "
                  "EVAL runs. 'Whole tick' is the timings.csv 'total' column: sense -> wheel command plus telemetry "
                  "encoding and debug crops. Telemetry is summarised only over ticks where it ran (column > 0.05 ms); "
                  "its per-tick median over all ticks is 0 because packets are built every 3rd tick.",
        "not_included": "Tier-0 frames carry disparity, no images: SGBM stereo matching, visual odometry and the terrain "
                        "segmenter do not run, so they are not in these timings. Separate stereo-mode measurements "
                        "(shared 4-core laptop, other jobs running): results/perception_timing.json stereo median "
                        "59.5 ms; results/localizer_timing.json VO mean 25.8 ms.",
        "tick_budget_ms": TICK_BUDGET_MS,
        "modules": rows,
        "whole_tick": total,
        "sense_to_wheel_command_compute_ms": {**sense_to_cmd, "source": "timings.csv compute_ms column",
                                              "cross_check": f"{EVAL_JSON} tier0.aggregate.FULL.all compute_ms_p50="
                                                             f"{ev['compute_ms_p50']} p95={ev['compute_ms_p95']} n_ticks={ev['n_ticks']}"},
        "derived": {"median_tick_share_of_budget": round(total["p50_ms"] / TICK_BUDGET_MS, 3),
                    "p95_headroom_ms": round(TICK_BUDGET_MS - total["p95_ms"], 1),
                    "frac_ticks_over_budget": round(over / n_ticks, 6)},
    }
    (OUT / "compute_budget.json").write_text(json.dumps(sidecar, indent=1))
    return sidecar


# ---- link budget ---------------------------------------------------------------------------
def link_budget() -> dict:
    from metagross.autonomy.link import codec
    from metagross.config.defaults import LINK_KBPS, TELEMETRY_HZ

    budget = codec.PACKET_BUDGET_B
    target = codec.PACKET_TARGET_B
    assert budget == int(LINK_KBPS * 1000 / 8 / TELEMETRY_HZ) == 600

    files = sorted((ROOT / RUNS).glob("*/autonomy/telemetry.jsonl"))
    onboard, reenc, rung_onboard, rung_reenc, dts, n_cm = [], [], [], [], [], 0
    for f in files:
        ts = []
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            onboard.append(int(d["packet_bytes"]))
            rung_onboard.append(int(d["link_rung"]))
            n_cm += d.get("costmap_cell_m") is not None
            ts.append(float(d["t"]))
            pkt, info = codec.encode_with_info(codec.telemetry_from_jsonable(d))
            reenc.append(len(pkt))
            rung_reenc.append(info.rung)
        dts += list(np.diff(ts))
    onboard, reenc = np.asarray(onboard), np.asarray(reenc)
    n_pk = int(onboard.size)
    period_s = float(np.median(dts))
    med, mx, mn = float(np.median(onboard)), int(onboard.max()), int(onboard.min())
    rc = Counter(rung_onboard)

    # ---- figure ----
    fig = plt.figure(figsize=(9.0, 5.6), dpi=DPI)
    ax = fig.add_axes([0.095, 0.215, 0.875, 0.70])
    bins = np.arange(240, target + 1, 8)  # numpy closes the last bin on the right, so 480 B lands in [472, 480]
    counts, edges, _ = ax.hist(onboard, bins=bins, color=NAVY, edgecolor=BG, linewidth=1.0, zorder=3)
    ymax = counts.max()
    ax.set_xlim(236, 640)
    ax.set_ylim(0, ymax * 1.42)
    ax.set_xticks([250, 300, 350, 400, 450, 480, 600])
    ax.tick_params(axis="both", labelsize=12, length=0, pad=6)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("Telemetry packet size (bytes)", fontsize=12.5, labelpad=8)
    ax.set_ylabel("Packets per 8-byte bin", fontsize=12.5, labelpad=8)

    top = ymax * 1.42
    # budget: hard line; encoder target: lighter line
    ax.axvline(budget, color=RED, lw=1.8, zorder=5)
    ax.text(budget - 4, top * 0.955, "600 B budget\n(9.6 kbit/s, 2 Hz)", ha="right", va="top", fontsize=12, color=INK,
            linespacing=1.3)
    ax.axvline(target, color=MUTED, lw=1.2, zorder=4)
    ax.annotate("", xy=(budget - 1.5, top * 0.50), xytext=(target + 1.5, top * 0.50),
                arrowprops=dict(arrowstyle="<|-|>", color=MUTED, lw=1.1, shrinkA=0, shrinkB=0))
    ax.text((target + budget) / 2, top * 0.475, f"{budget - target} B kept free\nfor radio framing\nand retries",
            ha="center", va="top", fontsize=11, color=MUTED, linespacing=1.25)
    # median and max
    ax.axvline(med, color=INK, lw=1.2, zorder=4)
    ax.text(med + 4, top * 0.955, f"median\n{med:.0f} B", ha="left", va="top", fontsize=12, color=INK, linespacing=1.25)
    ax.text(target + 4, top * 0.74, f"max {mx} B = encoder\ntarget (80 % of budget)", ha="left", va="top", fontsize=12, color=INK,
            linespacing=1.25)
    # what the packets carried (left empty area)
    full_res = rc.get(0, 0) + rc.get(1, 0)
    half_res = rc.get(2, 0) + rc.get(3, 0)
    ax.text(242, top * 0.955,
            f"{n_pk:,} packets, none over budget.\n"
            f"Every packet carried the 16 m\nego costmap: {100 * full_res / n_pk:.0f} % at 0.25 m cells,\n"
            f"{100 * half_res / n_pk:.0f} % at 0.5 m cells.",
            ha="left", va="top", fontsize=11.5, color=INK, linespacing=1.3)

    provenance(fig, [
        f"Simulated, tier-0, EVAL seeds 0-59, FULL · sizes as encoded onboard (codec v2), one packet every "
        f"{period_s:.1f} s",
        "results/runs_eval_tier0/FULL/*/autonomy/telemetry.jsonl",
    ])
    save(fig, "link_budget")

    sidecar = {
        "figure": "link_budget",
        "label": "Simulated (tier-0 synthetic depth sensor, no images)",
        "source": f"{RUNS}/*/autonomy/telemetry.jsonl ({len(files)} runs, {n_pk} packets)",
        "method": "Histogram (8-byte bins) of packet_bytes as logged by each run: len(codec.encode_with_info(tel)) "
                  "computed onboard in metagross/autonomy/process.py from the full-resolution costmap. Cross-check: "
                  "every logged line was rebuilt with codec.telemetry_from_jsonable and re-encoded with "
                  "codec.encode_with_info (as in tests/test_link_budget.py); the re-encoded sizes differ per packet "
                  "because the log holds the already-pooled/decoded costmap, so the onboard sizes are plotted.",
        "budget_bytes": budget,
        "budget_derivation": f"{LINK_KBPS} kbit/s / 8 / {TELEMETRY_HZ} Hz = {budget} B (codec.PACKET_BUDGET_B)",
        "encoder_target_bytes": target,
        "onboard_bytes": {"n": n_pk, "p50": med, "p95": round(pct(onboard, 95), 1), "max": mx, "min": mn,
                          "mean": round(float(onboard.mean()), 1), "n_over_budget": int((onboard > budget).sum()),
                          "n_over_target": int((onboard > target).sum())},
        "reencoded_bytes_check": {"n": int(reenc.size), "p50": float(np.median(reenc)), "p95": round(pct(reenc, 95), 1),
                                  "max": int(reenc.max()), "n_over_budget": int((reenc > budget).sum()),
                                  "rungs": {str(k): v for k, v in sorted(Counter(rung_reenc).items())}},
        "onboard_rungs": {str(k): v for k, v in sorted(rc.items())},
        "rung_meaning": {"0": "64x64 at 0.25 m, 8 ground levels", "1": "64x64 at 0.25 m, 4 levels",
                         "2": "32x32 at 0.5 m, 8 levels", "3": "32x32 at 0.5 m, 4 levels"},
        "packets_with_costmap": n_cm,
        "share_costmap_0p25m": round(full_res / n_pk, 4),
        "share_costmap_0p5m": round(half_res / n_pk, 4),
        "telemetry_period_s_median": round(period_s, 3),
        "note_period": "The 2 Hz telemetry timer runs on the 5 Hz tick, so packets go out every 0.6 s (1.67 Hz); "
                       "the 600 B budget assumes 2 Hz, so it is conservative.",
        "mean_bitrate_kbps_at_logged_period": round(8 * float(onboard.mean()) / period_s / 1000, 2),
        "worst_bitrate_kbps_at_2hz": round(8 * mx * TELEMETRY_HZ / 1000, 2),
    }
    (OUT / "link_budget.json").write_text(json.dumps(sidecar, indent=1))
    return sidecar


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    c = compute_budget()
    l = link_budget()
    print(json.dumps({"compute_whole_tick": c["whole_tick"], "link": l["onboard_bytes"]}, indent=1))
