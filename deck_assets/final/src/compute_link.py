"""Compute and operator-link budget figures for the final deck, from the held-out EVAL runs.

Run from the repo root:
    OMP_NUM_THREADS=2 python deck_assets/final/src/compute_link.py

Outputs (deck_assets/final/):
    compute_budget.png / .svg / .json   sense-to-wheel-command time per 5 Hz tick, per module, vs the 200 ms tick
    link_budget.png    / .svg / .json   telemetry packet sizes vs the 600 B per-packet budget

Sources (nothing else is read):
    results/runs_eval_tier0/FULL/*/autonomy/timings.csv     one row per control tick (60 EVAL runs)
    results/runs_eval_tier0/FULL/*/autonomy/telemetry.jsonl one line per telemetry packet, with the
                                                            size the onboard encoder produced
    results/closed_loop_eval.json                           tick count and compute p50/p95 (cross-check)
    results/claims.csv                                      the registered compute p50/p95 (cross-check)
    results/link_budget.json                                registered DEV link replay (quoted in the sidecar)
    results/perception_dev.json, results/vo_sim_dev.json    stereo-mode timings (sidecar and caption only)

What the timers cover (metagross/autonomy/node.py): ``compute_ms`` runs from frame arrival to the wheel
command and is the sum of the nine module timers (median un-timed remainder 0.04 ms). The ``telemetry``
column times only the Telemetry message build (ego costmap crop + waypoint resample), every 3rd tick,
after the command. ``total`` adds that and the debug-bundle prep. Packet encoding (codec.encode_with_info)
runs later, in process.py _Logs.write, and is in none of these columns.

Label: Simulated (tier-0 synthetic depth sensor, NO images: SGBM stereo, visual odometry and the terrain
segmenter do not run). Measured while the episodes ran: Linux cloud container, 4 vCPU, no GPU,
4 episodes in parallel (BUILD_LOG 2026-09-30). The embedded compute target is Proposed, not measured.
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
from matplotlib import transforms  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "deck_assets" / "final"
RUNS = "results/runs_eval_tier0/FULL"
EVAL_JSON = "results/closed_loop_eval.json"
CLAIMS_CSV = "results/claims.csv"
LINK_JSON = "results/link_budget.json"
PERC_DEV_JSON = "results/perception_dev.json"
VO_SIM_JSON = "results/vo_sim_dev.json"

# ---- palette (deck brief) ------------------------------------------------------------------
NAVY = "#1F497D"        # ours / FULL; median
NAVY_LIGHT = "#7A93B6"  # same hue, lighter step (p95). dataviz validator vs navy: CVD dE 24.9, contrast >= 3:1
INK = "#1A1A1A"         # text, and the budget/limit lines (a limit is not a hazard, so not red)
MUTED = "#5C5C5C"       # secondary text and provenance, 6.6:1 on white
GRID = "#E6E6E6"
BG = "#FFFFFF"

# type sizes in pt; the figures are 9 in wide at 200 dpi (~1800 px), so 1 pt = 2.78 px there and
# 1.39 px when the slide shows the image ~900 px wide
FS_TICK = 14.0   # tick and row labels (39 px saved, 19 px on the slide)
FS_VAL = 13.5    # value labels and annotations
FS_AXIS = 14.5   # axis titles
FS_NOTE = 13.0   # black notes under the axis
FS_PROV = 11.5   # provenance lines (MUTED)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FS_TICK,
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "svg.fonttype": "path",  # glyphs as outlines: renders identically without DejaVu Sans installed
    "figure.facecolor": BG,
    "axes.facecolor": BG,
})
DPI = 200
FIG_W_IN = 9.0
TICK_BUDGET_MS = 200.0  # 5 Hz control tick

MODULES = [  # (timings.csv column, label) in pipeline order: sense -> act; together they make compute_ms
    ("localizer", "Localisation (EKF + depth odometry)"),
    ("perception", "Perception (ground, hazards, ditches)"),
    ("map", "Rolling map"),
    ("costmap", "Costmap"),
    ("global", "Global planner"),
    ("supervisor", "Safety supervisor"),
    ("governor", "Speed governor"),
    ("mppi", "MPPI, 512 rollouts"),
    ("gate_mixer", "Safety gate + wheel mixer"),
]
TEL_EVERY_N_TICKS = 3  # 2 Hz timer on the 5 Hz tick: a message is built on ticks 0, 3, 6, ... (asserted below)


def notes(fig, lines: list[tuple[str, str, float]], y0_in: float, step_in: float) -> None:
    """Text lines stacked upward from ``y0_in`` inches above the bottom edge: (text, colour, size)."""
    h_in = fig.get_figheight()
    for i, (text, colour, size) in enumerate(reversed(lines)):
        fig.text(0.012, (y0_in + i * step_in) / h_in, text, fontsize=size, color=colour, ha="left", va="bottom")


def axes_in(fig, left: float, bottom_in: float, top_pad_in: float, width: float):
    """Axes with its bottom and top edges fixed in inches (so the notes block never collides with the x label)."""
    h_in = fig.get_figheight()
    return fig.add_axes([left, bottom_in / h_in, width, (h_in - bottom_in - top_pad_in) / h_in])


def save(fig, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.12, facecolor=BG)
    plt.close(fig)


def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q))


def registered(claim_id: str) -> str:
    with open(ROOT / CLAIMS_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["id"] == claim_id:
                return row["value"]
    raise KeyError(f"{claim_id} not in {CLAIMS_CSV}")


def claim(cid: str, value, unit: str, source: str, note: str) -> dict:
    """A row in the embedded-claims format metagross/eval/claims.py reads from results/*.json."""
    return {"id": cid, "value": value, "unit": unit, "label": "Simulated", "source": source, "note": note}


# ---- compute budget ------------------------------------------------------------------------
def load_timings() -> tuple[dict[str, np.ndarray], list[Path]]:
    files = sorted((ROOT / RUNS).glob("*/autonomy/timings.csv"))
    cols: dict[str, list] = {"run": []}
    for f in files:
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                cols["run"].append(f.parents[1].name)
                for k, v in row.items():
                    cols.setdefault(k, []).append(float(v))
    return {k: np.asarray(v) for k, v in cols.items()}, files


def telemetry_ticks(files: list[Path]) -> tuple[np.ndarray, int]:
    """Telemetry-message build times on the ticks that built one; asserts the count equals the packets logged."""
    out, n_packets = [], 0
    for f in files:
        with open(f, newline="") as fh:
            rows = list(csv.DictReader(fh))
        sel = [float(r["telemetry"]) for r in rows if int(float(r["tick"])) % TEL_EVERY_N_TICKS == 0]
        n_pk = sum(1 for line in (f.parent / "telemetry.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
        assert len(sel) == n_pk, (f, len(sel), n_pk)
        out += sel
        n_packets += n_pk
    return np.asarray(out), n_packets


def compute_budget() -> dict:
    tm, files = load_timings()
    n_runs, n_ticks = len(files), int(tm["compute_ms"].size)
    ev = json.loads((ROOT / EVAL_JSON).read_text())["tier0"]["aggregate"]["FULL"]["all"]
    cmd = tm["compute_ms"]
    p50, p95 = float(np.median(cmd)), pct(cmd, 95)
    # the drawn summary row must be the registered claim, via both the EVAL aggregate and the ledger
    reg50 = float(registered("closed_loop_eval_tier0_FULL_compute_p50"))
    reg95 = float(registered("closed_loop_eval_tier0_FULL_compute_p95"))
    assert ev["n_ticks"] == n_ticks, (ev["n_ticks"], n_ticks)
    assert abs(ev["compute_ms_p50"] - p50) < 0.06 and abs(reg50 - p50) < 0.06, (ev["compute_ms_p50"], reg50, p50)
    assert abs(ev["compute_ms_p95"] - p95) < 0.06 and abs(reg95 - p95) < 0.06, (ev["compute_ms_p95"], reg95, p95)
    # compute_ms is the nine module timers plus a negligible remainder
    remainder = cmd - np.sum([tm[k] for k, _ in MODULES], axis=0)
    assert float(np.median(remainder)) < 0.2, float(np.median(remainder))

    rows = []
    for key, label in MODULES:
        a = tm[key]
        rows.append({"key": key, "label": label, "n": int(a.size), "p50_ms": round(float(np.median(a)), 2),
                     "p95_ms": round(pct(a, 95), 2), "max_ms": round(float(a.max()), 2)})
    over = int((cmd > TICK_BUDGET_MS).sum())
    i_max = int(np.argmax(cmd))
    worst = {"compute_ms": round(float(cmd[i_max]), 1), "run": str(tm["run"][i_max]),
             "tick": int(tm["tick"][i_max]), "perception_ms": round(float(tm["perception"][i_max]), 1)}
    summary = {"label": "Sense to wheel command (compute_ms)", "n": n_ticks, "p50_ms": round(p50, 1),
               "p95_ms": round(p95, 1), "p99_ms": round(pct(cmd, 99), 1), "max_ms": round(float(cmd.max()), 1),
               "ticks_over_budget": over, "worst_tick": worst,
               "registered": {"closed_loop_eval_tier0_FULL_compute_p50": reg50,
                              "closed_loop_eval_tier0_FULL_compute_p95": reg95}}
    headroom = TICK_BUDGET_MS - reg95
    tel, n_packets = telemetry_ticks(files)
    tot = tm["total"]

    # ---- figure ----
    labels = [r["label"] for r in rows]
    v50 = [r["p50_ms"] for r in rows] + [reg50]
    v95 = [r["p95_ms"] for r in rows] + [reg95]
    n = len(v50)
    y = np.arange(n, dtype=float)
    y[-1] += 0.75  # air between the modules and the summary row
    yt = y[-1]
    h = 0.58

    fig = plt.figure(figsize=(FIG_W_IN, 7.8), dpi=DPI)
    ax = axes_in(fig, 0.43, bottom_in=1.92, top_pad_in=0.12, width=0.545)
    ax.barh(y, v95, height=h, color=NAVY_LIGHT, linewidth=0, zorder=2)
    ax.barh(y, v50, height=h, color=NAVY, linewidth=0, zorder=3)
    for yi, a, b in zip(y, v50, v95):
        fmt = (lambda v: f"{v:.2f}") if b < 1 else (lambda v: f"{v:.1f}")
        ax.text(b + 2.2, yi, f"{fmt(a)} / {fmt(b)}", va="center", ha="left", fontsize=FS_VAL, color=INK, zorder=4,
                fontweight="bold" if yi == yt else "normal")
    ax.set_yticks(y[:-1])
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    # summary row label: two lines, the scope in the second
    lab = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(-0.028, yt - 0.20, "Sense to wheel command", transform=lab, ha="right", va="center",
            fontsize=FS_TICK, fontweight="bold", color=INK)
    ax.text(-0.028, yt + 0.27, "tier-0, no images", transform=lab, ha="right", va="center",
            fontsize=FS_TICK - 1, color=INK)
    ax.set_xlim(0, 214)
    ax.set_ylim(yt + 1.55, -1.45)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.tick_params(axis="x", labelsize=FS_TICK, length=0, pad=6)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("Time per 5 Hz control tick (ms)", fontsize=FS_AXIS, labelpad=8)
    ysep = (y[-2] + y[-1]) / 2
    ax.plot([-0.78, 1.0], [ysep, ysep], transform=ax.get_yaxis_transform(), color=GRID, lw=1.0, clip_on=False)

    # the budget: a limit line in ink
    ax.axvline(TICK_BUDGET_MS, color=INK, lw=1.8, zorder=5)
    ax.text(TICK_BUDGET_MS - 2.5, -1.08, "200 ms tick budget (5 Hz)", ha="right", va="center", fontsize=FS_VAL,
            color=INK)

    # headroom bracket under the summary bar: from the p95 bar end to the budget line
    yb = yt + h / 2 + 0.22
    kw = dict(color=INK, lw=1.2, zorder=4, solid_capstyle="butt")
    ax.plot([reg95, TICK_BUDGET_MS], [yb, yb], **kw)
    for xe in (reg95, TICK_BUDGET_MS):
        ax.plot([xe, xe], [yb - 0.13, yb + 0.02], **kw)
    ax.text((reg95 + TICK_BUDGET_MS) / 2, yb + 0.12, f"{headroom:.0f} ms free at p95", ha="center", va="top",
            fontsize=FS_VAL, color=INK)

    # legend (two ordinal series), top-right empty area
    handles = [Patch(color=NAVY, label="median"), Patch(color=NAVY_LIGHT, label="95th percentile")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.43, 0.70), frameon=False, fontsize=FS_VAL,
              handlelength=1.2, handleheight=0.9, borderaxespad=0, labelspacing=0.35)
    ax.text(0.445, 0.575, "labels: median / p95, ms", transform=ax.transAxes, fontsize=FS_VAL - 0.5, color=MUTED,
            va="top")

    t_worst = float(tm["t"][i_max])
    notes(fig, [
        ("Not in this budget (tier-0 has no images): SGBM stereo, visual odometry, segmenter.", INK, FS_NOTE),
        (f"{over} of {n_ticks:,} ticks went over 200 ms ({worst['compute_ms']:.0f} ms, {t_worst:.1f} s after the "
         f"start of one run).", INK, FS_NOTE),
        ("Simulated, tier-0, EVAL seeds 0-59, FULL · cloud container, 4 vCPU shared by 4 episodes, no GPU;", MUTED,
         FS_PROV),
        ("embedded target not measured · results/runs_eval_tier0/FULL/*/autonomy/timings.csv", MUTED, FS_PROV),
    ], y0_in=0.04, step_in=0.29)
    save(fig, "compute_budget")

    # stereo-mode timings measured elsewhere (caption / sidecar only; NOT drawn, NOT additive)
    pdev = json.loads((ROOT / PERC_DEV_JSON).read_text())["timing_ab_stereo"]["after"]
    vo = json.loads((ROOT / VO_SIM_JSON).read_text())["localizer_timing_paired"]["current"]
    src_tim = f"{RUNS}/*/autonomy/timings.csv"
    note_c = "EVAL closed loop, tier-0 (no images), FULL, seeds 0-59, n=10454 ticks; 4 vCPU cloud container, 4 episodes in parallel"
    to_register = [claim(f"deck_compute_eval_tier0_FULL_{r['key']}_p50_ms", r["p50_ms"], "ms", src_tim,
                         f"{note_c}; median of the {r['key']} timer") for r in rows]
    to_register += [claim(f"deck_compute_eval_tier0_FULL_{r['key']}_p95_ms", r["p95_ms"], "ms", src_tim,
                          f"{note_c}; 95th percentile of the {r['key']} timer") for r in rows]
    to_register += [
        claim("deck_compute_eval_tier0_FULL_ticks_over_200ms", f"{over} of {n_ticks}", "ticks", src_tim,
              f"{note_c}; compute_ms > 200"),
        claim("deck_compute_eval_tier0_FULL_p95_headroom_ms", round(headroom, 1), "ms", EVAL_JSON,
              "200 ms tick minus registered closed_loop_eval_tier0_FULL_compute_p95"),
        claim("deck_telemetry_msg_build_eval_tier0_FULL_p50_p95_ms", f"{np.median(tel):.1f} / {pct(tel, 95):.1f}",
              "ms", src_tim, f"{note_c}; telemetry column on the {tel.size} ticks that built a message (tick % 3 == 0)"),
    ]
    sidecar = {
        "figure": "compute_budget",
        "label": "Simulated (tier-0 synthetic depth sensor, no images); compute measured while the EVAL runs ran",
        "machine": "Linux cloud container, 4 vCPU, no GPU, Python 3.11, OMP_NUM_THREADS=1, 4 EVAL episodes in parallel "
                   "(each episode = simulator process + autonomy process), per docs/BUILD_LOG.md 2026-09-30. "
                   "The embedded target (Jetson Orin Nano-class) is Proposed and not measured (README).",
        "source": f"{src_tim} ({n_runs} runs, {n_ticks} ticks)",
        "method": "Per module: numpy median and 95th percentile (linear interpolation) over every tick of all 60 FULL "
                  "EVAL runs. Summary row = timings.csv compute_ms (frame in -> wheel command), which is the sum of "
                  "the nine module timers (median remainder 0.04 ms); its median/p95 equal the registered claims "
                  "closed_loop_eval_tier0_FULL_compute_p50/p95 (asserted against results/closed_loop_eval.json and "
                  "results/claims.csv). Headroom = 200 ms - registered p95.",
        "timers": "node.py: compute_ms stops at the wheel command. The telemetry column times only the Telemetry "
                  "message build (ego_costmap_u4 costmap crop + resample_path waypoints), after the command. 'total' "
                  "= compute_ms + that + debug-bundle prep (rollout transform, map crop). Packet encoding "
                  "(codec.encode_with_info) runs later in process.py _Logs.write and is in none of these columns.",
        "not_included": {
            "what": "Tier-0 frames carry disparity, no images: SGBM stereo matching, visual odometry and the terrain "
                    "segmenter do not run, so they are not in these timings.",
            "stereo_mode_measured_separately": {
                "perception_incl_sgbm_and_segmenter_1of3_p50_p95_ms": [pdev["p50_ms"], pdev["p95_ms"]],
                "perception_source": f"{PERC_DEV_JSON}#timing_ab_stereo.after (registered "
                                     "perc_dev_stereo_perception_p50_ms_after / p95_ms_after); 85 DEV stereo frames, "
                                     "shared loaded laptop",
                "localizer_update_incl_vo_p50_p95_ms": [round(vo["localizer_ms_median"], 1),
                                                         round(vo["localizer_ms_p95"], 1)],
                "localizer_source": f"{VO_SIM_JSON}#localizer_timing_paired.current (p95 registered as "
                                    "localizer_p95_ms_sim_paired); loaded laptop, inflated vs idle (file note: 23 / 41 ms)",
            },
            "warning": "These are different machines and loads: do not add them to the tier-0 bars and call the sum "
                       "measured. The 111 ms p95 headroom applies to tier-0 only.",
        },
        "tick_budget_ms": TICK_BUDGET_MS,
        "modules": rows,
        "summary_row_sense_to_wheel_command": summary,
        "whole_tick_total_column": {"p50_ms": round(float(np.median(tot)), 2), "p95_ms": round(pct(tot, 95), 2),
                                    "max_ms": round(float(tot.max()), 2),
                                    "ticks_over_budget": int((tot > TICK_BUDGET_MS).sum()),
                                    "note": "not drawn; see 'timers'"},
        "telemetry_message_build": {"n": int(tel.size), "n_packets_logged": n_packets,
                                    "p50_ms": round(float(np.median(tel)), 2), "p95_ms": round(pct(tel, 95), 2),
                                    "note": "not drawn: built after the wheel command, on ticks with tick % 3 == 0 "
                                            "(count asserted equal to the logged packets per run)"},
        "derived": {"p95_headroom_ms": round(headroom, 1),
                    "median_share_of_budget": round(reg50 / TICK_BUDGET_MS, 3),
                    "frac_ticks_over_budget": round(over / n_ticks, 6)},
        "registration_status": "Only the summary row (62.5 / 88.7 ms) is in results/claims.csv. Every other drawn "
                               "number is listed in claims_to_register (embedded-claims format of "
                               "metagross/eval/claims.py) and must be registered before the slide ships.",
        "claims_to_register": to_register,
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
    assert n_cm == n_pk  # every packet carried a costmap
    period_s = float(np.median(dts))
    med, mx, mn = float(np.median(onboard)), int(onboard.max()), int(onboard.min())
    n_over = int((onboard > budget).sum())
    rc = Counter(rung_onboard)
    full_res = rc.get(0, 0) + rc.get(1, 0)
    half_res = rc.get(2, 0) + rc.get(3, 0)
    assert full_res + half_res == n_pk
    dev = json.loads((ROOT / LINK_JSON).read_text())["groups"]["tier0_FULL"]

    # ---- figure ----
    fig = plt.figure(figsize=(FIG_W_IN, 6.5), dpi=DPI)
    ax = axes_in(fig, 0.105, bottom_in=1.92, top_pad_in=0.12, width=0.865)
    bins = np.arange(240, target + 1, 8)  # numpy closes the last bin on the right, so 480 B lands in [472, 480]
    counts, _, _ = ax.hist(onboard, bins=bins, color=NAVY, edgecolor=BG, linewidth=1.0, zorder=3)
    ymax = float(counts.max())
    top = ymax * 1.62
    ax.set_xlim(236, 640)
    ax.set_ylim(0, top)
    ax.set_xticks([250, 300, 350, 400, 450, 480, 600])
    ax.tick_params(axis="both", labelsize=FS_TICK, length=0, pad=6)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("Telemetry packet size (bytes)", fontsize=FS_AXIS, labelpad=8)
    ax.set_ylabel("Packets per 8-byte bin", fontsize=FS_AXIS, labelpad=8)

    # budget (solid ink) and encoder target (dashed ink)
    ax.axvline(budget, color=INK, lw=1.9, zorder=5)
    ax.text(budget - 6, top * 0.975, "600 B budget\nper packet on the\nassumed 9.6 kbit/s\nlink at 2 Hz",
            ha="right", va="top", fontsize=FS_VAL, color=INK, linespacing=1.3)
    ax.axvline(target, color=INK, lw=1.3, ls=(0, (5, 3)), zorder=4)
    ax.text(target + 6, top * 0.62, f"max {mx} B =\nencoder target\n(80 % of budget)", ha="left", va="top",
            fontsize=FS_VAL, color=INK, linespacing=1.3)
    ya = top * 0.33
    ax.annotate("", xy=(budget - 1.5, ya), xytext=(target + 1.5, ya),
                arrowprops=dict(arrowstyle="<|-|>", color=MUTED, lw=1.1, shrinkA=0, shrinkB=0))
    ax.text((target + budget) / 2, ya - top * 0.03, f"{budget - target} B margin for\nlink framing, retries",
            ha="center", va="top", fontsize=FS_VAL - 0.5, color=MUTED, linespacing=1.25)
    # median: a short line through the histogram, labelled above it
    ax.plot([med, med], [0, ymax * 1.06], color=INK, lw=1.2, zorder=4)
    ax.text(med, ymax * 1.08, f"median {med:.0f} B", ha="center", va="bottom", fontsize=FS_VAL, color=INK)
    # what the packets carried (top-left empty area)
    ax.text(243, top * 0.975,
            f"{n_pk:,} packets, none over budget.\n"
            f"Each carried the 16 m ego costmap:\n"
            f"{100 * full_res / n_pk:.0f} % at 0.25 m cells, {100 * half_res / n_pk:.0f} % at 0.5 m.",
            ha="left", va="top", fontsize=FS_VAL, color=INK, linespacing=1.3)

    notes(fig, [
        ("Sizes only: packets were encoded and logged onboard; no radio or link", INK, FS_NOTE),
        ("emulator was in the loop, so delivery and loss are not shown.", INK, FS_NOTE),
        (f"Simulated, tier-0, EVAL seeds 0-59, FULL · onboard codec v2 sizes, one packet every {period_s:.1f} s",
         MUTED, FS_PROV),
        ("results/runs_eval_tier0/FULL/*/autonomy/telemetry.jsonl", MUTED, FS_PROV),
    ], y0_in=0.04, step_in=0.29)
    save(fig, "link_budget")

    src_tel = f"{RUNS}/*/autonomy/telemetry.jsonl"
    note_l = ("EVAL closed loop, tier-0, FULL, seeds 0-59; packet_bytes as encoded onboard (codec v2); "
              "sizes only, link not in the loop; budget 600 B = 9.6 kbit/s at 2 Hz (design assumption)")
    to_register = [
        claim("deck_link_eval_tier0_FULL_n_packets", n_pk, "packets", src_tel, note_l),
        claim("deck_link_eval_tier0_FULL_p50_bytes", med, "B", src_tel, note_l),
        claim("deck_link_eval_tier0_FULL_p95_bytes", round(pct(onboard, 95), 1), "B", src_tel, note_l),
        claim("deck_link_eval_tier0_FULL_max_bytes", mx, "B", src_tel, note_l),
        claim("deck_link_eval_tier0_FULL_n_over_budget", n_over, "packets", src_tel, note_l),
        claim("deck_link_eval_tier0_FULL_share_costmap_0p25m", round(full_res / n_pk, 3), "fraction", src_tel,
              f"{note_l}; link_rung 0 or 1"),
        claim("deck_link_eval_tier0_FULL_share_costmap_0p5m", round(half_res / n_pk, 3), "fraction", src_tel,
              f"{note_l}; link_rung 2 or 3"),
        claim("deck_link_eval_tier0_FULL_period_s", round(period_s, 2), "s", src_tel,
              "median spacing of logged packets (2 Hz timer on the 5 Hz tick)"),
    ]
    sidecar = {
        "figure": "link_budget",
        "label": "Simulated (tier-0 synthetic depth sensor, no images); packet sizes only, no link in the loop",
        "source": f"{src_tel} ({len(files)} runs, {n_pk} packets)",
        "method": "Histogram (8-byte bins) of packet_bytes as logged by each run: len(codec.encode_with_info(tel)) "
                  "computed onboard in metagross/autonomy/process.py from the full-resolution costmap. Cross-check: "
                  "every logged line was rebuilt with codec.telemetry_from_jsonable and re-encoded with "
                  "codec.encode_with_info (as in tests/test_link_budget.py); the re-encoded sizes differ per packet "
                  "because the log holds the already-pooled/decoded costmap, so the onboard sizes are plotted.",
        "scope": "The 9.6 kbit/s link (with 20 % loss, 0.4 s latency) is a design assumption (README). In these EVAL "
                 "runs packets were only encoded and logged; the link emulator is BUILT, NOT IN LOOP "
                 "(docs/PIPELINE_STATUS.md row 14). 'Every packet carried the costmap' is true of packets sent, not of "
                 "packets an operator would receive: the registered DEV emulation delivered "
                 f"{dev['emulated_link_after']['delivered']} of {dev['emulated_link_after']['sent']} "
                 f"({LINK_JSON}#groups.tier0_FULL.emulated_link_after).",
        "budget_bytes": budget,
        "budget_derivation": f"{LINK_KBPS} kbit/s / 8 / {TELEMETRY_HZ} Hz = {budget} B (codec.PACKET_BUDGET_B)",
        "encoder_target_bytes": target,
        "encoder_target_reason": "codec.py: 80 % of the budget, headroom for link framing / retries",
        "onboard_bytes": {"n": n_pk, "p50": med, "p95": round(pct(onboard, 95), 1), "max": mx, "min": mn,
                          "mean": round(float(onboard.mean()), 1), "n_over_budget": n_over,
                          "n_over_target": int((onboard > target).sum())},
        "reencoded_bytes_check": {"n": int(reenc.size), "p50": float(np.median(reenc)), "p95": round(pct(reenc, 95), 1),
                                  "max": int(reenc.max()), "n_over_budget": int((reenc > budget).sum()),
                                  "rungs": {str(k): v for k, v in sorted(Counter(rung_reenc).items())}},
        "registered_dev_replay_for_comparison": {
            "source": f"{LINK_JSON}#groups.tier0_FULL.after_bytes",
            "claims": "link_packet_after_tier0_FULL_p50_bytes / p100_bytes / frac_over_budget",
            "n": dev["n_packets"], "p50": dev["after_bytes"]["p50"], "p100": dev["after_bytes"]["p100"],
            "frac_over_budget": dev["after_frac_over_budget"]},
        "onboard_rungs": {str(k): v for k, v in sorted(rc.items())},
        "rung_meaning": {"0": "64x64 at 0.25 m, 8 ground levels", "1": "64x64 at 0.25 m, 4 levels",
                         "2": "32x32 at 0.5 m, 8 levels", "3": "32x32 at 0.5 m, 4 levels"},
        "packets_with_costmap": n_cm,
        "share_costmap_0p25m": round(full_res / n_pk, 4),
        "share_costmap_0p5m": round(half_res / n_pk, 4),
        "telemetry_period_s_median": round(period_s, 3),
        "note_period": "The 2 Hz telemetry timer runs on the 5 Hz tick, so packets go out every 0.6 s (1.67 Hz); "
                       "the 600 B budget assumes 2 Hz, so it is conservative. Do not say 'exactly 2 Hz'.",
        "mean_bitrate_kbps_at_logged_period": round(8 * float(onboard.mean()) / period_s / 1000, 2),
        "worst_bitrate_kbps_at_2hz": round(8 * mx * TELEMETRY_HZ / 1000, 2),
        "registration_status": "None of the EVAL-onboard numbers drawn here are in results/claims.csv (the registered "
                               "link claims are the DEV replay above). Register claims_to_register before the slide "
                               "ships, or quote only the registered DEV numbers.",
        "claims_to_register": to_register,
    }
    (OUT / "link_budget.json").write_text(json.dumps(sidecar, indent=1))
    return sidecar


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    c = compute_budget()
    l = link_budget()
    print(json.dumps({"compute_summary": c["summary_row_sense_to_wheel_command"], "link": l["onboard_bytes"]}, indent=1))
