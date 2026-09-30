"""Deck figure for the closed-loop result (slide 2 "RESULT" slot, 588 x 220 slide px).

Reads a closed-loop summary written by :mod:`metagross.sim.closed_loop_summary`
(``results/closed_loop_eval.json`` for the final figure) and draws, for the hazard families
(F2 ditch field, F3 crest + ditch, F4 sudden obstacle), how often each configuration reached B and
how many unsafe outcomes it had (ditch entries + collisions + off-map exits + water entries), plus the
all-family FULL success count in the title. Every number drawn is read from the summary file; the
honesty chip says Simulated and names the split.

Usage::

    python deck_assets/eval_figures.py --summary results/closed_loop_eval.json --mode stereo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from metagross.eval import plot_style as ps  # noqa: E402

LOG = logging.getLogger("eval_figures")

HAZARD_FAMILIES = ("F2_ditch_field", "F3_crest_ditch", "F4_sudden_obstacle")
SLOT_PX = (588, 220)  # slide-2 RESULT slot, 1x slide pixels
CONFIG_LABELS = {"FULL": "METAGROSS", "T1_TYPICAL": "Typical stack"}
CONFIG_COLORS = {"FULL": ps.TOKENS["accent"], "T1_TYPICAL": "#94A3B8"}
FONT_PT = 8.0  # ~33 px at 2x export = ~17 px on the slide (>= 14 px legibility floor)


def hazard_totals(by_family: dict[str, Any]) -> dict[str, int]:
    """Sum n / successes / unsafe outcomes over the hazard families present in ``by_family``."""
    tot = {"n": 0, "success": 0, "unsafe": 0}
    for fam in HAZARD_FAMILIES:
        a = by_family.get(fam)
        if not a:
            continue
        tot["n"] += int(a["n"])
        tot["success"] += int(a["n_success"])
        tot["unsafe"] += int(a["ditch_entries"]) + int(a["collisions"]) + int(a["out_of_bounds"]) + int(a.get("water_entries", 0))
    return tot


def render(summary: dict[str, Any], mode: str, out_name: str) -> list[Path]:
    """Draw the slot figure from ``summary[mode]['aggregate']``; returns written paths."""
    agg = summary[mode]["aggregate"]
    split = str(summary.get("split", "dev")).upper()
    if "T1" in agg and "T1_TYPICAL" not in agg:  # DEV verifier runs used the short name
        agg = {**agg, "T1_TYPICAL": agg["T1"]}
    cfgs = [c for c in ("FULL", "T1_TYPICAL") if c in agg]
    if "FULL" not in cfgs:
        raise ValueError(f"no FULL config in {mode} aggregate: {sorted(agg)}")
    full_all = agg["FULL"]["all"]
    tots = {c: hazard_totals(agg[c]["by_family"]) for c in cfgs}

    with ps.deck_style():
        fig, axes = ps.new_figure(SLOT_PX, nrows=1, ncols=2, gridspec_kw={"width_ratios": [1, 1]})
        metrics = (("success", "Reached B, F2-F4"), ("unsafe", "Unsafe outcomes, F2-F4"))
        for k_ax, (ax, (key, title)) in enumerate(zip(axes, metrics)):
            vals = [tots[c][key] for c in cfgs]
            ys = list(range(len(cfgs)))[::-1]
            ax.barh(ys, vals, height=0.62, color=[CONFIG_COLORS[c] for c in cfgs])
            n = max(tots[c]["n"] for c in cfgs)
            xmax = max(max(vals), 1) * 1.45 if key == "unsafe" else n * 1.3
            ax.set_xlim(0, xmax)
            for y, v, c in zip(ys, vals, cfgs):
                txt = f"{v}/{tots[c]['n']}" if key == "success" else f"{v}"
                ax.text(v + xmax * 0.03, y, txt, va="center", ha="left", fontsize=FONT_PT, fontweight="bold",
                        color=ps.TOKENS["text"], family=ps.mono_family())
            ax.set_yticks(ys, [CONFIG_LABELS[c] for c in cfgs] if k_ax == 0 else [""] * len(cfgs), fontsize=FONT_PT - 0.5)
            ax.set_xticks([])
            ax.set_title(title, fontsize=FONT_PT, loc="left", pad=4)
            ax.spines["bottom"].set_visible(False)
            ps.style_axes(ax, grid="none")
        fig.suptitle(f"Reached B in {full_all['n_success']}/{full_all['n']} {split} scenarios", x=0.01, ha="left",
                     fontsize=FONT_PT + 1.5, fontweight="bold", color=ps.TOKENS["text"])
        ps.add_honesty_chip(fig, "Simulated", split)
        paths = ps.save_fig(fig, out_name)
    LOG.info("hazard totals: %s; FULL all: %d/%d", tots, full_all["n_success"], full_all["n"])
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, default=REPO / "results" / "closed_loop_eval.json")
    ap.add_argument("--mode", choices=("stereo", "tier0"), default="stereo")
    ap.add_argument("--out-name", default="slots/s2_result")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    render(json.loads(a.summary.read_text(encoding="utf-8")), a.mode, a.out_name)


if __name__ == "__main__":
    main()
