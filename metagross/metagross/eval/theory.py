"""Analytic "theory" figures for the deck: why unknown is never free.

Label for every number produced here: **Estimated — analytic model** (closed-form
geometry, no measurement). All platform numbers come from
:mod:`metagross.config.defaults`; nothing is duplicated.

Models
------
Ditch (negative obstacle) detectability, Matthies & Rankin (IROS 2003), Eq. (1).
A camera at height ``H`` [m] looking at a ditch of width ``w`` [m] whose near lip
is at horizontal range ``R`` [m] sees the opening under the angle::

    theta_ditch(R) ~= H w / (R (R + w))            [rad]   (falls as 1/R^2)

The exact ground-plane geometry is ``atan(H/R) - atan(H/(R+w))``; both are
provided. A positive obstacle of height ``h`` subtends ``theta_pos ~= h / R`` (1/R).
An object is *detectable* while it covers at least ``n`` pixels, i.e.
``theta >= n * IFOV`` with ``IFOV = 1 / fx`` [rad/px] (``n = MIN_PIXELS_ON_TARGET``).
Solving ``R^2 + w R - H w fx / n = 0`` gives the first-detection range::

    R_det(w, H) = ( -w + sqrt(w^2 + 4 H w fx / n) ) / 2

Seen-distance speed governor. The vehicle must stop inside the range it has
certified::

    d_stop(v) = v^2 / (2 a) + v T_r + B  <=  R          ->
    v_max(R)  = a ( -T_r + sqrt(T_r^2 + 2 (R - B) / a) ),   0 if R <= B

with ``a = BRAKE_DECEL_MPS2``, ``T_r`` [s] reaction time, ``B = GOVERNOR_MARGIN_M``.
The *safe-speed envelope* is ``v_max(R_det(w, H))`` over mast height and design
ditch width.

Stereo depth error (first-order propagation of disparity noise)::

    sigma_Z(Z) = Z^2 sigma_d / (fx b)          [m]

with ``b`` the stereo baseline [m] and ``sigma_d`` the disparity noise [px].

Run ``python -m metagross.eval.theory`` to write ``deck_assets/theory/*.png|svg``
and ``results/theory.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from metagross.config.defaults import (
    BASELINE_M,
    BRAKE_DECEL_MPS2,
    CAM_HEIGHT_M,
    CAM_MAX_RANGE_M,
    DESIGN_DITCH_WIDTH_M,
    FX,
    GOVERNOR_MARGIN_M,
    MIN_PIXELS_ON_TARGET,
    VEHICLE,
)
from metagross.contracts.messages import CELL_COLORS, CellState
from metagross.eval import plot_style as ps

LOG = logging.getLogger(__name__)

HONESTY_LABEL = "Estimated"
HONESTY_DETAIL = "analytic model"

# Reaction time used by the deck envelope [s]: ~one 5 Hz frame + compute + 0.2 s actuator lag
# (the live governor measures T_r each tick; this is the deck's fixed design value).
T_REACTION_S = 0.6
# Typical sub-pixel disparity noise of SGBM on textured outdoor scenes [px]; an assumption,
# bracketed in the figure by SIGMA_D_BAND_PX.
SIGMA_D_PX = 0.25
SIGMA_D_BAND_PX = (0.1, 0.5)
# OAK-D Lite class baseline [m] (docs/REFERENCES.md [OAK-D-Lite]); comparison curve only.
BASELINE_OAKD_LITE_M = 0.075
# BEL Robotic Surveillance Platform maximum speed, 3.6 km/h (docs/REFERENCES.md [BEL-RSP]).
BEL_RSP_MAX_MPS = 3.6 / 3.6
# Ditch widths and positive-obstacle height shown in figure (a) [m].
DITCH_WIDTHS_M = (0.3, 0.5, 1.0)
ROCK_HEIGHT_M = 0.3
# Speeds whose stopping distance is marked in figure (a) [m/s].
MARK_SPEEDS_MPS = (1.0, 2.0)
# Envelope grid (figure b): mast height [m] x design ditch width [m].
ENVELOPE_H_RANGE_M = (0.4, 1.6)
ENVELOPE_W_RANGE_M = (0.2, 1.2)
ENVELOPE_N = 121  # grid points per axis (0.01 m steps)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_JSON = REPO_ROOT / "results" / "theory.json"
THEORY_DIR = "theory"  # sub-directory of deck_assets/


@dataclass(frozen=True, slots=True)
class TheoryParams:
    """Inputs of the analytic models (defaults from metagross.config.defaults)."""

    fx_px: float = FX  # focal length [px]
    n_px: float = float(MIN_PIXELS_ON_TARGET)  # pixels on target for detection
    cam_height_m: float = CAM_HEIGHT_M  # optical centre above ground [m]
    brake_decel_mps2: float = BRAKE_DECEL_MPS2  # a [m/s^2]
    t_reaction_s: float = T_REACTION_S  # T_r [s]
    margin_m: float = GOVERNOR_MARGIN_M  # B [m]
    v_platform_mps: float = VEHICLE.max_speed_mps  # platform speed cap [m/s]
    baseline_m: float = BASELINE_M  # stereo baseline [m]
    sigma_d_px: float = SIGMA_D_PX  # disparity noise [px]
    design_ditch_w_m: float = DESIGN_DITCH_WIDTH_M  # governor design ditch width [m]
    max_range_m: float = CAM_MAX_RANGE_M  # stereo range cut-off [m]

    @property
    def ifov_rad(self) -> float:
        """Instantaneous field of view of one pixel [rad/px]."""
        return 1.0 / self.fx_px

    @property
    def theta_min_rad(self) -> float:
        """Detection threshold angle n * IFOV [rad]."""
        return self.n_px * self.ifov_rad


# =========================================================================== formulas
def ditch_angle_rad(r_m: np.ndarray | float, w_m: float, h_m: float) -> np.ndarray:
    """Matthies & Rankin visible angle of a ditch opening, ``H w / (R (R + w))`` [rad].

    ``r_m``: horizontal range from the camera to the near lip [m] (> 0).
    """
    r = np.asarray(r_m, dtype=np.float64)
    return h_m * w_m / (r * (r + w_m))


def ditch_angle_exact_rad(r_m: np.ndarray | float, w_m: float, h_m: float) -> np.ndarray:
    """Exact flat-ground angle between the near and far lip, ``atan(H/R) - atan(H/(R+w))`` [rad]."""
    r = np.asarray(r_m, dtype=np.float64)
    return np.arctan(h_m / r) - np.arctan(h_m / (r + w_m))


def positive_angle_rad(r_m: np.ndarray | float, obstacle_h_m: float) -> np.ndarray:
    """Small-angle subtense of a positive obstacle of height h at range R, ``h / R`` [rad]."""
    return obstacle_h_m / np.asarray(r_m, dtype=np.float64)


def ditch_detection_range_m(w_m: np.ndarray | float, h_m: np.ndarray | float, fx_px: float, n_px: float) -> np.ndarray:
    """First-detection range [m] of a ditch: positive root of ``R^2 + w R - H w fx / n = 0``.

    Vectorised over ``w_m`` and ``h_m`` (broadcasting)."""
    w = np.asarray(w_m, dtype=np.float64)
    h = np.asarray(h_m, dtype=np.float64)
    k = h * w * fx_px / n_px
    return (-w + np.sqrt(w * w + 4.0 * k)) / 2.0


def positive_detection_range_m(obstacle_h_m: float, fx_px: float, n_px: float) -> float:
    """First-detection range [m] of a positive obstacle: ``h fx / n``."""
    return float(obstacle_h_m * fx_px / n_px)


def stopping_distance_m(v_mps: np.ndarray | float, a_mps2: float, t_r_s: float, margin_m: float) -> np.ndarray:
    """``v^2 / (2 a) + v T_r + B`` [m] for speed v [m/s]."""
    v = np.asarray(v_mps, dtype=np.float64)
    return v * v / (2.0 * a_mps2) + v * t_r_s + margin_m


def max_safe_speed_mps(r_m: np.ndarray | float, a_mps2: float, t_r_s: float, margin_m: float) -> np.ndarray:
    """Largest v [m/s] with ``d_stop(v) <= R`` (closed form, 0 where R <= B). Vectorised."""
    r = np.asarray(r_m, dtype=np.float64)
    usable = np.maximum(r - margin_m, 0.0)
    return a_mps2 * (-t_r_s + np.sqrt(t_r_s * t_r_s + 2.0 * usable / a_mps2))


def safe_speed_envelope(h_m: np.ndarray, w_m: np.ndarray, p: TheoryParams) -> np.ndarray:
    """Uncapped safe speed [m/s] on the grid ``(len(h_m), len(w_m))`` (rows = mast height)."""
    hh, ww = np.meshgrid(np.asarray(h_m, np.float64), np.asarray(w_m, np.float64), indexing="ij")
    r_det = ditch_detection_range_m(ww, hh, p.fx_px, p.n_px)
    return max_safe_speed_mps(r_det, p.brake_decel_mps2, p.t_reaction_s, p.margin_m)


def min_mast_height_m(w_m: float, v_mps: float, p: TheoryParams) -> float:
    """Lowest mast height [m] at which a ditch of width w is detected in time to stop from v.

    Inverts ``R_det(w, H) = d_stop(v)``: ``H = R (R + w) n / (w fx)``."""
    r = float(stopping_distance_m(v_mps, p.brake_decel_mps2, p.t_reaction_s, p.margin_m))
    return r * (r + w_m) * p.n_px / (w_m * p.fx_px)


def stereo_depth_sigma_m(z_m: np.ndarray | float, fx_px: float, baseline_m: float, sigma_d_px: float) -> np.ndarray:
    """First-order stereo range noise ``Z^2 sigma_d / (fx b)`` [m]."""
    z = np.asarray(z_m, dtype=np.float64)
    return z * z * sigma_d_px / (fx_px * baseline_m)


# =========================================================================== figures
def _hex(state: CellState) -> str:
    return ps.rgb_to_hex(CELL_COLORS[state])


# R_det label placement per ditch series (points dx, dy, ha): below-left / below-right / above-right
# keeps each label clear of its neighbours' curves on the log-scale figure.
_RDET_LABEL_OFFSETS = ((-3, -12, "right"), (3, -12, "left"), (3, 5, "left"))
# Ditch series share the missing-ground magenta hue, darker = wider (ordinal magnitude).
_DITCH_SERIES_HEX = {0.3: _hex(CellState.DITCH_CANDIDATE), 0.5: "#9A1E7C", 1.0: "#5F1150"}


def fig_ditch_detectability(p: TheoryParams) -> Any:
    """Figure (a): visible angle vs range for ditches and a rock, with the detection threshold."""
    from matplotlib.ticker import FuncFormatter

    fig, ax = ps.new_figure("card")
    r = np.linspace(0.4, 24.0, 600)
    theta_min_mrad = p.theta_min_rad * 1e3
    mono, annot = ps.mono_family(), ps.FONT_PT["annot"]
    ax.axvspan(p.max_range_m, r[-1], color=ps.TOKENS["surface"], zorder=0, lw=0)
    ax.text(p.max_range_m + 0.4, 700, f"beyond stereo\nmax range ({p.max_range_m:.0f} m)", fontsize=annot,
            color=ps.TOKENS["text_muted"], va="top")
    ax.axhline(theta_min_mrad, color=ps.TOKENS["text"], lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text(p.max_range_m + 0.4, theta_min_mrad * 0.58,
            f"detection threshold: {p.n_px:.0f} px x IFOV = {theta_min_mrad:.1f} mrad", fontsize=annot,
            color=ps.TOKENS["text"], va="center")

    # R_det labels alternate below / above the threshold line so neighbours never collide.
    for i, w in enumerate(DITCH_WIDTHS_M):
        color = _DITCH_SERIES_HEX.get(w, _hex(CellState.DITCH_CANDIDATE))
        lw = 2.2 if math.isclose(w, p.design_ditch_w_m) else 1.6
        ax.plot(r, ditch_angle_rad(r, w, p.cam_height_m) * 1e3, color=color, lw=lw, label=f"ditch w = {w:.1f} m", zorder=4)
        r_det = float(ditch_detection_range_m(w, p.cam_height_m, p.fx_px, p.n_px))
        ax.plot([r_det], [theta_min_mrad], "o", ms=5, color=color, mec=ps.TOKENS["bg"], mew=1.2, zorder=5)
        dx, dy, ha = _RDET_LABEL_OFFSETS[i % len(_RDET_LABEL_OFFSETS)]
        ax.annotate(f"{r_det:.1f} m", (r_det, theta_min_mrad), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    fontsize=annot, color=ps.TOKENS["text"], family=mono, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.1", fc=ps.TOKENS["bg"], ec="none"))
    rock_hex = _hex(CellState.POSITIVE)
    ax.plot(r, positive_angle_rad(r, ROCK_HEIGHT_M) * 1e3, color=rock_hex, lw=1.6, ls=(0, (6, 2)),
            label=f"rock h = {ROCK_HEIGHT_M:.1f} m", zorder=4)
    r_rock = positive_detection_range_m(ROCK_HEIGHT_M, p.fx_px, p.n_px)
    ax.plot([r_rock], [theta_min_mrad], "o", ms=5, color=rock_hex, mec=ps.TOKENS["bg"], mew=1.2, zorder=5)
    ax.annotate(f"{r_rock:.1f} m", (r_rock, theta_min_mrad), xytext=(3, 5), textcoords="offset points", ha="left",
                fontsize=annot, color=ps.TOKENS["text"], family=mono, zorder=6)

    # Stopping distances: labels at staggered heights to the right of each line.
    for v, y_lab in zip(MARK_SPEEDS_MPS, (3.2, 1.2)):
        d = float(stopping_distance_m(v, p.brake_decel_mps2, p.t_reaction_s, p.margin_m))
        ax.axvline(d, color=ps.TOKENS["accent"], lw=1.0, zorder=3)
        ax.text(d + 0.2, y_lab, f"stop from {v:.0f} m/s\n= {d:.1f} m", fontsize=annot, color=ps.TOKENS["accent"],
                va="center", bbox=dict(boxstyle="round,pad=0.15", fc=ps.TOKENS["bg"], ec="none", alpha=0.9), zorder=6)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_ylim(0.5, 1000)
    ax.set_xlim(0, r[-1])
    ax.set_xlabel("Range to near lip / obstacle, R (m)")
    ax.set_ylabel("Visible angle (mrad, log scale)")
    ax.set_title("A ditch fades as 1/R², a rock only as 1/R")
    ax.legend(loc="upper right", handlelength=2.4)
    ps.style_axes(ax)
    ps.add_honesty_chip(fig, HONESTY_LABEL, HONESTY_DETAIL)
    return fig


def fig_safe_speed_envelope(p: TheoryParams) -> Any:
    """Figure (b): max safe speed [m/s] vs camera mast height and design ditch width."""
    from matplotlib.colors import Normalize

    fig, ax = ps.new_figure("card_tall")
    h = np.linspace(*ENVELOPE_H_RANGE_M, ENVELOPE_N)
    w = np.linspace(*ENVELOPE_W_RANGE_M, ENVELOPE_N)
    v = safe_speed_envelope(h, w, p)
    annot = ps.FONT_PT["annot"]
    vmin, vmax = BEL_RSP_MAX_MPS, float(np.ceil(v.max() * 2.0) / 2.0)
    mesh = ax.pcolormesh(w, h, v, cmap=ps.sequential_cmap(), norm=Normalize(vmin, vmax), shading="gouraud", rasterized=True)

    # Iso-speed contours, labelled along the top band (H ~ 1.5 m) where nothing else is drawn.
    levels = [float(lv) for lv in np.arange(2.5, vmax, 0.5)]
    cs = ax.contour(w, h, v, levels=levels, colors=ps.TOKENS["bg"], linewidths=0.8)
    row = int(np.argmin(np.abs(h - 1.5)))
    manual = [(float(np.interp(lv, v[row], w)), 1.5) for lv in levels if v[row, 0] <= lv <= v[row, -1]]
    ax.clabel(cs, fmt=lambda x: f"{x:.1f} m/s", fontsize=annot, inline_spacing=4, manual=manual)

    # Platform cap: region where the envelope is below it is washed out and labelled.
    ax.contourf(w, h, v, levels=[0.0, p.v_platform_mps], colors=[ps.TOKENS["bg"]], alpha=0.55)
    ax.contour(w, h, v, levels=[p.v_platform_mps], colors=ps.TOKENS["text"], linewidths=1.6)
    w_cap = float(np.interp(p.v_platform_mps, v[0], w))  # crossing on the lowest mast row
    ax.annotate(f"below {p.v_platform_mps:.1f} m/s platform cap", (w_cap - 0.02, h[0] + 0.03), xytext=(18, 14),
                textcoords="offset points", fontsize=annot, color=ps.TOKENS["text"],
                bbox=dict(boxstyle="round,pad=0.25", fc=ps.TOKENS["bg"], ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color=ps.TOKENS["text"], lw=0.8))

    # Our platform: mast height from defaults, design ditch width from the governor.
    v_ours = float(safe_speed_envelope(np.array([p.cam_height_m]), np.array([p.design_ditch_w_m]), p)[0, 0])
    ax.axhline(p.cam_height_m, color=ps.TOKENS["text"], lw=0.8, ls=(0, (2, 2)))
    ax.plot([p.design_ditch_w_m], [p.cam_height_m], "o", ms=8, color=ps.TOKENS["bg"], mec=ps.TOKENS["text"], mew=1.8, zorder=6)
    ax.annotate(
        f"METAGROSS: H = {p.cam_height_m:.1f} m mast, w = {p.design_ditch_w_m:.1f} m design ditch\n"
        f"envelope {v_ours:.2f} m/s, capped to {min(v_ours, p.v_platform_mps):.1f} m/s by the platform",
        (p.design_ditch_w_m, p.cam_height_m), xytext=(16, -46), textcoords="offset points", fontsize=annot,
        color=ps.TOKENS["text"], bbox=dict(boxstyle="round,pad=0.4", fc=ps.TOKENS["bg"], ec=ps.TOKENS["border"], lw=0.8),
        arrowprops=dict(arrowstyle="-", color=ps.TOKENS["text"], lw=0.8), zorder=7,
    )
    ax.set_xlabel("Design ditch width, w (m)")
    ax.set_ylabel("Camera mast height, H (m)")
    ax.set_title("Safe-speed envelope: only as fast as it can see")
    ax.grid(False)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.05, aspect=30)
    cbar.outline.set_visible(False)
    ticks = sorted({BEL_RSP_MAX_MPS, p.v_platform_mps, *np.arange(3.0, vmax + 1e-9, 1.0)})
    cbar.set_ticks(ticks)
    names = {BEL_RSP_MAX_MPS: "BEL RSP max (3.6 km/h)", p.v_platform_mps: "platform cap"}
    cbar.set_ticklabels([f"{t:.1f}  {names[t]}" if t in names else f"{t:.1f}" for t in ticks])
    cbar.ax.tick_params(labelsize=annot, colors=ps.TOKENS["border"], labelcolor=ps.TOKENS["text_secondary"])
    for val in names:
        cbar.ax.axhline(val, color=ps.TOKENS["text"], lw=1.4)
    cbar.set_label("Max safe speed (m/s)", color=ps.TOKENS["text_secondary"])
    ps.add_honesty_chip(fig, HONESTY_LABEL, HONESTY_DETAIL)
    ps.add_footnote(fig, f"v²/(2a) + v·T_r + B ≤ R_det(w, H);  a = {p.brake_decel_mps2:.1f} m/s², T_r = {p.t_reaction_s:.1f} s, "
                    f"B = {p.margin_m:.1f} m, {p.n_px:.0f} px on target, fx = {p.fx_px:.0f} px")
    return fig


def fig_stereo_depth_error(p: TheoryParams) -> Any:
    """Figure (c): stereo range noise sigma_Z vs range for our baseline (and an OAK-D Lite class one)."""
    fig, ax = ps.new_figure("card")
    z = np.linspace(0.5, p.max_range_m, 300)
    lo, hi = SIGMA_D_BAND_PX
    blue = ps.TOKENS["vo_blue"]
    ax.fill_between(z, stereo_depth_sigma_m(z, p.fx_px, p.baseline_m, lo), stereo_depth_sigma_m(z, p.fx_px, p.baseline_m, hi),
                    color=blue, alpha=0.12, lw=0, label=f"σ_d = {lo:g}–{hi:g} px")
    ax.plot(z, stereo_depth_sigma_m(z, p.fx_px, p.baseline_m, p.sigma_d_px), color=blue, lw=2.0,
            label=f"b = {p.baseline_m * 100:.0f} cm (ours), σ_d = {p.sigma_d_px:g} px")
    ax.plot(z, stereo_depth_sigma_m(z, p.fx_px, BASELINE_OAKD_LITE_M, p.sigma_d_px), color=ps.TOKENS["text_secondary"],
            lw=1.4, ls=(0, (5, 2)), label=f"b = {BASELINE_OAKD_LITE_M * 100:.1f} cm (OAK-D Lite class)")
    r_det = float(ditch_detection_range_m(p.design_ditch_w_m, p.cam_height_m, p.fx_px, p.n_px))
    s_det = float(stereo_depth_sigma_m(r_det, p.fx_px, p.baseline_m, p.sigma_d_px))
    ax.plot([r_det], [s_det], "o", ms=6, color=blue, mec=ps.TOKENS["bg"], mew=1.2, zorder=5)
    ax.annotate(f"design ditch first seen at {r_det:.1f} m\nσ_Z = {s_det * 100:.0f} cm", (r_det, s_det), xytext=(-8, 26),
                textcoords="offset points", ha="right", fontsize=ps.FONT_PT["annot"], color=ps.TOKENS["text"],
                arrowprops=dict(arrowstyle="-", color=ps.TOKENS["text_secondary"], lw=0.8))
    s_max = float(stereo_depth_sigma_m(p.max_range_m, p.fx_px, p.baseline_m, p.sigma_d_px))
    ax.annotate(f"{s_max:.2f} m at {p.max_range_m:.0f} m", (p.max_range_m, s_max), xytext=(p.max_range_m - 0.15, 0.55 * s_max),
                textcoords="data", ha="right", va="center", fontsize=ps.FONT_PT["annot"], color=blue, family=ps.mono_family(),
                arrowprops=dict(arrowstyle="-", color=blue, lw=0.8, shrinkA=2, shrinkB=3))
    ax.set_xlim(0, p.max_range_m)
    ax.set_ylim(0, None)
    ax.set_xlabel("Range, Z (m)")
    ax.set_ylabel("Range noise σ_Z (m)")
    ax.set_title("Stereo range noise grows as Z²")
    ax.legend(loc="upper left")
    ps.style_axes(ax)
    ps.add_honesty_chip(fig, HONESTY_LABEL, HONESTY_DETAIL)
    ps.add_footnote(fig, f"σ_Z = Z² σ_d / (fx b),  fx = {p.fx_px:.0f} px, b = stereo baseline, σ_d = disparity noise")
    return fig


# --------------------------------------------------------------------------- slot-sized versions
# Re-composed (not rescaled) for a fixed box on the 1920x1080 slide, written to deck_assets/slots/ by
# deck_assets/diagrams/build_diagrams.py. At 2x / 300 dpi one point is 300 / 72 / 2 = 2.08 slide px, so
# 7 pt = 14.6 px is the floor for any text and 9 pt = 18.8 px is used for titles.
SLOT_PT = {"title": 9.0, "label": 7.5, "tick": 7.0, "annot": 7.0}
SLOT_SIZES = {"envelope": (732, 348), "ditch": (588, 220)}  # [slide px]
# Iso-speed contour labels of the slot envelope sit on this mast-height row [m] (clear of the annotations).
_SLOT_CLABEL_H_M = 1.42
# Ditch R_det labels of the slot chart, per ditch series: (label row as an axes fraction, alignment to its
# dotted leader). Rows are staggered and the outer labels flag outwards, so neighbours never touch.
_SLOT_RDET_LABELS = ((0.74, "right"), (0.90, "center"), (0.74, "left"))
_RELPOS = {"right": (1.0, 0.0), "center": (0.5, 0.0), "left": (0.0, 0.0)}  # leader attaches at the label's bottom


def _slot_figure(key: str) -> tuple[Any, Any]:
    """Figure + axes at slot size ``SLOT_SIZES[key]``. The size gets a quarter-pixel guard so the saved PNG is
    exactly EXPORT_SCALE x the slot (``w * 2 / 300`` in can land a hair under the integer and be truncated)."""
    w, h = SLOT_SIZES[key]
    fig, ax = ps.new_figure((w, h))
    fig.set_size_inches((w * ps.EXPORT_SCALE + 0.25) / ps.DPI, (h * ps.EXPORT_SCALE + 0.25) / ps.DPI)
    return fig, ax


def _slot_chip(fig: Any, key: str, inset_px: tuple[float, float] = (10.0, 9.0)) -> None:
    """Honesty chip in the top-right corner, inset by a fixed number of slide px so its outline is never clipped."""
    w, h = SLOT_SIZES[key]
    ps.add_honesty_chip(fig, HONESTY_LABEL, HONESTY_DETAIL, loc=(1.0 - inset_px[0] / w, 1.0 - inset_px[1] / h))


def _slot_text_style(ax: Any) -> None:
    """Slot typography on one axes: tick and axis-label sizes from :data:`SLOT_PT`."""
    ax.tick_params(labelsize=SLOT_PT["tick"])
    ax.xaxis.label.set_size(SLOT_PT["label"])
    ax.yaxis.label.set_size(SLOT_PT["label"])


def fig_safe_speed_envelope_slot(p: TheoryParams) -> Any:
    """Landscape slot version of figure (b) (732x348 slide px): max safe speed vs mast height and ditch width."""
    from matplotlib.colors import Normalize

    fig, ax = _slot_figure("envelope")
    h = np.linspace(*ENVELOPE_H_RANGE_M, ENVELOPE_N)
    w = np.linspace(*ENVELOPE_W_RANGE_M, ENVELOPE_N)
    v = safe_speed_envelope(h, w, p)
    annot, ink, bg = SLOT_PT["annot"], ps.TOKENS["text"], ps.TOKENS["bg"]
    vmin, vmax = BEL_RSP_MAX_MPS, float(np.ceil(v.max() * 2.0) / 2.0)
    mesh = ax.pcolormesh(w, h, v, cmap=ps.sequential_cmap(), norm=Normalize(vmin, vmax), shading="gouraud", rasterized=True)

    levels = [float(lv) for lv in np.arange(2.5, vmax, 0.5)]
    cs = ax.contour(w, h, v, levels=levels, colors=bg, linewidths=0.8)
    row = int(np.argmin(np.abs(h - _SLOT_CLABEL_H_M)))
    manual = [(float(np.interp(lv, v[row], w)), float(h[row])) for lv in levels if v[row, 0] <= lv <= v[row, -1]]
    ax.clabel(cs, fmt=lambda x: f"{x:.1f}", fontsize=annot, inline_spacing=3, manual=manual)

    # Platform cap: region below it washed out, boundary drawn and labelled.
    ax.contourf(w, h, v, levels=[0.0, p.v_platform_mps], colors=[bg], alpha=0.55)
    ax.contour(w, h, v, levels=[p.v_platform_mps], colors=ink, linewidths=1.4)
    h_cap = float(h[4])
    w_cap = float(np.interp(p.v_platform_mps, v[4], w))
    ax.annotate(f"{p.v_platform_mps:.1f} m/s platform cap", (w_cap, h_cap), xytext=(34, 2), textcoords="offset points",
                va="center", fontsize=annot, color=ink,
                bbox=dict(boxstyle="round,pad=0.2", fc=bg, ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color=ink, lw=0.8, shrinkA=0, shrinkB=2))

    # Our platform: mast height from defaults, design ditch width from the governor.
    v_ours = float(safe_speed_envelope(np.array([p.cam_height_m]), np.array([p.design_ditch_w_m]), p)[0, 0])
    ax.axhline(p.cam_height_m, color=ink, lw=0.8, ls=(0, (2, 2)))
    ax.plot([p.design_ditch_w_m], [p.cam_height_m], "o", ms=7, color=bg, mec=ink, mew=1.6, zorder=6)
    ax.annotate(f"METAGROSS: H = {p.cam_height_m:.1f} m, w = {p.design_ditch_w_m:.1f} m\n"
                f"{v_ours:.2f} m/s, capped {min(v_ours, p.v_platform_mps):.1f} by platform",
                (p.design_ditch_w_m, p.cam_height_m), xytext=(26, 10), textcoords="offset points", va="bottom",
                fontsize=annot, color=ink, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.35", fc=bg, ec=ps.TOKENS["border"], lw=0.8),
                arrowprops=dict(arrowstyle="-", color=ink, lw=0.8, shrinkA=0, shrinkB=4), zorder=7)

    ax.set_xlabel("Design ditch width, w (m)")
    ax.set_ylabel("Mast height, H (m)")
    ax.set_title("Max safe speed (m/s)", fontsize=SLOT_PT["title"])
    ax.grid(False)
    _slot_text_style(ax)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.05, aspect=16)
    cbar.outline.set_visible(False)
    ticks = sorted({BEL_RSP_MAX_MPS, p.v_platform_mps, *np.arange(3.0, vmax + 1e-9, 1.0)})
    cbar.set_ticks(ticks)
    names = {BEL_RSP_MAX_MPS: "BEL RSP max", p.v_platform_mps: "platform cap"}
    cbar.set_ticklabels([f"{t:.1f}  {names[t]}" if t in names else f"{t:.1f}" for t in ticks])
    cbar.ax.tick_params(labelsize=SLOT_PT["tick"], colors=ps.TOKENS["border"], labelcolor=ps.TOKENS["text_secondary"])
    for val in names:
        cbar.ax.axhline(val, color=ink, lw=1.4)
    _slot_chip(fig, "envelope")
    return fig


def fig_ditch_detectability_slot(p: TheoryParams) -> Any:
    """Compact slot version of figure (a) (588x220 slide px): ditches vs a rock, log angle, R_det markers."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, ax = _slot_figure("ditch")
    r = np.linspace(0.4, 24.0, 600)
    theta_min_mrad = p.theta_min_rad * 1e3
    annot, ink, bg, mono = SLOT_PT["annot"], ps.TOKENS["text"], ps.TOKENS["bg"], ps.mono_family()
    ax.axhline(theta_min_mrad, color=ink, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text(20.8, theta_min_mrad * 0.82, f"{p.n_px:.0f} px threshold", fontsize=annot, color=ink, ha="right", va="top")

    for i, w in enumerate(DITCH_WIDTHS_M):
        color = _DITCH_SERIES_HEX.get(w, _hex(CellState.DITCH_CANDIDATE))
        lw = 2.0 if math.isclose(w, p.design_ditch_w_m) else 1.5
        ax.plot(r, ditch_angle_rad(r, w, p.cam_height_m) * 1e3, color=color, lw=lw, label=f"ditch {w:.1f} m", zorder=4)
        r_det = float(ditch_detection_range_m(w, p.cam_height_m, p.fx_px, p.n_px))
        ax.plot([r_det], [theta_min_mrad], "o", ms=4.5, color=color, mec=bg, mew=1.0, zorder=5)
        row, ha = _SLOT_RDET_LABELS[i % len(_SLOT_RDET_LABELS)]
        ax.annotate(f"{r_det:.1f} m", (r_det, theta_min_mrad), xytext=(r_det, row), textcoords=ax.get_xaxis_transform(),
                    ha=ha, va="bottom", fontsize=annot, fontweight="bold", color=color, family=mono, zorder=6,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8, ls=(0, (1, 2)), relpos=_RELPOS[ha], shrinkA=0,
                                    shrinkB=3))
    rock_hex = _hex(CellState.POSITIVE)
    ax.plot(r, positive_angle_rad(r, ROCK_HEIGHT_M) * 1e3, color=rock_hex, lw=1.5, ls=(0, (5, 2)),
            label=f"rock {ROCK_HEIGHT_M:.1f} m", zorder=4)
    r_rock = positive_detection_range_m(ROCK_HEIGHT_M, p.fx_px, p.n_px)
    ax.plot([r_rock], [theta_min_mrad], "o", ms=4.5, color=rock_hex, mec=bg, mew=1.0, zorder=5)
    ax.annotate(f"{r_rock:.0f} m", (r_rock, theta_min_mrad), xytext=(0, 7), textcoords="offset points", ha="center",
                va="bottom", fontsize=annot, fontweight="bold", color=rock_hex, family=mono, zorder=6)

    ax.set_yscale("log")
    ax.set_ylim(1.0, 2000.0)
    ax.set_xlim(0, r[-1])
    ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel("Range R (m)")
    ax.set_ylabel("Angle (mrad)")
    ax.set_title("Ditch fades as 1/R², rock as 1/R", fontsize=SLOT_PT["title"])
    ax.legend(loc="upper right", ncol=2, fontsize=annot, handlelength=1.6, columnspacing=1.0, borderaxespad=0.2,
              handletextpad=0.5, labelspacing=0.3)
    ps.style_axes(ax, grid="y")
    _slot_text_style(ax)
    _slot_chip(fig, "ditch")
    return fig


#: Slot figures by output name (deck_assets/slots/<name>.png), built by deck_assets/diagrams/build_diagrams.py.
SLOT_FIGURES = {"s4_envelope": fig_safe_speed_envelope_slot, "s2_ditch_theory": fig_ditch_detectability_slot}


# =========================================================================== summary
@dataclass(slots=True)
class TheorySummary:
    """Numbers written to results/theory.json (all Estimated, analytic)."""

    label: str
    detail: str
    params: dict[str, float]
    ditch_detection: list[dict[str, float]] = field(default_factory=list)
    positive_detection: list[dict[str, float]] = field(default_factory=list)
    stopping: list[dict[str, float]] = field(default_factory=list)
    envelope: dict[str, Any] = field(default_factory=dict)
    stereo: list[dict[str, float]] = field(default_factory=list)
    claims: list[dict[str, str]] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)


def compute_summary(p: TheoryParams) -> TheorySummary:
    """Evaluate every model at the deck's reference points."""
    a, tr, b = p.brake_decel_mps2, p.t_reaction_s, p.margin_m
    s = TheorySummary(
        label=HONESTY_LABEL,
        detail=HONESTY_DETAIL,
        params={**{k: float(v) for k, v in asdict(p).items()}, "ifov_mrad": p.ifov_rad * 1e3,
                "theta_min_mrad": p.theta_min_rad * 1e3},
    )
    widths = sorted({*DITCH_WIDTHS_M, p.design_ditch_w_m, 0.8})
    for w in widths:
        r = float(ditch_detection_range_m(w, p.cam_height_m, p.fx_px, p.n_px))
        s.ditch_detection.append({
            "width_m": w, "cam_height_m": p.cam_height_m, "r_det_m": round(r, 3),
            "exact_angle_at_r_det_mrad": round(float(ditch_angle_exact_rad(r, w, p.cam_height_m)) * 1e3, 3),
            "v_safe_mps": round(float(max_safe_speed_mps(r, a, tr, b)), 3),
        })
    s.positive_detection.append({"height_m": ROCK_HEIGHT_M,
                                 "r_det_m": round(positive_detection_range_m(ROCK_HEIGHT_M, p.fx_px, p.n_px), 3)})
    for v in sorted({*MARK_SPEEDS_MPS, BEL_RSP_MAX_MPS, p.v_platform_mps}):
        s.stopping.append({"v_mps": v, "d_stop_m": round(float(stopping_distance_m(v, a, tr, b)), 3),
                           "min_mast_h_for_design_ditch_m": round(min_mast_height_m(p.design_ditch_w_m, v, p), 3)})
    h = np.linspace(*ENVELOPE_H_RANGE_M, ENVELOPE_N)
    w = np.linspace(*ENVELOPE_W_RANGE_M, ENVELOPE_N)
    env = safe_speed_envelope(h, w, p)
    v_ours = float(safe_speed_envelope(np.array([p.cam_height_m]), np.array([p.design_ditch_w_m]), p)[0, 0])
    s.envelope = {
        "h_range_m": list(ENVELOPE_H_RANGE_M), "w_range_m": list(ENVELOPE_W_RANGE_M), "n": ENVELOPE_N,
        "v_min_mps": round(float(env.min()), 3), "v_max_mps": round(float(env.max()), 3),
        "v_ours_mps": round(v_ours, 3), "v_ours_capped_mps": round(min(v_ours, p.v_platform_mps), 3),
        "frac_grid_below_platform_cap": round(float(np.mean(env < p.v_platform_mps)), 4),
        "frac_grid_below_bel_rsp": round(float(np.mean(env < BEL_RSP_MAX_MPS)), 4),
        "bel_rsp_max_mps": BEL_RSP_MAX_MPS,
    }
    for z in (2.0, 4.0, 6.0, 8.0, 10.0, p.max_range_m):
        s.stereo.append({"z_m": z, "sigma_z_m": round(float(stereo_depth_sigma_m(z, p.fx_px, p.baseline_m, p.sigma_d_px)), 4),
                         "sigma_z_oakd_lite_m": round(float(stereo_depth_sigma_m(z, p.fx_px, BASELINE_OAKD_LITE_M, p.sigma_d_px)), 4)})
    r_design = s.ditch_detection[[d["width_m"] for d in s.ditch_detection].index(p.design_ditch_w_m)]["r_det_m"]
    s.claims = [
        {"claim": f"A {p.design_ditch_w_m:.1f} m ditch is first resolvable at {r_design:.1f} m from a {p.cam_height_m:.1f} m mast",
         "value": f"{r_design:.2f} m", "label": HONESTY_LABEL, "source": "results/theory.json#ditch_detection"},
        {"claim": f"A {ROCK_HEIGHT_M:.1f} m rock is resolvable at {s.positive_detection[0]['r_det_m']:.1f} m (1/R vs 1/R^2)",
         "value": f"{s.positive_detection[0]['r_det_m']:.1f} m", "label": HONESTY_LABEL,
         "source": "results/theory.json#positive_detection"},
        {"claim": "Stopping distance from 2 m/s (a=1.5 m/s^2, T_r=0.6 s, B=0.5 m)",
         "value": f"{s.stopping[1]['d_stop_m']:.2f} m", "label": HONESTY_LABEL, "source": "results/theory.json#stopping"},
        {"claim": "Safe-speed envelope at our mast/design ditch (before 2 m/s platform cap)",
         "value": f"{v_ours:.2f} m/s", "label": HONESTY_LABEL, "source": "results/theory.json#envelope"},
    ]
    return s


def build_all(out_dir: Optional[Path] = None, results_json: Optional[Path] = RESULTS_JSON,
              params: TheoryParams = TheoryParams()) -> TheorySummary:
    """Render the three theory figures and write the summary JSON. Returns the summary."""
    summary = compute_summary(params)
    with ps.deck_style():
        for name, maker in (("ditch_detectability", fig_ditch_detectability),
                            ("safe_speed_envelope", fig_safe_speed_envelope),
                            ("stereo_depth_error", fig_stereo_depth_error)):
            paths = ps.save_fig(maker(params), f"{THEORY_DIR}/{name}", out_dir=out_dir)
            summary.figures.extend(str(pth.relative_to(REPO_ROOT)) if pth.is_relative_to(REPO_ROOT) else str(pth)
                                   for pth in paths)
    if results_json is not None:
        results_json.parent.mkdir(parents=True, exist_ok=True)
        results_json.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        LOG.info("wrote %s", results_json)
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=None, help="figure root (default deck_assets/)")
    ap.add_argument("--json", type=Path, default=RESULTS_JSON, help="summary JSON path")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    s = build_all(args.out_dir, args.json)
    for d in s.ditch_detection:
        LOG.info("ditch w=%.1f m: R_det=%.2f m, v_safe=%.2f m/s", d["width_m"], d["r_det_m"], d["v_safe_mps"])
    LOG.info("envelope at ours: %.2f m/s", s.envelope["v_ours_mps"])


if __name__ == "__main__":
    main()
