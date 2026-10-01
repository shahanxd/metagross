"""Deck diagrams for METAGROSS (SIH26126) as hand-laid SVG, rendered to 2x PNG.

Outputs (``deck_assets/diagrams/``)::

    how_it_works.svg|png        slide-2 hero: 6 stages + LINK branch + health band
    system_architecture.svg|png world | firewall | autonomy | operator | evaluator
    model_pipeline.svg|png      segmentation model, integrity monitor, MPPI loop
    missing_ground.svg|png      side-view concept: why a ditch is missing ground
    legend_cell_states.svg|png  colour key strip (all states) + legend_cell_states_core6
    chips/chip_<label>.svg|png  honesty chips (transparent background)

Slot-sized versions (``deck_assets/slots/``) are re-composed, not rescaled, for the exact box each figure
occupies on a 1920x1080 slide: every text element is >= ``MIN_SLOT_TEXT_PX`` (14) slide px and main labels
>= 18 px. The canvas is the slot size in slide px; the PNG is 2x::

    s2_how_it_works    1792x356  six stages in one row + link strip + health-mode chips
    s3_missing_ground  1180x384  side-view schematic + measured-range-per-row mini plot
    s3_models          1180x286  terrain model | integrity monitor | MPPI, three compact blocks
    s4_envelope         732x348  landscape safe-speed envelope   (metagross.eval.theory)
    s2_ditch_theory     588x220  compact ditch detectability chart (metagross.eval.theory)

Colours come from ``metagross.eval.plot_style`` (tokens) and
``metagross.contracts.messages.CELL_COLORS`` (cell-state key). Every number on a
diagram is read from ``metagross.config.defaults`` or from the module that owns it
(MPPI, supervisor, integrity monitor, model card) — see the ``FACTS`` dict.

Usage (the PNG step launches headless Chromium, so run it from PowerShell)::

    python deck_assets/diagrams/build_diagrams.py              # SVG + PNG, full-size and slot versions
    python deck_assets/diagrams/build_diagrams.py --svg-only   # SVG only (no browser)
    python deck_assets/diagrams/build_diagrams.py --slots-only # only deck_assets/slots/
"""

from __future__ import annotations

import argparse
import functools
import logging
import re
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metagross.config import defaults as D  # noqa: E402
from metagross.contracts.messages import CellState  # noqa: E402
from metagross.eval import plot_style as ps  # noqa: E402

LOG = logging.getLogger("build_diagrams")
OUT_DIR = Path(__file__).resolve().parent

T = ps.TOKENS
# Telemetry goes out on the control tick that follows each 1/TELEMETRY_HZ timer expiry, so the real
# period is a whole number of ticks: every 3rd 5 Hz tick = 0.6 s (median 0.6 s in the EVAL logs,
# results/deck_extras.json#closed_loop_eval_tier0_FULL_telemetry_period_s).
TELEMETRY_PERIOD_S = math.ceil(D.CAMERA_HZ_BATCH / D.TELEMETRY_HZ) / D.CAMERA_HZ_BATCH
C = {s: ps.CELL_HEX[s] for s in CellState}
MODE = ps.MODE_COLORS
FONT = "Inter, 'Segoe UI', Arial, sans-serif"
MONO = "'JetBrains Mono', 'Cascadia Mono', Consolas, monospace"

# MPPI horizon T [steps] and step dt [s] (metagross/autonomy/planning/mppi.py MppiParams; checked by the tests).
_MPPI_HORIZON, _MPPI_DT = 30, 0.1

# Facts shown on diagrams, each traceable to its owner module (kept here, not re-derived).
FACTS: dict[str, str] = {
    "cam": f"{D.IMG_W}×{D.IMG_H}",
    "cam_hz": f"{D.CAMERA_HZ_BATCH:g}–{D.CAMERA_HZ_DEMO:g} Hz",
    "baseline": f"{D.BASELINE_M * 100:.0f} cm",
    "telemetry_hz": f"{D.TELEMETRY_HZ:g} Hz",
    "link_kbps": f"{D.LINK_KBPS:g} kbps",
    "link_loss": f"{D.LINK_LOSS * 100:.0f} %",
    "link_latency": f"{D.LINK_LATENCY_S:g} s",
    "physics_hz": f"{D.PHYSICS_HZ:g} Hz",
    "gt_hz": f"{D.GT_LOG_HZ:g} Hz",
    "map_res": f"{D.MAP_RES_M:g} m",
    "map_size": f"{D.MAP_SIZE_M:g} m",
    "v_max": f"{D.VEHICLE.max_speed_mps:g} m/s",
    # metagross/autonomy/planning/mppi.py MppiParams
    "mppi_k": "512",
    "mppi_t": f"{_MPPI_HORIZON} × {_MPPI_DT:g} s",
    "mppi_span": f"{_MPPI_HORIZON * _MPPI_DT:g} s",
    # metagross/autonomy/localization/health.py FEATURE_NAMES (6 VO + 6 image features)
    "integrity_features": "12",
    # metagross/autonomy/safety/supervisor.py SupervisorParams
    "q_nominal": "0.7",
    "q_caution": "0.4",
    "q_degraded": "0.2",
    # docs/MODEL_CARD.md
    "seg_params": "3.22 M",
    "seg_input": "320×416",
}


# =========================================================================== SVG builder
@dataclass
class Svg:
    """Minimal SVG writer with the deck's typography and arrow markers."""

    w: int
    h: int
    bg: Optional[str] = T["bg"]
    parts: list[str] = field(default_factory=list)
    markers: set[str] = field(default_factory=set)

    # ---------------------------------------------------------------- primitives
    def rect(self, x: float, y: float, w: float, h: float, fill: str = "none", stroke: str = "none", sw: float = 1.0,
             rx: float = 0.0, dash: Optional[str] = None, opacity: Optional[float] = None) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += f' opacity="{opacity}"' if opacity is not None else ""
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}"{extra}/>')

    def circle(self, cx: float, cy: float, r: float, fill: str = "none", stroke: str = "none", sw: float = 1.0) -> None:
        self.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x: float, y: float, s: str, size: float = 16, weight: int = 400, fill: str = T["text"],
             anchor: str = "start", mono: bool = False, ls: float = 0.0, opacity: Optional[float] = None) -> None:
        fam = MONO if mono else FONT
        extra = f' letter-spacing="{ls}"' if ls else ""
        extra += f' opacity="{opacity}"' if opacity is not None else ""
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
                          f'fill="{fill}" text-anchor="{anchor}"{extra}>{escape(s)}</text>')

    def lines(self, x: float, y: float, rows: Sequence[str], size: float = 15, lh: float = 21, **kw: object) -> float:
        """Stacked text rows; returns the baseline y after the last row."""
        for i, row in enumerate(rows):
            self.text(x, y + i * lh, row, size=size, **kw)  # type: ignore[arg-type]
        return y + len(rows) * lh

    def _marker(self, color: str) -> str:
        mid = "arr" + color.lstrip("#")
        self.markers.add(color)
        return mid

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = T["text_secondary"], sw: float = 1.6,
             dash: Optional[str] = None, arrow: bool = False) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += f' marker-end="url(#{self._marker(stroke)})"' if arrow else ""
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
                          f'stroke-width="{sw}" stroke-linecap="round"{extra}/>')

    def path(self, d: str, fill: str = "none", stroke: str = "none", sw: float = 1.5, dash: Optional[str] = None,
             arrow: bool = False, opacity: Optional[float] = None) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += f' marker-end="url(#{self._marker(stroke)})"' if arrow else ""
        extra += f' opacity="{opacity}"' if opacity is not None else ""
        self.parts.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" '
                          f'stroke-linejoin="round"{extra}/>')

    def polygon(self, pts: Iterable[tuple[float, float]], fill: str, stroke: str = "none", sw: float = 1.0,
                opacity: Optional[float] = None) -> None:
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        extra = f' opacity="{opacity}"' if opacity is not None else ""
        self.parts.append(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>')

    def raw(self, s: str) -> None:
        self.parts.append(s)

    # ---------------------------------------------------------------- composites
    def card(self, x: float, y: float, w: float, h: float, fill: str = T["bg"], stroke: str = T["border"], sw: float = 1.5,
             rx: float = 14) -> None:
        self.rect(x, y, w, h, fill=fill, stroke=stroke, sw=sw, rx=rx)

    def chip(self, x: float, y: float, s: str, color: str, size: float = 13, fill: str = T["bg"], mono: bool = False,
             weight: int = 700, h: float = 26, pad: float = 11, text_color: Optional[str] = None) -> float:
        """Outlined pill; returns its width. Width is estimated from the character count."""
        w = text_width(s, size, mono) + 2 * pad
        self.rect(x, y, w, h, fill=fill, stroke=color, sw=1.4, rx=h / 2)
        self.text(x + w / 2, y + h / 2 + size * 0.36, s, size=size, weight=weight, fill=text_color or color,
                  anchor="middle", mono=mono)
        return w

    def badge(self, cx: float, cy: float, n: str, r: float = 17, fill: str = T["navy"]) -> None:
        self.circle(cx, cy, r, fill=fill)
        self.text(cx, cy + r * 0.36, n, size=r, weight=700, fill=T["bg"], anchor="middle")

    def swatch(self, x: float, y: float, color: str, size: float = 16, rx: float = 4) -> None:
        self.rect(x, y, size, size, fill=color, rx=rx)

    def bullet_rows(self, x: float, y: float, items: Sequence[Sequence[str]], size: float = 15, lh: float = 20.5,
                    gap: float = 9, dot: str = T["accent"], fill: str = T["text"]) -> float:
        """Bulleted list; each item is one or more wrapped rows. Returns y after the list."""
        for rows in items:
            self.circle(x + 3, y - size * 0.33, 3, fill=dot)
            y = self.lines(x + 14, y, rows, size=size, lh=lh, fill=fill) + gap
        return y

    def to_string(self) -> str:
        defs = "".join(
            f'<marker id="arr{c.lstrip("#")}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0.5 L9.5,5 L0,9.5 z" fill="{c}"/></marker>'
            for c in sorted(self.markers))
        bg = f'<rect width="100%" height="100%" fill="{self.bg}"/>' if self.bg else ""
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                f"<defs>{defs}</defs>{bg}" + "".join(self.parts) + "</svg>")


# Average glyph advance of Inter / a monospace face as a fraction of the font size.
_ADV_SANS, _ADV_MONO = 0.56, 0.60


def text_width(s: str, size: float, mono: bool = False) -> float:
    """Approximate rendered width [px] of ``s`` (layout aid for pills; not exact)."""
    return len(s) * size * (_ADV_MONO if mono else _ADV_SANS)


def arrow_h(svg: Svg, x1: float, x2: float, y: float, color: str = T["text_secondary"], sw: float = 2.0) -> None:
    svg.line(x1, y, x2, y, stroke=color, sw=sw, arrow=True)


# =========================================================================== (i) how it works
def diagram_how_it_works() -> Svg:
    W, H = 1840, 740
    s = Svg(W, H)
    card_w, gap, x0, y0, card_h = 236, 34, 20, 20, 560
    xs = [x0 + i * (card_w + gap) for i in range(6)]
    stages = [
        ("SENSE", "passive, camera-primary",
         [["Stereo pair " + FACTS["cam"], f"@ {FACTS['cam_hz']}, {FACTS['baseline']} base"], ["Wheel encoders + gyro"],
          ["Nothing emitted:", "no LiDAR, no radar"]],
         ["SensorFrame", "images · ticks · gyro"]),
        ("SEE", "depth + terrain meaning",
         [["SGBM stereo disparity"], ["Ground model", "(height, slope, step)"], ["5-class terrain", "segmentation (ONNX)"]],
         ["depth · ground ·", "terrain classes"]),
        ("KNOW THE UNSEEN", "unknown is never free",
         [["Seen-ground map:", "every cell has a state"], ["Missing-ground", "detector finds ditches"]],
         ["cell states + R_cert", ""]),
        ("LOCALIZE", "GPS-denied pose",
         [["Stereo visual odometry"], ["Integrity monitor", "(visual RAIM) → q"], ["Wheel / gyro EKF"],
          ["Slip check (VO vs wheels)"]],
         ["pose · σ · health q", ""]),
        ("DECIDE", "only as fast as it can see",
         [["Cost-to-go (global)"], ["MPPI on skid-steer", f"{FACTS['mppi_k']} rollouts × 3 s"], ["Seen-distance governor"],
          ["Safety supervisor"]],
         ["path · v_cap · mode", ""]),
        ("ACT", "wheel commands",
         [["Skid-steer mixer", "(v, ω) → ω_L, ω_R"], ["Wheel-speed + accel", "limits, watchdog"]],
         ["WheelCmd (rad/s)", ""]),
    ]
    for i, (x, (name, sub, bullets, out)) in enumerate(zip(xs, stages)):
        core = name == "KNOW THE UNSEEN"
        s.card(x, y0, card_w, card_h, stroke=T["accent"] if core else T["border"], sw=2.2 if core else 1.5)
        s.badge(x + 34, y0 + 40, str(i + 1), fill=T["accent"] if core else T["navy"])
        name_size = 16 if len(name) > 10 else 19
        s.text(x + 60, y0 + 47, name, size=name_size, weight=700, fill=T["navy"], ls=0 if len(name) > 10 else 0.3)
        s.text(x + 20, y0 + 84, sub, size=14, fill=T["text_secondary"])
        s.line(x + 20, y0 + 102, x + card_w - 20, y0 + 102, stroke=T["border"], sw=1.2)
        s.bullet_rows(x + 20, y0 + 132, bullets)
        # output chip
        rows = [r for r in out if r]
        oh = 30 + 17 * len(rows)
        oy = y0 + card_h - 14 - oh
        s.rect(x + 14, oy, card_w - 28, oh, fill=T["surface"], stroke=T["border"], rx=10)
        s.text(x + 26, oy + 19, "OUTPUT", size=10.5, weight=700, fill=T["text_muted"], ls=1)
        s.lines(x + 26, oy + 38, rows, size=12.5, lh=17, mono=True, fill=T["text"])
        if i < 5:
            arrow_h(s, x + card_w + 5, x + card_w + gap - 5, y0 + 250, color=T["text_secondary"])
    _how_visuals(s, xs, y0, card_w)

    # LINK side branch: from KNOW (map) down to the operator link box.
    lx, ly, lw, lh = xs[2], 620, xs[5] + card_w - xs[2], 100
    s.line(xs[2] + card_w / 2, y0 + card_h + 2, xs[2] + card_w / 2, ly - 6, stroke=T["accent"], sw=2, dash="6 5", arrow=True)
    s.card(lx, ly, lw, lh, fill=T["bg"], stroke=T["accent"], sw=1.5)
    s.text(lx + 150, ly + 36, "LINK → OPERATOR CONSOLE", size=17, weight=700, fill=T["navy"], ls=0.3)
    s.text(lx + 150, ly + 62, f"{FACTS['telemetry_hz']} costmap packet (64×64 cells × 4 bit, zlib + CRC) instead of video",
           size=14.5, fill=T["text"])
    s.text(lx + 150, ly + 84, f"sized for a {FACTS['link_kbps']} radio · uplink GO / HOLD / RESUME / E-STOP",
           size=14.5, fill=T["text_secondary"])
    # mini console glyph with a tiny costmap
    gx, gy = lx + 22, ly + 16
    s.rect(gx, gy, 108, 68, fill=T["surface"], stroke=T["text_secondary"], sw=1.4, rx=6)
    pattern = ["UUUUUUUUU", "UUGGGDDUU", "UGGGGDDGU", "GGGGGGGGG", "GGPGGGGGG", "GGGGGGGWW"]
    code = {"U": C[CellState.UNSEEN], "G": C[CellState.GROUND], "D": C[CellState.DITCH_CANDIDATE],
            "P": C[CellState.POSITIVE], "W": C[CellState.WATER]}
    for r, row in enumerate(pattern):
        for c, ch in enumerate(row):
            s.rect(gx + 9 + c * 10, gy + 5 + r * 10, 9, 9, fill=code[ch], rx=1.5)
    s.polygon([(gx + 49, gy + 64), (gx + 54, gy + 55), (gx + 59, gy + 64)], fill=T["path"])

    # Rates table under SENSE / SEE
    s.text(xs[0] + 4, ly + 22, "RATES", size=11, weight=700, fill=T["text_muted"], ls=1.2)
    rates = [("Camera", FACTS["cam_hz"]), ("Perception → control", "every frame"), ("Telemetry downlink", FACTS["telemetry_hz"]),
             ("Sim physics", FACTS["physics_hz"])]
    for k, (a, b) in enumerate(rates):
        yy = ly + 46 + k * 21
        s.text(xs[0] + 4, yy, a, size=14, fill=T["text_secondary"])
        s.text(xs[1] + card_w - 10, yy, b, size=13.5, fill=T["text"], anchor="end", mono=True)

    _health_band(s, x=1640, y=y0, w=180, h=700)
    return s


def _how_visuals(s: Svg, xs: list[int], y0: int, cw: int) -> None:
    """Small stage-specific glyphs in the lower half of each stage card."""
    vy = y0 + 318
    # 1 SENSE: two camera glyphs + crossed-out GNSS pill
    x = xs[0]
    for k in range(2):
        cx = x + 62 + k * 76
        s.rect(cx - 30, vy, 60, 38, fill=T["bg"], stroke=T["text_secondary"], sw=1.6, rx=8)
        s.circle(cx, vy + 19, 11, fill=T["surface"], stroke=T["accent"], sw=2)
        s.circle(cx, vy + 19, 4, fill=T["accent"])
        s.text(cx, vy + 58, "L" if k == 0 else "R", size=12, weight=700, fill=T["text_muted"], anchor="middle")
    gw = s.chip(x + 34, vy + 84, "GNSS", T["text_secondary"], size=14)
    s.line(x + 32, vy + 104, x + 36 + gw, vy + 90, stroke=C[CellState.POSITIVE], sw=2)
    s.text(x + 46 + gw, vy + 102, "DENIED", size=14, weight=700, fill=C[CellState.POSITIVE], ls=0.8)

    # 2 SEE: disparity strip (near = light) + five class ticks
    x = xs[1]
    s.raw('<defs><linearGradient id="dispgrad" x1="0" y1="0" x2="0" y2="1">'
          f'<stop offset="0" stop-color="{T["navy"]}"/><stop offset="1" stop-color="#DBE6FE"/></linearGradient></defs>')
    s.rect(x + 20, vy, 90, 70, fill="url(#dispgrad)", rx=6)
    s.text(x + 65, vy + 88, "disparity", size=12, fill=T["text_muted"], anchor="middle")
    classes = [("stable path", C[CellState.GROUND]), ("unstable", C[CellState.CREST_SHADOW]),
               ("water / mud", C[CellState.WATER]), ("obstacle", C[CellState.POSITIVE]), ("sky / bg", C[CellState.UNSEEN])]
    for k, (lab, col) in enumerate(classes):
        s.swatch(x + 124, vy + k * 16 - 2, col, size=11, rx=2)
        s.text(x + 140, vy + k * 16 + 8, lab, size=11.5, fill=T["text_secondary"])

    # 3 KNOW THE UNSEEN: mini seen-ground map + key
    x = xs[2]
    grid = ["UUUUUUUU", "UUUUUUUU", "UUCCCUUU", "GGGGGUUW", "GDDDDDGW", "GGGGGGGG", "GGGPGGGG", "GGGGGGGG"]
    code = {"U": C[CellState.UNSEEN], "G": C[CellState.GROUND], "D": C[CellState.DITCH_CANDIDATE],
            "C": C[CellState.CREST_SHADOW], "P": C[CellState.POSITIVE], "W": C[CellState.WATER]}
    gx, gy, cs = x + 20, vy - 58, 11
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            s.rect(gx + c * (cs + 1.5), gy + r * (cs + 1.5), cs, cs, fill=code[ch], rx=2)
    s.polygon([(gx + 3.5 * 12.5 + 5.5, gy + 8 * 12.5 + 2), (gx + 3.5 * 12.5, gy + 8 * 12.5 + 12),
               (gx + 3.5 * 12.5 + 11, gy + 8 * 12.5 + 12)], fill=T["path"])
    key = [("seen ground", CellState.GROUND), ("unseen", CellState.UNSEEN), ("ditch candidate", CellState.DITCH_CANDIDATE),
           ("crest", CellState.CREST_SHADOW), ("lethal", CellState.POSITIVE), ("water", CellState.WATER)]
    for k, (lab, st) in enumerate(key):
        kx, ky = x + 124, gy + 4 + k * 17
        s.swatch(kx, ky, C[st], size=11, rx=2)
        s.text(kx + 15, ky + 10, lab, size=11, fill=T["text_secondary"])
    s.text(x + 20, gy + 128, "unseen cells cost as lethal-but-", size=12.5, fill=T["text"])
    s.text(x + 20, gy + 145, "unknown: never planned as free", size=12.5, fill=T["text"])

    # 4 LOCALIZE: VO track with growing uncertainty ellipses
    x = xs[3]
    pts = [(x + 30, vy + 70), (x + 70, vy + 58), (x + 110, vy + 40), (x + 150, vy + 30), (x + 195, vy + 12)]
    d = "M" + " L".join(f"{px:.0f},{py:.0f}" for px, py in pts)
    s.path(d, stroke=T["vo_blue"], sw=2.4)
    for k, (px, py) in enumerate(pts):
        s.raw(f'<ellipse cx="{px}" cy="{py}" rx="{4 + 2.2 * k:.1f}" ry="{3 + 1.4 * k:.1f}" fill="{T["vo_blue"]}" '
              f'fill-opacity="0.10" stroke="{T["vo_blue"]}" stroke-width="1"/>')
        s.circle(px, py, 3.2, fill=T["vo_blue"])
    s.text(x + 20, vy + 104, "q = 1 − p_fail gates", size=12.5, fill=T["text"])
    s.text(x + 20, vy + 121, "VO trust and speed", size=12.5, fill=T["text"])

    # 5 DECIDE: governor inequality
    x = xs[4]
    s.rect(x + 16, vy + 2, cw - 32, 62, fill=T["surface"], stroke=T["border"], rx=8)
    s.text(x + cw / 2, vy + 29, "v²/2a + v·T_r + B", size=14, mono=True, anchor="middle", fill=T["text"])
    s.text(x + cw / 2, vy + 52, "≤ R_cert", size=14, mono=True, anchor="middle", fill=T["accent"], weight=700)
    s.text(x + 20, vy + 90, "stop inside ground", size=12.5, fill=T["text"])
    s.text(x + 20, vy + 107, "already certified", size=12.5, fill=T["text"])

    # 6 ACT: top-down skid-steer glyph
    x = xs[5]
    bx, by = x + 88, vy - 4
    s.rect(bx, by, 60, 92, fill=T["surface"], stroke=T["text_secondary"], sw=1.6, rx=8)
    for dx in (-14, 60):
        for dy in (6, 58):
            s.rect(bx + dx + (2 if dx < 0 else 2), by + dy, 10, 28, fill=T["text"], rx=3)
    s.line(bx - 26, by + 72, bx - 26, by + 18, stroke=T["accent"], sw=2.2, arrow=True)
    s.line(bx + 88, by + 72, bx + 88, by + 30, stroke=T["accent"], sw=2.2, arrow=True)
    s.text(bx - 26, by + 92, "ω_L", size=13, mono=True, anchor="middle", fill=T["accent"], weight=700)
    s.text(bx + 88, by + 92, "ω_R", size=13, mono=True, anchor="middle", fill=T["accent"], weight=700)


def _health_band(s: Svg, x: float, y: float, w: float, h: float) -> None:
    s.card(x, y, w, h, fill=T["surface"], stroke=T["border"])
    s.text(x + 16, y + 34, "HEALTH MODE", size=15, weight=700, fill=T["navy"], ls=0.6)
    s.text(x + 16, y + 55, "from integrity q", size=12.5, fill=T["text_secondary"])
    s.text(x + 16, y + 72, "and progress", size=12.5, fill=T["text_secondary"])
    modes = [
        ("NOMINAL", "NOMINAL", [f"q > {FACTS['q_nominal']}", "full governed speed"]),
        ("CAUTION", "CAUTION", [f"{FACTS['q_caution']} < q ≤ {FACTS['q_nominal']}", "×0.5 speed", "costs ×1.5"]),
        ("DEGRADED", "DEGRADED", [f"{FACTS['q_degraded']} < q ≤ {FACTS['q_caution']}", "×0.25 speed", "go / look hops"]),
        ("STOP_AND_LOOK", "STOP-AND-LOOK", ["no progress 3 s", "turn ±45° to", "certify ground"]),
        ("SAFE_STOP", "SAFE-STOP", [f"q < {FACTS['q_degraded']} for 3 s", "or E-stop:", "latched zero cmd"]),
    ]
    py, ph, pg = y + 90, 110, 12
    for k, (key, label, rows) in enumerate(modes):
        yy = py + k * (ph + pg)
        s.rect(x + 12, yy, w - 24, ph, fill=T["bg"], stroke=T["border"], rx=10)
        s.rect(x + 12, yy, 7, ph, fill=MODE[key], rx=3)
        s.text(x + 30, yy + 26, label, size=13.5 if len(label) > 10 else 14.5, weight=700, fill=T["text"], ls=0.4)
        s.lines(x + 30, yy + 49, rows, size=12.5, lh=18, fill=T["text_secondary"])
        if k < len(modes) - 1:
            s.line(x + w / 2, yy + ph + 1, x + w / 2, yy + ph + pg - 1, stroke=T["text_muted"], sw=1.2)


# =========================================================================== (ii) architecture
def diagram_architecture() -> Svg:
    W, H = 1840, 900
    s = Svg(W, H)

    def panel(x: float, y: float, w: float, h: float, title: str, tag: str, tag_color: str) -> None:
        s.card(x, y, w, h, fill=T["surface"], stroke=T["border"])
        s.text(x + 22, y + 38, title, size=18, weight=700, fill=T["navy"], ls=0.4)
        s.chip(x + w - 22 - text_width(tag, 12.5) - 22, y + 18, tag, tag_color, size=12.5)

    def module(x: float, y: float, w: float, title: str, detail: str, rate: Optional[str] = None, h: float = 70) -> None:
        s.card(x, y, w, h, fill=T["bg"], stroke=T["border"], rx=10)
        s.text(x + 18, y + 29, title, size=15.5, weight=700, fill=T["text"])
        s.text(x + 18, y + 52, detail, size=13.5, fill=T["text_secondary"])
        if rate:
            rw = text_width(rate, 12.5, mono=True) + 20
            s.rect(x + w - rw - 14, y + 14, rw, 24, fill=T["surface"], stroke=T["border"], rx=12)
            s.text(x + w - rw / 2 - 14, y + 31, rate, size=12.5, mono=True, anchor="middle", fill=T["accent"])

    # World / sim
    wx, wy, ww, wh = 20, 20, 390, 700
    panel(wx, wy, ww, wh, "WORLD (SIM)", "owns ground truth", C[CellState.POSITIVE])
    world = [
        ("Scenario generator", "seeded terrain, ditches, crests, water", "6 families"),
        ("Vehicle + terrain", "skid-steer, slip, 0.2 s actuator lag", FACTS["physics_hz"]),
        ("Stereo renderer", f"Three.js, {FACTS['cam']}, glare / dust", FACTS["cam_hz"]),
        ("Referee", "collisions, ditch entry, success", None),
        ("Tier-0 depth sensor", "fast synthetic disparity for batches", None),
        ("GT logger", "writes gt/; autonomy can't open it", FACTS["gt_hz"]),
    ]
    for k, (a, b, r) in enumerate(world):
        module(wx + 18, wy + 70 + k * 100, ww - 36, a, b, r, h=78)
    # (GT route drawn later along the bottom)

    # Firewall boundary around the autonomy process
    fx, fy, fw, fh = 510, 12, 680, 810
    s.rect(fx, fy, fw, fh, fill="none", stroke=T["navy"], sw=2, rx=20, dash="10 7")
    s.rect(fx + 24, fy - 12, 262, 24, fill=T["bg"])
    s.text(fx + 34, fy + 6, "GROUND-TRUTH FIREWALL", size=14, weight=700, fill=T["navy"], ls=1)

    ax, ay, aw, ah = 534, 40, 632, 560
    panel(ax, ay, aw, ah, "AUTONOMY PROCESS", "onboard stack", T["accent"])
    mods = [
        ("Perception", "SGBM depth · ground · missing ground · terrain seg", "every frame"),
        ("Localizer", "stereo VO · integrity q · wheel/gyro EKF · slip", "every frame"),
        ("Seen-ground map", f"rolling odometry-frame map, {FACTS['map_res']} cells, {FACTS['map_size']}", "every frame"),
        ("Planner", f"cost-to-go + MPPI ({FACTS['mppi_k']} × {FACTS['mppi_t']})", "every frame"),
        ("Governor + supervisor", "v_cap from R_cert · health modes · watchdog", "every frame"),
        ("Link codec", "pose, mode, v_cap, 64×64 costmap", FACTS["telemetry_hz"]),
    ]
    for k, (a, b, r) in enumerate(mods):
        module(ax + 18, ay + 66 + k * 80, aw - 36, a, b, r, h=68)

    # Firewall layers (inside the boundary, under the autonomy panel)
    ly = 618
    s.text(fx + 24, ly + 12, "6 LAYERS", size=12, weight=700, fill=T["text_muted"], ls=1.2)
    layers = [
        ("Static import scan", "autonomy never imports sim / eval"),
        ("Separate OS process", "a pipe carries messages only"),
        ("Slot whitelist", "SensorFrame fields only"),
        ("Runtime file guard", "audit-hook allow-list on open()"),
        ("Seed pre-registration", "eval seeds 0–59 never tuned on"),
        ("Post-hoc evaluation", "GT read only after the run"),
    ]
    for k, (a, b) in enumerate(layers):
        col, row = k // 3, k % 3
        lx, lyy = fx + 24 + col * 322, ly + 30 + row * 56
        s.badge(lx + 13, lyy + 16, str(k + 1), r=13, fill=T["navy"])
        s.text(lx + 34, lyy + 14, a, size=14, weight=700, fill=T["text"])
        s.text(lx + 34, lyy + 33, b, size=12.5, fill=T["text_secondary"])

    # What crosses the firewall
    mid = (wx + ww + fx) / 2  # labels sit between the world panel and the firewall line
    s.line(wx + ww + 4, 250, ax - 6, 250, stroke=T["accent"], sw=2.4, arrow=True)
    s.text(mid, 238, "SensorFrame", size=13.5, weight=700, fill=T["accent"], anchor="middle")
    s.lines(mid, 272, ["images, ticks,", "gyro"], size=12, lh=16, fill=T["text_secondary"], anchor="middle")
    s.line(ax - 4, 430, wx + ww + 6, 430, stroke=T["text"], sw=2.4, arrow=True)
    s.text(mid, 418, "WheelCmd", size=13.5, weight=700, fill=T["text"], anchor="middle")
    s.lines(mid, 452, ["ω_L, ω_R", "rad/s"], size=12, lh=16, fill=T["text_secondary"], anchor="middle")

    # Operator console
    ox, oy, ow, oh = 1300, 20, 520, 330
    panel(ox, oy, ow, oh, "OPERATOR CONSOLE", "no video", T["text_secondary"])
    s.lines(ox + 24, oy + 84, [f"Live costmap view at {FACTS['telemetry_hz']}", "pose, mode, v_cap, R_cert, reason code"],
            size=15, lh=24, fill=T["text"])
    s.lines(ox + 24, oy + 150, [f"Link emulator: {FACTS['link_kbps']}, {FACTS['link_loss']} loss,",
                                f"{FACTS['link_latency']} latency"], size=14, lh=22, fill=T["text_secondary"])
    bx = ox + 24
    for lab, col in (("GO", MODE["NOMINAL"]), ("HOLD", MODE["CAUTION"]), ("RESUME", T["accent"]), ("E-STOP", C[CellState.POSITIVE])):
        bx += s.chip(bx, oy + 230, lab, col, size=13.5, h=30) + 10
    s.text(ox + 24, oy + 294, "uplink: OperatorCmd", size=13, mono=True, fill=T["text_secondary"])
    # telemetry link autonomy <-> operator (from the Link codec module row)
    link_y = ay + 66 + 5 * 80 + 24
    v1, v2 = fx + fw + 36, fx + fw + 70  # vertical runs in the gap between firewall and right panels
    s.path(f"M{ax + aw + 4},{link_y} L{v1},{link_y} L{v1},{oy + 120} L{ox - 6},{oy + 120}", stroke=T["accent"], sw=2.2,
           dash="7 5", arrow=True)
    tmid = (link_y + oy + 120) / 2
    s.raw(f'<text x="{v1 - 10}" y="{tmid}" font-family="{FONT}" font-size="12.5" font-weight="700" '
          f'fill="{T["accent"]}" text-anchor="middle" transform="rotate(-90 {v1 - 10} {tmid})">'
          f'telemetry {FACTS["telemetry_hz"]}</text>')
    s.path(f"M{ox - 4},{oy + 270} L{v2},{oy + 270} L{v2},{link_y + 24} L{ax + aw + 6},{link_y + 24}", stroke=T["text"],
           sw=1.8, dash="3 4", arrow=True)
    umid = (link_y + oy + 290) / 2
    s.raw(f'<text x="{v2 + 18}" y="{umid}" font-family="{FONT}" font-size="12.5" font-weight="700" '
          f'fill="{T["text"]}" text-anchor="middle" transform="rotate(-90 {v2 + 18} {umid})">operator commands</text>')

    # Evaluator
    ex, ey, ew, eh = 1300, 390, 520, 330
    panel(ex, ey, ew, eh, "EVALUATOR", "offline, after the run", T["text_secondary"])
    s.lines(ex + 24, ey + 84, ["Reads gt/ and autonomy/ logs", "only after the run has ended"], size=15, lh=24, fill=T["text"])
    s.lines(ex + 24, ey + 150, ["Success, collisions, ditch entries,", "clearance, time, localisation error"], size=14, lh=22,
            fill=T["text_secondary"])
    hx = ex + 24
    for lab in ("Tested", "Simulated", "Estimated", "Proposed", "Literature"):
        hx += s.chip(hx, ey + 232, lab, ps.HONESTY_COLORS[lab], size=12, h=26, pad=9) + 8
    s.text(ex + 24, ey + 292, "every deck number → results/claims.csv", size=13, mono=True, fill=T["text_secondary"])
    # autonomy logs -> evaluator (short hop), GT logs -> evaluator (around the firewall, below it)
    s.path(f"M{ax + aw + 4},{ay + ah - 12} L{ex - 6},{ay + ah - 12}", stroke=T["text_secondary"], sw=1.8, dash="3 4",
           arrow=True)
    s.text((fx + fw + ex) / 2 + 6, ay + ah + 8, "autonomy/ logs", size=12, fill=T["text_secondary"], anchor="middle")
    gy = 862
    s.path(f"M{wx + ww / 2},{wy + wh + 2} L{wx + ww / 2},{gy} L{ex + ew / 2},{gy} L{ex + ew / 2},{ey + eh + 6}",
           stroke=C[CellState.POSITIVE], sw=2, dash="7 5", arrow=True)
    s.rect(fx + 150, gy - 14, 380, 28, fill=T["bg"])
    s.text(fx + fw / 2, gy + 5, "gt/ logs: post-hoc only, never through the autonomy", size=13.5, weight=700,
           fill=C[CellState.POSITIVE], anchor="middle")
    return s


# =========================================================================== (iii) model + pipeline blocks
def diagram_model_pipeline() -> Svg:
    W, H = 1840, 640
    s = Svg(W, H)
    pw, gap, py, ph = 580, 30, 20, 600
    xs = [20 + k * (pw + gap) for k in range(3)]
    titles = [("TERRAIN SEGMENTATION", "what the ground is"), ("INTEGRITY MONITOR", "visual RAIM: can VO be trusted?"),
              ("MPPI LOOP", "sampling planner on skid-steer")]
    for x, (t, sub) in zip(xs, titles):
        s.card(x, py, pw, ph, fill=T["surface"], stroke=T["border"])
        s.text(x + 24, py + 40, t, size=18, weight=700, fill=T["navy"], ls=0.4)
        s.text(x + 24, py + 64, sub, size=14, fill=T["text_secondary"])

    # ---- A: segmentation
    x = xs[0]
    y = py + 100
    cx = x + 24
    for name in ("RUGD", "RELLIS-3D", "GOOSE"):
        cx += s.chip(cx, y, name, T["text_secondary"], size=13.5, h=30) + 10
    s.line(cx + 4, y + 15, cx + 38, y + 15, stroke=T["text_secondary"], sw=2, arrow=True)
    s.chip(cx + 46, y, "OFFROAD5", T["accent"], size=13.5, h=30)
    s.text(x + 24, y + 56, "5-class re-labelled off-road training data", size=13, fill=T["text_secondary"])
    s.line(x + pw / 2, y + 70, x + pw / 2, y + 100, stroke=T["text_secondary"], sw=2, arrow=True)
    my = y + 108
    s.card(x + 24, my, pw - 48, 118, fill=T["bg"], stroke=T["accent"], sw=2, rx=12)
    s.text(x + 44, my + 36, "LR-ASPP  MobileNetV3-Large", size=18, weight=700, fill=T["text"])
    s.text(x + 44, my + 66, f"{FACTS['seg_params']} parameters", size=16, mono=True, fill=T["accent"], weight=700)
    s.text(x + 44, my + 94, f"input {FACTS['seg_input']} · ONNX opset 17 · ONNX Runtime CPU", size=13.5,
           fill=T["text_secondary"])
    s.line(x + pw / 2, my + 122, x + pw / 2, my + 150, stroke=T["text_secondary"], sw=2, arrow=True)
    cy = my + 170
    classes = [("0", "background / sky", C[CellState.UNSEEN]), ("1", "obstacle", C[CellState.POSITIVE]),
               ("2", "water / mud", C[CellState.WATER]), ("3", "unstable: grass, dirt", C[CellState.CREST_SHADOW]),
               ("4", "stable: path, gravel", C[CellState.GROUND])]
    for k, (i, lab, col) in enumerate(classes):
        cxx, cyy = x + 24 + (k % 2) * 270, cy + (k // 2) * 34
        s.swatch(cxx, cyy, col, size=18, rx=4)
        s.text(cxx + 28, cyy + 14, i, size=14, mono=True, fill=T["text_muted"], weight=700)
        s.text(cxx + 46, cyy + 14, lab, size=14.5, fill=T["text"])
    s.line(x + pw / 2, cy + 90, x + pw / 2, cy + 112, stroke=T["text_secondary"], sw=2, arrow=True)
    s.card(x + 24, cy + 118, pw - 48, 48, fill=T["bg"], stroke=T["border"], rx=10)
    s.text(x + 42, cy + 148, "class ids + per-pixel entropy → semantic cost in the map", size=14, fill=T["text"])
    s.text(x + 24, py + ph - 24, "runs at 1/3 camera rate · code BSD-3, runtime MIT", size=13, fill=T["text_muted"])

    # ---- B: integrity monitor
    x = xs[1]
    y = py + 96
    s.card(x + 24, y, pw - 48, 180, fill=T["bg"], stroke=T["border"], rx=12)
    s.text(x + 42, y + 28, "12 FEATURES / FRAME", size=12, weight=700, fill=T["text_muted"], ls=1.1)
    feats_vo = ["vo_inliers", "vo_inlier_ratio", "vo_reproj_rmse_px", "vo_coverage", "vo_track_age", "vo_hess_min_eig"]
    feats_im = ["img_lapvar", "img_sat_frac", "img_dark_frac", "img_rms_contrast", "img_dark_channel", "disp_density_ground"]
    for k, (a, b) in enumerate(zip(feats_vo, feats_im)):
        s.text(x + 42, y + 54 + k * 21, a, size=13, mono=True, fill=T["vo_blue"])
        s.text(x + 300, y + 54 + k * 21, b, size=13, mono=True, fill=T["text"])
    steps = [("standardise → logistic", "p_fail = σ(w·z + b)"), ("trust", "q = 1 − p_fail  (fast drop, slow recovery)")]
    yy = y + 188
    for a, b in steps:
        s.line(x + pw / 2, yy, x + pw / 2, yy + 22, stroke=T["text_secondary"], sw=2, arrow=True)
        s.card(x + 24, yy + 28, pw - 48, 54, fill=T["bg"], stroke=T["border"], rx=10)
        s.text(x + 42, yy + 50, a, size=12, weight=700, fill=T["text_muted"], ls=0.8)
        s.text(x + 42, yy + 72, b, size=14.5, mono=True, fill=T["text"])
        yy += 84
    s.line(x + pw / 2, yy, x + pw / 2, yy + 22, stroke=T["text_secondary"], sw=2, arrow=True)
    # mode bar over q in [0, 1]
    bx, by, bw, bh = x + 24, yy + 32, pw - 48, 30
    segs = [(0.0, 0.2, "SAFE_STOP", "SAFE-STOP"), (0.2, 0.4, "DEGRADED", "DEGRADED"), (0.4, 0.7, "CAUTION", "CAUTION"),
            (0.7, 1.0, "NOMINAL", "NOMINAL")]
    for a, b, key, lab in segs:
        s.rect(bx + a * bw + 1, by, (b - a) * bw - 2, bh, fill=MODE[key], rx=4)
        s.text(bx + (a + b) / 2 * bw, by + 20, lab, size=11.5, weight=700, fill=T["bg"], anchor="middle", ls=0.5)
    for q in (0.0, 0.2, 0.4, 0.7, 1.0):
        s.text(bx + q * bw, by + bh + 18, f"{q:g}", size=12, mono=True, fill=T["text_secondary"], anchor="middle")
    s.text(x + 24, py + ph - 24, "trained on real KITTI frames + synthetic degradations", size=13, fill=T["text_muted"])

    # ---- C: MPPI loop
    x = xs[2]
    # rollout fan
    ox, oy = x + 70, py + 222
    s.rect(x + 24, py + 92, pw - 48, 250, fill=T["bg"], stroke=T["border"], rx=12)
    ux = x + 452  # unseen ground beyond the certified range (right edge of the box)
    s.rect(ux, py + 93, x + pw - 25 - ux, 248, fill=C[CellState.UNSEEN])
    s.text((ux + x + pw - 25) / 2, py + 118, "unseen", size=12.5, weight=700, fill=T["text_secondary"], anchor="middle")
    s.rect(x + 300, py + 128, 38, 38, fill=C[CellState.POSITIVE], rx=8)
    s.text(x + 346, py + 152, "rock", size=12, fill=T["text_secondary"])
    s.rect(x + 318, py + 262, 120, 20, fill=C[CellState.DITCH_CANDIDATE], rx=4)
    s.text(x + 378, py + 300, "ditch", size=12, fill=T["text_secondary"], anchor="middle")
    for k in range(-6, 7):
        curv = k * 15
        d = f"M{ox},{oy} Q{ox + 190},{oy + curv * 0.35} {ox + 400},{oy + curv * 1.15}"
        s.path(d, stroke=T["vo_blue"], sw=1.2, opacity=0.35)
    s.path(f"M{ox},{oy} Q{ox + 190},{oy + 4} {ux - 74},{oy - 6}", stroke=T["path"], sw=3.4)
    s.circle(ux - 74, oy - 6, 5, fill=T["path"])
    s.rect(ox - 26, oy - 14, 30, 28, fill=T["surface"], stroke=T["text"], sw=1.6, rx=6)
    s.text(x + 44, py + 326, f"K = {FACTS['mppi_k']} rollouts × {FACTS['mppi_t']}", size=13.5, mono=True, fill=T["text_secondary"])
    s.rect(ux - 250, oy + 6, 172, 24, fill=T["bg"], rx=6)
    s.text(ux - 84, oy + 23, "plan ends on seen ground", size=12.5, weight=700, fill=T["path"], anchor="end")
    steps = [("1 sample", "K control sequences around U"), ("2 cost", "S_k: map · lethal · certification · cost-to-go"),
             ("3 weight", "w_k = exp(−(S_k − min S) / λ)"), ("4 update", "U += Σ w_k ε_k → mixer → wheel rad/s")]
    yy = py + 362
    for k, (a, b) in enumerate(steps):
        s.card(x + 24, yy, pw - 96, 46, fill=T["bg"], stroke=T["border"], rx=10)
        s.text(x + 42, yy + 29, a, size=13, weight=700, fill=T["accent"])
        s.text(x + 128, yy + 29, b, size=13.5, mono=k > 1, fill=T["text"])
        if k < len(steps) - 1:
            s.line(x + 60, yy + 47, x + 60, yy + 53, stroke=T["text_muted"], sw=1.2)
        yy += 54
    # loop arrow on the right: from the last step back to the first
    lx = x + pw - 50
    s.path(f"M{x + pw - 72},{py + 362 + 3 * 54 + 23} L{lx},{py + 362 + 3 * 54 + 23} L{lx},{py + 362 + 23} L{x + pw - 66},{py + 362 + 23}",
           stroke=T["accent"], sw=2, arrow=True)
    s.raw(f'<text x="{lx + 16}" y="{py + 362 + 100}" font-family="{FONT}" font-size="12" fill="{T["text_secondary"]}" '
          f'text-anchor="middle" transform="rotate(90 {lx + 16} {py + 362 + 100})">warm start, every tick</text>')
    return s


# =========================================================================== (iv) missing ground
def diagram_missing_ground() -> Svg:
    W, H = 1840, 720
    s = Svg(W, H)
    green, grey, mag = C[CellState.GROUND], C[CellState.UNSEEN], C[CellState.DITCH_CANDIDATE]
    g_y = 520.0  # ground line [px]
    cam = (190.0, 270.0)  # camera optical centre [px]
    lip, far = 820.0, 990.0  # near lip / far wall x [px] (w = far - lip)
    depth_y = 650.0

    s.card(20, 20, 1180, 680, fill=T["bg"], stroke=T["border"])
    s.text(44, 60, "SIDE VIEW (SCHEMATIC)", size=12, weight=700, fill=T["text_muted"], ls=1.2)
    s.text(44, 98, "A ditch is missing ground, not an object", size=24, weight=700, fill=T["navy"])
    s.text(44, 128, "Its opening shrinks as 1/R² and its interior is never observed.", size=16, fill=T["text_secondary"])

    # terrain body, then the surface outline
    s.polygon([(40, g_y), (lip, g_y), (lip, depth_y), (far, depth_y), (far, g_y), (1180, g_y), (1180, 690), (40, 690)],
              fill=T["surface"])
    # the grazing ray through the near lip meets the far wall at y_hit
    slope = (g_y - cam[1]) / (lip - cam[0])
    y_hit = g_y + slope * (far - lip)
    # visible-angle wedge between the near-lip and far-lip rays
    s.polygon([(cam[0] + 18, cam[1]), (lip, g_y), (far, g_y)], fill=T["accent"], opacity=0.10)
    # hidden interior (never observed), with a light hatch
    s.polygon([(lip, g_y), (far, y_hit), (far, depth_y), (lip, depth_y)], fill=grey)
    for k in range(10):
        x1 = lip + 6 + k * 17
        s.line(x1, depth_y - 3, min(x1 + 50, far - 3), depth_y - 3 - (min(x1 + 50, far - 3) - x1), stroke=T["bg"], sw=1.1)
    s.path(f"M40,{g_y} L{lip},{g_y} L{lip},{depth_y} L{far},{depth_y} L{far},{g_y} L1180,{g_y}", stroke=T["text"], sw=2.4)
    s.line(far, g_y, far, y_hit, stroke=mag, sw=7)  # visible part of the far wall
    s.line(330, g_y, lip, g_y, stroke=green, sw=7)
    s.line(far, g_y, 1178, g_y, stroke=green, sw=7)

    # vehicle, mast, camera
    s.rect(110, g_y - 60, 170, 42, fill=T["surface"], stroke=T["text_secondary"], sw=1.6, rx=10)
    for wx_ in (140, 250):
        s.circle(wx_, g_y - 16, 16, fill=T["text"], stroke=T["bg"], sw=2)
    s.line(cam[0], g_y - 60, cam[0], cam[1] + 12, stroke=T["text_secondary"], sw=4)
    s.rect(cam[0] - 26, cam[1] - 14, 52, 28, fill=T["bg"], stroke=T["text"], sw=2, rx=6)
    s.circle(cam[0] + 14, cam[1], 7, fill=T["accent"])

    # rays
    for gx in (380, 520, 670):
        s.line(cam[0] + 18, cam[1], gx, g_y, stroke=green, sw=1.4, dash="5 5")
        s.circle(gx, g_y, 4.5, fill=green)
    s.line(cam[0] + 18, cam[1], far, y_hit, stroke=mag, sw=2)
    s.circle(far, y_hit, 5, fill=mag)
    s.line(cam[0] + 18, cam[1], far, g_y, stroke=T["accent"], sw=1.6)
    for gx in (1070, 1150):
        s.line(cam[0] + 18, cam[1], gx, g_y, stroke=green, sw=1.4, dash="5 5")
        s.circle(gx, g_y, 4.5, fill=green)

    # theta callout, anchored on the wedge
    a1 = math.atan2(g_y - cam[1], lip - cam[0])
    a2 = math.atan2(g_y - cam[1], far - cam[0])
    r_arc = 600.0
    p1 = (cam[0] + 18 + r_arc * math.cos(a1), cam[1] + r_arc * math.sin(a1))
    p2 = (cam[0] + 18 + r_arc * math.cos(a2), cam[1] + r_arc * math.sin(a2))
    s.path(f"M{p2[0]:.1f},{p2[1]:.1f} A{r_arc},{r_arc} 0 0 1 {p1[0]:.1f},{p1[1]:.1f}", stroke=T["accent"], sw=3)
    bx, by = 560.0, 190.0
    s.rect(bx, by, 300, 88, fill=T["bg"], stroke=T["accent"], sw=1.5, rx=12)
    s.text(bx + 20, by + 38, "θ ≈ H·w / R²", size=24, weight=700, mono=True, fill=T["accent"])
    s.text(bx + 20, by + 68, "visible angle of the opening", size=14, fill=T["text_secondary"])
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    s.line(bx + 150, by + 90, mx, my - 6, stroke=T["accent"], sw=1.4)

    # dimensions: H (left of the vehicle), R (below ground), w (above the ditch)
    hx = 70.0
    s.line(hx, cam[1] + 2, hx, g_y - 2, stroke=T["text"], sw=1.4, arrow=True)
    s.line(hx, g_y - 2, hx, cam[1] + 2, stroke=T["text"], sw=1.4, arrow=True)
    s.line(hx - 8, cam[1], cam[0] - 30, cam[1], stroke=T["text_muted"], sw=1, dash="3 4")
    s.text(hx - 12, (cam[1] + g_y) / 2 + 7, "H", size=20, weight=700, anchor="end", fill=T["text"])
    ry = 590.0
    s.line(cam[0], ry, lip - 2, ry, stroke=T["text"], sw=1.4, arrow=True)
    s.line(lip - 2, ry, cam[0] + 2, ry, stroke=T["text"], sw=1.4, arrow=True)
    s.rect((cam[0] + lip) / 2 - 20, ry - 16, 40, 30, fill=T["surface"])
    s.text((cam[0] + lip) / 2, ry + 7, "R", size=20, weight=700, anchor="middle", fill=T["text"])
    wy_ = g_y - 36
    s.line(lip + 2, wy_, far - 2, wy_, stroke=T["text"], sw=1.4, arrow=True)
    s.line(far - 2, wy_, lip + 2, wy_, stroke=T["text"], sw=1.4, arrow=True)
    s.rect((lip + far) / 2 - 14, wy_ - 34, 28, 26, fill=T["bg"])
    s.text((lip + far) / 2, wy_ - 14, "w", size=20, weight=700, anchor="middle", fill=T["text"])

    # callouts
    s.text(lip - 10, g_y + 30, "near lip", size=14, anchor="end", fill=T["text"], weight=700)
    s.lines((lip + far) / 2, depth_y - 40, ["hidden interior:", "never observed"], size=14, lh=19, anchor="middle",
            fill=T["text"], weight=700)
    s.lines(far + 16, y_hit - 22, ["far wall reappears", "below lip height"], size=14, lh=19, fill=mag, weight=700)
    s.text(44, 640, "Seen ground is certified. The gap between lip and far wall is", size=15, fill=T["text"])
    s.text(44, 662, "MISSING GROUND: kept unseen / ditch, never planned as free.", size=15, weight=700, fill=T["text"])

    # ---------------------------------------------------------------- inset: column profile
    ix, iy, iw, ih = 1230, 20, 590, 680
    s.card(ix, iy, iw, ih, fill=T["bg"], stroke=T["border"])
    s.text(ix + 24, iy + 40, "ONE IMAGE COLUMN, BOTTOM → UP", size=12, weight=700, fill=T["text_muted"], ls=1.2)
    s.text(ix + 24, iy + 68, "Measured range per image row", size=18, weight=700, fill=T["text"])
    px0, py0, pw_, ph_ = ix + 70, iy + 100, iw - 110, 340  # plot area
    s.line(px0, py0 + ph_, px0 + pw_, py0 + ph_, stroke=T["border"], sw=1.4)
    s.line(px0, py0 + ph_, px0, py0, stroke=T["border"], sw=1.4)
    s.text(px0 + pw_ / 2, py0 + ph_ + 30, "image row  (near → far)", size=14, anchor="middle", fill=T["text_secondary"])
    s.raw(f'<text x="{px0 - 22}" y="{py0 + ph_ / 2}" font-family="{FONT}" font-size="14" fill="{T["text_secondary"]}" '
          f'text-anchor="middle" transform="rotate(-90 {px0 - 22} {py0 + ph_ / 2})">range</text>')
    # Schematic profile: ground rows rise smoothly; the vertical far wall is seen over several rows at a
    # constant range ~R + w, then far ground resumes. Illustrative numbers (not to scale).
    r_lip, w_d, n = 3.0, 1.4, 44

    def ground_range(u: float) -> float:  # flat-ground range at normalised row u (0 = bottom row)
        ang = math.radians(38.0) * (1 - u) + math.radians(9.0) * u
        return D.CAM_HEIGHT_M / math.tan(ang)

    r_top = ground_range(1.0) + 0.3

    def to_px(u: float, r: float) -> tuple[float, float]:
        return px0 + 10 + u * (pw_ - 20), py0 + ph_ - r / r_top * (ph_ - 10)

    us = [i / (n - 1) for i in range(n)]
    for u in us:
        r = ground_range(u)
        if r < r_lip:
            s.circle(*to_px(u, r), 4, fill=green)
        elif r < r_lip + w_d:
            s.circle(*to_px(u, r_lip + w_d), 4.5, fill=mag)  # far wall: constant range
        else:
            s.circle(*to_px(u, r), 4, fill=green)
    u_lip = next(u for u in us if ground_range(u) >= r_lip)
    ja, jb = to_px(u_lip, r_lip), to_px(u_lip, r_lip + w_d)
    s.line(ja[0] - 16, ja[1] + 2, ja[0] - 16, jb[1] + 4, stroke=T["text"], sw=1.6, arrow=True)
    s.lines(ja[0] - 26, (ja[1] + jb[1]) / 2 + 2, ["range", "jump ≈ w"], size=14, lh=18, anchor="end", fill=T["text"],
            weight=700)
    s.text(to_px(0.62, 0)[0], jb[1] - 16, "far wall: flat range", size=13, fill=mag, weight=700, anchor="middle")
    ty = py0 + ph_ + 76
    s.lines(ix + 24, ty, ["Ground returns rise smoothly with the row, then jump.",
                          "The rows in between would have seen the ditch floor."], size=14, lh=20, fill=T["text"])
    s.swatch(ix + 24, ty + 44, mag, size=14)
    s.text(ix + 46, ty + 56, "reappears near lip height → DITCH CANDIDATE", size=13.5, fill=T["text"])
    s.swatch(ix + 24, ty + 70, C[CellState.CREST_SHADOW], size=14)
    s.text(ix + 46, ty + 82, "reappears well below / slopes away → CREST", size=13.5, fill=T["text"])
    s.swatch(ix + 24, ty + 96, grey, size=14)
    s.text(ix + 46, ty + 108, "never seen → UNSEEN (never free)", size=13.5, fill=T["text"])
    return s


# =========================================================================== slot-sized versions
SLOT_DIR = REPO_ROOT / "deck_assets" / "slots"
MIN_SLOT_TEXT_PX = 14.0  # smallest font size on a slot figure [slide px] (28 px in the 2x PNG)
_MEASURE_SCALE = 8  # fonts are loaded at 8x the requested size for sub-pixel width measurement


@functools.lru_cache(maxsize=None)
def _font_path(bold: bool, mono: bool) -> Optional[str]:
    """Font file of the first installed family in the diagram font stack (None if none is found)."""
    from matplotlib import font_manager as fm

    for fam in ps.MONO_FAMILIES if mono else ps.SANS_FAMILIES:
        try:
            return fm.findfont(fm.FontProperties(family=fam, weight="bold" if bold else "normal"),
                               fallback_to_default=False)
        except ValueError:
            continue
    return None


@functools.lru_cache(maxsize=4096)
def measure(s: str, size: float, weight: int = 400, mono: bool = False, ls: float = 0.0) -> float:
    """Rendered width [px] of ``s`` from the installed font's advances (falls back to :func:`text_width`)."""
    width = text_width(s, size, mono)
    path = _font_path(weight >= 600, mono)
    if path:
        try:
            from PIL import ImageFont

            width = ImageFont.truetype(path, int(round(size * _MEASURE_SCALE))).getlength(s) / _MEASURE_SCALE
        except (ImportError, OSError):
            pass
    return width + ls * len(s)


def check_min_text(svg: Svg, name: str, min_px: float = MIN_SLOT_TEXT_PX) -> None:
    """Raise if any text element of ``svg`` is set below ``min_px`` (slot figures must stay legible)."""
    sizes = [float(v) for v in re.findall(r'font-size="([0-9.]+)"', svg.to_string())]
    small = [v for v in sizes if v < min_px]
    if small:
        raise ValueError(f"{name}: {len(small)} text element(s) below {min_px:g} px (smallest {min(small):g} px)")


def slot_how_it_works() -> Svg:
    """Slide-2 hero at slot size 1792x356: six stages in one row, link strip and health-mode chips."""
    W, H = 1792, 356
    s = Svg(W, H)
    cw, gap, x0, y0, ch = 274, 28, 4, 2, 266
    xs = [x0 + i * (cw + gap) for i in range(6)]
    stages: list[tuple[str, list[str]]] = [
        ("SENSE", [f"Stereo pair, {FACTS['cam_hz']}", "Wheel encoders + gyro"]),
        ("SEE", ["SGBM stereo depth", "Ground model", "5 terrain classes"]),
        ("KNOW THE UNSEEN", ["Seen-ground map", "Missing-ground detector"]),
        ("LOCALISE", ["Stereo visual odometry", "Integrity monitor → q", "Wheel / gyro EKF"]),
        ("DECIDE", ["MPPI planner", "Seen-distance governor", "Safety supervisor"]),
        ("ACT", ["Skid-steer mixer", "(v, ω) → wheel rad/s", "Rate limits + watchdog"]),
    ]
    for i, (x, (name, rows)) in enumerate(zip(xs, stages)):
        core = i == 2
        s.card(x, y0, cw, ch, stroke=T["accent"] if core else T["border"], sw=2.4 if core else 1.5)
        s.badge(x + 32, y0 + 34, str(i + 1), r=17, fill=T["accent"] if core else T["navy"])
        size = 20.0 if measure(name, 20, 700, ls=0.3) <= cw - 58 - 14 else 18.0
        s.text(x + 58, y0 + 41, name, size=size, weight=700, fill=T["navy"], ls=0.3)
        s.line(x + 20, y0 + 64, x + cw - 20, y0 + 64, stroke=T["border"], sw=1.2)
        y = y0 + 92
        for row in rows:
            s.circle(x + 25, y - 5.5, 3.2, fill=T["accent"])
            s.text(x + 37, y, row, size=18)
            y += 26
        if core:
            s.text(x + 20, y, "Unknown is never free", size=17, weight=700, fill=T["accent"])
        if i < 5:
            arrow_h(s, x + cw + 5, x + cw + gap - 5, y0 + 116)
    _slot_how_glyphs(s, xs, y0 + 160, cw)

    # bottom strip: link (fed by the seen-ground map) under stages 1-3, health modes under stages 4-6
    sy, sh = y0 + ch + 22, 58
    lx, lw = xs[0], xs[2] + cw - xs[0]
    mx = xs[2] + cw / 2
    s.line(mx, y0 + ch + 3, mx, sy - 4, stroke=T["accent"], sw=2, dash="3 3", arrow=True)
    s.card(lx, sy, lw, sh, fill=T["bg"], stroke=T["accent"], sw=1.5, rx=12)
    gx, gy = lx + 16, sy + 7
    s.rect(gx, gy, 68, 44, fill=T["surface"], stroke=T["text_secondary"], sw=1.3, rx=5)
    code = {"U": C[CellState.UNSEEN], "G": C[CellState.GROUND], "D": C[CellState.DITCH_CANDIDATE],
            "P": C[CellState.POSITIVE], "W": C[CellState.WATER]}
    for r, row in enumerate(["UUUUUU", "UGGDDU", "GGGGGG", "GGPGGW"]):
        for c, ch_ in enumerate(row):
            s.rect(gx + 4 + c * 10, gy + 2 + r * 10, 9, 9, fill=code[ch_], rx=1.5)
    tx, base = gx + 68 + 20, sy + sh / 2 + 6.5
    s.text(tx, base - 0.5, "LINK", size=16, weight=700, fill=T["text_secondary"], ls=1.2)
    tx += measure("LINK", 16, 700, ls=1.2) + 14
    main = f"costmap every {TELEMETRY_PERIOD_S:g} s → console replay"
    s.text(tx, base, main, size=18, weight=700, fill=T["navy"])
    tx += measure(main, 18, 700) + 16
    s.text(tx, base, f"{D.LINK_KBPS:g} kbit/s link, not video", size=16, fill=T["text_secondary"])

    hx = xs[3]
    hw = xs[5] + cw - hx
    s.card(hx, sy, hw, sh, fill=T["surface"], stroke=T["border"], rx=12)
    s.text(hx + 20, base - 0.5, "MODES", size=16, weight=700, fill=T["text_secondary"], ls=1.2)
    cx = hx + 20 + measure("MODES", 16, 700, ls=1.2) + 16
    modes = [("NOMINAL", "NOMINAL"), ("CAUTION", "CAUTION"), ("DEGRADED", "DEGRADED"),
             ("STOP_AND_LOOK", "STOP-AND-LOOK"), ("SAFE_STOP", "SAFE-STOP")]
    ph = 34
    py = sy + (sh - ph) / 2
    for key, lab in modes:
        w = 30 + measure(lab, 15, 700) + 15
        s.rect(cx, py, w, ph, fill=T["bg"], stroke=MODE[key], sw=1.6, rx=ph / 2)
        s.circle(cx + 17, py + ph / 2, 5.5, fill=MODE[key])
        s.text(cx + 30, py + ph / 2 + 5.4, lab, size=15, weight=700, fill=T["text"])
        cx += w + 10
    return s


def _sub_label(s: Svg, x: float, y: float, base: str, sub: str, size: float = 16, color: str = T["accent"]) -> None:
    """Centred bold label with a subscript (e.g. ω_L), drawn in the sans face so ω stays distinct from w."""
    s.raw(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" font-weight="700" fill="{color}" '
          f'text-anchor="middle">{escape(base)}<tspan dy="4" font-size="{MIN_SLOT_TEXT_PX:g}">{escape(sub)}</tspan></text>')


def _slot_how_glyphs(s: Svg, xs: list[int], vy: float, cw: int) -> None:
    """Stage glyphs of the slot hero, in the lower part of each card (vy = glyph band top)."""
    # 1 SENSE: stereo pair + crossed-out GNSS
    x = xs[0]
    for k, lab in enumerate("LR"):
        cx = x + 54 + k * 74
        s.rect(cx - 28, vy, 56, 36, fill=T["bg"], stroke=T["text_secondary"], sw=1.6, rx=8)
        s.circle(cx, vy + 18, 10, fill=T["surface"], stroke=T["accent"], sw=2)
        s.circle(cx, vy + 18, 4, fill=T["accent"])
        s.text(cx, vy + 56, lab, size=16, weight=700, fill=T["text_secondary"], anchor="middle")
    gy, gw = vy + 68, measure("GNSS", 15, 700) + 24
    s.rect(x + 26, gy, gw, 28, fill=T["bg"], stroke=T["text_secondary"], sw=1.4, rx=14)
    s.text(x + 26 + gw / 2, gy + 19.5, "GNSS", size=15, weight=700, fill=T["text_secondary"], anchor="middle")
    s.line(x + 22, gy + 25, x + 30 + gw, gy + 3, stroke=C[CellState.POSITIVE], sw=2.2)
    s.text(x + 40 + gw, gy + 19.5, "DENIED", size=15, weight=700, fill=C[CellState.POSITIVE], ls=0.8)

    # 2 SEE: disparity block (near = light) and a mini segmentation of the same view
    x = xs[1]
    bw, bh = 104, 62
    s.raw('<defs><linearGradient id="slotdisp" x1="0" y1="0" x2="0" y2="1">'
          f'<stop offset="0" stop-color="{T["navy"]}"/><stop offset="1" stop-color="#DBE6FE"/></linearGradient>'
          f'<clipPath id="slotseg"><rect x="{x + cw - 24 - bw}" y="{vy}" width="{bw}" height="{bh}" rx="6"/></clipPath></defs>')
    s.rect(x + 24, vy, bw, bh, fill="url(#slotdisp)", rx=6)
    s.text(x + 24 + bw / 2, vy + bh + 22, "depth", size=16, fill=T["text_secondary"], anchor="middle")
    sx = x + cw - 24 - bw
    s.raw('<g clip-path="url(#slotseg)">')
    s.rect(sx, vy, bw, bh, fill=C[CellState.UNSEEN])
    s.rect(sx, vy + 22, bw, bh - 22, fill=C[CellState.CREST_SHADOW])
    s.polygon([(sx + 30, vy + bh), (sx + 49, vy + 22), (sx + 57, vy + 22), (sx + 84, vy + bh)], fill=C[CellState.GROUND])
    s.raw(f'<ellipse cx="{sx + 14}" cy="{vy + 50}" rx="13" ry="6" fill="{C[CellState.WATER]}"/>')
    s.rect(sx + 74, vy + 12, 16, 17, fill=C[CellState.POSITIVE], rx=3)
    s.raw("</g>")
    s.text(sx + bw / 2, vy + bh + 22, "5 classes", size=16, fill=T["text_secondary"], anchor="middle")

    # 3 KNOW THE UNSEEN: mini seen-ground map + key
    x = xs[2]
    grid = ["UUUUUUUUU", "UUUUUUUUU", "UUUCCCUUU", "GGGGGGUUW", "GDDDDDDGW", "GGGGGGGGG", "GGGGPGGGG"]
    code = {"U": C[CellState.UNSEEN], "G": C[CellState.GROUND], "D": C[CellState.DITCH_CANDIDATE],
            "C": C[CellState.CREST_SHADOW], "P": C[CellState.POSITIVE], "W": C[CellState.WATER]}
    pitch, cs = 12.5, 11
    gx, gy = x + 24, vy - 6
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            s.rect(gx + c * pitch, gy + r * pitch, cs, cs, fill=code[ch], rx=2)
    tip = (gx + 4 * pitch + cs / 2, gy + len(grid) * pitch + 1)
    s.polygon([tip, (tip[0] - 6, tip[1] + 10), (tip[0] + 6, tip[1] + 10)], fill=T["path"])
    key = [("seen ground", CellState.GROUND), ("unseen", CellState.UNSEEN), ("ditch", CellState.DITCH_CANDIDATE),
           ("crest", CellState.CREST_SHADOW)]
    for k, (lab, st) in enumerate(key):
        kx, ky = x + 144, vy - 4 + k * 24
        s.swatch(kx, ky, C[st], size=14, rx=3)
        s.text(kx + 20, ky + 12.5, lab, size=16, fill=T["text_secondary"])

    # 4 LOCALIZE: VO track with growing uncertainty ellipses
    x = xs[3]
    pts = [(x + 34, vy + 58), (x + 82, vy + 46), (x + 130, vy + 30), (x + 178, vy + 22), (x + 228, vy + 8)]
    s.path("M" + " L".join(f"{px:.0f},{py:.0f}" for px, py in pts), stroke=T["vo_blue"], sw=2.4)
    for k, (px, py) in enumerate(pts):
        s.raw(f'<ellipse cx="{px}" cy="{py}" rx="{5 + 2.6 * k:.1f}" ry="{3.5 + 1.5 * k:.1f}" fill="{T["vo_blue"]}" '
              f'fill-opacity="0.10" stroke="{T["vo_blue"]}" stroke-width="1"/>')
        s.circle(px, py, 3.4, fill=T["vo_blue"])
    s.text(x + 22, vy + 94, "q gates VO trust + speed", size=16)

    # 5 DECIDE: seen-distance governor inequality
    x = xs[4]
    s.rect(x + 20, vy, cw - 40, 62, fill=T["surface"], stroke=T["border"], rx=10)
    s.text(x + cw / 2, vy + 26, "v²/2a + v·T_r + B", size=16, mono=True, anchor="middle")
    s.text(x + cw / 2, vy + 50, "≤ R_cert", size=16, mono=True, anchor="middle", fill=T["accent"], weight=700)
    s.text(x + 22, vy + 94, "stop inside seen ground", size=16)

    # 6 ACT: top-down skid-steer glyph with wheel-speed arrows
    x = xs[5]
    bx, by = x + cw / 2 - 26, vy - 2
    s.rect(bx, by, 52, 78, fill=T["surface"], stroke=T["text_secondary"], sw=1.6, rx=8)
    for dx in (-12, 54):
        for dy in (5, 47):
            s.rect(bx + dx, by + dy, 10, 26, fill=T["text"], rx=3)
    for ax_, top, sub in ((bx - 36, by + 12, "L"), (bx + 88, by + 24, "R")):
        s.line(ax_, by + 68, ax_, top, stroke=T["accent"], sw=2.2, arrow=True)
        _sub_label(s, ax_ - 2, by + 90, "ω", sub, size=18)


def slot_missing_ground() -> Svg:
    """Slide-3 concept at slot size 1180x384: side-view schematic (left) + range-per-row plot (right)."""
    W, H = 1180, 384
    s = Svg(W, H)
    green, grey, mag = C[CellState.GROUND], C[CellState.UNSEEN], C[CellState.DITCH_CANDIDATE]

    # ---------------------------------------------------------------- left: side view
    lx, ly, lw, lh = 2, 2, 724, 380
    s.card(lx, ly, lw, lh)
    s.text(lx + 24, ly + 44, "A ditch is missing ground, not an object", size=27, weight=700, fill=T["navy"])
    g_y = 280.0  # ground line
    cam = (150.0, 150.0)  # camera optical centre (mast top)
    lens = (cam[0] + 12, cam[1])
    lip, far = 455.0, 600.0  # near lip / far wall x (w = far - lip)
    depth_y = 368.0
    x_l, x_r, y_b = lx + 16, lx + lw - 16, ly + lh - 8
    s.polygon([(x_l, g_y), (lip, g_y), (lip, depth_y), (far, depth_y), (far, g_y), (x_r, g_y), (x_r, y_b), (x_l, y_b)],
              fill=T["surface"])
    slope = (g_y - lens[1]) / (lip - lens[0])
    y_hit = g_y + slope * (far - lip)  # grazing ray over the near lip meets the far wall here
    s.polygon([lens, (lip, g_y), (far, g_y)], fill=T["accent"], opacity=0.10)
    s.polygon([(lip, g_y), (far, y_hit), (far, depth_y), (lip, depth_y)], fill=grey)
    for k in range(int((far - lip) // 16)):
        x1 = lip + 6 + k * 16
        x2 = min(x1 + 40, far - 3)
        s.line(x1, depth_y - 3, x2, depth_y - 3 - (x2 - x1), stroke=T["bg"], sw=1.1)
    s.path(f"M{x_l},{g_y} L{lip},{g_y} L{lip},{depth_y} L{far},{depth_y} L{far},{g_y} L{x_r},{g_y}", stroke=T["text"],
           sw=2.4)
    s.line(far, g_y, far, y_hit, stroke=mag, sw=7)
    s.line(262, g_y, lip, g_y, stroke=green, sw=7)
    s.line(far, g_y, x_r - 2, g_y, stroke=green, sw=7)

    # vehicle, mast, camera
    s.rect(70, g_y - 46, 160, 32, fill=T["surface"], stroke=T["text_secondary"], sw=1.6, rx=9)
    for wx_ in (100, 200):
        s.circle(wx_, g_y - 13, 13, fill=T["text"], stroke=T["bg"], sw=2)
    s.line(cam[0], g_y - 46, cam[0], cam[1] + 12, stroke=T["text_secondary"], sw=4)
    s.rect(cam[0] - 23, cam[1] - 13, 46, 26, fill=T["bg"], stroke=T["text"], sw=2, rx=6)
    s.circle(lens[0], lens[1], 6.5, fill=T["accent"])

    # rays: seen ground (green), grazing ray into the ditch (magenta), far lip (accent)
    for gx in (282, 345, 405, 648, 694):
        s.line(lens[0] + 4, lens[1], gx, g_y, stroke=green, sw=1.4, dash="5 5")
        s.circle(gx, g_y, 4.5, fill=green)
    s.line(lens[0] + 4, lens[1], far, y_hit, stroke=mag, sw=2)
    s.circle(far, y_hit, 5, fill=mag)
    s.line(lens[0] + 4, lens[1], far, g_y, stroke=T["accent"], sw=1.6)

    # theta callout on the wedge
    a1 = math.atan2(g_y - lens[1], lip - lens[0])
    a2 = math.atan2(g_y - lens[1], far - lens[0])
    r_arc = 290.0
    p1 = (lens[0] + r_arc * math.cos(a1), lens[1] + r_arc * math.sin(a1))
    p2 = (lens[0] + r_arc * math.cos(a2), lens[1] + r_arc * math.sin(a2))
    s.path(f"M{p2[0]:.1f},{p2[1]:.1f} A{r_arc},{r_arc} 0 0 1 {p1[0]:.1f},{p1[1]:.1f}", stroke=T["accent"], sw=3)
    cap = "visible angle of the opening"
    bw_ = measure(cap, 18) + 40
    bx, by = 322.0, 66.0
    s.rect(bx, by, bw_, 74, fill=T["bg"], stroke=T["accent"], sw=1.5, rx=12)
    s.text(bx + 20, by + 35, "θ ≈ H·w / R²", size=27, weight=700, mono=True, fill=T["accent"])
    s.text(bx + 20, by + 62, cap, size=18, fill=T["text_secondary"])
    mx_, my_ = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    s.line(mx_ + 2, by + 75, mx_, my_ - 6, stroke=T["accent"], sw=1.4)

    # dimensions: H (left of the vehicle), R (below ground), w (above the ditch)
    hx = 46.0
    s.line(hx, cam[1] + 2, hx, g_y - 2, stroke=T["text"], sw=1.4, arrow=True)
    s.line(hx, g_y - 2, hx, cam[1] + 2, stroke=T["text"], sw=1.4, arrow=True)
    s.line(hx - 8, cam[1], cam[0] - 27, cam[1], stroke=T["text_muted"], sw=1, dash="3 4")
    s.text(hx - 10, (cam[1] + g_y) / 2 + 8, "H", size=23, weight=700, anchor="end")
    ry = 330.0
    s.line(cam[0], ry, lip - 2, ry, stroke=T["text"], sw=1.4, arrow=True)
    s.line(lip - 2, ry, cam[0] + 2, ry, stroke=T["text"], sw=1.4, arrow=True)
    s.rect((cam[0] + lip) / 2 - 18, ry - 15, 36, 28, fill=T["surface"])
    s.text((cam[0] + lip) / 2, ry + 8, "R", size=23, weight=700, anchor="middle")
    wy_ = g_y - 32
    s.line(lip + 2, wy_, far - 2, wy_, stroke=T["text"], sw=1.4, arrow=True)
    s.line(far - 2, wy_, lip + 2, wy_, stroke=T["text"], sw=1.4, arrow=True)
    s.rect((lip + far) / 2 - 13, wy_ - 30, 26, 24, fill=T["bg"])
    s.text((lip + far) / 2, wy_ - 11, "w", size=23, weight=700, anchor="middle")

    # callouts
    s.text(lip - 10, g_y + 26, "near lip", size=18, weight=700, anchor="end")
    s.text((lip + far) / 2, depth_y - 10, "never observed", size=18, weight=700, anchor="middle")
    s.lines(far + 12, g_y + 30, ["far wall", "seen below", "the lip"], size=18, lh=22, fill=mag, weight=700)

    # ---------------------------------------------------------------- right: one image column
    ix, iw = 740, 438
    s.card(ix, ly, iw, lh)
    s.text(ix + 22, ly + 42, "Range along one image column", size=21, weight=700)
    px0, py0, pw_, ph_ = ix + 50, ly + 66, iw - 72, 156  # plot area
    s.line(px0, py0 + ph_, px0 + pw_, py0 + ph_, stroke=T["border"], sw=1.4)
    s.line(px0, py0 + ph_, px0, py0, stroke=T["border"], sw=1.4)
    s.text(px0 + pw_ / 2, py0 + ph_ + 26, "image row, near to far", size=17, anchor="middle", fill=T["text_secondary"])
    s.raw(f'<text x="{px0 - 16}" y="{py0 + ph_ / 2}" font-family="{FONT}" font-size="17" fill="{T["text_secondary"]}" '
          f'text-anchor="middle" transform="rotate(-90 {px0 - 16} {py0 + ph_ / 2})">range</text>')
    # Schematic profile (illustrative, not to scale): ground rows rise smoothly, the far wall is seen over
    # several rows at a constant range ~R + w, then far ground resumes.
    r_lip, w_d, n = 3.0, 1.4, 30

    def ground_range(u: float) -> float:  # flat-ground range at normalised row u (0 = bottom row)
        ang = math.radians(38.0) * (1 - u) + math.radians(9.0) * u
        return D.CAM_HEIGHT_M / math.tan(ang)

    r_top = ground_range(1.0) + 0.3

    def to_px(u: float, r: float) -> tuple[float, float]:
        return px0 + 10 + u * (pw_ - 20), py0 + ph_ - r / r_top * (ph_ - 10)

    us = [i / (n - 1) for i in range(n)]
    for u in us:
        r = ground_range(u)
        if r_lip <= r < r_lip + w_d:
            s.circle(*to_px(u, r_lip + w_d), 4.2, fill=mag)  # far wall: constant range
        else:
            s.circle(*to_px(u, r), 3.8, fill=green)
    u_lip = next(u for u in us if ground_range(u) >= r_lip)
    ja, jb = to_px(u_lip, r_lip), to_px(u_lip, r_lip + w_d)
    s.line(ja[0] - 14, ja[1] + 2, ja[0] - 14, jb[1] + 4, stroke=T["text"], sw=1.6, arrow=True)
    s.lines(ja[0] - 42, (ja[1] + jb[1]) / 2 - 2, ["range", "jump ≈ w"], size=18, lh=21, anchor="end", weight=700)
    u_last = max(u for u in us if r_lip <= ground_range(u) < r_lip + w_d)  # last far-wall row
    s.text(to_px(u_last, 0)[0] - 2, jb[1] - 14, "far wall: flat range", size=17, fill=mag, weight=700, anchor="end")
    legend = [(green, "Ground", "range rises smoothly"),
              (mag, "Ditch", "far wall near lip height"),
              (C[CellState.CREST_SHADOW], "Crest", "ground reappears far below"),
              (grey, "Unseen", "never observed, never free")]
    why_x = ix + 50 + max(measure(name, 18, 700) for _, name, _ in legend) + 12  # shared column
    for k, (col, name, why) in enumerate(legend):
        yy = ly + 282 + k * 28
        s.swatch(ix + 22, yy - 15, col, size=18)
        s.text(ix + 50, yy, name, size=18, weight=700)
        s.text(why_x, yy, why, size=18, fill=T["text_secondary"])
    return s


def slot_models() -> Svg:
    """Slide-3 model strip at slot size 1180x286: terrain model | integrity monitor | MPPI, three compact blocks."""
    W, H = 1180, 286
    s = Svg(W, H)
    bw, gap, x0, y0, bh = 382, 14, 2, 2, 282
    xs = [x0 + k * (bw + gap) for k in range(3)]
    ra, rb, rc = (y0 + 54, 42), (y0 + 122, 72), (y0 + 220, 42)  # (top, height) of the three rows

    def frame(x: float, title: str) -> tuple[float, float]:
        s.card(x, y0, bw, bh, fill=T["surface"], stroke=T["border"])
        s.text(x + 20, y0 + 36, title, size=19, weight=700, fill=T["navy"], ls=0.4)
        ax_, aw = x + 16, bw - 32
        s.rect(ax_, ra[0], aw, ra[1], fill=T["bg"], stroke=T["border"], rx=10)
        s.rect(ax_, rb[0], aw, rb[1], fill=T["bg"], stroke=T["accent"], sw=2, rx=12)
        s.rect(ax_, rc[0], aw, rc[1], fill=T["bg"], stroke=T["border"], rx=10)
        for top, bot in ((ra[0] + ra[1], rb[0]), (rb[0] + rb[1], rc[0])):
            s.line(x + bw / 2, top + 3, x + bw / 2, bot - 3, stroke=T["text_secondary"], sw=2, arrow=True)
        return ax_ + 16, aw

    def row_y(row: tuple[float, float], size: float = 15) -> float:
        return row[0] + row[1] / 2 + size * 0.36

    def core(tx: float, head: str, detail: str) -> None:
        s.text(tx, rb[0] + 30, head, size=18, weight=700)
        s.text(tx, rb[0] + 57, detail, size=15, mono=True, weight=700, fill=T["accent"])

    # ---- terrain model
    tx, aw = frame(xs[0], "TERRAIN MODEL")
    # Trained so far on RUGD only (CPU run; RELLIS-3D + GOOSE need the GPU job) - keep the slide honest.
    src = "RUGD off-road images"
    s.text(tx, row_y(ra), src, size=15)
    ox = tx + measure(src, 15) + 8
    s.text(ox, row_y(ra), "→", size=15, fill=T["text_secondary"])
    s.text(ox + measure("→", 15) + 8, row_y(ra), "OFFROAD5 labels", size=15, weight=700, fill=T["accent"])
    core(tx, "LR-ASPP MobileNetV3-L", f"{FACTS['seg_params']} params")
    classes = [CellState.UNSEEN, CellState.POSITIVE, CellState.WATER, CellState.CREST_SHADOW, CellState.GROUND]
    for k, st in enumerate(classes):
        s.swatch(tx + k * 19, rc[0] + rc[1] / 2 - 7, C[st], size=14, rx=3)
    s.text(tx + 5 * 19 + 8, row_y(rc), "5 classes + entropy", size=15)

    # ---- integrity monitor
    tx, aw = frame(xs[1], "INTEGRITY MONITOR")
    s.text(tx, row_y(ra), f"{FACTS['integrity_features']} image + VO features", size=15)
    core(tx, "Logistic model", "p_fail = σ(w·z + b)")
    q = "q = 1 − p_fail"
    s.text(tx, row_y(rc), q, size=15, mono=True)
    s.text(tx + measure(q, 15, mono=True) + 14, row_y(rc), "→ health mode", size=15)

    # ---- MPPI
    tx, aw = frame(xs[2], "MPPI PLANNER")
    s.text(tx, row_y(ra), f"{FACTS['mppi_k']} rollouts × {FACTS['mppi_span']}", size=15)
    fx0, fy0 = xs[2] + bw - 132, ra[0] + ra[1] / 2  # mini rollout fan
    for k in range(-4, 5):
        s.path(f"M{fx0},{fy0} Q{fx0 + 50},{fy0 + k * 1.5} {fx0 + 98},{fy0 + k * 3.6}", stroke=T["vo_blue"], sw=1.4,
               opacity=0.55)
    s.path(f"M{fx0},{fy0} Q{fx0 + 50},{fy0 + 0.5} {fx0 + 98},{fy0 - 2}", stroke=T["path"], sw=2.6)
    s.circle(fx0 + 98, fy0 - 2, 3.2, fill=T["path"])
    s.rect(fx0 - 16, fy0 - 8, 16, 16, fill=T["surface"], stroke=T["text"], sw=1.5, rx=4)
    core(tx, "Cost incl. certification", "S_k: map · lethal · cost-to-go")
    s.text(tx, row_y(rc), "Softmax weights → wheel rad/s", size=15)
    return s


# =========================================================================== legend + chips
def legend_strip(states: Sequence[CellState], name: str) -> tuple[str, Svg]:
    labels = [ps.CELL_LABELS[st] if st != CellState.POSITIVE else "Lethal (step / rock / drop)" for st in states]
    widths = [34 + 0.9 * text_width(lab, 17) + 30 for lab in labels]  # Inter runs ~10 % narrower than the estimate
    W, H = int(sum(widths) + 40), 64
    s = Svg(W, H)
    x = 20.0
    for st, lab, w in zip(states, labels, widths):
        s.swatch(x, 20, C[st], size=24, rx=6)
        s.text(x + 34, 38, lab, size=17, fill=T["text"])
        x += w
    return name, s


def honesty_chip(label: str) -> Svg:
    size, h, pad = 20, 40, 18
    w = int(text_width(label, size) + 2 * pad + 4)
    s = Svg(w + 8, h + 8, bg=None)
    s.rect(4, 4, w, h, fill=T["bg"], stroke=ps.HONESTY_COLORS[label], sw=2, rx=h / 2)
    s.text(4 + w / 2, 4 + h / 2 + size * 0.36, label, size=size, weight=700, fill=ps.HONESTY_COLORS[label], anchor="middle")
    return s


# =========================================================================== build
def _rel(path: Path) -> str:
    """Repo-relative path for logs (absolute when the output lives elsewhere, e.g. in tests)."""
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def build_svgs(out_dir: Path = OUT_DIR) -> list[tuple[Path, bool]]:
    """Write every SVG; returns (path, transparent_background) pairs."""
    jobs: list[tuple[str, Svg, bool]] = [
        ("how_it_works", diagram_how_it_works(), False),
        ("system_architecture", diagram_architecture(), False),
        ("model_pipeline", diagram_model_pipeline(), False),
        ("missing_ground", diagram_missing_ground(), False),
    ]
    jobs.append((*legend_strip(ps.LEGEND_ORDER, "legend_cell_states"), False))
    core6 = (CellState.GROUND, CellState.UNSEEN, CellState.DITCH_CANDIDATE, CellState.CREST_SHADOW, CellState.POSITIVE,
             CellState.WATER)
    jobs.append((*legend_strip(core6, "legend_cell_states_core6"), False))
    for lab in ps.HONESTY_COLORS:
        jobs.append((f"chips/chip_{lab.lower()}", honesty_chip(lab), True))
    written = []
    for name, svg, transparent in jobs:
        path = out_dir / f"{name}.svg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg.to_string(), encoding="utf-8")
        written.append((path, transparent))
        LOG.info("wrote %s (%dx%d)", _rel(path), svg.w, svg.h)
    return written


def build_slot_svgs(out_dir: Path = SLOT_DIR) -> list[tuple[Path, bool]]:
    """Write the slot-sized diagram SVGs (canvas = slot size in slide px); returns (path, transparent) pairs."""
    jobs = [("s2_how_it_works", slot_how_it_works()), ("s3_missing_ground", slot_missing_ground()),
            ("s3_models", slot_models())]
    written = []
    for name, svg in jobs:
        check_min_text(svg, name)
        path = out_dir / f"{name}.svg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg.to_string(), encoding="utf-8")
        written.append((path, False))
        LOG.info("wrote %s (%dx%d)", _rel(path), svg.w, svg.h)
    return written


def build_slot_charts(out_dir: Path = SLOT_DIR) -> list[Path]:
    """Render the slot-sized theory charts (matplotlib, PNG at 2x + SVG) from ``metagross.eval.theory``."""
    from metagross.eval import theory as th

    params, written = th.TheoryParams(), []
    with ps.deck_style():
        for name, maker in th.SLOT_FIGURES.items():
            written += ps.save_fig(maker(params), name, out_dir=out_dir)
    for path in written:
        LOG.info("wrote %s", _rel(path))
    return written


def render_pngs(svgs: Sequence[tuple[Path, bool]], scale: int = 2) -> list[Path]:
    """Rasterise SVGs with headless Chromium at ``scale`` x (needs local sockets: run from PowerShell)."""
    from playwright.sync_api import sync_playwright

    out: list[Path] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(device_scale_factor=scale)
        for path, transparent in svgs:
            svg_text = path.read_text(encoding="utf-8")
            w = int(svg_text.split('width="', 1)[1].split('"', 1)[0])
            h = int(svg_text.split('height="', 1)[1].split('"', 1)[0])
            page.set_viewport_size({"width": w, "height": h})
            bg = "transparent" if transparent else T["bg"]
            page.set_content(f'<html><body style="margin:0;background:{bg}">{svg_text}</body></html>')
            page.evaluate("document.fonts.ready")
            png = path.with_suffix(".png")
            page.screenshot(path=str(png), clip={"x": 0, "y": 0, "width": w, "height": h}, omit_background=transparent)
            out.append(png)
            LOG.info("rendered %s (%dx%d px)", _rel(png), w * scale, h * scale)
        browser.close()
    return out


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Build METAGROSS deck diagrams (SVG + 2x PNG).")
    ap.add_argument("--svg-only", action="store_true", help="skip the Chromium PNG render")
    ap.add_argument("--scale", type=int, default=2, help="PNG pixels per SVG pixel")
    ap.add_argument("--slots-only", action="store_true", help="only build the slot-sized figures (deck_assets/slots/)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    svgs = [] if args.slots_only else build_svgs()
    svgs += build_slot_svgs()
    build_slot_charts()
    if not args.svg_only:
        render_pngs(svgs, scale=args.scale)


if __name__ == "__main__":
    main()
