"""Design tokens for the video compositor and cards (``docs/DESIGN_TOKENS.md``).

Two surfaces, one white-based look:

* **Dashboard** (``BG`` / ``PANEL`` / ...): white panels with hairline borders on a very light
  slate page, navy ink, one deep-blue accent. Used by the dashboard compositor.
* **Deck** (``DECK_*``, pure white) for full-screen title / evidence cards, matching the slides.

Neutrals, mode colours and honesty-chip colours mirror ``metagross/eval/plot_style.py``
(``TOKENS``, ``MODE_COLORS``, ``HONESTY_COLORS``); they are copied, not imported, so the video
does not pull in matplotlib, and ``tests/test_video_style.py`` keeps the two in sync.

Semantic colours are *not* defined here: cell states come from
:data:`metagross.contracts.messages.CELL_COLORS` so the console, video and deck share one
colour key. The 4-bit telemetry costmap codes (:mod:`metagross.autonomy.link.codec`) are
mapped onto that key by :func:`u4_palette`.

Fonts are real TrueType files (PIL renders them anti-aliased). Inter is preferred, then
Segoe UI, then Arial; numbers use a monospace face (JetBrains Mono, Cascadia Mono, Consolas).
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import ImageFont

from metagross.contracts.messages import CELL_COLORS, CellState, DriveMode

log = logging.getLogger(__name__)

RGB = tuple[int, int, int]


def hex_rgb(h: str) -> RGB:
    """'#RRGGBB' -> (r, g, b)."""
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ----------------------------------------------------------------------------- tokens (DESIGN_TOKENS.md)
TOKENS: dict[str, str] = {  # mirror of metagross.eval.plot_style.TOKENS
    "bg": "#FFFFFF", "surface": "#F8FAFC", "border": "#E2E8F0", "text": "#0F172A", "text_secondary": "#475569",
    "text_muted": "#94A3B8", "accent": "#1D4ED8", "navy": "#0B1F4B", "vo_blue": "#2563EB", "path": "#0F172A",
    "grid": "#EEF2F6",
}
HONESTY_COLORS: dict[str, str] = {  # mirror of metagross.eval.plot_style.HONESTY_COLORS
    "Tested": "#0F766E", "Simulated": "#1D4ED8", "Estimated": "#B45309", "Proposed": "#6B7280", "Literature": "#7C3AED",
}

# ----------------------------------------------------------------------------- dashboard theme (white)
BG = hex_rgb("#F1F5F9")  # page behind the panels: one step darker than `surface` so white panels read as cards
PANEL = hex_rgb(TOKENS["bg"])
PANEL_2 = hex_rgb(TOKENS["surface"])  # recessed wells inside panels (plots, bars, tiles)
BORDER = hex_rgb(TOKENS["border"])
GRID = hex_rgb(TOKENS["grid"])
TEXT = hex_rgb(TOKENS["text"])
TEXT_2 = hex_rgb(TOKENS["text_secondary"])
TEXT_3 = hex_rgb("#64748B")  # tertiary (ticks, units): one step darker than text_muted to survive H.264 at 13 px
ACCENT = hex_rgb(TOKENS["accent"])
ACCENT_LIGHT = hex_rgb(TOKENS["vo_blue"])  # localisation / VO trail
NAVY = hex_rgb(TOKENS["navy"])
WHITE = (255, 255, 255)
SIMULATED_RGB = hex_rgb(HONESTY_COLORS["Simulated"])

# ----------------------------------------------------------------------------- deck theme
DECK_BG = hex_rgb(TOKENS["bg"])
DECK_TEXT = hex_rgb(TOKENS["text"])
DECK_TEXT_2 = hex_rgb(TOKENS["text_secondary"])
DECK_ACCENT = hex_rgb(TOKENS["accent"])
DECK_BORDER = hex_rgb(TOKENS["border"])
DECK_SURFACE = hex_rgb(TOKENS["surface"])

# ----------------------------------------------------------------------------- drive modes
MODE_HEX: dict[str, str] = {  # mirror of metagross.eval.plot_style.MODE_COLORS (+ the two non-health modes)
    "NOMINAL": "#15803D", "CAUTION": "#CA8A04", "DEGRADED": "#EA580C", "STOP_AND_LOOK": "#DC2626", "SAFE_STOP": "#7F1D1D",
    "ARRIVED": TOKENS["accent"], "HOLD": "#64748B",
}
MODE_COLORS: dict[DriveMode, RGB] = {DriveMode(k): hex_rgb(v) for k, v in MODE_HEX.items()}


def mode_color(mode: DriveMode | str) -> RGB:
    """Colour of a drive mode (unknown strings -> HOLD grey)."""
    try:
        return MODE_COLORS[DriveMode(mode)]
    except ValueError:
        return MODE_COLORS[DriveMode.HOLD]


def mode_text_color(mode: DriveMode | str) -> RGB:
    """Legible text colour on top of a mode chip (white on every token mode colour)."""
    return WHITE


def tint(rgb: RGB, amount: float) -> RGB:
    """Mix ``rgb`` towards white by ``amount`` in [0, 1] (chip / banner backgrounds)."""
    return tuple(int(round(c + (255 - c) * amount)) for c in rgb)  # type: ignore[return-value]


# ----------------------------------------------------------------------------- cell states
CONFIRMED_DITCH_RGB: RGB = CELL_COLORS[CellState.DEPRESSION]  # confirmed ditch is lethal -> red
UNSEEN_RGB: RGB = CELL_COLORS[CellState.UNSEEN]
DITCH_RGB: RGB = CELL_COLORS[CellState.DITCH_CANDIDATE]  # missing ground (image-space mask + BEV)


def cell_palette() -> np.ndarray:
    """(256, 3) uint8 LUT: CellState value -> RGB (unused values -> UNSEEN)."""
    lut = np.tile(np.array(CELL_COLORS[CellState.UNSEEN], np.uint8), (256, 1))
    for st, rgb in CELL_COLORS.items():
        lut[int(st)] = rgb
    return lut


def u4_palette() -> np.ndarray:
    """(16, 3) uint8 LUT for the 4-bit telemetry costmap codes.

    Ground codes 1..8 (cost bins 0..1) shade the GROUND green towards a darker olive so the
    operator can still read cost, while every code keeps its colour-key identity.
    """
    from metagross.autonomy.link import codec  # local import: codec is the owner of the code table

    lut = np.zeros((16, 3), np.uint8)
    lut[codec.U4_UNSEEN] = CELL_COLORS[CellState.UNSEEN]
    g = np.array(CELL_COLORS[CellState.GROUND], np.float32)
    olive = np.array((110, 120, 40), np.float32)  # high-cost ground (still "seen", still green family)
    n = codec.U4_GROUND_MAX - codec.U4_GROUND_MIN
    for i, code in enumerate(range(codec.U4_GROUND_MIN, codec.U4_GROUND_MAX + 1)):
        lut[code] = np.round(g + (olive - g) * (i / n)).astype(np.uint8)
    lut[codec.U4_OCCLUDED] = CELL_COLORS[CellState.OCCLUDED]
    lut[codec.U4_CREST_SHADOW] = CELL_COLORS[CellState.CREST_SHADOW]
    lut[codec.U4_WATER] = CELL_COLORS[CellState.WATER]
    lut[codec.U4_DITCH_CANDIDATE] = CELL_COLORS[CellState.DITCH_CANDIDATE]
    lut[codec.U4_DYNAMIC] = CELL_COLORS[CellState.DYNAMIC]
    lut[codec.U4_DITCH] = CONFIRMED_DITCH_RGB
    lut[codec.U4_LETHAL] = CELL_COLORS[CellState.POSITIVE]
    return lut


LEGEND: tuple[tuple[str, RGB], ...] = (
    ("Seen ground", CELL_COLORS[CellState.GROUND]),
    ("Unseen", CELL_COLORS[CellState.UNSEEN]),
    ("Ditch (missing ground)", CELL_COLORS[CellState.DITCH_CANDIDATE]),
    ("Lethal", CELL_COLORS[CellState.POSITIVE]),
    ("Crest shadow", CELL_COLORS[CellState.CREST_SHADOW]),
    ("Water / mud", CELL_COLORS[CellState.WATER]),
    ("Occluded", CELL_COLORS[CellState.OCCLUDED]),
    ("Dynamic", CELL_COLORS[CellState.DYNAMIC]),
)

# Semantic tint for the camera panel (5-class scheme, interfaces.SEM_CLASSES); None = no tint.
SEM_TINT: dict[int, RGB | None] = {
    0: None,
    1: CELL_COLORS[CellState.POSITIVE],
    2: CELL_COLORS[CellState.WATER],
    3: (150, 170, 70),  # unstable grass/dirt: yellow-green, distinct from certified ground
    4: CELL_COLORS[CellState.GROUND],
}

# ----------------------------------------------------------------------------- fonts
FONT_DIR = Path("C:/Windows/Fonts")
_SANS_REGULAR = ("Inter-Regular.ttf", "Inter-Regular-slnt=0.ttf", "Inter.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")
# DESIGN_TOKENS allows Inter 400 / 700 only: without a Medium cut, 'medium' falls back to Inter Regular (never mix families)
_SANS_MEDIUM = ("Inter-Medium.ttf", "Inter-SemiBold.ttf", "Inter-Regular.ttf", "Inter-Regular-slnt=0.ttf", "seguisb.ttf",
                "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")
_SANS_BOLD = ("Inter-Bold.ttf", "Inter-Bold-slnt=0.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf")
_MONO = ("JetBrainsMono-Regular.ttf", "CascadiaMono.ttf", "consola.ttf", "cour.ttf", "DejaVuSansMono.ttf")
_MONO_BOLD = ("JetBrainsMono-Bold.ttf", "consolab.ttf", "CascadiaMono.ttf", "courbd.ttf", "DejaVuSansMono-Bold.ttf")
_FAMILIES = {"regular": _SANS_REGULAR, "medium": _SANS_MEDIUM, "bold": _SANS_BOLD, "mono": _MONO, "mono_bold": _MONO_BOLD}


def _candidates(names: Iterable[str]) -> Iterable[Path]:
    for n in names:
        yield FONT_DIR / n
        yield Path.home() / "AppData/Local/Microsoft/Windows/Fonts" / n


@functools.lru_cache(maxsize=256)
def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    """TrueType font of ``size`` px. ``weight`` in {'regular','medium','bold','mono','mono_bold'}."""
    for p in _candidates(_FAMILIES[weight]):
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                continue
    for n in _FAMILIES[weight]:  # let FreeType search its own paths (Linux CI)
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    log.warning("no TrueType font found for %s; using PIL default", weight)
    return ImageFont.load_default(size)  # type: ignore[return-value]
