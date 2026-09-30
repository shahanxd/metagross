"""Shared matplotlib style and design tokens for every METAGROSS deck chart.

The deck is assembled in Canva on 1920x1080 slides with a **white** background.
Every chart produced by ``metagross.eval`` should:

1. call :func:`apply_style` once (or use the :func:`deck_style` context manager),
2. create its figure with :func:`new_figure` using one of the named :data:`SIZES`,
3. write it with :func:`save_fig`, which exports PNG (300 dpi, 2x the slide size)
   and SVG to ``deck_assets/<name>.{png,svg}``.

Sizes are specified in **slide pixels** (the size the image occupies on a
1920x1080 slide). The PNG is rendered at :data:`EXPORT_SCALE` times that size so
it stays sharp when projected. Font sizes are chosen so that, at 1x slide size,
body text is ~17 px and titles ~22 px (legible from the back of a room).

Colour tokens mirror ``docs/DESIGN_TOKENS.md``; the semantic cell-state key is
taken from :data:`metagross.contracts.messages.CELL_COLORS` (never duplicated).
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Iterator, Literal, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless, deterministic rendering

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
from matplotlib import font_manager  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from metagross.contracts.messages import CELL_COLORS, CellState  # noqa: E402

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DECK_ASSETS_DIR = REPO_ROOT / "deck_assets"

# --------------------------------------------------------------------------- tokens
#: Neutral + brand colour tokens (hex). Source of truth: docs/DESIGN_TOKENS.md.
TOKENS: dict[str, str] = {
    "bg": "#FFFFFF",  # slide / figure background
    "surface": "#F8FAFC",  # subtle card / panel fill
    "border": "#E2E8F0",  # hairlines, card borders, grid
    "text": "#0F172A",  # primary ink
    "text_secondary": "#475569",  # axis labels, captions
    "text_muted": "#94A3B8",  # tick labels, footnotes
    "accent": "#1D4ED8",  # brand accent (deep blue)
    "navy": "#0B1F4B",  # heading accents
    "vo_blue": "#2563EB",  # localisation / VO
    "path": "#0F172A",  # planned path
    "grid": "#EEF2F6",  # chart grid (lighter than border so it recedes)
}

#: Honesty labels (claims ledger) -> chip colour. Rendered as small outlined pills.
HONESTY_COLORS: dict[str, str] = {
    "Tested": "#0F766E",
    "Simulated": "#1D4ED8",
    "Estimated": "#B45309",
    "Proposed": "#6B7280",
    "Literature": "#7C3AED",
}
HonestyLabel = Literal["Tested", "Simulated", "Estimated", "Proposed", "Literature"]

#: Supervisor drive modes (DriveMode values) -> status colour. Reserved for health state;
#: always shown with the mode name, never colour alone.
MODE_COLORS: dict[str, str] = {
    "NOMINAL": "#15803D",
    "CAUTION": "#CA8A04",
    "DEGRADED": "#EA580C",
    "STOP_AND_LOOK": "#DC2626",
    "SAFE_STOP": "#7F1D1D",
}


def rgb_to_hex(rgb: Sequence[int]) -> str:
    """(r, g, b) in 0..255 -> '#RRGGBB'."""
    r, g, b = (int(c) for c in rgb)
    if not all(0 <= c <= 255 for c in (r, g, b)):
        raise ValueError(f"RGB components must be in 0..255, got {rgb}")
    return f"#{r:02X}{g:02X}{b:02X}"


#: Cell-state key as hex, derived from the frozen contract (single source of truth).
CELL_HEX: dict[CellState, str] = {state: rgb_to_hex(rgb) for state, rgb in CELL_COLORS.items()}

#: Human-readable legend labels for the cell-state key (deck / console wording).
CELL_LABELS: dict[CellState, str] = {
    CellState.GROUND: "Seen ground",
    CellState.UNSEEN: "Unseen",
    CellState.DITCH_CANDIDATE: "Ditch / missing ground",
    CellState.CREST_SHADOW: "Crest shadow",
    CellState.POSITIVE: "Lethal obstacle",
    CellState.DEPRESSION: "Lethal depression",
    CellState.WATER: "Water / mud",
    CellState.OCCLUDED: "Occluded",
    CellState.DYNAMIC: "Dynamic",
}

#: Order in which the key is shown on slides (POSITIVE and DEPRESSION share a colour,
#: so the key shows one "Lethal" swatch).
LEGEND_ORDER: tuple[CellState, ...] = (
    CellState.GROUND,
    CellState.UNSEEN,
    CellState.DITCH_CANDIDATE,
    CellState.CREST_SHADOW,
    CellState.POSITIVE,
    CellState.WATER,
    CellState.OCCLUDED,
    CellState.DYNAMIC,
)

# --------------------------------------------------------------------------- typography
#: Sans families in preference order (Inter is the deck font; Segoe UI / Arial fall back).
SANS_FAMILIES: tuple[str, ...] = ("Inter", "Segoe UI", "Arial", "DejaVu Sans")
#: Monospace families for numbers.
MONO_FAMILIES: tuple[str, ...] = ("JetBrains Mono", "Cascadia Mono", "Consolas", "DejaVu Sans Mono")

FONT_PT = {"body": 8.5, "small": 7.5, "title": 10.5, "annot": 7.5}  # at 300 dpi, 2x -> ~17/15/22 slide px

# --------------------------------------------------------------------------- sizes
#: Named figure sizes in *slide pixels* (1x on a 1920x1080 slide).
SIZES: dict[str, tuple[int, int]] = {
    "card": (900, 520),  # half-width card
    "card_tall": (900, 700),
    "square": (700, 700),
    "wide": (1600, 600),  # full-width band
    "full": (1760, 860),  # nearly full slide
}
EXPORT_SCALE = 2  # PNG pixels per slide pixel
DPI = 300  # PNG dots per inch


def size_inches(size: str | tuple[int, int]) -> tuple[float, float]:
    """Figure size in inches such that a PNG saved at :data:`DPI` is EXPORT_SCALE x slide px."""
    w_px, h_px = SIZES[size] if isinstance(size, str) else size
    return (w_px * EXPORT_SCALE / DPI, h_px * EXPORT_SCALE / DPI)


def _available(families: Sequence[str]) -> list[str]:
    """Filter a family preference list down to the fonts matplotlib can actually find."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    found = [f for f in families if f in names]
    return found or ["DejaVu Sans"]


def sans_family() -> str:
    """First available sans family (Inter where installed)."""
    return _available(SANS_FAMILIES)[0]


def mono_family() -> str:
    """First available monospace family, for numbers."""
    fams = _available(MONO_FAMILIES)
    return fams[0] if fams[0] != "DejaVu Sans" else "DejaVu Sans Mono"


def style_rc() -> dict[str, object]:
    """rcParams implementing the deck chart style (white bg, no top/right spines, light grid)."""
    t = TOKENS
    return {
        "figure.facecolor": t["bg"],
        "axes.facecolor": t["bg"],
        "savefig.facecolor": t["bg"],
        "savefig.edgecolor": t["bg"],
        "font.family": "sans-serif",
        "font.sans-serif": _available(SANS_FAMILIES),
        "font.monospace": _available(MONO_FAMILIES),
        "font.size": FONT_PT["body"],
        "text.color": t["text"],
        "axes.edgecolor": t["border"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": t["text_secondary"],
        "axes.labelsize": FONT_PT["body"],
        "axes.titlesize": FONT_PT["title"],
        "axes.titleweight": "bold",
        "axes.titlecolor": t["text"],
        "axes.titlelocation": "left",
        "axes.titlepad": 10.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": t["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": t["border"],
        "ytick.color": t["border"],
        "xtick.labelcolor": t["text_secondary"],
        "ytick.labelcolor": t["text_secondary"],
        "xtick.labelsize": FONT_PT["small"],
        "ytick.labelsize": FONT_PT["small"],
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 0.0,
        "ytick.minor.size": 0.0,
        "legend.frameon": False,
        "legend.fontsize": FONT_PT["small"],
        "legend.labelcolor": t["text_secondary"],
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "lines.markersize": 5.0,
        "image.cmap": "Blues",
        "svg.fonttype": "none",  # keep text as text in SVG (editable in Figma/Canva)
        "pdf.fonttype": 42,
        "figure.dpi": 100,
        "savefig.dpi": DPI,
    }


def apply_style() -> None:
    """Apply the deck style globally (idempotent)."""
    plt.rcParams.update(style_rc())


@contextlib.contextmanager
def deck_style() -> Iterator[None]:
    """Context manager applying the deck style without leaking rcParams."""
    with plt.rc_context(style_rc()):
        yield


def new_figure(size: str | tuple[int, int] = "card", nrows: int = 1, ncols: int = 1, **kw: object) -> tuple[Figure, object]:
    """``plt.subplots`` at a named deck size, with constrained layout."""
    kw.setdefault("layout", "constrained")
    fig, axes = plt.subplots(nrows, ncols, figsize=size_inches(size), **kw)  # type: ignore[arg-type]
    return fig, axes


def save_fig(
    fig: Figure,
    name: str,
    out_dir: Optional[Path] = None,
    formats: Sequence[str] = ("png", "svg"),
    close: bool = True,
) -> list[Path]:
    """Save ``fig`` as ``<out_dir>/<name>.<fmt>`` for each format; returns written paths.

    ``name`` may contain sub-directories (e.g. ``"theory/ditch_detectability"``).
    PNGs are written at :data:`DPI` (the figure size from :func:`size_inches` makes that
    EXPORT_SCALE x the slide size). SVGs keep text as text.
    """
    base = (out_dir or DECK_ASSETS_DIR) / name
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        # Fixed metadata keeps outputs byte-stable across runs (no timestamps).
        meta = {"Software": None} if fmt == "png" else ({"Date": None} if fmt == "svg" else None)
        fig.savefig(path, format=fmt, dpi=DPI, facecolor=TOKENS["bg"], metadata=meta)
        written.append(path)
        LOG.info("wrote %s", path)
    if close:
        plt.close(fig)
    return written


# --------------------------------------------------------------------------- helpers
def honesty_text(label: HonestyLabel, detail: str = "") -> tuple[str, str]:
    """(chip text, chip colour) for a claims-ledger label; raises on unknown labels."""
    if label not in HONESTY_COLORS:
        raise ValueError(f"unknown honesty label {label!r}; expected one of {sorted(HONESTY_COLORS)}")
    return (label if not detail else f"{label} · {detail}"), HONESTY_COLORS[label]


def add_honesty_chip(fig: Figure, label: HonestyLabel, detail: str = "", loc: tuple[float, float] = (0.992, 0.985)) -> None:
    """Draw a small outlined honesty pill (e.g. 'Estimated · analytic model') on a figure.

    ``loc`` is the chip's top-right corner in figure-fraction coordinates; the default puts it
    in the top-right corner, level with the left-aligned axes title.
    """
    text, color = honesty_text(label, detail)
    fig.text(
        loc[0],
        loc[1],
        text,
        ha="right",
        va="top",
        fontsize=FONT_PT["small"],
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.8", fc=TOKENS["bg"], ec=color, lw=0.9),
        zorder=10,
    )


def add_footnote(fig: Figure, text: str) -> None:
    """Muted, left-aligned footnote under the axes (formula / assumptions). Uses ``supxlabel`` so
    constrained layout reserves space for it and it never collides with the x-axis label."""
    fig.supxlabel(text, x=0.01, ha="left", fontsize=FONT_PT["annot"], color=TOKENS["text_muted"])


def style_axes(ax: Axes, grid: Literal["both", "x", "y", "none"] = "both") -> None:
    """Final per-axes touches: recessive spines and the chosen grid direction."""
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TOKENS["border"])
    if grid == "none":
        ax.grid(False)
    else:
        ax.grid(True, axis=grid if grid != "both" else "both", which="major")


def sequential_cmap(name: str = "metagross_blue", hex_dark: str = TOKENS["accent"]) -> LinearSegmentedColormap:
    """Single-hue sequential colour map: near-white surface -> brand blue -> navy."""
    return LinearSegmentedColormap.from_list(name, ["#EFF4FF", "#93B4F5", hex_dark, TOKENS["navy"]])


def cell_state_cmap() -> ListedColormap:
    """Colour map indexed by ``CellState`` value (0..8) using the contract colours."""
    n = max(int(s) for s in CellState) + 1
    colors = [CELL_HEX.get(CellState(i), TOKENS["border"]) for i in range(n)]
    return ListedColormap(colors, name="cell_state")


def pill(ax: Axes, xy: tuple[float, float], w: float, h: float, color: str, fill: str = "none", lw: float = 1.0) -> FancyBboxPatch:
    """Rounded rectangle helper (data coords) used by legend/chip renderers."""
    patch = FancyBboxPatch(
        xy, w, h, boxstyle=f"round,pad=0,rounding_size={h / 2.0}", fc=fill, ec=color, lw=lw, mutation_aspect=1.0
    )
    ax.add_patch(patch)
    return patch
