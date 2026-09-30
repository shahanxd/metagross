"""Raster drawing primitives shared by the compositor and the cards.

Geometry is drawn with OpenCV on an (H, W, 3) uint8 RGB numpy canvas (anti-aliased lines,
fast image blits). Text is queued and composited at the end (:meth:`Painter.flush`) from
cached anti-aliased glyph-run masks rendered once per (text, size, weight, anchor) with real
TrueType fonts through PIL, so a frame never needs a full-frame numpy -> PIL round trip
(a 1080p round trip costs ~20 ms; the dashboard draws ~80 labels per frame, most repeated).

Coordinates are pixels, x right, y down. Colours are RGB tuples.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from video import style

RGB = tuple[int, int, int]
_SHIFT = 4  # sub-pixel bits for cv2 anti-aliased drawing
_SCALE = 1 << _SHIFT
TEXT_CACHE_SIZE = 8192  # cached text masks (a 3-minute video uses a few thousand distinct strings)


@functools.lru_cache(maxsize=TEXT_CACHE_SIZE)
def text_mask(s: str, size: int, weight: str, anchor: str) -> tuple[np.ndarray, int, int]:
    """Anti-aliased coverage mask of ``s`` (float32 in [0, 1], (h, w)) and its offset (dx, dy)
    from the anchor point, in pixels (PIL anchor semantics, e.g. 'la', 'mm', 'rs')."""
    f = style.font(size, weight)
    x0, y0, x1, y1 = f.getbbox(s, anchor=anchor)
    w, h = max(1, int(x1 - x0)), max(1, int(y1 - y0))
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((-x0, -y0), s, fill=255, font=f, anchor=anchor)
    return np.asarray(im, np.float32) / 255.0, int(x0), int(y0)


def draw_text(img: np.ndarray, x: float, y: float, s: str, size: int, color: RGB, weight: str = "regular",
              anchor: str = "la") -> None:
    """Composite text onto ``img`` (in place) at anchor point (x, y); clipped to the canvas."""
    m, dx, dy = text_mask(s, int(size), weight, anchor)
    X0, Y0 = int(round(x)) + dx, int(round(y)) + dy
    H, W = img.shape[:2]
    xa, ya, xb, yb = max(X0, 0), max(Y0, 0), min(X0 + m.shape[1], W), min(Y0 + m.shape[0], H)
    if xa >= xb or ya >= yb:
        return
    a = m[ya - Y0:yb - Y0, xa - X0:xb - X0, None]
    roi = img[ya:yb, xa:xb]
    roi[...] = (roi.astype(np.float32) * (1.0 - a) + np.asarray(color, np.float32) * a + 0.5).astype(np.uint8)


@dataclass
class _TextOp:
    xy: tuple[float, float]
    text: str
    size: int
    weight: str
    color: RGB
    anchor: str


def _fix(pts: np.ndarray) -> np.ndarray:
    """Float pixel coords -> int32 fixed-point for cv2 (shift=_SHIFT)."""
    return np.round(np.asarray(pts, np.float64) * _SCALE).astype(np.int32)


class Painter:
    """Draws on ``img`` in place; text is deferred until :meth:`flush`."""

    def __init__(self, img: np.ndarray) -> None:
        self.img = img
        self._text: list[_TextOp] = []

    # ------------------------------------------------------------------ shapes
    def rect(self, x0: int, y0: int, x1: int, y1: int, color: RGB, radius: int = 0) -> None:
        """Filled (optionally rounded) rectangle, inclusive-exclusive pixel bounds."""
        fill_round_rect(self.img, x0, y0, x1, y1, color, radius)

    def panel(self, x0: int, y0: int, x1: int, y1: int, fill: RGB = style.PANEL, border: RGB = style.BORDER,
              radius: int = 8) -> None:
        """Panel with a 1 px border."""
        fill_round_rect(self.img, x0, y0, x1, y1, border, radius)
        fill_round_rect(self.img, x0 + 1, y0 + 1, x1 - 1, y1 - 1, fill, max(radius - 1, 0))

    def outline(self, x0: int, y0: int, x1: int, y1: int, color: RGB, thickness: int = 1) -> None:
        cv2.rectangle(self.img, (x0, y0), (x1 - 1, y1 - 1), color, thickness, cv2.LINE_8)

    def line(self, pts: np.ndarray | Sequence[Sequence[float]], color: RGB, width: float = 1.0, closed: bool = False) -> None:
        """Anti-aliased polyline through float pixel points (N, 2)."""
        p = np.asarray(pts, np.float64)
        if len(p) < 2:
            return
        cv2.polylines(self.img, [_fix(p).reshape(-1, 1, 2)], closed, color, max(1, int(round(width))), cv2.LINE_AA, _SHIFT)

    def dashed(self, pts: np.ndarray, color: RGB, width: float = 1.0, dash: float = 8.0, gap: float = 6.0) -> None:
        """Anti-aliased dashed polyline (dash pattern continuous along the arc length)."""
        for seg in dash_segments(np.asarray(pts, np.float64), dash, gap):
            self.line(seg, color, width)

    def circle(self, c: tuple[float, float], r: float, color: RGB, width: float = 1.0, fill: bool = False) -> None:
        th = -1 if fill else max(1, int(round(width)))
        cv2.circle(self.img, tuple(int(v) for v in _fix(np.array(c))), int(round(r * _SCALE)), color, th, cv2.LINE_AA, _SHIFT)

    def poly(self, pts: np.ndarray, color: RGB) -> None:
        """Filled anti-aliased polygon."""
        cv2.fillPoly(self.img, [_fix(pts).reshape(-1, 1, 2)], color, cv2.LINE_AA, _SHIFT)

    def blit(self, src: np.ndarray, x: int, y: int, w: int, h: int, interp: int = cv2.INTER_AREA) -> None:
        """Resize ``src`` (RGB or gray) into the box (x, y, w, h)."""
        if src.ndim == 2:
            src = np.repeat(src[..., None], 3, axis=2)
        if src.shape[1] != w or src.shape[0] != h:
            src = cv2.resize(src, (w, h), interpolation=interp)
        self.img[y:y + h, x:x + w] = src[..., :3]

    # ------------------------------------------------------------------ text
    def text(self, x: float, y: float, s: str, size: int, color: RGB = style.TEXT, weight: str = "regular",
             anchor: str = "la") -> None:
        """Queue text. ``anchor`` is a PIL anchor (e.g. 'la' left-ascender, 'mm' middle, 'rs' right-baseline)."""
        if s:
            self._text.append(_TextOp((float(x), float(y)), s, size, weight, color, anchor))

    def flush(self) -> np.ndarray:
        """Composite queued text onto the canvas (in queue order); returns the canvas."""
        for op in self._text:
            draw_text(self.img, op.xy[0], op.xy[1], op.text, op.size, op.color, op.weight, op.anchor)
        self._text.clear()
        return self.img


@functools.lru_cache(maxsize=TEXT_CACHE_SIZE)
def text_width(s: str, size: int, weight: str = "regular") -> float:
    """Advance width of ``s`` in pixels."""
    return float(style.font(size, weight).getlength(s))


def fill_round_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: RGB, radius: int = 0) -> None:
    """Filled rectangle [x0, x1) x [y0, y1) with rounded corners of ``radius`` px."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if x1 <= x0 or y1 <= y0:
        return
    r = int(min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
    if r <= 0:
        img[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = color
        return
    img[y0 + r:y1 - r, x0:x1] = color
    img[y0:y1, x0 + r:x1 - r] = color
    for cx, cy in ((x0 + r, y0 + r), (x1 - r - 1, y0 + r), (x0 + r, y1 - r - 1), (x1 - r - 1, y1 - r - 1)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def blend(dst: np.ndarray, src: np.ndarray, alpha: float | np.ndarray, mask: Optional[np.ndarray] = None) -> None:
    """dst = (1 - a) * dst + a * src, in place; optional boolean ``mask`` restricts the blend.

    A scalar alpha on uint8 images uses OpenCV's saturating SIMD path (rounding within 1 DN of the
    float path); per-pixel alpha maps use float32."""
    a = np.asarray(alpha, np.float32)
    if a.ndim == 0 and dst.dtype == np.uint8 and src.dtype == np.uint8 and dst.shape == src.shape:
        out = cv2.addWeighted(dst, 1.0 - float(a), src, float(a), 0.0)
        if mask is None:
            dst[...] = out
        else:
            np.copyto(dst, out, where=np.asarray(mask, bool)[..., None])
        return
    if a.ndim == 2:
        a = a[..., None]
    out = dst.astype(np.float32) * (1.0 - a) + src.astype(np.float32) * a
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    if mask is None:
        dst[...] = out
    else:
        dst[mask] = out[mask]


def dash_segments(pts: np.ndarray, dash: float, gap: float) -> list[np.ndarray]:
    """Split a polyline into dash sub-polylines (lengths in pixels)."""
    if len(pts) < 2:
        return []
    seg = np.diff(pts, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    s = np.concatenate([[0.0], np.cumsum(L)])
    total = s[-1]
    out: list[np.ndarray] = []
    a = 0.0
    period = dash + gap
    while a < total:
        b = min(a + dash, total)
        ss = np.concatenate([[a], s[(s > a) & (s < b)], [b]])
        xs = np.interp(ss, s, pts[:, 0])
        ys = np.interp(ss, s, pts[:, 1])
        out.append(np.column_stack([xs, ys]))
        a += period
    return out


def arc_points(c: tuple[float, float], r: float, a0: float, a1: float, n: int = 48) -> np.ndarray:
    """Points of a circular arc (angles in radians, image convention: 0 = +x, +pi/2 = +y down)."""
    a = np.linspace(a0, a1, n)
    return np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)])
