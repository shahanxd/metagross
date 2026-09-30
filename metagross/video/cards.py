"""Full-screen 1920 x 1080 cards in the DECK style (white, navy text, one blue accent).

* :func:`title_card`    kicker, title, subtitle (opening / section cards).
* :func:`evidence_card` headline number + label + claim label chip ("Simulated", "Tested", ...)
                        + optional chart PNG on the right + source line.
* :func:`end_card`      closing card with a short list of lines.

Every number shown on an evidence card must come from ``results/`` and carry its claims.csv
label; the card prints that label so the viewer always knows what kind of number it is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from video import style
from video.draw import Painter, text_width

W, H = 1920, 1080
MX = 120  # horizontal margin
EVIDENCE_CHIP_Y = 200  # evidence cards: top of the claim-label chip, px
HEADLINE_PX = 150  # evidence headline number size (shrunk by 1/3 if it does not fit the left column)


def _canvas() -> tuple[np.ndarray, Painter]:
    img = np.empty((H, W, 3), np.uint8)
    img[...] = style.DECK_BG
    return img, Painter(img)


def _frame(p: Painter, kicker: str, footer: str) -> None:
    """Common chrome: wordmark + kicker on top, hairline + footer at the bottom."""
    p.rect(MX, 72, MX + 22, 94, style.DECK_ACCENT, radius=4)
    p.rect(MX + 6, 78, MX + 16, 88, style.DECK_BG, radius=2)
    p.text(MX + 34, 83, "METAGROSS", 20, style.DECK_TEXT, "bold", "lm")
    if kicker:
        p.text(W - MX, 83, kicker, 18, style.DECK_TEXT_2, "medium", "rm")
    p.rect(MX, H - 96, W - MX, H - 95, style.DECK_BORDER)
    if footer:
        p.text(MX, H - 64, footer, 18, style.DECK_TEXT_2, "regular", "lm")


def _wrap(text: str, size: int, weight: str, max_w: float) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if text_width(trial, size, weight) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def claim_chip(p: Painter, x: float, y: float, label: str) -> float:
    """Honesty chip (DESIGN_TOKENS): white fill, 2 px border and bold text in the label colour; top-left at (x, y)."""
    col = style.hex_rgb(style.HONESTY_COLORS.get(label, style.HONESTY_COLORS["Proposed"]))
    w = text_width(label.upper(), 16, "bold") + 32
    p.rect(int(x), int(y), int(x + w), int(y + 34), col, radius=17)
    p.rect(int(x) + 2, int(y) + 2, int(x + w) - 2, int(y + 34) - 2, style.WHITE, radius=15)
    p.text(x + w / 2, y + 18, label.upper(), 16, col, "bold", "mm")
    return x + w


def subtitle_strip(p: Painter, text: str) -> None:
    """Narration subtitle for cards (bottom centre, deck colours)."""
    if text:
        p.text(W / 2, H - 64, text, 24, style.DECK_TEXT, "regular", "mm")


def title_card(title: str, subtitle: str = "", kicker: str = "SIH26126 · Bharat Electronics Ltd",
               footer: str = "Vision-based autonomous navigation · GNSS-denied outdoor UGV", subtitle_text: str = "") -> np.ndarray:
    img, p = _canvas()
    _frame(p, kicker, "" if subtitle_text else footer)
    lines = _wrap(title, 84, "bold", W - 2 * MX)
    y = H / 2 - (len(lines) * 100) / 2 - (40 if subtitle else 0)
    p.rect(MX, int(y) - 40, MX + 64, int(y) - 34, style.DECK_ACCENT)
    for ln in lines:
        p.text(MX, y, ln, 84, style.DECK_TEXT, "bold", "la")
        y += 100
    if subtitle:
        for ln in _wrap(subtitle, 34, "regular", W - 2 * MX - 200):
            p.text(MX, y + 24, ln, 34, style.DECK_TEXT_2, "regular", "la")
            y += 48
    subtitle_strip(p, subtitle_text)
    return p.flush()


def _load_chart(path: Path, box_w: int, box_h: int) -> np.ndarray:
    im = np.asarray(Image.open(path).convert("RGB"))
    h, w = im.shape[:2]
    s = min(box_w / w, box_h / h)
    return cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def evidence_card(headline: str, unit: str, label: str, claim_label: str, chart_png: Optional[str | Path] = None,
                  kicker: str = "Evidence", bullets: Sequence[str] = (), source: str = "", subtitle_text: str = "") -> np.ndarray:
    """Headline number (e.g. '1.8', unit '%') with its label and claim label; chart on the right if given."""
    img, p = _canvas()
    _frame(p, kicker, "" if subtitle_text else (f"Source: {source}" if source else ""))
    left_w = 700 if chart_png else W - 2 * MX
    y = EVIDENCE_CHIP_Y
    claim_chip(p, MX, y, claim_label)
    size = HEADLINE_PX if text_width(headline, HEADLINE_PX, "bold") <= left_w - 120 else HEADLINE_PX * 2 // 3
    y += 34 + 44 + size * 0.73  # chip height + gap + cap height: the baseline clears the chip
    p.text(MX, y, headline, size, style.DECK_ACCENT, "bold", "ls")
    if unit:
        p.text(MX + text_width(headline, size, "bold") + 16, y, unit, 56, style.DECK_ACCENT, "medium", "ls")
    y += 60
    for ln in _wrap(label, 36, "medium", left_w):
        p.text(MX, y, ln, 36, style.DECK_TEXT, "medium", "la")
        y += 50
    y += 20
    for b in bullets:
        for i, ln in enumerate(_wrap(b, 24, "regular", left_w - 30)):
            if i == 0:
                p.rect(MX, int(y + 12), MX + 8, int(y + 20), style.DECK_TEXT_2, radius=2)
            p.text(MX + 26, y, ln, 24, style.DECK_TEXT_2, "regular", "la")
            y += 36
        y += 8
    if chart_png:
        bx0, by0, bx1, by1 = MX + left_w + 80, 180, W - MX, H - 140
        p.rect(bx0, by0, bx1, by1, style.DECK_BORDER, radius=10)
        p.rect(bx0 + 1, by0 + 1, bx1 - 1, by1 - 1, style.DECK_SURFACE, radius=9)
        path = Path(chart_png)
        if path.exists():
            ch = _load_chart(path, bx1 - bx0 - 48, by1 - by0 - 48)
            ox = bx0 + (bx1 - bx0 - ch.shape[1]) // 2
            oy = by0 + (by1 - by0 - ch.shape[0]) // 2
            p.img[oy:oy + ch.shape[0], ox:ox + ch.shape[1]] = ch
        else:
            p.text((bx0 + bx1) / 2, (by0 + by1) / 2, f"chart pending: {path.name}", 22, style.DECK_TEXT_2, "regular", "mm")
    subtitle_strip(p, subtitle_text)
    return p.flush()


IMAGE_CARD_TOP = 150  # image cards: top of the title line, px
IMAGE_CARD_BOTTOM = H - 120  # image cards: bottom of the image box (above the subtitle strip), px


def image_card(path: str | Path, title: str = "", kicker: str = "", subtitle_text: str = "") -> np.ndarray:
    """Deck-style card with an optional title and one diagram / chart fitted below it (aspect kept)."""
    img, p = _canvas()
    _frame(p, kicker, "")
    y = IMAGE_CARD_TOP
    if title:
        p.text(MX, y, title, 44, style.DECK_TEXT, "bold", "la")
        y += 80
    path = Path(path)
    if path.exists():
        ch = _load_chart(path, W - 2 * MX, IMAGE_CARD_BOTTOM - y)
        ox = (W - ch.shape[1]) // 2
        oy = y + (IMAGE_CARD_BOTTOM - y - ch.shape[0]) // 2
        p.img[oy:oy + ch.shape[0], ox:ox + ch.shape[1]] = ch
    else:
        p.text(W / 2, (y + IMAGE_CARD_BOTTOM) / 2, f"image pending: {path.name}", 22, style.DECK_TEXT_2, "regular", "mm")
    subtitle_strip(p, subtitle_text)
    return p.flush()


def end_card(title: str = "Unknown is never free.", lines: Sequence[str] = (), kicker: str = "SIH26126 · Team METAGROSS",
             subtitle_text: str = "") -> np.ndarray:
    img, p = _canvas()
    _frame(p, kicker, "")
    y = 360
    p.rect(MX, y - 40, MX + 64, y - 34, style.DECK_ACCENT)
    for ln in _wrap(title, 76, "bold", W - 2 * MX):
        p.text(MX, y, ln, 76, style.DECK_TEXT, "bold", "la")
        y += 92
    y += 30
    for ln in lines:
        p.text(MX, y, ln, 30, style.DECK_TEXT_2, "regular", "la")
        y += 46
    subtitle_strip(p, subtitle_text)
    return p.flush()
