"""Operator console static checks: colour key identical to the contract, self-contained assets."""

from __future__ import annotations

import json
import re
from pathlib import Path

from metagross.contracts.messages import CELL_COLORS, CellState
from video.style import MODE_HEX

UI = Path(__file__).resolve().parents[1] / "operator_ui"


def test_console_colour_key_matches_contract() -> None:
    js = (UI / "app.js").read_text(encoding="utf-8")
    block = re.search(r"/\*CELL_RGB_BEGIN\*/\s*const CELL_RGB = (\{.*?\});\s*/\*CELL_RGB_END\*/", js, re.S)
    assert block, "CELL_RGB block missing"
    obj = json.loads(re.sub(r"(\w+):", r'"\1":', block.group(1)))
    assert {k: tuple(v) for k, v in obj.items()} == {s.name: CELL_COLORS[s] for s in CellState}


def test_console_is_self_contained() -> None:
    html = (UI / "index.html").read_text(encoding="utf-8")
    for f in ("style.css", "app.js"):
        assert f in html and (UI / f).exists()
    for text in (html, (UI / "style.css").read_text(encoding="utf-8"), (UI / "app.js").read_text(encoding="utf-8")):
        assert not re.search(r"(src|href)=\"https?://", text), "no CDN / remote assets at runtime"
        assert "@import url(http" not in text


def test_console_exposes_the_input_api() -> None:
    js = (UI / "app.js").read_text(encoding="utf-8")
    for name in ("consoleReset", "pushTelemetry", "consoleReplay", "consoleSeek", "onOperatorCommand"):
        assert name in js
    for mode, hexcol in MODE_HEX.items():  # console and video share the DESIGN_TOKENS mode colours
        assert f'{mode}: "{hexcol}"' in js
