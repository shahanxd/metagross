"""Deck diagram generator (deck_assets/diagrams/build_diagrams.py): SVG validity and token discipline.

SVG only (no browser), so it runs fast in the sandbox."""

from __future__ import annotations

import importlib.util
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from metagross.eval import plot_style as ps

SCRIPT = Path(__file__).resolve().parents[1] / "deck_assets" / "diagrams" / "build_diagrams.py"


@pytest.fixture(scope="module")
def bd():
    spec = importlib.util.spec_from_file_location("build_diagrams", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["build_diagrams"] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _allowed_colours() -> set[str]:
    allowed = {c.upper() for c in ps.TOKENS.values()}
    allowed |= {c.upper() for c in ps.CELL_HEX.values()}
    allowed |= {c.upper() for c in ps.HONESTY_COLORS.values()}
    allowed |= {c.upper() for c in ps.MODE_COLORS.values()}
    allowed |= {"#DBE6FE"}  # light end of the disparity-strip gradient in the SEE glyph
    return allowed


def test_all_svgs_parse_and_use_only_token_colours(bd, tmp_path: Path) -> None:
    written = bd.build_svgs(tmp_path)
    names = {p.relative_to(tmp_path).as_posix() for p, _ in written}
    assert {"how_it_works.svg", "system_architecture.svg", "model_pipeline.svg", "missing_ground.svg",
            "legend_cell_states.svg", "chips/chip_tested.svg"} <= names
    allowed = _allowed_colours()
    for path, _ in written:
        text = path.read_text(encoding="utf-8")
        root = ET.fromstring(text)  # well-formed XML
        assert root.tag.endswith("svg")
        used = {c.upper() for c in re.findall(r"#[0-9A-Fa-f]{6}\b", text)}
        stray = used - allowed
        assert not stray, f"{path.name} uses non-token colours {stray}"


def test_chips_are_transparent_and_diagrams_white(bd, tmp_path: Path) -> None:
    written = dict((p.relative_to(tmp_path).as_posix(), t) for p, t in bd.build_svgs(tmp_path))
    assert written["chips/chip_estimated.svg"] is True
    assert written["how_it_works.svg"] is False


def test_facts_trace_to_defaults(bd) -> None:
    from metagross.config import defaults as D

    assert bd.FACTS["cam"] == f"{D.IMG_W}×{D.IMG_H}"
    assert bd.FACTS["telemetry_hz"] == f"{D.TELEMETRY_HZ:g} Hz"
    assert bd.FACTS["link_kbps"] == f"{D.LINK_KBPS:g} kbps"


def test_facts_match_owner_modules(bd) -> None:
    mppi = pytest.importorskip("metagross.autonomy.planning.mppi")
    sup = pytest.importorskip("metagross.autonomy.safety.supervisor")
    health = pytest.importorskip("metagross.autonomy.localization.health")
    p = mppi.MppiParams()
    assert bd.FACTS["mppi_k"] == str(p.n_samples)
    assert bd.FACTS["mppi_t"] == f"{p.horizon} × {p.dt:g} s"
    sp = sup.SupervisorParams()
    assert (float(bd.FACTS["q_nominal"]), float(bd.FACTS["q_caution"]), float(bd.FACTS["q_degraded"])) == (
        sp.q_nominal, sp.q_caution, sp.q_degraded)
    assert len(health.FEATURE_NAMES) == 12
