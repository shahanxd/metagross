"""docs/RESULTS.md renderer (metagross.eval.results_md): grouping, links, escaping, determinism."""

from __future__ import annotations

import csv
from pathlib import Path

from metagross.eval import results_md as rm

ROWS = [
    {"id": "b_sim", "value": "0.22", "unit": "px", "label": "Simulated", "source": "results/claims.csv", "note": "x"},
    {"id": "a_test", "value": "1.53", "unit": "%", "label": "Tested", "source": "results/claims.csv#t_err", "note": "a|b"},
    {"id": "c_est", "value": "4.30 m", "unit": "", "label": "Estimated", "source": "results/missing.json#k", "note": ""},
    {"id": "d_lit", "value": "1.15", "unit": "%", "label": "Literature", "source": "https://example.org/x", "note": ""},
    {"id": "e_odd", "value": "1", "unit": "", "label": "Guessed", "source": "", "note": "line1\nline2"},
]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rm.COLUMNS))
        w.writeheader()
        w.writerows(rows)
    return path


def test_groups_follow_label_order_and_unknown_last():
    groups = rm.group_by_label(ROWS)
    assert list(groups) == ["Tested", "Simulated", "Estimated", "Literature", rm.OTHER_LABEL]
    assert [r["id"] for r in groups[rm.OTHER_LABEL]] == ["e_odd"]


def test_render_links_escaping_and_sections():
    md = rm.render(ROWS, md_dir=rm.REPO / "docs")
    assert md.index("## Tested") < md.index("## Simulated") < md.index("## Estimated") < md.index("## Literature")
    # existing repo file -> relative link from docs/, JSON path kept in the link text
    assert "[results/claims.csv#t_err](../results/claims.csv)" in md
    # missing file -> plain text, URL -> link
    assert "results/missing.json#k" in md and "(../results/missing.json)" not in md
    assert "[https://example.org/x](https://example.org/x)" in md
    assert "a\\|b" in md  # pipe escaped inside a table cell
    assert "line1 line2" in md  # newline flattened
    assert "| total | 5 |" in md


def test_write_is_deterministic(tmp_path):
    src = _write_csv(tmp_path / "claims.csv", ROWS)
    out1 = rm.write_results_md(src, tmp_path / "docs" / "RESULTS.md").read_bytes()
    out2 = rm.write_results_md(src, tmp_path / "docs" / "RESULTS.md").read_bytes()
    assert out1 == out2 and b"\r\n" not in out1
    assert rm.load_claims(src)[0]["id"] == "b_sim"


def test_empty_ledger(tmp_path):
    src = _write_csv(tmp_path / "claims.csv", [])
    text = rm.write_results_md(src, tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "The ledger is empty." in text and "| total | 0 |" in text
