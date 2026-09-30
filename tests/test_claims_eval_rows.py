"""Claims ledger rows for the EVAL closed-loop file: overall, per family, failure types, family groups."""
from __future__ import annotations

import json
from pathlib import Path

from metagross.eval import claims


def _block(n: int, ok: int, failures: dict[str, int], **extra) -> dict:
    return {"n": n, "n_success": ok, "success_rate": ok / n, "ditch_entries": extra.get("ditch", 0),
            "collisions": extra.get("coll", 0), "water_entries": 0, "out_of_bounds": extra.get("oob", 0),
            "arrived_short": 0, "final_error_median_m": 1.5, "mean_speed_mps": 1.2,
            "false_stops": extra.get("false_stops", 0), "failure_types": {"None": ok, **failures}}


def _write_eval(results: Path) -> None:
    fams = {"F2_ditch_field": _block(10, 5, {"stuck": 5}),
            "F3_crest_ditch": _block(10, 6, {"ditch_entry": 2, "stuck": 2}, ditch=2, false_stops=3),
            "F4_dynamic": _block(10, 2, {"collision": 8}, coll=8)}
    doc = {"split": "eval", "tier0": {"aggregate": {"FULL": {
        "all": {**_block(30, 13, {"stuck": 7, "ditch_entry": 2, "collision": 8}, ditch=2, coll=8, false_stops=3),
                "seeds": list(range(30))},
        "by_family": fams}}}}
    (results / claims.CLOSED_LOOP_EVAL_FILE).write_text(json.dumps(doc), encoding="utf-8")


def test_family_rows_failure_types_and_groups(tmp_path: Path) -> None:
    _write_eval(tmp_path)
    rows = {r["id"]: r for r in claims.closed_loop_eval_rows(tmp_path)}
    assert rows["closed_loop_eval_tier0_FULL_success"]["value"] == "13"
    assert rows["closed_loop_eval_tier0_FULL_false_stops"]["value"] == "3"
    assert rows["closed_loop_eval_tier0_FULL_fail_stuck"]["value"] == "7"
    assert rows["closed_loop_eval_tier0_FULL_F4_dynamic_success"]["value"] == "2"
    assert rows["closed_loop_eval_tier0_FULL_F4_dynamic_fail_collision"]["value"] == "8"
    assert rows["closed_loop_eval_tier0_FULL_F3_crest_ditch_ditch"]["value"] == "2"
    # the summed ditch + crest group quoted on the deck
    assert rows["closed_loop_eval_tier0_FULL_F2F3_ditch_crest_success"]["value"] == "11"
    assert rows["closed_loop_eval_tier0_FULL_F2F3_ditch_crest_n"]["value"] == "20"
    # successes are not a failure type; family rows skip metrics outside the family subset
    assert "closed_loop_eval_tier0_FULL_fail_None" not in rows
    assert "closed_loop_eval_tier0_FULL_F4_dynamic_mean_speed" not in rows
    assert all(r["label"] == "Simulated" for r in rows.values())
    assert rows["closed_loop_eval_tier0_FULL_F4_dynamic_success"]["source"].endswith(
        "#tier0.aggregate.FULL.by_family.F4_dynamic.n_success")


def test_group_skipped_when_a_member_family_is_missing(tmp_path: Path) -> None:
    _write_eval(tmp_path)
    doc = json.loads((tmp_path / claims.CLOSED_LOOP_EVAL_FILE).read_text(encoding="utf-8"))
    del doc["tier0"]["aggregate"]["FULL"]["by_family"]["F2_ditch_field"]
    (tmp_path / claims.CLOSED_LOOP_EVAL_FILE).write_text(json.dumps(doc), encoding="utf-8")
    ids = {r["id"] for r in claims.closed_loop_eval_rows(tmp_path)}
    assert not any("F2F3" in i for i in ids)
    assert "closed_loop_eval_tier0_FULL_F3_crest_ditch_success" in ids
