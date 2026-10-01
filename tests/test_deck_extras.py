"""Derived deck numbers: packet period, distance to B, KITTI distance sum, envelope row."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from metagross.eval import deck_extras as dx


def test_median_period_ignores_order_and_uses_gaps() -> None:
    assert dx.median_period([1.2, 0.0, 0.6, 1.8, 3.0]) == pytest.approx(0.6)


def test_median_period_needs_two_stamps() -> None:
    with pytest.raises(ValueError):
        dx.median_period([0.0])


def test_dist_to_goal() -> None:
    assert dx.dist_to_goal([29.55, -5.3, -0.04], [54.1997, -2.1427]) == pytest.approx(math.hypot(24.6497, 3.1573))


def test_kitti_rows_sum_and_units(tmp_path: Path) -> None:
    for s, m in (("00", 2913.5), ("05", 2205.6), ("07", 694.7)):
        (tmp_path / f"kitti_vo_{s}.json").write_text(json.dumps({"path_length_m": m}))
    rows = {r["id"]: r for r in dx.kitti_rows(tmp_path)}
    assert rows["kitti_pooled_path_km"]["value"] == "5.81"
    assert rows["kitti00_path_length"]["value"] == "2.91" and rows["kitti00_path_length"]["unit"] == "km"
    assert rows["kitti07_path_length"]["value"] == "695" and rows["kitti07_path_length"]["unit"] == "m"
    assert all(r["label"] == "Tested" for r in rows.values())


def test_envelope_row_is_estimated(tmp_path: Path) -> None:
    (tmp_path / "theory.json").write_text(json.dumps({"envelope": {"v_min_mps": 1.607, "h_range_m": [0.4, 1.6],
                                                                   "w_range_m": [0.2, 1.2]}}))
    (row,) = dx.envelope_rows(tmp_path)
    assert row["value"] == "1.61" and row["label"] == "Estimated"


def test_eval_runtime_and_console_rows(tmp_path: Path) -> None:
    for seed in (0, 44):
        run = tmp_path / f"{seed:03d}" / "autonomy"
        run.mkdir(parents=True)
        (run / "timings.csv").write_text("tick,t,mppi\n0,0.0,5.0\n1,0.2,6.0\n2,0.4,5.5\n")
        lines = [{"t": round(0.6 * k, 1), "seq": k, "pose": [float(k), 0.0, 0.0]} for k in range(4)]
        (run / "telemetry.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
        (run.parent / "mission.json").write_text(json.dumps({"goal_xy_a": [10.0, 0.0]}))
    rows = {r["id"]: r for r in dx.eval_runtime_rows(tmp_path)}
    assert rows["closed_loop_eval_tier0_FULL_mppi_p50_ms"]["value"] == "5.5"
    assert rows["closed_loop_eval_tier0_FULL_telemetry_period_s"]["value"] == "0.6"
    (row,) = dx.console_rows(tmp_path, seed=44, seq=3)
    assert row["value"] == "7.0" and row["id"] == "console_eval_s044_seq3_dist_to_b_m"
