"""Worked-example numbers: trench confirmation matcher, post-confirmation window, ditch-entry lists, claim rows."""
from __future__ import annotations

import math

import numpy as np

from metagross.eval import example_runs as ex


def _tick(t: float, cells: list[tuple[int, int]], binding: str = "platform", cmd_w: float = 0.0) -> ex.Tick:
    conf = np.zeros((20, 20), bool)
    for i, j in cells:
        conf[i, j] = True
    return ex.Tick(t=t, confirmed=conf, crop_origin=(0.0, -2.0), res_m=0.2, gov_binding=binding, cmd_w=cmd_w)


def _trench_at_x(x0: float):
    """GT trench = the line x = x0 in the world."""
    return lambda xw, yw: np.abs(np.asarray(xw) - x0)


def test_confirmation_needs_enough_cells_on_the_trench() -> None:
    # trench at world x = 3.0 m; start at the origin facing +x, so A-frame = world
    off = [(10, 2)]  # x = 0.5 m: 2.5 m from the trench, never matched
    near = [(10, 14), (11, 14), (12, 14)]  # x = 2.9 m: on the trench
    ticks = [_tick(0.2, []), _tick(0.4, off), _tick(0.6, off + near[:2]), _tick(0.8, off + near)]
    det = ex.first_confirmation(ticks, (0.0, 0.0), 0.0, _trench_at_x(3.0), tol_m=0.5, min_cells=3)
    assert det is not None
    assert det["t"] == 0.8
    assert det["n_on_trench"] == 3 and det["n_cells"] == 4
    assert np.allclose(det["xw"], 2.9)


def test_confirmation_rotates_the_a_frame_into_the_world() -> None:
    # start at (10, 5) facing +y: A-frame x (forward) maps to world +y
    cells = [(10, j) for j in (14, 15, 16)]  # A-frame x = 2.9-3.3 m, y = 0.1 m
    det = ex.first_confirmation([_tick(1.0, cells)], (10.0, 5.0), math.pi / 2,
                                lambda xw, yw: np.abs(np.asarray(yw) - 8.1), tol_m=0.3, min_cells=3)
    assert det is not None
    assert np.allclose(det["xw"], 9.9)
    assert np.allclose(sorted(det["yw"]), [7.9, 8.1, 8.3])


def test_no_confirmation_returns_none() -> None:
    assert ex.first_confirmation([_tick(0.2, [(1, 1)])], (0.0, 0.0), 0.0, _trench_at_x(50.0)) is None


def test_after_window_counts_binding_and_peak_turn_rate() -> None:
    ticks = [_tick(1.0, [], "platform", 0.1), _tick(1.2, [], "platform", -1.1), _tick(1.4, [], "r_cert", 0.4),
             _tick(5.0, [], "r_cert", 3.0)]
    w = ex.after_window(ticks, 1.0, span_s=3.0)
    assert w["n_ticks"] == 3
    assert w["gov_binding_counts"] == {"platform": 2, "r_cert": 1}
    assert w["max_abs_cmd_w_rad_s"] == 1.1


def test_ditch_entry_seeds_per_config() -> None:
    rows = [{"config_name": "FULL", "seed": "26", "failure_type": "ditch_entry"},
            {"config_name": "FULL", "seed": "14", "failure_type": "ditch_entry"},
            {"config_name": "FULL", "seed": "3", "failure_type": ""},
            {"config_name": "TYPICAL", "seed": "38", "failure_type": "ditch_entry"},
            {"config_name": "TYPICAL", "seed": "5", "failure_type": "out_of_bounds"}]
    assert ex.ditch_entry_seeds(rows) == {"FULL": [14, 26], "TYPICAL": [38]}


def test_ditch_entries_by_family_counts() -> None:
    rows = [{"config_name": "FULL", "seed": "26", "family": "F3_crest_ditch", "failure_type": "ditch_entry"},
            {"config_name": "FULL", "seed": "14", "family": "F3_crest_ditch", "failure_type": "ditch_entry"},
            {"config_name": "FULL", "seed": "3", "family": "F2_ditch_field", "failure_type": "collision"},
            {"config_name": "TYPICAL", "seed": "7", "family": "F2_ditch_field", "failure_type": "ditch_entry"},
            {"config_name": "TYPICAL", "seed": "38", "family": "F3_crest_ditch", "failure_type": "ditch_entry"}]
    assert ex.ditch_entries_by_family(rows) == {"FULL": {"F3_crest_ditch": 2},
                                                "TYPICAL": {"F2_ditch_field": 1, "F3_crest_ditch": 1}}


def test_claim_rows_are_simulated_and_sourced() -> None:
    doc = {"pair": {"seed": 38, "family": "F3_crest_ditch", "outcome": {"FULL": "success", "TYPICAL": "ditch_entry"},
                    "end_time_s": {"FULL": 33.02, "TYPICAL": 22.26},
                    "full_confirmation": {"t_s": 18.2, "n_confirmed_cells": 44, "n_on_gt_trench": 44,
                                          "range_to_nearest_cell_m": 3.17, "gt_speed_min_next3s_mps": 0.665,
                                          "gov_binding_counts": {"platform": 15}, "max_abs_cmd_w_rad_s": 1.2}},
           "ditch_entry_seeds": {"FULL": [14, 26]}, "ditch_entries_by_family": {"FULL": {"F3_crest_ditch": 2}}}
    rows = {r["id"]: r for r in ex.claims(doc)}
    assert rows["example_eval_ditch_entries_FULL_F3_crest_ditch"]["value"] == "2"
    assert rows["example_eval_s038_full_reached_b_s"]["value"] == "33.0"
    assert rows["example_eval_s038_typical_end_s"]["value"] == "22.3"
    assert rows["example_eval_s038_full_trench_confirmed_range_m"]["value"] == "3.2"
    assert rows["example_eval_s038_full_trench_cells_on_gt"]["value"] == "44/44"
    assert rows["example_eval_s038_full_speed_min_after_confirm_mps"]["value"] == "0.7"
    assert rows["example_eval_ditch_entry_seeds_FULL"]["value"] == "14, 26"
    assert all(r["label"] == "Simulated" and r["source"].startswith("results/example_runs_eval.json#") for r in rows.values())


def test_claim_rows_without_confirmation_skip_detection_rows() -> None:
    doc = {"pair": {"seed": 7, "family": "F2_ditch_field", "outcome": {"FULL": "success", "TYPICAL": "ditch_entry"},
                    "end_time_s": {"FULL": 42.7, "TYPICAL": 47.0}}, "ditch_entry_seeds": {}}
    ids = {r["id"] for r in ex.claims(doc)}
    assert ids == {"example_eval_s007_full_reached_b_s", "example_eval_s007_typical_end_s"}
