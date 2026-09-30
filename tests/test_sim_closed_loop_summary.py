"""closed_loop_summary: aggregation of synthetic run directories (no simulation)."""

from __future__ import annotations

import json

import numpy as np

from metagross.sim.closed_loop_summary import aggregate, collect, effective_failure_type


def _fake_run(root, cfg, seed, family, success, arrive_t=None):
    d = root / cfg / f"{seed:03d}"
    (d / "autonomy").mkdir(parents=True)
    (d / "gt").mkdir()
    res = {"seed": seed, "family": family, "success": success, "failure_type": None if success else "stuck", "time": 20.0,
           "path_length": 20.0, "spl": 0.9 if success else 0.0, "mean_speed": 1.0, "final_error": 1.5 if success else 3.0,
           "ditch_entries": 0, "collisions": 0, "water_entries": 0, "false_stops": 1, "compute_ms_mean": 50.0}
    (d / "result.json").write_text(json.dumps(res), encoding="utf-8")
    (d / "autonomy" / "timings.csv").write_text("tick,t,compute_ms\n0,0.0,40\n1,0.2,60\n", encoding="utf-8")
    tel = [{"t": 0.0, "mode": "NOMINAL"}] + ([{"t": arrive_t, "mode": "ARRIVED"}] if arrive_t is not None else [])
    (d / "autonomy" / "telemetry.jsonl").write_text("\n".join(json.dumps(x) for x in tel) + "\n", encoding="utf-8")
    t = np.arange(0.0, 20.01, 1.0)
    np.savez(d / "gt" / "states.npz", t=t, x=t * 1.0, y=np.zeros_like(t))


def test_aggregate_rates_speeds_and_eval_seed_exclusion(tmp_path):
    _fake_run(tmp_path, "FULL", 102, "F1_trail", True)
    _fake_run(tmp_path, "FULL", 108, "F1_trail", False, arrive_t=10.0)
    _fake_run(tmp_path, "FULL", 5, "F1_trail", True)  # EVAL seed: never aggregated
    rows = collect(tmp_path)
    assert sorted(int(r["seed"]) for r in rows) == [102, 108]
    agg = aggregate(rows)["FULL"]
    a = agg["all"]
    assert a["n"] == 2 and a["success_rate"] == 0.5 and a["arrived_est_rate"] == 0.5
    assert a["compute_ms_p50"] == 50.0 and a["false_stops"] == 2
    assert agg["by_family"]["F1_trail"]["n"] == 2
    r108 = next(r for r in rows if r["seed"] == 108)
    assert r108["drive_speed"] == 1.0  # 10 m of GT path in the 10 s before the estimated arrival


def test_arrived_short_relabel_and_quantiles(tmp_path):
    """'stuck' after an estimated arrival outside the success radius is odometry error, not a stall."""
    _fake_run(tmp_path, "FULL", 102, "F1_trail", True)
    _fake_run(tmp_path, "FULL", 108, "F1_trail", False, arrive_t=10.0)  # stuck, final error 3.0 > 2.0
    _fake_run(tmp_path, "FULL", 114, "F1_trail", False)  # stuck without an arrival: stays 'stuck'
    rows = {int(r["seed"]): r for r in collect(tmp_path)}
    assert rows[108]["failure_type"] == "arrived_short" and rows[108]["failure_type_referee"] == "stuck"
    assert rows[114]["failure_type"] == "stuck"
    a = aggregate(list(rows.values()))["FULL"]["all"]
    assert a["arrived_short"] == 1 and a["n_reach"] == 2 and a["out_of_bounds"] == 0
    assert a["final_error_median_m"] == 3.0 and a["final_error_p90_m"] == 3.0


def test_arrived_from_cmds_npz(tmp_path):
    """The per-command mode log (gt/cmds.npz) is the primary ARRIVED source."""
    _fake_run(tmp_path, "FULL", 108, "F1_trail", False)
    d = tmp_path / "FULL" / "108"
    np.savez(d / "gt" / "cmds.npz", t=np.array([0.0, 0.2, 0.4]), mode=np.array(["NOMINAL", "ARRIVED", "ARRIVED"]))
    (d / "mission.json").write_text(json.dumps({"success_radius_m": 2.5}), encoding="utf-8")
    r = collect(tmp_path)[0]
    assert r["arrived_est"] and r["t_arrived_est"] == 0.2 and r["failure_type"] == "arrived_short"


def test_effective_failure_type_rules():
    assert effective_failure_type("stuck", False, True, 3.0, 2.0) == "arrived_short"
    assert effective_failure_type("stuck", False, True, 1.5, 2.0) == "stuck"  # inside the radius: not odometry
    assert effective_failure_type("collision", False, True, 3.0, 2.0) == "collision"  # hazards keep their label
    assert effective_failure_type(None, True, True, 1.0, 2.0) is None
