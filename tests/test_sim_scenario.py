"""Scenario generator: schema, determinism, sha256, solvability and per-family features."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from metagross.contracts.scenario import FAMILIES, MATERIALS, SCHEMA
from metagross.sim.gridplan import CORRIDOR_WIDTH_M, clearance_field, plan_path
from metagross.sim.hazards import gt_hazard_raster
from metagross.sim.scenario import (START_GOAL_DIST_M, canonical_json, check_solvable, f3_has_trench, family_of,
                                    load_scenario, make_scenario, scenario_sha256, write_scenario)
from metagross.sim.terrain import Terrain

TOP_KEYS = {"schema", "seed", "family", "split", "sha256", "world_frame", "terrain", "objects", "hazards", "start", "goal",
            "mission", "lighting", "dynamic", "vehicle"}
# One seed per family + an F3 crest-only control (seed // 6 odd).
SEEDS = {0: "F1_trail", 1: "F2_ditch_field", 2: "F3_crest_ditch", 3: "F4_sudden_obstacle", 4: "F5_lighting",
         5: "F6_water_mud", 8: "F3_crest_ditch"}


@pytest.fixture(scope="module")
def scenarios() -> dict[int, dict]:
    return {s: make_scenario(s) for s in SEEDS}


def _straight_line_hits(scn: dict, mask: np.ndarray) -> bool:
    t = Terrain.from_scenario(scn)
    a, b = np.asarray(scn["start"]["xy"]), np.asarray(scn["goal"]["xy"])
    pts = a[None, :] + np.linspace(0, 1, 2000)[:, None] * (b - a)[None, :]
    i, j = t.grid.xy_to_ij(pts[:, 0], pts[:, 1])
    return bool(mask[i, j].any())


def test_schema_and_family_mapping(scenarios):
    for seed, scn in scenarios.items():
        assert set(scn) == TOP_KEYS
        assert scn["schema"] == SCHEMA and scn["family"] == FAMILIES[seed % 6] == SEEDS[seed] == family_of(seed)
        assert scn["split"] == "eval"
        tb = scn["terrain"]
        assert tb["res_m"] == 0.05 and tb["shape"] == [800, 1280]  # 40 x 64 m
        t = Terrain.from_scenario(scn)
        assert t.height.dtype == np.float32 and set(np.unique(t.material)) <= set(MATERIALS)
        for ob in scn["objects"]:
            assert ob["type"] in {"rock", "tree", "bush", "log"}
        for hz in scn["hazards"]:
            assert hz["type"] in {"ditch", "crest", "water"}
        assert set(scn["mission"]) == {"success_radius_m", "timeout_s", "heading_init_err_deg"}
        assert set(scn["lighting"]) == {"sun_elev_deg", "sun_azim_deg", "fog_density", "exposure", "events"}
        chi = scn["vehicle"]["chi_by_material"]
        assert all(1.3 <= v <= 1.7 for v in chi.values()) and 0.03 <= scn["vehicle"]["slip_long"] <= 0.08
        relief = float(t.height.max() - t.height.min())
        assert 0.25 < relief < 4.0


def test_start_goal_geometry(scenarios):
    for scn in scenarios.values():
        a, b = np.asarray(scn["start"]["xy"]), np.asarray(scn["goal"]["xy"])
        assert START_GOAL_DIST_M[0] - 1e-6 <= np.linalg.norm(b - a) <= START_GOAL_DIST_M[1] + 1e-6
        assert a[0] < 12.0  # A near the west end
        assert abs(scn["mission"]["heading_init_err_deg"]) <= 1.0


def test_determinism_sha_and_roundtrip(tmp_path, scenarios):
    a = scenarios[1]
    b = make_scenario(1)
    assert canonical_json(a) == canonical_json(b)
    assert a["sha256"] == b["sha256"] == scenario_sha256(a)
    assert make_scenario(7)["sha256"] != a["sha256"]
    p = write_scenario(a, tmp_path / "1.json")
    assert load_scenario(p)["sha256"] == a["sha256"]
    doc = json.loads(p.read_text())
    doc["goal"]["xy"][0] += 0.5
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="sha256"):
        load_scenario(p)


def test_every_family_is_solvable_with_corridor(scenarios):
    for seed, scn in scenarios.items():
        ok, length = check_solvable(scn)
        assert ok, f"seed {seed} not solvable"
        straight = math.dist(scn["start"]["xy"], scn["goal"]["xy"])
        assert straight - 0.5 <= length < 2.5 * straight
    # The returned path really keeps >= corridor/2 clearance everywhere (sampled densely).
    scn = scenarios[1]
    hz = gt_hazard_raster(scn)
    clear = clearance_field(hz.grid, hz.hazard)
    res = plan_path(hz.grid, hz.hazard, scn["start"]["xy"], scn["goal"]["xy"], clear=clear)
    for p, q in zip(res.path_xy[:-1], res.path_xy[1:]):
        pts = p[None, :] + np.linspace(0, 1, 50)[:, None] * (q - p)[None, :]
        i, j = hz.grid.xy_to_ij(pts[:, 0], pts[:, 1])
        assert clear[i, j].min() >= 0.5 * CORRIDOR_WIDTH_M - 1e-6


def test_f2_trenches_cross_the_line_with_one_gap(scenarios):
    scn = scenarios[1]
    ditches = [h for h in scn["hazards"] if h["type"] == "ditch"]
    assert 1 <= len(ditches) <= 3
    t = Terrain.from_scenario(scn)
    hz = gt_hazard_raster(scn, t)
    assert _straight_line_hits(scn, hz.ditch)
    for d in ditches:
        assert 0.3 <= d["width"] <= 1.2 and 0.4 <= d["depth"] <= 1.0
        assert len(d["gaps"]) == 1 and 1.5 <= d["gaps"][0][1] - d["gaps"][0][0] <= 3.0
    # Steep but resolved walls: max drop within the ditch reaches ~depth, and slopes exceed 45 deg.
    gx, gy = t.gradient()
    steep = np.degrees(np.arctan(np.hypot(gx, gy)))[hz.ditch]
    assert np.percentile(steep, 99) > 45.0
    assert hz.ditch_drop.max() == pytest.approx(max(d["depth"] for d in ditches), rel=0.05)


def test_f3_trench_and_crest_only_control(scenarios):
    with_trench, control = scenarios[2], scenarios[8]
    assert f3_has_trench(2) and not f3_has_trench(8)
    assert [h["type"] for h in with_trench["hazards"]] == ["crest", "ditch"]
    assert [h["type"] for h in control["hazards"]] == ["crest"]
    hz = gt_hazard_raster(control)
    assert not hz.ditch.any() and not _straight_line_hits(control, hz.slope)  # the crest itself is safe
    assert control["hazards"][0]["drop"] > 0.5


def test_f4_dynamic_obstacle_crosses_route(scenarios):
    scn = scenarios[3]
    (dyn,) = scn["dynamic"]
    assert dyn["type"] in {"walker", "box", "boulder"} and 4.0 <= dyn["trigger_dist_m"] <= 6.0
    a, b = np.asarray(scn["start"]["xy"]), np.asarray(scn["goal"]["xy"])
    n = np.array([-(b - a)[1], (b - a)[0]])
    p0, p1 = (np.asarray(p) for p in dyn["path"])
    s0, s1 = (p0 - a) @ n, (p1 - a) @ n
    assert s0 * s1 <= 0 or abs(s1) / np.linalg.norm(n) < 0.35  # crosses the A->B line or rests on it


def test_f5_lighting_events(scenarios):
    L = scenarios[4]["lighting"]
    assert {e["type"] for e in L["events"]} == {"glare", "dim", "dust"}
    dim = [e for e in L["events"] if e["type"] == "dim"][0]
    assert dim["gain"] == 0.4 and dim["t1"] - dim["t0"] == pytest.approx(3.0)
    assert L["sun_elev_deg"] < 10.0
    a, b = np.asarray(scenarios[4]["start"]["xy"]), np.asarray(scenarios[4]["goal"]["xy"])
    bearing = math.degrees(math.atan2(*(b - a)[::-1]))
    assert abs((L["sun_azim_deg"] - bearing + 180) % 360 - 180) <= 10.0 + 1e-6  # low sun ahead


def test_f6_water_is_flat_and_on_route(scenarios):
    scn = scenarios[5]
    t = Terrain.from_scenario(scn)
    hz = gt_hazard_raster(scn, t)
    assert _straight_line_hits(scn, hz.water)
    from scipy.ndimage import label

    lab, n = label(hz.water)
    for k in range(1, n + 1):
        assert t.height[lab == k].std() < 0.01  # flat: geometrically benign
    assert not (hz.lethal & hz.water).any()
