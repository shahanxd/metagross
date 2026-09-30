"""Unit tests: rolling map, costmap, global planner, governor, MPPI, mixer, supervisor."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from metagross.autonomy.control.mixer import SkidSteerMixer, body_to_wheels, saturate_wheels, wheels_to_body
from metagross.autonomy.planning.costmap import CostmapParams, build_costmap
from metagross.autonomy.planning.global_planner import GlobalPlanner, resample_path
from metagross.autonomy.planning.governor import (
    SpeedGovernor,
    ditch_detection_range,
    stopping_distance,
    v_from_range,
)
from metagross.autonomy.planning.mppi import MppiParams, MppiPlanner, rollout
from metagross.autonomy.planning.rolling_map import FRESH_S, BevGeometry, RollingMap
from metagross.autonomy.safety.supervisor import Supervisor, SupervisorDecision
from metagross.config.defaults import (
    BRAKE_DECEL_MPS2,
    CAM_HEIGHT_M,
    CHI_NOMINAL,
    DESIGN_DITCH_WIDTH_M,
    FX,
    GOVERNOR_MARGIN_M,
    MIN_PIXELS_ON_TARGET,
    VEHICLE,
)
from metagross.contracts.messages import CellState as S
from metagross.contracts.messages import DriveMode, OperatorAction, OperatorCmd

GEO = BevGeometry()


def bev(fill=S.UNSEEN) -> np.ndarray:
    return np.full((GEO.n_fwd, GEO.n_lat), int(fill), np.uint8)


def bev_patch(states: np.ndarray, x0, x1, y0, y1, st) -> np.ndarray:
    xb, yb = GEO.cell_centres_body()
    states[(xb >= x0) & (xb <= x1) & (yb >= y0) & (yb <= y1)] = int(st)
    return states


def map_state_at(rm: RollingMap, x, y) -> int:
    ix, iy = rm.geometry.to_index(np.array([x]), np.array([y]))
    return int(rm.state[iy[0], ix[0]])


# ------------------------------------------------------------------ rolling map
def test_bev_geometry_matches_perception_convention():
    xb, yb = GEO.cell_centres_body()
    assert xb[0, 0] < xb[1, 0]  # rows grow forward
    assert yb[0, 0] < yb[0, 1]  # columns grow to the left (+y)
    assert math.isclose(xb[0, 0], GEO.x_min_m + GEO.res_m / 2)


def test_ground_certified_and_stale_only_when_nominal():
    rm = RollingMap()
    rm.integrate(0.0, (0, 0, 0), bev_patch(bev(), 2, 4, -1, 1, S.GROUND))
    assert map_state_at(rm, 3.0, 0.0) == S.GROUND
    ix, iy = rm.geometry.to_index(np.array([3.0]), np.array([0.0]))
    assert rm.certified_mask(1.0, health_nominal=False)[iy[0], ix[0]]
    later = FRESH_S + 1.0
    assert rm.certified_mask(later, health_nominal=True)[iy[0], ix[0]]
    assert not rm.certified_mask(later, health_nominal=False)[iy[0], ix[0]]


def test_ditch_candidate_needs_two_of_three_and_persists_out_of_view():
    rm = RollingMap()
    obs = bev_patch(bev_patch(bev(), 2, 6, -1, 1, S.GROUND), 4.0, 4.4, -1, 1, S.DITCH_CANDIDATE)
    rm.integrate(0.0, (0, 0, 0), obs)
    static, _ = rm.lethal_mask()
    assert not static.any(), "a single candidate frame must not be lethal"
    rm.integrate(0.2, (0, 0, 0), obs)
    static, _ = rm.lethal_mask()
    assert static.any(), "2 of 3 frames confirm the ditch"
    rm.integrate(0.4, (0, 0, 0), bev())  # lip leaves the FOV: nothing observed
    static2, _ = rm.lethal_mask()
    assert np.array_equal(static, static2)
    # one GROUND re-observation is not enough to clear lethal memory; two are
    ground = bev_patch(bev(), 2, 6, -1, 1, S.GROUND)
    rm.integrate(0.6, (0, 0, 0), ground)
    assert rm.lethal_mask()[0].any()
    rm.integrate(0.8, (0, 0, 0), ground)
    assert not rm.lethal_mask()[0].any()


def test_negobs_off_candidates_not_lethal():
    rm = RollingMap()
    obs = bev_patch(bev(), 4.0, 4.4, -1, 1, S.DITCH_CANDIDATE)
    for k in range(3):
        rm.integrate(0.2 * k, (0, 0, 0), obs)
    assert rm.lethal_mask(use_negobs=True)[0].any()
    assert not rm.lethal_mask(use_negobs=False)[0].any()


def test_new_occupancy_is_static_when_inference_off(monkeypatch):
    import metagross.autonomy.planning.rolling_map as rmod

    monkeypatch.setattr(rmod, "INFER_DYNAMIC", False)
    rm = RollingMap()
    ground = bev_patch(bev(), 2, 6, -2, 2, S.GROUND)
    rm.integrate(0.0, (0, 0, 0), ground)
    rm.integrate(0.2, (0, 0, 0), ground.copy())
    occ = bev_patch(bev_patch(bev(), 2, 6, -2, 2, S.GROUND), 4, 4.5, 0, 0.5, S.POSITIVE)
    rm.integrate(0.4, (0, 0, 0), occ)
    rm.integrate(0.6, (0, 0, 0), occ)
    assert map_state_at(rm, 4.2, 0.2) == S.POSITIVE


def test_dynamic_on_recent_ground_and_occlusion_keeps_ground():
    rm = RollingMap()  # INFER_DYNAMIC on (default)
    ground = bev_patch(bev(), 2, 6, -2, 2, S.GROUND)
    rm.integrate(0.0, (0, 0, 0), ground)
    rm.integrate(0.2, (0, 0, 0), ground.copy())  # well observed free (2 frames)
    occ = bev_patch(bev_patch(bev(), 2, 6, -2, 2, S.GROUND), 4, 4.5, 0, 0.5, S.POSITIVE)
    rm.integrate(0.4, (0, 0, 0), occ)
    assert map_state_at(rm, 4.2, 0.2) == S.GROUND and rm.lethal_mask()[1].sum() == 0  # 1 frame: pending only
    ix, iy = rm.geometry.to_index(np.array([4.2]), np.array([0.2]))
    assert rm.pending[iy[0], ix[0]] and not rm.certified_mask(0.4, True)[iy[0], ix[0]]
    rm.integrate(0.6, (0, 0, 0), occ)
    assert map_state_at(rm, 4.2, 0.2) == S.DYNAMIC
    occl = bev_patch(bev(), 2, 6, -2, 2, S.OCCLUDED)
    rm.integrate(0.4, (0, 0, 0), occl)
    assert map_state_at(rm, 3.0, -1.0) == S.GROUND
    assert map_state_at(rm, 4.2, 0.2) == S.DYNAMIC  # lethal memory not erased by occlusion


def test_pose_transform_and_recentre():
    rm = RollingMap()
    obs = bev_patch(bev(), 3.0, 3.2, 1.0, 1.2, S.POSITIVE)
    rm.integrate(0.0, (5.0, 2.0, math.pi / 2), obs)  # facing +y: body (3.1, 1.1) -> A (3.9, 5.1)
    rm.integrate(0.1, (5.0, 2.0, math.pi / 2), obs)  # lethal needs support from 2 frames
    assert map_state_at(rm, 5.0 - 1.1, 2.0 + 3.1) == S.POSITIVE
    rm.integrate(0.2, (25.0, 0.0, 0.0), bev())
    assert rm.geometry.origin_x > -40.0  # scrolled
    assert map_state_at(rm, 5.0 - 1.1, 2.0 + 3.1) == S.POSITIVE  # memory survives the scroll


# ------------------------------------------------------------------ costmap + global planner
def _open_map(apron=25.0) -> RollingMap:
    rm = RollingMap()
    rm.seed_apron((0, 0, 0), 0.0, apron)
    return rm


def _put(rm: RollingMap, x0, x1, y0, y1, st) -> None:
    g = rm.geometry
    ix, iy = np.meshgrid(np.arange(g.n), np.arange(g.n))
    xc, yc = g.to_xy(ix, iy)
    rm.state[(xc >= x0) & (xc <= x1) & (yc >= y0) & (yc <= y1)] = int(st)


def test_costmap_inflation_profile_and_unknown_is_free():
    rm = _open_map()
    _put(rm, 4.95, 5.25, -0.15, 0.15, S.POSITIVE)  # cells centred at x=5.1, y=+-0.1
    _put(rm, -10, 10, 8, 10, S.UNSEEN)
    maps = build_costmap(rm, 0.0, True, CostmapParams())
    c = lambda x, y: float(maps.sample(maps.cost, np.array([x]), np.array([y]), 0.0)[0])  # noqa: E731
    assert c(5.1, 0.0) == 1.0
    assert c(5.1, 0.45) == 1.0  # within lethal radius of the edge
    assert 0.0 < c(5.1, 0.8) < 1.0  # decay band
    assert c(5.1, 1.6) == 0.0
    assert c(0.0, 9.0) > 0.0 and not maps.sample(maps.certified, np.array([0.0]), np.array([9.0]), False)[0]
    free = build_costmap(rm, 0.0, True, CostmapParams(unknown_is_free=True))
    assert float(free.sample(free.cost, np.array([0.0]), np.array([9.0]), 1.0)[0]) == 0.0
    assert free.sample(free.certified, np.array([0.0]), np.array([9.0]), False)[0]


def test_global_planner_routes_around_wall_and_replans_on_cut():
    rm = _open_map()
    _put(rm, 6.0, 6.4, -3.0, 3.0, S.POSITIVE)
    maps = build_costmap(rm, 0.0, True, CostmapParams())
    gp = GlobalPlanner()
    plan = gp.update(0.0, maps, (0.0, 0.0), (12.0, 0.0))
    assert plan.route_ok
    near_wall = plan.path_xy[np.abs(plan.path_xy[:, 0] - 6.2) < 0.5]
    assert np.all(np.abs(near_wall[:, 1]) > 3.0)
    assert plan.ctg_at(np.array([12.0]), np.array([0.0]))[0] < 1.0
    n = gp.n_recomputes
    _put(rm, -10.0, 20.0, 3.0, 3.4, S.POSITIVE)  # cut the detour on one side
    _put(rm, -10.0, 20.0, -3.4, -3.0, S.POSITIVE)
    maps = build_costmap(rm, 0.1, True, CostmapParams())
    plan = gp.update(0.1, maps, (0.0, 0.0), (12.0, 0.0))  # 0.1 s later: only the cut can trigger it
    assert gp.n_recomputes == n + 1
    assert not plan.route_ok or np.all(np.isfinite(plan.path_xy))
    wps = resample_path(np.array([[0.0, 0.0], [10.0, 0.0]]), 2.0, 5)
    assert wps.shape == (5, 2) and np.allclose(wps[:, 0], [2, 4, 6, 8, 10])


# ------------------------------------------------------------------ governor
def test_governor_closed_form_and_monotone():
    a, tr, b = 1.5, 0.5, 0.5
    for r in (0.6, 1.0, 3.0, 8.0):
        v = v_from_range(r, a, tr, b)
        assert math.isclose(stopping_distance(v, a, tr, b), r, rel_tol=1e-9)
    assert v_from_range(b, a, tr, b) == 0.0 and v_from_range(0.1, a, tr, b) == 0.0
    vs = [v_from_range(r, a, tr, b) for r in np.linspace(0, 12, 50)]
    assert all(np.diff(vs) >= 0)
    r_det = ditch_detection_range(CAM_HEIGHT_M, DESIGN_DITCH_WIDTH_M, FX, MIN_PIXELS_ON_TARGET)
    theta = CAM_HEIGHT_M * DESIGN_DITCH_WIDTH_M / (r_det * (r_det + DESIGN_DITCH_WIDTH_M))
    assert math.isclose(theta * FX, MIN_PIXELS_ON_TARGET, rel_tol=1e-9)


def test_governor_binding_terms():
    rm = _open_map(apron=3.0)
    maps = build_costmap(rm, 0.0, True, CostmapParams())
    gov = SpeedGovernor()
    res = gov.compute(maps, (0.0, 0.0, 0.0), np.zeros((0, 2)), r_vis_m=10.0, q_health=1.0, latency_s=0.05, frame_period_s=0.2)
    # certified to ~3 m -> R_cert ~ 2.6 m from the bumper
    assert res.binding == "r_cert" and 2.0 < res.r_cert_m < 3.0
    assert math.isclose(res.v_cap_mps, v_from_range(res.r_cert_m, BRAKE_DECEL_MPS2, res.t_r_s, GOVERNOR_MARGIN_M))
    low = gov.compute(maps, (0.0, 0.0, 0.0), np.zeros((0, 2)), 10.0, 0.35, 0.05, 0.2)
    assert low.v_cap_mps < res.v_cap_mps and low.health_factor == pytest.approx(0.5)
    rm2 = _open_map(apron=30.0)
    maps2 = build_costmap(rm2, 0.0, True, CostmapParams())
    open_res = gov.compute(maps2, (0.0, 0.0, 0.0), np.zeros((0, 2)), 10.0, 1.0, 0.05, 0.2)
    assert open_res.binding in ("platform", "ditch_det") and open_res.v_cap_mps == pytest.approx(min(open_res.terms.values()))
    off = gov.compute(maps, (0.0, 0.0, 0.0), np.zeros((0, 2)), 10.0, 1.0, 0.05, 0.2, enabled=False)
    assert off.v_cap_mps == VEHICLE.max_speed_mps and off.reason == "GOV_OFF"


# ------------------------------------------------------------------ MPPI
def _ctg_to(goal):
    return lambda x, y: np.hypot(x - goal[0], y - goal[1])


def test_mppi_avoids_lethal_blob_and_reaches_goal():
    rm = _open_map(apron=30.0)
    _put(rm, 3.5, 4.5, -0.5, 0.5, S.POSITIVE)
    maps = build_costmap(rm, 0.0, True, CostmapParams())
    goal = (8.0, 0.0)
    mp = MppiPlanner(MppiParams(), seed=1)
    x, y, yaw, v, w = 0.0, 0.0, 0.0, 0.0, 0.0
    min_clear = np.inf
    ms = []
    for k in range(120):
        res = mp.plan(0.1 * k, (x, y, yaw), v, w, 1.5, maps, _ctg_to(goal), t_r=0.5)
        ms.append(res.compute_ms)
        v, w = res.u0
        yaw += w * 0.1
        x += v * math.cos(yaw) * 0.1
        y += v * math.sin(yaw) * 0.1
        for off in (-0.27, 0.0, 0.27):
            cx, cy = x + off * math.cos(yaw), y + off * math.sin(yaw)
            min_clear = min(min_clear, float(maps.sample(maps.clearance_m, np.array([cx]), np.array([cy]), 1e3)[0]))
        if math.hypot(x - goal[0], y - goal[1]) < 0.5:
            break
    assert math.hypot(x - goal[0], y - goal[1]) < 0.5, (x, y)
    assert min_clear > 0.33, f"footprint circle entered lethal core (clearance {min_clear:.2f} m)"
    assert np.median(ms) < 100.0


def test_mppi_respects_limits_and_is_deterministic():
    rm = _open_map(apron=30.0)
    maps = build_costmap(rm, 0.0, True, CostmapParams())
    r1 = MppiPlanner(seed=3).plan(0.0, (0, 0, 0), 0.0, 0.0, 0.7, maps, _ctg_to((10, 5)), 0.5)
    r2 = MppiPlanner(seed=3).plan(0.0, (0, 0, 0), 0.0, 0.0, 0.7, maps, _ctg_to((10, 5)), 0.5)
    assert np.array_equal(r1.U, r2.U)
    assert np.all(r1.U[:, 0] <= 0.7 + 1e-9) and np.all(r1.U[:, 0] >= 0.0)
    assert np.all(np.abs(r1.U[:, 1]) <= VEHICLE.max_yaw_rate_rps + 1e-9)
    X, Y, _ = rollout((0.0, 0.0, 0.0), np.full(10, 1.0), np.zeros(10), 0.1)
    assert math.isclose(X[-1], 1.0) and math.isclose(Y[-1], 0.0, abs_tol=1e-12)


def test_mppi_certification_term_slows_toward_unseen():
    rm = _open_map(apron=2.5)  # everything beyond 2.5 m unseen
    maps = build_costmap(rm, 0.0, True, CostmapParams())
    fast = MppiPlanner(seed=0)
    slow = MppiPlanner(seed=0)
    for k in range(15):
        a = fast.plan(0.1 * k, (0, 0, 0), 1.0, 0.0, 2.0, maps, _ctg_to((10, 0)), 0.5, cert_enabled=False)
        b = slow.plan(0.1 * k, (0, 0, 0), 1.0, 0.0, 2.0, maps, _ctg_to((10, 0)), 0.5, cert_enabled=True)
    assert b.U[:10, 0].mean() < a.U[:10, 0].mean()


# ------------------------------------------------------------------ mixer
def test_mixer_round_trip_and_curvature_preserving_saturation():
    for v, w in [(1.0, 0.0), (0.5, 0.8), (-0.3, -0.5), (0.0, 1.0)]:
        wl, wr = body_to_wheels(v, w, CHI_NOMINAL, VEHICLE.track_width_m, VEHICLE.wheel_radius_m)
        v2, w2 = wheels_to_body(wl, wr, CHI_NOMINAL, VEHICLE.track_width_m, VEHICLE.wheel_radius_m)
        assert math.isclose(v, v2, abs_tol=1e-12) and math.isclose(w, w2, abs_tol=1e-12)
    wl, wr, k = saturate_wheels(30.0, 10.0, 20.0)
    assert k < 1 and max(abs(wl), abs(wr)) == pytest.approx(20.0) and wl / wr == pytest.approx(3.0)


def test_mixer_accel_jerk_limits_and_emergency():
    m = SkidSteerMixer()
    m.mix(0.0, 0.0, 0.0)
    vs = []
    for k in range(1, 40):
        m.mix(0.1 * k, 2.0, 0.0)
        vs.append(m.v)
    dv = np.diff([0.0] + vs) / 0.1
    assert dv.max() <= VEHICLE.max_accel_mps2 + 1e-9
    assert np.all(np.diff(dv) <= 4.0 * 0.1 + 1e-9)  # jerk limit
    v_before = m.v
    m.mix(4.0, 0.0, 0.0, emergency=True)
    assert v_before - m.v == pytest.approx(min(v_before, BRAKE_DECEL_MPS2 * 0.1))
    wl, wr = m.mix(4.1, 5.0, 3.0)
    assert max(abs(wl), abs(wr)) <= VEHICLE.max_wheel_rad_s + 1e-9


# ------------------------------------------------------------------ supervisor
def _sup(goal=(20.0, 0.0), dead_end_enabled: bool = True) -> Supervisor:
    s = Supervisor(dead_end_enabled=dead_end_enabled)
    s.reset(goal, 2.0)
    return s


def test_supervisor_health_modes_with_hysteresis_and_safe_stop():
    s = _sup()
    pose = (0.0, 0.0, 0.0)
    assert s.update(0.0, 0.9, pose, 0.0, 0.0, progress_metric=20.0).mode == DriveMode.NOMINAL
    d = s.update(0.1, 0.6, pose, 0.0, 0.0, progress_metric=19.0)
    assert d.mode == DriveMode.CAUTION and d.speed_factor == 0.5 and d.inflate_scale > 1
    assert s.update(0.2, 0.72, pose, 0.0, 0.0, progress_metric=18.0).mode == DriveMode.CAUTION  # inside hysteresis
    s.update(0.3, 0.8, pose, 0.0, 0.0, progress_metric=17.0)
    assert s.update(1.4, 0.8, pose, 0.0, 0.0, progress_metric=16.0).mode == DriveMode.NOMINAL  # held 1 s
    assert s.update(1.5, 0.3, pose, 0.0, 0.0, progress_metric=15.0).mode == DriveMode.DEGRADED
    s.update(1.6, 0.1, pose, 0.0, 0.0, progress_metric=14.0)
    d = s.update(4.7, 0.1, pose, 0.0, 0.0, progress_metric=13.0)
    assert d.mode == DriveMode.SAFE_STOP and d.reason.startswith("HEALTH")
    assert s.update(5.0, 0.9, pose, 0.0, 0.0, progress_metric=12.0).mode == DriveMode.SAFE_STOP  # latched
    s.operator(OperatorCmd(5.1, OperatorAction.RESUME))
    assert s.update(5.2, 0.9, pose, 0.0, 0.0, progress_metric=11.0).mode == DriveMode.DEGRADED  # climbs back
    modes = [s.update(5.2 + 0.1 * k, 0.9, pose, 0.0, 0.0, progress_metric=10.0 - 0.4 * k).mode for k in range(1, 25)]
    assert modes[-1] == DriveMode.NOMINAL and DriveMode.CAUTION in modes


def test_supervisor_stop_and_look_then_safe_stop():
    """Dead-end memory disabled: max_look_cycles looks, then SAFE_STOP."""
    s = _sup(dead_end_enabled=False)
    yaw = 0.0
    modes, t = [], 0.0
    while t < 40.0 and s.mode != DriveMode.SAFE_STOP:
        d = s.update(t, 0.9, (0.0, 0.0, yaw), 0.0, 0.0, progress_metric=10.0)
        if d.override is not None:
            yaw += d.override[1] * 0.1
        modes.append(d.mode)
        assert not d.dead_end
        t += 0.1
    assert DriveMode.STOP_AND_LOOK in modes
    assert s.mode == DriveMode.SAFE_STOP and s.reason == "NO_CERTIFIED_PROGRESS"
    assert abs(math.atan2(math.sin(yaw), math.cos(yaw))) < 0.2  # each look returns to the original heading


def test_supervisor_dead_end_cycle_then_safe_stop():
    """Dead-end memory enabled: look -> dead end -> look -> ...; max_dead_ends dead ends without
    distance progress latch SAFE_STOP; every dead end comes after a look."""
    s = _sup()
    yaw, t = 0.0, 0.0
    modes, dead_t, look_starts = [], [], []
    prev = None
    while t < 200.0 and s.mode != DriveMode.SAFE_STOP:
        d = s.update(t, 0.9, (0.0, 0.0, yaw), 0.0, 0.0, progress_metric=10.0)
        if d.override is not None:
            yaw += d.override[1] * 0.1
        if d.dead_end:
            dead_t.append(t)
            assert d.override == (0.0, 0.0)
        if d.mode == DriveMode.STOP_AND_LOOK and prev != DriveMode.STOP_AND_LOOK:
            look_starts.append(t)
        prev = d.mode
        modes.append(d.mode)
        t += 0.1
    assert s.mode == DriveMode.SAFE_STOP and s.reason == "NO_CERTIFIED_PROGRESS"
    assert len(dead_t) == s.p.max_dead_ends == s.n_dead_ends
    for td in dead_t:  # a look preceded every dead end
        assert any(tl < td for tl in look_starts)
    assert abs(math.atan2(math.sin(yaw), math.cos(yaw))) < 0.2


def test_supervisor_dead_end_count_resets_on_progress():
    s = _sup()
    t, metric, n_dead = 0.0, 10.0, 0
    while t < 60.0:
        d = s.update(t, 0.9, (0.0, 0.0, 0.0), 0.0, 0.0, progress_metric=metric)
        n_dead += d.dead_end
        if d.dead_end:
            metric -= 1.0  # the planner found another way: distance progress follows
        t += 0.1
    assert n_dead >= 2 and s.mode != DriveMode.SAFE_STOP


def test_supervisor_operator_arrival_watchdog():
    s = _sup(goal=(1.0, 0.0))
    assert s.update(0.0, 0.9, (0.2, 0.0, 0.0), 0.0, 0.0).mode == DriveMode.ARRIVED
    s = _sup()
    s.operator(OperatorCmd(0.0, OperatorAction.HOLD))
    assert s.update(0.1, 0.9, (0, 0, 0), 0.0, 0.0).mode == DriveMode.HOLD
    s.operator(OperatorCmd(0.2, OperatorAction.RESUME))
    s.operator(OperatorCmd(0.3, OperatorAction.ESTOP))
    d = s.update(0.4, 0.9, (0, 0, 0), 0.0, 0.0)
    assert d.mode == DriveMode.SAFE_STOP and d.override == (0.0, 0.0)
    s.operator(OperatorCmd(0.5, OperatorAction.GO))
    d = s.update(0.6, 0.9, (0, 0, 0), 0.0, 0.0, frame_gap_s=0.8)
    assert d.override == (0.0, 0.0) and d.reason.startswith("WATCHDOG")


def test_supervisor_forward_check_brakes_before_lethal():
    rm = _open_map()
    _put(rm, 1.2, 1.6, -1.0, 1.0, S.POSITIVE)
    maps = build_costmap(rm, 0.0, True, CostmapParams())
    s = _sup()
    d = s.update(0.0, 0.9, (0, 0, 0), 0.0, 0.0)
    v, w, emergency, reason = s.gate(d, 1.0, 0.0, (0.0, 0.0, 0.0), maps)
    assert v == 0.0 and emergency and reason.startswith("FWD_CHECK")
    v, w, emergency, reason = s.gate(d, 1.0, 0.0, (0.0, 0.0, math.pi), maps)  # driving away is allowed
    assert v == 1.0 and reason is None


def test_supervisor_gate_allows_clear_rotation_when_forward_blocked():
    """Obstacle 0.2 m ahead of the bumper: the footprint circles (front circle reaches 0.6 m) are
    already inside it, so forward is blocked; the in-place look rotation sweeps only the body
    rectangle (half-diagonal 0.5 m), which stays clear, so it must pass. A rotation whose swept
    body hits an obstacle beside a corner stays blocked."""
    rm = _open_map()
    _put(rm, 0.6, 1.0, -1.0, 1.0, S.POSITIVE)
    maps = build_costmap(rm, 0.0, True, CostmapParams())
    s = _sup()
    look = SupervisorDecision(DriveMode.STOP_AND_LOOK, "STOP_AND_LOOK 1/3", 0.0, 1.0, (0.0, 0.6))
    now, fut = s.forward_clearance(0.0, 0.6, (0.0, 0.0, 0.0), maps)
    assert fut < s.p.footprint_radius_m and fut < now - 1e-3  # the old circle check would block
    assert s.rotation_clear(0.6, (0.0, 0.0, 0.0), maps)
    v, w, emergency, reason = s.gate(look, 0.0, 0.0, (0.0, 0.0, 0.0), maps)
    assert v == 0.0 and w == 0.6 and not emergency and reason is None
    v, w, emergency, reason = s.gate(s.update(0.0, 0.9, (0, 0, 0), 0.0, 0.0), 1.0, 0.0, (0.0, 0.0, 0.0), maps)
    assert v == 0.0 and reason.startswith("FWD_CHECK")  # forward still blocked
    rm2 = _open_map()
    _put(rm2, 0.3, 0.7, 0.35, 0.8, S.POSITIVE)  # beside the front-left corner: a left turn sweeps into it
    maps2 = build_costmap(rm2, 0.0, True, CostmapParams())
    assert not s.rotation_clear(0.6, (0.0, 0.0, 0.0), maps2)
    v, w, emergency, reason = s.gate(look, 0.0, 0.0, (0.0, 0.0, 0.0), maps2)
    assert v == 0.0 and w == 0.0 and emergency and reason.startswith("FWD_CHECK")


def test_supervisor_immobilised():
    s = _sup()
    d = None
    for k in range(40):
        d = s.update(0.1 * k, 0.9, (0, 0, 0), 0.5, 0.0, progress_metric=10.0 - 0.1 * k)
    assert d.mode == DriveMode.SAFE_STOP and d.reason == "IMMOBILISED"


def test_plan_stack_speed_budget():
    """Costmap + global + MPPI on a full 80 m map stay well inside a 5 Hz tick."""
    rm = _open_map(apron=20.0)
    _put(rm, 5.0, 6.0, -1.0, 1.0, S.POSITIVE)
    gp, mp = GlobalPlanner(), MppiPlanner(seed=0)
    ts = []
    for k in range(10):
        t0 = time.perf_counter()
        maps = build_costmap(rm, 0.2 * k, True, CostmapParams())
        plan = gp.update(0.2 * k, maps, (0.0, 0.0), (15.0, 0.0))
        mp.plan(0.2 * k, (0, 0, 0), 0.5, 0.0, 1.5, maps, plan.ctg_at, 0.5)
        ts.append((time.perf_counter() - t0) * 1e3)
    assert np.median(ts) < 120.0
