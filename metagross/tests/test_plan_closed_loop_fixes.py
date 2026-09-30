"""Unit tests for the build-phase-II closed-loop fixes (planning / safety / map), synthetic inputs only.

* MPPI guided sampling (pure-pursuit guide) recovers from a wrong-way nominal;
* rolling map: vote aggregation, 2-frame lethal support, footprint clearing, certification layer;
* global planner: a few unconfirmed candidate cells do not push the route off the seen corridor;
* supervisor: turning onto the route counts as progress;
* governor: perception's measured r_det replaces the closed-form one.
"""

from __future__ import annotations

import math

import numpy as np

from metagross.autonomy.planning.costmap import CostmapParams, build_costmap
from metagross.autonomy.planning.global_planner import GlobalPlanner
from metagross.autonomy.planning.governor import SpeedGovernor
from metagross.autonomy.planning.mppi import MppiParams, MppiPlanner, pursuit_controls
from metagross.autonomy.planning.rolling_map import BevGeometry, RollingMap
from metagross.autonomy.safety.supervisor import Supervisor
from metagross.contracts.messages import CellState as S
from metagross.contracts.messages import DriveMode

GEO = BevGeometry()


def _bev(fill=S.UNSEEN) -> np.ndarray:
    return np.full((GEO.n_fwd, GEO.n_lat), int(fill), np.uint8)


def _patch(states, x0, x1, y0, y1, st):
    xb, yb = GEO.cell_centres_body()
    states[(xb >= x0) & (xb <= x1) & (yb >= y0) & (yb <= y1)] = int(st)
    return states


def _at(rm: RollingMap, layer: np.ndarray, x: float, y: float):
    ix, iy = rm.geometry.to_index(np.array([x]), np.array([y]))
    return layer[iy[0], ix[0]]


def _put(rm: RollingMap, x0, x1, y0, y1, st) -> None:
    g = rm.geometry
    ix, iy = np.meshgrid(np.arange(g.n), np.arange(g.n))
    xc, yc = g.to_xy(ix, iy)
    rm.state[(xc >= x0) & (xc <= x1) & (yc >= y0) & (yc <= y1)] = int(st)


# ------------------------------------------------------------------ MPPI guide
def test_pursuit_controls_straight_and_rotate_in_place():
    V, W = pursuit_controls((0.0, 0.0, 0.0), np.array([[0.0, 0.0], [10.0, 0.0]]), 1.5, 30, 0.1, 1.5, 0.8, 2.0, 1.2)
    assert np.allclose(V, 1.5, atol=1e-6) and np.allclose(W, 0.0, atol=1e-6)
    V, W = pursuit_controls((0.0, 0.0, 0.0), np.array([[0.0, 0.0], [0.0, 10.0]]), 1.5, 30, 0.1, 1.5, 0.8, 2.0, 1.2)
    assert V[0] == 0.0 and W[0] > 0.0  # route 90 deg left: rotate in place, to the left
    assert V[-1] > 0.5  # then drive


def _drive(mp: MppiPlanner, maps, goal, guide: bool, n: int = 25):
    """Unicycle toy loop at 10 Hz with a nominal initialised to turn the WRONG way (left)."""
    x, y, yaw, v, w = 0.0, 0.0, 0.0, 0.0, 0.0
    mp.U[:, 0], mp.U[:, 1] = 0.0, 0.9
    path = np.array([[0.0, 0.0], [0.0, -1.0], [goal[0], goal[1]]])
    ctg = lambda X, Y: np.hypot(X - goal[0], Y - goal[1])  # noqa: E731
    for k in range(n):
        res = mp.plan(0.1 * k, (x, y, yaw), v, w, 1.5, maps, ctg, 0.5, v_ref=1.5, cert_enabled=False,
                      guide_path_xy=path if guide else None)
        v, w = res.u0
        yaw += w * 0.1
        x += v * math.cos(yaw) * 0.1
        y += v * math.sin(yaw) * 0.1
    return math.hypot(x - goal[0], y - goal[1])


def test_mppi_guided_sampling_recovers_from_wrong_way_nominal():
    rm = RollingMap()
    rm.seed_apron((0, 0, 0), 0.0, 25.0)
    maps = build_costmap(rm, 0.0, True, CostmapParams())
    goal = (-1.0, -6.0)  # ~100 deg to the right of the heading
    d0 = math.hypot(*goal)  # 6.08 m
    d_guided = _drive(MppiPlanner(MppiParams(), seed=0), maps, goal, guide=True, n=40)
    d_plain = _drive(MppiPlanner(MppiParams(), seed=0), maps, goal, guide=False, n=40)
    assert d_guided < d0 - 2.5, f"guided MPPI made no headway ({d_guided:.2f} m left)"
    assert d_plain > d0 - 1.0  # the failure mode the guide fixes: the plain sampler stays stuck in 4 s


def test_mppi_hold_s_returns_command_at_end_of_hold():
    rm = RollingMap()
    rm.seed_apron((0, 0, 0), 0.0, 25.0)
    maps = build_costmap(rm, 0.0, True, CostmapParams())
    ctg = lambda X, Y: np.hypot(X - 10.0, Y)  # noqa: E731
    r = MppiPlanner(seed=2).plan(0.0, (0, 0, 0), 0.0, 0.0, 1.5, maps, ctg, 0.5, hold_s=0.2)
    assert r.u0[0] == r.U[1, 0] and r.u0[1] == r.U[1, 1]


def test_mppi_backs_off_only_when_reverse_allowed():
    """Stopped with the nose inside a wall's inflation: forward-only MPPI can only stand still,
    with ``allow_reverse`` it backs off; reverse is never used without the flag."""
    rm = RollingMap()
    rm.seed_apron((0, 0, 0), 0.0, 25.0)
    _put(rm, 0.9, 1.3, -3.0, 3.0, S.POSITIVE)  # wall 0.5 m ahead of the front bumper
    maps = build_costmap(rm, 0.0, True, CostmapParams())
    ctg = lambda X, Y: np.hypot(X - 8.0, Y)  # noqa: E731 - goal behind the wall
    params = MppiParams(v_reverse_max=0.3)
    fwd = MppiPlanner(params, seed=0).plan(0.0, (0, 0, 0), 0.0, 0.0, 1.5, maps, ctg, 0.5, cert_enabled=False)
    assert np.all(fwd.U[:, 0] >= 0.0)
    mp = MppiPlanner(params, seed=0)
    x = 0.0
    for k in range(10):
        r = mp.plan(0.1 * k, (x, 0.0, 0.0), 0.0, 0.0, 1.5, maps, ctg, 0.5, cert_enabled=False, allow_reverse=True)
        x += r.u0[0] * 0.1
    assert x < -0.05 and np.all(mp.U[:, 0] >= -0.3 - 1e-9)


# ------------------------------------------------------------------ rolling map
def test_single_noisy_subcell_is_not_lethal_but_rock_is_after_two_frames():
    rm = RollingMap()
    obs = _patch(_bev(), 2, 8, -2, 2, S.GROUND)
    xb, yb = GEO.cell_centres_body()
    i, j = np.unravel_index(np.argmin((xb - 4.05) ** 2 + (yb - 0.05) ** 2), xb.shape)
    obs[i, j] = int(S.POSITIVE)  # one 0.1 m noisy cell
    _patch(obs, 6.0, 6.4, -0.2, 0.2, S.POSITIVE)  # a 0.4 m rock
    for k in range(3):
        rm.integrate(0.2 * k, (0, 0, 0), obs.copy())
        if k == 0:
            assert not rm.lethal_mask()[0].any(), "one frame is never lethal"
            assert _at(rm, rm.pending, 6.2, 0.0)
    static, dyn = rm.lethal_mask()
    assert _at(rm, static, 6.2, 0.0) and not dyn.any()
    assert not _at(rm, static, 4.05, 0.05), "a single 0.1 m sub-cell out-voted by 3 ground sub-cells"


def test_footprint_clearing_removes_lethal_under_the_body():
    rm = RollingMap()
    _put(rm, 0.1, 0.3, -0.1, 0.1, S.POSITIVE)
    _put(rm, 3.0, 3.4, -0.2, 0.2, S.POSITIVE)
    n = rm.clear_footprint((0.0, 0.0, 0.0), 1.0, 0.8, 0.6)
    assert n > 0
    static, _ = rm.lethal_mask()
    assert not _at(rm, static, 0.2, 0.0) and _at(rm, static, 3.2, 0.0)
    assert _at(rm, rm.assumed, 0.2, 0.0)  # not evidence of free space for DYNAMIC detection


def test_certified_local_layer_gates_certification():
    rm = RollingMap()
    obs = _patch(_bev(), 2, 8, -2, 2, S.GROUND)
    cert = np.zeros(obs.shape, bool)
    xb, _ = GEO.cell_centres_body()
    cert[xb < 5.0] = True  # perception certifies only the first 5 m
    rm.integrate(0.0, (0, 0, 0), obs, certified_local=cert)
    maps = build_costmap(rm, 0.0, True, CostmapParams())
    c = lambda x: bool(maps.sample(maps.certified, np.array([x]), np.array([0.0]), False)[0])  # noqa: E731
    assert c(3.0) and not c(7.0)
    rm2 = RollingMap()
    rm2.integrate(0.0, (0, 0, 0), obs)  # key absent: previous behaviour
    maps2 = build_costmap(rm2, 0.0, True, CostmapParams())
    assert bool(maps2.sample(maps2.certified, np.array([7.0]), np.array([0.0]), False)[0])


def test_void_cells_in_view_wedge_become_expensive_for_the_planner():
    rm = RollingMap()
    obs = _patch(_bev(), 1.5, 12, -6, 6, S.GROUND)
    _patch(obs, 3.0, 4.0, -3.0, 3.0, S.UNSEEN)  # a dark band 3-4 m ahead (e.g. ditch interior)
    for k in range(3):
        rm.integrate(0.2 * k, (0, 0, 0), obs.copy())
    void = rm.void_mask()
    assert _at(rm, void, 3.5, 0.0) and not _at(rm, void, 3.5, 5.0)  # outside the wedge: plain unseen
    assert not _at(rm, void, 6.0, 0.0)  # observed ground
    maps = build_costmap(rm, 0.6, True, CostmapParams())
    steps, geo = GlobalPlanner.step_costs(maps, False)
    ix, iy = geo.to_index(np.array([3.5, -20.0]), np.array([0.0, 0.0]))
    assert steps[iy[0], ix[0]] > 2.0 * steps[iy[1], ix[1]]  # void >> plain unseen
    free = build_costmap(rm, 0.6, True, CostmapParams(unknown_is_free=True))
    assert free.void is None  # the typical stack does not reason about unknown space


# ------------------------------------------------------------------ global planner
def test_unconfirmed_candidates_do_not_push_route_off_seen_corridor():
    rm = RollingMap()  # everything unseen ...
    rm.seed_apron((0, 0, 0), 0.0, 1.0)
    _put(rm, -1.0, 16.0, -0.8, 0.8, S.GROUND)  # ... except a seen corridor
    _put(rm, 7.0, 7.4, -0.8, 0.8, S.DITCH_CANDIDATE)  # an unconfirmed candidate stripe across it
    maps = build_costmap(rm, 0.0, True, CostmapParams())
    plan = GlobalPlanner().update(0.0, maps, (0.0, 0.0), (15.0, 0.0))
    assert plan.route_ok
    assert np.all(np.abs(plan.path_xy[:, 1]) <= 1.2), "route left the seen corridor"


# ------------------------------------------------------------------ supervisor
def test_turning_onto_route_is_progress_but_standing_still_is_not():
    s = Supervisor()
    s.reset((20.0, 0.0), 2.0)
    herr = 2.5
    modes = []
    for k in range(50):  # 5 s of in-place rotation toward the route, no distance progress
        herr = max(herr - 0.06, 0.1)
        modes.append(s.update(0.1 * k, 0.9, (0.0, 0.0, 0.0), 0.0, 0.0, progress_metric=20.0, heading_err=herr).mode)
    assert DriveMode.STOP_AND_LOOK not in modes
    modes = [s.update(5.0 + 0.1 * k, 0.9, (0.0, 0.0, 0.0), 0.0, 0.0, progress_metric=20.0, heading_err=0.1).mode for k in range(40)]
    assert DriveMode.STOP_AND_LOOK in modes


# ------------------------------------------------------------------ governor
def test_governor_uses_measured_r_det_when_given():
    rm = RollingMap()
    rm.seed_apron((0, 0, 0), 0.0, 30.0)
    maps = build_costmap(rm, 0.0, True, CostmapParams())
    gov = SpeedGovernor()
    base = gov.compute(maps, (0.0, 0.0, 0.0), np.zeros((0, 2)), 12.0, 1.0, 0.05, 0.2)
    short = gov.compute(maps, (0.0, 0.0, 0.0), np.zeros((0, 2)), 12.0, 1.0, 0.05, 0.2, r_det_m=2.0)
    assert short.terms["ditch_det"] < base.terms["ditch_det"] and short.binding == "ditch_det"
    nan = gov.compute(maps, (0.0, 0.0, 0.0), np.zeros((0, 2)), 12.0, 1.0, 0.05, 0.2, r_det_m=float("nan"))
    assert nan.terms["ditch_det"] == base.terms["ditch_det"]
