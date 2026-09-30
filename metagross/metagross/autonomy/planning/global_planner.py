"""Global cost-to-go field and descent path (A-frame).

A cost-to-go (CTG) field is grown *from the goal* with
``skimage.graph.MCP_Geometric`` (8-connected Dijkstra with geometric edge
weights) on a ``DS_RES_M`` down-sample of the planning costmap. The field serves
two consumers:

* MPPI's terminal cost (bilinear lookup, :meth:`GlobalPlan.ctg_at`), and
* the descent path / lookahead point (MCP traceback from the vehicle cell).

Per-cell step cost (cost units ~ metres of easy ground), on 2x2 blocks of map cells:

    step = (1 + COST_GAIN * c_obs) * (1 + (UNSEEN_PENALTY - 1) * f_unseen + (SUSPICIOUS_PENALTY - 1) * f_susp)

* ``c_obs``: observed-terrain cost + lethal inflation (``PlanningMaps.obs_cost``, 2x2 max),
  i.e. *not* the costmap's base cost of unknown cells (which is an MPPI stage cost);
* ``f_unseen`` / ``f_susp``: fraction of the 4 children that are never observed /
  suspicious (occluded, crest shadow, unconfirmed ditch candidate, single-frame lethal).
  Seen GROUND costs 1 per metre, UNSEEN 1.5, fully suspicious 2.0: the route prefers
  seen ground, but one unconfirmed candidate cell adds only ~(2-1)/4 of a cell and cannot
  push the route off the seen corridor (the old rule charged 9 per suspicious cell
  against 1.5 per unseen cell, which routed the vehicle *around* the ground it had seen);
* ``f_void``: fraction of children that are *void* (unseen although they sat in the
  certain-view wedge for several frames: ditch interior, terrain edge), charged
  ``VOID_PENALTY`` instead of ``UNSEEN_PENALTY`` so the route stops aiming at the same
  dark spot and explores elsewhere;
* lethal (any 2x2 child lethal, incl. confirmed ditches): ``inf`` (wall).

With ``unknown_is_free`` (typical stack) unknown cells cost 0 and carry no penalty.

The field is recomputed at ``REPLAN_HZ`` or immediately when a lethal cell
appears on the current descent path ("the route got cut"). A goal outside the
rolling map is clamped to the nearest map cell.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from skimage.graph import MCP_Geometric

from metagross.autonomy.planning.costmap import PlanningMaps
from metagross.autonomy.planning.dead_end import DeadEndMemory
from metagross.autonomy.planning.rolling_map import GridGeometry

LOG = logging.getLogger(__name__)

DS_FACTOR = 2  # 0.2 m map -> 0.4 m planning grid
COST_GAIN = 10.0  # step cost = 1 + COST_GAIN * c
UNSEEN_PENALTY = 1.5  # step-cost multiplier of a never-observed cell (seen ground = 1)
SUSPICIOUS_PENALTY = 2.0  # multiplier of a suspicious cell (occluded / shadow / unconfirmed candidate)
VOID_PENALTY = 6.0  # multiplier of a "void" cell: unseen although it sat in the certain-view wedge (rolling_map)
REPLAN_HZ = 1.0
LOOKAHEAD_M = 3.0
UNREACHABLE_PAD = 50.0  # CTG assigned to unreachable cells = max finite CTG + this [cost units]
CROP_MARGIN_M = 16.0  # MCP runs on a square covering vehicle and goal plus this margin
CROP_QUANT_CELLS = 8  # crop origin snapped to multiples of this (keeps the crop stable between ticks)


@dataclass(slots=True)
class GlobalPlan:
    """Result of the global planner. ``ctg`` is on ``geometry`` (the down-sampled grid)."""

    geometry: GridGeometry
    ctg: np.ndarray  # float32 (m, m), finite everywhere (unreachable padded)
    reachable: np.ndarray  # bool (m, m)
    goal_xy: tuple[float, float]
    path_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # (N,2) A-frame, vehicle -> goal
    lookahead_xy: Optional[tuple[float, float]] = None
    route_ok: bool = False  # vehicle cell connected to the goal
    ctg_vehicle: float = math.inf
    t_computed: float = -math.inf
    compute_ms: float = 0.0
    ctg_jump: float = 0.0  # on a recompute: new minus old CTG at the same vehicle position (new map info)

    def ctg_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear CTG at A-frame points; outside the grid: edge value + distance to the grid."""
        g = self.geometry
        fx = (np.asarray(x, np.float64) - g.origin_x) / g.res - 0.5
        fy = (np.asarray(y, np.float64) - g.origin_y) / g.res - 0.5
        fxc = np.clip(fx, 0.0, g.n - 1.000001)
        fyc = np.clip(fy, 0.0, g.n - 1.000001)
        x0 = np.floor(fxc).astype(np.int32)
        y0 = np.floor(fyc).astype(np.int32)
        ax, ay = fxc - x0, fyc - y0
        c = self.ctg
        v = (
            c[y0, x0] * (1 - ax) * (1 - ay)
            + c[y0, x0 + 1] * ax * (1 - ay)
            + c[y0 + 1, x0] * (1 - ax) * ay
            + c[y0 + 1, x0 + 1] * ax * ay
        )
        outside = np.hypot(fx - fxc, fy - fyc) * g.res
        return v + outside


class GlobalPlanner:
    """Maintains the CTG field; call :meth:`update` every tick (it decides when to recompute)."""

    def __init__(self, replan_hz: float = REPLAN_HZ, lookahead_m: float = LOOKAHEAD_M) -> None:
        self.replan_period_s = 1.0 / replan_hz
        self.lookahead_m = lookahead_m
        self.plan: Optional[GlobalPlan] = None
        self._mcp: Optional[MCP_Geometric] = None
        self._steps: Optional[np.ndarray] = None
        self._path_idx: Optional[np.ndarray] = None  # (N,2) (iy, ix) on plan geometry
        self._dead_end_version = -1  # DeadEndMemory.version the current field was computed with
        self.n_recomputes = 0

    def reset(self) -> None:
        self.plan = None
        self._mcp = None
        self._steps = None
        self._path_idx = None
        self._dead_end_version = -1

    # ------------------------------------------------------------------ internals
    @staticmethod
    def step_costs(maps: PlanningMaps, unknown_is_free: bool,
                   window: Optional[tuple[int, int, int]] = None) -> tuple[np.ndarray, GridGeometry]:
        """Down-sampled per-cell step costs (inf = wall) and their geometry.

        ``window = (i0, j0, side)`` in down-sampled cells restricts the computation to that
        square (the planner's crop); None = the whole map."""
        g = maps.geometry
        k = DS_FACTOR
        if window is None:
            i0, j0, side = 0, 0, g.n // k
        else:
            i0, j0, side = window
        sl = (slice(i0 * k, (i0 + side) * k), slice(j0 * k, (j0 + side) * k))
        c = _pool_max(maps.cost[sl], k)
        lethal = c >= 1.0
        if unknown_is_free:
            steps = 1.0 + COST_GAIN * c.astype(np.float64)
        else:
            obs = maps.obs_cost if maps.obs_cost is not None else np.where(maps.unseen | maps.suspicious, 0.0, maps.cost)
            c_obs = _pool_max(obs[sl], k).astype(np.float64)
            f_unseen = _pool_mean(maps.unseen[sl], k)
            f_susp = _pool_mean(maps.suspicious[sl], k)
            f_void = _pool_mean(maps.void[sl], k) if maps.void is not None else 0.0  # void cells are also unseen
            steps = (1.0 + COST_GAIN * c_obs) * (1.0 + (UNSEEN_PENALTY - 1.0) * (f_unseen - f_void)
                                                 + (VOID_PENALTY - 1.0) * f_void + (SUSPICIOUS_PENALTY - 1.0) * f_susp)
        steps[lethal] = np.inf
        res = g.res * k
        return steps, GridGeometry(res, side, g.origin_x + j0 * res, g.origin_y + i0 * res)

    def _route_cut(self, steps: np.ndarray, geo: GridGeometry) -> bool:
        if self.plan is None or self._path_idx is None or len(self._path_idx) == 0:
            return False
        if geo != self.plan.geometry:
            return True  # map scrolled or crop moved: indices no longer comparable
        iy, ix = self._path_idx[:, 0], self._path_idx[:, 1]
        return bool(np.any(np.isinf(steps[iy, ix])))

    # ------------------------------------------------------------------ API
    def update(
        self,
        t: float,
        maps: PlanningMaps,
        pose_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        force: bool = False,
        dead_ends: Optional[DeadEndMemory] = None,
    ) -> GlobalPlan:
        """Recompute the CTG field if due (1 Hz, route cut, goal change, map scroll, dead-end memory
        changed), then trace the descent path from the current vehicle position.
        ``dead_ends``: optional blocked-disc memory applied to the step costs (see :mod:`dead_end`)."""
        g = maps.geometry
        ds_geo = GridGeometry(g.res * DS_FACTOR, g.n // DS_FACTOR, g.origin_x, g.origin_y)
        window = _crop_window(ds_geo, pose_xy, goal_xy, CROP_MARGIN_M)
        steps, geo = self.step_costs(maps, maps.unknown_is_free, window)
        de_version = -1
        if dead_ends is not None and dead_ends.active(t):
            i0, j0, side = window
            sl = (slice(i0 * DS_FACTOR, (i0 + side) * DS_FACTOR), slice(j0 * DS_FACTOR, (j0 + side) * DS_FACTOR))
            unknown = maps.unseen[sl] | maps.suspicious[sl]
            steps = dead_ends.apply(t, steps, geo, _pool_mean(unknown, DS_FACTOR))
        if dead_ends is not None:
            de_version = dead_ends.version
        due = (
            force
            or self.plan is None
            or (t - self.plan.t_computed) >= self.replan_period_s - 1e-6
            or tuple(goal_xy) != self.plan.goal_xy
            or de_version != self._dead_end_version
            or self._route_cut(steps, geo)
        )
        self._dead_end_version = de_version
        old = self.plan
        if due:
            self._recompute(t, steps, geo, goal_xy)
        assert self.plan is not None
        self._trace(pose_xy)
        if due and old is not None and old.route_ok and self.plan.route_ok and old.goal_xy == self.plan.goal_xy:
            prev = float(old.ctg_at(np.array([pose_xy[0]]), np.array([pose_xy[1]]))[0])
            self.plan.ctg_jump = self.plan.ctg_vehicle - prev
        return self.plan

    def _recompute(self, t: float, steps: np.ndarray, geo: GridGeometry, goal_xy: tuple[float, float]) -> None:
        t0 = time.perf_counter()
        gi, gj = self._clamped_index(geo, goal_xy)
        steps = steps.copy()
        if not np.isfinite(steps[gi, gj]):
            steps[gi, gj] = UNSEEN_PENALTY  # never wall-in the goal cell itself
        mcp = MCP_Geometric(steps, fully_connected=True)
        cum, _ = mcp.find_costs([(gi, gj)])
        reachable = np.isfinite(cum)
        pad = (float(cum[reachable].max()) if np.any(reachable) else 0.0) + UNREACHABLE_PAD
        ctg = np.where(reachable, cum, pad).astype(np.float32)
        self._mcp, self._steps = mcp, steps
        self.plan = GlobalPlan(geometry=geo, ctg=ctg, reachable=reachable, goal_xy=(float(goal_xy[0]), float(goal_xy[1])), t_computed=t)
        self.plan.compute_ms = (time.perf_counter() - t0) * 1e3
        self.n_recomputes += 1

    @staticmethod
    def _clamped_index(geo: GridGeometry, xy: tuple[float, float]) -> tuple[int, int]:
        ix, iy = geo.to_index(np.array([xy[0]]), np.array([xy[1]]))
        return int(np.clip(iy[0], 0, geo.n - 1)), int(np.clip(ix[0], 0, geo.n - 1))

    def _trace(self, pose_xy: tuple[float, float]) -> None:
        plan = self.plan
        assert plan is not None and self._mcp is not None
        geo = plan.geometry
        vi, vj = self._clamped_index(geo, pose_xy)
        if not plan.reachable[vi, vj]:
            # vehicle cell walled (e.g. inside inflation): start from nearest reachable cell within 1.2 m
            r = int(math.ceil(1.2 / geo.res))
            i0, i1, j0, j1 = max(vi - r, 0), min(vi + r + 1, geo.n), max(vj - r, 0), min(vj + r + 1, geo.n)
            sub = np.where(plan.reachable[i0:i1, j0:j1], plan.ctg[i0:i1, j0:j1], np.inf)
            if not np.any(np.isfinite(sub)):
                plan.route_ok = False
                plan.ctg_vehicle = math.inf
                plan.path_xy = np.zeros((0, 2))
                plan.lookahead_xy = None
                self._path_idx = None
                return
            ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
            score = sub + np.hypot(ii - vi, jj - vj) * geo.res * UNSEEN_PENALTY
            k = int(np.argmin(score))
            vi, vj = int(ii.ravel()[k]), int(jj.ravel()[k])
        idx = np.asarray(self._mcp.traceback((vi, vj)), dtype=np.int32)[::-1]  # vehicle -> goal
        self._path_idx = idx
        px, py = geo.to_xy(idx[:, 1], idx[:, 0])
        path = np.column_stack([px, py])
        path[0] = pose_xy
        plan.path_xy = path
        plan.route_ok = True
        plan.ctg_vehicle = float(plan.ctg[vi, vj])
        plan.lookahead_xy = _point_at_arclength(path, self.lookahead_m)


def _pool_mean(a: np.ndarray, k: int) -> np.ndarray:
    """k x k mean-pool (as float64) via strided views; bool input gives the true fraction."""
    m = a.shape[0] // k
    out = np.zeros((m, m), np.float64)
    for di in range(k):
        for dj in range(k):
            out += a[di: m * k: k, dj: m * k: k]
    return out / (k * k)


def _pool_max(a: np.ndarray, k: int) -> np.ndarray:
    """k x k max-pool via strided views (much faster than reshape().max for k=2)."""
    m = a.shape[0] // k
    out = a[0: m * k: k, 0: m * k: k].copy()
    for di in range(k):
        for dj in range(k):
            if di or dj:
                np.maximum(out, a[di: m * k: k, dj: m * k: k], out=out)
    return out


def _crop_window(geo: GridGeometry, a_xy: tuple[float, float], b_xy: tuple[float, float],
                 margin_m: float) -> tuple[int, int, int]:
    """(i0, j0, side) of the square sub-grid of ``geo`` covering both points plus ``margin_m``,
    origin snapped to CROP_QUANT_CELLS."""
    n = geo.n
    ia, ja = geo.to_index(np.array([a_xy[0], b_xy[0]]), np.array([a_xy[1], b_xy[1]]))
    ix = np.clip(ia, 0, n - 1)
    iy = np.clip(ja, 0, n - 1)
    mc = int(math.ceil(margin_m / geo.res))
    side = int(min(n, max(ix.max() - ix.min(), iy.max() - iy.min()) + 2 * mc + CROP_QUANT_CELLS))
    q = CROP_QUANT_CELLS
    j0 = int(np.clip((int((ix.min() + ix.max()) // 2 - side // 2) // q) * q, 0, n - side))
    i0 = int(np.clip((int((iy.min() + iy.max()) // 2 - side // 2) // q) * q, 0, n - side))
    return i0, j0, side


def _crop_square(steps: np.ndarray, geo: GridGeometry, a_xy: tuple[float, float], b_xy: tuple[float, float],
                 margin_m: float) -> tuple[np.ndarray, GridGeometry]:
    """Square sub-grid of ``steps`` covering both points plus ``margin_m`` (see :func:`_crop_window`)."""
    i0, j0, side = _crop_window(geo, a_xy, b_xy, margin_m)
    return steps[i0:i0 + side, j0:j0 + side], GridGeometry(geo.res, side, geo.origin_x + j0 * geo.res, geo.origin_y + i0 * geo.res)


def _point_at_arclength(path: np.ndarray, s: float) -> tuple[float, float]:
    """Point at arc length ``s`` along polyline ``path`` (clamped to its end)."""
    if len(path) == 1:
        return float(path[0, 0]), float(path[0, 1])
    seg = np.hypot(*np.diff(path, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if s >= cum[-1]:
        return float(path[-1, 0]), float(path[-1, 1])
    k = int(np.searchsorted(cum, s, side="right") - 1)
    a = (s - cum[k]) / max(seg[k], 1e-9)
    p = path[k] + a * (path[k + 1] - path[k])
    return float(p[0]), float(p[1])


def resample_path(path: np.ndarray, spacing_m: float, max_points: int) -> np.ndarray:
    """Points every ``spacing_m`` along ``path`` (excluding the start), at most ``max_points``."""
    if len(path) < 2:
        return np.zeros((0, 2))
    seg = np.hypot(*np.diff(path, axis=0).T)
    total = float(seg.sum())
    ss = np.arange(1, max_points + 1) * spacing_m
    ss = ss[ss <= total + 1e-9]
    if len(ss) < max_points and (len(ss) == 0 or ss[-1] < total - 1e-6):
        ss = np.append(ss, total)[:max_points]
    return np.array([_point_at_arclength(path, float(s)) for s in ss]).reshape(-1, 2)
