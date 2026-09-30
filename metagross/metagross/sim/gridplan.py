"""Grid A* on a GT hazard raster (ground-truth side only).

Used by the scenario generator to *guarantee solvability* (a collision-free corridor of width
``CORRIDOR_WIDTH_M`` from A to B) and by the runner to compute the optimal path length for SPL.

Method: the Euclidean clearance to the nearest hazard cell is computed on the 0.05 m terrain grid
(exact EDT). A* (8-connected, octile heuristic, no corner cutting) then runs on a
``PLAN_RES_M`` lattice of cell centres whose clearance is >= half the corridor width, and the
lattice path is shortened by greedy line-of-sight shortcutting against the same clearance field
(an any-angle approximation, so SPL is not penalised by the 8-connected metric).

Tolerance: clearance is enforced at lattice points and at line-of-sight samples every half terrain
cell; the corridor-width guarantee therefore holds to within one terrain cell (0.05 m).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from metagross.sim.geometry import GridSpec

CORRIDOR_WIDTH_M = 1.2  # required collision-free corridor width (m)
PLAN_RES_M = 0.10  # A* lattice spacing (m)
BORDER_MARGIN_M = 1.0  # the outer ring of the world is not drivable (m)
_SQRT2 = math.sqrt(2.0)


@dataclass
class PlanResult:
    ok: bool
    path_xy: np.ndarray  # (N, 2) world metres, shortcut path (empty if not ok)
    length_m: float  # length of ``path_xy`` (inf if not ok)
    lattice_length_m: float  # length of the raw 8-connected lattice path
    expanded: int  # A* node expansions


def clearance_field(grid: GridSpec, hazard: np.ndarray) -> np.ndarray:
    """Distance (m) from each cell centre to the nearest hazard cell centre (0 on hazards)."""
    if not hazard.any():
        return np.full(grid.shape, np.inf, np.float32)
    return (distance_transform_edt(~hazard) * grid.res).astype(np.float32)


def astar_lattice(free: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> tuple[list[int] | None, int]:
    """8-connected A* on a boolean grid. Returns (flat indices start->goal or None, expansions).

    The outer ring of ``free`` is forced to False so neighbour lookups need no bounds checks.
    """
    ny, nx = free.shape
    f = free.copy()
    f[0, :] = f[-1, :] = False
    f[:, 0] = f[:, -1] = False
    s = start[0] * nx + start[1]
    g = goal[0] * nx + goal[1]
    if not (f.flat[s] and f.flat[g]):
        return None, 0
    free_l = f.ravel().tolist()
    n = ny * nx
    gscore = [math.inf] * n
    parent = [-1] * n
    closed = bytearray(n)
    gi, gj = goal
    steps = ((-nx, 1.0, 0, 0), (nx, 1.0, 0, 0), (-1, 1.0, 0, 0), (1, 1.0, 0, 0),
             (-nx - 1, _SQRT2, -nx, -1), (-nx + 1, _SQRT2, -nx, 1), (nx - 1, _SQRT2, nx, -1), (nx + 1, _SQRT2, nx, 1))
    k_oct = _SQRT2 - 2.0

    def h(idx: int) -> float:
        i, j = divmod(idx, nx)
        di = abs(i - gi)
        dj = abs(j - gj)
        return di + dj + k_oct * (di if di < dj else dj)

    gscore[s] = 0.0
    heap: list[tuple[float, float, int]] = [(h(s), 0.0, s)]
    expanded = 0
    push, pop = heapq.heappush, heapq.heappop
    while heap:
        _, gc, u = pop(heap)
        if closed[u]:
            continue
        if u == g:
            path = [u]
            while path[-1] != s:
                path.append(parent[path[-1]])
            return path[::-1], expanded
        closed[u] = 1
        expanded += 1
        for off, cost, c1, c2 in steps:
            v = u + off
            if not free_l[v] or closed[v]:
                continue
            if c1 and not (free_l[u + c1] and free_l[u + c2]):
                continue  # no corner cutting on diagonal moves
            ng = gc + cost
            if ng < gscore[v]:
                gscore[v] = ng
                parent[v] = u
                push(heap, (ng + h(v), ng, v))
    return None, expanded


def _polyline_length(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def _line_clear(grid: GridSpec, clear: np.ndarray, a: np.ndarray, b: np.ndarray, need: float) -> bool:
    n = max(2, int(math.ceil(float(np.linalg.norm(b - a)) / (0.5 * grid.res))) + 1)
    ts = np.linspace(0.0, 1.0, n)
    pts = a[None, :] + ts[:, None] * (b - a)[None, :]
    i, j = grid.xy_to_ij(pts[:, 0], pts[:, 1])
    return bool(np.all(clear[i, j] >= need))


def shortcut_path(grid: GridSpec, clear: np.ndarray, pts: np.ndarray, need: float) -> np.ndarray:
    """Greedy line-of-sight shortcutting of a lattice path against a clearance field."""
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    i = 0
    n = len(pts)
    while i < n - 1:
        k = i + 1
        while k + 1 < n and _line_clear(grid, clear, pts[i], pts[k + 1], need):
            k += 1
        out.append(pts[k])
        i = k
    return np.asarray(out)


def plan_path(grid: GridSpec, hazard: np.ndarray, start_xy: tuple[float, float], goal_xy: tuple[float, float],
              corridor_width_m: float = CORRIDOR_WIDTH_M, plan_res_m: float = PLAN_RES_M,
              border_m: float = BORDER_MARGIN_M, clear: np.ndarray | None = None) -> PlanResult:
    """Shortest path from start to goal keeping ``corridor_width_m / 2`` clearance from hazards."""
    need = 0.5 * corridor_width_m
    if clear is None:
        clear = clearance_field(grid, hazard)
    factor = max(1, int(round(plan_res_m / grid.res)))
    lat = grid.subsample(factor)
    free = clear[::factor, ::factor] >= need
    xs, ys = lat.cell_centres()
    free &= grid.contains(xs, ys, margin=border_m)
    si, sj = lat.xy_to_ij(start_xy[0], start_xy[1])
    gi, gj = lat.xy_to_ij(goal_xy[0], goal_xy[1])
    path, expanded = astar_lattice(free, (int(si), int(sj)), (int(gi), int(gj)))
    if path is None:
        return PlanResult(False, np.zeros((0, 2)), math.inf, math.inf, expanded)
    ii, jj = np.divmod(np.asarray(path), lat.nx)
    px, py = lat.ij_to_xy(ii, jj)
    pts = np.stack([px, py], axis=1)
    lattice_len = _polyline_length(pts)
    # Replace the snapped end points by the exact start/goal (they lie within half a lattice cell).
    pts[0] = start_xy
    pts[-1] = goal_xy
    short = shortcut_path(grid, clear, pts, need)
    return PlanResult(True, short, _polyline_length(short), lattice_len, expanded)
