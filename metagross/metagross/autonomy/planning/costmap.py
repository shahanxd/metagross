"""Observation state -> traversal cost, with footprint inflation.

Produces :class:`PlanningMaps`, the single bundle of A-frame layers that the
global planner, MPPI, speed governor and safety supervisor read.

Cost semantics (float32 in [0, 1]; exactly 1.0 = lethal for the footprint):

* GROUND: perception cost (slope / roughness / semantics).
* WATER: at least ``water_min_cost``; never certified (not ground).
* UNSEEN: ``unseen_cost``; OCCLUDED / CREST_SHADOW / unconfirmed ditch candidate /
  single-frame ("pending") lethal observation: ``suspicious_cost``. With
  ``unknown_is_free`` (typical-stack baseline) all of these cost 0 and count as certified.
* Certified = GROUND (not pending; perception-certified when the map has that layer),
  fresh or health NOMINAL, and not inside the lethal inflation.
* ``obs_cost`` = ``cost`` without the unseen / suspicious base cost (terrain + inflation
  only), used by the global planner, which applies its own multiplicative penalties.
* Lethal cores (POSITIVE, DEPRESSION, confirmed ditch; DYNAMIC with an extra
  ``dynamic_extra_m``) are inflated: cost 1 within ``lethal_radius_m`` of a core
  cell edge, then decaying linearly to 0 over ``decay_m``.

Implementation notes (this runs every tick on a 400 x 400 map): per-cell classes
come from one 32-entry look-up table indexed by ``state | confirmed << 4``;
``cv2.distanceTransform`` (exact Euclidean) runs only on the bounding box of the
lethal cores plus the inflation reach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from metagross.autonomy.planning.rolling_map import FRESH_S, GridGeometry, RollingMap
from metagross.contracts.messages import CellState as S

LETHAL_RADIUS_M = 0.45  # centre-of-footprint-circle clearance: circle radius 0.33 m + 0.12 m margin
DECAY_M = 0.40  # linear decay band beyond the lethal radius
DYNAMIC_EXTRA_M = 0.50  # extra inflation for DYNAMIC cells (moving obstacles)
UNSEEN_COST = 0.30  # MPPI stage cost of never-observed cells (global planner uses its own rule)
SUSPICIOUS_COST = 0.50  # occluded / crest shadow / unconfirmed ditch candidate
WATER_MIN_COST = 0.90
BIG_DIST_M = 1e3  # clearance reported far from any lethal cell

# look-up-table bit flags, indexed by code = state | (confirmed << 4)
F_STATIC, F_DYNAMIC, F_UNSEEN, F_SUSPICIOUS, F_WATER, F_GROUND = 1, 2, 4, 8, 16, 32


@lru_cache(maxsize=4)
def class_lut(use_negobs: bool) -> np.ndarray:
    """(32,) uint8 flag table for code = state | confirmed << 4."""
    lut = np.zeros(32, np.uint8)
    for conf in (0, 1):
        for st in S:
            f = 0
            if st in (S.POSITIVE, S.DEPRESSION):
                f |= F_STATIC
            elif st == S.DITCH_CANDIDATE:
                f |= F_STATIC if (conf and use_negobs) else F_SUSPICIOUS
            elif st == S.DYNAMIC:
                f |= F_DYNAMIC
            elif st == S.UNSEEN:
                f |= F_UNSEEN
            elif st in (S.OCCLUDED, S.CREST_SHADOW):
                f |= F_SUSPICIOUS
            elif st == S.WATER:
                f |= F_WATER
            elif st == S.GROUND:
                f |= F_GROUND
            lut[int(st) | (conf << 4)] = f
    return lut


@dataclass(frozen=True, slots=True)
class CostmapParams:
    lethal_radius_m: float = LETHAL_RADIUS_M
    decay_m: float = DECAY_M
    dynamic_extra_m: float = DYNAMIC_EXTRA_M
    unseen_cost: float = UNSEEN_COST
    suspicious_cost: float = SUSPICIOUS_COST
    water_min_cost: float = WATER_MIN_COST
    inflate_scale: float = 1.0  # >1 in CAUTION mode (inflated costs)
    unknown_is_free: bool = False
    use_negobs: bool = True

    def scaled(self, inflate_scale: float) -> "CostmapParams":
        return replace(self, inflate_scale=float(inflate_scale))


@dataclass(slots=True)
class PlanningMaps:
    """A-frame planning layers, all ``(n, n)`` indexed ``[iy, ix]`` on ``geometry``."""

    geometry: GridGeometry
    cost: np.ndarray  # float32 [0,1], 1.0 = lethal (inflated)
    lethal_core: np.ndarray  # bool, un-inflated lethal cells (static | dynamic)
    clearance_m: np.ndarray  # float32, distance from cell centre to nearest lethal-core edge (dynamic minus extra)
    certified: np.ndarray  # bool, drivable ground certified for speed
    unseen: np.ndarray  # bool, never observed
    suspicious: np.ndarray  # bool, occluded / crest shadow / unconfirmed candidate
    mu: np.ndarray  # float32 braking friction proxy
    unknown_is_free: bool = False
    obs_cost: Optional[np.ndarray] = None  # float32: like ``cost`` but 0 base cost on unseen / suspicious cells (planner)
    void: Optional[np.ndarray] = None  # bool: unseen cells left dark inside the certain-view wedge (RollingMap.void_mask)

    def sample(self, layer: np.ndarray, x: np.ndarray, y: np.ndarray, fill: float) -> np.ndarray:
        """Nearest-cell lookup of ``layer`` at A-frame points (any shape); ``fill`` outside the map."""
        g = self.geometry
        ix, iy = g.to_index(x, y)
        inb = g.in_bounds(ix, iy)
        n = g.n
        vals = np.take(layer.ravel(), np.clip(iy, 0, n - 1) * n + np.clip(ix, 0, n - 1))
        return np.where(inb, vals, np.asarray(fill, dtype=vals.dtype))

    def flat_index(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fast flat cell index (clamped to the border) for hot loops (MPPI)."""
        g = self.geometry
        inv = 1.0 / g.res
        ix = np.clip(((x - g.origin_x) * inv).astype(np.int32), 0, g.n - 1)
        iy = np.clip(((y - g.origin_y) * inv).astype(np.int32), 0, g.n - 1)
        return iy * g.n + ix


def build_costmap(rmap: RollingMap, t: float, health_nominal: bool, params: CostmapParams) -> PlanningMaps:
    """Compute :class:`PlanningMaps` from the rolling map at time ``t``."""
    res = rmap.res
    code = rmap.state | (rmap.confirmed.view(np.uint8) << 4)
    flags = class_lut(params.use_negobs)[code]
    static = (flags & F_STATIC) > 0
    dynamic = (flags & F_DYNAMIC) > 0
    unseen = (flags & F_UNSEEN) > 0
    core = static | dynamic
    pending = getattr(rmap, "pending", None)
    # a lethal observation from a single frame is suspicious (not certified, not lethal)
    suspicious = ((flags & F_SUSPICIOUS) > 0) | (pending & ~core if pending is not None else False)

    base = rmap.cost.copy()
    water = (flags & F_WATER) > 0
    base[water] = np.maximum(base[water], params.water_min_cost)
    obs_base = np.where(unseen | suspicious, 0.0, base).astype(np.float32)  # observed-terrain cost only
    if params.unknown_is_free:
        base[unseen | suspicious] = 0.0
        certified = ~core
    else:
        base[unseen] = params.unseen_cost
        base[suspicious] = params.suspicious_cost
        certified = ((flags & F_GROUND) > 0) & ~suspicious
        if getattr(rmap, "has_cert_layer", False):
            certified &= rmap.cert_obs  # only ground perception certified in >= 1 frame
        if not health_nominal:
            certified &= (t - rmap.t_ground) <= FRESH_S

    clearance = np.full(core.shape, BIG_DIST_M, np.float32)
    cost = base
    r_l = params.lethal_radius_m * params.inflate_scale
    r_d = params.decay_m * params.inflate_scale
    rows = np.flatnonzero(core.any(axis=1))
    if rows.size:
        cols = np.flatnonzero(core.any(axis=0))
        m = int(math.ceil((r_l + r_d + params.dynamic_extra_m) / res)) + 2
        n = core.shape[0]
        i0, i1 = max(rows[0] - m, 0), min(rows[-1] + m + 1, n)
        j0, j1 = max(cols[0] - m, 0), min(cols[-1] + m + 1, n)
        sl = (slice(i0, i1), slice(j0, j1))
        clr = np.minimum(_edge_distance(static[sl], res), _edge_distance(dynamic[sl], res) - params.dynamic_extra_m)
        infl = np.clip(1.0 - (clr - r_l) / max(r_d, 1e-6), 0.0, 0.999).astype(np.float32)
        infl[clr <= r_l] = 1.0
        np.maximum(cost[sl], infl, out=cost[sl])
        np.maximum(obs_base[sl], infl, out=obs_base[sl])
        clearance[sl] = clr
        cost[core] = 1.0
        obs_base[core] = 1.0
    certified &= cost < 1.0
    return PlanningMaps(
        geometry=rmap.geometry, cost=cost, lethal_core=core, clearance_m=clearance, certified=certified,
        unseen=unseen, suspicious=suspicious, mu=rmap.mu, unknown_is_free=params.unknown_is_free, obs_cost=obs_base,
        void=None if (params.unknown_is_free or not hasattr(rmap, "void_mask")) else rmap.void_mask() & unseen,
    )


def _edge_distance(mask: np.ndarray, res: float) -> np.ndarray:
    """Distance [m] from each cell centre to the nearest edge of a ``mask`` cell (0 inside)."""
    if not np.any(mask):
        return np.full(mask.shape, BIG_DIST_M, np.float32)
    src = np.where(mask, 0, 255).astype(np.uint8)
    d = cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.maximum(d * res - 0.5 * res, 0.0).astype(np.float32)
