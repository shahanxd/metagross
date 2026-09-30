"""Odometry-frame ("A-frame") rolling observation map.

Every tick the perception module hands over an *egocentric* bird's-eye-view
(BEV) grid of observation states (:class:`~metagross.contracts.messages.CellState`),
traversal costs and (optionally) braking-friction estimates. This module fuses
those grids, using the localiser pose, into a square A-frame map of
``MAP_SIZE_M`` x ``MAP_SIZE_M`` at ``MAP_RES_M`` that scrolls with the vehicle.

Fusion rules (the "memory" half of the thesis *unknown is never free*):

* **Aggregation 0.1 m BEV -> 0.2 m map cell is a vote, not a max.** Per frame and map
  cell the BEV sub-cells are counted per state. The cell observes *lethal* this frame
  when at least ``LETHAL_MIN_SUBCELLS`` sub-cells, or at least half of its seen
  sub-cells, are POSITIVE / DEPRESSION / DYNAMIC (a single noisy 0.1 m cell does not
  make a 0.2 m cell lethal; any real obstacle >= 0.2 m covers two). Otherwise the
  non-lethal states vote (ties go to the more severe state), and the cell cost is the
  mean over its non-lethal sub-cells.
* **Lethal needs support from >= 2 frames.** A lethal observation enters a 3-bit
  history; the cell becomes POSITIVE / DEPRESSION / DYNAMIC once 2 of the last 3
  in-view frames saw it lethal, or when the previous in-view frame saw lethal within
  ``LETHAL_SUPPORT_RADIUS_CELLS`` (an object moving <= ~1 cell per frame still confirms
  after 2 frames). A single-frame lethal observation only marks the cell *pending*
  (not certified, costed like a suspicious cell, never lethal).
* **Lethal persists.** POSITIVE, DEPRESSION, confirmed ditches and DYNAMIC cells
  persist until they are re-observed as GROUND in 2 of the last 3 frames in which
  the cell was in view. A ditch lip leaves the lower edge of the vertical FOV on
  approach; without memory the vehicle would forget the ditch just before it.
* **Ditch candidates need confirmation.** A DITCH_CANDIDATE becomes a confirmed
  (lethal) ditch only after 2 of the last 3 in-view frames flag it. Unconfirmed
  candidates are *not certified* (the governor will not drive onto them) but
  are not lethal either.
* **Certification.** GROUND observed within ``FRESH_S`` seconds is certified.
  Older GROUND is *stale-certified*: it counts only while localisation health
  is NOMINAL (pose drift makes old observations untrustworthy). When perception
  supplies ``certified_local`` (per-cell "missing ground would have been detected
  here"), only ground that was certified in at least one frame (majority of its
  sub-cells) counts; without that key every GROUND cell does (previous behaviour).
* **Dynamic.** New confirmed occupancy where the cell was *well observed free* (GROUND in
  both in-view frames before the occupancy started, seen within ``DYNAMIC_RECENT_GROUND_S``,
  not merely assumed), or perception's own DYNAMIC state, is labelled DYNAMIC and held for
  ``DYNAMIC_HOLD_S``; the costmap inflates it by an extra 0.5 m. The map-side inference can be
  switched off with ``INFER_DYNAMIC`` (see the constant for the measured trade-off).
* **Footprint clearing.** :meth:`clear_footprint` marks the cells under the vehicle
  body as (assumed) GROUND: the vehicle stands on them, so any lethal label there is
  a false positive (without this a spurious cell under the body deadlocks the
  forward-footprint check).
* **Void.** Cells inside the certain-view wedge (2-5 m, +-30 deg of the heading) that got
  no observation for ``VOID_FRAMES`` frames are reported by :meth:`void_mask` (ditch
  interior behind its lip, terrain edge); the global planner charges them more than UNSEEN.
* **Occlusion / crest shadow / water** never overwrite a persistent lethal cell,
  and OCCLUDED / CREST_SHADOW never erase an earlier GROUND observation (the
  ground was seen; its certificate simply ages).

Frames and indexing
-------------------
* A-frame: x forward along launch heading, y left (metres).
* Map arrays are indexed ``[iy, ix]`` with ``x = origin_x + (ix + 0.5) * res`` and
  ``y = origin_y + (iy + 0.5) * res``. ``origin`` is the A-frame corner of cell
  ``[0, 0]`` and moves in whole cells when the map recentres.
* Egocentric BEV: see :class:`BevGeometry` (perception's ``BevSpec`` convention: row grows
  forward from ``x_min`` = -2 m, column grows to the left).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from metagross.config.defaults import BEV_LOCAL_SIZE_M, BEV_RES_M, MAP_RES_M, MAP_SIZE_M
from metagross.contracts.messages import CellState

LOG = logging.getLogger(__name__)

FRESH_S = 5.0  # GROUND observed within this window is certified regardless of health
DYNAMIC_HOLD_S = 2.0  # DYNAMIC cells stay lethal at least this long after last occupancy
DYNAMIC_RECENT_GROUND_S = 3.0  # occupancy counts as "new" if GROUND was seen this recently
CONFIRM_HITS = 2  # k-of-3 confirmation for ditch candidates / lethal cells and for clearing lethal memory
LETHAL_MIN_SUBCELLS = 2  # BEV sub-cells that make a map cell observe "lethal" in one frame (or >= half of seen ones)
LETHAL_SUPPORT_RADIUS_CELLS = 1  # previous-frame lethal within this many map cells supports a new one (moving objects)
# Map-side DYNAMIC inference ("new occupancy on well-observed ground" -> DYNAMIC: +0.5 m inflation,
# 2 s hold). Trade-off measured on DEV (results/closed_loop_dev.json, notes.dynamic_inference):
# ON  - tier0 FULL 30 seeds: 1 collision (F4 moving box); rendered stereo 102/103/104: flickering
#       static false positives become DYNAMIC (50-74 cells within +-4 m on 102/103, which have no
#       moving objects) and wall the vehicle in after ~2 m;
# OFF - tier0 FULL: 3 collisions (2 F4 moving obstacles, 1 static-rock graze); stereo 102 drove 8.4 m.
# Kept ON (collisions weigh more); turning it off needs a perception false-positive fix first.
INFER_DYNAMIC = True
LAUNCH_APRON_M = 2.0  # radius certified at launch (camera blind zone ~1.5 m ahead of body)
RECENTRE_FRACTION = 0.25  # recentre when vehicle is this fraction of map size off-centre
DEFAULT_MU = 0.4  # braking-friction proxy when perception gives none (SEM_MU of 'unstable')
FOOTPRINT_MARGIN_M = 0.0  # extra margin around the body rectangle cleared by clear_footprint [m]
# "Void": cells inside the camera's certain-view wedge (ground there should return points on any
# drivable surface) that stayed without any observation for VOID_FRAMES frames: ditch interiors
# behind the lip, terrain edges, deep dips. Not lethal, but the global planner charges them more
# than plain UNSEEN so it stops routing through the same dark spot (exploration dead ends).
VOID_R_NEAR_M = 2.0  # beyond the ~1.5 m near blind zone (camera 0.9 m high, pitched 12 deg down) + margin
VOID_R_FAR_M = 5.0  # ground within this range returns stereo points on flat-ish terrain (CAM_MAX_RANGE_M = 12)
VOID_HALF_ANGLE_RAD = math.radians(30.0)  # inside the 72 deg HFOV with a 6 deg margin
VOID_FRAMES = 3  # consecutive in-wedge frames without an observation
N_STATES = 16  # CellState codes fit in 4 bits

_S = CellState
# Severity rank: tie-break of the per-cell vote (the more severe state wins a tie).
_RANK = np.zeros(N_STATES, dtype=np.uint8)
for _rank, _state in enumerate(
    [_S.UNSEEN, _S.GROUND, _S.OCCLUDED, _S.CREST_SHADOW, _S.WATER, _S.DITCH_CANDIDATE, _S.DYNAMIC, _S.DEPRESSION, _S.POSITIVE]
):
    _RANK[int(_state)] = _rank
_POPCOUNT3 = np.array([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.uint8)
_LETHAL_STATES = (int(_S.POSITIVE), int(_S.DEPRESSION), int(_S.DYNAMIC))
_NONLETHAL_VOTERS = np.array([int(_S.GROUND), int(_S.OCCLUDED), int(_S.CREST_SHADOW), int(_S.WATER), int(_S.DITCH_CANDIDATE)])


try:  # perception owns the BEV grid; use its origin so integration is a no-op
    from metagross.autonomy.perception.bev import BEV_X_MIN_M as _BEV_X_MIN_M
except ImportError:  # perception package absent (e.g. isolated planning tests)
    _BEV_X_MIN_M = -2.0


@dataclass(frozen=True, slots=True)
class BevGeometry:
    """Geometry of perception's egocentric BEV grid ``(Hf, Wl)`` (same convention as
    ``metagross.autonomy.perception.bev.BevSpec``).

    ``grid[i, j]`` has its centre at body ``x = x_min_m + (i + 0.5) * res_m`` (row index grows
    forward) and ``y = y_min_m + (j + 0.5) * res_m`` (column index grows to the LEFT).
    Body frame: x forward, y left, origin at the wheelbase centre on the ground.
    """

    res_m: float = BEV_RES_M
    n_fwd: int = int(round(BEV_LOCAL_SIZE_M[0] / BEV_RES_M))
    n_lat: int = int(round(BEV_LOCAL_SIZE_M[1] / BEV_RES_M))
    x_min_m: float = _BEV_X_MIN_M
    y_min_m: float = -BEV_LOCAL_SIZE_M[1] / 2.0

    @classmethod
    def for_shape(cls, shape: tuple[int, int], res_m: float = BEV_RES_M) -> "BevGeometry":
        """Default geometry for a BEV array of ``shape`` (laterally centred)."""
        return cls(res_m=res_m, n_fwd=int(shape[0]), n_lat=int(shape[1]), y_min_m=-shape[1] * res_m / 2.0)

    @classmethod
    def from_perception(cls, out: Mapping[str, Any], shape: tuple[int, int]) -> "BevGeometry":
        """Geometry from an optional ``out['bev_spec']`` / ``out['bev_geometry']`` dict with
        ``BevSpec.as_dict()`` keys (``res_m``, ``x_min_m``, ``y_min_m``); defaults otherwise."""
        g = dict(out.get("bev_spec") or out.get("bev_geometry") or {})
        res = float(g.get("res_m", BEV_RES_M))
        base = cls.for_shape(shape, res)
        return cls(
            res_m=res, n_fwd=base.n_fwd, n_lat=base.n_lat,
            x_min_m=float(g.get("x_min_m", base.x_min_m)), y_min_m=float(g.get("y_min_m", base.y_min_m)),
        )

    def cell_centres_body(self) -> tuple[np.ndarray, np.ndarray]:
        """(x_body, y_body) of every BEV cell, each shaped ``(n_fwd, n_lat)``, metres."""
        xs = self.x_min_m + (np.arange(self.n_fwd, dtype=np.float64) + 0.5) * self.res_m
        ys = self.y_min_m + (np.arange(self.n_lat, dtype=np.float64) + 0.5) * self.res_m
        return np.broadcast_to(xs[:, None], (self.n_fwd, self.n_lat)), np.broadcast_to(ys[None, :], (self.n_fwd, self.n_lat))


@dataclass(frozen=True, slots=True)
class GridGeometry:
    """Square A-frame grid: ``n`` x ``n`` cells of ``res`` metres; ``origin`` = corner of cell [0,0]."""

    res: float
    n: int
    origin_x: float
    origin_y: float

    def to_index(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A-frame metres -> integer (ix, iy); may be out of bounds."""
        ix = np.floor((np.asarray(x) - self.origin_x) / self.res).astype(np.int32)
        iy = np.floor((np.asarray(y) - self.origin_y) / self.res).astype(np.int32)
        return ix, iy

    def to_xy(self, ix: np.ndarray, iy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Integer (ix, iy) -> A-frame cell-centre metres."""
        return self.origin_x + (np.asarray(ix) + 0.5) * self.res, self.origin_y + (np.asarray(iy) + 0.5) * self.res

    def in_bounds(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        return (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)


@dataclass(slots=True)
class IntegrateStats:
    """Diagnostics of one :meth:`RollingMap.integrate` call."""

    n_observed_cells: int = 0
    n_new_lethal: int = 0
    n_confirmed_ditch: int = 0
    n_dynamic: int = 0
    recentred: bool = False


class RollingMap:
    """A-frame observation map that fuses egocentric BEV grids (see module docstring)."""

    def __init__(self, size_m: float = MAP_SIZE_M, res_m: float = MAP_RES_M) -> None:
        self.n = int(round(size_m / res_m))
        self.res = float(res_m)
        self.geometry = GridGeometry(self.res, self.n, -size_m / 2.0, -size_m / 2.0)
        shape = (self.n, self.n)
        self.state = np.zeros(shape, np.uint8)  # CellState
        self.confirmed = np.zeros(shape, bool)  # DITCH_CANDIDATE confirmed (lethal)
        self.cost = np.zeros(shape, np.float32)  # perception cost [0,1] of last observation
        self.mu = np.full(shape, DEFAULT_MU, np.float32)  # braking friction proxy
        self.t_ground = np.full(shape, -np.inf)  # last time observed as GROUND [s]
        self.t_seen = np.full(shape, -np.inf)  # last time observed at all [s]
        self.dyn_until = np.full(shape, -np.inf)  # DYNAMIC hold expiry [s]
        self.cand_hist = np.zeros(shape, np.uint8)  # 3-bit DITCH_CANDIDATE history (in-view frames)
        self.ground_hist = np.zeros(shape, np.uint8)  # 3-bit GROUND history (in-view frames)
        self.assumed = np.zeros(shape, bool)  # GROUND by assumption (launch apron), never observed yet
        self.lethal_hist = np.zeros(shape, np.uint8)  # 3-bit lethal-observation history (in-view frames)
        self.pending = np.zeros(shape, bool)  # lethal seen in 1 frame only: not certified, not lethal
        self.cert_obs = np.zeros(shape, bool)  # GROUND certified by perception's certified_local in >= 1 frame
        self.has_cert_layer = False  # perception supplies certified_local (else all GROUND counts)
        self.void_count = np.zeros(shape, np.uint8)  # consecutive in-wedge frames without any observation
        self._bev_cache: dict[BevGeometry, tuple[np.ndarray, np.ndarray]] = {}
        self.version = 0  # bumps whenever the lethal layer may have changed

    # ------------------------------------------------------------------ geometry
    def recentre(self, x: float, y: float) -> bool:
        """Scroll the map (whole cells) so that A-frame point (x, y) is near the centre.
        Returns True if the map moved. Cells scrolled in are UNSEEN."""
        g = self.geometry
        half = g.n * g.res / 2.0
        cx, cy = g.origin_x + half, g.origin_y + half
        lim = RECENTRE_FRACTION * g.n * g.res
        if abs(x - cx) <= lim and abs(y - cy) <= lim:
            return False
        dx = int(round((x - cx) / g.res))
        dy = int(round((y - cy) / g.res))
        for name, fill in (
            ("state", 0), ("confirmed", False), ("cost", 0.0), ("mu", DEFAULT_MU), ("t_ground", -np.inf),
            ("t_seen", -np.inf), ("dyn_until", -np.inf), ("cand_hist", 0), ("ground_hist", 0), ("assumed", False),
            ("lethal_hist", 0), ("pending", False), ("cert_obs", False), ("void_count", 0),
        ):
            setattr(self, name, _shift(getattr(self, name), dy, dx, fill))
        self.geometry = GridGeometry(g.res, g.n, g.origin_x + dx * g.res, g.origin_y + dy * g.res)
        self.version += 1
        LOG.info("rolling map recentred by (%d, %d) cells; origin now (%.1f, %.1f)", dx, dy, self.geometry.origin_x, self.geometry.origin_y)
        return True

    def seed_apron(self, pose_xy_yaw: tuple[float, float, float], t: float, radius_m: float = LAUNCH_APRON_M) -> None:
        """Certify the launch apron: the vehicle stands on it and the operator placed it there.
        Needed because the camera cannot see the first ~1.5 m ahead of the body."""
        g = self.geometry
        ix, iy = np.meshgrid(np.arange(g.n), np.arange(g.n))
        xc, yc = g.to_xy(ix, iy)
        m = (xc - pose_xy_yaw[0]) ** 2 + (yc - pose_xy_yaw[1]) ** 2 <= radius_m**2
        self.state[m] = int(CellState.GROUND)
        self.cost[m] = 0.0
        self.t_ground[m] = t
        self.t_seen[m] = t
        self.ground_hist[m] = 0b111
        self.assumed[m] = True  # certified, but not evidence of free space for DYNAMIC detection
        self.cert_obs[m] = True
        self.version += 1

    def clear_footprint(self, pose_xy_yaw: tuple[float, float, float], t: float, length_m: float, width_m: float,
                        margin_m: float = FOOTPRINT_MARGIN_M) -> int:
        """Mark the map cells whose centres lie under the body rectangle (A-frame pose; body
        ``length_m`` x ``width_m`` centred on the body origin, + ``margin_m``) as assumed GROUND:
        the vehicle is standing there. Returns the number of lethal / pending cells cleared."""
        x, y, yaw = pose_xy_yaw
        g = self.geometry
        hl, hw = 0.5 * length_m + margin_m, 0.5 * width_m + margin_m
        r = math.hypot(hl, hw)
        ix0, iy0 = g.to_index(np.array([x - r]), np.array([y - r]))
        ix1, iy1 = g.to_index(np.array([x + r]), np.array([y + r]))
        j0, j1 = int(np.clip(ix0[0], 0, g.n)), int(np.clip(ix1[0] + 1, 0, g.n))
        i0, i1 = int(np.clip(iy0[0], 0, g.n)), int(np.clip(iy1[0] + 1, 0, g.n))
        if j1 <= j0 or i1 <= i0:
            return 0
        jj, ii = np.meshgrid(np.arange(j0, j1), np.arange(i0, i1))
        xc, yc = g.to_xy(jj, ii)
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = xc - x, yc - y
        m = (np.abs(c * dx + s * dy) <= hl) & (np.abs(-s * dx + c * dy) <= hw)
        sl = (slice(i0, i1), slice(j0, j1))
        S = self.state[sl]
        bad = m & (np.isin(S, _LETHAL_STATES) | self.confirmed[sl] | self.pending[sl])
        n_bad = int(np.count_nonzero(bad))
        S[m] = int(CellState.GROUND)
        self.confirmed[sl][m] = False
        self.pending[sl][m] = False
        self.lethal_hist[sl][m] = 0
        self.cand_hist[sl][m] = 0
        self.dyn_until[sl][m] = -np.inf
        self.t_ground[sl][m] = t
        self.assumed[sl][m] = True
        self.cert_obs[sl][m] = True
        if n_bad:
            self.version += 1
        return n_bad

    def _bev_xy(self, geo: BevGeometry) -> tuple[np.ndarray, np.ndarray]:
        cached = self._bev_cache.get(geo)
        if cached is None:
            xb, yb = geo.cell_centres_body()
            cached = (np.ascontiguousarray(xb).ravel(), np.ascontiguousarray(yb).ravel())
            self._bev_cache[geo] = cached
        return cached

    # ------------------------------------------------------------------ fusion
    def integrate(
        self,
        t: float,
        pose_xy_yaw: tuple[float, float, float],
        cell_state_local: np.ndarray,
        cost_local: Optional[np.ndarray] = None,
        mu_local: Optional[np.ndarray] = None,
        bev: Optional[BevGeometry] = None,
        certified_local: Optional[np.ndarray] = None,
    ) -> IntegrateStats:
        """Fuse one egocentric BEV observation taken at A-frame pose ``pose_xy_yaw``.

        Args:
            t: frame time [s].
            pose_xy_yaw: body pose in the A-frame (m, m, rad).
            cell_state_local: (Hf, Wl) uint8 CellState grid.
            cost_local: (Hf, Wl) float cost in [0, 1] (inf treated as 1); None -> 0.
            mu_local: (Hf, Wl) braking-friction proxy; None -> keep previous / default.
            certified_local: optional (Hf, Wl) bool, perception's per-cell certification
                (a ditch of the design width would have been detected there); None -> every
                GROUND cell counts as certified (previous behaviour).
            bev: BEV geometry; defaults to the contract-default geometry for this shape.
        """
        stats = IntegrateStats()
        stats.recentred = self.recentre(pose_xy_yaw[0], pose_xy_yaw[1])
        states = np.asarray(cell_state_local, dtype=np.uint8)
        geo = bev or BevGeometry.for_shape(states.shape)
        xb, yb = self._bev_xy(geo)
        s = states.ravel()
        valid = s != int(CellState.UNSEEN)
        if not np.any(valid):
            self._update_void(pose_xy_yaw, np.zeros(0, np.int32), np.zeros(0, np.int32))
            return stats
        s = s[valid]
        x, y, yaw = pose_xy_yaw
        c, sn = math.cos(yaw), math.sin(yaw)
        xv, yv = xb[valid], yb[valid]
        ix, iy = self.geometry.to_index(x + c * xv - sn * yv, y + sn * xv + c * yv)
        inb = self.geometry.in_bounds(ix, iy)
        if not np.all(inb):
            ix, iy, s = ix[inb], iy[inb], s[inb]
            valid_idx = np.flatnonzero(valid)[inb]
        else:
            valid_idx = np.flatnonzero(valid)
        self._update_void(pose_xy_yaw, ix, iy)
        if s.size == 0:
            return stats
        i0, i1, j0, j1 = int(iy.min()), int(iy.max()) + 1, int(ix.min()), int(ix.max()) + 1
        h, w = i1 - i0, j1 - j0
        loc = (iy - i0) * w + (ix - j0)

        # per-cell sub-cell counts per state: (h*w, 16)
        cnt = np.bincount(loc * N_STATES + s, minlength=h * w * N_STATES).reshape(h * w, N_STATES)
        n_seen = cnt.sum(axis=1)
        n_pos, n_dep, n_dyn = cnt[:, int(_S.POSITIVE)], cnt[:, int(_S.DEPRESSION)], cnt[:, int(_S.DYNAMIC)]
        n_leth = n_pos + n_dep + n_dyn
        leth_obs = (n_leth >= LETHAL_MIN_SUBCELLS) | ((n_leth > 0) & (2 * n_leth >= n_seen))
        # which lethal state: DYNAMIC if perception said so for most lethal sub-cells, else POSITIVE / DEPRESSION
        leth_state = np.where(2 * n_dyn > n_leth, int(_S.DYNAMIC), np.where(n_pos >= n_dep, int(_S.POSITIVE), int(_S.DEPRESSION)))
        votes = cnt[:, _NONLETHAL_VOTERS].astype(np.int32) * N_STATES + _RANK[_NONLETHAL_VOTERS][None, :]
        vote_state = _NONLETHAL_VOTERS[np.argmax(votes, axis=1)]
        vote_state = np.where(cnt[:, _NONLETHAL_VOTERS].sum(axis=1) > 0, vote_state, int(_S.OCCLUDED))
        seen = (n_seen > 0).reshape(h, w)
        leth_obs = leth_obs.reshape(h, w) & seen
        leth_state = leth_state.reshape(h, w).astype(np.uint8)
        obs = vote_state.reshape(h, w).astype(np.uint8)  # non-lethal vote (lethal handled separately)
        if cost_local is not None:
            cv = np.nan_to_num(np.asarray(cost_local, np.float32).ravel()[valid_idx], nan=1.0, posinf=1.0)
            nonleth = ~np.isin(s, _LETHAL_STATES)
            csum = np.bincount(loc[nonleth], weights=np.clip(cv[nonleth], 0.0, 1.0), minlength=h * w)
            ncnt = np.bincount(loc[nonleth], minlength=h * w)
            cst = np.where(ncnt > 0, csum / np.maximum(ncnt, 1), 1.0).astype(np.float32).reshape(h, w)
        else:
            cst = np.zeros((h, w), np.float32)
        cert_frac = None
        if certified_local is not None:
            cl = np.asarray(certified_local, bool).ravel()[valid_idx]
            nc = np.bincount(loc, weights=cl.astype(np.float64), minlength=h * w)
            cert_frac = (2 * nc > n_seen).reshape(h, w)  # majority of the sub-cells certified
            self.has_cert_layer = True

        sl = (slice(i0, i1), slice(j0, j1))
        S = self.state[sl]
        S0 = S.copy()
        CONF = self.confirmed[sl]
        TG = self.t_ground[sl]
        DU = self.dyn_until[sl]
        PEND = self.pending[sl]
        persist = np.isin(S0, (_S.POSITIVE, _S.DEPRESSION, _S.DYNAMIC)) | ((S0 == _S.DITCH_CANDIDATE) & CONF)

        LH = self.lethal_hist[sl]
        # support from the previous in-view frame of this cell OR of a neighbour within
        # LETHAL_SUPPORT_RADIUS_CELLS: a moving obstacle (<= ~1 cell per frame) still confirms
        prev_leth = (LH & 1).astype(bool)
        r = LETHAL_SUPPORT_RADIUS_CELLS
        if r > 0 and prev_leth.any():
            k = 2 * r + 1
            prev_leth = cv2.dilate(prev_leth.view(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        LH[seen] = ((LH[seen] << 1) | leth_obs[seen]) & 0b111
        leth_conf = leth_obs & ((_POPCOUNT3[LH] >= CONFIRM_HITS) | prev_leth)
        pend_now = leth_obs & ~leth_conf & ~persist

        is_g = seen & ~leth_obs & (obs == _S.GROUND)
        is_occ = leth_conf & (leth_state != _S.DYNAMIC)
        is_dyn = leth_conf & (leth_state == _S.DYNAMIC)
        is_cand = seen & ~leth_obs & (obs == _S.DITCH_CANDIDATE)
        is_water = seen & ~leth_obs & (obs == _S.WATER)
        is_hidden = seen & ~leth_obs & ((obs == _S.OCCLUDED) | (obs == _S.CREST_SHADOW))

        GH = self.ground_hist[sl]
        GH0 = GH.copy()
        CH = self.cand_hist[sl]
        GH[seen] = ((GH[seen] << 1) | is_g[seen]) & 0b111
        CH[seen] = ((CH[seen] << 1) | is_cand[seen]) & 0b111
        self.t_seen[sl][seen] = t
        self.cost[sl][seen] = cst[seen]
        PEND[seen] = pend_now[seen]
        if cert_frac is not None:
            CO = self.cert_obs[sl]
            CO[is_g & cert_frac] = True
        if mu_local is not None:
            mv = np.asarray(mu_local, np.float32).ravel()[valid_idx]
            mu = np.full(h * w, np.inf, np.float32)
            np.minimum.at(mu, loc, mv)
            mu = mu.reshape(h, w)
            upd = seen & np.isfinite(mu)
            self.mu[sl][upd] = mu[upd]

        # GROUND: clears non-persistent states at once, persistent ones after k-of-3 and hold expiry.
        hold_active = (S0 == _S.DYNAMIC) & (t < DU)
        g_clear = is_g & (~persist | ((_POPCOUNT3[GH] >= CONFIRM_HITS) & ~hold_active))
        S[g_clear] = _S.GROUND
        CONF[g_clear] = False
        # Only *observed* ground counts: an obstacle first seen inside the assumed launch apron
        # is static (it was never observed free), not a moving object.
        # "Well observed free": GROUND in both in-view frames before the (2-frame) occupancy began.
        AS = self.assumed[sl]
        well_free = (GH0 & 0b110) == 0b110
        recent_ground = (S0 == _S.GROUND) & ((t - TG) <= DYNAMIC_RECENT_GROUND_S) & ~AS & well_free
        AS[is_g] = False  # assumed ground becomes evidence of free space only once observed as ground
        TG[is_g] = t

        # Confirmed occupancy (step / rock / depression). New occupancy on well-observed free ground -> DYNAMIC.
        newly = is_occ & ((~persist & recent_ground) | (S0 == _S.DYNAMIC)) if INFER_DYNAMIC else np.zeros_like(is_occ)
        stats.n_new_lethal = int(np.count_nonzero(is_occ & ~persist))
        S[is_occ] = leth_state[is_occ]
        CONF[is_occ] = False
        dyn = newly | is_dyn
        S[dyn] = _S.DYNAMIC
        DU[dyn] = t + DYNAMIC_HOLD_S
        stats.n_dynamic = int(np.count_nonzero(dyn))

        # Ditch candidates: k-of-3 confirmation.
        conf_now = is_cand & (_POPCOUNT3[CH] >= CONFIRM_HITS)
        stats.n_confirmed_ditch = int(np.count_nonzero(conf_now & ~(CONF & (S0 == _S.DITCH_CANDIDATE))))
        S[conf_now] = _S.DITCH_CANDIDATE
        CONF[conf_now] = True
        S[is_cand & ~conf_now & ~persist] = _S.DITCH_CANDIDATE

        # Water, occlusion, crest shadow: never overwrite lethal memory; hidden never erases GROUND.
        S[is_water & ~persist] = _S.WATER
        hid = is_hidden & ~persist & (S0 != _S.GROUND)
        S[hid] = obs[hid]
        if self.has_cert_layer:
            CO = self.cert_obs[sl]
            CO[seen & (S != _S.GROUND)] = False  # a certificate belongs to ground that is still ground

        stats.n_observed_cells = int(np.count_nonzero(seen))
        if stats.n_new_lethal or stats.n_confirmed_ditch or stats.n_dynamic or np.any(g_clear & persist):
            self.version += 1
        return stats

    def _update_void(self, pose_xy_yaw: tuple[float, float, float], ix: np.ndarray, iy: np.ndarray) -> None:
        """Count frames in which map cells inside the certain-view wedge (A-frame pose; range
        [VOID_R_NEAR_M, VOID_R_FAR_M], |bearing| <= VOID_HALF_ANGLE_RAD) got no observation at all;
        ``ix, iy``: map indices of this frame's observed (non-UNSEEN) BEV cells."""
        g = self.geometry
        x, y, yaw = pose_xy_yaw
        r = VOID_R_FAR_M
        bx0, by0 = g.to_index(np.array([x - r]), np.array([y - r]))
        j0, i0 = max(int(bx0[0]), 0), max(int(by0[0]), 0)
        j1, i1 = min(int(bx0[0]) + int(2 * r / g.res) + 2, g.n), min(int(by0[0]) + int(2 * r / g.res) + 2, g.n)
        if j1 <= j0 or i1 <= i0:
            return
        jj, ii = np.meshgrid(np.arange(j0, j1), np.arange(i0, i1))
        xc, yc = g.to_xy(jj, ii)
        dx, dy = xc - x, yc - y
        rng = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx) - yaw
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        wedge = (rng >= VOID_R_NEAR_M) & (rng <= VOID_R_FAR_M) & (np.abs(bearing) <= VOID_HALF_ANGLE_RAD)
        obs = np.zeros(wedge.shape, bool)
        inb = (ix >= j0) & (ix < j1) & (iy >= i0) & (iy < i1)
        obs[iy[inb] - i0, ix[inb] - j0] = True
        VC = self.void_count[i0:i1, j0:j1]
        miss = wedge & ~obs
        VC[miss] = np.minimum(VC[miss].astype(np.int32) + 1, 255).astype(np.uint8)
        VC[obs] = 0

    def void_mask(self) -> np.ndarray:
        """Never-observed cells that sat unobserved inside the certain-view wedge for >= VOID_FRAMES frames."""
        return (self.void_count >= VOID_FRAMES) & (self.state == _S.UNSEEN)

    # ------------------------------------------------------------------ queries
    def lethal_mask(self, use_negobs: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(static_lethal, dynamic_lethal) boolean layers.

        Static lethal = POSITIVE | DEPRESSION | confirmed ditch (the latter only if the
        missing-ground detector is enabled). Dynamic lethal = DYNAMIC."""
        S = self.state
        static = (S == _S.POSITIVE) | (S == _S.DEPRESSION)
        if use_negobs:
            static |= (S == _S.DITCH_CANDIDATE) & self.confirmed
        return static, S == _S.DYNAMIC

    def certified_mask(self, t: float, health_nominal: bool) -> np.ndarray:
        """Cells certified drivable at time t: fresh GROUND, or stale GROUND while health is NOMINAL
        (not pending; only perception-certified ground when that layer is supplied)."""
        ground = (self.state == _S.GROUND) & ~self.pending
        if self.has_cert_layer:
            ground &= self.cert_obs
        if health_nominal:
            return ground
        return ground & ((t - self.t_ground) <= FRESH_S)

    def unknown_masks(self, use_negobs: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(unseen, suspicious): UNSEEN cells, and OCCLUDED / CREST_SHADOW / unconfirmed candidates."""
        S = self.state
        unseen = S == _S.UNSEEN
        cand = S == _S.DITCH_CANDIDATE
        suspicious = (S == _S.OCCLUDED) | (S == _S.CREST_SHADOW) | (cand & ~self.confirmed)
        if not use_negobs:
            suspicious |= cand
        return unseen, suspicious

    def crop(self, x: float, y: float, half_m: float) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        """Axis-aligned crop around (x, y): (state, confirmed, (origin_x, origin_y)) for debug/video."""
        g = self.geometry
        ix, iy = g.to_index(np.array([x - half_m, x + half_m]), np.array([y - half_m, y + half_m]))
        j0, j1 = int(np.clip(ix[0], 0, g.n)), int(np.clip(ix[1], 0, g.n))
        i0, i1 = int(np.clip(iy[0], 0, g.n)), int(np.clip(iy[1], 0, g.n))
        return (
            self.state[i0:i1, j0:j1].copy(),
            self.confirmed[i0:i1, j0:j1].copy(),
            (g.origin_x + j0 * g.res, g.origin_y + i0 * g.res),
        )


def _shift(a: np.ndarray, dy: int, dx: int, fill: Any) -> np.ndarray:
    """Return ``a`` shifted so that new[i, j] = old[i + dy, j + dx]; vacated cells get ``fill``."""
    out = np.full_like(a, fill)
    n0, n1 = a.shape
    if abs(dy) >= n0 or abs(dx) >= n1:
        return out
    src_i = slice(max(dy, 0), n0 + min(dy, 0))
    dst_i = slice(max(-dy, 0), n0 + min(-dy, 0))
    src_j = slice(max(dx, 0), n1 + min(dx, 0))
    dst_j = slice(max(-dx, 0), n1 + min(-dx, 0))
    out[dst_i, dst_j] = a[src_i, src_j]
    return out

