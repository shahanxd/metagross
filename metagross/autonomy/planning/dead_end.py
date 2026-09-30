"""Dead-end memory: regions the vehicle got stuck in, remembered in the odometry (A) frame.

Why: in a field of full-width ditch lines the crossing gap is often 10+ m off the straight route and
not visible from the start. The global planner then routes round the nearer (unseen) end of the
ditch line; at the end of the line the ground ends (terrain edge, persistent void), the vehicle
cannot certify anything ahead, STOP_AND_LOOK does not help, and without memory the planner keeps
sending it into the same corner. With this memory the corner is remembered as blocked and the
cost-to-go field sends the vehicle the other way along the ditch line, toward the gap.

When (decided by :class:`~metagross.autonomy.safety.supervisor.Supervisor`): no progress toward the
goal at the same place even after a STOP_AND_LOOK (no route, route into persistent void / an
unseen terrain edge, or a forward-check deadlock). The node then calls :meth:`DeadEndMemory.mark`.

What is remembered: a disc of radius ``BLOCK_RADIUS_M`` (A-frame metres) centred on the route's
first not-observed-free point within ``FRONTIER_SEARCH_M`` of the vehicle (the frontier it was
stuck at), or on the lookahead point when the route is observed free all the way.

Effect on the global planner (:meth:`apply`, on its down-sampled step-cost grid): inside an active
disc,

* cells that are mostly *unknown* (never observed, void, occluded, crest shadow, unconfirmed
  candidate) become walls: that unknown was looked at and could not be certified;
* observed cells get the multiplier ``BLOCK_SEEN_PENALTY`` (the corridor that led in stays drivable,
  so the vehicle can drive back out, but the route no longer prefers it);
* cells within ``EXEMPT_RADIUS_M`` of the vehicle position at marking time are untouched (the
  vehicle must be able to turn round where it stands).

The disc is shrunk so it keeps ``GOAL_KEEP_M`` clear of the goal (no mark closer than that).
Discs expire ``BLOCK_TTL_S`` after marking (decay: the world may have been misread, e.g. a gap
that could not be certified at the first attempt); at most ``MAX_BLOCKS`` are kept (oldest dropped).
Nothing here knows the world bounds: a terrain edge is only "ground that could not be seen".
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from metagross.autonomy.planning.rolling_map import GridGeometry

LOG = logging.getLogger(__name__)

BLOCK_RADIUS_M = 6.0  # disc radius: larger than the look-around reach (void wedge 2-5 m) so the route must leave the corner
BLOCK_SEEN_PENALTY = 2.0  # step-cost multiplier of observed cells inside a disc
EXEMPT_RADIUS_M = 1.5  # around the vehicle at marking time: untouched (turn-round room, ~ body diagonal + margin)
GOAL_KEEP_M = 3.0  # a disc never reaches closer than this to the goal [m]
MIN_RADIUS_M = 2.0  # below this (goal too close) no disc is recorded
BLOCK_TTL_S = 90.0  # disc lifetime [s] (a ditch-line walk to a gap 10-20 m away takes ~20-40 s at ~1 m/s)
MAX_BLOCKS = 12
FRONTIER_SEARCH_M = 8.0  # search the route this far for the first not-observed-free point
FRONTIER_STEP_M = 0.2  # route sampling step (= map resolution)
FALLBACK_AHEAD_M = 3.0  # centre when the route is observed free within the search distance
UNKNOWN_MAJORITY = 0.5  # a planning cell is "unknown" when at least this fraction of its map cells is


@dataclass(frozen=True, slots=True)
class BlockedDisc:
    """One remembered dead end (A-frame metres / seconds)."""

    x: float
    y: float
    r: float
    t: float
    exempt_x: float
    exempt_y: float


class DeadEndMemory:
    """Blocked-disc memory with expiry. ``version`` bumps whenever the active set changes."""

    def __init__(self, radius_m: float = BLOCK_RADIUS_M, ttl_s: float = BLOCK_TTL_S, max_blocks: int = MAX_BLOCKS,
                 seen_penalty: float = BLOCK_SEEN_PENALTY, exempt_m: float = EXEMPT_RADIUS_M) -> None:
        self.radius_m, self.ttl_s, self.max_blocks = float(radius_m), float(ttl_s), int(max_blocks)
        self.seen_penalty, self.exempt_m = float(seen_penalty), float(exempt_m)
        self.discs: list[BlockedDisc] = []
        self.version = 0
        self.n_marked = 0

    # ------------------------------------------------------------------ bookkeeping
    def expire(self, t: float) -> None:
        """Drop discs older than the TTL."""
        keep = [d for d in self.discs if t - d.t < self.ttl_s]
        if len(keep) != len(self.discs):
            self.discs = keep
            self.version += 1

    def active(self, t: float) -> list[BlockedDisc]:
        self.expire(t)
        return self.discs

    # ------------------------------------------------------------------ marking
    @staticmethod
    def frontier_point(route_xy: np.ndarray, observed_free: Callable[[np.ndarray, np.ndarray], np.ndarray],
                       search_m: float = FRONTIER_SEARCH_M,
                       step_m: float = FRONTIER_STEP_M, fallback_m: float = FALLBACK_AHEAD_M) -> Optional[tuple[float, float]]:
        """First point along ``route_xy`` ((N,2) A-frame, vehicle first) within ``search_m`` where
        ``observed_free(x, y) -> bool array`` is False; the point at ``fallback_m`` (or the route end)
        when the route is observed free that far; None for a degenerate route."""
        route = np.asarray(route_xy, float).reshape(-1, 2)
        if len(route) < 2:
            return None
        seg = np.hypot(*np.diff(route, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if cum[-1] <= 1e-6:
            return None
        s = np.arange(0.0, min(search_m, cum[-1]) + 1e-9, step_m)
        px, py = np.interp(s, cum, route[:, 0]), np.interp(s, cum, route[:, 1])
        free = np.asarray(observed_free(px, py), bool)
        bad = np.flatnonzero(~free)
        if len(bad):
            k = int(bad[0])
            return float(px[k]), float(py[k])
        sf = min(fallback_m, float(cum[-1]))
        return float(np.interp(sf, cum, route[:, 0])), float(np.interp(sf, cum, route[:, 1]))

    def mark(self, t: float, pose_xy: tuple[float, float], centre_xy: tuple[float, float],
             goal_xy: tuple[float, float]) -> Optional[BlockedDisc]:
        """Record a dead end centred at ``centre_xy`` seen from vehicle position ``pose_xy`` at time ``t``.
        Returns the disc, or None when the goal is too close for a meaningful mark."""
        d_goal = math.hypot(centre_xy[0] - goal_xy[0], centre_xy[1] - goal_xy[1])
        r = min(self.radius_m, d_goal - GOAL_KEEP_M)
        if r < MIN_RADIUS_M:
            LOG.info("t=%.2f dead end at (%.1f, %.1f) not recorded: goal %.1f m away", t, centre_xy[0], centre_xy[1], d_goal)
            return None
        disc = BlockedDisc(float(centre_xy[0]), float(centre_xy[1]), float(r), float(t), float(pose_xy[0]), float(pose_xy[1]))
        self.discs.append(disc)
        if len(self.discs) > self.max_blocks:
            self.discs = self.discs[-self.max_blocks:]
        self.version += 1
        self.n_marked += 1
        LOG.info("t=%.2f dead end #%d recorded: centre (%.1f, %.1f) r=%.1f m, vehicle (%.1f, %.1f)",
                 t, self.n_marked, disc.x, disc.y, disc.r, pose_xy[0], pose_xy[1])
        return disc

    # ------------------------------------------------------------------ planner hook
    def apply(self, t: float, steps: np.ndarray, geo: GridGeometry, unknown_frac: np.ndarray) -> np.ndarray:
        """Return ``steps`` (planner step costs on ``geo``, inf = wall) with the active discs applied;
        ``unknown_frac``: per planning cell, fraction of its map cells that are not observed free."""
        discs = self.active(t)
        if not discs:
            return steps
        out = steps.copy()
        n = geo.n
        for d in discs:
            ix0, iy0 = geo.to_index(np.array([d.x - d.r]), np.array([d.y - d.r]))
            ix1, iy1 = geo.to_index(np.array([d.x + d.r]), np.array([d.y + d.r]))
            j0, j1 = int(np.clip(ix0[0], 0, n)), int(np.clip(ix1[0] + 1, 0, n))
            i0, i1 = int(np.clip(iy0[0], 0, n)), int(np.clip(iy1[0] + 1, 0, n))
            if j1 <= j0 or i1 <= i0:
                continue
            jj, ii = np.meshgrid(np.arange(j0, j1), np.arange(i0, i1))
            xc, yc = geo.to_xy(jj, ii)
            inside = (np.hypot(xc - d.x, yc - d.y) <= d.r) & (np.hypot(xc - d.exempt_x, yc - d.exempt_y) > self.exempt_m)
            sub = out[i0:i1, j0:j1]
            unk = unknown_frac[i0:i1, j0:j1] >= UNKNOWN_MAJORITY
            sub[inside & unk] = np.inf
            seen = inside & ~unk
            sub[seen] = sub[seen] * self.seen_penalty
        return out
