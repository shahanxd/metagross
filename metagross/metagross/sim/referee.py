"""Referee: judges every physics step against ground truth and emits timestamped events.

Checks (per physics step)
-------------------------
* ``collision``     vehicle footprint (``VEHICLE.length_m`` x ``VEHICLE.width_m`` rectangle about the
                    body origin) overlaps a lethal static footprint or a dynamic obstacle.
* ``ditch_entry``   any wheel-contact point lies over a cell of a ditch mask, or more than
                    ``DEPRESSION_LETHAL_M`` below the local 1 m median terrain height.
* ``water_entry``   any wheel-contact point on water / mud material.
* ``tip_over``      |roll| > 25 deg or |pitch| > 30 deg.
* ``stuck``         less than 0.5 m displacement over the last 20 s.
* ``timeout``       mission ``timeout_s`` reached.
* ``out_of_bounds`` body origin leaves the terrain raster (0.5 m margin).
* ``arrived_short`` the autonomy declared ARRIVED (``WheelCmd.mode``, reported through
                    :meth:`Referee.note_command`), the vehicle has come to rest (|v| < STOP_SPEED_MPS)
                    and the true goal is still farther than ``success_radius_m``: the stack believes
                    it is at B but its pose estimate drifted (odometry error, not a planning stall).
* ``success``       body origin within ``success_radius_m`` of the true goal B.

Hazard checks take precedence over success on the same step. Which events end the episode is
configurable (default: all of the above). The minimum signed clearance between the vehicle
footprint and any lethal / dynamic footprint is tracked continuously. ``arrived_short`` needs the
world / runner to forward each command's mode via :meth:`Referee.note_command`; without that call
it never fires (post-hoc classification from ``gt/cmds.npz`` is done in
:mod:`metagross.sim.closed_loop_summary`).

:func:`count_false_stops` classifies stops post hoc from the GT log (definition in its docstring).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import maximum_filter

from metagross.config import defaults
from metagross.contracts.messages import VehicleSpec
from metagross.sim.hazards import HazardRaster
from metagross.sim.objects import Footprint, circle_rect_distance, rect_rect_distance
from metagross.sim.terrain import Terrain
from metagross.sim.vehicle import VehicleState

TIP_ROLL_DEG = 25.0
TIP_PITCH_DEG = 30.0
STUCK_WINDOW_S = 20.0
STUCK_MIN_PROGRESS_M = 0.5
OOB_MARGIN_M = 0.5
DEPRESSION_WINDOW_M = 1.0  # side of the local-median window
CLEARANCE_QUERY_M = 6.0  # objects further than this from the body origin are skipped
WATER_MATERIALS = (4, 5)
# Half-width (m) of the max-filter used to gate the median test: covers the +-0.5 m sample lattice,
# bilinear support and nearest-cell rounding, so the gate never hides a true depression.
_DEP_GATE_HALF_M = 0.6
DEFAULT_TERMINAL = frozenset({"collision", "ditch_entry", "water_entry", "tip_over", "stuck", "timeout", "out_of_bounds",
                              "success", "arrived_short"})
ARRIVED_MODE = "ARRIVED"  # DriveMode.ARRIVED.value, as carried by WheelCmd.mode
# False-stop classification
STOP_SPEED_MPS = 0.05
STOP_MIN_S = 1.0
STOP_IGNORE_BEFORE_S = 1.0
FALSE_STOP_LOOKAHEAD_M = 5.0
FALSE_STOP_CORRIDOR_M = 1.2


@dataclass
class Event:
    t: float
    type: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"t": round(self.t, 4), "type": self.type, **self.detail}


class Referee:
    def __init__(self, terrain: Terrain, hazards: HazardRaster, static_feet: list[Footprint], goal_xy: tuple[float, float],
                 success_radius_m: float, timeout_s: float, spec: VehicleSpec = defaults.VEHICLE,
                 terminal: frozenset[str] | set[str] = DEFAULT_TERMINAL):
        self.terrain = terrain
        self.hz = hazards
        # Exact gate for the depression rule: median(window) <= max(window), so a contact can only be
        # > DEPRESSION_LETHAL_M below the local median where it is that far below the local maximum.
        size = 2 * int(math.ceil(_DEP_GATE_HALF_M / terrain.grid.res)) + 1
        self._dep_candidate = (maximum_filter(terrain.height, size=size, mode="nearest") - terrain.height) > defaults.DEPRESSION_LETHAL_M
        self._water_lut = np.zeros(256, bool)
        self._water_lut[list(WATER_MATERIALS)] = True
        self.spec = spec
        self.goal = np.asarray(goal_xy, float)
        self.success_radius = float(success_radius_m)
        self.timeout_s = float(timeout_s)
        self.terminal = frozenset(terminal)
        lethal = [f for f in static_feet if f.lethal]
        circ = [f for f in lethal if f.kind == "circle"]
        self._circ = np.array([[f.cx, f.cy, f.r] for f in circ], float).reshape(-1, 3)
        self._circ_labels = [f.label for f in circ]
        self._rects = [f for f in lethal if f.kind == "rect"]
        self.events: list[Event] = []
        self.min_clearance = math.inf
        self.clearance = math.inf
        self.collisions = 0
        self.ditch_entries = 0
        self.water_entries = 0
        self._in_contact = False
        self._in_ditch = False
        self._in_water = False
        self._hist: deque[tuple[float, float, float]] = deque()
        self.done = False
        self.success = False
        self.failure_type: str | None = None
        self.t_end: float | None = None
        self.t_arrived_declared: float | None = None  # first time the autonomy reported mode ARRIVED [s]

    # ------------------------------------------------------------------ helpers
    def vehicle_footprint(self, s: VehicleState) -> Footprint:
        return Footprint("rect", s.x, s.y, hl=0.5 * self.spec.length_m, hw=0.5 * self.spec.width_m, yaw=s.yaw, label="ego")

    def clearance_to(self, ego: Footprint, dyn_feet: list[Footprint]) -> tuple[float, str]:
        """Minimum signed distance (m) from the ego footprint to lethal / dynamic footprints."""
        best, label = math.inf, ""
        if len(self._circ):
            d2 = (self._circ[:, 0] - ego.cx) ** 2 + (self._circ[:, 1] - ego.cy) ** 2
            near = np.flatnonzero(d2 < CLEARANCE_QUERY_M ** 2)
            if len(near):
                c = self._circ[near]
                d = circle_rect_distance(c[:, 0], c[:, 1], c[:, 2], ego)
                k = int(np.argmin(d))
                best, label = float(d[k]), self._circ_labels[near[k]]
        for fp in self._rects + list(dyn_feet):
            if math.hypot(fp.cx - ego.cx, fp.cy - ego.cy) - fp.bound_r > CLEARANCE_QUERY_M:
                continue
            d = circle_rect_distance(fp.cx, fp.cy, fp.r, ego) if fp.kind == "circle" else rect_rect_distance(fp, ego)
            d = float(d)
            if d < best:
                best, label = d, fp.label
        return best, label

    def contact_hazards(self, contacts: np.ndarray) -> tuple[bool, bool, dict]:
        """(ditch, water, detail) for the (4, 2) wheel-contact points."""
        i, j = self.hz.grid.xy_to_ij(contacts[:, 0], contacts[:, 1])
        in_mask = self.hz.ditch[i, j]
        cand = self._dep_candidate[i, j]
        if cand.any():
            h = self.terrain.height_at(contacts[:, 0], contacts[:, 1])
            med = self.terrain.local_median_height(contacts[:, 0], contacts[:, 1], 0.5 * DEPRESSION_WINDOW_M)
            depress = (med - h) > defaults.DEPRESSION_LETHAL_M
        else:
            h = med = np.zeros(len(contacts))
            depress = np.zeros(len(contacts), bool)
        ditch = bool(np.any(in_mask | depress))
        water = bool(self._water_lut[self.terrain.material[i, j]].any())
        detail = {}
        if ditch:
            if not cand.any():
                h = self.terrain.height_at(contacts[:, 0], contacts[:, 1])
                med = self.terrain.local_median_height(contacts[:, 0], contacts[:, 1], 0.5 * DEPRESSION_WINDOW_M)
            k = int(np.argmax(np.where(in_mask | depress, med - h, -np.inf)))
            detail = {"wheel": ["FL", "FR", "RL", "RR"][k], "depth_below_median_m": round(float(med[k] - h[k]), 3),
                      "in_ditch_mask": bool(in_mask[k])}
        return ditch, water, detail

    def _emit(self, t: float, typ: str, **detail) -> Event:
        ev = Event(t, typ, detail)
        self.events.append(ev)
        if typ in self.terminal and not self.done:
            self.done = True
            self.t_end = t
            if typ == "success":
                self.success = True
            else:
                self.failure_type = typ
        return ev

    def emit_external(self, t: float, typ: str, **detail) -> Event:
        """Record a non-referee event (e.g. dynamic trigger, lighting) in the same log."""
        ev = Event(t, typ, detail)
        self.events.append(ev)
        return ev

    def note_command(self, t: float, mode: object) -> None:
        """Record the drive mode of an autonomy command issued at sim time ``t`` [s].

        ``mode`` is a ``DriveMode`` or its string value. Only the first ARRIVED declaration is kept;
        :meth:`check` turns it into ``arrived_short`` once the vehicle is at rest outside the
        success radius."""
        m = getattr(mode, "value", mode)
        if str(m) == ARRIVED_MODE and self.t_arrived_declared is None:
            self.t_arrived_declared = float(t)

    def force_end(self, t: float, failure_type: str) -> None:
        """End the episode for a reason outside the referee (autonomy error, wall timeout)."""
        if not self.done:
            self.events.append(Event(t, failure_type))
            self.done = True
            self.t_end = t
            self.failure_type = failure_type

    # ------------------------------------------------------------------ per-step
    def check(self, s: VehicleState, contacts: np.ndarray, dyn_feet: list[Footprint]) -> list[Event]:
        if self.done:
            return []
        t = s.t
        n0 = len(self.events)
        ego = self.vehicle_footprint(s)
        clr, lab = self.clearance_to(ego, dyn_feet)
        self.clearance = clr
        self.min_clearance = min(self.min_clearance, clr)
        contact = clr <= 0.0
        if contact and not self._in_contact:
            self.collisions += 1
            self._emit(t, "collision", object=lab, penetration_m=round(-clr, 3), speed_mps=round(s.v, 3))
        self._in_contact = contact
        ditch, water, det = self.contact_hazards(contacts)
        if ditch and not self._in_ditch:
            self.ditch_entries += 1
            self._emit(t, "ditch_entry", **det)
        self._in_ditch = ditch
        if water and not self._in_water:
            self.water_entries += 1
            self._emit(t, "water_entry")
        self._in_water = water
        if abs(s.roll) > math.radians(TIP_ROLL_DEG) or abs(s.pitch) > math.radians(TIP_PITCH_DEG):
            self._emit(t, "tip_over", roll_deg=round(math.degrees(s.roll), 2), pitch_deg=round(math.degrees(s.pitch), 2))
        if not self.hz.grid.contains(s.x, s.y, margin=OOB_MARGIN_M):
            self._emit(t, "out_of_bounds")
        # Stuck: displacement over the trailing window (history sampled at ~1 Hz).
        if not self._hist or t - self._hist[-1][0] >= 1.0 - 1e-9:
            self._hist.append((t, s.x, s.y))
        while len(self._hist) > 1 and self._hist[1][0] <= t - STUCK_WINDOW_S:
            self._hist.popleft()
        t_old, x_old, y_old = self._hist[0]
        if t - t_old >= STUCK_WINDOW_S - 1e-9 and math.hypot(s.x - x_old, s.y - y_old) < STUCK_MIN_PROGRESS_M:
            self._emit(t, "stuck", window_s=STUCK_WINDOW_S)
        if t >= self.timeout_s - 1e-9:
            self._emit(t, "timeout")
        d_goal = math.hypot(s.x - self.goal[0], s.y - self.goal[1])
        if not self.done and d_goal <= self.success_radius:
            self._emit(t, "success")
        if (not self.done and self.t_arrived_declared is not None and abs(s.v) < STOP_SPEED_MPS
                and d_goal > self.success_radius):
            self._emit(t, "arrived_short", final_error_m=round(d_goal, 3), t_declared=round(self.t_arrived_declared, 3))
        return self.events[n0:]


def count_false_stops(t: np.ndarray, x: np.ndarray, y: np.ndarray, yaw: np.ndarray, v: np.ndarray, hz: HazardRaster,
                      goal_xy: tuple[float, float], success_radius_m: float,
                      dyn_xy: np.ndarray | None = None, lighting_events: list[dict] | None = None) -> tuple[int, list[dict]]:
    """Count stops that ground truth does not justify.

    A *stop* is a maximal interval with |v| < STOP_SPEED_MPS lasting >= STOP_MIN_S that starts after
    STOP_IGNORE_BEFORE_S and outside the goal radius. It is *justified* if, at its onset, any GT
    hazard (lethal / ditch / water / dynamic-rest cell) lies in the forward corridor
    FALSE_STOP_LOOKAHEAD_M long and FALSE_STOP_CORRIDOR_M wide, a dynamic obstacle is within the
    look-ahead distance, or a lighting event (glare / dim / dust) is active. Otherwise it is a false
    stop. ``dyn_xy``: (T, n_dyn, 2) dynamic positions aligned with ``t`` (optional).
    """
    stopped = np.abs(v) < STOP_SPEED_MPS
    out: list[dict] = []
    k, n = 0, len(t)
    along = np.arange(0.3, FALSE_STOP_LOOKAHEAD_M + 1e-9, 0.1)
    across = np.arange(-0.5 * FALSE_STOP_CORRIDOR_M, 0.5 * FALSE_STOP_CORRIDOR_M + 1e-9, 0.1)
    A, C = np.meshgrid(along, across)
    haz = hz.hazard
    while k < n:
        if not stopped[k]:
            k += 1
            continue
        k1 = k
        while k1 + 1 < n and stopped[k1 + 1]:
            k1 += 1
        dur = t[k1] - t[k]
        if dur >= STOP_MIN_S and t[k] >= STOP_IGNORE_BEFORE_S and math.hypot(x[k] - goal_xy[0], y[k] - goal_xy[1]) > success_radius_m:
            c, s_ = math.cos(yaw[k]), math.sin(yaw[k])
            px = x[k] + c * A - s_ * C
            py = y[k] + s_ * A + c * C
            ii, jj = hz.grid.xy_to_ij(px, py)
            reason = None
            if haz[ii, jj].any():
                reason = "hazard_ahead"
            elif dyn_xy is not None and dyn_xy.shape[1] > 0 and np.any(
                    np.hypot(dyn_xy[k, :, 0] - x[k], dyn_xy[k, :, 1] - y[k]) <= FALSE_STOP_LOOKAHEAD_M):
                reason = "dynamic_near"
            elif any(ev["t0"] <= t[k] <= ev["t1"] for ev in (lighting_events or [])):
                reason = "lighting_event"
            out.append({"t": round(float(t[k]), 3), "duration_s": round(float(dur), 3), "justified": reason is not None,
                        "reason": reason or "none"})
        k = k1 + 1
    return sum(1 for s in out if not s["justified"]), out
