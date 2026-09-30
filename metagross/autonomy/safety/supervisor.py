"""Safety supervisor: drive-mode state machine, watchdog and forward footprint check.

Modes (:class:`~metagross.contracts.messages.DriveMode`) from the localisation health
score q in [0, 1] (1 - p_fail), with hysteresis:

======================  ===============================  ===============================
mode                    entry                            effect
======================  ===============================  ===============================
NOMINAL                 q > 0.7 (upgrade: q > 0.75, 1 s)  full governor speed
CAUTION                 0.4 < q <= 0.7                   x0.5 speed, costs inflated x1.5
DEGRADED                0.2 < q <= 0.4                   x0.25 speed in hops (2 s go, 1 s look)
SAFE_STOP (latched)     q < 0.2 for 3 s, IMMOBILISED,    zero command until RESUME/GO
                        no certified progress after
                        ``max_dead_ends`` dead ends,
                        operator ESTOP
STOP_AND_LOOK           3 s without 0.3 m of progress    rotate +45, -45, back to 0 deg in
                        toward the goal (cost-to-go, or  place to certify adjacent ground
                        distance when there is no route;
                        turning onto the route counts
                        for up to ``align_max_s``)
DEAD_END (one tick,     still no progress after          decision.dead_end = True: the node
mode unchanged)         ``looks_before_dead_end`` looks  records the corner in the dead-end
                                                         memory, the planner routes elsewhere
ARRIVED (latched)       within arrive radius of goal     zero command
HOLD                    operator HOLD                    zero command until RESUME/GO
======================  ===============================  ===============================

Stall handling alternates look -> dead end -> look -> ...; distance progress resets the count, and
``max_dead_ends`` dead ends without distance progress latch SAFE_STOP. With the dead-end memory
disabled (``dead_end_enabled=False``) the previous rule applies: ``max_look_cycles`` looks, then
SAFE_STOP.

Downgrades are immediate; upgrades need q above threshold + ``hysteresis`` for
``upgrade_hold_s``. Every mode change is logged with a short reason string.

Independent of mode: the watchdog stops the vehicle when the sensor frame gap
exceeds ``watchdog_gap_s``, and :meth:`Supervisor.gate` simulates the final
(v, omega) for ``fwd_horizon_s`` and brakes if the footprint would enter a lethal
core (unless the motion increases clearance).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from metagross.autonomy.planning.costmap import PlanningMaps
from metagross.config.defaults import VEHICLE
from metagross.contracts.messages import DriveMode, OperatorAction, OperatorCmd

LOG = logging.getLogger(__name__)

ROT_MARGIN_M = 0.05  # in-place rotation: body rectangle grown by this much must not sweep a lethal core [m]
ROT_SAMPLE_M = 0.1  # body sample spacing for that check (half the 0.2 m map cell) [m]

_LEVELS = (DriveMode.NOMINAL, DriveMode.CAUTION, DriveMode.DEGRADED)


@dataclass(frozen=True, slots=True)
class SupervisorParams:
    q_nominal: float = 0.7
    q_caution: float = 0.4
    q_degraded: float = 0.2
    hysteresis: float = 0.05
    upgrade_hold_s: float = 1.0
    safe_stop_hold_s: float = 3.0
    caution_speed: float = 0.5
    caution_inflate: float = 1.5
    degraded_speed: float = 0.25
    hop_drive_s: float = 2.0
    hop_pause_s: float = 1.0
    no_progress_s: float = 3.0
    progress_min_m: float = 0.3
    heading_progress_rad: float = 0.15  # shrinking the heading error to the route by this much counts as progress
    launch_hold_s: float = 0.0  # stand still this long at launch (gyro-bias calibration); the node sets LAUNCH_HOLD_S
    stall_escape_m: float = 1.5  # look / dead-end counts reset only once the vehicle got this far from where it first stalled
    blind_stall_s: float = 2.0  # speed held at zero by "cannot see ahead" this long = stall (skip the align window)
    align_max_s: float = 5.0  # heading progress restarts the clock only this long after the last distance progress
    look_angle_rad: float = math.radians(45.0)
    look_yaw_rate: float = 0.6
    look_tol_rad: float = 0.08
    look_timeout_s: float = 10.0
    max_look_cycles: int = 2  # dead-end memory disabled: looks before SAFE_STOP
    looks_before_dead_end: int = 1  # dead-end memory enabled: looks before a stall is declared a dead end
    max_dead_ends: int = 6  # dead ends without distance progress before SAFE_STOP
    watchdog_gap_s: float = 0.5
    immobile_cmd_v: float = 0.2
    immobile_meas_v: float = 0.05
    immobile_s: float = 3.0
    arrive_fraction: float = 0.25  # stop at this fraction of the mission success radius (pose-error margin: DEV odometry drift ~1 m)
    fwd_horizon_s: float = 1.0
    fwd_dt_s: float = 0.1
    footprint_offsets_m: tuple[float, ...] = (-0.27, 0.0, 0.27)
    footprint_radius_m: float = 0.33
    body_length_m: float = VEHICLE.length_m  # in-place rotation check: the body rectangle itself
    body_width_m: float = VEHICLE.width_m
    rot_margin_m: float = ROT_MARGIN_M
    rot_sample_m: float = ROT_SAMPLE_M


@dataclass(slots=True)
class SupervisorDecision:
    mode: DriveMode
    reason: str
    speed_factor: float = 1.0  # multiplies v_cap
    inflate_scale: float = 1.0  # costmap inflation scale
    override: Optional[tuple[float, float]] = None  # (v, omega) replacing the planner output
    emergency: bool = False  # brake with full deceleration
    info: dict[str, float] = field(default_factory=dict)
    dead_end: bool = False  # this tick declared the current corner a dead end (node records it)


class Supervisor:
    """Stateful supervisor. One :meth:`update` per tick before planning, one :meth:`gate` after.
    ``dead_end_enabled``: stalls that a look did not resolve become dead ends (see module docstring)."""

    def __init__(self, params: SupervisorParams = SupervisorParams(), dead_end_enabled: bool = True) -> None:
        self.p = params
        self.dead_end_enabled = bool(dead_end_enabled)
        self.reset((0.0, 0.0), 2.0)

    def reset(self, goal_xy: tuple[float, float], success_radius_m: float) -> None:
        self.goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
        self.arrive_radius_m = self.p.arrive_fraction * success_radius_m
        self.mode = DriveMode.NOMINAL
        self.reason = "START"
        self._level = 0
        self._upgrade_since: Optional[float] = None
        self._lowq_since: Optional[float] = None
        self._immobile_since: Optional[float] = None
        self._latched: Optional[str] = None  # SAFE_STOP reason
        self._estop = False
        self._hold = False
        self._arrived = False
        self._anchor: Optional[tuple[float, float, str]] = None  # (t, progress metric, metric kind)
        self._anchor_herr: Optional[float] = None  # |heading error| to the route at the last (heading) progress [rad]
        self._t_dist_anchor = 0.0  # time of the last distance progress (bounds heading-progress credit) [s]
        self._look: Optional[dict] = None
        self._look_cycles = 0
        self._t_start: Optional[float] = None
        self._stall_xy: Optional[tuple[float, float]] = None  # where the current stall sequence began
        self._progress_xy: Optional[tuple[float, float]] = None  # vehicle position at the last real progress
        self._look_home: Optional[float] = None  # heading every look returns to (set at the first look since progress)
        self._dead_ends_since_progress = 0
        self.n_dead_ends = 0
        self._degraded_since: Optional[float] = None
        self._metric: tuple[float, str] = (math.inf, "dist")
        self.transitions: list[tuple[float, str, str]] = []  # (t, mode, reason)

    # ------------------------------------------------------------------ operator
    def operator(self, cmd: OperatorCmd) -> None:
        """Apply an operator command (HOLD / ESTOP latch; RESUME / GO clear latches)."""
        if cmd.action == OperatorAction.ESTOP:
            self._estop = True
        elif cmd.action == OperatorAction.HOLD:
            self._hold = True
        elif cmd.action in (OperatorAction.RESUME, OperatorAction.GO):
            self._estop = self._hold = False
            self._latched = None
            self._lowq_since = self._immobile_since = None
            self._look, self._look_cycles, self._anchor = None, 0, None
            if cmd.goal_xy_a is not None:
                self.goal_xy = (float(cmd.goal_xy_a[0]), float(cmd.goal_xy_a[1]))
                self._arrived = False
        LOG.info("operator %s", cmd.action.value)

    # ------------------------------------------------------------------ helpers
    def _set(self, t: float, mode: DriveMode, reason: str) -> None:
        if mode != self.mode:
            LOG.info("t=%.2f mode %s -> %s (%s)", t, self.mode.value, mode.value, reason)
            self.transitions.append((t, mode.value, reason))
        self.mode, self.reason = mode, reason

    def _health_level(self, t: float, q: float) -> int:
        p = self.p
        thr = (p.q_nominal, p.q_caution, p.q_degraded)  # lower bound of level 0, 1, 2
        target = 0 if q > thr[0] else 1 if q > thr[1] else 2
        if target > self._level:
            self._level, self._upgrade_since = target, None
        elif target < self._level:
            need = thr[self._level - 1] + p.hysteresis  # one level at a time
            if q > need:
                if self._upgrade_since is None:
                    self._upgrade_since = t
                elif t - self._upgrade_since >= p.upgrade_hold_s:
                    self._level, self._upgrade_since = self._level - 1, None
            else:
                self._upgrade_since = None
        else:
            self._upgrade_since = None
        return self._level

    def _stop(self, t: float, mode: DriveMode, reason: str) -> SupervisorDecision:
        self._set(t, mode, reason)
        self._anchor = None
        return SupervisorDecision(mode, reason, 0.0, 1.0, (0.0, 0.0), emergency=mode == DriveMode.SAFE_STOP)

    # ------------------------------------------------------------------ main update
    def update(
        self,
        t: float,
        q: float,
        pose: tuple[float, float, float],
        v_cmd_prev: float,
        speed_meas: float,
        frame_gap_s: float = 0.0,
        immobilised_hint: bool = False,
        progress_metric: Optional[float] = None,
        metric_kind: str = "dist",
        metric_shift: float = 0.0,
        heading_err: Optional[float] = None,
        blind_s: float = 0.0,
    ) -> SupervisorDecision:
        """Advance the state machine. ``q`` in [0,1]; pose A-frame; speeds [m/s]; gap [s];
        ``immobilised_hint``: the localiser's wheel-vs-VO immobilisation flag;
        ``progress_metric``: how far the goal still is (global cost-to-go ~ metres, kind "ctg");
        defaults to the straight-line distance to the goal (kind "dist"); ``metric_shift``: increase
        of the metric caused by new map information (a replan lengthening the route), which moves
        the progress baseline. Driving away from the goal never moves it.
        ``heading_err``: |angle| between the vehicle heading and the direction to the route's
        lookahead point [rad]; turning onto the route (the error shrinking by
        ``heading_progress_rad``) counts as progress, so the no-progress clock effectively starts
        after alignment (None: distance progress only).
        ``blind_s``: how long the governor has held the speed cap at ~0 because nothing ahead is
        certified or visible [s]; from ``blind_stall_s`` on, turning is no longer progress and the
        stall is declared after ``blind_stall_s`` (not ``no_progress_s``)."""
        p = self.p
        if self._estop:
            return self._stop(t, DriveMode.SAFE_STOP, "ESTOP operator")
        if self._hold:
            return self._stop(t, DriveMode.HOLD, "HOLD operator")
        if self._latched is not None:
            return self._stop(t, DriveMode.SAFE_STOP, self._latched)
        if self._t_start is None:
            self._t_start = t
        if t - self._t_start < p.launch_hold_s:
            # Gyro calibration at A: a 0.06 deg/s turn-on bias alone is ~2 m cross-track after 45 m
            self._anchor = None
            self._set(t, DriveMode.NOMINAL, "START gyro cal")
            return SupervisorDecision(DriveMode.NOMINAL, "START gyro cal", 0.0, 1.0, (0.0, 0.0))
        d_goal = math.hypot(self.goal_xy[0] - pose[0], self.goal_xy[1] - pose[1])
        if self._arrived or d_goal <= self.arrive_radius_m:
            self._arrived = True
            return self._stop(t, DriveMode.ARRIVED, f"ARRIVED d={d_goal:.1f}m")

        # health -> SAFE_STOP after sustained very low q
        if q < p.q_degraded:
            self._lowq_since = t if self._lowq_since is None else self._lowq_since
            if t - self._lowq_since >= p.safe_stop_hold_s:
                self._latched = f"HEALTH q={q:.2f}"
                return self._stop(t, DriveMode.SAFE_STOP, self._latched)
        else:
            self._lowq_since = None
        # immobilised: commanded motion but no measured motion (or persistent slip)
        if abs(v_cmd_prev) > p.immobile_cmd_v and (abs(speed_meas) < p.immobile_meas_v or immobilised_hint):
            self._immobile_since = t if self._immobile_since is None else self._immobile_since
            if t - self._immobile_since >= p.immobile_s:
                self._latched = "IMMOBILISED"
                return self._stop(t, DriveMode.SAFE_STOP, self._latched)
        else:
            self._immobile_since = None

        if frame_gap_s > p.watchdog_gap_s:
            dec = SupervisorDecision(self.mode, f"WATCHDOG gap={frame_gap_s:.2f}s", 0.0, 1.0, (0.0, 0.0), emergency=True)
            self.reason = dec.reason
            return dec

        level = self._health_level(t, q)
        mode = _LEVELS[level]
        speed = (1.0, p.caution_speed, p.degraded_speed)[level]
        inflate = (1.0, p.caution_inflate, p.caution_inflate)[level]
        reason = ("OK", f"CAUTION q={q:.2f}", f"DEGRADED q={q:.2f}")[level]

        # STOP_AND_LOOK: progress means the goal got closer, not merely that the wheels turned
        finite = progress_metric is not None and math.isfinite(progress_metric)
        metric = float(progress_metric) if finite else d_goal
        kind = metric_kind if finite else "dist"
        self._metric = (metric, kind)
        if self._look is not None:
            return self._look_step(t, pose, inflate)
        herr = abs(float(heading_err)) if heading_err is not None and math.isfinite(heading_err) else None
        if self._anchor is not None and self._anchor[2] == kind and metric_shift > 0.0:
            self._anchor = (self._anchor[0], self._anchor[1] + metric_shift, kind)  # route got longer: same clock
        moved = (math.inf if self._progress_xy is None
                 else math.hypot(pose[0] - self._progress_xy[0], pose[1] - self._progress_xy[1]))
        if self._anchor is None or self._anchor[2] != kind:
            self._anchor = (t, metric, kind)
            self._anchor_herr, self._t_dist_anchor = herr, t
            self._progress_xy = (pose[0], pose[1])
        elif metric <= self._anchor[1] - p.progress_min_m and moved < p.progress_min_m:
            # the goal got "closer" only because the map changed (a look revealed a shorter route):
            # lower the baseline, keep the clock and the look / dead-end counts
            self._anchor = (self._anchor[0], metric, kind)
        elif metric <= self._anchor[1] - p.progress_min_m:
            self._anchor = (t, metric, kind)
            self._anchor_herr, self._t_dist_anchor = herr, t
            self._progress_xy = (pose[0], pose[1])
            escaped = (self._stall_xy is None
                       or math.hypot(pose[0] - self._stall_xy[0], pose[1] - self._stall_xy[1]) >= p.stall_escape_m)
            if escaped:  # creeping a few decimetres after each look is not getting out of the corner
                self._look_cycles = 0
                self._dead_ends_since_progress = 0
                self._look_home = None
                self._stall_xy = None
        elif (herr is not None and t - self._t_dist_anchor < p.align_max_s and blind_s < p.blind_stall_s
              and (self._anchor_herr is None or herr <= self._anchor_herr - p.heading_progress_rad)):
            # turning onto the route is progress: restart the clock, keep the distance baseline
            # (bounded by align_max_s since the last distance progress, so oscillation cannot hide a stall)
            self._anchor = (t, self._anchor[1], kind)
            self._anchor_herr = herr
        elif (herr is not None and self._anchor_herr is not None and herr > self._anchor_herr
              and metric_shift > 0.0):
            # route swung (replan: the route got longer): measure turning progress from the new error.
            # Without a replan a growing error is the vehicle wiggling; raising the baseline then let the
            # next swing back count as progress and hid a stall facing an unseeable ditch (DEV 109/121/127).
            self._anchor_herr = herr
        elif t - self._anchor[0] >= (p.blind_stall_s if blind_s >= p.blind_stall_s else p.no_progress_s):
            if self.dead_end_enabled and self._look_cycles >= p.looks_before_dead_end:
                if self._dead_ends_since_progress >= p.max_dead_ends:
                    self._latched = "NO_CERTIFIED_PROGRESS"
                    return self._stop(t, DriveMode.SAFE_STOP, self._latched)
                # a look did not help: declare a dead end (the node remembers the corner and the
                # planner routes elsewhere); fresh progress window, next stall looks again first
                self._dead_ends_since_progress += 1
                self.n_dead_ends += 1
                self._look_cycles = 0
                self._anchor = (t, metric, kind)
                self._anchor_herr, self._t_dist_anchor = herr, t
                reason = f"DEAD_END {self._dead_ends_since_progress}"
                self._set(t, mode, reason)
                return SupervisorDecision(mode, reason, 0.0, inflate, (0.0, 0.0), dead_end=True,
                                          info={"q": q, "level": float(level)})
            if not self.dead_end_enabled and self._look_cycles >= p.max_look_cycles:
                self._latched = "NO_CERTIFIED_PROGRESS"
                return self._stop(t, DriveMode.SAFE_STOP, self._latched)
            # Return to one fixed heading across repeated looks: each look ends within look_tol_rad of
            # its target, and restarting from the reached heading let that error accumulate.
            if self._stall_xy is None:
                self._stall_xy = (pose[0], pose[1])
            y0 = pose[2] if self._look_home is None else self._look_home
            self._look_home = y0
            self._look = {"targets": [y0 + p.look_angle_rad, y0 - p.look_angle_rad, y0], "k": 0, "t0": t}
            self._set(t, DriveMode.STOP_AND_LOOK, "NO_PROGRESS look")
            return self._look_step(t, pose, inflate)

        if level == 2:
            self._degraded_since = t if self._degraded_since is None else self._degraded_since
            phase = (t - self._degraded_since) % (p.hop_drive_s + p.hop_pause_s)
            if phase >= p.hop_drive_s:
                speed, reason = 0.0, f"DEGRADED hop-pause q={q:.2f}"
                self._anchor = (t, metric, kind)  # pauses are not lack of progress
        else:
            self._degraded_since = None
        self._set(t, mode, reason)
        return SupervisorDecision(mode, reason, speed, inflate, None, info={"q": q, "level": float(level)})

    def _look_step(self, t: float, pose: tuple[float, float, float], inflate: float) -> SupervisorDecision:
        p = self.p
        lk = self._look
        assert lk is not None
        while lk["k"] < len(lk["targets"]):
            tgt = lk["targets"][lk["k"]]
            err = math.atan2(math.sin(tgt - pose[2]), math.cos(tgt - pose[2]))
            if abs(err) > p.look_tol_rad and t - lk["t0"] < p.look_timeout_s:
                w = float(np.clip(2.0 * err, -p.look_yaw_rate, p.look_yaw_rate))
                reason = f"STOP_AND_LOOK {lk['k'] + 1}/{len(lk['targets'])}"
                self._set(t, DriveMode.STOP_AND_LOOK, reason)
                return SupervisorDecision(DriveMode.STOP_AND_LOOK, reason, 0.0, inflate, (0.0, w))
            lk["k"] += 1
        # look finished: resume and give the planner a fresh progress window
        self._look = None
        self._look_cycles += 1
        self._anchor = (t, *self._metric)
        self._anchor_herr, self._t_dist_anchor = None, t
        mode = _LEVELS[self._level]
        reason = f"RESUME after look {self._look_cycles}"
        self._set(t, mode, reason)
        return SupervisorDecision(mode, reason, 0.0, inflate, (0.0, 0.0))

    # ------------------------------------------------------------------ final gate
    def forward_clearance(self, v: float, w: float, pose: tuple[float, float, float], maps: PlanningMaps) -> tuple[float, float]:
        """(clearance now, min clearance over the horizon) [m] of the footprint circles to lethal cores,
        for constant (v, w) over ``fwd_horizon_s``."""
        p = self.p
        tt = np.arange(0.0, p.fwd_horizon_s + 1e-9, p.fwd_dt_s)
        yaw = pose[2] + w * tt
        if abs(w) > 1e-6:
            x = pose[0] + v / w * (np.sin(yaw) - math.sin(pose[2]))
            y = pose[1] - v / w * (np.cos(yaw) - math.cos(pose[2]))
        else:
            x = pose[0] + v * tt * math.cos(pose[2])
            y = pose[1] + v * tt * math.sin(pose[2])
        off = np.asarray(p.footprint_offsets_m)
        cx = x[:, None] + off * np.cos(yaw)[:, None]
        cy = y[:, None] + off * np.sin(yaw)[:, None]
        clr = maps.sample(maps.clearance_m, cx, cy, fill=np.float32(1e3)).min(axis=1)
        return float(clr[0]), float(clr.min())

    def rotation_clear(self, w: float, pose: tuple[float, float, float], maps: PlanningMaps) -> bool:
        """True when rotating in place at ``w`` [rad/s] for ``fwd_horizon_s`` sweeps the body rectangle
        (grown by ``rot_margin_m``) over no lethal-core cell (un-inflated; outside the map counts as clear,
        like :meth:`forward_clearance`). This is the physical swept footprint of the turn, so it can
        allow a turn that the conservative footprint circles block when an obstacle is close ahead."""
        p = self.p
        hl, hw = 0.5 * p.body_length_m + p.rot_margin_m, 0.5 * p.body_width_m + p.rot_margin_m
        bx = np.linspace(-hl, hl, max(2, int(math.ceil(2 * hl / p.rot_sample_m)) + 1))
        by = np.linspace(-hw, hw, max(2, int(math.ceil(2 * hw / p.rot_sample_m)) + 1))
        gx, gy = (a.ravel() for a in np.meshgrid(bx, by))
        yaw = pose[2] + w * np.arange(0.0, p.fwd_horizon_s + 1e-9, p.fwd_dt_s)
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        x = pose[0] + c * gx - s * gy
        y = pose[1] + s * gx + c * gy
        return not bool(np.any(maps.sample(maps.lethal_core, x, y, fill=False)))

    def gate(self, dec: SupervisorDecision, v: float, w: float, pose: tuple[float, float, float],
             maps: PlanningMaps) -> tuple[float, float, bool, Optional[str]]:
        """Apply overrides and the forward footprint check. Returns (v, w, emergency, brake_reason).
        An in-place rotation (v = 0, e.g. STOP_AND_LOOK) blocked by the footprint circles is still
        allowed when its swept body rectangle is clear (:meth:`rotation_clear`): forward being
        blocked must not deadlock the look-around."""
        if dec.override is not None:
            v, w = dec.override
        if v == 0.0 and w == 0.0:
            return 0.0, 0.0, dec.emergency, None
        now, fut = self.forward_clearance(v, w, pose, maps)
        r = self.p.footprint_radius_m
        if v == 0.0 and fut < r:
            # In-place rotation near a lethal core: the circles barely move (and on the map grid often
            # show no decrease), but the body corners sweep; judge by the swept rectangle alone.
            if self.rotation_clear(w, pose, maps):
                return 0.0, w, dec.emergency, None
            return 0.0, 0.0, True, f"FWD_CHECK rot clr={max(fut, 0.0):.2f}m"
        if fut < r and fut < now - 1e-3:
            reason = f"FWD_CHECK clr={max(fut, 0.0):.2f}m"
            now_r, fut_r = self.forward_clearance(0.0, w, pose, maps)
            if not (fut_r < r and fut_r < now_r - 1e-3) or self.rotation_clear(w, pose, maps):
                return 0.0, w, True, reason
            return 0.0, 0.0, True, reason
        return v, w, dec.emergency, None
