"""The onboard autonomy node: wires localisation, perception, mapping, planning, safety
and control into one ``step(frame) -> (WheelCmd, Telemetry | None, DebugBundle)``.

Per tick::

    Perception.compute_disparity(frame) -> disparity (SGBM once per frame, shared with VO)
    Localizer.update(frame, disparity)  -> A-frame pose, health q = 1 - p_fail, chi_hat
                                          (tier0 frames: vo_available = 0 -> wheel+gyro EKF, q := 1)
    Perception.process(frame, pose)    -> egocentric BEV states / costs, r_vis (reuses the disparity;
                                          the segmenter runs at 1/3 camera rate, EveryNthSegmenter)
    RollingMap.integrate               -> A-frame seen-ground memory
    build_costmap                      -> inflated costs, certification, clearance
    GlobalPlanner.update (1 Hz / cut)  -> cost-to-go field, descent path, lookahead
    Supervisor.update                  -> drive mode, speed factor, cost inflation, overrides
    SpeedGovernor.compute              -> v_cap, R_cert (+ which term binds)
    MppiPlanner.plan                   -> (v, omega), nominal path, rollouts
    Supervisor.gate                    -> overrides + 1 s forward footprint check
    SkidSteerMixer                     -> wheel rad/s (WheelCmd)

The costmap uses the supervisor's cost-inflation scale from the previous tick, because the
supervisor needs the fresh cost-to-go for its progress monitor and so runs after the planner.

Ablation switches (``metagross.contracts.ipc.DEFAULT_AUTONOMY_CONFIG``):

* ``unknown_is_free``: unknown cells cost 0 and count as certified (typical stack).
* ``use_negobs``: forwarded to perception; when False, ditch candidates are never lethal.
* ``use_governor``: seen-distance cap + MPPI certification term; False -> platform cap only.
* ``use_health``: health q gates speed / modes / stale certification; False -> q := 1.
* ``use_semantics``, ``use_wheel_odom``: forwarded to perception / localiser via config.
* ``fixed_speed_mps``: constant cruise speed (MPPI speed reference; replaces the governor cap
  when the governor is off, otherwise min of the two).
* ``speed_cap_mps``: hard cap on top of everything.
* ``debug_every_n``: used by :mod:`metagross.autonomy.process` for logging.

Extra (optional) keys: ``stub_perception`` / ``stub_localizer`` (force stubs, for plumbing
tests), ``seed`` (MPPI noise seed, default 0), ``launch_apron_m``, ``fixed_latency_s`` (use
this compute latency in the governor's reaction time instead of the measured one; for
deterministic tests / replays only - live runs must use the measured latency, which is the
windowed p90 of :class:`~metagross.autonomy.planning.governor.LatencyEstimator`),
``dead_end_memory`` (default True: a stall that a STOP_AND_LOOK did not resolve is recorded as a
blocked disc in the A frame, :mod:`metagross.autonomy.planning.dead_end`, which the global planner
routes around until it expires; False: the supervisor's older look-then-SAFE_STOP rule).

The real Perception / Localizer / Segmenter are imported lazily; stub fallbacks from
:mod:`metagross.autonomy.node_stubs` are used only when the real module does not exist.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import math
import re
import sys
import time
from typing import Any, Optional

import cv2
import numpy as np

from metagross.autonomy.control.mixer import MixerParams, SkidSteerMixer
from metagross.autonomy.link.egomap import ego_costmap_u4
from metagross.autonomy.planning.costmap import LETHAL_RADIUS_M, CostmapParams, PlanningMaps, build_costmap
from metagross.autonomy.planning.dead_end import FALLBACK_AHEAD_M, DeadEndMemory
from metagross.autonomy.planning.global_planner import GlobalPlan, GlobalPlanner, resample_path
from metagross.autonomy.planning.governor import Q_NOMINAL, GovernorParams, LatencyEstimator, SpeedGovernor
from metagross.autonomy.planning.mppi import ESCAPE_REVERSE_MPS, MppiParams, MppiPlanner
from metagross.autonomy.planning.rolling_map import LAUNCH_APRON_M, BevGeometry, RollingMap
from metagross.autonomy.safety.supervisor import Supervisor
from metagross.config.defaults import CAM_MAX_RANGE_M, CAMERA_HZ_BATCH, CHI_NOMINAL, TELEMETRY_HZ, VEHICLE
from metagross.contracts.interfaces import DebugBundle, LocalizerProto, PerceptionProto
from metagross.contracts.ipc import DEFAULT_AUTONOMY_CONFIG
from metagross.contracts.messages import (
    DriveMode,
    MissionSpec,
    OperatorCmd,
    SensorFrame,
    StereoCalibration,
    Telemetry,
    VehicleSpec,
    WheelCmd,
)

LOG = logging.getLogger(__name__)

PERCEPTION_MODULE = "metagross.autonomy.perception.pipeline"
LOCALIZER_MODULE = "metagross.autonomy.localization.localizer"
SEGMENTER_MODULE = "metagross.autonomy.perception.semantics"
EMA_ALPHA = 0.2  # smoothing of the measured frame period
N_WAYPOINTS = 5
WAYPOINT_SPACING_M = 2.0
MAP_CROP_HALF_M = 10.0  # debug map crop around the vehicle
TIER0_SENSOR_MODE = "tier0_disparity"  # SensorFrame.sensor_mode of image-less Tier-0 frames
SEG_EVERY_N = 3  # segmenter runs at 1/3 of the camera rate; the last mask is reused in between
DEBUG_IMAGE_WH = (320, 200)  # left image stored in DebugBundle.extras['left_rgb'] (INTER_AREA)
TIMING_KEYS = ("localizer", "perception", "map", "costmap", "global", "supervisor", "governor", "mppi", "gate_mixer")
CERTIFIED_LOCAL_KEY = "certified_local"  # optional perception output: (nx, ny) bool BEV certification
R_DET_KEY = "r_det_m"  # optional perception output: measured design-ditch detection range [m]
HEADING_ERR_MIN_DIST_M = 0.5  # lookahead closer than this: heading error to it is undefined (goal reached)

_ALIASES = {
    "calib": "calib", "calibration": "calib", "stereo_calib": "calib", "cal": "calib",
    "vehicle": "vehicle", "vehicle_spec": "vehicle", "vspec": "vehicle",
    "config": "config", "cfg": "config", "segmenter": "segmenter", "mission": "mission", "seed": "seed",
}


class EveryNthSegmenter:
    """SegmenterProto wrapper that runs the network on every ``n``-th call and returns the
    cached (class_ids, entropy) otherwise (semantics at 1/n of the camera rate: terrain
    classes change slowly, the CPU budget does not). ``calls`` / ``runs`` count invocations."""

    def __init__(self, segmenter: Any, every_n: int = SEG_EVERY_N) -> None:
        self.inner = segmenter
        self.every_n = max(1, int(every_n))
        self.calls = 0
        self.runs = 0
        self._last: Optional[tuple[np.ndarray, np.ndarray]] = None

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._last is None or self.calls % self.every_n == 0 or self._last[0].shape[:2] != rgb.shape[:2]:
            self._last = self.inner(rgb)
            self.runs += 1
        self.calls += 1
        return self._last


def _load_class(module: str, name: str) -> Optional[type]:
    """Import ``module.name``; None only if ``module`` itself does not exist (errors inside it propagate)."""
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name is not None and (module == exc.name or module.startswith(exc.name + ".")):
            return None
        raise
    return getattr(mod, name, None)


def _adapt_config(cls: type, prm: inspect.Parameter, config: dict) -> tuple[bool, Any]:
    """Map the stack-level config dict onto a component's ``config`` parameter.

    If the annotation names a config class exposing ``from_autonomy_config(dict)`` (resolved in
    the component's module), that is used; dict / Mapping / Any / unannotated parameters get the
    dict itself; otherwise the parameter is left at its default. Returns (pass_it, value)."""
    ann = prm.annotation
    ann_s = ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))
    if ann is inspect.Parameter.empty or any(k in ann_s for k in ("dict", "Mapping", "Any")):
        return True, config
    mod = sys.modules.get(cls.__module__)
    for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", ann_s):
        target = getattr(mod, name, None)
        factory = getattr(target, "from_autonomy_config", None)
        if callable(factory):
            return True, factory(config)
    if prm.default is inspect.Parameter.empty:
        raise TypeError(f"{cls.__name__}: cannot adapt autonomy config to parameter annotated {ann_s!r}")
    LOG.warning("%s: config parameter annotated %r left at its default", cls.__name__, ann_s)
    return False, None


def _construct(cls: type, **available: Any) -> Any:
    """Instantiate ``cls`` passing the available objects whose parameter names match (see _ALIASES)."""
    sig = inspect.signature(cls)
    kwargs = {}
    for pname, prm in sig.parameters.items():
        if prm.kind in (prm.VAR_POSITIONAL, prm.VAR_KEYWORD):
            continue
        canon = _ALIASES.get(pname)
        if canon == "config" and "config" in available:
            ok, val = _adapt_config(cls, prm, available["config"])
            if ok:
                kwargs[pname] = val
        elif canon is not None and canon in available:
            kwargs[pname] = available[canon]
        elif prm.default is inspect.Parameter.empty:
            raise TypeError(f"cannot construct {cls.__name__}: no value for required parameter '{pname}'")
    return cls(**kwargs)


class AutonomyStack:
    """AutonomyStackProto implementation. Components can be injected (tests, replay)."""

    def __init__(self, perception: Optional[PerceptionProto] = None, localizer: Optional[LocalizerProto] = None) -> None:
        self._inj_perception = perception
        self._inj_localizer = localizer
        self.perception: Optional[PerceptionProto] = None
        self.localizer: Optional[LocalizerProto] = None
        self.impl: dict[str, str] = {}
        self.config: dict = dict(DEFAULT_AUTONOMY_CONFIG)

    # ------------------------------------------------------------------ setup
    def _build_components(self, calib: StereoCalibration, vehicle: VehicleSpec, mission: MissionSpec) -> None:
        cfg = self.config
        avail = {"calib": calib, "vehicle": vehicle, "config": cfg, "mission": mission, "seed": int(cfg.get("seed", 0))}
        from metagross.autonomy import node_stubs

        if self._inj_localizer is not None:
            self.localizer, self.impl["localizer"] = self._inj_localizer, type(self._inj_localizer).__name__
        else:
            cls = None if cfg.get("stub_localizer") else _load_class(LOCALIZER_MODULE, "Localizer")
            if cls is None:
                cls = node_stubs.StubWheelGyroLocalizer
            self.localizer = _construct(cls, **avail)
            self.impl["localizer"] = cls.__name__
        if self._inj_perception is not None:
            self.perception, self.impl["perception"] = self._inj_perception, type(self._inj_perception).__name__
        else:
            cls = None if cfg.get("stub_perception") else _load_class(PERCEPTION_MODULE, "Perception")
            if cls is None:
                cls = node_stubs.StubBlindPerception
            else:
                params = inspect.signature(cls).parameters
                if "segmenter" in params and cfg.get("use_semantics", True):
                    seg_cls = _load_class(SEGMENTER_MODULE, "Segmenter")
                    if seg_cls is not None:
                        try:
                            avail["segmenter"] = EveryNthSegmenter(_construct(seg_cls, **avail), SEG_EVERY_N)
                            self.impl["segmenter"] = f"{seg_cls.__name__}/{SEG_EVERY_N}"
                        except (TypeError, FileNotFoundError, OSError) as exc:
                            LOG.warning("segmenter unavailable (%s); perception runs without semantics", exc)
                            avail["segmenter"] = None
                    else:
                        avail["segmenter"] = None
                elif "segmenter" in params:
                    avail["segmenter"] = None
            self.perception = _construct(cls, **avail)
            self.impl["perception"] = cls.__name__
        LOG.info("autonomy components: %s", self.impl)

    def reset(self, mission: MissionSpec, calib: StereoCalibration, vehicle: VehicleSpec, config: dict) -> None:
        """Prepare for a new mission. ``config`` overrides ``DEFAULT_AUTONOMY_CONFIG``."""
        self.config = {**DEFAULT_AUTONOMY_CONFIG, **(config or {})}
        cfg = self.config
        self.mission, self.calib, self.vehicle = mission, calib, vehicle
        self._build_components(calib, vehicle, mission)
        self.map = RollingMap()
        self.cm_params = CostmapParams(unknown_is_free=bool(cfg["unknown_is_free"]), use_negobs=bool(cfg["use_negobs"]))
        self.global_planner = GlobalPlanner()
        self.governor = SpeedGovernor(GovernorParams(v_platform_mps=vehicle.max_speed_mps))
        self.mppi = MppiPlanner(
            MppiParams(max_yaw_rate=vehicle.max_yaw_rate_rps, max_accel=vehicle.max_accel_mps2, track_width_m=vehicle.track_width_m,
                       v_wheel_max=vehicle.max_wheel_rad_s * vehicle.wheel_radius_m, v_reverse_max=ESCAPE_REVERSE_MPS),
            seed=int(cfg.get("seed", 0)),
        )
        self.dead_ends: Optional[DeadEndMemory] = DeadEndMemory() if bool(cfg.get("dead_end_memory", True)) else None
        self.supervisor = Supervisor(dead_end_enabled=self.dead_ends is not None)
        self.supervisor.reset(mission.goal_xy_a, mission.success_radius_m)
        self.mixer = SkidSteerMixer(MixerParams.from_vehicle(vehicle))
        self._t_prev: Optional[float] = None
        self._pose_prev: Optional[tuple[float, float, float]] = None
        self._frame_period = 1.0 / CAMERA_HZ_BATCH
        self.latency = LatencyEstimator()  # measured compute latency -> governor reaction time
        self._speed_meas = 0.0
        self._nominal_xy = np.zeros((0, 2))
        self._t_telemetry = -math.inf
        self._tel_seq = 0
        self._apron_done = False
        self._inflate_scale = 1.0
        self._tick = 0
        self.last_maps: Optional[PlanningMaps] = None
        LOG.info("reset mission %s goal=%s config=%s", mission.mission_id, mission.goal_xy_a, cfg)

    def operator(self, cmd: OperatorCmd) -> None:
        """Forward an operator command (HOLD / RESUME / ESTOP / GO[, new goal])."""
        self.supervisor.operator(cmd)
        if cmd.goal_xy_a is not None:
            self.mission = MissionSpec(self.mission.mission_id, tuple(cmd.goal_xy_a), self.mission.success_radius_m, self.mission.timeout_s)

    def _mark_dead_end(self, t: float, pose: tuple[float, float, float], gplan: GlobalPlan, maps: PlanningMaps,
                       goal: tuple[float, float]) -> None:
        """Record the corner the supervisor declared a dead end (A frame): centred on the route's first
        not-certified point near the vehicle (the frontier it stalled at), else on the lookahead point,
        else ``FALLBACK_AHEAD_M`` straight ahead."""
        centre = None
        if gplan.route_ok and gplan.path_xy.shape[0] >= 1:
            route = np.vstack([np.array([[pose[0], pose[1]]]), gplan.path_xy])
            centre = DeadEndMemory.frontier_point(route, lambda x, y: maps.sample(maps.certified, x, y, fill=False))
        if centre is None and gplan.lookahead_xy is not None:
            centre = (float(gplan.lookahead_xy[0]), float(gplan.lookahead_xy[1]))
        if centre is None:
            centre = (pose[0] + FALLBACK_AHEAD_M * math.cos(pose[2]), pose[1] + FALLBACK_AHEAD_M * math.sin(pose[2]))
        self.dead_ends.mark(t, (pose[0], pose[1]), centre, goal)

    # ------------------------------------------------------------------ tick
    def step(self, frame: SensorFrame) -> tuple[WheelCmd, Optional[Telemetry], DebugBundle]:
        """Run one control tick on ``frame``."""
        cfg = self.config
        t_start = time.perf_counter()
        tm: dict[str, float] = {}
        t = float(frame.t)
        gap = 0.0 if self._t_prev is None else t - self._t_prev
        if self._t_prev is not None and 0.0 < gap < 1.0:
            self._frame_period += EMA_ALPHA * (gap - self._frame_period)

        # 1. perception stage 1 (stereo) first: its disparity is shared with VO, so SGBM runs
        # once per frame; 2. localisation; 3. the rest of perception with the fresh pose.
        t0 = time.perf_counter()
        disp_fn = getattr(self.perception, "compute_disparity", None)
        disp_shared = disp_fn(frame) if callable(disp_fn) else frame.disparity
        t_stereo = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        loc = self.localizer.update(frame, disp_shared)
        tm["localizer"] = (time.perf_counter() - t0) * 1e3
        pose = tuple(float(v) for v in loc["pose_xy_yaw"])
        loc_health = {k: float(v) for k, v in (loc.get("health") or {}).items() if isinstance(v, (int, float, np.floating, np.integer, bool))}
        p_fail = float(loc_health.get("p_fail", 0.0))
        chi = float(loc.get("chi_hat") or CHI_NOMINAL)
        slip = float(loc.get("slip", 0.0) or 0.0)  # slip ratio (0 = none)
        immobilised = bool(loc.get("immobilised", False))
        if not self._apron_done:
            self.map.seed_apron(pose, t, float(cfg.get("launch_apron_m", LAUNCH_APRON_M)))
            self._apron_done = True

        t0 = time.perf_counter()
        per = self.perception.process(frame, pose)
        tm["perception"] = t_stereo + (time.perf_counter() - t0) * 1e3
        states = per["cell_state_local"]
        r_vis = float(per.get("r_vis_m", CAM_MAX_RANGE_M) if per.get("r_vis_m") is not None else CAM_MAX_RANGE_M)
        q = float(loc_health["q"]) if "q" in loc_health else 1.0 - p_fail  # localiser's smoothed q if given
        vo_available = float(loc_health.get("vo_available", 1.0)) > 0.5
        if not vo_available and frame.sensor_mode == TIER0_SENSOR_MODE:
            # Tier-0 frames carry no images by construction: VO cannot run, localisation is the
            # wheel + gyro EKF. That is the configured sensor suite, not a VO integrity fault, so
            # the VO integrity score must not gate modes / speed (its growth in pos_sigma_m does
            # the job of flagging drift). With images present (stereo), q keeps its full gating.
            q, p_fail = 1.0, 0.0
        per_health = per.get("health") or {}
        if "q" in per_health:
            q = min(q, float(per_health["q"]))
        if not cfg["use_health"]:
            q = 1.0
        q = float(np.clip(q, 0.0, 1.0))

        # 3. rolling map
        t0 = time.perf_counter()
        cert_local = per.get(CERTIFIED_LOCAL_KEY)  # optional perception certification layer (fallback: all GROUND)
        stats = self.map.integrate(t, pose, states, per.get("cost_local"), per.get("mu_local"), BevGeometry.from_perception(per, states.shape),
                                   certified_local=cert_local if isinstance(cert_local, np.ndarray) and cert_local.shape == states.shape else None)
        n_fp_cleared = self.map.clear_footprint(pose, t, self.vehicle.length_m, self.vehicle.width_m)
        tm["map"] = (time.perf_counter() - t0) * 1e3

        # measured speed from pose increments
        if self._pose_prev is not None and gap > 1e-6:
            v_meas = math.hypot(pose[0] - self._pose_prev[0], pose[1] - self._pose_prev[1]) / gap
            self._speed_meas += 0.5 * (v_meas - self._speed_meas)
        self._pose_prev, self._t_prev = pose, t

        # 4. costmap (inflation scale from the supervisor's health level of the previous tick)
        t0 = time.perf_counter()
        nominal_health = (q > Q_NOMINAL) if cfg["use_health"] else True
        maps = build_costmap(self.map, t, nominal_health, self.cm_params.scaled(self._inflate_scale))
        self.last_maps = maps
        tm["costmap"] = (time.perf_counter() - t0) * 1e3

        # 5. global planner
        t0 = time.perf_counter()
        goal = self.supervisor.goal_xy
        gplan = self.global_planner.update(t, maps, (pose[0], pose[1]), goal, dead_ends=self.dead_ends)
        tm["global"] = (time.perf_counter() - t0) * 1e3

        # 6. supervisor (mode, speed factor, overrides); progress = cost-to-go when a route exists
        t0 = time.perf_counter()
        herr = None
        if gplan.route_ok and gplan.lookahead_xy is not None:
            lx, ly = gplan.lookahead_xy
            if math.hypot(lx - pose[0], ly - pose[1]) > HEADING_ERR_MIN_DIST_M:
                e = math.atan2(ly - pose[1], lx - pose[0]) - pose[2]
                herr = abs(math.atan2(math.sin(e), math.cos(e)))
        dec = self.supervisor.update(t, q, pose, self.mixer.v, self._speed_meas, gap, immobilised,
                                     progress_metric=gplan.ctg_vehicle if gplan.route_ok else None, metric_kind="ctg",
                                     metric_shift=max(gplan.ctg_jump, 0.0) if gplan.t_computed == t else 0.0,
                                     heading_err=herr)
        self._inflate_scale = dec.inflate_scale
        if dec.dead_end and self.dead_ends is not None:
            self._mark_dead_end(t, pose, gplan, maps, goal)  # the planner routes round it from the next tick
        tm["supervisor"] = (time.perf_counter() - t0) * 1e3

        # 7. governor
        t0 = time.perf_counter()
        latency = float(cfg["fixed_latency_s"]) if cfg.get("fixed_latency_s") is not None else self.latency.value()
        r_det = per.get(R_DET_KEY)
        gov = self.governor.compute(maps, pose, self._nominal_xy, r_vis, q, latency, self._frame_period,
                                    enabled=bool(cfg["use_governor"]), use_health=bool(cfg["use_health"]),
                                    r_det_m=float(r_det) if isinstance(r_det, (int, float, np.floating)) else None)
        v_cap, gov_reason = gov.v_cap_mps, gov.reason
        fixed = cfg.get("fixed_speed_mps")
        v_ref = None
        if fixed is not None:  # constant cruise speed (typical stack); the governor, if on, may still lower it
            v_ref = min(float(fixed), self.vehicle.max_speed_mps)
            if v_ref < v_cap:
                v_cap, gov_reason = v_ref, f"GOV_FIXED v={v_ref:.1f}"
        cap = cfg.get("speed_cap_mps")
        if cap is not None and float(cap) < v_cap:
            v_cap, gov_reason = float(cap), f"GOV_CAP v={float(cap):.1f}"
        v_cap_eff = v_cap * dec.speed_factor
        tm["governor"] = (time.perf_counter() - t0) * 1e3

        # 8. MPPI
        t0 = time.perf_counter()
        if gplan.route_ok:
            ctg_fn = gplan.ctg_at
            guide = gplan.path_xy
        else:
            gx, gy = goal
            ctg_fn = lambda x, y: np.hypot(x - gx, y - gy)  # noqa: E731 - no route: straight-line fallback
            guide = np.array([[pose[0], pose[1]], [gx, gy]])
        if self._nominal_xy.shape[0] == 0 and gplan.lookahead_xy is not None:
            self.mppi.initialise_towards(pose, gplan.lookahead_xy, 0.0)
        # back-off allowed only when the footprint already sits inside the lethal inflation
        clr_now = self.supervisor.forward_clearance(0.0, 0.0, pose, maps)[0]
        allow_reverse = clr_now < LETHAL_RADIUS_M * self._inflate_scale
        res = self.mppi.plan(t, pose, self.mixer.v, self.mixer.w, v_cap_eff, maps, ctg_fn, gov.t_r_s, v_ref=v_ref,
                             cert_enabled=bool(cfg["use_governor"]) and not bool(cfg["unknown_is_free"]),
                             brake_decel=gov.a_mps2, chi=chi, guide_path_xy=guide, hold_s=self._frame_period,
                             allow_reverse=allow_reverse)
        self._nominal_xy = res.traj[:, :2]
        tm["mppi"] = (time.perf_counter() - t0) * 1e3

        # 9. safety gate + 10. mixer
        t0 = time.perf_counter()
        v, w, emergency, brake_reason = self.supervisor.gate(dec, res.u0[0], res.u0[1], pose, maps)
        wl, wr = self.mixer.mix(t, v, w, chi, emergency)
        tm["gate_mixer"] = (time.perf_counter() - t0) * 1e3
        compute_ms = (time.perf_counter() - t_start) * 1e3
        self.latency.push(compute_ms / 1e3)
        mode = dec.mode
        cmd = WheelCmd(t=t, seq=int(frame.seq), omega_l_rad_s=float(wl), omega_r_rad_s=float(wr), mode=mode, compute_ms=compute_ms)

        if brake_reason:
            reason = brake_reason
        elif mode in (DriveMode.NOMINAL, DriveMode.CAUTION, DriveMode.DEGRADED) and not gplan.route_ok:
            reason = "NO_ROUTE"
        elif mode == DriveMode.NOMINAL and dec.reason == "OK":
            reason = gov_reason
        else:
            reason = dec.reason
        health = {
            "q": q, "p_fail": p_fail, "pos_sigma_m": float(loc.get("pos_sigma_m", 0.0)), "vo_ok": float(bool(loc.get("vo_ok", False))),
            "slip": slip, "chi_hat": chi, "r_vis_m": r_vis, "compute_ms": compute_ms,
        }
        for k, val in loc_health.items():
            health.setdefault(k, val)

        # telemetry at TELEMETRY_HZ
        t0 = time.perf_counter()
        tel: Optional[Telemetry] = None
        u4 = None
        if t - self._t_telemetry >= 1.0 / TELEMETRY_HZ - 1e-6:
            self._t_telemetry = t
            u4 = ego_costmap_u4(self.map, maps, pose)
            wps = resample_path(gplan.path_xy, WAYPOINT_SPACING_M, N_WAYPOINTS) if gplan.route_ok else np.zeros((0, 2))
            tel = Telemetry(
                t=t, seq=self._tel_seq, pose_xy_yaw=pose, pos_sigma_m=health["pos_sigma_m"], mode=mode, reason=reason,
                v_cap_mps=v_cap_eff, r_cert_m=gov.r_cert_m, speed_mps=self._speed_meas,
                waypoints_xy=[(float(a), float(b)) for a, b in wps], costmap_u4=u4, health=health,
            )
            self._tel_seq += 1
        tm["telemetry"] = (time.perf_counter() - t0) * 1e3

        # debug bundle
        c, s = math.cos(pose[2]), math.sin(pose[2])
        rel = res.rollouts_xy - np.array(pose[:2])
        rollouts_body = np.stack([c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]], axis=-1)
        crop, crop_conf, crop_origin = self.map.crop(pose[0], pose[1], MAP_CROP_HALF_M)
        per_t = {f"perception.{k}": float(v) for k, v in (per.get("timings_ms") or {}).items()}
        loc_t = {f"localizer.{k}": float(v) for k, v in (loc.get("timings_ms") or {}).items()}
        tm["compute"] = compute_ms
        tm["total"] = (time.perf_counter() - t_start) * 1e3
        dbg = DebugBundle(
            t=t, pose_xy_yaw=pose,
            cell_state_local=states, cost_local=per.get("cost_local"), semantic_mask=per.get("semantic_mask"),
            missing_ground_mask=per.get("missing_ground_mask"), disparity=per.get("disparity"),
            rollouts_xy=rollouts_body.astype(np.float32), plan_xy=res.traj[:, :2].astype(np.float32),
            global_path_xy=gplan.path_xy.astype(np.float32), v_cap_mps=v_cap_eff, r_cert_m=gov.r_cert_m, health=health,
            timings_ms={**tm, **per_t, **loc_t},
            extras={
                "mode": mode.value, "reason": reason, "cmd_v": float(self.mixer.v), "cmd_w": float(self.mixer.w),
                "gov_terms": dict(gov.terms), "gov_binding": gov.binding, "t_r_s": gov.t_r_s, "a_mps2": gov.a_mps2,
                "r_path_m": gov.r_path_m, "health_factor": gov.health_factor, "speed_factor": dec.speed_factor,
                "mppi_ess": res.ess, "mppi_s_min": res.s_min, "mppi_terms": res.cost_terms, "route_ok": gplan.route_ok,
                "ctg_vehicle": gplan.ctg_vehicle, "global_ms": gplan.compute_ms, "map_new_lethal": stats.n_new_lethal,
                "map_confirmed_ditch": stats.n_confirmed_ditch, "map_fp_cleared": n_fp_cleared, "map_crop_state": crop, "map_crop_confirmed": crop_conf,
                "map_crop_origin": crop_origin, "map_res_m": self.map.res, "costmap_u4": u4, "impl": dict(self.impl),
                "lookahead_xy": gplan.lookahead_xy, "speed_meas": float(self._speed_meas),
                "vo_available": bool(vo_available), "tick": self._tick, "latency_s": float(latency),
                "dead_ends_marked": int(self.dead_ends.n_marked) if self.dead_ends is not None else 0,
            },
        )
        n_img = int(cfg.get("debug_image_every_n", 2) or 0)
        if frame.left_rgb is not None and n_img > 0 and self._tick % n_img == 0:
            dbg.extras["left_rgb"] = cv2.resize(frame.left_rgb, DEBUG_IMAGE_WH, interpolation=cv2.INTER_AREA)
        self._tick += 1
        return cmd, tel, dbg
