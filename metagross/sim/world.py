"""The simulated world: ground-truth owner for one episode.

``World`` integrates the vehicle at ``PHYSICS_HZ``, scripts the dynamic obstacles and lighting,
runs the referee every physics step, logs GT at ``GT_LOG_HZ`` and produces contract
:class:`~metagross.contracts.messages.SensorFrame` s:

* ``sensor_mode='tier0'``: the Tier-0 **synthetic depth sensor** fills ``disparity``
  (``sensor_mode='tier0_disparity'`` in the frame, images are ``None``);
* ``sensor_mode='stereo'``: ``renderer.render_stereo(state)`` supplies rectified images.

Nothing in a SensorFrame or MissionSpec carries ground truth: the goal is expressed in the
A-frame (start pose; x along the launch heading, y left) with the scenario's launch-heading
error applied as a counter-clockwise rotation of the goal vector, exactly what an operator with
an imperfect heading reference would hand the vehicle.

Renderer state (``render_state()``): ``pose`` is the **body** pose ``[x, y, z, roll, pitch, yaw]``
(world frame, REP-103 angles, see :mod:`metagross.sim.geometry`); ``T_world_body`` and
``T_world_cam`` (4x4 nested lists, camera = left camera, OpenCV axes) are included so the
renderer never has to re-derive the extrinsics.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from metagross.config import defaults
from metagross.contracts.interfaces import RendererProto
from metagross.contracts.messages import MissionSpec, SensorFrame, StereoCalibration, WheelCmd
from metagross.sim.geometry import pose_matrix, rot2
from metagross.sim.hazards import gt_hazard_raster
from metagross.sim.objects import DynamicObject, Footprint, Primitive, parse_static_objects
from metagross.sim.referee import DEFAULT_TERMINAL, Event, Referee
from metagross.sim.scenario import scenario_sha256
from metagross.sim.sensors import Gyro, GyroParams, Tier0DepthSensor, Tier0Params, WheelEncoders
from metagross.sim.terrain import Terrain
from metagross.sim.vehicle import CommandQueue, SkidSteerVehicle

log = logging.getLogger(__name__)

SENSOR_MODES = ("tier0", "stereo")
GT_FIELDS = ("t", "x", "y", "z", "roll", "pitch", "yaw", "v", "omega", "wl", "wr")
MISSION_ID_HEX_CHARS = 8  # opaque mission id = first 8 hex chars of the scenario sha256 (GT firewall: no family / seed)


def opaque_mission_id(scenario: dict) -> str:
    """Mission id that names the scenario file (sha256 prefix) without revealing its family or seed."""
    sha = str(scenario.get("sha256") or "") or scenario_sha256(scenario)
    return sha[:MISSION_ID_HEX_CHARS]


def sun_direction(elev_deg: float, azim_deg: float) -> np.ndarray:
    """Unit vector towards the sun in the world frame (azimuth CCW from +x/east)."""
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


class World:
    def __init__(self, scenario: dict, sensor_mode: str = "tier0", renderer: RendererProto | None = None,
                 noise_seed: int = 0, tier0_params: Tier0Params = Tier0Params(), gyro_params: GyroParams = GyroParams(),
                 terminal: frozenset[str] = DEFAULT_TERMINAL, timeout_s: float | None = None):
        if sensor_mode not in SENSOR_MODES:
            raise ValueError(f"sensor_mode must be one of {SENSOR_MODES}")
        if sensor_mode == "stereo" and renderer is None:
            raise ValueError("stereo mode needs a renderer")
        self.scenario = scenario
        self.sensor_mode = sensor_mode
        self.renderer = renderer
        self.dt = 1.0 / defaults.PHYSICS_HZ
        self.step_index = 0
        self.terrain = Terrain.from_scenario(scenario)
        self.static_prims, self.static_feet = parse_static_objects(scenario, self.terrain)
        self.hazards = gt_hazard_raster(scenario, self.terrain)
        self.dynamic = [DynamicObject.from_dict(k, d) for k, d in enumerate(scenario.get("dynamic", []))]
        self.start_xy = np.asarray(scenario["start"]["xy"], float)
        self.start_yaw = float(scenario["start"]["yaw"])
        self.goal_xy = np.asarray(scenario["goal"]["xy"], float)
        m = scenario["mission"]
        self.timeout_s = float(m["timeout_s"]) if timeout_s is None else min(float(timeout_s), float(m["timeout_s"]))
        self.vehicle = SkidSteerVehicle(self.terrain, scenario.get("vehicle", {}))
        self.vehicle.reset(float(self.start_xy[0]), float(self.start_xy[1]), self.start_yaw)
        self.queue = CommandQueue(self.dt)
        self._t_last_cmd = 0.0  # time the last WheelCmd took effect (actuator watchdog)
        self._cmd_timed_out = False
        self.n_cmd_timeouts = 0
        ss = np.random.SeedSequence([int(scenario["seed"]), int(noise_seed), 0x5E45])
        rng_gyro, rng_depth = (np.random.default_rng(s) for s in ss.spawn(2))
        self.rng_depth = rng_depth
        self.encoders = WheelEncoders()
        self.gyro = Gyro(rng_gyro, gyro_params)
        self.referee = Referee(self.terrain, self.hazards, self.static_feet, tuple(self.goal_xy), float(m["success_radius_m"]),
                               self.timeout_s, terminal=terminal)
        self.depth_sensor = Tier0DepthSensor(self.terrain, self.static_prims, tier0_params) if sensor_mode == "tier0" else None
        if renderer is not None:
            renderer.load_scenario(scenario)
        self._last_frame_t: float | None = None
        self._last_frame_yaw = self.vehicle.state.yaw_unwrapped
        self._log: dict[str, list] = {k: [] for k in GT_FIELDS}
        self._log_dyn: list[list[list[float]]] = []
        self._next_log_k = 0
        self._light_active: set[int] = set()
        self.path_length_m = 0.0
        self.sensor_ms: list[float] = []
        self.sensor_breakdown: list[dict[str, float]] = []  # Tier-0 per-stage timings (ms)
        self.physics_ms: list[float] = []
        self._log_gt()

    # ------------------------------------------------------------------ time / state
    @property
    def t(self) -> float:
        return self.step_index * self.dt

    @property
    def state(self):
        return self.vehicle.state

    def camera_pose(self) -> np.ndarray:
        """T_world_cam (4x4): left camera -> world, from the full 6-DoF body pose."""
        return pose_matrix(*self.state.pose6()) @ defaults.camera_extrinsics()

    def calibration(self) -> StereoCalibration:
        """Calibration of the frames this world produces (sent to the autonomy at reset)."""
        if self.depth_sensor is not None:
            return self.depth_sensor.calibration()
        return defaults.stereo_calibration()

    # ------------------------------------------------------------------ mission
    def mission_spec(self, mission_id: str | None = None) -> MissionSpec:
        """The operator's mission as the autonomy receives it (A-frame goal, opaque id; no GT)."""
        m = self.scenario["mission"]
        g_a = rot2(-self.start_yaw) @ (self.goal_xy - self.start_xy)
        g_a = rot2(math.radians(float(m.get("heading_init_err_deg", 0.0)))) @ g_a
        mid = mission_id or opaque_mission_id(self.scenario)
        return MissionSpec(mid, (float(g_a[0]), float(g_a[1])), float(m["success_radius_m"]), self.timeout_s)

    # ------------------------------------------------------------------ lighting / dynamics
    def lighting_state(self, t: float | None = None) -> dict:
        t = self.t if t is None else t
        L = self.scenario.get("lighting", {})
        active = [ev for ev in L.get("events", []) if ev["t0"] <= t <= ev["t1"]]
        exposure = float(L.get("exposure", 1.0))
        for ev in active:
            if ev["type"] == "dim":
                exposure *= float(ev["gain"])
        return {"sun_elev_deg": L.get("sun_elev_deg", 45.0), "sun_azim_deg": L.get("sun_azim_deg", 0.0),
                "sun_dir_world": sun_direction(L.get("sun_elev_deg", 45.0), L.get("sun_azim_deg", 0.0)).tolist(),
                "fog_density": float(L.get("fog_density", 0.0)), "exposure": exposure, "active_events": active}

    def dynamic_prims(self, t: float | None = None) -> list[Primitive]:
        t = self.t if t is None else t
        return [d.primitive_at(t, self.terrain) for d in self.dynamic]

    def dynamic_feet(self, t: float | None = None) -> list[Footprint]:
        t = self.t if t is None else t
        return [d.footprint_at(t) for d in self.dynamic]

    def render_state(self) -> dict:
        s = self.state
        return {"pose": s.pose6(), "t": self.t, "dynamic": [d.state_dict(self.t, self.terrain) for d in self.dynamic],
                "lighting": self.lighting_state(), "T_world_body": pose_matrix(*s.pose6()).tolist(),
                "T_world_cam": self.camera_pose().tolist()}

    # ------------------------------------------------------------------ commands / physics
    def queue_command(self, cmd: WheelCmd) -> int:
        """Queue an autonomy command issued for the frame taken at the current step (latency injection)."""
        return self.queue.push(cmd, self.step_index)

    def _log_gt(self) -> None:
        s = self.state
        for k in GT_FIELDS:
            self._log[k].append(self.t if k == "t" else getattr(s, k))
        self._log_dyn.append([d.xy_at(self.t).tolist() for d in self.dynamic])

    def step(self) -> list[Event]:
        """Advance one physics step; returns the referee / world events raised during it."""
        t0 = time.perf_counter()
        cmd = self.queue.pop_due(self.step_index)
        if cmd is not None:
            self.vehicle.set_command(cmd.omega_l_rad_s, cmd.omega_r_rad_s)
            self._t_last_cmd = self.t
        elif self.t - self._t_last_cmd > defaults.WHEEL_CMD_TIMEOUT_S + 1e-9 and not self._cmd_timed_out:
            # Actuator watchdog (motor-controller side): a silent autonomy must not keep driving.
            self.vehicle.set_command(0.0, 0.0)
            self._cmd_timed_out = True
            self.n_cmd_timeouts += 1
        if cmd is not None:
            self._cmd_timed_out = False
        x0, y0 = self.state.x, self.state.y
        self.vehicle.step()
        self.step_index += 1
        self.state.t = self.t
        self.path_length_m += math.hypot(self.state.x - x0, self.state.y - y0)
        n0 = len(self.referee.events)
        xy = np.array([self.state.x, self.state.y])
        for d in self.dynamic:
            if d.maybe_trigger(self.t, xy):
                self.referee.emit_external(self.t, "dynamic_trigger", object=f"dyn_{d.type}_{d.index}",
                                           distance_m=round(d.trigger_dist_m, 2))
        for k, ev in enumerate(self.scenario.get("lighting", {}).get("events", [])):
            on = ev["t0"] <= self.t <= ev["t1"]
            if on and k not in self._light_active:
                self._light_active.add(k)
                self.referee.emit_external(self.t, "lighting_on", event=ev["type"], gain=ev["gain"])
            elif not on and k in self._light_active:
                self._light_active.discard(k)
                self.referee.emit_external(self.t, "lighting_off", event=ev["type"])
        self.referee.check(self.state, self.vehicle.contact_points(), self.dynamic_feet())
        if self.t * defaults.GT_LOG_HZ >= self._next_log_k + 1 - 1e-9:
            self._next_log_k = int(math.floor(self.t * defaults.GT_LOG_HZ + 1e-9))
            self._log_gt()
        self.physics_ms.append((time.perf_counter() - t0) * 1e3)
        return self.referee.events[n0:]

    @property
    def done(self) -> bool:
        return self.referee.done

    # ------------------------------------------------------------------ sensors
    def make_sensor_frame(self, t: float, seq: int) -> SensorFrame:
        """Sensor sample at the current world time ``t`` (must equal ``self.t``)."""
        if abs(t - self.t) > 0.5 * self.dt:
            raise ValueError(f"frame time {t} != world time {self.t}: frames are sampled at the current state")
        s = self.state
        enc_l, enc_r = self.encoders.read(s.enc_l, s.enc_r)
        if self._last_frame_t is None:
            rate, dt = s.omega, 1.0 / defaults.CAMERA_HZ_BATCH
        else:
            dt = max(self.t - self._last_frame_t, self.dt)
            rate = (s.yaw_unwrapped - self._last_frame_yaw) / dt
        gyro = self.gyro.sample(rate, dt)
        self._last_frame_t, self._last_frame_yaw = self.t, s.yaw_unwrapped
        t0 = time.perf_counter()
        if self.sensor_mode == "tier0":
            res = self.depth_sensor.render(self.camera_pose(), self.dynamic_prims(), self.lighting_state(), self.rng_depth)
            self.sensor_breakdown.append(res.timings_ms)
            frame = SensorFrame(t=self.t, seq=seq, left_rgb=None, right_gray=None, wheel_angle_l_rad=enc_l,
                                wheel_angle_r_rad=enc_r, gyro_z_rps=gyro, sensor_mode="tier0_disparity", disparity=res.disparity)
        else:
            left, right = self.renderer.render_stereo(self.render_state())
            frame = SensorFrame(t=self.t, seq=seq, left_rgb=left, right_gray=right, wheel_angle_l_rad=enc_l,
                                wheel_angle_r_rad=enc_r, gyro_z_rps=gyro, sensor_mode="stereo")
        self.sensor_ms.append((time.perf_counter() - t0) * 1e3)
        return frame

    def render_gt(self) -> dict[str, np.ndarray]:
        """Evaluator-only GT passes at the current state: {'depth': m, 'semantic': 5-class ids}."""
        if self.sensor_mode == "stereo":
            return self.renderer.render_gt(self.render_state())
        res = self.depth_sensor.render(self.camera_pose(), self.dynamic_prims(), None, None, noise=False, want_gt=True)
        return {"depth": res.depth_gt, "semantic": res.semantic_gt}

    # ------------------------------------------------------------------ logs
    def gt_log(self) -> dict[str, np.ndarray]:
        out = {k: np.asarray(v, np.float64) for k, v in self._log.items()}
        if self.dynamic:
            out["dyn_xy"] = np.asarray(self._log_dyn, np.float64)
        return out
