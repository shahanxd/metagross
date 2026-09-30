"""Run directory -> per-frame panel data for the dashboard compositor.

Input layout (written by :mod:`metagross.sim.runner` and :mod:`metagross.autonomy.process`)::

    run_dir/
      result.json                      referee summary (+ seed, config, family)
      mission.json        (optional)   {"mission_id", "goal_xy_a", "goal_sigma_m", "success_radius_m"}
      gt/states.npz                    GT log: t, x, y, z, roll, pitch, yaw, v, omega, wl, wr  (WORLD frame)
      gt/cmds.npz         (optional)   per camera tick: t, omega_l, omega_r [rad/s], mode
      gt/events.json      (optional)   referee events [{"t", "type", "detail"}]
      autonomy/telemetry.jsonl         decoded downlink packets (codec.to_jsonable)
      autonomy/debug/tick_XXXXXX.npz   DebugBundle arrays (process._bundle_arrays);
                                       ``autonomy/debug_*.npz`` is accepted too

Frames
------
The GT log is in the simulator WORLD frame. Everything the compositor draws is in the
A-frame (mission frame: origin at the launch point, x along the TRUE launch heading, y left),
obtained from the first GT sample. The autonomy's own pose estimate is already in its
A-frame, so "VO trail vs GT trail" is a like-for-like comparison.

Sampling
--------
GT states are interpolated to the output frame time (linear in x, y, v; yaw on the
unwrapped angle). Autonomy debug bundles and telemetry are *held* (latest sample at or
before t) because that is what the robot / operator actually had at that instant.

Chase and left-camera images come from injected callables (e.g. the Three.js renderer's
``render_chase`` / ``render_stereo``); with none attached the compositor draws a clean
placeholder. A debug bundle that carries ``extra_left_rgb`` (synthetic demo data) is used
directly.
"""

from __future__ import annotations

import functools
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np

from metagross.autonomy.link import codec
from metagross.config import defaults

log = logging.getLogger(__name__)

VIDEO_FPS = 30.0
WHEEL_WINDOW_S = 10.0  # wheel-speed trace history shown on the dashboard
EVENT_HOLD_S = 3.0  # referee events stay "recent" this long
MODE_LOG_LEN = 6  # mode transitions kept for the event log
DEBUG_CACHE = 6  # decoded debug bundles kept in memory
LINK_WINDOW_S = 5.0  # sliding window for the measured telemetry rate / link load, s

ChaseRenderer = Callable[[dict, int, int], np.ndarray]
"""(renderer state dict, width, height) -> (h, w, 3) uint8 RGB. See RendererProto.render_chase."""
LeftRenderer = Callable[[dict], np.ndarray]
"""renderer state dict -> (H, W, 3) uint8 RGB left image (first element of render_stereo)."""


# ----------------------------------------------------------------------------- data classes
@dataclass(frozen=True)
class MissionInfo:
    """What the dashboard needs to know about the mission / run (never fed to autonomy)."""

    mission_id: str = "MISSION"
    goal_xy_a: Optional[tuple[float, float]] = None  # A-frame metres
    goal_sigma_m: float = 2.0  # operator goal uncertainty (1-sigma radius drawn as a circle), m
    success_radius_m: float = 2.0
    seed: Optional[int] = None
    config_name: str = "FULL"
    simulated: bool = True
    demo_fake: bool = False
    sensor_mode: str = "stereo"  # result.json 'sensor_mode': 'stereo' (rendered images) or 'tier0' (depth-only)


@dataclass
class DebugTick:
    """One autonomy debug bundle (arrays decoded lazily by :class:`RunReplay`)."""

    index: int
    t: float
    pose_xy_yaw: tuple[float, float, float]  # A-frame estimate
    v_cap_mps: float
    r_cert_m: float
    health: dict[str, float]
    extras: dict[str, Any]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return str(self.extras.get("mode", "HOLD"))

    @property
    def reason(self) -> str:
        return str(self.extras.get("reason", ""))


@dataclass
class PanelData:
    """Everything one dashboard frame shows, at time ``t`` (seconds since run start)."""

    t: float
    mission: MissionInfo
    gt_pose_a: tuple[float, float, float]  # A-frame x, y [m], yaw [rad]
    gt_speed_mps: float
    gt_trail_a: np.ndarray  # (N, 2) A-frame, up to t
    est_pose_a: tuple[float, float, float]  # autonomy estimate (held)
    est_trail_a: np.ndarray  # (M, 2)
    mode: str
    reason: str
    v_cap_mps: float
    r_cert_m: float
    speed_mps: float
    q: float  # integrity 1 - p_fail
    pos_sigma_m: float
    tick: Optional[DebugTick]
    telemetry: Optional[dict]
    costmap_u4: Optional[np.ndarray]  # (64, 64) decoded telemetry costmap
    wheel_t: np.ndarray  # (K,) seconds relative to t (<= 0)
    wheel_l: np.ndarray  # (K,) commanded omega_L [rad/s]
    wheel_r: np.ndarray  # (K,) commanded omega_R [rad/s]
    mode_log: list[tuple[float, str, str]]  # recent (t, mode, reason) transitions
    events: list[dict]  # referee events within EVENT_HOLD_S
    world_state: dict  # RendererProto state (WORLD frame pose) for the chase/left renderers
    chase_rgb: Optional[np.ndarray] = None
    left_rgb: Optional[np.ndarray] = None
    left_tick: Optional[DebugTick] = None  # bundle the onboard image (and its overlays) comes from
    compute_ms: float = float("nan")
    packet_bytes: Optional[int] = None
    link_rate_hz: float = 0.0  # telemetry packets per second over the last LINK_WINDOW_S
    link_kbps: float = 0.0  # measured downlink load over the last LINK_WINDOW_S, kbit/s
    packet_age_s: float = float("nan")  # time since the held telemetry packet, s


# ----------------------------------------------------------------------------- helpers
def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap to (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def world_to_a(x: np.ndarray, y: np.ndarray, yaw: np.ndarray, origin: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WORLD-frame poses -> A-frame, given the launch pose ``origin`` = (x0, y0, yaw0) in WORLD."""
    x0, y0, a0 = origin
    c, s = math.cos(a0), math.sin(a0)
    dx, dy = np.asarray(x) - x0, np.asarray(y) - y0
    return c * dx + s * dy, -s * dx + c * dy, wrap_angle(np.asarray(yaw) - a0)


def decode_costmap_hex(hex_str: Optional[str]) -> Optional[np.ndarray]:
    """``costmap_u4_hex`` from telemetry.jsonl -> (64, 64) uint8 codes, or None."""
    if not hex_str:
        return None
    return codec.unpack_u4(bytes.fromhex(hex_str))


def _held_index(times: np.ndarray, t: float) -> int:
    """Index of the latest sample with time <= t, or -1."""
    return int(np.searchsorted(times, t + 1e-9, side="right")) - 1


def _json_or_empty(arr: Any) -> dict:
    try:
        return json.loads(str(arr))
    except (TypeError, ValueError):
        return {}


# ----------------------------------------------------------------------------- replay
class RunReplay:
    """Random-access replay of one run directory (see module docstring).

    Parameters
    ----------
    run_dir: directory of one episode.
    chase_renderer: optional ``(state, w, h) -> rgb`` for the large chase-camera panel.
    left_renderer: optional ``state -> left_rgb`` used when debug bundles carry no image.
    chase_size: (w, h) passed to ``chase_renderer``.
    mission: overrides ``mission.json`` / ``result.json`` metadata.
    """

    def __init__(self, run_dir: str | Path, chase_renderer: Optional[ChaseRenderer] = None,
                 left_renderer: Optional[LeftRenderer] = None, chase_size: tuple[int, int] = (1160, 652),
                 mission: Optional[MissionInfo] = None) -> None:
        self.run_dir = Path(run_dir)
        self.chase_renderer = chase_renderer
        self.left_renderer = left_renderer
        self.chase_size = chase_size
        self.result = self._read_json(self.run_dir / "result.json", {})
        self.mission = mission or self._mission_info()
        self._load_gt()
        self._load_cmds()
        self._load_telemetry()
        self._index_debug()
        self.events: list[dict] = self._read_json(self.run_dir / "gt" / "events.json", [])
        self._mode_changes = self._compile_mode_changes()
        self._left_cache: dict[int, np.ndarray] = {}
        log.info("replay %s: %.1f s GT, %d debug ticks, %d telemetry packets", self.run_dir.name, self.duration,
                 len(self.debug_t), len(self.tel_t))

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def _mission_info(self) -> MissionInfo:
        m = self._read_json(self.run_dir / "mission.json", {})
        cfg = self.result.get("config") or {}
        goal = m.get("goal_xy_a")
        return MissionInfo(
            mission_id=str(m.get("mission_id", f"RUN-{self.result.get('seed', 'NA')}")),
            goal_xy_a=(float(goal[0]), float(goal[1])) if goal is not None else None,
            goal_sigma_m=float(m.get("goal_sigma_m", m.get("success_radius_m", 2.0))),
            success_radius_m=float(m.get("success_radius_m", 2.0)),
            seed=self.result.get("seed"),
            config_name=str(cfg.get("name", "FULL")),
            simulated=True,
            demo_fake=bool(self.result.get("demo_fake", False) or m.get("demo_fake", False)),
            sensor_mode=str(self.result.get("sensor_mode", "stereo")),
        )

    def _load_gt(self) -> None:
        p = self.run_dir / "gt" / "states.npz"
        if not p.exists():
            raise FileNotFoundError(f"missing GT log {p}")
        with np.load(p) as z:
            gt = {k: np.asarray(z[k], np.float64) for k in z.files}
        self.gt_t = gt["t"]
        self.gt_world = gt
        self.origin_world = (float(gt["x"][0]), float(gt["y"][0]), float(gt["yaw"][0]))
        xa, ya, yawa = world_to_a(gt["x"], gt["y"], gt["yaw"], self.origin_world)
        self.gt_xa, self.gt_ya = xa, ya
        self.gt_yaw_unwrapped = np.unwrap(yawa)
        self.gt_v = gt.get("v", np.zeros_like(self.gt_t))

    def _load_cmds(self) -> None:
        p = self.run_dir / "gt" / "cmds.npz"
        if p.exists():
            with np.load(p) as z:
                self.cmd_t = np.asarray(z["t"], np.float64)
                self.cmd_l = np.asarray(z["omega_l"], np.float64)
                self.cmd_r = np.asarray(z["omega_r"], np.float64)
        else:  # fall back to measured wheel speeds from the GT log
            self.cmd_t = self.gt_t
            self.cmd_l = self.gt_world.get("wl", np.zeros_like(self.gt_t))
            self.cmd_r = self.gt_world.get("wr", np.zeros_like(self.gt_t))

    def _load_telemetry(self) -> None:
        p = self.run_dir / "autonomy" / "telemetry.jsonl"
        self.telemetry: list[dict] = []
        if p.exists():
            with p.open(encoding="utf-8") as f:
                self.telemetry = [json.loads(line) for line in f if line.strip()]
        self.tel_t = np.array([d["t"] for d in self.telemetry], np.float64)

    def _index_debug(self) -> None:
        d = self.run_dir / "autonomy"
        files = sorted((d / "debug").glob("tick_*.npz")) or sorted(d.glob("debug_*.npz"))
        self.debug_files = files
        ticks: list[DebugTick] = []
        has_left: list[bool] = []
        for i, f in enumerate(files):
            with np.load(f, allow_pickle=False) as z:
                has_left.append("extra_left_rgb" in z.files)  # the runner logs the image every debug_image_every_n ticks
                ticks.append(DebugTick(
                    index=i, t=float(z["t"]), pose_xy_yaw=tuple(float(v) for v in z["pose_xy_yaw"]),  # type: ignore[arg-type]
                    v_cap_mps=float(z["v_cap_mps"]), r_cert_m=float(z["r_cert_m"]),
                    health=_json_or_empty(z["health_json"]) if "health_json" in z.files else {},
                    extras=_json_or_empty(z["extras_json"]) if "extras_json" in z.files else {},
                ))
        order = np.argsort([k.t for k in ticks], kind="stable")
        self.debug_ticks = [ticks[i] for i in order]
        self.debug_files = [files[i] for i in order]
        self.debug_has_left = np.array([has_left[i] for i in order], bool)
        for i, k in enumerate(self.debug_ticks):
            k.index = i
        self.debug_t = np.array([k.t for k in self.debug_ticks], np.float64)
        self.debug_pose = np.array([k.pose_xy_yaw for k in self.debug_ticks], np.float64).reshape(-1, 3)

    def _compile_mode_changes(self) -> list[tuple[float, str, str]]:
        """(t, mode, reason) at every mode change, from debug extras (10 Hz) or telemetry (2 Hz)."""
        seq = [(k.t, k.mode, k.reason) for k in self.debug_ticks] or [(d["t"], d["mode"], d.get("reason", "")) for d in self.telemetry]
        out: list[tuple[float, str, str]] = []
        for t, m, r in seq:
            if not out or out[-1][1] != m:
                out.append((float(t), m, r))
        return out

    @functools.lru_cache(maxsize=DEBUG_CACHE)  # noqa: B019 - bounded, per-instance lifetime is fine here
    def _arrays(self, index: int) -> dict[str, np.ndarray]:
        with np.load(self.debug_files[index], allow_pickle=False) as z:
            return {k: z[k] for k in z.files if not k.endswith("_json")}

    # ------------------------------------------------------------------ sampling
    @property
    def t_start(self) -> float:
        return float(self.gt_t[0])

    @property
    def t_end(self) -> float:
        return float(self.gt_t[-1])

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def gt_pose_at(self, t: float) -> tuple[float, float, float, float]:
        """Interpolated GT (x_a, y_a, yaw_a, v) at time t (clamped to the log)."""
        tt = float(np.clip(t, self.gt_t[0], self.gt_t[-1]))
        x = float(np.interp(tt, self.gt_t, self.gt_xa))
        y = float(np.interp(tt, self.gt_t, self.gt_ya))
        yaw = float(wrap_angle(np.interp(tt, self.gt_t, self.gt_yaw_unwrapped)))
        v = float(np.interp(tt, self.gt_t, self.gt_v))
        return x, y, yaw, v

    def world_state_at(self, t: float) -> dict:
        """RendererProto state dict (WORLD frame) at time t.

        ``dynamic`` holds one ``{"id", "xy"}`` entry per scripted actor, interpolated from the GT
        ``dyn_xy`` log (WORLD metres) when the run has one; lighting events are evaluated by the
        renderer from ``t``, so ``lighting`` stays empty (no overrides)."""
        tt = float(np.clip(t, self.gt_t[0], self.gt_t[-1]))
        g = self.gt_world
        pose = [float(np.interp(tt, self.gt_t, g[k])) if k in g else 0.0 for k in ("x", "y", "z", "roll", "pitch")]
        yaw_w = float(wrap_angle(np.interp(tt, self.gt_t, np.unwrap(g["yaw"]))))
        dynamic = []
        dyn = g.get("dyn_xy")
        if dyn is not None and dyn.ndim == 3 and dyn.shape[0] == len(self.gt_t):
            for i in range(dyn.shape[1]):
                dynamic.append({"id": i, "xy": [float(np.interp(tt, self.gt_t, dyn[:, i, 0])),
                                                float(np.interp(tt, self.gt_t, dyn[:, i, 1]))]})
        return {"pose": pose + [yaw_w], "t": tt, "dynamic": dynamic, "lighting": {}}

    def tick_at(self, t: float) -> Optional[DebugTick]:
        """Latest debug bundle at or before t, with its arrays attached."""
        i = _held_index(self.debug_t, t)
        if i < 0:
            return None
        k = self.debug_ticks[i]
        k.arrays = self._arrays(i)
        return k

    def telemetry_at(self, t: float) -> Optional[dict]:
        i = _held_index(self.tel_t, t)
        return self.telemetry[i] if i >= 0 else None

    def link_stats_at(self, t: float, window_s: float = LINK_WINDOW_S) -> tuple[float, float]:
        """(packets per s, kbit/s) measured from the telemetry log over (t - window, t].

        The window is clipped to the start of the log so the first seconds are not under-reported."""
        hi = _held_index(self.tel_t, t) + 1
        if hi <= 0:
            return 0.0, 0.0
        lo = int(np.searchsorted(self.tel_t, t - window_s, side="right"))
        span = min(window_s, max(t - self.t_start, 1.0 / defaults.TELEMETRY_HZ))
        n = hi - lo
        bits = 8.0 * sum(int(d.get("packet_bytes") or 0) for d in self.telemetry[lo:hi])
        return n / span, bits / span / 1000.0

    def left_tick_at(self, t: float) -> Optional[DebugTick]:
        """Latest debug bundle at or before t that carries an onboard image (``extra_left_rgb``).

        Bundles log the image only every ``debug_image_every_n`` ticks; holding the last image (with
        that bundle's own overlays) avoids a flicker between image and no-image ticks."""
        i = _held_index(self.debug_t, t)
        while i >= 0 and not self.debug_has_left[i]:
            i -= 1
        if i < 0:
            return None
        k = self.debug_ticks[i]
        k.arrays = self._arrays(i)
        return k

    def _left_image(self, tick: Optional[DebugTick]) -> Optional[np.ndarray]:
        if tick is None:
            return None
        if "extra_left_rgb" in tick.arrays:
            return tick.arrays["extra_left_rgb"]
        if self.left_renderer is None:
            return None
        if tick.index not in self._left_cache:
            if len(self._left_cache) > DEBUG_CACHE:
                self._left_cache.clear()
            self._left_cache[tick.index] = self.left_renderer(self.world_state_at(tick.t))
        return self._left_cache[tick.index]

    def frame_at(self, t: float) -> PanelData:
        """All panel data at time t (seconds, run clock)."""
        x, y, yaw, v = self.gt_pose_at(t)
        n_gt = int(np.searchsorted(self.gt_t, t, side="right"))
        gt_trail = np.column_stack([self.gt_xa[:n_gt], self.gt_ya[:n_gt]])
        if n_gt and (gt_trail[-1, 0] != x or gt_trail[-1, 1] != y):
            gt_trail = np.vstack([gt_trail, [x, y]])
        tick = self.tick_at(t)
        left_tick = self.left_tick_at(t) if self.debug_has_left.any() else tick
        tel = self.telemetry_at(t)
        n_dbg = _held_index(self.debug_t, t) + 1
        if n_dbg > 0:
            est_trail = self.debug_pose[:n_dbg, :2]
            est_pose = tuple(self.debug_pose[n_dbg - 1])
        elif tel is not None:
            ti = _held_index(self.tel_t, t) + 1
            est_trail = np.array([d["pose"][:2] for d in self.telemetry[:ti]], np.float64)
            est_pose = tuple(tel["pose"])
        else:
            est_trail, est_pose = np.zeros((0, 2)), (0.0, 0.0, 0.0)

        # scalar state: prefer the 10 Hz debug bundle, fall back to 2 Hz telemetry
        if tick is not None:
            mode, reason, v_cap, r_cert = tick.mode, tick.reason, tick.v_cap_mps, tick.r_cert_m
            q = float(tick.health.get("q", 1.0 - float(tick.health.get("p_fail", 0.0))))
            sigma = float(tick.health.get("pos_sigma_m", float("nan")))
            speed = float(tick.extras.get("speed_meas", tick.extras.get("cmd_v", v)))
            compute_ms = float(tick.health.get("compute_ms", float("nan")))
        elif tel is not None:
            mode, reason, v_cap, r_cert = tel["mode"], tel.get("reason", ""), tel["v_cap_mps"], tel["r_cert_m"]
            q = float(tel["health"].get("q", 1.0 - float(tel["health"].get("p_fail", 0.0))))
            sigma, speed = float(tel["pos_sigma_m"]), float(tel["speed_mps"])
            compute_ms = float(tel["health"].get("compute_ms", float("nan")))
        else:
            mode, reason, v_cap, r_cert, q, sigma, speed, compute_ms = "HOLD", "", 0.0, 0.0, 1.0, float("nan"), v, float("nan")

        # wheel commands over the last WHEEL_WINDOW_S (step-held samples)
        lo = int(np.searchsorted(self.cmd_t, t - WHEEL_WINDOW_S, side="left"))
        hi = int(np.searchsorted(self.cmd_t, t, side="right"))
        lo = max(0, lo - 1)
        wt = self.cmd_t[lo:hi] - t
        wl, wr = self.cmd_l[lo:hi], self.cmd_r[lo:hi]

        mode_log = [m for m in self._mode_changes if m[0] <= t + 1e-9][-MODE_LOG_LEN:]
        events = [e for e in self.events if t - EVENT_HOLD_S <= float(e.get("t", -1e9)) <= t]
        ws = self.world_state_at(t)
        chase = None
        if self.chase_renderer is not None:
            chase = self.chase_renderer(ws, *self.chase_size)
        rate_hz, kbps = self.link_stats_at(t)
        return PanelData(
            t=float(t), mission=self.mission, gt_pose_a=(x, y, yaw), gt_speed_mps=v, gt_trail_a=gt_trail,
            est_pose_a=(float(est_pose[0]), float(est_pose[1]), float(est_pose[2])), est_trail_a=np.asarray(est_trail, np.float64),
            mode=mode, reason=reason, v_cap_mps=float(v_cap), r_cert_m=float(r_cert), speed_mps=speed, q=q, pos_sigma_m=sigma,
            tick=tick, telemetry=tel, costmap_u4=decode_costmap_hex(tel.get("costmap_u4_hex")) if tel else None,
            wheel_t=wt, wheel_l=wl, wheel_r=wr, mode_log=mode_log, events=events, world_state=ws,
            chase_rgb=chase, left_rgb=self._left_image(left_tick), left_tick=left_tick, compute_ms=compute_ms,
            packet_bytes=(tel or {}).get("packet_bytes"), link_rate_hz=rate_hz, link_kbps=kbps,
            packet_age_s=float(t - tel["t"]) if tel is not None else float("nan"),
        )

    def frame_times(self, t0: Optional[float] = None, t1: Optional[float] = None, fps: float = VIDEO_FPS) -> np.ndarray:
        """Output frame timestamps in [t0, t1) at ``fps``."""
        a = self.t_start if t0 is None else float(t0)
        b = self.t_end if t1 is None else float(t1)
        n = max(0, int(math.floor((b - a) * fps + 1e-9)))
        return a + np.arange(n, dtype=np.float64) / fps

    def frames(self, t0: Optional[float] = None, t1: Optional[float] = None, fps: float = VIDEO_FPS) -> Iterator[PanelData]:
        """Yield :class:`PanelData` at ``fps`` between t0 and t1 (run clock)."""
        for t in self.frame_times(t0, t1, fps):
            yield self.frame_at(float(t))
