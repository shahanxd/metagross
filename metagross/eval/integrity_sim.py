"""VO integrity monitor and stereo-VO drift on rendered DEV drives (label: Simulated).

Two steps (rendering needs Chrome; the analysis is pure CPU and can be re-run):

``render``   scripted drive on a DEV scenario (seeds 100-129 only): the simulated vehicle
             follows the GT oracle path (:func:`metagross.sim.gridplan.plan_path`) with a
             pure-pursuit controller at a fixed speed; at the camera rate the real
             :class:`~metagross.sim.world.World` produces stereo ``SensorFrame`` s through the
             Three.js renderer (auto-exposure, lighting events, dynamic obstacles included).
             Frames (PNG) and per-frame GT (``T_world_cam``, pose, active lighting events,
             dynamic-obstacle range/bearing, AE exposure) are cached under
             ``results/raw/integrity_sim/<seed>/``.
``analyse``  replays the cached frames through the onboard pipeline pieces the stack uses
             (``Perception.compute_disparity`` -> :class:`Localizer` with wheels + gyro) and
             scores: per-frame VO failure label vs GT (same definition as the KITTI study:
             no pose, or relative error > 0.10 m / 1 deg), the q the node feeds to the
             supervisor, supervisor health levels (NOMINAL / CAUTION / DEGRADED) replayed
             with the real hysteresis, false alarms per minute on nominal frames, detection
             of lighting / sudden-obstacle events, AUROC, VO-only drift (% of distance).

Frames: camera = left camera, OpenCV axes (x right, y down, z forward); ``T_world_cam``
maps camera coordinates to the world frame (m). Times in s, distances in m.

Also: ``timing`` (paired previous-vs-current Localizer wall-clock on cached frames) and
``kitti-check`` (KITTI 07 VO regression of several VO configurations sharing one SGBM
pass -> ``results/kitti_vo_07_check.json``; canonical KITTI results are not touched).

Outputs: ``results/integrity_sim.json`` (model candidates: KITTI-only, Platt-recalibrated
on sim DEV, retrained on KITTI 07 + sim DEV; KITTI 05 and one DEV family held out),
``results/vo_sim_dev.json`` (VO drift with / without photometric normalisation), and with
``--adopt`` the chosen model in ``models/integrity.json`` (KITTI-only copy kept in
``models/integrity_kitti.json``).

Usage (PowerShell)::

    .venv\\Scripts\\python.exe -m metagross.eval.integrity_sim render --seeds 102 103
    .venv\\Scripts\\python.exe -m metagross.eval.integrity_sim replay --seeds 102 103
    .venv\\Scripts\\python.exe -m metagross.eval.integrity_sim timing --seeds 102 103
    .venv\\Scripts\\python.exe -m metagross.eval.integrity_sim analyse --seeds 102 103 --chosen kitti_only
    .venv\\Scripts\\python.exe -m metagross.eval.integrity_sim kitti-check
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import cv2
import numpy as np

from metagross.autonomy.localization.health import FEATURE_NAMES
from metagross.autonomy.localization.vo import rotation_angle
from metagross.config import defaults

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCEN_DIR = REPO_ROOT / "data" / "scenarios" / "dev"
CACHE_ROOT = REPO_ROOT / "results" / "raw" / "integrity_sim"
RESULTS = REPO_ROOT / "results"

DEV_SEEDS = range(100, 130)  # never EVAL seeds 0-59

# --- scripted drive
LOOKAHEAD_M = 2.0  # pure-pursuit lookahead along the oracle path
MAX_YAW_RATE_RPS = 0.8  # commanded yaw-rate clip (well inside the platform limit)
TURN_IN_PLACE_RAD = math.radians(60.0)  # rotate in place when the target is further off-axis
GOAL_STOP_M = 1.5  # stop the drive this close to the goal
SPEEDS_MPS = (1.0, 1.5, 2.0)  # drive speed by seed % 3 (covers the platform range, cap 2.0 m/s)
MAX_DRIVE_S = 75.0  # per drive (F5 lighting events all end before ~70 s)
F5_MIN_SPEED_MPS = 0.5  # slowest scripted speed used to stretch an F5 drive over its lighting events

# --- event annotation (evaluation side, GT)
DYN_NEAR_M = 5.0  # a triggered dynamic obstacle this close to the camera ...
DYN_FOV_MARGIN_RAD = math.radians(10.0)  # ... and inside the horizontal FOV (+ margin) is an event frame
EVENT_TAIL_S = 1.0  # frames this long after an event are neither nominal nor event (AE / EMA recovery)

DEGRADE_SEED_OFFSET = 5000  # synthetic-degradation schedule seed = offset + DEV seed (deterministic)

# --- VO failure label (same as metagross.eval.integrity_train)
FAIL_TRANS_M = 0.10
FAIL_ROT_DEG = 1.0


# ============================================================================ render
def load_dev_scenario(seed: int) -> dict:
    """Scenario dict of a DEV seed (refuses EVAL seeds)."""
    if seed not in DEV_SEEDS:
        raise ValueError(f"seed {seed} is not a DEV seed (100-129)")
    from metagross.sim.scenario import load_scenario

    return load_scenario(SCEN_DIR / f"{seed}.json")


def drive_speed(seed: int, scenario: Optional[dict] = None, path_len_m: float = math.inf) -> float:
    """Deterministic scripted drive speed (m/s) for a seed.

    F5 (lighting) drives are slowed so the vehicle is still driving when the last lighting
    event ends (+ ``EVENT_TAIL_S``), otherwise late glare / dim / dust events would be missed."""
    v = SPEEDS_MPS[seed % len(SPEEDS_MPS)]
    events = (scenario or {}).get("lighting", {}).get("events", [])
    if events and math.isfinite(path_len_m):
        t_end = min(max(float(ev["t1"]) for ev in events) + EVENT_TAIL_S, MAX_DRIVE_S)
        v = min(v, max(path_len_m / t_end, F5_MIN_SPEED_MPS))
    return v


@dataclass
class PurePursuit:
    """Pure-pursuit follower of a world-frame polyline (m); outputs body (v m/s, omega rad/s)."""

    path: np.ndarray  # (N, 2) world xy
    speed: float
    lookahead: float = LOOKAHEAD_M
    _i: int = 0
    _arc: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self._arc = np.concatenate([[0.0], np.cumsum(seg)])

    def progress(self, p: np.ndarray) -> float:
        """Arc length (m) of the closest point of the polyline to ``p``, searching from the current
        segment onward (monotone progress)."""
        a, b = self.path[self._i:-1], self.path[self._i + 1:]
        ab = b - a
        r = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0.0, 1.0)
        d = np.linalg.norm(a + r[:, None] * ab - p, axis=1)
        j = int(np.argmin(d))
        self._i += j
        return float(self._arc[self._i] + r[j] * np.linalg.norm(ab[j]))

    def command(self, x: float, y: float, yaw: float) -> tuple[float, float]:
        p = np.array([x, y])
        s_target = self.progress(p) + self.lookahead
        k = int(np.searchsorted(self._arc, s_target))
        if k >= len(self.path):
            target = self.path[-1]
        else:
            s0, s1 = self._arc[k - 1], self._arc[k]
            r = (s_target - s0) / max(s1 - s0, 1e-9)
            target = self.path[k - 1] + r * (self.path[k] - self.path[k - 1])
        dx, dy = target - p
        c, s = math.cos(yaw), math.sin(yaw)
        bx, by = c * dx + s * dy, -s * dx + c * dy
        alpha = math.atan2(by, bx)
        if abs(alpha) > TURN_IN_PLACE_RAD:
            return 0.0, math.copysign(MAX_YAW_RATE_RPS, alpha)
        L = max(math.hypot(bx, by), 1e-3)
        omega = self.speed * 2.0 * math.sin(alpha) / L
        return self.speed, float(np.clip(omega, -MAX_YAW_RATE_RPS, MAX_YAW_RATE_RPS))


def wheel_rates(v: float, omega: float) -> tuple[float, float]:
    """Skid-steer inverse kinematics with the nominal chi: body (v m/s, omega rad/s) -> wheel rad/s."""
    veh = defaults.VEHICLE
    half = 0.5 * omega * veh.track_width_m * defaults.CHI_NOMINAL
    return (v - half) / veh.wheel_radius_m, (v + half) / veh.wheel_radius_m


def _dyn_geometry(world: Any, T_world_cam: np.ndarray) -> tuple[float, float, bool]:
    """(range m, bearing rad, triggered) of the nearest dynamic obstacle w.r.t. the camera."""
    best = (math.inf, 0.0, False)
    T_cam_world = np.linalg.inv(T_world_cam)
    for d in world.dynamic:
        xy = d.xy_at(world.t)
        z = float(world.terrain.height_at(float(xy[0]), float(xy[1]))) + 0.5 * d.size[2]
        pc = T_cam_world @ np.array([xy[0], xy[1], z, 1.0])
        rng = float(np.linalg.norm(pc[:3]))
        if rng < best[0]:
            best = (rng, math.atan2(float(pc[0]), float(pc[2])), d.t_trigger is not None)
    return best


def _exposure_value(renderer: Any) -> float:
    """Auto-exposure value reported by the renderer for the last stereo frame (NaN if unknown)."""
    ex = (getattr(renderer, "last_js", None) or {}).get("exposure")
    if isinstance(ex, dict):
        for key in ("E", "value", "gain"):
            if isinstance(ex.get(key), (int, float)):
                return float(ex[key])
        return math.nan
    return float(ex) if isinstance(ex, (int, float)) else math.nan


def render_drive(seed: int, renderer: Any, out_root: Path = CACHE_ROOT, max_s: float = MAX_DRIVE_S) -> Path:
    """Scripted oracle-path drive on DEV ``seed``; caches stereo frames + GT. Returns the cache dir."""
    from metagross.contracts.messages import DriveMode, WheelCmd
    from metagross.sim.gridplan import plan_path
    from metagross.sim.world import World

    sc = load_dev_scenario(seed)
    world = World(sc, sensor_mode="stereo", renderer=renderer, terminal=frozenset())
    hz = world.hazards
    plan = plan_path(hz.grid, hz.hazard, tuple(world.start_xy), tuple(world.goal_xy))
    if not plan.ok:
        raise RuntimeError(f"seed {seed}: no oracle path")
    speed = drive_speed(seed, sc, plan.length_m)
    ctrl = PurePursuit(plan.path_xy, speed)
    out = out_root / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    steps_per_frame = int(round(defaults.PHYSICS_HZ / defaults.CAMERA_HZ_BATCH))
    rec: dict[str, list] = {k: [] for k in ("t", "wl", "wr", "gyro", "T_world_cam", "pose6", "light", "dyn_range",
                                            "dyn_bearing", "dyn_triggered", "exposure", "odo_m")}
    k = 0
    t0 = time.perf_counter()
    while world.t <= max_s + 1e-9:
        frame = world.make_sensor_frame(world.t, k)
        Twc = world.camera_pose()
        cv2.imwrite(str(out / f"L_{k:04d}.png"), cv2.cvtColor(frame.left_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"R_{k:04d}.png"), frame.right_gray)
        rng, brg, trig = _dyn_geometry(world, Twc)
        light = world.lighting_state()
        rec["t"].append(frame.t)
        rec["wl"].append(frame.wheel_angle_l_rad)
        rec["wr"].append(frame.wheel_angle_r_rad)
        rec["gyro"].append(frame.gyro_z_rps)
        rec["T_world_cam"].append(Twc)
        rec["pose6"].append(world.state.pose6())
        rec["light"].append(",".join(sorted(ev["type"] for ev in light["active_events"])))
        rec["dyn_range"].append(rng)
        rec["dyn_bearing"].append(brg)
        rec["dyn_triggered"].append(trig)
        rec["exposure"].append(_exposure_value(renderer))
        rec["odo_m"].append(world.state.odo_m)
        s = world.state
        if math.hypot(world.goal_xy[0] - s.x, world.goal_xy[1] - s.y) < GOAL_STOP_M:
            break
        v, om = ctrl.command(s.x, s.y, s.yaw)
        wl, wr = wheel_rates(v, om)
        world.queue_command(WheelCmd(world.t, k, wl, wr, DriveMode.NOMINAL, 0.0))
        for _ in range(steps_per_frame):
            world.step()
        k += 1
    meta = {key: np.asarray(v) for key, v in rec.items()}
    np.savez_compressed(out / "meta.npz", seed=seed, family=sc["family"], speed_mps=speed, **meta)
    LOG.info("seed %d (%s): %d frames, %.1f m, %.0f s wall", seed, sc["family"], len(rec["t"]), world.state.odo_m,
             time.perf_counter() - t0)
    return out


def render_many(seeds: list[int], max_s: float = MAX_DRIVE_S, skip_existing: bool = True) -> None:
    """Render several DEV drives with one renderer instance."""
    from metagross.sim.render.bridge import ThreeRenderer

    cv2.setNumThreads(2)
    r = ThreeRenderer(defaults.stereo_calibration(), defaults.VEHICLE)
    try:
        for seed in seeds:
            if skip_existing and (CACHE_ROOT / str(seed) / "meta.npz").exists():
                LOG.info("seed %d cached, skipping", seed)
                continue
            render_drive(seed, r, max_s=max_s)
    finally:
        r.close()


# ============================================================================ replay
@dataclass
class Drive:
    """Cached rendered drive (see :func:`render_drive`)."""

    seed: int
    family: str
    speed_mps: float
    meta: dict[str, np.ndarray]
    dir: Path

    @classmethod
    def load(cls, seed: int, root: Path = CACHE_ROOT) -> "Drive":
        d = root / str(seed)
        z = np.load(d / "meta.npz", allow_pickle=False)
        meta = {k: z[k] for k in z.files}
        return cls(seed, str(meta.pop("family")), float(meta.pop("speed_mps")), meta, d)

    @property
    def n(self) -> int:
        return int(self.meta["t"].shape[0])

    def frames(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        for k in range(self.n):
            left = cv2.cvtColor(cv2.imread(str(self.dir / f"L_{k:04d}.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            right = cv2.imread(str(self.dir / f"R_{k:04d}.png"), cv2.IMREAD_GRAYSCALE)
            yield k, left, right


def event_masks(drive: Drive) -> dict[str, np.ndarray]:
    """Per-frame event annotation (GT side): lighting kinds, sudden obstacle near/in view, nominal."""
    m = drive.meta
    t = m["t"].astype(float)
    light = m["light"].astype(str)
    out: dict[str, np.ndarray] = {}
    for kind in ("glare", "dim", "dust"):
        out[kind] = np.array([kind in s.split(",") for s in light])
    hfov = math.radians(defaults.HFOV_DEG) / 2.0 + DYN_FOV_MARGIN_RAD
    out["obstacle"] = (m["dyn_triggered"].astype(bool) & (m["dyn_range"] < DYN_NEAR_M)
                       & (np.abs(m["dyn_bearing"]) < hfov))
    any_ev = out["glare"] | out["dim"] | out["dust"] | out["obstacle"]
    tail = np.zeros_like(any_ev)
    for k in np.nonzero(any_ev)[0]:
        tail |= (t > t[k]) & (t <= t[k] + EVENT_TAIL_S)
    out["any_event"] = any_ev
    out["nominal"] = ~any_ev & ~tail
    return out


def gt_relative(drive: Drive) -> np.ndarray:
    """(N, 4, 4) GT relative camera poses T_prev_cur (identity at k=0)."""
    T = drive.meta["T_world_cam"].astype(float)
    rel = np.tile(np.eye(4), (T.shape[0], 1, 1))
    for k in range(1, T.shape[0]):
        rel[k] = np.linalg.inv(T[k - 1]) @ T[k]
    return rel


def degraded_frames(drive: Drive, seed: int) -> Iterator[tuple[int, np.ndarray, np.ndarray, str]]:
    """Frames of ``drive`` with synthetic degradation segments (:mod:`metagross.eval.degrade`, same
    schedule generator as the KITTI study). Degraded frames are gray (3 equal channels); yields
    (k, left_rgb, right_gray, kind)."""
    from metagross.eval.degrade import Degrader, depth_from_sgbm, make_schedule

    sched = make_schedule(drive.n, seed)
    sched[0] = None  # the VO init frame stays clean
    degr: dict[int, Degrader] = {}
    calib = defaults.stereo_calibration()
    for k, left, right in drive.frames():
        item = sched[k]
        if item is None:
            yield k, left, right, "clean"
            continue
        deg, seg_seed = item
        gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        if seg_seed not in degr:
            degr[seg_seed] = Degrader(deg, gray.shape, seg_seed)
        depth = depth_from_sgbm(gray, right, calib.fx, calib.baseline_m) if deg.kind == "haze" else None
        gl, gr = degr[seg_seed](gray, right, depth, frame=k)
        yield k, np.dstack([gl] * 3), gr, deg.kind


def replay(drive: Drive, extra_vo: Optional[dict[str, Any]] = None,
           degrade_seed: Optional[int] = None) -> dict[str, np.ndarray]:
    """Run perception disparity + the onboard Localizer (default config) over a cached drive.

    ``extra_vo``: name -> :class:`VOConfig` of additional bare VO instances fed the same
    frames and disparity (ablations share one SGBM pass); their relative poses and VO times are
    returned as ``rel_vo_<name>`` / ``vo_ms_<name>``.

    Returns per-frame arrays: X (N, 12 raw features), p_fail, q_inst, q (smoothed, what the
    node uses), q_gate, vo_ok, vo_accepted, rel_vo (N, 4, 4; NaN where VO failed), ekf_pose
    (N, 3 A-frame x, y, yaw), loc_ms, vo_ms, disp_ms, photo_gain.
    """
    from metagross.autonomy.localization.localizer import Localizer
    from metagross.autonomy.localization.vo import StereoVO
    from metagross.autonomy.perception.pipeline import Perception
    from metagross.contracts.messages import SensorFrame

    calib = defaults.stereo_calibration()
    per = Perception(calib, defaults.VEHICLE)
    loc = Localizer(calib, defaults.VEHICLE)
    extras = {name: StereoVO(calib.K, calib.baseline_m, cfg) for name, cfg in (extra_vo or {}).items()}
    n = drive.n
    X = np.zeros((n, len(FEATURE_NAMES)))
    cols = {k: np.zeros(n) for k in ("p_fail", "q_inst", "q", "q_gate", "vo_ok", "vo_accepted", "loc_ms", "vo_ms",
                                     "disp_ms", "photo_gain")}
    rel_vo = np.full((n, 4, 4), np.nan)
    ekf_pose = np.zeros((n, 3))
    ext_rel = {name: np.full((n, 4, 4), np.nan) for name in extras}
    ext_ms = {name: np.zeros(n) for name in extras}
    kinds = np.array(["clean"] * n, dtype=object)
    m = drive.meta
    src = (((k, a, b, "clean") for k, a, b in drive.frames()) if degrade_seed is None
           else degraded_frames(drive, degrade_seed))
    for k, left, right, kind in src:
        kinds[k] = kind
        fr = SensorFrame(t=float(m["t"][k]), seq=k, left_rgb=left, right_gray=right,
                         wheel_angle_l_rad=float(m["wl"][k]), wheel_angle_r_rad=float(m["wr"][k]),
                         gyro_z_rps=float(m["gyro"][k]), sensor_mode="stereo")
        t0 = time.perf_counter()
        disp = per.compute_disparity(fr)
        cols["disp_ms"][k] = (time.perf_counter() - t0) * 1e3
        out = loc.update(fr, disp)
        h = out["health"]
        X[k] = [h[f] for f in FEATURE_NAMES]
        for key in ("p_fail", "q_inst", "q", "q_gate"):
            cols[key][k] = h[key]
        cols["vo_ok"][k] = float(out["vo_ok"])
        cols["vo_accepted"][k] = float(out["vo_accepted"])
        cols["loc_ms"][k] = out["timings_ms"]["total"]
        cols["vo_ms"][k] = out["timings_ms"].get("vo", np.nan)
        last = loc.last_vo
        if last is not None and last.ok:
            rel_vo[k] = last.T_prev_cur
        cols["photo_gain"][k] = last.stats.get("photo_gain", 1.0) if last is not None else 1.0
        ekf_pose[k] = out["pose_xy_yaw"]
        gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        for name, vo in extras.items():
            t1 = time.perf_counter()
            r = vo.process(gray, None, disp, fr.t)
            ext_ms[name][k] = (time.perf_counter() - t1) * 1e3
            if r.ok:
                ext_rel[name][k] = r.T_prev_cur
    out_d: dict[str, np.ndarray] = {"X": X, "rel_vo": rel_vo, "ekf_pose": ekf_pose, "degrade_kind": kinds.astype(str),
                                    **cols}
    for name in extras:
        out_d[f"rel_vo_{name}"] = ext_rel[name]
        out_d[f"vo_ms_{name}"] = ext_ms[name]
    return out_d


def replay_path(seed: int, tag: str = "") -> Path:
    return CACHE_ROOT / str(seed) / f"replay{('_' + tag) if tag else ''}.npz"


def replay_seed(seed: int, tag: str = "", ablations: bool = True, degrade: bool = False) -> Path:
    """Replay one cached drive and store the per-frame outputs (``replay[_tag].npz``). ``degrade``:
    add synthetic degradation segments (schedule seed = ``DEGRADE_SEED_OFFSET + seed``)."""
    from metagross.autonomy.localization.vo import VOConfig

    cv2.setNumThreads(2)
    drive = Drive.load(seed)
    extra = {"nophoto": VOConfig(photometric_norm=False)} if ablations else None
    t0 = time.perf_counter()
    rep = replay(drive, extra, degrade_seed=(DEGRADE_SEED_OFFSET + seed) if degrade else None)
    out = replay_path(seed, tag)
    np.savez_compressed(out, **rep)
    LOG.info("replayed seed %d: %d frames in %.0f s", seed, drive.n, time.perf_counter() - t0)
    return out


def vo_labels(rel_vo: np.ndarray, rel_gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame (label, trans err m, rot err deg); label = VO failure (no pose or error above limits)."""
    n = rel_gt.shape[0]
    terr, rerr = np.full(n, np.nan), np.full(n, np.nan)
    ok = np.isfinite(rel_vo[:, 0, 0])
    for k in range(1, n):
        if ok[k]:
            E = np.linalg.inv(rel_vo[k]) @ rel_gt[k]
            terr[k] = float(np.linalg.norm(E[:3, 3]))
            rerr[k] = math.degrees(rotation_angle(E[:3, :3]))
    label = (~ok) | (terr > FAIL_TRANS_M) | (rerr > FAIL_ROT_DEG)
    label[0] = False
    return label, terr, rerr


def vo_drift_pct(rel_vo: np.ndarray, rel_gt: np.ndarray) -> dict[str, float]:
    """VO-only drift: chain VO increments (failed frames bridged with the last VO motion) and compare
    the end position with GT; % of GT path length. Also the max position error along the drive."""
    n = rel_gt.shape[0]
    P_vo, P_gt = np.eye(4), np.eye(4)
    last = np.eye(4)
    path = 0.0
    err_max = 0.0
    for k in range(1, n):
        if np.isfinite(rel_vo[k, 0, 0]):
            last = rel_vo[k]
        P_vo = P_vo @ last
        P_gt = P_gt @ rel_gt[k]
        path += float(np.linalg.norm(rel_gt[k][:3, 3]))
        err_max = max(err_max, float(np.linalg.norm(P_vo[:3, 3] - P_gt[:3, 3])))
    end_err = float(np.linalg.norm(P_vo[:3, 3] - P_gt[:3, 3]))
    return {"path_m": path, "end_err_m": end_err, "drift_pct": 100.0 * end_err / max(path, 1e-9),
            "max_err_m": err_max, "vo_fail_frames": int((~np.isfinite(rel_vo[1:, 0, 0])).sum())}


def supervisor_levels(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Health level per frame (0 NOMINAL, 1 CAUTION, 2 DEGRADED) with the supervisor's real hysteresis."""
    from metagross.autonomy.safety.supervisor import Supervisor

    sup = Supervisor()
    return np.array([sup._health_level(float(tt), float(qq)) for tt, qq in zip(t, q)], int)  # noqa: SLF001


def level_entries(levels: np.ndarray, level: int) -> np.ndarray:
    """Frame indices where the level first reaches >= ``level`` (event onsets)."""
    at = levels >= level
    return np.nonzero(at & ~np.concatenate([[False], at[:-1]]))[0]


# ============================================================================ scoring
FRAME_DT_S = 1.0 / defaults.CAMERA_HZ_BATCH
LEVEL_CAUTION, LEVEL_DEGRADED = 1, 2


@dataclass
class SeedData:
    """Everything the scoring needs for one replayed drive."""

    seed: int
    family: str
    t: np.ndarray
    X: np.ndarray  # (N, 12) raw features
    label: np.ndarray  # VO failure vs GT
    terr: np.ndarray
    rerr: np.ndarray
    events: dict[str, np.ndarray]
    rep: dict[str, np.ndarray]
    rel_gt: np.ndarray

    @classmethod
    def load(cls, seed: int, tag: str = "") -> "SeedData":
        drive = Drive.load(seed)
        rep = dict(np.load(replay_path(seed, tag)))
        rel_gt = gt_relative(drive)
        label, terr, rerr = vo_labels(rep["rel_vo"], rel_gt)
        return cls(seed, drive.family, drive.meta["t"].astype(float), rep["X"], label, terr, rerr,
                   event_masks(drive), rep, rel_gt)


def monitor_q(model: Any, X: np.ndarray, config: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """(p_fail, smoothed q) per frame with the onboard monitor logic (frame 0 = VO warm-up)."""
    from metagross.autonomy.localization.health import IntegrityMonitor

    mon = IntegrityMonitor(model=model, config=config)
    p, q = np.zeros(X.shape[0]), np.zeros(X.shape[0])
    for k in range(X.shape[0]):
        h = mon.update_features(dict(zip(FEATURE_NAMES, X[k])), warmup=(k == 0))
        p[k], q[k] = h["p_fail"], h["q"]
    return p, q


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges of the True runs of ``mask``."""
    out, start = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    return out


def score_model(model: Any, data: list[SeedData], config: Any = None) -> dict[str, Any]:
    """False alarms on nominal frames, event detection and AUROC of one integrity model.

    A DEGRADED / CAUTION onset on a nominal frame is counted either way; it is a *false alarm*
    when no VO failure (GT label) occurred within ``EVENT_TAIL_S`` before the onset."""
    from sklearn.metrics import roc_auc_score

    nominal_frames = 0
    caution_onsets = degraded_onsets = caution_false = degraded_false = 0
    nom_q_caution = nom_q_degraded = 0
    events: list[dict[str, Any]] = []
    p_all, y_all, ok_all = [], [], []
    look = int(round(EVENT_TAIL_S / FRAME_DT_S))
    for d in data:
        p, q = monitor_q(model, d.X, config)
        lv = supervisor_levels(d.t, q)
        nom = d.events["nominal"].copy()
        nom[0] = False
        nominal_frames += int(nom.sum())
        nom_q_caution += int((nom & (q <= 0.7)).sum())
        nom_q_degraded += int((nom & (q <= 0.4)).sum())
        for lvl in (LEVEL_CAUTION, LEVEL_DEGRADED):
            on = [k for k in level_entries(lv, lvl) if nom[k]]
            false = [k for k in on if not d.label[max(k - look, 0):k + 1].any()]
            if lvl == LEVEL_CAUTION:
                caution_onsets += len(on)
                caution_false += len(false)
            else:
                degraded_onsets += len(on)
                degraded_false += len(false)
        tail = int(round(EVENT_TAIL_S / FRAME_DT_S))
        for kind in ("glare", "dim", "dust", "obstacle"):
            for a, b in _segments(d.events[kind]):
                w = slice(a, min(b + tail, d.t.size))
                events.append({
                    "seed": d.seed, "family": d.family, "kind": kind, "t0_s": round(float(d.t[a]), 2),
                    "dur_s": round(float((b - a) * FRAME_DT_S), 2),
                    "vo_fail_frames": int(d.label[w].sum()),
                    "vo_terr_max_m": float(np.nanmax(d.terr[w])) if np.isfinite(d.terr[w]).any() else None,
                    "q_min": float(q[w].min()), "max_level": int(lv[w].max()),
                })
        p_all.append(p[1:])
        y_all.append(d.label[1:])
        ok_all.append(np.isfinite(d.rep["rel_vo"][1:, 0, 0]) if "rel_vo" in d.rep else np.ones(d.t.size - 1, bool))
    y, pp, okv = np.concatenate(y_all), np.concatenate(p_all), np.concatenate(ok_all)
    minutes = nominal_frames * FRAME_DT_S / 60.0
    degr = [e for e in events if e["vo_fail_frames"] > 0]
    out = {
        "nominal_minutes": minutes,
        "nominal_frames": nominal_frames,
        "nominal_frac_q_le_caution_0p7": nom_q_caution / max(nominal_frames, 1),
        "nominal_frac_q_le_degraded_0p4": nom_q_degraded / max(nominal_frames, 1),
        "nominal_caution_onsets": caution_onsets,
        "nominal_degraded_onsets": degraded_onsets,
        "nominal_caution_false_alarms": caution_false,
        "nominal_degraded_false_alarms": degraded_false,
        "caution_onsets_per_min": caution_onsets / max(minutes, 1e-9),
        "caution_false_alarms_per_min": caution_false / max(minutes, 1e-9),
        "degraded_onsets_per_5min": 5.0 * degraded_onsets / max(minutes, 1e-9),
        "degraded_false_alarms_per_5min": 5.0 * degraded_false / max(minutes, 1e-9),
        "frames_scored": int(y.size),
        "vo_failure_frames": int(y.sum()),
        "auroc_vo_failure": float(roc_auc_score(y, pp)) if 0 < y.sum() < y.size else None,
        "silent_failure_frames": int((y & okv).sum()),
        "auroc_silent_only": (float(roc_auc_score(y[okv], pp[okv])) if 0 < int((y & okv).sum()) < int(okv.sum())
                              else None),
        "events": events,
        "events_total": len(events),
        "events_with_vo_failure": len(degr),
        "events_with_vo_failure_flagged_caution": sum(e["max_level"] >= LEVEL_CAUTION for e in degr),
        "events_with_vo_failure_flagged_degraded": sum(e["max_level"] >= LEVEL_DEGRADED for e in degr),
        "events_flagged_caution": sum(e["max_level"] >= LEVEL_CAUTION for e in events),
    }
    return out


def vo_drift_table(data: list[SeedData]) -> list[dict[str, Any]]:
    """VO-only drift per drive, with and without photometric normalisation (same frames/disparity)."""
    rows = []
    for d in data:
        row = {"seed": d.seed, "family": d.family, "frames": int(d.t.size)}
        for name, key in (("photometric", "rel_vo"), ("no_photometric", "rel_vo_nophoto")):
            if key in d.rep:
                lab, te, _ = vo_labels(d.rep[key], d.rel_gt)
                dr = vo_drift_pct(d.rep[key], d.rel_gt)
                row[name] = {**dr, "vo_failure_label_frames": int(lab.sum()),
                             "terr_median_mm": float(1e3 * np.nanmedian(te))}
        rows.append(row)
    return rows


def timing_summary(data: list[SeedData]) -> dict[str, float]:
    """Localizer / VO wall-clock per frame over all replayed frames (ms; shared, loaded machine)."""
    loc = np.concatenate([d.rep["loc_ms"][1:] for d in data])
    vo = np.concatenate([d.rep["vo_ms"][1:] for d in data])
    return {"frames": int(loc.size), "localizer_ms_median": float(np.median(loc)),
            "localizer_ms_p95": float(np.percentile(loc, 95)), "vo_ms_median": float(np.median(vo)),
            "vo_ms_p95": float(np.percentile(vo, 95))}


# ============================================================================ recalibration / retraining
KITTI_RAW = REPO_ROOT / "results" / "raw"
MODEL_DIR = REPO_ROOT / "models"


def transformed(X: np.ndarray) -> np.ndarray:
    """Raw feature rows (N, 12) -> transformed rows (the model's input before standardisation)."""
    from metagross.autonomy.localization.health import transform_features

    return np.stack([transform_features(dict(zip(FEATURE_NAMES, row))) for row in X]) if len(X) else X


def kitti_set(seq: str) -> tuple[np.ndarray, np.ndarray]:
    """(raw features, VO-failure label) of a collected degraded-KITTI sequence (frame 0 dropped)."""
    z = np.load(KITTI_RAW / f"integrity_{seq}.npz", allow_pickle=False)
    return z["X"][1:], z["label"][1:].astype(bool)


def sim_set(data: list[SeedData]) -> tuple[np.ndarray, np.ndarray]:
    """(raw features, VO-failure label) of rendered drives (frame 0 = VO warm-up dropped)."""
    if not data:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0, bool)
    return np.concatenate([d.X[1:] for d in data]), np.concatenate([d.label[1:] for d in data])


def auroc(model: Any, X: np.ndarray, y: np.ndarray) -> Optional[float]:
    """AUROC of p_fail for the VO-failure label (None if one class is absent)."""
    from sklearn.metrics import roc_auc_score

    if not (0 < int(y.sum()) < y.size):
        return None
    return float(roc_auc_score(y, model.logit(transformed(X))))


def platt_recalibrate(model: Any, X: np.ndarray, y: np.ndarray, C: float = 1.0) -> Any:
    """Platt scaling p' = sigmoid(a * logit + b) fitted on (X, y); folded into the model's weights
    (coef' = a coef, intercept' = a intercept + b), so the onboard schema is unchanged."""
    from sklearn.linear_model import LogisticRegression

    from metagross.autonomy.localization.health import LogisticIntegrityModel

    z = model.logit(transformed(X)).reshape(-1, 1)
    lr = LogisticRegression(C=C, max_iter=2000).fit(z, y.astype(int))
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    return LogisticIntegrityModel(model.mean, model.scale, a * model.coef, a * model.intercept + b,
                                  source="platt", meta={"platt_a": a, "platt_b": b})


def retrain(sets: list[tuple[np.ndarray, np.ndarray]], C: float = 1.0) -> Any:
    """Logistic model on several (X, y) sets, each set weighted to the same total weight (so the
    larger domain does not dominate); same 12 features and transforms as the onboard model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from metagross.autonomy.localization.health import LogisticIntegrityModel

    Xt = np.concatenate([transformed(X) for X, _ in sets])
    y = np.concatenate([yy for _, yy in sets]).astype(int)
    w = np.concatenate([np.full(len(yy), 1.0 / max(len(yy), 1)) for _, yy in sets])
    w *= len(w) / w.sum()
    sc = StandardScaler().fit(Xt)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Xt), y, sample_weight=w)
    return LogisticIntegrityModel(sc.mean_, sc.scale_, clf.coef_.ravel(), float(clf.intercept_[0]), source="retrained")


# ============================================================================ analysis driver
INTEGRITY_SIM_JSON = RESULTS / "integrity_sim.json"
VO_SIM_JSON = RESULTS / "vo_sim_dev.json"
TIMING_JSON = CACHE_ROOT / "localizer_timing_paired.json"  # merged into vo_sim_dev.json by analyse
KITTI_MODEL_PATH = MODEL_DIR / "integrity_kitti.json"  # KITTI-only model (the pre-sim onboard model)
ONBOARD_MODEL_PATH = MODEL_DIR / "integrity.json"
HOLDOUT_FAMILY = "F5_lighting"  # DEV family held out of every fit (glare / dim / dust: the hard case)
EMA_ALPHA_DOWN_ABLATION = (0.7, 0.55)  # HealthConfig.alpha_down values compared in the analysis
MIN_SIM_POSITIVES = 5  # VO-failure frames needed in the sim training set to fit a Platt recalibration
KITTI_TRAIN_SEQ, KITTI_TEST_SEQ = "07", "05"


DEGRADED_TAG = "deg"  # replay tag of the synthetic-degradation replays


def score_degraded(model: Any, data: list[SeedData]) -> dict[str, Any]:
    """Synthetic degradation segments on rendered frames: per kind, segments in which VO failed and
    how many of them drove the supervisor level to CAUTION or below; frame AUROC."""
    per_kind: dict[str, dict[str, int]] = {}
    tail = int(round(EVENT_TAIL_S / FRAME_DT_S))
    for d in data:
        _, q = monitor_q(model, d.X)
        lv = supervisor_levels(d.t, q)
        kinds = d.rep["degrade_kind"].astype(str)
        for kind in sorted(set(kinds.tolist()) - {"clean"}):
            for a, b in _segments(kinds == kind):
                w = slice(a, min(b + tail, d.t.size))
                e = per_kind.setdefault(kind, {"segments": 0, "segments_vo_failed": 0, "flagged_caution": 0,
                                               "flagged_degraded": 0, "flagged_caution_without_vo_failure": 0})
                e["segments"] += 1
                failed = bool(d.label[w].any())
                e["segments_vo_failed"] += int(failed)
                if failed:
                    e["flagged_caution"] += int(lv[w].max() >= LEVEL_CAUTION)
                    e["flagged_degraded"] += int(lv[w].max() >= LEVEL_DEGRADED)
                else:
                    e["flagged_caution_without_vo_failure"] += int(lv[w].max() >= LEVEL_CAUTION)
    X, y = sim_set(data)
    return {"frames": int(y.size), "vo_failure_frames": int(y.sum()), "auroc": auroc(model, X, y),
            "per_kind": per_kind,
            "segments_vo_failed": sum(v["segments_vo_failed"] for v in per_kind.values()),
            "segments_vo_failed_flagged_caution": sum(v["flagged_caution"] for v in per_kind.values())}


def _strip_events(score: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in score.items() if k != "events"}


def candidate_models(train: list[SeedData]) -> dict[str, Any]:
    """KITTI-only model, its Platt recalibration on sim DEV (train families), and a retrain on
    KITTI 07 + sim DEV (train families); KITTI 05 and the hold-out family are never used."""
    from metagross.autonomy.localization.health import LogisticIntegrityModel

    kitti = LogisticIntegrityModel.load(KITTI_MODEL_PATH if KITTI_MODEL_PATH.exists() else ONBOARD_MODEL_PATH)
    Xs, ys = sim_set(train)
    out = {"kitti_only": kitti}
    if int(ys.sum()) >= MIN_SIM_POSITIVES:
        out["kitti_platt_sim"] = platt_recalibrate(kitti, Xs, ys)
    else:
        LOG.warning("only %d VO-failure frames in the sim training set: no Platt fit", int(ys.sum()))
    out["retrained_kitti07_sim"] = retrain([kitti_set(KITTI_TRAIN_SEQ), (Xs, ys)])
    return out


def analyse(seeds: list[int], holdout_family: str = HOLDOUT_FAMILY, tag: str = "") -> dict[str, Any]:
    """Score the integrity model candidates on replayed drives; write ``results/integrity_sim.json``."""
    data = [SeedData.load(s, tag) for s in seeds if replay_path(s, tag).exists()]
    deg = [SeedData.load(s, DEGRADED_TAG) for s in seeds if replay_path(s, DEGRADED_TAG).exists()]
    train = [d for d in data if d.family != holdout_family]
    held = [d for d in data if d.family == holdout_family]
    kitti_sets = {seq: kitti_set(seq) for seq in (KITTI_TRAIN_SEQ, KITTI_TEST_SEQ)}
    Xh, yh = sim_set(held)
    Xa, ya = sim_set(data)
    models = candidate_models(train)
    out: dict[str, Any] = {
        "label": "Simulated",
        "what": "VO integrity monitor on rendered DEV stereo drives (ThreeRenderer, scripted oracle-path drives, "
                "real World physics / lighting events / dynamic obstacles); onboard Perception SGBM + Localizer "
                "replayed offline on the cached frames.",
        "seeds": [d.seed for d in data],
        "families": sorted({d.family for d in data}),
        "holdout_family": holdout_family,
        "kitti_train_seq": KITTI_TRAIN_SEQ, "kitti_holdout_seq": KITTI_TEST_SEQ,
        "vo_failure_label": f"no pose, or per-frame relative error > {FAIL_TRANS_M} m or > {FAIL_ROT_DEG} deg vs GT",
        "nominal_definition": f"no lighting event active, no triggered dynamic obstacle within {DYN_NEAR_M} m in the "
                              f"FOV, and > {EVENT_TAIL_S} s after either",
        "false_alarm_definition": f"onset (supervisor hysteresis replayed) on a nominal frame with no VO failure in "
                                  f"the preceding {EVENT_TAIL_S} s",
        "thresholds": {"caution_q_le": 0.7, "degraded_q_le": 0.4},
        "frames_total": int(sum(d.t.size for d in data)),
        "synthetic_degradation_seeds": [d.seed for d in deg],
        "synthetic_degradation_note": "metagross.eval.degrade segments (same generator as the KITTI study) applied "
                                      "to rendered DEV frames; never used for fitting",
        "models": {},
    }
    from metagross.autonomy.localization.health import HealthConfig

    for name, m in models.items():
        # EMA ablation: the onboard HealthConfig vs the previous alpha_down (0.7: one failed frame -> DEGRADED)
        out.setdefault("ema_ablation", {})[name] = {
            f"alpha_down_{a:g}": _strip_events(score_model(m, data, HealthConfig(alpha_down=a)))
            for a in EMA_ALPHA_DOWN_ABLATION}
        s_all = score_model(m, data)
        tr = [KITTI_TRAIN_SEQ] if name.startswith("retrained") else list(models["kitti_only"].meta.get("trained_on", []))
        held_seq = next((s for s in kitti_sets if s not in tr), None)
        out["models"][name] = {
            "kitti_trained_on": tr,
            "kitti_holdout_seq": held_seq,
            "auroc_kitti_holdout": auroc(m, *kitti_sets[held_seq]) if held_seq else None,
            **{f"auroc_kitti{seq}": auroc(m, *kitti_sets[seq]) for seq in kitti_sets},
            "auroc_kitti_note": "degraded KITTI (real images, synthetic degradations); only sequences the model "
                                "was not trained on are held-out",
            f"auroc_dev_{holdout_family}_holdout": auroc(m, Xh, yh),
            "auroc_dev_all": auroc(m, Xa, ya),
            "dev_all": _strip_events(s_all),
            f"dev_{holdout_family}_holdout": _strip_events(score_model(m, held)) if held else None,
            "events": s_all["events"],
            "coef": m.coef.tolist(), "intercept": m.intercept, "meta": m.meta,
        }
        if deg:
            out["models"][name]["dev_synthetic_degradations"] = score_degraded(m, deg)
        LOG.info("%s: %s", name, {k: out["models"][name][k] for k in ("auroc_kitti_holdout", "auroc_dev_all")})
    return out


def adopt_model(seeds: list[int], chosen: str, holdout_family: str, tag: str, res: dict[str, Any]) -> Path:
    """Write the chosen candidate (refitted exactly as scored) to ``models/integrity.json``; the KITTI-only
    model stays in ``models/integrity_kitti.json``."""
    data = [SeedData.load(s, tag) for s in seeds if replay_path(s, tag).exists()]
    model = candidate_models([d for d in data if d.family != holdout_family])[chosen]
    m = res["models"][chosen]
    meta = {
        "source": f"metagross.eval.integrity_sim candidate '{chosen}'",
        "trained_on": {"kitti": KITTI_TRAIN_SEQ, "dev_seeds": [d.seed for d in data if d.family != holdout_family]},
        "held_out": {"kitti": KITTI_TEST_SEQ, "dev_family": holdout_family},
        "label": f"VO failure: {res['vo_failure_label']}",
        "auroc_kitti_holdout": m["auroc_kitti_holdout"], "kitti_holdout_seq": m["kitti_holdout_seq"],
        f"auroc_dev_{holdout_family}_holdout": m[f"auroc_dev_{holdout_family}_holdout"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **(model.meta or {}),
    }
    ONBOARD_MODEL_PATH.write_text(json.dumps(model.to_json(meta), indent=2), encoding="utf-8")
    LOG.info("adopted %s -> %s", chosen, ONBOARD_MODEL_PATH)
    return ONBOARD_MODEL_PATH


def vo_sim_report(seeds: list[int], tag: str = "") -> dict[str, Any]:
    """VO-only drift on rendered DEV drives, with / without photometric normalisation."""
    data = [SeedData.load(s, tag) for s in seeds if replay_path(s, tag).exists()]
    rows = vo_drift_table(data)
    tot = {}
    for name in ("photometric", "no_photometric"):
        r = [x[name] for x in rows if name in x]
        path = sum(x["path_m"] for x in r)
        tot[name] = {"drives": len(r), "path_m": path,
                     "median_drift_pct": float(np.median([x["drift_pct"] for x in r])),
                     "mean_drift_pct": float(np.mean([x["drift_pct"] for x in r])),
                     "vo_failure_label_frames": int(sum(x["vo_failure_label_frames"] for x in r))}
    paired = json.loads(TIMING_JSON.read_text(encoding="utf-8")) if TIMING_JSON.exists() else None
    return {"label": "Simulated", "drives": rows, "summary": tot, "timing_replay": timing_summary(data),
            "localizer_timing_paired": paired,
            "method": "VO-only (no wheels/gyro) chained frame-to-frame relative poses vs GT camera poses; frames "
                      "without a VO pose bridged with the last VO motion; drift = end-point error / GT path length."}


def previous_vo_config() -> Any:
    """The VO configuration before this change set (no photometric normalisation, KLT 21 px / 30 it / 0.01)."""
    from metagross.autonomy.localization.vo import VOConfig

    return VOConfig(photometric_norm=False, klt_win=21, klt_iters=30, klt_eps=0.01)


def localizer_timing(seeds: list[int], max_frames: int = 150) -> dict[str, Any]:
    """Paired wall-clock of ``Localizer.update`` (stereo mode, wheels + gyro, disparity supplied) with the
    previous vs the current VO configuration, interleaved frame by frame on the same cached frames
    (the run order alternates), so machine load hits both alike. ms per frame."""
    from metagross.autonomy.localization.localizer import Localizer, LocalizerConfig
    from metagross.autonomy.perception.pipeline import Perception
    from metagross.contracts.messages import SensorFrame

    cv2.setNumThreads(2)
    calib = defaults.stereo_calibration()
    acc: dict[str, list[float]] = {"previous": [], "current": []}
    vo_acc: dict[str, list[float]] = {"previous": [], "current": []}
    for seed in seeds:
        drive = Drive.load(seed)
        per = Perception(calib, defaults.VEHICLE)
        cfg_prev = LocalizerConfig()
        cfg_prev.vo = previous_vo_config()
        locs = {"previous": Localizer(calib, defaults.VEHICLE, cfg_prev), "current": Localizer(calib, defaults.VEHICLE)}
        m = drive.meta
        for k, left, right in drive.frames():
            if k >= max_frames:
                break
            fr = SensorFrame(t=float(m["t"][k]), seq=k, left_rgb=left, right_gray=right,
                             wheel_angle_l_rad=float(m["wl"][k]), wheel_angle_r_rad=float(m["wr"][k]),
                             gyro_z_rps=float(m["gyro"][k]), sensor_mode="stereo")
            disp = per.compute_disparity(fr)
            order = ("previous", "current") if k % 2 else ("current", "previous")
            for name in order:
                out = locs[name].update(fr, disp)
                if k:
                    acc[name].append(out["timings_ms"]["total"])
                    vo_acc[name].append(out["timings_ms"].get("vo", math.nan))
    res: dict[str, Any] = {"seeds": seeds, "frames_per_config": len(acc["current"]), "opencv_threads": 2}
    for name in acc:
        a, v = np.asarray(acc[name]), np.asarray(vo_acc[name])
        res[name] = {"localizer_ms_median": float(np.median(a)), "localizer_ms_p95": float(np.percentile(a, 95)),
                     "localizer_ms_mean": float(a.mean()), "vo_ms_median": float(np.median(v)),
                     "vo_ms_p95": float(np.percentile(v, 95))}
    res["note"] = ("Shared 4-core laptop with 5 other agents running: absolute ms are inflated vs an idle machine "
                   "(integration log: 23 / 41 ms median / p95); the paired previous-vs-current ratio is the result.")
    return res


def integrity_claims(res: dict[str, Any], chosen: str) -> list[dict[str, Any]]:
    """Ledger rows (results/claims.csv via metagross.eval.claims) for the chosen onboard model."""
    m = res["models"][chosen]
    src = "results/integrity_sim.json"
    hold = res["holdout_family"]
    allm = m["dev_all"]
    rows = [
        {"id": "integrity_sim_nominal_minutes", "value": f"{allm['nominal_minutes']:.1f}", "unit": "min",
         "label": "Simulated", "source": f"{src}#models.{chosen}.dev_all.nominal_minutes",
         "note": f"nominal rendered DEV driving scored ({len(res['seeds'])} drives, DEV seeds only)"},
        {"id": "integrity_sim_degraded_per_5min", "value": f"{allm['degraded_onsets_per_5min']:.2f}", "unit": "/5 min",
         "label": "Simulated", "source": f"{src}#models.{chosen}.dev_all.degraded_onsets_per_5min",
         "note": f"DEGRADED onsets (q<=0.4, supervisor hysteresis) in nominal DEV driving, model '{chosen}'"},
        {"id": "integrity_sim_degraded_false_per_5min", "value": f"{allm['degraded_false_alarms_per_5min']:.2f}",
         "unit": "/5 min", "label": "Simulated",
         "source": f"{src}#models.{chosen}.dev_all.degraded_false_alarms_per_5min",
         "note": "DEGRADED onsets without any VO failure in the preceding 1 s (false alarms)"},
        {"id": "integrity_sim_caution_false_per_min", "value": f"{allm['caution_false_alarms_per_min']:.2f}",
         "unit": "/min", "label": "Simulated", "source": f"{src}#models.{chosen}.dev_all.caution_false_alarms_per_min",
         "note": "CAUTION onsets (q<=0.7) without VO failure in nominal DEV driving"},
        {"id": "integrity_sim_events_vo_fail_flagged",
         "value": f"{allm['events_with_vo_failure_flagged_caution']}/{allm['events_with_vo_failure']}", "unit": "",
         "label": "Simulated", "source": f"{src}#models.{chosen}.dev_all",
         "note": "lighting / sudden-obstacle events in which VO actually failed that drove q to CAUTION or below"},
    ]
    if m.get("auroc_dev_all") is not None:
        rows.append({"id": "integrity_sim_auroc_dev", "value": f"{m['auroc_dev_all']:.3f}", "unit": "",
                     "label": "Simulated", "source": f"{src}#models.{chosen}.auroc_dev_all",
                     "note": f"per-frame AUROC of p_fail vs GT VO failure, all rendered DEV frames "
                             f"({allm['vo_failure_frames']} failure frames, {allm['silent_failure_frames']} silent; "
                             f"silent-only AUROC {allm['auroc_silent_only'] if allm['auroc_silent_only'] is None else round(allm['auroc_silent_only'], 3)})"})
    if m.get(f"auroc_dev_{hold}_holdout") is not None:
        rows.append({"id": "integrity_sim_auroc_dev_holdout", "value": f"{m[f'auroc_dev_{hold}_holdout']:.3f}",
                     "unit": "", "label": "Simulated", "source": f"{src}#models.{chosen}.auroc_dev_{hold}_holdout",
                     "note": f"held-out DEV family {hold} (never used for fitting)"})
    if m.get("auroc_kitti_holdout") is not None:
        seq = m["kitti_holdout_seq"]
        rows.append({"id": "integrity_onboard_auroc_kitti_holdout", "value": f"{m['auroc_kitti_holdout']:.3f}",
                     "unit": "", "label": "Tested", "source": f"{src}#models.{chosen}.auroc_kitti_holdout",
                     "note": f"onboard model on held-out degraded KITTI {seq} (real images, synthetic degradations; "
                             f"see results/integrity.json for its silent-failure count)"})
    deg = m.get("dev_synthetic_degradations")
    if deg and deg.get("auroc") is not None:
        rows.append({"id": "integrity_sim_auroc_synthetic_degradations", "value": f"{deg['auroc']:.3f}", "unit": "",
                     "label": "Simulated", "source": f"{src}#models.{chosen}.dev_synthetic_degradations.auroc",
                     "note": "rendered DEV frames with synthetic degradation segments (flare, low light, blur, ...)"})
        rows.append({"id": "integrity_sim_synth_segments_flagged",
                     "value": f"{deg['segments_vo_failed_flagged_caution']}/{deg['segments_vo_failed']}", "unit": "",
                     "label": "Simulated", "source": f"{src}#models.{chosen}.dev_synthetic_degradations",
                     "note": "synthetic degradation segments with a VO failure that drove q to CAUTION or below"})
    return rows


def vo_claims(rep: dict[str, Any]) -> list[dict[str, Any]]:
    """Ledger rows for results/vo_sim_dev.json."""
    s = rep["summary"]["photometric"]
    rows = [{"id": "vo_sim_dev_median_drift_pct", "value": f"{s['median_drift_pct']:.2f}", "unit": "%",
             "label": "Simulated", "source": "results/vo_sim_dev.json#summary.photometric.median_drift_pct",
             "note": f"VO-only end-point drift, median over {s['drives']} rendered DEV drives ({s['path_m']:.0f} m)"}]
    p = rep.get("localizer_timing_paired")
    if p:
        rows.append({"id": "localizer_p95_ms_sim_paired",
                     "value": f"{p['previous']['localizer_ms_p95']:.1f} -> {p['current']['localizer_ms_p95']:.1f}",
                     "unit": "ms", "label": "Simulated",
                     "source": "results/vo_sim_dev.json#localizer_timing_paired",
                     "note": "stereo-mode Localizer.update p95, previous vs current VO config, paired on the same "
                             "rendered DEV frames on a loaded shared laptop (ratio is the result)"})
    return rows


HEALTH_TIMING_REPS = 200  # repetitions of the image-feature micro-benchmark


def health_feature_timing(seed: int = 102, frame: int = 50, reps: int = HEALTH_TIMING_REPS) -> dict[str, float]:
    """Integrity image features on one rendered frame: previous numpy channel-min path vs the current
    cv2.min path (identical values, checked here), mean ms per call over ``reps`` calls."""
    from metagross.autonomy.localization import health as H

    cv2.setNumThreads(2)
    d = CACHE_ROOT / str(seed)
    rgb = cv2.cvtColor(cv2.imread(str(d / f"L_{frame:04d}.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def previous() -> dict[str, float]:  # the pre-change implementation of the RGB dark channel
        small = cv2.resize(gray, (gray.shape[1] // H.IMG_DOWNSAMPLE, gray.shape[0] // H.IMG_DOWNSAMPLE),
                           interpolation=cv2.INTER_AREA)
        lap = cv2.Laplacian(small, cv2.CV_32F)
        src = cv2.resize(rgb, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA).min(axis=2)
        dark = cv2.erode(src, cv2.getStructuringElement(cv2.MORPH_RECT, (H.DARK_CHANNEL_PATCH, H.DARK_CHANNEL_PATCH)))
        return {
            "img_lapvar": float(lap.var()),
            "img_sat_frac": float(np.count_nonzero(small > H.SAT_LEVEL)) / small.size,
            "img_dark_frac": float(np.count_nonzero(small < H.DARK_LEVEL)) / small.size,
            "img_rms_contrast": float(small.std()) / 255.0,
            "img_dark_channel": float(dark.mean()) / 255.0,
        }

    def timed(f: Any) -> float:
        f()
        t0 = time.perf_counter()
        for _ in range(reps):
            f()
        return (time.perf_counter() - t0) / reps * 1e3

    cur = H.image_features(gray, rgb)
    ref = previous()
    same = all(abs(cur[k] - ref[k]) < 1e-12 for k in ref)
    return {"seed": seed, "frame": frame, "reps": reps, "previous_ms": timed(previous),
            "current_image_features_ms": timed(lambda: H.image_features(gray, rgb)), "values_identical": same}


# ============================================================================ KITTI regression check
KITTI_CHECK_JSON = RESULTS / "kitti_vo_07_check.json"


def kitti_vo_check(seq_id: str = "07", configs: Optional[dict[str, Any]] = None, threads: int = 2,
                   max_frames: Optional[int] = None, out_path: Path = KITTI_CHECK_JSON) -> dict[str, Any]:
    """Re-run KITTI stereo VO with several configurations sharing one SGBM pass per frame.

    Each configuration integrates its own trajectory exactly like
    :func:`metagross.eval.kitti_vo.run_vo` (failed frames bridged with the last motion) and is
    scored with the KITTI protocol (:func:`metagross.eval.traj_metrics.evaluate`). VO times
    exclude SGBM (it is shared; onboard it is perception's cost) and are measured interleaved
    frame by frame, so machine load affects all configurations alike. Writes ``out_path``
    only (the canonical ``results/kitti_vo_<seq>.json`` files are not touched).
    """
    import dataclasses
    import platform

    from metagross.autonomy.localization.vo import StereoVO, VOConfig, compute_disparity, make_sgbm
    from metagross.eval.kitti_vo import KittiSequence
    from metagross.eval.traj_metrics import evaluate

    cv2.setNumThreads(threads)
    seq = KittiSequence(seq_id)
    if not seq.available():
        raise FileNotFoundError(f"KITTI {seq_id} not downloaded")
    K, baseline = seq.calib()
    if configs is None:
        configs = {"baseline_no_photometric": dataclasses.replace(VOConfig.kitti(), photometric_norm=False),
                   "current": VOConfig.kitti()}
    vos = {name: StereoVO(K, baseline, cfg) for name, cfg in configs.items()}
    frames = seq.frames()[: max_frames or None]
    times = seq.times()
    n = len(frames)
    sgbm = make_sgbm(VOConfig.kitti().num_disparities, VOConfig.kitti().sgbm_block)
    poses = {name: np.tile(np.eye(4), (n, 1, 1)) for name in vos}
    last = {name: np.eye(4) for name in vos}
    ms = {name: np.zeros(n) for name in vos}
    fails = {name: 0 for name in vos}
    inl = {name: [] for name in vos}
    order = list(vos)
    t_wall = time.perf_counter()
    for k, (lp, rp) in enumerate(frames):
        left = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
        disp = compute_disparity(sgbm, left, right)
        t = None if times is None else float(times[k])
        order = order[1:] + order[:1]  # rotate the run order: no configuration always runs first
        for name in order:
            t0 = time.perf_counter()
            res = vos[name].process(left, None, disp, t)
            ms[name][k] = (time.perf_counter() - t0) * 1e3
            if k == 0:
                continue
            if res.ok:
                last[name] = res.T_prev_cur
                inl[name].append(res.stats["inliers"])
            else:
                fails[name] += 1
            poses[name][k] = poses[name][k - 1] @ last[name]
        if k and k % 200 == 0:
            LOG.info("kitti check %s: frame %d/%d (%.0f s)", seq_id, k, n, time.perf_counter() - t_wall)
    gt = seq.gt()[:n]
    out: dict[str, Any] = {
        "sequence": seq_id, "label": "Tested", "frames": n, "full_sequence": n == len(seq.frames()),
        "mode": "stereo VO, camera only, frame-to-frame; SGBM shared by all configurations (not timed)",
        "opencv_threads": threads, "cpu": platform.processor() or platform.machine(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "configs": {},
        "note": "Shared 4-core laptop (5 other agents running): VO-only ms are paired per frame (same load for "
                "all configurations). t_err/r_err: KITTI protocol, 100..800 m segments. Canonical "
                "results/kitti_vo_07.json (t_err 2.048 %) is not modified by this check.",
    }
    for name, cfg in configs.items():
        m = evaluate(gt, poses[name])
        a = ms[name][1:]
        out["configs"][name] = {
            **{k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in m.items()},
            "vo_failures": fails[name], "inliers_median": float(np.median(inl[name])) if inl[name] else 0.0,
            "vo_ms_median_excl_sgbm": float(np.median(a)), "vo_ms_p95_excl_sgbm": float(np.percentile(a, 95)),
            "vo_ms_mean_excl_sgbm": float(a.mean()),
            "vo_config": {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)},
        }
    out["claims"] = kitti_check_claims(out)
    out["current_defaults_equal"] = KITTI_CHECK_CURRENT
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    LOG.info("kitti check: %s", {k: (v["t_err_pct"], v["vo_ms_p95_excl_sgbm"]) for k, v in out["configs"].items()})
    return out


KITTI_CHECK_PREVIOUS = "baseline_no_photometric"  # config that reproduces results/kitti_vo_07.json
KITTI_CHECK_CURRENT = "photometric_w15_it10"  # config equal to the current VOConfig.kitti() defaults


def kitti_check_claims(res: dict[str, Any]) -> list[dict[str, Any]]:
    """Ledger rows of the KITTI regression check (previous vs current VO defaults)."""
    c = res["configs"]
    if KITTI_CHECK_PREVIOUS not in c or KITTI_CHECK_CURRENT not in c or c[KITTI_CHECK_CURRENT]["t_err_pct"] is None:
        return []
    prev, cur = c[KITTI_CHECK_PREVIOUS], c[KITTI_CHECK_CURRENT]
    src = "results/kitti_vo_07_check.json#configs"
    return [
        {"id": "kitti07_t_err_pct_current_vo", "value": f"{cur['t_err_pct']:.3f}", "unit": "%", "label": "Tested",
         "source": f"{src}.{KITTI_CHECK_CURRENT}.t_err_pct",
         "note": f"KITTI 07 with photometric normalisation + KLT 15 px/10 it (previous config "
                 f"{prev['t_err_pct']:.3f} % in the same run)"},
        {"id": "kitti07_vo_p95_ms_prev_vs_current",
         "value": f"{prev['vo_ms_p95_excl_sgbm']:.0f} -> {cur['vo_ms_p95_excl_sgbm']:.0f}", "unit": "ms",
         "label": "Tested", "source": f"{src}.*.vo_ms_p95_excl_sgbm",
         "note": "full-res KITTI VO excl. SGBM, paired per frame on a loaded shared laptop (ratio is the result)"},
    ]


# ============================================================================ CLI
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--seeds", type=int, nargs="+", required=True)
    r.add_argument("--max-s", type=float, default=MAX_DRIVE_S)
    r.add_argument("--force", action="store_true")
    kc = sub.add_parser("kitti-check")
    kc.add_argument("--seq", default="07")
    kc.add_argument("--max-frames", type=int, default=None)
    p = sub.add_parser("replay")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--no-ablations", action="store_true")
    p.add_argument("--degrade", action="store_true", help="add synthetic degradation segments (use with --tag)")
    p.add_argument("--skip-newer-than", type=float, default=None,
                   help="skip seeds whose replay file is newer than this UNIX time (parallel workers)")
    tm = sub.add_parser("timing")
    tm.add_argument("--seeds", type=int, nargs="+", required=True)
    tm.add_argument("--max-frames", type=int, default=150)
    tm.add_argument("--health-only", action="store_true", help="only add the image-feature micro-benchmark")
    a = sub.add_parser("analyse")
    a.add_argument("--seeds", type=int, nargs="+", required=True)
    a.add_argument("--tag", default="")
    a.add_argument("--holdout-family", default=HOLDOUT_FAMILY)
    a.add_argument("--chosen", default="", help="candidate reported as the onboard model (claims)")
    a.add_argument("--adopt", action="store_true", help="write the chosen candidate to models/integrity.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    if args.cmd == "render":
        render_many(args.seeds, args.max_s, skip_existing=not args.force)
    elif args.cmd == "replay":
        for s in args.seeds:
            rp = replay_path(s, args.tag)
            if args.skip_newer_than is not None and rp.exists() and rp.stat().st_mtime > args.skip_newer_than:
                LOG.info("seed %d already replayed, skipping", s)
                continue
            replay_seed(s, args.tag, ablations=not args.no_ablations, degrade=args.degrade)
    elif args.cmd == "analyse":
        res = analyse(args.seeds, args.holdout_family, args.tag)
        if args.chosen:
            res["onboard_model"] = args.chosen
            res["claims"] = integrity_claims(res, args.chosen)
            if args.adopt:
                adopt_model(args.seeds, args.chosen, args.holdout_family, args.tag, res)
        INTEGRITY_SIM_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
        vrep = vo_sim_report(args.seeds, args.tag)
        vrep["claims"] = vo_claims(vrep)
        VO_SIM_JSON.write_text(json.dumps(vrep, indent=2), encoding="utf-8")
    elif args.cmd == "timing":
        tres = localizer_timing(args.seeds, args.max_frames) if not args.health_only else (
            json.loads(TIMING_JSON.read_text(encoding="utf-8")) if TIMING_JSON.exists() else {})
        tres["health_image_features"] = health_feature_timing(args.seeds[0])
        TIMING_JSON.write_text(json.dumps(tres, indent=2), encoding="utf-8")
    elif args.cmd == "kitti-check":
        kitti_vo_check(args.seq, kitti_check_configs(), max_frames=args.max_frames)
    return 0


def kitti_check_configs() -> dict[str, Any]:
    """Configurations compared by ``kitti-check``: the pre-change VO (reproduces results/kitti_vo_07.json),
    the current defaults, and the KLT speed variants evaluated on rendered DEV drives."""
    import dataclasses

    from metagross.autonomy.localization.vo import VOConfig

    k = VOConfig.kitti()
    rep = dataclasses.replace
    return {
        "baseline_no_photometric": rep(k, photometric_norm=False, klt_win=21, klt_iters=30, klt_eps=0.01),
        "photometric_w21_it30": rep(k, klt_win=21, klt_iters=30, klt_eps=0.01),
        "photometric_w15_it10": rep(k, klt_win=15, klt_iters=10, klt_eps=0.03),
        "photometric_w15_it30": rep(k, klt_win=15, klt_iters=30, klt_eps=0.01),
        "photometric_w21_it10": rep(k, klt_win=21, klt_iters=10, klt_eps=0.03),
    }


if __name__ == "__main__":
    raise SystemExit(main())
