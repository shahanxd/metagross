"""Chase-camera replay for the demo video: re-render third-person frames from a run's GT log.

This is **offline visualisation tooling on the simulator side** (it reads the scenario file and
the GT log); the autonomy stack never imports it. It adds no new rendering: every frame comes from
:meth:`ThreeRenderer.render_chase` on the same Three.js scene the stereo cameras saw, so the
video's "simulator view" is exactly the simulated world.

* :func:`resolve_scenario` finds the scenario a run was made on (``result.json`` -> ``seed``,
  ``split``, ``sha256`` -> ``<scenario_root>/<split>/<seed>.json``) and verifies its sha256, so a
  replay can never silently render a different world.
* :class:`ThreeChaseFactory` owns **one** renderer (one headed Chrome) for a whole video and
  reloads the scene only when the next run uses a different scenario. ``for_run(run_dir)``
  returns a ``(state, w, h) -> rgb`` callable (``video.replay.ChaseRenderer``).

State dicts follow ``RendererProto`` (WORLD frame, REP-103 pose ``[x, y, z, roll, pitch, yaw]``
in metres / radians, ``t`` in seconds of sim time). Lighting events are evaluated by the web
renderer from ``t``; dynamic actors come from ``state['dynamic']`` (the replay fills it from the
GT ``dyn_xy`` log).

Chase camera model (:class:`ChaseCamParams`, :func:`plan_chase_camera`)
------------------------------------------------------------------------
The camera is planned once per run over the whole GT log, so a frame depends only on the run and
``t`` (deterministic, cacheable, order-independent):

1. **Smoothed heading.** The camera follows a critically damped (second-order, no overshoot)
   filter of the unwrapped GT yaw, so turning on the spot or steering jitter does not swing it.
2. **Occluder push-in.** The sightline from the rover (``target_up_m`` above its base) to the
   nominal camera position is tested against the scenario's objects (tree trunks and canopies,
   bushes, rocks, logs as inflated cylinders / spheres). If one cuts it, the camera slides in
   along that sightline to just before the occluder (never closer than ``min_back_frac``). The
   push-in starts ``anticipate_s`` early, attacks fast and releases slowly (asymmetric critically
   damped filter), and the final value is never farther out than the raw clear fraction.

The result is passed to the web renderer as ``state['chase'] = {back_m, up_m, yaw_smooth,
lookahead_m}`` (``renderChase`` in ``web/main.js``; absent = the legacy fixed 6 m / 3 m offset).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from metagross.contracts.interfaces import RendererProto
from metagross.sim.scenario import load_scenario

log = logging.getLogger(__name__)

DEFAULT_SCENARIO_ROOT = Path(__file__).resolve().parents[3] / "data" / "scenarios"  # repo/data/scenarios
RUN_SCENARIO_FILE = "scenario.json"  # optional copy of the scenario next to result.json (takes precedence)
CHASE_CAM_VERSION = 2  # bump when the camera model changes: it is part of the chase-frame cache key
_EPS = 1e-9
# Tree geometry, matched to web/objects.js buildStaticObjects: canopy centre at H - cr, lobes reach
# about 1.2 cr sideways, down to about H - 1.8 cr and up to about H + 0.2 cr (a vertical cylinder,
# because a sphere would reach far below the lowest lobes); trunk height max(H - 1.2 cr, 0.4 H).
TREE_CANOPY_RADIUS_K = 1.2
TREE_CANOPY_BOTTOM_K = 1.8  # canopy bottom at H - 1.8 cr
TREE_CANOPY_TOP_K = 0.2  # canopy top at H + 0.2 cr
TREE_TRUNK_RADIUS_K = 1.2  # trunk flare at the base (objects.js: tr * 1.2)
TREE_TRUNK_H_K = 1.2
TREE_TRUNK_MIN_H_FRAC = 0.4
LOG_SPHERE_SPACING_K = 1.0  # logs as a chain of spheres spaced one radius apart


@dataclass(frozen=True)
class ChaseCamParams:
    """Third-person camera parameters. Distances in metres, WORLD frame; angular rates in rad/s."""

    back_m: float = 6.0  # nominal horizontal distance behind the rover (legacy CHASE_BACK_M)
    up_m: float = 3.0  # nominal camera height above the rover base (legacy CHASE_UP_M)
    lookahead_m: float = 2.5  # look-at point ahead of the rover along the smoothed heading
    yaw_omega_rad_s: float = 3.0  # critically damped heading filter; 90 % of a step after ~1.3 s
    target_up_m: float = 0.5  # rover sight point above its base: the camera must see it
    clearance_m: float = 0.35  # extra radius added to every occluder
    min_back_frac: float = 0.4  # never slide closer than this fraction of the nominal offset (2.4 m at 6 m)
    anticipate_s: float = 0.6  # start pushing in this long before an occluder cuts the sightline
    attack_omega_rad_s: float = 8.0  # push-in filter rate (fast)
    release_omega_rad_s: float = 1.5  # return-to-nominal filter rate (slow, no pumping)

    def tag(self) -> str:
        """Short digest of the parameters (part of the chase-frame cache key)."""
        blob = json.dumps({"v": CHASE_CAM_VERSION, **asdict(self)}, sort_keys=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def critically_damped(t: np.ndarray, u: np.ndarray, omega: float | np.ndarray,
                      omega_down: Optional[float] = None) -> np.ndarray:
    """Critically damped second-order filter of samples ``u`` at times ``t`` (s), exact per step.

    ``y'' = omega^2 (u - y) - 2 omega y'`` with ``u`` held constant over each step; starts at rest on
    ``u[0]``. With ``omega_down`` the rate is ``omega_down`` while the input is below the output
    (asymmetric attack / release). Units of ``y`` are those of ``u``; ``omega`` in rad/s.
    """
    t = np.asarray(t, np.float64)
    u = np.asarray(u, np.float64)
    y = np.empty_like(u)
    if u.size == 0:
        return y
    yk, vk = float(u[0]), 0.0
    y[0] = yk
    for k in range(1, u.size):
        dt = max(float(t[k] - t[k - 1]), 0.0)
        target = float(u[k])
        w = float(omega_down) if (omega_down is not None and target < yk) else float(omega)
        e0 = yk - target
        b = vk + w * e0
        decay = np.exp(-w * dt)
        yk = target + (e0 + b * dt) * decay  # e(t) = (e0 + (v0 + w e0) t) exp(-w t)
        vk = (vk - w * b * dt) * decay
        y[k] = yk
    return y


def build_occluders(scenario: dict, height_at: Optional[Callable[[float, float], float]] = None) -> tuple[np.ndarray, np.ndarray]:
    """Scenario objects -> (cylinders (K, 5) ``[x, y, r, z0, z1]``, spheres (L, 4) ``[x, y, z, r]``), WORLD m.

    Vertical cylinders: tree trunks and canopies, bushes. Spheres: rocks, logs (sphere chains).
    ``height_at(x, y)`` is the terrain height (m); without it the ground is taken as z = 0.
    """
    hz = height_at or (lambda x, y: 0.0)
    cyl: list[list[float]] = []
    sph: list[list[float]] = []
    for o in scenario.get("objects", []) or []:
        typ = o.get("type")
        if typ == "tree":
            x, y = (float(v) for v in o["xy"])
            zg, h, cr, tr = float(hz(x, y)), float(o["height"]), float(o["canopy_r"]), float(o["trunk_r"])
            trunk_h = max(h - TREE_TRUNK_H_K * cr, TREE_TRUNK_MIN_H_FRAC * h)
            cyl.append([x, y, TREE_TRUNK_RADIUS_K * tr, zg, zg + trunk_h])
            cyl.append([x, y, TREE_CANOPY_RADIUS_K * cr, zg + h - TREE_CANOPY_BOTTOM_K * cr, zg + h + TREE_CANOPY_TOP_K * cr])
        elif typ == "bush":
            x, y = (float(v) for v in o["xy"])
            zg = float(hz(x, y))
            cyl.append([x, y, float(o["radius"]), zg, zg + float(o["height"])])
        elif typ == "rock":
            x, y, z = (float(v) for v in o["xyz"])
            sph.append([x, y, z, float(o["radius"])])
        elif typ == "log":
            x, y = (float(v) for v in o["xy"])
            r, length, yaw = float(o["radius"]), float(o["length"]), float(o.get("yaw", 0.0))
            zc = float(hz(x, y)) + r
            n = max(2, int(np.ceil(length / (LOG_SPHERE_SPACING_K * r))) + 1)
            for s in np.linspace(-length / 2, length / 2, n):
                sph.append([x + s * np.cos(yaw), y + s * np.sin(yaw), zc, r])
    return np.asarray(cyl, np.float64).reshape(-1, 5), np.asarray(sph, np.float64).reshape(-1, 4)


def _interval_quadratic(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[s_lo, s_hi] where a s^2 + b s + c <= 0 (a >= 0); empty (lo > hi) when there is none."""
    disc = b * b - 4.0 * a * c
    ok = disc >= 0.0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    safe_a = np.where(a > _EPS, a, 1.0)
    lo = np.where(a > _EPS, (-b - sq) / (2.0 * safe_a), np.where(c <= 0.0, -np.inf, np.inf))
    hi = np.where(a > _EPS, (-b + sq) / (2.0 * safe_a), np.where(c <= 0.0, np.inf, -np.inf))
    lo = np.where(ok | (a <= _EPS), lo, np.inf)
    hi = np.where(ok | (a <= _EPS), hi, -np.inf)
    return lo, hi


def sightline_clear_fraction(target: np.ndarray, eye: np.ndarray, cylinders: np.ndarray, spheres: np.ndarray,
                             clearance_m: float = 0.0, ignore_below: float | np.ndarray = 0.0) -> np.ndarray:
    """Fraction s in (0, 1] of each segment ``target -> eye`` (both (N, 3), WORLD m) that is free.

    Returns the smallest entry parameter into any inflated occluder (1.0 when the segment is clear).
    Occluders that already contain the target point, or that the segment enters before fraction
    ``ignore_below``, are ignored: moving the camera (never closer than that) cannot help.
    """
    target = np.asarray(target, np.float64).reshape(-1, 3)
    d = np.asarray(eye, np.float64).reshape(-1, 3) - target
    s_hit = np.ones(len(target))
    ib = np.broadcast_to(np.asarray(ignore_below, np.float64), (len(target),))[:, None]  # scalar or per segment
    if len(cylinders):
        cx, cy, r, z0, z1 = (cylinders[:, i][None, :] for i in range(5))
        ax, ay, az = (target[:, i][:, None] for i in range(3))
        dx, dy, dz = (d[:, i][:, None] for i in range(3))
        r = r + clearance_m
        a = dx * dx + dy * dy
        b = 2.0 * (dx * (ax - cx) + dy * (ay - cy))
        c = (ax - cx) ** 2 + (ay - cy) ** 2 - r * r
        lo, hi = _interval_quadratic(a, b, c)
        zlo, zhi = z0 - clearance_m, z1 + clearance_m
        with np.errstate(divide="ignore", invalid="ignore"):
            sa = np.where(np.abs(dz) > _EPS, (zlo - az) / dz, np.where((az >= zlo) & (az <= zhi), -np.inf, np.inf))
            sb = np.where(np.abs(dz) > _EPS, (zhi - az) / dz, np.where((az >= zlo) & (az <= zhi), np.inf, -np.inf))
        vlo, vhi = np.minimum(sa, sb), np.maximum(sa, sb)
        s0 = np.maximum(np.maximum(lo, vlo), 0.0)
        s1 = np.minimum(np.minimum(hi, vhi), 1.0)
        hit = (s0 <= s1) & (s0 > ib)
        s_hit = np.minimum(s_hit, np.where(hit, s0, 1.0).min(axis=1))
    if len(spheres):
        cx, cy, cz, r = (spheres[:, i][None, :] for i in range(4))
        ax, ay, az = (target[:, i][:, None] for i in range(3))
        dx, dy, dz = (d[:, i][:, None] for i in range(3))
        r = r + clearance_m
        a = dx * dx + dy * dy + dz * dz
        b = 2.0 * (dx * (ax - cx) + dy * (ay - cy) + dz * (az - cz))
        c = (ax - cx) ** 2 + (ay - cy) ** 2 + (az - cz) ** 2 - r * r
        lo, hi = _interval_quadratic(a, b, c)
        s0, s1 = np.maximum(lo, 0.0), np.minimum(hi, 1.0)
        hit = (s0 <= s1) & (s0 > ib)
        s_hit = np.minimum(s_hit, np.where(hit, s0, 1.0).min(axis=1))
    return s_hit


def _future_min(t: np.ndarray, x: np.ndarray, window_s: float) -> np.ndarray:
    """min of x over [t_k, t_k + window_s] for every sample k (t ascending)."""
    out = np.empty_like(x)
    j_end = np.searchsorted(t, t + window_s, side="right")
    for k in range(len(x)):
        out[k] = x[k:max(j_end[k], k + 1)].min()
    return out


def plan_chase_camera(t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, yaw: np.ndarray,
                      cylinders: np.ndarray, spheres: np.ndarray,
                      params: ChaseCamParams = ChaseCamParams()) -> dict[str, np.ndarray]:
    """Per-GT-sample chase camera: ``yaw_smooth`` (rad, unwrapped), ``back_m``, ``up_m``, ``clear_raw``.

    Inputs are the GT log in the WORLD frame (s, m, rad). See the module docstring for the model.
    """
    t = np.asarray(t, np.float64)
    yaw_s = critically_damped(t, np.unwrap(np.asarray(yaw, np.float64)), params.yaw_omega_rad_s)
    base = np.column_stack([x, y, z]).astype(np.float64)
    target = base + np.array([0.0, 0.0, params.target_up_m])
    eye = base + np.column_stack([-np.cos(yaw_s) * params.back_m, -np.sin(yaw_s) * params.back_m,
                                  np.full(len(t), params.up_m)])
    clear = sightline_clear_fraction(target, eye, cylinders, spheres, params.clearance_m, ignore_below=params.min_back_frac)
    raw = np.maximum(clear, params.min_back_frac)  # an occluder closer than that is accepted (rover brushing past it)
    ant = _future_min(t, raw, params.anticipate_s)
    frac = critically_damped(t, ant, params.release_omega_rad_s, omega_down=params.attack_omega_rad_s)
    frac = np.clip(np.minimum(frac, raw), params.min_back_frac, 1.0)
    # the look-at point moves in with the camera so a close camera still frames the whole rover
    return {"yaw_smooth": yaw_s, "back_m": frac * params.back_m, "lookahead_m": frac * params.lookahead_m,
            "up_m": params.target_up_m + frac * (params.up_m - params.target_up_m), "clear_raw": clear}


class ScenarioNotFound(FileNotFoundError):
    """The scenario of a run could not be located or does not match the run's sha256."""


def resolve_scenario(run_dir: str | Path, scenario_root: str | Path = DEFAULT_SCENARIO_ROOT) -> dict:
    """Scenario dict a run was produced on (sha256-verified against ``result.json``).

    Search order: ``run_dir/scenario.json``, then ``<scenario_root>/<split>/<seed>.json``.
    Raises :class:`ScenarioNotFound` if neither exists or the digest differs.
    """
    run = Path(run_dir)
    res_path = run / "result.json"
    if not res_path.exists():
        raise ScenarioNotFound(f"{res_path} missing: cannot tell which scenario {run} used")
    res = json.loads(res_path.read_text(encoding="utf-8"))
    want = str(res.get("sha256", ""))
    candidates = [run / RUN_SCENARIO_FILE]
    if res.get("seed") is not None:
        candidates.append(Path(scenario_root) / str(res.get("split", "dev")) / f"{int(res['seed'])}.json")
    for c in candidates:
        if c.exists():
            scn = load_scenario(c, verify=True)
            if want and scn["sha256"] != want:
                raise ScenarioNotFound(f"{c}: sha256 {scn['sha256'][:12]} != run's {want[:12]}")
            return scn
    raise ScenarioNotFound(f"no scenario for {run} (looked in {[str(c) for c in candidates]})")


def default_three_renderer() -> RendererProto:
    """Three.js renderer at the default calibration (same as the runner's stereo mode)."""
    from metagross.sim.render.bridge import ThreeRenderer

    return ThreeRenderer()


class ThreeChaseFactory:
    """One shared renderer for all chase replays of a video (see module docstring).

    Parameters
    ----------
    scenario_root: directory holding ``dev/`` and ``eval/`` scenario JSON files.
    renderer_factory: ``() -> RendererProto`` (default: :func:`default_three_renderer`);
        created lazily on the first frame so building a timeline stays cheap.
    """

    def __init__(self, scenario_root: str | Path = DEFAULT_SCENARIO_ROOT,
                 renderer_factory: Callable[[], RendererProto] = default_three_renderer,
                 params: Optional[ChaseCamParams] = ChaseCamParams()) -> None:
        self.scenario_root = Path(scenario_root)
        self.renderer_factory = renderer_factory
        self.params = params  # None = legacy fixed-offset camera (no state['chase'])
        self._renderer: Optional[RendererProto] = None
        self._loaded_sha: Optional[str] = None
        self.n_frames = 0
        self.n_loads = 0

    def _ensure(self, scn: dict) -> RendererProto:
        if self._renderer is None:
            self._renderer = self.renderer_factory()
        if self._loaded_sha != scn["sha256"]:
            self._renderer.load_scenario(scn)
            self._loaded_sha = scn["sha256"]
            self.n_loads += 1
            log.info("chase replay: loaded scenario seed %s (%s)", scn.get("seed"), scn["sha256"][:12])
        return self._renderer

    def for_run(self, run_dir: str | Path) -> "ChaseReplay":
        """Chase renderer bound to the scenario of ``run_dir`` (camera planned from its GT log)."""
        return ChaseReplay(self, resolve_scenario(run_dir, self.scenario_root), run_dir=run_dir, params=self.params)

    def render(self, scn: dict, state: dict, width: int, height: int) -> np.ndarray:
        """(height, width, 3) uint8 RGB chase frame of ``scn`` at ``state``."""
        r = self._ensure(scn)
        img = r.render_chase(state, int(width), int(height))
        self.n_frames += 1
        return img

    def close(self) -> None:
        """Close the browser (idempotent)."""
        if self._renderer is not None:
            try:
                self._renderer.close()
            finally:
                self._renderer = None
                self._loaded_sha = None

    def __enter__(self) -> "ThreeChaseFactory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _terrain_height_fn(scenario: dict) -> Optional[Callable[[float, float], float]]:
    """Terrain height (m) of a full scenario, or None for stub scenarios without a terrain block."""
    if "terrain" not in scenario:
        return None
    from metagross.sim.terrain import Terrain

    terr = Terrain.from_scenario(scenario)
    return lambda x, y: float(terr.height_at(x, y))


class ChaseReplay:
    """``(state, w, h) -> rgb`` bound to one scenario; the factory keeps the browser.

    With ``run_dir`` and ``params`` the camera is planned over the run's GT log
    (:func:`plan_chase_camera`) and every state passed to the renderer gets a ``chase`` block
    interpolated at ``state['t']``; ``cache_tag`` then identifies that plan for the frame cache.
    """

    def __init__(self, factory: ThreeChaseFactory, scenario: dict, run_dir: Optional[str | Path] = None,
                 params: Optional[ChaseCamParams] = None) -> None:
        self.factory = factory
        self.scenario = scenario
        self.params = params
        self._plan: Optional[dict[str, np.ndarray]] = None
        self.cache_tag = ""
        gt_path = Path(run_dir) / "gt" / "states.npz" if run_dir is not None else None
        if params is not None and gt_path is not None and gt_path.exists():
            with np.load(gt_path) as z:
                gt = {k: np.asarray(z[k], np.float64) for k in ("t", "x", "y", "z", "yaw") if k in z.files}
            if {"t", "x", "y", "yaw"} <= gt.keys() and len(gt["t"]):
                cyl, sph = build_occluders(scenario, _terrain_height_fn(scenario))
                self._plan = plan_chase_camera(gt["t"], gt["x"], gt["y"], gt.get("z", np.zeros_like(gt["t"])), gt["yaw"],
                                               cyl, sph, params)
                self._plan["t"] = gt["t"]
                digest = hashlib.sha1(gt_path.read_bytes()).hexdigest()[:10]
                self.cache_tag = f"cam{CHASE_CAM_VERSION}-{params.tag()}-{digest}"
                pushed = float(np.mean(self._plan["back_m"] < params.back_m - 1e-3))
                log.info("chase camera for %s: %d occluders, pushed in on %.0f %% of the log", Path(run_dir).name,
                         len(cyl) + len(sph), 100.0 * pushed)

    @property
    def scenario_sha(self) -> str:
        return str(self.scenario["sha256"])

    def chase_block(self, t: float) -> Optional[dict]:
        """``state['chase']`` at run time ``t`` (s), or None without a plan."""
        if self._plan is None or self.params is None:
            return None
        p, tt = self._plan, float(t)
        return {k: float(np.interp(tt, p["t"], p[k])) for k in ("back_m", "up_m", "yaw_smooth", "lookahead_m")}

    def __call__(self, state: dict, width: int, height: int) -> np.ndarray:
        block = self.chase_block(float(state.get("t", 0.0))) if "chase" not in state else None
        if block is not None:
            state = {**state, "chase": block}
        return self.factory.render(self.scenario, state, width, height)
