"""Model Predictive Path Integral control (Williams et al., ICRA 2016), vectorised numpy.

Model: unicycle in the A-frame, state (x, y, yaw), control u = (v [m/s], omega [rad/s]),
integrated with dt using the mid-point heading. Skid-steer limits applied to every
sample: ``-v_reverse_max <= v <= v_cap`` (reverse only as a slow escape, default off;
the live node allows ``ESCAPE_REVERSE_MPS``, penalised by ``w_reverse``, with the
certification probes placed behind the rear bumper), ``|omega| <= max_yaw_rate``,
per-side wheel speed ``|v| + |omega| * chi * B / 2 <= max_wheel_rad_s * r`` and an
acceleration envelope around the current speed.

Rollout cost S_k (all terms summed over the horizon, weights in :class:`MppiParams`):

* map cost at 3 footprint circles along the body axis (costmap already inflated by the
  circle radius + margin),
* ``w_lethal`` per circle-step inside a lethal cell,
* CERTIFICATION: ``w_cert`` per probe point that is not certified, where probes lie
  ahead of each rollout state at fractions of its stopping distance
  ``d_stop(v_t) = v_t^2/(2a) + v_t T_r + B`` (the MPPI half of the speed governor),
* smoothness on dv, domega; speed tracking toward ``v_ref``;
* terminal cost-to-go from the global planner;
* the MPPI control-cost term ``gamma * lambda * u^T Sigma^-1 eps``.

Weights ``w_k = exp(-(S_k - min S) / lambda)``; nominal update ``U += sum_k w_k eps_k``;
Savitzky-Golay smoothing; warm start by shifting the nominal by the elapsed time.

Exploration noise is time-correlated: i.i.d. N(0, sigma) values at knots every
``noise_knot_steps`` steps, linearly interpolated. With i.i.d. per-step noise the
horizon-average speed of a sample varies by only sigma / sqrt(T), so MPPI converges to
the cruise speed very slowly; correlated noise explores whole-horizon speed and
curvature changes (cf. the smooth / colored-noise MPPI variants).
Sample 0 is the (shifted) nominal and sample 1 is a full stop, so braking is always
in the candidate set.

Guided sampling (optional ``guide_path_xy``, the global planner's descent path): a
local sampler around a stale nominal cannot find a route that swings > ~90 deg (the
first DEV runs turned the wrong way in place for seconds after a replan). When a guide
path is given, a fraction ``guide_fraction`` of the samples is drawn around a
pure-pursuit rollout of that path (rotate in place while the heading error exceeds
``pursuit_rotate_rad``, else drive at ``v_ref`` scaled by cos(error)), and the exact
pursuit sequences at a few speed scales plus rotate-in-place left / right are added
noise-free. All samples are weighted by the same cost, so the guide only wins when it
is actually cheaper (Williams et al. 2018, "ancillary controller" sampling).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.signal import savgol_filter

from metagross.autonomy.planning.costmap import PlanningMaps
from metagross.config.defaults import BRAKE_DECEL_MPS2, CHI_NOMINAL, GOVERNOR_MARGIN_M, VEHICLE

CtgFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
ESCAPE_REVERSE_MPS = 0.3  # reverse speed the live stack allows (back off from a ditch lip / inflation it stopped in)


@dataclass(frozen=True, slots=True)
class MppiParams:
    n_samples: int = 512  # K
    horizon: int = 30  # T
    dt: float = 0.1  # [s]
    sigma_v: float = 0.3  # [m/s]
    sigma_w: float = 0.6  # [rad/s]
    lam: float = 3.0  # temperature (cost units); tuned on the toy loop in tests/test_plan_mppi.py
    gamma: float = 0.0  # control-cost fraction; 0: the i.i.d.-noise term over-penalises speed under correlated noise
    w_map: float = 4.0
    w_lethal: float = 1e6
    w_cert: float = 60.0
    w_dv: float = 1.0
    w_dw: float = 0.2
    w_speed: float = 1.0
    w_ctg: float = 8.0
    circle_offsets_m: tuple[float, ...] = (-0.27, 0.0, 0.27)  # along body x; radius ~0.33 m
    cert_fracs: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)  # probe fractions of d_stop
    front_m: float = VEHICLE.length_m / 2.0
    savgol_window: int = 9
    savgol_order: int = 3
    noise_knot_steps: int = 5  # correlated-noise knot spacing [steps]
    n_vis: int = 16  # rollouts returned for visualisation
    guide_fraction: float = 0.25  # share of samples drawn around the pure-pursuit guide (when a guide path is given)
    pursuit_lookahead_m: float = 1.5  # pure-pursuit lookahead along the guide path [m]
    pursuit_rotate_rad: float = 0.8  # heading error above which the guide rotates in place [rad]
    pursuit_gain: float = 2.0  # yaw-rate gain on the heading error [1/s]
    guide_speed_scales: tuple[float, ...] = (1.0, 0.6)  # noise-free pursuit candidates at these fractions of v_ref
    rotate_rate_frac: float = 0.7  # noise-free rotate-in-place candidates at this fraction of max_yaw_rate
    v_reverse_max: float = 0.0  # [m/s] allowed reverse speed (0: forward only); the node enables ESCAPE_REVERSE_MPS
    w_reverse: float = 2.0  # extra cost per (m/s * s) of reversing, on top of the speed-tracking term
    max_yaw_rate: float = VEHICLE.max_yaw_rate_rps
    max_accel: float = VEHICLE.max_accel_mps2
    brake_decel: float = BRAKE_DECEL_MPS2
    track_width_m: float = VEHICLE.track_width_m
    v_wheel_max: float = VEHICLE.max_wheel_rad_s * VEHICLE.wheel_radius_m


@dataclass(slots=True)
class MppiResult:
    u0: tuple[float, float]  # first control (v, omega)
    U: np.ndarray  # (T, 2) smoothed nominal controls
    traj: np.ndarray  # (T, 3) nominal states x, y, yaw (A-frame)
    rollouts_xy: np.ndarray  # (n_vis, T, 2) A-frame
    s_min: float
    ess: float  # effective sample size 1/sum(w^2)
    best_lethal: bool  # even the best sample hits lethal (collision unavoidable in model)
    compute_ms: float
    cost_terms: dict[str, float] = field(default_factory=dict)  # of the weighted-mean rollout


def rollout(x0: tuple[float, float, float], V: np.ndarray, W: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate unicycle controls (â€¦, T) from x0 with mid-point heading. Returns X, Y, YAW (â€¦, T)."""
    yaw = x0[2] + np.cumsum(W * dt, axis=-1)
    yaw_mid = yaw - 0.5 * W * dt
    X = x0[0] + np.cumsum(V * np.cos(yaw_mid) * dt, axis=-1)
    Y = x0[1] + np.cumsum(V * np.sin(yaw_mid) * dt, axis=-1)
    return X, Y, yaw


GUIDE_RESAMPLE_M = 0.2  # guide path resampling step [m]


def _resample_polyline(path: np.ndarray, step_m: float) -> np.ndarray:
    """Arc-length-uniform resampling of an (N, 2) polyline (metres)."""
    seg = np.hypot(*np.diff(path, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-9:
        return path[:1].copy()
    s = np.arange(0.0, cum[-1] + 1e-9, step_m)
    return np.column_stack([np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])])


def pursuit_controls(pose: tuple[float, float, float], path_xy: np.ndarray, v_ref: float, horizon: int, dt: float,
                     lookahead_m: float, rotate_rad: float, gain: float, max_yaw_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed-loop pure-pursuit rollout along ``path_xy`` (A-frame, vehicle -> goal) from ``pose``.

    Per step: carrot = path point ``lookahead_m`` beyond the closest point; heading error e;
    omega = clip(gain * e, +-max_yaw_rate); v = 0 if |e| > rotate_rad else v_ref * cos(e).
    Returns (V, W), each ``(horizon,)`` [m/s, rad/s]; unconstrained by accel limits (the
    caller clips)."""
    pts = _resample_polyline(np.asarray(path_xy, np.float64), GUIDE_RESAMPLE_M)
    n = len(pts)
    la = max(1, int(round(lookahead_m / GUIDE_RESAMPLE_M)))
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    V = np.zeros(horizon)
    W = np.zeros(horizon)
    k0 = 0
    win = la * 4
    for i in range(horizon):
        hi = min(n, k0 + win)
        d2 = (pts[k0:hi, 0] - x) ** 2 + (pts[k0:hi, 1] - y) ** 2
        k0 = k0 + int(np.argmin(d2))
        cx, cy = pts[min(k0 + la, n - 1)]
        e = math.atan2(cy - y, cx - x) - yaw
        e = math.atan2(math.sin(e), math.cos(e))
        w = float(np.clip(gain * e, -max_yaw_rate, max_yaw_rate))
        v = 0.0 if abs(e) > rotate_rad else v_ref * math.cos(e)
        if math.hypot(cx - x, cy - y) < 0.5 * GUIDE_RESAMPLE_M:  # at the end of the path
            v, w = 0.0, 0.0
        V[i], W[i] = v, w
        yaw_mid = yaw + 0.5 * w * dt
        x += v * math.cos(yaw_mid) * dt
        y += v * math.sin(yaw_mid) * dt
        yaw += w * dt
    return V, W


class MppiPlanner:
    """Stateful MPPI controller (keeps the warm-started nominal control sequence)."""

    def __init__(self, params: MppiParams = MppiParams(), seed: int = 0) -> None:
        self.p = params
        self.rng = np.random.default_rng(seed)
        self.U = np.zeros((params.horizon, 2))
        self._t_last: Optional[float] = None
        self.chi = CHI_NOMINAL
        self._v_rev = 0.0  # reverse speed allowed in the current plan() call [m/s]
        T = params.horizon
        # Savitzky-Golay as a fixed (T, T) linear operator (mode='nearest'); avoids per-call lstsq
        win = min(params.savgol_window, T - (1 - T % 2))
        self._sg = savgol_filter(np.eye(T), win, params.savgol_order, axis=0, mode="nearest") if win > params.savgol_order else np.eye(T)
        # knot -> step linear interpolation matrix (T, n_knots) for correlated noise
        kn = max(1, params.noise_knot_steps)
        knots = np.arange(0, T + kn, kn, dtype=float)
        steps = np.arange(T, dtype=float)
        self._interp = np.stack([np.interp(steps, knots, np.eye(len(knots))[i]) for i in range(len(knots))], axis=1)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.U[:] = 0.0
        self._t_last = None

    def initialise_towards(self, pose: tuple[float, float, float], target_xy: tuple[float, float], v: float) -> None:
        """Seed the nominal with a pure-pursuit-like arc toward ``target_xy`` at speed ``v``."""
        dx, dy = target_xy[0] - pose[0], target_xy[1] - pose[1]
        err = math.atan2(math.sin(math.atan2(dy, dx) - pose[2]), math.cos(math.atan2(dy, dx) - pose[2]))
        w = float(np.clip(2.0 * err, -self.p.max_yaw_rate, self.p.max_yaw_rate))
        self.U[:, 0] = v if abs(err) < 0.8 else 0.0
        self.U[:, 1] = w

    def _add_guided(self, V: np.ndarray, W: np.ndarray, eps: np.ndarray, pose: tuple[float, float, float],
                    path: np.ndarray, v_ref: float) -> None:
        """Overwrite the last samples (in place) with the pure-pursuit guide: noisy copies, then
        noise-free pursuit at ``guide_speed_scales`` and rotate-in-place left / right."""
        p = self.p
        K, T = V.shape
        gv, gw = pursuit_controls(pose, path, v_ref, T, p.dt, p.pursuit_lookahead_m, p.pursuit_rotate_rad,
                                  p.pursuit_gain, p.max_yaw_rate)
        n_g = int(p.guide_fraction * K)
        if n_g <= 0:
            return
        g0 = K - n_g
        V[g0:] = gv[None, :] + eps[g0:, :, 0]
        W[g0:] = gw[None, :] + eps[g0:, :, 1]
        fixed = [(gv * s, gw) for s in p.guide_speed_scales]
        wr = p.rotate_rate_frac * p.max_yaw_rate
        fixed += [(np.zeros(T), np.full(T, wr)), (np.zeros(T), np.full(T, -wr))]
        for k, (fv, fw) in enumerate(fixed[: max(n_g, 0)]):
            V[g0 + k], W[g0 + k] = fv, fw

    def _clip(self, V: np.ndarray, W: np.ndarray, v_cap: float, v0: float) -> tuple[np.ndarray, np.ndarray]:
        p = self.p
        tt = np.arange(1, p.horizon + 1) * p.dt
        lo = np.maximum(v0 - p.brake_decel * tt, -max(self._v_rev, 0.0))
        hi = np.minimum(v0 + p.max_accel * tt, max(v_cap, 0.0))
        V = np.clip(V, np.minimum(lo, hi), hi)
        half = self.chi * p.track_width_m / 2.0
        w_lim = np.clip((p.v_wheel_max - np.abs(V)) / half, 0.0, p.max_yaw_rate)
        W = np.clip(W, -w_lim, w_lim)
        return V, W

    def plan(
        self,
        t: float,
        pose: tuple[float, float, float],
        v0: float,
        w0: float,
        v_cap: float,
        maps: PlanningMaps,
        ctg: CtgFn,
        t_r: float,
        v_ref: Optional[float] = None,
        cert_enabled: bool = True,
        margin_m: float = GOVERNOR_MARGIN_M,
        brake_decel: Optional[float] = None,
        chi: float = CHI_NOMINAL,
        guide_path_xy: Optional[np.ndarray] = None,
        hold_s: Optional[float] = None,
        allow_reverse: bool = False,
    ) -> MppiResult:
        """One MPPI iteration from ``pose`` (A-frame) with current control (v0, w0).

        ``guide_path_xy``: optional (N, 2) A-frame path (vehicle -> goal) for guided sampling
        (see module docstring). ``hold_s``: how long the returned command will be held by the
        caller [s] (e.g. one camera frame); ``u0`` is then the nominal control at the end of
        that interval instead of the first 0.1 s step, so the command already contains the
        acceleration the nominal plans over the hold (None: first step). ``allow_reverse``:
        permit reversing down to ``-v_reverse_max`` this tick (the caller enables it only when
        the footprint is already inside the lethal inflation, i.e. to back off)."""
        t_start = time.perf_counter()
        p = self.p
        K, T, dt = p.n_samples, p.horizon, p.dt
        self.chi = chi
        self._v_rev = p.v_reverse_max if allow_reverse else 0.0
        a_brk = brake_decel if brake_decel is not None else p.brake_decel
        v_ref = v_cap if v_ref is None else v_ref

        # warm start: shift by elapsed control steps
        if self._t_last is not None:
            shift = int(np.clip(round((t - self._t_last) / dt), 0, T))
            if shift >= T:
                self.U[:] = self.U[-1]
            elif shift > 0:
                self.U[: T - shift] = self.U[shift:].copy()
                self.U[T - shift:] = self.U[T - shift - 1]
        self._t_last = t
        self.U[:, 0], self.U[:, 1] = self._clip(self.U[:, 0], self.U[:, 1], v_cap, v0)
        U = self.U

        knots = self.rng.standard_normal((K, self._interp.shape[1], 2)) * np.array([p.sigma_v, p.sigma_w])
        eps = np.matmul(self._interp, knots)  # (T,nk) @ (K,nk,2) -> (K,T,2)
        eps[0] = 0.0
        V = U[None, :, 0] + eps[..., 0]
        W = U[None, :, 1] + eps[..., 1]
        if guide_path_xy is not None and len(guide_path_xy) >= 2 and v_ref > 0.0:
            self._add_guided(V, W, eps, pose, np.asarray(guide_path_xy, np.float64), min(v_ref, max(v_cap, 0.0)))
        V[1], W[1] = 0.0, 0.0  # full-stop candidate
        if self._v_rev > 0.0:
            V[2], W[2] = -self._v_rev, 0.0  # straight back-off candidate (escape from inflation)
        V, W = self._clip(V, W, v_cap, v0)
        eps_v, eps_w = V - U[None, :, 0], W - U[None, :, 1]

        X, Y, YAW = rollout(pose, V, W, dt)
        c, s = np.cos(YAW), np.sin(YAW)

        # footprint circles (K, T, C)
        off = np.asarray(p.circle_offsets_m)
        CX = X[..., None] + off * c[..., None]
        CY = Y[..., None] + off * s[..., None]
        cmap = np.take(maps.cost.ravel(), maps.flat_index(CX, CY))
        lethal_hits = np.count_nonzero(cmap >= 1.0, axis=(1, 2))
        s_map = p.w_map * dt * cmap.sum(axis=(1, 2))
        s_lethal = p.w_lethal * lethal_hits

        # certification: probes ahead of each state at fractions of its stopping distance
        if cert_enabled:
            dstop = V * V / (2.0 * a_brk) + np.abs(V) * t_r + margin_m
            fr = np.asarray(p.cert_fracs)
            # probes ahead in the direction of travel (behind the rear bumper when reversing)
            dist = np.where(V < 0.0, -1.0, 1.0)[..., None] * (p.front_m + dstop[..., None] * fr)
            PX = X[..., None] + dist * c[..., None]
            PY = Y[..., None] + dist * s[..., None]
            cert = np.take(maps.certified.ravel(), maps.flat_index(PX, PY))
            s_cert = p.w_cert * dt * np.count_nonzero(~cert, axis=(1, 2))
        else:
            s_cert = np.zeros(K)

        dv = np.diff(np.concatenate([np.full((K, 1), v0), V], axis=1), axis=1)
        dw = np.diff(np.concatenate([np.full((K, 1), w0), W], axis=1), axis=1)
        s_smooth = p.w_dv * (dv * dv).sum(axis=1) + p.w_dw * (dw * dw).sum(axis=1)
        s_speed = p.w_speed * dt * ((V - v_ref) ** 2).sum(axis=1) + p.w_reverse * dt * np.maximum(-V, 0.0).sum(axis=1)
        s_term = p.w_ctg * ctg(X[:, -1], Y[:, -1])
        s_ctrl = p.gamma * p.lam * (
            (U[None, :, 0] * eps_v).sum(axis=1) / p.sigma_v**2 + (U[None, :, 1] * eps_w).sum(axis=1) / p.sigma_w**2
        )
        S = s_map + s_lethal + s_cert + s_smooth + s_speed + s_term + s_ctrl

        s_min = float(S.min())
        w = np.exp(-(S - s_min) / p.lam)
        w /= w.sum()
        ess = float(1.0 / np.sum(w * w))
        U_new = U.copy()
        U_new[:, 0] += w @ eps_v
        U_new[:, 1] += w @ eps_w
        U_new = self._sg @ U_new
        U_new[:, 0], U_new[:, 1] = self._clip(U_new[:, 0], U_new[:, 1], v_cap, v0)
        self.U = U_new

        nx, ny, nyaw = rollout(pose, U_new[:, 0], U_new[:, 1], dt)
        k_best = int(np.argmin(S))
        order = np.argsort(S)
        n_top = p.n_vis // 2
        rand = self.rng.choice(K, size=p.n_vis - n_top, replace=False)
        vis = np.concatenate([order[:n_top], rand])
        terms = {
            "map": float(w @ s_map), "lethal": float(w @ s_lethal), "cert": float(w @ s_cert), "smooth": float(w @ s_smooth),
            "speed": float(w @ s_speed), "ctg": float(w @ s_term),
        }
        i0 = 0 if hold_s is None else int(np.clip(round(hold_s / dt) - 1, 0, T - 1))
        return MppiResult(
            u0=(float(U_new[i0, 0]), float(U_new[i0, 1])),
            U=U_new.copy(),
            traj=np.column_stack([nx, ny, nyaw]),
            rollouts_xy=np.stack([X[vis], Y[vis]], axis=-1),
            s_min=s_min,
            ess=ess,
            best_lethal=bool(lethal_hits[k_best] > 0),
            compute_ms=(time.perf_counter() - t_start) * 1e3,
            cost_terms=terms,
        )
