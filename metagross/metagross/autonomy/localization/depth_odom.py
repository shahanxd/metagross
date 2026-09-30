"""Depth odometry: frame-to-frame 6-DoF registration of consecutive disparity maps.

Used where there are no images (Tier-0 synthetic depth sensor) so stereo VO cannot run, and as
an optional fallback when stereo VO fails. It measures *ground* motion from scene geometry
(rolling terrain, rocks, trees), so unlike the wheel encoders it does not over-count under
longitudinal slip.

Method (dense direct alignment on inverse depth, projective data association, cf. DIFODO,
Jaimez & Gonzalez-Jimenez, ICRA 2015):

1. Working map: the disparity at the sensor's native resolution (Tier-0 maps are x2
   nearest-upsampled, so ``decimate = 2`` recovers the native 320 x 200 map), smoothed by
   normalised convolution over valid pixels; gradients by central differences. A pixel is
   *good* when it and its 3x3 neighbours are valid and the local disparity spread is small
   (no depth discontinuity).
2. Points: good pixels of the previous map (stride-sampled) with depth in
   ``[z_min_m, z_max_m]``, back-projected to the previous camera frame (OpenCV axes, metres).
3. Gauss-Newton on ``T_cur_prev`` (SE(3), left perturbation ``(dt, dw)``) minimising the Huber-
   weighted residual ``r = D_cur(project(T p)) - f B / z(T p)`` (px, homogeneous noise), from the
   wheel + gyro prior. A flat, featureless ground plane leaves in-plane translation and yaw
   unobservable: the solution then stays at the prior and the covariance is large.
4. Covariance: ``sigma_r^2 (J^T W J)^-1`` (``sigma_r`` from the weighted residual), inflated by
   ``cov_inflation`` (residuals are spatially correlated), mapped to the planar body increment
   by a numerical Jacobian of the camera->body conjugation.

Output: planar body-frame increment of the current frame w.r.t. the previous one (dx m forward,
dy m left, dyaw rad CCW) with its 3x3 covariance, like the stereo VO increment the EKF fuses.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

LOG = logging.getLogger(__name__)

N_DOF = 6  # SE(3) parameters (tx, ty, tz, wx, wy, wz)
FD_STEP = 1e-6  # finite-difference step of the planar-projection Jacobian (m / rad)


@dataclass(slots=True)
class DepthOdomConfig:
    """Depth-odometry parameters. Pixel values are at the *working* resolution."""

    decimate: int = 2  # working map = disparity[::decimate, ::decimate] (Tier-0 native resolution)
    point_stride: int = 2  # sample every n-th working pixel (rows and columns) as a 3-D point
    max_points: int = 5000  # cap (deterministic stride thinning beyond this)
    min_points: int = 400  # fewer usable points -> no measurement
    z_min_m: float = 0.5  # nearer points are dropped (inside the body footprint / bumper zone)
    z_max_m: float = 10.0  # farther points carry little translation information (disparity ~ 1/Z)
    smooth_sigma_px: float = 1.0  # Gaussian sigma of the normalised-convolution smoothing
    edge_spread_px: float = 0.6  # max 3x3 disparity spread of a "good" pixel (depth discontinuity test)
    iters: int = 8
    conv_tol: float = 5e-5  # stop when max |delta| (m / rad) is below this
    huber_k: float = 1.5  # Huber threshold in robust sigmas
    sigma_floor_px: float = 0.03  # floor of the robust residual scale
    damping: float = 1e-6  # Levenberg damping relative to mean(diag(H))
    max_step_m: float = 0.02  # trust region per Gauss-Newton step (the wheel prior is within ~10 % of travel)
    max_step_rad: float = 0.01
    cov_inflation: float = 4.0  # residual correlation / model error inflation of the covariance
    max_sigma_fwd_m: float = 0.05  # forward 1-sigma above this: too little structure -> no measurement
    max_inlier_rmse_px: float = 0.5  # weighted residual RMS above this: registration failed


@dataclass(slots=True)
class DepthOdomResult:
    """One registration. ``dx, dy, dyaw``: body increment previous -> current frame (m, m, rad)."""

    ok: bool
    reason: str
    dx: float = 0.0
    dy: float = 0.0
    dyaw: float = 0.0
    cov: np.ndarray = field(default_factory=lambda: np.eye(3))
    T_prev_cur_cam: np.ndarray = field(default_factory=lambda: np.eye(4))
    stats: dict[str, float] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class _Map:
    """Prepared working disparity map (float32 px at working resolution, 0 = invalid)."""

    D: np.ndarray  # (H, W) smoothed disparity
    good: np.ndarray  # (H, W) bool: valid, no discontinuity, full 3x3 support
    stack: np.ndarray  # (H*W, 3) interleaved (D, dD/du, dD/dv) for bilinear gathers
    gu: np.ndarray  # (H, W) dD/du (view, for inspection)


def _exp_se3(delta: np.ndarray) -> np.ndarray:
    """4x4 transform of a small twist ``(tx, ty, tz, wx, wy, wz)`` (translation not coupled)."""
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(delta[3:], float).reshape(3, 1))[0]
    T[:3, 3] = delta[:3]
    return T


def planar_from_body(T_b: np.ndarray) -> np.ndarray:
    """(dx m, dy m, dyaw rad) of a 4x4 body-frame relative pose (previous -> current)."""
    return np.array([T_b[0, 3], T_b[1, 3], math.atan2(T_b[1, 0], T_b[0, 0])])


def body_from_planar(dx: float, dy: float, dyaw: float) -> np.ndarray:
    """4x4 body relative pose of a planar increment."""
    T = np.eye(4)
    c, s = math.cos(dyaw), math.sin(dyaw)
    T[:2, :2] = [[c, -s], [s, c]]
    T[0, 3], T[1, 3] = dx, dy
    return T


class DepthOdometry:
    """Stateful frame-to-frame depth odometry (one instance per mission).

    ``K``: 3x3 intrinsics of the *input* disparity map (px); ``baseline_m``: stereo baseline (m);
    ``T_body_cam``: 4x4 left camera -> body transform.
    """

    def __init__(self, K: np.ndarray, baseline_m: float, T_body_cam: np.ndarray,
                 config: Optional[DepthOdomConfig] = None) -> None:
        self.cfg = config or DepthOdomConfig()
        s = 1.0 / self.cfg.decimate
        K = np.asarray(K, float)
        # Pixel centres: working pixel j covers input pixels [j*dec, (j+1)*dec); centre mapping below.
        self.fx, self.fy = K[0, 0] * s, K[1, 1] * s
        self.cx, self.cy = (K[0, 2] + 0.5) * s - 0.5, (K[1, 2] + 0.5) * s - 0.5
        self.fB = self.fx * float(baseline_m)  # working-res disparity x depth (px m)
        self.T_bc = np.asarray(T_body_cam, float)
        self.T_cb = np.linalg.inv(self.T_bc)
        self.reset()

    def reset(self) -> None:
        self._prev: Optional[_Map] = None  # prepared map of the previous frame
        self._prev_raw: Optional[np.ndarray] = None  # raw previous map, prepared lazily (see observe)
        self.n_frames = 0
        self.n_ok = 0

    def observe(self, disparity: np.ndarray) -> None:
        """Store a frame without registering it (O(1): prepared only if the next frame registers).

        Used by the stereo VO fallback, where most frames do not need depth odometry."""
        self._prev, self._prev_raw = None, disparity
        self.n_frames += 1

    # ------------------------------------------------------------------ preprocessing
    def prepare(self, disparity: np.ndarray) -> _Map:
        """Working map from an input disparity map (px at input resolution, <= 0 invalid)."""
        c = self.cfg
        d = np.asarray(disparity, np.float32)[:: c.decimate, :: c.decimate] * np.float32(1.0 / c.decimate)
        valid = (d > 0).astype(np.float32)
        dz = np.where(d > 0, d, np.float32(0.0))
        ks = (0, 0)
        num = cv2.GaussianBlur(dz, ks, c.smooth_sigma_px, borderType=cv2.BORDER_CONSTANT)
        den = cv2.GaussianBlur(valid, ks, c.smooth_sigma_px, borderType=cv2.BORDER_CONSTANT)
        Ds = np.where(valid > 0, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
        k3 = np.ones((3, 3), np.uint8)
        full = cv2.erode(valid, k3, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        hi = cv2.dilate(Ds, k3)
        lo = cv2.erode(np.where(full, Ds, np.float32(1e6)), k3)
        good = full & ((hi - lo) < c.edge_spread_px)
        g = np.zeros(Ds.shape + (3,), np.float32)  # interleaved (D, dD/du, dD/dv): one gather per corner
        g[..., 0] = Ds
        g[:, 1:-1, 1] = 0.5 * (Ds[:, 2:] - Ds[:, :-2])
        g[1:-1, :, 2] = 0.5 * (Ds[2:, :] - Ds[:-2, :])
        return _Map(Ds, good, g.reshape(-1, 3), g[..., 1])

    def _points(self, m: _Map) -> np.ndarray:
        """(N, 3) camera-frame points (m) of the good pixels of a working map (stride-sampled)."""
        c = self.cfg
        st = c.point_stride
        sub = m.good[::st, ::st]
        vv, uu = np.nonzero(sub)
        vv, uu = vv * st, uu * st
        d = m.D[vv, uu]
        z = self.fB / np.maximum(d, 1e-6)
        keep = (z >= c.z_min_m) & (z <= c.z_max_m)
        vv, uu, z = vv[keep], uu[keep], z[keep]
        if len(z) > c.max_points:  # deterministic thinning
            idx = np.linspace(0, len(z) - 1, c.max_points).astype(np.intp)
            vv, uu, z = vv[idx], uu[idx], z[idx]
        x = (uu - self.cx) / self.fx * z
        y = (vv - self.cy) / self.fy * z
        return np.stack([x, y, z], axis=1).astype(np.float64)

    # ------------------------------------------------------------------ residuals
    def _residuals(self, cur: _Map, P: np.ndarray, T: np.ndarray, want_jac: bool
                   ) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Residual (px), Jacobian (N, 6) w.r.t. a left twist on ``T`` and the usable mask."""
        H, W = cur.D.shape
        Q = P @ T[:3, :3].T + T[:3, 3]
        z = Q[:, 2]
        front = z > 0.1
        iz = 1.0 / np.where(front, z, 1.0)
        u = self.fx * Q[:, 0] * iz + self.cx
        v = self.fy * Q[:, 1] * iz + self.cy
        inside = front & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        u0 = np.floor(np.where(inside, u, 0.0)).astype(np.intp)
        v0 = np.floor(np.where(inside, v, 0.0)).astype(np.intp)
        fu, fv = u - u0, v - v0
        idx = v0 * W + u0
        good = cur.good.ravel()
        ok = inside & good[idx] & good[idx + 1] & good[idx + W] & good[idx + W + 1]
        fu, fv = fu[:, None], fv[:, None]
        st = cur.stack
        b = ((1 - fu) * (1 - fv) * st[idx] + fu * (1 - fv) * st[idx + 1]
             + (1 - fu) * fv * st[idx + W] + fu * fv * st[idx + W + 1])
        r = b[:, 0] - self.fB * iz
        if not want_jac:
            return r, None, ok
        gu, gv = b[:, 1], b[:, 2]
        a = np.empty_like(Q)
        a[:, 0] = gu * self.fx * iz
        a[:, 1] = gv * self.fy * iz
        a[:, 2] = (-(gu * self.fx * Q[:, 0] + gv * self.fy * Q[:, 1]) + self.fB) * iz * iz
        J = np.empty((len(r), N_DOF))
        J[:, :3] = a
        J[:, 3] = Q[:, 1] * a[:, 2] - Q[:, 2] * a[:, 1]  # Q x a
        J[:, 4] = Q[:, 2] * a[:, 0] - Q[:, 0] * a[:, 2]
        J[:, 5] = Q[:, 0] * a[:, 1] - Q[:, 1] * a[:, 0]
        return r, J, ok

    # ------------------------------------------------------------------ main
    def cam_from_body_increment(self, dx: float, dy: float, dyaw: float) -> np.ndarray:
        """T_prev_cur in the camera frame of a planar body increment (4x4)."""
        return self.T_cb @ body_from_planar(dx, dy, dyaw) @ self.T_bc

    def process(self, disparity: np.ndarray, prior: tuple[float, float, float]) -> DepthOdomResult:
        """Register ``disparity`` against the previous frame's map.

        ``prior``: predicted body increment (dx m, dy m, dyaw rad) previous -> current frame
        (wheels + gyro). The first frame only stores the map (``reason='init'``).
        """
        c = self.cfg
        t0 = time.perf_counter()
        cur = self.prepare(disparity)
        prev = self._prev
        if prev is None and self._prev_raw is not None:
            prev = self.prepare(self._prev_raw)
        t_prep = (time.perf_counter() - t0) * 1e3
        self._prev, self._prev_raw = cur, None
        self.n_frames += 1
        if prev is None:
            return DepthOdomResult(False, "init", timings_ms={"prepare": t_prep, "total": t_prep})
        P = self._points(prev)
        stats: dict[str, float] = {"points": float(len(P))}
        if len(P) < c.min_points:
            tt = (time.perf_counter() - t0) * 1e3
            return DepthOdomResult(False, "few points", stats=stats, timings_ms={"prepare": t_prep, "total": tt})
        T = np.linalg.inv(self.cam_from_body_increment(*prior))  # T_cur_prev (camera)
        sigma = None
        n_it = 0
        r = J = ok = w = None
        for n_it in range(1, c.iters + 1):
            r, J, ok = self._residuals(cur, P, T, want_jac=True)
            if int(ok.sum()) < c.min_points:
                break
            if sigma is None:  # first pass: keep only points with a usable association (cheaper iterations)
                P, r, J, ok = P[ok], r[ok], J[ok], ok[ok]
            ro = r[ok]
            if sigma is None:  # robust scale from the prior's residuals, then fixed
                sigma = max(1.4826 * float(np.median(np.abs(ro - np.median(ro)))), c.sigma_floor_px)
            thr = c.huber_k * sigma
            aro = np.abs(ro)
            w = np.where(aro <= thr, 1.0, thr / np.maximum(aro, 1e-12))
            Jo = J[ok]
            Hm = (Jo * w[:, None]).T @ Jo
            g = (Jo * w[:, None]).T @ ro
            lam = c.damping * float(np.trace(Hm)) / N_DOF
            delta = -np.linalg.solve(Hm + lam * np.eye(N_DOF), g)
            # trust region: a weakly constrained direction must not throw the solution out of the basin
            shrink = max(float(np.max(np.abs(delta[:3]))) / c.max_step_m,
                         float(np.max(np.abs(delta[3:]))) / c.max_step_rad, 1.0)
            delta /= shrink
            T = _exp_se3(delta) @ T
            if float(np.max(np.abs(delta))) < c.conv_tol:
                break
        t_reg = (time.perf_counter() - t0) * 1e3 - t_prep
        # final residuals and Hessian at the solution
        r, J, ok = self._residuals(cur, P, T, want_jac=True)
        n_ok = int(ok.sum())
        stats.update({"used": float(n_ok), "iters": float(n_it)})
        if sigma is None or n_ok < c.min_points:
            tt = (time.perf_counter() - t0) * 1e3
            return DepthOdomResult(False, "few overlaps", stats=stats,
                                   timings_ms={"prepare": t_prep, "register": t_reg, "total": tt})
        ro, Jo = r[ok], J[ok]
        thr = c.huber_k * sigma
        aro = np.abs(ro)
        w = np.where(aro <= thr, 1.0, thr / np.maximum(aro, 1e-12))
        Hm = (Jo * w[:, None]).T @ Jo
        rmse = math.sqrt(float(np.sum(w * ro * ro)) / max(float(np.sum(w)), 1.0))
        s2 = float(np.sum(w * ro * ro)) / max(float(np.sum(w)) - N_DOF, 1.0)
        lam = c.damping * float(np.trace(Hm)) / N_DOF
        cov6 = s2 * np.linalg.inv(Hm + lam * np.eye(N_DOF)) * c.cov_inflation
        T_prev_cur = np.linalg.inv(T)
        T_b = self.T_bc @ T_prev_cur @ self.T_cb
        z = planar_from_body(T_b)
        # planar Jacobian w.r.t. the left twist on T_cur_prev (numerical, 6 small products)
        Jp = np.empty((3, N_DOF))
        for k in range(N_DOF):
            e = np.zeros(N_DOF)
            e[k] = FD_STEP
            Tk = self.T_bc @ np.linalg.inv(_exp_se3(e) @ T) @ self.T_cb
            dz = planar_from_body(Tk) - z
            dz[2] = math.atan2(math.sin(dz[2]), math.cos(dz[2]))
            Jp[:, k] = dz / FD_STEP
        cov = Jp @ cov6 @ Jp.T
        cov = 0.5 * (cov + cov.T)
        sig_fwd = math.sqrt(max(float(cov[0, 0]), 0.0))
        stats.update({"rmse_px": rmse, "sigma_px": sigma, "sigma_fwd_m": sig_fwd,
                      "sigma_yaw_rad": math.sqrt(max(float(cov[2, 2]), 0.0)),
                      "inlier_frac": float(np.mean(aro <= thr))})
        ok_meas, reason = True, ""
        if rmse > c.max_inlier_rmse_px:
            ok_meas, reason = False, f"rmse {rmse:.2f} px"
        elif sig_fwd > c.max_sigma_fwd_m:
            ok_meas, reason = False, "low structure"
        if ok_meas:
            self.n_ok += 1
        tt = (time.perf_counter() - t0) * 1e3
        return DepthOdomResult(ok_meas, reason, float(z[0]), float(z[1]), float(z[2]), cov, T_prev_cur, stats,
                               {"prepare": t_prep, "register": t_reg, "total": tt})


__all__ = ["DepthOdomConfig", "DepthOdomResult", "DepthOdometry", "body_from_planar", "planar_from_body"]
