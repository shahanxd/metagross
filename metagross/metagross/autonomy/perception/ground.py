"""Robust ground model: v-disparity profile + continuity-checked RANSAC plane bands.

Two complementary estimates are produced every frame:

1. **v-disparity profile** (image space). For every image row v the dominant disparity
   is taken from a per-row histogram; a RANSAC line ``d = a v + b`` through the row
   peaks (weighted by peak support) is refined by a Tukey-IRLS quadratic. The line
   absorbs camera pitch (``a`` and the horizon row ``-b/a`` move with pitch); the
   quadratic term absorbs gentle rolling terrain. Output: expected ground disparity
   per row, ``VDisparityProfile.expected(v)`` in px (<= 0 above the horizon).

2. **Piecewise planar ground in the body frame.** Ground candidates (points whose
   disparity agrees with the v-disparity profile) are split into range bands of
   ``BAND_WIDTH_M`` along body x; each band gets a vectorised RANSAC plane
   ``z = a x + b y + c`` refined by least squares. Bands are fitted near-to-far and a
   band is accepted only if it is *continuous* with the accepted band before it (height
   jump at the shared edge < ``BAND_CONTINUITY_M`` and normal change <
   ``BAND_MAX_TILT_CHANGE_DEG``); otherwise the previous plane is extrapolated. The
   model therefore describes ground *connected to where the vehicle stands*: terrain
   beyond a drop-off does not silently become "ground". Planes are blended linearly
   between band centres, giving a continuous height function ``height(x, y)`` (m, body
   frame) and, by iterated ray / tangent-plane intersection, the expected ground
   disparity ``expected_disparity(v, u)`` (px) for any pixel.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from metagross.autonomy.perception.geometry import CameraGeometry, PointSet
from metagross.config import defaults

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------- v-disparity
VD_BIN_PX = 0.25  # histogram bin width (px)
VD_COL_STRIDE = 4  # columns sampled for the histogram
VD_MIN_PEAK_FRAC = 0.08  # row peak must hold >= this fraction of the row's sampled pixels
VD_MIN_PEAK_COUNT = 6
VD_RANSAC_ITERS = 128
VD_INLIER_PX = 0.6  # absolute inlier tolerance (px) ...
VD_INLIER_REL = 0.06  # ... or relative to disparity, whichever is larger
VD_MIN_ROWS = 12  # minimum inlier rows for a usable profile
VD_TUKEY_C = 2.5  # Tukey biweight constant in units of the inlier tolerance
VD_QUAD_MIN_SPAN_ROWS = 60  # quadratic term only when inliers span this many rows

# ---------------------------------------------------------------- plane bands
BAND_WIDTH_M = 2.0  # range band width along body x (m)
BAND_X_MAX_M = defaults.CAM_MAX_RANGE_M
BAND_MIN_POINTS = 40
BAND_RANSAC_ITERS = 32
BAND_RANSAC_MAX_EVAL = 512  # points used to score hypotheses
BAND_MAX_POINTS = 3000  # points per band kept for the refinement (deterministic stride subsample)
BAND_PRIOR_WEIGHT = 10.0  # ridge weight (points*m^2) pulling band slopes to the nearer band's
BAND_TOL_M = 0.04  # plane inlier tolerance at 0 m ...
BAND_TOL_PER_M = 0.006  # ... growing with range (stereo depth noise grows with Z^2)
BAND_MAX_TILT_DEG = 30.0  # reject near-vertical hypotheses (walls, rock faces)
BAND_CONTINUITY_M = 0.15  # max height step between consecutive band planes at their shared edge
BAND_MAX_TILT_CHANGE_DEG = 15.0
CANDIDATE_DISP_TOL_PX = 1.5  # ground-candidate prefilter vs v-disparity profile (abs) ...
CANDIDATE_DISP_TOL_REL = 0.3  # ... or relative
BAND_CANDIDATE_Z_TOL_M = 0.4  # per band: within this height of the extrapolated nearer plane
FIRST_BAND_MAX_OFFSET_M = 0.35  # |z(0,0)| of the nearest plane: the vehicle stands on the ground
RAY_ITERS = 2  # fixed-point iterations for ray / piecewise-ground intersection


@dataclass
class VDisparityProfile:
    """Expected ground disparity per image row (column independent)."""

    coeffs: np.ndarray  # polynomial in v (highest power first), np.polyval convention
    line: np.ndarray  # RANSAC line (a, b): d = a v + b
    v_min: int  # inlier row span; outside it the line is used
    v_max: int
    n_inlier_rows: int
    height: int

    def expected(self, v: np.ndarray) -> np.ndarray:
        """Expected ground disparity (px) at rows v; <= 0 above the horizon."""
        v = np.asarray(v, dtype=np.float64)
        inside = (v >= self.v_min) & (v <= self.v_max)
        d = np.where(inside, np.polyval(self.coeffs, v), np.polyval(self.line, v))
        # Keep the curve continuous at the span ends by matching the line offset there.
        lo = np.polyval(self.coeffs, self.v_min) - np.polyval(self.line, self.v_min)
        hi = np.polyval(self.coeffs, self.v_max) - np.polyval(self.line, self.v_max)
        d = np.where(v < self.v_min, d + lo, d)
        d = np.where(v > self.v_max, d + hi, d)
        return np.maximum(d, 0.0)

    @property
    def horizon_row(self) -> float:
        a, b = self.line
        return float(-b / a) if a > 0 else float("nan")


def fit_v_disparity(disp: np.ndarray, geom: CameraGeometry, rng: np.random.Generator) -> Optional[VDisparityProfile]:
    """Robust v-disparity ground profile from a disparity image (px). None if unusable."""
    h, w = disp.shape
    d = disp[:, VD_COL_STRIDE // 2 :: VD_COL_STRIDE]
    d_hi = float(np.nanmax(d)) if d.size else 0.0
    if d_hi <= geom.min_disparity_px:
        return None
    nb = int(math.ceil((d_hi + 1.0) / VD_BIN_PX))
    valid = d >= geom.min_disparity_px
    vv = np.broadcast_to(np.arange(h)[:, None], d.shape)[valid]
    bins = np.minimum((d[valid] / VD_BIN_PX).astype(np.int64), nb - 1)
    hist = np.bincount(vv * nb + bins, minlength=h * nb).reshape(h, nb).astype(np.float32)
    # [1 2 1] smoothing along disparity makes the peak robust to sub-bin jitter.
    hs = hist.copy()
    hs[:, 1:] += hist[:, :-1]
    hs[:, :-1] += hist[:, 1:]
    peak_bin = np.argmax(hs, axis=1)
    peak_cnt = hist[np.arange(h), peak_bin] + 0.5 * (hist[np.arange(h), np.maximum(peak_bin - 1, 0)] + hist[np.arange(h), np.minimum(peak_bin + 1, nb - 1)])
    row_cnt = valid.sum(axis=1)
    ok = (peak_cnt >= VD_MIN_PEAK_COUNT) & (peak_cnt >= VD_MIN_PEAK_FRAC * d.shape[1])
    rows = np.nonzero(ok)[0]
    if rows.size < VD_MIN_ROWS:
        return None
    # Sub-bin peak: weighted mean of the three bins around the peak.
    pb = peak_bin[rows]
    idx = np.stack([np.maximum(pb - 1, 0), pb, np.minimum(pb + 1, nb - 1)], axis=1)
    wts = hist[rows[:, None], idx]
    peak_d = ((idx + 0.5) * VD_BIN_PX * wts).sum(1) / np.maximum(wts.sum(1), 1e-6)
    weight = peak_cnt[rows] / np.maximum(row_cnt[rows], 1)
    vr = rows.astype(np.float64)

    # --- RANSAC line through (v, peak_d)
    i = rng.integers(0, rows.size, size=(VD_RANSAC_ITERS, 2))
    v1, v2, d1, d2 = vr[i[:, 0]], vr[i[:, 1]], peak_d[i[:, 0]], peak_d[i[:, 1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (d2 - d1) / (v2 - v1)
    b = d1 - a * v1
    good = np.isfinite(a) & (a > 0)
    if not np.any(good):
        return None
    a, b = a[good], b[good]
    tol = np.maximum(VD_INLIER_PX, VD_INLIER_REL * peak_d)
    res = np.abs(peak_d[None, :] - (a[:, None] * vr[None, :] + b[:, None]))
    score = ((res < tol[None, :]) * weight[None, :]).sum(1)
    k = int(np.argmax(score))
    inl = res[k] < tol
    if inl.sum() < VD_MIN_ROWS:
        return None
    line = np.polyfit(vr[inl], peak_d[inl], 1, w=np.sqrt(weight[inl]))
    if line[0] <= 0:
        return None
    # --- Tukey IRLS refinement (quadratic if the inliers span enough rows)
    inl = np.abs(peak_d - np.polyval(line, vr)) < tol
    span = vr[inl].max() - vr[inl].min()
    deg = 2 if span >= VD_QUAD_MIN_SPAN_ROWS else 1
    coeffs = np.polyfit(vr[inl], peak_d[inl], deg, w=np.sqrt(weight[inl]))
    for _ in range(3):
        r = (peak_d - np.polyval(coeffs, vr)) / (VD_TUKEY_C * tol)
        tw = np.where(np.abs(r) < 1.0, (1.0 - r**2) ** 2, 0.0) * weight
        if (tw > 0).sum() < VD_MIN_ROWS:
            break
        coeffs = np.polyfit(vr, peak_d, deg, w=np.sqrt(tw))
    inl = np.abs(peak_d - np.polyval(coeffs, vr)) < tol
    if deg == 1:
        coeffs = np.concatenate([[0.0], coeffs])
    return VDisparityProfile(coeffs=coeffs, line=line, v_min=int(vr[inl].min()), v_max=int(vr[inl].max()),
                             n_inlier_rows=int(inl.sum()), height=h)


# ---------------------------------------------------------------- band planes
@dataclass
class BandPlane:
    """Plane z = a x + b y + c (body frame, m) fitted to one range band."""

    x0: float
    x1: float
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    n_points: int = 0
    n_inliers: int = 0
    fitted: bool = False  # True if accepted from this band's own data, False if inherited

    def z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.a * x + self.b * y + self.c

    def normal(self) -> np.ndarray:
        n = np.array([-self.a, -self.b, 1.0])
        return n / np.linalg.norm(n)


def _tilt_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan(np.hypot(a, b)))


def _ransac_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray, tol: np.ndarray, rng: np.random.Generator,
                  prior_ab: tuple[float, float] = (0.0, 0.0)) -> Optional[tuple[float, float, float, np.ndarray]]:
    """Vectorised RANSAC for z = a x + b y + c. Returns (a, b, c, inlier_mask) or None.

    Hypotheses are scored on at most ``BAND_RANSAC_MAX_EVAL`` points; the winner is refined
    by ridge-regularised least squares on all inliers, pulling (a, b) towards ``prior_ab``
    (the nearer band's slopes) so that a narrow strip of points cannot produce an
    arbitrary tilt.
    """
    n = x.size
    ev = rng.choice(n, BAND_RANSAC_MAX_EVAL, replace=False) if n > BAND_RANSAC_MAX_EVAL else np.arange(n)
    idx = ev[rng.integers(0, ev.size, size=(BAND_RANSAC_ITERS, 3))]
    P = np.stack([x[idx], y[idx], z[idx]], axis=-1)  # (K, 3, 3)
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    nz = nrm[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = -nrm[:, 0] / nz
        b = -nrm[:, 1] / nz
    c = P[:, 0, 2] - a * P[:, 0, 0] - b * P[:, 0, 1]
    ok = np.isfinite(a) & np.isfinite(b) & (_tilt_deg(a, b) < BAND_MAX_TILT_DEG)
    if not np.any(ok):
        return None
    a, b, c = a[ok], b[ok], c[ok]
    xe, ye, ze, te = x[ev], y[ev], z[ev], tol[ev]
    res = np.abs(ze[None, :] - (a[:, None] * xe[None, :] + b[:, None] * ye[None, :] + c[:, None]))
    k = int(np.argmax((res < te[None, :]).sum(1)))
    inl = np.abs(z - (a[k] * x + b[k] * y + c[k])) < tol
    lam = math.sqrt(BAND_PRIOR_WEIGHT)
    for _ in range(2):  # regularised least-squares refinement on the inliers
        ni = int(inl.sum())
        if ni < 3:
            return None
        A = np.zeros((ni + 2, 3))
        A[:ni, 0], A[:ni, 1], A[:ni, 2] = x[inl], y[inl], 1.0
        A[ni, 0] = lam
        A[ni + 1, 1] = lam
        rhs = np.concatenate([z[inl], [lam * prior_ab[0], lam * prior_ab[1]]])
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        a_k, b_k, c_k = (float(s) for s in sol)
        r = np.abs(z - (a_k * x + b_k * y + c_k))
        mad = 1.4826 * float(np.median(r[inl]))
        inl = r < np.minimum(tol, np.maximum(0.02, 3.0 * mad))
    return a_k, b_k, c_k, inl


@dataclass
class GroundModel:
    """Continuous piecewise-planar ground in the body frame (see module docstring)."""

    bands: list[BandPlane]
    vdisp: Optional[VDisparityProfile] = None
    centres: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        self.centres = np.array([0.5 * (b.x0 + b.x1) for b in self.bands])
        self._A = np.array([b.a for b in self.bands])
        self._B = np.array([b.b for b in self.bands])
        self._C = np.array([b.c for b in self.bands])
        # Per-interval coefficient increments between consecutive band centres.
        self._step = float(self.centres[1] - self.centres[0]) if self.centres.size > 1 else 1.0
        inc = lambda v: np.append(np.diff(v), 0.0)  # noqa: E731  (last entry unused)
        self._dA, self._dB, self._dC = inc(self._A), inc(self._B), inc(self._C)

    # -------------------------------------------------------------- queries
    # The blended surface is z = A(x) x + B(x) y + C(x) with A, B, C the piecewise-linear
    # interpolants of the band coefficients over the (uniformly spaced) band centres,
    # clamped at the ends - algebraically identical to blending the two neighbouring planes.
    # Uniform spacing lets us index intervals with floor() instead of a binary search.
    def _interval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cen = self.centres
        step = float(cen[1] - cen[0]) if cen.size > 1 else 1.0
        f = (x - cen[0]) * (1.0 / step)
        k = np.clip(np.floor(f), 0, max(cen.size - 2, 0)).astype(np.intp)
        t_raw = f - k
        t = np.clip(t_raw, 0.0, 1.0)
        inner = (t_raw > 0.0) & (t_raw < 1.0)
        return k, t, inner

    def _coeffs(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        k, t, inner = self._interval(x)
        a = np.take(self._A, k) + t * np.take(self._dA, k)
        b = np.take(self._B, k) + t * np.take(self._dB, k)
        c = np.take(self._C, k) + t * np.take(self._dC, k)
        return a, b, c, k, inner

    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Expected ground height z (m, body frame) at body (x, y)."""
        x = np.asarray(x, dtype=np.float64)
        a, b, c, _, _ = self._coeffs(x)
        return a * x + b * np.asarray(y, dtype=np.float64) + c

    def height_and_gradient(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(z, dz/dx, dz/dy) of the blended ground at body (x, y)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        a, b, c, k, inner = self._coeffs(x)
        z = a * x + b * y + c
        inv = 1.0 / self._step
        dslope = (x * np.take(self._dA, k) + y * np.take(self._dB, k) + np.take(self._dC, k)) * inv
        gx = a + np.where(inner, dslope, 0.0)
        return z, gx, b

    def gradient(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(dz/dx, dz/dy) of the blended ground (dimensionless)."""
        _, gx, gy = self.height_and_gradient(x, y)
        return gx, gy

    def expected_depth(self, geom: CameraGeometry, v: np.ndarray, u: np.ndarray,
                       rays: Optional[np.ndarray] = None) -> np.ndarray:
        """Optical depth Z_c (m) where pixel rays (v, u) meet the ground; inf if they miss.

        ``rays`` (..., 3) may be passed to skip the lookup ``geom.rays_body[v, u]``.
        Starts from the nearest band plane and iterates ray / tangent-plane intersections.
        """
        if rays is None:
            rays = geom.rays_body[v, u]
        rx, ry, rz = (np.ascontiguousarray(rays[..., i], dtype=np.float64) for i in range(3))
        tx, ty, tz = geom.t_bc
        def plane(bp: BandPlane) -> np.ndarray:
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (bp.a * tx + bp.b * ty + bp.c - tz) / (rz - bp.a * rx - bp.b * ry)
            return np.where((z > 0) & np.isfinite(z), z, np.inf)

        zc = plane(self.bands[0])
        z_far = plane(self.bands[-1])
        for _ in range(RAY_ITERS):
            fin = np.isfinite(zc)
            zq = np.where(fin, zc, np.where(np.isfinite(z_far), z_far, 0.0))
            x = tx + zq * rx
            y = ty + zq * ry
            z0, gx, gy = self.height_and_gradient(x, y)
            num = z0 - gx * x - gy * y + gx * tx + gy * ty - tz
            den = rz - gx * rx - gy * ry
            with np.errstate(divide="ignore", invalid="ignore"):
                zn = num / den
            zc = np.where((zn > 0) & np.isfinite(zn), zn, np.inf)
        return zc

    def expected_disparity(self, geom: CameraGeometry, v: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Expected ground disparity (px) at pixels (v, u); 0 where the ray misses the ground."""
        zc = self.expected_depth(geom, v, u)
        with np.errstate(divide="ignore"):
            return np.where(np.isfinite(zc) & (zc > 0), geom.fxb / zc, 0.0)

    @property
    def n_fitted(self) -> int:
        return int(sum(b.fitted for b in self.bands))


def seed_plane_from_profile(profile: Optional[VDisparityProfile], geom: CameraGeometry) -> tuple[float, float, float]:
    """Body-frame plane (a, b, c) implied by the v-disparity line (no roll assumed).

    Falls back to the nominal plane z = 0 when the profile is missing or implausible.
    """
    if profile is None:
        return 0.0, 0.0, 0.0
    a_l, b_l = profile.line
    # d = (fxB/h) (n_y (v - cy)/fy + n_z)  for camera-frame plane n.P = h, n_x = 0.
    ny_h = a_l * geom.fy / geom.fxb
    nz_h = (a_l * geom.cy + b_l) / geom.fxb
    inv_h = math.hypot(ny_h, nz_h)
    if inv_h <= 0:
        return 0.0, 0.0, 0.0
    hgt = 1.0 / inv_h
    n_cam = np.array([0.0, ny_h * hgt, nz_h * hgt])  # unit normal pointing from camera to ground
    n_body = geom.R_bc @ n_cam  # points "down" in the body frame
    if n_body[2] >= -0.5:
        return 0.0, 0.0, 0.0
    # Plane: n_body . (p - t) = hgt  ->  z = a x + b y + c
    nx, ny, nz = n_body
    a = -nx / nz
    b = -ny / nz
    c = (hgt + n_body @ geom.t_bc) / nz
    if abs(c) > FIRST_BAND_MAX_OFFSET_M or _tilt_deg(a, b) > BAND_MAX_TILT_DEG:
        return 0.0, 0.0, 0.0
    return float(a), float(b), float(c)


def fit_ground(pts: PointSet, disp: np.ndarray, geom: CameraGeometry, seed: int = 0) -> GroundModel:
    """Fit the v-disparity profile and the continuity-checked band planes for one frame.

    pts: back-projected (strided) valid points, body frame. disp: full disparity (px).
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    profile = fit_v_disparity(disp, geom, rng)
    sa, sb, sc = seed_plane_from_profile(profile, geom)
    # Ground candidates: agree with the row profile where it was measured (rows outside its
    # inlier span are not judged by it). Height gating happens per band (see below), the
    # nearest band being gated by the seed plane implied by the profile.
    cand = np.ones(len(pts), dtype=bool)
    if profile is not None:
        de = profile.expected(pts.v)
        in_span = (pts.v >= profile.v_min) & (pts.v <= profile.v_max)
        cand &= ~in_span | (np.abs(pts.d - de) < np.maximum(CANDIDATE_DISP_TOL_PX, CANDIDATE_DISP_TOL_REL * de))
    x, y, z = pts.x[cand].astype(np.float64), pts.y[cand].astype(np.float64), pts.z[cand].astype(np.float64)

    edges = np.arange(0.0, BAND_X_MAX_M + 1e-9, BAND_WIDTH_M)
    bands: list[BandPlane] = []
    prev = BandPlane(-BAND_WIDTH_M, 0.0, sa, sb, sc, fitted=False)
    for x0, x1 in zip(edges[:-1], edges[1:]):
        band = BandPlane(float(x0), float(x1), prev.a, prev.b, prev.c)
        m = (x >= x0) & (x < x1)
        # Far bands: only points near the extrapolated nearer plane can be connected ground.
        m &= np.abs(z - prev.z(x, y)) < BAND_CANDIDATE_Z_TOL_M
        band.n_points = int(m.sum())
        if band.n_points > BAND_MAX_POINTS:
            idx = np.nonzero(m)[0]
            m = np.zeros_like(m)
            m[idx[:: int(np.ceil(idx.size / BAND_MAX_POINTS))]] = True
        if band.n_points >= BAND_MIN_POINTS:
            tol = BAND_TOL_M + BAND_TOL_PER_M * x[m]
            fit = _ransac_plane(x[m], y[m], z[m], tol, rng, prior_ab=(prev.a, prev.b))
            if fit is not None:
                a, b, c, inl = fit
                cand_plane = BandPlane(float(x0), float(x1), a, b, c, band.n_points, int(inl.sum()), True)
                first = not any(bb.fitted for bb in bands)
                if int(inl.sum()) >= BAND_MIN_POINTS and _continuous(prev, cand_plane, float(x0), first=first):
                    band = cand_plane
        bands.append(band)
        prev = band
    model = GroundModel(bands=bands, vdisp=profile)
    LOG.debug("ground: %d/%d bands fitted, horizon row %.1f", model.n_fitted, len(bands),
              profile.horizon_row if profile else float("nan"))
    return model


def _continuous(prev: BandPlane, cur: BandPlane, x_edge: float, first: bool) -> bool:
    """Accept ``cur`` only if it continues ``prev`` without a step or a sharp kink."""
    ys = np.array([-1.5, 0.0, 1.5])
    if first:
        # Nearest fitted band: the vehicle stands on the ground, so z(0, 0) must be ~0.
        return abs(cur.c) < FIRST_BAND_MAX_OFFSET_M and float(_tilt_deg(cur.a, cur.b)) < BAND_MAX_TILT_DEG
    dz = np.max(np.abs(cur.z(x_edge, ys) - prev.z(x_edge, ys)))
    ang = math.degrees(math.acos(float(np.clip(prev.normal() @ cur.normal(), -1.0, 1.0))))
    return bool(dz < BAND_CONTINUITY_M and ang < BAND_MAX_TILT_CHANGE_DEG)
