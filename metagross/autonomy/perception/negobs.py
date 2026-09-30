"""Missing-ground detector: ditches, crest shadows and occlusion shadows from stereo.

A ditch is *missing ground*, not an object. Seen from a camera at height H, a ditch of
width w at range R subtends only ``H*w / (R*(R+w))`` rad of elevation (Matthies & Rankin
2003), so at useful ranges it is a handful of image rows in which the ground "should"
be visible but is not: the measured disparity **drops** (the ray passes over the near
lip and lands on the far wall, farther away) or becomes invalid.

Per image column band (every ``col_stride`` columns, disparity averaged over the band),
walking from the bottom of the image up, each sample is labelled against the expected
ground disparity ``d_g(v)`` of the continuity-checked ground model
(:mod:`metagross.autonomy.perception.ground`):

* ``GROUND``  -tau(v) <= d - d_g <= tau_a(v); tau from a height tolerance (grows with d_g);
* ``ABOVE``   d - d_g > tau_a(v): closer than the ground -> something stands on it;
* ``MISSING`` d - d_g < -tau(v) (a range jump: below the ground surface) or invalid;
* ``OUT``     the ground model is beyond the range cap / above the horizon.

Runs of labels are then analysed, vectorised over all columns:

* A ``MISSING`` run right after >= ``N_LIP_ROWS`` of ``GROUND`` is a *gap* if the ground
  model says it should have covered >= ``MIN_GAP_ROWS`` image rows. For a range jump the
  rows are counted up to where the ground model catches up with the farthest measured
  disparity, i.e. the size of the **discontinuity**, not only the flagged rows (which
  SGBM's smoothing erodes).
* Gap classification. If ground reappears after the gap, it is a ``DITCH_CANDIDATE`` only when
  every noise-aware test passes (sigma_d = per-frame band noise estimated from the data,
  :func:`band_noise_px`; depth noise sigma_Z = Z^2 sigma_d / (fx B)):
  - the measured gap points lie below the lip->reappearance chord by more than
    ``K_CHORD_SIGMA`` sigmas of their median height (and ``CHORD_DROP_M``): the far wall;
  - the range step from the lip to the far wall, measured against the lip's OWN residual
    (immune to ground-model bias on rolling terrain), exceeds ``K_STEP_SIGMA`` sigmas and its
    depth equivalent (~ the ditch width) reaches ``MIN_DITCH_WIDTH_M``;
  - the far wall is a disparity plateau and the relative range jump >= ``JUMP_REL_MIN``;
  - >= ``LAT_MIN_SUPPORT`` neighbouring column bands hold a ditch gap at a similar lip range.
  Gaps without any valid disparity stay UNSEEN (unknown) unless ``NO_POINT_GAPS_ARE_DITCH``.
  A 3x3 median on the disparity removes 2x2 gross mismatches first. In the BEV, clusters too
  narrow across the line of sight are dropped (:func:`filter_ditch_cells`);
  - the reappearing ground is clearly below the lip, or the gap points are below the
    chord but spread along the ground (a foreshortened slope falling away, barely
    sampled) -> ``CREST_SHADOW``;
  - otherwise the "gap" is foreshortened but continuous terrain (points on or above the
    chord, e.g. the back of a gentle hump) and nothing is flagged.
  If ground does not reappear (drop-off, far side beyond range) -> ``CREST_SHADOW``.
* A gap right behind ``ABOVE`` pixels is an occlusion shadow -> ``OCCLUDED`` (never
  lethal), from the occluder's base to where ground is seen again.
* ``GROUND`` runs certify the ground between consecutive rows (seen-ground fill).

Results are ground-track segments along each column's ray (body frame, metres) for BEV
rasterisation, plus an image-space mask of gap pixels for overlays.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.autonomy.perception.ground import GroundModel
from metagross.config import defaults

LOG = logging.getLogger(__name__)

COL_STRIDE = 4  # analyse every 4th column; disparity is averaged over the 4-px band
ROW_SUB = 4  # the ground model is evaluated every 4th row and interpolated (d_g ~ linear in v)
TAU_MIN_PX = 0.3  # ground-consistency floor (px): ~3 sigma of band-averaged SGBM noise
GROUND_TOL_M = 0.04  # height tolerance converted to disparity: tau = d_g * tol / H_cam
ABOVE_TOL_M = 0.08  # samples higher than this above the ground are occluders
N_LIP_ROWS = 3  # observed ground rows needed before a gap (the lip)
N_LIP_AFTER_OBJECT_ROWS = 6  # ... or this many when an occluder is just below the lip
N_REAPPEAR_ROWS = 2  # ground rows needed after a gap to call it "reappeared"
MIN_FLAGGED_ROWS = 2  # measured MISSING rows needed (robust to single-row noise)
MIN_GAP_ROWS = 3  # expected ground rows the gap must span (>= 3 rows)
DZ_DITCH_MAX_M = 0.3  # reappearance more than this below the lip -> crest shadow
CHORD_DROP_M = 0.015  # gap points this far below the lip-reappearance chord -> trench wall
JUMP_REL_MIN = 0.06  # far side of a gap must be >= 6 % farther than the lip (d_lip / d_far - 1)
PLATEAU_RATIO = 0.4  # ... and on a disparity plateau: over 3 rows d falls < 40 % of the model's fall
LIP_RANGE_MARGIN_M = 0.5  # gaps whose lip is this close to the range cap are not classified
MIN_BAND_VALID = 2  # valid pixels needed in a column band for an averaged disparity
INVALID_RESIDUAL_PX = -1.0e3  # residual assigned to invalid samples (acts as "missing")
TOP_MARGIN_ROWS = 5
# ---- noise-aware ditch evidence (range step lip -> far wall, measured against the lip itself)
SIGMA_D_PX = 0.25  # fallback disparity noise of one band sample (px); = eval.theory.SIGMA_D_PX
SIGMA_D_MIN_PX = 0.08  # the per-frame noise estimate (band_noise_px) is clipped to this range (px)
SIGMA_D_MAX_PX = 0.5
NOISE_LAG_ROWS = 3  # row lag of the residual differences used for the noise estimate
NOISE_MIN_SAMPLES = 200  # fewer GROUND pairs than this -> fallback SIGMA_D_PX
MEDIAN_EFF = 1.2533  # std of a median of n Gaussian samples ~ 1.2533 sigma / sqrt(n)
K_STEP_SIGMA = 3.0  # the lip -> far-side step must exceed this many sigmas of its own noise (DEV-tuned, tier0)
K_CHORD_SIGMA = 2.0  # gap points must lie this many sigmas (of their median height) below the chord (DEV-tuned)
K_TAU_SIGMA = 2.0  # MISSING needs a residual below -K_TAU_SIGMA x the per-frame band noise (px) as well as -tau
LAT_HALF_COLS = 3  # lateral consistency: neighbouring column bands examined on each side
LAT_MIN_SUPPORT = 2  # ... of which at least this many must hold a ditch gap at a similar lip range
LAT_BIN_M = 0.25  # lip-range bin (m) for the lateral consistency test (+-1 bin tolerance)
N_STEP_ROWS = 5  # valid gap rows (from the lip upwards) searched for the far-side plateau
STEP_PAIR_ROWS = 2  # the far side is the lowest mean of this many consecutive gap rows
MIN_DITCH_WIDTH_M = (2.0 / 3.0) * defaults.DESIGN_DITCH_WIDTH_M  # range step (~ ditch width, m) below which a gap is no ditch (0.2 m)
NO_POINT_GAPS_ARE_DITCH = False  # gaps without any valid disparity stay UNSEEN (unknown, never free, never certified)
MIN_DITCH_LATERAL_M = 0.4  # a ditch cluster must span this much across the line of sight (m) - see filter_ditch_cells
DITCH_EXTENT_K = 1.5  # a ditch segment spans at most this x the step-based width estimate (which reads ~20 % low)
OUTLIER_MEDIAN_KSIZE = 3  # median filter (px) before the column walk: a 2x2 outlier blob is 4 of 9 px -> removed

LAB_OUT, LAB_GROUND, LAB_ABOVE, LAB_MISSING = 0, 1, 2, 3


@dataclass
class Segments:
    """Ground-track segments: column-band index, body-x start/end (m)."""

    col: np.ndarray
    x0: np.ndarray
    x1: np.ndarray

    @staticmethod
    def empty() -> "Segments":
        z = np.zeros(0)
        return Segments(np.zeros(0, np.int64), z, z)

    def __len__(self) -> int:
        return int(self.col.size)


@dataclass
class NegObsResult:
    """Output of :meth:`MissingGroundDetector.detect` (see module docstring)."""

    ground: Segments
    ditch: Segments
    crest: Segments
    occluded: Segments
    image_mask: np.ndarray  # (H, W) bool: ditch / crest gap pixels
    track_y0: np.ndarray  # (C,) ground track of each column band: y = y0 + ky * x (body, m)
    track_ky: np.ndarray  # (C,)
    track_half_w: np.ndarray  # (C,) half band width per metre of range (m/m)
    labels: np.ndarray  # (R, C) uint8 bottom-up labels (debug / tests)
    rows: np.ndarray  # (R,) image row of each label row
    col_u: np.ndarray  # (C,) centre pixel column of each band
    gap_debug: Optional[dict] = None  # per-candidate-gap diagnostics (arrays), for tests / tuning
    sigma_d_px: float = SIGMA_D_PX  # per-frame noise of one band-averaged disparity sample (px, band_noise_px)


def _nanmedian0(a: np.ndarray) -> np.ndarray:
    """Median over axis 0 ignoring NaN (NaN where a column has no finite value). Sort based:
    ~10x faster than ``np.nanmedian`` (masked-array path) for the short gap / lip windows."""
    a = np.asarray(a, dtype=np.float64)
    srt = np.sort(a, axis=0)  # NaN sort last
    n = np.isfinite(a).sum(0)
    cols = np.arange(a.shape[1])
    lo = np.clip((n - 1) // 2, 0, a.shape[0] - 1)
    hi = np.clip(n // 2, 0, a.shape[0] - 1)
    return np.where(n > 0, 0.5 * (srt[lo, cols] + srt[hi, cols]), np.nan)


def _med3(a: np.ndarray) -> np.ndarray:
    """Median of 3 along axis 0 (edges keep their value)."""
    out = a.copy()
    x, y, z = a[:-2], a[1:-1], a[2:]
    out[1:-1] = np.maximum(np.minimum(x, y), np.minimum(np.maximum(x, y), z))
    return out


def filter_ditch_cells(ditch: np.ndarray, X: np.ndarray, Y: np.ndarray, cam_xy: tuple[float, float],
                       min_lateral_m: Optional[float] = None) -> np.ndarray:
    """Keep only ditch-candidate cell clusters that extend across the line of sight.

    ``ditch``: (nx, ny) bool BEV grid; ``X``, ``Y``: cell centres (body frame, m); ``cam_xy``:
    camera position (body, m). Clusters are 8-connected components of the grid dilated by one
    cell (bridges the column bands a noisy frame misses); a cluster is kept when its angular
    extent seen from the camera times its mean range (the lateral arc, m) reaches
    ``min_lateral_m`` (default ``MIN_DITCH_LATERAL_M``). A trench crossing the path is wide
    across the line of sight; stereo mismatches rasterise into narrow radial streaks.
    """
    min_lat = MIN_DITCH_LATERAL_M if min_lateral_m is None else float(min_lateral_m)
    if not ditch.any():
        return ditch
    grown = cv2.dilate(ditch.astype(np.uint8), np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(grown, connectivity=8)
    li = lab[ditch]
    az = np.arctan2(Y[ditch] - cam_xy[1], X[ditch] - cam_xy[0])
    rr = np.hypot(X[ditch] - cam_xy[0], Y[ditch] - cam_xy[1])
    amin = np.full(n, np.inf)
    amax = np.full(n, -np.inf)
    np.minimum.at(amin, li, az)
    np.maximum.at(amax, li, az)
    cnt = np.bincount(li, minlength=n)
    rmean = np.bincount(li, weights=rr, minlength=n) / np.maximum(cnt, 1)
    span = np.where(cnt > 0, amax - amin, 0.0)
    ok = span * rmean >= min_lat
    ok[0] = False  # background label of the dilated grid
    out = np.zeros_like(ditch)
    out[ditch] = ok[li]
    return out


def band_noise_px(raw_res: np.ndarray, lab: np.ndarray, lag: int = NOISE_LAG_ROWS) -> float:
    """Per-frame noise (px) of one band-averaged disparity sample, estimated from the data:
    robust (MAD) spread of residual differences ``lag`` rows apart on GROUND-labelled rows,
    divided by sqrt(2). A lag of several rows includes SGBM's row-correlated noise.
    ``raw_res``: (R, C) residual d - d_ground (px, NaN invalid); ``lab``: (R, C) labels.
    Clipped to [SIGMA_D_MIN_PX, SIGMA_D_MAX_PX]; SIGMA_D_PX when there is too little ground."""
    ok = (lab[lag:] == LAB_GROUND) & (lab[:-lag] == LAB_GROUND)
    dd = (raw_res[lag:] - raw_res[:-lag])[ok]
    dd = dd[np.isfinite(dd)]
    if dd.size < NOISE_MIN_SAMPLES:
        return SIGMA_D_PX
    mad = float(np.median(np.abs(dd - np.median(dd))))
    return float(np.clip(1.4826 * mad / math.sqrt(2.0), SIGMA_D_MIN_PX, SIGMA_D_MAX_PX))


def _lateral_support(col: np.ndarray, x_lip: np.ndarray, sel: np.ndarray, n_cols: int) -> np.ndarray:
    """For every gap: number of OTHER column bands within +-LAT_HALF_COLS that hold a selected
    gap whose lip range is within +-1 bin (LAT_BIN_M) of its own. Vectorised; (G,) int."""
    out = np.zeros(col.size, dtype=np.int64)
    if not np.any(sel):
        return out
    nb = int(np.ceil(max(float(np.max(x_lip[sel])), 0.0) / LAT_BIN_M)) + 3
    b = np.clip(np.floor(x_lip / LAT_BIN_M).astype(np.int64) + 1, 0, nb - 2)
    occ = np.zeros((n_cols + 2 * LAT_HALF_COLS, nb), dtype=np.int32)
    occ[col[sel] + LAT_HALF_COLS, b[sel]] = 1
    occ_d = occ.copy()
    occ_d[:, 1:] |= occ[:, :-1]
    occ_d[:, :-1] |= occ[:, 1:]
    cs = np.cumsum(np.vstack([np.zeros((1, nb), np.int32), occ_d]), axis=0)
    w = 2 * LAT_HALF_COLS + 1
    win = cs[w:] - cs[:-w]  # (n_cols, nb): columns c-LAT_HALF .. c+LAT_HALF
    out = win[col, b] - occ_d[col + LAT_HALF_COLS, b]
    return out


class MissingGroundDetector:
    """Column-walk missing-ground detector bound to one camera geometry."""

    def __init__(self, geom: CameraGeometry, col_stride: int = COL_STRIDE) -> None:
        self.geom = geom
        self.stride = int(col_stride)
        self.C = geom.width // self.stride
        self.u0 = np.arange(self.C) * self.stride
        self.uc = self.u0 + self.stride // 2
        # Band-centre rays, rows bottom-up (index 0 = last image row), contiguous float64.
        r = geom.rays_body[::-1][:, self.uc, :].astype(np.float64)
        self.rx, self.ry, self.rz = (np.ascontiguousarray(r[..., i]) for i in range(3))

    # ------------------------------------------------------------------ helpers
    def _first_row(self, model: GroundModel) -> int:
        """Highest image row that can still see ground within the range cap."""
        geom = self.geom
        xr = geom.max_range_m + geom.t_bc[0]
        z_far = max(0.0, float(np.max(model.height(np.full(3, xr), np.array([-3.0, 0.0, 3.0]))))) + 0.5
        _, v, zc = geom.project_body(np.array([[xr, 0.0, z_far]]))
        if not np.isfinite(v[0]) or zc[0] <= 0:
            return 0
        return int(np.clip(np.floor(v[0]) - TOP_MARGIN_ROWS, 0, geom.height - 1))

    def _expected_ground(self, model: GroundModel, R: int) -> np.ndarray:
        """Expected ground disparity (R, C) px, 0 where not evaluable; model on every
        ``ROW_SUB``-th row, linearly interpolated in between."""
        geom = self.geom
        rs = np.arange(0, R + ROW_SUB, ROW_SUB)
        rs = np.minimum(rs, R - 1)
        rays = np.stack([self.rx[rs], self.ry[rs], self.rz[rs]], axis=-1)
        zg = model.expected_depth(geom, None, None, rays=rays)
        ok = np.isfinite(zg) & (zg <= geom.max_range_m)
        dg_c = np.where(ok, geom.fxb / np.where(ok, zg, 1.0), 0.0)
        r = np.arange(R)
        i0 = r // ROW_SUB
        t = ((r - rs[i0]) / np.maximum(rs[i0 + 1] - rs[i0], 1))[:, None]
        a, b = dg_c[i0], dg_c[i0 + 1]
        dg = a + t * (b - a)
        dg[(a <= 0) | (b <= 0)] = 0.0
        return dg

    # ------------------------------------------------------------------ main
    def detect(self, disp: np.ndarray, model: GroundModel, use_negobs: bool = True) -> NegObsResult:
        """Run the column walk on a disparity image (px, <= 0 invalid)."""
        geom, S, C = self.geom, self.stride, self.C
        h, w = disp.shape
        tx, ty, tz = geom.t_bc
        v_top = self._first_row(model)
        R = h - v_top
        rows = np.arange(h - 1, v_top - 1, -1)  # bottom-up image rows
        # A 3x3 median removes isolated gross mismatches (a Tier-0 / SGBM outlier covers up to
        # 2x2 px, i.e. 2 rows, which survives the 3-row label median and read as a range step).
        crop = np.ascontiguousarray(disp[v_top:, : C * S], dtype=np.float32)
        if OUTLIER_MEDIAN_KSIZE > 1:
            crop = cv2.medianBlur(crop, OUTLIER_MEDIAN_KSIZE)
        blk = crop[::-1].reshape(R, C, S)
        valid = blk > 0
        nv = valid.sum(2)
        Dm = np.where(nv >= MIN_BAND_VALID, np.where(valid, blk, 0.0).sum(2) / np.maximum(nv, 1), -1.0)
        rx, ry = self.rx[:R], self.ry[:R]

        DG = self._expected_ground(model, R)
        evaluable = DG >= geom.min_disparity_px
        Zq = np.where(evaluable, geom.fxb / np.where(evaluable, DG, 1.0), 0.0)
        XG = tx + Zq * rx
        YG = ty + Zq * ry

        h_cam = max(float(tz - model.height(np.array(tx), np.array(ty))), 0.3)
        tau = np.maximum(TAU_MIN_PX, DG * (GROUND_TOL_M / h_cam))
        tau_a = np.maximum(tau, DG * (ABOVE_TOL_M / h_cam))
        meas = Dm > 0
        res = _med3(np.where(meas, Dm - DG, INVALID_RESIDUAL_PX))
        lab = np.full((R, C), LAB_GROUND, dtype=np.uint8)
        lab[res > tau_a] = LAB_ABOVE
        lab[res < -tau] = LAB_MISSING
        lab[~evaluable] = LAB_OUT
        sig_d = band_noise_px(np.where(meas, Dm - DG, np.nan), lab)
        # Noise-aware MISSING label: a residual must also fall below -K_TAU_SIGMA sigma of this
        # frame's measured band noise. SGBM pixel-locking on rendered stereo leaves a ~2 px
        # sawtooth on smooth ground whose lower teeth otherwise read as missing ground (and
        # their plateaus as far walls); on Tier-0 (sigma ~0.1 px) the TAU_MIN_PX floor rules.
        tau_n = K_TAU_SIGMA * sig_d
        if tau_n > TAU_MIN_PX:
            relax = (lab == LAB_MISSING) & (res >= -np.maximum(tau, tau_n))
            lab[relax] = LAB_GROUND

        # -------------------------------------------------------------- runs
        T = lab.T  # (C, R)
        start = np.ones_like(T, dtype=bool)
        start[:, 1:] = T[:, 1:] != T[:, :-1]
        ci, ri = np.nonzero(start)
        rl = T[ci, ri]
        n_runs = ci.size
        same_next = np.r_[ci[1:] == ci[:-1], False]
        end = np.where(same_next, np.r_[ri[1:], 0] - 1, R - 1)
        length = end - ri + 1

        def shifted(a: np.ndarray, k: int, fill: int) -> np.ndarray:
            """Value of run (i + k) when it is in the same column, else ``fill``."""
            j = np.arange(n_runs) + k
            jj = np.clip(j, 0, max(n_runs - 1, 0))
            ok = (j >= 0) & (j < n_runs) & (ci[jj] == ci)
            return np.where(ok, a[jj], fill)

        cols = np.arange(C)
        last_eval = evaluable.sum(0) - 1  # bottom-up index of the last evaluable row (-1: none)
        le = np.clip(last_eval, 0, R - 1)
        col_end_x = XG[le, cols]

        gk = np.nonzero(rl == LAB_GROUND)[0]
        ground = Segments(ci[gk], XG[ri[gk], ci[gk]], XG[end[gk], ci[gk]])

        # Per-column straight ground tracks y = y0 + ky x (first and last evaluable rows).
        xa, ya, xb, yb = XG[0], YG[0], XG[le, cols], YG[le, cols]
        dx = np.where(np.abs(xb - xa) > 1e-6, xb - xa, 1.0)
        ky = np.where(le > 0, (yb - ya) / dx, 0.0)
        y0 = ya - ky * xa
        half_w = np.full(C, 0.5 * S / geom.fx)

        image_mask = np.zeros((h, w), dtype=bool)
        empty = Segments.empty()
        if not use_negobs or n_runs == 0:
            return NegObsResult(ground, empty, empty, empty, image_mask, y0, ky, half_w, lab, rows, self.uc,
                                sigma_d_px=sig_d)

        prev_lab, prev_len, prev2_lab = shifted(rl, -1, 255), shifted(length, -1, 0), shifted(rl, -2, 255)
        next_lab, next_len, next_start = shifted(rl, 1, 255), shifted(length, 1, 0), shifted(ri, 1, 0)

        # -------------------------------------------------------------- gaps
        obj_below = (prev2_lab == LAB_ABOVE) & (prev_len < N_LIP_AFTER_OBJECT_ROWS)
        cand = (rl == LAB_MISSING) & (prev_lab == LAB_GROUND) & (prev_len >= N_LIP_ROWS) & ~obj_below
        cand &= length >= MIN_FLAGGED_ROWS
        k = np.nonzero(cand)[0]
        ditch, crest = empty, empty
        gap_debug: Optional[dict] = None
        if k.size:
            c, s, e = ci[k], ri[k], end[k]
            lip = s - 1
            rz_all = self.rz[:R]

            def gather(start_row: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                """Measured (d, x, z, valid) on rows start_row + [0, n) of the gap columns, (n, G)."""
                idx = np.clip(start_row[None, :] + np.arange(n)[:, None], 0, R - 1)
                d = Dm[idx, c[None, :]]
                ok = d > 0
                zc = np.where(ok, geom.fxb / np.where(ok, d, 1.0), np.nan)
                return d, tx + zc * rx[idx, c[None, :]], tz + zc * rz_all[idx, c[None, :]], ok

            # Gap rows, gathered compactly: (L, G) with L the longest gap.
            L = int(length[k].max())
            dk_, xm, zm, okm = gather(s, L)
            vk = okm & (np.arange(L)[:, None] < length[k][None, :])
            xm = np.where(vk, xm, np.nan)
            zm = np.where(vk, zm, np.nan)
            has_pts = vk.any(0)
            d_far = np.where(vk, dk_, np.inf).min(0)  # farthest measured disparity in the gap (px)
            rr = np.arange(R)[:, None]
            DGk = DG[:, c]
            catch_up = ((DGk >= d_far[None, :]) & (rr > lip[None, :]) & (DGk > 0)).sum(0)
            n_exp = np.where(np.isfinite(d_far), np.maximum(catch_up, length[k]), length[k])
            reap = (next_lab[k] == LAB_GROUND) & ((next_len[k] >= N_REAPPEAR_ROWS) | (next_start[k] + next_len[k] - 1 >= last_eval[c]))
            rs = np.clip(np.where(reap, next_start[k], e), 0, R - 1)
            x_l, y_l = XG[lip, c], YG[lip, c]
            x_r, y_r = XG[rs, c], YG[rs, c]
            dz_rep = model.height(x_r, y_r) - model.height(x_l, y_l)  # reappearance vs lip (model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _, xl_all, zl_all, _ = gather(lip - (N_LIP_ROWS - 1), N_LIP_ROWS)
                _, xr_all, zr_all, _ = gather(rs, N_REAPPEAR_ROWS)
                xl_m, zl_m = _nanmedian0(xl_all), _nanmedian0(zl_all)
                dl_all, _, _, _ = gather(lip - (N_LIP_ROWS - 1), N_LIP_ROWS)
                d_lip = _nanmedian0(np.where(dl_all > 0, dl_all, np.nan))
                xr_m, zr_m = _nanmedian0(xr_all), _nanmedian0(zr_all)
                span = np.where(np.abs(xr_m - xl_m) > 1e-3, xr_m - xl_m, 1e-3)
                chord = zl_m[None, :] + (zr_m - zl_m)[None, :] * (xm - xl_m[None, :]) / span[None, :]
                dzc = _nanmedian0(zm - chord)  # gap points vs lip->reappearance chord (m)
            dzc = np.where(np.isfinite(dzc), dzc, 0.0)
            # Noise-aware chord test: a gap point's height error along a grazing ray is
            # sigma_Z * h/R = R h sigma_d / (fx B); the median of n points shrinks it by sqrt(n).
            n_pts = vk.sum(0)
            r_lip = np.maximum(XG[lip, c] - tx, 0.1)
            sig_h = MEDIAN_EFF * r_lip * h_cam * sig_d / geom.fxb / np.sqrt(np.maximum(n_pts, 1))
            below = has_pts & (dzc < -np.maximum(CHORD_DROP_M, K_CHORD_SIGMA * sig_h))
            # Range step lip -> far wall measured against the lip's OWN residual (removes any
            # ground-model bias, e.g. rolling terrain between band planes): median residual of
            # the lip rows minus the far-side residual (below) in px, which must clear
            # K_STEP_SIGMA of its noise, and whose depth equivalent (~ the ditch width,
            # sigma_Z = Z^2 sigma_d / (fx B)) must reach MIN_DITCH_WIDTH_M.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                raw_l = np.where(dl_all > 0, dl_all - DG[np.clip(lip - (N_LIP_ROWS - 1), 0, R - 1)[None, :] + np.arange(N_LIP_ROWS)[:, None], c[None, :]], np.nan)
                res_lip = _nanmedian0(raw_l)
                n_lip = np.isfinite(raw_l).sum(0)
                # Far side = lowest mean of STEP_PAIR_ROWS consecutive valid residuals among the
                # first N_STEP_ROWS valid gap rows: SGBM smears the lip -> wall transition over
                # 1-2 rows, the wall plateau follows.
                first = vk & (np.cumsum(vk, 0) <= N_STEP_ROWS)
                dg_rows = DG[np.clip(s[None, :] + np.arange(L)[:, None], 0, R - 1), c[None, :]]
                rg = np.where(first, dk_ - dg_rows, np.nan)
                dg_m = np.where(first, dk_, np.nan)
                if L >= STEP_PAIR_ROWS:
                    pair = 0.5 * (rg[:-1] + rg[1:])
                    j = np.argmin(np.where(np.isfinite(pair), pair, np.inf), axis=0)
                    ok_pair = np.isfinite(pair[j, np.arange(k.size)])
                    res_gap = np.where(ok_pair, pair[j, np.arange(k.size)], _nanmedian0(rg))
                    d_gap = np.where(ok_pair, 0.5 * (dg_m[j, np.arange(k.size)] + dg_m[np.minimum(j + 1, L - 1), np.arange(k.size)]),
                                     _nanmedian0(dg_m))
                    n_gap = np.where(ok_pair, STEP_PAIR_ROWS, first.sum(0))
                else:
                    res_gap, d_gap, n_gap = _nanmedian0(rg), _nanmedian0(dg_m), first.sum(0)
            step_px = np.where(np.isfinite(res_lip) & np.isfinite(res_gap), res_lip - res_gap, 0.0)
            sig_step = sig_d * np.sqrt(MEDIAN_EFF**2 / np.maximum(n_lip, 1) + 1.0 / np.maximum(n_gap, 1))
            significant = has_pts & (step_px >= K_STEP_SIGMA * sig_step)
            d_gap = np.where(np.isfinite(d_gap) & (d_gap > 0), d_gap, 1.0)
            width_est = geom.fxb / d_gap - geom.fxb / (d_gap + np.maximum(step_px, 0.0))
            wide = width_est >= MIN_DITCH_WIDTH_M
            # A trench's far wall is a (near-)vertical surface at one range: somewhere in the gap
            # the measured disparity stays on a plateau while the ground model keeps falling.
            # Tested on 3-row windows (2-row differences); gaps with < 3 valid rows pass.
            idx = np.clip(s[None, :] + np.arange(L)[:, None], 0, R - 1)
            dgk = DG[idx, c[None, :]]
            win = vk[:-2] & vk[1:-1] & vk[2:] if L >= 3 else np.zeros((0, k.size), bool)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (dk_[:-2] - dk_[2:]) / (dgk[:-2] - dgk[2:]) if L >= 3 else np.zeros((0, k.size))
            ratio = np.where(win & np.isfinite(ratio), ratio, np.inf)
            plateau = (ratio.min(0) < PLATEAU_RATIO) if L >= 3 else np.zeros(k.size, bool)
            plateau |= ~win.any(0)
            # The range jump must be a meaningful fraction of the range (w / R >~ 6 %): SGBM's
            # fronto-parallel "staircase" errors on steeply slanted near ground are ~1-2 px, i.e.
            # a few % of the near-range disparity, and must not read as trenches.
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_jump = np.where(np.isfinite(d_far) & (d_far > 0), d_lip / d_far - 1.0, np.inf)
            big_jump = ~(rel_jump < JUMP_REL_MIN)
            is_ditch = reap & below & plateau & big_jump & significant & wide
            if NO_POINT_GAPS_ARE_DITCH:
                is_ditch |= reap & ~has_pts & (dz_rep > -DZ_DITCH_MAX_M)
            is_crest = ~is_ditch & big_jump & (~reap | (dz_rep <= -DZ_DITCH_MAX_M) | below)
            keep = (n_exp >= MIN_GAP_ROWS) & (x_l < geom.max_range_m - LIP_RANGE_MARGIN_M)

            x_lip = 0.5 * (x_l + XG[np.minimum(lip + 1, R - 1), c])
            x_catch = XG[np.clip(lip + n_exp, 0, R - 1), c]
            x_rep = 0.5 * (x_r + XG[np.maximum(rs - 1, 0), c])
            x_after = np.where(next_lab[k] == LAB_OUT, col_end_x[c], XG[e, c])
            # A ditch segment ends at the far lip; one far outlier in the gap must not stretch
            # it into a long radial streak: cap its length at DITCH_EXTENT_K x the width estimate.
            x_ditch_end = np.minimum(np.maximum(x_rep, x_catch),
                                     x_lip + np.maximum(DITCH_EXTENT_K * width_est, MIN_DITCH_WIDTH_M))
            x_end = np.where(is_ditch, x_ditch_end, np.where(reap, x_rep, x_after))
            keep &= x_end > x_lip
            dk, ck = keep & is_ditch, keep & is_crest
            # A ditch that matters crosses the path: it spans several neighbouring column bands
            # at about the same range. Isolated single-band "ditches" are noise.
            dk &= _lateral_support(c, x_lip, dk, C) >= LAT_MIN_SUPPORT
            gap_debug = {"col": c, "x_lip": x_lip, "x_end": x_end, "n_exp": n_exp, "length": length[k], "reap": reap,
                         "has_pts": has_pts, "below": below, "plateau": plateau, "rel_jump": rel_jump, "dzc": dzc,
                         "dz_rep": dz_rep, "is_ditch": is_ditch, "is_crest": is_crest, "keep": keep,
                         "step_px": step_px, "sig_step": sig_step, "width_est": width_est, "sig_h": sig_h,
                         "sig_d": sig_d}
            ditch = Segments(c[dk], x_lip[dk], x_end[dk])
            crest = Segments(c[ck], x_lip[ck], x_end[ck])
            top = np.clip(lip + np.maximum(n_exp, length[k]), 0, R - 1)
            for g in np.nonzero(dk | ck)[0]:
                image_mask[rows[top[g]] : rows[min(lip[g] + 1, R - 1)] + 1, self.u0[c[g]] : self.u0[c[g]] + S] = True

        # -------------------------------------------------------------- occlusion shadows
        ak = np.nonzero(rl == LAB_ABOVE)[0]
        occluded = empty
        if ak.size:
            found = np.zeros(ak.size, dtype=bool)
            x_res = col_end_x[ci[ak]].copy()
            for hop in range(1, 5):
                j = ak + hop
                jj = np.clip(j, 0, n_runs - 1)
                ok = (j < n_runs) & ~found & (ci[jj] == ci[ak])
                hit = ok & (rl[jj] == LAB_GROUND) & (length[jj] >= N_LIP_ROWS)
                x_res = np.where(hit, XG[ri[jj], ci[jj]], x_res)
                found |= hit | (ok & (rl[jj] == LAB_OUT))
            x_base = XG[ri[ak], ci[ak]]
            okk = x_res > x_base
            occluded = Segments(ci[ak][okk], x_base[okk], x_res[okk])

        LOG.debug("negobs: %d ditch, %d crest, %d occluded segments", len(ditch), len(crest), len(occluded))
        return NegObsResult(ground, ditch, crest, occluded, image_mask, y0, ky, half_w, lab, rows, self.uc, gap_debug,
                            sigma_d_px=sig_d)
