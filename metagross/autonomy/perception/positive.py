"""Positive-obstacle, slope and roughness costs on the egocentric BEV grid.

Inputs are per-cell point statistics (:class:`~metagross.autonomy.perception.bev.BevStats`,
heights relative to the local ground model, metres) and the ground-model height at the
cell centres (body frame, metres).

* **Terrain level** (:func:`terrain_level`) - morphological open-close (``TERRAIN_WINDOW_M``)
  of denoised absolute heights: follows rolling terrain, removes objects narrower than the
  kernel, fills narrow pits.
* **Step** - a cell is ``POSITIVE`` (lethal) when >= ``MIN_POINTS_POSITIVE`` of its points
  stand more than the noise-aware step threshold (:func:`step_threshold_m`:
  ``max(STEP_LETHAL_M, K_POS_SIGMA * sigma_h(r))`` with the ground-point height noise
  ``sigma_h(r) = r H sigma_d / (fx B)`` at camera range r) above the terrain level
  (:func:`count_outside`); without those counts, its highest point is used.
* **Minimum physical size** (:func:`vertical_extent_m`) - a point may only vote for a step when
  it lies on an image surface that is at least ``MIN_OBSTACLE_EXTENT_M`` tall: its vertical run of
  row-to-row consistent disparity, converted to metres (``rows * B / d``). SGBM mismatches in the
  sky / on the far horizon (a few rows of large disparity) back-project to "obstacles" ~1 m tall
  right in front of the vehicle; a real rock or trunk is a tall run of constant disparity that
  continues into its ground contact. The pipeline drops such floating points
  (:func:`floating_points`) before binning.
* **Slope** - measured on the terrain level (open-close, objects narrower than
  ``TERRAIN_WINDOW_M`` removed, ``SLOPE_ON_LEVEL``) box-smoothed over ``SLOPE_WINDOW_M`` and
  differentiated with a 3x3 Sobel; slopes above ``SLOPE_LETHAL_DEG`` in cells with
  >= ``MIN_POINTS_POSITIVE`` points are lethal (reported through the POSITIVE state). Objects are
  the step test's job: smoothing raw heights spread every rock's flank over ~0.4 m of the
  surrounding ground as a lethal "slope" halo.
* **Roughness** - neighbourhood RMS of the per-cell height standard deviation in excess of
  the expected stereo height noise at that range.

Non-lethal terrain cost in [0, ``GROUND_COST_MAX``] combines slope and roughness.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from metagross.autonomy.perception.bev import BevSpec, BevStats
from metagross.config import defaults

SLOPE_WINDOW_M = 0.6  # smoothing window for slope (m)
MIN_POINTS_POSITIVE = 2  # points needed before a cell can be declared a lethal step
ROUGH_REF_M = 0.05  # roughness (m RMS) that saturates the roughness cost term
W_SLOPE = 0.6  # weights of the non-lethal terrain cost terms
W_ROUGH = 0.4
GROUND_COST_MAX = 0.7  # non-lethal terrain never costs more than this
TERRAIN_WINDOW_M = 1.5  # opening kernel of the local terrain level: objects narrower than this are removed
SMOOTH_CELLS = 5  # count-weighted box mean (cells) that denoises heights before min filters
STEREO_SIGMA_D_PX = 0.5  # disparity noise (px) assumed when discounting stereo noise from roughness (~2x tier-0 / SGBM sub-pixel)
# ---- noise-aware step test and minimum physical size (see module docstring)
K_POS_SIGMA = 3.0  # a step must exceed this many sigmas of the ground-point height noise at its range
SIGMA_D_POS_MIN_PX = 0.25  # floor (px) of the per-frame disparity noise used by the step threshold
MIN_OBSTACLE_EXTENT_M = 0.5 * defaults.STEP_LETHAL_M  # a step point must lie on a surface >= this tall (m)
VSUPPORT_TOL_ABS_PX = 2.5  # row-to-row disparity continuity (px): > the ~2 px SGBM pixel-locking staircase + noise
VSUPPORT_TOL_REL = 0.08  # ... or this fraction of the disparity (a vertical face keeps d constant)
SLOPE_ON_LEVEL = True  # slope from the terrain level (True) or from the smoothed raw heights (legacy)


@dataclass
class TerrainCosts:
    """Per-cell terrain analysis (all arrays shaped like the BEV grid)."""

    positive: np.ndarray  # bool: lethal step or lethal slope
    step_m: np.ndarray  # float32 step height above local ground (m), NaN where empty
    slope_deg: np.ndarray  # float32 smoothed terrain slope (deg)
    roughness_m: np.ndarray  # float32 RMS roughness (m)
    cost: np.ndarray  # float32 non-lethal terrain cost in [0, GROUND_COST_MAX]


def _odd(n: float) -> int:
    k = max(int(round(n)), 1)
    return k if k % 2 == 1 else k + 1


def smoothed_heights(stats: BevStats, k: int = SMOOTH_CELLS) -> tuple[np.ndarray, np.ndarray]:
    """Point-count-weighted ``k`` x ``k`` box mean of the cell mean heights (m, relative to the
    model) and its support mask. Far cells hold 1-3 noisy points; min filters on the raw means
    are biased low by ~2 sigma (~0.1 m at 8 m range in the tier-0 noise model), which turned
    ordinary ground into 15 cm "steps"."""
    cnt = stats.count.astype(np.float32)
    num = cv2.boxFilter(np.where(cnt > 0, stats.h_mean * cnt, 0.0).astype(np.float32), -1, (k, k), normalize=False,
                        borderType=cv2.BORDER_CONSTANT)
    den = cv2.boxFilter(cnt, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)
    sup = den > 0
    return np.where(sup, num / np.maximum(den, 1e-6), 0.0).astype(np.float32), sup


def terrain_level(stats: BevStats, spec: BevSpec, ground_z: np.ndarray, window_m: float = TERRAIN_WINDOW_M) -> np.ndarray:
    """Local terrain level relative to the ground model (m, (nx, ny) float32; 0 where unsupported).

    Morphological open-close (square ``window_m`` kernel) of the absolute cell mean heights
    ``ground_z + h_mean`` - the progressive-morphological-filter idea of Zhang et al. (2003):
    the opening removes every bump narrower than the kernel (rocks, trunks, bushes, logs), the
    closing then fills every pit narrower than it (narrow ditch bottoms), and both reproduce
    planar ramps exactly, so the level follows rolling terrain that the piecewise-planar band
    model cannot represent laterally. Cells without support fall back to the model (0)."""
    hs, sup = smoothed_heights(stats)
    k = np.ones((_odd(window_m / spec.res),) * 2, np.uint8)
    z_abs = (ground_z + np.where(sup, hs, 0.0)).astype(np.float32)
    # The min filter picks noise minima: it runs on denoised heights (see smoothed_heights).
    ero = cv2.erode(np.where(sup, z_abs, np.inf).astype(np.float32), k, borderType=cv2.BORDER_REPLICATE)
    opened = cv2.dilate(np.where(np.isfinite(ero), ero, -np.inf).astype(np.float32), k, borderType=cv2.BORDER_REPLICATE)
    dil = cv2.dilate(opened, k, borderType=cv2.BORDER_REPLICATE)
    closed = cv2.erode(np.where(np.isfinite(dil), dil, np.inf).astype(np.float32), k, borderType=cv2.BORDER_REPLICATE)
    rel = np.where(np.isfinite(opened), closed, np.nan) - ground_z
    return np.where(np.isfinite(rel), rel, 0.0).astype(np.float32)


def vertical_extent_m(disp: np.ndarray, fxb: float, fy: float, stride_v: int = 1, stride_u: int = 1,
                      row_start: int = 0) -> np.ndarray:
    """Metric height (m) of the vertical disparity run each sampled pixel belongs to.

    ``disp``: (H, W) disparity (px, <= 0 invalid). The image is sampled like
    :meth:`CameraGeometry.points_from_disparity` (rows ``row_start::stride_v``, columns
    ``::stride_u``); the result has that sampled shape. Two vertically adjacent samples are
    connected when both are valid and differ by <= ``max(VSUPPORT_TOL_ABS_PX, VSUPPORT_TOL_REL d)``.
    A run of n connected samples spans ``n * stride_v`` image rows, i.e. ``n stride_v Z_c / fy``
    metres at its depth ``Z_c = fxb / d`` (``fxb = fx B``, px m). Invalid samples get 0.
    """
    d = np.ascontiguousarray(disp[row_start::stride_v, ::stride_u], dtype=np.float32)
    rows, cols = d.shape
    if rows == 0:
        return np.zeros_like(d)
    valid = d > 0
    tol = np.maximum(VSUPPORT_TOL_ABS_PX, VSUPPORT_TOL_REL * np.minimum(d[:-1], d[1:]))
    cont = valid[:-1] & valid[1:] & (np.abs(d[1:] - d[:-1]) <= tol)
    brk = np.ones((rows, cols), dtype=np.int32)
    brk[1:] = ~cont
    lab = np.cumsum(brk, axis=0) + (np.arange(cols, dtype=np.int64) * (rows + 1))[None, :]
    run = np.bincount(lab.ravel()).astype(np.float32)[lab]
    with np.errstate(divide="ignore", invalid="ignore"):
        ext = np.where(valid, run * (stride_v * fxb / fy) / np.where(valid, d, 1.0), 0.0)
    return ext.astype(np.float32)


def floating_points(ext_samples: np.ndarray, h_rel: np.ndarray, min_extent_m: float = MIN_OBSTACLE_EXTENT_M,
                    above_m: float = defaults.STEP_LETHAL_M) -> np.ndarray:
    """Bool mask of points that stand more than ``above_m`` above the ground model (``h_rel``, m)
    yet lie on an image surface shorter than ``min_extent_m`` (``ext_samples``, m, per point from
    :func:`vertical_extent_m`): physically too small to be an obstacle, i.e. stereo mismatches."""
    return (np.asarray(h_rel) > above_m) & (np.asarray(ext_samples) < min_extent_m)


def step_threshold_m(spec: BevSpec, sigma_d_px: float, k: float = K_POS_SIGMA) -> np.ndarray:
    """Per-cell lethal step threshold (m, (nx, ny) float32): ``max(STEP_LETHAL_M, k sigma_h)``.

    ``sigma_h = r H sigma_d / (fx B)`` is the height noise of a ground point at camera range r
    (m, along body x from the camera; H camera height): a ray grazing the ground at depression
    angle ~H/r turns the depth noise ``sigma_Z = r^2 sigma_d / (fx B)`` into height noise
    ``sigma_Z H / r``. ``sigma_d_px`` is the per-frame disparity noise estimate (floored at
    ``SIGMA_D_POS_MIN_PX``)."""
    X, _ = spec.centres()
    r = np.maximum(X - defaults.CAM_FORWARD_M, 0.0)
    sig_d = max(float(sigma_d_px), SIGMA_D_POS_MIN_PX)
    sig_h = r * defaults.CAM_HEIGHT_M * sig_d / (defaults.FX * defaults.BASELINE_M)
    return np.maximum(defaults.STEP_LETHAL_M, k * sig_h).astype(np.float32)


def count_outside(spec: BevSpec, x: np.ndarray, y: np.ndarray, h: np.ndarray, level: np.ndarray,
                  above_m: float | np.ndarray, below_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell numbers of points more than ``above_m`` above / ``below_m`` below the local
    terrain ``level`` (heights relative to the model, metres). ``above_m`` is a scalar or a
    per-cell (nx, ny) threshold (:func:`step_threshold_m`). Returns two (nx, ny) int32 grids."""
    idx = spec.flat_index(x, y)
    ok = idx >= 0
    idx, hh = idx[ok], np.asarray(h, np.float32)[ok]
    rel = hh - level.ravel()[idx]
    n = spec.n_cells
    thr = np.asarray(above_m, np.float32).ravel()[idx] if np.ndim(above_m) else above_m
    up = np.bincount(idx[rel > thr], minlength=n).reshape(spec.shape).astype(np.int32)
    dn = np.bincount(idx[rel < -below_m], minlength=n).reshape(spec.shape).astype(np.int32)
    return up, dn


def terrain_costs(stats: BevStats, spec: BevSpec, ground_z: np.ndarray, observed: np.ndarray,
                  level: np.ndarray | None = None, n_above: np.ndarray | None = None) -> TerrainCosts:
    """Step / slope / roughness analysis.

    stats: point statistics. ground_z: (nx, ny) model ground height at cell centres (m).
    observed: (nx, ny) bool, cells whose surface has been seen (points or ground fill).
    level: local terrain level relative to the model (:func:`terrain_level`); computed if None.
    n_above: (nx, ny) points more than STEP_LETHAL_M above the level (:func:`count_outside`);
    None falls back to the single-point ``h_max`` test.
    """
    res = spec.res
    has = stats.count > 0
    lvl = terrain_level(stats, spec, ground_z) if level is None else level
    # ---- step: max height above the local terrain level (not above the band model, which
    # cannot follow lateral undulation: rolling terrain standing > 15 cm above a band plane
    # used to become a field of false POSITIVE cells). The level's closing fills narrow pits,
    # so the rim of a depression is not a step.
    # With per-cell counts of points above the step height (``n_above``), a step needs
    # MIN_POINTS_POSITIVE such points: a single gross stereo mismatch no longer makes a lethal
    # cell (near cells hold ~100 points, so h_max alone was an outlier detector).
    step = np.where(has, stats.h_max - lvl, np.nan).astype(np.float32)
    if n_above is not None:
        lethal_step = has & (n_above >= MIN_POINTS_POSITIVE)
    else:
        lethal_step = has & (stats.count >= MIN_POINTS_POSITIVE) & (step > defaults.STEP_LETHAL_M)

    # ---- slope on absolute, denoised heights (unsupported cells take the terrain level);
    # steps too sharp for the smoothed slope are caught by the step test above.
    if SLOPE_ON_LEVEL:
        z = (ground_z + lvl).astype(np.float32)
    else:
        hs, sup = smoothed_heights(stats)
        z = (ground_z + np.where(sup, hs, lvl)).astype(np.float32)
    k_sl = _odd(SLOPE_WINDOW_M / res)
    zs = cv2.blur(z, (k_sl, k_sl), borderType=cv2.BORDER_REPLICATE)
    gx = cv2.Sobel(zs, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE) / (8.0 * res)
    gy = cv2.Sobel(zs, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE) / (8.0 * res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    # Lethal slope needs direct evidence in the cell (>= MIN_POINTS_POSITIVE points): a 1-point
    # cell beside a tall object inherits a smoothed "slope" that is really the object's side.
    lethal_slope = observed & (stats.count >= MIN_POINTS_POSITIVE) & (slope > defaults.SLOPE_LETHAL_DEG)

    # ---- roughness
    # Roughness is the height spread in EXCESS of the stereo height noise expected at that
    # range (sigma_h ~ sigma_d * x * H_cam / (fx B)); raw spread made far, perfectly smooth
    # ground look rough, i.e. costlier than unseen cells, and the route planner turned away.
    var = np.nan_to_num(np.where(has, stats.h_std, 0.0), nan=0.0).astype(np.float32) ** 2
    X, _ = spec.centres()
    sig_n = STEREO_SIGMA_D_PX * np.maximum(X, 0.0) * defaults.CAM_HEIGHT_M / (defaults.FX * defaults.BASELINE_M)
    var_x = np.maximum(cv2.blur(var, (3, 3), borderType=cv2.BORDER_REPLICATE) - (sig_n**2).astype(np.float32), 0.0)
    rough = np.sqrt(var_x).astype(np.float32)

    c = W_SLOPE * (slope / defaults.SLOPE_LETHAL_DEG) ** 2 + W_ROUGH * (rough / ROUGH_REF_M)
    cost = (GROUND_COST_MAX * np.clip(c, 0.0, 1.0)).astype(np.float32)
    return TerrainCosts(lethal_step | lethal_slope, step, slope, rough, cost)
