"""Frame-to-frame stereo visual odometry (onboard, CPU, OpenCV).

Pipeline per stereo frame k (all image quantities in pixels of the rectified left
camera; 3-D quantities in metres in the OpenCV camera frame: x right, y down,
z forward):

1. Disparity of frame k: taken from the caller if given (perception already
   computed it, or the Tier-0 sensor supplied it), otherwise SGBM (3-way).
2. Track the persistent 2-D feature set from frame k-1 to k with pyramidal KLT
   (21x21 window), seeded with the flow predicted by the previous motion, and
   keep only tracks passing a forward-backward check (< ``fb_max_px``).
   Photometric normalisation: KLT minimises an SSD that is not gain-invariant, so
   if fewer than half of the tracks survive, a global brightness change
   ``I_k ~ g I_{k-1} + o`` is estimated (quantile matching of 5x5 patch means at
   the surviving correspondences - the same scene points, so content changes such
   as the sky fraction while pitching do not bias it; whole-image moments if too
   few survived). If it departs from identity (auto-exposure step, dimming, veiling
   glare) KLT is re-run on the k-1 image mapped into the photometric frame of k and
   the pass keeping more tracks wins. Healthy frames are never modified.
3. Back-project the k-1 end of every surviving track to 3-D with the k-1
   disparity map (Z = fx * B / d), rejecting invalid, too-small and
   depth-discontinuous disparities.
4. PnP RANSAC (AP3P minimal solver, ``ransac_px`` reprojection threshold), then
   Levenberg-Marquardt refinement (``cv2.solvePnPRefineLM``) on the inliers.
5. Motion-sanity gate (per-frame translation / rotation bounded by speed and
   yaw-rate limits times dt, minimum inlier count).
6. Track management ("keyframing"): PnP outliers are dropped; when fewer than
   ``min_tracks`` tracks remain, new Shi-Tomasi corners are detected with an
   8x5 bucketing grid (``max_corners`` total) away from existing tracks.

The output is the 6-DoF relative pose ``T_prev_cur`` (pose of the current
camera expressed in the previous camera frame, 4x4) and a statistics dict used
by the integrity monitor (:mod:`metagross.autonomy.localization.health`).
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

# Default reprojection RMSE reported when VO produced no pose (px). Chosen well
# above any healthy value (healthy KITTI frames sit near 0.3-0.6 px).
RMSE_WHEN_FAILED_PX = 5.0
COVERAGE_GRID = (4, 4)  # (rows, cols) grid used for the feature-coverage statistic
COVERAGE_MIN_PTS = 2  # inliers a coverage cell needs to count as covered
PHOTO_STD_FLOOR_DN = 2.0  # below this intensity IQR (8-bit DN) the gain is not estimable -> offset only
PHOTO_QUANTILES = (25.0, 50.0, 75.0)  # quantiles matched by the photometric gain/offset estimate


@dataclass(slots=True)
class VOConfig:
    """Tunable VO parameters. Defaults target the METAGROSS UGV camera
    (640x400, 0.12 m baseline, <= 2 m/s); :meth:`kitti` gives the KITTI preset."""

    # --- detection (Shi-Tomasi / GFTT with bucketing)
    max_corners: int = 1000
    grid_cols: int = 8
    grid_rows: int = 5
    quality_level: float = 0.01  # relative to the strongest corner in the image
    min_distance_px: int = 7
    corner_block_size: int = 3
    border_px: int = 10  # ignore corners this close to the image border
    min_tracks: int = 300  # re-detect when fewer tracks survive ("keyframing")
    # --- KLT
    # KLT window / termination. 15 px, 10 iterations, eps 0.03 px: ~30 % less VO time than the
    # previous 21 px / 30 / 0.01 on rendered DEV drives at equal or better per-frame accuracy; smaller
    # windows also reduce the near-ground perspective bias. KITTI 07 check: results/kitti_vo_07_check.json.
    klt_win: int = 15  # px
    klt_max_level: int = 3  # OpenCV maxLevel: 3 pyramid levels above full resolution
    klt_iters: int = 10
    klt_eps: float = 0.03  # px
    fb_max_px: float = 1.0  # forward-backward consistency threshold
    # --- photometric normalisation before KLT (auto-exposure / dimming / veiling glare)
    photometric_norm: bool = True
    photo_retry_good_frac: float = 0.5  # KLT is re-run on a normalised k-1 image only if fewer tracks survived
    photo_box_px: int = 5  # patch (px) averaged around each matched track for the gain/offset estimate
    photo_min_samples: int = 30  # fewer surviving tracks: estimate from whole-image moments instead
    photo_downsample: int = 4  # whole-image moment estimate on a 1/4-resolution copy
    # |g - 1| below this and |o| below the offset limit: images left untouched. KLT copes with
    # small changes; the estimate itself has a few % noise (texture scale change under motion).
    photo_min_gain_dev: float = 0.10
    photo_min_offset_dn: float = 8.0  # 8-bit DN
    photo_gain_min: float = 0.25  # clip of the estimated gain (AE range of the UGV camera is x1/8..x8
    photo_gain_max: float = 4.0  # per frame pair; larger jumps are not a photometric change)
    # --- stereo / 3-D
    num_disparities: int = 64  # SGBM search range (multiple of 16)
    sgbm_block: int = 5
    min_disp_px: float = 1.5  # ignore farther points (Z > fx*B/min_disp)
    max_disp_jump_px: float = 1.5  # reject points on depth discontinuities (3x3 range)
    # --- PnP
    ransac_px: float = 1.5
    ransac_iters: int = 200
    ransac_conf: float = 0.999
    min_inliers: int = 15
    refine_px: float = 2.0  # inlier threshold after LM refinement
    # --- motion sanity gate
    max_speed_mps: float = 4.0  # 2x the platform cap (VEHICLE.max_speed_mps = 2.0)
    max_yaw_rate_rps: float = 2.4  # 2x the platform cap
    gate_slack_m: float = 0.10
    gate_slack_rad: float = math.radians(5.0)
    default_dt_s: float = 0.2  # used when timestamps are unavailable

    @classmethod
    def kitti(cls) -> "VOConfig":
        """Preset for KITTI odometry (1241x376, 0.54 m baseline, cars up to ~25 m/s @ 10 Hz)."""
        return cls(num_disparities=128, max_speed_mps=40.0, max_yaw_rate_rps=math.radians(120.0),
                   gate_slack_m=0.5, default_dt_s=0.1)


@dataclass(slots=True)
class VOResult:
    """Output of :meth:`StereoVO.process` for one frame.

    ``T_prev_cur`` maps points from the current camera frame into the previous
    camera frame (i.e. it is the current camera pose expressed in the previous
    camera frame). It is ``None`` when ``ok`` is False.
    """

    ok: bool
    T_prev_cur: Optional[np.ndarray]
    stats: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    disparity: Optional[np.ndarray] = None  # disparity of the current frame (px, <=0 invalid)
    timings_ms: dict[str, float] = field(default_factory=dict)


def empty_stats(n_tracks: int = 0) -> dict[str, float]:
    """VO statistics for a frame where no pose was estimated (feeds the integrity monitor)."""
    return {
        "inliers": 0.0,
        "inlier_ratio": 0.0,
        "reproj_rmse_px": RMSE_WHEN_FAILED_PX,
        "coverage": 0.0,
        "track_age": 0.0,
        "hess_min_eig": 0.0,
        "n_tracks": float(n_tracks),
        "n_3d": 0.0,
    }


def make_sgbm(num_disparities: int, block: int) -> cv2.StereoSGBM:
    """3-way SGBM matcher for rectified 8-bit gray images."""
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        disp12MaxDiff=1,
        preFilterCap=63,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disparity(sgbm: cv2.StereoSGBM, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Disparity (float32 px, <= 0 invalid) of the rectified pair."""
    raw = sgbm.compute(left, right)
    disp = raw.astype(np.float32) * (1.0 / 16.0)
    disp[disp <= 0.0] = 0.0
    return disp


def sample_patch_means(img: np.ndarray, pts: np.ndarray, box_px: int) -> np.ndarray:
    """Mean intensity (float32, 8-bit DN) of the ``box_px`` x ``box_px`` patch at each pixel ``pts`` (N, 2)
    (x, y); points outside the image are clamped to the border."""
    h, w = img.shape
    blur = cv2.boxFilter(img, cv2.CV_32F, (box_px, box_px))  # float output: no 8-bit rounding of the means
    u = np.clip(np.round(pts[:, 0]).astype(np.int64), 0, w - 1)
    v = np.clip(np.round(pts[:, 1]).astype(np.int64), 0, h - 1)
    return blur[v, u].astype(np.float32)


def photometric_gain_offset(prev_vals: np.ndarray, cur_vals: np.ndarray, gain_min: float = 0.25,
                            gain_max: float = 4.0) -> tuple[float, float]:
    """Affine brightness change ``cur ~ g * prev + o`` (o in 8-bit DN) from intensities of the SAME
    scene points in two frames (patch means at a track and at its predicted position).

    Robust quantile matching: ``g`` = ratio of inter-quartile ranges (clipped to
    ``[gain_min, gain_max]``), ``o`` = median(cur) - g * median(prev). Using corresponding
    points (not whole-image moments) keeps the estimate unbiased when the image content
    changes, e.g. the sky fraction while pitching on rough ground. If the previous samples are
    almost uniform only the offset is estimated.
    """
    qa = np.percentile(prev_vals, PHOTO_QUANTILES)
    qb = np.percentile(cur_vals, PHOTO_QUANTILES)
    iqr_a, iqr_b = float(qa[2] - qa[0]), float(qb[2] - qb[0])
    g = 1.0 if iqr_a < PHOTO_STD_FLOOR_DN else min(max(iqr_b / iqr_a, gain_min), gain_max)
    return g, float(qb[1]) - g * float(qa[1])


def image_gain_offset(prev: np.ndarray, cur: np.ndarray, downsample: int = 4, gain_min: float = 0.25,
                      gain_max: float = 4.0) -> tuple[float, float]:
    """Fallback ``cur ~ g * prev + o`` from whole-image mean / standard deviation at ``1/downsample``
    resolution (biased by content change; used only when too few tracks survived to match points)."""
    h, w = prev.shape
    size = (max(w // downsample, 1), max(h // downsample, 1))
    ma, sa = (float(v[0, 0]) for v in cv2.meanStdDev(cv2.resize(prev, size, interpolation=cv2.INTER_AREA)))
    mb, sb = (float(v[0, 0]) for v in cv2.meanStdDev(cv2.resize(cur, size, interpolation=cv2.INTER_AREA)))
    g = 1.0 if sa < PHOTO_STD_FLOOR_DN else min(max(sb / sa, gain_min), gain_max)
    return g, mb - g * ma


def apply_gain_offset(img: np.ndarray, g: float, o: float) -> np.ndarray:
    """``clip(round(g * img + o), 0, 255)`` for a uint8 image (LUT, ~0.1 ms at 640x400)."""
    lut = np.clip(np.round(g * np.arange(256, dtype=np.float32) + o), 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def rotation_angle(R: np.ndarray) -> float:
    """Geodesic angle (rad) of a rotation matrix."""
    c = (np.trace(R) - 1.0) * 0.5
    return float(math.acos(max(-1.0, min(1.0, c))))


def pose_hessian_min_eig(P_cur: np.ndarray, fx: float, fy: float) -> float:
    """Smallest eigenvalue of the Gauss-Newton pose Hessian J^T J.

    ``P_cur``: (N, 3) inlier points in the *current* camera frame (m). J is the
    Jacobian of the pixel reprojection w.r.t. a left se(3) perturbation
    [rho (m), phi (rad)], so the eigenvalue has mixed units (px^2/m^2 for the
    translation directions, px^2/rad^2 for rotation); in practice the minimum is
    set by the depth-direction translation, i.e. it measures how well the
    geometry constrains forward motion.
    """
    if P_cur.shape[0] < 3:
        return 0.0
    X, Y, Z = P_cur[:, 0], P_cur[:, 1], np.maximum(P_cur[:, 2], 1e-3)
    iz, iz2 = 1.0 / Z, 1.0 / (Z * Z)
    n = P_cur.shape[0]
    J = np.zeros((n, 2, 6))
    # du/dP and dv/dP
    J[:, 0, 0] = fx * iz
    J[:, 0, 2] = -fx * X * iz2
    J[:, 1, 1] = fy * iz
    J[:, 1, 2] = -fy * Y * iz2
    # rotation part: dP/dphi = -[P]x  ->  J_rot = J_trans @ (-[P]x)
    Jt = J[:, :, :3]
    Px = np.zeros((n, 3, 3))
    Px[:, 0, 1], Px[:, 0, 2] = -P_cur[:, 2], P_cur[:, 1]
    Px[:, 1, 0], Px[:, 1, 2] = P_cur[:, 2], -P_cur[:, 0]
    Px[:, 2, 0], Px[:, 2, 1] = -P_cur[:, 1], P_cur[:, 0]
    J[:, :, 3:] = -np.einsum("nij,njk->nik", Jt, Px)
    H = np.einsum("nki,nkj->ij", J, J)
    return float(np.linalg.eigvalsh(H)[0])


def coverage_fraction(pts: np.ndarray, width: int, height: int,
                      grid: tuple[int, int] = COVERAGE_GRID, min_pts: int = COVERAGE_MIN_PTS) -> float:
    """Fraction of cells of a ``grid`` (rows, cols) over the image holding >= ``min_pts`` points."""
    if pts.shape[0] == 0:
        return 0.0
    rows, cols = grid
    cx = np.clip((pts[:, 0] * cols / width).astype(np.int64), 0, cols - 1)
    cy = np.clip((pts[:, 1] * rows / height).astype(np.int64), 0, rows - 1)
    counts = np.bincount(cy * cols + cx, minlength=rows * cols)
    return float(np.count_nonzero(counts >= min_pts)) / float(rows * cols)


class StereoVO:
    """Stateful frame-to-frame stereo VO. Call :meth:`process` once per stereo frame, in order."""

    def __init__(self, K: np.ndarray, baseline_m: float, config: Optional[VOConfig] = None) -> None:
        self.cfg = config or VOConfig()
        self.K = np.asarray(K, dtype=np.float64)
        self.fx, self.fy = float(self.K[0, 0]), float(self.K[1, 1])
        self.cx, self.cy = float(self.K[0, 2]), float(self.K[1, 2])
        self.baseline_m = float(baseline_m)
        self._sgbm = make_sgbm(self.cfg.num_disparities, self.cfg.sgbm_block)
        self._lk_params = dict(
            winSize=(self.cfg.klt_win, self.cfg.klt_win),
            maxLevel=self.cfg.klt_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.cfg.klt_iters, self.cfg.klt_eps),
        )
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._prev_img: Optional[np.ndarray] = None
        self._prev_disp: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None
        self._pts = np.zeros((0, 2), np.float32)  # tracks in the previous image (px)
        self._age = np.zeros(0, np.int32)  # frames each track has survived
        self._last_T_cur_prev: Optional[np.ndarray] = None  # last accepted motion (for KLT seeding)
        self.frames = 0

    @property
    def n_tracks(self) -> int:
        return int(self._pts.shape[0])

    # ------------------------------------------------------------------ helpers
    def _detect(self, img: np.ndarray, existing: np.ndarray) -> np.ndarray:
        """Bucketed Shi-Tomasi corners (N, 2) float32 avoiding ``existing`` tracks."""
        c = self.cfg
        h, w = img.shape
        eig = cv2.cornerMinEigenVal(img, c.corner_block_size)
        r = c.min_distance_px
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
        local_max = eig >= cv2.dilate(eig, kernel)
        thr = c.quality_level * float(eig.max())
        cand = local_max & (eig > thr)
        b = c.border_px
        cand[:b, :] = False
        cand[-b:, :] = False
        cand[:, :b] = False
        cand[:, -b:] = False
        if existing.shape[0]:
            occ = np.zeros((h, w), np.uint8)
            ex = np.round(existing).astype(np.int64)
            ok = (ex[:, 0] >= 0) & (ex[:, 0] < w) & (ex[:, 1] >= 0) & (ex[:, 1] < h)
            occ[ex[ok, 1], ex[ok, 0]] = 255
            occ = cv2.dilate(occ, kernel)
            cand &= occ == 0
        ys, xs = np.nonzero(cand)
        if ys.size == 0:
            return np.zeros((0, 2), np.float32)
        score = eig[ys, xs]
        cell = (ys * c.grid_rows // h) * c.grid_cols + (xs * c.grid_cols // w)
        n_cells = c.grid_rows * c.grid_cols
        budget = max(c.max_corners - existing.shape[0], 0)
        per_cell = int(math.ceil(budget / n_cells))
        if per_cell == 0:
            return np.zeros((0, 2), np.float32)
        # Sort by (cell asc, score desc), keep the first `per_cell` of each cell.
        order = np.lexsort((-score, cell))
        cell_sorted = cell[order]
        starts = np.searchsorted(cell_sorted, np.arange(n_cells))
        rank = np.arange(order.size) - starts[cell_sorted]
        keep = order[rank < per_cell]
        # Existing tracks are uneven across cells; cap the total at the budget by score.
        if keep.size > budget:
            keep = keep[np.argsort(-score[keep])[:budget]]
        return np.stack([xs[keep], ys[keep]], axis=1).astype(np.float32)

    def _backproject(self, pts: np.ndarray, disp: np.ndarray, disp_range: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """3-D points (N, 3) in the camera frame for pixels ``pts``; returns (points, valid mask)."""
        h, w = disp.shape
        u = np.clip(np.round(pts[:, 0]).astype(np.int64), 0, w - 1)
        v = np.clip(np.round(pts[:, 1]).astype(np.int64), 0, h - 1)
        d = disp[v, u]
        valid = (d >= self.cfg.min_disp_px) & (disp_range[v, u] <= self.cfg.max_disp_jump_px)
        d_safe = np.where(valid, d, 1.0)
        Z = self.fx * self.baseline_m / d_safe
        X = (pts[:, 0] - self.cx) * Z / self.fx
        Y = (pts[:, 1] - self.cy) * Z / self.fy
        return np.stack([X, Y, Z], axis=1), valid

    def _project(self, P: np.ndarray) -> np.ndarray:
        Z = np.maximum(P[:, 2], 1e-6)
        return np.stack([self.fx * P[:, 0] / Z + self.cx, self.fy * P[:, 1] / Z + self.cy], axis=1)

    def _track(self, prev_img: np.ndarray, left: np.ndarray, p1_init: Optional[np.ndarray],
               flags: int) -> tuple[np.ndarray, np.ndarray]:
        """KLT of the current tracks from ``prev_img`` to ``left`` with the forward-backward check.
        Returns (p1 (N, 2) px, good (N,) bool)."""
        h, w = left.shape
        p0 = self._pts.reshape(-1, 1, 2)
        init = None if p1_init is None else p1_init.copy()
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_img, left, p0, init, flags=flags, **self._lk_params)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(left, prev_img, p1, p0.copy(),
                                               flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self._lk_params)
        p1 = p1.reshape(-1, 2)
        fb = np.linalg.norm(p0b.reshape(-1, 2) - self._pts, axis=1)
        inside = (p1[:, 0] >= 0) & (p1[:, 0] < w - 1) & (p1[:, 1] >= 0) & (p1[:, 1] < h - 1)
        return p1, (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.cfg.fb_max_px) & inside

    def _gain_offset(self, left: np.ndarray, p1: np.ndarray, good: np.ndarray) -> tuple[float, float]:
        """Brightness change k-1 -> k: from the tracks that survived the first KLT pass (exact
        correspondences) when there are enough, else from whole-image moments (tracking fully broken,
        e.g. a x0.4 dimming step; content changes then bias the estimate, but the step dominates)."""
        cfg = self.cfg
        if int(good.sum()) >= cfg.photo_min_samples:
            a = sample_patch_means(self._prev_img, self._pts[good], cfg.photo_box_px)
            b = sample_patch_means(left, p1[good], cfg.photo_box_px)
            return photometric_gain_offset(a, b, cfg.photo_gain_min, cfg.photo_gain_max)
        return image_gain_offset(self._prev_img, left, cfg.photo_downsample, cfg.photo_gain_min, cfg.photo_gain_max)

    @staticmethod
    def _disp_range(disp: np.ndarray) -> np.ndarray:
        k = np.ones((3, 3), np.uint8)
        return cv2.dilate(disp, k) - cv2.erode(disp, k)

    # ------------------------------------------------------------------ main
    def process(self, left: np.ndarray, right: Optional[np.ndarray] = None,
                disparity: Optional[np.ndarray] = None, t: Optional[float] = None) -> VOResult:
        """Process one rectified stereo frame.

        Args:
            left: (H, W) uint8 gray left image.
            right: (H, W) uint8 gray right image; required if ``disparity`` is None.
            disparity: optional (H, W) float32 disparity of this frame (px, <=0 invalid).
            t: timestamp (s) used by the motion-sanity gate.
        """
        cfg = self.cfg
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        if left.ndim != 2 or left.dtype != np.uint8:
            raise ValueError("left must be a (H, W) uint8 gray image")
        if disparity is None:
            if right is None:
                raise ValueError("either right image or disparity is required")
            disparity = compute_disparity(self._sgbm, left, right)
        else:
            disparity = np.asarray(disparity, np.float32)
        timings["disparity"] = (time.perf_counter() - t0) * 1e3
        self.frames += 1

        if self._prev_img is None:
            self._pts = self._detect(left, np.zeros((0, 2), np.float32))
            self._age = np.zeros(self._pts.shape[0], np.int32)
            self._store(left, disparity, t)
            timings["total"] = (time.perf_counter() - t0) * 1e3
            return VOResult(False, None, empty_stats(self.n_tracks), "init", disparity, timings)

        dt = cfg.default_dt_s if (t is None or self._prev_t is None or t <= self._prev_t) else t - self._prev_t
        h, w = left.shape

        # --- 3-D points at k-1 (needed for KLT seeding and PnP)
        t1 = time.perf_counter()
        disp_rng = self._disp_range(self._prev_disp)
        P0, has3d = self._backproject(self._pts, self._prev_disp, disp_rng)

        # --- KLT with forward-backward check, seeded by the previous motion
        p0 = self._pts.reshape(-1, 1, 2)
        flags = 0
        p1_init = None
        if self._last_T_cur_prev is not None and p0.shape[0]:
            R, tt = self._last_T_cur_prev[:3, :3], self._last_T_cur_prev[:3, 3]
            P_pred = P0 @ R.T + tt
            seedable = has3d & (P_pred[:, 2] > 0.1)
            pred = np.where(seedable[:, None], self._project(P_pred), self._pts)
            p1_init = pred.astype(np.float32).reshape(-1, 1, 2)
            flags = cv2.OPTFLOW_USE_INITIAL_FLOW
        if p0.shape[0] == 0:
            return self._fail(left, disparity, t, timings, t0, "no_tracks")
        p1, good = self._track(self._prev_img, left, p1_init, flags)
        photo = {"photo_gain": 1.0, "photo_offset_dn": 0.0}
        if cfg.photometric_norm and good.mean() < cfg.photo_retry_good_frac:
            # Tracking was poor: if the pair shows a real brightness change, retry on a normalised k-1 image.
            g, o = self._gain_offset(left, p1, good)
            if abs(g - 1.0) > cfg.photo_min_gain_dev or abs(o) > cfg.photo_min_offset_dn:
                p1n, goodn = self._track(apply_gain_offset(self._prev_img, g, o), left, p1_init, flags)
                if goodn.sum() > good.sum():
                    p1, good = p1n, goodn
                    photo = {"photo_gain": g, "photo_offset_dn": o}
        timings["klt"] = (time.perf_counter() - t1) * 1e3

        # --- PnP RANSAC + LM refinement
        t2 = time.perf_counter()
        use = good & has3d
        idx = np.nonzero(use)[0]
        n3d = int(idx.size)
        if n3d < max(cfg.min_inliers, 6):
            self._pts, self._age = p1[good], self._age[good] + 1
            return self._fail(left, disparity, t, timings, t0, f"few_3d={n3d}", n3d=n3d)
        obj = P0[idx].astype(np.float64)
        img = p1[idx].astype(np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, img, self.K, None, iterationsCount=cfg.ransac_iters, reprojectionError=cfg.ransac_px,
            confidence=cfg.ransac_conf, flags=cv2.SOLVEPNP_AP3P)
        if not ok or inl is None or len(inl) < cfg.min_inliers:
            self._pts, self._age = p1[good], self._age[good] + 1
            n_in = 0 if inl is None else len(inl)
            return self._fail(left, disparity, t, timings, t0, f"pnp_fail inl={n_in}", n3d=n3d)
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], self.K, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        tv = tvec.ravel()
        P1 = obj @ R.T + tv
        err = np.linalg.norm(self._project(P1) - img, axis=1)
        inl_mask = (err < cfg.refine_px) & (P1[:, 2] > 0)
        n_inl = int(inl_mask.sum())
        if n_inl < cfg.min_inliers:
            self._pts, self._age = p1[good], self._age[good] + 1
            return self._fail(left, disparity, t, timings, t0, f"refine_fail inl={n_inl}", n3d=n3d)
        # second LM pass on the refined inlier set
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl_mask], img[inl_mask], self.K, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        tv = tvec.ravel()
        P1 = obj @ R.T + tv
        err = np.linalg.norm(self._project(P1) - img, axis=1)
        inl_mask = (err < cfg.refine_px) & (P1[:, 2] > 0)
        n_inl = int(inl_mask.sum())
        timings["pnp"] = (time.perf_counter() - t2) * 1e3

        T_cur_prev = np.eye(4)
        T_cur_prev[:3, :3], T_cur_prev[:3, 3] = R, tv
        T_prev_cur = np.linalg.inv(T_cur_prev)

        stats = {
            "inliers": float(n_inl),
            "inlier_ratio": float(n_inl) / float(n3d),
            "reproj_rmse_px": float(np.sqrt(np.mean(err[inl_mask] ** 2))) if n_inl else RMSE_WHEN_FAILED_PX,
            "coverage": coverage_fraction(img[inl_mask], w, h),
            "track_age": float(np.mean(self._age[idx][inl_mask])) + 1.0 if n_inl else 0.0,
            "hess_min_eig": pose_hessian_min_eig(P1[inl_mask], self.fx, self.fy),
            "n_tracks": float(good.sum()),
            "n_3d": float(n3d),
            **photo,
        }

        # --- motion-sanity gate
        trans = float(np.linalg.norm(T_prev_cur[:3, 3]))
        ang = rotation_angle(R)
        max_trans = cfg.max_speed_mps * dt + cfg.gate_slack_m
        max_ang = cfg.max_yaw_rate_rps * dt + cfg.gate_slack_rad
        sane = trans <= max_trans and ang <= max_ang

        # --- track management: drop PnP outliers, keep tracks without depth
        outlier = np.zeros(p1.shape[0], bool)
        outlier[idx[~inl_mask]] = True
        keep = good & ~outlier
        self._pts, self._age = p1[keep], self._age[keep] + 1
        if not sane:
            reason = f"motion_gate trans={trans:.2f}m ang={math.degrees(ang):.1f}deg"
            return self._fail(left, disparity, t, timings, t0, reason, n3d=n3d, stats=stats)

        self._last_T_cur_prev = T_cur_prev
        t3 = time.perf_counter()
        self._maybe_redetect(left)
        timings["detect"] = (time.perf_counter() - t3) * 1e3
        self._store(left, disparity, t)
        timings["total"] = (time.perf_counter() - t0) * 1e3
        return VOResult(True, T_prev_cur, stats, "", disparity, timings)

    def _maybe_redetect(self, img: np.ndarray) -> None:
        if self._pts.shape[0] < self.cfg.min_tracks:
            new = self._detect(img, self._pts)
            if new.shape[0]:
                self._pts = np.concatenate([self._pts, new], axis=0)
                self._age = np.concatenate([self._age, np.zeros(new.shape[0], np.int32)])

    def _store(self, img: np.ndarray, disp: np.ndarray, t: Optional[float]) -> None:
        self._prev_img, self._prev_disp, self._prev_t = img, disp, t

    def _fail(self, img: np.ndarray, disp: np.ndarray, t: Optional[float], timings: dict[str, float],
              t0: float, reason: str, n3d: int = 0, stats: Optional[dict[str, float]] = None) -> VOResult:
        """Common failure path: keep the image stream going and re-detect features."""
        s = empty_stats(self.n_tracks) if stats is None else dict(stats)
        s["n_3d"] = float(n3d)
        self._last_T_cur_prev = None
        self._maybe_redetect(img)
        self._store(img, disp, t)
        timings["total"] = (time.perf_counter() - t0) * 1e3
        LOG.debug("VO frame %d rejected: %s", self.frames, reason)
        return VOResult(False, None, s, reason, disp, timings)
