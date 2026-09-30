"""Onboard perception node: stereo -> ground -> seen-ground BEV with observation states.

Implements :class:`metagross.contracts.interfaces.PerceptionProto`::

    Perception(calib, vehicle, config, segmenter=None).process(frame, pose_xy_yaw) -> dict

Output dictionary (egocentric BEV arrays follow :class:`~.bev.BevSpec`: ``grid[i, j]``,
i forward from ``x_min``, j to the left from ``y_min``; see ``out['bev_spec']``):

=====================  ======================================================================
``disparity``          (H, W) float32 px at calibrated resolution, <= 0 invalid
``cell_state_local``   (nx, ny) uint8 :class:`~metagross.contracts.messages.CellState`
``cost_local``         (nx, ny) float32 in [0, 1]; 1.0 = lethal (POSITIVE / DEPRESSION / DITCH)
``height_local``       (nx, ny) float32 mean terrain height z (m, body frame); certified-ground
                       cells without points take the ground-model height; NaN = unknown
``mu_local``           (nx, ny) float32 braking-friction proxy (``interfaces.SEM_MU``)
``semantic_mask``      (H, W) uint8 class ids or None (no segmenter / semantics disabled)
``missing_ground_mask``(H, W) bool image-space ditch / crest-shadow / depression pixels
``r_vis_m``            float: farthest forward range up to which >= 60 % of the
                       expected-visible ground cells of the forward sector were observed
``certified_local``    (nx, ny) bool: GROUND that is *certified*: a ditch of the design width
                       (``defaults.DESIGN_DITCH_WIDTH_M``) would already cover >=
                       ``defaults.MIN_PIXELS_ON_TARGET`` px there (Matthies-Rankin, camera height
                       above that cell's terrain level; :mod:`~.certify`). GROUND outside it is
                       seen but NOT proof of drivable ground: for certification (governor arc /
                       speed) treat it like unknown; it may still be routed through at unknown cost
``r_det_m``            float: design-ditch detection range (m, horizontal from the camera) on
                       level ground at the current camera height; certified cells lie within ~it
``timings_ms``         dict of stage wall-clock times (ms)
``bev_spec``           dict describing the grid; ``ground`` / ``expected_px`` / ``counts`` extras
=====================  ======================================================================

Unknown handling (thesis: *unknown is never free*): UNSEEN / OCCLUDED / CREST_SHADOW cells
cost ``UNKNOWN_COST``; the ablation switch ``config['unknown_is_free']`` sets it to 0.
``config['use_negobs'] = False`` disables gap classification (no DITCH / CREST / OCCLUDED:
missing ground simply stays UNSEEN). ``config['use_semantics'] = False`` skips the segmenter.
``config['ditch_persistence']`` (default True): a DITCH_CANDIDATE must be supported by a raw
candidate of one of the last 2 frames within 0.25 m in the odometry frame (``pose_xy_yaw``, the
localiser pose); unconfirmed candidates are output as UNSEEN (:mod:`~.persistence`). The first
frame of a Perception instance is not filtered.

Latency: the segmenter (ONNX Runtime, releases the GIL) runs on a worker thread that is started
as soon as the frame is known (:meth:`Perception.compute_disparity` or the top of
:meth:`Perception.process`), so it overlaps SGBM / geometry instead of adding to them
(``config['async_semantics']``, default True; outputs are identical either way).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional

import numpy as np

from metagross.autonomy.perception.bev import BevSpec, accumulate, expected_coverage, rasterize_tracks, visible_range
from metagross.autonomy.perception.certify import GroundCertifier
from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.autonomy.perception.ground import GroundModel, fit_ground
from metagross.autonomy.perception.negobs import MissingGroundDetector, NegObsResult, Segments, filter_ditch_cells
from metagross.autonomy.perception.persistence import DitchPersistence
from metagross.autonomy.perception.positive import (MIN_POINTS_POSITIVE, count_outside, floating_points, step_threshold_m, terrain_costs,
                                                    terrain_level, vertical_extent_m)
from metagross.autonomy.perception.semantic_bev import MU_DEFAULT, fuse_semantics, project_semantics
from metagross.autonomy.perception.stereo import SGBMParams, StereoMatcher, disparity_for_frame
from metagross.config import defaults
from metagross.contracts.interfaces import SegmenterProto
from metagross.contracts.messages import CellState, SensorFrame, StereoCalibration, VehicleSpec

LOG = logging.getLogger(__name__)

UNKNOWN_COST = 0.5  # cost of UNSEEN / OCCLUDED / CREST_SHADOW cells (0 if unknown_is_free)
LETHAL_COST = 1.0
POINT_STRIDE = (2, 2)  # (rows, cols) sampling of disparity pixels for the BEV
GROUND_FIT_STRIDE = (4, 4)  # the ground fit uses every 2nd BEV sample row and column
MIN_POINTS_DEPRESSION = 2  # points needed to declare a cell a lethal depression
RASTER_STEP_M = 0.05  # sampling step along hazard ground-track segments (half a cell)
GROUND_FILL_STEP_M = defaults.BEV_RES_M  # seen-ground fill is sampled once per cell
HORIZON_MARGIN_ROWS = 40  # rows above the nominal horizon still back-projected (uphill terrain)

LETHAL_STATES = (CellState.POSITIVE, CellState.DEPRESSION, CellState.DITCH_CANDIDATE)
UNKNOWN_STATES = (CellState.UNSEEN, CellState.OCCLUDED, CellState.CREST_SHADOW)

DEFAULT_PERCEPTION_CONFIG: dict[str, Any] = {
    "use_negobs": True,
    "use_semantics": True,
    "unknown_is_free": False,
    "ditch_persistence": True,  # confirm ditch candidates across frames in the odometry frame (persistence.py)
    "async_semantics": True,  # run the segmenter on a worker thread, overlapping SGBM (same outputs)
    "min_obstacle_size": True,  # drop step points on image surfaces < positive.MIN_OBSTACLE_EXTENT_M tall
    "noise_aware_step": True,  # lethal step threshold max(STEP_LETHAL_M, K sigma_h(r)) (positive.step_threshold_m)
    "positive_persistence": True,  # confirm POSITIVE cells across 2 frames in the odometry frame, like ditches
    "seed": 0,
}


class Perception:
    """Seen-ground perception (see module docstring)."""

    def __init__(self, calib: StereoCalibration, vehicle: VehicleSpec = defaults.VEHICLE, config: Optional[dict] = None,
                 segmenter: Optional[SegmenterProto] = None, sgbm: Optional[SGBMParams] = None) -> None:
        self.calib = calib
        self.vehicle = vehicle
        self.config = {**DEFAULT_PERCEPTION_CONFIG, **(config or {})}
        self.segmenter = segmenter
        self.geom = CameraGeometry(calib)
        # Rows far above the flat-ground horizon cannot hold ground within range; skip them.
        zc_flat = self.geom.flat_ground_depth()
        ground_rows = np.nonzero(np.isfinite(zc_flat).any(axis=1))[0]
        horizon = int(ground_rows[0]) if ground_rows.size else 0
        self.row_start = max(0, horizon - HORIZON_MARGIN_ROWS)
        self.matcher = StereoMatcher(sgbm or SGBMParams(row_start=self.row_start))
        self.spec = BevSpec()
        self.expected_px, self.expected_rows = expected_coverage(self.geom, self.spec)
        self.detector = MissingGroundDetector(self.geom)
        self.certifier = GroundCertifier(self.geom, self.spec)
        self.persistence = DitchPersistence(self.spec)
        self.pos_persistence = DitchPersistence(self.spec)  # same world-frame test for POSITIVE cells
        cx, cy = self.spec.centres()
        self._cell_x, self._cell_y = cx, cy
        self._n = 0
        self._disp_cache: Optional[tuple[int, float, np.ndarray, float]] = None  # (seq, t, disparity, ms)
        self._seg_pool: Optional[ThreadPoolExecutor] = None
        if segmenter is not None and self.config.get("async_semantics", True):
            self._seg_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="perception-seg")
        self._seg_job: Optional[tuple[int, float, Future]] = None  # (seq, t, future of segmenter(left_rgb))

    def _start_semantics(self, frame: SensorFrame) -> None:
        """Submit the segmenter for this frame to the worker thread (once per frame)."""
        if self._seg_pool is None or frame.left_rgb is None or not self.config.get("use_semantics", True):
            return
        if self._seg_job is not None and self._seg_job[0] == frame.seq and self._seg_job[1] == frame.t:
            return
        self._seg_job = (frame.seq, frame.t, self._seg_pool.submit(self.segmenter, frame.left_rgb))

    def _semantics_result(self, frame: SensorFrame) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """(class ids, entropy) of this frame: from the worker if submitted, else computed inline."""
        job, self._seg_job = self._seg_job, None
        if job is not None and job[0] == frame.seq and job[1] == frame.t:
            return job[2].result()
        return self.segmenter(frame.left_rgb)

    def compute_disparity(self, frame: SensorFrame) -> np.ndarray:
        """Stage 1 of a tick: the frame's disparity (SGBM on stereo frames, the Tier-0 map
        otherwise), cached so that the following :meth:`process` of the same frame reuses it.
        The node calls this first and hands the result to the localiser (one SGBM per frame)."""
        self._start_semantics(frame)
        disp, ms = disparity_for_frame(frame, self.calib, self.matcher)
        self._disp_cache = (frame.seq, frame.t, disp, ms)
        return disp

    # ------------------------------------------------------------------ helpers
    def _raster(self, seg: Segments, res: NegObsResult, step_m: float = RASTER_STEP_M) -> np.ndarray:
        """Boolean grid of cells touched by ground-track segments."""
        grid = np.zeros(self.spec.n_cells, dtype=bool)
        if len(seg):
            cells, _ = rasterize_tracks(self.spec, seg.x0, seg.x1, (res.track_y0[seg.col], res.track_ky[seg.col]),
                                        res.track_half_w[seg.col], step_m)
            grid[cells] = True
        return grid.reshape(self.spec.shape)

    # ------------------------------------------------------------------ main
    def process(self, frame: SensorFrame, pose_xy_yaw: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict[str, Any]:
        """One perception tick. All outputs are egocentric (body frame, see the module docstring,
        including ``certified_local`` / ``r_det_m``). ``pose_xy_yaw`` (odometry frame: m, m, rad)
        is used only to confirm ditch candidates across frames (``ditch_persistence``)."""
        t0 = time.perf_counter()
        tm: dict[str, float] = {}
        cfg = self.config
        geom, spec = self.geom, self.spec
        self._start_semantics(frame)  # no-op if compute_disparity() already started it

        cached, self._disp_cache = self._disp_cache, None
        if cached is not None and cached[0] == frame.seq and cached[1] == frame.t:
            disp, tm["stereo"] = cached[2], cached[3]  # computed by compute_disparity() this tick
        else:
            disp, tm["stereo"] = disparity_for_frame(frame, self.calib, self.matcher)

        t = time.perf_counter()
        pts = geom.points_from_disparity(disp, POINT_STRIDE[0], POINT_STRIDE[1], row_start=self.row_start)
        fit_pts = pts.subset((pts.u % GROUND_FIT_STRIDE[1] == 0) & (pts.v % GROUND_FIT_STRIDE[0] == 0))
        tm["geometry"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        model: GroundModel = fit_ground(fit_pts, disp, geom, seed=int(cfg.get("seed", 0)) + self._n)
        tm["ground"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        h_rel = (pts.z - model.height(pts.x, pts.y)).astype(np.float32)
        n_floating = 0
        floating_cells = np.zeros(spec.shape, dtype=bool)
        if cfg.get("min_obstacle_size", True):
            # Minimum physical size: points high above the ground on an image surface shorter than
            # MIN_OBSTACLE_EXTENT_M are stereo mismatches (sky / horizon streaks back-project to
            # ~1 m "obstacles" right in front of the vehicle); drop them before binning.
            # Unknown is never free: dropping them must not turn their cells into GROUND. A cell
            # holding >= MIN_POINTS_POSITIVE of them (the count that would have made it a lethal
            # step) is UNSEEN below - never certified - instead of POSITIVE. (Dropping them
            # outright certified the hidden F3 trench on DEV 122 at close range.)
            ext = vertical_extent_m(disp, geom.fxb, geom.fy, POINT_STRIDE[0], POINT_STRIDE[1], self.row_start)
            ext_pts = ext[(pts.v - self.row_start) // POINT_STRIDE[0], pts.u // POINT_STRIDE[1]]
            floating = floating_points(ext_pts, h_rel)
            n_floating = int(floating.sum())
            if n_floating:
                fidx = spec.flat_index(pts.x[floating], pts.y[floating])
                n_fl = np.bincount(fidx[fidx >= 0], minlength=spec.n_cells).reshape(spec.shape)
                floating_cells = n_fl >= MIN_POINTS_POSITIVE
                pts, h_rel = pts.subset(~floating), h_rel[~floating]
        stats = accumulate(spec, pts.x, pts.y, pts.z, h_rel)
        ground_z = model.height(self._cell_x, self._cell_y).astype(np.float32)
        tm["bev"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        use_negobs = bool(cfg.get("use_negobs", True))
        neg = self.detector.detect(disp, model, use_negobs=use_negobs)
        ground_fill = self._raster(neg.ground, neg, GROUND_FILL_STEP_M)
        ditch = filter_ditch_cells(self._raster(neg.ditch, neg), self._cell_x, self._cell_y, (geom.t_bc[0], geom.t_bc[1]))
        ditch_unconfirmed = np.zeros_like(ditch)
        if cfg.get("ditch_persistence", True) and use_negobs:
            ditch, ditch_unconfirmed = self.persistence.filter(ditch, pose_xy_yaw)
        crest = self._raster(neg.crest, neg)
        occluded = self._raster(neg.occluded, neg)
        tm["negobs"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        has = stats.count > 0
        # Heights are judged against the local terrain level (band model + morphological
        # open-close of the measured heights), so rolling terrain is not a step / depression.
        # A cell is a step / depression only with >= 2 points beyond the threshold (robust to
        # single gross stereo mismatches).
        level = terrain_level(stats, spec, ground_z)
        step_thr = step_threshold_m(spec, neg.sigma_d_px) if cfg.get("noise_aware_step", True) else defaults.STEP_LETHAL_M
        n_up, n_dn = count_outside(spec, pts.x, pts.y, h_rel, level, step_thr, defaults.DEPRESSION_LETHAL_M)
        pts_ground = has & (n_up < MIN_POINTS_DEPRESSION) & (n_dn < MIN_POINTS_DEPRESSION)
        observed = has | ground_fill
        terrain = terrain_costs(stats, spec, ground_z, observed, level, n_up)
        depression = n_dn >= MIN_POINTS_DEPRESSION
        positive = terrain.positive
        positive_unconfirmed = np.zeros_like(positive)
        if cfg.get("positive_persistence", True):
            positive, positive_unconfirmed = self.pos_persistence.filter(positive, pose_xy_yaw)

        state = np.full(spec.shape, CellState.UNSEEN, dtype=np.uint8)
        state[occluded] = CellState.OCCLUDED
        state[pts_ground | (ground_fill & ~has)] = CellState.GROUND
        state[crest & ~observed] = CellState.CREST_SHADOW
        state[has & ~pts_ground & ~depression & ~positive & ~positive_unconfirmed] = CellState.GROUND  # mild outliers
        state[floating_cells & (state == CellState.GROUND)] = CellState.UNSEEN  # unexplained high points: unknown
        state[ditch_unconfirmed] = CellState.UNSEEN  # not (yet) persistent: unknown, not ground
        state[ditch] = CellState.DITCH_CANDIDATE
        state[depression] = CellState.DEPRESSION
        state[positive_unconfirmed] = CellState.UNSEEN  # a step seen in one frame only: unknown, not ground
        state[positive] = CellState.POSITIVE
        if not use_negobs:
            state[np.isin(state, (CellState.OCCLUDED, CellState.CREST_SHADOW, CellState.DITCH_CANDIDATE))] = CellState.UNSEEN

        unknown_cost = 0.0 if cfg.get("unknown_is_free", False) else UNKNOWN_COST
        cost = np.full(spec.shape, unknown_cost, dtype=np.float32)
        gmask = state == CellState.GROUND
        cost[gmask] = terrain.cost[gmask]
        cost[np.isin(state, np.array(LETHAL_STATES, dtype=np.uint8))] = LETHAL_COST
        tm["cells"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        sem_mask: Optional[np.ndarray] = None
        mu = np.full(spec.shape, MU_DEFAULT, dtype=np.float32)
        if self.segmenter is not None and cfg.get("use_semantics", True) and frame.left_rgb is not None:
            sem_mask, entropy = self._semantics_result(frame)
            sem = project_semantics(sem_mask, disp, geom, model, spec, entropy=entropy)
            state, cost = fuse_semantics(state, cost, sem)
            mu = sem.mu
        tm["semantics"] = (time.perf_counter() - t) * 1e3

        height = np.where(has, stats.z_mean, np.where(ground_fill, ground_z, np.nan)).astype(np.float32)
        r_vis = visible_range(spec, state, self.expected_px)
        # Certification uses the smooth local terrain level (ground model + open-close level),
        # not the noisy per-cell mean: H = camera height above that cell's terrain.
        certified = self.certifier.certify(state, ground_z + level)
        tx, ty, tz = geom.t_bc
        r_det = self.certifier.r_det_m(float(tz - model.height(np.array(tx), np.array(ty))))
        tm["total"] = (time.perf_counter() - t0) * 1e3
        self._n += 1
        return {
            "disparity": disp,
            "cell_state_local": state,
            "cost_local": cost,
            "height_local": height,
            "mu_local": mu.astype(np.float32),
            "semantic_mask": sem_mask,
            "missing_ground_mask": neg.image_mask,
            "r_vis_m": r_vis,
            "certified_local": certified,
            "r_det_m": r_det,
            "timings_ms": tm,
            "bev_spec": spec.as_dict(),
            "pose_xy_yaw": tuple(pose_xy_yaw),
            "ground_bands": [(b.x0, b.x1, b.a, b.b, b.c, b.fitted) for b in model.bands],
            "n_ditch_columns": int(len(neg.ditch)),
            "n_crest_columns": int(len(neg.crest)),
            "n_floating_points": n_floating,
            "sigma_d_px": float(neg.sigma_d_px),
        }
