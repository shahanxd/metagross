"""Dense stereo disparity for the perception channel.

Image frame: rectified left camera, pixel (row v, column u), OpenCV convention.
Output disparity is ``float32`` in **pixels of the returned array** (``d = fx * B / Z_c``);
any value ``<= 0`` means *invalid / no measurement*.

Two sources are supported:

* ``stereo`` sensor mode: CLAHE-equalised grey images -> OpenCV ``StereoSGBM``
  (``MODE_SGBM_3WAY``, 64 disparities, 5x5 blocks, P1 = 8*cn*25, P2 = 32*cn*25,
  uniqueness 10, speckle filter, ``disp12MaxDiff = 1``) with an optional explicit
  left-right consistency check.
* ``tier0_disparity`` sensor mode: the frame already carries a disparity image
  (possibly at a lower resolution). It is resampled to the calibrated resolution with
  nearest-neighbour interpolation and its values are multiplied by the resize factor,
  i.e. the tier-0 array is assumed to be expressed in pixels of its own resolution
  (a camera with intrinsics scaled by the same factor).

The CLAHE parameters are shared with visual odometry so both channels see the same
photometric normalisation (:func:`make_clahe`).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from metagross.contracts.messages import SensorFrame, StereoCalibration

LOG = logging.getLogger(__name__)

# Shared photometric normalisation (also used by VO): contrast-limited adaptive
# histogram equalisation on 8x8 tiles, clip limit 2.0 (OpenCV default scale).
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

SGBM_DISP_SCALE = 16.0  # OpenCV SGBM returns fixed-point disparity * 16
INVALID_DISPARITY = -1.0  # value written to invalid pixels (any <= 0 is invalid)


def make_clahe() -> "cv2.CLAHE":
    """Return the project-wide CLAHE operator (shared by stereo and VO)."""
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)


def to_gray(img: np.ndarray) -> np.ndarray:
    """(H,W,3) RGB uint8 or (H,W) uint8 -> (H,W) uint8 grey."""
    if img.ndim == 2:
        return img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


@dataclass(frozen=True)
class SGBMParams:
    """StereoSGBM configuration (all disparities in pixels, block size in pixels)."""

    num_disparities: int = 64
    block_size: int = 5
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100  # px area of a connected blob below which it is removed
    speckle_range: int = 2  # max disparity variation inside a blob (x16 internally by OpenCV)
    disp12_max_diff: int = 1  # OpenCV internal left-right check tolerance, px
    pre_filter_cap: int = 31
    use_clahe: bool = True
    lr_check: bool = False  # explicit (second-pass) left-right consistency check
    lr_tol_px: float = 1.0
    row_start: int = 0  # rows above this are not matched (sky crop); 0 = full image

    def p1(self, channels: int = 1) -> int:
        return 8 * channels * self.block_size * self.block_size

    def p2(self, channels: int = 1) -> int:
        return 32 * channels * self.block_size * self.block_size


class StereoMatcher:
    """CLAHE + StereoSGBM(3-way) disparity with optional left-right check.

    ``compute(left, right)`` returns float32 disparity (px) at the input resolution,
    ``<= 0`` invalid. ``last_ms`` holds the wall-clock time of the last call.
    """

    def __init__(self, params: Optional[SGBMParams] = None) -> None:
        self.params = params or SGBMParams()
        p = self.params
        self._clahe = make_clahe()
        self._sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=p.num_disparities,
            blockSize=p.block_size,
            P1=p.p1(1),
            P2=p.p2(1),
            disp12MaxDiff=p.disp12_max_diff,
            preFilterCap=p.pre_filter_cap,
            uniquenessRatio=p.uniqueness_ratio,
            speckleWindowSize=p.speckle_window_size,
            speckleRange=p.speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.last_ms = 0.0

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        g = to_gray(img)
        return self._clahe.apply(g) if self.params.use_clahe else g

    def _raw(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        d16 = self._sgbm.compute(left, right)
        disp = d16.astype(np.float32) / SGBM_DISP_SCALE
        disp[d16 <= 0] = INVALID_DISPARITY
        return disp

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Rectified left/right (RGB or grey uint8) -> float32 disparity (px), <=0 invalid."""
        t0 = time.perf_counter()
        gl, gr = self.preprocess(left), self.preprocess(right)
        h = gl.shape[0]
        r0 = int(np.clip(self.params.row_start, 0, h - self.params.block_size - 1))
        disp = np.full(gl.shape, INVALID_DISPARITY, dtype=np.float32)
        dl = self._raw(gl[r0:], gr[r0:])
        if self.params.lr_check:
            dl = self._left_right_check(dl, gl[r0:], gr[r0:])
        disp[r0:] = dl
        self.last_ms = (time.perf_counter() - t0) * 1e3
        return disp

    def _left_right_check(self, dl: np.ndarray, gl: np.ndarray, gr: np.ndarray) -> np.ndarray:
        """Invalidate pixels whose right-image disparity disagrees by > lr_tol_px.

        The right disparity map is obtained by matching the horizontally flipped pair
        (right becomes the reference image), which needs no opencv-contrib.
        """
        dr = self._raw(np.ascontiguousarray(gr[:, ::-1]), np.ascontiguousarray(gl[:, ::-1]))[:, ::-1]
        h, w = dl.shape
        uu = np.broadcast_to(np.arange(w, dtype=np.float32), (h, w))
        ur = np.rint(uu - np.maximum(dl, 0.0)).astype(np.int32)
        ok = (dl > 0) & (ur >= 0)
        ur = np.clip(ur, 0, w - 1)
        d_back = np.take_along_axis(dr, ur, axis=1)
        ok &= (d_back > 0) & (np.abs(d_back - dl) <= self.params.lr_tol_px)
        out = dl.copy()
        out[~ok] = INVALID_DISPARITY
        return out


def resample_disparity(disp: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a disparity image (px of its own resolution) to (height, width).

    Nearest-neighbour keeps discontinuities sharp; values are scaled by the horizontal
    resize factor so they are expressed in pixels of the target resolution.
    """
    disp = np.asarray(disp, dtype=np.float32)
    h0, w0 = disp.shape
    if (h0, w0) == (height, width):
        return disp
    scale = width / float(w0)
    out = cv2.resize(disp, (width, height), interpolation=cv2.INTER_NEAREST)
    valid = out > 0
    out = np.where(valid, out * scale, INVALID_DISPARITY).astype(np.float32)
    return out


def disparity_for_frame(frame: SensorFrame, calib: StereoCalibration, matcher: StereoMatcher) -> tuple[np.ndarray, float]:
    """Disparity (px at calib resolution, float32, <=0 invalid) for a sensor frame, plus ms.

    Honours ``frame.sensor_mode``: tier-0 frames carry their own disparity, stereo frames
    are matched with SGBM.
    """
    t0 = time.perf_counter()
    if frame.disparity is not None and (frame.sensor_mode == "tier0_disparity" or frame.left_rgb is None):
        disp = resample_disparity(frame.disparity, calib.width, calib.height)
    elif frame.left_rgb is not None and frame.right_gray is not None:
        disp = matcher.compute(frame.left_rgb, frame.right_gray)
        if disp.shape != (calib.height, calib.width):
            disp = resample_disparity(disp, calib.width, calib.height)
    else:
        raise ValueError(f"frame seq={frame.seq}: no disparity and no stereo pair (mode={frame.sensor_mode})")
    return disp, (time.perf_counter() - t0) * 1e3
