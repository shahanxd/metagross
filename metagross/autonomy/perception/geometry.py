"""Camera geometry: disparity -> 3-D points in the camera frame and the body frame.

Frames
------
* Camera (left, rectified): OpenCV, x right, y down, z forward (optical axis), metres.
* Body: x forward, y left, z up, origin at the wheelbase centre on the ground plane.
  ``p_body = R_bc @ p_cam + t_bc`` with ``T_body_cam = calib.T_body_cam``.

For a pixel (u, v) we pre-compute the body-frame ray ``r(u, v) = R_bc @ [x_n, y_n, 1]``
(x_n = (u - cx)/fx, y_n = (v - cy)/fy). A point at optical depth Z_c on that pixel is
``t_bc + Z_c * r(u, v)``, and its disparity is ``d = fx * B / Z_c`` (pixels).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metagross.config import defaults
from metagross.contracts.messages import StereoCalibration


@dataclass
class PointSet:
    """Sparse set of back-projected pixels (all 1-D arrays of equal length N).

    x, y, z: body frame, metres. zc: optical depth, metres. d: disparity, px. v, u: pixel.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    zc: np.ndarray
    d: np.ndarray
    v: np.ndarray
    u: np.ndarray

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def subset(self, mask: np.ndarray) -> "PointSet":
        return PointSet(*(a[mask] for a in (self.x, self.y, self.z, self.zc, self.d, self.v, self.u)))


class CameraGeometry:
    """Pinhole stereo geometry of the rectified left camera plus its mounting."""

    def __init__(self, calib: StereoCalibration, max_range_m: float = defaults.CAM_MAX_RANGE_M) -> None:
        self.calib = calib
        self.width, self.height = int(calib.width), int(calib.height)
        self.fx, self.fy, self.cx, self.cy = float(calib.fx), float(calib.fy), float(calib.cx), float(calib.cy)
        self.baseline_m = float(calib.baseline_m)
        self.fxb = self.fx * self.baseline_m  # px*m: d = fxb / Z_c
        T = np.asarray(calib.T_body_cam, dtype=np.float64)
        self.R_bc = T[:3, :3].copy()
        self.t_bc = T[:3, 3].copy()
        self.max_range_m = float(max_range_m)
        self.min_disparity_px = self.fxb / self.max_range_m  # below this: beyond the range cap
        uu = (np.arange(self.width, dtype=np.float64) - self.cx) / self.fx
        vv = (np.arange(self.height, dtype=np.float64) - self.cy) / self.fy
        xn, yn = np.meshgrid(uu, vv)
        rays_cam = np.stack([xn, yn, np.ones_like(xn)], axis=-1)  # (H, W, 3), z_c = 1
        self.rays_body = (rays_cam @ self.R_bc.T).astype(np.float32)  # (H, W, 3)

    # ------------------------------------------------------------------ conversions
    def depth_from_disparity(self, d: np.ndarray) -> np.ndarray:
        """Optical depth Z_c (m) from disparity (px); NaN where d <= 0."""
        d = np.asarray(d, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(d > 0, self.fxb / d, np.nan).astype(np.float32)

    def points_from_disparity(self, disp: np.ndarray, stride_v: int = 1, stride_u: int = 1, row_start: int = 0) -> PointSet:
        """Back-project valid pixels on a (stride_v, stride_u) grid into the body frame.

        Pixels with d <= 0 or Z_c > max_range_m are dropped.
        """
        d = disp[row_start::stride_v, ::stride_u]
        vs = np.arange(row_start, disp.shape[0], stride_v)
        us = np.arange(0, disp.shape[1], stride_u)
        valid = d >= self.min_disparity_px
        iv, iu = np.nonzero(valid)
        dv = d[iv, iu].astype(np.float32)
        v = vs[iv].astype(np.int32)
        u = us[iu].astype(np.int32)
        zc = (self.fxb / dv).astype(np.float32)
        r = self.rays_body[v, u]  # (N, 3)
        x = self.t_bc[0] + zc * r[:, 0]
        y = self.t_bc[1] + zc * r[:, 1]
        z = self.t_bc[2] + zc * r[:, 2]
        return PointSet(x.astype(np.float32), y.astype(np.float32), z.astype(np.float32), zc, dv, v, u)

    def point_image(self, disp: np.ndarray) -> np.ndarray:
        """(H, W, 3) float32 body-frame XYZ per pixel, NaN where invalid / beyond range."""
        zc = self.depth_from_disparity(disp)
        zc[zc > self.max_range_m] = np.nan
        return (self.t_bc.astype(np.float32) + zc[..., None] * self.rays_body).astype(np.float32)

    def cam_to_body(self, p_cam: np.ndarray) -> np.ndarray:
        """(N, 3) camera-frame points -> (N, 3) body-frame points (m)."""
        return np.asarray(p_cam) @ self.R_bc.T + self.t_bc

    def body_to_cam(self, p_body: np.ndarray) -> np.ndarray:
        """(N, 3) body-frame points -> (N, 3) camera-frame points (m)."""
        return (np.asarray(p_body) - self.t_bc) @ self.R_bc

    def project_body(self, p_body: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Body points (N, 3) -> (u, v, Z_c). Points behind the camera get Z_c <= 0."""
        pc = self.body_to_cam(p_body)
        zc = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * pc[:, 0] / zc + self.cx
            v = self.fy * pc[:, 1] / zc + self.cy
        return u, v, zc

    # ------------------------------------------------------------------ ground rays
    def plane_depth(self, rays: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Optical depth Z_c at which body-frame rays hit the plane z = a x + b y + c.

        ``rays``: (..., 3) body-frame rays with unit z_c; a, b, c broadcastable to rays[..., 0].
        Returns +inf where the ray does not hit the plane in front of the camera.
        """
        tx, ty, tz = self.t_bc
        num = a * tx + b * ty + c - tz
        den = rays[..., 2] - a * rays[..., 0] - b * rays[..., 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            zc = num / den
        return np.where((zc > 0) & np.isfinite(zc), zc, np.inf)

    def flat_ground_depth(self) -> np.ndarray:
        """(H, W) optical depth of the nominal flat ground plane z = 0 (inf above horizon)."""
        return self.plane_depth(self.rays_body.astype(np.float64), 0.0, 0.0, 0.0)
