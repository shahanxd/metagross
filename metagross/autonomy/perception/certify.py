"""Ground certification: seen GROUND counts as *certified* only where a design-width ditch
would already have been detectable from the current viewpoint.

Thesis: *unknown is never free* - a cell that looks like ground is not proof of ground if a
ditch there would still be invisible. Matthies & Rankin (2003): a ditch of width ``w`` whose
near lip is at horizontal range ``R`` from a camera at height ``H`` above it subtends
``~ H w / (R (R + w))`` rad of elevation; with ``n`` pixels needed on target
(``defaults.MIN_PIXELS_ON_TARGET``) and focal length ``fx`` (px) it is detectable up to

    r_det(w, H) = (-w + sqrt(w^2 + 4 H w fx / n)) / 2      [m, horizontal, from the camera]

(identical to ``metagross.eval.theory.ditch_detection_range_m``; re-implemented here because
the onboard stack must not import ``metagross.eval``; a unit test checks the two agree).

Per BEV cell (egocentric grid of :mod:`metagross.autonomy.perception.bev`, body frame, metres):
``H = z_cam - z_cell`` (camera height above that cell's terrain: measured mean height where
the cell has points, else the ground model), ``R`` = horizontal distance camera -> cell centre.

``certified_local = (state == GROUND) & (H > 0) & (R <= r_det(DESIGN_DITCH_WIDTH_M, H))``.

Uphill cells (smaller ``H``) certify closer, downhill cells farther, exactly as the geometry
of the missing-ground signature dictates.
"""

from __future__ import annotations

import cv2
import numpy as np

from metagross.autonomy.perception.bev import BevSpec
from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.config import defaults
from metagross.contracts.messages import CellState

MIN_CAM_HEIGHT_M = 0.05  # cells this close below the camera height (or above it) are never certified
CERT_CLEARANCE_M = 0.3  # GROUND this close (Chebyshev, m) to lethal evidence is not certified (3 cells = vehicle half width)
LETHAL_EVIDENCE = np.array([CellState.POSITIVE, CellState.DEPRESSION, CellState.DITCH_CANDIDATE], dtype=np.uint8)


def ditch_detection_range_m(w_m: float, h_m: np.ndarray | float, fx_px: float,
                            n_px: float = float(defaults.MIN_PIXELS_ON_TARGET)) -> np.ndarray:
    """Horizontal range (m) up to which a ditch of width ``w_m`` seen from height ``h_m`` (m)
    covers >= ``n_px`` pixels; 0 where ``h_m <= 0``. Vectorised over ``h_m``."""
    h = np.maximum(np.asarray(h_m, dtype=np.float64), 0.0)
    k = h * w_m * fx_px / n_px
    return (-w_m + np.sqrt(w_m * w_m + 4.0 * k)) / 2.0


class GroundCertifier:
    """Pre-computes the camera -> cell horizontal ranges of one BEV grid / camera mount."""

    def __init__(self, geom: CameraGeometry, spec: BevSpec, design_width_m: float = defaults.DESIGN_DITCH_WIDTH_M) -> None:
        self.geom = geom
        self.w = float(design_width_m)
        X, Y = spec.centres()
        tx, ty, _ = geom.t_bc
        self.r_cam = np.hypot(X - tx, Y - ty).astype(np.float32)  # (nx, ny) m
        k = 2 * int(round(CERT_CLEARANCE_M / spec.res)) + 1
        self._kernel = np.ones((k, k), np.uint8)

    def r_det_m(self, cam_height_m: float) -> float:
        """Design-ditch detection range (m, horizontal from the camera) on level ground."""
        return float(ditch_detection_range_m(self.w, cam_height_m, self.geom.fx))

    def certify(self, state: np.ndarray, cell_z: np.ndarray) -> np.ndarray:
        """(nx, ny) bool certified ground. ``cell_z``: terrain height per cell (m, body frame;
        NaN allowed only where the state is not GROUND). GROUND within ``CERT_CLEARANCE_M`` of
        lethal evidence (POSITIVE / DEPRESSION / DITCH_CANDIDATE) is not certified: hazard
        boundaries are only known to about a cell, and a lip or rock rim is not proof of ground."""
        h = self.geom.t_bc[2] - np.nan_to_num(cell_z, nan=self.geom.t_bc[2])
        r_det = ditch_detection_range_m(self.w, h, self.geom.fx)
        cert = (state == CellState.GROUND) & (h > MIN_CAM_HEIGHT_M) & (self.r_cam <= r_det)
        lethal = np.isin(state, LETHAL_EVIDENCE).astype(np.uint8)
        if lethal.any():
            cert &= cv2.dilate(lethal, self._kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) == 0
        return cert
