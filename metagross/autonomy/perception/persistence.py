"""World-frame persistence of per-frame ditch candidates.

A trench is fixed in the world; a stereo mismatch on repetitive texture (gravel, grass) moves or
vanishes when the viewpoint changes by a few decimetres. A DITCH_CANDIDATE cell of the current
frame is *confirmed* when a raw candidate of one of the last ``PERSIST_HISTORY`` frames lies
within ``PERSIST_RADIUS_M`` of it in the odometry frame (pose from the localiser, ``(x, y, yaw)``
in metres / radians). Unconfirmed candidates are reported as UNSEEN - unknown, never certified,
never free - instead of lethal. The first frame (empty history) is passed through unfiltered, so
single-frame use (tests, figures) is unchanged. Cost: one rasterisation + one dilation per frame.

Grids follow the egocentric BEV convention of :mod:`metagross.autonomy.perception.bev`.
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from metagross.autonomy.perception.bev import BevSpec

PERSIST_HISTORY = 2  # previous frames searched for support (0.4 s at 5 Hz)
PERSIST_RADIUS_M = 0.25  # support radius (m): pose drift over 0.4 s + one cell + lip-range noise


class DitchPersistence:
    """Keeps the raw ditch candidates of recent frames in the odometry frame (see module doc)."""

    def __init__(self, spec: BevSpec, history: int = PERSIST_HISTORY, radius_m: float = PERSIST_RADIUS_M) -> None:
        self.spec = spec
        X, Y = spec.centres()
        self._X, self._Y = X, Y
        self._hist: deque[np.ndarray] = deque(maxlen=max(int(history), 1))
        k = 2 * int(math.ceil(radius_m / spec.res)) + 1
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def reset(self) -> None:
        self._hist.clear()

    def filter(self, ditch: np.ndarray, pose_xy_yaw: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
        """Split raw candidates (nx, ny) bool into (confirmed, unconfirmed) and remember them."""
        px, py, yaw = (float(v) for v in pose_xy_yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        X, Y = self._X[ditch], self._Y[ditch]
        world = np.stack([px + c * X - s * Y, py + s * X + c * Y], axis=1)
        if not self._hist:
            self._hist.append(world)
            return ditch.copy(), np.zeros_like(ditch)
        prev = np.concatenate(list(self._hist), axis=0)
        self._hist.append(world)
        if prev.size == 0 or not ditch.any():
            return np.zeros_like(ditch), ditch.copy()
        dx, dy = prev[:, 0] - px, prev[:, 1] - py
        idx = self.spec.flat_index(c * dx + s * dy, -s * dx + c * dy)
        sup = np.zeros(self.spec.n_cells, np.uint8)
        sup[idx[idx >= 0]] = 1
        sup = cv2.dilate(sup.reshape(self.spec.shape), self._kernel) > 0
        return ditch & sup, ditch & ~sup
