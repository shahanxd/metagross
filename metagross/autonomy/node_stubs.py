"""Clearly-named fallback implementations used by :mod:`metagross.autonomy.node` ONLY when
the real module is absent (or when a config explicitly forces them for plumbing tests).

* :class:`StubWheelGyroLocalizer` — dead reckoning from wheel encoders (distance) and the
  gyro (yaw). No visual odometry, no integrity monitor. Uses only SensorFrame fields.
* :class:`StubBlindPerception` — reports every BEV cell as UNSEEN. With the seen-ground
  policy the vehicle therefore never leaves its launch apron: failing safe.

Both log a WARNING on construction so a run can never silently use them.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Optional

import numpy as np

from metagross.config.defaults import BEV_LOCAL_SIZE_M, BEV_RES_M, CHI_NOMINAL, VEHICLE
from metagross.contracts.messages import CellState, SensorFrame, StereoCalibration, VehicleSpec

LOG = logging.getLogger(__name__)

ODOM_SIGMA_PER_M = 0.02  # assumed 2 % of distance travelled position drift for wheel odometry


class StubWheelGyroLocalizer:
    """LocalizerProto stub: wheel-encoder distance + gyro-integrated yaw, A-frame from the first frame."""

    is_stub = True

    def __init__(self, calib: Optional[StereoCalibration] = None, vehicle: VehicleSpec = VEHICLE, config: Optional[dict] = None) -> None:
        LOG.warning("USING STUB LOCALIZER (wheel+gyro dead reckoning, no VO, no integrity monitor)")
        self.vehicle = vehicle
        self._prev: Optional[SensorFrame] = None
        self.x = self.y = self.yaw = 0.0
        self.dist = 0.0

    def update(self, frame: SensorFrame, disparity: Optional[np.ndarray]) -> dict[str, Any]:
        t0 = time.perf_counter()
        if self._prev is not None:
            dt = max(frame.t - self._prev.t, 0.0)
            r = self.vehicle.wheel_radius_m
            ds = r * ((frame.wheel_angle_l_rad - self._prev.wheel_angle_l_rad) + (frame.wheel_angle_r_rad - self._prev.wheel_angle_r_rad)) / 2.0
            dyaw = 0.5 * (frame.gyro_z_rps + self._prev.gyro_z_rps) * dt  # trapezoidal gyro integration
            mid = self.yaw + 0.5 * dyaw
            self.x += ds * math.cos(mid)
            self.y += ds * math.sin(mid)
            self.yaw = math.atan2(math.sin(self.yaw + dyaw), math.cos(self.yaw + dyaw))
            self.dist += abs(ds)
        self._prev = frame
        return {
            "pose_xy_yaw": (self.x, self.y, self.yaw),
            "pos_sigma_m": ODOM_SIGMA_PER_M * self.dist,
            "health": {"p_fail": 0.0, "stub": 1.0},
            "vo_ok": False,
            "slip": False,
            "chi_hat": CHI_NOMINAL,
            "timings_ms": {"stub_localizer": (time.perf_counter() - t0) * 1e3},
        }


class StubBlindPerception:
    """PerceptionProto stub: sees nothing (all UNSEEN). The vehicle stays on its launch apron."""

    is_stub = True

    def __init__(self, calib: Optional[StereoCalibration] = None, config: Optional[dict] = None) -> None:
        LOG.warning("USING STUB PERCEPTION (blind: every BEV cell UNSEEN)")
        self.shape = (int(round(BEV_LOCAL_SIZE_M[0] / BEV_RES_M)), int(round(BEV_LOCAL_SIZE_M[1] / BEV_RES_M)))

    def process(self, frame: SensorFrame, pose_xy_yaw: tuple[float, float, float]) -> dict[str, Any]:
        return {
            "disparity": frame.disparity,
            "cell_state_local": np.full(self.shape, int(CellState.UNSEEN), np.uint8),
            "cost_local": np.zeros(self.shape, np.float32),
            "height_local": np.full(self.shape, np.nan, np.float32),
            "semantic_mask": None,
            "missing_ground_mask": None,
            "r_vis_m": 0.0,
            "timings_ms": {},
        }
