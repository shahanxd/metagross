"""Message contracts between the simulated world, the onboard autonomy stack,
the operator console and the evaluator.

These dataclasses are the ONLY things that cross process boundaries. The
autonomy process must be able to run on a real UGV with nothing but these
inputs, so nothing here may carry simulator ground truth.

Frames
------
* A-frame (mission frame): origin at launch point A, x forward along the
  launch heading, y left, z up. Goals are expressed in this frame.
* Body frame: x forward, y left, z up, origin at the centre of the wheelbase
  on the ground plane.
* Camera frame: OpenCV convention, x right, y down, z forward (optical axis).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

CONTRACT_VERSION = "1.0"


class DriveMode(str, Enum):
    """Health / supervisory state reported by the autonomy stack."""

    NOMINAL = "NOMINAL"
    CAUTION = "CAUTION"
    DEGRADED = "DEGRADED"
    STOP_AND_LOOK = "STOP_AND_LOOK"
    SAFE_STOP = "SAFE_STOP"
    ARRIVED = "ARRIVED"
    HOLD = "HOLD"  # operator hold


class OperatorAction(str, Enum):
    GO = "GO"
    HOLD = "HOLD"
    RESUME = "RESUME"
    ESTOP = "ESTOP"


@dataclass(slots=True, frozen=True)
class StereoCalibration:
    """Rectified stereo rig calibration (known a priori on real hardware)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float
    # 4x4 homogeneous transform: point in camera(left) frame -> body frame.
    T_body_cam: np.ndarray

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


@dataclass(slots=True, frozen=True)
class VehicleSpec:
    """Nominal platform parameters the autonomy is allowed to know."""

    track_width_m: float  # B, lateral distance between wheel centre lines
    wheel_radius_m: float
    length_m: float
    width_m: float
    ground_clearance_m: float
    max_speed_mps: float
    max_yaw_rate_rps: float
    max_accel_mps2: float
    max_wheel_rad_s: float
    encoder_ticks_per_rev: int


@dataclass(slots=True, frozen=True)
class MissionSpec:
    """Mission handed to the UGV by the operator before GO."""

    mission_id: str
    goal_xy_a: tuple[float, float]  # goal in A-frame metres (heading-init error already baked in)
    success_radius_m: float
    timeout_s: float


@dataclass(slots=True)
class SensorFrame:
    """One synchronized sensor sample delivered to the onboard stack.

    `left_rgb` / `right_gray` are rectified images. In the Tier-0 synthetic depth
    mode, `disparity` is filled by the sensor model instead of images and
    `sensor_mode` says so; perception must honour whichever is present.

    Tier-0 disparity units: float32 **pixels at the calibrated resolution** (the
    StereoCalibration sent at reset, 640 x 400), left-image referenced, so depth is
    ``Z = fx * baseline_m / d`` metres along the optical axis; ``<= 0`` means invalid
    (no return / dropout / beyond range). It already contains the sensor noise model
    (sub-pixel noise, gross mismatches, dropouts) - treat it like an SGBM output.
    With no images, VO cannot run: the localiser reports ``health['vo_available'] = 0``
    and localisation is wheel + gyro dead reckoning (not a VO integrity fault).
    """

    t: float
    seq: int
    left_rgb: Optional[np.ndarray]  # (H, W, 3) uint8, RGB
    right_gray: Optional[np.ndarray]  # (H, W) uint8
    wheel_angle_l_rad: float  # cumulative encoder angle, left side
    wheel_angle_r_rad: float  # cumulative encoder angle, right side
    gyro_z_rps: float  # yaw rate with bias + noise
    sensor_mode: str = "stereo"  # "stereo" | "tier0_disparity"
    disparity: Optional[np.ndarray] = None  # (H, W) float32 px, only in tier0 mode, <=0 invalid


@dataclass(slots=True)
class WheelCmd:
    """Actuator command produced by the onboard stack for one control tick."""

    t: float
    seq: int
    omega_l_rad_s: float
    omega_r_rad_s: float
    mode: DriveMode
    compute_ms: float  # measured wall-clock compute for this tick (for latency injection)


@dataclass(slots=True)
class Telemetry:
    """Low-bandwidth downlink packet content (encoded by autonomy.link.codec)."""

    t: float
    seq: int
    pose_xy_yaw: tuple[float, float, float]  # A-frame
    pos_sigma_m: float
    mode: DriveMode
    reason: str  # short human-readable reason code, e.g. "GLARE sat=31%"
    v_cap_mps: float
    r_cert_m: float
    speed_mps: float
    waypoints_xy: list[tuple[float, float]] = field(default_factory=list)  # next <=5, A-frame
    costmap_u4: Optional[np.ndarray] = None  # (64, 64) uint8 in [0, 15], ego-centred, 0.25 m cells (see COSTMAP_U4_CODES)
    health: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class OperatorCmd:
    t: float
    action: OperatorAction
    goal_xy_a: Optional[tuple[float, float]] = None


# ---------------------------------------------------------------------------
# Observation states for BEV cells. Shared by perception, planning, operator
# console, video and deck (same colour key everywhere).
# ---------------------------------------------------------------------------


class CellState(int, Enum):
    UNSEEN = 0  # never resolved by the cameras
    GROUND = 1  # observed ground, certified drivable (cost from slope/roughness/semantics)
    POSITIVE = 2  # step / rock / trunk above clearance: lethal
    DEPRESSION = 3  # points measured below local ground: lethal
    DITCH_CANDIDATE = 4  # missing ground, reappears near lip height
    CREST_SHADOW = 5  # missing ground, reappears well below / slopes away
    OCCLUDED = 6  # hidden behind an above-ground object: unknown, never lethal
    WATER = 7  # semantic water / mud
    DYNAMIC = 8  # newly occupied where free space was observed


# Colour key (RGB) used by console, video and deck. Keep in sync with deck tokens.
CELL_COLORS: dict[CellState, tuple[int, int, int]] = {
    CellState.UNSEEN: (203, 208, 214),
    CellState.GROUND: (34, 160, 90),
    CellState.POSITIVE: (220, 50, 47),
    CellState.DEPRESSION: (220, 50, 47),
    CellState.DITCH_CANDIDATE: (200, 40, 160),
    CellState.CREST_SHADOW: (240, 160, 30),
    CellState.OCCLUDED: (150, 156, 164),
    CellState.WATER: (20, 140, 190),
    CellState.DYNAMIC: (250, 90, 20),
}

SENSOR_FRAME_FIELDS = frozenset(SensorFrame.__slots__)

# Telemetry.costmap_u4 code table (documentation of metagross.autonomy.link.codec U4_*).
# Layout: 64 x 64 cells of 0.25 m, body-aligned and vehicle-centred; row 0 = farthest FORWARD,
# column 0 = farthest LEFT (i.e. already in picture orientation). Lethal codes (>= 13) are
# max-dilated by one rolling-map cell before resampling so they cannot be dropped.
COSTMAP_U4_CODES: dict[int, str] = {
    0: "UNSEEN",
    1: "GROUND cost bin 0 (cost < 1/8)", 2: "GROUND cost bin 1", 3: "GROUND cost bin 2", 4: "GROUND cost bin 3",
    5: "GROUND cost bin 4", 6: "GROUND cost bin 5", 7: "GROUND cost bin 6", 8: "GROUND cost bin 7 (cost >= 7/8)",
    9: "OCCLUDED",
    10: "CREST_SHADOW",
    11: "WATER",
    12: "DITCH_CANDIDATE (unconfirmed, not lethal, not certified)",
    13: "DYNAMIC (lethal)",
    14: "DITCH (confirmed k-of-3, lethal)",
    15: "LETHAL (POSITIVE / DEPRESSION)",
}
