"""Single source of truth for platform, camera and timing parameters.

Every module (sim, renderer, autonomy, eval, video) reads these values; do not
hard-code duplicates elsewhere.
"""

from __future__ import annotations

import math

import numpy as np

from metagross.contracts.messages import StereoCalibration, VehicleSpec

# --------------------------------------------------------------------------- timing
PHYSICS_HZ = 50.0
CAMERA_HZ_BATCH = 5.0
CAMERA_HZ_DEMO = 10.0
ACTUATOR_LAG_S = 0.2  # first-order wheel-speed response time constant
GT_LOG_HZ = 30.0
WHEEL_CMD_TIMEOUT_S = 0.5  # world-side actuator watchdog: no new WheelCmd for this long -> wheels commanded to 0

# --------------------------------------------------------------------------- camera
IMG_W, IMG_H = 640, 400
HFOV_DEG = 72.0
FX = (IMG_W / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))  # ~440.4 px
FY = FX
CX, CY = (IMG_W - 1) / 2.0, (IMG_H - 1) / 2.0
BASELINE_M = 0.12  # ZED 2i class
CAM_HEIGHT_M = 0.9  # optical centre above ground
CAM_FORWARD_M = 0.30  # ahead of body origin
CAM_PITCH_DEG = -12.0  # negative = looking down
CAM_MAX_RANGE_M = 12.0


def camera_extrinsics() -> np.ndarray:
    """T_body_cam: maps points from left-camera frame (x right, y down, z fwd) to body frame
    (x fwd, y left, z up)."""
    # Rotation cam->body for a level camera: z_cam -> x_body, x_cam -> -y_body, y_cam -> -z_body
    R_level = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    p = math.radians(-CAM_PITCH_DEG)  # positive pitch-down angle
    # Pitch down about body y-axis (left): rotate forward axis toward -z
    R_pitch = np.array([[math.cos(p), 0.0, math.sin(p)], [0.0, 1.0, 0.0], [-math.sin(p), 0.0, math.cos(p)]])
    T = np.eye(4)
    T[:3, :3] = R_pitch @ R_level
    # Left camera sits baseline/2 to the left of the rig centre.
    T[:3, 3] = [CAM_FORWARD_M, BASELINE_M / 2.0, CAM_HEIGHT_M]
    return T


def stereo_calibration() -> StereoCalibration:
    return StereoCalibration(
        width=IMG_W,
        height=IMG_H,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        baseline_m=BASELINE_M,
        T_body_cam=camera_extrinsics(),
    )


# --------------------------------------------------------------------------- vehicle
VEHICLE = VehicleSpec(
    track_width_m=0.50,
    wheel_radius_m=0.13,
    length_m=0.80,
    width_m=0.60,
    ground_clearance_m=0.15,
    max_speed_mps=2.0,  # Scout-Mini class platform cap
    max_yaw_rate_rps=1.2,
    max_accel_mps2=1.0,
    max_wheel_rad_s=2.0 / 0.13 * 1.6,
    encoder_ticks_per_rev=4096,
)

# Skid-steer effective-track factor chi (true value sampled per terrain in sim; autonomy estimates).
CHI_NOMINAL = 1.4

# --------------------------------------------------------------------------- mapping
BEV_RES_M = 0.10  # local grid resolution
BEV_LOCAL_SIZE_M = (16.0, 12.0)  # forward, lateral extent of egocentric grid
# Egocentric BEV grid convention (perception -> planning -> console / video), body frame metres:
#   grid[i, j] (shape (BEV_NX, BEV_NY) = (160, 120)), cell centre
#   x = BEV_LOCAL_X_MIN_M + (i + 0.5) * BEV_RES_M   (row index i grows FORWARD),
#   y = BEV_LOCAL_Y_MIN_M + (j + 0.5) * BEV_RES_M   (column index j grows to the LEFT).
# Picture orientation (forward up, vehicle-left on the left) is grid[::-1, ::-1].
BEV_LOCAL_X_MIN_M = -2.0  # grid starts 2 m behind the body origin (footprint + margin)
BEV_LOCAL_X_MAX_M = BEV_LOCAL_X_MIN_M + BEV_LOCAL_SIZE_M[0]  # 14.0 m
BEV_LOCAL_Y_MIN_M = -BEV_LOCAL_SIZE_M[1] / 2.0  # -6.0 m (right edge)
BEV_LOCAL_Y_MAX_M = BEV_LOCAL_SIZE_M[1] / 2.0  # +6.0 m (left edge)
BEV_NX = int(round(BEV_LOCAL_SIZE_M[0] / BEV_RES_M))  # rows (forward)
BEV_NY = int(round(BEV_LOCAL_SIZE_M[1] / BEV_RES_M))  # columns (lateral)
BEV_INDEX_ORDER = "ij: i forward (x), j left (y)"
MAP_RES_M = 0.20  # rolling odometry-frame map
MAP_SIZE_M = 80.0
STEP_LETHAL_M = 0.15  # = ground clearance
SLOPE_LETHAL_DEG = 20.0
DEPRESSION_LETHAL_M = 0.15

# --------------------------------------------------------------------------- governor
BRAKE_DECEL_MPS2 = 1.5
GOVERNOR_MARGIN_M = 0.5
DESIGN_DITCH_WIDTH_M = 0.3
MIN_PIXELS_ON_TARGET = 6  # Matthies & Rankin detectability criterion

# --------------------------------------------------------------------------- link
TELEMETRY_HZ = 2.0
LINK_KBPS = 9.6
LINK_LOSS = 0.2
LINK_LATENCY_S = 0.4
