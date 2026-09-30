"""Module-level interfaces. Implementations live in their own packages; these
Protocols pin the call signatures so parallel work integrates cleanly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np

from metagross.contracts.messages import (
    MissionSpec,
    SensorFrame,
    StereoCalibration,
    Telemetry,
    VehicleSpec,
    WheelCmd,
)


# ----------------------------------------------------------------- autonomy side
@dataclass
class DebugBundle:
    """Per-tick internals the autonomy writes to ITS OWN log (for video/eval
    replay). Never fed back into the world."""

    t: float
    pose_xy_yaw: tuple[float, float, float]
    # Egocentric BEV arrays follow defaults.BEV_LOCAL_* (grid[i, j]: i forward from x = -2 m,
    # j LEFT from y = -6 m, 0.1 m cells, shape (160, 120)). extras may hold 'left_rgb'
    # ((200, 320, 3) uint8, every debug_image_every_n ticks in stereo mode) and 'speed_meas' (m/s).
    cell_state_local: Optional[np.ndarray] = None  # (Hf, Wl) uint8 CellState, egocentric BEV
    cost_local: Optional[np.ndarray] = None  # (Hf, Wl) float32 [0,1], inf/1.0 lethal
    semantic_mask: Optional[np.ndarray] = None  # (H, W) uint8 class ids (5-class scheme)
    missing_ground_mask: Optional[np.ndarray] = None  # (H, W) bool, image-space ditch/crest pixels
    disparity: Optional[np.ndarray] = None  # (H, W) float32
    rollouts_xy: Optional[np.ndarray] = None  # (K_vis, T, 2) sampled MPPI rollouts, body frame
    plan_xy: Optional[np.ndarray] = None  # (T, 2) chosen trajectory, A-frame
    global_path_xy: Optional[np.ndarray] = None  # (N, 2) cost-to-go descent path, A-frame
    v_cap_mps: float = 0.0
    r_cert_m: float = 0.0
    health: dict[str, float] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


class AutonomyStackProto(Protocol):
    def reset(self, mission: MissionSpec, calib: StereoCalibration, vehicle: VehicleSpec, config: dict) -> None: ...

    def step(self, frame: SensorFrame) -> tuple[WheelCmd, Optional[Telemetry], DebugBundle]: ...


class PerceptionProto(Protocol):
    """Stereo + semantics -> egocentric BEV observation states and costs."""

    def process(self, frame: SensorFrame, pose_xy_yaw: tuple[float, float, float]) -> dict[str, Any]:
        """Returns at least: 'disparity', 'cell_state_local', 'cost_local', 'height_local',
        'semantic_mask' (or None), 'missing_ground_mask', 'r_vis_m', 'timings_ms'."""
        ...


class LocalizerProto(Protocol):
    def update(self, frame: SensorFrame, disparity: Optional[np.ndarray]) -> dict[str, Any]:
        """Returns 'pose_xy_yaw', 'pos_sigma_m', 'health' (dict of features + 'p_fail'),
        'vo_ok', 'slip', 'chi_hat', 'timings_ms'."""
        ...


class SegmenterProto(Protocol):
    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """rgb (H,W,3) uint8 -> (class_ids (H,W) uint8, entropy (H,W) float32)."""
        ...


# ----------------------------------------------------------------- world side
class RendererProto(Protocol):
    """World-side renderer (``metagross.sim.render.bridge.ThreeRenderer``).

    Pose convention of ``state`` (built by ``metagross.sim.world.World.render_state``):

    * ``pose = [x, y, z, roll, pitch, yaw]`` is the **body** pose (origin at the wheelbase
      centre on the ground) in the scenario **world** frame (x east, y north, z up, metres);
      angles are REP-103 (roll about x, pitch about y, yaw about z; R = Rz(yaw) Ry(pitch)
      Rx(roll), radians). Positive pitch = nose DOWN (right-hand rule about +y, left).
    * ``T_world_body`` (4x4 nested lists) is that pose as a matrix, and
      ``T_world_cam = T_world_body @ config.defaults.camera_extrinsics()`` maps LEFT-camera
      points (OpenCV axes: x right, y down, z forward) to the world. The right camera is
      ``baseline_m`` along the left camera's +x. Renderers should use ``T_world_cam`` rather
      than re-deriving the extrinsics.
    * Images are rectified, row 0 at the top, intrinsics = ``defaults.stereo_calibration()``.
    """

    def load_scenario(self, scenario: dict) -> None: ...

    def render_stereo(self, state: dict) -> tuple[np.ndarray, np.ndarray]:
        """state: {'pose': [x,y,z,roll,pitch,yaw], 't': float, 'dynamic': [...], 'lighting': {...},
        'T_world_body': 4x4, 'T_world_cam': 4x4} -> (left_rgb (H,W,3) uint8, right_gray (H,W) uint8)."""
        ...

    def render_gt(self, state: dict) -> dict[str, np.ndarray]:
        """Evaluator-only passes: {'depth': (H,W) float32 metres, 'semantic': (H,W) uint8}."""
        ...

    def render_chase(self, state: dict, width: int, height: int) -> np.ndarray: ...

    def close(self) -> None: ...


# Five-class terrain scheme shared by training, sim GT semantic pass and perception
# (matches GAIA-URJC OFFROAD5 label ids).
SEM_CLASSES = {0: "background/sky", 1: "obstacle", 2: "water/mud", 3: "unstable (grass/dirt)", 4: "stable (asphalt/concrete/gravel path)"}
SEM_COST = {0: 0.0, 1: 0.8, 2: 0.95, 3: 0.2, 4: 0.0}
SEM_MU = {0: 0.4, 1: 0.4, 2: 0.2, 3: 0.4, 4: 0.6}  # braking friction proxy for governor
