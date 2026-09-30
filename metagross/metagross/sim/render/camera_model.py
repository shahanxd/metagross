"""Reference pinhole + pose model for the stereo renderer.

This is the Python ground truth that the Three.js renderer is tested against
(``tests/test_render_projection.py``): a world point projected here must land on
the same pixel the WebGL pipeline draws it on, to within 0.5 px.

Frames
------
* World: x east, y north, z up (metres). The Three.js scene is built directly in
  this z-up frame; the renderer never uses Three's y-up convention for geometry.
* Body: x forward, y left, z up; origin at the wheelbase centre on the ground.
* Camera: OpenCV, x right, y down, z forward (optical axis). Pixel (u, v) has
  its centre at integer coordinates, u to the right, v down; ``cx = (W-1)/2``
  therefore means "optical axis through the exact image centre".

Pose convention
---------------
``pose = [x, y, z, roll, pitch, yaw]`` (metres, radians) gives the body origin in
world coordinates and the body attitude as ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``
(intrinsic Z-Y'-X'', right-handed). With body y pointing left, a positive pitch
rotates the nose DOWN (REP-103 / ROS convention).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from metagross.config import defaults
from metagross.contracts.messages import StereoCalibration

# OpenCV camera (x right, y down, z fwd) -> OpenGL camera (x right, y up, z back).
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])

# Clip planes of the stereo cameras (metres). Near is below the smallest
# obstacle distance the vehicle can physically reach; far bounds the fog/sky.
NEAR_M = 0.05
FAR_M = 200.0


def rot_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (radians)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def pose_to_T_world_body(pose: np.ndarray | list[float]) -> np.ndarray:
    """4x4 homogeneous transform body -> world for ``pose = [x,y,z,roll,pitch,yaw]``."""
    p = np.asarray(pose, dtype=np.float64).reshape(6)
    T = np.eye(4)
    T[:3, :3] = rot_zyx(p[3], p[4], p[5])
    T[:3, 3] = p[:3]
    return T


def T_world_cam(
    pose: np.ndarray | list[float],
    eye: Literal["left", "right"] = "left",
    T_body_cam: np.ndarray | None = None,
    baseline_m: float = defaults.BASELINE_M,
) -> np.ndarray:
    """4x4 transform OpenCV camera -> world for the left or right camera.

    The right camera is the left camera translated by ``+baseline_m`` along the
    left camera's +x axis (rectified rig, OpenCV convention).
    """
    Tbc = defaults.camera_extrinsics() if T_body_cam is None else np.asarray(T_body_cam, dtype=np.float64)
    T = pose_to_T_world_body(pose) @ Tbc
    if eye == "right":
        shift = np.eye(4)
        shift[0, 3] = baseline_m
        T = T @ shift
    elif eye != "left":
        raise ValueError(f"eye must be 'left' or 'right', got {eye!r}")
    return T


def projection_matrix_gl(
    fx: float, fy: float, cx: float, cy: float, width: int, height: int, near: float = NEAR_M, far: float = FAR_M
) -> np.ndarray:
    """OpenGL clip-space projection reproducing the OpenCV pinhole exactly.

    Derivation: with pixel centres at integer (u, v), NDC x = -1 is the left edge of
    pixel 0, i.e. ``u = (x_ndc + 1) * W / 2 - 0.5`` and ``v = (1 - y_ndc) * H / 2 - 0.5``
    (v down). Solving ``u = fx X/Z + cx`` for a GL camera (x_gl = X, y_gl = -Y,
    z_gl = -Z) gives the terms below. Row-major (numpy) layout.
    """
    P = np.zeros((4, 4))
    P[0, 0] = 2.0 * fx / width
    P[0, 2] = 1.0 - 2.0 * (cx + 0.5) / width
    P[1, 1] = 2.0 * fy / height
    P[1, 2] = 2.0 * (cy + 0.5) / height - 1.0
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


def project_world_points(
    points_world: np.ndarray,
    pose: np.ndarray | list[float],
    calib: StereoCalibration | None = None,
    eye: Literal["left", "right"] = "left",
) -> tuple[np.ndarray, np.ndarray]:
    """Project (N,3) world points (metres) into the given camera.

    Returns ``(uv (N,2) px, z (N,) metres along the optical axis)``. Points with
    ``z <= 0`` are behind the camera; their ``uv`` is meaningless.
    """
    cal = defaults.stereo_calibration() if calib is None else calib
    T = T_world_cam(pose, eye, cal.T_body_cam, cal.baseline_m)
    T_cam_world = np.linalg.inv(T)
    P = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    Pc = P @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
    z = Pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cal.fx * Pc[:, 0] / z + cal.cx
        v = cal.fy * Pc[:, 1] / z + cal.cy
    return np.stack([u, v], axis=1), z


def calib_to_js(calib: StereoCalibration | None = None) -> dict:
    """Serialisable camera description handed to ``window.configureCameras``.

    ``T_body_cam`` is sent row-major (16 floats). Keeping this the only path by
    which numbers reach the browser means the JS side never duplicates defaults.
    """
    cal = defaults.stereo_calibration() if calib is None else calib
    return {
        "width": int(cal.width),
        "height": int(cal.height),
        "fx": float(cal.fx),
        "fy": float(cal.fy),
        "cx": float(cal.cx),
        "cy": float(cal.cy),
        "baseline": float(cal.baseline_m),
        "T_body_cam": [float(v) for v in np.asarray(cal.T_body_cam, dtype=np.float64).reshape(16)],
        "near": NEAR_M,
        "far": FAR_M,
    }
