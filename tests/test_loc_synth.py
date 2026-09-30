"""Synthetic stereo scene for localisation tests (no simulator, no dataset).

The scene is a textured ground plane (camera frame y = +cam_height, i.e. below
the camera) closed by a textured wall at z = wall_z. Images are ray-cast per
pixel, so the geometry and disparity are exact; this lets the VO / localiser
tests check recovered motion against a known answer.

Camera frame: OpenCV (x right, y down, z forward). Poses ``T_w_c`` map camera
points into the world frame, which is the camera frame of the first view.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

W, H = 320, 240
FX = FY = 220.0
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0
BASELINE_M = 0.12
CAM_HEIGHT_M = 0.9
WALL_Z_M = 14.0
TEX_RES_M = 0.01  # metres per texture pixel
TEX_SIZE = 2048


def make_texture(seed: int = 0) -> np.ndarray:
    """Multi-scale random texture (uint8), rich in corners."""
    rng = np.random.default_rng(seed)
    tex = np.zeros((TEX_SIZE, TEX_SIZE), np.float32)
    for sigma, amp in ((1.0, 0.5), (3.0, 0.8), (9.0, 1.0)):
        n = rng.standard_normal((TEX_SIZE, TEX_SIZE)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        tex += amp * n / (n.std() + 1e-6)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    return (tex * 255).astype(np.uint8)


def K() -> np.ndarray:
    return np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])


def render(T_w_c: np.ndarray, tex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ray-cast the scene from camera pose ``T_w_c``.

    Returns (gray image (H, W) uint8, depth z in the camera frame (H, W) float32 m).
    """
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    d_c = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u)], axis=-1)
    R, c = T_w_c[:3, :3], T_w_c[:3, 3]
    d_w = d_c @ R.T
    s_ground = np.where(d_w[..., 1] > 1e-6, (CAM_HEIGHT_M - c[1]) / np.maximum(d_w[..., 1], 1e-9), np.inf)
    s_wall = np.where(d_w[..., 2] > 1e-6, (WALL_Z_M - c[2]) / np.maximum(d_w[..., 2], 1e-9), np.inf)
    s = np.minimum(s_ground, s_wall)
    hit = c + s[..., None] * d_w
    on_ground = s_ground <= s_wall
    # texture coordinates: ground uses (x, z), wall uses (x, y)
    tu = hit[..., 0] / TEX_RES_M + TEX_SIZE / 2
    tv = np.where(on_ground, hit[..., 2] / TEX_RES_M, (hit[..., 1] + 6.0) / TEX_RES_M)
    img = cv2.remap(tex, tu.astype(np.float32), (tv % TEX_SIZE).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    depth = (s * d_c[..., 2]).astype(np.float32)  # d_c z-component is 1 -> depth = s
    return img, depth


def disparity_from_depth(depth: np.ndarray) -> np.ndarray:
    return (FX * BASELINE_M / np.maximum(depth, 1e-3)).astype(np.float32)


def pose(x: float, y: float, z: float, yaw_rad: float = 0.0) -> np.ndarray:
    """Camera pose with rotation about the camera y axis (down) = vehicle yaw (sign flipped)."""
    T = np.eye(4)
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    # rotation about camera -y (i.e. about "up"): positive yaw turns the optical axis to the left (-x)
    T[:3, :3] = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    T[:3, 3] = [x, y, z]
    return T


def test_render_is_consistent():
    tex = make_texture(0)
    img, depth = render(np.eye(4), tex)
    assert img.shape == (H, W) and img.dtype == np.uint8
    assert np.isfinite(depth).all() and depth.min() > 0.5 and depth.max() <= WALL_Z_M + 1e-3
    assert img.std() > 20  # textured
    # bottom rows see the ground at z = h*fy/(v-cy)
    v = H - 1
    assert abs(depth[v, W // 2] - CAM_HEIGHT_M * FY / (v - CY)) < 1e-3
