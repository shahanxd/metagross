"""Analytic synthetic scenes for perception built-in tests (BIT), unit tests and demos.

This is *not* the simulator: it lives in the onboard package so that the perception
channel can be self-tested from the camera model alone (no scenario files, no ground
truth from ``metagross.sim``). Scenes are heightfields ``z = h(x, y)`` in the **body
frame** (x forward, y left, z up, metres) plus vertical-walled boxes and trenches.

Rendering ray-marches every left-camera pixel ray (and, for image pairs, every right-
camera ray) against the heightfield with a coarse step followed by bisection, giving
optical depth Z_c to sub-millimetre precision, hence exact disparity d = fx*B/Z_c.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from metagross.autonomy.perception.geometry import CameraGeometry

HeightFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

MARCH_STEP_M = 0.04  # coarse ray-march step in optical depth (m); thinnest feature > 2 steps
MARCH_ZC_MIN_M = 0.3
MARCH_ZC_MAX_M = 16.0
BISECT_ITERS = 10  # final precision = step / 2**iters (~0.04 mm)
# (texel size m, amplitude grey levels) of the stereo-pair renderer's texture octaves
TEXTURE_OCTAVES = ((0.004, 26.0), (0.015, 24.0), (0.05, 20.0), (0.18, 16.0))


# ----------------------------------------------------------------------------- terrain
@dataclass(frozen=True)
class Plane:
    """z = tan(pitch_deg) * x + tan(roll_deg) * y + z0 (positive pitch = uphill ahead)."""

    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    z0: float = 0.0

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return math.tan(math.radians(self.pitch_deg)) * x + math.tan(math.radians(self.roll_deg)) * y + self.z0


@dataclass(frozen=True)
class Rolling:
    """Rolling terrain z = amp * (1 - cos(2 pi x / wavelength)) (zero height at the vehicle)."""

    amp_m: float = 0.15
    wavelength_m: float = 8.0

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.amp_m * (1.0 - np.cos(2.0 * np.pi * np.asarray(x) / self.wavelength_m))


@dataclass(frozen=True)
class Crest:
    """Flat ground up to x_c, then a downslope of slope_deg until drop_m below, then flat."""

    x_c: float = 6.0
    drop_m: float = 1.5
    slope_deg: float = 35.0

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        fall = np.clip((np.asarray(x) - self.x_c) * math.tan(math.radians(self.slope_deg)), 0.0, self.drop_m)
        return -fall + 0.0 * y


@dataclass(frozen=True)
class Box:
    """Vertical-walled block of height h_m above the base terrain over [x0,x1] x [y0,y1]."""

    x0: float
    x1: float
    y0: float
    y1: float
    h_m: float


@dataclass(frozen=True)
class Trench:
    """Vertical-walled trench: base terrain lowered by depth_m over [x0, x0+width] x [y0, y1]."""

    x0: float
    width_m: float
    depth_m: float
    y0: float = -1e3
    y1: float = 1e3


@dataclass
class Scene:
    """Heightfield scene: base terrain + boxes (raise) + trenches (lower)."""

    base: HeightFn = field(default_factory=Plane)
    boxes: Sequence[Box] = ()
    trenches: Sequence[Trench] = ()
    name: str = "scene"

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        z = np.asarray(self.base(x, y), dtype=np.float64)
        for t in self.trenches:
            inside = (x >= t.x0) & (x <= t.x0 + t.width_m) & (y >= t.y0) & (y <= t.y1)
            z = np.where(inside, z - t.depth_m, z)
        for b in self.boxes:
            inside = (x >= b.x0) & (x <= b.x1) & (y >= b.y0) & (y <= b.y1)
            z = np.where(inside, z + b.h_m, z)
        return z

    def max_height(self) -> float:
        """Loose upper bound of the scene height used to terminate upward rays (m)."""
        xs = np.linspace(0.0, MARCH_ZC_MAX_M, 64)
        ys = np.linspace(-8.0, 8.0, 33)
        X, Y = np.meshgrid(xs, ys)
        return float(np.max(self(X, Y))) + 0.05


# ----------------------------------------------------------------------------- rendering
def march_depth(origin: np.ndarray, rays: np.ndarray, height: Scene, step_m: float = MARCH_STEP_M,
                zc_max: float = MARCH_ZC_MAX_M) -> np.ndarray:
    """First intersection of body-frame rays with the heightfield.

    origin: (3,) body-frame ray origin (m). rays: (N, 3) body rays with unit optical depth.
    Returns (N,) optical depth Z_c (m), +inf where nothing is hit before zc_max.
    Each ray starts where it descends below the scene's maximum height, so flat scenes
    need only a handful of steps per ray.
    """
    n = rays.shape[0]
    out = np.full(n, np.inf)
    zmax = height.max_height()
    rz = rays[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        start = np.where(rz < 0, (zmax - origin[2]) / rz, np.inf)
    if origin[2] <= zmax:
        start = np.full(n, MARCH_ZC_MIN_M)
    start = np.maximum(start, MARCH_ZC_MIN_M)
    active = np.nonzero(start <= zc_max)[0]
    zc = start[active].copy()  # current sample depth per active ray
    while active.size:
        r = rays[active]
        pz = origin[2] + zc * r[:, 2]
        below = pz <= height(origin[0] + zc * r[:, 0], origin[1] + zc * r[:, 1])
        if np.any(below):
            hit = active[below]
            b = zc[below].copy()
            a = np.maximum(b - step_m, MARCH_ZC_MIN_M)
            rh = rays[hit]
            for _ in range(BISECT_ITERS):
                m = 0.5 * (a + b)
                inside = origin[2] + m * rh[:, 2] <= height(origin[0] + m * rh[:, 0], origin[1] + m * rh[:, 1])
                b = np.where(inside, m, b)
                a = np.where(inside, a, m)
            out[hit] = b
        escaped = (r[:, 2] >= 0) & (pz > zmax)
        keep = ~below & ~escaped & (zc + step_m <= zc_max)
        active = active[keep]
        zc = zc[keep] + step_m
    return out


def render_disparity(geom: CameraGeometry, scene: Scene, noise_px: float = 0.0, seed: int = 0,
                     row_start: int = 0) -> np.ndarray:
    """Exact tier-0 style disparity (H, W) float32 px for the left camera; -1 where no hit.

    Optional additive Gaussian noise of ``noise_px`` (deterministic given ``seed``).
    """
    h, w = geom.height, geom.width
    rays = geom.rays_body[row_start:].reshape(-1, 3).astype(np.float64)
    zc = march_depth(geom.t_bc, rays, scene)
    disp = np.full((h, w), -1.0, dtype=np.float32)
    with np.errstate(divide="ignore"):
        d = np.where(np.isfinite(zc), geom.fxb / zc, -1.0)
    disp[row_start:] = d.reshape(h - row_start, w).astype(np.float32)
    if noise_px > 0:
        rng = np.random.default_rng(seed)
        valid = disp > 0
        disp[valid] += rng.normal(0.0, noise_px, size=int(valid.sum())).astype(np.float32)
        disp[valid & (disp <= 0)] = -1.0
    return disp


def _texture(seed: int, size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fine = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 1.0)
    coarse = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 1.5)
    norm = lambda t: (t - t.mean()) / (t.std() + 1e-6)  # noqa: E731
    return norm(fine), norm(coarse)


def _shade(points: np.ndarray, valid: np.ndarray, footprint_m: np.ndarray, tex: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Intensity (uint8) of body points from a multi-octave procedural texture; sky where invalid.

    Natural ground has texture at every scale; a pixel integrates it over its footprint
    (``footprint_m`` = metres per pixel at the point). Each octave is attenuated by
    exp(-(footprint / texel)^2), a cheap anti-aliasing pre-filter, so near ground shows
    fine grain and far ground stays alias-free and consistent between the two views.
    """
    fine, coarse = tex
    size = fine.shape[0]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    n = x.shape[0]
    cols = 1024  # remap maps must stay below SHRT_MAX in each dimension
    rows = -(-n // cols)

    def m(a: np.ndarray) -> np.ndarray:
        buf = np.zeros(rows * cols, dtype=np.float32)
        buf[:n] = np.nan_to_num(a, nan=0.0)
        return buf.reshape(rows, cols)

    val = np.full(n, 120.0)
    for i, (texel, amp) in enumerate(TEXTURE_OCTAVES):
        src = fine if i % 2 == 0 else coarse
        # Texture coordinates vary on horizontal and vertical faces alike; offsets decorrelate octaves.
        s_ = ((x + (0.7 - 0.4 * i) * z) / texel + 97.0 * i) % size
        t_ = ((y + (1.3 - 0.2 * i) * z) / texel + 31.0 * i) % size
        o = cv2.remap(src, m(s_), m(t_), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP).ravel()[:n]
        att = np.exp(-np.square(np.nan_to_num(footprint_m, nan=1.0) / texel))
        val += amp * att * o
    val = np.where(valid, val, 205.0)  # flat, textureless sky
    return np.clip(val, 0, 255).astype(np.uint8)


def render_stereo_pair(geom: CameraGeometry, scene: Scene, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Textured rectified stereo pair of the scene.

    Returns (left_rgb (H,W,3) uint8, right_gray (H,W) uint8, true_disparity (H,W) float32).
    """
    h, w = geom.height, geom.width
    rays = geom.rays_body.reshape(-1, 3).astype(np.float64)
    tex = _texture(seed)
    images = []
    zc_left = None
    for side in (0, 1):
        origin = geom.t_bc + (geom.R_bc @ np.array([geom.baseline_m, 0.0, 0.0])) * side  # right cam: +B along x_cam
        zc = march_depth(origin, rays, scene)
        valid = np.isfinite(zc)
        pts = origin + np.where(valid, zc, 0.0)[:, None] * rays
        footprint = np.where(valid, zc, np.nan) / geom.fx
        images.append(_shade(pts, valid, footprint, tex).reshape(h, w))
        if side == 0:
            zc_left = zc
    with np.errstate(divide="ignore"):
        disp = np.where(np.isfinite(zc_left), geom.fxb / zc_left, -1.0).reshape(h, w).astype(np.float32)
    left_rgb = np.repeat(images[0][..., None], 3, axis=2)
    return left_rgb, images[1], disp


# ----------------------------------------------------------------------------- catalogue
def scene_catalogue() -> dict[str, Scene]:
    """Named reference scenes used by tests and the demo (body frame, metres)."""
    return {
        "flat": Scene(name="flat"),
        "box": Scene(boxes=[Box(5.0, 5.4, -0.3, 0.3, 0.40)], name="box"),
        "trench": Scene(trenches=[Trench(5.0, 0.5, 0.6)], name="trench"),
        "crest": Scene(base=Crest(6.0, 1.5, 35.0), name="crest"),
        "rock_occlusion": Scene(boxes=[Box(4.0, 4.4, -0.3, 0.3, 0.50)], name="rock_occlusion"),
        "tilted": Scene(base=Plane(pitch_deg=4.0, roll_deg=3.0), name="tilted"),
        "rolling": Scene(base=Rolling(0.15, 8.0), name="rolling"),
    }


def make_scene(name: str, **kwargs: float) -> Scene:
    """Scene by name; ``trench`` accepts x0/width/depth overrides."""
    if name == "trench" and kwargs:
        return Scene(trenches=[Trench(kwargs.get("x0", 5.0), kwargs.get("width", 0.5), kwargs.get("depth", 0.6))], name="trench")
    return scene_catalogue()[name]


def optional_seeded_noise(disp: np.ndarray, sigma_px: Optional[float], seed: int) -> np.ndarray:
    """Return a noisy copy of ``disp`` (Gaussian, px) or the input when sigma is None/0."""
    if not sigma_px:
        return disp
    out = disp.copy()
    rng = np.random.default_rng(seed)
    valid = out > 0
    out[valid] += rng.normal(0.0, sigma_px, int(valid.sum())).astype(np.float32)
    out[valid & (out <= 0)] = -1.0
    return out
