"""Photometric / optical degradations applied to real stereo frames.

Used to stress the VO and to train / test the integrity monitor on *real*
KITTI images. All functions take and return 8-bit gray images (H, W) and are
deterministic given the ``rng`` passed in.

Kinds and severity levels (1 = mild, 2 = moderate, 3 = severe):

============== ================================================ =====================
kind           model                                            parameter per level
============== ================================================ =====================
gamma_dark     I' = 255 (I/255)^gamma                           gamma 1.8 / 2.4 / 3.0
gamma_bright   same, gamma < 1                                  gamma 0.6 / 0.45 / 0.3
exposure_clip  I' = clip(g I, 0, 255) (over-exposure)           g 1.8 / 2.6 / 3.5
motion_blur    linear PSF, per-frame jittered length/direction  9 / 15 / 21 px (+-35 %)
low_light      gamma darkening + per-frame Gaussian sensor noise gamma 2.4 / 3.0 / 3.5,
                                                                noise 6 / 10 / 16 DN
sun_flare      additive glow + veiling glare + ghost disks       peak 0.5 / 0.8 / 1.0
haze           Koschmieder: I = J t + A (1 - t), t = e^(-beta d) beta 0.02 / 0.05 / 0.10 1/m
smudge         smooth blobs mixing a heavily blurred, brightened left lens only;
               image (left lens only)                           coverage 0.1 / 0.25 / 0.4
============== ================================================ =====================

Haze needs a per-pixel depth ``d`` (m); callers compute it from SGBM on the
clean pair (:func:`depth_from_sgbm`); pixels without valid depth (sky) are
treated as ``FAR_DEPTH_M`` away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

KINDS: tuple[str, ...] = ("gamma_dark", "gamma_bright", "exposure_clip", "motion_blur", "sun_flare", "haze", "smudge",
                          "low_light")
LEVELS: dict[str, tuple[float, float, float]] = {
    "gamma_dark": (1.8, 2.4, 3.0),
    "gamma_bright": (0.6, 0.45, 0.3),
    "exposure_clip": (1.8, 2.6, 3.5),
    "motion_blur": (9, 15, 21),
    "sun_flare": (0.5, 0.8, 1.0),
    "haze": (0.02, 0.05, 0.10),
    "smudge": (0.10, 0.25, 0.40),
    "low_light": (2.4, 3.0, 3.5),  # gamma; paired with LOW_LIGHT_NOISE_DN
}
LOW_LIGHT_NOISE_DN = (6.0, 10.0, 16.0)  # per-pixel Gaussian read/shot noise (8-bit DN) at high gain
BLUR_JITTER = 0.35  # per-frame relative jitter of blur length (vibration: blur changes frame to frame)
BLUR_ANGLE_JITTER_RAD = math.radians(25.0)
AIRLIGHT = 230.0  # 8-bit atmospheric light A
FAR_DEPTH_M = 200.0  # depth assumed where stereo has no valid disparity (sky, far field)


@dataclass(frozen=True, slots=True)
class Degradation:
    kind: str
    severity: int  # 1..3

    def __post_init__(self) -> None:
        if self.kind not in KINDS or self.severity not in (1, 2, 3):
            raise ValueError(f"bad degradation {self.kind}/{self.severity}")

    @property
    def param(self) -> float:
        return float(LEVELS[self.kind][self.severity - 1])


def gamma(img: np.ndarray, g: float) -> np.ndarray:
    lut = (255.0 * (np.arange(256) / 255.0) ** g).round().clip(0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def low_light(img: np.ndarray, g: float, noise_dn: float, rng: np.random.Generator) -> np.ndarray:
    """Dark scene at high sensor gain: gamma darkening plus independent per-pixel noise."""
    dark = gamma(img, g).astype(np.float32)
    return np.clip(dark + rng.normal(0.0, noise_dn, img.shape).astype(np.float32), 0, 255).astype(np.uint8)


def exposure_clip(img: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def motion_blur_kernel(length_px: int, angle_rad: float) -> np.ndarray:
    k = np.zeros((length_px, length_px), np.float32)
    c = (length_px - 1) / 2.0
    dx, dy = math.cos(angle_rad), math.sin(angle_rad)
    for s in np.linspace(-c, c, 4 * length_px):
        x, y = int(round(c + s * dx)), int(round(c + s * dy))
        k[y, x] = 1.0
    return k / k.sum()


def motion_blur(img: np.ndarray, length_px: int, angle_rad: float) -> np.ndarray:
    return cv2.filter2D(img, -1, motion_blur_kernel(int(length_px), angle_rad), borderType=cv2.BORDER_REFLECT)


def flare_overlay(shape: tuple[int, int], peak: float, rng: np.random.Generator) -> np.ndarray:
    """Additive flare layer in [0, 255*peak] (float32): sun glow in the upper image,
    global veiling glare and ghost disks mirrored through the image centre."""
    h, w = shape
    sx, sy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.0, 0.35) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = (xx - sx) ** 2 + (yy - sy) ** 2
    glow = np.exp(-r2 / (2 * (0.12 * w) ** 2)) + 0.5 * np.exp(-r2 / (2 * (0.35 * w) ** 2))
    layer = glow * 255.0 * peak + 0.25 * 255.0 * peak  # veiling glare lifts the whole frame
    cx, cy = w / 2.0, h / 2.0
    for frac, rad in ((0.6, 0.05), (1.2, 0.08), (1.6, 0.03)):
        gx, gy = cx + (cx - sx) * frac, cy + (cy - sy) * frac
        disk = ((xx - gx) ** 2 + (yy - gy) ** 2) < (rad * w) ** 2
        layer += disk * 60.0 * peak
    return cv2.GaussianBlur(layer, (0, 0), 3.0)


def sun_flare(img: np.ndarray, layer: np.ndarray) -> np.ndarray:
    return np.clip(img.astype(np.float32) + layer, 0, 255).astype(np.uint8)


def haze(img: np.ndarray, depth_m: np.ndarray, beta_per_m: float, airlight: float = AIRLIGHT) -> np.ndarray:
    """Koschmieder model with per-pixel transmission t = exp(-beta d)."""
    d = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m, FAR_DEPTH_M)
    t = np.exp(-beta_per_m * d)
    out = img.astype(np.float32) * t + airlight * (1.0 - t)
    return np.clip(out, 0, 255).astype(np.uint8)


def smudge_mask(shape: tuple[int, int], coverage: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth random blob mask in [0, 1] covering roughly ``coverage`` of the frame."""
    h, w = shape
    noise = rng.standard_normal((max(h // 16, 2), max(w // 16, 2))).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), 8.0)
    thr = np.quantile(noise, 1.0 - coverage)
    m = np.clip((noise - thr) / (noise.std() * 0.5 + 1e-6), 0.0, 1.0)
    return cv2.GaussianBlur(m, (0, 0), 6.0)


def smudge(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), 9.0).astype(np.float32)
    smeared = 0.7 * blurred + 0.3 * 200.0  # grease scatters light: blur + brighten
    out = img.astype(np.float32) * (1.0 - mask) + smeared * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def depth_from_sgbm(left: np.ndarray, right: np.ndarray, fx: float, baseline_m: float,
                    num_disparities: int = 64) -> np.ndarray:
    """Coarse depth (m) for the haze model: SGBM at half resolution, upsampled; inf where invalid."""
    h, w = left.shape
    ls, rs = cv2.pyrDown(left), cv2.pyrDown(right)
    sgbm = cv2.StereoSGBM_create(0, num_disparities, 5, P1=8 * 25, P2=32 * 25, uniquenessRatio=10,
                                 speckleWindowSize=50, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = sgbm.compute(ls, rs).astype(np.float32) / 16.0 * 2.0  # back to full-resolution pixels
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
    with np.errstate(divide="ignore"):
        return np.where(d > 0.5, fx * baseline_m / np.maximum(d, 1e-6), np.inf).astype(np.float32)


class Degrader:
    """Applies one :class:`Degradation` to a stereo pair, holding per-segment random state
    (flare position, smudge mask, blur direction) so a segment is temporally coherent.

    Per-frame randomness (blur jitter, sensor noise) is drawn from a generator seeded
    by (segment seed, frame index), so results do not depend on call order."""

    def __init__(self, deg: Degradation, shape: tuple[int, int], seed: int) -> None:
        self.deg = deg
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.angle = float(rng.uniform(0.0, math.pi))
        self.flare = flare_overlay(shape, deg.param, rng) if deg.kind == "sun_flare" else None
        self.mask = smudge_mask(shape, deg.param, rng) if deg.kind == "smudge" else None

    def __call__(self, left: np.ndarray, right: np.ndarray, depth_m: Optional[np.ndarray] = None,
                 frame: int = 0) -> tuple[np.ndarray, np.ndarray]:
        k, p = self.deg.kind, self.deg.param
        frng = np.random.default_rng([self.seed, frame])
        if k in ("gamma_dark", "gamma_bright"):
            return gamma(left, p), gamma(right, p)
        if k == "low_light":
            n = LOW_LIGHT_NOISE_DN[self.deg.severity - 1]
            return low_light(left, p, n, frng), low_light(right, p, n, frng)
        if k == "exposure_clip":
            return exposure_clip(left, p), exposure_clip(right, p)
        if k == "motion_blur":
            length = max(3, int(round(p * (1.0 + frng.uniform(-BLUR_JITTER, BLUR_JITTER)))) | 1)
            ang = self.angle + frng.uniform(-BLUR_ANGLE_JITTER_RAD, BLUR_ANGLE_JITTER_RAD)
            return motion_blur(left, length, ang), motion_blur(right, length, ang)
        if k == "sun_flare":
            return sun_flare(left, self.flare), sun_flare(right, self.flare)
        if k == "haze":
            if depth_m is None:
                raise ValueError("haze needs a depth map")
            return haze(left, depth_m, p), haze(right, depth_m, p)
        if k == "smudge":
            return smudge(left, self.mask), right
        raise ValueError(k)


def make_schedule(n_frames: int, seed: int, clean_len: tuple[int, int] = (15, 45),
                  bad_len: tuple[int, int] = (20, 50)) -> list[Optional[tuple[Degradation, int]]]:
    """Per-frame schedule alternating clean and degraded segments.

    Returns a list with ``None`` for clean frames and ``(Degradation, segment_seed)``
    for degraded ones. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    out: list[Optional[tuple[Degradation, int]]] = []
    k = 0
    while k < n_frames:
        n_clean = int(rng.integers(*clean_len))
        out.extend([None] * n_clean)
        n_bad = int(rng.integers(*bad_len))
        deg = Degradation(KINDS[int(rng.integers(len(KINDS)))], int(rng.integers(1, 4)))
        seg_seed = int(rng.integers(2 ** 31))
        out.extend([(deg, seg_seed)] * n_bad)
        k += n_clean + n_bad
    return out[:n_frames]
