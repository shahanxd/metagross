"""Seeded image + label augmentations for terrain segmentation (numpy / OpenCV only).

Two policies:

* ``clean``  - geometry + light photometrics: horizontal flip, random scale 0.75-1.25
  with aspect jitter, random crop (pad with the ignore label), light colour jitter.
* ``robust`` - ``clean`` + the camera failure modes the vehicle actually meets outdoors:
  gamma 0.4-2.5, exposure clipping (over/under-exposure with sensor saturation),
  motion blur, synthetic lens flare, haze, JPEG artefacts and sensor noise. Each is
  applied independently with its own probability.

Every random draw comes from the ``numpy.random.Generator`` passed in, so a sample's
augmentation is a pure function of ``(seed, epoch, index)`` regardless of DataLoader
worker scheduling.

Images are (H, W, 3) uint8 RGB; labels are (H, W) uint8 class ids with
:data:`~metagross.train.data_index.IGNORE_INDEX` (255) for padded pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from metagross.train.data_index import IGNORE_INDEX

POLICIES = ("clean", "robust", "none")

# --------------------------------------------------------------------------- parameters
SCALE_RANGE = (0.75, 1.25)  # isotropic scale relative to the crop size
# Aspect jitter (x-scale / y-scale). The deploy camera (640x400) is squashed to the model
# input (416x320) with a factor 0.81 relative to RUGD's 400x320; 0.8-1.25 covers it.
ASPECT_RANGE = (0.8, 1.25)
JITTER_BRIGHTNESS = 0.2  # +-20 % multiplicative
JITTER_CONTRAST = 0.2
JITTER_SATURATION = 0.2
JITTER_HUE_DEG = 7.0  # OpenCV hue units are degrees / 2

GAMMA_RANGE = (0.4, 2.5)  # sampled log-uniformly
P_GAMMA = 0.5
P_EXPOSURE = 0.3
OVEREXPOSE_GAIN = (1.6, 3.0)
UNDEREXPOSE_GAIN = (0.15, 0.45)
P_MOTION_BLUR = 0.25
MOTION_BLUR_LEN_PX = (5, 17)
P_FLARE = 0.2
P_HAZE = 0.2
HAZE_BETA = (0.6, 2.0)  # extinction coefficient of the row-based depth proxy
P_JPEG = 0.3
JPEG_QUALITY = (12, 60)
P_NOISE = 0.35
NOISE_SIGMA = (2.0, 12.0)  # grey levels, additive Gaussian (plus shot noise below)


@dataclass(frozen=True)
class AugConfig:
    """Output crop size (height, width) in pixels and policy name."""

    out_hw: tuple[int, int]
    policy: str = "clean"

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {self.policy!r}")


# --------------------------------------------------------------------------- geometry
def resize_pair(img: np.ndarray, lab: np.ndarray, out_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Resize image (area/linear) and label (nearest) to exactly ``out_hw`` (validation)."""
    h, w = out_hw
    interp = cv2.INTER_AREA if img.shape[0] > h and img.shape[1] > w else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp), cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)


def random_scale_crop(img: np.ndarray, lab: np.ndarray, out_hw: tuple[int, int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Scale so the image height maps to ``out_h * s`` (aspect-jittered), then random crop.

    Regions outside the scaled image are padded with 0 (image) / IGNORE_INDEX (label).
    """
    oh, ow = out_hw
    h, w = lab.shape
    s = rng.uniform(*SCALE_RANGE)
    a = math.exp(rng.uniform(math.log(ASPECT_RANGE[0]), math.log(ASPECT_RANGE[1])))
    sy = s * oh / h / math.sqrt(a)
    sx = sy * a
    nh, nw = max(1, int(round(h * sy))), max(1, int(round(w * sx)))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    lab = cv2.resize(lab, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ph, pw = max(0, oh - nh), max(0, ow - nw)
    if ph or pw:
        top, left = int(rng.integers(0, ph + 1)), int(rng.integers(0, pw + 1))
        img = cv2.copyMakeBorder(img, top, ph - top, left, pw - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        lab = cv2.copyMakeBorder(lab, top, ph - top, left, pw - left, cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
    y0 = int(rng.integers(0, img.shape[0] - oh + 1))
    x0 = int(rng.integers(0, img.shape[1] - ow + 1))
    return img[y0 : y0 + oh, x0 : x0 + ow], lab[y0 : y0 + oh, x0 : x0 + ow]


# --------------------------------------------------------------------------- photometric
def color_jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light brightness / contrast / saturation / hue jitter."""
    x = img.astype(np.float32)
    b = 1.0 + rng.uniform(-JITTER_BRIGHTNESS, JITTER_BRIGHTNESS)
    c = 1.0 + rng.uniform(-JITTER_CONTRAST, JITTER_CONTRAST)
    mean = x.mean()
    x = (x - mean) * c + mean * b
    x = np.clip(x, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= 1.0 + rng.uniform(-JITTER_SATURATION, JITTER_SATURATION)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-JITTER_HUE_DEG, JITTER_HUE_DEG) / 2.0) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """out = 255 * (in / 255) ** gamma (gamma > 1 darkens, < 1 brightens)."""
    lut = np.clip(255.0 * (np.arange(256) / 255.0) ** gamma, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def exposure_clip(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Over-exposure (gain > 1, highlights saturate at 255) or under-exposure (crushed shadows)."""
    over = rng.random() < 0.5
    g = rng.uniform(*(OVEREXPOSE_GAIN if over else UNDEREXPOSE_GAIN))
    return np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)


def motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Linear motion-blur kernel of random length and direction."""
    length = int(rng.integers(MOTION_BLUR_LEN_PX[0], MOTION_BLUR_LEN_PX[1] + 1))
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D(((length - 1) / 2.0, (length - 1) / 2.0), float(rng.uniform(0, 180)), 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= max(k.sum(), 1e-6)
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)


def synthetic_flare(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Additive sun glare: radial glow + a streak + a few ghost discs along the optical axis."""
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0, w), rng.uniform(0, 0.6 * h)  # sun is usually in the upper image
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xx - cx, yy - cy)
    radius = rng.uniform(0.15, 0.45) * max(h, w)
    glow = np.exp(-((r / radius) ** 2)) * rng.uniform(120, 255)
    ang = rng.uniform(0, math.pi)
    d = np.abs((xx - cx) * math.sin(ang) - (yy - cy) * math.cos(ang))
    streak = np.exp(-((d / (0.01 * max(h, w))) ** 2)) * np.exp(-r / (0.6 * max(h, w))) * rng.uniform(60, 160)
    add = glow + streak
    ox, oy = w / 2.0 - cx, h / 2.0 - cy  # ghosts mirror through the image centre
    for t in rng.uniform(0.6, 1.8, size=int(rng.integers(1, 4))):
        gr = rng.uniform(0.02, 0.06) * max(h, w)
        add += (np.hypot(xx - (cx + 2 * t * ox), yy - (cy + 2 * t * oy)) < gr) * rng.uniform(20, 60)
    tint = np.array([1.0, rng.uniform(0.85, 1.0), rng.uniform(0.6, 0.9)], np.float32)  # warm
    return np.clip(img.astype(np.float32) + add[..., None] * tint, 0, 255).astype(np.uint8)


def haze(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Koschmieder haze I = J t + A (1 - t) with a row-based depth proxy (top = far)."""
    h = img.shape[0]
    depth = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None, None]  # 1 at top row, 0 at bottom
    t = np.exp(-rng.uniform(*HAZE_BETA) * depth)
    airlight = rng.uniform(170, 240)
    return np.clip(img.astype(np.float32) * t + airlight * (1 - t), 0, 255).astype(np.uint8)


def jpeg(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """JPEG re-encode at a random low quality (RGB in / RGB out)."""
    q = int(rng.integers(JPEG_QUALITY[0], JPEG_QUALITY[1] + 1))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return img
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def sensor_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian read noise + signal-dependent shot noise."""
    x = img.astype(np.float32)
    sigma = rng.uniform(*NOISE_SIGMA)
    shot = np.sqrt(np.maximum(x, 0.0) * rng.uniform(0.02, 0.2))
    x = x + rng.standard_normal(x.shape).astype(np.float32) * (sigma + shot)
    return np.clip(x, 0, 255).astype(np.uint8)


def robust_photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The ROBUST-only corruption chain (each op drawn independently)."""
    if rng.random() < P_GAMMA:
        img = apply_gamma(img, math.exp(rng.uniform(math.log(GAMMA_RANGE[0]), math.log(GAMMA_RANGE[1]))))
    if rng.random() < P_EXPOSURE:
        img = exposure_clip(img, rng)
    if rng.random() < P_HAZE:
        img = haze(img, rng)
    if rng.random() < P_FLARE:
        img = synthetic_flare(img, rng)
    if rng.random() < P_MOTION_BLUR:
        img = motion_blur(img, rng)
    if rng.random() < P_NOISE:
        img = sensor_noise(img, rng)
    if rng.random() < P_JPEG:
        img = jpeg(img, rng)
    return img


def augment(img: np.ndarray, lab: np.ndarray, cfg: AugConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``cfg.policy`` and return an (out_h, out_w) image/label pair."""
    if cfg.policy == "none":
        return resize_pair(img, lab, cfg.out_hw)
    if rng.random() < 0.5:
        img, lab = img[:, ::-1], lab[:, ::-1]
    img, lab = random_scale_crop(np.ascontiguousarray(img), np.ascontiguousarray(lab), cfg.out_hw, rng)
    img = color_jitter(np.ascontiguousarray(img), rng)
    if cfg.policy == "robust":
        img = robust_photometric(img, rng)
    return np.ascontiguousarray(img), np.ascontiguousarray(lab)
