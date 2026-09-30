"""VO integrity monitor ("visual RAIM"): per-frame probability that VO has failed.

Twelve cheap features are computed every frame (names are the keys of the
returned dict and of the logged ``health`` telemetry):

======================  =====================================================  =========
feature                 meaning                                                transform
======================  =====================================================  =========
vo_inliers              PnP inliers after LM refinement                        log1p
vo_inlier_ratio         inliers / tracked points with valid depth              identity
vo_reproj_rmse_px       inlier reprojection RMSE (px)                          identity
vo_coverage             fraction of a 4x4 image grid holding inliers           identity
vo_track_age            mean age of inlier tracks (frames)                     log1p
vo_hess_min_eig         min eigenvalue of the pose Hessian J^T J               log10(1+x)
img_lapvar              variance of the Laplacian (blur; low = blurred)         log10(1+x)
img_sat_frac            fraction of pixels > 250 (glare / over-exposure)       identity
img_dark_frac           fraction of pixels < 10 (under-exposure)               identity
img_rms_contrast        std(I) / 255                                           identity
img_dark_channel        mean 15x15 local-min intensity / 255 (haze proxy,      identity
                        He et al. 2009 dark-channel prior)
disp_density_ground     valid-disparity fraction in the ground ROI             identity
======================  =====================================================  =========

A logistic model maps the standardised, transformed features to
``p_fail = sigmoid(w . z + b)``. Weights come from ``models/integrity.json``
(trained offline by ``metagross.eval.integrity_train`` on real KITTI frames
with synthetic degradations); if the file is absent a hand-set fallback with
the physically expected signs is used. ``q = 1 - p_fail`` is smoothed with an
asymmetric EMA (fast to drop, slow to recover) so that trust is regained only
after several healthy frames.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from metagross.autonomy.localization.vo import empty_stats

LOG = logging.getLogger(__name__)

MODEL_SCHEMA = "metagross.integrity/1"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "integrity.json"

FEATURE_NAMES: tuple[str, ...] = (
    "vo_inliers", "vo_inlier_ratio", "vo_reproj_rmse_px", "vo_coverage", "vo_track_age", "vo_hess_min_eig",
    "img_lapvar", "img_sat_frac", "img_dark_frac", "img_rms_contrast", "img_dark_channel", "disp_density_ground",
)
TRANSFORMS: dict[str, str] = {
    "vo_inliers": "log1p", "vo_track_age": "log1p", "vo_hess_min_eig": "log10p", "img_lapvar": "log10p",
}

SAT_LEVEL = 250  # 8-bit level counted as saturated
DARK_LEVEL = 10  # 8-bit level counted as black
DARK_CHANNEL_PATCH = 15  # px (at half resolution), He et al. use 15 at full res
IMG_DOWNSAMPLE = 2  # image statistics are computed at 1/2 resolution for speed
# Ground ROI for disparity density, as fractions of (height, width)
GROUND_ROI_ROWS = (0.60, 0.95)
GROUND_ROI_COLS = (0.10, 0.90)

# Hand-set fallback model: (mean, scale, coef) per transformed feature. Means / scales are
# typical healthy values; signs encode physics (more inliers / texture -> lower p_fail).
FALLBACK_MODEL: dict[str, tuple[float, float, float]] = {
    "vo_inliers": (5.0, 1.0, -1.5),
    "vo_inlier_ratio": (0.7, 0.2, -0.8),
    "vo_reproj_rmse_px": (0.6, 0.5, 0.8),
    "vo_coverage": (0.6, 0.2, -0.6),
    "vo_track_age": (1.5, 0.7, -0.3),
    "vo_hess_min_eig": (4.0, 1.0, -0.5),
    "img_lapvar": (2.5, 0.5, -0.4),
    "img_sat_frac": (0.02, 0.05, 0.5),
    "img_dark_frac": (0.02, 0.05, 0.5),
    "img_rms_contrast": (0.2, 0.07, -0.4),
    "img_dark_channel": (0.15, 0.1, 0.3),
    "disp_density_ground": (0.8, 0.2, -0.4),
}
FALLBACK_INTERCEPT = -3.0  # p_fail ~ 0.05 for a typical healthy frame


@dataclass(slots=True)
class HealthConfig:
    alpha_down: float = 0.7  # EMA gain when q drops (react within ~1-2 frames)
    alpha_up: float = 0.2  # EMA gain when q recovers (~5 frames to regain trust)
    q_reject: float = 0.4  # VO rejected below this (Localizer)


def transform_features(feats: Mapping[str, float], names: tuple[str, ...] = FEATURE_NAMES) -> np.ndarray:
    """Raw feature dict -> transformed vector (order = ``names``)."""
    out = np.empty(len(names))
    for i, n in enumerate(names):
        v = float(feats[n])
        tr = TRANSFORMS.get(n, "id")
        if tr == "log1p":
            v = math.log1p(max(v, 0.0))
        elif tr == "log10p":
            v = math.log10(1.0 + max(v, 0.0))
        out[i] = v
    return out


def image_features(gray: np.ndarray, rgb: Optional[np.ndarray] = None) -> dict[str, float]:
    """Photometric features of one frame. ``gray`` (H, W) uint8; ``rgb`` optional (H, W, 3) uint8."""
    small = cv2.resize(gray, (gray.shape[1] // IMG_DOWNSAMPLE, gray.shape[0] // IMG_DOWNSAMPLE),
                       interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(small, cv2.CV_32F)
    if rgb is not None:
        s3 = cv2.resize(rgb, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
        # per-pixel channel minimum; cv2.min gives exactly numpy's min(axis=2) at ~1/10 of the cost
        src = cv2.min(cv2.min(s3[..., 0], s3[..., 1]), s3[..., 2])
    else:
        src = small
    dark = cv2.erode(src, cv2.getStructuringElement(cv2.MORPH_RECT, (DARK_CHANNEL_PATCH, DARK_CHANNEL_PATCH)))
    return {
        "img_lapvar": float(lap.var()),
        "img_sat_frac": float(np.count_nonzero(small > SAT_LEVEL)) / small.size,
        "img_dark_frac": float(np.count_nonzero(small < DARK_LEVEL)) / small.size,
        "img_rms_contrast": float(small.std()) / 255.0,
        "img_dark_channel": float(dark.mean()) / 255.0,
    }


def disparity_density(disparity: Optional[np.ndarray]) -> float:
    """Fraction of valid (> 0) disparities in the ground ROI; 0 when no disparity."""
    if disparity is None:
        return 0.0
    h, w = disparity.shape[:2]
    roi = disparity[int(GROUND_ROI_ROWS[0] * h):int(GROUND_ROI_ROWS[1] * h),
                    int(GROUND_ROI_COLS[0] * w):int(GROUND_ROI_COLS[1] * w)]
    return float(np.count_nonzero(roi > 0)) / max(roi.size, 1)


def compute_features(vo_stats: Optional[Mapping[str, float]], gray: np.ndarray,
                     disparity: Optional[np.ndarray], rgb: Optional[np.ndarray] = None) -> dict[str, float]:
    """All 12 raw features for one frame (VO stats of a failed frame -> failure defaults)."""
    s = dict(empty_stats()) if vo_stats is None else vo_stats
    feats = {
        "vo_inliers": float(s["inliers"]),
        "vo_inlier_ratio": float(s["inlier_ratio"]),
        "vo_reproj_rmse_px": float(s["reproj_rmse_px"]),
        "vo_coverage": float(s["coverage"]),
        "vo_track_age": float(s["track_age"]),
        "vo_hess_min_eig": float(s["hess_min_eig"]),
    }
    feats.update(image_features(gray, rgb))
    feats["disp_density_ground"] = disparity_density(disparity)
    return feats


class LogisticIntegrityModel:
    """``p_fail = sigmoid(coef . (transform(f) - mean) / scale + intercept)``."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray, coef: np.ndarray, intercept: float,
                 names: tuple[str, ...] = FEATURE_NAMES, source: str = "fallback",
                 meta: Optional[dict[str, Any]] = None) -> None:
        self.names = names
        self.mean = np.asarray(mean, float)
        self.scale = np.where(np.asarray(scale, float) > 1e-12, np.asarray(scale, float), 1.0)
        self.coef = np.asarray(coef, float)
        self.intercept = float(intercept)
        self.source = source
        self.meta = meta or {}

    @classmethod
    def fallback(cls) -> "LogisticIntegrityModel":
        m = np.array([FALLBACK_MODEL[n][0] for n in FEATURE_NAMES])
        s = np.array([FALLBACK_MODEL[n][1] for n in FEATURE_NAMES])
        c = np.array([FALLBACK_MODEL[n][2] for n in FEATURE_NAMES])
        return cls(m, s, c, FALLBACK_INTERCEPT, source="fallback")

    @classmethod
    def load(cls, path: Optional[Path] = DEFAULT_MODEL_PATH) -> "LogisticIntegrityModel":
        """Load a trained model; fall back to the hand-set model if absent or invalid."""
        if path is None or not Path(path).exists():
            LOG.info("integrity model %s not found; using hand-set fallback", path)
            return cls.fallback()
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            if d.get("schema") != MODEL_SCHEMA or tuple(d["features"]) != FEATURE_NAMES:
                raise ValueError("schema / feature list mismatch")
            return cls(np.array(d["mean"]), np.array(d["scale"]), np.array(d["coef"]), float(d["intercept"]),
                       source=str(path), meta=d.get("meta", {}))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            LOG.warning("integrity model %s invalid (%s); using hand-set fallback", path, exc)
            return cls.fallback()

    def to_json(self, meta: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return {
            "schema": MODEL_SCHEMA,
            "features": list(self.names),
            "transforms": [TRANSFORMS.get(n, "id") for n in self.names],
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "meta": meta if meta is not None else self.meta,
        }

    def logit(self, x_transformed: np.ndarray) -> np.ndarray:
        z = (np.atleast_2d(x_transformed) - self.mean) / self.scale
        return z @ self.coef + self.intercept

    def p_fail(self, feats: Mapping[str, float]) -> float:
        a = float(self.logit(transform_features(feats, self.names))[0])
        return 1.0 / (1.0 + math.exp(-max(min(a, 50.0), -50.0)))


class IntegrityMonitor:
    """Stateful wrapper: features -> p_fail -> smoothed health score q."""

    def __init__(self, model: Optional[LogisticIntegrityModel] = None, config: Optional[HealthConfig] = None,
                 model_path: Optional[Path] = DEFAULT_MODEL_PATH) -> None:
        self.model = model if model is not None else LogisticIntegrityModel.load(model_path)
        self.cfg = config or HealthConfig()
        self.reset()

    def reset(self) -> None:
        self.q_smooth: Optional[float] = None

    def update(self, vo_stats: Optional[Mapping[str, float]], gray: np.ndarray,
               disparity: Optional[np.ndarray], rgb: Optional[np.ndarray] = None,
               warmup: bool = False) -> dict[str, float]:
        """Returns the 12 features plus 'p_fail', 'q_inst', 'q' (smoothed) and 'q_gate'.

        ``q_gate = min(q_inst, q)``: VO is rejected if either this frame looks bad
        or the recent history has not yet recovered. ``warmup=True`` marks a frame
        on which VO cannot produce a pose by construction (the first frame after
        a reset): its features are reported but it does not enter the EMA.
        """
        return self.update_features(compute_features(vo_stats, gray, disparity, rgb), warmup=warmup)

    def update_features(self, feats: Mapping[str, float], warmup: bool = False) -> dict[str, float]:
        """Same as :meth:`update` from precomputed raw features (offline replay)."""
        p = self.model.p_fail(feats)
        q_inst = 1.0 - p
        if warmup:
            q_s = 1.0 if self.q_smooth is None else self.q_smooth
            out = dict(feats)
            out.update({"p_fail": p, "q_inst": q_inst, "q": q_s, "q_gate": q_s})
            return out
        if self.q_smooth is None:
            self.q_smooth = q_inst
        else:
            a = self.cfg.alpha_down if q_inst < self.q_smooth else self.cfg.alpha_up
            self.q_smooth += a * (q_inst - self.q_smooth)
        out = dict(feats)
        out.update({"p_fail": p, "q_inst": q_inst, "q": self.q_smooth, "q_gate": min(q_inst, self.q_smooth)})
        return out
