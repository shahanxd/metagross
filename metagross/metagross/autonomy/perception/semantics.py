"""Terrain semantic segmentation for the onboard stack (ONNX Runtime, CPU).

Implements :class:`metagross.contracts.interfaces.SegmenterProto`:

    rgb (H, W, 3) uint8  ->  (class_ids (H, W) uint8, entropy (H, W) float32)

Class ids follow the project's five-class terrain scheme
(:data:`metagross.contracts.interfaces.SEM_CLASSES`, identical to the GAIA-URJC
OFFROAD5 / RUGD-5Labels label ids):

    0 background/sky, 1 obstacle, 2 water/mud, 3 unstable (grass/dirt), 4 stable (paved/path)

Two model families are supported, selected by the sidecar JSON written next to the
ONNX file (``<model>.json``) or, failing that, by the number of output channels:

* ``direct5`` - our trained LR-ASPP MobileNetV3-Large export (5 logits per pixel,
  output at the input resolution). Produced by ``metagross.train.export_onnx``.
* ``ade150``  - a zero-shot SegFormer-B0 ADE20K export (150 logits at 1/4 input
  resolution). ADE20K probabilities are summed into the five classes through
  :data:`ADE20K_TO_SEM5` (see the table for the verified ADE ids); any ADE class not
  listed is treated as **obstacle** - the conservative choice for a vehicle whose
  thesis is "unknown is never free".

``entropy`` is the Shannon entropy of the five-class posterior, normalised by
``ln(5)`` so it lies in [0, 1] (0 = certain, 1 = uniform). It is computed at the
model output resolution and bilinearly resized to the image.

Image frame: pixel (row v, column u) of the rectified left camera (OpenCV camera
frame: x right, y down, z forward). The segmenter is purely 2-D; projection into the
BEV grid is done by the perception node.

This module must never import ``metagross.sim``, ``metagross.eval`` or
``metagross.train``; it depends only on numpy, OpenCV, ONNX Runtime and contracts.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from metagross.contracts.interfaces import SEM_CLASSES
from metagross.contracts.messages import CELL_COLORS, CellState

LOG = logging.getLogger(__name__)

NUM_SEM_CLASSES = len(SEM_CLASSES)  # 5
SEM_BACKGROUND, SEM_OBSTACLE, SEM_WATER, SEM_UNSTABLE, SEM_STABLE = 0, 1, 2, 3, 4
TRAVERSABLE_CLASSES: tuple[int, ...] = (SEM_UNSTABLE, SEM_STABLE)
HAZARD_CLASSES: tuple[int, ...] = (SEM_OBSTACLE, SEM_WATER)

# Project colour key for the five semantic classes (RGB). Re-uses the BEV cell colours
# where the meaning coincides so console, video and deck read the same:
# sky -> UNSEEN grey, obstacle -> POSITIVE red, water -> WATER blue, stable -> GROUND green.
# "Unstable" (drivable at a cost) gets ochre: a yellow-green was checked with the dataviz
# palette validator and failed the normal-vision floor against GROUND green (dE 14.6 < 15).
# Note the frozen POSITIVE red vs GROUND green pair is weak under deuteranopia, so every
# figure carries a text legend as secondary encoding.
SEM_COLORS: dict[int, tuple[int, int, int]] = {
    SEM_BACKGROUND: CELL_COLORS[CellState.UNSEEN],
    SEM_OBSTACLE: CELL_COLORS[CellState.POSITIVE],
    SEM_WATER: CELL_COLORS[CellState.WATER],
    SEM_UNSTABLE: (217, 180, 58),
    SEM_STABLE: CELL_COLORS[CellState.GROUND],
}
SEM_PALETTE = np.array([SEM_COLORS[i] for i in range(NUM_SEM_CLASSES)], dtype=np.uint8)  # (5, 3)

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------- ADE20K -> 5
ADE20K_NUM_CLASSES = 150
# ADE20K (SceneParse150, 0-based ids as in the HF `nvidia/segformer-*-ade-512-512` config).
# Every id below was checked against the checkpoint's id2label table.
ADE20K_TO_SEM5: dict[int, int] = {
    # stable, paved / prepared surfaces -> 4
    6: SEM_STABLE,  # road
    11: SEM_STABLE,  # sidewalk
    52: SEM_STABLE,  # path
    54: SEM_STABLE,  # runway
    3: SEM_STABLE,  # floor
    # unstable, natural ground -> 3
    9: SEM_UNSTABLE,  # grass
    13: SEM_UNSTABLE,  # earth
    29: SEM_UNSTABLE,  # field
    46: SEM_UNSTABLE,  # sand
    91: SEM_UNSTABLE,  # dirt track
    94: SEM_UNSTABLE,  # land
    # water -> 2
    21: SEM_WATER,  # water
    26: SEM_WATER,  # sea
    60: SEM_WATER,  # river
    109: SEM_WATER,  # swimming pool
    113: SEM_WATER,  # waterfall
    128: SEM_WATER,  # lake
    # background (far, never driven on) -> 0
    2: SEM_BACKGROUND,  # sky
    16: SEM_BACKGROUND,  # mountain
    68: SEM_BACKGROUND,  # hill
    # explicit obstacles -> 1 (listed for documentation; the default is also 1)
    0: SEM_OBSTACLE,  # wall
    1: SEM_OBSTACLE,  # building
    4: SEM_OBSTACLE,  # tree
    12: SEM_OBSTACLE,  # person
    17: SEM_OBSTACLE,  # plant
    20: SEM_OBSTACLE,  # car
    32: SEM_OBSTACLE,  # fence
    34: SEM_OBSTACLE,  # rock
    43: SEM_OBSTACLE,  # signboard
    69: SEM_OBSTACLE,  # bench
    72: SEM_OBSTACLE,  # palm
    80: SEM_OBSTACLE,  # bus
    83: SEM_OBSTACLE,  # truck
    93: SEM_OBSTACLE,  # pole
    102: SEM_OBSTACLE,  # van
    126: SEM_OBSTACLE,  # animal
    127: SEM_OBSTACLE,  # bicycle
}
ADE20K_DEFAULT_SEM = SEM_OBSTACLE  # any unlisted ADE class: conservative


def ade20k_lut() -> np.ndarray:
    """(150,) uint8 lookup table ADE20K id -> 5-class id (unlisted ids -> obstacle)."""
    lut = np.full(ADE20K_NUM_CLASSES, ADE20K_DEFAULT_SEM, dtype=np.uint8)
    for ade_id, sem in ADE20K_TO_SEM5.items():
        lut[ade_id] = sem
    return lut


def ade20k_group_matrix() -> np.ndarray:
    """(5, 150) float32 0/1 matrix M so that p5 = M @ p150 sums probabilities per group."""
    lut = ade20k_lut()
    m = np.zeros((NUM_SEM_CLASSES, ADE20K_NUM_CLASSES), dtype=np.float32)
    m[lut, np.arange(ADE20K_NUM_CLASSES)] = 1.0
    return m


# --------------------------------------------------------------------------- helpers
def colorize(class_ids: np.ndarray) -> np.ndarray:
    """(H, W) uint8 class ids -> (H, W, 3) uint8 RGB using :data:`SEM_PALETTE`.

    Ids outside 0..4 (e.g. an ignore label 255) are drawn black.
    """
    ids = np.asarray(class_ids)
    out = np.zeros(ids.shape + (3,), dtype=np.uint8)
    valid = ids < NUM_SEM_CLASSES
    out[valid] = SEM_PALETTE[ids[valid]]
    return out


def softmax_channels(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over axis 0 of a (C, h, w) float32 array."""
    z = logits - logits.max(axis=0, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=0, keepdims=True)
    return z


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    """(C, h, w) probabilities -> (h, w) float32 entropy / ln(C) in [0, 1]."""
    c = probs.shape[0]
    p = np.clip(probs, 1e-8, 1.0)
    ent = -(p * np.log(p)).sum(axis=0) / math.log(c)
    return np.clip(ent, 0.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class SegModelSpec:
    """Everything the runtime needs to know about an exported segmentation model.

    Attributes:
        kind: ``"direct5"`` (5 logits) or ``"ade150"`` (ADE20K 150 logits, zero-shot).
        input_hw: model input (height, width) in pixels.
        mean, std: per-channel RGB normalisation applied to ``rgb / 255``.
        name: human-readable model name (from the sidecar).
        meta: the full sidecar dictionary (training data, metrics, licence ...).
    """

    kind: str
    input_hw: tuple[int, int]
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


DEFAULT_INPUT_HW = (320, 416)  # used only when neither sidecar nor static ONNX shape gives one


def sidecar_path(model_path: Path) -> Path:
    """``models/foo.onnx`` -> ``models/foo.json``."""
    return model_path.with_suffix(".json")


def load_model_spec(model_path: Path, session: Any = None) -> SegModelSpec:
    """Read the sidecar JSON if present, otherwise infer the spec from the ONNX graph."""
    sc = sidecar_path(model_path)
    meta: dict[str, Any] = {}
    if sc.exists():
        meta = json.loads(sc.read_text(encoding="utf-8"))
    kind = meta.get("kind")
    input_hw = meta.get("input_hw")
    if session is not None and (kind is None or input_hw is None):
        shp_in = session.get_inputs()[0].shape
        shp_out = session.get_outputs()[0].shape
        if input_hw is None and all(isinstance(s, int) for s in shp_in[2:4]):
            input_hw = [int(shp_in[2]), int(shp_in[3])]
        if kind is None:
            n_out = shp_out[1] if len(shp_out) > 1 else None
            kind = "ade150" if n_out == ADE20K_NUM_CLASSES else "direct5"
    if kind not in ("direct5", "ade150"):
        raise ValueError(f"unknown segmentation model kind {kind!r} for {model_path}")
    hw = tuple(int(v) for v in (input_hw or DEFAULT_INPUT_HW))
    return SegModelSpec(
        kind=kind,
        input_hw=(hw[0], hw[1]),
        mean=tuple(meta.get("mean", IMAGENET_MEAN)),  # type: ignore[arg-type]
        std=tuple(meta.get("std", IMAGENET_STD)),  # type: ignore[arg-type]
        name=str(meta.get("name", model_path.stem)),
        meta=meta,
    )


# --------------------------------------------------------------------------- segmenter
REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models"
# First existing file wins when the config does not name a model: the trained AWS
# models, then the CPU smoke model. The zero-shot SegFormer baseline is deliberately NOT
# a fallback (NVIDIA non-commercial licence, ~4x slower); it runs only when named in
# config["seg_model_path"].
DEFAULT_MODEL_CANDIDATES: tuple[str, ...] = (
    "lraspp_offroad5_robust.onnx",
    "lraspp_offroad5_clean.onnx",
    "lraspp_smoke.onnx",
)
DEFAULT_THREADS = 2  # ORT intra-op threads; the live loop shares 4 cores with stereo/VO


def resolve_model_path(config: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
    """Model path from ``config['seg_model_path']`` or the first existing default candidate."""
    cfg = config or {}
    explicit = cfg.get("seg_model_path")
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else (REPO_ROOT / p)
    for name in DEFAULT_MODEL_CANDIDATES:
        p = MODELS_DIR / name
        if p.exists():
            return p
    return None


class Segmenter:
    """ONNX Runtime CPU terrain segmenter (implements ``SegmenterProto``).

    Args:
        model_path: path to the ``.onnx`` file (sidecar ``.json`` optional). If ``None``,
            resolved from ``config`` (see :func:`resolve_model_path`).
        intra_op_threads: ORT intra-op thread count; ``None`` -> ``config['seg_threads']``
            or 2.
        input_hw: optional override of the model input (height, width); only valid for
            models exported with dynamic spatial axes (the SegFormer export).
        config: autonomy config dict. This lets ``AutonomyStack`` build the segmenter with
            its generic keyword wiring (``Segmenter(config=cfg)``).

    Raises ``FileNotFoundError`` when no model file exists (the node then runs without
    semantics). Call ``seg(rgb)`` with an (H, W, 3) uint8 RGB image. The node is
    responsible for the call rate (1/3 of the camera rate in the live loop).
    """

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        intra_op_threads: Optional[int] = None,
        input_hw: Optional[tuple[int, int]] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        import onnxruntime as ort  # local import keeps module import cheap for tests that only need tables

        cfg = config or {}
        if model_path is None:
            model_path = resolve_model_path(cfg)
            if model_path is None:
                raise FileNotFoundError(f"no segmentation model found (config seg_model_path={cfg.get('seg_model_path')!r}, searched {MODELS_DIR})")
        if intra_op_threads is None:
            intra_op_threads = int(cfg.get("seg_threads", DEFAULT_THREADS))
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(intra_op_threads)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self._sess = ort.InferenceSession(str(self.model_path), so, providers=["CPUExecutionProvider"])
        self._in_name = self._sess.get_inputs()[0].name
        spec = load_model_spec(self.model_path, self._sess)
        if input_hw is not None:
            static = self._sess.get_inputs()[0].shape[2:4]
            if all(isinstance(s, int) for s in static) and tuple(static) != tuple(input_hw):
                raise ValueError(f"model has static input {static}; cannot override to {input_hw}")
            spec = SegModelSpec(spec.kind, (int(input_hw[0]), int(input_hw[1])), spec.mean, spec.std, spec.name, spec.meta)
        self.spec = spec
        self.threads = int(intra_op_threads)
        mean = np.asarray(spec.mean, dtype=np.float32)
        std = np.asarray(spec.std, dtype=np.float32)
        # x_norm = rgb * scale + bias   with scale = 1 / (255 std), bias = -mean / std
        self._scale = (1.0 / (255.0 * std)).reshape(1, 1, 3)
        self._bias = (-mean / std).reshape(1, 1, 3)
        self._group = ade20k_group_matrix() if spec.kind == "ade150" else None
        self.last_timings_ms: dict[str, float] = {}
        LOG.info("Segmenter %s kind=%s input_hw=%s threads=%d", self.model_path.name, spec.kind, spec.input_hw, self.threads)

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]] = None) -> "Segmenter":
        """Build from an autonomy config dict.

        Keys (all optional): ``seg_model_path`` (str, absolute or repo-relative) and
        ``seg_threads`` (int, default 2). Raises ``FileNotFoundError`` if no model exists.
        """
        return cls(config=config or {})

    # ----------------------------------------------------------------- stages
    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 RGB -> (1, 3, h_in, w_in) float32 normalised NCHW."""
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"expected (H, W, 3) uint8 RGB, got {rgb.shape} {rgb.dtype}")
        h_in, w_in = self.spec.input_hw
        h, w = rgb.shape[:2]
        interp = cv2.INTER_AREA if (h > h_in and w > w_in) else cv2.INTER_LINEAR
        img = cv2.resize(rgb, (w_in, h_in), interpolation=interp) if (h, w) != (h_in, w_in) else rgb
        x = img.astype(np.float32) * self._scale + self._bias
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def infer_logits(self, x: np.ndarray) -> np.ndarray:
        """Run the ONNX graph; returns (C, h_out, w_out) float32 logits."""
        return self._sess.run(None, {self._in_name: x})[0][0]

    def probs5(self, logits: np.ndarray) -> np.ndarray:
        """Model logits -> (5, h_out, w_out) float32 five-class posterior."""
        p = softmax_channels(logits.astype(np.float32, copy=False))
        if self._group is not None:
            c, h, w = p.shape
            p = (self._group @ p.reshape(c, h * w)).reshape(NUM_SEM_CLASSES, h, w)
        return p

    def _to_input_res(self, p: np.ndarray) -> np.ndarray:
        """Bilinearly resize a (5, h, w) posterior to the model input size (SegFormer outputs 1/4 res)."""
        h_in, w_in = self.spec.input_hw
        if p.shape[1:] != (h_in, w_in):
            p = cv2.resize(p.transpose(1, 2, 0), (w_in, h_in), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        return p

    def predict_proba(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 -> (5, h_in, w_in) float32 posterior at the model input resolution."""
        return np.ascontiguousarray(self._to_input_res(self.probs5(self.infer_logits(self.preprocess(rgb)))))

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(H, W, 3) uint8 RGB -> (class_ids (H, W) uint8, entropy (H, W) float32 in [0, 1])."""
        t0 = time.perf_counter()
        x = self.preprocess(rgb)
        t1 = time.perf_counter()
        logits = self.infer_logits(x)
        t2 = time.perf_counter()
        p = self._to_input_res(self.probs5(logits))
        h, w = rgb.shape[:2]
        ids_small = p.argmax(axis=0).astype(np.uint8)
        ent_small = normalized_entropy(p)
        ids = cv2.resize(ids_small, (w, h), interpolation=cv2.INTER_NEAREST)
        ent = cv2.resize(ent_small, (w, h), interpolation=cv2.INTER_LINEAR)
        t3 = time.perf_counter()
        self.last_timings_ms = {
            "seg_pre": (t1 - t0) * 1e3,
            "seg_infer": (t2 - t1) * 1e3,
            "seg_post": (t3 - t2) * 1e3,
            "seg_total": (t3 - t0) * 1e3,
        }
        return ids, ent.astype(np.float32, copy=False)


__all__ = [
    "ADE20K_TO_SEM5",
    "ADE20K_DEFAULT_SEM",
    "HAZARD_CLASSES",
    "NUM_SEM_CLASSES",
    "SEM_COLORS",
    "SEM_PALETTE",
    "SegModelSpec",
    "Segmenter",
    "TRAVERSABLE_CLASSES",
    "ade20k_group_matrix",
    "ade20k_lut",
    "colorize",
    "load_model_spec",
    "normalized_entropy",
    "resolve_model_path",
    "softmax_channels",
]
