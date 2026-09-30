"""Segmentation model factory.

* ``lraspp`` - torchvision LR-ASPP MobileNetV3-Large (BSD-3-Clause code, ~3.2 M params).
  The **deploy** model: small, CPU-fast, exports cleanly to ONNX opset 17.
  ``init='imagenet'`` loads the ImageNet-1k MobileNetV3 backbone (torchvision download);
  ``init='coco'`` loads the full COCO-with-VOC-labels segmentation weights and replaces
  the two classifier convs; ``init='none'`` is random (tests).
* ``segformer_b0`` - optional research comparison (HF ``transformers``, NVIDIA
  non-commercial licence). Not used for deployment. Needs ``pip install transformers``
  (installed by ``aws/setup_and_train.sh``, not in the laptop venv).

Every model is wrapped in :class:`SegNet`, whose ``forward`` maps a normalised
(N, 3, H, W) batch to (N, 5, H, W) logits at the input resolution.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

LOG = logging.getLogger(__name__)

MODELS = ("lraspp", "segformer_b0")
INITS = ("imagenet", "coco", "none")


class SegNet(nn.Module):
    """Uniform wrapper: normalised image batch -> logits at input resolution."""

    def __init__(self, core: nn.Module, kind: str) -> None:
        super().__init__()
        self.core = core
        self.kind = kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "lraspp":
            return self.core(x)["out"]
        logits = self.core(pixel_values=x).logits  # segformer: (N, C, H/4, W/4)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def _lraspp(num_classes: int, init: str) -> nn.Module:
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights, lraspp_mobilenet_v3_large

    if init == "coco":
        m = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
        low, high = m.classifier.low_classifier, m.classifier.high_classifier
        m.classifier.low_classifier = nn.Conv2d(low.in_channels, num_classes, 1)
        m.classifier.high_classifier = nn.Conv2d(high.in_channels, num_classes, 1)
        return m
    wb = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if init == "imagenet" else None
    return lraspp_mobilenet_v3_large(weights=None, weights_backbone=wb, num_classes=num_classes)


def _segformer_b0(num_classes: int, init: str) -> nn.Module:
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("segformer_b0 needs `pip install transformers` (AWS setup installs it)") from exc
    if init == "none":
        return SegformerForSemanticSegmentation(SegformerConfig.from_pretrained("nvidia/mit-b0", num_labels=num_classes))
    # MiT-B0 encoder is ImageNet-1k pretrained; the decode head is new.
    return SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=num_classes)


def build_model(name: str, num_classes: int = 5, init: str = "imagenet") -> SegNet:
    """Construct a wrapped segmentation model (see module docstring)."""
    if name not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if init not in INITS:
        raise ValueError(f"init must be one of {INITS}")
    core = _lraspp(num_classes, init) if name == "lraspp" else _segformer_b0(num_classes, init)
    net = SegNet(core, name)
    LOG.info("built %s (init=%s): %.2f M params", name, init, sum(p.numel() for p in net.parameters()) / 1e6)
    return net


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
