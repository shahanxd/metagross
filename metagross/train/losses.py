"""Loss: class-weighted cross-entropy + 0.5 x soft multi-class Dice.

CE with ENet class weights keeps the rare, safety-relevant classes (water/mud is ~0.1-1 %
of RUGD pixels) from being ignored; the Dice term optimises region overlap directly and
is insensitive to class frequency. Pixels labelled ``ignore_index`` contribute to neither.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

DICE_WEIGHT = 0.5
DICE_SMOOTH = 1.0  # Laplace smoothing; keeps absent classes well-defined


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255, smooth: float = DICE_SMOOTH) -> torch.Tensor:
    """1 - mean_k Dice_k over all classes; logits (N, C, H, W), target (N, H, W) int64."""
    n_cls = logits.shape[1]
    valid = (target != ignore_index).unsqueeze(1).to(logits.dtype)  # (N, 1, H, W)
    probs = logits.float().softmax(dim=1) * valid
    onehot = F.one_hot(target.clamp(0, n_cls - 1), n_cls).permute(0, 3, 1, 2).to(probs.dtype) * valid
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


class CEDiceLoss(nn.Module):
    """``CE(weight=class_weights) + dice_weight * Dice``; returns (total, parts dict)."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None, dice_weight: float = DICE_WEIGHT, ignore_index: int = 255) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.empty(0))
        self.dice_weight = float(dice_weight)
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        w = self.class_weights if self.class_weights.numel() else None
        ce = F.cross_entropy(logits.float(), target, weight=w, ignore_index=self.ignore_index)
        dice = soft_dice_loss(logits, target, self.ignore_index)
        total = ce + self.dice_weight * dice
        return total, {"ce": float(ce.detach()), "dice": float(dice.detach())}
