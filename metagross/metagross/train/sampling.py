"""Step-indexed, deterministic sample streams for budgeted (CPU) training.

A *stream* is the concatenation of per-pass shuffles of the training-set indices. Pass
``p`` is drawn from ``numpy.random.default_rng([seed, STREAM_SALT, p])``, so the stream
is a pure function of ``(n, seed, repeat factors)``. Optimiser step ``k`` with batch size
``B`` consumes stream positions ``[k*B, (k+1)*B)``: training can stop at any step (a
wall-clock deadline, a killed process) and ``--resume`` continues with exactly the
batches an uninterrupted run would have seen. Each stream position ``q`` also keys the
augmentation draw of its sample, so an image repeated within a pass gets a different
augmentation each time and results do not depend on the number of DataLoader workers.

Repeat-factor sampling (RFS; Gupta et al., "LVIS", CVPR 2019), adapted to semantic
segmentation at image level:

* class ``c`` is *present* in an image if it covers at least ``min_frac`` of the image's
  labelled pixels (``min_frac`` is a pixel fraction, 0..1);
* ``f_c`` = fraction of training images in which ``c`` is present;
* class repeat factor ``r_c = max(1, sqrt(t / f_c))`` with threshold ``t`` (an image
  fraction, 0..1); image repeat factor ``r_i = max(1, max_{c present in i} r_c)``;
* each pass contains image ``i`` ``floor(r_i)`` times plus once more with probability
  ``r_i - floor(r_i)`` (stochastic rounding), then is shuffled.

Rare, safety-relevant classes (water/mud) are therefore seen more often without
changing the loss or the label set.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from metagross.autonomy.perception.semantics import NUM_SEM_CLASSES
from metagross.train.data_index import SampleRef, read_mask

LOG = logging.getLogger(__name__)

STREAM_SALT = 7_919  # separates stream shuffles from other seeded draws of the same seed
RFS_THRESH_DEFAULT = 0.2  # t: classes present in fewer than 20 % of images are repeated
RFS_MIN_FRAC_DEFAULT = 0.005  # a class counts as present if it covers >= 0.5 % of labelled pixels
_EPS = 1e-12


def image_class_fractions(samples: Sequence[SampleRef], num_classes: int = NUM_SEM_CLASSES) -> np.ndarray:
    """(n, C) float64: per image, the fraction of labelled pixels (ids < C) of each class."""
    out = np.zeros((len(samples), num_classes), np.float64)
    for i, s in enumerate(samples):
        m = read_mask(s.mask)
        counts = np.bincount(m.ravel(), minlength=256)[:num_classes].astype(np.float64)
        out[i] = counts / max(counts.sum(), 1.0)
    return out


@dataclass(frozen=True)
class RepeatFactors:
    """Result of :func:`repeat_factors` (all arrays indexed by class or image)."""

    image: np.ndarray  # (n,) float64 >= 1, per-image repeat factor r_i
    class_factor: np.ndarray  # (C,) float64 >= 1, r_c
    class_image_freq: np.ndarray  # (C,) float64, f_c = share of images where c is present

    def summary(self) -> dict[str, object]:
        """JSON-ready summary for training metadata."""
        return {
            "class_repeat_factor": np.round(self.class_factor, 4).tolist(),
            "class_image_freq": np.round(self.class_image_freq, 5).tolist(),
            "mean_image_factor": round(float(self.image.mean()), 4),
            "max_image_factor": round(float(self.image.max()), 4),
        }


def repeat_factors(fractions: np.ndarray, thresh: float = RFS_THRESH_DEFAULT, min_frac: float = RFS_MIN_FRAC_DEFAULT) -> RepeatFactors:
    """Image-level RFS factors from per-image class fractions (see module docstring)."""
    if not 0.0 < thresh <= 1.0:
        raise ValueError(f"thresh must be in (0, 1], got {thresh}")
    fr = np.asarray(fractions, np.float64)
    present = (fr > 0.0) & (fr >= min_frac)
    f_c = present.mean(axis=0) if len(fr) else np.zeros(fr.shape[1])
    r_c = np.where(f_c > 0, np.maximum(1.0, np.sqrt(thresh / np.maximum(f_c, _EPS))), 1.0)
    r_i = np.max(np.where(present, r_c[None, :], 1.0), axis=1) if len(fr) else np.zeros(0)
    return RepeatFactors(image=np.maximum(r_i, 1.0), class_factor=r_c, class_image_freq=f_c)


def pass_indices(n: int, rng: np.random.Generator, factors: Optional[np.ndarray] = None) -> np.ndarray:
    """One shuffled pass over ``n`` images (each repeated per ``factors`` if given)."""
    if factors is None:
        return rng.permutation(n).astype(np.int64)
    f = np.asarray(factors, np.float64)
    if f.shape != (n,) or (f < 1.0).any():
        raise ValueError("factors must be shape (n,) and >= 1")
    base = np.floor(f)
    reps = (base + (rng.random(n) < (f - base))).astype(np.int64)
    idx = np.repeat(np.arange(n, dtype=np.int64), reps)
    return idx[rng.permutation(len(idx))]


def build_stream(n: int, length: int, seed: int = 0, factors: Optional[np.ndarray] = None) -> np.ndarray:
    """First ``length`` positions of the sample stream (int64 dataset indices)."""
    if n <= 0:
        raise ValueError("empty dataset")
    parts: list[np.ndarray] = []
    total, p = 0, 0
    while total < length:
        part = pass_indices(n, np.random.default_rng([seed, STREAM_SALT, p]), factors)
        parts.append(part)
        total += len(part)
        p += 1
    return np.concatenate(parts)[:length] if parts else np.zeros(0, np.int64)


def stream_length(total_iters: int, batch_size: int) -> int:
    """Stream positions needed for ``total_iters`` optimiser steps."""
    return int(total_iters) * int(batch_size)


def epoch_step_range(epoch: int, epoch_iters: int, global_it: int, total_iters: int) -> tuple[int, int]:
    """Optimiser steps ``[start, end)`` still to run in ``epoch`` (empty if start >= end)."""
    start = max(global_it, epoch * epoch_iters)
    end = min((epoch + 1) * epoch_iters, total_iters)
    return start, end


def n_epochs_for(total_iters: int, epoch_iters: int) -> int:
    """Number of (virtual) epochs that cover ``total_iters`` steps."""
    return int(math.ceil(total_iters / max(1, epoch_iters)))


class StreamDataset(Dataset):
    """View of a :class:`~metagross.train.datasets.SegDataset` indexed by stream position.

    ``ds[q]`` returns ``base.item(stream[q], aug_key=q)`` = (image, label, base index).
    """

    def __init__(self, base: Dataset, stream: np.ndarray) -> None:
        self.base = base
        self.stream = np.asarray(stream, np.int64)

    def __len__(self) -> int:
        return len(self.stream)

    def __getitem__(self, q: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.base.item(int(self.stream[q]), aug_key=int(q))  # type: ignore[attr-defined]
