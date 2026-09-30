"""PyTorch dataset for the GAIA-URJC five-label layouts (RUGD-5L resized, OFFROAD5).

``SegDataset`` wraps a list of :class:`~metagross.train.data_index.SampleRef` and returns

    image  float32 (3, H, W)  ImageNet-normalised RGB
    label  int64   (H, W)     class ids 0..4, 255 = ignore
    index  int                position in the sample list (to look up source tags)

Augmentation randomness is derived from ``(seed, epoch, index)`` so results do not
depend on the number of DataLoader workers. Call :meth:`SegDataset.set_epoch` before
each epoch (the DataLoader must not use persistent workers).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from metagross.autonomy.perception.semantics import IMAGENET_MEAN, IMAGENET_STD, NUM_SEM_CLASSES
from metagross.train.augment import AugConfig, augment, resize_pair
from metagross.train.data_index import IGNORE_INDEX, SampleRef, list_samples, read_mask, read_rgb, subset

LOG = logging.getLogger(__name__)

DATASETS = ("rugd5", "offroad5")
CLASS_FREQ_MAX_MASKS = 400  # masks sampled to estimate class frequencies (deterministic)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> (3, H, W) float32 ImageNet-normalised."""
    mean = np.asarray(IMAGENET_MEAN, np.float32) * 255.0
    std = np.asarray(IMAGENET_STD, np.float32) * 255.0
    return ((img.astype(np.float32) - mean) / std).transpose(2, 0, 1)


class SegDataset(Dataset):
    """Image/label pairs with a seeded augmentation policy.

    Args:
        samples: indexed files (see :func:`metagross.train.data_index.list_samples`).
        out_hw: (height, width) of the returned tensors.
        policy: ``clean`` | ``robust`` (training) or ``none`` (plain resize, validation).
        seed: base seed for augmentation draws.
    """

    def __init__(self, samples: Sequence[SampleRef], out_hw: tuple[int, int], policy: str = "none", seed: int = 0) -> None:
        self.samples = list(samples)
        self.cfg = AugConfig(out_hw=out_hw, policy=policy)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def sources(self) -> list[str]:
        return [s.source for s in self.samples]

    def load_raw(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.samples[index]
        img, lab = read_rgb(s.image), read_mask(s.mask)
        if img.shape[:2] != lab.shape:
            img = resize_pair(img, lab, lab.shape)[0]
        lab = np.where(lab < NUM_SEM_CLASSES, lab, IGNORE_INDEX).astype(np.uint8)
        return img, lab

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.item(index, aug_key=self.epoch)

    def item(self, index: int, aug_key: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Sample ``index`` with the augmentation drawn from ``rng([seed, aug_key, index])``.

        ``aug_key`` is the epoch for ordinary loaders and the stream position for
        :class:`~metagross.train.sampling.StreamDataset`.
        """
        img, lab = self.load_raw(index)
        if self.cfg.policy == "none":
            img, lab = resize_pair(img, lab, self.cfg.out_hw)
        else:
            rng = np.random.default_rng([self.seed, int(aug_key), index])
            img, lab = augment(img, lab, self.cfg, rng)
        return torch.from_numpy(normalize_image(img)), torch.from_numpy(lab.astype(np.int64)), index


def build_splits(data: str, root: Path, train_subset: int = 0, val_subset: int = 0, seed: int = 0) -> tuple[list[SampleRef], list[SampleRef]]:
    """Dataset-provided train / val lists (optionally seeded subsets)."""
    if data not in DATASETS:
        raise ValueError(f"data must be one of {DATASETS}")
    train = subset(list_samples(root, "train"), train_subset, seed)
    val = subset(list_samples(root, "val"), val_subset, seed + 1)
    return train, val


def class_frequencies(samples: Sequence[SampleRef], max_masks: int = CLASS_FREQ_MAX_MASKS, seed: int = 0) -> np.ndarray:
    """Pixel frequency of each of the 5 classes over a seeded subset of masks (sums to 1)."""
    counts = np.zeros(NUM_SEM_CLASSES, np.float64)
    for s in subset(samples, max_masks, seed):
        m = read_mask(s.mask)
        counts += np.bincount(m[m < NUM_SEM_CLASSES].ravel(), minlength=NUM_SEM_CLASSES)[:NUM_SEM_CLASSES]
    return counts / max(counts.sum(), 1.0)


def enet_class_weights(freq: np.ndarray, c: float = 1.02) -> np.ndarray:
    """ENet weighting w_k = 1 / ln(c + f_k), normalised to mean 1 (bounded for rare classes)."""
    w = 1.0 / np.log(c + np.asarray(freq, np.float64))
    return (w / w.mean()).astype(np.float32)
