"""Tiny synthetic datasets in the GAIA on-disk layouts (for tests and pipeline dry-runs).

Each image is a toy outdoor scene: sky band (0) at the top, an obstacle blob (1),
a water patch (2), grass (3) and a paved strip (4) on the ground, with colours that
correlate with the class so a model can actually learn something in a few iterations.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from metagross.train.data_index import LAYOUTS

# Mean RGB per class for the toy scenes.
CLASS_RGB = np.array([[150, 190, 235], [70, 60, 50], [40, 80, 150], [70, 140, 50], [120, 120, 120]], np.float32)


def synthetic_pair(hw: tuple[int, int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One (rgb uint8 (H, W, 3), label uint8 (H, W)) toy scene."""
    h, w = hw
    lab = np.full((h, w), 3, np.uint8)
    horizon = int(h * rng.uniform(0.3, 0.45))
    lab[:horizon] = 0
    x0 = int(rng.uniform(0.3, 0.6) * w)
    lab[horizon:, x0 : x0 + max(2, w // 6)] = 4  # paved strip
    cy, cx = int(rng.uniform(horizon + 2, h - 2)), int(rng.uniform(0, w))
    cv2.circle(lab, (cx, cy), max(2, h // 8), 1, -1)  # obstacle
    wy, wx = int(rng.uniform(horizon + 2, h - 2)), int(rng.uniform(0, w))
    cv2.ellipse(lab, (wx, wy), (max(2, w // 10), max(1, h // 16)), 0, 0, 360, 2, -1)  # water
    rgb = CLASS_RGB[lab] + rng.normal(0, 12, (h, w, 3))
    return np.clip(rgb, 0, 255).astype(np.uint8), lab


def write_synthetic_dataset(root: Path, layout: str = "rugd5", n_per_split: int = 6, hw: tuple[int, int] = (64, 80), seed: int = 0) -> Path:
    """Write train/val/test splits under ``root`` in ``layout``; returns ``root``.

    For ``offroad5`` the names cycle through RUGD, RELLIS-3D and GOOSE conventions so
    per-source tagging can be tested.
    """
    img_dir_name, mask_dir_name = LAYOUTS[layout]
    rng = np.random.default_rng(seed)
    for split in ("train", "val", "test"):
        img_dir, mask_dir = root / split / img_dir_name, root / split / mask_dir_name
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_split):
            rgb, lab = synthetic_pair(hw, rng)
            kind = i % 3 if layout == "offroad5" else 0
            if kind == 0:
                stem, suffix = f"trail-{i % 2 + 1}_{i:05d}", "_group5"
            elif kind == 1:
                stem, suffix = f"frame{i:06d}-1581623950_{i:03d}", "_group5"
            else:
                stem, suffix = f"2022-11-11_aying__{i:04d}_1668158754128964651_windshield_vis", "_group5b"
            cv2.imwrite(str(img_dir / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mask_dir / f"{stem}{suffix}.png"), lab)
    return root
