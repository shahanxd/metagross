"""Photometric gain/offset normalisation before KLT (synthetic, exact geometry)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metagross.autonomy.localization.vo import (
    StereoVO,
    VOConfig,
    apply_gain_offset,
    photometric_gain_offset,
    rotation_angle,
    sample_patch_means,
)
from tests.test_loc_synth import BASELINE_M, K, disparity_from_depth, make_texture, pose, render


@pytest.fixture(scope="module")
def tex() -> np.ndarray:
    return make_texture(5)


def _drive(tex: np.ndarray, gains: list[float], cfg: VOConfig, offsets: list[float] | None = None):
    """Forward motion 0.3 m / frame; frame k is rendered then brightness-mapped g_k * I + o_k."""
    offsets = offsets or [0.0] * len(gains)
    vo = StereoVO(K(), BASELINE_M, cfg)
    poses = [pose(0.02 * k, 0, 0.3 * k, math.radians(1.0 * k)) for k in range(len(gains))]
    out = []
    for k, T in enumerate(poses):
        img, depth = render(T, tex)
        img = apply_gain_offset(img, gains[k], offsets[k])
        out.append(vo.process(img, disparity=disparity_from_depth(depth), t=0.2 * k))
    return poses, out


def test_gain_offset_estimate_recovers_affine_change(tex):
    img, _ = render(np.eye(4), tex)
    rng = np.random.default_rng(1)
    pts = np.stack([rng.uniform(20, img.shape[1] - 20, 400), rng.uniform(20, img.shape[0] - 20, 400)], axis=1)
    a = sample_patch_means(img, pts, 9)
    b = sample_patch_means(apply_gain_offset(img, 0.5, 10.0), pts, 9)
    g, o = photometric_gain_offset(a, b)
    assert g == pytest.approx(0.5, abs=0.03)  # 8-bit rounding of the darkened image
    assert o == pytest.approx(10.0, abs=2.0)
    g1, o1 = photometric_gain_offset(a, a)
    assert g1 == pytest.approx(1.0, abs=1e-6) and o1 == pytest.approx(0.0, abs=1e-6)


def test_estimate_ignores_content_change_outside_the_tracks(tex):
    """Whole-image moments would see the sky patch; samples at the tracks do not."""
    img, _ = render(np.eye(4), tex)
    cur = img.copy()
    cur[:60, :] = 250  # bright 'sky' enters the top of the image, exposure unchanged
    rng = np.random.default_rng(2)
    pts = np.stack([rng.uniform(20, img.shape[1] - 20, 300), rng.uniform(90, img.shape[0] - 20, 300)], axis=1)
    g, o = photometric_gain_offset(sample_patch_means(img, pts, 9), sample_patch_means(cur, pts, 9))
    assert g == pytest.approx(1.0, abs=0.02) and abs(o) < 2.0


def test_gain_clipped_and_flat_samples_offset_only():
    g, o = photometric_gain_offset(np.full(50, 100.0), np.full(50, 130.0))
    assert g == 1.0 and o == pytest.approx(30.0)
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 255, 200)
    g, _ = photometric_gain_offset(a, a / 50.0, gain_min=0.25)
    assert g == pytest.approx(0.25)


def test_patch_means_clamp_to_border():
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    v = sample_patch_means(img, np.array([[-5.0, -5.0], [4.0, 4.0]]), 1)
    assert v.tolist() == [0.0, 44.0]


def test_apply_gain_offset_is_clipped_lut():
    x = np.arange(256, dtype=np.uint8).reshape(16, 16)
    y = apply_gain_offset(x, 2.0, -10.0)
    assert y.dtype == np.uint8
    assert y[0, 0] == 0 and y.max() == 255 and y.flat[20] == 30


def test_exposure_step_tracked_with_normalisation(tex):
    """A x0.4 dimming step (auto-exposure / F5 dim event) between frames 1 and 2."""
    gains = [1.0, 1.0, 0.4, 0.4]
    poses, res = _drive(tex, gains, VOConfig())
    r = res[2]
    assert r.ok, r.reason
    assert r.stats["photo_gain"] == pytest.approx(0.4, abs=0.05)
    T_true = np.linalg.inv(poses[1]) @ poses[2]
    assert np.linalg.norm(r.T_prev_cur[:3, 3] - T_true[:3, 3]) < 0.03
    assert rotation_angle(r.T_prev_cur[:3, :3].T @ T_true[:3, :3]) < math.radians(0.5)
    # without normalisation the same step keeps fewer tracks through the forward-backward check
    _, raw = _drive(tex, gains, VOConfig(photometric_norm=False))
    assert r.stats["n_tracks"] >= raw[2].stats["n_tracks"]


def test_constant_exposure_leaves_images_untouched(tex):
    _, res = _drive(tex, [1.0, 1.0, 1.0], VOConfig())
    assert res[2].ok
    assert res[2].stats["photo_gain"] == 1.0 and res[2].stats["photo_offset_dn"] == 0.0
