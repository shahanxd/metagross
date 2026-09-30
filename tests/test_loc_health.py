"""Integrity monitor: features respond to degradations, p_fail ordering, model IO."""

from __future__ import annotations

import json

import numpy as np
import pytest

from metagross.autonomy.localization.health import (
    FEATURE_NAMES,
    MODEL_SCHEMA,
    IntegrityMonitor,
    LogisticIntegrityModel,
    compute_features,
    disparity_density,
    image_features,
)
from tests.test_loc_synth import make_texture

GOOD_STATS = {"inliers": 300.0, "inlier_ratio": 0.9, "reproj_rmse_px": 0.4, "coverage": 0.8,
              "track_age": 5.0, "hess_min_eig": 5e4, "n_tracks": 400.0, "n_3d": 330.0}


@pytest.fixture(scope="module")
def img() -> np.ndarray:
    return make_texture(1)[:240, :320].copy()


def test_twelve_features(img):
    f = compute_features(GOOD_STATS, img, np.ones_like(img, np.float32))
    assert tuple(sorted(f)) == tuple(sorted(FEATURE_NAMES)) and len(f) == 12
    assert f["disp_density_ground"] == pytest.approx(1.0)


def test_photometric_features_react(img):
    base = image_features(img)
    dark = image_features((img * 0.02).astype(np.uint8))
    sat = image_features(np.clip(img.astype(int) * 4, 0, 255).astype(np.uint8))
    blur = image_features(np.full_like(img, 128))
    assert dark["img_dark_frac"] > 0.9 > base["img_dark_frac"]
    assert sat["img_sat_frac"] > 0.5 > base["img_sat_frac"]
    assert blur["img_lapvar"] < 1e-6 < base["img_lapvar"]
    assert blur["img_rms_contrast"] == pytest.approx(0.0)
    haze = image_features((0.3 * img + 0.7 * 230).astype(np.uint8))
    assert haze["img_dark_channel"] > base["img_dark_channel"]


def test_rgb_dark_channel_matches_numpy_reference():
    """The cv2.min channel minimum must reproduce the original numpy min(axis=2) feature exactly."""
    import cv2

    rng = np.random.default_rng(4)
    rgb = rng.integers(0, 256, (120, 160, 3)).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small3 = cv2.resize(rgb, (80, 60), interpolation=cv2.INTER_AREA)
    ref = cv2.erode(small3.min(axis=2), cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))).mean() / 255.0
    assert image_features(gray, rgb)["img_dark_channel"] == pytest.approx(float(ref), abs=1e-12)


def test_disparity_density():
    d = np.zeros((100, 100), np.float32)
    assert disparity_density(d) == 0.0 and disparity_density(None) == 0.0
    d[60:, :] = 5.0
    assert disparity_density(d) == pytest.approx(1.0)


def test_fallback_orders_good_vs_failed(img):
    m = LogisticIntegrityModel.fallback()
    p_good = m.p_fail(compute_features(GOOD_STATS, img, np.ones_like(img, np.float32)))
    p_bad = m.p_fail(compute_features(None, np.full_like(img, 3), None))
    assert p_good < 0.2 and p_bad > 0.95


def test_monitor_asymmetric_ema(img):
    mon = IntegrityMonitor(model=LogisticIntegrityModel.fallback())
    disp = np.ones_like(img, np.float32)
    q_good = mon.update(GOOD_STATS, img, disp)["q"]
    bad = mon.update(None, img, None)
    assert bad["q"] < q_good and bad["q_gate"] == pytest.approx(bad["q_inst"])
    qs = [mon.update(GOOD_STATS, img, disp)["q"] for _ in range(6)]
    assert qs[0] < q_good - 0.1  # recovery is slow...
    assert qs[-1] > qs[0]  # ...but monotone


def test_model_roundtrip(tmp_path, img):
    m = LogisticIntegrityModel.fallback()
    p = tmp_path / "integrity.json"
    p.write_text(json.dumps(m.to_json({"note": "test"})))
    m2 = LogisticIntegrityModel.load(p)
    assert m2.source == str(p)
    f = compute_features(GOOD_STATS, img, None)
    assert m2.p_fail(f) == pytest.approx(m.p_fail(f))
    bad = json.loads(p.read_text())
    bad["schema"] = "other"
    p.write_text(json.dumps(bad))
    assert LogisticIntegrityModel.load(p).source == "fallback"
    assert LogisticIntegrityModel.load(tmp_path / "missing.json").source == "fallback"
    assert m.to_json()["schema"] == MODEL_SCHEMA
