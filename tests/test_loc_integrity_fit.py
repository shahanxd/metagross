"""Integrity-monitor training pipeline on synthetic, separable per-frame data."""

from __future__ import annotations

import json

import numpy as np
import pytest

import metagross.eval.integrity_train as it
from metagross.autonomy.localization.health import FEATURE_NAMES, LogisticIntegrityModel


def _fake_npz(path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    label = np.zeros(n, bool)
    label[1:] = rng.random(n - 1) < 0.2
    X = np.empty((n, len(FEATURE_NAMES)))
    good = {"vo_inliers": 300, "vo_inlier_ratio": 0.85, "vo_reproj_rmse_px": 0.4, "vo_coverage": 0.8,
            "vo_track_age": 5, "vo_hess_min_eig": 5e4, "img_lapvar": 400, "img_sat_frac": 0.01,
            "img_dark_frac": 0.01, "img_rms_contrast": 0.2, "img_dark_channel": 0.1, "disp_density_ground": 0.8}
    for j, name in enumerate(FEATURE_NAMES):
        X[:, j] = good[name] * (1.0 + 0.1 * rng.standard_normal(n))
    j_in = FEATURE_NAMES.index("vo_inliers")
    X[label, j_in] = rng.uniform(15, 40, label.sum())  # failures have few inliers
    ok = np.ones(n, bool)
    ok[0] = False
    step = np.eye(4)
    step[2, 3] = 1.0
    gt = np.stack([np.eye(4) @ np.linalg.matrix_power(step, k) for k in range(n)])
    rel_vo = np.tile(step, (n, 1, 1))
    rel_vo[label, 2, 3] = 1.5  # failed frames are 50 % too long
    np.savez_compressed(path, X=X, ok=ok, label=label, terr=np.zeros(n), rerr=np.zeros(n), rel_vo=rel_vo, gt=gt,
                        kinds=np.array(["clean"] * n), severity=np.zeros(n, int), vo_ms=np.zeros(n), seed=seed)


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    for attr, sub in (("RAW", "raw"), ("RESULTS", "results"), ("DECK", "deck")):
        monkeypatch.setattr(it, attr, tmp_path / sub)
    monkeypatch.setattr(it, "MODEL_PATH", tmp_path / "models" / "integrity.json")
    (tmp_path / "raw").mkdir()
    _fake_npz(tmp_path / "raw" / "integrity_91.npz", 400, 1)
    _fake_npz(tmp_path / "raw" / "integrity_92.npz", 300, 2)
    return tmp_path


def test_fit_learns_separable_signal(sandbox):
    res = it.fit(["91"], "92")
    assert res["meta"]["auroc_test"] > 0.95
    assert res["coef"]["vo_inliers"] < 0  # more inliers -> lower failure probability
    model = LogisticIntegrityModel.load(sandbox / "models" / "integrity.json")
    assert model.source.endswith("integrity.json")
    assert (sandbox / "deck" / "integrity_roc.png").exists()
    saved = json.loads((sandbox / "results" / "integrity.json").read_text())
    d = saved["drift_test"]
    # gating rejects the 50 %-too-long frames, so the gated trajectory drifts less
    assert d["health_gating"]["ate_rmse_m"] < d["no_gating"]["ate_rmse_m"]


def test_fit_refuses_same_sequence(sandbox):
    with pytest.raises(ValueError):
        it.fit(["91"], "91")
