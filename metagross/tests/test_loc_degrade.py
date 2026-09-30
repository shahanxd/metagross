"""Degradation models (eval) and integrity-training helpers, on synthetic images."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metagross.autonomy.localization.health import image_features
from metagross.eval.degrade import (
    AIRLIGHT,
    KINDS,
    Degradation,
    Degrader,
    haze,
    make_schedule,
    motion_blur_kernel,
)
from metagross.eval.integrity_train import integrate
from tests.test_loc_synth import make_texture


@pytest.fixture(scope="module")
def pair() -> tuple[np.ndarray, np.ndarray]:
    t = make_texture(5)
    return t[:240, :320].copy(), t[:240, 8:328].copy()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("sev", [1, 3])
def test_every_kind_runs_and_preserves_shape(pair, kind, sev):
    left, right = pair
    depth = np.full(left.shape, 20.0, np.float32)
    dl, dr = Degrader(Degradation(kind, sev), left.shape, seed=3)(left, right, depth)
    assert dl.shape == left.shape and dl.dtype == np.uint8 and dr.shape == right.shape
    assert not np.array_equal(dl, left)


def test_severity_is_monotone(pair):
    left, right = pair
    f = image_features(left)
    lap = [image_features(Degrader(Degradation("motion_blur", s), left.shape, 0)(left, right)[0])["img_lapvar"]
           for s in (1, 2, 3)]
    assert f["img_lapvar"] > lap[0] > lap[1] > lap[2]
    dark = [image_features(Degrader(Degradation("gamma_dark", s), left.shape, 0)(left, right)[0])["img_dark_frac"]
            for s in (1, 2, 3)]
    assert dark[0] < dark[1] < dark[2]


def test_haze_koschmieder_limits():
    J = np.full((10, 10), 50, np.uint8)
    near = haze(J, np.full((10, 10), 1e-3, np.float32), 0.1)  # depth <= 0 would mean "invalid"
    far = haze(J, np.full((10, 10), np.inf, np.float32), 0.1)  # invalid depth -> far
    assert np.all(near == 50)
    assert np.all(np.abs(far.astype(int) - AIRLIGHT) <= 1)
    mid = haze(J, np.full((10, 10), 10.0, np.float32), 0.1)
    t = math.exp(-1.0)
    assert int(mid[0, 0]) == pytest.approx(50 * t + AIRLIGHT * (1 - t), abs=1)


def test_smudge_only_left(pair):
    left, right = pair
    dl, dr = Degrader(Degradation("smudge", 3), left.shape, 1)(left, right)
    assert np.array_equal(dr, right) and not np.array_equal(dl, left)


def test_blur_kernel_normalised():
    k = motion_blur_kernel(15, 0.3)
    assert k.sum() == pytest.approx(1.0) and k.shape == (15, 15)


def test_schedule_deterministic_and_mixed():
    a, b = make_schedule(500, 7), make_schedule(500, 7)
    assert [None if x is None else (x[0], x[1]) for x in a] == [None if x is None else (x[0], x[1]) for x in b]
    frac = np.mean([x is not None for x in a])
    assert 0.2 < frac < 0.6 and len(a) == 500
    with pytest.raises(ValueError):
        Degradation("rain", 1)


def test_integrate_bridges_rejected_frames():
    step = np.eye(4)
    step[2, 3] = 1.0
    rel = np.tile(step, (6, 1, 1))
    rel[3, 2, 3] = 50.0  # a gross VO failure
    use = np.array([False, True, True, False, True, True])
    poses = integrate(rel, use)
    assert poses[-1, 2, 3] == pytest.approx(5.0)  # frame 3 bridged with the previous 1 m step
