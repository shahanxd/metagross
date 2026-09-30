"""Perception: ground certification, ditch persistence, ditch cluster filter, noise estimate."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metagross.autonomy.perception import synthetic as syn
from metagross.autonomy.perception.bev import BevSpec
from metagross.autonomy.perception.certify import CERT_CLEARANCE_M, GroundCertifier, ditch_detection_range_m
from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.autonomy.perception.negobs import LAB_GROUND, band_noise_px, filter_ditch_cells
from metagross.autonomy.perception.persistence import DitchPersistence
from metagross.autonomy.perception.pipeline import Perception
from metagross.config import defaults
from metagross.contracts.messages import CellState, SensorFrame
from metagross.eval.theory import ditch_detection_range_m as theory_r_det

CAL = defaults.stereo_calibration()


@pytest.fixture(scope="module")
def geom() -> CameraGeometry:
    return CameraGeometry(CAL)


def test_detection_range_matches_theory():
    for w in (0.3, 0.5, 1.0):
        for h in (0.5, 0.9, 1.4):
            assert ditch_detection_range_m(w, h, defaults.FX) == pytest.approx(
                float(theory_r_det(w, h, defaults.FX, float(defaults.MIN_PIXELS_ON_TARGET))), rel=1e-9)
    assert float(ditch_detection_range_m(0.3, 0.0, defaults.FX)) == 0.0


def test_certified_only_within_design_ditch_range(geom):
    spec = BevSpec()
    cert = GroundCertifier(geom, spec)
    state = np.full(spec.shape, CellState.GROUND, np.uint8)
    out = cert.certify(state, np.zeros(spec.shape, np.float32))
    r_det = cert.r_det_m(geom.t_bc[2])
    assert r_det == pytest.approx(float(theory_r_det(defaults.DESIGN_DITCH_WIDTH_M, defaults.CAM_HEIGHT_M, defaults.FX, 6.0)), rel=1e-6)
    assert np.all(cert.r_cam[out] <= r_det + 1e-6)
    assert out[cert.r_cam < r_det - 0.2].all()
    # Uphill terrain (camera closer to it) certifies closer; non-GROUND is never certified.
    up = cert.certify(state, np.full(spec.shape, 0.4, np.float32))
    assert up.sum() < out.sum()
    state[:] = CellState.UNSEEN
    assert not cert.certify(state, np.zeros(spec.shape, np.float32)).any()


def test_no_certification_next_to_lethal_evidence(geom):
    spec = BevSpec()
    cert = GroundCertifier(geom, spec)
    X, Y = spec.centres()
    state = np.full(spec.shape, CellState.GROUND, np.uint8)
    rock = (np.abs(X - 2.5) < 0.15) & (np.abs(Y) < 0.15)
    state[rock] = CellState.POSITIVE
    out = cert.certify(state, np.zeros(spec.shape, np.float32))
    near = (np.abs(X - 2.5) < 0.15 + CERT_CLEARANCE_M) & (np.abs(Y) < 0.15 + CERT_CLEARANCE_M)
    assert not out[near].any()
    assert out[(np.abs(X - 2.5) < 0.1) & (np.abs(Y - 1.0) < 0.1)].all()


def test_pipeline_outputs_certification_and_trench_cells_uncertified():
    p = Perception(CAL, defaults.VEHICLE, {})
    d = syn.render_disparity(p.geom, syn.make_scene("trench", x0=3.0, width=0.5, depth=0.6), noise_px=0.1, seed=7)
    out = p.process(SensorFrame(0.0, 0, None, None, 0.0, 0.0, 0.0, sensor_mode="tier0_disparity", disparity=d))
    cert = out["certified_local"]
    assert cert.dtype == bool and cert.shape == out["cell_state_local"].shape
    assert 3.5 < out["r_det_m"] < 5.0
    X, Y = p.spec.centres()
    trench = (X >= 3.0) & (X <= 3.5) & (np.abs(Y) < 1.0)
    assert not cert[trench].any()
    assert cert[(X > 1.5) & (X < 2.5) & (np.abs(Y) < 0.5)].mean() > 0.9
    assert not cert[X > out["r_det_m"] + defaults.CAM_FORWARD_M + 0.3].any()


def test_persistence_confirms_static_and_drops_flicker():
    spec = BevSpec()
    per = DitchPersistence(spec)
    X, Y = spec.centres()
    band = (np.abs(X - 5.0) < 0.2) & (np.abs(Y) < 1.0)
    conf, unc = per.filter(band, (0.0, 0.0, 0.0))  # first frame: bootstrap, passed through
    assert conf.sum() == band.sum() and not unc.any()
    # Vehicle moved 0.2 m forward: the static trench is 0.2 m closer in the body frame.
    band2 = (np.abs(X - 4.8) < 0.2) & (np.abs(Y) < 1.0)
    flicker = (np.abs(X - 8.0) < 0.1) & (np.abs(Y - 2.0) < 0.3)
    conf, unc = per.filter(band2 | flicker, (0.2, 0.0, 0.0))
    assert conf[band2].mean() > 0.95 and not conf[flicker].any() and unc[flicker].all()


def test_persistence_uses_yaw():
    spec = BevSpec()
    per = DitchPersistence(spec)
    X, Y = spec.centres()
    per.filter((np.abs(X - 5.0) < 0.2) & (np.abs(Y) < 0.5), (0.0, 0.0, 0.0))
    # Same world patch seen after a 90 deg left turn in place: it is now on the right (y < 0).
    yaw = math.pi / 2
    seen = (np.abs(X - 0.0) < 0.5) & (np.abs(Y + 5.0) < 0.2)
    conf, _ = per.filter(seen, (0.0, 0.0, yaw))
    assert conf[seen].mean() > 0.9


def test_filter_ditch_cells_keeps_crossing_band_drops_radial_streak():
    spec = BevSpec()
    X, Y = spec.centres()
    crossing = (np.abs(X - 5.0) < 0.2) & (np.abs(Y) < 0.8)
    cam = (defaults.CAM_FORWARD_M, 0.0)
    r = np.hypot(X - cam[0], Y - cam[1])
    az = np.arctan2(Y - cam[1], X - cam[0])
    streak = (np.abs(az - 0.25) < 0.01) & (r > 6.0) & (r < 9.0)  # along one line of sight
    out = filter_ditch_cells(crossing | streak, X, Y, cam)
    assert out[crossing].all() and not out[streak].any()
    assert not filter_ditch_cells(np.zeros_like(crossing), X, Y, (0.3, 0.0)).any()


def test_fast_nanmedian_matches_numpy():
    import warnings

    from metagross.autonomy.perception.negobs import _nanmedian0

    rng = np.random.default_rng(1)
    for shape in ((3, 40), (5, 9), (1, 4)):
        a = rng.normal(size=shape)
        a[rng.random(shape) < 0.3] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = np.nanmedian(a, 0)
        assert np.allclose(_nanmedian0(a), ref, equal_nan=True)


def test_band_noise_estimate_tracks_injected_noise():
    rng = np.random.default_rng(0)
    lab = np.full((200, 160), LAB_GROUND, np.uint8)
    for sigma in (0.1, 0.3):
        res = rng.normal(0.0, sigma, lab.shape)
        assert band_noise_px(res, lab) == pytest.approx(sigma, rel=0.15)
    assert band_noise_px(np.zeros((5, 5)), np.zeros((5, 5), np.uint8)) > 0  # fallback when no ground
