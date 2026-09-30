"""Perception: missing-ground detector (ditch / crest shadow / occlusion) on analytic scenes."""

from __future__ import annotations

import numpy as np
import pytest

from metagross.autonomy.perception import synthetic as syn
from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.autonomy.perception.ground import fit_ground
from metagross.autonomy.perception.negobs import MissingGroundDetector, NegObsResult
from metagross.config import defaults


@pytest.fixture(scope="module")
def geom() -> CameraGeometry:
    return CameraGeometry(defaults.stereo_calibration())


@pytest.fixture(scope="module")
def det(geom) -> MissingGroundDetector:
    return MissingGroundDetector(geom)


def _run(geom, det, scene, noise=0.0, seed=3, use_negobs=True) -> NegObsResult:
    d = syn.render_disparity(geom, scene, noise_px=noise, seed=seed)
    pts = geom.points_from_disparity(d, 2, 4)
    model = fit_ground(pts, d, geom, seed=0)
    return det.detect(d, model, use_negobs=use_negobs)


def _central(res: NegObsResult, seg, geom, half_width_px: int = 200):
    m = np.abs(res.col_u[seg.col] - geom.cx) < half_width_px
    return seg.x0[m], seg.x1[m]


@pytest.mark.parametrize("noise", [0.0, 0.15])
def test_flat_ground_has_no_false_gaps(geom, det, noise):
    r = _run(geom, det, syn.make_scene("flat"), noise)
    assert len(r.ditch) == 0 and len(r.crest) == 0 and len(r.occluded) == 0
    assert not r.image_mask.any()
    assert len(r.ground) >= det.C  # every column certifies a ground run


@pytest.mark.parametrize("noise", [0.0, 0.15])
def test_trench_at_5m_is_a_ditch_candidate(geom, det, noise):
    scene = syn.make_scene("trench", x0=5.0, width=0.5, depth=0.6)
    r = _run(geom, det, scene, noise)
    x0, x1 = _central(r, r.ditch, geom)
    assert x0.size >= 0.8 * 100  # >= 80 % of the 100 central column bands
    assert abs(np.median(x0) - 5.0) < 0.15
    assert abs(np.median(x1) - 5.5) < 0.15
    assert r.image_mask.any()


def test_crest_is_shadow_not_ditch(geom, det):
    r = _run(geom, det, syn.make_scene("crest"), 0.1)
    assert len(r.ditch) == 0
    x0, x1 = _central(r, r.crest, geom)
    assert x0.size > 80
    assert abs(np.median(x0) - 6.0) < 0.2 and np.median(x1) > 9.0


def test_rock_occlusion_is_occluded_never_ditch(geom, det):
    r = _run(geom, det, syn.make_scene("rock_occlusion"), 0.1)
    assert len(r.ditch) == 0 and len(r.crest) == 0
    assert len(r.occluded) > 5
    assert np.all(r.occluded.x0 > 3.9) and np.median(r.occluded.x1) > 6.0


def test_tilted_ground_no_false_gaps(geom, det):
    r = _run(geom, det, syn.make_scene("tilted"), 0.1)
    assert len(r.ditch) == 0 and len(r.crest) == 0


def test_ablation_switch_disables_classification(geom, det):
    r = _run(geom, det, syn.make_scene("trench"), 0.0, use_negobs=False)
    assert len(r.ditch) == 0 and len(r.crest) == 0 and len(r.occluded) == 0
    assert len(r.ground) > 0


def test_narrow_trench_detected_near_theory_range(geom, det):
    # 0.5 m wide trench at 5.5 m subtends ~6 px (Matthies-Rankin criterion) -> must be found.
    r = _run(geom, det, syn.make_scene("trench", x0=5.5, width=0.5, depth=0.6))
    x0, _ = _central(r, r.ditch, geom)
    assert x0.size > 50
