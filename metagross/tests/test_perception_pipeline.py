"""Perception: end-to-end Perception.process on synthetic tier-0 and stereo frames."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from metagross.autonomy.perception import synthetic as syn
from metagross.autonomy.perception.pipeline import LETHAL_COST, UNKNOWN_COST, Perception
from metagross.config import defaults
from metagross.contracts.interfaces import SEM_COST, SEM_MU
from metagross.contracts.messages import CellState, SensorFrame

cv2.setNumThreads(2)
CAL = defaults.stereo_calibration()


@pytest.fixture(scope="module")
def perception() -> Perception:
    # Unrelated scenes at the same pose: disable cross-frame ditch persistence (tested separately).
    return Perception(CAL, defaults.VEHICLE, {"use_negobs": True, "ditch_persistence": False})


def _tier0(p: Perception, name: str, noise: float = 0.1, **kw) -> SensorFrame:
    d = syn.render_disparity(p.geom, syn.make_scene(name, **kw), noise_px=noise, seed=7)
    return SensorFrame(0.0, 0, None, None, 0.0, 0.0, 0.0, sensor_mode="tier0_disparity", disparity=d)


def _corridor(p: Perception, state: np.ndarray, s: CellState, half_w: float = 1.0) -> np.ndarray:
    X, Y = p.spec.centres()
    return X[(state == s) & (np.abs(Y) < half_w)]


def _steady(p: Perception, frame: SensorFrame) -> dict:
    """Output of a static scene once positive persistence has seen it twice (a POSITIVE cell is
    confirmed across 2 frames in the odometry frame, like a ditch): feed the frame twice at the
    same pose and keep the second output."""
    p.process(frame, (0.0, 0.0, 0.0))
    return p.process(frame, (0.0, 0.0, 0.0))


@pytest.fixture(scope="module")
def outputs(perception):
    return {n: _steady(perception, _tier0(perception, n))
            for n in ("flat", "box", "trench", "crest", "rock_occlusion", "tilted")}


def test_output_contract(outputs):
    out = outputs["flat"]
    nx, ny = int(round(defaults.BEV_LOCAL_SIZE_M[0] / defaults.BEV_RES_M)), int(round(defaults.BEV_LOCAL_SIZE_M[1] / defaults.BEV_RES_M))
    for k in ("disparity", "cell_state_local", "cost_local", "height_local", "mu_local", "semantic_mask",
              "missing_ground_mask", "r_vis_m", "timings_ms"):
        assert k in out
    assert out["cell_state_local"].dtype == np.uint8 and out["cell_state_local"].shape == (nx, ny)
    assert out["cost_local"].dtype == np.float32 and out["cost_local"].min() >= 0 and out["cost_local"].max() <= 1
    assert out["missing_ground_mask"].shape == (CAL.height, CAL.width) and out["missing_ground_mask"].dtype == bool
    assert out["semantic_mask"] is None
    assert np.allclose(out["mu_local"], SEM_MU[0])
    assert "total" in out["timings_ms"]


def test_flat_all_ground_no_hazard(perception, outputs):
    st = outputs["flat"]["cell_state_local"]
    lethal = np.isin(st, [CellState.POSITIVE, CellState.DEPRESSION, CellState.DITCH_CANDIDATE, CellState.CREST_SHADOW])
    assert not lethal.any()
    g = _corridor(perception, st, CellState.GROUND)
    assert g.min() < 2.0 and g.max() > 11.0
    assert outputs["flat"]["r_vis_m"] >= 10.0


def test_box_is_positive(perception, outputs):
    out = outputs["box"]
    pos = _corridor(perception, out["cell_state_local"], CellState.POSITIVE, 0.3)
    assert pos.size > 5 and pos.min() > 4.7 and pos.max() < 6.0
    assert np.all(out["cost_local"][out["cell_state_local"] == CellState.POSITIVE] == LETHAL_COST)
    assert _corridor(perception, out["cell_state_local"], CellState.OCCLUDED, 0.2).size > 0


def test_trench_ditch_cells_at_right_range(perception, outputs):
    out = outputs["trench"]
    dit = _corridor(perception, out["cell_state_local"], CellState.DITCH_CANDIDATE)
    assert dit.size > 50
    assert 4.9 <= dit.min() and dit.max() <= 5.7
    assert out["missing_ground_mask"].sum() > 500
    assert _corridor(perception, out["cell_state_local"], CellState.GROUND).max() > 10.0  # far side seen


def test_crest_shadow_not_ditch_and_limits_visibility(perception, outputs):
    out = outputs["crest"]
    st = out["cell_state_local"]
    assert _corridor(perception, st, CellState.DITCH_CANDIDATE).size == 0
    cr = _corridor(perception, st, CellState.CREST_SHADOW)
    assert cr.size > 100 and cr.min() > 5.8
    assert out["r_vis_m"] <= 6.5
    assert np.all(out["cost_local"][st == CellState.CREST_SHADOW] == pytest.approx(UNKNOWN_COST))


def test_rock_shadow_is_occluded_not_lethal(perception, outputs):
    out = outputs["rock_occlusion"]
    st = out["cell_state_local"]
    occ = _corridor(perception, st, CellState.OCCLUDED, 0.2)
    assert occ.size > 20 and occ.min() > 4.3
    assert _corridor(perception, st, CellState.DITCH_CANDIDATE).size == 0
    lethal = _corridor(perception, st, CellState.POSITIVE, 1.0)
    assert lethal.max() < 5.0  # only the rock itself (3.9 .. 4.4 m + inflation)
    assert np.all(out["cost_local"][st == CellState.OCCLUDED] < LETHAL_COST)


def test_tilted_ground_no_lethal(outputs):
    st = outputs["tilted"]["cell_state_local"]
    assert not np.isin(st, [CellState.POSITIVE, CellState.DEPRESSION, CellState.DITCH_CANDIDATE]).any()


def test_negobs_ablation_and_unknown_is_free():
    p = Perception(CAL, defaults.VEHICLE, {"use_negobs": False, "unknown_is_free": True})
    out = p.process(_tier0(p, "trench"))
    st = out["cell_state_local"]
    assert not np.isin(st, [CellState.DITCH_CANDIDATE, CellState.CREST_SHADOW, CellState.OCCLUDED]).any()
    assert np.all(out["cost_local"][st == CellState.UNSEEN] == 0.0)


class _WaterStub:
    """Segmenter stub: water in the lower-left image quadrant, stable path elsewhere."""

    def __call__(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        cls = np.full((h, w), 4, np.uint8)
        cls[: h // 3] = 0
        cls[h // 2 :, : w // 3] = 2
        return cls, np.zeros((h, w), np.float32)


def test_semantics_monotone_fusion():
    p = Perception(CAL, defaults.VEHICLE, {}, segmenter=_WaterStub())
    d = syn.render_disparity(p.geom, syn.make_scene("box"), noise_px=0.1, seed=7)
    rgb = np.zeros((CAL.height, CAL.width, 3), np.uint8)
    frame = SensorFrame(0.0, 0, rgb, None, 0.0, 0.0, 0.0, sensor_mode="tier0_disparity", disparity=d)
    out = p.process(frame)
    st, cost = out["cell_state_local"], out["cost_local"]
    assert out["semantic_mask"] is not None
    water = st == CellState.WATER
    X, Y = p.spec.centres()
    assert water.sum() > 20 and np.all(Y[water] > 0)  # left of the vehicle
    assert np.all(cost[water] >= SEM_COST[2])
    assert np.all(cost[st == CellState.POSITIVE] == LETHAL_COST)  # geometry never cleared
    assert np.all(out["mu_local"][water] < SEM_MU[4])


def test_stereo_sgbm_trench_end_to_end():
    p = Perception(CAL, defaults.VEHICLE, {})
    left, right, _ = syn.render_stereo_pair(p.geom, syn.make_scene("trench"), seed=5)
    out = p.process(SensorFrame(0.0, 0, left, right, 0.0, 0.0, 0.0))
    dit = _corridor(p, out["cell_state_local"], CellState.DITCH_CANDIDATE)
    # Noise-aware gap test (2026-09-30): SGBM smears the 5 m trench step to ~3 sigma, so fewer
    # column bands pass (12 corridor cells measured vs > 30 with the old, noise-blind rule that
    # also flagged 22-30 % of off-hazard DEV cells). Location must stay right, nothing elsewhere.
    assert dit.size >= 8 and 4.8 <= np.median(dit) <= 5.6
    X, _ = p.spec.centres()
    assert not np.any((out["cell_state_local"] == CellState.DITCH_CANDIDATE) & (X > 6.5))
    assert out["timings_ms"]["stereo"] > 0
