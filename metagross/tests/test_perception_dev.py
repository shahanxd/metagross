"""Eval: DEV perception harness scoring logic (synthetic arrays, no simulator)."""

from __future__ import annotations

import numpy as np
import pytest

from metagross.autonomy.perception.bev import BevSpec
from metagross.config import defaults
from metagross.contracts.messages import CellState
from metagross.eval import perception_dev as pd


def _gt(shape, lethal=None, ditch=None):
    lethal = np.zeros(shape, bool) if lethal is None else lethal
    did = np.full(shape, -1, np.int16)
    if ditch is not None:
        did[ditch] = 0
    return {"lethal": lethal | (did >= 0), "ditch_id": did, "off_hazard": ~(lethal | (did >= 0))}


def test_refuses_eval_seeds():
    with pytest.raises(ValueError):
        pd.load_dev_scenario(5)


def test_score_frame_rates_and_certification():
    spec = BevSpec()
    X, Y = spec.centres()
    st = np.full(spec.shape, CellState.GROUND, np.uint8)
    st[(X > 8) & (X < 8.2)] = CellState.DITCH_CANDIDATE  # false candidates (off hazard)
    ditch = (X > 2.0) & (X < 2.5) & (np.abs(Y) < 1.0)
    st[ditch & (Y > 0)] = CellState.DITCH_CANDIDATE
    out = {"cell_state_local": st, "certified_local": st == CellState.GROUND, "timings_ms": {"total": 5.0}}
    f = pd.score_frame(out, _gt(spec.shape, ditch=ditch), X, Y, 0, 5.0, pd.stopping_envelope_m())
    assert f.n_cand_off == int(((X > 8) & (X < 8.2)).sum())
    assert f.n_obs_off == int((~ditch).sum())
    assert f.ditch_path > 0 and 0.4 < f.ditch_flagged / f.ditch_path < 0.6
    assert f.haz_env > 0 and f.haz_env_cert == f.haz_env_cert_ditch > 0  # right half of the path is certified
    assert f.ditch_lip_range_m == pytest.approx(2.05 - defaults.CAM_FORWARD_M, abs=0.06)


def test_detection_ranges_first_and_stable():
    def fs(r, frac):
        return pd.FrameScore(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, int(round(frac * 10)), r, 1.0)
    scores = [fs(9.0, 0.0), fs(8.0, 0.6), fs(7.0, 0.2), fs(6.0, 0.7), fs(5.0, 0.9), fs(1.5, 0.0)]
    d = pd.detection_ranges(scores)
    assert d["r_first_m"] == 8.0 and d["r_stable_m"] == 6.0  # 1.5 m is inside the blind zone: not scored
    assert pd.detection_ranges([])["n_frames"] == 0


def test_theory_and_envelope_values():
    assert pd.theory_r_det_m(0.3) == pytest.approx(4.304, abs=0.01)  # results/theory.json
    assert pd.stopping_envelope_m(2.0) == pytest.approx(0.4 + 3.033, abs=0.01)
