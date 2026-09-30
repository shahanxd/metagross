"""Renderer <-> world convention check (real Chrome + hardware WebGL2; skipped when unavailable).

1. A known world point, projected with the world's ``T_world_cam`` (from ``World.render_state()``)
   and the calibration K, lands on the same pixel as the renderer's own GL projection.
2. SGBM disparity on the rendered stereo pair agrees with the Tier-0 sensor's disparity at the same
   pose on ground pixels: median |difference| < 1 px (pose, extrinsics, baseline and sign
   conventions all consistent).

Run from PowerShell: ``& .venv\\Scripts\\python.exe -m pytest tests\\test_render_world_consistency.py``
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from metagross.autonomy.perception.stereo import SGBMParams, StereoMatcher, disparity_for_frame
from metagross.config import defaults
from metagross.contracts.messages import SensorFrame
from metagross.sim.scenario import load_scenario
from metagross.sim.world import World

cv2.setNumThreads(2)
REPO = Path(__file__).resolve().parents[1]
DEV_SCENARIO = REPO / "data" / "scenarios" / "dev" / "102.json"
CAL = defaults.stereo_calibration()


@pytest.fixture(scope="module")
def renderer():
    try:
        from metagross.sim.render.bridge import ThreeRenderer

        r = ThreeRenderer()
    except Exception as e:  # noqa: BLE001 - any launch failure means "not available here"
        pytest.skip(f"renderer unavailable: {e}")
    yield r
    r.close()


@pytest.fixture(scope="module")
def worlds(renderer):
    if not DEV_SCENARIO.exists():
        pytest.skip("DEV scenarios not generated")
    scn = load_scenario(DEV_SCENARIO)
    return World(scn, sensor_mode="stereo", renderer=renderer), World(scn, sensor_mode="tier0")


def test_known_world_point_projects_to_same_pixel(worlds, renderer):
    ws, _ = worlds
    st = ws.render_state()
    T_wc = np.asarray(st["T_world_cam"])
    # ground point 6 m along the camera's forward axis projected onto the terrain height
    p_cam = np.array([0.4, 0.5, 6.0, 1.0])
    p_w = (T_wc @ p_cam)[:3]
    p_c = np.linalg.inv(T_wc) @ np.r_[p_w, 1.0]
    uv_world = (CAL.K @ (p_c[:3] / p_c[2]))[:2]
    uv_gl = renderer.project_points(st["pose"], p_w[None, :], eye="left")[0]
    assert np.hypot(*(uv_world - uv_gl)) < 0.5, (uv_world, uv_gl)


def test_sgbm_on_rendered_pair_matches_tier0_disparity_on_ground(worlds):
    ws, wt = worlds
    left, right = ws.renderer.render_stereo(ws.render_state())
    fr = SensorFrame(0.0, 0, left, right, 0.0, 0.0, 0.0, "stereo")
    d_sgbm, _ = disparity_for_frame(fr, CAL, StereoMatcher(SGBMParams()))
    d_t0 = wt.make_sensor_frame(wt.t, 0).disparity
    gt = wt.render_gt()
    ground = gt["semantic"] >= 2  # terrain classes (not sky / object)
    rows = np.zeros_like(ground)
    rows[int(0.6 * CAL.height):, :] = True  # near ground, well below the horizon
    m = ground & rows & (d_sgbm > 0) & (d_t0 > 0)
    assert m.sum() > 5000
    diff = np.abs(d_sgbm[m] - d_t0[m])
    assert float(np.median(diff)) < 1.0, float(np.median(diff))
