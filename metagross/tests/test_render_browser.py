"""Browser-backed renderer tests (real Chrome/Edge + hardware WebGL2).

Skipped automatically when Playwright, Chrome/Edge or a hardware WebGL2 context is
unavailable. Run from a shell that allows the browser to start, e.g. PowerShell:
    & .venv\\Scripts\\python.exe -m pytest tests\\test_render_browser.py
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from metagross.config import defaults
from metagross.sim.render.camera_model import project_world_points
from metagross.sim.render.testscene import make_test_scenario

cv2.setNumThreads(2)
CAL = defaults.stereo_calibration()
BOX_XY = (5.0, 0.0)  # box centre (m, world); scene is flat at z = 0
BOX_SIZE = (1.0, 1.0, 1.0)
WALL_X = 9.0


def _flat_scene() -> dict:
    sc = make_test_scenario(102, flat=True, extent_m=(24.0, 12.0))
    sc["dynamic"] = [
        {"type": "box", "size": list(BOX_SIZE), "trigger_dist_m": 0.0, "path": [list(BOX_XY), [BOX_XY[0] + 1, BOX_XY[1]]], "speed": 0.0},
        {"type": "box", "size": [0.4, 7.0, 4.0], "trigger_dist_m": 0.0, "path": [[WALL_X, 0.0], [WALL_X + 1, 0.0]], "speed": 0.0},
    ]
    sc["lighting"]["fog_density"] = 0.0
    return sc


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
def flat_loaded(renderer):
    renderer.load_scenario(_flat_scene())
    return renderer


POSE0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _state(pose, t=0.0):
    return {"pose": pose, "t": t, "seq": 0, "dynamic": [{"xy": list(BOX_XY), "yaw": 0.0}, {"xy": [WALL_X, 0.0], "yaw": 0.0}]}


def test_renderer_is_hardware(renderer) -> None:
    rs = renderer.renderer_string.lower()
    assert rs and "swiftshader" not in rs


def test_js_projection_matches_python_pinhole(flat_loaded) -> None:
    pose = [1.0, -0.5, 0.05, 0.03, -0.02, 0.4]
    rng = np.random.default_rng(1)
    pts = np.c_[rng.uniform(3, 15, 50), rng.uniform(-4, 4, 50), rng.uniform(-0.5, 2.0, 50)]
    pts[:, :2] = pts[:, :2] @ np.array([[math.cos(0.4), math.sin(0.4)], [-math.sin(0.4), math.cos(0.4)]]) + [1.0, -0.5]
    for eye in ("left", "right"):
        uv_py, z = project_world_points(pts, pose, CAL, eye)
        uv_js = flat_loaded.project_points(pose, pts, eye)
        ok = z > 0.1
        assert np.max(np.abs(uv_js[ok] - uv_py[ok])) < 0.01, eye


def test_rendered_box_edges_land_on_predicted_pixels(flat_loaded) -> None:
    """Rasterised object edges vs the Python pinhole. GL covers a pixel iff its centre (integer
    u, v in OpenCV convention) lies inside the projected face, so the first/last covered
    column must be exactly ceil(u_left) / floor(u_right) (small tolerance for near-integers)."""
    gt = flat_loaded.render_gt(_state(POSE0))
    box = (gt["semantic"] == 1) & (gt["depth"] < WALL_X - 2.0)  # the near box, not the far wall
    x_front = BOX_XY[0] - BOX_SIZE[0] / 2
    face = np.array([[x_front, -0.5, 0.5], [x_front, 0.5, 0.5]])
    uv, _ = project_world_points(face, POSE0, CAL, "left")
    row = int(round(uv[0, 1]))
    cols = np.flatnonzero(box[row])
    u_left, u_right = uv[1, 0], uv[0, 0]  # +y (left in world) projects to smaller u
    eps = 0.02
    assert cols.min() - 1 - eps < u_left <= cols.min() + eps, (cols.min(), u_left)
    assert cols.max() - eps <= u_right < cols.max() + 1 + eps, (cols.max(), u_right)
    # bottom edge: the box stands on z = 0 at x_front
    bottom, _ = project_world_points(np.array([[x_front, 0.0, 0.0]]), POSE0, CAL, "left")
    last = np.flatnonzero(box[:, int(round(bottom[0, 0]))]).max()
    assert last - eps <= bottom[0, 1] < last + 1 + eps, (last, bottom[0, 1])


def test_stereo_disparity_matches_fx_b_over_z(flat_loaded) -> None:
    st = _state(POSE0, t=0.2)
    left, right = flat_loaded.render_stereo(st)
    assert left.shape == (CAL.height, CAL.width, 3) and left.dtype == np.uint8
    assert right.shape == (CAL.height, CAL.width) and right.dtype == np.uint8
    gt = flat_loaded.render_gt(st)
    b = 5
    sgbm = cv2.StereoSGBM_create(0, 64, b, P1=8 * b * b, P2=32 * b * b, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                                 disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(cv2.cvtColor(left, cv2.COLOR_RGB2GRAY), right).astype(np.float32) / 16
    d_gt = CAL.fx * CAL.baseline_m / gt["depth"]
    region = np.isfinite(gt["depth"]) & (gt["depth"] > 2.0) & (gt["depth"] < 10.0)
    region[:, :64] = False
    ok = region & (disp > 0)
    assert ok.sum() > 0.5 * region.sum()
    err = disp[ok] - d_gt[ok]
    assert abs(np.median(err)) < 0.3, np.median(err)
    # the far 'wall' (fronto-parallel-ish plane at ~WALL_X): disparity ~ fx*B/Z there too
    wall = ok & (gt["semantic"] == 1) & (gt["depth"] > WALL_X - 1.5)
    if wall.sum() > 200:
        assert abs(np.median(disp[wall] - d_gt[wall])) < 0.3


def test_gt_depth_is_optical_axis_distance_to_ground_plane(flat_loaded) -> None:
    """Back-project GT depth through the Python pinhole; ground pixels must land on z = 0."""
    gt = flat_loaded.render_gt(_state(POSE0))
    depth, sem = gt["depth"], gt["semantic"]
    v, u = np.nonzero((sem == 4) & np.isfinite(depth) & (depth < 15))
    sel = np.random.default_rng(0).choice(len(u), 500, replace=False)
    u, v, z = u[sel].astype(float), v[sel].astype(float), depth[v[sel].astype(int), u[sel].astype(int)]
    Xc = np.c_[(u - CAL.cx) / CAL.fx * z, (v - CAL.cy) / CAL.fy * z, z]
    T = defaults.camera_extrinsics()
    Pw = Xc @ T[:3, :3].T + T[:3, 3]
    assert np.max(np.abs(Pw[:, 2])) < 0.01


def test_semantic_ids_present(renderer) -> None:
    sc = make_test_scenario(100)
    renderer.load_scenario(sc)
    # look at the pond from the trail: expect sky, objects, water, grass, gravel
    x, y = 6.0, -3.0
    yaw = math.atan2(-7.5 - y, 12.0 - x) + 0.25
    gt = renderer.render_gt({"pose": [x, y, 0.0, 0.0, 0.0, yaw], "t": 0.0})
    ids = set(np.unique(gt["semantic"]).tolist())
    assert ids <= {0, 1, 2, 3, 4}
    assert {0, 1, 2, 3}.issubset(ids), ids
    sky = gt["semantic"] == 0
    assert np.all(np.isinf(gt["depth"][sky])) and np.all(np.isfinite(gt["depth"][~sky]))


def test_chase_render_shape(renderer) -> None:
    img = renderer.render_chase({"pose": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0], "t": 0.0}, 320, 180)
    assert img.shape == (180, 320, 3) and img.dtype == np.uint8
    assert img.std() > 5.0


def test_render_is_deterministic_for_same_seed(renderer) -> None:
    sc = _flat_scene()
    outs = []
    for _ in range(2):
        renderer.load_scenario(sc)
        outs.append(renderer.render_stereo(_state([0.5, 0.2, 0.0, 0.0, 0.0, 0.1], t=1.0)))
    assert np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])
