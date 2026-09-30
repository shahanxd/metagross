"""World: mission spec (A-frame + heading-init error), SensorFrame production, stereo renderer hook,
GT logging rate, lighting state."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metagross.config import defaults
from metagross.contracts.messages import SENSOR_FRAME_FIELDS
from metagross.sim.miniworld import mini_scenario
from metagross.sim.world import World

FLAT = np.zeros((200, 600), np.float32)


def test_mission_goal_in_a_frame_with_heading_error():
    yaw, err = 0.4, 1.0
    scn = mini_scenario(FLAT, start=(2.0, 3.0, yaw), goal=(22.0, 7.0), heading_init_err_deg=err)
    m = World(scn).mission_spec()
    g = np.array([20.0, 4.0])
    c, s = math.cos(-yaw + math.radians(err)), math.sin(-yaw + math.radians(err))
    expected = np.array([[c, -s], [s, c]]) @ g
    np.testing.assert_allclose(m.goal_xy_a, expected, atol=1e-9)
    assert math.hypot(*m.goal_xy_a) == pytest.approx(math.hypot(*g))
    assert m.success_radius_m == scn["mission"]["success_radius_m"]


def test_tier0_sensor_frame_contents():
    w = World(mini_scenario(FLAT, start=(2.0, 5.0, 0.0), goal=(20.0, 5.0)))
    f = w.make_sensor_frame(w.t, 0)
    assert set(f.__slots__) == SENSOR_FRAME_FIELDS
    assert f.sensor_mode == "tier0_disparity" and f.left_rgb is None and f.right_gray is None
    assert f.disparity.shape == (defaults.IMG_H, defaults.IMG_W) and f.disparity.dtype == np.float32
    assert f.wheel_angle_l_rad == 0.0 and abs(f.gyro_z_rps) < 0.02
    with pytest.raises(ValueError):
        w.make_sensor_frame(w.t + 0.1, 1)  # frames are only taken at the current state


class _FakeRenderer:
    def __init__(self):
        self.loaded = None
        self.states = []

    def load_scenario(self, scenario):
        self.loaded = scenario["seed"]

    def render_stereo(self, state):
        self.states.append(state)
        return np.zeros((defaults.IMG_H, defaults.IMG_W, 3), np.uint8), np.zeros((defaults.IMG_H, defaults.IMG_W), np.uint8)

    def render_gt(self, state):
        return {"depth": np.zeros((4, 4), np.float32), "semantic": np.zeros((4, 4), np.uint8)}

    def render_chase(self, state, width, height):
        return np.zeros((height, width, 3), np.uint8)

    def close(self):
        pass


def test_stereo_mode_uses_renderer_with_body_and_camera_pose():
    r = _FakeRenderer()
    w = World(mini_scenario(FLAT, start=(2.0, 5.0, 0.3), goal=(20.0, 5.0), seed=4242), sensor_mode="stereo", renderer=r)
    f = w.make_sensor_frame(w.t, 0)
    assert r.loaded == 4242 and f.sensor_mode == "stereo" and f.disparity is None
    st = r.states[0]
    assert st["pose"][:2] == [2.0, 5.0] and st["pose"][5] == pytest.approx(0.3)
    T_wc = np.asarray(st["T_world_cam"])
    np.testing.assert_allclose(T_wc, np.asarray(st["T_world_body"]) @ defaults.camera_extrinsics(), atol=1e-12)
    assert {"lighting", "dynamic", "t"} <= set(st)
    with pytest.raises(ValueError):
        World(mini_scenario(FLAT), sensor_mode="stereo")


def test_gt_log_rate_and_lighting_state():
    light = {"sun_elev_deg": 5.0, "sun_azim_deg": 0.0, "fog_density": 0.0, "exposure": 1.0,
             "events": [{"type": "dim", "t0": 0.5, "t1": 1.0, "gain": 0.4}]}
    w = World(mini_scenario(FLAT, start=(2.0, 5.0, 0.0), goal=(20.0, 5.0), lighting=light))
    for _ in range(100):
        w.step()
    log = w.gt_log()
    assert len(log["t"]) == 61  # t = 0 plus 30 Hz for 2 s
    assert w.lighting_state(0.7)["exposure"] == pytest.approx(0.4)
    assert w.lighting_state(1.5)["active_events"] == []
    types = [e.type for e in w.referee.events]
    assert types == ["lighting_on", "lighting_off"]
    np.testing.assert_allclose(w.lighting_state()["sun_dir_world"], [math.cos(math.radians(5)), 0.0, math.sin(math.radians(5))])
