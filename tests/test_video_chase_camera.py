"""Chase-camera planning (chase_replay): critically damped yaw, occluder push-in, per-run chase block. No browser."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from metagross.sim.render.chase_replay import (ChaseCamParams, ThreeChaseFactory, build_occluders, critically_damped,
                                                plan_chase_camera, sightline_clear_fraction)
from tests.test_video_chase import FakeRenderer, _run_with_scenario, _scenario
from video.chase import make_chase, state_key

NO_CYL = np.zeros((0, 5))
NO_SPH = np.zeros((0, 4))


def test_critically_damped_step_has_no_overshoot_and_settles() -> None:
    t = np.arange(0.0, 6.0, 0.04)
    u = np.where(t >= 1.0, 1.0, 0.0)
    y = critically_damped(t, u, omega=2.0)
    assert y[0] == 0.0 and y.max() <= 1.0 + 1e-9  # critically damped: never overshoots
    assert np.all(np.diff(y) >= -1e-12)  # monotone for a step input
    i = int(np.searchsorted(t, 1.0 + 1.9))  # 1 - (1 + wt) e^-wt = 0.9 at w t ~ 3.9
    assert y[i] == pytest.approx(0.9, abs=0.03)
    assert y[-1] == pytest.approx(1.0, abs=1e-3)


def test_critically_damped_asymmetric_attack_is_faster() -> None:
    t = np.arange(0.0, 3.0, 0.04)
    down = critically_damped(t, np.where(t >= 0.5, 0.0, 1.0), omega=1.0, omega_down=8.0)
    up = critically_damped(t, np.where(t >= 0.5, 1.0, 0.0), omega=1.0, omega_down=8.0)
    k = int(np.searchsorted(t, 1.0))
    assert down[k] < 0.1 and up[k] < 0.2  # half a second after the step: attack done, release barely started


def test_sightline_hits_a_trunk_at_the_right_fraction() -> None:
    target = np.array([[0.0, 0.0, 0.5]])
    eye = np.array([[-6.0, 0.0, 3.0]])
    trunk = np.array([[-3.0, 0.0, 0.2, 0.0, 4.0]])  # x, y, r, z0, z1
    s = sightline_clear_fraction(target, eye, trunk, NO_SPH, clearance_m=0.3)
    assert s[0] == pytest.approx((3.0 - 0.5) / 6.0, abs=1e-6)
    off = np.array([[-3.0, 2.0, 0.2, 0.0, 4.0]])  # 2 m to the side: clear
    assert sightline_clear_fraction(target, eye, off, NO_SPH, 0.3)[0] == 1.0
    low = np.array([[-3.0, 0.0, 0.2, 0.0, 0.5]])  # too short to reach the sightline (z ~ 1.75 m there)
    assert sightline_clear_fraction(target, eye, low, NO_SPH, 0.3)[0] == 1.0


def test_sightline_sphere_and_target_inside_is_ignored() -> None:
    target = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    eye = np.array([[-6.0, 0.0, 3.0], [-6.0, 0.0, 3.0]])
    canopy = np.array([[-4.0, 0.0, 2.3, 1.0]])  # centred on the sightline at x = -4
    s = sightline_clear_fraction(target, eye, NO_CYL, canopy, 0.0)
    assert 0.4 < s[0] < 0.67
    around_target = np.array([[0.0, 0.0, 0.5, 1.0]])  # the rover sits inside: moving the camera cannot help
    assert sightline_clear_fraction(target, eye, NO_CYL, around_target, 0.0)[0] == 1.0


def test_build_occluders_from_scenario_objects() -> None:
    scn = {"objects": [
        {"type": "tree", "xy": [1.0, 2.0], "height": 6.0, "canopy_r": 2.0, "trunk_r": 0.2},
        {"type": "bush", "xy": [3.0, 0.0], "radius": 0.5, "height": 1.0},
        {"type": "rock", "xyz": [4.0, 1.0, 0.1], "radius": 0.3},
        {"type": "log", "xy": [0.0, 0.0], "radius": 0.2, "length": 2.0, "yaw": 0.0},
    ]}
    cyl, sph = build_occluders(scn, lambda x, y: 1.0)
    assert cyl.shape == (3, 5) and cyl[0, 3] == 1.0 and cyl[0, 4] == pytest.approx(1.0 + 3.6)  # trunk
    assert cyl[1].tolist() == pytest.approx([1.0, 2.0, 2.4, 1.0 + 6.0 - 3.6, 1.0 + 6.0 + 0.4])  # canopy
    assert len(sph) >= 1 + 3  # rock + a chain of log spheres


def test_plan_pushes_in_past_a_tree_and_returns_to_nominal() -> None:
    p = ChaseCamParams()
    t = np.arange(0.0, 20.0, 0.04)
    x = 1.0 * t  # 1 m/s along +x, yaw 0: the camera trails 6 m behind along -x
    zeros = np.zeros_like(t)
    trunk = np.array([[7.0, 0.0, 0.25, 0.0, 5.0]])  # the rover passes it at t = 7 s; it cuts the sightline until ~13 s
    plan = plan_chase_camera(t, x, zeros, zeros, zeros, trunk, NO_SPH, p)
    back = plan["back_m"]
    assert back[0] == pytest.approx(p.back_m) and back[-1] == pytest.approx(p.back_m, abs=0.05)
    blocked = plan["clear_raw"] < 1.0
    assert blocked.any()
    fixable = blocked & (plan["clear_raw"] >= p.min_back_frac)  # closer occluders are accepted by design
    assert fixable.any()
    assert np.all(back[fixable] <= plan["clear_raw"][fixable] * p.back_m + 1e-9)  # never behind the occluder
    assert back.min() >= p.min_back_frac * p.back_m - 1e-9
    first = int(np.argmax(blocked))
    k_early = int(np.searchsorted(t, t[first] - 0.3))
    assert back[k_early] < p.back_m - 0.5  # anticipation: already moving in before the trunk cuts the view
    # recheck: the planned eye really sees the rover
    ys = plan["yaw_smooth"]
    eye = np.column_stack([x - np.cos(ys) * back, -np.sin(ys) * back, plan["up_m"]])
    target = np.column_stack([x, zeros, zeros + p.target_up_m])
    # no occluder beyond the closest allowed camera distance cuts the planned sightline
    clear = sightline_clear_fraction(target, eye, trunk, NO_SPH, 0.0, ignore_below=(p.min_back_frac * p.back_m + p.clearance_m) / back)
    inside = np.hypot(x - trunk[0, 0], zeros - trunk[0, 1]) < trunk[0, 2] + p.clearance_m  # rover touching it: ignored
    assert np.all(clear[~inside] >= 1.0 - 1e-9)


def test_yaw_smoothing_lags_a_turn_on_the_spot_without_overshoot() -> None:
    p = ChaseCamParams()
    t = np.arange(0.0, 6.0, 0.04)
    yaw = np.where(t < 1.0, 0.0, np.minimum((t - 1.0) * 2.0, math.pi / 2))  # 90 deg pivot at 2 rad/s
    zeros = np.zeros_like(t)
    plan = plan_chase_camera(t, zeros, zeros, zeros, yaw, NO_CYL, NO_SPH, p)
    ys = plan["yaw_smooth"]
    assert ys.max() <= math.pi / 2 + 1e-6
    k = int(np.searchsorted(t, 1.785))  # the pivot ends here
    assert ys[k] < yaw[k] - 0.3  # the camera swings more slowly than the rover
    assert ys[-1] == pytest.approx(math.pi / 2, abs=0.05)


def test_chase_replay_adds_the_chase_block_and_a_cache_tag(tmp_path: Path) -> None:
    fake = FakeRenderer()
    seen: list[dict] = []
    orig = fake.render_chase

    def spy(state: dict, w: int, h: int) -> np.ndarray:
        seen.append(state)
        return orig(state, w, h)

    fake.render_chase = spy  # type: ignore[method-assign]
    scn = _scenario(101, "a")
    run = _run_with_scenario(tmp_path / "r", scn, tmp_path / "scn")
    fac = ThreeChaseFactory(tmp_path / "scn", renderer_factory=lambda: fake)
    rep = fac.for_run(run)
    assert rep.cache_tag.startswith("cam")
    rep({"pose": [0, 0, 0, 0, 0, 0], "t": 2.0}, 16, 8)
    block = seen[-1]["chase"]
    assert set(block) == {"back_m", "up_m", "yaw_smooth", "lookahead_m"}
    assert block["back_m"] == pytest.approx(6.0) and block["yaw_smooth"] == pytest.approx(0.5, abs=1e-3)  # run heads 0.5 rad
    ch = make_chase(fac, run, None, 30.0)
    assert ch.cache_tag == rep.cache_tag
    legacy = ThreeChaseFactory(tmp_path / "scn", renderer_factory=lambda: fake, params=None).for_run(run)
    legacy({"pose": [0, 0, 0, 0, 0, 0], "t": 2.0}, 16, 8)
    assert "chase" not in seen[-1] and legacy.cache_tag == ""


def test_state_key_includes_the_chase_block() -> None:
    s = {"pose": [1.0, 2.0, 0.0, 0.0, 0.0, 0.5], "t": 3.0}
    with_block = {**s, "chase": {"back_m": 4.0, "up_m": 2.0, "yaw_smooth": 0.4, "lookahead_m": 2.5}}
    assert state_key(s, 10, 10, "x") != state_key(with_block, 10, 10, "x")
    assert state_key(with_block, 10, 10, "x") == state_key(json.loads(json.dumps(with_block)), 10, 10, "x")


def test_real_scenario_occluders_when_available() -> None:
    p = Path(__file__).resolve().parents[1] / "data" / "scenarios" / "dev" / "103.json"
    if not p.exists():
        pytest.skip("DEV scenario 103 not generated")
    scn = json.loads(p.read_text(encoding="utf-8"))
    cyl, sph = build_occluders(scn)
    n_tree = sum(o["type"] == "tree" for o in scn["objects"])
    assert len(cyl) >= 2 * n_tree
