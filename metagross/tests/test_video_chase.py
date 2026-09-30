"""Chase replay plumbing: scenario resolution, shared renderer, frame hold and disk cache (no browser)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from metagross.sim.render.chase_replay import ScenarioNotFound, ThreeChaseFactory, resolve_scenario
from metagross.sim.scenario import SCHEMA, scenario_sha256, write_scenario
from tests.test_video_replay import _make_run
from video.chase import CachedChase, make_chase, state_key
from video.replay import RunReplay


class FakeRenderer:
    """RendererProto stand-in: records calls, returns a frame whose value encodes the call count."""

    def __init__(self) -> None:
        self.loads: list[str] = []
        self.calls = 0
        self.closed = False

    def load_scenario(self, scn: dict) -> None:
        self.loads.append(scn["sha256"])

    def render_chase(self, state: dict, w: int, h: int) -> np.ndarray:
        self.calls += 1
        img = np.zeros((h, w, 3), np.uint8)
        img[..., 0] = self.calls % 256
        img[..., 1] = int(state["t"] * 10) % 256
        return img

    def close(self) -> None:
        self.closed = True


def _scenario(seed: int, tag: str) -> dict:
    scn = {"schema": SCHEMA, "seed": seed, "family": "F1_trail", "split": "dev", "sha256": "", "tag": tag}
    scn["sha256"] = scenario_sha256(scn)
    return scn


def _run_with_scenario(root: Path, scn: dict, split_dir: Path) -> Path:
    run = _make_run(root)
    res = json.loads((run / "result.json").read_text(encoding="utf-8"))
    res.update({"seed": scn["seed"], "split": "dev", "sha256": scn["sha256"]})
    (run / "result.json").write_text(json.dumps(res), encoding="utf-8")
    write_scenario(scn, split_dir / "dev" / f"{scn['seed']}.json")
    return run


def test_resolve_scenario_checks_the_digest(tmp_path: Path) -> None:
    scn = _scenario(101, "a")
    run = _run_with_scenario(tmp_path / "r", scn, tmp_path / "scn")
    assert resolve_scenario(run, tmp_path / "scn")["sha256"] == scn["sha256"]
    other = _scenario(101, "b")  # same seed, different world: must be refused
    write_scenario(other, tmp_path / "scn" / "dev" / "101.json")
    with pytest.raises(ScenarioNotFound):
        resolve_scenario(run, tmp_path / "scn")
    with pytest.raises(ScenarioNotFound):
        resolve_scenario(tmp_path / "nope", tmp_path / "scn")


def test_factory_shares_one_renderer_and_reloads_only_on_scene_change(tmp_path: Path) -> None:
    fake = FakeRenderer()
    s1, s2 = _scenario(101, "a"), _scenario(102, "b")
    r1 = _run_with_scenario(tmp_path / "r1", s1, tmp_path / "scn")
    r2 = _run_with_scenario(tmp_path / "r2", s2, tmp_path / "scn")
    fac = ThreeChaseFactory(tmp_path / "scn", renderer_factory=lambda: fake)
    c1, c1b, c2 = fac.for_run(r1), fac.for_run(r1), fac.for_run(r2)
    state = {"pose": [0, 0, 0, 0, 0, 0], "t": 0.0}
    for c in (c1, c1b, c1, c2, c1):
        assert c(state, 32, 18).shape == (18, 32, 3)
    assert fake.loads == [s1["sha256"], s2["sha256"], s1["sha256"]]
    assert fac.n_frames == 5
    fac.close()
    assert fake.closed


def test_cached_chase_holds_frames_and_uses_the_disk_cache(tmp_path: Path) -> None:
    fake = FakeRenderer()
    ch = CachedChase(lambda s, w, h: fake.render_chase(s, w, h), scene_id="abc", cache_dir=tmp_path / "cache", chase_fps=15.0)
    ts = np.arange(0, 1.0, 1 / 30)  # 30 video frames in one run second
    for t in ts:
        ch({"pose": [t, 0, 0, 0, 0, 0], "t": float(t)}, 16, 8)
    assert fake.calls == 15  # held every other frame at 15 fps
    again = CachedChase(lambda s, w, h: fake.render_chase(s, w, h), scene_id="abc", cache_dir=tmp_path / "cache", chase_fps=15.0)
    for t in ts:
        again({"pose": [t, 0, 0, 0, 0, 0], "t": float(t)}, 16, 8)
    assert fake.calls == 15 and again.hits == 15  # second pass: all from disk


def test_cached_chase_falls_back_after_a_renderer_failure() -> None:
    calls = []

    def broken(state: dict, w: int, h: int) -> np.ndarray:
        calls.append(state["t"])
        raise RuntimeError("WebGL context lost")

    ch = CachedChase(broken, scene_id="x")
    assert ch({"pose": [0] * 6, "t": 0.0}, 8, 8) is None and ch.failed
    assert ch({"pose": [0] * 6, "t": 1.0}, 8, 8) is None and len(calls) == 1  # not retried every frame


def test_state_key_is_stable_and_sensitive() -> None:
    s = {"pose": [1.0, 2.0, 0.0, 0.0, 0.0, 0.5], "t": 3.0, "dynamic": [{"id": 0, "xy": [4.0, 5.0]}]}
    assert state_key(s, 10, 10, "x") == state_key(json.loads(json.dumps(s)), 10, 10, "x")
    assert state_key(s, 10, 10, "x") != state_key({**s, "t": 3.1}, 10, 10, "x")
    assert state_key(s, 10, 10, "x") != state_key(s, 10, 12, "x")
    assert state_key(s, 10, 10, "x") != state_key(s, 10, 10, "y")


def test_make_chase_accepts_per_run_factories_and_plain_callables(tmp_path: Path) -> None:
    s1 = _scenario(101, "a")
    run = _run_with_scenario(tmp_path / "r", s1, tmp_path / "scn")
    fac = ThreeChaseFactory(tmp_path / "scn", renderer_factory=FakeRenderer)
    ch = make_chase(fac, run, None, 30.0)
    assert ch.scene_id == s1["sha256"]
    plain = make_chase(lambda: (lambda s, w, h: np.zeros((h, w, 3), np.uint8)), run, None, 30.0)
    rp = RunReplay(run, chase_renderer=plain, chase_size=(40, 20))
    assert rp.frame_at(1.0).chase_rgb.shape == (20, 40, 3)


def test_world_state_carries_dynamic_actors(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    with np.load(run / "gt" / "states.npz") as z:
        d = {k: z[k] for k in z.files}
    n = len(d["t"])
    dyn = np.zeros((n, 1, 2))
    dyn[:, 0, 0] = np.linspace(0.0, 10.0, n)
    np.savez_compressed(run / "gt" / "states.npz", **d, dyn_xy=dyn)
    st = RunReplay(run).world_state_at(float(d["t"][-1]) / 2)
    assert len(st["dynamic"]) == 1 and st["dynamic"][0]["id"] == 0
    assert st["dynamic"][0]["xy"][0] == pytest.approx(5.0, abs=0.05)
