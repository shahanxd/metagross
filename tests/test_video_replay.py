"""Replay: run-dir loading, WORLD->A transform, GT interpolation, held debug/telemetry."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from metagross.autonomy.link import codec
from metagross.contracts.messages import DriveMode, Telemetry
from video.replay import WHEEL_WINDOW_S, RunReplay, decode_costmap_hex, world_to_a

START = (5.0, 3.0, 0.5)  # WORLD pose of launch point A
V = 1.0  # m/s straight-line speed in the synthetic run


def _make_run(root: Path, yaw_rate: float = 0.0, duration: float = 12.0) -> Path:
    run = root / "run"
    (run / "gt").mkdir(parents=True)
    (run / "autonomy" / "debug").mkdir(parents=True)
    t = np.arange(0.0, duration + 1e-9, 1.0 / 30.0)
    a = START[2] + yaw_rate * t
    x = START[0] + V * t * math.cos(START[2]) if yaw_rate == 0 else START[0] + np.cumsum(np.full_like(t, V / 30.0) * np.cos(a))
    y = START[1] + V * t * math.sin(START[2]) if yaw_rate == 0 else START[1] + np.cumsum(np.full_like(t, V / 30.0) * np.sin(a))
    z = np.zeros_like(t)
    np.savez_compressed(run / "gt" / "states.npz", t=t, x=x, y=y, z=z, roll=z, pitch=z, yaw=np.arctan2(np.sin(a), np.cos(a)),
                        v=np.full_like(t, V), omega=np.full_like(t, yaw_rate), wl=np.full_like(t, 7.0), wr=np.full_like(t, 8.0))
    tc = np.arange(0.0, duration + 1e-9, 0.1)
    np.savez_compressed(run / "gt" / "cmds.npz", t=tc, seq=np.arange(len(tc)), omega_l=np.full_like(tc, 7.5),
                        omega_r=np.full_like(tc, 7.7), compute_ms=np.full_like(tc, 100.0), apply_t=tc, mode=np.array(["NOMINAL"] * len(tc)))
    for i, tt in enumerate(tc[:30]):  # debug ticks for the first 3 s only
        mode = "NOMINAL" if tt < 1.0 else "CAUTION"
        np.savez_compressed(run / "autonomy" / "debug" / f"tick_{i:06d}.npz", t=np.float64(tt),
                            pose_xy_yaw=np.array([V * tt + 0.01, 0.02, 0.0]), v_cap_mps=np.float64(1.5), r_cert_m=np.float64(6.0),
                            plan_xy=np.array([[V * tt, 0.0], [V * tt + 2.0, 0.0]], np.float32),
                            health_json=np.array(json.dumps({"q": 0.9, "pos_sigma_m": 0.1})),
                            extras_json=np.array(json.dumps({"mode": mode, "reason": "OK" if mode == "NOMINAL" else "GOV_DITCH_DET d=5m"})),
                            timings_json=np.array("{}"))
    cm = (np.arange(64 * 64) % 16).reshape(64, 64).astype(np.uint8)
    lines = []
    for k, tt in enumerate(np.arange(0.0, duration + 1e-9, 0.5)):
        tel = Telemetry(t=float(tt), seq=k, pose_xy_yaw=(V * tt, 0.0, 0.0), pos_sigma_m=0.1, mode=DriveMode.NOMINAL, reason="OK",
                        v_cap_mps=1.5, r_cert_m=6.0, speed_mps=V, costmap_u4=cm, health={"q": 0.9})
        pkt = codec.encode(tel)
        lines.append(json.dumps(codec.to_jsonable(codec.decode(pkt), len(pkt))))
    (run / "autonomy" / "telemetry.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run / "result.json").write_text(json.dumps({"seed": 101, "config": {"name": "FULL"}}), encoding="utf-8")
    (run / "mission.json").write_text(json.dumps({"mission_id": "T-1", "goal_xy_a": [20.0, 1.0], "goal_sigma_m": 1.5}), encoding="utf-8")
    return run


def test_onboard_image_is_held_across_ticks_without_one(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    for f in sorted((run / "autonomy" / "debug").glob("tick_*.npz")):
        i = int(f.stem.split("_")[1])
        if i % 2 == 0:  # the runner logs the image every 2nd tick
            with np.load(f) as z:
                d = {k: z[k] for k in z.files}
            np.savez_compressed(f, **d, extra_left_rgb=np.full((20, 32, 3), i, np.uint8))
    rp = RunReplay(run)
    pd = rp.frame_at(0.15)  # tick 1 (t = 0.1) has no image: tick 0's is held
    assert pd.tick.index == 1 and pd.left_tick.index == 0 and int(pd.left_rgb[0, 0, 0]) == 0
    assert int(rp.frame_at(0.2).left_rgb[0, 0, 0]) == 2


@pytest.fixture()
def replay(tmp_path: Path) -> RunReplay:
    return RunReplay(_make_run(tmp_path))


def test_world_to_a_origin_and_axes() -> None:
    xa, ya, yawa = world_to_a(np.array([5.0, 5.0 + math.cos(0.5)]), np.array([3.0, 3.0 + math.sin(0.5)]), np.array([0.5, 0.5]), START)
    np.testing.assert_allclose(xa, [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(ya, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(yawa, [0.0, 0.0], atol=1e-12)


def test_gt_interpolation_between_samples(replay: RunReplay) -> None:
    x, y, yaw, v = replay.gt_pose_at(1.0 / 60.0)  # half-way between the first two 30 Hz samples
    assert x == pytest.approx(V / 60.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert yaw == pytest.approx(0.0, abs=1e-9)
    assert v == pytest.approx(V)
    assert replay.frame_at(0.0).gt_pose_a == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_yaw_interpolation_across_pi(tmp_path: Path) -> None:
    rp = RunReplay(_make_run(tmp_path, yaw_rate=1.0, duration=6.0))
    # A-frame yaw = t; around t = pi the wrapped yaw jumps from +pi to -pi: interpolation must not pass through 0
    _, _, yaw, _ = rp.gt_pose_at(math.pi)
    assert abs(abs(yaw) - math.pi) < 0.05


def test_debug_is_held_between_ticks(replay: RunReplay) -> None:
    assert replay.frame_at(0.15).tick.t == pytest.approx(0.1)
    assert replay.frame_at(0.1).tick.t == pytest.approx(0.1)
    f = replay.frame_at(5.0)  # after the last debug tick: last one is held
    assert f.tick.t == pytest.approx(2.9)
    assert "plan_xy" in f.tick.arrays
    assert f.est_pose_a[0] == pytest.approx(V * 2.9 + 0.01)


def test_mode_log_and_scalars(replay: RunReplay) -> None:
    f = replay.frame_at(1.55)
    assert f.mode == "CAUTION" and f.reason.startswith("GOV_DITCH_DET")
    assert [m[1] for m in f.mode_log] == ["NOMINAL", "CAUTION"]
    assert f.q == pytest.approx(0.9) and f.v_cap_mps == pytest.approx(1.5) and f.r_cert_m == pytest.approx(6.0)
    assert f.mission.goal_xy_a == (20.0, 1.0) and f.mission.mission_id == "T-1"


def test_telemetry_hold_and_costmap_roundtrip(replay: RunReplay) -> None:
    f = replay.frame_at(1.2)
    assert f.telemetry["t"] == pytest.approx(1.0)
    np.testing.assert_array_equal(f.costmap_u4, (np.arange(64 * 64) % 16).reshape(64, 64))
    assert decode_costmap_hex(None) is None


def test_frame_times_and_wheel_window(replay: RunReplay) -> None:
    ts = replay.frame_times(1.0, 3.0)
    assert len(ts) == 60 and ts[1] - ts[0] == pytest.approx(1 / 30)
    f = replay.frame_at(11.0)
    assert f.wheel_t.max() <= 1e-9 and f.wheel_t.min() >= -WHEEL_WINDOW_S - 0.1 - 1e-9
    assert np.all(f.wheel_l == 7.5) and np.all(f.wheel_r == 7.7)
    assert len(list(replay.frames(0.0, 0.5))) == 15


def test_missing_gt_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RunReplay(tmp_path)
