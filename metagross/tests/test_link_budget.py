"""Operator-link budget: codec v2 fidelity ladder, conservative pooling, and a replay of REAL DEV telemetry.

The replay test feeds real ``autonomy/telemetry.jsonl`` logs from DEV closed-loop runs (the vendored
stereo log ``results/link_replay/stereo_FULL_102.telemetry.jsonl`` plus tier0 FULL logs when present)
through the codec and asserts every packet fits 9.6 kbit/s at 2 Hz (600 B)."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from metagross.autonomy.link import codec
from metagross.autonomy.link.budget_report import hazard_downgrades
from metagross.autonomy.link.emulator import LinkEmulator
from metagross.config.defaults import LINK_KBPS, TELEMETRY_HZ
from metagross.contracts.messages import DriveMode, Telemetry

REPO = Path(__file__).resolve().parents[1]
VENDORED = REPO / "results" / "link_replay" / "stereo_FULL_102.telemetry.jsonl"
TIER0_GLOB = "results/runs_dev_tier0/FULL/*/autonomy/telemetry.jsonl"
N_TIER0_LOGS = 4  # keep the test fast: a few full-length tier0 logs besides the stereo one


def _real_logs() -> list[Path]:
    logs = [VENDORED] if VENDORED.exists() else []
    logs += sorted(REPO.glob(TIER0_GLOB))[:N_TIER0_LOGS]
    return logs


def _tel(u4=None, health=None, reason="GOV_RCERT R=3.1m") -> Telemetry:
    return Telemetry(t=3.5, seq=7, pose_xy_yaw=(1.0, -2.0, 0.3), pos_sigma_m=0.05, mode=DriveMode.NOMINAL, reason=reason,
                     v_cap_mps=1.2, r_cert_m=4.0, speed_mps=1.1, waypoints_xy=[(2.0, -2.0), (4.0, -1.5)], costmap_u4=u4,
                     health=health if health is not None else {"q": 0.9, "vo_inliers": 300.0})


def test_budget_constants_match_link():
    assert codec.PACKET_BUDGET_B == int(LINK_KBPS * 1000 / 8 / TELEMETRY_HZ) == 600
    assert codec.PACKET_TARGET_B < codec.PACKET_BUDGET_B


@pytest.mark.skipif(not _real_logs(), reason="no real DEV telemetry.jsonl available")
def test_real_dev_telemetry_fits_600_bytes_at_2hz():
    n = 0
    for log in _real_logs():
        for line in log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            tel = codec.telemetry_from_jsonable(d)
            pkt, info = codec.encode_with_info(tel)
            bits_per_s = 8 * len(pkt) * TELEMETRY_HZ
            assert len(pkt) <= codec.PACKET_BUDGET_B and bits_per_s <= LINK_KBPS * 1000, (log.name, d["seq"], len(pkt))
            out = codec.decode(pkt)
            assert out.seq == tel.seq and out.mode == tel.mode and out.reason == tel.reason
            assert np.allclose(out.waypoints_xy, tel.waypoints_xy, atol=0.006) if tel.waypoints_xy else not out.waypoints_xy
            if tel.costmap_u4 is not None and info.costmap_cell_m is not None:
                assert out.costmap_u4.shape == (codec.COSTMAP_N, codec.COSTMAP_N)
                assert hazard_downgrades(tel.costmap_u4, out.costmap_u4) == 0
            for k in codec.CORE_HEALTH_KEYS & set(tel.health):
                assert k in out.health
            n += 1
    assert n > 100


@pytest.mark.skipif(not VENDORED.exists(), reason="vendored stereo DEV log missing")
def test_real_stereo_stream_does_not_queue_on_emulated_radio():
    link = LinkEmulator(seed=0, loss=0.0)
    t_last = 0.0
    for line in VENDORED.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        t_last = float(d["t"])
        link.send(t_last, codec.encode(codec.telemetry_from_jsonable(d)))
        link.poll(t_last)
    link.poll(t_last + 60.0)
    assert link.stats.dropped_queue == 0
    assert max(link.stats.latencies_s) < 1.0 / TELEMETRY_HZ + link.latency_s + 1e-6


def test_pool2_is_conservative():
    a = np.zeros((4, 4), np.uint8)
    a[0:2, 0:2] = [[1, 3], [2, 2]]  # all ground -> ground with the highest bin
    a[0:2, 2:4] = [[1, 0], [1, 1]]  # one unseen sub-cell -> unseen
    a[2:4, 0:2] = [[1, codec.U4_DITCH_CANDIDATE], [0, 1]]  # hazard wins over unseen / ground
    a[2:4, 2:4] = [[codec.U4_WATER, codec.U4_LETHAL], [1, 1]]  # worst hazard
    p = codec.pool2_conservative(a)
    assert p.tolist() == [[3, codec.U4_UNSEEN], [codec.U4_DITCH_CANDIDATE, codec.U4_LETHAL]]
    up = codec.upsample2(p)
    assert up.shape == (4, 4) and hazard_downgrades(a, up) == 0


def test_quantize_ground_rounds_up_and_keeps_hazards():
    a = np.arange(16, dtype=np.uint8).reshape(4, 4)
    q4 = codec.quantize_ground(a, 4)
    assert q4.ravel()[1:9].tolist() == [2, 2, 4, 4, 6, 6, 8, 8]
    q2 = codec.quantize_ground(a, 2)
    assert q2.ravel()[1:9].tolist() == [4, 4, 4, 4, 8, 8, 8, 8]
    for q in (q4, q2):
        assert q.ravel()[0] == codec.U4_UNSEEN and np.array_equal(q.ravel()[9:], a.ravel()[9:])
    with pytest.raises(ValueError):
        codec.quantize_ground(a, 3)


def test_ladder_prefers_full_resolution_when_it_fits():
    u4 = np.zeros((64, 64), np.uint8)
    u4[10:30, 20:44] = 2
    u4[15:18, 30:33] = codec.U4_LETHAL
    pkt, info = codec.encode_with_info(_tel(u4))
    assert info.rung == 0 and info.costmap_cell_m == codec.COSTMAP_RES_M and info.ground_levels == 8
    assert np.array_equal(codec.decode(pkt).costmap_u4, u4)


def test_ladder_degrades_textured_map_to_fit_budget():
    rng = np.random.default_rng(3)
    u4 = rng.integers(1, 9, (64, 64)).astype(np.uint8)  # incompressible ground texture
    u4[5:9, 5:9] = codec.U4_DITCH
    pkt, info = codec.encode_with_info(_tel(u4))
    assert len(pkt) <= codec.PACKET_TARGET_B and info.rung > 0
    out = codec.decode(pkt)
    if out.costmap_u4 is not None:
        assert hazard_downgrades(u4, out.costmap_u4) == 0


def test_ladder_last_rung_drops_costmap_and_extended_health():
    rng = np.random.default_rng(4)
    u4 = rng.integers(0, 16, (64, 64)).astype(np.uint8)
    health = {"q": 0.5, **{f"extra_key_{i:02d}_with_a_long_name": float(i) for i in range(20)}}
    pkt, info = codec.encode_with_info(_tel(u4, health), budget_bytes=200)
    assert info.rung == len(codec.LADDER) - 1 and info.costmap_cell_m is None and not info.all_health
    out = codec.decode(pkt)
    assert out.costmap_u4 is None and out.health == pytest.approx({"q": 0.5}, rel=1e-3)


def test_extended_health_keys_are_not_free_text():
    keys = ("vo_inliers", "vo_inlier_ratio", "img_lapvar", "disp_density_ground", "q_gate", "vo_available")
    small = len(codec.encode(_tel(None, {k: 1.0 for k in keys})))
    free = len(codec.encode(_tel(None, {f"{k}_x": 1.0 for k in keys})))
    assert free - small >= len(keys) * 4


def test_decode_rejects_v1_packets():
    body = bytes([codec.MAGIC, 1]) + bytes(20)
    with pytest.raises(codec.CodecError):
        codec.decode(body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF))


def test_jsonable_round_trip_matches_telemetry():
    u4 = np.zeros((64, 64), np.uint8)
    u4[20:40, 20:40] = 3
    tel = _tel(u4)
    pkt, info = codec.encode_with_info(tel)
    d = json.loads(json.dumps(codec.to_jsonable(codec.decode(pkt), len(pkt), info)))
    assert d["link_rung"] == info.rung and d["packet_bytes"] == len(pkt)
    back = codec.telemetry_from_jsonable(d)
    assert np.array_equal(back.costmap_u4, u4) and back.seq == tel.seq and back.mode == tel.mode
