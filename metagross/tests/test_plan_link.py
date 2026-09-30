"""Telemetry codec round trip / size, ego costmap rendering and the link emulator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metagross.autonomy.link import codec
from metagross.autonomy.link.egomap import ego_costmap_u4
from metagross.autonomy.link.emulator import LinkEmulator
from metagross.autonomy.planning.costmap import CostmapParams, build_costmap
from metagross.autonomy.planning.rolling_map import RollingMap
from metagross.config.defaults import LINK_KBPS, TELEMETRY_HZ
from metagross.contracts.messages import CellState, DriveMode, Telemetry


def _realistic_u4(seed=0) -> np.ndarray:
    """Ego costmap resembling a real one: unseen behind/sides, a ground wedge ahead with
    low-cost texture, two obstacles and a ditch band."""
    rng = np.random.default_rng(seed)
    n = codec.COSTMAP_N
    r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    fwd = (n / 2 - r) * codec.COSTMAP_RES_M
    lat = (n / 2 - c) * codec.COSTMAP_RES_M
    u4 = np.zeros((n, n), np.uint8)
    wedge = (fwd > 1.5) & (np.abs(np.arctan2(lat, fwd)) < math.radians(36))
    u4[wedge] = codec.U4_GROUND_MIN + rng.integers(0, 3, wedge.sum())
    u4[(np.hypot(fwd - 5, lat - 1) < 0.6)] = codec.U4_LETHAL
    u4[(np.hypot(fwd - 3, lat + 2) < 0.4)] = codec.U4_DYNAMIC
    u4[wedge & (np.abs(fwd - 6.5) < 0.3)] = codec.U4_DITCH
    return u4


def _tel(u4=None) -> Telemetry:
    return Telemetry(
        t=12.345, seq=77, pose_xy_yaw=(12.34, -5.67, 2.5), pos_sigma_m=0.123, mode=DriveMode.CAUTION,
        reason="GOV_RCERT R=3.1m", v_cap_mps=1.23, r_cert_m=3.1, speed_mps=0.98,
        waypoints_xy=[(13.0, -5.0), (15.0, -4.0), (17.0, -3.5), (19.0, -3.0), (21.0, -2.0)],
        costmap_u4=u4, health={"q": 0.62, "p_fail": 0.38, "chi_hat": 1.43, "my_custom": 7.5},
    )


def test_codec_round_trip():
    u4 = _realistic_u4()
    tel = _tel(u4)
    pkt = codec.encode(tel)
    out = codec.decode(pkt)
    assert out.seq == tel.seq and out.mode == tel.mode and out.reason == tel.reason
    assert out.t == pytest.approx(tel.t, abs=1e-3)
    assert out.pose_xy_yaw[0] == pytest.approx(12.34, abs=0.005) and out.pose_xy_yaw[1] == pytest.approx(-5.67, abs=0.005)
    assert out.pose_xy_yaw[2] == pytest.approx(2.5, abs=0.005)
    assert out.pos_sigma_m == pytest.approx(0.123, abs=5e-4)
    assert out.v_cap_mps == pytest.approx(1.23, abs=0.005) and out.r_cert_m == pytest.approx(3.1, abs=0.005)
    assert len(out.waypoints_xy) == 5 and np.allclose(out.waypoints_xy, tel.waypoints_xy, atol=0.005)
    assert np.array_equal(out.costmap_u4, u4)
    for k, v in tel.health.items():
        assert out.health[k] == pytest.approx(v, rel=2e-3)


def test_codec_reason_free_text_and_no_costmap():
    tel = _tel(None)
    tel.reason = "something new happened"
    out = codec.decode(codec.encode(tel))
    assert out.reason == "something new happened" and out.costmap_u4 is None


def test_codec_rejects_corruption():
    pkt = bytearray(codec.encode(_tel(_realistic_u4())))
    pkt[20] ^= 0xFF
    with pytest.raises(codec.CodecError):
        codec.decode(bytes(pkt))
    with pytest.raises(codec.CodecError):
        codec.decode(b"\x00\x01")


def test_packet_size_fits_link_budget():
    """At TELEMETRY_HZ the packets must fit the LINK_KBPS radio with >= 25 % headroom, even for a
    textured costmap (random ground-cost bins are the worst case for zlib)."""
    sizes = [len(codec.encode(_tel(_realistic_u4(s)))) for s in range(10)]
    budget_bytes = LINK_KBPS * 1000.0 / 8.0 / TELEMETRY_HZ
    assert max(sizes) < 0.75 * budget_bytes, (sizes, budget_bytes)
    assert len(codec.encode(_tel(None))) < 120


def test_ego_costmap_orientation():
    rm = RollingMap()
    rm.seed_apron((0, 0, 0), 0.0, 20.0)
    g = rm.geometry
    ix, iy = g.to_index(np.array([3.0]), np.array([10.0]))  # A-frame point 3 m fwd-x, 10 m left
    rm.state[iy[0], ix[0]] = int(CellState.POSITIVE)
    maps = build_costmap(rm, 0.0, True, CostmapParams())
    # vehicle at (0, 8) facing +y: the point is 2 m ahead, 3 m to the right
    u4 = ego_costmap_u4(rm, maps, (0.0, 8.0, math.pi / 2))
    rr, cc = np.nonzero(u4 == codec.U4_LETHAL)
    assert len(rr)
    fwd = codec.COSTMAP_N / 2 * codec.COSTMAP_RES_M - (rr.mean() + 0.5) * codec.COSTMAP_RES_M
    lat = codec.COSTMAP_N / 2 * codec.COSTMAP_RES_M - (cc.mean() + 0.5) * codec.COSTMAP_RES_M
    assert fwd == pytest.approx(2.0, abs=0.3) and lat == pytest.approx(-3.0, abs=0.3)


def test_link_emulator_deterministic_bandwidth_loss_latency():
    def run(seed):
        link = LinkEmulator(bandwidth_bps=9600, loss=0.2, latency_s=0.4, seed=seed)
        got = []
        for k in range(200):
            link.send(0.5 * k, bytes(300))
            got += link.poll(0.5 * k)
        got += link.poll(1e9)
        return link, got
    a, ga = run(1)
    b, gb = run(1)
    assert [g[0] for g in ga] == [g[0] for g in gb]
    assert 0.1 < a.stats.dropped_loss / a.stats.sent < 0.3
    tx = 300 * 8 / 9600
    assert min(a.stats.latencies_s) == pytest.approx(0.4 + tx)
    slow = LinkEmulator(bandwidth_bps=1000, loss=0.0, latency_s=0.0, max_queue_s=2.0)
    ok = [slow.send(0.0, bytes(125)) for _ in range(5)]  # each needs 1 s on air
    assert ok == [True, True, True, False, False] and slow.stats.dropped_queue == 2
