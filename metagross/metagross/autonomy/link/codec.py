"""Telemetry <-> bytes for the low-bandwidth operator downlink (codec version 2).

Instead of video the UGV sends a ``TELEMETRY_HZ`` (2 Hz) packet with pose, mode, the
speed-governor state, the next waypoints, a 16 m x 16 m ego costmap and a health vector.
The radio is ``LINK_KBPS`` (9.6 kbit/s), so one packet may use at most
:data:`PACKET_BUDGET_B` = 9600 / 8 / 2 = **600 bytes**. :func:`encode` targets
:data:`PACKET_TARGET_B` (80 % of that, leaving air time for radio framing and a busy channel)
by walking a fixed *fidelity ladder* (:data:`LADDER`) and sending the first rung that fits:

====  =============  ===========  ==============  ===========================================
rung  costmap cells  cell size    ground levels   health
====  =============  ===========  ==============  ===========================================
0     64 x 64        0.25 m       8               all keys
1     64 x 64        0.25 m       4               all keys
2     32 x 32        0.50 m       8               all keys
3     32 x 32        0.50 m       4               all keys
4     32 x 32        0.50 m       2               all keys
5     32 x 32        0.50 m       2               core keys only (:data:`CORE_HEALTH_KEYS`)
6     none           -            -               core keys only
====  =============  ===========  ==============  ===========================================

Hazard classes (codes >= :data:`U4_OCCLUDED`) are never quantised. 2 x 2 pooling is
*conservative*: a pooled cell takes the most severe hazard code of its four sub-cells; with no
hazard it is GROUND only if all four sub-cells are GROUND (else UNSEEN), with the highest
ground-cost bin. Ground-cost quantisation rounds each bin *up* to the top of its coarser bin.
The decoder always returns the contract's (64, 64) grid (pooled maps are nearest-upsampled).

Packet layout (little-endian)::

    offset  type      field
    0       u8        magic 0x4D ('M')
    1       u8        codec version (2)
    2       u16       seq (mod 65536)
    4       u32       t [ms]
    8       u8        DriveMode index (order of DriveMode)
    9       u8        reason code (REASON_CODES index; 0 = free text)
    10      u8 + n    reason detail, ASCII, n <= 40
    ..      i16 i16   pose x, y [cm], A-frame (clipped to +-327 m)
    ..      i16       yaw [0.01 rad], wrapped to (-pi, pi]
    ..      u16       pos sigma [mm]
    ..      u16 x3    v_cap [cm/s], R_cert [cm], speed [cm/s] (speed clipped >= 0)
    ..      u8 + 4n   waypoints: n <= 5, each (i16 x, i16 y) [cm], A-frame
    ..      u8        costmap format: 0xFF = no costmap; else bit 0 = pooled 2 x 2 (32 x 32 at 0.5 m),
                      bits 1-2 = ground-level code (0: 8 levels, 1: 4, 2: 2), bits 3-5 = ladder rung
    ..      u16 + m   (only if a costmap is sent) raw DEFLATE of the cells, 4 bit, two per byte,
                      high nibble first, row-major
    ..      u8 + ...  health: n entries of (u8 key id, f16 value); key id 255 -> (u8 len, name) follows the id
    end-4   u32       CRC-32 of all preceding bytes

The 4-bit ego costmap is *body-aligned*: row 0 is the far-forward edge
(x_body = +8 m), column 0 the left edge (y_body = +8 m), vehicle at the centre. Cell codes
(:data:`U4_*`) are ordered by severity so max-pooling is conservative.
"""

from __future__ import annotations

import logging
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from metagross.config.defaults import LINK_KBPS, TELEMETRY_HZ
from metagross.contracts.messages import DriveMode, Telemetry

LOG = logging.getLogger(__name__)

MAGIC = 0x4D
CODEC_VERSION = 2
COSTMAP_N = 64
COSTMAP_RES_M = 0.25
MAX_WAYPOINTS = 5
MAX_REASON_DETAIL = 40
_MODES = list(DriveMode)

PACKET_BUDGET_B = int(LINK_KBPS * 1000.0 / 8.0 / TELEMETRY_HZ)  # 600 B: the radio's capacity per packet period
PACKET_TARGET_FRACTION = 0.8  # use at most 80 % of the air time: headroom for radio framing / retries
PACKET_TARGET_B = int(PACKET_BUDGET_B * PACKET_TARGET_FRACTION)  # 480 B

# 4-bit ego-costmap cell codes (severity ordered).
U4_UNSEEN = 0
U4_GROUND_MIN, U4_GROUND_MAX = 1, 8  # GROUND with cost bins 0..1
U4_OCCLUDED = 9
U4_CREST_SHADOW = 10
U4_WATER = 11
U4_DITCH_CANDIDATE = 12  # unconfirmed
U4_DYNAMIC = 13
U4_DITCH = 14  # confirmed ditch (lethal)
U4_LETHAL = 15  # POSITIVE / DEPRESSION

REASON_CODES: tuple[str, ...] = (
    "",  # 0: free text in detail
    "OK", "START", "GOV_RCERT", "GOV_RVIS", "GOV_DITCH_DET", "GOV_PLATFORM", "GOV_OFF", "GOV_FIXED", "GOV_CAP",
    "CAUTION", "DEGRADED", "STOP_AND_LOOK", "NO_PROGRESS", "RESUME", "NO_CERTIFIED_PROGRESS", "HEALTH", "IMMOBILISED",
    "ESTOP", "HOLD", "WATCHDOG", "FWD_CHECK", "ARRIVED", "NO_ROUTE", "GLARE", "DIM", "DUST", "VO_FAIL", "SLIP",
)
_REASON_INDEX = {c: i for i, c in enumerate(REASON_CODES)}

# Health key table. Ids are stable: append only. Ids 0..15 are codec v1; 16.. were added in v2 so the
# localiser / image-health keys that every real packet carries no longer travel as free text.
HEALTH_KEYS: tuple[str, ...] = (
    "q", "p_fail", "pos_sigma_m", "vo_ok", "slip", "chi_hat", "r_vis_m", "compute_ms", "inliers", "inlier_ratio",
    "reproj_px", "sat_frac", "dark_frac", "texture", "ess", "latency_ms",
    "vo_inliers", "vo_inlier_ratio", "vo_reproj_rmse_px", "vo_coverage", "vo_track_age", "vo_hess_min_eig",
    "img_lapvar", "img_sat_frac", "img_dark_frac", "img_rms_contrast", "img_dark_channel", "disp_density_ground",
    "q_inst", "q_gate", "vo_available",
)
_HEALTH_INDEX = {k: i for i, k in enumerate(HEALTH_KEYS)}
CORE_HEALTH_KEYS: frozenset[str] = frozenset(HEALTH_KEYS[:8])  # what the operator console shows first
_FREE_KEY = 255
_MAX_FREE_KEY_LEN = 32
_F16_MAX = 65504.0

CM_NONE = 0xFF
_CM_POOLED_BIT = 0x01
_CM_LEVEL_SHIFT = 1
_CM_RUNG_SHIFT = 3
_GROUND_LEVEL_CODES = {8: 0, 4: 1, 2: 2}
_GROUND_LEVELS_OF_CODE = {v: k for k, v in _GROUND_LEVEL_CODES.items()}
_DEFLATE_LEVEL = 9
_DEFLATE_WBITS = -15  # raw DEFLATE: the packet CRC already protects the payload


@dataclass(frozen=True)
class Rung:
    """One step of the fidelity ladder (see module docstring)."""

    pool: int  # 1: 64 x 64 at 0.25 m; 2: 32 x 32 at 0.5 m; 0: no costmap
    ground_levels: int  # 8, 4 or 2 ground-cost levels (ignored without costmap)
    all_health: bool  # False: CORE_HEALTH_KEYS only


LADDER: tuple[Rung, ...] = (
    Rung(1, 8, True), Rung(1, 4, True), Rung(2, 8, True), Rung(2, 4, True), Rung(2, 2, True),
    Rung(2, 2, False), Rung(0, 8, False),
)


@dataclass(frozen=True)
class PacketInfo:
    """What the encoder chose for one packet (for logs and the link-budget report)."""

    n_bytes: int
    rung: int
    costmap_cell_m: Optional[float]  # None: no costmap in this packet
    ground_levels: Optional[int]
    all_health: bool


_HDR = struct.Struct("<BBHIBB")
_POSE = struct.Struct("<hhhH")
_SPD = struct.Struct("<HHH")


class CodecError(ValueError):
    """Raised on malformed / corrupted packets."""


def _cm_i16(v: float) -> int:
    return int(np.clip(round(v * 100.0), -32768, 32767))


def _u16(v: float) -> int:
    return int(np.clip(round(v), 0, 65535))


def pack_u4(cells: np.ndarray) -> bytes:
    """(H, W) uint8 in [0,15] -> H*W/2 bytes, two cells per byte (high nibble = even index)."""
    a = np.asarray(cells, np.uint8).ravel()
    if a.size % 2:
        raise ValueError("u4 grid must have an even number of cells")
    if a.max(initial=0) > 15:
        raise ValueError("u4 values must be in [0, 15]")
    return ((a[0::2] << 4) | a[1::2]).astype(np.uint8).tobytes()


def unpack_u4(data: bytes, shape: tuple[int, int] = (COSTMAP_N, COSTMAP_N)) -> np.ndarray:
    """Inverse of :func:`pack_u4`."""
    b = np.frombuffer(data, np.uint8)
    out = np.empty(b.size * 2, np.uint8)
    out[0::2] = b >> 4
    out[1::2] = b & 0x0F
    return out.reshape(shape)


def pool2_conservative(cells: np.ndarray) -> np.ndarray:
    """(2n, 2n) codes -> (n, n): worst hazard code wins; otherwise GROUND only if all 4 sub-cells are GROUND."""
    a = np.asarray(cells, np.uint8)
    h, w = a.shape
    blocks = a.reshape(h // 2, 2, w // 2, 2).transpose(0, 2, 1, 3).reshape(h // 2, w // 2, 4)
    worst = blocks.max(axis=-1)
    any_unseen = (blocks == U4_UNSEEN).any(axis=-1)
    out = np.where(worst >= U4_OCCLUDED, worst, np.where(any_unseen, U4_UNSEEN, worst))
    return out.astype(np.uint8)


def upsample2(cells: np.ndarray) -> np.ndarray:
    """(n, n) -> (2n, 2n) nearest-neighbour."""
    return np.repeat(np.repeat(np.asarray(cells, np.uint8), 2, axis=0), 2, axis=1)


def quantize_ground(cells: np.ndarray, levels: int) -> np.ndarray:
    """Coarsen GROUND cost bins (codes 1..8) to ``levels`` bins, rounding each bin *up* (conservative)."""
    if levels not in _GROUND_LEVEL_CODES:
        raise ValueError(f"ground levels must be one of {sorted(_GROUND_LEVEL_CODES)}")
    a = np.asarray(cells, np.uint8).copy()
    n_bins = U4_GROUND_MAX - U4_GROUND_MIN + 1
    step = n_bins // levels
    g = (a >= U4_GROUND_MIN) & (a <= U4_GROUND_MAX)
    a[g] = U4_GROUND_MIN + ((a[g] - U4_GROUND_MIN) // step) * step + (step - 1)
    return a


def _deflate(data: bytes) -> bytes:
    c = zlib.compressobj(_DEFLATE_LEVEL, zlib.DEFLATED, _DEFLATE_WBITS)
    return c.compress(data) + c.flush()


def _inflate(data: bytes) -> bytes:
    return zlib.decompress(data, _DEFLATE_WBITS)


def split_reason(reason: str) -> tuple[int, str]:
    """'GOV_RCERT R=3.1m' -> (code, 'R=3.1m'); unknown prefixes -> (0, full text)."""
    head, _, tail = reason.partition(" ")
    code = _REASON_INDEX.get(head, 0) if head else 0
    detail = tail if code else reason
    return code, detail.encode("ascii", "replace")[:MAX_REASON_DETAIL].decode("ascii")


def join_reason(code: int, detail: str) -> str:
    if code == 0:
        return detail
    head = REASON_CODES[code] if code < len(REASON_CODES) else f"R{code}"
    return f"{head} {detail}" if detail else head


def _head_bytes(tel: Telemetry) -> bytes:
    """Everything before the costmap: header, reason, pose, speeds, waypoints."""
    code, detail = split_reason(tel.reason or "")
    db = detail.encode("ascii")
    x, y, yaw = tel.pose_xy_yaw
    yaw_w = math.atan2(math.sin(yaw), math.cos(yaw))
    wps = list(tel.waypoints_xy or [])[:MAX_WAYPOINTS]
    parts = [
        _HDR.pack(MAGIC, CODEC_VERSION, int(tel.seq) & 0xFFFF, int(round(max(tel.t, 0.0) * 1000.0)) & 0xFFFFFFFF,
                  _MODES.index(DriveMode(tel.mode)), code),
        bytes([len(db)]), db,
        _POSE.pack(_cm_i16(x), _cm_i16(y), int(np.clip(round(yaw_w * 100.0), -32768, 32767)), _u16(tel.pos_sigma_m * 1000.0)),
        _SPD.pack(_u16(tel.v_cap_mps * 100.0), _u16(tel.r_cert_m * 100.0), _u16(max(tel.speed_mps, 0.0) * 100.0)),
        bytes([len(wps)]),
    ]
    parts += [struct.pack("<hh", _cm_i16(wx), _cm_i16(wy)) for wx, wy in wps]
    return b"".join(parts)


def _health_bytes(health: dict, all_keys: bool) -> bytes:
    items = [(k, v) for k, v in (health or {}).items()
             if v is not None and np.isfinite(float(v)) and (all_keys or k in CORE_HEALTH_KEYS)][:255]
    parts = [bytes([len(items)])]
    for k, v in items:
        kid = _HEALTH_INDEX.get(k)
        if kid is None:
            kb = str(k).encode("ascii", "replace")[:_MAX_FREE_KEY_LEN]
            parts += [bytes([_FREE_KEY, len(kb)]), kb]
        else:
            parts.append(bytes([kid]))
        parts.append(np.float16(np.clip(float(v), -_F16_MAX, _F16_MAX)).tobytes())
    return b"".join(parts)


def _costmap_bytes(cm: Optional[np.ndarray], rung: int) -> bytes:
    r = LADDER[rung]
    if cm is None or r.pool == 0:
        return bytes([CM_NONE])
    cells = pool2_conservative(cm) if r.pool == 2 else np.asarray(cm, np.uint8)
    if r.ground_levels != 8:
        cells = quantize_ground(cells, r.ground_levels)
    fmt = (_CM_POOLED_BIT if r.pool == 2 else 0) | (_GROUND_LEVEL_CODES[r.ground_levels] << _CM_LEVEL_SHIFT) | (rung << _CM_RUNG_SHIFT)
    z = _deflate(pack_u4(cells))
    return bytes([fmt]) + struct.pack("<H", len(z)) + z


def encode_with_info(tel: Telemetry, budget_bytes: int = PACKET_TARGET_B) -> tuple[bytes, PacketInfo]:
    """Encode :class:`Telemetry` at the highest-fidelity ladder rung that fits ``budget_bytes``.

    If even the last rung (no costmap, core health) exceeds the budget (only possible with a
    huge free-text health dict) that rung is returned anyway and a warning is logged."""
    cm = None
    if tel.costmap_u4 is not None:
        cm = np.asarray(tel.costmap_u4)
        if cm.shape != (COSTMAP_N, COSTMAP_N):
            raise ValueError(f"costmap_u4 must be {(COSTMAP_N, COSTMAP_N)}, got {cm.shape}")
    head = _head_bytes(tel)
    health_cache: dict[bool, bytes] = {}
    body = b""
    rung = 0
    for rung, r in enumerate(LADDER):
        if cm is None and r.pool != 0 and rung > 0:
            continue  # without a costmap only the health choice matters
        hb = health_cache.setdefault(r.all_health, _health_bytes(tel.health, r.all_health))
        body = head + _costmap_bytes(cm, rung) + hb
        if len(body) + 4 <= budget_bytes:
            break
    else:
        LOG.warning("telemetry seq %s: %d B exceeds the %d B budget even without a costmap", tel.seq, len(body) + 4, budget_bytes)
    pkt = body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
    r = LADDER[rung]
    sent_cm = cm is not None and r.pool != 0
    info = PacketInfo(n_bytes=len(pkt), rung=rung, costmap_cell_m=COSTMAP_RES_M * r.pool if sent_cm else None,
                      ground_levels=r.ground_levels if sent_cm else None, all_health=r.all_health)
    return pkt, info


def encode(tel: Telemetry, budget_bytes: int = PACKET_TARGET_B) -> bytes:
    """Encode :class:`Telemetry` into a compact packet (see module docstring)."""
    return encode_with_info(tel, budget_bytes)[0]


def decode(packet: bytes) -> Telemetry:
    """Inverse of :func:`encode`. Raises :class:`CodecError` on bad magic / CRC / truncation.

    The returned costmap is always (64, 64) (pooled maps are upsampled, quantised bins stay quantised)."""
    if len(packet) < _HDR.size + 4:
        raise CodecError("packet too short")
    body, crc = packet[:-4], struct.unpack("<I", packet[-4:])[0]
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        raise CodecError("CRC mismatch")
    try:
        magic, ver, seq, t_ms, mode_i, code = _HDR.unpack_from(body, 0)
        if magic != MAGIC or ver != CODEC_VERSION:
            raise CodecError(f"bad magic/version {magic:#x}/{ver}")
        o = _HDR.size
        n = body[o]; o += 1
        detail = body[o:o + n].decode("ascii"); o += n
        xc, yc, yawc, sig = _POSE.unpack_from(body, o); o += _POSE.size
        vc, rc, sc = _SPD.unpack_from(body, o); o += _SPD.size
        nw = body[o]; o += 1
        wps = []
        for _ in range(nw):
            wx, wy = struct.unpack_from("<hh", body, o); o += 4
            wps.append((wx / 100.0, wy / 100.0))
        fmt = body[o]; o += 1
        costmap: Optional[np.ndarray] = None
        if fmt != CM_NONE:
            (m,) = struct.unpack_from("<H", body, o); o += 2
            pooled = bool(fmt & _CM_POOLED_BIT)
            side = COSTMAP_N // 2 if pooled else COSTMAP_N
            cells = unpack_u4(_inflate(body[o:o + m]), (side, side)); o += m
            costmap = upsample2(cells) if pooled else cells
        nh = body[o]; o += 1
        health: dict[str, float] = {}
        for _ in range(nh):
            kid = body[o]; o += 1
            if kid == _FREE_KEY:
                kl = body[o]; o += 1
                key = body[o:o + kl].decode("ascii"); o += kl
            else:
                key = HEALTH_KEYS[kid]
            health[key] = float(np.frombuffer(body[o:o + 2], np.float16)[0]); o += 2
    except (struct.error, IndexError, ValueError, zlib.error, UnicodeDecodeError) as exc:
        if isinstance(exc, CodecError):
            raise
        raise CodecError(f"malformed packet: {exc}") from exc
    return Telemetry(
        t=t_ms / 1000.0, seq=seq, pose_xy_yaw=(xc / 100.0, yc / 100.0, yawc / 100.0), pos_sigma_m=sig / 1000.0,
        mode=_MODES[mode_i], reason=join_reason(code, detail), v_cap_mps=vc / 100.0, r_cert_m=rc / 100.0,
        speed_mps=sc / 100.0, waypoints_xy=wps, costmap_u4=costmap, health=health,
    )


def to_jsonable(tel: Telemetry, packet_bytes: Optional[int] = None, info: Optional[PacketInfo] = None) -> dict:
    """JSON-friendly dict of a (decoded) telemetry packet; costmap as hex of the packed nibbles (64 x 64).

    ``info`` (from :func:`encode_with_info`) adds ``link_rung``, ``costmap_cell_m`` and ``ground_levels``."""
    d = {
        "t": round(tel.t, 3), "seq": tel.seq, "mode": DriveMode(tel.mode).value, "reason": tel.reason,
        "pose": [round(tel.pose_xy_yaw[0], 2), round(tel.pose_xy_yaw[1], 2), round(tel.pose_xy_yaw[2], 2)],
        "pos_sigma_m": round(tel.pos_sigma_m, 3), "v_cap_mps": round(tel.v_cap_mps, 2), "r_cert_m": round(tel.r_cert_m, 2),
        "speed_mps": round(tel.speed_mps, 2), "waypoints": [[round(a, 2), round(b, 2)] for a, b in tel.waypoints_xy],
        "health": {k: round(float(v), 4) for k, v in tel.health.items()},
        "costmap_u4_hex": pack_u4(tel.costmap_u4).hex() if tel.costmap_u4 is not None else None,
    }
    if packet_bytes is not None:
        d["packet_bytes"] = int(packet_bytes)
    if info is not None:
        d["link_rung"] = info.rung
        d["costmap_cell_m"] = info.costmap_cell_m
        d["ground_levels"] = info.ground_levels
    return d


def telemetry_from_jsonable(d: dict) -> Telemetry:
    """Rebuild a :class:`Telemetry` from one ``telemetry.jsonl`` line (inverse of :func:`to_jsonable`)."""
    hx = d.get("costmap_u4_hex")
    return Telemetry(
        t=float(d["t"]), seq=int(d["seq"]), pose_xy_yaw=tuple(float(v) for v in d["pose"]), pos_sigma_m=float(d["pos_sigma_m"]),
        mode=DriveMode(d["mode"]), reason=str(d.get("reason") or ""), v_cap_mps=float(d["v_cap_mps"]), r_cert_m=float(d["r_cert_m"]),
        speed_mps=float(d["speed_mps"]), waypoints_xy=[(float(a), float(b)) for a, b in d.get("waypoints", [])],
        costmap_u4=unpack_u4(bytes.fromhex(hx)) if hx else None, health={k: float(v) for k, v in (d.get("health") or {}).items()},
    )
