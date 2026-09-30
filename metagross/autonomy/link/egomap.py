"""Render the rolling map into the 64 x 64 x 4-bit ego costmap carried by telemetry.

See :mod:`metagross.autonomy.link.codec` for the cell codes and orientation
(body-aligned, row 0 = far forward, column 0 = left, 0.25 m cells, vehicle centred).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from metagross.autonomy.link import codec
from metagross.autonomy.planning.costmap import PlanningMaps
from metagross.autonomy.planning.rolling_map import RollingMap
from metagross.contracts.messages import CellState as S


def state_codes(rmap: RollingMap, maps: PlanningMaps) -> np.ndarray:
    """Per-map-cell u4 code layer on the rolling-map grid."""
    st = rmap.state
    code = np.zeros(st.shape, np.uint8)
    ground = st == S.GROUND
    bins = codec.U4_GROUND_MIN + np.clip((maps.cost * 8.0).astype(np.int32), 0, 7)
    code[ground] = bins[ground].astype(np.uint8)
    code[st == S.OCCLUDED] = codec.U4_OCCLUDED
    code[st == S.CREST_SHADOW] = codec.U4_CREST_SHADOW
    code[st == S.WATER] = codec.U4_WATER
    cand = st == S.DITCH_CANDIDATE
    code[cand & ~rmap.confirmed] = codec.U4_DITCH_CANDIDATE
    code[cand & rmap.confirmed] = codec.U4_DITCH
    code[st == S.DYNAMIC] = codec.U4_DYNAMIC
    code[(st == S.POSITIVE) | (st == S.DEPRESSION)] = codec.U4_LETHAL
    return code


def ego_costmap_u4(rmap: RollingMap, maps: PlanningMaps, pose_xy_yaw: tuple[float, float, float],
                   n: int = codec.COSTMAP_N, res_m: float = codec.COSTMAP_RES_M) -> np.ndarray:
    """(n, n) uint8 codes in [0, 15], body-aligned around ``pose_xy_yaw`` (A-frame)."""
    code = state_codes(rmap, maps)
    # lethal codes are max-dilated by one map cell so that the 0.25 m resampling cannot drop them
    leth = np.where(code >= codec.U4_DYNAMIC, code, 0).astype(np.uint8)
    leth = cv2.dilate(leth, np.ones((3, 3), np.uint8))
    code = np.where(leth > 0, leth, code).astype(np.uint8)
    g = rmap.geometry
    px, py, yaw = pose_xy_yaw
    c, s = math.cos(yaw), math.sin(yaw)
    half = n * res_m / 2.0 - res_m / 2.0  # body coordinate of the centre of row 0 / column 0
    k = res_m / g.res
    M = np.array(
        [
            [k * s, -k * c, (px + c * half - s * half - g.origin_x) / g.res - 0.5],
            [-k * c, -k * s, (py + s * half + c * half - g.origin_y) / g.res - 0.5],
        ],
        dtype=np.float64,
    )
    return cv2.warpAffine(code, M, (n, n), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=codec.U4_UNSEEN)
