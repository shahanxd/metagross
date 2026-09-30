"""Rigid-body and raster geometry helpers shared by the simulator (ground-truth side only).

Conventions
-----------
* World frame: x east, y north, z up (metres).
* Body frame: x forward, y left, z up (REP-103), origin at the wheelbase centre on the ground.
* Euler angles are intrinsic Z-Y-X: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (body -> world).
  With this convention **positive pitch = nose down** and **positive roll = left side up**
  (right side down), exactly as REP-103 prescribes for a FLU body frame.
* Raster grids: row index ``i`` runs along +y, column index ``j`` along +x; the centre of cell
  ``(0, 0)`` is at ``origin``. This matches the scenario ``terrain`` block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# Fixed-point shift used for sub-cell polygon rasterisation with cv2.fillPoly (1/16 cell).
_POLY_SHIFT = 4


def rot_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body->world rotation matrix for intrinsic Z-Y-X Euler angles (radians)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def pose_matrix(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """4x4 homogeneous transform T_world_body from a 6-DoF pose (metres, radians)."""
    t = np.eye(4)
    t[:3, :3] = rot_zyx(roll, pitch, yaw)
    t[:3, 3] = (x, y, z)
    return t


def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle (radians) to [-pi, pi); works on scalars and arrays."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rot2(a: float) -> np.ndarray:
    """2x2 counter-clockwise rotation matrix."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def polyline_arclength(pts: np.ndarray) -> np.ndarray:
    """Cumulative arc length (m) at each vertex of an (N, 2) polyline; first entry is 0."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points p (..., 2) to segments a->b (broadcastable (..., 2))."""
    ab = b - a
    denom = np.maximum(np.sum(ab * ab, axis=-1), 1e-12)
    t = np.clip(np.sum((p - a) * ab, axis=-1) / denom, 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.linalg.norm(p - proj, axis=-1)


@dataclass(frozen=True)
class GridSpec:
    """Axis-aligned raster over the world xy plane.

    ``origin`` is the world position (m) of the centre of cell (0, 0); ``res`` the cell size (m).
    """

    origin_x: float
    origin_y: float
    res: float
    ny: int
    nx: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) of the cell centres (m)."""
        return (self.origin_x, self.origin_x + (self.nx - 1) * self.res,
                self.origin_y, self.origin_y + (self.ny - 1) * self.res)

    def xy_to_ij_float(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Continuous (row, col) coordinates of world points; integer values are cell centres."""
        return (np.asarray(y) - self.origin_y) / self.res, (np.asarray(x) - self.origin_x) / self.res

    def xy_to_ij(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest cell (row, col), clipped to the grid."""
        fi, fj = self.xy_to_ij_float(x, y)
        i = np.clip(np.rint(fi).astype(np.int64), 0, self.ny - 1)
        j = np.clip(np.rint(fj).astype(np.int64), 0, self.nx - 1)
        return i, j

    def ij_to_xy(self, i: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.origin_x + np.asarray(j) * self.res, self.origin_y + np.asarray(i) * self.res

    def contains(self, x: np.ndarray, y: np.ndarray, margin: float = 0.0) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.extent
        x = np.asarray(x)
        y = np.asarray(y)
        return (x >= xmin + margin) & (x <= xmax - margin) & (y >= ymin + margin) & (y <= ymax - margin)

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Full-size (ny, nx) arrays of cell-centre x and y (float32, m)."""
        xs = (self.origin_x + np.arange(self.nx) * self.res).astype(np.float32)
        ys = (self.origin_y + np.arange(self.ny) * self.res).astype(np.float32)
        return np.broadcast_to(xs[None, :], self.shape), np.broadcast_to(ys[:, None], self.shape)

    def subsample(self, factor: int) -> "GridSpec":
        """Grid made of every ``factor``-th cell centre (same origin)."""
        return GridSpec(self.origin_x, self.origin_y, self.res * factor,
                        (self.ny - 1) // factor + 1, (self.nx - 1) // factor + 1)


def fill_polygon(grid: GridSpec, polygon_xy: np.ndarray, mask: np.ndarray | None = None, value: int = 1) -> np.ndarray:
    """Rasterise a world-frame polygon (N, 2) into a uint8 mask (cells whose centre is inside)."""
    if mask is None:
        mask = np.zeros(grid.shape, np.uint8)
    fi, fj = grid.xy_to_ij_float(polygon_xy[:, 0], polygon_xy[:, 1])
    pts = np.round(np.stack([fj, fi], axis=1) * (1 << _POLY_SHIFT)).astype(np.int32)
    cv2.fillPoly(mask, [pts], int(value), lineType=cv2.LINE_8, shift=_POLY_SHIFT)
    return mask


def fill_circle(grid: GridSpec, cx: float, cy: float, r: float, mask: np.ndarray, value: int = 1) -> np.ndarray:
    """Rasterise a filled world-frame circle into ``mask`` (in place)."""
    fi, fj = grid.xy_to_ij_float(cx, cy)
    s = 1 << _POLY_SHIFT
    cv2.circle(mask, (int(round(float(fj) * s)), int(round(float(fi) * s))), int(round(r / grid.res * s)),
               int(value), thickness=-1, lineType=cv2.LINE_8, shift=_POLY_SHIFT)
    return mask


def oriented_rect_corners(cx: float, cy: float, half_l: float, half_w: float, yaw: float) -> np.ndarray:
    """(4, 2) corners of a rectangle centred at (cx, cy), long axis along ``yaw`` (CCW order)."""
    local = np.array([[half_l, half_w], [-half_l, half_w], [-half_l, -half_w], [half_l, -half_w]])
    return local @ rot2(yaw).T + np.array([cx, cy])


def polyline_band(grid: GridSpec, polyline: np.ndarray, max_dist: float) -> tuple[np.ndarray, np.ndarray]:
    """Distance (m) of every cell centre to a polyline and the arc length (m) of the closest point.

    Only cells within ``max_dist`` of some segment are evaluated (others get +inf / nan), so the
    cost is proportional to the band area, not the grid.
    """
    dist = np.full(grid.shape, np.inf, np.float32)
    arc = np.full(grid.shape, np.nan, np.float32)
    s_vert = polyline_arclength(polyline)
    for k in range(len(polyline) - 1):
        a, b = polyline[k], polyline[k + 1]
        lo = np.minimum(a, b) - max_dist
        hi = np.maximum(a, b) + max_dist
        i0, j0 = grid.xy_to_ij(lo[0], lo[1])
        i1, j1 = grid.xy_to_ij(hi[0], hi[1])
        if i1 < i0 or j1 < j0:
            continue
        ii = np.arange(int(i0), int(i1) + 1)
        jj = np.arange(int(j0), int(j1) + 1)
        xs = grid.origin_x + jj * grid.res
        ys = grid.origin_y + ii * grid.res
        px = xs[None, :] - a[0]
        py = ys[:, None] - a[1]
        ab = b - a
        L2 = max(float(ab @ ab), 1e-12)
        t = np.clip((px * ab[0] + py * ab[1]) / L2, 0.0, 1.0)
        dx = px - t * ab[0]
        dy = py - t * ab[1]
        d = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        sub_d = dist[i0:i1 + 1, j0:j1 + 1]
        sub_s = arc[i0:i1 + 1, j0:j1 + 1]
        better = d < sub_d
        sub_d[better] = d[better]
        sub_s[better] = (s_vert[k] + t * math.sqrt(L2)).astype(np.float32)[better]
    return dist, arc
