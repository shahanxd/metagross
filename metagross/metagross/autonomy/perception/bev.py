"""Egocentric bird's-eye-view (BEV) grid: point statistics, expected coverage, visibility.

Grid convention (body frame, metres)
------------------------------------
``grid[i, j]`` covers ``x in [x_min + i*res, x_min + (i+1)*res)`` (forward) and
``y in [y_min + j*res, y_min + (j+1)*res)`` (left positive), i.e. row index grows
forward and column index grows to the LEFT. ``BevSpec.display(a)`` flips an array to
the usual picture (forward up, left on the left). Size = ``defaults.BEV_LOCAL_SIZE_M``
(forward x lateral) at ``defaults.BEV_RES_M``; the grid starts ``BEV_X_MIN_M`` behind the
body origin so the vehicle footprint is inside it.

Per-cell statistics use heights *relative to the local ground model* (m, + up).
The expected-coverage LUT says how many image pixels (and rows) a patch of nominal flat
ground in each cell would occupy in the left image; it is computed once per camera
geometry and makes "unseen" vs "should have been seen" decidable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.config import defaults
from metagross.contracts.messages import CellState

BEV_X_MIN_M = -2.0  # grid starts 2 m behind the body origin (vehicle footprint + margin)
COVERAGE_SUBSAMPLES = 4  # per-axis sub-samples per cell for the FOV fraction of the LUT
EXPECTED_VISIBLE_MIN_PX = 1.0  # a cell "should have been seen" if flat ground there covers >= 1 px
R_VIS_SECTOR_HALF_DEG = 20.0  # forward sector for r_vis
R_VIS_MIN_HALF_WIDTH_M = 0.6  # ... but never narrower than the vehicle corridor
R_VIS_RING_M = 0.5  # ring width along body x for the r_vis test
R_VIS_MIN_FRACTION = 0.6  # >= 60 % of expected-visible cells observed
R_VIS_MIN_CELLS = 4  # rings with fewer expected-visible cells are skipped


@dataclass(frozen=True)
class BevSpec:
    """Geometry of the egocentric grid (see module docstring)."""

    x_min: float = BEV_X_MIN_M
    size_x: float = defaults.BEV_LOCAL_SIZE_M[0]
    size_y: float = defaults.BEV_LOCAL_SIZE_M[1]
    res: float = defaults.BEV_RES_M

    @property
    def y_min(self) -> float:
        return -0.5 * self.size_y

    @property
    def shape(self) -> tuple[int, int]:
        return int(round(self.size_x / self.res)), int(round(self.size_y / self.res))

    @property
    def n_cells(self) -> int:
        nx, ny = self.shape
        return nx * ny

    def flat_index(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Flat cell index for body (x, y); -1 outside the grid."""
        nx, ny = self.shape
        i = np.floor((np.asarray(x) - self.x_min) / self.res).astype(np.int64)
        j = np.floor((np.asarray(y) - self.y_min) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        return np.where(ok, i * ny + j, -1)

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        """(X, Y) body coordinates of cell centres, each (nx, ny)."""
        nx, ny = self.shape
        xs = self.x_min + (np.arange(nx) + 0.5) * self.res
        ys = self.y_min + (np.arange(ny) + 0.5) * self.res
        return np.meshgrid(xs, ys, indexing="ij")

    def display(self, grid: np.ndarray) -> np.ndarray:
        """Flip to picture orientation: forward up, vehicle-left on the left."""
        return grid[::-1, ::-1]

    def as_dict(self) -> dict:
        nx, ny = self.shape
        return {"x_min_m": self.x_min, "y_min_m": self.y_min, "res_m": self.res, "shape": (nx, ny),
                "index": "grid[i, j]: x = x_min + (i + 0.5) res (fwd), y = y_min + (j + 0.5) res (left)"}


@dataclass
class BevStats:
    """Per-cell statistics of back-projected points (arrays shaped like the grid)."""

    count: np.ndarray  # int32 points per cell
    z_mean: np.ndarray  # float32 mean body-frame height (m), NaN if empty
    h_min: np.ndarray  # float32 min height above local ground (m), NaN if empty
    h_max: np.ndarray  # float32 max height above local ground (m), NaN if empty
    h_mean: np.ndarray  # float32 mean height above local ground (m), NaN if empty
    h_std: np.ndarray  # float32 std of height above ground (m), NaN if empty


def accumulate(spec: BevSpec, x: np.ndarray, y: np.ndarray, z: np.ndarray, h: np.ndarray) -> BevStats:
    """Bin body-frame points (x, y, z) with heights above ground h into the grid."""
    n = spec.n_cells
    idx = spec.flat_index(x, y)
    ok = idx >= 0
    idx, z, h = idx[ok], z[ok].astype(np.float64), h[ok].astype(np.float64)
    cnt = np.bincount(idx, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        zs = np.bincount(idx, weights=z, minlength=n) / cnt
        hs = np.bincount(idx, weights=h, minlength=n) / cnt
        hq = np.bincount(idx, weights=h * h, minlength=n) / cnt
    hmin = np.full(n, np.inf)
    hmax = np.full(n, -np.inf)
    np.minimum.at(hmin, idx, h)
    np.maximum.at(hmax, idx, h)
    empty = cnt == 0
    hmin[empty] = np.nan
    hmax[empty] = np.nan
    std = np.sqrt(np.maximum(hq - hs * hs, 0.0))
    shp = spec.shape
    f32 = lambda a: a.reshape(shp).astype(np.float32)  # noqa: E731
    return BevStats(cnt.reshape(shp).astype(np.int32), f32(zs), f32(hmin), f32(hmax), f32(hs), f32(std))


def expected_coverage(geom: CameraGeometry, spec: BevSpec) -> tuple[np.ndarray, np.ndarray]:
    """LUT of image coverage of nominal flat ground (z = 0) per cell.

    Returns (pixels, rows), both float32 (nx, ny): the pixel area and the number of image
    rows the cell's ground patch would occupy in the left image, scaled by the fraction
    of the patch that is inside the image and within the camera range cap. Zero for
    cells the camera cannot see.
    """
    X, Y = spec.centres()
    r = spec.res
    corners = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    us, vs, zs = [], [], []
    for dx, dy in corners:
        p = np.stack([X + dx * r, Y + dy * r, np.zeros_like(X)], axis=-1).reshape(-1, 3)
        u, v, zc = geom.project_body(p)
        us.append(u), vs.append(v), zs.append(zc)
    U, V, Zc = np.stack(us, 1), np.stack(vs, 1), np.stack(zs, 1)  # (N, 4)
    front = np.all(Zc > 0.05, axis=1)
    area = 0.5 * np.abs(np.sum(U * np.roll(V, -1, axis=1) - np.roll(U, -1, axis=1) * V, axis=1))
    rows = np.max(V, axis=1) - np.min(V, axis=1)
    # Fraction of the cell visible (inside the image and within range).
    s = (np.arange(COVERAGE_SUBSAMPLES) + 0.5) / COVERAGE_SUBSAMPLES - 0.5
    frac = np.zeros(X.size)
    for dx in s:
        for dy in s:
            p = np.stack([X.ravel() + dx * r, Y.ravel() + dy * r, np.zeros(X.size)], axis=-1)
            u, v, zc = geom.project_body(p)
            inside = (zc > 0.05) & (zc <= geom.max_range_m) & (u >= 0) & (u <= geom.width - 1) & (v >= 0) & (v <= geom.height - 1)
            frac += inside
    frac /= COVERAGE_SUBSAMPLES**2
    px = np.where(front, area * frac, 0.0).reshape(X.shape).astype(np.float32)
    rw = np.where(front, rows * frac, 0.0).reshape(X.shape).astype(np.float32)
    return px, rw


# States that count as "observed" for the visibility range (direct evidence exists).
OBSERVED_STATES = np.array([CellState.GROUND, CellState.POSITIVE, CellState.DEPRESSION, CellState.DITCH_CANDIDATE,
                            CellState.WATER, CellState.DYNAMIC], dtype=np.uint8)


def visible_range(spec: BevSpec, states: np.ndarray, expected_px: np.ndarray) -> float:
    """Farthest forward range (m, body x) up to which every ring of the forward sector has
    >= R_VIS_MIN_FRACTION of its expected-visible cells actually observed.

    Rings are ``R_VIS_RING_M`` wide along x; rings with too few expected-visible cells
    (e.g. under the camera's near blind zone) are skipped at the start. Returns 0.0 when
    even the first visible ring fails.
    """
    X, Y = spec.centres()
    half = np.maximum(R_VIS_MIN_HALF_WIDTH_M, X * np.tan(np.radians(R_VIS_SECTOR_HALF_DEG)))
    sector = (np.abs(Y) <= half) & (X > 0)
    expv = sector & (expected_px >= EXPECTED_VISIBLE_MIN_PX)
    obs = np.isin(states, OBSERVED_STATES) & expv
    ring = np.floor(X / R_VIS_RING_M).astype(np.int64)
    ring = np.where(expv, ring, -1)
    n_ring = int(ring.max()) + 1 if np.any(expv) else 0
    if n_ring <= 0:
        return 0.0
    n_exp = np.bincount(ring[expv], minlength=n_ring)
    n_obs = np.bincount(ring[obs], minlength=n_ring)
    r_vis = 0.0
    started = False
    for k in range(n_ring):
        if n_exp[k] < R_VIS_MIN_CELLS:
            if started:
                break
            continue
        started = True
        if n_obs[k] < R_VIS_MIN_FRACTION * n_exp[k]:
            break
        r_vis = (k + 1) * R_VIS_RING_M
    return float(r_vis)


def rasterize_tracks(spec: BevSpec, x0: np.ndarray, x1: np.ndarray, y_at: tuple[np.ndarray, np.ndarray],
                     half_width: np.ndarray, step_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Cells covered by straight ground-track segments.

    Segment k runs along body x from x0[k] to x1[k] with lateral position
    ``y = y_at[0][k] + y_at[1][k] * x`` and a lateral half-width that grows linearly with x:
    ``half_width[k] * x`` (m). Returns (flat cell indices, segment ids) for every sample
    (duplicates allowed; outside-grid samples dropped).
    """
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    n = np.maximum(np.ceil((x1 - x0) / step_m).astype(np.int64) + 1, 1)
    seg = np.repeat(np.arange(x0.size), n)
    if seg.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    offs = np.arange(seg.size) - np.repeat(np.cumsum(n) - n, n)
    xs = np.minimum(x0[seg] + offs * step_m, x1[seg])
    yc = y_at[0][seg] + y_at[1][seg] * xs
    hw = half_width[seg] * np.maximum(xs, 0.0)
    cells, ids = [], []
    for f in (-1.0, 0.0, 1.0):
        idx = spec.flat_index(xs, yc + f * hw)
        ok = idx >= 0
        cells.append(idx[ok])
        ids.append(seg[ok])
    return np.concatenate(cells), np.concatenate(ids)
