"""Project a 5-class semantic mask into the egocentric BEV and fuse it monotonically.

Pixels (sub-sampled on a stride grid) are placed in the BEV by:

* valid disparity -> the measured 3-D point (body frame);
* no disparity but the pixel ray hits the ground model within range (below the horizon)
  -> the ray / ground intersection (water and dark mud often give no stereo match).

Each cell accumulates a class histogram (``interfaces.SEM_CLASSES``). From it:

* semantic cost = histogram-weighted ``interfaces.SEM_COST`` (sky pixels ignored);
* ``WATER`` when water/mud holds >= ``WATER_MIN_FRACTION`` of the labelled pixels;
* braking-friction proxy mu = histogram-weighted ``interfaces.SEM_MU``
  (``MU_DEFAULT`` where nothing was labelled).

Fusion is **monotone**: semantics may only raise a cell's cost and may only turn
"soft" states (UNSEEN / GROUND / OCCLUDED / CREST_SHADOW) into WATER; a geometric lethal
state (POSITIVE, DEPRESSION, DITCH_CANDIDATE) is never cleared or downgraded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from metagross.autonomy.perception.bev import BevSpec
from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.autonomy.perception.ground import GroundModel
from metagross.contracts.interfaces import SEM_CLASSES, SEM_COST, SEM_MU
from metagross.contracts.messages import CellState

N_CLASSES = len(SEM_CLASSES)
SEM_SKY, SEM_WATER = 0, 2
PIXEL_STRIDE = 4  # semantic pixels sampled every 4th row and column
WATER_MIN_FRACTION = 0.5
MIN_LABELLED_PX = 2  # labelled pixels needed before a cell's semantics count
ENTROPY_MAX = 0.8  # pixels with normalised entropy above this are ignored (if entropy given)
MU_DEFAULT = SEM_MU[0]  # unknown surface: conservative friction proxy
_COST_LUT = np.array([SEM_COST[i] for i in range(N_CLASSES)], dtype=np.float32)
_MU_LUT = np.array([SEM_MU[i] for i in range(N_CLASSES)], dtype=np.float32)
SOFT_STATES = (CellState.UNSEEN, CellState.GROUND, CellState.OCCLUDED, CellState.CREST_SHADOW)


@dataclass
class SemanticBev:
    """Per-cell semantic layer (arrays shaped like the BEV grid)."""

    hist: np.ndarray  # (nx, ny, 5) int32 pixel counts per class
    cost: np.ndarray  # float32 semantic cost in [0, 1], 0 where unlabelled
    water: np.ndarray  # bool: water / mud dominant
    mu: np.ndarray  # float32 friction proxy
    labelled: np.ndarray  # bool: enough labelled pixels


def project_semantics(classes: np.ndarray, disp: np.ndarray, geom: CameraGeometry, model: GroundModel, spec: BevSpec,
                      entropy: Optional[np.ndarray] = None) -> SemanticBev:
    """Class-id mask (H, W) uint8 + disparity (px) -> per-cell semantic layer."""
    s = PIXEL_STRIDE
    cls = classes[::s, ::s].astype(np.int64)
    d = disp[::s, ::s]
    vv, uu = np.mgrid[0 : disp.shape[0] : s, 0 : disp.shape[1] : s]
    keep = (cls > SEM_SKY) & (cls < N_CLASSES)
    if entropy is not None:
        keep &= entropy[::s, ::s] <= ENTROPY_MAX
    v, u, c, dd = vv[keep], uu[keep], cls[keep], d[keep]
    zc = np.where(dd > 0, geom.fxb / np.where(dd > 0, dd, 1.0), np.nan)
    need = ~(dd > 0)
    if np.any(need):
        zc[need] = model.expected_depth(geom, v[need], u[need])
    ok = np.isfinite(zc) & (zc <= geom.max_range_m)
    r = geom.rays_body[v[ok], u[ok]]
    x = geom.t_bc[0] + zc[ok] * r[:, 0]
    y = geom.t_bc[1] + zc[ok] * r[:, 1]
    idx = spec.flat_index(x, y)
    m = idx >= 0
    nx, ny = spec.shape
    hist = np.bincount(idx[m] * N_CLASSES + c[ok][m], minlength=spec.n_cells * N_CLASSES)
    hist = hist.reshape(nx, ny, N_CLASSES).astype(np.int32)
    tot = hist.sum(2)
    labelled = tot >= MIN_LABELLED_PX
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = hist / np.maximum(tot, 1)[..., None]
    cost = np.where(labelled, (frac * _COST_LUT).sum(2), 0.0).astype(np.float32)
    mu = np.where(labelled, (frac * _MU_LUT).sum(2), MU_DEFAULT).astype(np.float32)
    water = labelled & (frac[..., SEM_WATER] >= WATER_MIN_FRACTION)
    return SemanticBev(hist, cost, water, mu, labelled)


def fuse_semantics(states: np.ndarray, cost: np.ndarray, sem: SemanticBev) -> tuple[np.ndarray, np.ndarray]:
    """Monotone fusion: returns (states, cost) with semantics only ever raising cost."""
    states = states.copy()
    soft = np.isin(states, np.array(SOFT_STATES, dtype=np.uint8))
    states[soft & sem.water] = CellState.WATER
    cost = np.maximum(cost, np.where(sem.labelled, sem.cost, 0.0))
    cost = np.where(states == CellState.WATER, np.maximum(cost, SEM_COST[SEM_WATER]), cost).astype(np.float32)
    return states, cost
