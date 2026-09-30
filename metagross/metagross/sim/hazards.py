"""Ground-truth hazard rasters for the referee, the evaluator and the scenario generator.

The ditch cross-section is defined here, once, and used both to *carve* the terrain in
:mod:`metagross.sim.scenario` and to derive the ditch mask from a stored scenario, so the GT mask
and the heightmap can never disagree.

Masks (all ``(ny, nx)`` bool on the terrain grid, 0.05 m cells):

* ``object``  -- footprints of static objects protruding more than the ground clearance.
* ``ditch``   -- cells excavated more than ``DITCH_MASK_MIN_DROP_M`` by a ditch hazard.
* ``water``   -- water or mud material (flat, geometrically benign, semantically hazardous).
* ``slope``   -- vehicle-scale slope above ``defaults.SLOPE_LETHAL_DEG``.
* ``dynamic_rest`` -- final resting footprints of scripted dynamic obstacles.

``lethal = object | ditch | slope`` and ``hazard = lethal | water | dynamic_rest``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from metagross.config import defaults
from metagross.sim.geometry import GridSpec, fill_circle, fill_polygon, polyline_band
from metagross.sim.objects import DynamicObject, Footprint, parse_static_objects, rect_corners
from metagross.sim.terrain import Terrain

# Ditch cross-section: nominal wall angle and the minimum horizontal wall run (>= 2 terrain
# cells at 0.05 m so the wall is resolved by the raster instead of being a 1-cell step).
DITCH_WALL_ANGLE_DEG = 72.0
DITCH_MIN_WALL_RUN_M = 0.10
DITCH_MAX_RUN_FRAC = 0.45  # wall run never exceeds this fraction of the top width
# Length over which a ditch shallows to zero at each end of a bypass gap (m).
DITCH_GAP_TAPER_M = 0.30
# Cells excavated more than this (m) count as ditch in the GT mask.
DITCH_MASK_MIN_DROP_M = 0.05
# Gaussian smoothing scale used for the vehicle-scale slope mask (~ half the wheelbase).
SLOPE_SMOOTH_SIGMA_M = 0.25
WATER_MATERIALS = (4, 5)  # mud, water


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ditch_wall_run(width: float, depth: float) -> float:
    """Horizontal run (m) of one ditch wall for a given top width and depth."""
    return max(DITCH_MIN_WALL_RUN_M, min(DITCH_MAX_RUN_FRAC * width, depth / math.tan(math.radians(DITCH_WALL_ANGLE_DEG))))


def ditch_drop(grid: GridSpec, ditch: dict) -> np.ndarray:
    """Excavation depth field (m, >= 0) of one ditch hazard on ``grid``.

    Cross-section: ``depth * smoothstep((w/2 - rho) / run)`` where ``rho`` is the distance to the
    centre polyline, i.e. a flat bottom with steep, rounded walls (``DITCH_WALL_ANGLE_DEG``).
    Along the polyline the depth is multiplied by 0 inside every bypass gap ``[s0, s1]`` (arc
    length, m) and ramps back to full depth over ``DITCH_GAP_TAPER_M`` outside it.
    """
    poly = np.asarray(ditch["polyline"], float)
    w, depth = float(ditch["width"]), float(ditch["depth"])
    run = ditch_wall_run(w, depth)
    dist, arc = polyline_band(grid, poly, 0.5 * w + grid.res)
    across = depth * _smoothstep((0.5 * w - dist) / run)
    along = np.ones_like(across)
    band = np.isfinite(dist)
    s = arc[band]
    f = np.ones_like(s)
    for s0, s1 in ditch.get("gaps", []):
        d_out = np.maximum(s0 - s, s - s1)  # <0 inside the gap, distance outside it
        f = np.minimum(f, _smoothstep(d_out / DITCH_GAP_TAPER_M))
    along[band] = f
    out = np.where(band, across * along, 0.0).astype(np.float32)
    return out


def footprint_mask(grid: GridSpec, feet: list[Footprint], mask: np.ndarray | None = None, lethal_only: bool = True) -> np.ndarray:
    """Rasterise circle / rectangle footprints into a uint8 mask."""
    if mask is None:
        mask = np.zeros(grid.shape, np.uint8)
    for fp in feet:
        if lethal_only and not fp.lethal:
            continue
        if fp.kind == "circle":
            fill_circle(grid, fp.cx, fp.cy, fp.r, mask)
        else:
            fill_polygon(grid, rect_corners(fp), mask)
    return mask


@dataclass
class HazardRaster:
    """GT hazard masks on the terrain grid (see module docstring)."""

    grid: GridSpec
    object: np.ndarray
    ditch: np.ndarray
    water: np.ndarray
    slope: np.ndarray
    dynamic_rest: np.ndarray
    ditch_drop: np.ndarray  # (ny, nx) float32 m, total excavation depth of all ditches

    @property
    def lethal(self) -> np.ndarray:
        return self.object | self.ditch | self.slope

    @property
    def hazard(self) -> np.ndarray:
        return self.lethal | self.water | self.dynamic_rest


def gt_hazard_raster(scenario: dict, terrain: Terrain | None = None) -> HazardRaster:
    """Build the GT hazard raster of a scenario dict (deterministic, ~0.3 s for 64 x 40 m)."""
    terrain = terrain or Terrain.from_scenario(scenario)
    grid = terrain.grid
    drop = np.zeros(grid.shape, np.float32)
    for hz in scenario.get("hazards", []):
        if hz["type"] == "ditch":
            drop = np.maximum(drop, ditch_drop(grid, hz))
    ditch = drop > DITCH_MASK_MIN_DROP_M
    _, feet = parse_static_objects(scenario, terrain)
    obj = footprint_mask(grid, feet).astype(bool)
    dyn_feet = []
    for k, d in enumerate(scenario.get("dynamic", [])):
        dob = DynamicObject.from_dict(k, d)
        dob.t_trigger = 0.0
        dyn_feet.append(dob.footprint_at(1e9))  # resting pose at the end of its path
    dyn = footprint_mask(grid, dyn_feet, lethal_only=False).astype(bool)
    water = np.isin(terrain.material, WATER_MATERIALS)
    # Slope of the landform *before* excavation: ditch walls are covered by the ditch mask, and
    # smoothing them into the slope mask would wrongly narrow the bypass gaps.
    landform = gaussian_filter(terrain.height + drop, SLOPE_SMOOTH_SIGMA_M / grid.res, mode="nearest")
    gy, gx = np.gradient(landform, grid.res)
    slope = np.hypot(gx, gy) > math.tan(math.radians(defaults.SLOPE_LETHAL_DEG))
    return HazardRaster(grid, obj, ditch, water, slope, dyn, drop)
