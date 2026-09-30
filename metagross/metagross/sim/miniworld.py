"""Hand-built mini scenarios for unit tests (valid ``metagross.scenario/1`` dicts).

Example::

    h = np.zeros((200, 400), np.float32)                       # 20 x 10 m flat at 0.05 m
    scn = mini_scenario(h, start=(2, 5, 0), goal=(18, 5),
                        objects=[{"type": "rock", "xyz": [8, 5, -0.05], "radius": 0.4, "squash": 0.7, "seed": 0}])
"""

from __future__ import annotations

import numpy as np

from metagross.contracts.scenario import FAMILIES, SCHEMA
from metagross.sim.geometry import GridSpec
from metagross.sim.hazards import ditch_drop
from metagross.sim.scenario import scenario_sha256
from metagross.sim.terrain import encode_array_b64

MINI_RES_M = 0.05


def mini_scenario(height: np.ndarray, material: np.ndarray | None = None, *, res: float = MINI_RES_M,
                  origin: tuple[float, float] = (0.0, 0.0), start: tuple[float, float, float] = (1.0, 1.0, 0.0),
                  goal: tuple[float, float] = (5.0, 1.0), objects: list[dict] | None = None, hazards: list[dict] | None = None,
                  dynamic: list[dict] | None = None, lighting: dict | None = None, seed: int = 9999, family: str = FAMILIES[0],
                  timeout_s: float = 60.0, success_radius_m: float = 2.0, heading_init_err_deg: float = 0.0,
                  chi: float = 1.4, slip: float = 0.05, carve_ditches: bool = True) -> dict:
    """Build a scenario dict from explicit rasters. Ditch hazards are carved into ``height``
    (with the same cross-section as the generator) unless ``carve_ditches`` is False."""
    ny, nx = height.shape
    h = np.asarray(height, np.float32).copy()
    mat = np.zeros((ny, nx), np.uint8) if material is None else np.asarray(material, np.uint8)
    grid = GridSpec(float(origin[0]), float(origin[1]), res, ny, nx)
    for hz in hazards or []:
        if hz["type"] == "ditch" and carve_ditches:
            h -= ditch_drop(grid, hz)
    scn = {
        "schema": SCHEMA, "seed": int(seed), "family": family, "split": "dev", "sha256": "",
        "world_frame": "x east, y north, z up (metres)",
        "terrain": {"origin_xy": [float(origin[0]), float(origin[1])], "res_m": res, "shape": [ny, nx],
                    "height_b64": encode_array_b64(h, "float32"), "material_b64": encode_array_b64(mat, "uint8")},
        "objects": list(objects or []), "hazards": list(hazards or []),
        "start": {"xy": [float(start[0]), float(start[1])], "yaw": float(start[2])}, "goal": {"xy": [float(goal[0]), float(goal[1])]},
        "mission": {"success_radius_m": success_radius_m, "timeout_s": timeout_s, "heading_init_err_deg": heading_init_err_deg},
        "lighting": lighting or {"sun_elev_deg": 45.0, "sun_azim_deg": 0.0, "fog_density": 0.0, "exposure": 1.0, "events": []},
        "dynamic": list(dynamic or []),
        "vehicle": {"chi_by_material": {str(k): chi for k in range(6)}, "slip_long": slip},
    }
    scn["sha256"] = scenario_sha256(scn)
    return scn
