"""Scenario JSON schema (``metagross.scenario/1``), produced by metagross.sim.scenario
and consumed by the world, the renderer, the referee and the evaluator.

The AUTONOMY NEVER RECEIVES THIS. It gets only MissionSpec + sensor frames.

Top-level keys
--------------
schema        "metagross.scenario/1"
seed          int
family        one of FAMILIES
split         "dev" | "eval"
sha256        hex digest of canonical JSON of the scenario with sha256 set to ""
world_frame   world x east, y north, z up (metres)
terrain       {
                "origin_xy": [x0, y0],          # world coords of cell (0,0) centre
                "res_m": 0.05,
                "shape": [ny, nx],              # rows along +y, cols along +x
                "height_b64": str,              # float32 little-endian, row-major, metres
                "material_b64": str,            # uint8 per cell, MATERIALS ids
              }
objects       list of
                {"type": "rock",  "xyz": [x,y,z], "radius": r, "squash": s, "seed": k}
                {"type": "tree",  "xy": [x,y], "trunk_r": r, "height": h, "canopy_r": c, "seed": k}
                {"type": "bush",  "xy": [x,y], "radius": r, "height": h, "seed": k}
                {"type": "log",   "xy": [x,y], "yaw": a, "length": L, "radius": r}
hazards       list of (for referee / evaluator / plots only)
                {"type": "ditch", "polyline": [[x,y],...], "width": w, "depth": d, "gaps": [[s0,s1],...]}
                {"type": "crest", "polyline": [[x,y],...], "drop": d}
                {"type": "water", "polygon": [[x,y],...]}
start         {"xy": [x,y], "yaw": rad}
goal          {"xy": [x,y]}
mission       {"success_radius_m": 2.0, "timeout_s": float, "heading_init_err_deg": float}
lighting      {"sun_elev_deg": e, "sun_azim_deg": a, "fog_density": f, "exposure": 1.0,
               "events": [{"type": "dim"|"glare"|"dust", "t0": s, "t1": s, "gain": g}]}
dynamic       list of {"type": "walker"|"box"|"boulder", "size": [sx,sy,sz],
                       "trigger_dist_m": d, "path": [[x,y],[x,y]], "speed": v}
vehicle       {"chi_by_material": {material_id: chi}, "slip_long": s}
"""

from __future__ import annotations

SCHEMA = "metagross.scenario/1"

FAMILIES = (
    "F1_trail",  # trail with rocks and trees
    "F2_ditch_field",  # 1-3 trenches with a bypass gap
    "F3_crest_ditch",  # ditch hidden behind a crest + crest-only controls
    "F4_sudden_obstacle",  # obstacle enters path at 4-6 m
    "F5_lighting",  # glare / dimming / dust events
    "F6_water_mud",  # water and mud patches
)

EVAL_SEEDS = tuple(range(0, 60))  # 10 per family, family = FAMILIES[seed % 6]
DEV_SEEDS = tuple(range(100, 130))

# Ground materials (terrain cells)
MATERIALS = {
    0: "grass",
    1: "dirt",
    2: "gravel_trail",
    3: "rocky_ground",
    4: "mud",
    5: "water",
}

# Material / object -> 5-class semantic id (see interfaces.SEM_CLASSES)
MATERIAL_TO_SEM = {0: 3, 1: 3, 2: 4, 3: 3, 4: 2, 5: 2}
OBJECT_SEM = 1  # rocks, trees, bushes, logs, dynamic obstacles
SKY_SEM = 0
