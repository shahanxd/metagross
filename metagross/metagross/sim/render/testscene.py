"""Self-contained synthetic scenarios for renderer tests and the GO/NO-GO bench.

These follow the ``metagross.scenario/1`` schema (contracts/scenario.py) but are
NOT the official scenario families (those come from ``metagross.sim.scenario``);
they exist so the renderer can be validated without depending on other modules.

World frame: x east, y north, z up (metres). Heightmap rows run along +y, columns
along +x, cell (0, 0) centre at ``origin_xy``.
"""

from __future__ import annotations

import base64
import math

import numpy as np

from metagross.contracts.scenario import SCHEMA

RES_M = 0.05  # heightmap resolution fixed by the schema (m)

GRASS, DIRT, GRAVEL, ROCKY, MUD, WATER = 0, 1, 2, 3, 4, 5


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], cell: int) -> np.ndarray:
    """Band-limited noise: random grid every `cell` samples, bicubic-ish upsampled (via separable cosine interp)."""
    ny, nx = shape
    gy, gx = ny // cell + 3, nx // cell + 3
    g = rng.standard_normal((gy, gx))
    ys = np.arange(ny) / cell
    xs = np.arange(nx) / cell
    iy, fy = np.floor(ys).astype(int), ys - np.floor(ys)
    ix, fx = np.floor(xs).astype(int), xs - np.floor(xs)
    wy = (1 - np.cos(np.pi * fy)) / 2
    wx = (1 - np.cos(np.pi * fx)) / 2
    a = g[iy][:, ix] * (1 - wx) + g[iy][:, ix + 1] * wx
    b = g[iy + 1][:, ix] * (1 - wx) + g[iy + 1][:, ix + 1] * wx
    return a * (1 - wy[:, None]) + b * wy[:, None]


def make_test_scenario(seed: int = 100, flat: bool = False, extent_m: tuple[float, float] = (50.0, 30.0)) -> dict:
    """Trail scene with a ditch (with a bypass gap), pond, mud, rocks, trees, bushes,
    logs and one walker. ``flat=True`` gives a featureless-geometry flat gravel plane
    (used by the stereo-baseline test)."""
    rng = np.random.default_rng(seed)
    x0, y0 = -5.0, -extent_m[1] / 2
    nx, ny = int(round(extent_m[0] / RES_M)) + 1, int(round(extent_m[1] / RES_M)) + 1
    xs = x0 + np.arange(nx) * RES_M
    ys = y0 + np.arange(ny) * RES_M
    X, Y = np.meshgrid(xs, ys)
    objects: list[dict] = []
    hazards: list[dict] = []
    dynamic: list[dict] = []
    if flat:
        h = np.zeros((ny, nx), np.float32)
        mat = np.full((ny, nx), GRAVEL, np.uint8)
        mat[np.abs(Y) > 3.0] = GRASS
        mat[(np.abs(Y) > 1.5) & (np.abs(Y) <= 3.0)] = DIRT
    else:
        h = 0.35 * _smooth_noise(rng, (ny, nx), 160) + 0.06 * _smooth_noise(rng, (ny, nx), 30) + 0.012 * _smooth_noise(rng, (ny, nx), 5)
        h += 0.012 * X  # gentle up-slope along the route
        trail_c = 1.8 * np.sin(X / 9.0)
        d_trail = np.abs(Y - trail_c)
        h -= 0.05 * np.exp(-(d_trail / 0.9) ** 2)  # trail is slightly worn in
        mat = np.full((ny, nx), GRASS, np.uint8)
        dirt_mask = _smooth_noise(rng, (ny, nx), 60) > 0.7
        mat[dirt_mask] = DIRT
        mat[d_trail < 1.6] = DIRT
        mat[d_trail < 1.0] = GRAVEL
        rocky_mask = (_smooth_noise(rng, (ny, nx), 80) > 1.1) & (d_trail > 2.5)
        mat[rocky_mask] = ROCKY
        # pond (flat water surface) + mud rim
        pc, pr = np.array([12.0, -7.5]), 2.6
        dp = np.hypot(X - pc[0], Y - pc[1])
        surf = float(np.median(h[(dp > pr) & (dp < pr + 0.3)])) - 0.08
        h = np.where(dp < pr + 0.6, np.minimum(h, surf + 0.08 * np.clip((dp - pr) / 0.6, 0, 1)), h)
        mat[dp < pr + 0.7] = MUD
        mat[dp < pr] = WATER
        h[dp < pr] = surf
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        hazards.append({"type": "water", "polygon": [[float(pc[0] + pr * math.cos(a)), float(pc[1] + pr * math.sin(a))] for a in ang]})
        # ditch across the route at x = 24 with a bypass gap at y in [5.5, 8.0]
        xd, wd, dd = 24.0, 0.8, 0.6
        gap = (5.5, 8.0)
        in_gap = (Y > gap[0]) & (Y < gap[1])
        dx = np.abs(X - xd)
        wall = 0.08  # wall horizontal run (m): ~82 deg walls
        prof = np.clip((wd / 2 + wall - dx) / wall, 0, 1)
        h = np.where(in_gap, h, h - dd * prof)
        mat[(dx < wd / 2 + wall) & ~in_gap] = DIRT
        s0 = gap[0] - y0
        hazards.append({"type": "ditch", "polyline": [[xd, float(y0)], [xd, float(y0 + extent_m[1])]], "width": wd, "depth": dd,
                        "gaps": [[s0, s0 + gap[1] - gap[0]]]})
        h = h.astype(np.float32)

        def hz(x: float, y: float) -> float:
            j = int(round((x - x0) / RES_M)); i = int(round((y - y0) / RES_M))
            return float(h[min(max(i, 0), ny - 1), min(max(j, 0), nx - 1)])

        for k in range(40):  # rocks, biased to trail edges
            x = float(rng.uniform(1, 44)); side = rng.choice([-1, 1])
            y = float(1.8 * math.sin(x / 9.0) + side * rng.uniform(1.3, 9.0))
            r = float(rng.uniform(0.08, 0.45))
            objects.append({"type": "rock", "xyz": [x, y, hz(x, y) + 0.15 * r], "radius": r, "squash": float(rng.uniform(0.5, 0.85)), "seed": int(rng.integers(1 << 30))})
        for k in range(26):
            x = float(rng.uniform(-3, 45)); side = rng.choice([-1, 1])
            y = float(1.8 * math.sin(x / 9.0) + side * rng.uniform(3.8, 14.0))
            if abs(x - xd) < 2.0:
                continue
            H = float(rng.uniform(4.0, 8.0))
            objects.append({"type": "tree", "xy": [x, y], "trunk_r": float(rng.uniform(0.1, 0.22)), "height": H, "canopy_r": float(H * rng.uniform(0.22, 0.32)), "seed": int(rng.integers(1 << 30))})
        for k in range(30):
            x = float(rng.uniform(-3, 45)); side = rng.choice([-1, 1])
            y = float(1.8 * math.sin(x / 9.0) + side * rng.uniform(2.2, 12.0))
            objects.append({"type": "bush", "xy": [x, y], "radius": float(rng.uniform(0.3, 0.8)), "height": float(rng.uniform(0.4, 1.1)), "seed": int(rng.integers(1 << 30))})
        objects.append({"type": "log", "xy": [9.0, 3.6], "yaw": 0.4, "length": 2.4, "radius": 0.16})
        objects.append({"type": "log", "xy": [30.0, -3.2], "yaw": -0.9, "length": 1.8, "radius": 0.12})
        dynamic.append({"type": "walker", "size": [0.45, 0.35, 1.72], "trigger_dist_m": 6.0, "path": [[16.5, -3.5], [16.5, 4.0]], "speed": 1.2})
    return {
        "schema": SCHEMA,
        "seed": int(seed),
        "family": "F2_ditch_field",
        "split": "dev",
        "sha256": "",
        "world_frame": "x east, y north, z up",
        "terrain": {"origin_xy": [x0, y0], "res_m": RES_M, "shape": [ny, nx], "height_b64": _b64(h.astype("<f4")), "material_b64": _b64(mat)},
        "objects": objects,
        "hazards": hazards,
        "start": {"xy": [0.0, 0.0], "yaw": 0.0},
        "goal": {"xy": [42.0, 0.0]},
        "mission": {"success_radius_m": 2.0, "timeout_s": 120.0, "heading_init_err_deg": 0.0},
        "lighting": {"sun_elev_deg": 38.0, "sun_azim_deg": 140.0, "fog_density": 0.006, "exposure": 1.0, "events": []},
        "dynamic": dynamic,
        "vehicle": {"chi_by_material": {str(k): 1.4 for k in range(6)}, "slip_long": 0.05},
    }


def terrain_height(scenario: dict, x: float, y: float) -> float:
    """Bilinear height of the scenario terrain at world (x, y) (m), clamped to the grid."""
    t = scenario["terrain"]
    ny, nx = t["shape"]
    h = np.frombuffer(base64.b64decode(t["height_b64"]), dtype="<f4").reshape(ny, nx)
    fx = min(max((x - t["origin_xy"][0]) / t["res_m"], 0.0), nx - 1.001)
    fy = min(max((y - t["origin_xy"][1]) / t["res_m"], 0.0), ny - 1.001)
    j, i = int(fx), int(fy)
    u, v = fx - j, fy - i
    return float((h[i, j] * (1 - u) + h[i, j + 1] * u) * (1 - v) + (h[i + 1, j] * (1 - u) + h[i + 1, j + 1] * u) * v)


def route_poses(scenario: dict, n: int, x_start: float = 0.0, x_end: float = 22.0) -> list[list[float]]:
    """Vehicle poses [x,y,z,roll,pitch,yaw] along the synthetic trail centreline."""
    poses = []
    for k in range(n):
        x = x_start + (x_end - x_start) * k / max(n - 1, 1)
        y = 1.8 * math.sin(x / 9.0)
        yaw = math.atan(1.8 / 9.0 * math.cos(x / 9.0))
        poses.append([x, y, terrain_height(scenario, x, y), 0.0, 0.0, yaw])
    return poses
