"""Deterministic scenario generator (schema ``metagross.scenario/1``, see
:mod:`metagross.contracts.scenario`).

``make_scenario(seed, split)`` builds a ~64 x 40 m outdoor world at 0.05 m terrain resolution:

* fBm hills (0.3-1.2 m relief), material-dependent micro-roughness (cm scale), a winding gravel
  trail from A to B, grass / dirt / rocky-ground patches;
* family-specific features (``family = FAMILIES[seed % 6]``):

  - ``F1_trail``            extra rocks (0.2-1.0 m) on / beside the trail and trees along it;
  - ``F2_ditch_field``      1-3 trenches (0.3-1.2 m wide, 0.4-1.0 m deep, steep walls) crossing the
                            straight A->B line, spanning the world, each with ONE 1.5-3 m bypass gap;
  - ``F3_crest_ditch``      a crest across the route; for even ``seed // 6`` a trench sits on the
                            back slope just behind it (hidden until the crest), otherwise the crest
                            is a safe control (crest / drop only);
  - ``F4_sudden_obstacle``  a walker / box / boulder whose path crosses the route, triggered when the
                            vehicle is within 4-6 m, starting behind an occluding bush;
  - ``F5_lighting``         low sun ahead + glare, dimming (gain 0.4 for 3 s) and dust events;
  - ``F6_water_mud``        flat water / mud patches on the route with a dry way around;

* generic rocks / trees / bushes / logs at modest density everywhere.

Solvability is guaranteed: after building a candidate, grid A* on the GT hazard raster
(:mod:`metagross.sim.gridplan`) must find a corridor >= 1.2 m wide from A to B, otherwise the
candidate is discarded and regenerated from the next attempt sub-seed. Everything is a pure
function of ``seed`` (NumPy ``SeedSequence([seed, attempt])``).

Conventions: world x east, y north, z up (m). ``sun_azim_deg`` is measured counter-clockwise from
world +x (east) in the xy plane; ``sun_elev_deg`` above the horizon. ``heading_init_err_deg`` is a
counter-clockwise rotation applied to the A-frame goal vector handed to the autonomy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from metagross.contracts.scenario import DEV_SEEDS, EVAL_SEEDS, FAMILIES, MATERIALS, SCHEMA
from metagross.sim.geometry import GridSpec, fill_polygon, polyline_band, rot2
from metagross.sim.gridplan import CORRIDOR_WIDTH_M, plan_path
from metagross.sim.hazards import ditch_drop, gt_hazard_raster
from metagross.sim.terrain import Terrain, encode_array_b64

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- world layout
WORLD_SIZE_M = (64.0, 40.0)  # x (east) extent, y (north) extent
TERRAIN_RES_M = 0.05
START_GOAL_DIST_M = (40.0, 55.0)
EDGE_MARGIN_M = 4.0  # A and B keep this far from the world edge
START_CLEAR_M = 3.0  # no objects / hazards within this radius of A
GOAL_CLEAR_M = 2.5  # ... and of B
MAX_ATTEMPTS = 30
SUCCESS_RADIUS_M = 2.0
MIN_MEAN_SPEED_FOR_TIMEOUT = 0.4  # timeout_s = 60 s + straight distance / this (m/s)
HEADING_INIT_ERR_SIGMA_DEG = 0.5  # operator launch-heading alignment error, clipped at 2 sigma
HILL_RELIEF_M = (0.3, 1.2)
HILL_WAVELENGTH_M = (6.0, 64.0)
# Micro-roughness std (m) per material id: grass, dirt, gravel trail, rocky, mud, water.
MICRO_ROUGHNESS_M = {0: 0.012, 1: 0.010, 2: 0.005, 3: 0.025, 4: 0.004, 5: 0.0}
TRAIL_WIDTH_M = (1.6, 2.4)
TRAIL_DEPTH_M = 0.02  # worn trail sits slightly below the grass
# Families
F2_N_TRENCH = (1, 3)
F2_WIDTH_M = (0.3, 1.2)
F2_DEPTH_M = (0.4, 1.0)
GAP_WIDTH_M = (1.5, 3.0)
GAP_OFFSET_M = (3.0, 12.0)  # gap centre distance from the A->B crossing, along the trench
F3_CREST_HEIGHT_M = (0.4, 0.8)
F3_EXTRA_DROP_M = (0.2, 0.5)
F3_BACK_SLOPE_DEG = (12.0, 15.0)  # max slope of the back side (below SLOPE_LETHAL_DEG: crest-only is safe)
F3_TRENCH_BEHIND_M = (1.2, 2.5)
F4_TRIGGER_M = (4.0, 6.0)
F6_PATCH_RADIUS_M = (1.5, 3.2)
WATER_FEATHER_M = 1.5  # blend distance from flat water level back to the terrain


# ----------------------------------------------------------------------------- helpers
def _r(x: float, nd: int = 4) -> float:
    """Round for storage (0.1 mm); the stored value is what every consumer uses."""
    return float(round(float(x), nd))


def _rl(p, nd: int = 4) -> list:
    return [_r(v, nd) for v in p]


def family_of(seed: int) -> str:
    return FAMILIES[seed % len(FAMILIES)]


def split_of(seed: int) -> str:
    if seed in EVAL_SEEDS:
        return "eval"
    if seed in DEV_SEEDS:
        return "dev"
    return "dev"


def f3_has_trench(seed: int) -> bool:
    """F3 seeds alternate trench / crest-only control by ``seed // 6`` parity."""
    return (seed // len(FAMILIES)) % 2 == 0


def canonical_json(scn: dict) -> str:
    return json.dumps(scn, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def scenario_sha256(scn: dict) -> str:
    """sha256 hex digest of the canonical JSON of ``scn`` with ``sha256`` set to ""."""
    tmp = dict(scn)
    tmp["sha256"] = ""
    return hashlib.sha256(canonical_json(tmp).encode("ascii")).hexdigest()


def write_scenario(scn: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(scn), encoding="ascii")
    return path


def load_scenario(path: str | Path, verify: bool = True) -> dict:
    scn = json.loads(Path(path).read_text(encoding="ascii"))
    if scn.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unexpected schema {scn.get('schema')!r}")
    if verify and scenario_sha256(scn) != scn["sha256"]:
        raise ValueError(f"{path}: sha256 mismatch (file modified?)")
    return scn


def _fbm(rng: np.random.Generator, shape: tuple[int, int], res: float, lam_min: float, lam_max: float, beta: float) -> np.ndarray:
    """Band-limited fractional-Brownian surface by spectral synthesis, normalised to [0, 1]."""
    ny, nx = shape
    ky = np.fft.fftfreq(ny, d=res)[:, None]
    kx = np.fft.rfftfreq(nx, d=res)[None, :]
    k = np.sqrt(kx * kx + ky * ky)
    k_lo, k_hi = 1.0 / lam_max, 1.0 / lam_min
    with np.errstate(divide="ignore"):
        amp = np.where(k > 0, k ** (-beta / 2.0), 0.0)
    amp *= (k >= k_lo) * np.exp(-((np.maximum(k - k_hi, 0.0) / (0.25 * k_hi)) ** 2))
    spec = (rng.standard_normal(amp.shape) + 1j * rng.standard_normal(amp.shape)) * amp
    f = np.fft.irfft2(spec, s=shape)
    f -= f.min()
    return f / max(float(f.max()), 1e-12)


def _line_across_world(p: np.ndarray, direction: np.ndarray, normal: np.ndarray, grid: GridSpec,
                       wiggle_amp: float = 0.0, wiggle_lam: float = 20.0, wiggle_phase: float = 0.0,
                       step: float = 1.0, margin: float = 2.0) -> tuple[np.ndarray, float]:
    """Polyline through ``p`` along ``direction`` clipped to the world (+margin).

    Returns (polyline (N,2), arc length of the point closest to ``p``)."""
    s = np.arange(-80.0, 80.0 + 1e-9, step)
    off = wiggle_amp * np.sin(2 * math.pi * s / wiggle_lam + wiggle_phase)
    pts = p[None, :] + s[:, None] * direction[None, :] + off[:, None] * normal[None, :]
    inside = grid.contains(pts[:, 0], pts[:, 1], margin=-margin)
    idx = np.flatnonzero(inside)
    lo, hi = idx.min(), idx.max()
    pts = pts[lo:hi + 1]
    s_cross = float(np.linalg.norm(np.diff(pts[: (int(np.argmin(np.abs(s[lo:hi + 1]))) + 1)], axis=0), axis=1).sum())
    return pts, s_cross


@dataclass
class _Layout:
    grid: GridSpec
    A: np.ndarray
    B: np.ndarray
    yaw: float
    D: float
    e: np.ndarray  # unit A->B
    n: np.ndarray  # unit left normal
    keepout: list[tuple[float, float, float]] = field(default_factory=list)  # (x, y, r) no generic objects

    def at(self, frac: float, lateral: float = 0.0) -> np.ndarray:
        return self.A + frac * self.D * self.e + lateral * self.n


# ----------------------------------------------------------------------------- generator
class _Builder:
    """One generation attempt. All randomness flows from ``rng``."""

    def __init__(self, seed: int, split: str, attempt: int):
        self.seed = seed
        self.split = split
        self.family = family_of(seed)
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(attempt), 0x4D47]))
        nx = int(round(WORLD_SIZE_M[0] / TERRAIN_RES_M))
        ny = int(round(WORLD_SIZE_M[1] / TERRAIN_RES_M))
        self.grid = GridSpec(0.0, 0.0, TERRAIN_RES_M, ny, nx)
        self.objects: list[dict] = []
        self.hazards: list[dict] = []
        self.dynamic: list[dict] = []
        self._occupied: list[tuple[float, float, float]] = []  # (x, y, r) of placed objects

    # -------------------------------------------------------------- layout
    def layout(self) -> _Layout:
        rng = self.rng
        xmin, xmax, ymin, ymax = self.grid.extent
        for _ in range(1000):
            A = np.array([xmin + EDGE_MARGIN_M + rng.uniform(1.0, 3.0), rng.uniform(ymin + 12.0, ymax - 12.0)])
            D = rng.uniform(*START_GOAL_DIST_M)
            psi = rng.uniform(-0.35, 0.35)
            B = A + D * np.array([math.cos(psi), math.sin(psi)])
            if self.grid.contains(B[0], B[1], margin=EDGE_MARGIN_M):
                break
        else:  # pragma: no cover - the ranges above always admit a solution
            raise RuntimeError("layout sampling failed")
        e = (B - A) / D
        n = np.array([-e[1], e[0]])
        yaw = math.atan2(e[1], e[0]) + rng.uniform(-math.radians(10), math.radians(10))
        lay = _Layout(self.grid, A, B, yaw, float(D), e, n)
        lay.keepout += [(A[0], A[1], START_CLEAR_M), (B[0], B[1], GOAL_CLEAR_M)]
        return lay

    # -------------------------------------------------------------- terrain
    def base_terrain(self, lay: _Layout) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Landform heights (float64), materials (uint8), trail distance field, trail width."""
        rng, g = self.rng, self.grid
        relief = rng.uniform(*HILL_RELIEF_M)
        h = _fbm(rng, g.shape, g.res, HILL_WAVELENGTH_M[0], HILL_WAVELENGTH_M[1], beta=3.2) * relief
        mat = np.zeros(g.shape, np.uint8)  # grass
        patches = _fbm(rng, g.shape, g.res, 2.0, 20.0, beta=2.5)
        mat[patches > rng.uniform(0.62, 0.70)] = 1  # dirt
        rocky = _fbm(rng, g.shape, g.res, 2.0, 16.0, beta=2.5)
        mat[rocky > rng.uniform(0.72, 0.80)] = 3  # rocky ground
        # Winding trail A -> B.
        xi = np.arange(0.0, lay.D + 1e-9, 0.5)
        amp = rng.uniform(1.0, 4.0) * rng.choice([-1.0, 1.0])
        periods = rng.choice([1.0, 1.5, 2.0])
        phase = rng.uniform(0, 2 * math.pi)
        lat = amp * np.sin(math.pi * xi / lay.D) * np.sin(2 * math.pi * periods * xi / lay.D + phase)
        trail = lay.A[None, :] + xi[:, None] * lay.e[None, :] + lat[:, None] * lay.n[None, :]
        self.trail = trail
        tw = rng.uniform(*TRAIL_WIDTH_M)
        tdist, _ = polyline_band(g, trail, 0.5 * tw + 0.5)
        on_trail = tdist <= 0.5 * tw
        mat[on_trail] = 2
        edge = np.clip((0.5 * tw + 0.3 - tdist) / 0.3, 0.0, 1.0)  # worn-trail depression with soft edge
        h -= TRAIL_DEPTH_M * np.where(np.isfinite(tdist), edge, 0.0)
        return h, mat, tdist, tw

    def micro_roughness(self, mat: np.ndarray) -> np.ndarray:
        g = self.grid
        noise = gaussian_filter(self.rng.standard_normal(g.shape), 1.0)
        noise /= max(float(noise.std()), 1e-9)
        sigma = np.zeros(g.shape, np.float64)
        for mid, s in MICRO_ROUGHNESS_M.items():
            sigma[mat == mid] = s
        sigma = gaussian_filter(sigma, 2.0)  # no seams at material borders
        return noise * sigma

    # -------------------------------------------------------------- family features
    def _trench(self, lay: _Layout, p_cross: np.ndarray, direction: np.ndarray, width: float, depth: float,
                wiggle: bool = True) -> dict | None:
        rng = self.rng
        normal = np.array([-direction[1], direction[0]])
        amp = rng.uniform(0.2, 0.8) if wiggle else 0.0
        poly, s_cross = _line_across_world(p_cross, direction, normal, self.grid, wiggle_amp=amp,
                                           wiggle_lam=rng.uniform(15.0, 30.0), wiggle_phase=rng.uniform(0, 2 * math.pi))
        s_tot = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
        gw = rng.uniform(*GAP_WIDTH_M)
        side = rng.choice([-1.0, 1.0])
        for _ in range(2):
            sc = s_cross + side * rng.uniform(*GAP_OFFSET_M)
            if 3.0 < sc < s_tot - 3.0:
                # Gap centre must lie inside the world with margin.
                k = np.searchsorted(np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]), sc)
                q = poly[min(k, len(poly) - 1)]
                if self.grid.contains(q[0], q[1], margin=3.0):
                    self_gap = (sc - 0.5 * gw, sc + 0.5 * gw)
                    lay.keepout.append((float(q[0]), float(q[1]), 0.5 * gw + 1.5))
                    return {"type": "ditch", "polyline": [_rl(p) for p in poly], "width": _r(width), "depth": _r(depth),
                            "gaps": [[_r(self_gap[0]), _r(self_gap[1])]]}
            side = -side
        return None

    def family_features(self, lay: _Layout, h: np.ndarray) -> np.ndarray:
        rng, fam = self.rng, self.family
        if fam == "F1_trail":
            self._f1_trail_objects(lay)
        elif fam == "F2_ditch_field":
            n_tr = int(rng.integers(F2_N_TRENCH[0], F2_N_TRENCH[1] + 1))
            fracs: list[float] = []
            for _ in range(200):
                if len(fracs) == n_tr:
                    break
                f = rng.uniform(0.25, 0.8)
                if all(abs(f - q) * lay.D >= 8.0 for q in fracs):
                    fracs.append(f)
            base = rng.uniform(-math.radians(25), math.radians(25))
            for f in sorted(fracs):
                ang = base + rng.uniform(-math.radians(5), math.radians(5))
                direction = rot2(ang) @ lay.n
                hz = self._trench(lay, lay.at(f), direction, rng.uniform(*F2_WIDTH_M), rng.uniform(*F2_DEPTH_M))
                if hz is not None:
                    self.hazards.append(hz)
        elif fam == "F3_crest_ditch":
            h = self._f3_crest(lay, h)
        elif fam == "F4_sudden_obstacle":
            self._f4_dynamic(lay)
        elif fam == "F6_water_mud":
            self._f6_patches(lay)
        return h

    def _f1_trail_objects(self, lay: _Layout) -> None:
        rng = self.rng
        tr = self.trail
        seg = np.diff(tr, axis=0)
        tang = seg / np.linalg.norm(seg, axis=1, keepdims=True)
        for _ in range(int(rng.integers(8, 15))):
            k = int(rng.integers(10, len(tr) - 8))
            nrm = np.array([-tang[k, 1], tang[k, 0]])
            r = 0.2 + 0.8 * rng.uniform() ** 2
            p = tr[k] + rng.uniform(-3.0, 3.0) * nrm
            self._place("rock", p, r, allow_trail=True)
        for _ in range(int(rng.integers(6, 13))):
            k = int(rng.integers(6, len(tr) - 4))
            nrm = np.array([-tang[k, 1], tang[k, 0]])
            p = tr[k] + rng.choice([-1.0, 1.0]) * rng.uniform(1.8, 4.5) * nrm
            self._place("tree", p, None)

    def _f3_crest(self, lay: _Layout, h: np.ndarray) -> np.ndarray:
        rng, g = self.rng, self.grid
        xi_c = rng.uniform(0.4, 0.6) * lay.D
        Hc = rng.uniform(*F3_CREST_HEIGHT_M)
        L_up = rng.uniform(5.0, 7.0)
        drop = Hc + rng.uniform(*F3_EXTRA_DROP_M)
        back = math.radians(rng.uniform(*F3_BACK_SLOPE_DEG))
        L_down = 1.5 * drop / math.tan(back)  # smoothstep max slope = 1.5 * drop / L
        xs, ys = g.cell_centres()
        xi = (xs - lay.A[0]) * lay.e[0] + (ys - lay.A[1]) * lay.e[1]
        up = np.clip((xi - (xi_c - L_up)) / L_up, 0.0, 1.0)
        dn = np.clip((xi - xi_c) / L_down, 0.0, 1.0)
        prof = Hc * up * up * (3 - 2 * up) - drop * dn * dn * (3 - 2 * dn)
        h = h + prof
        p_c = lay.at(xi_c / lay.D)
        crest_line, _ = _line_across_world(p_c, lay.n, lay.e, g)
        self.hazards.append({"type": "crest", "polyline": [_rl(p) for p in crest_line], "drop": _r(drop)})
        lay.keepout.append((float(p_c[0]), float(p_c[1]), 1.0))
        if f3_has_trench(self.seed):
            delta = min(rng.uniform(*F3_TRENCH_BEHIND_M), 0.6 * L_down)
            hz = self._trench(lay, p_c + delta * lay.e, lay.n, rng.uniform(0.4, 1.0), rng.uniform(0.4, 0.8), wiggle=False)
            if hz is not None:
                self.hazards.append(hz)
        return h

    def _f4_dynamic(self, lay: _Layout) -> None:
        rng = self.rng
        typ = str(rng.choice(["walker", "box", "boulder"]))
        f = rng.uniform(0.35, 0.7)
        P = lay.at(f)
        side = rng.choice([-1.0, 1.0])
        mdir = rot2(rng.uniform(-math.radians(15), math.radians(15))) @ (-side * lay.n)  # towards/over the route
        off = rng.uniform(3.0, 5.0)
        p0 = P - off * mdir
        if typ == "walker":
            size, speed, p1 = [0.5, 0.5, 1.7], rng.uniform(1.0, 1.5), P + rng.uniform(2.5, 4.0) * mdir
        elif typ == "box":
            s = rng.uniform(0.5, 0.8)
            size, speed, p1 = [s, s, rng.uniform(0.4, 0.7)], rng.uniform(0.8, 1.2), P + rng.uniform(-0.3, 0.3) * mdir
        else:
            s = rng.uniform(0.6, 0.9)
            size, speed, p1 = [s, s, rng.uniform(0.5, 0.8)], rng.uniform(1.2, 2.0), P + rng.uniform(-0.3, 0.3) * mdir
        self.dynamic.append({"type": typ, "size": _rl(size), "trigger_dist_m": _r(rng.uniform(*F4_TRIGGER_M)),
                             "path": [_rl(p0), _rl(p1)], "speed": _r(speed)})
        # Occluding bush between the start point and the approaching vehicle (hides the object).
        q = P + 0.75 * off * (-mdir) - 1.5 * lay.e
        self._place("bush", q, rng.uniform(0.55, 0.8), force=True, height=rng.uniform(0.9, 1.3))
        for pt, r in ((p0, 1.0), (p1, 1.0)):
            lay.keepout.append((float(pt[0]), float(pt[1]), r))
        # Keep the swept corridor free of generic clutter.
        for s in np.linspace(0.0, 1.0, 6):
            pt = p0 + s * (p1 - p0)
            lay.keepout.append((float(pt[0]), float(pt[1]), 1.2))

    def _f6_patches(self, lay: _Layout) -> None:
        rng = self.rng
        n = int(rng.integers(1, 4))
        fracs: list[float] = []
        for _ in range(200):
            if len(fracs) == n:
                break
            f = rng.uniform(0.25, 0.8)
            if all(abs(f - q) * lay.D >= 8.0 for q in fracs):
                fracs.append(f)
        kinds = ["water"] + [str(rng.choice(["water", "mud"])) for _ in range(len(fracs) - 1)]
        rng.shuffle(kinds)
        self._patches: list[tuple[np.ndarray, int]] = []
        for f, kind in zip(sorted(fracs), kinds):
            c = lay.at(f, rng.uniform(-1.0, 1.0))
            R = rng.uniform(*F6_PATCH_RADIUS_M)
            ang = np.linspace(0, 2 * math.pi, 16, endpoint=False) + rng.uniform(0, 2 * math.pi)
            rad = R * (1.0 + 0.2 * np.convolve(np.r_[rng.standard_normal(16), rng.standard_normal(2)], np.ones(3) / 3, "valid"))
            poly = c[None, :] + np.stack([np.cos(ang), np.sin(ang)], 1) * np.clip(rad, 0.6 * R, 1.3 * R)[:, None]
            poly_l = [_rl(p) for p in poly]
            self.hazards.append({"type": "water", "polygon": poly_l})
            self._patches.append((np.asarray(poly_l), 5 if kind == "water" else 4))
            lay.keepout.append((float(c[0]), float(c[1]), 1.3 * R + 0.5))

    def apply_water(self, h: np.ndarray, mat: np.ndarray) -> np.ndarray:
        """Flatten every water / mud patch to a level surface and paint its material."""
        g = self.grid
        for poly, mid in getattr(self, "_patches", []):
            m = fill_polygon(g, poly).astype(bool)
            ii, jj = np.nonzero(m)
            pad = int(math.ceil(WATER_FEATHER_M / g.res)) + 2
            i0, i1 = max(ii.min() - pad, 0), min(ii.max() + pad + 1, g.ny)
            j0, j1 = max(jj.min() - pad, 0), min(jj.max() + pad + 1, g.nx)
            sub_m = m[i0:i1, j0:j1]
            level = float(np.median(h[i0:i1, j0:j1][sub_m])) - 0.03
            dist = distance_transform_edt(~sub_m) * g.res
            w = np.clip(dist / WATER_FEATHER_M, 0.0, 1.0)
            w = w * w * (3 - 2 * w)
            h[i0:i1, j0:j1] = level * (1 - w) + h[i0:i1, j0:j1] * w
            mat[i0:i1, j0:j1][sub_m] = mid
        return h

    # -------------------------------------------------------------- objects
    def _place(self, typ: str, p: np.ndarray, r: float | None, allow_trail: bool = False, force: bool = False,
               height: float | None = None) -> bool:
        rng = self.rng
        x, y = float(p[0]), float(p[1])
        if typ == "rock":
            rr = float(r)
            spec = {"type": "rock", "radius": rr, "squash": rng.uniform(0.45, 0.8), "seed": int(rng.integers(0, 2**31 - 1))}
        elif typ == "tree":
            cr = rng.uniform(1.0, 2.2)
            rr = rng.uniform(0.10, 0.25)
            spec = {"type": "tree", "trunk_r": rr, "height": 2 * cr + rng.uniform(1.6, 3.0), "canopy_r": cr,
                    "seed": int(rng.integers(0, 2**31 - 1))}
        elif typ == "bush":
            rr = float(r) if r is not None else rng.uniform(0.3, 0.8)
            spec = {"type": "bush", "radius": rr, "height": height if height is not None else rng.uniform(0.4, 1.2),
                    "seed": int(rng.integers(0, 2**31 - 1))}
        elif typ == "log":
            L = rng.uniform(1.0, 3.0)
            rr = 0.5 * L
            spec = {"type": "log", "yaw": rng.uniform(-math.pi, math.pi), "length": L, "radius": rng.uniform(0.10, 0.22)}
        else:
            raise ValueError(typ)
        if not force:
            if not self.grid.contains(x, y, margin=1.0 + rr):
                return False
            for kx, ky, kr in self._keepout:
                if math.hypot(x - kx, y - ky) < kr + rr:
                    return False
            for ox, oy, orr in self._occupied:
                if math.hypot(x - ox, y - oy) < orr + rr + 0.8:
                    return False
            if not allow_trail and typ != "tree":
                i, j = self.grid.xy_to_ij(x, y)
                if self._trail_dist[i, j] < 0.5 * self._trail_w + rr:
                    return False
        self._occupied.append((x, y, rr))
        spec["_xy"] = (x, y)
        self.objects.append(spec)
        return True

    def generic_objects(self) -> None:
        rng = self.rng
        xmin, xmax, ymin, ymax = self.grid.extent
        counts = {"rock": rng.integers(12, 23), "tree": rng.integers(6, 15), "bush": rng.integers(6, 15), "log": rng.integers(1, 5)}
        for typ, n in counts.items():
            placed, tries = 0, 0
            while placed < n and tries < 60 * n:
                tries += 1
                p = np.array([rng.uniform(xmin + 1, xmax - 1), rng.uniform(ymin + 1, ymax - 1)])
                r = 0.1 + 0.35 * rng.uniform() ** 2 if typ == "rock" else None
                placed += self._place(typ, p, r)

    def finalize_objects(self, terrain: Terrain, near_hazard: np.ndarray) -> list[dict]:
        """Drop objects that ended up in ditches / water, resolve z, round everything."""
        out: list[dict] = []
        foot_r = {"rock": lambda s: s["radius"], "tree": lambda s: s["trunk_r"], "bush": lambda s: s["radius"],
                  "log": lambda s: 0.5 * s["length"]}
        for spec in self.objects:
            x, y = spec.pop("_xy")
            i, j = self.grid.xy_to_ij(x, y)
            if near_hazard[i, j] < foot_r[spec["type"]](spec) + 0.2:
                continue  # never leave an object standing in a ditch or a water patch
            g = float(terrain.height_at(x, y))
            if spec["type"] == "rock":
                r, sq = spec["radius"], spec["squash"]
                spec = {"type": "rock", "xyz": _rl([x, y, g - 0.25 * r * sq]), "radius": _r(r), "squash": _r(sq), "seed": spec["seed"]}
            elif spec["type"] == "tree":
                spec = {"type": "tree", "xy": _rl([x, y]), "trunk_r": _r(spec["trunk_r"]), "height": _r(spec["height"]),
                        "canopy_r": _r(spec["canopy_r"]), "seed": spec["seed"]}
            elif spec["type"] == "bush":
                spec = {"type": "bush", "xy": _rl([x, y]), "radius": _r(spec["radius"]), "height": _r(spec["height"]), "seed": spec["seed"]}
            else:
                spec = {"type": "log", "xy": _rl([x, y]), "yaw": _r(spec["yaw"]), "length": _r(spec["length"]), "radius": _r(spec["radius"])}
            out.append(spec)
        return out

    # -------------------------------------------------------------- lighting / vehicle / mission
    def lighting(self, lay: _Layout) -> dict:
        rng = self.rng
        if self.family == "F5_lighting":
            bearing = math.degrees(math.atan2(lay.e[1], lay.e[0]))
            t_glare = rng.uniform(5.0, 20.0)
            t_dim = rng.uniform(15.0, 40.0)
            t_dust = rng.uniform(25.0, 60.0)
            events = [
                {"type": "glare", "t0": _r(t_glare), "t1": _r(t_glare + rng.uniform(4.0, 8.0)), "gain": _r(rng.uniform(1.5, 3.0))},
                {"type": "dim", "t0": _r(t_dim), "t1": _r(t_dim + 3.0), "gain": 0.4},
                {"type": "dust", "t0": _r(t_dust), "t1": _r(t_dust + rng.uniform(4.0, 8.0)), "gain": _r(rng.uniform(0.05, 0.2))},
            ]
            return {"sun_elev_deg": _r(rng.uniform(3.0, 8.0)), "sun_azim_deg": _r((bearing + rng.uniform(-10, 10)) % 360.0),
                    "fog_density": _r(rng.uniform(0.005, 0.02)), "exposure": 1.0, "events": sorted(events, key=lambda ev: ev["t0"])}
        return {"sun_elev_deg": _r(rng.uniform(25.0, 65.0)), "sun_azim_deg": _r(rng.uniform(0.0, 360.0)),
                "fog_density": _r(rng.uniform(0.0, 0.005)), "exposure": 1.0, "events": []}

    def vehicle(self) -> dict:
        rng = self.rng
        chi_rng = {0: (1.45, 1.60), 1: (1.40, 1.55), 2: (1.30, 1.40), 3: (1.35, 1.50), 4: (1.55, 1.70), 5: (1.60, 1.70)}
        return {"chi_by_material": {str(k): _r(rng.uniform(*chi_rng[k])) for k in sorted(MATERIALS)},
                "slip_long": _r(rng.uniform(0.03, 0.08))}

    # -------------------------------------------------------------- build
    def build(self) -> dict:
        lay = self.layout()
        self._keepout = lay.keepout
        h, mat, tdist, tw = self.base_terrain(lay)
        self._trail_dist, self._trail_w = tdist, tw
        h = self.family_features(lay, h)
        h = h + self.micro_roughness(mat)
        h = self.apply_water(h, mat)
        drop = np.zeros(self.grid.shape, np.float32)
        for hz in self.hazards:
            if hz["type"] == "ditch":
                drop = np.maximum(drop, ditch_drop(self.grid, hz))
        h = h - drop
        mat[drop > 0.05] = 1  # excavated soil
        self.generic_objects()
        h32 = h.astype(np.float32)
        terrain = Terrain(self.grid, h32, mat)
        near_hazard = distance_transform_edt(~((drop > 0.05) | np.isin(mat, (4, 5)))) * self.grid.res
        objects = self.finalize_objects(terrain, near_hazard)
        lighting = self.lighting(lay)
        vehicle = self.vehicle()
        err = float(np.clip(self.rng.normal(0.0, HEADING_INIT_ERR_SIGMA_DEG), -2 * HEADING_INIT_ERR_SIGMA_DEG, 2 * HEADING_INIT_ERR_SIGMA_DEG))
        scn = {
            "schema": SCHEMA,
            "seed": int(self.seed),
            "family": self.family,
            "split": self.split,
            "sha256": "",
            "world_frame": "x east, y north, z up (metres)",
            "terrain": {"origin_xy": [self.grid.origin_x, self.grid.origin_y], "res_m": self.grid.res,
                        "shape": [self.grid.ny, self.grid.nx], "height_b64": encode_array_b64(h32, "float32"),
                        "material_b64": encode_array_b64(mat, "uint8")},
            "objects": objects,
            "hazards": self.hazards,
            "start": {"xy": _rl(lay.A), "yaw": _r(lay.yaw)},
            "goal": {"xy": _rl(lay.B)},
            "mission": {"success_radius_m": SUCCESS_RADIUS_M, "timeout_s": _r(60.0 + lay.D / MIN_MEAN_SPEED_FOR_TIMEOUT, 1),
                        "heading_init_err_deg": _r(err, 3)},
            "lighting": lighting,
            "dynamic": self.dynamic,
            "vehicle": vehicle,
        }
        return scn


def check_solvable(scn: dict, terrain: Terrain | None = None) -> tuple[bool, float]:
    """True if a >= CORRIDOR_WIDTH_M corridor links start and goal on the GT hazard raster.

    Returns (ok, shortcut path length in m)."""
    hz = gt_hazard_raster(scn, terrain)
    res = plan_path(hz.grid, hz.hazard, tuple(scn["start"]["xy"]), tuple(scn["goal"]["xy"]), CORRIDOR_WIDTH_M)
    return res.ok, res.length_m


def make_scenario(seed: int, split: str | None = None) -> dict:
    """Generate the scenario for ``seed`` (deterministic; family = FAMILIES[seed % 6]).

    ``split`` defaults to 'eval' for EVAL_SEEDS and 'dev' otherwise. Raises RuntimeError if no
    solvable candidate is found within MAX_ATTEMPTS (never observed for the 90 published seeds).
    """
    split = split or split_of(seed)
    if split not in ("dev", "eval"):
        raise ValueError(f"split must be 'dev' or 'eval', got {split!r}")
    for attempt in range(MAX_ATTEMPTS):
        scn = _Builder(seed, split, attempt).build()
        ok, length = check_solvable(scn)
        if ok:
            scn["sha256"] = scenario_sha256(scn)
            log.info("seed %d (%s) solvable on attempt %d, A* corridor path %.1f m", seed, scn["family"], attempt, length)
            return scn
        log.info("seed %d attempt %d not solvable, regenerating", seed, attempt)
    raise RuntimeError(f"seed {seed}: no solvable scenario in {MAX_ATTEMPTS} attempts")


def scenario_seeds(split: str) -> tuple[int, ...]:
    """Pre-registered seeds of a split ('eval' = 0-59, 'dev' = 100-129)."""
    return {"eval": EVAL_SEEDS, "dev": DEV_SEEDS}[split]
