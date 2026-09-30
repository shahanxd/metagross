"""Synthetic **DEMO_FAKE** run directory for building and testing the video toolchain.

Nothing here is a measurement. It fabricates a plausible-looking run so the compositor,
encoder and console can be developed before real closed-loop runs exist; every output is
marked ``demo_fake`` (result.json, mission.json, and the dashboard's honesty chip).

World (A-frame, metres): a vehicle follows a smooth S-curve from A = (0, 0) to B ~ (30, 3).
A 0.7 m wide ditch band crosses the route at x = 14.0..14.7 with a bypass gap at
y = 3.2..5.8; three rocks and a water patch add the other colour-key states. A toy
visibility model (camera FOV, range, rock shadows, ditch lip occlusion) fills a seen-ground
map so the BEV shows the thesis visually: the ditch is an unseen grey band, turns magenta
(missing ground) at <= 9 m and red (confirmed) after 1 s or at <= 6 m. A toy governor
(stopping distance <= R_cert) sets v_cap; the mode goes NOMINAL -> CAUTION -> NOMINAL ->
ARRIVED.

Images come from a flat-ground ray caster: every pixel of a rigidly mounted camera hits the
ground plane at a fixed body-frame point, so a frame is one ``cv2.remap`` of a procedural
ground texture. The same caster provides a 3rd-person chase view (vehicle drawn as a box),
the left image, semantic mask, missing-ground mask and disparity.

Output layout matches :mod:`video.replay` (runner + autonomy-process formats).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.interpolate import CubicSpline

from metagross.autonomy.link import codec
from metagross.config import defaults
from metagross.contracts.messages import CellState as S
from metagross.contracts.messages import DriveMode, Telemetry

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- world constants
ROUTE_PTS = np.array([(0, 0), (6, 0), (10.5, 1.2), (14.3, 4.4), (18.5, 4.8), (24, 3.4), (30, 3.0)], np.float64)
DITCH_X = (14.0, 14.7)  # A-frame x extent of the ditch band, m
DITCH_GAP_Y = (3.2, 5.8)  # bypass gap, m
DITCH_Y = (-14.0, 14.0)
LIP_SHADOW_M = 1.6  # ground hidden behind the far lip until close
LIP_VISIBLE_RANGE_M = 3.5  # far side of the trench (incl. the bypass gap) hidden until this close
DITCH_DETECT_RANGE_M = 9.0  # missing ground classified at <= this range
DITCH_CONFIRM_RANGE_M = 6.0
DITCH_CONFIRM_S = 1.0
ROCKS = ((9.0, -2.4, 0.45), (20.5, 1.6, 0.40), (26.0, 6.2, 0.50))  # x, y, radius
WATER = ((22.0, -4.0), (3.0, 1.8))  # centre, semi-axes
GOAL_OFFSET = (0.4, -0.3)  # operator-entered B vs true route end (heading-init + range error)
GOAL_SIGMA_M = 1.5
LAUNCH_APRON_M = 2.5  # ground ahead of the start pose certified at launch (camera blind zone ~1.6 m)
DISP_QUANT = 1.0 / 16.0  # disparity quantisation, px (SGBM-style fixed point)
START_WORLD = (12.0, -4.0, 0.35)  # world pose of launch point A (exercises the world->A transform)

WORLD_X0, WORLD_Y0 = -10.0, -20.0  # corner of the truth grids (A-frame)
WORLD_W_M, WORLD_H_M = 60.0, 40.0
TEX_RES = 0.04  # texture resolution, m/px
CRUISE_MPS = 1.4
# Toy governor constants (FAKE, chosen so the cap visibly binds in the demo; the real governor
# lives in metagross.autonomy.planning.governor and uses defaults.BRAKE_DECEL_MPS2).
T_R_S = 0.4  # reaction time, s
TOY_DECEL_MPS2 = 0.6  # braking deceleration, m/s^2
SEED = 7
FOG_START_M, FOG_SCALE_M, FOG_MAX = 6.0, 30.0, 0.9  # synthetic haze, m / m / fraction


# ----------------------------------------------------------------------------- route
@dataclass
class Route:
    """Arc-length parametrised reference route (A-frame)."""

    s: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray

    @classmethod
    def build(cls, pts: np.ndarray = ROUTE_PTS, ds: float = 0.05) -> "Route":
        chord = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        cs = CubicSpline(chord, pts, bc_type="natural")
        u = np.linspace(0, chord[-1], int(chord[-1] / ds) * 4)
        p = cs(u)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
        ss = np.arange(0.0, s[-1], ds)
        xy = np.column_stack([np.interp(ss, s, p[:, 0]), np.interp(ss, s, p[:, 1])])
        d = np.gradient(xy, axis=0)
        return cls(ss, xy, np.unwrap(np.arctan2(d[:, 1], d[:, 0])))

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def at(self, s: float) -> tuple[float, float, float]:
        s = float(np.clip(s, 0.0, self.s[-1]))
        return (float(np.interp(s, self.s, self.xy[:, 0])), float(np.interp(s, self.s, self.xy[:, 1])),
                float(np.interp(s, self.s, self.yaw)))

    def segment(self, s0: float, s1: float, step: float = 0.2) -> np.ndarray:
        ss = np.arange(max(s0, 0.0), min(s1, self.length), step)
        return np.column_stack([np.interp(ss, self.s, self.xy[:, 0]), np.interp(ss, self.s, self.xy[:, 1])])


# ----------------------------------------------------------------------------- truth + texture
class GroundScene:
    """Procedural ground truth (class grid) and texture in the A-frame, plus the flat-ground ray caster."""

    def __init__(self, route: Route, seed: int = SEED) -> None:
        self.route = route
        rng = np.random.default_rng(seed)
        nx, ny = int(WORLD_W_M / TEX_RES), int(WORLD_H_M / TEX_RES)
        xs = WORLD_X0 + (np.arange(nx) + 0.5) * TEX_RES
        ys = WORLD_Y0 + (np.arange(ny) + 0.5) * TEX_RES
        X, Y = np.meshgrid(xs, ys)  # rows = +y, cols = +x
        self.X, self.Y = X, Y

        def noise(cells: int, amp: float) -> np.ndarray:
            small = rng.standard_normal((max(2, ny // cells), max(2, nx // cells))).astype(np.float32)
            return cv2.resize(small, (nx, ny), interpolation=cv2.INTER_CUBIC) * amp

        n = noise(6, 10.0) + noise(30, 14.0) + noise(120, 8.0)
        grass = np.stack([86 + n, 112 + n * 1.1, 58 + n * 0.6], -1)
        # distance to route (coarse, via distance transform on a route raster)
        route_mask = np.ones((ny, nx), np.uint8)
        rp = ((route.xy - (WORLD_X0, WORLD_Y0)) / TEX_RES).astype(np.int32)
        ok = (rp[:, 0] >= 0) & (rp[:, 0] < nx) & (rp[:, 1] >= 0) & (rp[:, 1] < ny)
        route_mask[rp[ok, 1], rp[ok, 0]] = 0
        dist = cv2.distanceTransform(route_mask, cv2.DIST_L2, 5) * TEX_RES
        trail = dist < (0.85 + noise(40, 0.012) / 10.0)
        dirt = np.stack([150 + n * 0.8, 128 + n * 0.8, 96 + n * 0.6], -1)
        tex = np.where(trail[..., None], dirt, grass)
        cls = np.where(trail, 4, 3).astype(np.uint8)  # 5-class semantics: 4 stable path, 3 grass
        (wx, wy), (ax, ay) = WATER
        water = ((X - wx) / ax) ** 2 + ((Y - wy) / ay) ** 2 <= 1.0
        tex[water] = np.stack([48 + n[water] * 0.3, 70 + n[water] * 0.3, 84 + n[water] * 0.3], -1)
        cls[water] = 2
        self.ditch = self.ditch_mask(X, Y)
        tex[self.ditch] = np.stack([52 + n[self.ditch] * 0.4, 42 + n[self.ditch] * 0.3, 34 + n[self.ditch] * 0.2], -1)
        self.rock = np.zeros_like(water)
        for rx, ry, rr in ROCKS:
            d = np.hypot(X - rx, Y - ry)
            m = d <= rr
            shade = 1.0 - 0.45 * (d[m] / rr) ** 2
            tex[m] = np.stack([128 * shade + n[m] * 0.5, 124 * shade + n[m] * 0.5, 118 * shade + n[m] * 0.5], -1)
            self.rock |= m
        cls[self.rock] = 1
        self.tex = np.clip(tex, 0, 255).astype(np.uint8)
        self.cls = cls
        self.water = water
        self._caster_cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @staticmethod
    def ditch_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        in_band = (x >= DITCH_X[0]) & (x <= DITCH_X[1]) & (y >= DITCH_Y[0]) & (y <= DITCH_Y[1])
        return in_band & ~((y > DITCH_GAP_Y[0]) & (y < DITCH_GAP_Y[1]))

    # ------------------------------------------------------------------ ray casting
    @staticmethod
    def _ground_lut(K: np.ndarray, T_body_cam: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-pixel body-frame ground hit (x, y) and camera depth z; sky pixels -> NaN."""
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
        R, t = T_body_cam[:3, :3], T_body_cam[:3, 3]
        d = rays @ R.T
        lam = np.where(d[..., 2] < -1e-6, -t[2] / np.minimum(d[..., 2], -1e-6), np.nan)
        return (t[0] + lam * d[..., 0]).astype(np.float32), (t[1] + lam * d[..., 1]).astype(np.float32), lam.astype(np.float32)

    def _lut(self, key: str, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = (key, w, h)
        if k not in self._caster_cache:
            if key == "left":
                calib = defaults.stereo_calibration()
                sx, sy = w / calib.width, h / calib.height
                K = np.array([[calib.fx * sx, 0, calib.cx * sx], [0, calib.fy * sy, calib.cy * sy], [0, 0, 1]])
                self._caster_cache[k] = self._ground_lut(K, calib.T_body_cam, w, h)
            else:  # chase camera: 4.2 m behind, 2.4 m up, pitched 18 deg down, 64 deg HFOV
                f = (w / 2) / math.tan(math.radians(32.0))
                K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
                self._caster_cache[k] = self._ground_lut(K, chase_extrinsics(), w, h)
        return self._caster_cache[k]

    def _sample(self, lut: tuple[np.ndarray, np.ndarray, np.ndarray], pose_a: tuple[float, float, float],
                layer: np.ndarray, interp: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bx, by, lam = lut
        c, s = math.cos(pose_a[2]), math.sin(pose_a[2])
        ax = pose_a[0] + c * bx - s * by
        ay = pose_a[1] + s * bx + c * by
        mx = ((ax - WORLD_X0) / TEX_RES - 0.5).astype(np.float32)
        my = ((ay - WORLD_Y0) / TEX_RES - 0.5).astype(np.float32)
        out = cv2.remap(layer, np.nan_to_num(mx, nan=-10.0), np.nan_to_num(my, nan=-10.0), interp,
                        borderMode=cv2.BORDER_REPLICATE)
        return out, ax, ay

    def render(self, key: str, pose_a: tuple[float, float, float], w: int, h: int) -> np.ndarray:
        """RGB view from the left camera (``key='left'``) or the chase camera (``'chase'``)."""
        lut = self._lut(key, w, h)
        img, _, _ = self._sample(lut, pose_a, self.tex, cv2.INTER_LINEAR)
        lam = lut[2]
        sky = np.isnan(lam)
        rng = np.nan_to_num(lam, nan=1e3)
        fog = np.clip((rng - FOG_START_M) / FOG_SCALE_M, 0.0, FOG_MAX)[..., None]  # hides far-field texture aliasing
        rows = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        sky_rgb = np.array([170, 190, 210], np.float32) * (1 - 0.15 * (1 - rows)) + 0 * rows
        sky_img = np.broadcast_to(sky_rgb, (h, w, 3))
        out = img.astype(np.float32) * (1 - fog) + np.array([180, 194, 206], np.float32) * fog
        out = np.where(sky[..., None], sky_img, out)
        out = np.clip(out, 0, 255).astype(np.uint8)
        if key == "chase":
            draw_vehicle_chase(out, w, h)
        return out

    def perception(self, pose_a: tuple[float, float, float], rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Left image + semantic mask + missing-ground mask + disparity at the camera resolution."""
        calib = defaults.stereo_calibration()
        w, h = calib.width, calib.height
        lut = self._lut("left", w, h)
        left = self.render("left", pose_a, w, h)
        sem, ax, ay = self._sample(lut, pose_a, self.cls, cv2.INTER_NEAREST)
        lam = lut[2]
        sky = np.isnan(lam)
        sem = np.where(sky, 0, sem).astype(np.uint8)
        ditch = self.ditch_mask(np.nan_to_num(ax, nan=-1e3), np.nan_to_num(ay, nan=-1e3)) & ~sky & (lam < defaults.CAM_MAX_RANGE_M)
        disp = np.where(sky | (lam > defaults.CAM_MAX_RANGE_M), 0.0, calib.fx * calib.baseline_m / np.nan_to_num(lam, nan=1e9))
        disp = np.where(ditch, disp * 0.86, disp)  # points below the ground plane appear farther
        disp = (np.round(disp / DISP_QUANT) * DISP_QUANT).astype(np.float32)
        return {"left_rgb": left, "semantic_mask": sem, "missing_ground_mask": ditch, "disparity": disp}


def chase_extrinsics() -> np.ndarray:
    """T_body_cam of the virtual chase camera (OpenCV camera axes)."""
    p = math.radians(18.0)
    R_level = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    R_pitch = np.array([[math.cos(p), 0.0, math.sin(p)], [0.0, 1.0, 0.0], [-math.sin(p), 0.0, math.cos(p)]])
    T = np.eye(4)
    T[:3, :3] = R_pitch @ R_level
    T[:3, 3] = [-4.2, 0.0, 2.4]
    return T


def draw_vehicle_chase(img: np.ndarray, w: int, h: int) -> None:
    """Draw the UGV (box + mast) into a chase frame; the vehicle is rigid w.r.t. the camera."""
    f = (w / 2) / math.tan(math.radians(32.0))
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
    T = chase_extrinsics()
    R, t = T[:3, :3], T[:3, 3]
    L, Wd, Hh = defaults.VEHICLE.length_m, defaults.VEHICLE.width_m, 0.38

    def proj(pts: np.ndarray) -> np.ndarray:
        pc = (pts - t) @ R
        return np.column_stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]])

    def box(x0, x1, y0, y1, z0, z1):
        return {"top": [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
                "back": [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
                "left": [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
                "right": [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]}

    shadow = proj(np.array([(-L / 2 - 0.1, Wd / 2 + 0.12, 0), (L / 2 + 0.1, Wd / 2 + 0.12, 0), (L / 2 + 0.1, -Wd / 2 - 0.12, 0),
                            (-L / 2 - 0.1, -Wd / 2 - 0.12, 0)], float))
    ov = img.copy()
    cv2.fillPoly(ov, [np.round(shadow * 16).astype(np.int32)], (20, 20, 20), cv2.LINE_AA, 4)
    cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)
    # wheels
    for yy in (Wd / 2 + 0.02, -Wd / 2 - 0.02):
        for xx in (-L / 2 + 0.16, L / 2 - 0.16):
            q = proj(np.array([(xx - 0.13, yy, 0.0), (xx + 0.13, yy, 0.0), (xx + 0.13, yy, 0.26), (xx - 0.13, yy, 0.26)]))
            cv2.fillPoly(img, [np.round(q * 16).astype(np.int32)], (28, 30, 34), cv2.LINE_AA, 4)
    faces = box(-L / 2, L / 2, -Wd / 2, Wd / 2, 0.12, Hh)
    shade = {"right": (58, 66, 80), "left": (58, 66, 80), "back": (44, 52, 64), "top": (92, 102, 118)}
    for name in ("left", "right", "back", "top"):
        q = proj(np.array(faces[name], float))
        cv2.fillPoly(img, [np.round(q * 16).astype(np.int32)], shade[name], cv2.LINE_AA, 4)
        cv2.polylines(img, [np.round(q * 16).astype(np.int32)], True, (24, 28, 36), 1, cv2.LINE_AA, 4)
    mast = box(0.18, 0.30, -0.1, 0.1, Hh, Hh + 0.42)
    for name in ("back", "left", "top"):
        q = proj(np.array(mast[name], float))
        cv2.fillPoly(img, [np.round(q * 16).astype(np.int32)], shade[name], cv2.LINE_AA, 4)
    cam = box(0.24, 0.34, -0.14, 0.14, Hh + 0.42, Hh + 0.5)
    for name in ("back", "top"):
        q = proj(np.array(cam[name], float))
        cv2.fillPoly(img, [np.round(q * 16).astype(np.int32)], (30, 34, 44) if name == "back" else (70, 78, 92), cv2.LINE_AA, 4)
    # accent stripe on the back face (identification)
    q = proj(np.array([(-L / 2 - 0.001, -Wd / 2 + 0.05, 0.28), (-L / 2 - 0.001, Wd / 2 - 0.05, 0.28),
                       (-L / 2 - 0.001, Wd / 2 - 0.05, 0.31), (-L / 2 - 0.001, -Wd / 2 + 0.05, 0.31)]))
    cv2.fillPoly(img, [np.round(q * 16).astype(np.int32)], (59, 130, 246), cv2.LINE_AA, 4)


# ----------------------------------------------------------------------------- seen-ground map (toy)
class ToyMap:
    """A-frame seen-ground map at MAP_RES_M with the toy visibility model (module docstring)."""

    def __init__(self, scene: GroundScene) -> None:
        self.res = defaults.MAP_RES_M
        self.nx, self.ny = int(WORLD_W_M / self.res), int(WORLD_H_M / self.res)
        xs = WORLD_X0 + (np.arange(self.nx) + 0.5) * self.res
        ys = WORLD_Y0 + (np.arange(self.ny) + 0.5) * self.res
        self.X, self.Y = np.meshgrid(xs, ys)
        self.state = np.zeros((self.ny, self.nx), np.uint8)
        self.confirmed = np.zeros_like(self.state, bool)
        self.first_cand_t = np.full(self.state.shape, np.inf)
        self.is_ditch = GroundScene.ditch_mask(self.X, self.Y)
        (wx, wy), (ax, ay) = WATER
        self.is_water = ((self.X - wx) / ax) ** 2 + ((self.Y - wy) / ay) ** 2 <= 1.0
        self.is_rock = np.zeros_like(self.is_water)
        for rx, ry, rr in ROCKS:
            self.is_rock |= np.hypot(self.X - rx, self.Y - ry) <= rr
        lip = (self.X > DITCH_X[1]) & (self.X <= DITCH_X[1] + LIP_SHADOW_M) & (self.Y >= DITCH_Y[0]) & (self.Y <= DITCH_Y[1])
        self.lip_shadow = lip
        # launch apron: the stack starts with the ground under/around the vehicle certified
        self.state[(self.X > -1.5) & (self.X < LAUNCH_APRON_M) & (np.abs(self.Y) < 1.0)] = S.GROUND

    def observe(self, pose: tuple[float, float, float], t: float) -> None:
        x, y, yaw = pose
        cx, cy = x + defaults.CAM_FORWARD_M * math.cos(yaw), y + defaults.CAM_FORWARD_M * math.sin(yaw)
        dx, dy = self.X - cx, self.Y - cy
        r = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx) - yaw
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        vis = (np.abs(bearing) <= math.radians(defaults.HFOV_DEG / 2)) & (r >= 1.3) & (r <= defaults.CAM_MAX_RANGE_M)
        occl = np.zeros_like(vis)
        for rx, ry, rr in ROCKS:
            rr_c = math.hypot(rx - cx, ry - cy)
            if rr_c < 1e-3:
                continue
            b = math.atan2(ry - cy, rx - cx) - yaw
            half = math.asin(min(1.0, rr / rr_c))
            db = np.arctan2(np.sin(bearing - b), np.cos(bearing - b))
            occl |= (np.abs(db) <= half) & (r > rr_c + rr * 0.5) & (r < rr_c + 4.0)
        lip_hidden = self.lip_shadow & (r > LIP_VISIBLE_RANGE_M)
        seen = vis & ~occl & ~lip_hidden
        ditch_far = self.is_ditch & (r > DITCH_DETECT_RANGE_M)
        st = self.state
        # ground / water / rocks
        g = seen & ~self.is_ditch & ~self.is_rock & ~self.is_water
        st[g] = S.GROUND
        st[seen & self.is_water] = S.WATER
        st[(vis & self.is_rock)] = S.POSITIVE
        # occluded (unknown, never lethal): only where nothing better is known
        st[vis & occl & (st == S.UNSEEN) & ~self.is_rock] = S.OCCLUDED
        # ditch: missing ground once close enough, confirmed after dwell or at short range
        cand = seen & self.is_ditch & ~ditch_far
        st[cand] = S.DITCH_CANDIDATE
        self.first_cand_t = np.where(cand & np.isinf(self.first_cand_t), t, self.first_cand_t)
        conf = cand & ((t - self.first_cand_t >= DITCH_CONFIRM_S) | (r <= DITCH_CONFIRM_RANGE_M))
        self.confirmed |= conf

    def index(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ix = np.floor((np.asarray(x) - WORLD_X0) / self.res).astype(np.int64)
        iy = np.floor((np.asarray(y) - WORLD_Y0) / self.res).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return np.clip(ix, 0, self.nx - 1), np.clip(iy, 0, self.ny - 1), ok

    def crop(self, x: float, y: float, half_m: float = 10.0) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        j0 = int(np.clip(math.floor((x - half_m - WORLD_X0) / self.res), 0, self.nx))
        j1 = int(np.clip(math.floor((x + half_m - WORLD_X0) / self.res), 0, self.nx))
        i0 = int(np.clip(math.floor((y - half_m - WORLD_Y0) / self.res), 0, self.ny))
        i1 = int(np.clip(math.floor((y + half_m - WORLD_Y0) / self.res), 0, self.ny))
        return (self.state[i0:i1, j0:j1].copy(), self.confirmed[i0:i1, j0:j1].copy(),
                (WORLD_X0 + j0 * self.res, WORLD_Y0 + i0 * self.res))

    def sample_states(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix, iy, ok = self.index(x, y)
        st = np.where(ok, self.state[iy, ix], S.UNSEEN).astype(np.uint8)
        return st, np.where(ok, self.confirmed[iy, ix], False)

    def r_cert(self, path_xy: np.ndarray, half_width: float = 0.45, step: float = 0.2) -> float:
        """Distance along ``path_xy`` (from its start) to the first non-GROUND cell in a corridor."""
        if len(path_xy) < 2:
            return 0.0
        d = np.gradient(path_xy, axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        nrm = np.column_stack([-d[:, 1], d[:, 0]])
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path_xy, axis=0).T))])
        offs = np.array([-half_width, 0.0, half_width])
        pts = path_xy[:, None, :] + offs[None, :, None] * nrm[:, None, :]
        st, _ = self.sample_states(pts[..., 0], pts[..., 1])
        bad = np.any(st != S.GROUND, axis=1)
        if not bad.any():  # corridor certified up to the end of the checked path
            return float(defaults.CAM_MAX_RANGE_M)
        return float(min(s[int(np.argmax(bad))], defaults.CAM_MAX_RANGE_M))


def u4_codes(state: np.ndarray, confirmed: np.ndarray, trail_cost: np.ndarray) -> np.ndarray:
    """CellState (+ confirmation) -> 4-bit telemetry codes (see codec)."""
    out = np.zeros(state.shape, np.uint8)
    g = state == S.GROUND
    out[g] = (codec.U4_GROUND_MIN + np.clip((trail_cost * 8).astype(int), 0, 7))[g]
    out[state == S.OCCLUDED] = codec.U4_OCCLUDED
    out[state == S.CREST_SHADOW] = codec.U4_CREST_SHADOW
    out[state == S.WATER] = codec.U4_WATER
    c = state == S.DITCH_CANDIDATE
    out[c & ~confirmed] = codec.U4_DITCH_CANDIDATE
    out[c & confirmed] = codec.U4_DITCH
    out[state == S.DYNAMIC] = codec.U4_DYNAMIC
    out[(state == S.POSITIVE) | (state == S.DEPRESSION)] = codec.U4_LETHAL
    return out


def v_cap_from_rcert(r_cert: float, a: float = TOY_DECEL_MPS2, t_r: float = T_R_S,
                     margin: float = defaults.GOVERNOR_MARGIN_M) -> float:
    """Largest v with v^2/(2a) + v t_r + margin <= r_cert (m/s), clipped to the platform cap."""
    rr = max(r_cert - margin, 0.0)
    v = -a * t_r + math.sqrt((a * t_r) ** 2 + 2 * a * rr)
    return float(np.clip(v, 0.0, defaults.VEHICLE.max_speed_mps))


# ----------------------------------------------------------------------------- run synthesis
def _compose(p: tuple[float, float, float], q: tuple[float, float, float]) -> tuple[float, float, float]:
    c, s = math.cos(p[2]), math.sin(p[2])
    return p[0] + c * q[0] - s * q[1], p[1] + s * q[0] + c * q[1], p[2] + q[2]


def _to_world(pa: tuple[float, float, float]) -> tuple[float, float, float]:
    return _compose(START_WORLD, pa)


def build_demo_run(out_dir: str | Path, max_t: float = 40.0, seed: int = SEED, write_images: bool = True) -> Path:
    """Synthesise a DEMO_FAKE run directory at ``out_dir`` (see module docstring). Returns the path."""
    out = Path(out_dir)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    (out / "autonomy" / "debug").mkdir(parents=True, exist_ok=True)
    for f in (out / "autonomy" / "debug").glob("tick_*.npz"):
        f.unlink()
    rng = np.random.default_rng(seed)
    route = Route.build()
    scene = GroundScene(route, seed)
    tmap = ToyMap(scene)
    goal_a = (float(route.xy[-1, 0] + GOAL_OFFSET[0]), float(route.xy[-1, 1] + GOAL_OFFSET[1]))
    dt = 1.0 / defaults.GT_LOG_HZ
    ticks_every = int(round(defaults.GT_LOG_HZ / defaults.CAMERA_HZ_DEMO))
    tel_every = int(round(defaults.CAMERA_HZ_DEMO / defaults.TELEMETRY_HZ))
    B, r_w, chi = defaults.VEHICLE.track_width_m, defaults.VEHICLE.wheel_radius_m, defaults.CHI_NOMINAL
    gt: dict[str, list] = {k: [] for k in ("t", "x", "y", "z", "roll", "pitch", "yaw", "v", "omega", "wl", "wr")}
    cmds: dict[str, list] = {k: [] for k in ("t", "seq", "omega_l", "omega_r", "compute_ms", "apply_t", "mode")}
    tel_lines: list[str] = []
    s_pos, v, v_cmd, w_cmd = 0.0, 0.0, 0.0, 0.0
    mode, reason = DriveMode.NOMINAL, "START"
    tick = 0
    tel_seq = 0
    t = 0.0
    arrived_t: Optional[float] = None
    wl_cmd = wr_cmd = 0.0
    while t <= max_t:
        pa = route.at(s_pos)
        k = 0.0
        if s_pos < route.length - 0.1:
            y0, y1 = route.at(s_pos)[2], route.at(s_pos + 0.2)[2]
            k = (y1 - y0) / 0.2
        omega = v * k
        if int(round(t / dt)) % ticks_every == 0:
            # ---- estimated pose (toy VO drift that grows with distance)
            err = (0.004 * s_pos + 0.03 * math.sin(0.4 * t), 0.006 * s_pos, 0.0015 * s_pos)
            est = _compose(pa, err)
            tmap.observe(pa, t)
            ahead = route.segment(s_pos, s_pos + 12.5)
            r_cert = tmap.r_cert(ahead) if len(ahead) > 1 else 0.0
            v_cap = v_cap_from_rcert(r_cert)
            # CAUTION while missing ground is within 9 m of the vehicle and ahead of it
            near = np.hypot(tmap.X - pa[0], tmap.Y - pa[1])
            dit = (tmap.state == S.DITCH_CANDIDATE) & (near <= DITCH_DETECT_RANGE_M) & (tmap.X > pa[0] - 0.5)
            d_ditch = float(near[dit].min()) if dit.any() else math.inf
            dist_goal = math.hypot(goal_a[0] - pa[0], goal_a[1] - pa[1])
            if arrived_t is not None or s_pos >= route.length - 0.15:
                arrived_t = arrived_t if arrived_t is not None else t
                mode, reason = DriveMode.ARRIVED, f"ARRIVED err={dist_goal:.1f}m"
                v_target = 0.0
            elif math.isfinite(d_ditch):
                mode, reason = DriveMode.CAUTION, f"GOV_DITCH_DET d={d_ditch:.1f}m"
                v_target = min(v_cap, 0.65 * CRUISE_MPS)
            else:
                mode = DriveMode.NOMINAL
                binding = v_cap < CRUISE_MPS
                reason = f"GOV_RCERT R={r_cert:.1f}m" if binding else "OK"
                v_target = min(v_cap, CRUISE_MPS)
            v_cmd, w_cmd = v_target, v_target * k
            wl_cmd = (v_cmd - w_cmd * chi * B / 2) / r_w
            wr_cmd = (v_cmd + w_cmd * chi * B / 2) / r_w
            q = float(np.clip(0.965 + rng.normal(0, 0.008) - (0.05 if mode == DriveMode.CAUTION else 0.0), 0, 1))
            sigma = 0.05 + 0.009 * s_pos
            compute = float(np.clip(rng.normal(118, 9), 90, 170))
            health = {"q": q, "p_fail": 1 - q, "pos_sigma_m": sigma, "vo_ok": 1.0, "slip": 0.02, "chi_hat": chi,
                      "r_vis_m": float(min(defaults.CAM_MAX_RANGE_M, r_cert + 2.0)), "compute_ms": compute}
            cmds_row = (t, tick, wl_cmd, wr_cmd, compute, t + compute / 1e3, mode.value)
            for key, val in zip(cmds.keys(), cmds_row):
                cmds[key].append(val)
            # ---- debug bundle
            corr = _compose(est, (-pa[0] * math.cos(pa[2]) - pa[1] * math.sin(pa[2]),
                                  pa[0] * math.sin(pa[2]) - pa[1] * math.cos(pa[2]), -pa[2]))  # est o gt^-1

            def to_est(xy: np.ndarray) -> np.ndarray:
                c, s_ = math.cos(corr[2]), math.sin(corr[2])
                return np.column_stack([corr[0] + c * xy[:, 0] - s_ * xy[:, 1], corr[1] + s_ * xy[:, 0] + c * xy[:, 1]])

            plan = to_est(route.segment(s_pos, s_pos + max(0.5, v_cmd * 2.5 + 0.6), 0.1)) if v_cmd > 0.02 else np.array([est[:2]])
            glob = to_est(route.segment(s_pos, route.length, 0.5))
            K_r, T_r = 32, 20
            wk = w_cmd + rng.normal(0, 0.35, K_r)
            vk = np.clip(v_cmd * (1 + rng.normal(0, 0.12, K_r)), 0, None)
            th = np.cumsum(np.tile(wk[:, None] * 0.1, (1, T_r)), axis=1)
            ro = np.stack([np.cumsum(vk[:, None] * np.cos(th) * 0.1, axis=1), np.cumsum(vk[:, None] * np.sin(th) * 0.1, axis=1)], -1)
            crop, conf, origin = tmap.crop(est[0], est[1])
            gx = defaults.BEV_RES_M
            xs = -2.0 + (np.arange(160) + 0.5) * gx
            ys = -6.0 + (np.arange(120) + 0.5) * gx
            XB, YB = np.meshgrid(xs, ys, indexing="ij")
            c, s_ = math.cos(est[2]), math.sin(est[2])
            loc_st, _ = tmap.sample_states(est[0] + c * XB - s_ * YB, est[1] + s_ * XB + c * YB)
            arrays = {"t": np.float64(t), "pose_xy_yaw": np.asarray(est, np.float64), "v_cap_mps": np.float64(v_cap),
                      "r_cert_m": np.float64(r_cert), "cell_state_local": loc_st, "rollouts_xy": ro.astype(np.float32),
                      "plan_xy": plan.astype(np.float32), "global_path_xy": glob.astype(np.float32),
                      "extra_map_crop_state": crop, "extra_map_crop_confirmed": conf}
            if write_images:
                per = scene.perception(pa, rng)
                arrays.update({"semantic_mask": per["semantic_mask"], "missing_ground_mask": per["missing_ground_mask"],
                               "disparity": per["disparity"], "extra_left_rgb": per["left_rgb"]})
            extras = {"mode": mode.value, "reason": reason, "cmd_v": v_cmd, "cmd_w": w_cmd, "speed_meas": v, "t_r_s": T_R_S,
                      "a_mps2": TOY_DECEL_MPS2, "map_crop_origin": list(origin), "map_res_m": tmap.res,
                      "gov_binding": "r_cert", "demo_fake": True}
            arrays["health_json"] = np.array(json.dumps(health))
            arrays["timings_json"] = np.array(json.dumps({"compute": compute}))
            arrays["extras_json"] = np.array(json.dumps(extras))
            np.savez_compressed(out / "autonomy" / "debug" / f"tick_{tick:06d}.npz", **arrays)
            # ---- telemetry at 2 Hz through the real codec
            if tick % tel_every == 0:
                n, res = codec.COSTMAP_N, codec.COSTMAP_RES_M
                half = n * res / 2
                xb = half - (np.arange(n) + 0.5) * res
                yb = half - (np.arange(n) + 0.5) * res
                XB2, YB2 = np.meshgrid(xb, yb, indexing="ij")
                ax_, ay_ = est[0] + c * XB2 - s_ * YB2, est[1] + s_ * XB2 + c * YB2
                st2, cf2 = tmap.sample_states(ax_, ay_)
                ix, iy, _ = tmap.index(ax_, ay_)
                cost = np.where(scene.cls[np.clip((iy * tmap.res / TEX_RES).astype(int), 0, scene.cls.shape[0] - 1),
                                          np.clip((ix * tmap.res / TEX_RES).astype(int), 0, scene.cls.shape[1] - 1)] == 4, 0.0, 0.25)
                wps = [tuple(map(float, p)) for p in glob[::4][1:6]]
                tel = Telemetry(t=t, seq=tel_seq, pose_xy_yaw=est, pos_sigma_m=sigma, mode=mode, reason=reason, v_cap_mps=v_cap,
                                r_cert_m=r_cert, speed_mps=v, waypoints_xy=wps, costmap_u4=u4_codes(st2, cf2, cost), health=health)
                pkt = codec.encode(tel)
                tel_lines.append(json.dumps(codec.to_jsonable(codec.decode(pkt), len(pkt))))
                tel_seq += 1
            tick += 1
        # ---- GT log (world frame) + kinematics
        pw = _to_world(pa)
        wl_m = (v - omega * chi * B / 2) / r_w
        wr_m = (v + omega * chi * B / 2) / r_w
        for key, val in zip(gt.keys(), (t, pw[0], pw[1], 0.0, 0.0, 0.0, pw[2], v, omega, wl_m, wr_m)):
            gt[key].append(val)
        acc = defaults.VEHICLE.max_accel_mps2 if v_cmd > v else -defaults.BRAKE_DECEL_MPS2
        v = float(np.clip(v + acc * dt, min(v, v_cmd), max(v, v_cmd)))
        s_pos = min(s_pos + v * dt, route.length)
        t = round(t + dt, 6)
        if arrived_t is not None and t > arrived_t + 3.0:
            break
    np.savez_compressed(out / "gt" / "states.npz", **{k: np.asarray(v_, np.float64) for k, v_ in gt.items()})
    np.savez_compressed(out / "gt" / "cmds.npz", **{k: np.asarray(v_) for k, v_ in cmds.items()})
    (out / "gt" / "events.json").write_text(json.dumps([{"t": 0.0, "type": "start", "detail": {}}]), encoding="utf-8")
    (out / "autonomy" / "telemetry.jsonl").write_text("\n".join(tel_lines) + "\n", encoding="utf-8")
    end_a = route.at(s_pos)
    result = {"demo_fake": True, "NOTE": "SYNTHETIC DEMO DATA - not a measurement", "success": True, "failure_type": None,
              "time": round(t, 3), "path_length": round(route.length, 3),
              "final_error": round(math.hypot(end_a[0] - route.xy[-1, 0], end_a[1] - route.xy[-1, 1]), 3),
              "config": {"name": "DEMO_FAKE"}, "seed": seed, "family": "DEMO_FAKE", "split": "none",
              "sensor_mode": "demo_fake", "camera_hz": defaults.CAMERA_HZ_DEMO, "n_frames": tick, "n_telemetry": tel_seq}
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    mission = {"mission_id": f"MSN-DEMO-{seed:02d}", "goal_xy_a": list(goal_a), "goal_sigma_m": GOAL_SIGMA_M,
               "success_radius_m": 2.0, "timeout_s": 120.0, "demo_fake": True,
               "goal_entry": {"range_m": round(math.hypot(*goal_a), 2), "bearing_deg": round(math.degrees(math.atan2(goal_a[1], goal_a[0])), 1)}}
    (out / "mission.json").write_text(json.dumps(mission, indent=1), encoding="utf-8")
    log.info("demo_fake run written to %s (%.1f s, %d ticks, %d telemetry packets)", out, t, tick, tel_seq)
    return out


class DemoChaseRenderer:
    """``(state, w, h) -> rgb`` chase renderer for a demo_fake run (WORLD-frame state -> A-frame view)."""

    def __init__(self, seed: int = SEED) -> None:
        self.scene = GroundScene(Route.build(), seed)

    def __call__(self, state: dict, w: int, h: int) -> np.ndarray:
        x, y, _, _, _, yaw = state["pose"]
        x0, y0, a0 = START_WORLD
        c, s = math.cos(a0), math.sin(a0)
        pa = (c * (x - x0) + s * (y - y0), -s * (x - x0) + c * (y - y0), yaw - a0)
        return self.scene.render("chase", pa, w, h)
