"""Simulated proprioceptive sensors and the **Tier-0 synthetic depth sensor**.

Wheel encoders
    Report the cumulative *wheel* rotation (rad) quantised to ``encoder_ticks_per_rev``. Because
    they measure shaft rotation, slip shows up as encoder travel exceeding ground travel.

Gyro (z)
    Mean yaw rate over the last camera interval (a pre-integrated MEMS gyro) plus a turn-on bias,
    a bias random walk and white noise (see :class:`GyroParams`).

Tier-0 synthetic depth sensor
    Produces what a rectified stereo matcher would output, *without images*: a float32 disparity
    map (px, ``<= 0`` invalid) from the true scene geometry. It is always labelled
    "synthetic depth sensor" / ``sensor_mode='tier0_disparity'``.

    Rendering (internal resolution ``internal_scale`` x the calibrated size, default 320 x 200):

    1. Terrain: the heightmap is sampled on a camera-centred polar lattice (azimuth step 1/1.3
       pixel over the fan spanned by the image border, range step max(5 cm, 1.5 % of range) out to
       CAM_MAX_RANGE_M + 1 m). Consecutive range
       samples of one azimuth form 3-D segments that are projected and rasterised (DDA, one
       fragment per pixel step, inverse depth interpolated linearly in screen space, which is exact
       for straight segments) into an inverse-depth z-buffer (``np.maximum.at``). Steep walls
       therefore cover every pixel row they span, and a ditch interior hidden behind its near lip
       loses the depth test exactly as it would in a real camera.
    2. Objects (rocks = ellipsoids, trees = trunk cylinder + canopy sphere, bushes, logs, dynamic
       obstacles) are ray-cast analytically inside their projected bounding boxes; min depth wins.
    3. Leak repair (a pixel > ~11 % farther than both opposite neighbours is a hidden surface seen
       through a splatting gap), small hole fill (holes with >= 5 of 8 valid neighbours), left/right occlusion check
       (points not visible from the right camera are invalid), range cap CAM_MAX_RANGE_M.
    4. Measurement model: Gaussian disparity noise sigma_d (default 0.25 px at *output*
       resolution), sparse gross outliers, block dropout by surface texture (water / mud / dim
       dirt), sky = invalid, and lighting effects (glare disc around the sun, dimming inflates
       noise and dropout, dust / fog range-dependent dropout plus dust phantom returns).

    Output: by default the internal map is nearest-upsampled x2 to the calibrated 640 x 400 and the
    disparity is rescaled to **full-resolution pixels**, so the frames match
    ``defaults.stereo_calibration()`` exactly (``output='internal'`` returns 320 x 200 with
    :meth:`Tier0DepthSensor.calibration` scaled accordingly).

    A GT pass (noise-free depth in metres and 5-class semantic ids via MATERIAL_TO_SEM /
    OBJECT_SEM / SKY_SEM) comes from the same raster for evaluator use only.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from metagross.config import defaults
from metagross.contracts.messages import StereoCalibration
from metagross.contracts.scenario import MATERIAL_TO_SEM, OBJECT_SEM, SKY_SEM
from metagross.sim.objects import Primitive, raycast_primitive
from metagross.sim.terrain import Terrain

# Label codes inside the renderer: 0..5 material ids, then these two.
CODE_OBJECT = 10
CODE_NONE = 255
# Leak repair: a pixel whose inverse depth is below this fraction of both opposite neighbours'
# (i.e. > ~11 % farther) is a hidden surface seen through a splatting gap.
_LEAK_RATIO = np.float32(0.9)


# =========================================================================== proprioception
class WheelEncoders:
    """Quantised cumulative wheel angles (rad)."""

    def __init__(self, ticks_per_rev: int = defaults.VEHICLE.encoder_ticks_per_rev):
        self.tick_rad = 2.0 * math.pi / ticks_per_rev

    def read(self, angle_l: float, angle_r: float) -> tuple[float, float]:
        q = self.tick_rad
        return math.floor(angle_l / q) * q, math.floor(angle_r / q) * q


@dataclass(frozen=True)
class GyroParams:
    """MEMS-class z-gyro error model (all rad/s based)."""

    bias0_sigma: float = 1.0e-3  # turn-on bias std (rad/s) ~ 0.06 deg/s
    bias_rw: float = 1.0e-4  # bias random walk (rad/s/sqrt(s))
    noise_density: float = 2.0e-3  # white-noise density (rad/s/sqrt(Hz))


class Gyro:
    """Returns the mean yaw rate over the sample interval + bias + noise."""

    def __init__(self, rng: np.random.Generator, params: GyroParams = GyroParams()):
        self.rng = rng
        self.p = params
        self.bias = float(rng.normal(0.0, params.bias0_sigma))

    def sample(self, mean_rate: float, dt: float) -> float:
        dt = max(dt, 1e-3)
        self.bias += float(self.rng.normal(0.0, self.p.bias_rw * math.sqrt(dt)))
        return mean_rate + self.bias + float(self.rng.normal(0.0, self.p.noise_density / math.sqrt(dt)))


# =========================================================================== Tier-0 depth
@dataclass(frozen=True)
class Tier0Params:
    internal_scale: float = 0.5  # render at 320 x 200 for a 640 x 400 calibration
    output: str = "full"  # 'full' (upsample to calibrated size) | 'internal'
    max_range_m: float = defaults.CAM_MAX_RANGE_M
    az_oversample: float = 1.3  # azimuth samples per pixel column
    az_margin_deg: float = 3.0  # added to the azimuth fan spanned by the image border rays
    r_min_m: float = 0.15
    dr_min_m: float = 0.05  # = terrain resolution
    dr_rel: float = 0.015  # range step grows with range (segments are rasterised, so no holes)
    max_frag_per_seg: int = 256
    z_near_m: float = 0.05
    hole_fill_min_nbrs: int = 5
    lr_check: bool = True
    sigma_d_px: float = 0.25  # disparity noise std at OUTPUT resolution (px)
    outlier_frac: float = 0.002  # fraction of gross mismatches
    outlier_px: tuple[float, float] = (2.0, 8.0)  # |error| range of a mismatch at output res (px)
    dropout_block_px: int = 4  # dropout blobs (internal px)
    dropout_iid: float = 0.005
    # Low-texture dropout probability per label code (materials 0..5, objects).
    dropout_by_code: tuple[tuple[int, float], ...] = ((0, 0.02), (1, 0.04), (2, 0.01), (3, 0.01), (4, 0.12), (5, 0.60),
                                                      (CODE_OBJECT, 0.01))
    glare_core_deg: float = 8.0  # saturated disc radius per unit glare gain
    glare_halo_deg: float = 20.0  # halo decay scale for extra dropout
    dust_phantom_frac_per_density: float = 0.1  # phantom fraction = this * dust density (1/m)


@dataclass
class Tier0Result:
    disparity: np.ndarray  # (H, W) float32 px at output resolution, <= 0 invalid
    depth_gt: np.ndarray | None = None  # (H, W) float32 m, 0 = no surface within range (evaluator only)
    semantic_gt: np.ndarray | None = None  # (H, W) uint8 5-class ids (evaluator only)
    timings_ms: dict[str, float] = field(default_factory=dict)


def _sem_lut() -> np.ndarray:
    lut = np.full(256, SKY_SEM, np.uint8)
    for m, s in MATERIAL_TO_SEM.items():
        lut[m] = s
    lut[CODE_OBJECT] = OBJECT_SEM
    return lut


_SEM_LUT = _sem_lut()


class Tier0DepthSensor:
    """Synthetic depth sensor (see module docstring). Not an image-based stereo pipeline."""

    def __init__(self, terrain: Terrain, static_prims: list[Primitive], params: Tier0Params = Tier0Params(),
                 calib: StereoCalibration | None = None):
        self.terrain = terrain
        self.prims = static_prims
        self.p = params
        self.calib_out = calib or defaults.stereo_calibration()
        s = params.internal_scale
        c = self.calib_out
        self.W = int(round(c.width * s))
        self.H = int(round(c.height * s))
        self.fx, self.fy = c.fx * s, c.fy * s
        self.cx, self.cy = (c.cx + 0.5) * s - 0.5, (c.cy + 0.5) * s - 0.5
        self.baseline = c.baseline_m
        # Radial samples of the camera-centred polar lattice (ground range, m).
        r = [params.r_min_m]
        r_max = params.max_range_m + 1.0
        while r[-1] < r_max:
            r.append(r[-1] + max(params.dr_min_m, params.dr_rel * r[-1]))
        self.r_samples = np.asarray(r, np.float32)
        # Pixel rays in the camera frame with z = 1 (ray parameter == depth).
        uu, vv = np.meshgrid(np.arange(self.W), np.arange(self.H))
        self.dcam = np.stack([(uu - self.cx) / self.fx, (vv - self.cy) / self.fy, np.ones_like(uu, float)], axis=-1)
        border = np.concatenate([self.dcam[0, :], self.dcam[-1, :], self.dcam[:, 0], self.dcam[:, -1]])
        self._border_rays = border[:: max(1, len(border) // 64)]
        self._dcam_unit: np.ndarray | None = None
        self._drop_lut = np.zeros(256, np.float32)
        for code, pr in params.dropout_by_code:
            self._drop_lut[code] = pr
        self.last_ms = 0.0

    # ------------------------------------------------------------------ calibration
    def calibration(self) -> StereoCalibration:
        """Calibration matching the frames this sensor outputs."""
        if self.p.output == "full":
            return self.calib_out
        return replace(self.calib_out, width=self.W, height=self.H, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)

    # ------------------------------------------------------------------ rendering core
    def _azimuth_fan(self, R: np.ndarray) -> tuple[float, float]:
        """(centre azimuth, half-width) in rad of the ground fan seen by the image border rays."""
        fwd = R[:, 2]
        az0 = math.atan2(fwd[1], fwd[0])
        d = self._border_rays @ R.T
        az = np.arctan2(d[:, 1], d[:, 0]) - az0
        az = (az + math.pi) % (2 * math.pi) - math.pi
        return az0, min(float(np.abs(az).max()) + math.radians(self.p.az_margin_deg), math.radians(89.0))

    def _terrain_pass(self, R: np.ndarray, C: np.ndarray, zbuf: np.ndarray, lab: np.ndarray) -> int:
        p, W, H = self.p, self.W, self.H
        f32 = np.float32
        az0, half = self._azimuth_fan(R)
        # Uniform in tan(azimuth) ~ uniform in image u, so the column spacing stays below
        # 1 / az_oversample px across the whole width (uniform angles stretch to 1/cos^2 at the edges).
        th = math.tan(half)
        n_az = int(math.ceil(2 * th * self.fx * p.az_oversample)) + 1
        az = (az0 + np.arctan(np.linspace(-th, th, n_az))).astype(f32)
        ca, sa = np.cos(az)[:, None], np.sin(az)[:, None]
        r = self.r_samples[None, :]
        dx = ca * r
        dy = sa * r
        Z, mat, inside = self.terrain.sample_fast(dx + f32(C[0]), dy + f32(C[1]))
        dz = Z - f32(C[2])
        Rf = R.astype(f32)
        xc = dx * Rf[0, 0] + dy * Rf[1, 0] + dz * Rf[2, 0]
        yc = dx * Rf[0, 1] + dy * Rf[1, 1] + dz * Rf[2, 1]
        zc = dx * Rf[0, 2] + dy * Rf[1, 2] + dz * Rf[2, 2]
        ok = inside & (zc > p.z_near_m)
        iz = f32(1.0) / np.where(ok, zc, f32(1.0))
        u = f32(self.fx) * xc * iz + f32(self.cx)
        v = f32(self.fy) * yc * iz + f32(self.cy)
        ua, ub, va, vb = u[:, :-1], u[:, 1:], v[:, :-1], v[:, 1:]
        seg = ok[:, :-1] & ok[:, 1:]
        seg &= ~(((ua < -1) & (ub < -1)) | ((ua > W) & (ub > W)) | ((va < -1) & (vb < -1)) | ((va > H) & (vb > H)))
        seg &= np.minimum(zc[:, :-1], zc[:, 1:]) <= p.max_range_m + 1.0
        ua, ub, va, vb = ua[seg], ub[seg], va[seg], vb[seg]
        iza, izb = iz[:, :-1][seg], iz[:, 1:][seg]
        code = mat[:, :-1][seg]
        du, dv, diz = ub - ua, vb - va, izb - iza
        n = np.minimum(np.ceil(np.maximum(np.abs(du), np.abs(dv))), f32(p.max_frag_per_seg))
        np.maximum(n, f32(1.0), out=n)
        n_int = n.astype(np.intp)
        total = int(n_int.sum())
        start = (np.cumsum(n_int) - n_int).astype(f32)
        # One gather of all per-segment attributes (better locality than one gather per array).
        attrs = np.stack([ua + f32(0.5), va + f32(0.5), iza, du, dv, diz, start, n, code.astype(f32)], axis=1)
        sid = np.repeat(np.arange(len(n_int)), n_int)
        a = attrs[sid]
        tau = (np.arange(total, dtype=f32) - a[:, 6]) / a[:, 7]
        iu = np.floor(a[:, 0] + tau * a[:, 3]).astype(np.intp)
        iv = np.floor(a[:, 1] + tau * a[:, 4]).astype(np.intp)
        izf = a[:, 2] + tau * a[:, 5]
        inb = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
        pix = iv[inb] * W + iu[inb]  # intp indices: required for numpy's fast ufunc.at path
        izf = izf[inb]
        np.maximum.at(zbuf, pix, izf)
        win = izf >= zbuf[pix]
        lab[pix[win]] = a[:, 8][inb][win].astype(np.uint8)
        return total

    def _object_pass(self, prims: list[Primitive], R: np.ndarray, C: np.ndarray, zbuf2d: np.ndarray, lab2d: np.ndarray) -> int:
        p, W, H = self.p, self.W, self.H
        n_cast = 0
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        for prim in prims:
            pc = R.T @ (prim.center - C)
            b = prim.bound_r
            if pc[2] + b < p.z_near_m or pc[2] - b > p.max_range_m + 1.0:
                continue
            if pc[2] - b > p.z_near_m:
                cc = pc[None, :] + b * corners
                uu = self.fx * cc[:, 0] / cc[:, 2] + self.cx
                vv = self.fy * cc[:, 1] / cc[:, 2] + self.cy
                u0, u1 = int(math.floor(uu.min())), int(math.ceil(uu.max()))
                v0, v1 = int(math.floor(vv.min())), int(math.ceil(vv.max()))
            else:
                u0, u1, v0, v1 = 0, W - 1, 0, H - 1
            u0, u1 = max(u0, 0), min(u1, W - 1)
            v0, v1 = max(v0, 0), min(v1, H - 1)
            if u1 < u0 or v1 < v0:
                continue
            d = self.dcam[v0:v1 + 1, u0:u1 + 1].reshape(-1, 3) @ R.T
            t = raycast_primitive(prim, C, d)
            n_cast += len(t)
            izo = np.where(np.isfinite(t), 1.0 / np.where(np.isfinite(t), t, 1.0), 0.0).astype(np.float32)
            izo = izo.reshape(v1 - v0 + 1, u1 - u0 + 1)
            blk = zbuf2d[v0:v1 + 1, u0:u1 + 1]
            better = izo > blk
            blk[better] = izo[better]
            lab2d[v0:v1 + 1, u0:u1 + 1][better] = CODE_OBJECT
        return n_cast

    @staticmethod
    def _repair_leaks(zb: np.ndarray, lab: np.ndarray) -> int:
        """Splatting safety net: a pixel much farther than BOTH horizontal (or both vertical)
        neighbours shows a hidden surface through a coverage gap; replace it by the nearer pair."""
        k = _LEAK_RATIO
        c = zb[1:-1, 1:-1]
        l, r, u, d = zb[1:-1, :-2], zb[1:-1, 2:], zb[:-2, 1:-1], zb[2:, 1:-1]
        leak_h = (l > 0) & (r > 0) & (c < k * np.minimum(l, r))
        leak_v = (u > 0) & (d > 0) & (c < k * np.minimum(u, d)) & ~leak_h
        n = int(leak_h.sum() + leak_v.sum())
        if n:
            lab_c = lab[1:-1, 1:-1]
            c[leak_h] = 0.5 * (l[leak_h] + r[leak_h])
            lab_c[leak_h] = lab[1:-1, :-2][leak_h]
            c[leak_v] = 0.5 * (u[leak_v] + d[leak_v])
            lab_c[leak_v] = lab[:-2, 1:-1][leak_v]
        return n

    def _hole_fill(self, zb: np.ndarray, lab: np.ndarray) -> None:
        valid = (zb > 0).astype(np.float32)
        k = np.ones((3, 3), np.float32)
        cnt = cv2.filter2D(valid, -1, k, borderType=cv2.BORDER_CONSTANT)
        s = cv2.filter2D(zb, -1, k, borderType=cv2.BORDER_CONSTANT)
        holes = (valid == 0) & (cnt >= self.p.hole_fill_min_nbrs)
        if not holes.any():
            return
        zb[holes] = s[holes] / cnt[holes]
        # Label of a valid 4-neighbour (up, down, left, right priority).
        fill = np.full(lab.shape, CODE_NONE, np.uint8)
        for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            nb_lab = np.roll(lab, sh, axis=ax)
            nb_ok = np.roll(valid, sh, axis=ax) > 0
            take = (fill == CODE_NONE) & nb_ok
            fill[take] = nb_lab[take]
        lab[holes] = fill[holes]

    def _lr_occlusion(self, disp: np.ndarray) -> np.ndarray:
        """Mask of pixels (with disparity) that the right camera cannot see."""
        H, W = disp.shape
        vv, uu = np.nonzero(disp > 0)
        d = disp[vv, uu]
        ur = np.floor(uu - d + 0.5).astype(np.int64)
        out = ur < 0
        key = vv * W + np.clip(ur, 0, W - 1)
        rmax = np.zeros(H * W, np.float32)
        np.maximum.at(rmax, key, d)
        occ = (d < rmax[key] - 0.5) | out
        mask = np.zeros((H, W), bool)
        mask[vv[occ], uu[occ]] = True
        return mask

    # ------------------------------------------------------------------ public
    def render(self, T_world_cam: np.ndarray, dynamic_prims: list[Primitive] | None = None, lighting: dict | None = None,
               rng: np.random.Generator | None = None, noise: bool = True, want_gt: bool = False) -> Tier0Result:
        """Render one synthetic disparity frame from a camera pose (4x4, camera->world)."""
        t0 = time.perf_counter()
        p, W, H = self.p, self.W, self.H
        R = np.asarray(T_world_cam[:3, :3], float)
        C = np.asarray(T_world_cam[:3, 3], float)
        zbuf = np.zeros(H * W, np.float32)
        lab = np.full(H * W, CODE_NONE, np.uint8)
        n_frag = self._terrain_pass(R, C, zbuf, lab)
        t1 = time.perf_counter()
        zb2, lab2 = zbuf.reshape(H, W), lab.reshape(H, W)
        n_cast = self._object_pass(self.prims + list(dynamic_prims or []), R, C, zb2, lab2)
        t2 = time.perf_counter()
        n_leak = self._repair_leaks(zb2, lab2)
        self._hole_fill(zb2, lab2)
        depth = np.where(zb2 > 0, 1.0 / np.maximum(zb2, 1e-9), 0.0).astype(np.float32)
        in_range = (depth > 0) & (depth <= p.max_range_m)
        disp = np.where(in_range, self.fx * self.baseline * zb2, 0.0).astype(np.float32)
        if p.lr_check:
            disp[self._lr_occlusion(disp)] = 0.0
        if noise:
            disp = self._measurement_model(disp, depth, lab2, R, lighting or {}, rng or np.random.default_rng(0))
        t3 = time.perf_counter()
        scale = 1.0 / p.internal_scale
        if p.output == "full":
            disp = cv2.resize(disp, (self.calib_out.width, self.calib_out.height), interpolation=cv2.INTER_NEAREST) * np.float32(scale)
        res = Tier0Result(disparity=disp)
        if want_gt:
            sem = _SEM_LUT[lab2]
            if p.output == "full":
                size = (self.calib_out.width, self.calib_out.height)
                depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
                sem = cv2.resize(sem, size, interpolation=cv2.INTER_NEAREST)
            res.depth_gt, res.semantic_gt = depth, sem
        t4 = time.perf_counter()
        res.timings_ms = {"terrain": (t1 - t0) * 1e3, "objects": (t2 - t1) * 1e3, "post": (t3 - t2) * 1e3,
                          "output": (t4 - t3) * 1e3, "total": (t4 - t0) * 1e3, "fragments": float(n_frag), "obj_rays": float(n_cast),
                          "leaks_repaired": float(n_leak)}
        self.last_ms = res.timings_ms["total"]
        return res

    # ------------------------------------------------------------------ noise / dropout / lighting
    def _measurement_model(self, disp: np.ndarray, depth: np.ndarray, lab: np.ndarray, R: np.ndarray, lighting: dict,
                           rng: np.random.Generator) -> np.ndarray:
        p, H, W = self.p, self.H, self.W
        s = p.internal_scale
        valid = disp > 0
        p_drop = self._drop_lut[lab].copy()
        sigma = np.full((H, W), p.sigma_d_px * s, np.float32)
        z = np.where(valid, depth, 0.0)
        fog = float(lighting.get("fog_density", 0.0))
        if fog > 0:
            p_drop += 1.0 - np.exp(-fog * z)
        phantom = None
        for ev in lighting.get("active_events", []):
            typ, gain = ev["type"], float(ev["gain"])
            if typ == "dim":
                g = max(gain, 0.05)
                sigma /= g
                low_tex = np.isin(lab, (0, 1, 4, 5))
                p_drop += np.where(low_tex, 0.25, 0.10) * (1.0 - g)
            elif typ == "dust":
                p_drop += 1.0 - np.exp(-gain * z)
                phantom = p.dust_phantom_frac_per_density * gain
            elif typ == "glare":
                sun = lighting.get("sun_dir_world")
                if sun is not None:
                    sc = R.T @ np.asarray(sun, float)
                    if sc[2] > 0:
                        if self._dcam_unit is None:
                            self._dcam_unit = self.dcam / np.linalg.norm(self.dcam, axis=-1, keepdims=True)
                        dn = self._dcam_unit
                        ang = np.degrees(np.arccos(np.clip(dn @ sc, -1.0, 1.0)))
                        core = ang < p.glare_core_deg * gain
                        p_drop += 0.3 * min(gain, 3.0) / 3.0 * np.exp(-ang / p.glare_halo_deg)
                        sigma *= 1.0 + 0.5 * np.exp(-ang / p.glare_halo_deg).astype(np.float32)
                        p_drop[core] = 1.0
        b = p.dropout_block_px
        blocks = rng.random((-(-H // b), -(-W // b)), dtype=np.float32)
        field_ = np.repeat(np.repeat(blocks, b, axis=0), b, axis=1)[:H, :W]
        drop = (field_ < p_drop) | (rng.random((H, W), dtype=np.float32) < p.dropout_iid)
        out = disp + rng.standard_normal((H, W), dtype=np.float32) * sigma
        n_out = rng.random((H, W), dtype=np.float32) < p.outlier_frac
        if n_out.any():
            mag = rng.uniform(*p.outlier_px, size=int(n_out.sum())).astype(np.float32) * s
            out[n_out] += mag * rng.choice(np.array([-1.0, 1.0], np.float32), size=mag.shape)
        out[drop | ~valid] = 0.0
        if phantom:
            ph_blocks = rng.random(blocks.shape, dtype=np.float32) < phantom
            ph = np.repeat(np.repeat(ph_blocks, b, axis=0), b, axis=1)[:H, :W]
            z_ph = rng.uniform(1.5, 5.0, size=blocks.shape).astype(np.float32)
            z_ph = np.repeat(np.repeat(z_ph, b, axis=0), b, axis=1)[:H, :W]
            out = np.where(ph, self.fx * self.baseline / z_ph, out)
        out[out <= 0] = 0.0
        return out.astype(np.float32)
