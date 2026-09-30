"""Hero visuals for the deck and video, from the REAL renderer and the REAL perception stack.

Everything here is *Simulated*: frames come from the Three.js stereo renderer
(:class:`metagross.sim.render.bridge.ThreeRenderer`) on DEV scenarios (seeds 100-129 only) and
the BEV maps come from :class:`metagross.autonomy.perception.pipeline.Perception` run on the
rendered stereo pair at that pose (single frame, no temporal map). The simulator and the
autonomy code are used read-only.

Outputs (PNG, deck_assets/hero/):

* ``title_hero.png``        chase view of the rover approaching a ditch + inset BEV card
* ``perception_strip.png``  left image | cell states in the image | SGBM disparity | BEV + legend
* ``ditch_sequence.png``    BEV at 9 / 7 / 5 / 4 m before a trench (unseen -> ditch -> lethal)
* ``sim_montage.png``       2x3 camera views, one per scenario family (DEV seeds 100-105)

Run (PowerShell, launches Chrome)::

    & .venv\\Scripts\\python.exe scripts\\make_hero_visuals.py            # render + compose all
    & .venv\\Scripts\\python.exe scripts\\make_hero_visuals.py --compose  # re-compose from cache

Rendered frames and perception outputs are cached in ``deck_assets/hero/_cache/*.npz`` so the
layout can be iterated without re-rendering.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("OMP_NUM_THREADS", "2")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from metagross.config import defaults  # noqa: E402
from metagross.contracts.messages import CELL_COLORS, CellState, SensorFrame  # noqa: E402

LOG = logging.getLogger("make_hero_visuals")

SCEN_DIR = REPO / "data" / "scenarios" / "dev"
OUT_DIR = REPO / "deck_assets" / "hero"
CACHE_DIR = OUT_DIR / "_cache"
DEV_SEEDS = range(100, 130)  # never EVAL seeds 0-59

# --------------------------------------------------------------------------- scene selection
DITCH_GAP_MARGIN_M = 2.0  # crossing must be this far (arc length) from a ditch bypass gap
AE_SETTLE_DT_S = (-12.0, -6.0, 0.0)  # stereo renders before the kept frame (> 5 s apart: AE jumps to target)


def load_dev_scenario(seed: int) -> dict:
    """Scenario dict of a DEV seed (refuses EVAL seeds)."""
    if seed not in DEV_SEEDS:
        raise ValueError(f"seed {seed} is not a DEV seed (100-129)")
    return json.loads((SCEN_DIR / f"{seed}.json").read_text())


@dataclass
class Crossing:
    """Where the straight start->goal line crosses a ditch polyline (world frame)."""

    s_m: float  # arc length along start->goal line (m)
    xy: np.ndarray  # crossing point (m)
    angle_deg: float  # angle between line and ditch (90 = perpendicular)
    hazard: int  # index into scenario['hazards']


def _line_dir(sc: dict) -> tuple[np.ndarray, np.ndarray, float]:
    a = np.asarray(sc["start"]["xy"], float)
    b = np.asarray(sc["goal"]["xy"], float)
    L = float(np.linalg.norm(b - a))
    return a, (b - a) / L, L


def ditch_crossings(sc: dict) -> list[Crossing]:
    """All crossings of the start->goal line with ditch polylines, outside bypass gaps."""
    a, u, L = _line_dir(sc)
    out: list[Crossing] = []
    for k, hz in enumerate(sc.get("hazards", [])):
        if hz.get("type") != "ditch":
            continue
        P = np.asarray(hz["polyline"], float)
        seg = np.diff(P, axis=0)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
        for i in range(len(seg)):
            d = seg[i]
            M = np.array([[u[0], -d[0]], [u[1], -d[1]]])
            if abs(np.linalg.det(M)) < 1e-9:
                continue
            s, r = np.linalg.solve(M, P[i] - a)
            if 0.0 <= r <= 1.0 and 0.0 <= s <= L:
                s_arc = arc[i] + r * float(np.linalg.norm(d))
                if any(g0 - DITCH_GAP_MARGIN_M <= s_arc <= g1 + DITCH_GAP_MARGIN_M for g0, g1 in hz.get("gaps", [])):
                    continue
                cosang = abs(float(np.dot(u, d / np.linalg.norm(d))))
                out.append(Crossing(float(s), a + s * u, math.degrees(math.acos(min(cosang, 1.0))), k))
    return sorted(out, key=lambda c: c.s_m)


class ScenePoser:
    """Settles the simulated vehicle on the terrain at a point of the start->goal line and
    returns the renderer state (``World.render_state()`` format)."""

    def __init__(self, sc: dict) -> None:
        from metagross.sim.world import World  # read-only use of the simulator

        self.sc = sc
        self.world = World(sc, sensor_mode="tier0")
        self.a, self.u, self.L = _line_dir(sc)
        self.yaw = math.atan2(self.u[1], self.u[0])

    def state_at(self, s_m: float, t: float = 0.0, lateral_m: float = 0.0, yaw_offset_deg: float = 0.0,
                 dyn_trigger_t: Optional[float] = None) -> dict:
        n = np.array([-self.u[1], self.u[0]])
        xy = self.a + s_m * self.u + lateral_m * n
        self.world.vehicle.reset(float(xy[0]), float(xy[1]), self.yaw + math.radians(yaw_offset_deg))
        st = self.world.render_state()
        st["t"] = float(t)
        st["lighting"] = self.world.lighting_state(t)
        if dyn_trigger_t is not None:
            for d in self.world.dynamic:
                d.t_trigger = dyn_trigger_t
        st["dynamic"] = [d.state_dict(t, self.world.terrain) for d in self.world.dynamic]
        return st

    def world_to_body(self, pose6: list[float], xy: np.ndarray) -> np.ndarray:
        """World (N,2) -> body-frame (N,2) using the pose's x, y, yaw (planar)."""
        c, s = math.cos(pose6[5]), math.sin(pose6[5])
        d = np.asarray(xy, float) - np.asarray(pose6[:2], float)
        return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)


# --------------------------------------------------------------------------- render + perceive
class Capture:
    """Renders stereo (AE-settled) + optional chase at a state and runs Perception on the pair."""

    def __init__(self) -> None:
        from metagross.autonomy.perception.pipeline import Perception
        from metagross.sim.render.bridge import ThreeRenderer

        cv2.setNumThreads(2)
        self.calib = defaults.stereo_calibration()
        self.r = ThreeRenderer(self.calib, defaults.VEHICLE, shadows=True, color_transport="rgb")
        self.Perception = Perception
        self._loaded: Optional[int] = None
        LOG.info("renderer: %s", self.r.renderer_string)

    def load(self, sc: dict) -> None:
        if self._loaded != sc["seed"]:
            self.r.load_scenario(sc)
            self._loaded = sc["seed"]

    def stereo(self, state: dict) -> tuple[np.ndarray, np.ndarray]:
        t = float(state["t"])
        for k, dt in enumerate(AE_SETTLE_DT_S):
            st = dict(state, t=t + dt, seq=k)
            left, right = self.r.render_stereo(st)
        return left.copy(), right.copy()

    def chase(self, state: dict, w: int, h: int) -> np.ndarray:
        return self.r.render_chase(state, w, h).copy()

    def perceive(self, left: np.ndarray, right: np.ndarray, t: float) -> dict[str, Any]:
        per = self.Perception(self.calib, defaults.VEHICLE)  # fresh node: single-frame output
        frame = SensorFrame(t=t, seq=0, left_rgb=left, right_gray=right, wheel_angle_l_rad=0.0, wheel_angle_r_rad=0.0,
                            gyro_z_rps=0.0, sensor_mode="stereo")
        return per.process(frame, (0.0, 0.0, 0.0))

    def close(self) -> None:
        self.r.close()


def save_npz(name: str, **arrays: Any) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{name}.npz"
    np.savez_compressed(p, **{k: (np.asarray(v) if not isinstance(v, dict) else np.asarray(json.dumps(v))) for k, v in arrays.items()})
    return p


def load_npz(name: str) -> dict[str, Any]:
    z = np.load(CACHE_DIR / f"{name}.npz", allow_pickle=False)
    out = {}
    for k in z.files:
        a = z[k]
        out[k] = json.loads(str(a)) if a.dtype.kind == "U" else a
    return out


def per_arrays(out: dict[str, Any]) -> dict[str, Any]:
    """Subset of a Perception output that the figures need (cache-friendly)."""
    return {"disparity": out["disparity"], "cell_state": out["cell_state_local"], "missing": out["missing_ground_mask"],
            "r_vis": np.float32(out["r_vis_m"]), "bev_spec": out["bev_spec"],
            "timings": {k: float(v) for k, v in out["timings_ms"].items()}}


# --------------------------------------------------------------------------- scene plan (DEV only)
HERO_SEED = 103  # F2 ditch field: trench crosses the gravel trail ~10.8 m from the start
HERO_D_M = 5.0  # rover body origin this far (along the path) before the trench centreline
HERO_T_S = 20.0  # scene time (no lighting events in seed 103)
CHASE_SUPERSAMPLE = 2  # chase rendered at 2x the output size, then area-downsampled
TITLE_SIZE = (1600, 1200)
APPROACH_D0_M, APPROACH_D1_M = 12.0, 3.0  # approach run for the fused-map sequence (m before trench)
APPROACH_SPEED_MPS = 1.0  # 0.2 m between frames at CAMERA_HZ_BATCH
SEQ_D_M = (9.0, 7.0, 5.0, 3.0)  # snapshots shown in ditch_sequence.png (m before the trench centreline)
MONTAGE_SCALE = 2  # montage frames rendered at 2x the onboard resolution (same FOV / mount)
MONTAGE = {  # seed -> (s along start->goal [m] or None, scene time [s], label)
    102: (16.0, 20.0, "F1 · Trail"),
    103: (5.8, 20.0, "F2 · Ditch field"),
    104: (25.0, 20.0, "F3 · Crest (drop-off ahead)"),
    105: (None, 20.0, "F4 · Sudden obstacle"),
    100: (12.0, 15.5, "F5 · Lighting (sun glare)"),
    101: (14.0, 20.0, "F6 · Water / mud"),
}
F4_LEAD_M = 6.0  # F4: rover this far before the dynamic obstacle's path
F4_PATH_FRACTION = 0.55  # ... with the obstacle this far along its path


def capture_hero(cap: Capture) -> None:
    """Title hero + perception strip frame (single-frame Perception at the hero pose)."""
    sc = load_dev_scenario(HERO_SEED)
    cr = ditch_crossings(sc)[0]
    poser = ScenePoser(sc)
    cap.load(sc)
    st = poser.state_at(cr.s_m - HERO_D_M, t=HERO_T_S)
    left, right = cap.stereo(st)
    out = cap.perceive(left, right, HERO_T_S)
    w, h = TITLE_SIZE
    chase = cap.chase(st, w * CHASE_SUPERSAMPLE, h * CHASE_SUPERSAMPLE)
    chase = cv2.resize(chase, (w, h), interpolation=cv2.INTER_AREA)
    ditch = sc["hazards"][cr.hazard]
    edges = ditch_edges_body(poser, st["pose"], ditch)
    save_npz("hero", left=left, right=right, chase=chase, pose=np.asarray(st["pose"]), edges=edges,
             meta={"seed": HERO_SEED, "family": sc["family"], "d_m": HERO_D_M, "crossing_s_m": cr.s_m,
                   "ditch_width_m": ditch["width"], "ditch_depth_m": ditch["depth"]}, **per_arrays(out))
    LOG.info("hero: seed %d d=%.1f m, r_vis %.1f m, perception %.0f ms", HERO_SEED, HERO_D_M, out["r_vis_m"], out["timings_ms"]["total"])


def ditch_edges_body(poser: ScenePoser, pose6: list[float], ditch: dict) -> np.ndarray:
    """(2, N, 2) GT ditch top edges (centreline +- width/2) in the body frame at ``pose6``."""
    P = np.asarray(ditch["polyline"], float)
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    hw = 0.5 * float(ditch["width"])
    return np.stack([poser.world_to_body(pose6, P + hw * n), poser.world_to_body(pose6, P - hw * n)])


def fused_display(rmap: Any, pose6: list[float], x_rng: tuple[float, float], y_rng: tuple[float, float],
                  res: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample the RollingMap on a body-aligned grid (forward up, vehicle-left on the left).

    Returns (state, confirmed) arrays shaped (rows far->near, cols left->right)."""
    xs = np.arange(x_rng[1] - 0.5 * res, x_rng[0], -res)
    ys = np.arange(y_rng[1] - 0.5 * res, y_rng[0], -res)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    c, s = math.cos(pose6[5]), math.sin(pose6[5])
    wx, wy = pose6[0] + c * X - s * Y, pose6[1] + s * X + c * Y
    ix, iy = rmap.geometry.to_index(wx, wy)
    ok = rmap.geometry.in_bounds(ix, iy)
    ixc, iyc = np.clip(ix, 0, rmap.n - 1), np.clip(iy, 0, rmap.n - 1)
    state = np.where(ok, rmap.state[iyc, ixc], int(CellState.UNSEEN)).astype(np.uint8)
    conf = np.where(ok, rmap.confirmed[iyc, ixc], False)
    return state, conf


SEQ_X = (-1.0, 11.0)  # body-frame display window of the fused map (m)
SEQ_Y = (-4.0, 4.0)
SEQ_RES = 0.025  # display sampling of the 0.2 m map (crisp rotated cells)


def capture_approach(cap: Capture) -> None:
    """Drive the rover straight at the trench (GT poses from the simulator), fuse every
    single-frame Perception BEV into the stack's RollingMap, snapshot at SEQ_D_M."""
    from metagross.autonomy.planning.rolling_map import BevGeometry, RollingMap

    sc = load_dev_scenario(HERO_SEED)
    cr = ditch_crossings(sc)[0]
    poser = ScenePoser(sc)
    cap.load(sc)
    rmap = RollingMap()
    dt = 1.0 / defaults.CAMERA_HZ_BATCH
    step = APPROACH_SPEED_MPS * dt
    n = int(round((APPROACH_D0_M - APPROACH_D1_M) / step)) + 1
    per = cap.Perception(cap.calib, defaults.VEHICLE)  # one node for the whole run (as onboard)
    ditch = sc["hazards"][cr.hazard]
    snaps: dict[str, Any] = {}
    for k in range(n):
        d = APPROACH_D0_M - k * step
        t = HERO_T_S + k * dt
        st = poser.state_at(cr.s_m - d, t=t)
        p = st["pose"]
        if k == 0:
            left, right = cap.stereo(st)  # AE settled at the first pose
            rmap.recentre(p[0], p[1])
            rmap.seed_apron((p[0], p[1], p[5]), t)
        else:
            left, right = cap.r.render_stereo(dict(st, seq=k))
        frame = SensorFrame(t=t, seq=k, left_rgb=left, right_gray=right, wheel_angle_l_rad=0.0, wheel_angle_r_rad=0.0,
                            gyro_z_rps=0.0, sensor_mode="stereo")
        out = per.process(frame, (p[0], p[1], p[5]))
        rmap.integrate(t, (p[0], p[1], p[5]), out["cell_state_local"], out.get("cost_local"), out.get("mu_local"),
                       BevGeometry.from_perception(out, out["cell_state_local"].shape))
        for ds in SEQ_D_M:
            if abs(d - ds) < 0.5 * step:
                fs, fc = fused_display(rmap, p, SEQ_X, SEQ_Y, SEQ_RES)
                key = f"{ds:g}"
                snaps[f"left_{key}"] = left.copy()
                snaps[f"state_{key}"] = fs
                snaps[f"conf_{key}"] = fc
                snaps[f"single_{key}"] = out["cell_state_local"]
                snaps[f"edges_{key}"] = ditch_edges_body(poser, p, ditch)
                snaps["bev_spec"] = out["bev_spec"]
                LOG.info("approach snapshot d=%.1f m (frame %d)", d, k)
    save_npz("approach", **snaps, meta={"seed": HERO_SEED, "family": sc["family"], "n_frames": n, "step_m": step,
                                         "d0_m": APPROACH_D0_M, "d1_m": APPROACH_D1_M, "ditch_width_m": ditch["width"],
                                         "ditch_depth_m": ditch["depth"], "pose_source": "simulator ground truth"})


def capture_montage() -> None:
    """One onboard-camera view per scenario family at MONTAGE_SCALE x the onboard resolution."""
    import dataclasses

    from metagross.sim.render.bridge import ThreeRenderer

    base = defaults.stereo_calibration()
    k = MONTAGE_SCALE
    W, H = base.width * k, base.height * k
    cal = dataclasses.replace(base, width=W, height=H, fx=base.fx * k, fy=base.fy * k, cx=(W - 1) / 2.0, cy=(H - 1) / 2.0)
    r = ThreeRenderer(cal, defaults.VEHICLE, shadows=True, color_transport="rgb")
    try:
        arrays: dict[str, np.ndarray] = {}
        for seed, (s, t, _label) in MONTAGE.items():
            sc = load_dev_scenario(seed)
            poser = ScenePoser(sc)
            r.load_scenario(sc)
            trig = None
            if s is None:
                dyn = sc["dynamic"][0]
                P = np.asarray(dyn["path"], float)
                s = float(((P - poser.a) @ poser.u).mean()) - F4_LEAD_M
                trig = t - F4_PATH_FRACTION * float(np.linalg.norm(P[1] - P[0])) / float(dyn["speed"])
            st = poser.state_at(s, t=t, dyn_trigger_t=trig)
            for j, dt in enumerate(AE_SETTLE_DT_S):
                left, _ = r.render_stereo(dict(st, t=t + dt, seq=j))
            arrays[f"left_{seed}"] = left.copy()
            LOG.info("montage seed %d (%s) s=%.1f t=%.1f", seed, sc["family"], s, t)
        save_npz("montage", **arrays)
    finally:
        r.close()


def capture_all(which: set[str]) -> None:
    if which & {"hero", "approach"}:
        cap = Capture()
        try:
            if "hero" in which:
                capture_hero(cap)
            if "approach" in which:
                capture_approach(cap)
        finally:
            cap.close()
    if "montage" in which:
        capture_montage()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compose", action="store_true", help="only re-compose PNGs from the cache (no rendering)")
    ap.add_argument("--capture", default="hero,approach,montage", help="capture stages to run")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for name in ("metagross.sim.render.bridge",):
        logging.getLogger(name).setLevel(logging.WARNING)
    if not args.compose:
        capture_all({s.strip() for s in args.capture.split(",") if s.strip()})
    compose_all()
    return 0


# =========================================================================== composition
PX_PER_PT = 100.0 / 72.0  # figures are 100 dpi and saved at 100 dpi: 1 px == 1 output pixel
LETHAL_RGB = CELL_COLORS[CellState.POSITIVE]
UNSEEN_RGB = CELL_COLORS[CellState.UNSEEN]
SHADOW_ALPHA = 0.22  # inset card drop shadow strength on the photo
SHADOW_BLUR_PX = 18


def _style() -> dict[str, Any]:
    from metagross.eval.plot_style import CELL_LABELS, HONESTY_COLORS, TOKENS, mono_family, sans_family

    return {"T": TOKENS, "chip": HONESTY_COLORS["Simulated"], "labels": CELL_LABELS, "sans": sans_family(), "mono": mono_family()}


class Canvas:
    """Pixel-exact matplotlib canvas: top-left pixel coordinates, 1 figure px == 1 PNG px."""

    def __init__(self, w: int, h: int, bg: Optional[str] = None) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.transforms import IdentityTransform

        self.S = _style()
        self.plt = plt
        self.w, self.h = int(w), int(h)
        self.fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100, facecolor=bg or self.S["T"]["bg"])
        self.T = IdentityTransform()

    # geometry helpers (x, y = top-left in px)
    def _yb(self, y: float, hh: float = 0.0) -> float:
        return self.h - y - hh

    def rrect(self, x: float, y: float, w: float, h: float, r: float = 14, fc: str = "none", ec: str = "none",
              lw_px: float = 0.0, z: float = 2, alpha: float = 1.0, ls: str = "-"):
        from matplotlib.patches import FancyBboxPatch

        p = FancyBboxPatch((x, self._yb(y, h)), w, h, boxstyle=f"round,pad=0,rounding_size={r}", transform=self.T,
                           fc=fc, ec=ec, lw=lw_px / PX_PER_PT, zorder=z, alpha=alpha, ls=ls)
        self.fig.add_artist(p)
        return p

    def image(self, img: np.ndarray, x: int, y: int, radius: float = 0.0, z: float = 1, border: Optional[str] = None):
        im = self.fig.figimage(img, xo=int(x), yo=int(self.h - y - img.shape[0]), origin="upper", zorder=z)
        if radius > 0:
            clip = self.rrect(x, y, img.shape[1], img.shape[0], r=radius, fc="none", z=z)
            clip.set_visible(False)
            im.set_clip_path(clip)
        if border:
            self.rrect(x + 0.5, y + 0.5, img.shape[1] - 1, img.shape[0] - 1, r=radius, ec=border, lw_px=1.5, z=z + 0.1)
        return im

    def text(self, x: float, y: float, s: str, size_px: float, color: Optional[str] = None, weight: str = "normal",
             ha: str = "left", va: str = "top", mono: bool = False, z: float = 5, **kw: Any):
        fam = self.S["mono"] if mono else self.S["sans"]
        return self.fig.text(x, self._yb(y), s, transform=self.T, fontsize=size_px / PX_PER_PT, color=color or self.S["T"]["text"],
                             fontweight=weight, ha=ha, va=va, family=fam, zorder=z, **kw)

    def chip(self, x: float, y: float, s: str = "Simulated", size_px: float = 20, ha: str = "right", fc: Optional[str] = None):
        """Outlined honesty pill (white fill, 2 px border, bold text in the chip colour)."""
        col = self.S["chip"]
        return self.text(x, y, s, size_px, color=col, weight="bold", ha=ha, va="top",
                         bbox=dict(boxstyle="round,pad=0.42,rounding_size=0.95", fc=fc or "#FFFFFF", ec=col, lw=2.0 / PX_PER_PT))

    def line(self, xs: np.ndarray, ys: np.ndarray, color: str, lw_px: float = 1.5, ls: str = "-", z: float = 3, alpha: float = 1.0):
        from matplotlib.lines import Line2D

        ln = Line2D(np.asarray(xs, float), self.h - np.asarray(ys, float), transform=self.T, color=color,
                    lw=lw_px / PX_PER_PT, ls=ls, zorder=z, alpha=alpha, solid_capstyle="round", dash_capstyle="round")
        self.fig.add_artist(ln)
        return ln

    def swatch_legend(self, x: float, y: float, items: list[tuple[str, Any]], size_px: float = 19, sw: int = 22,
                      row_h: int = 36, cols: int = 1, col_w: int = 300) -> None:
        """items: (label, rgb tuple | 'gt' for a dashed ground-truth edge)."""
        T = self.S["T"]
        for k, (label, rgb) in enumerate(items):
            cx = x + (k % cols) * col_w
            cy = y + (k // cols) * row_h
            if rgb == "gt":
                self.line([cx, cx + sw], [cy + sw / 2, cy + sw / 2], T["text"], lw_px=2.2, ls=(0, (2, 1.6)))
            else:
                self.rrect(cx, cy, sw, sw, r=5, fc=_hex(rgb), ec=T["border"], lw_px=1.0)
            self.text(cx + sw + 10, cy + sw / 2, label, size_px, color=T["text_secondary"], va="center_baseline")

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=100, facecolor=self.fig.get_facecolor(), metadata={"Software": None})
        self.plt.close(self.fig)
        LOG.info("wrote %s (%dx%d)", path, self.w, self.h)
        return path


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#%02X%02X%02X" % tuple(int(c) for c in rgb)


def state_lut(confirmed_is_lethal: bool = False) -> np.ndarray:
    lut = np.zeros((256, 3), np.uint8)
    for s, c in CELL_COLORS.items():
        lut[int(s)] = c
    return lut


def colour_states(state: np.ndarray, confirmed: Optional[np.ndarray] = None) -> np.ndarray:
    """CellState grid -> RGB using the contract colours; confirmed ditch candidates are lethal red."""
    rgb = state_lut()[state]
    if confirmed is not None:
        rgb[(state == int(CellState.DITCH_CANDIDATE)) & confirmed] = LETHAL_RGB
    return rgb


@dataclass
class BevView:
    """Body-frame display window of the egocentric BEV (forward up, left on the left)."""

    x0: float
    x1: float
    y0: float
    y1: float
    px_per_m: float

    @property
    def size(self) -> tuple[int, int]:
        return int(round((self.y1 - self.y0) * self.px_per_m)), int(round((self.x1 - self.x0) * self.px_per_m))

    def to_px(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Body (x fwd, y left) -> panel pixel (col, row)."""
        return (self.y1 - np.asarray(y)) * self.px_per_m, (self.x1 - np.asarray(x)) * self.px_per_m


def bev_panel(cell_state: np.ndarray, spec: dict, view: BevView) -> np.ndarray:
    """Crop + nearest-upsample the perception BEV (``BevSpec`` convention) to ``view`` (RGB)."""
    return colour_states(bev_states(cell_state, spec, view))


def bev_states(cell_state: np.ndarray, spec: dict, view: BevView) -> np.ndarray:
    """Crop + nearest-upsample the perception BEV to ``view`` (CellState per display pixel)."""
    res, xm, ym = float(spec["res_m"]), float(spec["x_min_m"]), float(spec["y_min_m"])
    nx, ny = cell_state.shape
    w, h = view.size
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    y = view.y1 - cols / view.px_per_m
    x = view.x1 - rows / view.px_per_m
    i = np.floor((x - xm) / res).astype(int)
    j = np.floor((y - ym) / res).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    st = np.full((h, w), int(CellState.UNSEEN), np.uint8)
    st[ok] = cell_state[i[ok], j[ok]]
    return st


def draw_rover(cv: Canvas, ox: float, oy: float, view: BevView, color: Optional[str] = None) -> None:
    """Rover footprint (VEHICLE length x width, body origin at the wheelbase centre) + heading tick."""
    v = defaults.VEHICLE
    col = color or cv.S["T"]["navy"]
    c0, r0 = view.to_px(np.array([v.length_m / 2]), np.array([v.width_m / 2]))
    w = v.width_m * view.px_per_m
    h = v.length_m * view.px_per_m
    cv.rrect(ox + c0[0], oy + r0[0], w, h, r=min(w, h) * 0.18, fc=col, ec="#FFFFFF", lw_px=1.5, z=4)


def draw_gt_edges(cv: Canvas, ox: float, oy: float, view: BevView, edges: np.ndarray, color: str) -> None:
    for e in edges:
        c, r = view.to_px(e[:, 0], e[:, 1])
        ok = (c > -50) & (c < view.size[0] + 50) & (r > -50) & (r < view.size[1] + 50)
        if ok.sum() < 2:
            continue
        c, r = np.clip(c[ok], 0, view.size[0]), np.clip(r[ok], 0, view.size[1])
        cv.line(ox + c, oy + r, color, lw_px=2.0, ls=(0, (2.2, 1.8)), z=3.5, alpha=0.85)


def overlay_states_in_image(left: np.ndarray, disp: np.ndarray, cell_state: np.ndarray, missing: np.ndarray, spec: dict,
                            alpha: float = 0.58) -> np.ndarray:
    """Tint each pixel by the BEV cell its stereo point falls in (perception's own output
    back-projected), plus the image-space missing-ground mask in magenta."""
    from metagross.autonomy.perception.geometry import CameraGeometry

    geom = CameraGeometry(defaults.stereo_calibration())
    P = geom.point_image(disp)
    res, xm, ym = float(spec["res_m"]), float(spec["x_min_m"]), float(spec["y_min_m"])
    nx, ny = cell_state.shape
    with np.errstate(invalid="ignore"):
        i = np.floor((P[..., 0] - xm) / res)
        j = np.floor((P[..., 1] - ym) / res)
    ok = np.isfinite(i) & np.isfinite(j)
    ok &= (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    st = np.full(disp.shape, 255, np.uint8)
    st[ok] = cell_state[i[ok].astype(int), j[ok].astype(int)]
    col = np.zeros(left.shape, np.float32)
    tint = np.zeros(disp.shape, bool)
    for s in (CellState.GROUND, CellState.POSITIVE, CellState.DEPRESSION, CellState.DITCH_CANDIDATE, CellState.CREST_SHADOW,
              CellState.OCCLUDED, CellState.WATER):
        m = st == int(s)
        col[m] = CELL_COLORS[s]
        tint |= m
    col[missing] = CELL_COLORS[CellState.DITCH_CANDIDATE]
    tint |= missing
    gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY).astype(np.float32)[..., None]
    base = 0.55 * left.astype(np.float32) + 0.45 * gray  # partly desaturated so the key colours read
    out = base.copy()
    out[tint] = (1 - alpha) * base[tint] + alpha * col[tint]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def disparity_rgb(disp: np.ndarray, invalid_hex: str) -> np.ndarray:
    import matplotlib

    valid = disp > 0
    vmax = float(np.percentile(disp[valid], 99.5)) if valid.any() else 1.0
    rgb = (matplotlib.colormaps["turbo"](np.clip(disp / vmax, 0, 1))[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = [int(invalid_hex[k:k + 2], 16) for k in (1, 3, 5)]
    return rgb


def upscale(img: np.ndarray, w: int, h: int) -> np.ndarray:
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LANCZOS4
    return cv2.resize(img, (w, h), interpolation=interp)


def drop_shadow(img: np.ndarray, x: int, y: int, w: int, h: int, r: int) -> np.ndarray:
    """Darken a photo under a soft rounded-rect shadow (offset down a little)."""
    m = np.zeros(img.shape[:2], np.float32)
    cv2.rectangle(m, (x + r, y + 6), (x + w - r, y + h + 10), 1.0, -1)
    cv2.rectangle(m, (x, y + r + 6), (x + w, y + h - r + 10), 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), SHADOW_BLUR_PX)
    return np.clip(img.astype(np.float32) * (1.0 - SHADOW_ALPHA * m[..., None]), 0, 255).astype(np.uint8)


CORE_LEGEND = (CellState.GROUND, CellState.UNSEEN, CellState.DITCH_CANDIDATE, CellState.POSITIVE, CellState.CREST_SHADOW,
               CellState.OCCLUDED)


def legend_items(states: tuple[CellState, ...], labels: dict) -> list[tuple[str, Any]]:
    names = {CellState.POSITIVE: "Lethal (step / rock / depression)"}
    return [(names.get(s, labels[s]), CELL_COLORS[s]) for s in states]


# --------------------------------------------------------------------------- 1. title hero
def compose_title() -> Path:
    d = load_npz("hero")
    meta = d["meta"]
    img = d["chase"]
    H, W = img.shape[:2]
    view = BevView(-0.5, 11.5, -6.0, 6.0, 30.0)  # 360 x 360 px
    bw, bh = view.size
    pad = 22
    card_w = bw + 2 * pad
    legend_rows = 2
    card_h = pad + 58 + bh + 18 + legend_rows * 34 + pad - 6
    cx, cy = 40, H - card_h - 40
    img = drop_shadow(img, cx, cy, card_w, card_h, 16)
    cv = Canvas(W, H)
    T = cv.S["T"]
    cv.image(img, 0, 0)
    cv.rrect(cx, cy, card_w, card_h, r=16, fc="#FFFFFF", ec=T["border"], lw_px=1.5, z=2, alpha=0.97)
    cv.text(cx + pad, cy + pad, "SEEN-GROUND MAP", 15, color=T["text_muted"], weight="bold")
    cv.text(cx + pad, cy + pad + 22, "This frame, from the rover's stereo pair", 17, color=T["text"], weight="bold")
    by = cy + pad + 58
    cv.image(bev_panel(d["cell_state"], d["bev_spec"], view), cx + pad, by, radius=10, z=3)
    draw_rover(cv, cx + pad, by, view)
    items = legend_items((CellState.GROUND, CellState.DITCH_CANDIDATE, CellState.UNSEEN, CellState.POSITIVE), cv.S["labels"])
    items[1] = ("Ditch / missing ground", items[1][1])
    items[3] = ("Lethal", items[3][1])
    cv.swatch_legend(cx + pad, by + bh + 18, items, size_px=16, sw=18, row_h=34, cols=2, col_w=bw // 2 - 10)
    cv.chip(W - 36, 34, "Simulated", size_px=22)
    cv.text(W - 36, H - 34, f"DEV seed {meta['seed']} · {meta['family'].replace('_', ' ')} · {meta['d_m']:.0f} m before the trench",
            16, color="#FFFFFF", ha="right", va="bottom", weight="bold",
            bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.9", fc="#0F172A99", ec="none"))
    return cv.save(OUT_DIR / "title_hero.png")


# --------------------------------------------------------------------------- 2. perception strip
def compose_strip() -> Path:
    d = load_npz("hero")
    meta = d["meta"]
    S = _style()
    T = S["T"]
    iw, ih = 832, 520
    view = BevView(-0.5, 12.5, -6.5, 6.5, 40.0)  # 520 x 520 px, 4 px per 0.1 m cell
    bw, bh = view.size
    M, G = 48, 32
    W = M + 3 * iw + 3 * G + bw + M
    top = 44
    title_h = 64
    leg_rows = 4
    H = top + title_h + ih + 26 + leg_rows * 38 + 20
    cv = Canvas(W, H)
    left, disp = d["left"], d["disparity"]
    panels = [
        ("Left camera", "Rectified 640×400 frame from the stereo pair", upscale(left, iw, ih)),
        ("Cell states in the image", "Magenta = missing ground · red = lethal · green = seen ground",
         upscale(overlay_states_in_image(left, disp, d["cell_state"], d["missing"], d["bev_spec"]), iw, ih)),
        ("Stereo disparity (SGBM)", "Warm = near · dark = no reliable match", upscale(disparity_rgb(disp, T["text"]), iw, ih)),
    ]
    x = M
    y0 = top + title_h
    for k, (title, sub, img) in enumerate(panels):
        _panel_title(cv, x, top, k + 1, title, sub)
        cv.image(img, x, y0, radius=12, border=T["border"])
        x += iw + G
    _panel_title(cv, x, top, 4, "Seen-ground BEV", "0.1 m cells · forward up · rover at the bottom")
    cv.image(bev_panel(d["cell_state"], d["bev_spec"], view), x, y0, radius=12, border=T["border"])
    draw_gt_edges(cv, x, y0, view, d["edges"], T["text"])
    draw_rover(cv, x, y0, view)
    for r_m in (3, 6, 9):
        _, rr = view.to_px(np.array([r_m]), np.array([0.0]))
        cv.text(x + 12, y0 + rr[0], f"{r_m} m", 15, color=T["text_secondary"], va="center", mono=True,
                bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.6", fc="#FFFFFFE0", ec="none"))
    names = {CellState.POSITIVE: "Lethal", CellState.DITCH_CANDIDATE: "Ditch / missing ground"}
    items = [(names.get(s, S["labels"][s]), CELL_COLORS[s]) for s in CORE_LEGEND] + [("Ditch edge (ground truth)", "gt")]
    items = [items[k] for k in (0, 4, 1, 5, 2, 6, 3)]  # column-major reading order in a 2-column grid
    cv.swatch_legend(x, y0 + ih + 26, items, size_px=18, sw=20, row_h=38, cols=2, col_w=bw // 2)
    yb = y0 + ih + 26
    cv.chip(M + 12, yb, "Simulated", size_px=19, ha="left")
    cv.text(M + 160, yb + 3, f"One rendered stereo frame · DEV seed {meta['seed']} ({meta['family'].replace('_', ' ')}) · rover "
            f"{meta['d_m']:.0f} m before a {meta['ditch_width_m']:.2f} m wide, {meta['ditch_depth_m']:.2f} m deep trench · "
            "processed by the onboard Perception node (single frame)", 17, color=T["text_secondary"], va="top")
    return cv.save(OUT_DIR / "perception_strip.png")


def _panel_title(cv: Canvas, x: float, y: float, n: int, title: str, sub: str) -> None:
    T = cv.S["T"]
    cv.rrect(x, y, 28, 28, r=14, fc=T["navy"], z=3)
    cv.text(x + 14, y + 14.5, str(n), 16, color="#FFFFFF", weight="bold", ha="center", va="center", mono=True)
    cv.text(x + 40, y + 1, title, 22, weight="bold", va="top")
    cv.text(x + 40, y + 31, sub, 16, color=T["text_secondary"], va="top")


# --------------------------------------------------------------------------- 3. ditch sequence
def ditch_mask_display(edges: np.ndarray, view: BevView) -> np.ndarray:
    w, h = view.size
    c0, r0 = view.to_px(edges[0][:, 0], edges[0][:, 1])
    c1, r1 = view.to_px(edges[1][:, 0], edges[1][:, 1])
    poly = np.concatenate([np.stack([c0, r0], 1), np.stack([c1, r1], 1)[::-1]]).astype(np.float64)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(poly * 4).astype(np.int32)], 1, shift=2)
    return m.astype(bool)


SEQ_CORRIDOR_HALF_M = 1.0  # path corridor used for the per-snapshot ditch readout (rover half-width + margin)


def compose_sequence(mode: str = "single") -> Path:
    """``mode='single'``: single-frame Perception BEVs (deck figure). ``mode='fused'``: the stack's
    RollingMap after fusing every frame of the approach (diagnostic figure in ``_diag/``)."""
    d = load_npz("approach")
    meta = d["meta"]
    S = _style()
    T = S["T"]
    pview = BevView(SEQ_X[0], SEQ_X[1], SEQ_Y[0], SEQ_Y[1], 50.0)  # 400 x 600 px
    pw, ph = pview.size
    cam_w, cam_h = 336, 210
    col_w = pw + 20 + cam_w
    M, G = 48, 40
    W = 2 * M + 4 * col_w + 3 * G
    top = 40
    head_h = 86 if mode == "single" else 116
    H = top + head_h + ph + 128
    cv = Canvas(W, H)
    x = M
    y0 = top + head_h
    cols, rows = np.meshgrid(np.arange(pw) + 0.5, np.arange(ph) + 0.5)
    lateral = SEQ_Y[1] - cols / pview.px_per_m
    corridor = np.abs(lateral) <= SEQ_CORRIDOR_HALF_M
    if mode == "fused":
        cv.text(M, top - 22, "DIAGNOSTIC · NOT FOR THE DECK · RollingMap (0.2 m) fused over the approach, confirmed ditch = red, "
                "orange = DYNAMIC", 16, color="#B45309", weight="bold")
    for k, dm in enumerate(SEQ_D_M):
        key = f"{dm:g}"
        if mode == "fused":
            st = cv2.resize(d[f"state_{key}"], (pw, ph), interpolation=cv2.INTER_NEAREST)
            cf = cv2.resize(d[f"conf_{key}"].astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            st = bev_states(d[f"single_{key}"], d["bev_spec"], pview)
            cf = np.zeros(st.shape, bool)
        rgb = colour_states(st, cf)
        hy = top + (30 if mode == "fused" else 0)
        cv.text(x, hy, f"{dm:.0f} m", 44, weight="bold", mono=True, va="top")
        cv.text(x + 100, hy + 13, "to the trench", 19, color=T["text_secondary"], va="top")
        cv.image(rgb, x, y0, radius=12, border=T["border"])
        draw_gt_edges(cv, x, y0, pview, d[f"edges_{key}"], T["text"])
        draw_rover(cv, x, y0, pview)
        cx = x + pw + 20
        cv.image(upscale(d[f"left_{key}"], cam_w, cam_h), cx, y0, radius=10, border=T["border"])
        cv.text(cx, y0 + cam_h + 18, "TRENCH CELLS ON THE PATH", 13, color=T["text_muted"], weight="bold")
        dm_mask = ditch_mask_display(d[f"edges_{key}"], pview) & corridor
        n = max(int(dm_mask.sum()), 1)
        s_in = st[dm_mask]
        c_in = cf[dm_mask]
        lethal = ((s_in == int(CellState.DITCH_CANDIDATE)) & c_in) | np.isin(s_in, (int(CellState.POSITIVE), int(CellState.DEPRESSION)))
        cand = (s_in == int(CellState.DITCH_CANDIDATE)) & ~c_in
        unseen = np.isin(s_in, (int(CellState.UNSEEN), int(CellState.OCCLUDED), int(CellState.CREST_SHADOW)))
        ground = s_in == int(CellState.GROUND)
        fr = [(unseen.sum() / n, UNSEEN_RGB, "unseen"), (ground.sum() / n, CELL_COLORS[CellState.GROUND], "seen as ground"),
              (cand.sum() / n, CELL_COLORS[CellState.DITCH_CANDIDATE], "candidate"), (lethal.sum() / n, LETHAL_RGB, "lethal")]
        _stacked_bar(cv, cx, y0 + cam_h + 42, cam_w, 18, fr)
        yy = y0 + cam_h + 74
        for f, rgb_c, lab in fr:
            if f >= 0.005:
                cv.rrect(cx, yy + 2, 14, 14, r=4, fc=_hex(rgb_c), z=3)
                cv.text(cx + 22, yy, f"{100 * f:.0f} %", 16, weight="bold", mono=True, va="top")
                cv.text(cx + 80, yy + 1, lab, 16, color=T["text_secondary"], va="top")
                yy += 26
        if k < len(SEQ_D_M) - 1:
            ax = x + col_w + G / 2
            cv.text(ax, y0 + ph / 2, "›", 40, color=T["text_muted"], ha="center", va="center", weight="bold")
        x += col_w + G
    if mode == "fused":
        items = [("Unseen", UNSEEN_RGB), ("Seen ground", CELL_COLORS[CellState.GROUND]),
                 ("Ditch candidate (unconfirmed)", CELL_COLORS[CellState.DITCH_CANDIDATE]),
                 ("Confirmed ditch / lethal", LETHAL_RGB), ("Trench edge (ground truth)", "gt")]
        foot = (f"RollingMap fused from {meta['n_frames']} stereo frames · approach at 1 m/s, 5 Hz · DEV seed {meta['seed']} · "
                "poses from the simulator")
    else:
        items = [("Unseen", UNSEEN_RGB), ("Seen ground", CELL_COLORS[CellState.GROUND]),
                 ("Ditch / missing ground", CELL_COLORS[CellState.DITCH_CANDIDATE]),
                 ("Lethal (points below ground)", LETHAL_RGB), ("Trench edge (ground truth)", "gt")]
        foot = (f"Each panel: one stereo frame through the onboard Perception node · DEV seed {meta['seed']} · "
                f"{meta['ditch_width_m']:.2f} m wide, {meta['ditch_depth_m']:.2f} m deep trench · readout = ground-truth trench "
                f"footprint within ±{SEQ_CORRIDOR_HALF_M:.0f} m of the path")
    cv.swatch_legend(M, H - 84, items, size_px=18, sw=20, row_h=36, cols=5, col_w=400)
    cv.chip(W - M, top + 4, "Simulated", size_px=19)
    cv.text(M, H - 22, foot, 16, color=T["text_secondary"], ha="left", va="bottom")
    out = OUT_DIR / ("ditch_sequence.png" if mode == "single" else "_diag/ditch_sequence_rolling_map.png")
    return cv.save(out)


def _stacked_bar(cv: Canvas, x: float, y: float, w: float, h: float, parts: list[tuple[float, Any, str]]) -> None:
    T = cv.S["T"]
    cv.rrect(x, y, w, h, r=h / 2, fc=T["surface"], ec=T["border"], lw_px=1.0, z=2)
    xx = x
    bar = cv.rrect(x, y, w, h, r=h / 2, fc="none", z=2)
    bar.set_visible(False)
    from matplotlib.patches import Rectangle

    for f, rgb, _ in parts:
        if f <= 0:
            continue
        r = Rectangle((xx, cv._yb(y, h)), f * w, h, transform=cv.T, fc=_hex(rgb), ec="none", zorder=3)
        r.set_clip_path(bar)
        cv.fig.add_artist(r)
        xx += f * w


# --------------------------------------------------------------------------- 4. montage
def compose_montage() -> Path:
    d = load_npz("montage")
    S = _style()
    T = S["T"]
    tw, th = 960, 600
    M, G = 48, 24
    top = 40
    head_h = 70
    W = 2 * M + 3 * tw + 2 * G
    H = top + head_h + 2 * th + G + M
    cv = Canvas(W, H)
    cv.text(M, top, "SIX SCENARIO FAMILIES · ONBOARD LEFT CAMERA", 16, color=T["text_muted"], weight="bold")
    cv.text(M, top + 24, "Three.js stereo renderer, procedurally generated DEV worlds (seeds 100–105)", 22, weight="bold")
    cv.chip(W - M, top + 4, "Simulated", size_px=19)
    order = sorted(MONTAGE.items(), key=lambda kv: kv[1][2])
    for k, (seed, (_s, _t, label)) in enumerate(order):
        x = M + (k % 3) * (tw + G)
        y = top + head_h + (k // 3) * (th + G)
        cv.image(upscale(d[f"left_{seed}"], tw, th), x, y, radius=14, border=T["border"])
        cv.text(x + 18, y + 18, f"{label}", 20, weight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.9", fc="#FFFFFFEE", ec="none"))
        cv.text(x + tw - 18, y + th - 18, f"seed {seed}", 15, color="#FFFFFF", weight="bold", ha="right", va="bottom", mono=True,
                bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.8", fc="#0F172A99", ec="none"))
    return cv.save(OUT_DIR / "sim_montage.png")


def compose_all() -> None:
    out = []
    for name, fn in (("hero", compose_title), ("hero", compose_strip), ("approach", compose_sequence),
                     ("approach", lambda: compose_sequence("fused")), ("montage", compose_montage)):
        if (CACHE_DIR / f"{name}.npz").exists():
            out.append(fn())
        else:
            LOG.warning("cache %s missing: skipped a figure", name)
    for p in out:
        LOG.info("output %s", p)


if __name__ == "__main__":
    raise SystemExit(main())
