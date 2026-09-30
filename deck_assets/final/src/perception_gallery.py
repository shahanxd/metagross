"""Perception gallery: what the onboard Perception node makes of ONE rendered stereo frame.

Run from the repo root (Linux, CPU only: software WebGL, ~20 s per scene):

    MG_ANGLE=swiftshader MG_RENDER_HEADLESS=1 MG_ALLOW_SOFTWARE_GL=1 OMP_NUM_THREADS=2 \\
        python deck_assets/final/src/perception_gallery.py            # render (if not cached) + compose
    python deck_assets/final/src/perception_gallery.py --compose      # re-compose from the cache only
    python deck_assets/final/src/perception_gallery.py --recapture    # force a fresh render

Three DEV scenarios (seeds 100-129 only, never EVAL), one row each:

    row 1  F1 trail        seed 120  boulder on the trail ahead        -> lethal obstacle (red) + occluded wedge
    row 2  F2 ditch field  seed 109  trench across the path ahead      -> missing ground / ditch (magenta)
    row 3  F3 crest        seed 116  crest (terrain drops away) ahead  -> crest shadow (amber)

Columns: rendered left camera image | SGBM disparity | bird's-eye cell states (Perception output).

How each frame is produced (simulator used read-only, exactly the closed-loop stereo path):

    renderer = ThreeRenderer(stereo_calibration(), VEHICLE)            # = runner.default_renderer_factory()
    world = World(scenario, sensor_mode="stereo", renderer=renderer)
    world.vehicle.reset(x, y, yaw)   # rover on the straight start->goal line, S_M metres from the start,
                                     # heading along the line, settled on the terrain by the simulator
    renderer.render_stereo(state @ t=-12 s, -6 s)                      # auto-exposure pre-roll at the new pose
    frame = world.make_sensor_frame(world.t, 0)                        # contract SensorFrame, sensor_mode="stereo"
    out = Perception(calib, VEHICLE).process(frame, (0, 0, 0))         # fresh node: single-frame output

No terrain segmenter is attached (its deploy model is not trained yet), so there is no water / semantic cue.
Hazard distances printed on the images are ground truth, computed from data/scenarios/dev/<seed>.json.

Outputs (deck_assets/final/): perception_gallery.{png,svg,json}, perception_row_{1,2,3}.{png,svg,json}.
Label: Simulated (rendered camera, real stereo matching).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from metagross.config import defaults  # noqa: E402
from metagross.contracts.messages import CellState  # noqa: E402

LOG = logging.getLogger("perception_gallery")
OUT = ROOT / "deck_assets" / "final"
SCEN_DIR = "data/scenarios/dev"
DEV_SEEDS = range(100, 130)
CACHE_DEFAULT = Path("/tmp/claude-0/final/perception/cache")
GENERATOR = "deck_assets/final/src/perception_gallery.py"
LABEL = "Simulated (rendered camera, real stereo matching)"

# ---- palette (deck brief) ------------------------------------------------------------------
NAVY = "#1F497D"
INK = "#1A1A1A"
MUTED = "#5C5C5C"
FAINT = "#8A8A8A"
GRID = "#E6E6E6"
BG = "#FFFFFF"
NO_MATCH = "#D6D6D6"  # disparity: no stereo match (sky, left border, textureless)
# Bird's-eye cell-state colours. Ground uses a light tint of the deck green (#2E7D32): with the full-strength
# green, red-vs-green separation for deuteranopes is only dE 4.2 (validator), with this tint it is >= 10.
CELL_HEX: dict[CellState, str] = {
    CellState.GROUND: "#81C784",
    CellState.UNSEEN: "#DADADA",
    CellState.OCCLUDED: "#8F8F8F",
    CellState.DITCH_CANDIDATE: "#C828A0",  # magenta (contract colour CELL_COLORS[DITCH_CANDIDATE])
    CellState.POSITIVE: "#C62828",
    CellState.DEPRESSION: "#C62828",
    CellState.CREST_SHADOW: "#D98E04",
    CellState.WATER: "#1E88E5",
    CellState.DYNAMIC: "#C62828",
}
LEGEND = (
    (CellState.GROUND, "Seen ground"),
    (CellState.UNSEEN, "Unseen"),
    (CellState.OCCLUDED, "Occluded"),
    (CellState.DITCH_CANDIDATE, "Ditch / missing ground"),
    (CellState.POSITIVE, "Lethal (obstacle or hole)"),
    (CellState.CREST_SHADOW, "Crest shadow"),
)


@dataclass(frozen=True)
class Scene:
    key: str
    seed: int
    s_m: float  # rover body origin: arc length along the straight start->goal line (m)
    title: str  # short label drawn on the camera image


SCENES = (
    Scene("trail", 120, 8.0, "F1 trail"),
    Scene("ditch", 109, 13.8, "F2 ditch field"),
    Scene("crest", 116, 17.3, "F3 crest"),
)
AE_PREROLL_T_S = (-12.0, -6.0)  # renders before the kept frame (t = 0); > 5 s apart so AE jumps to its target

# ---- layout (pixels at the saved resolution; figure is W_PX wide at DPI) ------------------------
DPI = 200
W_PX = 1800
PT = 72.0 / DPI  # points per pixel
M_X = 38
GAP_X = 30
CAM_W, CAM_H = 640, 400  # native onboard resolution (no resampling of the camera image)
BEV_X = (-0.5, 12.0)  # body-frame display window (m): forward
BEV_Y = (-6.0, 6.0)  # lateral (left positive); the perception grid is +-6 m wide
BEV_PX_PER_M = CAM_H / (BEV_X[1] - BEV_X[0])  # 32 px/m -> panel 384 x 400 px
BEV_W = int(round((BEV_Y[1] - BEV_Y[0]) * BEV_PX_PER_M))
HEAD_H = 64
ROW_GAP = 22
RANGE_TICKS_M = (4, 8, 12)
FS_HEAD, FS_CHIP, FS_LEG, FS_TICK, FS_PROV = 31, 25, 26, 24, 23  # font sizes in px (>= 20 px)


def col_x(k: int) -> int:
    return M_X + k * (CAM_W + GAP_X)


# =========================================================================== scenario geometry (GT)
def load_dev(seed: int) -> dict:
    if seed not in DEV_SEEDS:
        raise ValueError(f"seed {seed} is not a DEV seed (100-129)")
    return json.loads((ROOT / SCEN_DIR / f"{seed}.json").read_text())


def line_frame(sc: dict) -> tuple[np.ndarray, np.ndarray, float]:
    a = np.asarray(sc["start"]["xy"], float)
    b = np.asarray(sc["goal"]["xy"], float)
    L = float(np.linalg.norm(b - a))
    return a, (b - a) / L, L


def polyline_crossings(sc: dict, typ: str) -> list[tuple[float, int]]:
    """(arc length along start->goal, hazard index) where the straight line crosses a hazard polyline."""
    a, u, L = line_frame(sc)
    out = []
    for k, hz in enumerate(sc.get("hazards", [])):
        if hz.get("type") != typ:
            continue
        P = np.asarray(hz["polyline"], float)
        for i in range(len(P) - 1):
            d = P[i + 1] - P[i]
            M = np.array([[u[0], -d[0]], [u[1], -d[1]]])
            if abs(np.linalg.det(M)) < 1e-9:
                continue
            s, r = np.linalg.solve(M, P[i] - a)
            if 0.0 <= r <= 1.0 and 0.0 <= s <= L:
                out.append((float(s), k))
    return sorted(out)


def hazard_facts(sc: dict, scene: Scene) -> dict[str, Any]:
    """Ground-truth description of the hazard ahead of the rover (from the scenario file)."""
    if scene.key in ("ditch", "crest"):
        typ = "ditch" if scene.key == "ditch" else "crest"
        s, k = next((s, k) for s, k in polyline_crossings(sc, typ) if s > scene.s_m)
        hz = sc["hazards"][k]
        f = {"type": typ, "distance_ahead_m": round(s - scene.s_m, 2),
             "how": "straight start->goal line intersected with the hazard centreline polyline; minus the rover's arc length"}
        if typ == "ditch":
            f.update(width_m=round(float(hz["width"]), 2), depth_m=round(float(hz["depth"]), 2))
            f["text"] = f"Trench {f['width_m']:.2f} m wide, {f['depth_m']:.2f} m deep, {f['distance_ahead_m']:.1f} m ahead"
        else:
            f.update(drop_m=round(float(hz["drop"]), 2))
            f["text"] = f"Crest {f['distance_ahead_m']:.1f} m ahead, then a {f['drop_m']:.1f} m drop"
        return f
    # trail: rocks that protrude above the lethal step height inside the displayed window
    from metagross.sim.terrain import Terrain  # read-only: terrain height under each rock

    ter = Terrain.from_scenario(sc)
    a, u, _ = line_frame(sc)
    n = np.array([-u[1], u[0]])
    rocks = []
    for ob in sc.get("objects", []):
        if ob["type"] != "rock":
            continue
        x, y, z = (float(v) for v in ob["xyz"])
        d = np.array([x, y]) - (a + scene.s_m * u)
        bx, by = float(d @ u), float(d @ n)
        protr = z + float(ob["radius"]) * float(ob["squash"]) - float(ter.height_at(x, y))
        if BEV_X[0] < bx < BEV_X[1] and BEV_Y[0] < by < BEV_Y[1]:
            rocks.append({"ahead_m": round(bx, 2), "left_m": round(by, 2), "radius_m": round(float(ob["radius"]), 2),
                          "protrusion_m": round(protr, 2), "lethal": bool(protr > defaults.STEP_LETHAL_M)})
    rocks.sort(key=lambda r: r["ahead_m"])
    others = []
    for ob in sc.get("objects", []):
        if ob["type"] == "rock":
            continue
        d = np.asarray(ob.get("xy") or ob["xyz"][:2], float) - (a + scene.s_m * u)
        bx, by = float(d @ u), float(d @ n)
        if BEV_X[0] < bx < BEV_X[1] and BEV_Y[0] < by < BEV_Y[1]:
            others.append({"type": ob["type"], "ahead_m": round(bx, 2), "left_m": round(by, 2)})
    big = max(rocks, key=lambda r: r["radius_m"])
    return {"type": "rocks", "rocks_in_window": rocks, "other_objects_in_window": sorted(others, key=lambda o: o["ahead_m"]), "step_lethal_m": defaults.STEP_LETHAL_M,
            "text": f"Boulder {big['protrusion_m']:.2f} m tall, centre {big['ahead_m']:.1f} m ahead",
            "how": "rock centres (scenario objects) in the rover body frame; protrusion = rock top - terrain height"}


# =========================================================================== capture (renderer + Perception)
def capture(cache: Path) -> None:
    from metagross.autonomy.perception.pipeline import Perception
    from metagross.contracts.messages import SensorFrame  # noqa: F401  (type of the frame used below)
    from metagross.sim.render.bridge import ThreeRenderer
    from metagross.sim.world import World

    cv2.setNumThreads(2)
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    r = ThreeRenderer(defaults.stereo_calibration(), defaults.VEHICLE)  # same settings as runner.default_renderer_factory
    LOG.info("renderer up in %.0f s: %s", time.perf_counter() - t0, r.renderer_string)
    try:
        for sc_ in SCENES:
            t1 = time.perf_counter()
            sc = load_dev(sc_.seed)
            w = World(sc, sensor_mode="stereo", renderer=r)
            a, u, _ = line_frame(sc)
            xy = a + sc_.s_m * u
            w.vehicle.reset(float(xy[0]), float(xy[1]), math.atan2(u[1], u[0]))
            for tt in AE_PREROLL_T_S:
                r.render_stereo(dict(w.render_state(), t=tt))
            frame = w.make_sensor_frame(w.t, 0)
            per = Perception(w.calibration(), defaults.VEHICLE)
            out = per.process(frame, (0.0, 0.0, 0.0))
            meta = {"seed": sc_.seed, "family": sc["family"], "s_m": sc_.s_m, "pose6_world": [float(v) for v in w.state.pose6()],
                    "renderer": r.renderer_string, "exposure": r.last_js.get("exposure"),
                    "r_vis_m": float(out["r_vis_m"]), "timings_ms": {k: float(v) for k, v in out["timings_ms"].items()},
                    "bev_spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in out["bev_spec"].items()
                                 if k in ("x_min_m", "y_min_m", "res_m", "shape", "index")}}
            np.savez_compressed(cache / f"{sc_.key}.npz", left=frame.left_rgb, right=frame.right_gray,
                                disparity=out["disparity"], cell_state=out["cell_state_local"],
                                missing=out["missing_ground_mask"], meta=np.asarray(json.dumps(meta)))
            LOG.info("%s: seed %d s=%.1f m captured in %.0f s (r_vis %.1f m)", sc_.key, sc_.seed, sc_.s_m,
                     time.perf_counter() - t1, out["r_vis_m"])
    finally:
        r.close()


def load_cache(cache: Path, key: str) -> dict[str, Any]:
    z = np.load(cache / f"{key}.npz", allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


# =========================================================================== panels
def hex_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)


def bev_display(cell_state: np.ndarray, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Sample the egocentric grid (grid[i, j], i forward, j left) at every display pixel centre of the
    BEV panel (forward up, vehicle-left on the left). Returns (RGB image, CellState per pixel)."""
    res, xm, ym = float(spec["res_m"]), float(spec["x_min_m"]), float(spec["y_min_m"])
    nx, ny = cell_state.shape
    cols, rows = np.meshgrid(np.arange(BEV_W) + 0.5, np.arange(CAM_H) + 0.5)
    y = BEV_Y[1] - cols / BEV_PX_PER_M
    x = BEV_X[1] - rows / BEV_PX_PER_M
    i = np.floor((x - xm) / res).astype(int)
    j = np.floor((y - ym) / res).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    st = np.full((CAM_H, BEV_W), int(CellState.UNSEEN), np.uint8)
    st[ok] = cell_state[i[ok], j[ok]]
    lut = np.tile(hex_rgb(CELL_HEX[CellState.UNSEEN]), (256, 1))
    for s, h in CELL_HEX.items():
        lut[int(s)] = hex_rgb(h)
    return lut[st], st


def disparity_rgb(disp: np.ndarray, norm: Normalize) -> np.ndarray:
    valid = disp > 0
    rgb = (matplotlib.colormaps["magma"](norm(np.where(valid, disp, 0.0)))[..., :3] * 255 + 0.5).astype(np.uint8)
    rgb[~valid] = hex_rgb(NO_MATCH)
    return rgb


def bev_to_px(x_m: float, y_m: float) -> tuple[float, float]:
    """Body (x fwd, y left) -> BEV panel pixel (col, row)."""
    return (BEV_Y[1] - y_m) * BEV_PX_PER_M, (BEV_X[1] - x_m) * BEV_PX_PER_M


# =========================================================================== figure helpers
class PxFig:
    """Matplotlib figure addressed in saved-image pixels (x right, y down from the top-left)."""

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI, facecolor=BG)

    def axes(self, x: float, y: float, w: float, h: float) -> plt.Axes:
        ax = self.fig.add_axes((x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h))
        ax.set_axis_off()
        return ax

    def image(self, img: np.ndarray, x: float, y: float, w: float, h: float, nearest: bool = False) -> plt.Axes:
        ax = self.axes(x, y, w, h)
        ax.imshow(img, interpolation="nearest" if nearest else "antialiased", aspect="auto", extent=(0, w, h, 0))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        return ax

    def text(self, x: float, y: float, s: str, px: float, **kw: Any):
        kw.setdefault("color", INK)
        kw.setdefault("va", "top")
        kw.setdefault("ha", "left")
        return self.fig.text(x / self.w, 1 - y / self.h, s, fontsize=px * PT, **kw)

    def save(self, name: str) -> list[str]:
        paths = []
        for ext in ("png", "svg"):
            p = OUT / f"{name}.{ext}"
            self.fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=0.10, facecolor=BG,
                             metadata={"Software": None} if ext == "png" else {"Creator": None, "Date": None})
            paths.append(str(p.relative_to(ROOT)))
        plt.close(self.fig)
        return paths


def chip(ax: plt.Axes, x: float, y: float, s: str, dark: bool = False, weight: str = "normal") -> None:
    """Small label on a photo (x, y in panel pixels, top-left anchored)."""
    ax.text(x, y, s, fontsize=FS_CHIP * PT, color="#FFFFFF" if dark else INK, ha="left", va="top", fontweight=weight,
            bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.5", fc="#1A1A1AB0" if dark else "#FFFFFFE6", ec="none"),
            zorder=5)


def draw_row(pf: PxFig, y: float, d: dict, facts: dict, scene: Scene, norm: Normalize) -> dict[str, Any]:
    meta = d["meta"]
    # 1. camera (native 640 x 400, no resampling)
    ax = pf.image(d["left"], col_x(0), y, CAM_W, CAM_H)
    chip(ax, 12, 12, f"{scene.title} · DEV seed {scene.seed}", weight="bold")
    chip(ax, 12, CAM_H - 50, facts["text"], dark=True)
    # 2. disparity (shared colour scale across rows)
    pf.image(disparity_rgb(d["disparity"], norm), col_x(1), y, CAM_W, CAM_H)
    # 3. bird's-eye cell states
    rgb, st = bev_display(d["cell_state"], meta["bev_spec"])
    ax = pf.image(rgb, col_x(2), y, BEV_W, CAM_H, nearest=True)
    for r_m in RANGE_TICKS_M:
        _, rr = bev_to_px(r_m, 0.0)
        ax.plot([0, BEV_W], [rr, rr], color=INK, lw=0.6, alpha=0.28, zorder=3)
        ax.text(8, rr + 3, f"{r_m} m", fontsize=FS_TICK * PT, color=INK, ha="left", va="top", zorder=4)
    v = defaults.VEHICLE
    c0, r0 = bev_to_px(v.length_m / 2, v.width_m / 2)
    ax.add_patch(Rectangle((c0, r0), v.width_m * BEV_PX_PER_M, v.length_m * BEV_PX_PER_M, fc=NAVY, ec="#FFFFFF", lw=0.8, zorder=4))
    spec = meta["bev_spec"]
    nx, ny = d["cell_state"].shape
    xc = float(spec["x_min_m"]) + (np.arange(nx) + 0.5) * float(spec["res_m"])
    yc = float(spec["y_min_m"]) + (np.arange(ny) + 0.5) * float(spec["res_m"])
    win = ((xc > BEV_X[0]) & (xc < BEV_X[1]))[:, None] & ((yc > BEV_Y[0]) & (yc < BEV_Y[1]))[None, :]
    counts = {CellState(int(s)).name: int(n) for s, n in zip(*np.unique(d["cell_state"][win], return_counts=True))}
    valid = d["disparity"] > 0
    return {"cell_counts_displayed_window_0p1m_cells": counts, "cell_counts_full_grid": {
        CellState(int(s)).name: int(n) for s, n in zip(*np.unique(d["cell_state"], return_counts=True))},
        "disparity_valid_fraction": round(float(valid.mean()), 4),
        "disparity_px_p1_p99": [round(float(np.percentile(d["disparity"][valid], q)), 2) for q in (1, 99)],
        "missing_ground_pixels": int(d["missing"].sum())}


def frame_boxes(pf: PxFig, ys: list[float]) -> None:
    for y in ys:
        for k in range(3):
            w = BEV_W if k == 2 else CAM_W
            ax = pf.axes(col_x(k) - 0.5, y - 0.5, w + 1, CAM_H + 1)
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc="none", ec="#BDBDBD", lw=0.6))


def draw_header(pf: PxFig, y: float) -> None:
    heads = ("Left camera (rendered)", "Disparity from SGBM stereo", "Bird's-eye cell states")
    for k, h in enumerate(heads):
        pf.text(col_x(k), y, h, FS_HEAD, fontweight="bold")


def draw_band(pf: PxFig, y: float, norm: Normalize, seeds: list[int], rows_txt: str) -> None:
    """Legend (cell states), disparity colour bar and provenance below the image grid."""
    # legend: one line, swatch + label, spaced by measured text widths
    r = pf.fig.canvas.get_renderer()
    x = col_x(0)
    sw = 26
    for s, lab in LEGEND:
        ax = pf.axes(x, y + 2, sw, sw)
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc=CELL_HEX[s], ec="#9E9E9E", lw=0.5))
        t = pf.text(x + sw + 10, y + sw / 2 + 2, lab, FS_LEG, va="center")
        wpx = t.get_window_extent(renderer=r).width
        x += sw + 10 + wpx + 38
    # disparity colour bar under the disparity column
    yb = y + 64
    cax = pf.axes(col_x(1), yb + 4, CAM_W, 16)
    grad = np.linspace(norm.vmin, norm.vmax, 512)[None, :]
    cax.imshow(grad, cmap="magma", norm=norm, aspect="auto", extent=(norm.vmin, norm.vmax, 0, 1))
    cax.set_axis_on()
    cax.set_yticks([])
    cax.set_xticks(np.arange(0, norm.vmax + 1e-6, 10 if norm.vmax <= 60 else 20))
    cax.tick_params(axis="x", labelsize=FS_TICK * PT, length=3, width=0.6, color=MUTED, labelcolor=INK, pad=2)
    for sp in cax.spines.values():
        sp.set_visible(False)
    pf.text(col_x(1), yb + 58, "Disparity (px): brighter = nearer, light grey = no valid disparity", FS_TICK, color=MUTED)
    # honesty label under the camera column, provenance line at the bottom-left
    pf.text(col_x(0), yb, "Simulated", FS_CHIP + 2, color=INK, fontweight="bold")
    pf.text(col_x(0), yb + 36, "rendered camera, real stereo matching", FS_CHIP, color=MUTED)
    prov = (f"Simulated, rendered stereo, DEV seed{'s' if len(seeds) > 1 else ''} {rows_txt} · one frame{' each' if len(seeds) > 1 else ''}, onboard "
            f"Perception node, no segmenter · hazard distances: data/scenarios/dev · {GENERATOR}")
    t = pf.text(col_x(0), yb + 104, prov, FS_PROV, color=FAINT)
    grid_w = col_x(2) + BEV_W - col_x(0)
    if t.get_window_extent(renderer=pf.fig.canvas.get_renderer()).width > grid_w:  # wrap at the last separator that fits
        t.remove()
        parts = prov.split(" · ")
        cut = len(parts) - 1
        while cut > 1 and len(" · ".join(parts[:cut])) * FS_PROV * 0.56 > grid_w:
            cut -= 1
        pf.text(col_x(0), yb + 104, " · ".join(parts[:cut]), FS_PROV, color=FAINT)
        pf.text(col_x(0), yb + 134, " · ".join(parts[cut:]), FS_PROV, color=FAINT)


# =========================================================================== compose
def compose(cache: Path) -> None:
    data = {s.key: load_cache(cache, s.key) for s in SCENES}
    facts = {s.key: hazard_facts(load_dev(s.seed), s) for s in SCENES}
    all_valid = np.concatenate([d["disparity"][d["disparity"] > 0] for d in data.values()])
    vmax = float(np.ceil(np.percentile(all_valid, 99.5) / 10.0) * 10.0)
    norm = Normalize(0.0, vmax)
    band_h = 64 + 134 + 30
    outputs: dict[str, Any] = {}

    def build(scenes: tuple[Scene, ...], name: str) -> dict[str, Any]:
        n = len(scenes)
        H = HEAD_H + n * CAM_H + (n - 1) * ROW_GAP + 30 + band_h + 20
        pf = PxFig(W_PX, H)
        draw_header(pf, 14)
        ys = [HEAD_H + k * (CAM_H + ROW_GAP) for k in range(n)]
        rows = []
        for y, sc_ in zip(ys, scenes):
            stats = draw_row(pf, y, data[sc_.key], facts[sc_.key], sc_, norm)
            m = data[sc_.key]["meta"]
            rows.append({"row_key": sc_.key, "seed": sc_.seed, "split": "dev", "family": m["family"],
                         "scenario": f"{SCEN_DIR}/{sc_.seed}.json", "rover_s_along_start_goal_m": sc_.s_m,
                         "rover_pose6_world": [round(v, 4) for v in m["pose6_world"]], "hazard_ground_truth": facts[sc_.key],
                         "perception_r_vis_m": round(m["r_vis_m"], 2),
                         "perception_timings_ms_this_container": {k: round(v, 1) for k, v in m["timings_ms"].items()},
                         **stats})
        frame_boxes(pf, ys)
        draw_band(pf, ys[-1] + CAM_H + 30, norm, [s.seed for s in scenes], ", ".join(str(s.seed) for s in scenes))
        files = pf.save(name)
        rec = {"figure": name, "label": LABEL, "generator": GENERATOR, "files": files,
               "what": "one rendered stereo frame per DEV scenario -> SGBM disparity -> onboard Perception.process() "
                       "(fresh node, single frame, no temporal fusion, no segmenter)",
               "pipeline": {"renderer": "metagross.sim.render.bridge.ThreeRenderer(defaults.stereo_calibration(), defaults.VEHICLE)",
                            "renderer_backend_this_run": data[scenes[0].key]["meta"]["renderer"],
                            "world": "metagross.sim.world.World(scenario, sensor_mode='stereo', renderer)",
                            "placement": "world.vehicle.reset(x, y, yaw) on the straight start->goal line (simulator settles z, roll, pitch)",
                            "auto_exposure_preroll_t_s": list(AE_PREROLL_T_S),
                            "frame": "world.make_sensor_frame(world.t, 0)",
                            "perception": "metagross.autonomy.perception.pipeline.Perception(calib, VEHICLE).process(frame, (0, 0, 0))",
                            "segmenter": None},
               "drawn_numbers": {
                   "hazard_text_on_images": {r["row_key"]: r["hazard_ground_truth"]["text"] for r in rows},
                   "bev_range_ticks_m": list(RANGE_TICKS_M),
                   "bev_window_body_m": {"x_forward": list(BEV_X), "y_left": list(BEV_Y), "px_per_m": BEV_PX_PER_M},
                   "disparity_colour_scale_px": [0.0, vmax],
                   "disparity_scale_how": "0 to the 99.5th percentile of valid disparity over all three frames, rounded up to 10 px",
                   "rover_footprint_m": {"length": defaults.VEHICLE.length_m, "width": defaults.VEHICLE.width_m}},
               "colours": {"cell_states": {s.name: h for s, h in CELL_HEX.items()},
                           "note": "ground = light tint of deck green #2E7D32 for red/green CVD separation (validator: deutan dE 4.2 -> >= 10)",
                           "disparity_cmap": "magma", "no_match": NO_MATCH},
               "caveats": ["Simulated: procedurally generated Three.js worlds, not real imagery.",
                           "Rendered with software WebGL (SwiftShader) in a CPU container; same scene and shaders as the GPU path.",
                           "Single frame from a fresh Perception node: ditch persistence filtering does not apply to a node's first frame; "
                           "the closed loop fuses frames in the rolling map.",
                           "No terrain segmenter (deploy model not trained), so there is no water / semantic cue.",
                           "Closed-loop results are tier-0 only; this gallery shows perception on rendered stereo, not a closed-loop run.",
                           "Perception timings were measured in this 4-vCPU container and are not a claim."],
               "rows": rows}
        (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1))
        LOG.info("wrote %s", files)
        return rec

    outputs["gallery"] = build(SCENES, "perception_gallery")
    for k, sc_ in enumerate(SCENES, start=1):
        outputs[f"row_{k}"] = build((sc_,), f"perception_row_{k}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compose", action="store_true", help="only compose from the cache (no rendering)")
    ap.add_argument("--recapture", action="store_true", help="render even if the cache exists")
    ap.add_argument("--cache", type=Path, default=Path(os.environ.get("MG_GALLERY_CACHE", CACHE_DEFAULT)))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("metagross.sim.render.bridge").setLevel(logging.WARNING)
    missing = [s.key for s in SCENES if not (args.cache / f"{s.key}.npz").exists()]
    if not args.compose and (args.recapture or missing):
        capture(args.cache)
    compose(args.cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
