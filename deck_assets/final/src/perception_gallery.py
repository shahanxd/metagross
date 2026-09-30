"""Perception gallery: what the onboard Perception node makes of ONE rendered stereo frame.

Run from the repo root:

    OMP_NUM_THREADS=2 python deck_assets/final/src/perception_gallery.py              # compose from the stored frames
    OMP_NUM_THREADS=2 python deck_assets/final/src/perception_gallery.py --reprocess  # re-run Perception on the stored images
    MG_ANGLE=swiftshader MG_RENDER_HEADLESS=1 MG_ALLOW_SOFTWARE_GL=1 OMP_NUM_THREADS=2 \\
        python deck_assets/final/src/perception_gallery.py --recapture                # render the frames again (~20 s each)

Three DEV scenarios (seeds 100-129 only, never EVAL), one frame each:

    trail  F1_trail        seed 120  boulder on the trail ahead          -> lethal obstacle (red) + occluded wedge behind it
    ditch  F2_ditch_field  seed 109  1.04 m trench across the path       -> missing ground / ditch (magenta, lethal)
    crest  F3_crest_ditch  seed 116  crest-only CONTROL (no trench)      -> crest shadow (amber, unknown) past the crest top

The frames are ILLUSTRATIVE and HAND-PICKED: 3 of the 20 poses rendered while choosing them (``POSES_RENDERED``).
They are single frames from a fresh Perception node (no temporal fusion; the ditch / positive persistence filters pass a
node's first frame through unfiltered) with no terrain segmenter (deploy model not trained): not a closed-loop result.

How each frame was produced (``capture``; the simulator is used read-only, the same stereo path as the closed-loop runner):

    renderer = ThreeRenderer(stereo_calibration(), VEHICLE)            # = runner.default_renderer_factory()
    world = World(scenario, sensor_mode="stereo", renderer=renderer)
    world.vehicle.reset(x, y, yaw)   # rover on the straight start->goal line, S_M metres from the start
    renderer.render_stereo(state @ t=-12 s, -6 s)                      # auto-exposure pre-roll at the new pose
    frame = world.make_sensor_frame(world.t, 0)
    out = Perception(calib, VEHICLE).process(frame, (0, 0, 0))

The stored frames (``deck_assets/final/perception_frames/*.npz``: left RGB, right grey, disparity, cell states,
missing-ground mask, certified-ground mask, r_det, metadata) were rendered in a CPU container with software WebGL
(SwiftShader). ``--reprocess`` re-runs Perception on the stored images and checks that disparity and cell states are
bit-identical before it stores the certified-ground mask, so the figures can be rebuilt without a renderer.

Ground-truth numbers drawn (hazard distances) are computed here from ``data/scenarios/dev/<seed>.json``; the cell
counts quoted in the captions are computed from the stored cell states. Everything is written to the JSON sidecars.

Outputs (deck_assets/final/): perception_gallery.{png,svg,json} (camera + bird's-eye, 3 scenes side by side) and
perception_row_{1,2,3}.{png,svg,json} (camera | disparity | bird's-eye for one scene).
Label: Simulated (software-rendered camera images, real SGBM stereo matching, onboard Perception node).
"""
from __future__ import annotations

import argparse
import hashlib
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
FRAMES = OUT / "perception_frames"  # stored rendered frames + Perception outputs (in the repo; results/**/*.npz is git-ignored)
SCEN_DIR = "data/scenarios/dev"
DEV_SEEDS = range(100, 130)
GENERATOR = "deck_assets/final/src/perception_gallery.py"
LABEL = "Simulated (software-rendered camera images, real SGBM stereo matching, onboard Perception node)"

# Every pose rendered while choosing the gallery frames (seed, s along start->goal m, lateral m): 16 in the first
# search, 4 more (DEV 110 / 122: crest WITH a hidden trench) in the revision. DEV 110 was not used: in
# results/perception_dev.json its trench is detected only ~2.8 m ahead in tier-0 and some trench cells were certified
# inside the stopping envelope (documented safety gap), so a far-away frame of it would flatter the stack.
POSES_RENDERED = ((102, 8.0, 0.0), (103, 5.8, 0.0), (104, 21.9, 0.0), (108, 0.0, 0.0), (108, 1.0, 0.0), (108, 1.0, 0.8),
                  (108, 25.0, 0.0), (109, 13.8, 0.0), (109, 14.8, 0.0), (116, 17.3, 0.0), (116, 18.3, 0.0), (120, 7.0, 0.0),
                  (120, 8.0, 0.0), (126, 5.0, 0.0), (126, 20.0, 0.0), (128, 20.9, 0.0), (110, 20.4, 0.0), (110, 20.9, 0.0),
                  (122, 17.5, 0.0), (122, 18.0, 0.0))

# ---- palette (deck brief) --------------------------------------------------------------------------------------
NAVY = "#1F497D"
INK = "#1A1A1A"
MUTED = "#5C5C5C"  # secondary text and provenance: 6.7:1 on white
BG = "#FFFFFF"
FRAME_EDGE = "#BDBDBD"
NO_MATCH = "#D6D6D6"  # disparity: no valid stereo match
GREEN_HATCH = "#2E7D32"  # deck green: stripes on certified ground
# Seen ground is a light tint of the deck green: red vs full-strength #2E7D32 is only dE 4.2 for deuteranopes, the tint
# is >= 21 (validator). Certified ground is the tint hatched with deck-green stripes (texture, so CVD-safe).
# Occluded is a cool slate (not the warm #9A9A9A that means TYPICAL elsewhere in the deck).
CELL_HEX: dict[CellState, str] = {
    CellState.GROUND: "#81C784",
    CellState.UNSEEN: "#E6E6E6",
    CellState.OCCLUDED: "#546E7A",
    CellState.DITCH_CANDIDATE: "#C828A0",  # magenta = contract colour CELL_COLORS[DITCH_CANDIDATE]
    CellState.POSITIVE: "#C62828",
    CellState.DEPRESSION: "#C62828",
    CellState.CREST_SHADOW: "#D98E04",
    CellState.WATER: "#1E88E5",
    CellState.DYNAMIC: "#C62828",
}
LETHAL = (CellState.POSITIVE, CellState.DEPRESSION, CellState.DITCH_CANDIDATE)  # = pipeline.LETHAL_STATES (cost 1)
RED = (CellState.POSITIVE, CellState.DEPRESSION)
HATCH_PERIOD_PX, HATCH_ON_PX = 12, 5  # certified-ground stripes (saved-image px)
LEGEND_GROUPS = (
    ("Seen ground", (("ground", "Not certified"), ("certified", "Certified"))),
    ("Lethal", ((CellState.POSITIVE, "Obstacle or step"), (CellState.DITCH_CANDIDATE, "Ditch (missing ground)"))),
    ("Unknown, not free", ((CellState.UNSEEN, "Unseen"), (CellState.OCCLUDED, "Occluded"), (CellState.CREST_SHADOW, "Crest shadow"))),
)


@dataclass(frozen=True)
class Scene:
    key: str
    seed: int
    s_m: float  # rover body origin: arc length along the straight start->goal line (m)
    family_txt: str  # family as printed on the figure


SCENES = (
    Scene("trail", 120, 8.0, "F1 trail"),
    Scene("ditch", 109, 13.8, "F2 ditch field"),
    Scene("crest", 116, 17.3, "F3 crest only"),
)
AE_PREROLL_T_S = (-12.0, -6.0)  # renders before the kept frame (t = 0); > 5 s apart so AE jumps to its target

# ---- ground-truth analysis constants ----------------------------------------------------------------------------
ROCK_MARGIN_M = 0.3  # a lethal cell within rock radius + this is attributed to that rock (3 cells)
TRUNK_MARGIN_M = 0.6  # ... within trunk radius + this to a tree trunk
BUSH_MARGIN_M = 0.3
DITCH_MARGIN_M = 0.3  # a ditch cell within half-width + this of the GT trench centreline is "on the trench"
HALF_CELL = 0.5 * defaults.BEV_RES_M  # cell centre -> cell edge (m), for ranges quoted in captions
FLOOR_INSET_M = 0.1  # stereo points this far inside the GT trench edges count as "inside the trench"
SPURIOUS_BEYOND_M = 1.0  # a connected ditch component reaching farther than this beyond the trench edge is spurious
CORRIDOR_HALF_M = 2.0  # |y| band used for per-column band edges (m)
PROFILE_STEP_M = 0.05  # terrain profile sampling along the line
SLOPE_BASELINE_M = 1.0  # slope measured over this horizontal baseline
DROP_FRACTION = 0.95  # "the ground falls D m over L m": L = distance past the top to reach this fraction of D

# ---- layout (saved-image pixels; the figure is W_PX wide at DPI, shown ~900 px wide on a 1080p slide) --------------
DPI = 200
W_PX = 1800
PT = 72.0 / DPI  # points per pixel
FS_HEAD, FS_TXT, FS_PROV = 40, 36, 32  # px at 1800 px width -> 20 / 18 / 16 px at 900 px
BEV_X0, BEV_X1 = -0.6, 10.6  # body-frame display window, forward (m)
BEV_Y = (-6.0, 6.0)  # lateral window (m) = the full perception grid width
RANGE_LINES_M = (2, 4, 6, 8, 10)
TICK_MARGIN = 92  # px left of each bird's-eye grid for the range labels


def fs(px: float) -> float:
    return px * PT


# =========================================================================== ground truth (scenario files, read-only)
def load_dev(seed: int) -> dict:
    if seed not in DEV_SEEDS:
        raise ValueError(f"seed {seed} is not a DEV seed (100-129)")
    return json.loads((ROOT / SCEN_DIR / f"{seed}.json").read_text())


def line_frame(sc: dict) -> tuple[np.ndarray, np.ndarray, float]:
    a = np.asarray(sc["start"]["xy"], float)
    b = np.asarray(sc["goal"]["xy"], float)
    L = float(np.linalg.norm(b - a))
    return a, (b - a) / L, L


def polyline_crossings(sc: dict, typ: str) -> list[tuple[float, int, float]]:
    """(arc length along start->goal, hazard index, angle line/polyline in deg) where the line crosses a polyline."""
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
                ang = math.degrees(math.acos(min(1.0, abs(float(u @ d) / float(np.linalg.norm(d))))))
                out.append((float(s), k, ang))
    return sorted(out)


def to_body(pose6: list[float], xy: np.ndarray) -> np.ndarray:
    """World (x, y) -> rover body frame (x forward, y left), using the rover's x, y, yaw."""
    x0, y0, yaw = pose6[0], pose6[1], pose6[5]
    d = np.asarray(xy, float) - np.array([x0, y0])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], axis=-1)


def seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance of points p (N, 2) to segment a-b."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def poly_dist(p: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.min(np.stack([seg_dist(p, P[i], P[i + 1]) for i in range(len(P) - 1)]), axis=0)


def hazard_facts(sc: dict, scene: Scene, pose6: list[float]) -> dict[str, Any]:
    """Ground-truth description of the hazard ahead of the rover (scenario file + terrain heightmap)."""
    from metagross.sim.terrain import Terrain  # read-only ground truth (figure generator, not the onboard stack)

    ter = Terrain.from_scenario(sc)
    a, u, L = line_frame(sc)
    f: dict[str, Any] = {"rover_s_m": scene.s_m}
    if scene.key == "ditch":
        s, k, ang = next(c for c in polyline_crossings(sc, "ditch") if c[0] > scene.s_m)
        hz = sc["hazards"][k]
        w = float(hz["width"])
        half_along = 0.5 * w / math.sin(math.radians(ang))
        f.update(type="ditch", width_m=round(w, 3), depth_m=round(float(hz["depth"]), 3),
                 centre_ahead_m=round(s - scene.s_m, 2), crossing_angle_deg=round(ang, 1),
                 near_edge_ahead_m=round(s - half_along - scene.s_m, 2), far_edge_ahead_m=round(s + half_along - scene.s_m, 2),
                 gaps_arc_m=hz.get("gaps"), polyline_body=to_body(pose6, np.asarray(hz["polyline"], float)).round(3).tolist(),
                 how="start->goal line intersected with the trench centreline polyline; edges = centre -/+ (width / 2) / "
                     "sin(crossing angle); minus the rover's arc length")
        f["label"] = f"Trench {f['near_edge_ahead_m']:.1f}–{f['far_edge_ahead_m']:.1f} m ahead"
        return f
    if scene.key == "crest":
        s, k, _ = next(c for c in polyline_crossings(sc, "crest") if c[0] > scene.s_m)
        hz = sc["hazards"][k]
        ss = np.arange(scene.s_m, min(L, scene.s_m + 20.0), PROFILE_STEP_M)
        P = a[None, :] + ss[:, None] * u[None, :]
        z = ter.height_at(P[:, 0], P[:, 1])
        near = ss <= scene.s_m + 10.0
        i_top = int(np.argmax(np.where(near, z, -np.inf)))
        after = slice(i_top, len(ss))
        i_min = i_top + int(np.argmin(z[after][: int(round(12.0 / PROFILE_STEP_M))]))
        drop = float(z[i_top] - z[i_min])
        i_frac = i_top + int(np.argmax(z[i_top:] <= z[i_top] - DROP_FRACTION * drop))
        nb = int(round(SLOPE_BASELINE_M / PROFILE_STEP_M))
        dz = z[i_top + nb: i_min + 1] - z[i_top: i_min + 1 - nb]
        max_down_deg = float(np.degrees(np.arctan(-dz.min() / SLOPE_BASELINE_M))) if dz.size else 0.0
        from metagross.sim.scenario import f3_has_trench

        f.update(type="crest", has_trench=bool(f3_has_trench(scene.seed)), polyline_ahead_m=round(s - scene.s_m, 2),
                 scenario_drop_m=round(float(hz["drop"]), 3), top_ahead_m=round(float(ss[i_top] - scene.s_m), 2),
                 height_drop_m=round(drop, 2), drop_run_m=round(float(ss[i_frac] - ss[i_top]), 1),
                 lowest_ahead_m=round(float(ss[i_min] - scene.s_m), 2), max_downhill_slope_deg_1m=round(max_down_deg, 1),
                 slope_lethal_deg=float(defaults.SLOPE_LETHAL_DEG),
                 how=f"terrain heightmap sampled every {PROFILE_STEP_M} m along the start->goal line: top = highest point within "
                     f"10 m ahead; drop = top minus the lowest point within 12 m past it; run = distance past the top to lose "
                     f"{int(DROP_FRACTION * 100)} % of the drop; slope over a {SLOPE_BASELINE_M} m baseline")
        f["label"] = f"Crest top {f['top_ahead_m']:.1f} m ahead"
        return f
    # trail: rocks / trees / bushes / logs in the displayed window, in the body frame of the actual rover pose
    objs = []
    for ob in sc.get("objects", []):
        xy = np.asarray(ob.get("xy") or ob["xyz"][:2], float)
        bx, by = (float(v) for v in to_body(pose6, xy))
        if not (BEV_X0 - 3 < bx < BEV_X1 + 3 and BEV_Y[0] - 3 < by < BEV_Y[1] + 3):
            continue
        o = {"type": ob["type"], "ahead_m": round(bx, 2), "left_m": round(by, 2)}
        if ob["type"] == "rock":
            protr = float(ob["xyz"][2]) + float(ob["radius"]) * float(ob["squash"]) - float(ter.height_at(xy[0], xy[1]))
            o.update(radius_m=round(float(ob["radius"]), 3), protrusion_m=round(protr, 3), lethal=bool(protr > defaults.STEP_LETHAL_M))
        elif ob["type"] == "tree":
            o.update(trunk_r_m=round(float(ob["trunk_r"]), 3), canopy_r_m=round(float(ob["canopy_r"]), 3), height_m=round(float(ob["height"]), 2))
        elif ob["type"] == "bush":
            o.update(radius_m=round(float(ob["radius"]), 3), height_m=round(float(ob["height"]), 2))
        elif ob["type"] == "log":
            o.update(length_m=round(float(ob["length"]), 3), radius_m=round(float(ob["radius"]), 3), yaw_world=float(ob["yaw"]),
                     xy_world=[float(v) for v in xy])
        objs.append(o)
    objs.sort(key=lambda o: o["ahead_m"])
    rocks = [o for o in objs if o["type"] == "rock" and BEV_X0 < o["ahead_m"] < BEV_X1 and abs(o["left_m"]) < 6]
    big = min((r for r in rocks if r["lethal"]), key=lambda r: r["ahead_m"])
    f.update(type="rocks", boulder=big, objects_near_window=objs, step_lethal_m=defaults.STEP_LETHAL_M,
             how="scenario objects in the body frame of the rendered rover pose; rock protrusion = rock top - terrain height")
    f["label"] = f"Boulder {big['ahead_m']:.1f} m ahead"
    return f


# =========================================================================== frames: capture (renderer) / reprocess
def capture(frames: Path, keys: list[str]) -> None:
    """Render the scenes and run a fresh Perception node on each frame (needs Chrome / SwiftShader)."""
    from metagross.autonomy.perception.pipeline import Perception
    from metagross.sim.render.bridge import ThreeRenderer
    from metagross.sim.world import World

    cv2.setNumThreads(2)
    frames.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    r = ThreeRenderer(defaults.stereo_calibration(), defaults.VEHICLE)  # same settings as runner.default_renderer_factory
    LOG.info("renderer up in %.0f s: %s", time.perf_counter() - t0, r.renderer_string)
    try:
        for sc_ in (s for s in SCENES if s.key in keys):
            sc = load_dev(sc_.seed)
            w = World(sc, sensor_mode="stereo", renderer=r)
            a, u, _ = line_frame(sc)
            xy = a + sc_.s_m * u
            w.vehicle.reset(float(xy[0]), float(xy[1]), math.atan2(u[1], u[0]))
            for tt in AE_PREROLL_T_S:
                r.render_stereo(dict(w.render_state(), t=tt))
            frame = w.make_sensor_frame(w.t, 0)
            out = Perception(w.calibration(), defaults.VEHICLE).process(frame, (0.0, 0.0, 0.0))
            meta = {"seed": sc_.seed, "family": sc["family"], "s_m": sc_.s_m, "pose6_world": [float(v) for v in w.state.pose6()],
                    "renderer": r.renderer_string, "exposure": r.last_js.get("exposure"),
                    "r_vis_m": float(out["r_vis_m"]), "r_det_m": float(out["r_det_m"]),
                    "timings_ms": {k: float(v) for k, v in out["timings_ms"].items()},
                    "bev_spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in out["bev_spec"].items()
                                 if k in ("x_min_m", "y_min_m", "res_m", "shape", "index")}}
            np.savez_compressed(frames / f"{sc_.key}.npz", left=frame.left_rgb, right=frame.right_gray, disparity=out["disparity"],
                                cell_state=out["cell_state_local"], missing=out["missing_ground_mask"],
                                certified=out["certified_local"], meta=np.asarray(json.dumps(meta)))
            LOG.info("%s: seed %d s=%.1f m captured", sc_.key, sc_.seed, sc_.s_m)
    finally:
        r.close()


def load_frame(frames: Path, key: str) -> dict[str, Any]:
    z = np.load(frames / f"{key}.npz", allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


def reprocess(frames: Path, store: bool) -> dict[str, dict]:
    """Re-run a fresh Perception node on the stored left / right images and require bit-identical disparity and cell
    states (and certified mask, once stored). With ``store`` the certified-ground mask and r_det are (re)written into the
    frame files, so the figures need no renderer."""
    from metagross.autonomy.perception.pipeline import Perception
    from metagross.contracts.messages import SensorFrame

    cv2.setNumThreads(2)
    report = {}
    for sc_ in SCENES:
        d = load_frame(frames, sc_.key)
        f = SensorFrame(t=0.0, seq=0, left_rgb=d["left"], right_gray=d["right"], wheel_angle_l_rad=0.0, wheel_angle_r_rad=0.0,
                        gyro_z_rps=0.0, sensor_mode="stereo")
        out = Perception(defaults.stereo_calibration(), defaults.VEHICLE).process(f, (0.0, 0.0, 0.0))
        n_disp = int(np.sum(out["disparity"] != d["disparity"]))
        n_state = int(np.sum(out["cell_state_local"] != d["cell_state"]))
        if n_disp or n_state:
            raise RuntimeError(f"{sc_.key}: re-run differs from the stored frame ({n_disp} disparity px, {n_state} cells); "
                               "the Perception code changed since capture - re-capture or review before using the figure")
        n_cert = int(np.sum(out["certified_local"] != d["certified"])) if "certified" in d else None
        if n_cert and not store:
            raise RuntimeError(f"{sc_.key}: certified mask differs from the stored one ({n_cert} cells); run with --reprocess")
        if store:
            meta = dict(d["meta"], r_det_m=float(out["r_det_m"]), n_floating_points=int(out["n_floating_points"]),
                        reprocessed=time.strftime("%Y-%m-%d %H:%M:%S"))
            arrays = {k: d[k] for k in ("left", "right", "disparity", "cell_state", "missing")}
            np.savez_compressed(frames / f"{sc_.key}.npz", **arrays, certified=out["certified_local"], meta=np.asarray(json.dumps(meta)))
        report[sc_.key] = {"disparity_px_mismatch": n_disp, "cell_state_mismatch": n_state, "certified_mismatch": n_cert,
                           "r_det_m": round(float(out["r_det_m"]), 3), "r_vis_m": float(out["r_vis_m"]),
                           "certified_cells": int(out["certified_local"].sum()), "stored": store}
        LOG.info("%s: re-run identical; r_det %.2f m, %d certified cells", sc_.key, out["r_det_m"], out["certified_local"].sum())
    return report


# =========================================================================== analysis of the stored cell states
def cell_centres(spec: dict, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    res, xm, ym = float(spec["res_m"]), float(spec["x_min_m"]), float(spec["y_min_m"])
    xc = xm + (np.arange(shape[0]) + 0.5) * res
    yc = ym + (np.arange(shape[1]) + 0.5) * res
    return np.meshgrid(xc, yc, indexing="ij")


def per_column_edges(mask: np.ndarray, X: np.ndarray, Y: np.ndarray) -> dict[str, Any]:
    """Median over lateral columns (|y| <= CORRIDOR_HALF_M) of the nearest / farthest x of the mask (cell centres)."""
    near, far = [], []
    for j in range(mask.shape[1]):
        if abs(Y[0, j]) > CORRIDOR_HALF_M or not mask[:, j].any():
            continue
        xs = X[mask[:, j], j]
        near.append(xs.min())
        far.append(xs.max())
    if not near:
        return {}
    return {"n_columns": len(near), "median_near_m": round(float(np.median(near)), 2), "median_far_m": round(float(np.median(far)), 2),
            "p5_near_m": round(float(np.percentile(near, 5)), 2), "p95_far_m": round(float(np.percentile(far, 95)), 2)}


def rng(v: np.ndarray) -> list[float]:
    return [round(float(v.min()), 2), round(float(v.max()), 2)] if v.size else []


def trench_floor_check(d: dict, facts: dict) -> dict[str, Any]:
    """Does stereo see the trench floor? Every valid disparity pixel is back-projected (onboard camera geometry, body
    frame); points strictly inside the GT trench footprint (|y| <= CORRIDOR_HALF_M) are compared with the local lip level
    (median height of points 0.2-1.0 m outside the trench edges)."""
    from metagross.autonomy.perception.pipeline import Perception

    per = Perception(defaults.stereo_calibration(), defaults.VEHICLE)
    pts = per.geom.points_from_disparity(d["disparity"], 1, 1, row_start=per.row_start)
    P = np.stack([pts.x, pts.y], axis=-1)
    dist = poly_dist(P, np.asarray(facts["polyline_body"], float))
    half = 0.5 * facts["width_m"]
    corr = np.abs(pts.y) <= CORRIDOR_HALF_M
    inside = (dist <= half - FLOOR_INSET_M) & corr
    ring = (dist > half + 0.2) & (dist < half + 1.0) & corr
    z_lip = float(np.median(pts.z[ring]))
    dz = pts.z[inside] - z_lip
    return {"points_inside_footprint": int(inside.sum()), "lip_level_z_m": round(z_lip, 3),
            "deeper_than_0p3m_below_lip": int((dz < -0.3).sum()), "min_below_lip_m": round(float(dz.min()), 3) if dz.size else None,
            "trench_depth_m": facts["depth_m"],
            "how": f"all valid disparity pixels back-projected with the onboard CameraGeometry; inside = more than {FLOOR_INSET_M} m "
                   f"inside the GT trench edges and |y| <= {CORRIDOR_HALF_M} m; lip level = median z of points 0.2-1.0 m outside "
                   "the edges. No point more than 0.3 m below the lip means stereo returns nothing from the floor."}


def analyse(d: dict, facts: dict, scene: Scene) -> dict[str, Any]:
    """Cell counts in the displayed window, and attribution of hazard cells to ground-truth objects."""
    cs, cert = d["cell_state"], d["certified"]
    X, Y = cell_centres(d["meta"]["bev_spec"], cs.shape)
    win = (X > BEV_X0) & (X < BEV_X1) & (Y > BEV_Y[0]) & (Y < BEV_Y[1])
    counts = {CellState(int(s)).name: int(n) for s, n in zip(*np.unique(cs[win], return_counts=True))}
    red = np.isin(cs, RED) & win
    mag = (cs == CellState.DITCH_CANDIDATE) & win
    amb = (cs == CellState.CREST_SHADOW) & win
    cx = X[cert]
    a: dict[str, Any] = {
        "window_body_m": {"x_forward": [BEV_X0, BEV_X1], "y_left": list(BEV_Y)},
        "cell_counts_window": counts,
        "certified_cells_window": int((cert & win).sum()),
        "certified_x_max_m": round(float(cx.max()), 2) if cx.size else None,
        "certified_x_max_central_m": round(float(X[cert & (np.abs(Y) <= 1.0)].max()), 2) if (cert & (np.abs(Y) <= 1.0)).any() else None,
        "r_det_m": round(float(d["meta"]["r_det_m"]), 2),
        "r_vis_m": float(d["meta"]["r_vis_m"]),
        "red_cells": int(red.sum()), "magenta_cells": int(mag.sum()), "amber_cells": int(amb.sum()),
        "amber_x_range_m": rng(X[amb]), "amber_x_median_m": round(float(np.median(X[amb])), 2) if amb.any() else None,
    }
    P = np.stack([X, Y], axis=-1)
    if scene.key == "trail":
        objs = facts["objects_near_window"]
        red_p = P[red]
        label = np.full(len(red_p), "none", dtype=object)
        rocks = [o for o in objs if o["type"] == "rock"]
        trees = [o for o in objs if o["type"] == "tree"]
        bushes = [o for o in objs if o["type"] == "bush"]
        big = facts["boulder"]
        # priority: rocks, trunks, bushes, canopy
        for o in rocks:
            dd = np.hypot(red_p[:, 0] - o["ahead_m"], red_p[:, 1] - o["left_m"])
            tag = "boulder" if o is big or (o["ahead_m"], o["left_m"]) == (big["ahead_m"], big["left_m"]) else \
                f"rock {o['protrusion_m']:.2f} m tall at {o['ahead_m']:.1f} m ahead, {o['left_m']:+.1f} m left"
            label[(label == "none") & (dd <= o["radius_m"] + ROCK_MARGIN_M)] = tag
        for o in trees:
            dd = np.hypot(red_p[:, 0] - o["ahead_m"], red_p[:, 1] - o["left_m"])
            label[(label == "none") & (dd <= o["trunk_r_m"] + TRUNK_MARGIN_M)] = "tree trunk"
        for o in bushes:
            dd = np.hypot(red_p[:, 0] - o["ahead_m"], red_p[:, 1] - o["left_m"])
            label[(label == "none") & (dd <= o["radius_m"] + BUSH_MARGIN_M)] = "bush"
        for o in trees:
            dd = np.hypot(red_p[:, 0] - o["ahead_m"], red_p[:, 1] - o["left_m"])
            label[(label == "none") & (dd <= o["canopy_r_m"])] = "under a tree canopy (not near the trunk)"
        u_, n_ = np.unique(label, return_counts=True)
        a["red_attribution"] = {str(k): int(v) for k, v in zip(u_, n_)}
        a["red_attribution_how"] = (f"each red cell (POSITIVE/DEPRESSION) in the window is assigned, in this order, to a rock within "
                                    f"radius + {ROCK_MARGIN_M} m, a tree trunk within trunk radius + {TRUNK_MARGIN_M} m, a bush within "
                                    f"radius + {BUSH_MARGIN_M} m, or a tree canopy (within canopy radius); else 'none'")
        a["red_none_x_range_m"] = rng(red_p[label == "none", 0])
        mp = P[mag]
        dmag = np.hypot(mp[:, 0] - big["ahead_m"], mp[:, 1] - big["left_m"]) - big["radius_m"]
        a["magenta"] = {"x_range_m": rng(mp[:, 0]), "y_range_m": rng(mp[:, 1]),
                        "distance_outside_boulder_footprint_m": rng(dmag),
                        "note": "ditch cells at the near (rover-facing) base of the boulder, inside the rover's corridor"}
        a["occluded_behind_boulder"] = int(((cs == CellState.OCCLUDED) & win & (X > big["ahead_m"]) &
                                            (np.abs(Y - big["left_m"]) < big["radius_m"] + 1.0)).sum())
    elif scene.key == "ditch":
        from scipy import ndimage

        body_poly = np.asarray(facts["polyline_body"], float)
        beyond = np.full(cs.shape, np.inf)  # distance beyond the GT trench edge (m), for ditch cells
        beyond[mag] = poly_dist(P[mag], body_poly) - 0.5 * facts["width_m"]
        on_grid = mag & (beyond <= DITCH_MARGIN_M)
        lab, n = ndimage.label(mag, structure=np.ones((3, 3)))
        comps = []
        for k in range(1, n + 1):
            mk = lab == k
            comps.append({"cells": int(mk.sum()), "x_range_m": rng(X[mk]), "y_range_m": rng(Y[mk]),
                          "max_beyond_trench_edge_m": round(float(beyond[mk].max()), 2),
                          "spurious": bool(beyond[mk].max() > SPURIOUS_BEYOND_M)})
        spur = [c for c in comps if c["spurious"]]
        spur_mask = np.isin(lab, [k + 1 for k, c in enumerate(comps) if c["spurious"]])
        on_grid &= ~spur_mask
        a["ditch_components"] = comps
        a["ditch_on_trench_cells"] = int(on_grid.sum())
        a["ditch_spurious_cells"] = int(sum(c["cells"] for c in spur))
        a["ditch_spurious_x_range_m"] = [min(c["x_range_m"][0] for c in spur), max(c["x_range_m"][1] for c in spur)] if spur else []
        a["ditch_spurious_y_range_m"] = [min(c["y_range_m"][0] for c in spur), max(c["y_range_m"][1] for c in spur)] if spur else []
        a["ditch_band_overshoot_cells"] = int((mag & ~spur_mask & (beyond > DITCH_MARGIN_M)).sum())
        a["ditch_band_edges_corridor"] = per_column_edges(on_grid, X, Y)
        a["ditch_how"] = (f"8-connected components of ditch cells; a component reaching more than {SPURIOUS_BEYOND_M} m beyond the GT "
                          f"trench edge is spurious; band edges: cells within {DITCH_MARGIN_M} m of the GT trench edges, per lateral "
                          f"column with |y| <= {CORRIDOR_HALF_M} m the nearest / farthest cell centre, median over columns; "
                          f"overshoot = band cells 0.3 to {SPURIOUS_BEYOND_M} m beyond the GT edges")
        a["red_x_range_m"] = rng(X[red])
        a["stereo_points_in_trench"] = trench_floor_check(d, facts)
    elif scene.key == "crest":
        g = (cs == CellState.GROUND) & win
        last_g, first_c = [], []
        for j in range(cs.shape[1]):
            if abs(Y[0, j]) > CORRIDOR_HALF_M:
                continue
            i_sh = np.nonzero(amb[:, j])[0]
            if not i_sh.size:
                continue
            first_c.append(X[i_sh[0], j])
            i_g = np.nonzero(g[: i_sh[0], j])[0]
            if i_g.size:
                last_g.append(X[i_g[-1], j])
        a["seen_ground_last_before_shadow_median_m"] = round(float(np.median(last_g)), 2)
        a["crest_shadow_first_x_median_m"] = round(float(np.median(first_c)), 2)
        a["crest_how"] = (f"per lateral column with |y| <= {CORRIDOR_HALF_M} m: nearest CREST_SHADOW cell centre, and the last GROUND "
                          f"cell centre before it; median over columns")
        a["red_x_range_m"] = rng(X[red])
    if scene.key == "trail":
        a["amber_at_sides_far"] = int((amb & (np.abs(Y) > 3.0) & (X > 7.0)).sum())
        a["amber_at_sides_far_how"] = "amber cells with |y| > 3 m and x > 7 m"
    return a


# =========================================================================== rasters
def hex_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)


def bev_raster(cell_state: np.ndarray, certified: np.ndarray, spec: dict, w: int, h: int, ppm: float) -> np.ndarray:
    """RGB bird's-eye image (forward up, vehicle-left on the left) sampled at every display pixel centre; certified
    ground is hatched with deck-green stripes."""
    res, xm, ym = float(spec["res_m"]), float(spec["x_min_m"]), float(spec["y_min_m"])
    nx, ny = cell_state.shape
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    y = BEV_Y[1] - cols / ppm
    x = BEV_X0 + (h - rows) / ppm
    i = np.floor((x - xm) / res).astype(int)
    j = np.floor((y - ym) / res).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    st = np.full((h, w), int(CellState.UNSEEN), np.uint8)
    st[ok] = cell_state[i[ok], j[ok]]
    ce = np.zeros((h, w), bool)
    ce[ok] = certified[i[ok], j[ok]]
    lut = np.tile(hex_rgb(CELL_HEX[CellState.UNSEEN]), (256, 1))
    for s, hx in CELL_HEX.items():
        lut[int(s)] = hex_rgb(hx)
    img = lut[st]
    stripe = ((np.arange(w)[None, :] + np.arange(h)[:, None]) % HATCH_PERIOD_PX) < HATCH_ON_PX
    img[ce & stripe] = hex_rgb(GREEN_HATCH)
    return img


def hatch_swatch(n: int) -> np.ndarray:
    img = np.tile(hex_rgb(CELL_HEX[CellState.GROUND]), (n, n, 1))
    stripe = ((np.arange(n)[None, :] + np.arange(n)[:, None]) % HATCH_PERIOD_PX) < HATCH_ON_PX
    img[stripe] = hex_rgb(GREEN_HATCH)
    return img


def disparity_rgb(disp: np.ndarray, norm: Normalize) -> np.ndarray:
    valid = disp > 0
    rgb = (matplotlib.colormaps["magma"](norm(np.where(valid, disp, 0.0)))[..., :3] * 255 + 0.5).astype(np.uint8)
    rgb[~valid] = hex_rgb(NO_MATCH)
    return rgb


# =========================================================================== figure helpers
class PxFig:
    """Matplotlib figure addressed in saved-image pixels (x right, y down from the top-left)."""

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI, facecolor=BG)
        self.fits: list[tuple[Any, float, str]] = []  # (text, x_max px, what): right edge checked before saving
        self.font_px: list[float] = []

    def axes(self, x: float, y: float, w: float, h: float) -> plt.Axes:
        ax = self.fig.add_axes((x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h))
        ax.set_axis_off()
        return ax

    def image(self, img: np.ndarray, x: float, y: float, w: float, h: float, nearest: bool = False, edge: bool = True) -> plt.Axes:
        ax = self.axes(x, y, w, h)
        ax.imshow(img, interpolation="nearest" if nearest else "antialiased", aspect="auto", extent=(0, w, h, 0))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        if edge:
            ax.add_patch(Rectangle((0, 0), w, h, fc="none", ec=FRAME_EDGE, lw=0.8, zorder=10))
        return ax

    def text(self, x: float, y: float, s: str, px: float, x_max: float | None = None, **kw: Any):
        kw.setdefault("color", INK)
        kw.setdefault("va", "top")
        kw.setdefault("ha", "left")
        t = self.fig.text(x / self.w, 1 - y / self.h, s, fontsize=fs(px), **kw)
        self.font_px.append(px)
        if x_max is not None:
            self.fits.append((t, x_max, s))
        return t

    def check_fits(self) -> None:
        r = self.fig.canvas.get_renderer()
        for t, x1, s in self.fits:
            bb = t.get_window_extent(renderer=r)
            if bb.x1 > x1 + 1:
                raise RuntimeError(f"text overflows its box ({bb.x1:.0f} > {x1:.0f} px): {s!r}")

    def save(self, name: str) -> list[str]:
        self.check_fits()
        paths = []
        for ext in ("png", "svg"):
            p = OUT / f"{name}.{ext}"
            self.fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=0.08, facecolor=BG,
                             metadata={"Software": None} if ext == "png" else {"Creator": None, "Date": None})
            paths.append(str(p.relative_to(ROOT)))
        plt.close(self.fig)
        return paths


def chip(pf: PxFig, ax: plt.Axes, x: float, y: float, s: str, x_max: float, dark: bool = False, weight: str = "normal") -> None:
    """Label on a photo (x, y in panel pixels, top-left anchored); x_max in figure pixels for the fit check."""
    t = ax.text(x, y, s, fontsize=fs(FS_TXT), color="#FFFFFF" if dark else INK, ha="left", va="top", fontweight=weight,
                bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.4", fc="#1A1A1AC8" if dark else "#FFFFFFEB", ec="none"), zorder=5)
    pf.font_px.append(FS_TXT)
    pf.fits.append((t, x_max, s))


def draw_bev(pf: PxFig, d: dict, x: float, y: float, w: int, h: int, ppm: float, rover_label: bool) -> None:
    """Bird's-eye grid at (x + TICK_MARGIN, y) with range lines, range labels in the left margin and the rover."""
    gx = x + TICK_MARGIN
    ax = pf.image(bev_raster(d["cell_state"], d["certified"], d["meta"]["bev_spec"], w, h, ppm), gx, y, w, h, nearest=True)
    for r_m in RANGE_LINES_M:
        rr = h - (r_m - BEV_X0) * ppm
        ax.plot([0, w], [rr, rr], color=INK, lw=0.7, alpha=0.30, zorder=3)
        pf.text(gx - 12, y + rr, f"{r_m} m", FS_TXT, color=MUTED, ha="right", va="center")
    v = defaults.VEHICLE
    c0 = (BEV_Y[1] - v.width_m / 2) * ppm
    r0 = h - (v.length_m / 2 - BEV_X0) * ppm
    ax.add_patch(Rectangle((c0, r0), v.width_m * ppm, v.length_m * ppm, fc=NAVY, ec="#FFFFFF", lw=0.8, zorder=4))
    if rover_label:
        t = ax.text(c0 + v.width_m * ppm + 12, r0 + v.length_m * ppm / 2, "rover", fontsize=fs(FS_TXT), color=NAVY,
                    fontweight="bold", ha="left", va="center", zorder=5)
        pf.font_px.append(FS_TXT)
        pf.fits.append((t, gx + w, "rover"))


def draw_legend(pf: PxFig, xs: list[float], y: float, col_w: float) -> float:
    """Three grouped legend columns at xs; returns the bottom y."""
    sw, row = 36, 50
    bottom = y
    for (head, items), x in zip(LEGEND_GROUPS, xs):
        pf.text(x, y, head, FS_TXT, x_max=x + col_w, fontweight="bold")
        for k, (key, lab) in enumerate(items):
            yy = y + 52 + k * row
            if key == "ground":
                ax = pf.axes(x, yy, sw, sw)
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc=CELL_HEX[CellState.GROUND], ec=FRAME_EDGE, lw=0.6))
            elif key == "certified":
                pf.image(hatch_swatch(sw), x, yy, sw, sw, nearest=True)
            else:
                ax = pf.axes(x, yy, sw, sw)
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc=CELL_HEX[key], ec=FRAME_EDGE, lw=0.6))
            pf.text(x + sw + 14, yy + sw / 2, lab, FS_TXT, x_max=x + col_w, va="center")
            bottom = max(bottom, yy + sw)
    return bottom


def draw_provenance(pf: PxFig, x: float, y: float, x_max: float, lines: list[str]) -> float:
    t = pf.text(x, y, "Simulated", FS_PROV, fontweight="bold", color=INK)
    wpx = t.get_window_extent(renderer=pf.fig.canvas.get_renderer()).width
    pf.text(x + wpx + 10, y, lines[0], FS_PROV, x_max=x_max, color=MUTED)
    for k, s in enumerate(lines[1:], start=1):
        pf.text(x, y + k * 42, s, FS_PROV, x_max=x_max, color=MUTED)
    return y + len(lines) * 42


def provenance_lines(seeds: list[int]) -> list[str]:
    many = len(seeds) > 1
    return [f"· software-rendered stereo (SwiftShader), DEV seed{'s' if many else ''} {', '.join(map(str, seeds))}, "
            f"one frame{' each' if many else ''}",
            f"Illustrative, hand-picked from {len(POSES_RENDERED)} rendered poses · fresh Perception node, no fusion, no segmenter",
            f"Hazard ground truth: data/scenarios/dev · {GENERATOR}"]


# =========================================================================== the two figure types
GAL_COL_W, GAL_GAP, GAL_M = 560, 36, 24
GAL_CAM_H = int(round(GAL_COL_W / 1.6))
GAL_BEV_W = GAL_COL_W - TICK_MARGIN
GAL_PPM = GAL_BEV_W / (BEV_Y[1] - BEV_Y[0])
GAL_BEV_H = int(round((BEV_X1 - BEV_X0) * GAL_PPM))


def gallery(data: dict, facts: dict, scenes: tuple[Scene, ...]) -> tuple[list[str], dict]:
    cx = [GAL_M + k * (GAL_COL_W + GAL_GAP) for k in range(3)]
    y_cam = 108
    y_bev = y_cam + GAL_CAM_H + 16
    y_leg = y_bev + GAL_BEV_H + 34
    H = y_leg + 50 + 3 * 50 + 30 + 3 * 42 + 16
    pf = PxFig(W_PX, H)
    for k, sc_ in enumerate(scenes):
        d, f = data[sc_.key], facts[sc_.key]
        pf.text(cx[k], 0, f["label"], FS_HEAD, x_max=cx[k] + GAL_COL_W, fontweight="bold")
        pf.text(cx[k], 54, f"DEV seed {sc_.seed} · {sc_.family_txt}", FS_TXT, x_max=cx[k] + GAL_COL_W, color=MUTED)
        pf.image(d["left"], cx[k], y_cam, GAL_COL_W, GAL_CAM_H)
        draw_bev(pf, d, cx[k], y_bev, GAL_BEV_W, GAL_BEV_H, GAL_PPM, rover_label=(k == 0))
    bottom = draw_legend(pf, cx, y_leg, GAL_COL_W)
    draw_provenance(pf, cx[0], bottom + 30, W_PX - GAL_M, provenance_lines([s.seed for s in scenes]))
    layout = {"columns": "one scene per column; top: rendered left camera image (640 x 400 shown at 560 x 350), bottom: bird's-eye "
                         "cell states", "bev_px_per_m": round(GAL_PPM, 2)}
    return pf.save("perception_gallery"), {"layout": layout, "min_font_px_at_1800": min(pf.font_px)}


ROW_M, ROW_GAP = 24, 30
ROW_PH = int((W_PX - 2 * ROW_M - 2 * ROW_GAP - TICK_MARGIN) / (3.2 + (BEV_Y[1] - BEV_Y[0]) / (BEV_X1 - BEV_X0)))
ROW_CAM_W = int(round(ROW_PH * 1.6))
ROW_PPM = ROW_PH / (BEV_X1 - BEV_X0)
ROW_BEV_W = int(round((BEV_Y[1] - BEV_Y[0]) * ROW_PPM))


def row_figure(d: dict, f: dict, sc_: Scene, norm: Normalize, k: int) -> tuple[list[str], dict]:
    xc = [ROW_M, ROW_M + ROW_CAM_W + ROW_GAP, ROW_M + 2 * (ROW_CAM_W + ROW_GAP)]
    y_p = 62
    y_b = y_p + ROW_PH + 22
    y_leg = y_b + 150
    H = y_leg + 50 + 3 * 50 + 30 + 3 * 42 + 16
    pf = PxFig(W_PX, H)
    for x, s, xm in ((xc[0], "Left camera (rendered)", xc[0] + ROW_CAM_W), (xc[1], "Disparity (SGBM stereo)", xc[1] + ROW_CAM_W),
                     (xc[2] + TICK_MARGIN, "Bird's-eye map", W_PX)):
        pf.text(x, 0, s, FS_HEAD, x_max=xm, fontweight="bold")
    ax = pf.image(d["left"], xc[0], y_p, ROW_CAM_W, ROW_PH)
    chip(pf, ax, 10, 10, f"DEV seed {sc_.seed} · {sc_.family_txt}", xc[0] + ROW_CAM_W)
    chip(pf, ax, 10, ROW_PH - 56, f["label"], xc[0] + ROW_CAM_W, dark=True)
    pf.image(disparity_rgb(d["disparity"], norm), xc[1], y_p, ROW_CAM_W, ROW_PH)
    draw_bev(pf, d, xc[2], y_p, ROW_BEV_W, ROW_PH, ROW_PPM, rover_label=True)
    # under the camera: honesty label; under the disparity: colour bar
    pf.text(xc[0], y_b, "Software-rendered images,", FS_TXT, x_max=xc[0] + ROW_CAM_W, color=MUTED)
    pf.text(xc[0], y_b + 46, "real stereo matching", FS_TXT, x_max=xc[0] + ROW_CAM_W, color=MUTED)
    cax = pf.axes(xc[1], y_b + 4, ROW_CAM_W, 18)
    grad = np.linspace(norm.vmin, norm.vmax, 512)[None, :]
    cax.imshow(grad, cmap="magma", norm=norm, aspect="auto", extent=(norm.vmin, norm.vmax, 0, 1))
    cax.set_axis_on()
    cax.set_yticks([])
    cax.set_xticks(np.arange(0, norm.vmax + 1e-6, 10))
    cax.tick_params(axis="x", labelsize=fs(FS_TXT), length=4, width=0.6, color=MUTED, labelcolor=INK, pad=3)
    pf.font_px.append(FS_TXT)
    for sp in cax.spines.values():
        sp.set_visible(False)
    pf.text(xc[1], y_b + 76, "Disparity (px), brighter = nearer", FS_TXT, x_max=xc[1] + ROW_CAM_W, color=MUTED)
    lx = [ROW_M, ROW_M + 600, ROW_M + 1200]
    bottom = draw_legend(pf, lx, y_leg, 560)
    draw_provenance(pf, ROW_M, bottom + 30, W_PX - ROW_M, provenance_lines([sc_.seed]))
    return pf.save(f"perception_row_{k}"), {"layout": {"panels": "left camera | SGBM disparity | bird's-eye cell states",
                                                       "camera_shown_px": [ROW_CAM_W, ROW_PH], "bev_px_per_m": round(ROW_PPM, 2)},
                                            "min_font_px_at_1800": min(pf.font_px)}


# =========================================================================== captions (numbers filled from the analysis)
def texts(facts: dict, an: dict) -> dict[str, dict[str, str]]:
    """Slide headlines and captions; every number is filled from the ground-truth facts or the cell analysis."""
    ft, fd, fc = facts["trail"], facts["ditch"], facts["crest"]
    at, ad, ac = an["trail"], an["ditch"], an["crest"]
    b = ft["boulder"]
    o2 = next(o for o in ft["objects_near_window"] if o["type"] == "rock" and o.get("lethal") and o is not b
              and BEV_X0 < o["ahead_m"] < BEV_X1 and abs(o["left_m"]) < BEV_Y[1])
    ra = at["red_attribution"]
    n_rock2 = sum(v for k, v in ra.items() if k.startswith("rock "))
    n_trees = ra.get("tree trunk", 0) + ra.get("under a tree canopy (not near the trunk)", 0)
    eb = ad["ditch_band_edges_corridor"]
    band = f"{eb['median_near_m'] - HALF_CELL:.1f} to {eb['median_far_m'] + HALF_CELL:.1f} m"  # cell centres -> cell edges
    sy, sx = ad["ditch_spurious_y_range_m"], ad["ditch_spurious_x_range_m"]
    cert = {k: an[k]["certified_x_max_central_m"] + HALF_CELL for k in an}  # far edge of the farthest certified cell

    def cert_txt(keys: list[str]) -> str:
        v = sorted(cert[k] for k in keys)
        rng_txt = f"about {v[0]:.1f} m" if len(v) == 1 else f"about {v[0]:.1f} to {v[-1]:.1f} m"
        return (f"Hatched: certified ground, where a 0.3 m design ditch would already be visible (to {rng_txt} here; the design "
                f"ditch resolves to 4.3 m on level ground, Estimated). The speed governor counts only certified ground.")

    one = ("Illustrative frame, hand-picked from 20 rendered poses; a single frame from a fresh Perception node (no fusion, "
           "no segmenter), software-rendered (SwiftShader). Not a closed-loop result.")
    many = ("Illustrative frames, hand-picked from 20 rendered poses; single frames from a fresh Perception node (no fusion, "
            "no segmenter), software-rendered (SwiftShader). Not a closed-loop result.")
    row1 = (f"DEV seed 120 (F1 trail). The boulder ({b['protrusion_m']:.2f} m tall, centre {b['ahead_m']:.1f} m ahead, "
            f"{b['left_m']:.1f} m left) is red (lethal) and the ground behind it is a slate occluded wedge (unknown, not free). "
            f"Of the {at['red_cells']} red cells, {ra.get('boulder', 0)} are at the boulder, {n_rock2} at a second "
            f"{o2['protrusion_m']:.2f} m rock {o2['ahead_m']:.0f} m ahead, {n_trees} at trees (trunks and low canopy) and "
            f"{ra.get('none', 0)} match no object ({at['red_none_x_range_m'][0] - HALF_CELL:.0f} to "
            f"{at['red_none_x_range_m'][1] + HALF_CELL:.1f} m out). "
            f"{at['magenta_cells']} magenta ditch cells sit at the boulder's near base (also lethal). {at['amber_cells']} amber "
            f"crest-shadow cells ({at['amber_at_sides_far']} of them at the far left and right edges, beyond 7 m) are unknown. "
            + cert_txt(["trail"]) + " " + one)
    row2 = (f"DEV seed 109 (F2 ditch field). The trench is {fd['width_m']:.2f} m wide, {fd['depth_m']:.2f} m deep and spans "
            f"{fd['near_edge_ahead_m']:.1f} to {fd['far_edge_ahead_m']:.1f} m ahead (ground truth). Stereo returns no point more than "
            f"0.3 m below the lip inside it (it cannot see the floor), and the missing-ground detector marks the gap as a lethal "
            f"magenta band at about {band} (central 4 m). "
            f"Artefacts: a spurious {ad['ditch_spurious_cells']}-cell ditch streak {-sy[1] - HALF_CELL:.1f} to "
            f"{-sy[0] + HALF_CELL:.1f} m to the right, out to {sx[1] + HALF_CELL:.1f} m; {ad['amber_cells']} amber and "
            f"{ad['red_cells']} red cells farther out; bright horizon mismatches at "
            f"the top of the disparity map. The trench is 3.5 times as wide as the 0.3 m design ditch, so this frame is not a "
            f"detection-range result. " + cert_txt(["ditch"]) + " " + one)
    row3 = (f"DEV seed 116 (F3 crest-ditch family, crest-only control: no trench). Seen ground gives way to amber crest shadow "
            f"about {ac['crest_shadow_first_x_median_m'] - HALF_CELL:.1f} m ahead, right at the crest top ({fc['top_ahead_m']:.1f} m, "
            f"ground truth): unknown, not free. Behind the crest the ground falls {fc['height_drop_m']:.1f} m over about "
            f"{fc['drop_run_m']:.0f} m (steepest {fc['max_downhill_slope_deg_1m']:.0f}°, below the {fc['slope_lethal_deg']:.0f}° "
            f"lethal slope), so it is drivable here, but the rover cannot know that from this viewpoint. Bright specks at the top "
            f"of the disparity map are horizon mismatches. " + cert_txt(["crest"]) + " " + one)
    gal = (f"Three DEV worlds, one column each: the rendered left camera image and the onboard Perception node's bird's-eye cell "
           f"states (forward is up, rover in navy). Trail (seed 120): the boulder {b['ahead_m']:.1f} m ahead is red and the ground "
           f"behind it is occluded; the other red cells are trees, a second {o2['protrusion_m']:.2f} m rock at {o2['ahead_m']:.0f} m "
           f"and {ra.get('none', 0)} cells that match no object. Ditch field (seed 109): the {fd['width_m']:.2f} m trench "
           f"({fd['near_edge_ahead_m']:.1f} to {fd['far_edge_ahead_m']:.1f} m ahead) becomes a magenta band at about {band}, plus a "
           f"spurious {ad['ditch_spurious_cells']}-cell streak to the right. Crest-only control (seed 116, no trench): crest shadow "
           f"starts at the crest top {fc['top_ahead_m']:.1f} m ahead; the ground beyond falls {fc['height_drop_m']:.1f} m over about "
           f"{fc['drop_run_m']:.0f} m and is drivable, but the rover cannot see it. " + cert_txt(["trail", "ditch", "crest"]) + " " + many)
    return {
        "gallery": {"headline": "From one rendered stereo frame, the perception node marks the boulder and the trench as lethal, and "
                                "marks ground it could not see (behind the boulder, past the crest) as unknown, not free",
                    "caption": gal},
        "row_1": {"headline": "A boulder on the trail becomes a lethal obstacle, and the ground hidden behind it stays unknown "
                              "instead of free", "caption": row1},
        "row_2": {"headline": "Stereo gets nothing back from the trench floor; the missing-ground detector turns that gap into a "
                              "lethal ditch band about 5 m ahead", "caption": row2},
        "row_3": {"headline": "Past a crest the ground drops out of view, so the rover marks it as unknown, not free, even though "
                              "here it happens to be drivable", "caption": row3},
    }


# =========================================================================== compose
def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compose(frames: Path, reprocess_report: dict | None) -> None:
    data = {s.key: load_frame(frames, s.key) for s in SCENES}
    facts = {s.key: hazard_facts(load_dev(s.seed), s, data[s.key]["meta"]["pose6_world"]) for s in SCENES}
    an = {s.key: analyse(data[s.key], facts[s.key], s) for s in SCENES}
    all_valid = np.concatenate([d["disparity"][d["disparity"] > 0] for d in data.values()])
    p995 = float(np.percentile(all_valid, 99.5))
    vmax = float(np.ceil(p995 / 10.0) * 10.0)
    norm = Normalize(0.0, vmax)
    tx = texts(facts, an)

    top_mismatch = {}
    for s in SCENES:  # bright disparity near the top of the searched rows = horizon mismatches (true disparity there < 2 px)
        dsp = data[s.key]["disparity"]
        rows = np.nonzero((dsp > 0).any(axis=1))[0]
        band = dsp[rows[0]: rows[0] + 40] if rows.size else dsp[:0]
        top_mismatch[s.key] = {"first_valid_row": int(rows[0]) if rows.size else None,
                               "px_above_20px_in_top_40_rows": int((band > 20).sum())}

    def record(name: str, files: list[str], extra: dict, scenes: tuple[Scene, ...], tkey: str) -> None:
        rows = []
        for s in scenes:
            m = data[s.key]["meta"]
            rows.append({"row_key": s.key, "seed": s.seed, "split": "dev", "family": m["family"], "scenario": f"{SCEN_DIR}/{s.seed}.json",
                         "frame_npz": str((frames / f"{s.key}.npz").relative_to(ROOT)) if frames.is_relative_to(ROOT) else str(frames / f"{s.key}.npz"),
                         "frame_npz_sha256": sha256(frames / f"{s.key}.npz"),
                         "rover_s_along_start_goal_m": s.s_m, "rover_pose6_world": [round(v, 4) for v in m["pose6_world"]],
                         "renderer_backend": m["renderer"], "hazard_ground_truth": facts[s.key], "cell_analysis": an[s.key],
                         "disparity_valid_fraction": round(float((data[s.key]["disparity"] > 0).mean()), 4),
                         "disparity_top_band": top_mismatch[s.key],
                         "perception_timings_ms_this_container_not_a_claim": {k: round(v, 1) for k, v in m["timings_ms"].items()}})
        drawn = [{"what": f"{r['row_key']}: hazard label", "value": facts[r["row_key"]]["label"],
                  "source": f"{SCEN_DIR}/{r['seed']}.json", "how": facts[r["row_key"]]["how"]} for r in rows]
        drawn += [
            {"what": "bird's-eye range lines", "value": list(RANGE_LINES_M), "unit": "m ahead of the body origin", "source": "display"},
            {"what": "bird's-eye window", "value": {"x_forward_m": [BEV_X0, BEV_X1], "y_left_m": list(BEV_Y)}, "source": "display; "
             "perception grid is x -2..14 m, y +/-6 m, 0.1 m cells (defaults.BEV_LOCAL_*)"},
            {"what": "certified ground (hatched)", "value": {r["row_key"]: an[r["row_key"]]["certified_x_max_central_m"] for r in rows},
             "unit": "farthest certified cell centre within |y| <= 1 m (m); captions quote its far edge (+0.05 m)",
             "source": "Perception.process()['certified_local'] on the stored frame (re-run and checked at every build)"},
        ]
        if tkey != "gallery":  # the gallery has no disparity column
            drawn.append({"what": "disparity colour scale", "value": [0.0, vmax], "unit": "px",
                          "source": f"99.5th percentile of valid disparity over the three frames ({p995:.2f} px), rounded up to 10 px"})
        rec = {"figure": name, "label": LABEL, "generator": GENERATOR, "files": files, **extra,
               "headline": tx[tkey]["headline"], "caption": tx[tkey]["caption"],
               "selection": {"hand_picked": True, "poses_rendered": len(POSES_RENDERED), "seeds_rendered": sorted({p[0] for p in POSES_RENDERED}),
                             "poses": [list(p) for p in POSES_RENDERED],
                             "criterion": "clean single-frame bird's-eye maps that show one hazard type each (not random)"},
               "pipeline": {"renderer": "metagross.sim.render.bridge.ThreeRenderer(defaults.stereo_calibration(), defaults.VEHICLE)",
                            "world": "metagross.sim.world.World(scenario, sensor_mode='stereo', renderer)",
                            "placement": "world.vehicle.reset(x, y, yaw) on the straight start->goal line (simulator settles z, roll, pitch)",
                            "auto_exposure_preroll_t_s": list(AE_PREROLL_T_S), "frame": "world.make_sensor_frame(world.t, 0)",
                            "perception": "metagross.autonomy.perception.pipeline.Perception(calib, VEHICLE).process(frame, (0, 0, 0))",
                            "segmenter": None, "reprocess_check": reprocess_report},
               "drawn_numbers": drawn,
               "caption_numbers": caption_numbers(facts, an, [s.key for s in scenes]),
               "registered_claims_quoted": [{"text": "a 0.3 m design ditch resolves to 4.3 m on level ground (0.9 m mast)", "value": "4.30 m",
                                             "label": "Estimated", "source": "results/claims.csv row 2 -> results/theory.json#ditch_detection"}],
               "colours": {"cell_states": {s.name: h for s, h in CELL_HEX.items()}, "certified_hatch": GREEN_HATCH, "rover": NAVY,
                           "disparity_cmap": "magma", "no_match": NO_MATCH,
                           "note": "occluded is slate #546E7A (not the deck's TYPICAL grey); ground is a tint of deck green for red/green CVD "
                                   "separation; certified ground is hatched (texture) instead of a second green"},
               "caveats": ["Simulated: procedurally generated Three.js worlds, not real imagery.",
                           "Software-rendered (SwiftShader) in a CPU container; CLAUDE.md calls this path smoke-test grade. Same scenes and "
                           "shaders as the GPU path, but no GPU-rendered comparison was made for these frames.",
                           "Hand-picked illustrative frames (see selection); single frames from a fresh node: the ditch and positive "
                           "persistence filters pass a node's first frame through unfiltered and nothing is fused over time.",
                           "No terrain segmenter (deploy model not trained): no water / semantic cue.",
                           "Not a closed-loop result: closed-loop results are tier-0 only.",
                           "Numbers here are not yet in results/claims.csv; see proposed_claims."],
               "proposed_claims": proposed_claims(facts, an, scenes),
               "rows": rows}
        (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1, default=str))
        LOG.info("wrote %s", files)

    files, extra = gallery(data, facts, SCENES)
    record("perception_gallery", files, extra, SCENES, "gallery")
    for k, s in enumerate(SCENES, start=1):
        files, extra = row_figure(data[s.key], facts[s.key], s, norm, k)
        record(f"perception_row_{k}", files, extra, (s,), f"row_{k}")


def caption_numbers(facts: dict, an: dict, keys: list[str]) -> list[dict]:
    """Every number the captions quote, with where it lives in this JSON (rows[*].hazard_ground_truth / cell_analysis)."""
    out = []

    def add(key: str, what: str, value: Any, where: str) -> None:
        if key in keys:
            out.append({"row": key, "what": what, "value": value, "json": where})

    ft, fd, fc, at, ad, ac = facts["trail"], facts["ditch"], facts["crest"], an["trail"], an["ditch"], an["crest"]
    add("trail", "boulder protrusion, centre ahead, left (m)", [ft["boulder"]["protrusion_m"], ft["boulder"]["ahead_m"], ft["boulder"]["left_m"]],
        "hazard_ground_truth.boulder")
    add("trail", "red cells in the window and their attribution", {"total": at["red_cells"], **at["red_attribution"]}, "cell_analysis.red_attribution")
    add("trail", "red cells matching no object: x range (cell centres, m)", at["red_none_x_range_m"], "cell_analysis.red_none_x_range_m")
    add("trail", "magenta cells at the boulder base", at["magenta_cells"], "cell_analysis.magenta")
    add("trail", "amber cells; of them at |y| > 3 m and x > 7 m", [at["amber_cells"], at["amber_at_sides_far"]], "cell_analysis.amber_at_sides_far")
    add("ditch", "trench width, depth, near / far edge ahead (m)", [fd["width_m"], fd["depth_m"], fd["near_edge_ahead_m"], fd["far_edge_ahead_m"]],
        "hazard_ground_truth")
    add("ditch", "magenta band median near / far cell centre, central 4 m (m)",
        [ad["ditch_band_edges_corridor"]["median_near_m"], ad["ditch_band_edges_corridor"]["median_far_m"]], "cell_analysis.ditch_band_edges_corridor")
    add("ditch", "spurious streak: cells, x range, y range (cell centres, m)",
        [ad["ditch_spurious_cells"], ad["ditch_spurious_x_range_m"], ad["ditch_spurious_y_range_m"]], "cell_analysis.ditch_components")
    add("ditch", "amber and red cells", [ad["amber_cells"], ad["red_cells"]], "cell_analysis.cell_counts_window")
    add("ditch", "stereo points > 0.3 m below the lip inside the trench", ad["stereo_points_in_trench"]["deeper_than_0p3m_below_lip"],
        "cell_analysis.stereo_points_in_trench")
    add("ditch", "trench width / 0.3 m design ditch", round(fd["width_m"] / defaults.DESIGN_DITCH_WIDTH_M, 2), "hazard_ground_truth.width_m")
    add("crest", "crest top ahead; drop over run; steepest slope; lethal slope",
        [fc["top_ahead_m"], fc["height_drop_m"], fc["drop_run_m"], fc["max_downhill_slope_deg_1m"], fc["slope_lethal_deg"]], "hazard_ground_truth")
    add("crest", "first crest-shadow cell centre, median over the central 4 m (m)", ac["crest_shadow_first_x_median_m"],
        "cell_analysis.crest_shadow_first_x_median_m")
    for k in ("trail", "ditch", "crest"):
        add(k, "certified ground far edge (m)", round(an[k]["certified_x_max_central_m"] + HALF_CELL, 2), "cell_analysis.certified_x_max_central_m")
    out.append({"row": "all", "what": "0.3 m design ditch resolvable range (m), Estimated", "value": 4.30,
                "json": "registered: results/claims.csv row 2 -> results/theory.json#ditch_detection"})
    out.append({"row": "all", "what": "poses rendered while choosing the frames", "value": len(POSES_RENDERED), "json": "selection"})
    return out


def proposed_claims(facts: dict, an: dict, scenes: tuple[Scene, ...]) -> list[dict]:
    """Rows for results/claims.csv (not written here: this generator only writes deck_assets/final)."""
    out = []
    keys = {s.key for s in scenes}
    if "trail" in keys:
        b = facts["trail"]["boulder"]
        out += [{"id": "percgal_trail120_boulder_protrusion_m", "value": b["protrusion_m"], "label": "Simulated", "source": "data/scenarios/dev/120.json (GT)"},
                {"id": "percgal_trail120_boulder_ahead_m", "value": b["ahead_m"], "label": "Simulated", "source": "data/scenarios/dev/120.json (GT)"}]
    if "ditch" in keys:
        f, e = facts["ditch"], an["ditch"]["ditch_band_edges_corridor"]
        out += [{"id": "percgal_ditch109_width_m", "value": f["width_m"], "label": "Simulated", "source": "data/scenarios/dev/109.json (GT)"},
                {"id": "percgal_ditch109_edges_ahead_m", "value": [f["near_edge_ahead_m"], f["far_edge_ahead_m"]], "label": "Simulated",
                 "source": "data/scenarios/dev/109.json (GT)"},
                {"id": "percgal_ditch109_band_median_edges_m", "value": [e["median_near_m"], e["median_far_m"]], "label": "Simulated",
                 "source": "Perception single frame, deck_assets/final/perception_frames/ditch.npz"}]
    if "crest" in keys:
        f = facts["crest"]
        out += [{"id": "percgal_crest116_top_ahead_m", "value": f["top_ahead_m"], "label": "Simulated", "source": "data/scenarios/dev/116.json (GT heightmap)"},
                {"id": "percgal_crest116_drop_m_over_run_m", "value": [f["height_drop_m"], f["drop_run_m"]], "label": "Simulated",
                 "source": "data/scenarios/dev/116.json (GT heightmap)"}]
    for s in scenes:
        out.append({"id": f"percgal_{s.key}{s.seed}_certified_to_m", "value": round(an[s.key]["certified_x_max_central_m"] + HALF_CELL, 2), "label": "Simulated",
                    "source": f"Perception single frame, deck_assets/final/perception_frames/{s.key}.npz"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recapture", action="store_true", help="render the frames again (needs Chrome / SwiftShader)")
    ap.add_argument("--reprocess", action="store_true", help="re-run Perception on the stored images (checks bit-identity)")
    ap.add_argument("--frames", type=Path, default=FRAMES)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("metagross.sim.render.bridge").setLevel(logging.WARNING)
    frames = args.frames.resolve()
    missing = [s.key for s in SCENES if not (frames / f"{s.key}.npz").exists()]
    if args.recapture or missing:
        capture(frames, [s.key for s in SCENES] if args.recapture else missing)
    store = args.reprocess or any("certified" not in np.load(frames / f"{s.key}.npz").files for s in SCENES)
    report = reprocess(frames, store=store)  # always re-run Perception on the stored images and require identity
    compose(frames, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
