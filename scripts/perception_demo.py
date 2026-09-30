"""Perception demo, ditch detection-range sweep and timing benchmark.

Usage (from the repo root)::

    python scripts/perception_demo.py            # synthetic (+ KITTI if data/kitti exists) figures
    python scripts/perception_demo.py --sweep    # first-detection range vs Matthies-Rankin theory
    python scripts/perception_demo.py --bench    # process() timing, cv2 threads = 2

Outputs: ``deck_assets/perception_demo_*.png``, ``results/perception_ditch_range.csv``,
``results/perception_timing.json``. All synthetic numbers are *Simulated* (analytic scenes).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere, like the other scripts

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from metagross.autonomy.perception import synthetic as syn  # noqa: E402
from metagross.autonomy.perception.geometry import CameraGeometry  # noqa: E402
from metagross.autonomy.perception.ground import fit_ground  # noqa: E402
from metagross.autonomy.perception.negobs import MissingGroundDetector  # noqa: E402
from metagross.autonomy.perception.pipeline import Perception  # noqa: E402
from metagross.autonomy.perception.stereo import StereoMatcher  # noqa: E402
from metagross.config import defaults  # noqa: E402
from metagross.contracts.messages import CELL_COLORS, CellState, SensorFrame, StereoCalibration  # noqa: E402

LOG = logging.getLogger("perception_demo")
ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "deck_assets"
RESULTS = ROOT / "results"
KITTI_SEQ = ROOT / "data" / "kitti" / "sequences" / "07"
KITTI_CAM_HEIGHT_M = 1.65  # KITTI rig: cameras ~1.65 m above ground, near-level
KITTI_SCALE = 0.5  # half resolution so 64 disparities reach ~3 m
SWEEP_WIDTHS_M = (0.3, 0.5, 0.8)
SWEEP_DEPTH_M = 0.6
SWEEP_X0 = np.round(np.arange(2.5, 11.51, 0.25), 2)
DETECT_FRACTION = 0.5  # >= 50 % of the central column bands must flag the trench
CENTRAL_HALF_WIDTH_PX = 100


# ----------------------------------------------------------------------------- rendering
def state_rgb(state: np.ndarray) -> np.ndarray:
    lut = np.zeros((256, 3), np.uint8)
    for s, col in CELL_COLORS.items():
        lut[int(s)] = col
    return lut[state]


def figure(out: dict, left_rgb: np.ndarray, spec_dict: dict, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    over = left_rgb.copy()
    m = out["missing_ground_mask"]
    over[m] = (0.45 * over[m] + 0.55 * np.array(CELL_COLORS[CellState.DITCH_CANDIDATE])).astype(np.uint8)
    disp = out["disparity"]
    st = out["cell_state_local"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.6, 1.6, 1.0]})
    ax[0].imshow(over)
    ax[0].set_title("left image + missing-ground mask")
    ax[1].imshow(np.where(disp > 0, disp, np.nan), cmap="turbo")
    ax[1].set_title("disparity (px)")
    x_min, y_min, res = spec_dict["x_min_m"], spec_dict["y_min_m"], spec_dict["res_m"]
    nx, ny = spec_dict["shape"]
    ax[2].imshow(state_rgb(st)[::-1, ::-1], extent=(-(y_min + ny * res), -y_min, x_min, x_min + nx * res))
    ax[2].set_xlabel("lateral (m, right +)")
    ax[2].set_ylabel("forward x (m)")
    ax[2].set_title(f"BEV states   r_vis = {out['r_vis_m']:.1f} m")
    handles = [Patch(color=np.array(CELL_COLORS[s]) / 255.0, label=s.name) for s in CellState]
    ax[2].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    for a in ax[:2]:
        a.axis("off")
    t = out["timings_ms"]
    fig.suptitle(f"{title}   (process {t['total']:.0f} ms: stereo {t['stereo']:.0f}, negobs {t['negobs']:.0f})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    LOG.info("wrote %s", path)


# ----------------------------------------------------------------------------- demos
def demo_synthetic() -> None:
    calib = defaults.stereo_calibration()
    p = Perception(calib, defaults.VEHICLE, {})
    scene = syn.Scene(boxes=[syn.Box(3.3, 3.8, 0.8, 1.4, 0.45)], trenches=[syn.Trench(4.6, 0.5, 0.6, -2.5, 3.0)],
                      name="demo")
    left, right, _ = syn.render_stereo_pair(p.geom, scene, seed=11)
    frame = SensorFrame(0.0, 0, left, right, 0.0, 0.0, 0.0)
    p.process(frame)  # warm-up
    out = p.process(frame)
    figure(out, left, out["bev_spec"], "synthetic: rock at 3.3 m + 0.5 m wide, 0.6 m deep trench at 4.6 m (SGBM)", DECK / "perception_demo_synthetic.png")


def kitti_calib(path: Path, scale: float) -> StereoCalibration:
    P = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            P[k.strip()] = np.array([float(x) for x in v.split()]).reshape(3, 4)
    fx, cx, cy = P["P0"][0, 0], P["P0"][0, 2], P["P0"][1, 2]
    baseline = -P["P1"][0, 3] / P["P1"][0, 0]
    img = cv2.imread(str(KITTI_SEQ / "image_0" / "000000.png"), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # level camera
    T[:3, 3] = [0.0, 0.0, KITTI_CAM_HEIGHT_M]
    return StereoCalibration(int(round(w * scale)), int(round(h * scale)), fx * scale, fx * scale, cx * scale, cy * scale,
                             baseline, T)


def demo_kitti(frame_id: int = 0) -> None:
    if not (KITTI_SEQ / "calib.txt").exists():
        LOG.info("no KITTI data at %s - skipping", KITTI_SEQ)
        return
    calib = kitti_calib(KITTI_SEQ / "calib.txt", KITTI_SCALE)
    p = Perception(calib, defaults.VEHICLE, {})
    size = (calib.width, calib.height)
    l = cv2.resize(cv2.imread(str(KITTI_SEQ / "image_0" / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE), size, interpolation=cv2.INTER_AREA)
    r = cv2.resize(cv2.imread(str(KITTI_SEQ / "image_1" / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE), size, interpolation=cv2.INTER_AREA)
    left_rgb = np.repeat(l[..., None], 3, axis=2)
    out = p.process(SensorFrame(0.0, frame_id, left_rgb, r, 0.0, 0.0, 0.0))
    out = p.process(SensorFrame(0.0, frame_id, left_rgb, r, 0.0, 0.0, 0.0))
    figure(out, left_rgb, out["bev_spec"], f"KITTI odometry seq 07 frame {frame_id} (half res, SGBM)", DECK / "perception_demo_kitti.png")


# ----------------------------------------------------------------------------- sweep
def theory_range(width_m: float, cam_h_m: float, fx: float, n_px: float = defaults.MIN_PIXELS_ON_TARGET) -> float:
    """Camera range R solving H*w / (R (R + w)) = n_px / fx (Matthies & Rankin 2003)."""
    k = cam_h_m * width_m * fx / n_px
    return (-width_m + math.sqrt(width_m**2 + 4.0 * k)) / 2.0


def sweep(noises: tuple[float, ...] = (0.0, 0.15)) -> list[dict]:
    geom = CameraGeometry(defaults.stereo_calibration())
    det = MissingGroundDetector(geom)
    rows: list[dict] = []
    central = np.abs(det.uc - geom.cx) < CENTRAL_HALF_WIDTH_PX
    for noise in noises:
        for w in SWEEP_WIDTHS_M:
            hits = []
            for x0 in SWEEP_X0:
                d = syn.render_disparity(geom, syn.make_scene("trench", x0=float(x0), width=w, depth=SWEEP_DEPTH_M),
                                         noise_px=noise, seed=int(x0 * 100))
                model = fit_ground(geom.points_from_disparity(d, 4, 4), d, geom, seed=0)
                res = det.detect(d, model)
                seg = res.ditch
                ok = (seg.x1 > x0 - 0.3) & (seg.x0 < x0 + w + 0.3)
                cols = np.unique(seg.col[ok])
                frac = float(np.isin(np.nonzero(central)[0], cols).mean())
                hits.append((float(x0), frac))
            det_x = [x for x, f in hits if f >= DETECT_FRACTION]
            farthest = max(det_x) if det_x else float("nan")
            # Reliable first-detection range: the vehicle approaching from far first sees the
            # trench at x and keeps seeing it at every closer tested range.
            reliable = float("nan")
            for x, f in sorted(hits):
                if f < DETECT_FRACTION:
                    break
                reliable = x
            first = reliable
            r_th = theory_range(w, defaults.CAM_HEIGHT_M, geom.fx)
            rows.append({
                "mode": "tier0_analytic", "noise_px": noise, "width_m": w, "depth_m": SWEEP_DEPTH_M,
                "theory_range_cam_m": round(r_th, 2), "theory_x_body_m": round(r_th + defaults.CAM_FORWARD_M, 2),
                "first_detection_x_body_m": first, "first_detection_range_cam_m": round(first - defaults.CAM_FORWARD_M, 2),
                "farthest_sporadic_hit_x_body_m": farthest, "detect_fraction_threshold": DETECT_FRACTION,
                "sweep_step_m": 0.25,
                "frac_by_x0": ";".join(f"{x:.2f}:{f:.2f}" for x, f in hits),
            })
            LOG.info("noise %.2f w %.1f: first detection x=%.2f m (theory x=%.2f m)", noise, w, first, r_th + defaults.CAM_FORWARD_M)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "perception_ditch_range.csv"
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    LOG.info("wrote %s", out)
    return rows


# ----------------------------------------------------------------------------- bench
def bench(n: int = 30) -> dict:
    cv2.setNumThreads(2)
    calib = defaults.stereo_calibration()
    p = Perception(calib, defaults.VEHICLE, {})
    left, right, truth = syn.render_stereo_pair(p.geom, syn.make_scene("trench"), seed=5)
    frame = SensorFrame(0.0, 0, left, right, 0.0, 0.0, 0.0)
    tier0 = SensorFrame(0.0, 0, None, None, 0.0, 0.0, 0.0, sensor_mode="tier0_disparity", disparity=truth)
    sgbm_only = StereoMatcher()
    for _ in range(3):
        p.process(frame)
    res: dict[str, list[float]] = {}
    base: list[float] = []
    t0_list: list[float] = []
    for _ in range(n):
        t = time.perf_counter()
        sgbm_only.compute(left, right)  # load reference: full-frame SGBM alone
        base.append((time.perf_counter() - t) * 1e3)
        out = p.process(frame)
        for k, v in out["timings_ms"].items():
            res.setdefault(k, []).append(v)
        t0_list.append(p.process(tier0)["timings_ms"]["total"])
    summary = {k: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90))} for k, v in res.items()}
    summary["tier0_total"] = {"median": float(np.median(t0_list)), "p90": float(np.percentile(t0_list, 90))}
    summary["reference_full_frame_sgbm_alone"] = {"median": float(np.median(base)), "p90": float(np.percentile(base, 90))}
    summary["meta"] = {"n": n, "resolution": [calib.width, calib.height], "cv2_threads": cv2.getNumThreads(),
                       "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                       "note": "wall-clock on a shared 4-core laptop CPU; other jobs were running concurrently"}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "perception_timing.json").write_text(json.dumps(summary, indent=2))
    LOG.info("timing: %s", json.dumps(summary))
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--kitti-frame", type=int, default=0)
    a = ap.parse_args()
    cv2.setNumThreads(2)
    if a.sweep:
        sweep()
    if a.bench:
        bench()
    if not (a.sweep or a.bench):
        demo_synthetic()
        demo_kitti(a.kitti_frame)


if __name__ == "__main__":
    main()
