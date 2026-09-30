"""Tier-0 odometry scale error on DEV drives: wheel + gyro vs wheel + gyro + depth odometry (Simulated).

Two steps:

``record``   scripted drive on a DEV scenario (seeds 100-129 only) in **Tier-0** mode through the real
             :class:`~metagross.sim.world.World` (terrain-following vehicle with the scenario's
             longitudinal slip, encoders, gyro, Tier-0 synthetic depth sensor with its noise /
             dropout / lighting model and dynamic obstacles). The vehicle follows the GT oracle path
             (:func:`metagross.sim.gridplan.plan_path`) with the pure-pursuit controller and speed
             schedule of :mod:`metagross.eval.integrity_sim`. Per frame the sensor outputs (disparity at
             the sensor's native 320 x 200, float32 px; wheel angles; gyro) and GT (pose6) are cached in
             ``results/raw/odometry_dev/<seed>.npz`` (git-ignored).
``analyse``  replays the cached frames through the onboard :class:`Localizer` twice (depth odometry
             off / on; everything else identical, wheel + gyro EKF) and scores against GT:

             * along-track scale error = (estimated planar path length / GT planar path length - 1),
               in % (positive = over-count), path lengths summed frame to frame;
             * final position error (m and % of GT path length), A-frame (launch pose) coordinates;
             * depth-odometry acceptance and compute (ms per frame, this machine, shared load).

             Seeds are split into *tuning* seeds (parameters were chosen looking at these) and
             *holdout* seeds (never looked at before the final run); both are reported.

Frames: A-frame = launch pose (x forward, y left, z up), metres / rad. The replayed disparity is the
cached native map nearest-upsampled x2, which reproduces the sensor's 640 x 400 output exactly.

Usage (PowerShell)::

    .venv\\Scripts\\python.exe -m metagross.eval.odometry_dev record --seeds 100 101
    .venv\\Scripts\\python.exe -m metagross.eval.odometry_dev analyse --tune 100 101 --holdout 120 121
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from metagross.config import defaults

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / "results" / "raw" / "odometry_dev"
RESULTS_JSON = REPO_ROOT / "results" / "odometry_dev.json"
DEV_SEEDS = range(100, 130)  # never EVAL seeds 0-59
MAX_DRIVE_S = 75.0  # per scripted drive (s)
UPSAMPLE = 2  # Tier-0 output = native map nearest-upsampled by this factor
THREADS = 2  # shared machine: OpenCV / BLAS threads


# ============================================================================ record
def record_drive(seed: int, out_root: Path = CACHE_ROOT, max_s: float = MAX_DRIVE_S) -> Path:
    """Scripted oracle-path Tier-0 drive on DEV ``seed``; caches sensor data + GT. Returns the file."""
    from metagross.contracts.messages import DriveMode, WheelCmd
    from metagross.eval.integrity_sim import PurePursuit, drive_speed, load_dev_scenario, wheel_rates
    from metagross.sim.gridplan import plan_path
    from metagross.sim.world import World

    sc = load_dev_scenario(seed)
    world = World(sc, sensor_mode="tier0", terminal=frozenset())
    hz = world.hazards
    plan = plan_path(hz.grid, hz.hazard, tuple(world.start_xy), tuple(world.goal_xy))
    if not plan.ok:
        raise RuntimeError(f"seed {seed}: no oracle path")
    speed = drive_speed(seed, sc, plan.length_m)
    ctrl = PurePursuit(plan.path_xy, speed)
    steps_per_frame = int(round(defaults.PHYSICS_HZ / defaults.CAMERA_HZ_BATCH))
    rec: dict[str, list] = {k: [] for k in ("t", "wl", "wr", "gyro", "pose6", "odo_m", "slip", "disp")}
    k = 0
    t0 = time.perf_counter()
    goal_stop = 1.5  # m, as integrity_sim.GOAL_STOP_M
    while world.t <= max_s + 1e-9:
        fr = world.make_sensor_frame(world.t, k)
        s = world.state
        rec["t"].append(fr.t)
        rec["wl"].append(fr.wheel_angle_l_rad)
        rec["wr"].append(fr.wheel_angle_r_rad)
        rec["gyro"].append(fr.gyro_z_rps)
        rec["pose6"].append(s.pose6())
        rec["odo_m"].append(s.odo_m)
        rec["slip"].append(float(getattr(s, "slip", 0.0)))
        rec["disp"].append(np.asarray(fr.disparity, np.float32)[::UPSAMPLE, ::UPSAMPLE].copy())
        if math.hypot(world.goal_xy[0] - s.x, world.goal_xy[1] - s.y) < goal_stop:
            break
        v, om = ctrl.command(s.x, s.y, s.yaw)
        wl, wr = wheel_rates(v, om)
        world.queue_command(WheelCmd(world.t, k, wl, wr, DriveMode.NOMINAL, 0.0))
        for _ in range(steps_per_frame):
            world.step()
        k += 1
    out_root.mkdir(parents=True, exist_ok=True)
    out = out_root / f"{seed}.npz"
    arrays = {key: np.asarray(v) for key, v in rec.items()}
    np.savez_compressed(out, seed=seed, family=sc["family"], speed_mps=speed,
                        slip_long=float(sc.get("vehicle", {}).get("slip_long", 0.0)), **arrays)
    LOG.info("seed %d (%s): %d frames, %.1f m, %.0f s wall", seed, sc["family"], len(rec["t"]),
             world.state.odo_m, time.perf_counter() - t0)
    return out


# ============================================================================ replay
@dataclass
class Drive:
    """Cached Tier-0 drive (see :func:`record_drive`)."""

    seed: int
    family: str
    speed_mps: float
    slip_long: float
    t: np.ndarray
    wl: np.ndarray
    wr: np.ndarray
    gyro: np.ndarray
    pose6: np.ndarray
    disp: np.ndarray  # (N, H/2, W/2) float32 native disparity

    @classmethod
    def load(cls, seed: int, root: Path = CACHE_ROOT) -> "Drive":
        z = np.load(root / f"{seed}.npz", allow_pickle=False)
        return cls(int(z["seed"]), str(z["family"]), float(z["speed_mps"]), float(z["slip_long"]), z["t"], z["wl"],
                   z["wr"], z["gyro"], z["pose6"], z["disp"])

    def gt_xy_a(self) -> np.ndarray:
        """(N, 2) GT planar positions in the A-frame (launch pose)."""
        x0, y0, yaw0 = self.pose6[0, 0], self.pose6[0, 1], self.pose6[0, 5]
        c, s = math.cos(yaw0), math.sin(yaw0)
        d = self.pose6[:, :2] - np.array([x0, y0])
        return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)


def replay(drive: Drive, use_depth_odom: bool, loc_config: Any = None) -> dict[str, np.ndarray]:
    """Run the onboard Localizer (wheel + gyro EKF, depth odometry on/off) over a cached drive.

    Returns per-frame arrays: pose (N, 3), loc_ms, do_ms, do_accepted, do_ran, do_sigma_fwd, do_dx, slip."""
    import cv2

    from metagross.autonomy.localization.localizer import Localizer, LocalizerConfig
    from metagross.contracts.messages import SensorFrame

    cv2.setNumThreads(THREADS)
    calib = defaults.stereo_calibration()
    cfg = loc_config if loc_config is not None else LocalizerConfig()
    cfg.use_depth_odom = use_depth_odom
    loc = Localizer(calib, defaults.VEHICLE, cfg)
    n = drive.t.size
    out = {"pose": np.zeros((n, 3)), "loc_ms": np.zeros(n), "do_ms": np.zeros(n), "do_accepted": np.zeros(n, bool),
           "do_ran": np.zeros(n, bool), "do_sigma_fwd": np.full(n, np.nan), "do_dx": np.full(n, np.nan),
           "slip": np.zeros(n)}
    reasons: dict[str, int] = {}
    for k in range(n):
        disp = np.repeat(np.repeat(drive.disp[k], UPSAMPLE, axis=0), UPSAMPLE, axis=1)
        fr = SensorFrame(t=float(drive.t[k]), seq=k, left_rgb=None, right_gray=None,
                         wheel_angle_l_rad=float(drive.wl[k]), wheel_angle_r_rad=float(drive.wr[k]),
                         gyro_z_rps=float(drive.gyro[k]), sensor_mode="tier0_disparity", disparity=disp)
        res = loc.update(fr, disp)
        out["pose"][k] = res["pose_xy_yaw"]
        out["loc_ms"][k] = res["timings_ms"]["total"]
        out["slip"][k] = res["slip"]
        di = res.get("depth_odom") or {}
        out["do_ms"][k] = float(di.get("ms", 0.0))
        out["do_accepted"][k] = bool(di.get("accepted", False))
        out["do_ran"][k] = bool(di.get("ran", False))
        out["do_sigma_fwd"][k] = float(di.get("sigma_fwd_m", np.nan))
        out["do_dx"][k] = float(di.get("dx", np.nan))
        if di.get("ran") and not di.get("accepted"):
            key = str(di.get("reason", "")).split(" ")[0] or "gate"
            reasons[key] = reasons.get(key, 0) + 1
    out["reasons"] = reasons  # type: ignore[assignment]
    return out


def path_length(xy: np.ndarray) -> float:
    """Planar polyline length (m)."""
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0


def score(drive: Drive, rep: dict[str, Any]) -> dict[str, Any]:
    """Scale / final-position errors of one replay vs GT (A-frame, planar)."""
    gt = drive.gt_xy_a()
    est = rep["pose"][:, :2]
    L_gt, L_est = path_length(gt), path_length(est)
    e = est[-1] - gt[-1]
    # along / cross-track split of the final error w.r.t. the GT start->end chord
    chord = gt[-1] - gt[0]
    u = chord / max(float(np.linalg.norm(chord)), 1e-9)
    return {
        "gt_path_m": L_gt,
        "scale_err_pct": 100.0 * (L_est / max(L_gt, 1e-9) - 1.0),
        "final_err_m": float(np.linalg.norm(e)),
        "final_err_pct": 100.0 * float(np.linalg.norm(e)) / max(L_gt, 1e-9),
        "final_along_m": float(e @ u),
        "final_cross_m": float(e[0] * -u[1] + e[1] * u[0]),
        "max_err_m": float(np.max(np.linalg.norm(est - gt, axis=1))),
    }


def analyse_seed(seed: int, loc_config_factory: Any = None) -> dict[str, Any]:
    """Both replays of one cached drive, scored."""
    drive = Drive.load(seed)
    row: dict[str, Any] = {"seed": seed, "family": drive.family, "frames": int(drive.t.size),
                           "speed_mps": drive.speed_mps, "scenario_slip_long": drive.slip_long}
    for name, use in (("wheel_gyro", False), ("wheel_gyro_depth", True)):
        cfg = loc_config_factory() if loc_config_factory else None
        rep = replay(drive, use, cfg)
        sc = score(drive, rep)
        ran = rep["do_ran"][1:]
        sc.update({
            "loc_ms_median": float(np.median(rep["loc_ms"][1:])),
            "loc_ms_p95": float(np.percentile(rep["loc_ms"][1:], 95)),
            "slip_final": float(rep["slip"][-1]),
        })
        if use:
            sc.update({
                "do_frames": int(ran.sum()),
                "do_accepted_frac": float(rep["do_accepted"][1:].sum() / max(int(ran.sum()), 1)),
                "do_ms_median": float(np.median(rep["do_ms"][1:][ran])) if ran.any() else None,
                "do_ms_p95": float(np.percentile(rep["do_ms"][1:][ran], 95)) if ran.any() else None,
                "do_sigma_fwd_median_m": float(np.nanmedian(rep["do_sigma_fwd"][1:])) if ran.any() else None,
                "do_not_accepted_reasons": rep["reasons"],
            })
        row[name] = sc
    LOG.info("seed %d %s: scale %.2f%% -> %.2f%%, final %.2f -> %.2f m, accepted %.0f%%, do p95 %.1f ms", seed,
             drive.family, row["wheel_gyro"]["scale_err_pct"], row["wheel_gyro_depth"]["scale_err_pct"],
             row["wheel_gyro"]["final_err_m"], row["wheel_gyro_depth"]["final_err_m"],
             100 * row["wheel_gyro_depth"]["do_accepted_frac"], row["wheel_gyro_depth"]["do_ms_p95"] or -1)
    return row


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Median / mean / max of the per-drive errors for both configurations."""
    out: dict[str, Any] = {"drives": len(rows), "gt_path_m": float(sum(r["wheel_gyro"]["gt_path_m"] for r in rows))}
    for name in ("wheel_gyro", "wheel_gyro_depth"):
        sc = np.array([r[name]["scale_err_pct"] for r in rows])
        fe = np.array([r[name]["final_err_m"] for r in rows])
        fp = np.array([r[name]["final_err_pct"] for r in rows])
        out[name] = {"scale_err_pct_median": float(np.median(sc)), "abs_scale_err_pct_median": float(np.median(np.abs(sc))),
                     "abs_scale_err_pct_max": float(np.max(np.abs(sc))), "final_err_m_median": float(np.median(fe)),
                     "final_err_m_max": float(np.max(fe)), "final_err_pct_median": float(np.median(fp))}
    ms = [r["wheel_gyro_depth"]["do_ms_p95"] for r in rows if r["wheel_gyro_depth"].get("do_ms_p95") is not None]
    md = [r["wheel_gyro_depth"]["do_ms_median"] for r in rows if r["wheel_gyro_depth"].get("do_ms_median") is not None]
    out["depth_odom_ms_median_of_drive_medians"] = float(np.median(md)) if md else None
    out["depth_odom_ms_max_of_drive_p95"] = float(np.max(ms)) if ms else None
    out["depth_odom_accepted_frac_median"] = float(np.median([r["wheel_gyro_depth"]["do_accepted_frac"] for r in rows]))
    return out


def analyse(tune: list[int], holdout: list[int]) -> dict[str, Any]:
    """Score tuning and holdout seeds; returns the results dict (written by ``main``)."""
    rows_t = [analyse_seed(s) for s in tune if (CACHE_ROOT / f"{s}.npz").exists()]
    rows_h = [analyse_seed(s) for s in holdout if (CACHE_ROOT / f"{s}.npz").exists()]
    from metagross.autonomy.localization.depth_odom import DepthOdomConfig
    from metagross.autonomy.localization.localizer import LocalizerConfig

    lc = LocalizerConfig()
    return {
        "label": "Simulated",
        "what": "Tier-0 open-loop drives along the GT oracle path through the real World (scenario longitudinal slip, "
                "encoders, gyro, Tier-0 synthetic depth sensor); onboard Localizer replayed with depth odometry off "
                "(wheel + gyro EKF) and on (wheel + gyro + depth odometry).",
        "metric_scale": "along-track scale error = estimated planar path length / GT planar path length - 1 (%), "
                        "positive = over-count",
        "metric_final": "final position error in the A-frame (m, and % of GT path length)",
        "timing_note": f"ms per frame on this 4-core laptop shared with other jobs, OpenCV/BLAS threads = {THREADS}",
        "tune_seeds": [r["seed"] for r in rows_t],
        "holdout_seeds": [r["seed"] for r in rows_h],
        "holdout_note": "holdout seeds were recorded and scored only after the depth-odometry parameters were frozen",
        "config": {"depth_odom": {k: getattr(DepthOdomConfig(), k) for k in DepthOdomConfig.__slots__},
                   "yaw_gate": lc.depth_odom_yaw_gate, "fwd_over_frac": lc.depth_odom_fwd_over_frac,
                   "fwd_margin_m": lc.depth_odom_fwd_margin_m, "slip_sigma_m": lc.depth_odom_slip_sigma_m},
        "summary_tune": summarise(rows_t) if rows_t else None,
        "summary_holdout": summarise(rows_h) if rows_h else None,
        "summary_all": summarise(rows_t + rows_h) if rows_t or rows_h else None,
        "drives_tune": rows_t,
        "drives_holdout": rows_h,
    }


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--seeds", type=int, nargs="+", required=True)
    r.add_argument("--max-s", type=float, default=MAX_DRIVE_S)
    r.add_argument("--force", action="store_true")
    a = sub.add_parser("analyse")
    a.add_argument("--tune", type=int, nargs="*", default=[])
    a.add_argument("--holdout", type=int, nargs="*", default=[])
    a.add_argument("--out", type=Path, default=RESULTS_JSON)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
    seeds = args.seeds if args.cmd == "record" else args.tune + args.holdout
    bad = [s for s in seeds if s not in DEV_SEEDS]
    if bad:
        raise SystemExit(f"not DEV seeds (100-129): {bad}")
    if args.cmd == "record":
        for s in args.seeds:
            if (CACHE_ROOT / f"{s}.npz").exists() and not args.force:
                LOG.info("seed %d cached", s)
                continue
            record_drive(s, max_s=args.max_s)
    else:
        res = analyse(args.tune, args.holdout)
        prev = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
        if "claims" in prev:
            res["claims"] = prev["claims"]
        args.out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
        LOG.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
