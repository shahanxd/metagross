"""Evaluate the onboard stereo VO on real KITTI odometry sequences (camera only).

KITTI has no wheel encoders, so this runs :class:`StereoVO` exactly as the
onboard stack does in camera-only mode: full-resolution grayscale stereo, SGBM
disparity computed per frame, PnP-RANSAC + LM. When VO rejects a frame (motion
gate / too few inliers) the pose is bridged with the last accepted relative
motion (constant velocity) and the frame is counted as a failure.

Outputs
-------
* ``results/kitti_vo_<seq>.json`` - KITTI t_err / r_err, segment drift at
  100/500/1000 m, ATE, failure count, timing (fps) and run configuration.
* ``results/kitti_vo.csv`` - one row per evaluated sequence (rebuilt from all JSONs).
* ``results/raw/kitti_vo_<seq>_poses.txt`` - estimated poses (KITTI format).
* ``deck_assets/kitti_<seq>_traj.png`` - top-view trajectory (GT black, ours blue).

Usage::

    python -m metagross.eval.kitti_vo --seqs 07 05 --threads 2
    python -m metagross.eval.kitti_vo --seqs 00 --threads 2 --first-frame 1101

KITTI 00 from the Hugging Face mirror (scripts/download_kitti.py) is corrupted: frames 000000-001100
are a different (highway) drive and times.txt has 1101 lines, so 00 is evaluated from frame 1101
(see results/kitti_summary.json, ``data_issues``). Deck figures: ``deck_assets/kitti_figures.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from metagross.autonomy.localization.health import compute_features
from metagross.autonomy.localization.vo import StereoVO, VOConfig
from metagross.eval.traj_metrics import evaluate, load_kitti_poses, plot_trajectory, save_kitti_poses

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
KITTI_ROOT = REPO_ROOT / "data" / "kitti"
RESULTS = REPO_ROOT / "results"
DECK = REPO_ROOT / "deck_assets"

FrameTransform = Callable[[int, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class KittiSequence:
    seq: str
    root: Path = KITTI_ROOT

    @property
    def dir(self) -> Path:
        return self.root / "sequences" / self.seq

    @property
    def gt_path(self) -> Path:
        return self.root / "poses" / f"{self.seq}.txt"

    def available(self) -> bool:
        return (self.dir / ".complete").exists() and self.gt_path.exists()

    def n_contiguous_frames(self) -> int:
        """Number of stereo pairs present from frame 0 without a gap (for partial downloads)."""
        if not self.gt_path.exists() or not (self.dir / "calib.txt").exists():
            return 0
        n = 0
        for lp, rp in self.frames():
            if not rp.exists():
                break
            n += 1
        return n

    def calib(self) -> tuple[np.ndarray, float]:
        """(K 3x3, baseline m) of the rectified gray pair (P0, P1)."""
        P = {}
        for line in (self.dir / "calib.txt").read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                P[k.strip()] = np.array([float(x) for x in v.split()]).reshape(3, 4)
        K = P["P0"][:, :3].copy()
        baseline = -P["P1"][0, 3] / P["P1"][0, 0]
        return K, float(baseline)

    def frames(self) -> list[tuple[Path, Path]]:
        left = sorted((self.dir / "image_0").glob("*.png"))
        return [(p, self.dir / "image_1" / p.name) for p in left]

    def times(self) -> Optional[np.ndarray]:
        """Per-frame timestamps (s), or None if times.txt is missing or its length differs from the
        number of left frames (the HF mirror of 00 ships a 1101-line times.txt for 4541 frames, which
        used to raise IndexError at frame 1101). With None, VO uses ``VOConfig.default_dt_s``."""
        f = self.dir / "times.txt"
        if not f.exists():
            return None
        t = np.atleast_1d(np.loadtxt(f))
        n = len(self.frames())
        if t.size != n:
            LOG.warning("seq %s: times.txt has %d entries for %d frames; using nominal dt", self.seq, t.size, n)
            return None
        return t

    def gt(self) -> np.ndarray:
        return load_kitti_poses(self.gt_path)


@dataclass
class VORun:
    """Per-frame outputs of a VO run over a sequence."""

    poses: np.ndarray  # (N, 4, 4) camera-to-world (world = first camera)
    ok: np.ndarray  # (N,) bool, frame 0 is False (init)
    rel: np.ndarray  # (N, 4, 4) relative pose actually used for integration
    rel_vo: np.ndarray  # (N, 4, 4) VO relative pose (NaN where VO failed)
    stats: list[dict[str, float]]
    reasons: list[str]
    vo_ms: np.ndarray  # (N,) VO wall-clock incl. SGBM (ms)
    io_ms: np.ndarray  # (N,) PNG decode (ms)
    features: list[dict[str, float]] = field(default_factory=list)


def run_vo(seq: KittiSequence, max_frames: Optional[int] = None, vo_config: Optional[VOConfig] = None,
           transform: Optional[FrameTransform] = None, with_features: bool = False,
           log_every: int = 200, first_frame: int = 0) -> VORun:
    """Run stereo VO over a sequence, bridging failures with constant velocity.

    ``first_frame`` starts VO at that frame index (output index 0 = sequence frame ``first_frame``); used to
    skip a corrupted prefix of a mirrored sequence (see ``evaluate_sequence``).
    """
    K, baseline = seq.calib()
    vo = StereoVO(K, baseline, vo_config or VOConfig.kitti())
    frames = seq.frames()[first_frame:][: max_frames or None]
    times = seq.times()
    if times is not None:
        times = times[first_frame:]
    n = len(frames)
    poses = np.tile(np.eye(4), (n, 1, 1))
    rel = np.tile(np.eye(4), (n, 1, 1))
    rel_vo = np.full((n, 4, 4), np.nan)
    ok = np.zeros(n, bool)
    vo_ms, io_ms = np.zeros(n), np.zeros(n)
    stats: list[dict[str, float]] = []
    reasons: list[str] = []
    feats: list[dict[str, float]] = []
    last_rel = np.eye(4)
    t_wall = time.perf_counter()
    for k, (lp, rp) in enumerate(frames):
        t0 = time.perf_counter()
        left = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            raise FileNotFoundError(f"missing frame {lp} / {rp}")
        if transform is not None:
            left, right = transform(k, left, right)
        io_ms[k] = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        res = vo.process(left, right, t=None if times is None else float(times[k]))
        vo_ms[k] = (time.perf_counter() - t1) * 1e3
        if with_features:
            feats.append(compute_features(res.stats, left, res.disparity))
        stats.append(res.stats)
        reasons.append(res.reason)
        if k == 0:
            continue
        if res.ok:
            ok[k] = True
            rel_vo[k] = res.T_prev_cur
            last_rel = res.T_prev_cur
        rel[k] = last_rel
        poses[k] = poses[k - 1] @ rel[k]
        if log_every and k % log_every == 0:
            el = time.perf_counter() - t_wall
            LOG.info("seq %s frame %d/%d  %.1f fps  fails=%d", seq.seq, k, n, (k + 1) / el, int(k - ok[1:k + 1].sum()))
    return VORun(poses, ok, rel, rel_vo, stats, reasons, vo_ms, io_ms, feats)


def _timing_summary(vo_ms: np.ndarray, io_ms: np.ndarray) -> dict[str, float]:
    v = vo_ms[1:] if vo_ms.size > 1 else vo_ms
    tot = (vo_ms + io_ms)[1:] if vo_ms.size > 1 else vo_ms + io_ms
    return {
        "vo_ms_mean": float(v.mean()),
        "vo_ms_median": float(np.median(v)),
        "vo_ms_p95": float(np.percentile(v, 95)),
        "fps_vo": float(1000.0 / v.mean()),
        "fps_with_png_decode": float(1000.0 / tot.mean()),
    }


def evaluate_sequence(seq_id: str, max_frames: Optional[int] = None, threads: int = 2, first_frame: int = 0) -> dict:
    """Run + score one sequence; writes JSON, poses and plot. Returns the result dict.

    ``first_frame`` > 0 evaluates the sub-sequence ``[first_frame, first_frame + n)`` against the matching GT
    slice (all metrics are relative or first-pose aligned, so they stay valid); it is recorded in the JSON and
    ``full_sequence`` is then False.
    """
    seq = KittiSequence(seq_id)
    if not seq.available():
        raise FileNotFoundError(f"KITTI sequence {seq_id} not downloaded (run scripts/download_kitti.py)")
    cv2.setNumThreads(threads)
    K, baseline = seq.calib()
    run = run_vo(seq, max_frames=max_frames, first_frame=first_frame)
    n = run.poses.shape[0]
    gt = seq.gt()[first_frame:first_frame + n]
    metrics = evaluate(gt, run.poses)
    inl = np.array([s["inliers"] for s in run.stats[1:]])
    h, w = cv2.imread(str(seq.frames()[0][0]), cv2.IMREAD_GRAYSCALE).shape
    res = {
        "sequence": seq_id,
        "label": "Tested",
        "frames": int(n),
        "first_frame": int(first_frame),
        "full_sequence": first_frame == 0 and (max_frames is None or n == len(seq.frames())),
        "resolution": [int(w), int(h)],
        "mode": "stereo VO, camera only (no wheels/IMU), frame-to-frame, no loop closure, no bundle adjustment",
        "timestamps": ("times.txt" if seq.times() is not None
                       else f"nominal dt {VOConfig.kitti().default_dt_s} s (times.txt missing or length mismatch)"),
        **{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in metrics.items()},
        "vo_failures": int(n - 1 - run.ok[1:].sum()),
        "vo_failure_pct": float(100.0 * (n - 1 - run.ok[1:].sum()) / max(n - 1, 1)),
        "inliers_median": float(np.median(inl)) if inl.size else 0.0,
        **_timing_summary(run.vo_ms, run.io_ms),
        "opencv_threads": threads,
        "cpu": platform.processor() or platform.machine(),
        "K": K.tolist(),
        "baseline_m": baseline,
        "vo_config": {k: getattr(VOConfig.kitti(), k) for k in VOConfig.__slots__},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "fps measured single-process on a shared 4-core laptop CPU (other jobs running); "
                "t_err/r_err: KITTI protocol, segments 100..800 m, step 10 frames; ATE after first-pose alignment only.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"kitti_vo_{seq_id}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    save_kitti_poses(RESULTS / "raw" / f"kitti_vo_{seq_id}_poses.txt", run.poses)
    _plot(seq_id, gt, run.poses, res)
    LOG.info("seq %s: %s", seq_id, {k: res[k] for k in ("t_err_pct", "r_err_deg_per_100m", "ate_rmse_m", "fps_vo")})
    write_combined_csv()
    return res


def _plot(seq_id: str, gt: np.ndarray, est: np.ndarray, res: dict) -> None:
    t_err = res.get("t_err_pct")
    sub = (f"t_err {t_err:.2f} %   r_err {res['r_err_deg_per_100m']:.2f} deg/100 m   "
           f"ATE {res['ate_rmse_m']:.1f} m   (KITTI protocol, 100-800 m segments)" if t_err is not None else "")
    ff = int(res.get("first_frame", 0))
    rng = f"frames {ff}-{ff + res['frames'] - 1}" if ff else f"{res['frames']} frames"
    plot_trajectory(gt, est, DECK / f"kitti_{seq_id}_traj.png",
                    title=f"KITTI {seq_id} - stereo VO, camera only ({rng}, "
                          f"{res['path_length_m']:.0f} m)", subtitle=sub)


def replot(seq_id: str) -> None:
    """Redraw the trajectory figure from the saved JSON + poses (no VO re-run)."""
    from metagross.eval.traj_metrics import load_kitti_poses

    res = json.loads((RESULTS / f"kitti_vo_{seq_id}.json").read_text(encoding="utf-8"))
    est = load_kitti_poses(RESULTS / "raw" / f"kitti_vo_{seq_id}_poses.txt")
    ff = int(res.get("first_frame", 0))
    _plot(seq_id, KittiSequence(seq_id).gt()[ff: ff + est.shape[0]], est, res)


def benchmark_vo(seq_id: str = "07", n_frames: int = 200, threads: tuple[int, ...] = (2, 4)) -> dict:
    """Full-resolution VO throughput (SGBM + KLT + PnP) at several OpenCV thread counts.

    Images are decoded before timing. Writes ``results/kitti_vo_timing.json``.
    """
    seq = KittiSequence(seq_id)
    K, baseline = seq.calib()
    pairs = [(cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE), cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE))
             for lp, rp in seq.frames()[:n_frames]]
    out: dict = {"label": "Tested", "sequence": seq_id, "frames": len(pairs),
                 "resolution": [int(pairs[0][0].shape[1]), int(pairs[0][0].shape[0])], "runs": {},
                 "note": "shared 4-core/8-thread laptop CPU (Iris Xe laptop, other users' jobs running)"}
    for th in threads:
        cv2.setNumThreads(th)
        vo = StereoVO(K, baseline, VOConfig.kitti())
        acc: dict[str, list[float]] = {}
        for left, right in pairs:
            res = vo.process(left, right)
            for key, v in res.timings_ms.items():
                acc.setdefault(key, []).append(v)
        stage = {k: float(np.mean(v[1:])) for k, v in acc.items() if len(v) > 1}
        out["runs"][str(th)] = {**{f"{k}_ms_mean": v for k, v in stage.items()},
                                "fps_vo": 1000.0 / stage["total"]}
        LOG.info("VO timing @%d threads: %s", th, out["runs"][str(th)])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "kitti_vo_timing.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


LIVE_CROP_W = 640  # UGV image width (defaults.IMG_W); KITTI rows (370-376) stand in for the UGV's 400


def benchmark_localizer(seq_id: str = "07", n_frames: int = 150, threads: int = 2) -> dict:
    """Live-loop cost of :class:`Localizer.update` on real frames at UGV-like resolution.

    KITTI frames are centre-cropped to 640 px wide. Disparity is precomputed (not
    timed) because onboard it comes from perception's SGBM; camera-only mode
    (KITTI has no wheels). Writes ``results/localizer_timing.json``.
    """
    from metagross.autonomy.localization.localizer import Localizer, LocalizerConfig
    from metagross.autonomy.localization.vo import compute_disparity, make_sgbm
    from metagross.config.defaults import VEHICLE
    from metagross.contracts.messages import SensorFrame, StereoCalibration

    cv2.setNumThreads(threads)
    seq = KittiSequence(seq_id)
    K, baseline = seq.calib()
    frames = seq.frames()[:n_frames]
    w_full = cv2.imread(str(frames[0][0]), cv2.IMREAD_GRAYSCALE).shape[1]
    x0 = (w_full - LIVE_CROP_W) // 2
    T_bc = np.eye(4)
    T_bc[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]  # level camera -> body
    h = cv2.imread(str(frames[0][0]), cv2.IMREAD_GRAYSCALE).shape[0]
    calib = StereoCalibration(LIVE_CROP_W, h, K[0, 0], K[1, 1], K[0, 2] - x0, K[1, 2], baseline, T_bc)
    cfg = LocalizerConfig(use_wheel_odom=False)
    cfg.vo = VOConfig.kitti()
    loc = Localizer(calib, VEHICLE, cfg)
    sgbm = make_sgbm(cfg.vo.num_disparities, cfg.vo.sgbm_block)
    times = seq.times()
    acc: dict[str, list[float]] = {}
    sgbm_ms: list[float] = []
    for k, (lp, rp) in enumerate(frames):
        left = np.ascontiguousarray(cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)[:, x0:x0 + LIVE_CROP_W])
        right = np.ascontiguousarray(cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)[:, x0:x0 + LIVE_CROP_W])
        t0 = time.perf_counter()
        disp = compute_disparity(sgbm, left, right)
        sgbm_ms.append((time.perf_counter() - t0) * 1e3)
        fr = SensorFrame(t=float(times[k]), seq=k, left_rgb=np.dstack([left] * 3), right_gray=right,
                         wheel_angle_l_rad=0.0, wheel_angle_r_rad=0.0, gyro_z_rps=0.0)
        out = loc.update(fr, disp)
        if k > 0:
            for key, v in out["timings_ms"].items():
                acc.setdefault(key, []).append(v)
    res = {
        "label": "Tested", "sequence": seq_id, "frames": len(frames), "resolution": [LIVE_CROP_W, int(h)],
        "opencv_threads": threads, "mode": "camera-only Localizer.update (VO + integrity + EKF), disparity supplied",
        **{f"{k}_ms_mean": float(np.mean(v)) for k, v in acc.items()},
        **{f"{k}_ms_p95": float(np.percentile(v, 95)) for k, v in acc.items()},
        "sgbm_ms_mean_not_in_localizer": float(np.mean(sgbm_ms[1:])),
        "hz_localizer_only": float(1000.0 / np.mean(acc["total"])),
        "vo_accept_rate": float(loc.n_vo_accepted / max(len(frames) - 1, 1)),
        "note": "shared 4-core laptop CPU with other jobs running; SGBM is perception's cost onboard",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "localizer_timing.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    LOG.info("localizer timing: %s", res)
    return res


CSV_COLUMNS = ("sequence", "frames", "first_frame", "full_sequence", "path_length_m", "t_err_pct", "r_err_deg_per_100m",
               "drift_100m_m", "drift_100m_pct", "drift_500m_m", "drift_500m_pct", "drift_1000m_m", "drift_1000m_pct",
               "ate_rmse_m", "final_err_m", "vo_failures", "inliers_median", "vo_ms_mean", "fps_vo",
               "fps_with_png_decode", "opencv_threads", "label")


def write_combined_csv() -> Path:
    """Rebuild results/kitti_vo.csv from every results/kitti_vo_<seq>.json."""
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("kitti_vo_[0-9][0-9].json"))]
    out = RESULTS / "kitti_vo.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in CSV_COLUMNS})
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="KITTI stereo VO evaluation (camera only)")
    ap.add_argument("--seqs", nargs="+", default=["07"])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--first-frame", type=int, default=0,
                    help="start VO at this frame (skip a corrupted prefix; recorded in the JSON)")
    ap.add_argument("--threads", type=int, default=2, help="OpenCV threads (shared machine: keep low)")
    ap.add_argument("--skip-missing", action="store_true", help="skip sequences not yet downloaded")
    ap.add_argument("--benchmark-localizer", action="store_true",
                    help="only time Localizer.update on 640-px crops of the first sequence")
    ap.add_argument("--replot", action="store_true", help="redraw figures from saved results only")
    ap.add_argument("--benchmark-vo", action="store_true", help="only time full-resolution VO at 2 and 4 threads")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    if args.benchmark_localizer:
        benchmark_localizer(f"{int(args.seqs[0]):02d}", args.max_frames or 150, args.threads)
        return 0
    if args.benchmark_vo:
        benchmark_vo(f"{int(args.seqs[0]):02d}", args.max_frames or 200)
        return 0
    if args.replot:
        for s in args.seqs:
            replot(f"{int(s):02d}")
        return 0
    for s in args.seqs:
        s = f"{int(s):02d}"
        if args.skip_missing and not KittiSequence(s).available():
            LOG.warning("sequence %s not available; skipped", s)
            continue
        evaluate_sequence(s, args.max_frames, args.threads, args.first_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
