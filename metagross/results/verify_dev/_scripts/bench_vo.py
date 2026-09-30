"""Interleaved VO config benchmark on cached rendered DEV frames (paired timing, accuracy vs GT)."""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, r"D:\Downloads\sih again\metagross")
from metagross.autonomy.localization.vo import StereoVO, VOConfig  # noqa: E402
from metagross.autonomy.perception.pipeline import Perception  # noqa: E402
from metagross.config import defaults  # noqa: E402
from metagross.contracts.messages import SensorFrame  # noqa: E402
import metagross.eval.integrity_sim as S  # noqa: E402

cv2.setNumThreads(2)
seeds = [int(s) for s in sys.argv[1].split(",")]
nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 150
CFGS = {
    "base": VOConfig(photometric_norm=False),
    "photo": VOConfig(),
    "it10": VOConfig(klt_iters=10, klt_eps=0.03),
    "w15it10": VOConfig(klt_win=15, klt_iters=10, klt_eps=0.03),
    "it15": VOConfig(klt_iters=15, klt_eps=0.03),
}
calib = defaults.stereo_calibration()
for seed in seeds:
    d = S.Drive.load(seed)
    per = Perception(calib, defaults.VEHICLE)
    vos = {k: StereoVO(calib.K, calib.baseline_m, c) for k, c in CFGS.items()}
    n = min(d.n, nmax)
    ms = {k: np.zeros(n) for k in CFGS}
    rel = {k: np.full((n, 4, 4), np.nan) for k in CFGS}
    order = list(CFGS)
    for k, left, right in d.frames():
        if k >= n:
            break
        fr = SensorFrame(t=float(d.meta["t"][k]), seq=k, left_rgb=left, right_gray=right, wheel_angle_l_rad=0.0,
                         wheel_angle_r_rad=0.0, gyro_z_rps=0.0, sensor_mode="stereo")
        disp = per.compute_disparity(fr)
        gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        order = order[1:] + order[:1]  # rotate the run order to spread load bias
        for name in order:
            t0 = time.perf_counter()
            r = vos[name].process(gray, None, disp, fr.t)
            ms[name][k] = (time.perf_counter() - t0) * 1e3
            if r.ok:
                rel[name][k] = r.T_prev_cur
    gt = S.gt_relative(d)[:n]
    print(f"seed {seed} ({d.family}) n={n}")
    for name in CFGS:
        lab, te, re = S.vo_labels(rel[name], gt)
        dr = S.vo_drift_pct(rel[name], gt)
        a = ms[name][1:]
        print(f"  {name:10s} ms med {np.median(a):6.2f} p95 {np.percentile(a, 95):6.2f} | terr med {1e3*np.nanmedian(te):.2f} mm "
              f"p95 {1e3*np.nanpercentile(te, 95):.2f} mm | fails {int(lab.sum())} | drift {dr['drift_pct']:.3f} %")
