"""Profile Localizer.update on a cached rendered DEV drive (disparity precomputed, not timed)."""
import cProfile
import pstats
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, r"D:\Downloads\sih again\metagross")
from metagross.autonomy.localization.localizer import Localizer  # noqa: E402
from metagross.autonomy.perception.pipeline import Perception  # noqa: E402
from metagross.config import defaults  # noqa: E402
from metagross.contracts.messages import SensorFrame  # noqa: E402
import metagross.eval.integrity_sim as S  # noqa: E402

cv2.setNumThreads(2)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 102
nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 120
d = S.Drive.load(seed)
calib = defaults.stereo_calibration()
per = Perception(calib, defaults.VEHICLE)
frames = []
for k, left, right in d.frames():
    if k >= nmax:
        break
    fr = SensorFrame(t=float(d.meta["t"][k]), seq=k, left_rgb=left, right_gray=right,
                     wheel_angle_l_rad=float(d.meta["wl"][k]), wheel_angle_r_rad=float(d.meta["wr"][k]),
                     gyro_z_rps=float(d.meta["gyro"][k]), sensor_mode="stereo")
    frames.append((fr, per.compute_disparity(fr)))

def run(prof=None):
    loc = Localizer(calib, defaults.VEHICLE)
    acc = {}
    for fr, disp in frames:
        if prof: prof.enable()
        out = loc.update(fr, disp)
        if prof: prof.disable()
        for key, v in out["timings_ms"].items():
            acc.setdefault(key, []).append(v)
        for key, v in loc.last_vo.timings_ms.items():
            acc.setdefault("vo_" + key, []).append(v)
    return acc

run()  # warm
acc = run()
for key, v in acc.items():
    a = np.asarray(v[1:])
    print(f"{key:14s} med {np.median(a):6.2f} p95 {np.percentile(a, 95):6.2f} mean {a.mean():6.2f} n {a.size}")
pr = cProfile.Profile()
run(pr)
pstats.Stats(pr).sort_stats("tottime").print_stats(18)
