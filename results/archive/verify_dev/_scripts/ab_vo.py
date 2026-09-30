"""Scratch A/B of VO configs on the first N frames of KITTI 07 (dev only, not a deliverable)."""
import dataclasses
import sys

import cv2
import numpy as np

sys.path.insert(0, r"D:\Downloads\sih again\metagross")
from metagross.autonomy.localization.vo import VOConfig, rotation_angle  # noqa: E402
from metagross.eval.kitti_vo import KittiSequence, run_vo  # noqa: E402
from metagross.eval.traj_metrics import ate_rmse  # noqa: E402

cv2.setNumThreads(2)
N = 300
seq = KittiSequence("07")
gt = seq.gt()[:N]
base = VOConfig.kitti()
for name, cfg in [("base", base), ("min_disp3", dataclasses.replace(base, min_disp_px=3.0)),
                  ("win15", dataclasses.replace(base, klt_win=15)),
                  ("min_disp6", dataclasses.replace(base, min_disp_px=6.0))]:
    run = run_vo(seq, max_frames=N, vo_config=cfg, log_every=0)
    yaw = []
    for k in range(1, N):
        E = run.rel[k][:3, :3].T @ (np.linalg.inv(gt[k - 1]) @ gt[k])[:3, :3]
        yaw.append(np.degrees(np.arctan2(E[0, 2], E[2, 2])))
    print(name, "ATE", round(ate_rmse(gt, run.poses)["ate_rmse_m"], 3), "final", round(ate_rmse(gt, run.poses)["final_err_m"], 3),
          "yaw cum deg", round(float(np.sum(yaw)), 3), "vo ms", round(float(run.vo_ms[1:].mean()), 1), flush=True)
