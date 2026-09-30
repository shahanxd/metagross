"""Ad-hoc checks of the kitti_vo.py changes (times-length fallback, first_frame) on a fake sequence."""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(r"D:\Downloads\sih again\metagross")
sys.path.insert(0, str(REPO))
cv2.setNumThreads(2)

from metagross.autonomy.localization.vo import VOConfig  # noqa: E402
from metagross.eval.kitti_vo import KittiSequence, run_vo  # noqa: E402
from metagross.eval.traj_metrics import save_kitti_poses  # noqa: E402
from tests.test_loc_synth import BASELINE_M, CX, CY, FX, FY, make_texture, pose, render  # noqa: E402

N, STEP = 6, 0.3
root = Path(tempfile.mkdtemp())
sd = root / "sequences" / "99"
(sd / "image_0").mkdir(parents=True)
(sd / "image_1").mkdir()
tex = make_texture(11)
poses = []
for k in range(N):
    T = pose(0.0, 0.0, STEP * k)
    left, _ = render(T, tex)
    Tr = T.copy()
    Tr[:3, 3] += np.array([BASELINE_M, 0, 0])
    right, _ = render(Tr, tex)
    cv2.imwrite(str(sd / "image_0" / f"{k:06d}.png"), left)
    cv2.imwrite(str(sd / "image_1" / f"{k:06d}.png"), right)
    poses.append(T)
(sd / "calib.txt").write_text(f"P0: {FX} 0 {CX} 0 0 {FY} {CY} 0 0 0 1 0\nP1: {FX} 0 {CX} {-FX * BASELINE_M} 0 {FY} {CY} 0 0 0 1 0\n")
(sd / ".complete").write_text("ok")
save_kitti_poses(root / "poses" / "99.txt", np.stack(poses))
seq = KittiSequence("99", root)

np.savetxt(sd / "times.txt", np.arange(N) * 0.1)
assert seq.times() is not None and seq.times().size == N
np.savetxt(sd / "times.txt", np.arange(N - 2) * 0.1)  # mismatched length -> None (used to IndexError in run_vo)
assert seq.times() is None
run = run_vo(seq, vo_config=VOConfig(num_disparities=64), log_every=0)
assert run.ok[1:].all(), run.reasons
np.savetxt(sd / "times.txt", np.arange(N) * 0.1)
run2 = run_vo(seq, vo_config=VOConfig(num_disparities=64), log_every=0, first_frame=2)
assert run2.poses.shape[0] == N - 2 and run2.ok[1:].all()
print("final z first_frame=2:", run2.poses[-1, 2, 3], "expected", STEP * (N - 3))
assert abs(run2.poses[-1, 2, 3] - STEP * (N - 3)) < 0.05
print("OK")
