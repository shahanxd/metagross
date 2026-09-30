"""KITTI evaluation plumbing on a tiny fake sequence rendered from the synthetic scene."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from metagross.eval.kitti_vo import KittiSequence, run_vo
from metagross.eval.traj_metrics import ate_rmse, save_kitti_poses
from metagross.autonomy.localization.vo import VOConfig
from tests.test_loc_synth import BASELINE_M, CX, CY, FX, FY, make_texture, pose, render

N_FRAMES = 6
STEP_M = 0.3


@pytest.fixture(scope="module")
def fake_kitti(tmp_path_factory):
    root = tmp_path_factory.mktemp("kitti")
    sd = root / "sequences" / "99"
    (sd / "image_0").mkdir(parents=True)
    (sd / "image_1").mkdir()
    tex = make_texture(11)
    poses = []
    for k in range(N_FRAMES):
        T = pose(0.0, 0.0, STEP_M * k)
        left, _ = render(T, tex)
        T_r = T.copy()
        T_r[:3, 3] += np.array([BASELINE_M, 0, 0])
        right, _ = render(T_r, tex)
        cv2.imwrite(str(sd / "image_0" / f"{k:06d}.png"), left)
        cv2.imwrite(str(sd / "image_1" / f"{k:06d}.png"), right)
        poses.append(T)
    P0 = f"{FX} 0 {CX} 0 0 {FY} {CY} 0 0 0 1 0"
    P1 = f"{FX} 0 {CX} {-FX * BASELINE_M} 0 {FY} {CY} 0 0 0 1 0"
    (sd / "calib.txt").write_text(f"P0: {P0}\nP1: {P1}\n")
    np.savetxt(sd / "times.txt", np.arange(N_FRAMES) * 0.1)
    (sd / ".complete").write_text("ok")
    save_kitti_poses(root / "poses" / "99.txt", np.stack(poses))
    return KittiSequence("99", root)


def test_calib_parsing(fake_kitti):
    K, b = fake_kitti.calib()
    assert K[0, 0] == pytest.approx(FX) and K[1, 2] == pytest.approx(CY) and b == pytest.approx(BASELINE_M)
    assert fake_kitti.available() and fake_kitti.n_contiguous_frames() == N_FRAMES


def test_run_vo_on_fake_sequence(fake_kitti):
    run = run_vo(fake_kitti, vo_config=VOConfig(num_disparities=64), log_every=0)
    assert run.ok[1:].all() and not run.ok[0]
    gt = fake_kitti.gt()
    assert ate_rmse(gt, run.poses)["final_err_m"] < 0.05 * STEP_M * (N_FRAMES - 1) + 0.02
    assert run.vo_ms.shape == (N_FRAMES,)


def test_transform_hook_and_features(fake_kitti):
    seen = []

    def blank_frame_3(k, left, right):
        seen.append(k)
        return (np.full_like(left, 128), right) if k == 3 else (left, right)

    run = run_vo(fake_kitti, vo_config=VOConfig(num_disparities=64), transform=blank_frame_3,
                 with_features=True, log_every=0)
    assert seen == list(range(N_FRAMES))
    assert not run.ok[3] and len(run.features) == N_FRAMES
    assert run.features[3]["img_rms_contrast"] == pytest.approx(0.0)
    # the bridged frame reuses the previous motion, so the trajectory keeps going
    assert run.poses[3, 2, 3] == pytest.approx(3 * STEP_M, abs=0.05)
