"""Trajectory accuracy metrics (KITTI odometry protocol) and trajectory plots.

Poses are (N, 4, 4) homogeneous camera-to-world transforms in metres, KITTI
convention: the world is the camera frame of the first image (x right, y down,
z forward), so the ground plane is x-z.

* :func:`kitti_errors` ports ``kitti-odom-eval`` (Huangying-Zhan, MIT licence;
  itself a port of the official KITTI devkit): for every 10th start frame and
  every segment length L in 100..800 m, the relative pose error of the segment
  is ``E = (P_gt[a]^-1 P_gt[b])^-1 (P_est[a]^-1 P_est[b])``; translation error is
  ``|t(E)| / L`` and rotation error ``angle(R(E)) / L``. Averaging over all
  segments gives t_err (%) and r_err (deg / 100 m).
* :func:`segment_drift` reports the same relative error for single segment
  lengths (e.g. 100 / 500 / 1000 m) as mean end-point error in metres and %.
* :func:`ate_rmse` is the absolute trajectory error after aligning the first
  poses only (no scale, no best-fit rotation), i.e. the drift a vehicle would
  actually accumulate from its start pose.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

LOG = logging.getLogger(__name__)

KITTI_LENGTHS_M: tuple[int, ...] = (100, 200, 300, 400, 500, 600, 700, 800)
STEP_SIZE = 10  # start-frame stride of the KITTI protocol (frames)
KITTI_FRAME_DT_S = 0.1  # 10 Hz; only used for the per-segment speed column


def load_kitti_poses(path: Path) -> np.ndarray:
    """Read a KITTI pose file (N lines of 12 floats, row-major 3x4) -> (N, 4, 4)."""
    data = np.loadtxt(path, dtype=np.float64).reshape(-1, 3, 4)
    poses = np.tile(np.eye(4), (data.shape[0], 1, 1))
    poses[:, :3, :] = data
    return poses


def save_kitti_poses(path: Path, poses: np.ndarray) -> None:
    """Write (N, 4, 4) poses in KITTI 12-float format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, poses[:, :3, :].reshape(-1, 12), fmt="%.9e")


def trajectory_distances(poses: np.ndarray) -> np.ndarray:
    """Cumulative path length (m) at every frame, from consecutive translations."""
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _rotation_error(E: np.ndarray) -> float:
    d = 0.5 * (E[0, 0] + E[1, 1] + E[2, 2] - 1.0)
    return float(np.arccos(max(min(d, 1.0), -1.0)))


def _last_frame_from_segment_length(dist: np.ndarray, first: int, length: float) -> int:
    idx = np.searchsorted(dist, dist[first] + length, side="right")
    return int(idx) if idx < dist.size else -1


def segment_errors(gt: np.ndarray, est: np.ndarray, lengths: Sequence[float] = KITTI_LENGTHS_M,
                   step: int = STEP_SIZE) -> np.ndarray:
    """Per-segment errors, columns: [first_frame, r_err (rad/m), t_err (m/m), length (m), speed (m/s)]."""
    if gt.shape != est.shape:
        raise ValueError(f"pose arrays differ: {gt.shape} vs {est.shape}")
    dist = trajectory_distances(gt)
    rows: list[list[float]] = []
    for first in range(0, gt.shape[0], step):
        for L in lengths:
            last = _last_frame_from_segment_length(dist, first, L)
            if last == -1:
                continue
            d_gt = np.linalg.inv(gt[first]) @ gt[last]
            d_est = np.linalg.inv(est[first]) @ est[last]
            E = np.linalg.inv(d_est) @ d_gt  # devkit: inv(pose_delta_result) * pose_delta_gt
            n_frames = last - first + 1
            speed = L / (KITTI_FRAME_DT_S * n_frames)
            rows.append([first, _rotation_error(E) / L, float(np.linalg.norm(E[:3, 3])) / L, float(L), speed])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def kitti_errors(gt: np.ndarray, est: np.ndarray) -> dict[str, float]:
    """KITTI t_err (%) and r_err (deg/100 m), averaged over all segments of 100..800 m.

    Returns NaN for both if the trajectory is shorter than 100 m.
    """
    seg = segment_errors(gt, est)
    if seg.shape[0] == 0:
        return {"t_err_pct": float("nan"), "r_err_deg_per_100m": float("nan"), "n_segments": 0}
    return {
        "t_err_pct": float(seg[:, 2].mean() * 100.0),
        "r_err_deg_per_100m": float(np.degrees(seg[:, 1].mean()) * 100.0),
        "n_segments": int(seg.shape[0]),
    }


def segment_drift(gt: np.ndarray, est: np.ndarray, lengths: Iterable[float] = (100.0, 500.0, 1000.0),
                  step: int = STEP_SIZE) -> dict[str, Optional[float]]:
    """Mean end-point translation error (m and % of L) over all segments of each length L.

    Lengths longer than the trajectory are reported as None.
    """
    out: dict[str, Optional[float]] = {}
    for L in lengths:
        seg = segment_errors(gt, est, lengths=(L,), step=step)
        key = f"{int(L)}m"
        if seg.shape[0] == 0:
            out[f"drift_{key}_m"] = None
            out[f"drift_{key}_pct"] = None
            out[f"drift_{key}_n"] = 0
        else:
            out[f"drift_{key}_m"] = float(seg[:, 2].mean() * L)
            out[f"drift_{key}_pct"] = float(seg[:, 2].mean() * 100.0)
            out[f"drift_{key}_n"] = int(seg.shape[0])
    return out


def align_first(gt: np.ndarray, est: np.ndarray) -> np.ndarray:
    """Express ``est`` so that its first pose coincides with the first GT pose (SE(3), no scale)."""
    return gt[0] @ np.linalg.inv(est[0]) @ est


def ate_rmse(gt: np.ndarray, est: np.ndarray) -> dict[str, float]:
    """ATE after first-pose alignment only: RMSE / max / final position error (m)."""
    est_a = align_first(gt, est)
    e = np.linalg.norm(gt[:, :3, 3] - est_a[:, :3, 3], axis=1)
    return {"ate_rmse_m": float(np.sqrt(np.mean(e ** 2))), "ate_max_m": float(e.max()),
            "final_err_m": float(e[-1]), "path_length_m": float(trajectory_distances(gt)[-1])}


def evaluate(gt: np.ndarray, est: np.ndarray) -> dict[str, float | int | None]:
    """All metrics in one flat dict (JSON-serialisable)."""
    res: dict[str, float | int | None] = {}
    res.update(kitti_errors(gt, est))
    res.update(segment_drift(gt, est))
    res.update(ate_rmse(gt, est))
    return res


def plot_trajectory(gt: np.ndarray, est: np.ndarray, out_png: Path, title: str,
                    extra: Optional[dict[str, np.ndarray]] = None, subtitle: str = "") -> None:
    """Top-view (x-z) trajectory plot: clean white style, GT black, ours blue."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    est_a = align_first(gt, est)
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=150)
    ax.plot(gt[:, 0, 3], gt[:, 2, 3], color="black", lw=1.8, label="Ground truth")
    ax.plot(est_a[:, 0, 3], est_a[:, 2, 3], color="#1f5fbf", lw=1.6, label="METAGROSS stereo VO")
    for name, traj in (extra or {}).items():
        tr = align_first(gt, traj)
        ax.plot(tr[:, 0, 3], tr[:, 2, 3], lw=1.2, ls="--", label=name)
    ax.plot(gt[0, 0, 3], gt[0, 2, 3], "o", color="black", ms=6)
    ax.annotate("start", (gt[0, 0, 3], gt[0, 2, 3]), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(title, fontsize=11, loc="left", pad=18 if subtitle else 6)
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, fontsize=8, color="#555555", va="bottom")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    LOG.info("wrote %s", out_png)
