"""Onboard localiser: stereo VO + integrity monitor + wheel/gyro SE(2) EKF + slip check.

Implements :class:`metagross.contracts.interfaces.LocalizerProto`.

Per frame (``update``):

1. **Predict** the planar pose from the encoder increments (skid-steer model
   with the online chi estimate) and, if enabled, the bias-corrected gyro yaw.
   Camera-only mode (``use_wheel_odom=False``) predicts with a constant-velocity
   model instead.
2. **VO**: frame-to-frame stereo VO (:class:`StereoVO`) on the left gray image,
   re-using the caller's disparity map when given (perception computes it).
3. **Integrity**: 12 features -> ``p_fail`` -> smoothed ``q``
   (:class:`IntegrityMonitor`). VO is *rejected* when ``q_gate < q_reject``
   (0.4): the EKF then bridges on wheel odometry (or constant velocity).
4. **Update**: an accepted VO increment is converted camera -> body frame
   (``T_b0_b1 = T_bc T_c0_c1 T_bc^-1``), projected to SE(2) and fused with a
   covariance inflated by ``1/inliers`` and ``1/q``.
5. **Slip**: slip ratio and IMMOBILISED flag from VO vs wheel distance; online
   chi from wheel-differential vs VO yaw rate.

Depth odometry (:mod:`depth_odom`): with no images (Tier-0 disparity-only frames) VO cannot
run, so consecutive disparity maps are registered instead (prior = the wheel + gyro increment).
An increment is fused like a VO increment (covariance from the registration residual and the
amount of structure) after two physical gates: its yaw must agree with the gyro, and its forward
travel must lie between ``-depth_odom_fwd_margin_m`` and the wheel travel x
``(1 + depth_odom_fwd_over_frac)`` + margin (wheels over-count under slip, they do not
under-count). Featureless terrain yields no measurement (``low structure``): the EKF then
dead-reckons on wheels + gyro. Well-constrained increments also feed the slip monitor. In
stereo mode VO stays primary; ``depth_odom_vo_fallback`` (off by default) registers depth
only on frames where VO was not accepted.

Output frame: A-frame (x forward at launch, y left, z up), starting at (0, 0, 0).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from metagross.autonomy.localization.depth_odom import DepthOdomConfig, DepthOdometry, DepthOdomResult
from metagross.autonomy.localization.ekf import EKFConfig, PlanarEKF, wrap_angle
from metagross.autonomy.localization.health import (
    DEFAULT_MODEL_PATH,
    FEATURE_NAMES,
    HealthConfig,
    IntegrityMonitor,
)
from metagross.autonomy.localization.slip import SlipConfig, SlipEstimator
from metagross.autonomy.localization.vo import StereoVO, VOConfig, VOResult
from metagross.config.defaults import CHI_NOMINAL
from metagross.contracts.messages import SensorFrame, StereoCalibration, VehicleSpec

LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class LocalizerConfig:
    use_wheel_odom: bool = True  # False = camera-only (constant velocity + VO)
    use_gyro: bool = True  # gyro yaw replaces wheel-differential yaw (wheel mode only)
    use_health: bool = True  # integrity monitor gates VO
    use_chi_hat: bool = True  # use the online chi estimate in the wheel model
    q_reject: float = 0.4  # reject VO when q_gate is below this
    # gyro-bias Kalman filter (scalar). Datasheet-class MEMS numbers, not simulator internals:
    gyro_density_rps_rthz: float = 2.0e-3  # white-noise density (rad/s/sqrt(Hz))
    gyro_bias_init_sigma_rps: float = 1.0e-2  # turn-on bias 1-sigma (rad/s); consumer MEMS specs quote up to ~1 deg/s
    gyro_bias_rw_rps_rts: float = 1.0e-4  # bias random walk (rad/s/sqrt(s))
    # Depth-odometry yaw is not a reliable bias reference (DEV replay, seeds 100-129: median final error 1.05 -> 1.55 m
    # when it feeds the bias), so by default the bias is learned standing still (launch hold) and from VO only.
    gyro_bias_do_yaw_sigma_max_rad: float = 0.0
    gyro_bias_gate: float = 4.0  # reject a bias observation beyond this many sigma
    stationary_wheel_rad: float = 1e-3  # per-tick wheel increment below which the UGV is still
    slip_noise_per_unit: float = 10.0  # wheel-noise inflation = 1 + this * slip (slip 0.1 -> x2)
    slip_noise_max: float = 20.0
    default_dt_s: float = 0.2
    # depth odometry (disparity-map registration) where VO has no images (Tier-0)
    use_depth_odom: bool = True
    depth_odom_vo_fallback: bool = False  # stereo mode: register depth when VO is not accepted
    depth_odom_yaw_gate: float = 4.0  # reject if |dyaw - gyro dyaw| > this x combined sigma ...
    depth_odom_yaw_gate_floor_rad: float = math.radians(0.3)  # ... + this floor
    depth_odom_fwd_over_frac: float = 0.5  # forward travel may not exceed wheel travel x (1 + this) + margin
    depth_odom_fwd_margin_m: float = 0.05
    # Depth-odometry yaw is biased on some terrain (DEV replay: -1.4 -> -3.2 deg on seed 124), so heading comes
    # mainly from the gyro: floor its yaw sigma. Chosen on DEV seeds 100-129 (tail of final error).
    depth_odom_yaw_sigma_floor_rad: float = math.radians(3.0)
    depth_odom_slip_sigma_m: float = 0.02  # increments this well constrained (forward 1-sigma) feed the slip monitor
    depth_odom: DepthOdomConfig = field(default_factory=DepthOdomConfig)
    vo: VOConfig = field(default_factory=VOConfig)
    ekf: EKFConfig = field(default_factory=EKFConfig)
    slip: SlipConfig = field(default_factory=SlipConfig)
    health: HealthConfig = field(default_factory=HealthConfig)

    @classmethod
    def from_autonomy_config(cls, config: Optional[Mapping[str, Any]]) -> "LocalizerConfig":
        """Map the stack-level switches (``contracts.ipc.DEFAULT_AUTONOMY_CONFIG``) onto this config."""
        c = cls()
        if config:
            c.use_wheel_odom = bool(config.get("use_wheel_odom", c.use_wheel_odom))
            c.use_health = bool(config.get("use_health", c.use_health))
            c.use_depth_odom = bool(config.get("use_depth_odom", c.use_depth_odom))
        return c


def camera_to_body_increment(T_prev_cur_cam: np.ndarray, T_body_cam: np.ndarray) -> tuple[float, float, float]:
    """Planar body-frame increment (dx m, dy m, dyaw rad) from a camera-frame relative pose."""
    T_b = T_body_cam @ T_prev_cur_cam @ np.linalg.inv(T_body_cam)
    return float(T_b[0, 3]), float(T_b[1, 3]), float(math.atan2(T_b[1, 0], T_b[0, 0]))


class Localizer:
    """Stateful onboard localiser (one instance per mission)."""

    def __init__(self, calib: StereoCalibration, vehicle: VehicleSpec, config: Optional[LocalizerConfig] = None,
                 model_path: Optional[Path] = DEFAULT_MODEL_PATH) -> None:
        self.calib = calib
        self.vehicle = vehicle
        self.cfg = config or LocalizerConfig()
        self.T_body_cam = np.asarray(calib.T_body_cam, float)
        self.vo = StereoVO(calib.K, calib.baseline_m, self.cfg.vo)
        self.ekf = PlanarEKF(self.cfg.ekf)
        self.slip = SlipEstimator(self.cfg.slip, chi0=CHI_NOMINAL)
        self.health = IntegrityMonitor(config=self.cfg.health, model_path=model_path)
        self.depth_odom = DepthOdometry(calib.K, calib.baseline_m, self.T_body_cam, self.cfg.depth_odom)
        self.reset()

    def reset(self) -> None:
        self.vo.reset()
        self.depth_odom.reset()
        self.n_do_accepted = 0
        self.n_do_rejected_gate = 0
        self.last_do: Optional[DepthOdomResult] = None  # last depth-odometry result (offline evaluation)
        self.ekf.reset()
        self.slip.reset()
        self.health.reset()
        self._prev_t: Optional[float] = None
        self._prev_wheels: Optional[tuple[float, float]] = None
        self.gyro_bias = 0.0
        self.gyro_bias_var = self.cfg.gyro_bias_init_sigma_rps ** 2
        self.n_bias_updates = 0
        self.n_frames = 0
        self.n_vo_accepted = 0
        self.n_vo_rejected_health = 0
        self.last_vo: Optional[VOResult] = None  # raw VO output of the last update (offline evaluation)

    # ------------------------------------------------------------------ main
    def update(self, frame: SensorFrame, disparity: Optional[np.ndarray] = None) -> dict[str, Any]:
        """Fuse one sensor frame. See :class:`LocalizerProto` for the returned keys.

        Extra keys: 'immobilised' (bool), 'vo_accepted' (bool), 'reason' (str),
        'pose_cov' (3x3 ndarray, x/y/yaw).
        """
        cfg = self.cfg
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        dt = cfg.default_dt_s if self._prev_t is None else max(frame.t - self._prev_t, 1e-3)
        first = self._prev_t is None
        wheels = (frame.wheel_angle_l_rad, frame.wheel_angle_r_rad)
        dphi_l = 0.0 if first else wheels[0] - self._prev_wheels[0]
        dphi_r = 0.0 if first else wheels[1] - self._prev_wheels[1]
        r, B = self.vehicle.wheel_radius_m, self.vehicle.track_width_m
        chi = self.slip.chi_hat if cfg.use_chi_hat else CHI_NOMINAL

        # 1. predict
        t0 = time.perf_counter()
        prior = (0.0, 0.0, 0.0)  # predicted body increment (m, m, rad), prior of depth odometry
        if not first:
            if cfg.use_wheel_odom:
                gyro_dyaw = (frame.gyro_z_rps - self.gyro_bias) * dt if cfg.use_gyro else None
                inflate = min(1.0 + cfg.slip_noise_per_unit * max(self.slip.slip, 0.0), cfg.slip_noise_max)
                d_p, dyaw_p = self.ekf.predict_wheels(dphi_l, dphi_r, r, B, chi, gyro_dyaw=gyro_dyaw, dt=dt,
                                                      noise_scale=inflate)
                prior = (d_p, 0.0, dyaw_p)
            else:
                v, w = self.ekf.v_body
                self.ekf.predict_constant_velocity(dt)
                prior = (float(v) * dt, 0.0, float(w) * dt)
        timings["predict"] = (time.perf_counter() - t0) * 1e3

        # 2. VO
        gray = self._gray(frame)
        disp_in = disparity if disparity is not None else frame.disparity
        vo_res = None
        if gray is not None and (frame.right_gray is not None or disp_in is not None):
            vo_res = self.vo.process(gray, frame.right_gray, disp_in, frame.t)
            self.last_vo = vo_res
            timings["vo"] = vo_res.timings_ms.get("total", 0.0)
            timings["vo_disparity"] = vo_res.timings_ms.get("disparity", 0.0)

        # 3. integrity
        t0 = time.perf_counter()
        if vo_res is not None:
            rgb = frame.left_rgb if frame.left_rgb is not None and frame.left_rgb.ndim == 3 else None
            health = self.health.update(vo_res.stats, gray, vo_res.disparity, rgb, warmup=vo_res.reason == "init")
            health["vo_available"] = 1.0
        else:  # no images (e.g. Tier-0 disparity-only mode): VO unavailable, dead reckoning
            health = {n: 0.0 for n in FEATURE_NAMES}
            health.update({"p_fail": 1.0, "q_inst": 0.0, "q": 0.0, "q_gate": 0.0, "vo_available": 0.0})
        timings["health"] = (time.perf_counter() - t0) * 1e3

        # 4. update
        t0 = time.perf_counter()
        vo_ok = bool(vo_res is not None and vo_res.ok)
        q = health["q"] if cfg.use_health else 1.0
        accepted = vo_ok and (not cfg.use_health or health["q_gate"] >= cfg.q_reject)
        reason = "" if vo_res is None else vo_res.reason
        d_vo = dyaw_vo = None
        if vo_ok and not accepted:
            self.n_vo_rejected_health += 1
            reason = f"health q={health['q_gate']:.2f}"
        if accepted:
            dx, dy, dyaw = camera_to_body_increment(vo_res.T_prev_cur, self.T_body_cam)
            Rm = self.ekf.vo_noise(dx, dy, dyaw, vo_res.stats["inliers"], q)
            accepted = self.ekf.update_relative(dx, dy, dyaw, Rm, dt=dt)
            if accepted:
                self.n_vo_accepted += 1
                d_vo, dyaw_vo = math.copysign(math.hypot(dx, dy), dx), dyaw
        timings["ekf"] = (time.perf_counter() - t0) * 1e3

        # 4b. depth odometry: no images (Tier-0), or optional fallback when VO was not accepted
        do_info = self._depth_odometry(disp_in, vo_res is None, bool(accepted), first, prior, dt)
        if do_info.get("feed_slip"):
            d_vo = do_info["dx"]  # ground distance for the slip monitor (yaw not used for chi)
        timings["depth_odom"] = do_info.get("ms", 0.0)
        self.ekf.clone()

        # 5. slip / chi / gyro bias (need wheels)
        slip_out = {"slip": 0.0, "immobilised": 0.0, "chi_hat": chi}
        if cfg.use_wheel_odom and not first:
            d_wheel = r * (dphi_l + dphi_r) * 0.5
            dyaw_wd = r * (dphi_r - dphi_l) / B
            slip_out = self.slip.update(frame.t, d_wheel, dyaw_wd, dt, d_vo, dyaw_vo)
            still = abs(dphi_l) < cfg.stationary_wheel_rad and abs(dphi_r) < cfg.stationary_wheel_rad
            self._update_gyro_bias(frame.gyro_z_rps, dt, still, dyaw_vo, do_info)

        self._prev_t, self._prev_wheels = frame.t, wheels
        self.n_frames += 1
        timings["total"] = (time.perf_counter() - t_start) * 1e3
        return {
            "pose_xy_yaw": self.ekf.pose,
            "pos_sigma_m": self.ekf.pos_sigma_m(),
            "pose_cov": self.ekf.cov,
            "health": health,
            "vo_ok": vo_ok,
            "vo_accepted": bool(accepted),
            "slip": float(slip_out["slip"]),
            "immobilised": bool(slip_out["immobilised"]),
            "chi_hat": float(slip_out["chi_hat"]),
            "gyro_bias_rps": self.gyro_bias,
            "reason": reason,
            "timings_ms": timings,
            "depth_odom": {k: v for k, v in do_info.items() if k != "feed_slip"},
        }

    def _update_gyro_bias(self, gyro_rps: float, dt: float, still: bool, dyaw_vo: Optional[float],
                          do_info: Mapping[str, Any]) -> None:
        """Scalar Kalman filter on the gyro bias (rad/s).

        Observations: (a) standing still, the gyro reads the bias (yaw rate is zero; VO, if any, must
        agree); (b) moving, an accepted increment whose yaw is well constrained on its own (VO, or
        depth odometry with small yaw covariance) gives ``bias = gyro - dyaw/dt``. Without (b) a
        turn-on bias of 0.06 deg/s alone turns into ~2 m of cross-track error over a 45 m mission."""
        cfg = self.cfg
        self.gyro_bias_var += cfg.gyro_bias_rw_rps_rts ** 2 * dt
        white_var = cfg.gyro_density_rps_rthz ** 2 / dt  # variance of the interval-mean rate
        z = R = None
        if still and (dyaw_vo is None or abs(dyaw_vo) < math.radians(0.05)):
            z, R = gyro_rps, white_var
        elif dyaw_vo is not None:
            z, R = gyro_rps - dyaw_vo / dt, white_var + (self.ekf.cfg.vo_yaw_floor_rad / dt) ** 2
        elif do_info.get("accepted") and self.last_do is not None:
            var_do = float(self.last_do.cov[2, 2])
            if var_do <= cfg.gyro_bias_do_yaw_sigma_max_rad ** 2:
                z, R = gyro_rps - self.last_do.dyaw / dt, white_var + var_do / dt ** 2
        if z is None:
            return
        S = self.gyro_bias_var + R
        innov = z - self.gyro_bias
        if not still and innov * innov > cfg.gyro_bias_gate ** 2 * S:  # standing still the gyro reads the bias: never gated
            return
        k = self.gyro_bias_var / S
        self.gyro_bias += k * innov
        self.gyro_bias_var *= (1.0 - k)
        self.n_bias_updates += 1

    def _depth_odometry(self, disp: Optional[np.ndarray], no_images: bool, vo_accepted: bool, first: bool,
                        prior: tuple[float, float, float], dt: float) -> dict[str, Any]:
        """Run / gate / fuse depth odometry for this frame; returns a small info dict.

        Keys: 'ran', 'accepted' (bool), 'reason' (str), 'dx' (m), 'sigma_fwd_m', 'ms', 'feed_slip'."""
        cfg = self.cfg
        info: dict[str, Any] = {"ran": False, "accepted": False, "reason": "off"}
        if not cfg.use_depth_odom or disp is None:
            return info
        if not no_images and not cfg.depth_odom_vo_fallback:
            return info
        if not no_images and vo_accepted:  # stereo fallback mode: VO is fine, only keep the map
            self.depth_odom.observe(disp)
            info["reason"] = "vo ok"
            return info
        res = self.depth_odom.process(disp, prior)
        self.last_do = res
        info.update({"ran": True, "reason": res.reason, "ms": res.timings_ms.get("total", 0.0),
                     "dx": res.dx, "sigma_fwd_m": res.stats.get("sigma_fwd_m", math.nan)})
        if first or not res.ok:
            return info
        # physical gates: gyro yaw agreement, forward travel bounded by the (over-counting) wheels
        c = self.ekf.cfg
        sig_gyro = c.gyro_noise_rps * math.sqrt(max(dt, 1e-6)) + c.gyro_bias_sigma_rps * dt
        yaw_tol = cfg.depth_odom_yaw_gate * math.sqrt(float(res.cov[2, 2]) + sig_gyro ** 2) \
            + cfg.depth_odom_yaw_gate_floor_rad
        d_prior = abs(prior[0])
        fwd_hi = d_prior * (1.0 + cfg.depth_odom_fwd_over_frac) + cfg.depth_odom_fwd_margin_m
        if abs(wrap_angle(res.dyaw - prior[2])) > yaw_tol:
            info["reason"] = "yaw gate"
        elif not (-cfg.depth_odom_fwd_margin_m <= res.dx * (1.0 if prior[0] >= 0.0 else -1.0) <= fwd_hi):
            info["reason"] = "forward gate"
        elif self.ekf.update_relative(res.dx, res.dy, res.dyaw, self._depth_odom_cov(res.cov), dt=dt):
            self.n_do_accepted += 1
            info["accepted"] = True
            info["feed_slip"] = info["sigma_fwd_m"] <= cfg.depth_odom_slip_sigma_m
            return info
        self.n_do_rejected_gate += 1
        return info

    def _depth_odom_cov(self, cov: np.ndarray) -> np.ndarray:
        """Depth-odometry covariance with the yaw sigma floored (cross terms scaled to keep it PSD)."""
        floor = self.cfg.depth_odom_yaw_sigma_floor_rad
        if floor <= 0.0 or cov[2, 2] >= floor ** 2:
            return cov
        c = np.array(cov, float, copy=True)
        k = floor / math.sqrt(max(float(c[2, 2]), 1e-18))
        c[2, :2] *= k
        c[:2, 2] *= k
        c[2, 2] = floor ** 2
        return c

    @staticmethod
    def _gray(frame: SensorFrame) -> Optional[np.ndarray]:
        if frame.left_rgb is None:
            return None
        img = frame.left_rgb
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


__all__ = ["Localizer", "LocalizerConfig", "camera_to_body_increment", "wrap_angle"]
