"""Seen-distance speed governor: *only as fast as it can see*.

The vehicle must always be able to stop inside ground it has certified:

    d_stop(v) = v^2 / (2 a) + v * T_r + B  <=  R_cert

Solving for v gives the closed-form cap (``v_from_range``):

    v_cap = a * ( -T_r + sqrt(T_r^2 + 2 (R_cert - B) / a) ),   0 if R_cert <= B

with
* ``a = min(BRAKE_DECEL_MPS2, mu * g)`` (mu = lowest friction proxy on the probed path),
* ``T_r`` = compute latency + one camera frame + actuator lag. The compute latency is the
  ``LATENCY_QUANTILE`` percentile of the last ``LATENCY_WINDOW_N`` measured ticks, clamped to
  [``LATENCY_MIN_S``, ``LATENCY_MAX_S``] (:class:`LatencyEstimator`). The floor is the 200 ms design
  compute budget: while the stack runs inside its budget, T_r (and so the speed cap) does not
  follow machine load, which makes closed-loop outcomes reproducible; above it T_r grows with the
  measured tail latency (a slow stack must drive slower), up to the ceiling,
* ``B = GOVERNOR_MARGIN_M``,
* ``R_cert = min(arc length along the MPPI nominal path to the first non-certified
  cell, r_vis) * health_factor``. The nominal path is extended straight along its
  final heading so that a stationary vehicle still probes the ground ahead.

A static cap comes from ditch detectability (Matthies & Rankin 2003): a ditch of
width w at range R subtends ``theta ~ H w / (R (R + w))`` rad for a camera at
height H; it is detectable while ``theta * fx >= MIN_PIXELS_ON_TARGET``. Solving
the quadratic gives ``R_det``; the vehicle must be able to stop within ``R_det``.
Finally the platform cap ``VehicleSpec.max_speed_mps`` applies. The binding term
is reported for telemetry.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from metagross.autonomy.planning.costmap import PlanningMaps
from metagross.config.defaults import (
    ACTUATOR_LAG_S,
    BRAKE_DECEL_MPS2,
    CAM_HEIGHT_M,
    CAM_MAX_RANGE_M,
    DESIGN_DITCH_WIDTH_M,
    FX,
    GOVERNOR_MARGIN_M,
    MIN_PIXELS_ON_TARGET,
    VEHICLE,
)

G_MPS2 = 9.81
Q_NOMINAL = 0.7  # health q above which the full certified range is trusted
PROBE_STEP_M = 0.1  # arc-length sampling step when searching for the first non-certified cell
FRONT_OFFSET_M = VEHICLE.length_m / 2.0  # probe starts at the front bumper
LATENCY_WINDOW_N = 25  # ticks in the reaction-latency window (5 s at the 5 Hz batch camera rate)
LATENCY_QUANTILE = 90.0  # percentile of the window used as the compute latency
LATENCY_MIN_S = 0.20  # floor = design compute budget per tick [s]
LATENCY_MAX_S = 0.60  # ceiling [s]; beyond it the watchdog / health, not the governor, must act


class LatencyEstimator:
    """Compute latency used in the reaction time: p``quantile`` of the last ``window_n`` ticks,
    clamped to [``min_s``, ``max_s``] seconds; ``min_s`` before the first sample."""

    def __init__(self, window_n: int = LATENCY_WINDOW_N, quantile: float = LATENCY_QUANTILE,
                 min_s: float = LATENCY_MIN_S, max_s: float = LATENCY_MAX_S) -> None:
        self._buf: deque[float] = deque(maxlen=max(1, int(window_n)))
        self.quantile, self.min_s, self.max_s = float(quantile), float(min_s), float(max_s)

    def push(self, latency_s: float) -> None:
        """Add one measured tick latency [s] (non-finite values are ignored)."""
        if math.isfinite(latency_s) and latency_s >= 0.0:
            self._buf.append(float(latency_s))

    def value(self) -> float:
        """Clamped windowed percentile [s]."""
        if not self._buf:
            return self.min_s
        return float(np.clip(np.percentile(np.fromiter(self._buf, float), self.quantile), self.min_s, self.max_s))


def stopping_distance(v: float | np.ndarray, a: float, t_r: float, margin: float) -> float | np.ndarray:
    """Distance [m] to stop from speed v [m/s] with decel a [m/s^2], reaction time t_r [s], margin [m]."""
    return np.asarray(v) ** 2 / (2.0 * a) + np.asarray(v) * t_r + margin


def v_from_range(r_m: float, a: float, t_r: float, margin: float) -> float:
    """Largest speed [m/s] whose stopping distance fits in range ``r_m`` [m] (closed form)."""
    if r_m <= margin:
        return 0.0
    return float(a * (-t_r + math.sqrt(t_r * t_r + 2.0 * (r_m - margin) / a)))


def ditch_detection_range(h_m: float, w_m: float, fx_px: float, n_px: float) -> float:
    """Range [m] out to which a ditch of width w is resolved by >= n_px pixels.

    Solves ``h w / (R (R + w)) = n_px / fx`` for R > 0."""
    k = h_m * w_m * fx_px / n_px
    return float((-w_m + math.sqrt(w_m * w_m + 4.0 * k)) / 2.0)


@dataclass(frozen=True, slots=True)
class GovernorParams:
    brake_decel_mps2: float = BRAKE_DECEL_MPS2
    margin_m: float = GOVERNOR_MARGIN_M
    actuator_lag_s: float = ACTUATOR_LAG_S
    cam_height_m: float = CAM_HEIGHT_M
    fx_px: float = FX
    design_ditch_w_m: float = DESIGN_DITCH_WIDTH_M
    min_px: float = MIN_PIXELS_ON_TARGET
    v_platform_mps: float = VEHICLE.max_speed_mps
    max_probe_m: float = CAM_MAX_RANGE_M


@dataclass(slots=True)
class GovernorResult:
    v_cap_mps: float
    r_cert_m: float
    r_path_m: float  # certified arc length along the probe path (before r_vis / health)
    r_vis_m: float
    health_factor: float
    a_mps2: float
    t_r_s: float
    terms: dict[str, float] = field(default_factory=dict)  # candidate caps [m/s]
    binding: str = ""  # key of the binding term
    reason: str = ""  # short telemetry reason, e.g. "GOV_RCERT R=3.1m"


def probe_path(nominal_xy: np.ndarray, pose_xy_yaw: tuple[float, float, float], length_m: float, step_m: float = PROBE_STEP_M) -> np.ndarray:
    """Arc-length-uniform probe points starting at the front bumper.

    Follows the MPPI nominal path (A-frame (T,2)), then continues straight along its final
    heading (or the vehicle heading when the path is degenerate) up to ``length_m``."""
    x, y, yaw = pose_xy_yaw
    pts = [np.array([x, y])]
    if nominal_xy is not None and len(nominal_xy) > 0:
        pts.extend(np.asarray(nominal_xy, np.float64))
    poly = np.array(pts)
    seg = np.hypot(*np.diff(poly, axis=0).T) if len(poly) > 1 else np.zeros(0)
    keep = np.concatenate([[True], seg > 1e-6])
    poly = poly[keep]
    if len(poly) >= 2:
        d = poly[-1] - poly[-2]
        hd = math.atan2(d[1], d[0])
    else:
        hd = yaw
    total_needed = length_m + FRONT_OFFSET_M
    seg = np.hypot(*np.diff(poly, axis=0).T) if len(poly) > 1 else np.zeros(0)
    have = float(seg.sum())
    if have < total_needed:
        poly = np.vstack([poly, poly[-1] + (total_needed - have + step_m) * np.array([math.cos(hd), math.sin(hd)])])
        seg = np.hypot(*np.diff(poly, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(FRONT_OFFSET_M, total_needed + 1e-9, step_m)
    px = np.interp(s, cum, poly[:, 0])
    py = np.interp(s, cum, poly[:, 1])
    return np.column_stack([px, py])


def certified_arclength(maps: PlanningMaps, probe_xy: np.ndarray, step_m: float = PROBE_STEP_M) -> tuple[float, float]:
    """(arc length [m] from the bumper to the first non-certified probe point, min mu on the way)."""
    cert = maps.sample(maps.certified, probe_xy[:, 0], probe_xy[:, 1], fill=False)
    bad = np.flatnonzero(~cert)
    n_ok = int(bad[0]) if len(bad) else len(cert)
    mu_vals = maps.sample(maps.mu, probe_xy[: max(n_ok, 1), 0], probe_xy[: max(n_ok, 1), 1], fill=np.float32(0.4))
    return n_ok * step_m, float(np.min(mu_vals)) if len(mu_vals) else 0.4


class SpeedGovernor:
    """Computes ``v_cap`` and ``R_cert`` each tick; stateless apart from parameters."""

    def __init__(self, params: GovernorParams = GovernorParams()) -> None:
        self.p = params
        self.r_det_m = ditch_detection_range(params.cam_height_m, params.design_ditch_w_m, params.fx_px, params.min_px)

    def reaction_time(self, latency_s: float, frame_period_s: float) -> float:
        """T_r = measured compute latency + one frame + actuator lag [s]."""
        return float(latency_s + frame_period_s + self.p.actuator_lag_s)

    def compute(
        self,
        maps: PlanningMaps,
        pose_xy_yaw: tuple[float, float, float],
        nominal_xy: np.ndarray,
        r_vis_m: float,
        q_health: float,
        latency_s: float,
        frame_period_s: float,
        enabled: bool = True,
        use_health: bool = True,
        r_det_m: Optional[float] = None,
    ) -> GovernorResult:
        """Evaluate all caps. With ``enabled=False`` only the platform cap applies.
        ``r_det_m``: perception's measured detection range of the design ditch [m]; when given
        (finite, > 0) it replaces the closed-form Matthies-Rankin range in the ``ditch_det`` term."""
        p = self.p
        r_det = float(r_det_m) if r_det_m is not None and np.isfinite(r_det_m) and r_det_m > 0.0 else self.r_det_m
        t_r = self.reaction_time(latency_s, frame_period_s)
        probe = probe_path(nominal_xy, pose_xy_yaw, p.max_probe_m)
        r_path, mu_min = certified_arclength(maps, probe)
        a = min(p.brake_decel_mps2, mu_min * G_MPS2)
        hf = min(1.0, max(q_health, 0.0) / Q_NOMINAL) if use_health else 1.0
        r_vis = float(r_vis_m) if np.isfinite(r_vis_m) else p.max_probe_m
        r_cert = min(r_path, r_vis) * hf
        terms = {"platform": p.v_platform_mps}
        if enabled:
            terms["r_cert"] = v_from_range(r_cert, a, t_r, p.margin_m)
            terms["ditch_det"] = v_from_range(r_det, a, t_r, p.margin_m)
        binding = min(terms, key=terms.__getitem__)
        v_cap = max(0.0, terms[binding])
        if binding == "r_cert":
            src = "RVIS" if r_vis < r_path else "RCERT"
            reason = f"GOV_{src} R={r_cert:.1f}m"
            binding = "r_vis" if src == "RVIS" else "r_cert"
        elif binding == "ditch_det":
            reason = f"GOV_DITCH_DET R={r_det:.1f}m"
        else:
            reason = "GOV_PLATFORM" if enabled else "GOV_OFF"
        return GovernorResult(
            v_cap_mps=v_cap, r_cert_m=r_cert, r_path_m=r_path, r_vis_m=r_vis, health_factor=hf, a_mps2=a,
            t_r_s=t_r, terms=terms, binding=binding, reason=reason,
        )
