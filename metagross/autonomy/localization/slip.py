"""Wheel-slip monitor and online skid-steer effective-track (chi) estimator.

Inputs per tick (all planar, body frame):
* ``d_wheel`` (m): distance from the encoders, ``r (dphi_L + dphi_R) / 2``;
* ``dyaw_wheel_diff`` (rad): wheel-differential yaw *without* chi, ``r (dphi_R - dphi_L) / B``;
* ``d_vo`` / ``dyaw_vo``: the accepted VO increment for the same tick (None if VO
  was rejected - such ticks are excluded from both windows).

Slip ratio over a sliding ``window_s`` window: ``s = 1 - sum(d_VO) / sum(d_wheel)``
(0 = pure rolling, 1 = wheels spinning in place). The vehicle is declared
IMMOBILISED when ``s > immobilised_ratio`` continuously for ``immobilised_hold_s``
while motion is commanded.

Effective-track factor: for a skid-steer vehicle the true yaw increment is
``dyaw = r (dphi_R - dphi_L) / (chi B)``, so ``chi = dyaw_wheel_diff / dyaw_true``.
``chi_hat`` is an EMA of that ratio of rates over ticks with a clear VO yaw
rate, clipped to ``[chi_min, chi_max]``.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

from metagross.config.defaults import CHI_NOMINAL

LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class SlipConfig:
    window_s: float = 2.0
    immobilised_ratio: float = 0.6
    immobilised_hold_s: float = 3.0
    min_window_wheel_m: float = 0.05  # below this the slip ratio is undefined (reported 0)
    commanded_speed_mps: float = 0.05  # wheel speed that counts as "motion commanded" if not told
    chi_min: float = 1.0
    chi_max: float = 2.0
    chi_alpha: float = 0.05  # EMA gain per usable tick
    chi_min_yaw_rate_rps: float = 0.15  # need a clear turn to observe chi
    chi_ratio_clip: tuple[float, float] = (0.5, 3.0)  # reject absurd instantaneous ratios


class SlipEstimator:
    """Sliding-window slip ratio + IMMOBILISED latch + online chi estimate."""

    def __init__(self, config: Optional[SlipConfig] = None, chi0: float = CHI_NOMINAL) -> None:
        self.cfg = config or SlipConfig()
        self.chi0 = chi0
        self.reset()

    def reset(self) -> None:
        self._win: deque[tuple[float, float, float]] = deque()  # (t, d_vo, d_wheel)
        self.chi_hat = self.chi0
        self.slip = 0.0
        self.immobilised = False
        self._slip_since: Optional[float] = None
        self.n_chi_updates = 0

    def update(self, t: float, d_wheel: float, dyaw_wheel_diff: float, dt: float,
               d_vo: Optional[float], dyaw_vo: Optional[float], commanded: Optional[bool] = None) -> dict[str, float]:
        """Add one tick; returns {'slip', 'immobilised', 'chi_hat'}.

        ``commanded`` = motion was commanded this tick; if None it is inferred from
        the wheel speed (wheels turning implies a command reached the motors).
        """
        c = self.cfg
        if commanded is None:
            commanded = dt > 0 and abs(d_wheel) / dt > c.commanded_speed_mps
        if d_vo is not None:
            self._win.append((t, d_vo, d_wheel))
        while self._win and self._win[0][0] < t - c.window_s:
            self._win.popleft()
        sum_vo = sum(abs(w[1]) for w in self._win)
        sum_wh = sum(abs(w[2]) for w in self._win)
        self.slip = (1.0 - sum_vo / sum_wh) if sum_wh >= c.min_window_wheel_m else 0.0

        if commanded and self.slip > c.immobilised_ratio:
            if self._slip_since is None:
                self._slip_since = t
            if t - self._slip_since >= c.immobilised_hold_s and not self.immobilised:
                self.immobilised = True
                LOG.warning("IMMOBILISED: slip=%.2f for %.1f s", self.slip, t - self._slip_since)
        else:
            self._slip_since = None
            self.immobilised = False

        if dyaw_vo is not None and dt > 0 and abs(dyaw_vo) / dt >= c.chi_min_yaw_rate_rps:
            ratio = dyaw_wheel_diff / dyaw_vo  # (wheel-diff yaw rate) / (VO yaw rate)
            lo, hi = c.chi_ratio_clip
            if lo <= ratio <= hi:
                self.chi_hat += c.chi_alpha * (ratio - self.chi_hat)
                self.chi_hat = min(max(self.chi_hat, c.chi_min), c.chi_max)
                self.n_chi_updates += 1
        return {"slip": self.slip, "immobilised": float(self.immobilised), "chi_hat": self.chi_hat}
