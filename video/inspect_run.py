"""Print a one-line-per-second summary of a run, to pick scene start times (``t0_*``) for the video.

    python -m video.inspect_run results/golden/stereo/FULL/104 [--every 1.0]

Columns: run time t [s], GT distance travelled [m], GT speed [m/s], drive mode, reason,
v_cap [m/s], R_cert [m], integrity q, and referee events (ditch entry, collision, lighting on/off,
dynamic trigger) that happened in that second. Offline tooling: reads GT, never used by autonomy.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np

from video.replay import RunReplay

log = logging.getLogger(__name__)


def summary_rows(run_dir: str | Path, every_s: float = 1.0) -> list[dict]:
    """Per-``every_s`` rows (see module docstring)."""
    rp = RunReplay(run_dir)
    dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(rp.gt_xa), np.diff(rp.gt_ya)))])
    rows = []
    for t in np.arange(rp.t_start, rp.t_end + 1e-9, every_s):
        pd = rp.frame_at(float(t))
        ev = [str(e.get("type", "")) for e in rp.events if t - every_s < float(e.get("t", -1e9)) <= t]
        rows.append({"t": float(t), "dist_m": float(np.interp(t, rp.gt_t, dist)), "v": pd.gt_speed_mps, "mode": pd.mode,
                     "reason": pd.reason, "v_cap": pd.v_cap_mps, "r_cert": pd.r_cert_m, "q": pd.q, "events": ev})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--every", type=float, default=1.0, help="row spacing, s")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rp = RunReplay(a.run_dir)
    res = rp.result
    log.info(f"{a.run_dir}: seed {res.get('seed')} {res.get('family')} config {rp.mission.config_name} sensor {res.get('sensor_mode')} "
          f"success {res.get('success')} failure {res.get('failure_type')} time {res.get('time')} s ditch {res.get('ditch_entries')}")
    log.info(f"{'t':>6} {'dist':>6} {'v':>5} {'mode':<14} {'v_cap':>5} {'Rcert':>5} {'q':>5}  reason / events")
    for r in summary_rows(a.run_dir, a.every):
        q = f"{r['q']:.2f}" if math.isfinite(r["q"]) else "  -  "
        log.info(f"{r['t']:6.1f} {r['dist_m']:6.1f} {r['v']:5.2f} {r['mode']:<14} {r['v_cap']:5.2f} {r['r_cert']:5.1f} {q:>5}  "
              f"{r['reason']}{'  <' + ', '.join(r['events']) + '>' if r['events'] else ''}")


if __name__ == "__main__":
    main()
