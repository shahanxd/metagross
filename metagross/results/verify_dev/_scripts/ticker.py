"""Compact per-debug-bundle timeline of one run (verifier). Usage: ticker.py <run_dir> t0 t1"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\Downloads\sih again\metagross")
from metagross.contracts.messages import CellState as S  # noqa: E402

run, t0, t1 = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
for f in sorted((run / "autonomy" / "debug").glob("tick_*.npz")):
    z = np.load(f, allow_pickle=True)
    t = float(z["t"])
    if not (t0 <= t <= t1):
        continue
    ex = json.loads(str(z["extras_json"]))
    cs = z["cell_state_local"]
    crop = z["extra_map_crop_state"]
    conf = z["extra_map_crop_confirmed"]
    gp = z["global_path_xy"]
    pose = z["pose_xy_yaw"]
    end = gp[-1] if len(gp) else [np.nan, np.nan]
    mid = gp[min(len(gp) - 1, 25)] if len(gp) else [np.nan, np.nan]
    print(f"t={t:5.1f} pose=({pose[0]:.1f},{pose[1]:.1f},{np.degrees(pose[2]):.0f}) {ex['mode']:13s} {ex['reason']:18s} route={int(ex['route_ok'])} "
          f"ctg={ex['ctg_vehicle']:.0f} v_cap={float(z['v_cap_mps']):.2f} Rc={float(z['r_cert_m']):.1f} cmd_v={ex['cmd_v']:.2f} | "
          f"BEV ditch={(cs == S.DITCH_CANDIDATE).sum()} dep={(cs == S.DEPRESSION).sum()} pos={(cs == S.POSITIVE).sum()} crest={(cs == S.CREST_SHADOW).sum()} "
          f"| map ditch_conf={((crop == S.DITCH_CANDIDATE) & conf).sum()} ditch_all={(crop == S.DITCH_CANDIDATE).sum()} dep={(crop == S.DEPRESSION).sum()} "
          f"pos={(crop == S.POSITIVE).sum()} | path@5m=({mid[0]:.1f},{mid[1]:.1f}) end=({end[0]:.1f},{end[1]:.1f}) n={len(gp)}")
