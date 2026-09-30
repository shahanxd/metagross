"""In-process closed loop (World + AutonomyStack) with per-tick instrumentation. DEV seeds only."""
import json, logging, math, sys, time
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(2)
sys.path.insert(0, r"D:\Downloads\sih again\metagross")
from metagross.config import defaults
from metagross.contracts import ipc
from metagross.contracts.messages import OperatorCmd, OperatorAction
from metagross.sim.scenario import load_scenario
from metagross.sim.world import World
from metagross.autonomy.node import AutonomyStack

logging.basicConfig(level=logging.WARNING)


def run(scn, cfg, max_s=40.0, verbose=True, every=1):
    assert 100 <= int(scn["seed"]) <= 129 or int(scn["seed"]) >= 9000, "DEV only"
    w = World(scn, sensor_mode="tier0", timeout_s=max_s)
    st = AutonomyStack()
    c = {**ipc.DEFAULT_AUTONOMY_CONFIG, **cfg}
    st.reset(w.mission_spec(), w.calibration(), defaults.VEHICLE, c)
    st.operator(OperatorCmd(t=0.0, action=OperatorAction.GO))
    spf = int(round(defaults.PHYSICS_HZ / defaults.CAMERA_HZ_BATCH))
    seq = 0
    rows = []
    while not w.done:
        if w.step_index % spf == 0:
            fr = w.make_sensor_frame(w.t, seq)
            cmd, tel, dbg = st.step(fr)
            w.queue_command(cmd)
            ex = dbg.extras
            m = st.last_maps
            pose = dbg.pose_xy_yaw
            clr = float(m.sample(m.clearance_m, np.array([pose[0]]), np.array([pose[1]]), 1e3)[0])
            # lethal cells within 3 m
            g = m.geometry
            ix, iy = g.to_index(np.array([pose[0]]), np.array([pose[1]]))
            r = int(3.0 / g.res)
            sub = m.lethal_core[max(iy[0]-r,0):iy[0]+r, max(ix[0]-r,0):ix[0]+r]
            n_leth = int(sub.sum())
            s = w.state
            row = dict(t=round(w.t, 2), x=round(s.x, 2), y=round(s.y, 2), v_gt=round(s.v, 3), w_gt=round(s.omega, 3),
                       cmd_v=round(ex["cmd_v"], 3), cmd_w=round(ex["cmd_w"], 3), vcap=round(dbg.v_cap_mps, 2),
                       mode=ex["mode"], reason=ex["reason"], gov=ex["gov_binding"], rcert=round(dbg.r_cert_m, 1),
                       route=ex["route_ok"], ctg=round(ex["ctg_vehicle"], 1) if math.isfinite(ex["ctg_vehicle"]) else None,
                       clr=round(clr, 2), nleth3=n_leth, ess=round(ex["mppi_ess"], 1),
                       terms={k: round(v, 1) for k, v in ex["mppi_terms"].items()},
                       wl=round(cmd.omega_l_rad_s, 2), wr=round(cmd.omega_r_rad_s, 2), cms=round(cmd.compute_ms, 0),
                       la=None if ex["lookahead_xy"] is None else [round(ex["lookahead_xy"][0], 1), round(ex["lookahead_xy"][1], 1)],
                       pose=[round(p, 2) for p in pose])
            rows.append(row)
            if verbose and seq % every == 0:
                print(row["t"], row["v_gt"], row["cmd_v"], row["cmd_w"], row["vcap"], row["mode"][:6], row["reason"][:24], row["ctg"], row["clr"], row["nleth3"], row["rcert"], row["la"], row["pose"])
            seq += 1
        w.step()
    ref = w.referee
    res = dict(success=ref.success, failure=ref.failure_type, t=round(w.t, 1), L=round(w.path_length_m, 2),
               v_mean=round(w.path_length_m / max(w.t, 1e-6), 3), ditch=ref.ditch_entries, coll=ref.collisions, final_err=round(float(np.hypot(w.state.x-w.goal_xy[0], w.state.y-w.goal_xy[1])),2), pose_est=rows[-1]["pose"], gt=[round(w.state.x,2), round(w.state.y,2)], slip=scn["vehicle"])
    print("RESULT", json.dumps(res))
    return rows, res


if __name__ == "__main__":
    seed = int(sys.argv[1])
    cfgname = sys.argv[2]
    max_s = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    every = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    cfgs = {
        "FULL": {},
        "FULL_NONEG": {"use_negobs": False},
        "TYPICAL": {"name": "TYPICAL", "unknown_is_free": True, "use_negobs": False, "use_governor": False, "fixed_speed_mps": 1.5},
    }
    scn = load_scenario(Path(r"D:\Downloads\sih again\metagross\data\scenarios\dev") / f"{seed}.json")
    run(scn, cfgs[cfgname], max_s, every=every)
