"""In-process diagnostic closed loop (world + AutonomyStack in one process)."""
import os, sys, math, time, json
os.environ.setdefault("OMP_NUM_THREADS", "2")
import cv2
cv2.setNumThreads(2)
from pathlib import Path
import numpy as np
REPO = Path(r"D:\Downloads\sih again\metagross")
sys.path.insert(0, str(REPO))
from metagross.sim.scenario import load_scenario
from metagross.sim.world import World
from metagross.autonomy.node import AutonomyStack
from metagross.config import defaults
from metagross.contracts.messages import CellState, OperatorCmd, OperatorAction

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 102
T = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
cfg = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
every = int(sys.argv[4]) if len(sys.argv) > 4 else 5
scn = load_scenario(REPO / "data" / "scenarios" / "dev" / f"{seed}.json")
world = World(scn, sensor_mode="tier0", timeout_s=T)
stack = AutonomyStack()
stack.reset(world.mission_spec(), world.calibration(), defaults.VEHICLE, {"fixed_latency_s": None, **cfg})
stack.operator(OperatorCmd(0.0, OperatorAction.GO))
print("impl", stack.impl, "goal", world.mission_spec().goal_xy_a)
spf = int(round(defaults.PHYSICS_HZ / defaults.CAMERA_HZ_BATCH))
seq = 0
while not world.done:
    if world.step_index % spf == 0:
        fr = world.make_sensor_frame(world.t, seq)
        cmd, tel, dbg = stack.step(fr)
        world.queue_command(cmd)
        if seq % every == 0:
            st = dbg.cell_state_local
            ng = int((st == CellState.GROUND).sum()) if st is not None else -1
            ex = dbg.extras
            s = world.state
            print(f"t={world.t:5.1f} gt=({s.x:.1f},{s.y:.1f},{math.degrees(s.yaw):.0f}) pose=({dbg.pose_xy_yaw[0]:.2f},{dbg.pose_xy_yaw[1]:.2f},{math.degrees(dbg.pose_xy_yaw[2]):.0f}) "
                  f"mode={ex['mode']} reason={ex['reason']} vcap={dbg.v_cap_mps:.2f} rcert={dbg.r_cert_m:.2f} rpath={ex['r_path_m']:.2f} "
                  f"rvis={dbg.health.get('r_vis_m', 0):.1f} q={dbg.health['q']:.2f} nG={ng} route={ex['route_ok']} cmd_v={ex['cmd_v']:.2f} w={ex['cmd_w']:.2f} "
                  f"gov={ex['gov_binding']} ms={dbg.timings_ms.get('compute',0):.0f}")
        seq += 1
    world.step()
r = world.referee
print("done", r.success, r.failure_type, "t", round(world.t, 1), "L", round(world.path_length_m, 1), "ditch", r.ditch_entries, "coll", r.collisions)
