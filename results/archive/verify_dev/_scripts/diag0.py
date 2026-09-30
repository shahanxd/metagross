"""Inspect tick-0 perception/map/costmap along the probe for a DEV seed."""
import os, sys, math, json
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
from metagross.autonomy.planning.governor import probe_path
from metagross.config import defaults
from metagross.contracts.messages import CellState

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 102
scn = load_scenario(REPO / "data" / "scenarios" / "dev" / f"{seed}.json")
world = World(scn, sensor_mode="tier0")
stack = AutonomyStack()
stack.reset(world.mission_spec(), world.calibration(), defaults.VEHICLE, {"use_health": False})
fr = world.make_sensor_frame(world.t, 0)
d = fr.disparity
print("disp valid frac", float((d > 0).mean()), "rows with valid", np.nonzero((d > 0).any(1))[0][[0, -1]])
cmd, tel, dbg = stack.step(fr)
st = dbg.cell_state_local
for s in CellState:
    print(f"  {s.name:16s} {(st == s).sum()}")
# where are lethal cells in body frame?
from metagross.autonomy.perception.bev import BevSpec
X, Y = BevSpec().centres()
for s in (CellState.POSITIVE, CellState.DEPRESSION, CellState.DITCH_CANDIDATE):
    m = st == s
    if m.any():
        print(s.name, "x range", X[m].min(), X[m].max(), "median x", np.median(X[m]), "y range", Y[m].min(), Y[m].max())
# ground cells in x bins
m = st == CellState.GROUND
print("GROUND x hist", np.histogram(X[m], bins=[-2, 0, 1, 2, 3, 4, 6, 8, 10, 14])[0])
maps = stack.last_maps
pose = dbg.pose_xy_yaw
probe = probe_path(np.zeros((0, 2)), pose, 12.0)
for k in range(0, 60, 3):
    x, y = probe[k]
    cert = maps.sample(maps.certified, np.array([x]), np.array([y]), fill=False)[0]
    c = maps.sample(maps.cost, np.array([x]), np.array([y]), fill=np.float32(-1))[0]
    s = maps.sample(stack.map.state, np.array([x]), np.array([y]), fill=np.uint8(99))[0]
    clr = maps.sample(maps.clearance_m, np.array([x]), np.array([y]), fill=np.float32(-1))[0]
    print(f"probe x={x:.1f} y={y:.2f} cert={cert} cost={c:.2f} state={s} clr={clr:.2f}")
gp = stack.global_planner.plan
print("route_ok", gp.route_ok, "lookahead", gp.lookahead_xy, "ctg", gp.ctg_vehicle)
print("path head", np.round(gp.path_xy[:12], 2).tolist())
print("mppi plan head", np.round(dbg.plan_xy[:8], 2).tolist())
steps, geo = stack.global_planner.step_costs(maps, False)
for (x, y) in [(1, 0), (2, 0), (3, 0), (4, 0), (6, 0), (8, 0), (-1, 0), (-3, 0), (0, 3), (0, -3)]:
    ix, iy = geo.to_index(np.array([x]), np.array([y]))
    print(f"step cost at ({x},{y}) = {steps[iy[0], ix[0]]:.2f}  map cost={maps.sample(maps.cost, np.array([x]), np.array([y]), fill=np.float32(-1))[0]:.2f} unseen={maps.sample(maps.unseen, np.array([x]), np.array([y]), fill=False)[0]}")
print("health",{k: round(v, 3) for k, v in dbg.health.items()})
print("timings", {k: round(v, 1) for k, v in dbg.timings_ms.items()})
