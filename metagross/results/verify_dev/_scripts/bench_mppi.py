import sys, time
sys.path.insert(0, r"D:\Downloads\sih again\metagross")
import numpy as np, cv2
cv2.setNumThreads(2)
from metagross.autonomy.planning.rolling_map import RollingMap
from metagross.autonomy.planning.costmap import build_costmap, CostmapParams
from metagross.autonomy.planning.global_planner import GlobalPlanner
from metagross.autonomy.planning.mppi import MppiPlanner, MppiParams
from metagross.contracts.messages import CellState
rm = RollingMap(); rm.seed_apron((0,0,0), 0.0, 30.0)
g = rm.geometry
ix, iy = g.to_index(np.array([7.0]), np.array([0.0]))
rm.state[iy[0]-4:iy[0]+4, ix[0]-4:ix[0]+4] = CellState.POSITIVE
maps = build_costmap(rm, 0.0, True, CostmapParams())
gp = GlobalPlanner(); plan = gp.update(0.0, maps, (0,0), (14,0))
mp = MppiPlanner(MppiParams(), seed=0)
def tm(f, n=30):
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); f(); ts.append((time.perf_counter()-t)*1e3)
    return np.median(ts), np.percentile(ts, 90)
print("costmap", tm(lambda: build_costmap(rm, 0.0, True, CostmapParams())))
print("global step_costs+trace", tm(lambda: gp.update(0.5, maps, (0,0), (14,0))))
print("global recompute", tm(lambda: gp.update(0.5, maps, (0,0), (14,0), force=True), 10))
k=[0]
def run():
    k[0]+=1; mp.plan(k[0]*0.2, (0.0,0.0,0.0), 1.0, 0.0, 2.0, maps, plan.ctg_at, 0.5)
print("mppi", tm(run))
import cProfile, pstats
pr=cProfile.Profile(); pr.enable()
for _ in range(20): run()
pr.disable(); pstats.Stats(pr).sort_stats("tottime").print_stats(12)
