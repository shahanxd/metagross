"""Split POSITIVE cells by cause (step vs slope) at tick 0."""
import os, sys, math
os.environ.setdefault("OMP_NUM_THREADS", "2")
import cv2
cv2.setNumThreads(2)
from pathlib import Path
import numpy as np
REPO = Path(r"D:\Downloads\sih again\metagross")
sys.path.insert(0, str(REPO))
from metagross.sim.scenario import load_scenario
from metagross.sim.world import World
from metagross.autonomy.perception import pipeline as P
from metagross.autonomy.perception import positive as POS
from metagross.autonomy.perception.bev import BevSpec
from metagross.config import defaults

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 102
scn = load_scenario(REPO / "data" / "scenarios" / "dev" / f"{seed}.json")
world = World(scn, sensor_mode="tier0")
cap = {}
orig = POS.terrain_costs
def spy(stats, spec, ground_z, observed, level=None):
    r = orig(stats, spec, ground_z, observed, level)
    cap.update(stats=stats, gz=ground_z, obs=observed, lvl=level, r=r)
    return r
P.terrain_costs = spy
per = P.Perception(world.calibration(), defaults.VEHICLE, {"use_semantics": False})
out = per.process(world.make_sensor_frame(0.0, 0), (0, 0, 0))
st, r, stats = cap["stats"], cap["r"], cap["stats"]
has = stats.count > 0
slope_l = cap["obs"] & (r.slope_deg > defaults.SLOPE_LETHAL_DEG)
step_l = r.positive & ~slope_l
print("positive", r.positive.sum(), "slope-lethal", slope_l.sum(), "slope-lethal & no points", (slope_l & ~has).sum(), "step-only", step_l.sum())
X, Y = BevSpec().centres()
print("slope-lethal x hist", np.histogram(X[slope_l], bins=[0, 2, 4, 6, 8, 10, 12, 14])[0])
print("level stats (has):", np.percentile(cap["lvl"][has], [5, 50, 95]))
print("h_mean stats:", np.nanpercentile(stats.h_mean[has], [5, 50, 95]))
from scipy.ndimage import binary_dilation
s = world.state
c, sn = math.cos(s.yaw), math.sin(s.yaw)
wx = s.x + c * X - sn * Y; wy = s.y + sn * X + c * Y
g = world.hazards.grid
L = binary_dilation(world.hazards.lethal, iterations=3)
i, j = g.xy_to_ij(wx, wy)
ok = (i >= 0) & (i < L.shape[0]) & (j >= 0) & (j < L.shape[1])
GT = np.zeros(X.shape, bool); GT[ok] = L[i[ok], j[ok]]
lvl = cap["lvl"]
for name, m in (("FP", step_l & ~GT), ("TP", step_l & GT)):
    print(name, m.sum(), "count med", np.median(stats.count[m]), "hmax-lvl med", np.median((stats.h_max - lvl)[m]),
          "hmean-lvl med", np.median((stats.h_mean - lvl)[m]), "hstd med", np.median(stats.h_std[m]), "x med", np.median(X[m]))
    print("   hmean-lvl pct", np.percentile((stats.h_mean - lvl)[m], [10, 50, 90]), "count pct", np.percentile(stats.count[m], [10, 50, 90]))
m = slope_l & has & ~GT
print("slope FP cells: hmax-lvl pct", np.percentile((stats.h_max - lvl)[m], [10, 50, 90]), "hmean-lvl", np.percentile((stats.h_mean - lvl)[m], [10, 50, 90]))
print("slope FP: slope pct", np.percentile(r.slope_deg[m], [10, 50, 90]), "count", np.percentile(stats.count[m], [10, 50, 90]))
print("all has cells hmax-lvl > 1.0:", int((has & (stats.h_max - lvl > 1.0)).sum()))
near = r.positive & (X < 4.0)
for ii, jj in zip(*np.nonzero(near)):
    print(f"  near POS x={X[ii,jj]:.2f} y={Y[ii,jj]:.2f} cnt={stats.count[ii,jj]} hmax={stats.h_max[ii,jj]:.3f} hmean={stats.h_mean[ii,jj]:.3f} lvl={lvl[ii,jj]:.3f} slope={r.slope_deg[ii,jj]:.1f} GT={GT[ii,jj]}")
dep = out["cell_state_local"] == 3
for ii, jj in list(zip(*np.nonzero(dep & (X < 4))))[:10]:
    print(f"  near DEP x={X[ii,jj]:.2f} y={Y[ii,jj]:.2f} cnt={stats.count[ii,jj]} hmin={stats.h_min[ii,jj]:.3f} lvl={lvl[ii,jj]:.3f} GT={GT[ii,jj]}")
GTc = GT & has & (X > 1.5)
print("GT cells with points", GTc.sum(), "detected", (GTc & r.positive).sum())
