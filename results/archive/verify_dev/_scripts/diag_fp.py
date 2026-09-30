"""False-positive audit of perception lethal cells vs GT hazard raster (diagnostic only)."""
import os, sys, math, json
os.environ.setdefault("OMP_NUM_THREADS", "2")
import cv2
cv2.setNumThreads(2)
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
REPO = Path(r"D:\Downloads\sih again\metagross")
sys.path.insert(0, str(REPO))
from metagross.sim.scenario import load_scenario
from metagross.sim.world import World
from metagross.autonomy.perception.pipeline import Perception
from metagross.autonomy.perception.bev import BevSpec
from metagross.config import defaults
from metagross.contracts.messages import CellState

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 102
scn = load_scenario(REPO / "data" / "scenarios" / "dev" / f"{seed}.json")
world = World(scn, sensor_mode="tier0")
per = Perception(world.calibration(), defaults.VEHICLE, {"use_semantics": False})
fr = world.make_sensor_frame(world.t, 0)
out = per.process(fr, (0, 0, 0))
st = out["cell_state_local"]
X, Y = BevSpec().centres()
s = world.state
c, sn = math.cos(s.yaw), math.sin(s.yaw)
wx = s.x + c * X - sn * Y
wy = s.y + sn * X + c * Y
hz = world.hazards
g = hz.grid
lethal = binary_dilation(hz.lethal, iterations=3)
ditch = binary_dilation(hz.ditch, iterations=3)
obj = binary_dilation(hz.object, iterations=3)
i, j = g.xy_to_ij(wx, wy)
print("grid res", g.res, "hazards:", [h["type"] for h in scn["hazards"]], "n objects", len(scn["objects"]))
ok = (i >= 0) & (i < lethal.shape[0]) & (j >= 0) & (j < lethal.shape[1])
def at(mask):
    out = np.zeros(st.shape, bool)
    out[ok] = mask[i[ok], j[ok]]
    return out
L, D, O = at(lethal), at(ditch), at(obj)
H = np.zeros(st.shape); H[ok] = world.terrain.height[i[ok], j[ok]]
for name in ("POSITIVE", "DEPRESSION", "DITCH_CANDIDATE"):
    m = st == CellState[name]
    print(f"{name}: n={m.sum()} in GT lethal(dil)={int((m & L).sum())} in GT ditch={int((m & D).sum())} in GT object={int((m & O).sum())}")
print("GT lethal cells in view grid (x>1.5):", int((L & (X > 1.5)).sum()))
print("terrain height range in grid:", H[ok].min(), H[ok].max(), "std", H[ok].std())
# slope near vehicle
print("height along x (y=0):", [round(float(H[k, 60]), 2) for k in range(20, 160, 10)])
print("gt pitch/roll", math.degrees(s.pitch), math.degrees(s.roll))
m = st == CellState.POSITIVE
fp = m & ~L
print("FP POSITIVE x hist", np.histogram(X[fp], bins=[0, 2, 4, 6, 8, 10, 12, 14])[0])
print("ground bands", out["ground_bands"])
print("timings", {k: round(v, 1) for k, v in out["timings_ms"].items()})
