"""Where do ditch-candidate segments come from on a ditch-free F1 scene?"""
import os, sys
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
from metagross.config import defaults

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 102
world = World(load_scenario(REPO / "data" / "scenarios" / "dev" / f"{seed}.json"), sensor_mode="tier0")
cap = {}
orig = P.MissingGroundDetector.detect
def spy(self, disp, model, use_negobs=True):
    r = orig(self, disp, model, use_negobs)
    cap["r"] = r
    return r
P.MissingGroundDetector.detect = spy
per = P.Perception(world.calibration(), defaults.VEHICLE, {"use_semantics": False})
per.process(world.make_sensor_frame(0.0, 0), (0, 0, 0))
g = cap["r"].gap_debug
d = g["is_ditch"] & g["keep"]
print("gaps", len(g["col"]), "ditch", int(d.sum()), "crest", int((g["is_crest"] & g["keep"]).sum()))
print("x_lip hist of ditch", np.histogram(g["x_lip"][d], bins=np.arange(0, 13, 1))[0])
print("ditch via below&plateau&jump:", int((d & g["has_pts"]).sum()), " via no-points:", int((d & ~g["has_pts"]).sum()))
print("rel_jump pct", np.percentile(g["rel_jump"][d & g["has_pts"]], [10, 50, 90]) if (d & g["has_pts"]).any() else None)
print("dzc pct", np.percentile(g["dzc"][d & g["has_pts"]], [10, 50, 90]) if (d & g["has_pts"]).any() else None)
print("n_exp pct", np.percentile(g["n_exp"][d], [10, 50, 90]), "length pct", np.percentile(g["length"][d], [10, 50, 90]))
