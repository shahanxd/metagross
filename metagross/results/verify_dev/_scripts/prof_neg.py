import numpy as np, time, cProfile, pstats
from metagross.config import defaults
from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.autonomy.perception import synthetic as syn
from metagross.autonomy.perception.ground import fit_ground
from metagross.autonomy.perception.negobs import MissingGroundDetector
g = CameraGeometry(defaults.stereo_calibration()); det = MissingGroundDetector(g)
d = syn.render_disparity(g, syn.make_scene("trench"), noise_px=0.15, seed=1)
pts = g.points_from_disparity(d, 2, 4, row_start=100)
gm = fit_ground(pts, d, g)
ts=[]
for i in range(8):
    t=time.perf_counter(); det.detect(d, gm); ts.append((time.perf_counter()-t)*1e3)
print("negobs ms", np.round(ts,1))
cProfile.run("for i in range(5): det.detect(d, gm)", "C:/Users/Asus/AppData/Local/Temp/claude/D--Downloads-sih-again/4eb046a2-5c84-43b5-91e3-06a29db230da/scratchpad/p2.out")
pstats.Stats("C:/Users/Asus/AppData/Local/Temp/claude/D--Downloads-sih-again/4eb046a2-5c84-43b5-91e3-06a29db230da/scratchpad/p2.out").sort_stats("tottime").print_stats(10)
