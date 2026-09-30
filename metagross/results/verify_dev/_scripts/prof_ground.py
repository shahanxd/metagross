import numpy as np, time, cProfile, pstats
from metagross.config import defaults
from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.autonomy.perception import synthetic as syn, ground as gr
g = CameraGeometry(defaults.stereo_calibration())
d = syn.render_disparity(g, syn.make_scene("box"), noise_px=0.1, seed=1)
pts = g.points_from_disparity(d, 2, 4, row_start=100)
ts=[]
for i in range(10):
    t=time.perf_counter(); gr.fit_ground(pts, d, g); ts.append((time.perf_counter()-t)*1e3)
print("fit_ground ms", np.round(ts,1))
ts=[]
for i in range(10):
    t=time.perf_counter(); gr.fit_v_disparity(d, g, np.random.default_rng(0)); ts.append((time.perf_counter()-t)*1e3)
print("vdisp ms", np.round(ts,1))
ts=[]
for i in range(10):
    t=time.perf_counter(); g.points_from_disparity(d, 2, 4, row_start=100); ts.append((time.perf_counter()-t)*1e3)
print("points ms", np.round(ts,1))
cProfile.run("for i in range(10): gr.fit_ground(pts, d, g)", "C:/Users/Asus/AppData/Local/Temp/claude/D--Downloads-sih-again/4eb046a2-5c84-43b5-91e3-06a29db230da/scratchpad/p.out")
pstats.Stats("C:/Users/Asus/AppData/Local/Temp/claude/D--Downloads-sih-again/4eb046a2-5c84-43b5-91e3-06a29db230da/scratchpad/p.out").sort_stats("tottime").print_stats(12)
