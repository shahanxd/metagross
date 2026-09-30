import numpy as np, cv2, time, cProfile, pstats
cv2.setNumThreads(2)
from metagross.config import defaults
from metagross.contracts.messages import SensorFrame
from metagross.autonomy.perception.pipeline import Perception
from metagross.autonomy.perception import synthetic as syn
P = Perception(defaults.stereo_calibration())
L, R, _ = syn.render_stereo_pair(P.geom, syn.make_scene("trench"), seed=5)
f = SensorFrame(0.0, 0, L, R, 0, 0, 0)
for i in range(3): P.process(f)
tt = {}
for i in range(15):
    out = P.process(f)
    for k, v in out["timings_ms"].items(): tt.setdefault(k, []).append(v)
print({k: round(float(np.median(v)),1) for k, v in tt.items()})
cProfile.run("for i in range(10): P.process(f)", "C:/Users/Asus/AppData/Local/Temp/claude/D--Downloads-sih-again/4eb046a2-5c84-43b5-91e3-06a29db230da/scratchpad/p3.out")
pstats.Stats("C:/Users/Asus/AppData/Local/Temp/claude/D--Downloads-sih-again/4eb046a2-5c84-43b5-91e3-06a29db230da/scratchpad/p3.out").sort_stats("tottime").print_stats(22)
