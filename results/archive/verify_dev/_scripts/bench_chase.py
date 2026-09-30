import logging, sys, time
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, r"D:\Downloads\sih again\metagross")
cv2.setNumThreads(2)
logging.basicConfig(level=logging.INFO)
from metagross.sim.render.chase_replay import ThreeChaseFactory
from video.replay import RunReplay

run = Path(r"D:\Downloads\sih again\metagross\results\runs_integration\stereo\FULL\102")
out = Path(r"C:\Users\Asus\AppData\Local\Temp\claude\D--Downloads-sih-again\4eb046a2-5c84-43b5-91e3-06a29db230da\scratchpad")
t0 = time.perf_counter()
with ThreeChaseFactory() as fac:
    ch = fac.for_run(run)
    rp = RunReplay(run)
    img = ch(rp.world_state_at(0.0), 1152, 648)
    t1 = time.perf_counter()
    print("first frame (incl browser + load) s", t1 - t0)
    ts = []
    for k, t in enumerate(np.linspace(0, 20, 20)):
        a = time.perf_counter()
        img = ch(rp.world_state_at(float(t)), 1152, 648)
        ts.append(time.perf_counter() - a)
    print("chase 1152x648 ms mean %.1f p50 %.1f max %.1f" % (1e3 * np.mean(ts), 1e3 * np.median(ts), 1e3 * np.max(ts)))
    cv2.imwrite(str(out / "chase_102_t20.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    ts = []
    for k, t in enumerate(np.linspace(0, 20, 10)):
        a = time.perf_counter()
        img = ch(rp.world_state_at(float(t)), 936, 526)
        ts.append(time.perf_counter() - a)
    print("chase 936x526 ms mean %.1f" % (1e3 * np.mean(ts)))
