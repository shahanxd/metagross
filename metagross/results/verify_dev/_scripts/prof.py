import cProfile, pstats, sys, time, logging
from pathlib import Path
import cv2
sys.path.insert(0, r"D:\Downloads\sih again\metagross")
cv2.setNumThreads(2)
from video import layout
from video.replay import RunReplay

run = Path(r"D:\Downloads\sih again\metagross\results\runs_integration\stereo\FULL\102")
rp = RunReplay(run)
dash = layout.Dashboard(gt_at=lambda t: rp.gt_pose_at(t)[:3])
dash.compose(rp.frame_at(1.0))
def go():
    for k in range(30):
        dash.compose(rp.frame_at(12.0 + k / 30), subtitle="The vehicle drives only on ground it has seen.")
t0 = time.perf_counter(); go(); print("ms/frame no chase", (time.perf_counter() - t0) / 30 * 1e3)
cProfile.run("go()", r"C:\Users\Asus\AppData\Local\Temp\claude\D--Downloads-sih-again\4eb046a2-5c84-43b5-91e3-06a29db230da\scratchpad\prof.out")
pstats.Stats(r"C:\Users\Asus\AppData\Local\Temp\claude\D--Downloads-sih-again\4eb046a2-5c84-43b5-91e3-06a29db230da\scratchpad\prof.out").sort_stats("cumulative").print_stats(25)
