import logging, sys, time
from pathlib import Path
import cv2
sys.path.insert(0, r"D:\Downloads\sih again\metagross")
cv2.setNumThreads(2)
logging.basicConfig(level=logging.WARNING)
from video import layout
from video.chase import CachedChase
from metagross.sim.render.chase_replay import ThreeChaseFactory
from video.replay import RunReplay

R = Path(r"D:\Downloads\sih again\metagross\results\runs_integration")
out = Path(r"C:\Users\Asus\AppData\Local\Temp\claude\D--Downloads-sih-again\4eb046a2-5c84-43b5-91e3-06a29db230da\scratchpad")
cache = Path(r"D:\Downloads\sih again\metagross\video\out\cache\chase")
which = sys.argv[1] if len(sys.argv) > 1 else "both"
with ThreeChaseFactory() as fac:
    if which in ("both", "dash"):
        run = R / "stereo/FULL/102"
        ch = CachedChase(fac.for_run(run), fac.for_run(run).scenario_sha, cache)
        rp = RunReplay(run, chase_renderer=ch, chase_size=layout.CHASE_SIZE)
        dash = layout.Dashboard(gt_at=lambda t: rp.gt_pose_at(t)[:3])
        t0 = time.perf_counter()
        img = dash.compose(rp.frame_at(12.0), caption=layout.Caption("Only on ground it has seen", "Green: measured ground · grey: not yet seen"), subtitle="The vehicle drives only on ground it has seen.")
        print("compose ms", (time.perf_counter() - t0) * 1e3)
        cv2.imwrite(str(out / "still_dash.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        t0 = time.perf_counter()
        for k in range(10):
            img = dash.compose(rp.frame_at(12.0 + k / 30), subtitle="The vehicle drives only on ground it has seen.")
        print("compose ms/frame (incl chase)", (time.perf_counter() - t0) * 1e2)
    if which in ("both", "split"):
        ra_dir, rb_dir = R / "tier0/TYPICAL/103", R / "stereo/FULL/103"
        size = (layout.SPLIT_W, layout.SPLIT_CHASE_H)
        ra = RunReplay(ra_dir, chase_renderer=CachedChase(fac.for_run(ra_dir), "103", cache), chase_size=size)
        rb = RunReplay(rb_dir, chase_renderer=CachedChase(fac.for_run(rb_dir), "103", cache), chase_size=size)
        da = layout.Dashboard("TYPICAL", gt_at=lambda t: ra.gt_pose_at(t)[:3])
        db = layout.Dashboard("FULL", gt_at=lambda t: rb.gt_pose_at(t)[:3])
        img = layout.compose_split(da, ra.frame_at(15.0), db, rb.frame_at(15.0), "TYPICAL STACK · unknown = free", "METAGROSS · unknown is never free",
                                   caption=layout.Caption("Same seed, same ditch field", "Draft: non-golden DEV runs"), subtitle="Same seed. Two stacks.")
        cv2.imwrite(str(out / "still_split.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
