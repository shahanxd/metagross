import logging, sys, time, os
from pathlib import Path
sys.path.insert(0, r"D:\Downloads\sih again\metagross")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, cv2
cv2.setNumThreads(2)
logging.basicConfig(level=logging.INFO)
from metagross.sim.render.bridge import ThreeRenderer
from metagross.sim.render.testscene import make_test_scenario, route_poses

OUT = Path(r"C:\Users\Asus\AppData\Local\Temp\claude\D--Downloads-sih-again\4eb046a2-5c84-43b5-91e3-06a29db230da\scratchpad\smoke")
OUT.mkdir(exist_ok=True)
t0 = time.time()
sc = make_test_scenario(100)
print("scenario built", time.time() - t0)
with ThreeRenderer() as r:
    print("renderer", r.info)
    r.load_scenario(sc)
    print("load", r.last_load)
    poses = route_poses(sc, 6, 0, 20)
    for k, p in enumerate(poses):
        st = {"pose": p, "t": k * 0.2, "seq": k}
        L, R = r.render_stereo(st)
        print(k, r.last_js)
        if k in (0, 3, 5):
            cv2.imwrite(str(OUT / f"L{k}.png"), cv2.cvtColor(L, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(OUT / f"R{k}.png"), R)
            gt = r.render_gt(st)
            d = gt["depth"]; s = gt["semantic"]
            print("depth", np.nanmin(d), np.nanmax(d[np.isfinite(d)]), "sem ids", np.unique(s))
            dv = np.where(np.isfinite(d), np.clip(d / 30 * 255, 0, 255), 255).astype(np.uint8)
            cv2.imwrite(str(OUT / f"D{k}.png"), dv)
            cv2.imwrite(str(OUT / f"S{k}.png"), (s * 60).astype(np.uint8))
    ch = r.render_chase({"pose": poses[3], "t": 0.6}, 960, 540)
    cv2.imwrite(str(OUT / "chase.png"), cv2.cvtColor(ch, cv2.COLOR_RGB2BGR))
    print(r.stats())
