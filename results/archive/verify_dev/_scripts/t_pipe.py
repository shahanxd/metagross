import time, numpy as np, cv2, os
os.environ.setdefault("OMP_NUM_THREADS", "2")
cv2.setNumThreads(2)
from metagross.config import defaults
from metagross.contracts.messages import SensorFrame, CellState
from metagross.autonomy.perception.pipeline import Perception
from metagross.autonomy.perception import synthetic as syn
calib = defaults.stereo_calibration()
P = Perception(calib)
g = P.geom
names = {int(s): s.name for s in CellState}
for name in ["flat", "trench", "crest", "rock_occlusion", "box", "tilted", "rolling"]:
    sc = syn.make_scene(name)
    for mode in ["tier0", "stereo"]:
        if mode == "tier0":
            d = syn.render_disparity(g, sc, noise_px=0.1, seed=5)
            f = SensorFrame(0.0, 0, None, None, 0, 0, 0, sensor_mode="tier0_disparity", disparity=d)
        else:
            L, Rg, _ = syn.render_stereo_pair(g, sc, seed=5)
            f = SensorFrame(0.0, 0, L, Rg, 0, 0, 0)
        ts = []
        for i in range(3):
            out = P.process(f, (0, 0, 0)); ts.append(out["timings_ms"]["total"])
        st = out["cell_state_local"]
        X, Y = P.spec.centres()
        corridor = (np.abs(Y) < 1.0)
        cnt = {names[k]: int(((st == k) & corridor).sum()) for k in range(9) if ((st == k) & corridor).any()}
        # range of ditch/crest/positive cells in corridor
        def rng(k):
            m = (st == k) & corridor
            return (round(float(X[m].min()),2), round(float(X[m].max()),2)) if m.any() else None
        print(f"{name:14s} {mode:6s} total={np.median(ts):5.1f}ms r_vis={out['r_vis_m']:.1f} ditch={rng(4)} crest={rng(5)} pos={rng(2)} dep={rng(3)} occ={rng(6)}")
        print("      ", {k: round(v,1) for k, v in out["timings_ms"].items()}, cnt)
