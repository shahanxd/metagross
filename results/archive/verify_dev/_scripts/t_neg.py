import time, numpy as np
from metagross.config import defaults
from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.autonomy.perception import synthetic as syn
from metagross.autonomy.perception.ground import fit_ground
from metagross.autonomy.perception.negobs import MissingGroundDetector
g = CameraGeometry(defaults.stereo_calibration()); det = MissingGroundDetector(g)
for name in ["flat", "trench", "crest", "rock_occlusion", "box", "tilted", "rolling"]:
    for noise in [0.0, 0.15]:
        sc = syn.make_scene(name)
        d = syn.render_disparity(g, sc, noise_px=noise, seed=3)
        pts = g.points_from_disparity(d, 2, 4, row_start=100)
        gm = fit_ground(pts, d, g, seed=0)
        t = time.perf_counter()
        r = det.detect(d, gm)
        ms = (time.perf_counter()-t)*1e3
        def summ(sg):
            if len(sg)==0: return "-"
            return f"n={len(sg)} x0 med={np.median(sg.x0):.2f} [{sg.x0.min():.2f},{sg.x0.max():.2f}] x1 med={np.median(sg.x1):.2f}"
        print(f"{name:14s} noise={noise} {ms:5.1f}ms ditch: {summ(r.ditch)} | crest: {summ(r.crest)} | occl: {summ(r.occluded)} | mask px={r.image_mask.sum()}")
