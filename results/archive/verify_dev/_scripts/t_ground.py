import time, numpy as np, cv2
cv2.setNumThreads(2)
from metagross.config import defaults
from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.autonomy.perception import synthetic as syn
from metagross.autonomy.perception.ground import fit_ground
g = CameraGeometry(defaults.stereo_calibration())
for name in ["flat", "tilted", "rolling", "trench", "crest", "box"]:
    sc = syn.make_scene(name)
    d = syn.render_disparity(g, sc, noise_px=0.1, seed=1)
    t = time.perf_counter()
    pts = g.points_from_disparity(d, 2, 4, row_start=100)
    gm = fit_ground(pts, d, g, seed=0)
    ms = (time.perf_counter()-t)*1e3
    xs = np.array([2, 4, 6, 8, 10, 11.5]); ys = np.array([-1, 0, 1.0])
    X, Y = np.meshgrid(xs, ys)
    err = gm.height(X, Y) - sc(X, Y)
    vv, uu = np.meshgrid(np.arange(150, 400, 10), np.arange(0, 640, 40), indexing="ij")
    de = gm.expected_disparity(g, vv, uu)
    ref = syn.render_disparity(g, syn.Scene(base=sc.base))[vv, uu]
    m = (ref > 4.4)
    print(f"{name:8s} {ms:5.1f}ms fitted={[b.fitted for b in gm.bands]} hz={gm.vdisp.horizon_row if gm.vdisp else None:.1f} "
          f"max|dz|={np.abs(err).max():.3f} disp err p95={np.percentile(np.abs(de[m]-ref[m]),95):.3f}")
    print("   dz by x:", np.round(err[1], 3))
