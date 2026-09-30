import numpy as np, math
from metagross.config import defaults
from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.autonomy.perception import synthetic as syn, ground as gr
g = CameraGeometry(defaults.stereo_calibration())
sc = syn.make_scene("rolling")
d = syn.render_disparity(g, sc, noise_px=0.1, seed=1)
pts = g.points_from_disparity(d, 2, 4, row_start=100)
rng = np.random.default_rng(0)
prof = gr.fit_v_disparity(d, g, rng)
print("profile", prof.coeffs, prof.v_min, prof.v_max)
sa, sb, sc0 = gr.seed_plane_from_profile(prof, g); print("seed", sa, sb, sc0)
de = prof.expected(pts.v)
cand = np.abs(pts.d - de) < np.maximum(1.5, 0.3*de)
for x0 in range(0, 12, 2):
    m = (pts.x >= x0) & (pts.x < x0+2)
    print(x0, "pts", m.sum(), "cand", (m&cand).sum())
gm = gr.fit_ground(pts, d, g)
for b in gm.bands: print(b)
# ground truth planes per band
for x0 in range(0,12,2):
    m = (pts.x >= x0) & (pts.x < x0+2)
    if m.sum()>3:
        A = np.stack([pts.x[m], pts.y[m], np.ones(m.sum())],1); s,*_ = np.linalg.lstsq(A, pts.z[m], rcond=None); print("LS", x0, np.round(s,3), "rms", np.sqrt(np.mean((A@s-pts.z[m])**2)))
print("----")
b6 = gm.bands[3]
m = (pts.x >= 8) & (pts.x < 10)
x,y,z = pts.x[m].astype(float), pts.y[m].astype(float), pts.z[m].astype(float)
mm = np.abs(z - b6.z(x,y)) < 0.4
print("cand", mm.sum())
fit = gr._ransac_plane(x[mm], y[mm], z[mm], 0.04+0.006*x[mm], np.random.default_rng(0))
print(fit[:3], fit[3].sum())
cp = gr.BandPlane(8, 10, *fit[:3], fitted=True)
ys = np.array([-1.5,0,1.5]); print("dz edge", cp.z(8, ys) - b6.z(8, ys), "ang", math.degrees(math.acos(b6.normal()@cp.normal())))
