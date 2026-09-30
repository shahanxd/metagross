import numpy as np, math
from metagross.config import defaults
from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.autonomy.perception import synthetic as syn, ground as gr
g = CameraGeometry(defaults.stereo_calibration())
sc = syn.make_scene("rolling")
d = syn.render_disparity(g, sc, noise_px=0.1, seed=1)
pts = g.points_from_disparity(d, 2, 4, row_start=100)
gm = gr.fit_ground(pts, d, g)
for b in gm.bands: print(np.round([b.x0, b.a, b.b, b.c],3), b.n_points, b.n_inliers, b.fitted)
b6 = gm.bands[3]
m = (pts.x >= 8) & (pts.x < 10)
x,y,z = pts.x[m].astype(float), pts.y[m].astype(float), pts.z[m].astype(float)
print("pts in 8-10", m.sum(), "z range", z.min(), z.max(), "pred", b6.z(np.array([8,10.]),0))
mm = np.abs(z - b6.z(x,y)) < 0.4
fit = gr._ransac_plane(x[mm], y[mm], z[mm], 0.04+0.006*x[mm], np.random.default_rng(0), (b6.a,b6.b))
print(np.round(fit[:3],3), fit[3].sum())
cp = gr.BandPlane(8, 10, *fit[:3], fitted=True)
ys = np.array([-1.5,0,1.5]); print("dz edge", cp.z(8, ys) - b6.z(8, ys), "ang", math.degrees(math.acos(b6.normal()@cp.normal())))
