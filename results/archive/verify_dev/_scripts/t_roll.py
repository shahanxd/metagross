import numpy as np
from metagross.config import defaults
from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.autonomy.perception import synthetic as syn
from metagross.autonomy.perception.ground import fit_ground
from metagross.autonomy.perception.negobs import MissingGroundDetector
g = CameraGeometry(defaults.stereo_calibration()); det = MissingGroundDetector(g)
sc = syn.make_scene("rolling")
d = syn.render_disparity(g, sc, noise_px=0.0, seed=3)
pts = g.points_from_disparity(d, 2, 4, row_start=100)
gm = fit_ground(pts, d, g, seed=0)
r = det.detect(d, gm)
c = 80
DG = det._expected_ground(gm, r.rows.size)
for i in range(175, 262):
    v = r.rows[i]; dd = d[v, c*4:c*4+4].mean()
    zc = g.fxb/dd; X = g.t_bc + zc*g.rays_body[v, c*4+2]
    print(v, round(dd,3), round(DG[i,c],3), r.labels[i,c], np.round(X,3), "true z", round(float(sc(X[0], X[1])),3))
