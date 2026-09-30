import numpy as np, cv2
cv2.setNumThreads(2)
from metagross.config import defaults
from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.autonomy.perception import synthetic as syn
from metagross.autonomy.perception.stereo import StereoMatcher
from metagross.autonomy.perception.ground import fit_ground
from metagross.autonomy.perception.negobs import MissingGroundDetector, LAB_MISSING
g = CameraGeometry(defaults.stereo_calibration()); det = MissingGroundDetector(g)
L, R, D = syn.render_stereo_pair(g, syn.make_scene("flat"), seed=5)
d = StereoMatcher().compute(L, R)
pts = g.points_from_disparity(d, 2, 4, row_start=66)
gm = fit_ground(pts, d, g)
r = det.detect(d, gm)
print("ditch cols", r.ditch.col, np.round(r.ditch.x0,2), np.round(r.ditch.x1,2))
c = int(r.ditch.col[0]) if len(r.ditch) else 80
DG = det._expected_ground(gm, r.rows.size)
for i in range(0, 40):
    v = r.rows[i]; band = d[v, c*4:c*4+4]
    print(v, np.round(band,2), round(DG[i,c],2), "true", np.round(D[v, c*4:c*4+4].mean(),2), r.labels[i,c])
gd = r.gap_debug
m = gd["is_ditch"] & gd["keep"]
for key in gd: print(key, np.round(np.asarray(gd[key][m], dtype=float), 3)[:12])
