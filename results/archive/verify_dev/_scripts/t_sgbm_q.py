import numpy as np, cv2
cv2.setNumThreads(2)
from metagross.config import defaults
from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.autonomy.perception import synthetic as syn
from metagross.autonomy.perception.stereo import StereoMatcher
g = CameraGeometry(defaults.stereo_calibration())
L, R, D = syn.render_stereo_pair(g, syn.make_scene("flat"), seed=5)
cv2.imwrite("C:/Users/Asus/AppData/Local/Temp/claude/D--Downloads-sih-again/4eb046a2-5c84-43b5-91e3-06a29db230da/scratchpad/L2.png", np.hstack([L[...,0], R]))
d = StereoMatcher().compute(L, R)
for r0, r1 in [(140, 200), (200, 280), (280, 340), (340, 400)]:
    m = (D[r0:r1, 70:] > 0); e = np.abs(d[r0:r1, 70:] - D[r0:r1, 70:])[m]; ok = d[r0:r1,70:][m] > 0
    print(r0, r1, "valid", round(ok.mean(),3), "med err", round(np.median(e[ok]),3), "p95", round(np.percentile(e[ok],95),3), ">1px", round((e[ok]>1).mean(),4))
