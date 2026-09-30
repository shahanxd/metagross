import time, numpy as np, cv2
cv2.setNumThreads(2)
from metagross.config import defaults
from metagross.autonomy.perception.geometry import CameraGeometry
from metagross.autonomy.perception import synthetic as syn
g = CameraGeometry(defaults.stereo_calibration())
t=time.perf_counter(); d = syn.render_disparity(g, syn.make_scene("flat")); print("render ms", (time.perf_counter()-t)*1e3)
zc = g.flat_ground_depth(); da = np.where(np.isfinite(zc), g.fxb/zc, -1)
m = (d>0)&(da>0)
print("max err", np.abs(d[m]-da[m]).max(), "valid rows", np.nonzero((d>0).any(1))[0][[0,-1]])
print("d bottom", d[-1, 320], "d at v=141", d[141,320], "horizon est", np.nonzero((da>0).any(1))[0][0])
t=time.perf_counter(); d = syn.render_disparity(g, syn.make_scene("trench")); print("render trench ms", (time.perf_counter()-t)*1e3)
print(np.round(d[150:260, 320][::-1],2))
t=time.perf_counter(); L,R,D = syn.render_stereo_pair(g, syn.make_scene("trench")); print("pair ms", (time.perf_counter()-t)*1e3)
cv2.imwrite("C:/Users/Asus/AppData/Local/Temp/claude/D--Downloads-sih-again/4eb046a2-5c84-43b5-91e3-06a29db230da/scratchpad/L.png", np.hstack([L[...,0], R]))
from metagross.autonomy.perception.stereo import StereoMatcher
m = StereoMatcher(); ds = m.compute(L, R); print("sgbm ms", m.last_ms)
ok = (ds>0)&(D>0)&(D>4.4)
print("sgbm err median/p90", np.median(np.abs(ds[ok]-D[ok])), np.percentile(np.abs(ds[ok]-D[ok]),90), "valid frac", ok.sum()/((D>4.4).sum()))
print(np.round(ds[150:260, 320][::-1],2))
