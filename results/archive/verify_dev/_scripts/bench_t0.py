import numpy as np, time, cv2, sys
cv2.setNumThreads(2)
from metagross.config import defaults as D
from metagross.sim.scenario import make_scenario
from metagross.sim.terrain import Terrain
from metagross.sim.objects import parse_static_objects
from metagross.sim.sensors import Tier0DepthSensor
from metagross.sim.geometry import pose_matrix, GridSpec
g = GridSpec(-20,-20,0.05,800,800); T = Terrain(g, np.zeros(g.shape,np.float32), np.zeros(g.shape,np.uint8))
sen = Tier0DepthSensor(T, [])
Twc = pose_matrix(0,0,0,0,0,0) @ D.camera_extrinsics()
r = sen.render(Twc, noise=False, want_gt=True)
calib = D.stereo_calibration(); H,W = calib.height, calib.width
uu,vv = np.meshgrid(np.arange(W), np.arange(H))
d = np.stack([(uu-calib.cx)/calib.fx, (vv-calib.cy)/calib.fy, np.ones_like(uu,float)],-1) @ Twc[:3,:3].T
t = -Twc[2,3]/d[...,2]; ok = (d[...,2]<0)&(t<=12)
dan = np.where(ok, calib.fx*calib.baseline_m/np.where(ok,t,1), 0)
m = ok & (r.disparity>0); err = r.disparity[m]-dan[m]
print('flat: valid frac %.3f err mean %.3f p95 %.3f max %.3f' % (m.sum()/ok.sum(), err.mean(), np.percentile(np.abs(err),95), np.abs(err).max()))
s = make_scenario(int(sys.argv[1]) if len(sys.argv) > 1 else 101); T = Terrain.from_scenario(s); prims, feet = parse_static_objects(s, T)
sen = Tier0DepthSensor(T, prims)
x,y = s['start']['xy']; yaw = s['start']['yaw']
Twc = pose_matrix(x,y,float(T.height_at(x,y)),0,0,yaw) @ D.camera_extrinsics()
rng = np.random.default_rng(0)
for i in range(3): r = sen.render(Twc, rng=rng)
ts=[]; parts = {}
for i in range(20):
    t0=time.perf_counter(); r = sen.render(Twc, rng=rng); ts.append((time.perf_counter()-t0)*1e3)
    for k,v in r.timings_ms.items(): parts.setdefault(k, []).append(v)
print('scenario frame ms mean %.1f median %.1f min %.1f' % (np.mean(ts), np.median(ts), np.min(ts)), {k: round(float(np.median(v)),1) for k,v in parts.items()}, 'valid %.2f' % (r.disparity>0).mean())
