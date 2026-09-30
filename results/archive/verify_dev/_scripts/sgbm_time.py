import time, numpy as np, cv2
cv2.setNumThreads(2)
rng = np.random.default_rng(0)
tex = cv2.GaussianBlur((rng.random((400, 720))*255).astype(np.uint8), (0,0), 1.2)
L = tex[:, 40:680].copy(); R = tex[:, 48:688].copy()
for mode_name, mode in [("3WAY", cv2.STEREO_SGBM_MODE_SGBM_3WAY), ("SGBM", cv2.STEREO_SGBM_MODE_SGBM)]:
    m = cv2.StereoSGBM_create(0, 64, 5, P1=200, P2=800, disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, mode=mode)
    ts=[]
    for i in range(6):
        t=time.perf_counter(); d=m.compute(L,R); ts.append((time.perf_counter()-t)*1e3)
    print(mode_name, np.round(ts,1), np.median(d[:,100:]/16.))
    ts=[]
    for i in range(6):
        t=time.perf_counter(); d=m.compute(L[100:],R[100:]); ts.append((time.perf_counter()-t)*1e3)
    print(mode_name, "crop", np.round(ts,1))
