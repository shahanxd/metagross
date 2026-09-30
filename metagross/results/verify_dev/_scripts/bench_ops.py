import sys, time
sys.path.insert(0, r"D:\Downloads\sih again\metagross")
import numpy as np
from metagross.autonomy.planning.mppi import rollout
K,T=512,30
rng=np.random.default_rng(0)
interp=np.random.rand(T,7)
def t(label, f, n=50):
    ts=[]
    for _ in range(n):
        a=time.perf_counter(); f(); ts.append((time.perf_counter()-a)*1e3)
    print(f"{label:30s} {np.median(ts):7.3f} ms")
kn = rng.standard_normal((K,7,2))
t("randn", lambda: rng.standard_normal((K,7,2)))
t("matmul", lambda: np.matmul(interp, kn))
V=rng.random((K,T)); W=rng.random((K,T))
t("rollout", lambda: rollout((0,0,0),V,W,0.1))
X,Y,YAW=rollout((0,0,0),V,W,0.1)
c,s=np.cos(YAW),np.sin(YAW)
off=np.array([-0.27,0,0.27])
t("circles", lambda: (X[...,None]+off*c[...,None], Y[...,None]+off*s[...,None]))
CX=X[...,None]+off*c[...,None]
lay=np.random.rand(400*400).astype(np.float32)
def fi(x,y):
    ix=np.clip(((x+40)*5).astype(np.int32),0,399); iy=np.clip(((y+40)*5).astype(np.int32),0,399); return iy*400+ix
t("flat_index 46k", lambda: fi(CX,CX))
idx=fi(CX,CX)
t("take 46k", lambda: np.take(lay, idx))
t("count_nonzero axis", lambda: np.count_nonzero(np.take(lay, idx)>=1.0, axis=(1,2)))
t("sum axis12", lambda: np.take(lay, idx).sum(axis=(1,2)))
fr=np.array([.25,.5,.75,1.])
t("probes", lambda: X[...,None] + (0.4 + V[...,None]*fr)*c[...,None])
t("exp", lambda: np.exp(-rng.random(K)))
t("choice", lambda: rng.choice(K, size=8, replace=False))
t("argsort", lambda: np.argsort(rng.random(K)))
