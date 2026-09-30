"""Scratch: class presence statistics on RUGD-5L train masks (to choose the repeat-factor threshold)."""
import time
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)
root = Path(r"D:\Downloads\sih again\metagross\data\rugd5\train\labelids_res")
t0 = time.perf_counter()
fr = []
names = sorted(root.iterdir())
for p in names:
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    fr.append(np.bincount(m.ravel(), minlength=256)[:5] / m.size)
fr = np.array(fr)
print("read", len(fr), "masks in", round(time.perf_counter() - t0, 1), "s")
for thr in (0.0, 0.001, 0.005, 0.01):
    pres = fr > thr
    print("min_frac", thr, "image freq per class", np.round(pres.mean(0), 4).tolist())
for t in (0.05, 0.1, 0.2):
    for thr in (0.005,):
        f = (fr > thr).mean(0)
        r_c = np.maximum(1.0, np.sqrt(t / np.maximum(f, 1e-9)))
        r_i = np.max(np.where(fr > thr, r_c[None], 1.0), axis=1)
        print("t", t, "r_c", np.round(r_c, 2).tolist(), "mean r_i", round(r_i.mean(), 3), "water img share after", round(((fr[:, 2] > thr) * r_i).sum() / r_i.sum(), 4))
seqs = {}
for p, f in zip(names, fr):
    s = p.stem.rsplit("_", 2)[0]
    seqs.setdefault(s, []).append(f[2])
print({k: (len(v), round(float(np.mean(v)), 4)) for k, v in seqs.items()})
