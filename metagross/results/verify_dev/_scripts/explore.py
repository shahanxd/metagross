import logging
import sys
import time
from pathlib import Path

REPO = Path(r"D:\Downloads\sih again\metagross")
sys.path.insert(0, str(REPO / "scripts"))
import make_hero_visuals as m  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent / "explore"
OUT.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO)

lut = np.zeros((256, 3), np.uint8)
for s, c in m.CELL_COLORS.items():
    lut[int(s)] = c

cands = [(103, 0), (109, 0), (115, 0), (122, 0), (110, 0), (121, 0), (127, 0)]
cap = m.Capture()
try:
    for seed, ci in cands:
        sc = m.load_dev_scenario(seed)
        cr = m.ditch_crossings(sc)[ci]
        poser = m.ScenePoser(sc)
        cap.load(sc)
        for dback in (5.0,):
            st = poser.state_at(cr.s_m - dback, t=20.0)
            t0 = time.perf_counter()
            left, right = cap.stereo(st)
            ch = cap.chase(st, 960, 640)
            out = cap.perceive(left, right, 20.0)
            print(seed, "d", dback, "render+perc %.0f ms" % ((time.perf_counter() - t0) * 1e3), "r_vis", out["r_vis_m"],
                  {int(k): int(v) for k, v in zip(*np.unique(out["cell_state_local"], return_counts=True))})
            bev = lut[out["cell_state_local"]][::-1, ::-1]
            bev = cv2.resize(bev, (bev.shape[1] * 4, bev.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
            h = 640
            L = cv2.resize(left, (1024, 640))
            B = cv2.resize(bev, (int(bev.shape[1] * h / bev.shape[0]), h), interpolation=cv2.INTER_NEAREST)
            sheet = np.concatenate([ch, L, B], axis=1)
            cv2.imwrite(str(OUT / f"s{seed}_d{int(dback)}.png"), sheet[:, :, ::-1])
            m.save_npz(f"explore_{seed}_{int(dback)}", left=left, right=right, chase=ch, **m.per_arrays(out))
finally:
    cap.close()
