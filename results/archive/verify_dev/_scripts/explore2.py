import logging
import sys
import time
from pathlib import Path

import math
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


def bevimg(st, h=400):
    b = lut[st][::-1, ::-1]
    return cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h), interpolation=cv2.INTER_NEAREST)


cap = m.Capture()
try:
    for seed in (103, 127):
        sc = m.load_dev_scenario(seed)
        cr = m.ditch_crossings(sc)[0]
        poser = m.ScenePoser(sc)
        cap.load(sc)
        row = []
        for d in (10, 9, 8, 7, 6, 5, 4, 3):
            st = poser.state_at(cr.s_m - d, t=20.0)
            left, right = cap.stereo(st)
            out = cap.perceive(left, right, 20.0)
            u, c = np.unique(out["cell_state_local"], return_counts=True)
            print(seed, d, {int(a): int(b) for a, b in zip(u, c)})
            m.save_npz(f"seq_{seed}_{d}", left=left, right=right, pose=np.array(st["pose"]), **m.per_arrays(out))
            row.append(bevimg(out["cell_state_local"]))
        cv2.imwrite(str(OUT / f"seq_{seed}.png"), np.concatenate(row, axis=1)[:, :, ::-1])
    # montage candidates
    mont = {100: dict(s=12.0, t=15.5), 101: dict(s=14.0, t=20.0), 102: dict(s=10.0, t=20.0), 103: dict(s=10.8 - 5, t=20.0),
            104: dict(s=26.4 - 4.5, t=20.0), 105: dict(s=None, t=20.0)}
    tiles, chases = [], []
    for seed, p in mont.items():
        sc = m.load_dev_scenario(seed)
        poser = m.ScenePoser(sc)
        cap.load(sc)
        s = p["s"]
        trig = None
        if s is None:  # F4: vehicle 6 m before the dynamic path, box mid-crossing
            dyn = sc["dynamic"][0]
            P = np.asarray(dyn["path"], float)
            along = (P - poser.a) @ poser.u
            s = float(along.mean()) - 6.0
            L = float(np.linalg.norm(P[1] - P[0]))
            trig = p["t"] - 0.55 * L / dyn["speed"]
        st = poser.state_at(s, t=p["t"], dyn_trigger_t=trig)
        left, right = cap.stereo(st)
        ch = cap.chase(st, 1280, 800)
        m.save_npz(f"mont_{seed}", left=left, chase=ch, pose=np.array(st["pose"]))
        tiles.append(left)
        chases.append(cv2.resize(ch, (640, 400), interpolation=cv2.INTER_AREA))
    g = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
    cv2.imwrite(str(OUT / "mont_left.png"), g[:, :, ::-1])
    g = np.concatenate([np.concatenate(chases[:3], 1), np.concatenate(chases[3:], 1)], 0)
    cv2.imwrite(str(OUT / "mont_chase.png"), g[:, :, ::-1])
    # big chase timing test
    sc = m.load_dev_scenario(103)
    poser = m.ScenePoser(sc)
    cap.load(sc)
    cr = m.ditch_crossings(sc)[0]
    st = poser.state_at(cr.s_m - 5.0, t=20.0)
    cap.stereo(st)
    t0 = time.perf_counter()
    big = cap.chase(st, 3200, 2400)
    print("big chase ms", (time.perf_counter() - t0) * 1e3)
    cv2.imwrite(str(OUT / "big_chase_103.png"), cv2.resize(big, (1600, 1200), interpolation=cv2.INTER_AREA)[:, :, ::-1])
finally:
    cap.close()
