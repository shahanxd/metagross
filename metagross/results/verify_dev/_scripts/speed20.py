"""Red-team: GT path length / 20 s for the first 20 s of the final DEV tier0 runs (offline eval)."""
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Downloads\sih again\metagross\results\runs_dev_tier0")
T_CUT_S = 20.0
for cfg, seeds in (("FULL", range(100, 106)), ("TYPICAL", (103, 104, 109, 110))):
    for s in seeds:
        st = np.load(ROOT / cfg / str(s) / "gt" / "states.npz")
        m = st["t"] <= T_CUT_S
        L = float(np.sum(np.hypot(np.diff(st["x"][m]), np.diff(st["y"][m]))))
        print(f"{cfg} {s}: path {L:.2f} m in {st['t'][m][-1]:.1f} s -> {L / T_CUT_S:.3f} m/s")
