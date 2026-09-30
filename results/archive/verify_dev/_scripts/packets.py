"""Red-team: telemetry packet sizes in DEV closed-loop runs vs the 600 B per-packet budget
(9.6 kbit/s / 8 / 2 Hz). Reads autonomy/telemetry.jsonl 'packet_bytes'."""
import json
from pathlib import Path

import numpy as np

BUDGET_B = 9600 / 8 / 2
for root in ("results/runs_dev_tier0/FULL", "results/runs_dev_tier0/TYPICAL", "results/runs_dev_stereo/FULL"):
    sizes = []
    over_runs = 0
    runs = 0
    for tel in sorted(Path(r"D:\Downloads\sih again\metagross", root).glob("*/autonomy/telemetry.jsonl")):
        b = [json.loads(l)["packet_bytes"] for l in tel.read_text(encoding="utf-8").splitlines() if l.strip()]
        sizes += b
        runs += 1
        over_runs += int(max(b) > BUDGET_B)
    s = np.array(sizes)
    print(f"{root}: runs={runs} packets={s.size} median={np.median(s):.0f} p95={np.percentile(s, 95):.0f} max={s.max()} "
          f"frac>600B={np.mean(s > BUDGET_B):.3f} runs_with_any>600B={over_runs}")
