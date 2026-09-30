# METAGROSS — engineering rules for every contributor (human or agent)

Project: SIH 2026 PS **SIH26126** (BEL) — vision-based autonomous navigation for a UGV in GPS-denied outdoor terrain.
Thesis: **"Unknown is never free. It drives only on ground it has seen, and only as fast as it can see."**

## Layout (ownership matters — only edit files you own)
- `metagross/contracts/` — message dataclasses + Protocols. **Frozen. Do not edit**; request changes in your report.
- `metagross/config/defaults.py` — single source of truth for camera/vehicle/timing numbers. **Do not duplicate constants.**
- `metagross/sim/` — world (ground truth owner), vehicle, referee, scenarios, tier-0 depth sensor, runner.
- `metagross/sim/render/` — Three.js stereo renderer + Playwright bridge.
- `metagross/autonomy/` — onboard stack. **Must never import `metagross.sim`, `metagross.eval`, or read scenario/GT files.**
- `metagross/eval/` — offline evaluation (KITTI VO, seg metrics, closed-loop metrics, plots, claims ledger).
- `metagross/train/` — segmentation training/export (runs on AWS GPU; CPU smoke mode).
- `operator_ui/` — static operator console. `video/` — replay + dashboard compositor. `deck_assets/` — figures/diagrams.
- `tests/` — pytest. `results/` — generated CSV/JSON (never hand-edited). `data/` — downloaded datasets (git-ignored).

## Environment
- Windows 11, Python 3.10 venv at `.venv` (`.venv\Scripts\python.exe`). Path contains a space: always use `pathlib`, quote paths in shell commands.
- The Bash tool's sandbox blocks local sockets; launch long-running Python/Playwright/servers via the PowerShell tool.
- Do not `pip install` into the venv without it being listed in your task; report missing deps instead.
- 4 cores / 8 threads, no NVIDIA GPU locally. Keep live-loop code CPU-friendly (numpy-vectorised, OpenCV, ONNX Runtime).

## Code quality bar
- Typed, small modules, docstrings stating units and frames. No magic numbers — use `defaults.py` or a module-level constant with a comment.
- Every module gets fast pytest unit tests (synthetic inputs) under `tests/`.
- Log with `logging`, not print. Deterministic given a seed.
- Frames: A-frame (mission) x fwd / y left / z up; body same axes; camera OpenCV (x right, y down, z fwd).

## Honesty rules (non-negotiable)
- Every number destined for the deck/video comes from a file in `results/` and is registered in `results/claims.csv` with a label:
  `Tested` (real data) · `Simulated` · `Estimated` · `Proposed` · `Literature`.
- Never tune on EVAL seeds (0–59). Tune only on DEV seeds (100–129).
- Never fabricate metrics. If something was not measured, say so.
