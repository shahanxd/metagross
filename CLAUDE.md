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
- `metagross/train/` — segmentation training/export (runs on the laptop GPU via `scripts/train_seg_gpu.ps1`; CPU smoke mode).
- `operator_ui/` — static operator console. `video/` — replay + dashboard compositor. `deck_assets/` — figures/diagrams.
- `tests/` — pytest. `results/` — generated CSV/JSON (never hand-edited). `data/` — downloaded datasets (git-ignored).

## Environment
- Repo root = this folder (`pip install -e ".[dev]"`; tests: `python -m pytest -q`). Work lands on `main`.
- Two machines have been used: the team's Windows 11 laptop (Python 3.10 venv `.venv`, path with a space: use
  `pathlib`, quote paths) and Linux cloud containers (Python 3.11, 4 vCPU, no GPU, Hugging Face / KITTI hosts blocked).
- GPU work runs on the team laptop's NVIDIA GPU (RTX 3050 Laptop, 4 GB): `scripts/train_seg_gpu.ps1`. AWS (`aws/`) is dropped
  and kept for reference only.
- Rendered-stereo mode: Chrome/Edge + GPU on Windows by default; on Linux set `MG_RENDER_HEADLESS=1` and, CPU-only,
  `MG_ANGLE=swiftshader MG_ALLOW_SOFTWARE_GL=1` (seconds per frame, smoke tests only).
- Keep live-loop code CPU-friendly (numpy-vectorised, OpenCV, ONNX Runtime).
- Status of every component: `docs/PIPELINE_STATUS.md`. Superseded evidence: `results/archive/` (never quote it).

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
