# METAGROSS

Vision-based autonomous navigation for a small unmanned ground vehicle (UGV) in GPS-denied outdoor terrain.
Submission for Smart India Hackathon 2026, problem statement **SIH26126** (Bharat Electronics Ltd.).

The repository holds three things:

1. **The onboard stack** (`metagross/autonomy/`). It drives from launch point A to a goal B using a stereo camera,
   wheel encoders and a z-gyro. No GNSS, no LiDAR, no radar, no map of the area.
2. **A closed-loop simulator** (`metagross/sim/`). Procedural outdoor worlds with ditches, crests, rocks, water and
   moving obstacles; a synthetic depth sensor for fast batches; a Three.js stereo renderer for camera images; a
   referee that judges every run against ground truth.
3. **An evaluation harness** (`metagross/eval/`). Real-data tests (KITTI stereo VO, RUGD terrain segmentation),
   analytic models, and a claims ledger that every reported number must pass through.

## Thesis

> **Unknown is never free. It drives only on ground it has seen, and only as fast as it can see.**

Most planners treat unobserved cells as free space. For a UGV that is how it drives into a ditch: a ditch is
*missing ground*, and it looks like "no data", not like an obstacle. METAGROSS therefore:

* certifies as drivable only ground that the stereo camera actually observed (`CellState.GROUND`);
* treats never-observed, occluded and crest-shadowed cells as unknown, never as free;
* detects missing ground (ditches, crest drop-offs) explicitly from stereo disparity;
* caps speed so the vehicle can always stop inside certified ground:
  `v^2/(2a) + v*T_r + B <= R_cert` (`metagross/autonomy/planning/governor.py`).

## What is real, simulated, estimated or proposed

Every number on the slides and in the video carries one of these labels and is listed in
[docs/RESULTS.md](docs/RESULTS.md), which is generated from `results/claims.csv`. The documents in `docs/` name the
results file next to every number they quote.

| Part | Status | Evidence |
|---|---|---|
| Stereo visual odometry | **Tested** on real KITTI odometry drives (sequences 00, 05, 07), camera only | `results/kitti_vo_*.json`, `results/kitti_summary.json` |
| VO integrity monitor | **Tested** on real KITTI frames with synthetic image degradations | `results/integrity_07to05.json` |
| Terrain segmentation | **Tested** on RUGD-5L validation/test images: a zero-shot baseline and a CPU "smoke" model (not the trained deploy model) | `results/seg_zeroshot.json`, `results/seg_smoke.json`, `docs/MODEL_CARD.md` |
| Onboard compute timings | **Tested** on the development laptop (Intel i5-1135G7, 4 cores, shared with other jobs) | `results/localizer_timing.json`, `results/perception_timing.json` |
| Missing-ground detection range | **Simulated** (analytic synthetic scenes) | `results/perception_ditch_range.csv` |
| Stereo renderer realism (SGBM on rendered images) | **Simulated** | `results/renderer_bench.json` |
| Closed-loop A to B driving | **Simulated** (tier-0 depth sensor, no images): held-out EVAL seeds 0-59, FULL 33/60 reached B, TYPICAL baseline 31/60; DEV 100-129 FULL 20/30. A first EVAL run on a regressed commit (17/60) is kept and disclosed | `docs/EVAL_PREREGISTRATION.md`, `results/closed_loop_eval.json`, `results/closed_loop_dev.json` |
| Detectability and safe-speed envelope | **Estimated** (closed-form geometry) | `results/theory.json` |
| Trained off-road deploy segmenter (OFFROAD5, GPU) | **Proposed**: training scripts exist (`aws/`); no trained model is in `models/` yet | `aws/README.md` |
| Operator console and narrow-band link | Implemented and unit-tested (codec, link emulator, replay console); not tested over a real radio | `tests/test_plan_link.py`, `operator_ui/` |
| Real vehicle, real camera, embedded computer | **Proposed**; nothing has run on hardware | `docs/QA.md` (field-test plan) |

## Hardware assumptions

The values below are defined in `metagross/config/defaults.py`, which every module reads.

| Item | Assumption |
|---|---|
| Stereo camera | ZED 2i-class rig: 12 cm baseline, rectified 640 x 400 px, 72 deg horizontal FOV (fx = fy ~ 440 px), calibration known a priori |
| Camera mount | 0.9 m above ground, 0.30 m ahead of the body origin, pitched 12 deg down; useful depth range capped at 12 m |
| Vehicle | Scout-Mini-class skid-steer: track 0.50 m, wheel radius 0.13 m, footprint 0.80 x 0.60 m, ground clearance 0.15 m, speed cap 2.0 m/s, yaw rate cap 1.2 rad/s, acceleration 1.0 m/s^2 |
| Proprioception | Wheel encoders (4096 ticks/rev), MEMS z-gyro (bias, bias random walk, white noise) |
| Rates | Physics 50 Hz; camera and control 5 Hz in batch runs (10 Hz demo setting); telemetry 2 Hz |
| Operator link | Design assumption for the emulator: 9.6 kbit/s, 20 % packet loss, 0.4 s latency; no video is sent |
| Compute target | Jetson Orin Nano-class embedded computer (proposed; not measured). All timings so far are from the laptop |

A cheaper 7.5 cm-baseline camera (OAK-D Lite class) is analysed as an option in `results/theory.json` (depth
error grows 1.6x at the same range); it is not the modelled sensor.

## Quickstart (Linux / macOS)

```bash
git clone https://github.com/shahanxd/metagross && cd metagross
pip install -e ".[dev]"
python -m pytest -q                                              # ~420 tests, ~75 s on 4 cores
python scripts/gen_scenarios.py --split all --workers 4          # 90 pre-registered scenarios (hashes: results/scenario_manifest.json)
python -m metagross.sim.batch --split dev --seeds 102 --workers 1 --out runs/demo   # one closed-loop episode, A -> B
```

## Quickstart (Windows, PowerShell)

The repository path contains a space; keep the quotes. Every command below was run on 2026-09-30 from the
repository root with the existing `.venv` (Python 3.10).

```powershell
cd "D:\Downloads\sih again\metagross"
$env:OMP_NUM_THREADS = 2

# 1. Fast tests: GT firewall import scan, analytic formulas, results renderer
.venv\Scripts\python.exe -m pytest -q tests\test_gt_separation.py tests\test_theory_formulas.py tests\test_results_md.py

# 2. DEV scenarios (seeds 100-129 are for development; 0-59 are held-out EVAL seeds)
.venv\Scripts\python.exe scripts\gen_scenarios.py --split dev --seeds 100 101 102 --workers 1

# 3. One closed-loop episode: DEV seed 102, Tier-0 depth sensor, 20 s of simulated time
.venv\Scripts\python.exe -m metagross.sim.batch --split dev --seeds 102 --workers 1 --max-sim-s 20 --out runs\quickstart

# 4. Analytic figures into a scratch folder (does not touch deck_assets/)
.venv\Scripts\python.exe -m metagross.eval.theory --out-dir runs\quickstart\theory --json runs\quickstart\theory\theory.json

# 5. Claims ledger, then the results page
.venv\Scripts\python.exe -m metagross.eval.claims
.venv\Scripts\python.exe -m metagross.eval.results_md
```

Step 3 took about one minute of wall time on the shared laptop. It writes `runs\quickstart\FULL\102\` with
`result.json` (referee verdict; with the 20 s cap a run that has not arrived ends as `timeout`), `mission.json`,
`gt\` (ground truth, simulator side) and `autonomy\` (the stack's own logs: `telemetry.jsonl`, `timings.csv`, debug
bundles). A batch skips seeds whose `result.json` already exists, so delete `runs\quickstart` to re-run it.
`runs/` is git-ignored.

The full test suite takes several minutes and opens a Chrome window for the renderer tests:

```powershell
$env:OMP_NUM_THREADS = 2; .venv\Scripts\python.exe -m pytest -q
```

To view a run in the operator console, open `operator_ui\index.html` in Chrome or Edge and drag the run's
`autonomy\telemetry.jsonl` onto the page (the console only replays; it is not connected to a live run).

Setting up a new machine: create a Python 3.10 venv and install the packages listed with versions in
[LICENSES/THIRD_PARTY.md](LICENSES/THIRD_PARTY.md). Playwright's own browser download is not needed, because the
renderer drives the installed Chrome or Edge. There is no `requirements.txt` yet, and this setup path was not
re-run on a clean machine.

Rules for contributors (human or agent) are in [CLAUDE.md](CLAUDE.md): never tune on EVAL seeds 0-59, never let
`metagross/autonomy` import `metagross.sim` or `metagross.eval`, never hand-edit `results/`.

## Repository map

| Path | Contents |
|---|---|
| `metagross/contracts/` | Frozen message dataclasses (`SensorFrame`, `WheelCmd`, `Telemetry`, `MissionSpec`, ...), Protocols, the world-autonomy pipe protocol, the scenario schema |
| `metagross/config/defaults.py` | Single source of truth for camera, vehicle, timing, mapping, governor and link numbers |
| `metagross/autonomy/` | Onboard stack: `perception/` (SGBM stereo, ground model, missing-ground detector, BEV, segmentation), `localization/` (stereo VO, integrity monitor, EKF, slip), `planning/` (rolling map, costmap, global planner, speed governor, MPPI), `safety/` (supervisor), `control/` (skid-steer mixer), `link/` (telemetry codec, link emulator), `node.py`, `process.py` |
| `metagross/sim/` | Ground-truth owner: scenario generator, terrain, vehicle model, Tier-0 depth sensor, referee, world, runner, batch runner; `render/` holds the Three.js stereo renderer and its Playwright bridge |
| `metagross/eval/` | Offline evaluation: KITTI VO, trajectory metrics, integrity training, segmentation metrics, analytic theory, plot style, claims ledger, `results_md.py` |
| `metagross/train/` | Segmentation training and ONNX export (GPU box; CPU smoke mode) |
| `scripts/` | Data download, scenario generation, renderer benchmark, perception demo, hero visuals, demo data |
| `aws/` | One-command GPU training of the terrain segmenter |
| `operator_ui/` | Static operator console (no build step, no CDN) |
| `video/` | Replay, dashboard compositor, cards, encoder for the demo video |
| `deck_assets/` | Figures and diagrams for the slides; `deck_assets/slots/` holds slot-sized images |
| `models/` | ONNX models (git-ignored) with JSON sidecars; `integrity.json` (VO integrity weights) |
| `results/` | Generated JSON/CSV only; `claims.csv` is the ledger |
| `tests/` | pytest suite (synthetic inputs; a few tests need Chrome or downloaded data) |
| `docs/` | This documentation |
| `data/`, `runs/` | Downloaded datasets, scenarios and run outputs (git-ignored) |

## Documentation

| Document | For |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Processes, message contracts, ground-truth firewall, per-module data flow, where each PS requirement is met |
| [docs/SIMULATION.md](docs/SIMULATION.md) | Renderer, Tier-0 sensor model, scenario families, DEV/EVAL split, referee metrics |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Commands that regenerate every results file and figure |
| [docs/RESULTS.md](docs/RESULTS.md) | Every registered number, grouped by label, with its source (generated) |
| [docs/QA.md](docs/QA.md) | Hard questions and honest answers |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Terrain segmenter: data, recipe, metrics, limits |
| [docs/REFERENCES.md](docs/REFERENCES.md) | Verified bibliography |
| [docs/BUILD_LOG.md](docs/BUILD_LOG.md) | Engineering log (what broke, what was fixed, what is open) |
| [LICENSES/THIRD_PARTY.md](LICENSES/THIRD_PARTY.md) | Dependencies, datasets and their licences |
