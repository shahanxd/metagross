# Pipeline status: built, running, or not

This page records which parts of METAGROSS were run and what each one produced. It is based on an audit on
2026-09-30, where every row was executed in a Linux container (4 vCPU, no GPU). Status labels:

- **RUNS IN LOOP**: executes on every tick of the closed loop, and its output is used.
- **BUILT, NOT IN LOOP**: the code works when called, but the live loop does not use it.
- **MISSING**: no working artefact exists.

## What the system is
It is a software module. The inputs are a stereo camera, wheel encoders and a z-gyro, with no GNSS. The output is
left and right wheel speeds at 5 Hz, plus 2 Hz telemetry to an operator.

## Two sensor modes in the simulator
| Mode | What the stack receives | Where it runs |
|---|---|---|
| **tier-0** | A synthetic disparity map computed from true geometry with a noise, dropout and lighting model. No images. | Any CPU. This mode produced all DEV and EVAL closed-loop numbers. |
| **stereo** | Real rendered left and right images from a Three.js renderer. SGBM and VO run on them. | A GPU with a browser. It also runs on Linux with software WebGL (`MG_ANGLE=swiftshader MG_RENDER_HEADLESS=1 MG_ALLOW_SOFTWARE_GL=1`), at about 3 s per frame. |

## Component status

| # | Component | Status | Evidence |
|---|---|---|---|
| 1 | Stereo depth (SGBM) | RUNS IN LOOP (stereo mode) | Rendered DEV 102: 69-73 % valid disparity, 24-50 ms |
| 2 | Ground model, BEV map, positive obstacles, certification | RUNS IN LOOP (both modes) | Stage timings in `autonomy/timings.csv`; unit tests |
| 3 | Missing-ground (ditch) detector | RUNS IN LOOP (both modes) | DITCH_CANDIDATE cells in both modes; 0 ditch entries in 30 DEV runs |
| 4 | **Terrain segmenter (ML)** | **TRAINED, NOT IN LOOP**: deploy weights exist since 2026-10-01 but are not wired into the closed loop yet | SageMaker job `metagross-seg-20261001-055202` (`ml.g4dn.12xlarge`, 4x T4, fp16), 40/40 epochs, 1.05 billable h. OFFROAD5 test (n=2405) mIoU 0.825 (clean aug) / 0.800 (robust aug), false-safe 3.8 % / 3.9 %; RUGD-5L test (n=733) 0.768 / 0.761 (`results/seg_lraspp_offroad5_*.json`). Water is the weak class (RUGD-5L val water IoU 0.02 / 0.00). The ONNX files are git-ignored: they sit in `models/` of the machine that fetched them and on S3 under the job's `checkpoints/out/`. CPU latency on the laptop is not measured yet. In tier-0 there are no images, so it cannot run there. |
| 5 | Semantic fusion (WATER cells) | BUILT, NOT IN LOOP | Needs item 4 and images. WATER was never produced in any closed-loop run. |
| 6 | Stereo visual odometry | RUNS IN LOOP (stereo mode only) | Rendered DEV 102: 0.34 % drift over 13.5 m. KITTI 00/05/07: 1.53-2.11 % (`results/kitti_*.json`, measured on another machine; KITTI is not reachable from this container) |
| 7 | Depth odometry (tier-0) | RUNS IN LOOP (tier-0) | DEV drives: along-track error 6.3 % → 1.5 % (median). No unit tests yet. |
| 8 | **Integrity monitor (ML, logistic)** | RUNS IN LOOP in stereo mode only | `models/integrity.json` loads (trained on KITTI 05, tested on 07). In tier-0 it is not called and health is fixed at q = 1. |
| 9 | EKF (wheels + gyro + VO / depth odometry), gyro-bias filter, slip | RUNS IN LOOP | Unit tests; closed-loop runs |
| 10 | Rolling map, costmap, global planner, MPPI, speed governor, dead-end memory | RUNS IN LOOP | DEV 102: map 2.5, costmap 2.8, global 5.4, MPPI 6.0 ms per tick |
| 11 | Safety supervisor and forward gate | RUNS IN LOOP | Stop-and-look, dead-end and safe-stop paths all occur in DEV runs. The operator HOLD, ESTOP and RESUME paths are only unit-tested: the runner sends one GO and nothing else. |
| 12 | Skid-steer mixer → wheel commands | RUNS IN LOOP | Drives the simulated vehicle |
| 13 | Telemetry codec (600 B / 2 Hz budget) | RUNS IN LOOP | All 111 real packets fit (`tests/test_link_budget.py`) |
| 14 | Link emulator (loss, latency) | BUILT, NOT IN LOOP | Used offline only |
| 15 | **Operator console** (`operator_ui/`) | **REPLAY ONLY** | Plays back `telemetry.jsonl`. It has no live connection, and its buttons are not wired to the autonomy. |
| 16 | Simulator: worlds, vehicle, referee, 90 pre-registered scenarios | RUNS | `results/scenario_manifest.json` |

## Results and where they come from
| Result | Mode | Label |
|---|---|---|
| EVAL closed loop: FULL 33/60, TYPICAL 31/60 | tier-0 (no images, so no segmenter and no VO) | Simulated |
| DEV closed loop: FULL 20/30 | tier-0 | Simulated |
| KITTI VO drift 1.53-2.11 % | real images | Tested (other machine) |
| Segmenter mIoU, deploy model (`seg_clean_*`, `seg_robust_*`): OFFROAD5 test 0.825 / 0.800, RUGD-5L test 0.768 / 0.761 | real RUGD + RELLIS-3D images, offline | Tested (accuracy only; not yet in the loop, latency not yet measured) |
| Segmenter mIoU (`seg_smoke`, `seg_zeroshot`, `seg_cpu`) | real RUGD images | Tested, but the models are smoke or zero-shot runs, not the deploy model |
| Ditch visibility range 4.3 m, speed envelope | analytic | Estimated |

## Gaps to close for a fully built end-to-end pipeline
1. ~~**Train the terrain segmenter on a GPU**~~ Done 2026-10-01 on SageMaker (row 4). Left: measure its CPU latency on
   the laptop, pick clean vs robust for deployment, and wire it into the loop (enables water and obstacle semantics).
2. **Run the closed loop in stereo mode on the laptop GPU**, so that SGBM, VO, the integrity monitor and the
   segmenter all run on rendered images. The published numbers are tier-0 only.
3. **Connect the operator console live**, with a telemetry stream from the running episode and operator commands
   back to the autonomy.
4. Make the node fail loudly when the segmenter is missing, and record `impl` in `result.json`.

## Next steps (handoff, 2026-10-01)
- **Segmenter trained on SageMaker (2026-10-01).** Account with free credits, ap-southeast-2. The 4-GPU L4/A10G
  sizes (`ml.g6.24xlarge`, `ml.g6.12xlarge`, `ml.g5.12xlarge`) and `ml.g6.16xlarge` sat in "waiting for capacity"
  for 15-35 min each; `ml.g4dn.12xlarge` (4x T4) got an instance within seconds. The first T4 job failed at step 1
  because torch reports emulated bf16 on Turing and cuDNN has no bf16 engine for MobileNetV3's convs; fixed in
  `train_seg.amp_dtype_for_cuda` (bf16 only on sm_80+, else fp16 + GradScaler) and in the setup CUDA sanity check.
  The relaunch (`metagross-seg-20261001-055202`) trained both runs for 40 epochs at ~70 s/epoch (0.24 s/it, batch 32,
  320x416) and exported + evaluated them: 1.05 billable hours in total. Best checkpoints: clean epoch 31, robust
  epoch 32 (1-based) by val mIoU. Results: `results/seg_lraspp_offroad5_{clean,robust}_{offroad5,rugd5}.json`,
  figures `deck_assets/seg_lraspp_offroad5_*`, sidecars `models/lraspp_offroad5_{clean,robust}.json`, training
  metrics `runs/seg/lraspp_offroad5_{clean,robust}/metrics.json`; 24 rows `seg_{clean,robust}_*` in
  `results/claims.csv` (Tested). Note: the test splits score higher than val (OFFROAD5 test 0.825 vs val 0.739,
  clean) because the splits differ in source mix; quote test numbers with their split and n.
- **Next for the segmenter:**
  1. Copy `models/lraspp_offroad5_{clean,robust}.onnx` to the laptop (or re-run
     `python aws/ec2_run.py fetch --backend sagemaker --job seg --name metagross-seg-20261001-055202` there) and
     measure CPU latency (`aws/README.md` section 3); register the latency rows.
  2. Choose the deploy model: clean scores higher on every test split; the onboard `Segmenter` loads
     `lraspp_offroad5_robust.onnx` first. Decide on DEV data, not EVAL seeds.
  3. Wire it into the stereo-mode loop (WATER cells, item 5).
- **Laptop GPU training (fallback, no longer needed for the deploy model)** (PowerShell, repo root). The script uses its own venv `.venv-gpu` with CUDA
  torch and leaves the CPU `.venv` untouched:
  1. `powershell -ExecutionPolicy Bypass -File scripts\train_seg_gpu.ps1 -Setup`: CUDA torch, dependencies, CUDA
     check, unit tests, OFFROAD5 (~8.5 GB) + RUGD-5L (~2.1 GB) download into `data/`.
  2. `... -Bench`: 100-step timing run; prints s/step and the projected hours. Timing only.
  3. `... -Launch`: detached, resumable training of `lraspp_offroad5_robust` (25 epochs x 1000 steps, batch 8,
     320x416, mixed precision). A watcher keeps Windows awake and runs the finish step when training ends.
     `... -Status` shows progress. Re-running `-Launch` after an interruption resumes from `last.pt`.
  4. Finish (automatic, or `... ` with no switch): ONNX export to `models/lraspp_offroad5_robust.onnx` (the first
     file the onboard `Segmenter` looks for), evaluation on OFFROAD5 and RUGD-5L val + test with laptop-CPU latency,
     `results/seg_lraspp_offroad5_robust_{offroad5,rugd5}.json` and figures in `deck_assets/`. Register the
     JSON's `claims` rows in `results/claims.csv` as **Tested**.
  5. Optional ablation: `-Launch -Aug clean`, then finish with `-Aug clean`.
- Then: (2) wire the ONNX model into the loop; (3) run the stereo-mode DEV/EVAL loop on the laptop GPU;
  (4) connect the live console. After that, the video and slides.
