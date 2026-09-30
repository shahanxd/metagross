# Reproducing the results and figures

All commands run from the repository root in PowerShell on Windows with the project venv. The path contains a
space, so quote it. Set the thread count first; the numbers in `results/` were measured with 2 threads on a
shared 4-core laptop.

```powershell
cd "D:\Downloads\sih again\metagross"
$env:OMP_NUM_THREADS = 2
$py = ".venv\Scripts\python.exe"
```

Requirement column: **CPU** = laptop only; **Data** = needs a downloaded dataset under `data/`; **Chrome** = opens
a headed Chrome or Edge window through Playwright (run from PowerShell; the Bash tool's sandbox blocks the local
pipe); **GPU** = NVIDIA GPU box (see `aws/README.md`); **Net** = network access.

Timing numbers depend on machine load and will not repeat exactly. Accuracy numbers are deterministic given the
same data, code and seeds.

## 0. Data

| Data | Command | Requirement | Writes |
|---|---|---|---|
| KITTI odometry 00, 05, 07 (grayscale stereo + GT poses) | `& $py scripts\download_kitti.py` | Net (several GB; exact size not recorded) | `data/kitti/sequences/XX/`, `data/kitti/poses/XX.txt` |
| RUGD-5Labels + zero-shot SegFormer ONNX | `& $py scripts\download_seg_data.py --dataset rugd5 --zero-shot-model` | Net, ~2 GB | `data/rugd5/`, `models/segformer_b0_ade.onnx` |
| OFFROAD5 (GPU training set) | `python scripts/download_seg_data.py --dataset offroad5 --root /data/seg` | Net, ~8.5 GB, GPU box | dataset root |
| Scenarios (DEV and EVAL) | `& $py scripts\gen_scenarios.py --split all --workers 2` | CPU | `data/scenarios/{dev,eval}/<seed>.json`, `manifest.json` |

KITTI 00 note: frames 0-1100 of the Hugging Face mirror are a different drive, so 00 is evaluated on frames
1101-4540 only (`--first-frame 1101`; evidence in `results/kitti_summary.json#data_issues`).

## 1. Results files

| Output | Command | Requirement | Label |
|---|---|---|---|
| `results/kitti_vo_07.json`, `results/kitti_vo_05.json`, `results/raw/kitti_vo_0X_poses.txt`, `deck_assets/kitti_0X_traj.png`, `results/kitti_vo.csv` | `& $py -m metagross.eval.kitti_vo --seqs 07 05 --threads 2` | CPU, Data (KITTI) | Tested |
| `results/kitti_vo_00.json` (+ poses, figure) | `& $py -m metagross.eval.kitti_vo --seqs 00 --threads 2 --first-frame 1101` | CPU, Data | Tested |
| `results/kitti_vo_timing.json` | `& $py -m metagross.eval.kitti_vo --seqs 07 --benchmark-vo` | CPU, Data | Tested (timing) |
| `results/localizer_timing.json` | `& $py -m metagross.eval.kitti_vo --seqs 07 --benchmark-localizer --threads 2` | CPU, Data | Tested (timing) |
| `results/kitti_summary.json`, `deck_assets/kitti_traj_panel.*`, `kitti_drift_table.*`, `kitti_drift_card.*`, `deck_assets/slots/s3_kitti.png` | `& $py deck_assets\kitti_figures.py` (only the slot image: `--slot-only`) | CPU, Data (reads GT poses; does not re-run VO) | Tested |
| `results/raw/integrity_05.npz`, `results/raw/integrity_07.npz` | `& $py -m metagross.eval.integrity_train collect --seq 05 --max-frames 1500` then `... collect --seq 07` | CPU, Data | Tested |
| `models/integrity.json`, `results/integrity.json`, `deck_assets/integrity_roc.png`, `integrity_timeline.png` | `& $py -m metagross.eval.integrity_train fit --train 05 --test 07` | CPU | Tested |
| `results/integrity_07to05.json`, `deck_assets/integrity_roc_07to05.png`, `integrity_timeline_07to05.png` | `& $py -m metagross.eval.integrity_train fit --train 07 --test 05 --tag 07to05` | CPU | Tested |
| `results/seg_zeroshot.json`, `deck_assets/seg_zeroshot_*.png` | `& $py -m metagross.eval.seg_eval --model models/segformer_b0_ade.onnx --data-root data/rugd5 --splits val test --out results/seg_zeroshot.json --fig-prefix deck_assets/seg_zeroshot --label "Zero-shot SegFormer-B0 (ADE20K -> 5 classes)"` | CPU, Data | Tested |
| `models/lraspp_smoke.onnx` (+ `.json`) | `& $py -m metagross.train.train_seg --model lraspp --data rugd5 --aug robust --device cpu --threads 3 --workers 1 --img 256x320 --bs 8 --epochs 3 --train-subset 800 --val-subset 240 --lr 1e-3 --warmup-iters 20 --out runs/seg/lraspp_smoke` then `& $py -m metagross.train.export_onnx --ckpt runs/seg/lraspp_smoke/best.pt --out models/lraspp_smoke.onnx --data-root data/rugd5` | CPU, Data (smoke run, not a trained model) | - |
| `results/seg_smoke.json`, `deck_assets/seg_smoke_*.png` | `& $py -m metagross.eval.seg_eval --model models/lraspp_smoke.onnx --data-root data/rugd5 --splits val test --smoke --out results/seg_smoke.json --fig-prefix deck_assets/seg_smoke --label "LR-ASPP SMOKE (300 CPU iters, 800 RUGD imgs)"` | CPU, Data | Tested (SMOKE) |
| `models/lraspp_offroad5_{clean,robust}.onnx`, `results/seg_lraspp_offroad5_*.json` | `bash aws/setup_and_train.sh` then `bash aws/fetch_results.sh ubuntu@<ip> <key.pem> ~/metagross` | GPU, Net | not run yet |
| `docs/MODEL_CARD.md` results section | `& $py -m metagross.train.model_card` | CPU | - |
| `results/theory.json`, `deck_assets/theory/*.png/svg` | `& $py -m metagross.eval.theory` (the slot versions `deck_assets/slots/s4_envelope.png`, `s2_ditch_theory.png` are drawn by `build_diagrams.py`, section 3) | CPU | Estimated |
| `results/perception_ditch_range.csv`, `deck_assets/perception_demo_*.png` | `$env:PYTHONPATH = "."; & $py scripts\perception_demo.py --sweep` (figures: without flags). This script does not add the repo to `sys.path` itself | CPU (KITTI panel only if `data/kitti` exists) | Simulated |
| `results/perception_timing.json` | `$env:PYTHONPATH = "."; & $py scripts\perception_demo.py --bench` | CPU | Tested (timing) |
| `results/sim_benchmark.json` | `& $py -m metagross.sim.bench --seed 102 --frames 60 --episode-target test_sim_runner:dummy_autonomy_main --extra-sys-path tests --episode-sim-s 60 --out results/sim_benchmark.json` | CPU | Simulated (timing) |
| `results/renderer_bench.json`, `deck_assets/render_check.png`, `render_chase.png`, `render_events.png` | `& $py scripts\bench_renderer.py --n 50` | Chrome, GPU-backed WebGL | Simulated |
| `results/claims.csv`, `results/claims_index.json` | `& $py -m metagross.eval.claims` | CPU | ledger |
| `docs/RESULTS.md` | `& $py -m metagross.eval.results_md` | CPU | generated page |

Run the last two after any other command in this table: the ledger is rebuilt from `results/*.json` claims lists and
the extractors in `metagross/eval/claims.py`, and `RESULTS.md` is rendered from the ledger.

## 2. Closed-loop runs (in progress)

Closed-loop numbers are still changing and are not registered in the ledger yet. The commands are:

```powershell
# FULL stack, Tier-0 sensor, all DEV seeds, 2 workers
& $py -m metagross.sim.batch --split dev --workers 2 --out runs\dev_tier0

# FULL vs a "typical stack" ablation: write the configs to a JSON file first
'[{"name": "FULL"}, {"name": "TYPICAL", "unknown_is_free": true, "use_negobs": false, "use_governor": false, "fixed_speed_mps": 1.5}]' |
    Set-Content -Encoding ascii runs\configs_full_typical.json
& $py -m metagross.sim.batch --split dev --workers 2 --configs runs\configs_full_typical.json --out runs\dev_ablation

# Rendered stereo (Chrome, one worker: one browser per worker)
& $py -m metagross.sim.batch --split dev --seeds 102 103 --workers 1 --sensor-mode stereo --max-sim-s 20 --out runs\dev_stereo
```

Each run directory holds `result.json`, `mission.json`, `gt/` and `autonomy/`; each batch writes `summary.csv`.
EVAL seeds (`--split eval`) are for the final, frozen stack only.

## 3. Figures and media without a results file

| Output | Command | Requirement |
|---|---|---|
| `deck_assets/diagrams/*.svg/png`, `deck_assets/slots/s2_how_it_works.png`, `s3_missing_ground.png`, `s3_models.png`, `s4_envelope.png`, `s2_ditch_theory.png` | `& $py deck_assets\diagrams\build_diagrams.py` (`--svg-only` needs no browser; `--slots-only` for slots) | Chrome (PNG step) |
| `deck_assets/hero/*.png` | `& $py scripts\make_hero_visuals.py` (`--compose` re-composes from the cache) | Chrome, DEV seeds only |
| `deck_assets/dashboard_preview.png`, `card_preview.png`, `video/out/test_clip.mp4` | `& $py scripts\make_demo_data.py` | CPU; output is **DEMO_FAKE** synthetic data, never a result |
| `deck_assets/console_preview.png` | `& $py scripts\make_demo_data.py --console` | Chrome |
| Demo video | `& $py -m video.render_timeline video/scenes/golden_demo.yaml video/out/golden_demo.mp4 --values <values.json>` (`--plan-only` prints scene durations without rendering) | CPU; chase views need Chrome; narration needs Net (edge-tts), `--no-tts` otherwise. Unfilled number placeholders abort the render |

## 4. Tests

```powershell
$env:OMP_NUM_THREADS = 2; & $py -m pytest -q
```

Tests that need Chrome: `tests/test_render_browser.py`, `tests/test_render_world_consistency.py`; they skip when no
renderer can start. The segmentation tests build small synthetic datasets in the RUGD layout, so they do not need
`data/rugd5`.

## 5. What was re-run for this document (2026-09-30)

These were executed while writing the docs: `scripts\gen_scenarios.py --split dev --seeds 100 101 102`,
`-m metagross.eval.theory` (to a scratch folder), `-m metagross.sim.batch --split dev --seeds 102 --workers 1
--max-sim-s 20`, `deck_assets\kitti_figures.py --slot-only`, `-m metagross.eval.claims`,
`-m metagross.eval.results_md`, and the `--help` of every CLI named above (which is how the `PYTHONPATH` need of
`scripts\perception_demo.py` was found). The long evaluations (KITTI VO,
integrity training, segmentation evaluation, renderer benchmark) were not re-run; their commands are taken from the
module docstrings, `docs/MODEL_CARD.md` and the provenance fields of the results files.
