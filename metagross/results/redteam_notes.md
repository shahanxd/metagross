# Red-team notes (honesty and correctness), 2026-09-30, 06:05-06:30

Verifier only: I edited no code. The one generated artefact I refreshed is `results/claims.csv` +
`claims_index.json` (via `python -m metagross.eval.claims`, as instructed). Scratch scripts live outside the repo.
DEV seeds only. The one closed-loop episode I ran was DEV 102, capped at 2 s, written to scratch.

Full suite at 06:14-06:17: `421 passed, 2 warnings in 218.46 s` (OMP_NUM_THREADS=2, PowerShell).

## (a) GT firewall

| # | Finding | Where | How verified |
|---|---|---|---|
| F1 | Autonomy receives `mission_id = "<family>-<seed>"`, for example `F2_ditch_field-103`, so the scenario family name and seed cross into the onboard process. Today it is only logged (node.py:256), never branched on. | `metagross/sim/world.py:127`, sent at `sim/runner.py:178` | read code; grep of `metagross/autonomy` for `mission_id` / family names |
| F2 | spawn re-imports the parent's `__main__` (`metagross.sim.batch`) into the autonomy process before `autonomy_main` runs. Resident in the autonomy process: `metagross.sim`, `.scenario`, `.terrain`, `.hazards`, `.gridplan`, `.objects`, `.geometry`. The file guard still blocks reading scenario / GT files. | `sim/runner.py:110-113` | probe autonomy target (scratch `probe_target.py`) through `python -m metagross.sim.batch --seeds 102 --max-sim-s 2 --autonomy-target probe_target:probe_main`; it dumped `sys.modules` |
| F3 | The AST import scan misses relative imports (`from ..sim import x` gives `module == "sim"`), `importlib.import_module("metagross.sim...")` strings and `__import__`. There are 0 relative imports today, and `node.py:127` uses importlib with config-provided names. | `tests/test_gt_separation.py:15-32` | read test; grep |
| F4 | The file guard hooks only the `open` event. `os.listdir` / `os.scandir` / `subprocess` / `socket` are not blocked. | `autonomy/process.py:77-92` | read code |
| F5 | The launch apron (2 m) is certified without observation. The sim guarantees `START_CLEAR_M = 3.0` around A. This is a documented operator assumption, but it is a deliberate "unknown is free" exception. | `planning/rolling_map.py:248`, `sim/scenario.py:58` | grep |

No `metagross.sim` / `metagross.eval` imports and no scenario / GT paths were found in `metagross/autonomy`. The
DEV-tuned constants (negobs `K_STEP_SIGMA`, `K_CHORD_SIGMA`) are generic noise multipliers, not scenario geometry.

## (b) EVAL seeds

All 141 `result.json` under results/, runs/ and video/ have `split == "dev"` and seed >= 100. Every `summary.csv` has 0
rows with seed < 100. The `results/raw/*` seed folders are all 100-124. Nothing in scripts/ passes `--split eval`.
**Clean.**

## (c) Claims ledger

Regenerated: **148 claims (85 Tested, 59 Simulated, 4 Estimated)**; no duplicate ids across files. I checked 10
random claims (seed 20260930) against their sources, and all 10 values match:
- seg_cpu: val hazard_to_stable delta 0.0053; test mIoU delta -0.0314; val obstacle_to_stable 0.0066; test water_to_stable 0.0.
- seg_zeroshot val mIoU 0.5426.
- perception_dev: stereo before 18; tier0 before 195; ditch 115 before 7.35.
- closed_loop: stereo p50 232.3 / p95 420.0; tier0 drive speed 1.2065.

Issues:
- `seg_cpu_*`: 47 **Tested** rows come from a **PARTIAL iteration-250 checkpoint**. Their notes say "iter 250/4000", but the
  training report says the plan is now 2750. They must be regenerated after the auto-finish, and must not be quoted before then.
- `integrity_onboard_auroc_kitti_holdout = 1.000` (Tested): real images with synthetic degradations. The KITTI 07 holdout has
  20 failure frames, all hard, 0 silent. Label it "Tested (semi-synthetic)" or quote it next to the 07->05 number (0.957 / silent 0.849).
- `claims.py:32` LABELS adds `Implemented`, which is not one of the 5 CLAUDE.md labels. No row uses it today.
- Some sources are not machine-resolvable (`sequences[seed=115]`; seg_* rows have no JSON path).
- The video expects ids that do not exist (`closed_loop_success_full`, `closed_loop_n_runs_full`, ...). The ledger has
  `closed_loop_tier0_FULL_success` etc.
- docs/RESULTS.md was rendered from 42 claims and is now stale; re-run `python -m metagross.eval.results_md`.

## (d) Docs vs code / results (factual errors)

- README.md:43: "Closed-loop ... results are not final and are not quoted here". Results now exist
  (`results/closed_loop_dev.json`, 15 ledger rows): tier0 FULL 3/30, TYPICAL 4/30, stereo FULL 0/3.
- README.md:45: "no trained model is in models/ yet". `models/lraspp_offroad5_cpu.onnx` exists: a PARTIAL iteration-250
  checkpoint trained on RUGD-5L, not OFFROAD5, despite its name.
- README.md:38, ARCHITECTURE.md:215, QA.md:72-74: these describe the integrity model "trained on KITTI 07, tested on 05,
  AUROC 0.957". The **onboard** model `models/integrity.json` (health.py:50) is trained on **05** and tested on 07
  (`results/integrity.json`: AUROC 1.000, 0 silent failures). The 07->05 file is the reverse cross-check.
- QA.md:168-171 and ARCHITECTURE.md:218: the packets "designed to fit" the 600 B / packet budget. **Real DEV telemetry
  is 1.8-2.5x over budget:**
  - tier0 FULL: median 1094 B, p95 1329 B, 95.9 % of 2255 packets > 600 B;
  - tier0 TYPICAL: median 875 B;
  - stereo FULL: median 1522 B, 99.2 % > 600 B.

  The unit test uses synthetic packets. Measured with scratch `packets.py` over `autonomy/telemetry.jsonl`.
- QA.md:80-84: "position error at goal ... 1-2 % drift -> 0.5-1 m, inside 2 m". In tier0 closed loop, wheel odometry
  over-counts distance by **3.2-8.8 % (median 6.3 %, n=30)**. 14/30 FULL runs stop 2.25-4.44 m from the true goal.
  Measured as telemetry-pose path length / GT path length at the same 2 Hz timestamps.
- QA.md:135, ARCHITECTURE.md:204: "3 s without 0.3 m of progress -> STOP_AND_LOOK". Heading progress now restarts the clock
  for up to 9 s (`safety/supervisor.py:63-64, 251`).
- ARCHITECTURE.md:172: GROUND is "certified while fresh or health nominal". It now also requires perception
  `certified_local` in at least one frame (`costmap.py:151-152`).
- ARCHITECTURE.md:182-186: missing the 2-frame lethal support ('pending' cells, costed as suspicious), vote aggregation
  and the void layer. The global cost is now `(1+10c)(1+0.5 f_unseen+1.0 f_susp+5 f_void)` (global_planner.py:137-146).
- ARCHITECTURE.md:216: "typical-stack ablation enters the ditch" is true only in the toy test with synthetic perception. In
  the simulator, TYPICAL entered **0 ditches in 30 DEV runs** (same as FULL).
- QA.md:44-48 / 108-112 / 213: the water, F4 and "closed-loop not done" answers are now measured:
  - FULL enters water in 5/5 F6 tier0 runs;
  - FULL and TYPICAL each have 1 F4 collision.
- QA.md:152-153: perception 123 ms and localiser 31.5 Hz predate today's perception and KLT changes. Closed-loop tier0
  compute is now p50 151 / p95 238 ms, and stereo 232 / 420 ms (> 200 ms budget), on a loaded machine.
- BUILD_LOG.md:6: "closed-loop A->B NOT yet achieved on any DEV seed". It is now 3/30 (the lead appends).

## (e) Five upgrade claims, reproduced

1. **Tier0 FULL 3/30 success (106, 108, 126); TYPICAL 4/30 (100, 106, 108, 126).** CONFIRMED from `runs_dev_tier0/summary.csv`.
   FULL failures: 21 stuck, 5 water, 1 collision.
2. **Odometry over-count is the dominant FULL failure.** CONFIRMED:
   - est/GT distance ratio 1.032-1.088 (median 1.063) on all 30 FULL runs;
   - the 3 successes have ratios of 1.032-1.053;
   - the 14 "arrived in estimate but failed" runs have final errors of 2.25-4.44 m.

   Their referee label is "stuck", because they sit still for 20 s.
3. **Crawl fix: GT speed over the first 20 s.** CONFIRMED from `gt/states.npz`:
   - FULL 100-105: 0.98-1.40 m/s (seed 101 enters water at 14.8 s);
   - TYPICAL 103/104/109/110: 0.93-1.28 m/s.

   Flat-world test re-run: FULL 1.619 m/s (report: 1.616, a small non-determinism), TYPICAL 1.441 m/s.
4. **Perception tier0 F1 off-hazard DITCH_CANDIDATE 0.13 %.** REPRODUCED EXACTLY by re-running
   `perception_dev --section redteam --modes tier0` into scratch:
   - cand 0.0013, lethal 0.01158;
   - certified hazard in envelope 25/2846;
   - r_stable 6.65 / 6.85 / 5.95 / 2.75 m.

   Caveat: the thresholds were DEV-tuned on these same frames, so this is an in-sample result.
5. **Integrity: 0.35 DEGRADED / 5 min, 0 false alarms.** CONFIRMED from `integrity_sim.json`. But this is 1 onset in
   14.35 min, and the Poisson 95 % upper bound is 4.74 / 2.87 = **1.65 per 5 min**, so the "< 1 per 5 min" target is not
   statistically demonstrated. VO failed in only 3 real events.

## (f) Top 10 weaknesses a BEL/DRDO judge will attack (ranked), each with a fix under 1 h

1. **A->B is barely demonstrated.** tier0 3/30, and stereo, the real sensor, 0/3 (stalls within about 2 m on false
   POSITIVE cells).
   Fix:
   - Add noise-aware POSITIVE gating: height excess > k·sigma_h(Z), with sigma_h from Z²σd/(fB), the same approach as negobs.
   - Re-run stereo DEV 102-104.
   - Headline "arrived in own estimate 16/30 + GT error", with the cause.
2. **The thesis ablation shows nothing.** TYPICAL 4/30 vs FULL 3/30, 0 vs 0 ditch entries (DEPRESSION catches visible
   ditch interiors). Fix:
   - Run the F3 crest-ditch DEV seeds with TYPICAL at 2.0 m/s and FULL, and report ditch entries.
   - Show the per-frame certified-hazard metric (195 -> 25 cells in the envelope) as the thesis evidence.
3. **Link budget is false in practice** (packets 1.8-2.5x the 600 B budget). Fix:
   - zlib/RLE the 4-bit costmap, or send it every other packet or at 32x32.
   - Add a test that replays a real DEV `telemetry.jsonl`.
4. **Water: FULL drives into water in 5/5 F6 runs.** The smoke segmenter's water IoU is about 0 (6e-5 val). Fix: a
   return-density rule. Cells where expected coverage is high but valid disparity is below X % become suspicious / never
   certified, using the existing bev.py coverage LUT.
5. **Localisation in closed loop:** 6.3 % median odometry scale error in tier0, and VO never exercised over distance in
   closed loop. Fix:
   - State it plainly.
   - Quote the rendered-DEV VO drift (median 0.62 %, `vo_sim_dev.json`) as the stereo-mode expectation.
   - Add an "arrived_short" failure type so these runs stop showing as "stuck".
6. **Compute:** stereo p50 232 / p95 420 ms and tier0 p50 151 ms vs a 200 ms budget; tier0 doubled vs the 70.6 ms
   integration baseline. Fix: one quiet-machine A/B, MPPI 512 -> 256 samples, and global replans only on route cut.
7. **GT firewall gaps (F1-F4).** Fix:
   - Use an opaque mission_id.
   - Assert in `autonomy_main` that no `metagross.sim*` module is resident, and start the child from a neutral main.
   - Extend the AST test to relative imports and importlib literals.
   - Add `os.listdir` / `os.scandir` / `subprocess.Popen` / `socket.connect` to the audit hook.
8. **In-sample tuning / no EVAL run.**
   - The perception thresholds were tuned and scored on the same DEV frames.
   - 4 KLT configs were compared on KITTI 07, which is a Tested sequence.
   - No EVAL seed has ever run.

   Fix: score perception_dev on untouched DEV seeds (120 / 126 / 121 / 127), and publish the EVAL scenario sha256 list now.
9. **Segmentation:**
   - The deploy model is untrained.
   - The "offroad5_cpu" model is a PARTIAL RUGD checkpoint, worse than smoke on mIoU.
   - The in-loop model is the smoke model.

   Fix: rename or label it, regenerate the claims after the auto-finish, and do not quote the PARTIAL rows.
10. **Moving obstacles and stale certification.**
    - F4 side impacts.
    - While health is NOMINAL, certified ground never expires (`costmap.py:153-154`), so "seen once = free forever" in the
      side blind zone.

    Fix: expire certification outside the current FOV after `FRESH_S` regardless of health, and inflate DYNAMIC cells
    along their observed displacement.
