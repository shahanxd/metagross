# METAGROSS build log

## 2026-09-30 — Build phase II, integration pass (integration owner)

Goal: the real `AutonomyStack` drives A->B on DEV scenarios in closed loop (tier0, then rendered stereo).
**Status: plumbing complete and tested; closed-loop A->B NOT yet achieved on any DEV seed (all 10 tier0
episodes end `stuck`).** Root causes found so far are listed below with what was fixed and what remains.

### What was broken, why, and the fix

1. **Tier-0 health -> SAFE_STOP (verified).** In tier0 mode frames carry no images, the localizer reports
   `vo_available=0, q=0`, and the supervisor latched `SAFE_STOP HEALTH q=0.00` after 3 s (seed 102: SAFE_STOP at
   t=3.0 s). Fix (`autonomy/node.py`): when `health['vo_available']==0` **and** `frame.sensor_mode=='tier0_disparity'`,
   localisation is the wheel+gyro EKF by design, so q := 1 / p_fail := 0 for modes and the governor. Stereo frames
   keep full integrity gating (test: black stereo images -> q < 0.7). The localizer itself is unchanged (still
   honest: VO unavailable).
2. **`use_health=False` only rotates, never drives (seed 102).** Instrumented tick 0: 4840 GROUND cells in view but
   the governor's certified arc was 0.1 m. Two causes, both fixed:
   * *Perception false lethal cells on rolling terrain.* Audit vs the GT hazard raster (diagnostic script, sim side
     only): 1137 of 1216 POSITIVE cells were false (terrain 0.05-0.75 m within the grid; the 2 m band-plane ground
     model cannot follow lateral undulation, raw per-cell `h_max`/`h_min` are outlier detectors with ~100 points per
     near cell, and min-filters on noisy far cells are biased ~0.1 m low). Fix (`perception/positive.py`,
     `pipeline.py`): local terrain level = morphological open-close (1.5 m kernel) of count-weighted, 5x5-denoised
     absolute heights (progressive-morphological-filter idea); step/depression need >= 2 points beyond +-15 cm of that
     level; lethal slope needs >= 2 points in the cell; roughness is the excess over expected stereo height noise.
     Seed 102 tick 0: POSITIVE 1216 -> 141 (41 on GT objects), DEPRESSION 32 -> 1. Unit test: synthetic 0.3 m rolling
     terrain gives 0 lethal cells away from a rock, the 0.4 m rock stays lethal (>80 % of its cells).
   * *Launch apron made static obstacles DYNAMIC.* `seed_apron` stamps "recently observed GROUND", so any lethal cell
     first seen inside the 2 m apron was labelled DYNAMIC (+0.5 m inflation) and wiped certification ahead. Fix
     (`planning/rolling_map.py`): new `assumed` layer; apron ground is certified but is not evidence of free space.
   After both: seed 102 tick 0 R_cert 0.1 m -> 4.1 m, v_cap 0 -> 2.0 m/s.
3. **Remaining blocker (not fixed): missing-ground (ditch) false positives on rolling terrain.** Seed 102 (F1, no
   ditches) tick 0: 170 ditch gaps / 1743 DITCH_CANDIDATE cells, 0 on GT ditches; 118 via "points below chord +
   plateau + jump" (median gap points only 5.6 cm below the lip-reappearance chord, rel. jump 0.12), 52 via
   invalid-disparity gaps. They form stripes every ~2 m (band edges of the ground model). In the global planner a
   suspicious cell costs (1+10*0.5)*1.5 = 9 per cell vs 1.5 for UNSEEN, so the route leaves the seen corridor
   (seed 103 lookahead at 45 deg right, then the vehicle turns away from the goal) and the 3 s no-progress monitor
   fires STOP_AND_LOOK during the turn. Even the TYPICAL config (no negobs, unknown free) only crawls (0.12-0.19 m/s
   mean) and ends `stuck`, so a second, not yet isolated cause sits in MPPI/near-field lethal cells. **This is the
   next thing to fix; the headline FULL-vs-TYPICAL ditch comparison is not yet real (0 ditch entries for both
   because neither drives far enough).**
4. **Runner stereo mode.** `run_episode(..., sensor_mode='stereo')` now builds `default_renderer_factory()`
   (ThreeRenderer at the default calibration) when no factory is given; World calls `load_scenario` and feeds
   `render_state()` (body pose [x,y,z,roll,pitch,yaw], REP-103, plus `T_world_body`, `T_world_cam`) at camera_hz;
   the renderer is closed on any failure. Convention check `tests/test_render_world_consistency.py` (real Chrome):
   a world point projected with `T_world_cam`+K lands within 0.5 px of the renderer's GL projection, and SGBM on
   the rendered pair matches tier0 disparity on ground pixels (median |diff| < 1 px) — both pass.
5. **mission.json + actuator watchdog.** `run_dir/mission.json` = {mission_id, goal_xy_a, success_radius_m,
   timeout_s, goal_sigma_m = 0.5 + |heading_init_err| * range, goal_entry{range_m, bearing_deg}}. World stops the
   wheels after `defaults.WHEEL_CMD_TIMEOUT_S = 0.5` s without a new command; `result.json` has `n_cmd_timeouts`.
   (`tests/test_sim_referee.py::_drive` now re-issues its open-loop command at 5 Hz.)
6. **Video logging.** `DebugBundle.extras['left_rgb']` (320x200 INTER_AREA) every `debug_image_every_n` ticks
   (stereo only), `extras['speed_meas']`, `extras['vo_available']`, `extras['tick']`.
7. **Contracts / config (additions only).** `defaults.BEV_LOCAL_{X,Y}_{MIN,MAX}_M`, `BEV_NX/NY`, `BEV_INDEX_ORDER`,
   `WHEEL_CMD_TIMEOUT_S`; `messages.COSTMAP_U4_CODES` + SensorFrame tier0 disparity units note; RendererProto pose
   convention docstring; DebugBundle BEV/extras note; `DEFAULT_AUTONOMY_CONFIG` keys seg_model_path=None,
   seg_threads=2, launch_apron_m=2.0, seed=0, fixed_latency_s=None, stub_perception=False, stub_localizer=False,
   debug_image_every_n=2.
8. **Semantics in the loop.** Segmenter (default resolves `models/lraspp_smoke.onnx`) wrapped in
   `EveryNthSegmenter(3)`: network at 1/3 camera rate, last mask reused.
9. **Shared disparity.** Perception now runs first (its outputs are egocentric; pose was only echoed) and the
   localizer gets `per['disparity']`, so SGBM runs once per stereo frame. SGBM already crops rows above the
   horizon (perception `row_start`).

### DEV results, tier0 (measured 2026-09-30, `results/runs_integration/tier0/summary.csv`)

Runner + separate autonomy process, 5 Hz, 2 batch workers in parallel on the shared 4-core laptop.
"stuck" = referee: no progress for 20 s.

| config | seed | family | success | failure | time s | path m | SPL | mean speed m/s | ditch | coll | compute ms (mean) | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FULL | 100 | F5_lighting | no | stuck | 20.0 | 0.20 | 0 | 0.010 | 0 | 0 | 73.1 | 16.3 |
| FULL | 101 | F6_water_mud | no | stuck | 20.0 | 0.21 | 0 | 0.010 | 0 | 0 | 70.7 | 15.8 |
| FULL | 102 | F1_trail | no | stuck | 20.0 | 0.43 | 0 | 0.022 | 0 | 0 | 74.4 | 17.6 |
| FULL | 103 | F2_ditch_field | no | stuck | 20.0 | 0.50 | 0 | 0.025 | 0 | 0 | 71.7 | 15.9 |
| FULL | 104 | F3_crest_ditch | no | stuck | 20.0 | 0.44 | 0 | 0.022 | 0 | 0 | 76.8 | 16.2 |
| FULL | 105 | F4_sudden_obstacle | no | stuck | 20.0 | 0.42 | 0 | 0.021 | 0 | 0 | 73.3 | 17.5 |
| TYPICAL | 103 | F2_ditch_field | no | stuck | 25.0 | 2.97 | 0 | 0.119 | 0 | 0 | 62.1 | 18.7 |
| TYPICAL | 104 | F3_crest_ditch | no | stuck | 24.0 | 2.61 | 0 | 0.109 | 0 | 0 | 63.3 | 18.4 |
| TYPICAL | 109 | F2_ditch_field | no | stuck | 26.0 | 5.05 | 0 | 0.194 | 0 | 0 | 61.8 | 20.7 |
| TYPICAL | 110 | F3_crest_ditch | no | stuck | 24.0 | 3.48 | 0 | 0.145 | 0 | 0 | 62.4 | 19.1 |

TYPICAL = unknown_is_free, use_negobs=False, use_governor=False, fixed_speed_mps=1.5.

Per-module ms, tier0 FULL, 606 ticks over seeds 100-105 (from `autonomy/timings.csv`): compute median 70.6
(p95 99.0); perception 49.2 (68.1); localizer 0.2 (no VO in tier0); map 1.9; costmap 4.4; global 2.0 (p95 14.9);
MPPI 9.6 (14.0); others < 1.

### DEV results, rendered stereo (ThreeRenderer, `results/runs_integration/stereo/summary.csv`)

Only short smoke episodes (sim capped at 20 s via `max_sim_s`, 1 worker) because tier0 does not drive yet:

| config | seed | family | success | failure | time s | path m | mean speed m/s | ditch | coll | compute ms (mean) | sensor (render) ms mean | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FULL | 102 | F1_trail | no | timeout (20 s cap) | 20.0 | 0.80 | 0.040 | 0 | 0 | 151.6 | 77.7 | 35.9 |
| FULL | 103 | F2_ditch_field | no | timeout (20 s cap) | 20.0 | 0.04 | 0.002 | 0 | 0 | 153.9 | 79.7 | 36.3 |

Seed 102 stereo, 98 ticks: compute median 142.0 ms (p95 209.8); perception (SGBM + BEV + semantics at 1/3 rate)
86.7 (138.3); localizer (VO on the shared disparity + integrity) 23.1 (41.4); map 1.9; costmap 4.9; global 2.1;
MPPI 11.0. Health in stereo: `vo_available=1`, q = 1.0 on the first bundle; modes NOMINAL 42 / STOP_AND_LOOK 58
ticks. The 5 Hz budget (<= 200 ms) holds at the median, not at p95. Debug bundles with `left_rgb` average 205 KB.
Render time measured world-side (~78 ms incl. transfer) is sim time only and does not count against autonomy.

Note: these DEV runs were made before the final reorder commit in `node.step` (perception stage 1
`compute_disparity` -> VO -> `perception.process(frame, pose)` with the fresh pose); the real `Perception` only
echoes the pose, so its outputs are identical; the change restores the pose semantics for pose-using test oracles.

### Tests
`python -m pytest -q` (PowerShell): 335 passed. New: `tests/test_integration_phase2.py` (10 tests),
`tests/test_render_world_consistency.py` (2, real Chrome); updated `test_sim_runner.py` (mission.json, watchdog
counter) and `test_sim_referee.py::_drive` (5 Hz re-issue under the actuator watchdog).

### Next (ordered)
1. Negobs on rolling terrain: gate the "below chord" ditch rule on a depth that can actually be lethal / demand
   persistence across ranges before k-of-3 confirmation; examine the band-edge stripes (every 2 m).
2. Global planner: suspicious step cost 9 vs unseen 1.5 makes the route leave seen ground; revisit relative costs.
3. Supervisor: do not start the 3 s no-progress clock while the vehicle is still turning onto the route.
4. Then re-run DEV 100-129 tier0 + stereo, and the FULL vs TYPICAL F2/F3 ditch comparison.

## 2026-09-30 — Documentation pass (documentation owner, appended)

New: `README.md`, `docs/ARCHITECTURE.md`, `docs/SIMULATION.md`, `docs/REPRODUCE.md`, `docs/QA.md` (30 questions),
`LICENSES/THIRD_PARTY.md`, `docs/RESULTS.md` (generated by the new `metagross/eval/results_md.py` from
`results/claims.csv`; test `tests/test_results_md.py`), and `deck_assets/slots/s3_kitti.png` (1128 x 652 px, every
text >= 26 px; `python deck_assets/kitti_figures.py --slot-only`; test `tests/test_kitti_slot.py`). No closed-loop
numbers are stated in the docs.

Found while running every documented command:
* `scripts/perception_demo.py` does not put the repo on `sys.path`; it fails with `ModuleNotFoundError: metagross`
  unless `PYTHONPATH=.` is set (documented in REPRODUCE.md; a one-line fix like the other scripts' `sys.path` insert
  would remove the need).
* There is no `requirements.txt` and no project licence file.
* `imageio-ffmpeg` bundles a GPL-3.0 ffmpeg build (`--enable-gpl --enable-version3 --enable-libx264`); it is used only
  by the offline video tools, never onboard.
* The first quickstart episode (04:33, DEV seed 102) ended `autonomy_error` with `NameError: certified_local` while
  `rolling_map.py` was being edited in the shared tree; the re-run at 04:45 ran to the 20 s cap without error.
