# METAGROSS architecture

This document describes the code as it is in the repository. Module paths are given so each statement can be
checked. Numeric parameters come from `metagross/config/defaults.py` unless another file is named.

## 1. Two processes

```
 WORLD PROCESS (simulator, owns ground truth)                 AUTONOMY PROCESS (onboard stack)
 metagross/sim/runner.py                                      metagross/autonomy/process.py
 +--------------------------------------------+   Pipe     +------------------------------------------+
 | Scenario (terrain, objects, hazards)       |  (spawn)   | file guard (audit hook) installed first  |
 | SkidSteerVehicle  50 Hz physics            |            | AutonomyStack.step(frame)                |
 | Tier-0 depth sensor | Three.js renderer    | SensorFrame|   stereo -> localiser -> perception ->   |
 | encoders, gyro                             | ---------> |   map -> costmap -> global planner ->    |
 | Referee (every physics step)               |            |   supervisor -> governor -> MPPI ->      |
 | actuator watchdog 0.5 s                    | <--------- |   gate -> skid-steer mixer               |
 | writes run_dir/gt/, result.json            |  WheelCmd  | writes run_dir/autonomy/ only            |
 +--------------------------------------------+ +Telemetry +------------------------------------------+
            |                                                          | telemetry.jsonl (2 Hz packets)
            v  after the run                                           v
 EVALUATOR (metagross/eval, video/)  reads gt/ and autonomy/      OPERATOR CONSOLE (operator_ui/) replays packets
```

* The **runner** (`metagross/sim/runner.py`) is the world process. It owns the `World`
  (`metagross/sim/world.py`): terrain, vehicle, sensors, referee, and ground-truth (GT) logs.
* The **autonomy** runs in a separate OS process started with the `multiprocessing` *spawn* context. Its entry
  point is `metagross.autonomy.process:autonomy_main(conn, run_dir, config)`. It sees only the messages below.
* The loop is **lock-step**. Every `round(PHYSICS_HZ / camera_hz)` physics steps (10 steps at 5 Hz) the world
  builds a `SensorFrame` at the current state, sends it, and blocks until the command arrives. Simulated time is
  frozen while the autonomy computes. The autonomy reports its measured `compute_ms`; the world applies the command
  `ceil(compute_ms / 20 ms)` physics steps after the frame, so compute latency is paid in simulated time.
* The world stops the wheels if no new command arrived for `WHEEL_CMD_TIMEOUT_S = 0.5 s` (actuator watchdog,
  counted in `result.json` as `n_cmd_timeouts`).
* The same `AutonomyStack` class can be driven in-process by tests and replay tools; the process wrapper adds only
  the file guard and logging.

### 1.1 Pipe protocol (`metagross/contracts/ipc.py`)

| Direction | Message | When |
|---|---|---|
| runner -> autonomy | `("reset", MissionSpec, StereoCalibration, VehicleSpec, config)` | once per episode |
| runner -> autonomy | `("frame", SensorFrame)` | every camera tick |
| runner -> autonomy | `("operator", OperatorCmd)` | on operator action (the runner sends `GO` after `ready`) |
| runner -> autonomy | `("close",)` | end of episode |
| autonomy -> runner | `("ready",)` | after reset |
| autonomy -> runner | `("cmd", WheelCmd, Telemetry or None)` | one per frame, in order |
| autonomy -> runner | `("error", traceback_text)` | fatal error; the referee ends the run as `autonomy_error` |

`config` is merged over `DEFAULT_AUTONOMY_CONFIG` (same file). Its switches build the ablations: `unknown_is_free`,
`use_negobs`, `use_governor`, `use_health`, `use_semantics`, `use_wheel_odom`, `fixed_speed_mps`, `speed_cap_mps`.

### 1.2 Message contracts (`metagross/contracts/messages.py`, version `CONTRACT_VERSION = "1.0"`)

These dataclasses are the only things that cross the process boundary. They are frozen; changes must be additive
and documented.

| Message | Fields (units, frame) |
|---|---|
| `MissionSpec` | `mission_id`; `goal_xy_a` (m, A-frame; the launch-heading error is already applied); `success_radius_m`; `timeout_s` |
| `StereoCalibration` | `width`, `height` (px); `fx`, `fy`, `cx`, `cy` (px); `baseline_m`; `T_body_cam` (4x4, left camera to body) |
| `VehicleSpec` | track width, wheel radius, length, width, ground clearance (m); max speed (m/s), yaw rate (rad/s), acceleration (m/s^2), wheel rate (rad/s); encoder ticks per revolution |
| `SensorFrame` | `t` (s), `seq`; `left_rgb` (H x W x 3 uint8, rectified) and `right_gray` (H x W uint8) in stereo mode; `wheel_angle_l_rad`, `wheel_angle_r_rad` (cumulative encoder angle, rad, quantised); `gyro_z_rps` (rad/s, with bias and noise); `sensor_mode` (`"stereo"` or `"tier0_disparity"`); `disparity` (H x W float32 px, only in Tier-0 mode, `<= 0` invalid) |
| `WheelCmd` | `t`, `seq`; `omega_l_rad_s`, `omega_r_rad_s` (wheel angular rate, rad/s); `mode` (`DriveMode`); `compute_ms` (measured wall-clock) |
| `Telemetry` | `t`, `seq`; `pose_xy_yaw` (m, m, rad, A-frame); `pos_sigma_m`; `mode`; `reason` (short text); `v_cap_mps`; `r_cert_m`; `speed_mps`; up to 5 `waypoints_xy` (A-frame); `costmap_u4` (64 x 64, 4-bit codes, 0.25 m cells, body-aligned); `health` (name -> float) |
| `OperatorCmd` | `t`; `action` (`GO`, `HOLD`, `RESUME`, `ESTOP`); optional new `goal_xy_a` |

`DebugBundle` (`metagross/contracts/interfaces.py`) is the autonomy's per-tick internal state. It is written only to
the autonomy's own log directory for replay and video, and is never sent back to the world.

### 1.3 Frames

| Frame | Axes and origin | Used by |
|---|---|---|
| A-frame (mission) | x forward along the launch heading, y left, z up; origin at launch point A | goal, pose, rolling map, telemetry |
| Body | x forward, y left, z up (REP-103); origin at the wheelbase centre on the ground | BEV grid, MPPI footprint, mixer |
| Camera | OpenCV: x right, y down, z forward (optical axis); left camera | disparity, VO |
| World (simulator only) | x east, y north, z up | scenario, renderer, referee; never seen by the autonomy |
| Egocentric BEV grid | `grid[i, j]`, i forward from x = -2 m, j to the left from y = -6 m, 0.1 m cells, 160 x 120 | perception to planning |
| Rolling map | `[iy, ix]`, A-frame, 0.2 m cells, 80 m square (400 x 400), recentred in whole cells | planning |
| Telemetry costmap | 64 x 64, 0.25 m cells, row 0 = 8 m ahead, column 0 = 8 m left, vehicle at the centre | operator console |

The camera sits 0.9 m above ground, 0.30 m ahead of the body origin, pitched 12 deg down
(`defaults.camera_extrinsics()`).

## 2. Ground-truth firewall (six layers)

The autonomy must be able to run on a real UGV with nothing but the contract messages. Six independent layers keep
simulator ground truth out of it.

| # | Layer | Mechanism | Where it is enforced / tested |
|---|---|---|---|
| 1 | Static import scan | No file under `metagross/autonomy/` may import `metagross.sim`, `metagross.eval` or `metagross.train` | `tests/test_gt_separation.py::test_autonomy_never_imports_ground_truth_modules` (AST scan of every file) |
| 2 | Separate OS process | The stack runs in a spawned process; only the pipe tuples of section 1.1 cross | `metagross/sim/runner.py` (`_AutonomyLink`), `tests/test_node_process.py::test_process_protocol_with_stubs`, `tests/test_sim_runner.py` |
| 3 | Slot whitelist | `SensorFrame` is a `slots` dataclass; its field set is pinned, so adding a field fails a test and forces a review for GT leakage. The goal is handed over in the A-frame with the scenario's heading-init error applied, as an operator would enter it | `tests/test_gt_separation.py::test_sensor_frame_whitelist`, `World.mission_spec()` |
| 4 | Runtime file guard | Before importing the stack, the process installs a `sys.addaudithook` hook. Every `open` outside an allow-list raises `PermissionError`. Allowed: the Python installation and venv, `metagross/autonomy`, `metagross/contracts`, `metagross/config`, `models/`, and `<run_dir>/autonomy/`. `<run_dir>/gt/` and `data/scenarios/` are therefore unreadable | `metagross/autonomy/process.py::install_file_guard`, `tests/test_node_process.py::test_audit_hook_blocks_forbidden_open` |
| 5 | Seed pre-registration | EVAL seeds 0-59 and DEV seeds 100-129 are fixed in the contract. Scenarios are generated deterministically from the seed, hashed (sha256 of canonical JSON) and verified on load. Tuning is allowed on DEV seeds only | `metagross/contracts/scenario.py`, `metagross/sim/scenario.py::load_scenario`, `data/scenarios/manifest.json` |
| 6 | Post-hoc evaluation | The referee judges online on the world side, but no GT reaches the autonomy. Evaluation, plots and video read `gt/` and `autonomy/` logs only after the run has ended | `metagross/sim/runner.py::_finalize`, `metagross/eval/`, `video/replay.py` |

Two further rules support the firewall. The onboard perception self-test scenes
(`metagross/autonomy/perception/synthetic.py`) are built from the camera model alone, not from the simulator. The
Tier-0 frames carry disparity that already contains the sensor noise model (section 3 of `docs/SIMULATION.md`), not
clean GT depth.

## 3. Per-module data flow

One control tick (`metagross/autonomy/node.py::AutonomyStack.step`) runs these stages in order on every camera
frame (5 Hz in batch runs). Times are measured per stage and logged to `autonomy/timings.csv`.

| # | Stage (module) | Input | Output | Rate |
|---|---|---|---|---|
| 1 | Stereo (`perception/stereo.py`) | rectified left RGB + right gray, or Tier-0 disparity | disparity (H x W float32 px; `Z = fx * B / d` m; `<= 0` invalid). CLAHE then OpenCV `StereoSGBM` (3-way, 64 disparities, 5 x 5 blocks); rows above the horizon are cropped | every frame; computed once and shared with VO |
| 2 | Localiser (`localization/localizer.py`, `vo.py`, `ekf.py`, `health.py`, `slip.py`) | frame + shared disparity | A-frame pose (x, y m; yaw rad), `pos_sigma_m`, health features and `p_fail`, smoothed `q = 1 - p_fail`, slip ratio, `chi_hat` | every frame |
| 3 | Perception (`perception/pipeline.py`, `ground.py`, `bev.py`, `positive.py`, `negobs.py`, `semantic_bev.py`) | frame, disparity, pose | egocentric BEV (160 x 120, 0.1 m, body frame): `CellState` per cell, cost in [0, 1] (1 = lethal), height, braking-friction proxy, image-space missing-ground mask, `r_vis_m` | every frame |
| 3a | Segmenter (`perception/semantics.py`) | left RGB | 5-class ids + normalised entropy (H x W); ONNX Runtime CPU, 2 threads | every 3rd frame (`EveryNthSegmenter(3)`); the last mask is reused |
| 4 | Rolling map (`planning/rolling_map.py`) | BEV grids + pose | A-frame map (0.2 m, 80 m square): state, cost, certification age, ditch confirmation, lethal memory | every frame |
| 5 | Costmap (`planning/costmap.py`) | rolling map, health | `PlanningMaps`: inflated cost in [0, 1], certified mask, lethal cores, clearance | every frame |
| 6 | Global planner (`planning/global_planner.py`) | costmap, pose, goal | cost-to-go field grown from the goal (8-connected Dijkstra, 0.4 m grid), descent path, 3 m lookahead point | recomputed at 1 Hz, or at once when a lethal cell cuts the current path |
| 7 | Supervisor (`safety/supervisor.py`) | q, pose, speeds, frame gap, immobilised flag, cost-to-go progress | drive mode, speed factor, cost-inflation scale, overrides | every frame |
| 8 | Speed governor (`planning/governor.py`) | costmap, pose, previous MPPI path, `r_vis`, q, measured latency | `v_cap` (m/s), `R_cert` (m), binding term | every frame |
| 9 | MPPI (`planning/mppi.py`) | pose, current (v, omega), `v_cap`, costmap, cost-to-go | first control (v m/s, omega rad/s), nominal path (A-frame), sampled rollouts | every frame; 512 samples x 30 steps x 0.1 s (3 s horizon) |
| 10 | Gate (`Supervisor.gate`) | (v, omega), pose, costmap | braked (v, omega) if the footprint would enter a lethal core within 1 s | every frame |
| 11 | Mixer (`control/mixer.py`) | (v, omega), `chi_hat` | wheel rates `omega_L`, `omega_R` (rad/s) = `WheelCmd` | every frame |
| 12 | Link (`link/egomap.py`, `link/codec.py`) | map, plan, health | `Telemetry` and its binary packet (CRC-32, zlib-compressed 4-bit costmap) | 2 Hz |

Per-stage timings on the development laptop are in `results/perception_timing.json` (perception at 640 x 400) and
`results/localizer_timing.json` (localiser with shared disparity). Closed-loop per-module timings are in each
run's `autonomy/timings.csv`; the engineering log (`docs/BUILD_LOG.md`) summarises the DEV runs.

### 3.1 Localisation

* **VO** (`vo.py`): frame-to-frame stereo odometry. KLT tracks (forward-backward checked) are back-projected with
  the previous frame's disparity, then PnP-RANSAC (AP3P) and Levenberg-Marquardt refinement give the 6-DoF relative
  camera motion. A motion-sanity gate rejects implausible motions. No loop closure, no bundle adjustment.
* **Integrity monitor** (`health.py`): 12 cheap features per frame (inlier count and ratio, reprojection RMSE,
  feature coverage, track age, pose-Hessian eigenvalue, blur, saturation, darkness, contrast, dark channel, ground
  disparity density) feed a logistic model, `p_fail = sigmoid(w . z + b)`. Weights are in `models/integrity.json`,
  trained on real KITTI frames with synthetic degradations (`metagross/eval/integrity_train.py`). `q` is smoothed
  asymmetrically: it drops fast and recovers slowly.
* **EKF** (`ekf.py`): planar SE(2) state `[x, y, yaw]` in the A-frame, starting at (0, 0, 0). Prediction from wheel
  odometry with an effective-track factor chi (skid-steer) and the bias-corrected gyro. VO relative motions are
  fused by stochastic cloning, with covariance inflated by `1/inliers` and `1/q`. VO is rejected when `q` is below
  0.4; the filter then bridges on wheels and gyro.
* **Slip** (`slip.py`): slip ratio `1 - sum(d_VO) / sum(d_wheel)` over a sliding window; `IMMOBILISED` when it stays
  high while motion is commanded; online chi from wheel-differential versus VO yaw rate.
* In Tier-0 frames there are no images, so VO cannot run (`health['vo_available'] = 0`). Localisation is then the
  wheel + gyro EKF by design, and the node does not let the missing VO score gate speed (`node.py`, step 2).

### 3.2 Perception and the observation states

* **Ground model** (`ground.py`): a v-disparity profile (RANSAC line + robust quadratic) in image space, and
  piecewise planes in 2 m range bands in the body frame. A band is accepted only if it is continuous with the
  nearer band, so terrain beyond a drop-off does not become "ground".
* **Positive obstacles, slope, roughness** (`positive.py`): heights relative to a local terrain level
  (morphological open-close of denoised heights). A cell is `POSITIVE` (lethal) when at least 2 points stand more
  than 0.15 m (the ground clearance) above that level, or when the slope exceeds 20 deg.
* **Missing ground** (`negobs.py`): per image column band, walking up from the bottom, disparity is compared with the
  expected ground disparity. A run of "missing" rows after observed ground is a gap. It becomes `DITCH_CANDIDATE`
  when the far wall is visible below the lip-to-reappearance chord (or the gap has no valid disparity and ground
  reappears near lip height), `CREST_SHADOW` when ground reappears well below the lip or not at all, and `OCCLUDED`
  when the gap starts behind an above-ground object. The theory is Matthies and Rankin (2003): a ditch of width w at
  range R subtends about `H w / R^2` rad, so it shrinks with the square of range.
* **Semantics** (`semantics.py`, `semantic_bev.py`): 5 classes (sky, obstacle, water/mud, unstable, stable). Fusion is
  monotone: semantics can raise a cost or mark `WATER`, never clear a geometric hazard.
* **Visibility** (`bev.py`): an expected-coverage lookup table says how many pixels flat ground in each cell would
  occupy, which makes "unseen" distinguishable from "should have been seen". `r_vis_m` is the farthest forward range
  up to which at least 60 % of the expected-visible cells were observed.

| `CellState` | Meaning | Planning treatment |
|---|---|---|
| `UNSEEN` | never resolved by the cameras | not certified; costed (never free unless `unknown_is_free`) |
| `GROUND` | observed ground | certified while fresh (5 s), or while health is nominal |
| `POSITIVE`, `DEPRESSION` | step or rock above clearance; points below local ground | lethal, persists in memory |
| `DITCH_CANDIDATE` | missing ground, reappears near lip height | not certified; lethal after 2 of 3 in-view frames confirm it |
| `CREST_SHADOW` | missing ground that falls away | not certified, costed as suspicious |
| `OCCLUDED` | hidden behind an object | not certified, never lethal |
| `WATER` | semantic water or mud | high cost, never certified |
| `DYNAMIC` | new occupancy where free ground was seen recently | lethal, extra 0.5 m inflation, held at least 2 s |

### 3.3 Planning, speed and safety

* **Certification and memory** (`rolling_map.py`): lethal cells persist until re-observed as ground in 2 of the last
  3 in-view frames; a 2 m launch apron is certified at start (the camera cannot see the first ~1.5 m) but is not
  evidence of free space.
* **Global planner**: step cost `1 + 10 c` on observed ground, `1.5` on never-observed cells (optimistic but
  penalised), higher on suspicious cells, infinite on lethal cells.
* **Speed governor** ("only as fast as it can see"): the stopping distance must fit inside certified ground,

  `d_stop(v) = v^2 / (2 a) + v T_r + B <= R_cert`, so `v_cap = a ( -T_r + sqrt(T_r^2 + 2 (R_cert - B) / a) )`,

  with `a = min(1.5 m/s^2, mu g)`, `T_r` = measured compute latency + one frame + actuator lag (0.2 s), `B = 0.5 m`,
  and `R_cert` = arc length along the planned path to the first non-certified cell, capped by `r_vis` and scaled by
  health. A static cap comes from ditch detectability (a 0.3 m design ditch must cover at least 6 pixels), and the
  platform cap of 2.0 m/s applies last.
* **MPPI** (Williams et al. 2016): unicycle rollouts with skid-steer wheel limits, cost = map cost at 3 footprint
  circles + lethal penalty + a certification term (probes at fractions of each rollout's stopping distance must lie
  on certified ground) + smoothness + speed tracking + terminal cost-to-go. Sample 0 is the shifted nominal and
  sample 1 is a full stop, so braking is always a candidate. Exploration noise is time-correlated. When the node
  passes the global planner's path as a guide, a share of the samples is drawn around a pure-pursuit rollout of that
  path, so the planner can find routes that swing far from the previous plan; all samples are still ranked by the
  same cost.
* **Supervisor**: modes from q with hysteresis (NOMINAL q > 0.7; CAUTION 0.4-0.7 at half speed with costs x1.5;
  DEGRADED 0.2-0.4 at quarter speed in go/look hops; SAFE_STOP latched on q < 0.2 for 3 s, immobilised, no progress
  after a look, or ESTOP). `STOP_AND_LOOK` rotates in place to certify adjacent ground after 3 s without 0.3 m of
  progress. A frame gap above 0.5 s stops the vehicle. The final gate simulates the command for 1 s and brakes if the
  footprint would enter a lethal core.
* **Mixer**: `v_L,R = v -/+ omega chi B / 2`, `omega_L,R = v_L,R / r`, with yaw-rate, acceleration, jerk and
  curvature-preserving wheel saturation limits.

## 4. Where each problem-statement requirement is met

| Requirement | How | Code | Evidence and status |
|---|---|---|---|
| Path / traversable-ground detection | Stereo ground model + observation states + semantic cost; only observed ground is certified | `autonomy/perception/`, `planning/rolling_map.py`, `planning/costmap.py` | Missing-ground detection range vs theory in analytic scenes (`results/perception_ditch_range.csv`, Simulated); segmentation on RUGD (`results/seg_*.json`, Tested; deploy model trained offline, not yet in the loop) |
| Visual localisation without GNSS | Stereo VO + integrity monitor + wheel/gyro EKF + slip check | `autonomy/localization/` | KITTI drift, camera only (`results/kitti_vo_*.json`, Tested); integrity AUROC on degraded KITTI (`results/integrity_07to05.json`, Tested). Closed-loop localisation error in simulation: in progress |
| Collision and hazard avoidance | Lethal cells from geometry, missing-ground detector, DYNAMIC state, costmap inflation, MPPI lethal and certification terms, 1 s forward gate, seen-distance speed cap | `perception/positive.py`, `perception/negobs.py`, `planning/`, `safety/supervisor.py` | Unit tests (`tests/test_plan_units.py`, `tests/test_perception_*.py`) and toy closed-loop tests with synthetic perception (`tests/test_node_closed_loop.py`: obstacle bypass, stop before a ditch band, find the gap, blind vehicle stays on the apron, typical-stack ablation enters the ditch); simulator referee results in progress (`docs/BUILD_LOG.md`) |
| Wheel commands | MPPI (v, omega) -> skid-steer mixer -> `WheelCmd` rad/s per side, 5 Hz, actuator watchdog on the world side | `control/mixer.py`, `node.py` | Unit tests (`tests/test_plan_units.py`, `tests/test_node_closed_loop.py`) |
| Operator link | 2 Hz binary telemetry (pose, mode, reason, v_cap, R_cert, waypoints, 64 x 64 4-bit costmap, health) designed for 9.6 kbit/s; operator GO / HOLD / RESUME / ESTOP / new goal; static console | `autonomy/link/`, `safety/supervisor.py::operator`, `operator_ui/` | Codec round-trip, corruption rejection, packet-size budget and link-emulator unit tests (`tests/test_plan_link.py`); console colour key and input API (`tests/test_video_console.py`). The console replays `telemetry.jsonl`; it is not connected live to a running episode, and nothing has been tested on a radio |

## 5. Logs written per run

| Path | Writer | Content |
|---|---|---|
| `run_dir/mission.json` | runner | mission as the operator entered it, goal uncertainty |
| `run_dir/result.json` | runner | referee verdict and metrics (see `docs/SIMULATION.md`) |
| `run_dir/gt/states.npz`, `cmds.npz`, `events.json` | runner | GT trajectory at 30 Hz, commands as applied, referee events |
| `run_dir/autonomy/telemetry.jsonl` | autonomy | every telemetry packet, encoded then decoded, with its size in bytes |
| `run_dir/autonomy/timings.csv` | autonomy | per-tick per-stage milliseconds |
| `run_dir/autonomy/debug/tick_*.npz` | autonomy | `DebugBundle` (BEV, costs, rollouts, plan, health, optional 320 x 200 left image) |
| `run_dir/autonomy/autonomy.log` | autonomy | log, including every mode change with its reason |

The command is sent to the world before logs are written, so logging adds no control latency.
