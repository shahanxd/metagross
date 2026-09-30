# METAGROSS simulation

The simulator exists to test the onboard stack in closed loop against ground truth (GT) it never sees. It is not a
physics engine. It is a kinematic vehicle, a procedural terrain, and two sensor models: a fast synthetic depth
sensor for batches, and a real-GPU stereo renderer for camera images. Everything here lives in `metagross/sim/`
and is on the world side of the ground-truth firewall (`docs/ARCHITECTURE.md`, section 2).

## 1. Sensor tiers

| Tier | `run_episode(sensor_mode=...)` | What the autonomy receives | Use |
|---|---|---|---|
| Tier 0 | `"tier0"` | `SensorFrame.disparity` (float32 px at 640 x 400), no images; `sensor_mode = "tier0_disparity"` | Fast batch runs of geometry, planning and control. VO cannot run (no images); localisation is wheel + gyro |
| Stereo | `"stereo"` | rectified `left_rgb` + `right_gray` at 640 x 400 from the Three.js renderer | Full stack: SGBM, VO, integrity monitor, segmentation |

Both tiers also provide wheel encoder angles and a z-gyro sample in every frame (`metagross/sim/sensors.py`):

* **Encoders**: cumulative wheel rotation quantised to 4096 ticks per revolution. They measure shaft rotation, so
  slip appears as encoder travel exceeding ground travel.
* **Gyro**: mean yaw rate over the camera interval plus a turn-on bias (1e-3 rad/s std), a bias random walk
  (1e-4 rad/s/sqrt(s)) and white noise (2e-3 rad/s/sqrt(Hz)) (`GyroParams`).

### 1.1 Tier-0 synthetic depth sensor (`metagross/sim/sensors.py::Tier0DepthSensor`)

It produces what a rectified stereo matcher would output, without rendering images. It is always labelled
"synthetic depth sensor".

1. **Terrain raster.** The heightmap is sampled on a camera-centred polar lattice (1.3 samples per pixel column in
   azimuth; range step max(5 cm, 1.5 % of range) out to 13 m). Consecutive samples form 3-D segments that are
   projected and rasterised into an inverse-depth z-buffer at 320 x 200. Steep walls cover every row they span, and
   a ditch interior hidden behind its near lip loses the depth test, as it would for a real camera.
2. **Objects.** Rocks (ellipsoids), trees (trunk cylinder + canopy sphere), bushes, logs and dynamic obstacles are
   ray-cast analytically; the nearest surface wins.
3. **Clean-up like a matcher.** Leak repair, small-hole fill, a left/right occlusion check (points the right camera
   cannot see are invalid), range cap 12 m.
4. **Measurement model** (`Tier0Params`): Gaussian disparity noise 0.25 px (at 640 x 400), 0.2 % gross mismatches of
   2-8 px, block dropout by surface texture (probability 0.60 on water, 0.12 on mud, 0.01-0.04 elsewhere), sky
   invalid, and lighting effects: a glare disc around the sun, dimming that inflates noise and dropout, dust and fog
   dropout that grows with range, and dust phantom returns.
5. **Output.** Nearest-upsampled x2 to 640 x 400, with disparity rescaled to full-resolution pixels, so frames match
   `defaults.stereo_calibration()` exactly.

A GT pass (noise-free depth and 5-class semantics) comes from the same raster for the evaluator only.

Measured cost on the development laptop, seed 102, 60 frames: mean 55.9 ms, median 50.0 ms per frame; physics mean
0.44 ms per step (`results/sim_benchmark.json`, shared 4-core laptop, 2 threads).

### 1.2 Three.js stereo renderer (`metagross/sim/render/`)

* **Engine.** Three.js r186 (WebGL2, MIT licence, vendored in `render/web/vendor/`). The scene is built directly in
  the world frame (x east, y north, z up).
* **Driver.** `bridge.py::ThreeRenderer` drives a real, headed Chrome or Edge through Playwright's synchronous API.
  Each render is one `page.evaluate` call, so the simulation stays in lock-step. The bridge refuses to run on the
  SwiftShader software rasteriser, so images always come from the GPU (on the laptop: ANGLE / Direct3D 11 on Intel
  Iris Xe). Assets are served from disk through `page.route` on a fake origin; no local socket is opened.
* **Cameras.** Intrinsics and `T_body_cam` come from `metagross/config/defaults.py`. The world passes the full 6-DoF
  body pose and `T_world_cam`, so the renderer never re-derives extrinsics. The right camera is 12 cm along the
  left camera's +x.
* **Appearance.** Procedural, hash-based ground albedo per material (identical on every GPU), band-limited so that
  sub-pixel detail does not alias differently in the two eyes; Preetham sky; sun shadows; fog layer; dynamic
  obstacles.
* **Camera effects** (constants at the top of `render/web/main.js`): first-order auto-exposure (1 s time constant),
  exposure-dependent sensor noise (2 DN read noise at unity gain), vignetting, about 0.5 px optical blur, lens flare
  and veiling glare for glare events, dimming and dust events with 0.5 s ramps.
* **GT passes** (evaluator only): depth in metres along the optical axis and 5-class semantic ids, plus a chase
  camera for the video (`render_chase`).
* **Convention checks** (`tests/test_render_world_consistency.py`, needs Chrome): a world point projected with
  `T_world_cam` and K lands within 0.5 px of where WebGL draws it, and SGBM on a rendered pair matches the Tier-0
  disparity on ground pixels (median absolute difference below 1 px). `tests/test_render_camera_model.py` checks
  the reference pinhole model.

Renderer benchmark (`scripts/bench_renderer.py`, 50 stereo pairs, DEV seed 100; `results/renderer_bench.json`,
label Simulated): browser-side render and readback 31.6 ms mean per pair; 120.5 ms mean per pair including the
Playwright transfer and decode in Python, which misses the 120 ms GO target recorded in the same file; SGBM on
rendered pairs gives valid disparity on 99.8 % of ground pixels within 10 m, with a median error of 0.22 px against
the renderer's GT depth.

## 2. Vehicle model (`metagross/sim/vehicle.py`)

Kinematic skid-steer at 50 Hz:

* wheel rates follow the command through a first-order lag (0.2 s) and an angular-acceleration limit;
* `v = (1 - s) r (w_l + w_r) / 2`, `omega = (1 - s) r (w_r - w_l) / (chi B)` (ICR model, Mandow et al. 2007), where
  `s` is longitudinal slip (0.03-0.08 per scenario, x2 on mud, x2.5 on water) and `chi` is the true effective-track
  factor of the material under the vehicle (1.30-1.70 by material). The autonomy only knows the nominal
  `CHI_NOMINAL = 1.4` and estimates chi online;
* terrain following from the four wheel-contact heights gives z, pitch and roll (REP-103 signs);
* encoders integrate the actual wheel rotation, including the part lost to slip.

The command is applied after the measured compute latency (`docs/ARCHITECTURE.md`, section 1). A world-side
watchdog zeroes the wheels 0.5 s after the last command.

## 3. Scenarios (`metagross/sim/scenario.py`, schema `metagross.scenario/1`)

Each scenario is a pure function of its seed. The world is 64 m x 40 m at 0.05 m terrain resolution: fBm hills
(0.3-1.2 m relief), material-dependent micro-roughness, a winding gravel trail from A to B, and patches of grass,
dirt and rocky ground, with generic rocks, trees, bushes and logs. A and B are 40-55 m apart. Materials: grass,
dirt, gravel trail, rocky ground, mud, water.

| Family (`seed % 6`) | Content | What it tests |
|---|---|---|
| `F1_trail` | extra rocks (0.2-1.0 m) on and beside the trail, trees along it | positive obstacles, trail following |
| `F2_ditch_field` | 1-3 trenches, 0.3-1.2 m wide, 0.4-1.0 m deep, steep walls, crossing the straight A-B line across the whole world, each with one 1.5-3 m bypass gap | missing-ground detection, finding the gap |
| `F3_crest_ditch` | a 0.4-0.8 m crest across the route; for even `seed // 6` a trench sits 1.2-2.5 m behind it on the back slope (hidden until the crest), otherwise the crest is a safe control | crest shadow vs ditch, not stopping forever at safe crests |
| `F4_sudden_obstacle` | a walker, box or boulder that starts behind an occluding bush and crosses the route when the vehicle is 4-6 m away | dynamic obstacles, reaction time |
| `F5_lighting` | low sun (3-8 deg) roughly ahead, glare (gain 1.5-3.0 for 4-8 s), dimming (gain 0.4 for 3 s), dust (4-8 s) | integrity monitor, graceful degradation |
| `F6_water_mud` | flat water and mud patches on the route with a dry way around | semantic hazards that geometry sees as flat |

Mission: success radius 2.0 m; timeout 60 s plus the straight A-B distance at 0.4 m/s; the goal handed to the
autonomy is rotated by a launch-heading error (0.5 deg std, clipped at 2 std), as an operator with an imperfect
heading reference would enter it.

**Solvability.** After building a candidate, grid A* on the GT hazard raster (`metagross/sim/gridplan.py`) must find
a corridor at least 1.2 m wide from A to B. Otherwise the candidate is discarded and regenerated from the next
attempt sub-seed (`SeedSequence([seed, attempt])`, at most 30 attempts).

**Hazard raster** (`metagross/sim/hazards.py`): `lethal = object | ditch | slope` (objects taller than the 0.15 m
ground clearance, ditch cells more than a set depth below the surface, slopes above 20 deg) and
`hazard = lethal | water | dynamic_rest`. The ditch cross-section is defined once and used both to carve the
terrain and to derive the mask, so mask and heightmap cannot disagree.

## 4. DEV / EVAL split and hashing

| Split | Seeds | Per family | Use |
|---|---|---|---|
| DEV | 100-129 | 5 | development, debugging and tuning |
| EVAL | 0-59 | 10 | held out: final numbers only, never tuned on |

* The seed lists are part of the frozen contract (`metagross/contracts/scenario.py`), so they cannot move quietly.
* `scripts/gen_scenarios.py` writes `data/scenarios/<split>/<seed>.json` and `data/scenarios/manifest.json`
  (seed, family, split, sha256, path).
* The sha256 is taken over the canonical JSON of the scenario (sorted keys, compact separators, ASCII, no NaN) with
  the `sha256` field set to `""`. `load_scenario()` recomputes it and refuses a modified file.
* Every `result.json` records `seed`, `family`, `split` and `sha256`, so any result can be traced to the exact
  world it ran on, and the video chase replay (`metagross/sim/render/chase_replay.py`) re-verifies the hash before
  rendering.

## 5. Referee and metrics (`metagross/sim/referee.py`, `metagross/sim/runner.py`)

The referee runs on the world side at every physics step (50 Hz) and emits timestamped events.

| Event | Rule |
|---|---|
| `collision` | the 0.80 x 0.60 m vehicle rectangle overlaps a lethal static footprint or a dynamic obstacle |
| `ditch_entry` | any wheel-contact point is on a ditch-mask cell, or more than 0.15 m below the local 1 m median height |
| `water_entry` | any wheel-contact point is on water or mud |
| `tip_over` | roll above 25 deg or pitch above 30 deg |
| `stuck` | less than 0.5 m of displacement over the last 20 s |
| `timeout` | mission timeout reached |
| `out_of_bounds` | body origin leaves the terrain (0.5 m margin) |
| `success` | body origin within the success radius of the true goal B |

Hazard events take precedence over success on the same step. By default every event above ends the episode. A
failure of the autonomy process ends it as `autonomy_error`; `wall_timeout` is used when a wall-clock limit is set.

`result.json` fields:

| Field | Definition |
|---|---|
| `success`, `failure_type`, `time` | verdict, first terminal event, end time (s) |
| `path_length`, `mean_speed` | GT path length (m), path length / time (m/s) |
| `optimal_path_length`, `spl` | shortest 1.2 m-wide corridor on the GT hazard raster (A* with line-of-sight shortcutting); SPL = success x optimal / max(path, optimal) |
| `min_clearance` | minimum signed distance (m) between the vehicle footprint and any lethal or dynamic footprint |
| `final_error` | distance (m) from the vehicle to B at the end |
| `ditch_entries`, `collisions`, `water_entries` | counts |
| `false_stops`, `stops` | stops (speed below 0.05 m/s for at least 1 s, after the first second, outside the goal radius) not justified by GT: no hazard in the 5 m x 1.2 m forward corridor, no dynamic obstacle within 5 m, no lighting event active |
| `compute_ms_mean`, `compute_ms_p95` | autonomy compute per tick, as reported in `WheelCmd.compute_ms` |
| `sensor_ms_mean`, `physics_ms_per_step`, `sensor_breakdown_ms` | world-side costs (not charged to the autonomy) |
| `n_frames`, `n_telemetry`, `n_cmd_timeouts` | frames sent, telemetry packets produced, actuator-watchdog trips |
| `config`, `seed`, `family`, `split`, `sha256`, `sensor_mode`, `camera_hz`, `noise_seed`, `machine`, `wall_time_s`, `error` | provenance |

Batches (`python -m metagross.sim.batch`) run many (seed, config) pairs over worker processes, skip jobs whose
`result.json` exists, and rebuild `summary.csv` from the result files. Closed-loop results are still being produced
and are not quoted in this document; see `docs/BUILD_LOG.md` for the engineering state and `docs/RESULTS.md` for
registered numbers.

## 6. Known gaps between this simulator and a real vehicle

* The vehicle is kinematic: no suspension, no tyre force model, no wheel sinkage. Slip is a scalar per scenario and
  material.
* Tier-0 disparity is a noise model, not a matcher: it has no real texture-dependent failures beyond the dropout
  table and no calibration error.
* Rendered images are procedural. They have no real vegetation, no motion blur from vibration, no rolling shutter,
  no lens distortion residuals, no rain or night.
* Terrain has no overhanging vegetation and no deformable ground (tall grass that is actually passable is not
  modelled).
* The operator link is emulated (bandwidth, loss, latency), not a radio.
