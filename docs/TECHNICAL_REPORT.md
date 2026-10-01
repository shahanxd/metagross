# METAGROSS: technical report

**Seen-ground autonomy for an unmanned ground vehicle in GPS-denied outdoor terrain**

Smart India Hackathon 2026, problem statement SIH26126, Bharat Electronics Ltd. [1] · 1 October 2026 · source code and all results files: [2]

This report goes with our slide deck. It covers what we built, why we built it that way, and what we have and have not measured.

**How to read the numbers.** Every number has a label and a source.

| Label | Meaning |
|---|---|
| **Tested** | measured on real data (real camera images with ground truth) |
| **Simulated** | measured in our simulator |
| **Estimated** | closed-form analysis, no measurement |
| **Proposed** | planned; not built or not run |
| **Literature** | someone else's published number, for context only |

The claims ledger `results/claims.csv` [3] lists each registered number with its label and source. On 1 October 2026 it held 366 claims: 89 Tested, 271 Simulated, 5 Estimated and 1 Proposed (counted from the CSV). Numbers marked **†** are not registered in the ledger: they come from repository run logs, build notes or figure sidecars, or we computed them from files in `results/`; the text says how we computed each one. Other numbers come from the `results/` file cited next to them; not all of them are individually in the ledger. Figures are PNG files in the repository and are referred to by path from the repository root.

---

## 1. Summary

**What it is.** METAGROSS is the onboard software for a small skid-steer UGV. It drives from a launch point A to a goal B in outdoor terrain without GNSS, LiDAR, radar or a prior map. Its inputs are a stereo camera, wheel encoders and a z-axis gyro. Its outputs are left and right wheel speeds at 5 Hz, plus a telemetry packet about every 0.6 s (median; a 2 Hz timer on the 5 Hz tick), sized for an assumed 9.6 kbit/s operator link. The design follows one rule, which is our thesis and not a proven property: unknown is never free. The vehicle should drive only on ground its cameras have observed, and only as fast as it can stop inside that ground.

Most planners treat unobserved cells as free. For a ground vehicle this is how it drives into a ditch, because in stereo a ditch appears as *missing ground*, not as an obstacle. So we do three things:
- certify as drivable only the ground the cameras have observed;
- detect missing ground explicitly;
- cap speed so that the modelled stopping distance fits inside certified ground: `v²/(2a) + v·T_r + d_m ≤ R_cert`.

**What works today.**

- **Closed loop in simulation.** On 60 pre-registered EVAL worlds (run 2: we re-ran EVAL once after a gyro-bias fix made on DEV, a disclosed protocol deviation, section 7.5), with a synthetic depth sensor ("tier-0", no images), the full stack (FULL) reached B in **33/60** runs. A baseline that treats unknown ground as free, with no missing-ground detector and a fixed 1.5 m/s (TYPICAL), reached B in **31/60**. FULL never left the map; TYPICAL did so **7** times, and on those 7 seeds FULL reached B twice and was stuck 4 times (Simulated, `results/closed_loop_eval.json`).
- **Visual odometry on real images.** Camera-only stereo VO on real KITTI driving, KITTI-protocol translational error 1.53 % (05), 2.05 % (07) and 2.11 % (00, frames 1101-4540 only); 1.87 % pooled over 5.81 km, all with the previous VO settings (the current settings give 2.04 % on 07). No loop closure; VO did not fail on any of the 7,302 frames (Tested, `results/kitti_summary.json`).
- **Integrity monitor (offline only).** Trained on KITTI 07 and tested on KITTI 05, both with synthetic degradations, a learned monitor ranks failed VO frames above good ones with AUROC **0.957** under a lenient label (no pose, or > 0.1 m or > 1.0° per frame); 0.849 on the 6 silent failures; 0.73 under a strict 0.05 m / 0.25° label. It has never gated anything in a closed loop: tier-0 fixes q = 1 (Tested, `results/integrity_07to05.json`).
- **Compute and link.** In tier-0 simulation (no images, so no SGBM, VO or segmenter) on a 4-vCPU container, sensing to wheel command takes 62.5 ms median and 88.7 ms p95 of the 200 ms tick; a full stereo + VO tick has not been timed end to end (Simulated). All 4,437 replayed held-out DEV packets (registered) and the 3,505 EVAL packets (†) met the 480 B encoder target inside the 600 B budget, by coarsening the costmap to 0.5 m cells when needed (82 % of EVAL packets). The link emulator and console are replay only; no radio was used (Simulated).

**What the evidence does not show.**
- The +2/60 difference in reached-B is within batch noise (exact McNemar p = 0.79, †), so **the seen-ground design is not shown to raise the success rate**.
- Other safety events did not improve: collisions 7 vs 5 (all of FULL's in F4), ditch entries 2 vs 2, stuck runs 5 vs 1, stops not justified by ground truth 5 vs 2.
- Both of FULL's ditch entries (seeds 14 and 26, both F3) were a trench just behind a crest, where certification wrongly assumed the ground continued.
- On sudden obstacles (family F4), FULL did *worse* than the baseline: 2/10 against 5/10. We have not diagnosed why.
- Without images neither stack can see water: FULL 1/10, TYPICAL 0/10.
- All closed-loop numbers are tier-0, so VO, the integrity monitor and the segmenter have never run in a scored closed loop.

**What is pending.**
- GPU training of the deploy terrain segmenter on the team laptop (RTX 3050) is the next step. No trained deploy model exists yet, and no number in this report comes from one.
- A closed loop in rendered-stereo mode.
- A live operator console; today the console only replays logs.
- Timing on an embedded computer.
- Any run on hardware.

## 2. Problem and constraints

**The problem statement.** SIH26126 asks for vision-based autonomous navigation of a UGV in outdoor terrain [1]: the vehicle must find traversable ground, localise itself without GNSS, avoid obstacles and hazards, and command its wheels.

**Why GPS-denied (more precisely, GNSS-denied).** GNSS interference in India is routine and documented:
- **Nov 2023.** The DGCA issued an advisory circular on GNSS interference in airspace [4].
- **Mar 2025.** The Ministry of Civil Aviation reported 465 interference or spoofing incidents from November 2023 to February 2025, mostly in the Amritsar and Jammu region [5].
- **Mar 2026.** The count had reached 2,354 reports up to December 2025, plus 623 within 60 NM of Delhi in January and February 2026 alone [6].
- **Delhi.** Spoofing hit GPS-based approaches at Delhi's IGI airport, and the DGCA issued an SOP for real-time reporting [7].
- **NavIC.** India's NavIC constellation lost IRNSS-1F to a clock failure in March 2026, leaving three satellites providing positioning [8]. A standalone 3-D fix normally needs four, so NavIC alone cannot give one (our inference).

These are aviation reports, not ground measurements, but a ground vehicle near a border cannot count on satellite positioning.

**Use cases.** BEL's Robotic Surveillance Platform (RSP) is a UGV listed with a 3.6 km/h (1.0 m/s) top speed, up to 4 h endurance, and waypoint navigation with obstacle detection and avoidance [9]. We designed for that class of mission: patrols, surveillance and short point-to-point moves near borders or installations, on unpaved ground, with GNSS jammed or spoofed. The mission we test is one-way: reach B from A, given B's position relative to the launch pose.

**Constraints we set.**

| Constraint | Choice | Why |
|---|---|---|
| Sensing | Passive stereo camera, wheel encoders, z-gyro; no LiDAR, no radar | Vision-based per [1]. Military UGVs often need a non-emitting mode [10]. Cost and mass |
| Camera | ZED 2i class [11]: 12 cm baseline, 640 × 400 px rectified, 72° HFOV (fx ≈ 440 px), 0.9 m high, pitched 12° down | Longer baseline than a 7.5 cm OAK-D Lite class camera [12]: modelled depth σ at 10 m is 0.47 m vs 0.76 m (Estimated, `results/theory.json#stereo`) |
| Vehicle | Scout-Mini class skid-steer: track 0.50 m, wheel radius 0.13 m, footprint 0.80 × 0.60 m, clearance 0.15 m, caps 2.0 m/s and 1.2 rad/s | A small platform typical of the target class |
| Compute | Jetson Orin Nano class [13] (Proposed, not measured). All timings so far come from a laptop or a cloud CPU | Low-power embedded target |
| Operator link | 9.6 kbit/s, 20 % loss, 0.4 s latency (a design assumption for our emulator); no video | Narrow-band radio |
| Map | None. The launch pose defines the mission frame | GNSS-denied, unknown area |

## 3. System design

### 3.1 Processes, sensors and rates

The onboard stack (`metagross/autonomy/`) runs in its own OS process and sees only frozen message types: `SensorFrame` in, `WheelCmd` and `Telemetry` out. In simulation the world process owns ground truth and runs lock-step with the stack. Each command is applied `ceil(compute_ms / 20 ms)` physics steps after its frame, so compute latency costs simulated time.

| Signal | Rate | Content |
|---|---|---|
| Camera frame | 5 Hz in batch runs (10 Hz demo setting) | Rectified left RGB and right grey at 640 × 400 (stereo mode), or a synthetic disparity map (tier-0) |
| Encoders, gyro | every frame | Cumulative wheel angle (4096 ticks/rev); mean z-rate with bias, bias random walk and white noise |
| Wheel command | 5 Hz | Left and right wheel rates (rad/s) |
| Segmenter | every 3rd frame (design; not in the loop, no trained weights) | 5-class mask; the last mask is reused on the two frames in between |
| Global planner | 1 Hz, or at once when a lethal cell cuts the route | Cost-to-go field |
| Telemetry | 2 Hz timer on the 5 Hz tick: one packet every 0.6 s median (Simulated, ledger `closed_loop_eval_tier0_FULL_telemetry_period_s`); the 600 B budget assumes 2 Hz, so it is conservative | Pose, mode, speed cap, certified range, waypoints, 64 × 64 costmap, health |

### 3.2 Data flow in one control tick

```
 stereo pair (or tier-0 disparity), encoders, gyro
 [1]  Stereo: CLAHE + SGBM -> disparity (shared with VO)
 [2]  Localiser: stereo VO | depth odometry -> integrity monitor -> SE(2) EKF (wheels + gyro) -> pose, q
 [3]  Perception: ground model -> positive obstacles -> missing-ground detector -> BEV cell states, r_vis
      [3a] Segmenter (every 3rd frame) -> semantic cost, WATER
 [4]  Rolling map (0.2 m cells, 80 m square, mission frame): states, certification age, ditch confirmation
 [5]  Costmap: inflated cost, certified mask, lethal cores
 [6]  Global planner: cost-to-go from B, descent path, 3 m lookahead
 [7]  Supervisor: drive mode from health, progress and stalls
 [8]  Speed governor: R_cert, v_cap
 [9]  MPPI: 512 rollouts x 30 steps x 0.1 s -> (v, omega)
 [10] Forward gate: simulate 1 s, brake if the footprint would enter a lethal core
 [11] Skid-steer mixer -> WheelCmd
 [12] Telemetry encoder (2 Hz timer; one packet per 0.6 s in practice) -> log (replay only, no radio)
```

**Status.** In every scored (tier-0) run, stereo matching (step 1), the VO branch of step 2, the integrity monitor (q fixed at 1) and the segmenter (step 3a) did not run. Stereo matching and VO run only in rendered-stereo mode, which has no scored batch; the segmenter has no trained weights.

### 3.3 Perception modules

**Stereo depth.** We apply contrast-limited histogram equalisation [14], then OpenCV's [15] semi-global matching [16] (3-way, 64 disparities, 5 × 5 blocks). Rows above the horizon are cropped. Depth is `Z = f·b/d`, with stereo baseline `b` and noise `σ_Z = Z²·σ_d/(f·b)`. At `σ_d = 0.25 px` that is 0.47 m at 10 m (Estimated). Usable range is capped at 12 m. SGM is the standard dense matcher that fits a CPU budget. On a 50-pair rendered benchmark (DEV seed 100, ground pixels within 10 m by ground-truth mask) it gives valid disparity on 99.8 % of ground pixels with a median error of 0.22 px (Simulated, `results/renderer_bench.json`). On a full rendered DEV 102 drive, valid disparity is 69-73 % (†, `docs/PIPELINE_STATUS.md`). SGBM has not run in any scored closed-loop batch.

**Ground model.** A v-disparity profile [17] (a RANSAC line [18] plus a robust quadratic) in the image, and piecewise planes in 2 m range bands in the body frame. A band counts as ground only if it continues the nearer band, so terrain beyond a drop-off does not become "ground".

**Positive obstacles and slope.** Heights are measured against a local terrain level: a morphological open-close of denoised heights with one fixed 1.5 m window, after the progressive morphological filter [19]. A cell is lethal when at least 2 points stand more than 0.15 m (the ground clearance) above that level, or when the slope exceeds 20°. Before this change we used raw per-cell height extremes, and 1,137 of 1,216 POSITIVE cells on DEV seed 102, tick 0 were false (Simulated, DEV, †, `docs/BUILD_LOG.md`).

**Terrain segmenter.** The segmenter labels five classes: sky, obstacle, water/mud, unstable and stable. It is LR-ASPP on MobileNetV3-Large [20] (3.2 M parameters), designed to run through ONNX Runtime [21] on the CPU. Fusion is monotone: semantics can raise a cell's cost or mark it WATER, but can never clear a geometric hazard.
- **Why a small segmenter.** Learned systems predict traversability directly: GA-Nav [22], WVN [23], TerrainNet [24], and Velociraptor [25] (which uses camera plus LiDAR). We could not validate any of them in our loop and want geometry to keep the veto, so the segmenter is a cost layer under geometry.
- **Status.** Built, not in the loop. The deploy model is not trained yet, so every closed-loop run in this report ran without semantics.

## 4. The seen-ground idea

### 4.1 Cell states

Each cell of the 0.1 m egocentric bird's-eye grid gets one state. Only `GROUND` can be certified.

| State | Meaning | Planning treatment |
|---|---|---|
| `UNSEEN` | never resolved by the cameras | not certified; given a cost, never treated as free (except in the TYPICAL baseline) |
| `GROUND` | observed ground | certified while fresh (5 s) or while localisation health is nominal |
| `POSITIVE`, `DEPRESSION` | step or rock above clearance; points below local ground | lethal; kept in memory |
| `DITCH_CANDIDATE` | missing ground that reappears near lip height | not certified; lethal once confirmed in 2 of 3 in-view frames |
| `CREST_SHADOW` | missing ground that falls away | not certified; costed as suspicious |
| `OCCLUDED` | hidden behind an object | not certified; never lethal |
| `WATER` | semantic water or mud | high cost; never certified |
| `DYNAMIC` | new occupancy where free ground was seen recently | lethal, extra 0.5 m inflation, held ≥ 2 s |

**Unseen vs should-have-been-seen.** A lookup table gives how many pixels flat ground in each cell should occupy, which separates "unseen" from "should have been seen". `r_vis` is the farthest range up to which at least 60 % of the expected-visible cells were observed.

**Launch apron.** The camera cannot see the first ~1.5 m, so a 2 m launch apron is certified at the start. It is not treated as evidence of free space.

### 4.2 Missing ground

**Geometry.** From a camera at height `H`, a ditch of width `w` at range `R` subtends about `θ ≈ H·w / (R·(R + w))` [26]. This falls off as `1/R²`; a positive obstacle of height `h` subtends about `h/R`. Requiring 6 pixels on target, `θ·f ≥ 6`, gives the first resolvable range `R_det = (−w + sqrt(w² + 4·H·w·f/6)) / 2`. From our 0.9 m mast:

| Target | Analytic first resolution (Estimated, `results/theory.json`) | Detector on synthetic scenes, no noise (Simulated, `results/perception_ditch_range.csv`) |
|---|---|---|
| 0.3 m wide ditch (design ditch) | 4.3 m | 4.45 m |
| 0.5 m wide ditch | 5.5 m | 5.45 m |
| 0.8 m wide ditch | 6.9 m | 7.2 m |
| 0.3 m tall rock | 22 m (beyond our 12 m cap) | — |

**Detector.** `perception/negobs.py` walks up each 4-pixel column band from the bottom of the image, comparing measured disparity with the ground model's expected disparity. A run of "missing" rows after observed ground is a gap. The gap becomes a `DITCH_CANDIDATE` only if all of these noise-aware tests pass: the far wall lies below the chord from the lip to where ground reappears; the range step is significant against the lip's own residual; the far wall is a disparity plateau; and neighbouring column bands agree. Ground that reappears well below the lip, or not at all, is `CREST_SHADOW`; a gap right behind an object is `OCCLUDED`. The approach follows JPL's stereo negative-obstacle detection [27] and uses the geometry of [26].

**Figure 1** (`deck_assets/final/perception_gallery.png`; single rows in `perception_row_1.png` to `perception_row_3.png`) shows single rendered frames from three DEV worlds:
- A boulder becomes lethal, and the ground behind it stays occluded, not free.
- A 1.04 m trench becomes a lethal band at about 4.6-5.7 m.
- Past a crest, the ground is marked unknown even though here it is drivable.

The frames are hand-picked, show artefacts (for example a spurious 50-cell ditch streak), and are not a closed-loop result.

**Before and after the noise-aware detector.** We ran the perception of commit e2cbefd and the current perception back to back on the same DEV frames (Simulated, `results/perception_dev.json`). On 85 rendered stereo frames (the tuning set, so in-sample):
- The share of true ditch cells on the path that were certified as ground fell from 31.5 % to 0.0 % (0 of 2,244).
- Hazard cells certified inside the 2 m/s stopping envelope fell from 18 to 0.

In tier-0 the same change still left 25 hazard cells certified inside the envelope, down from 195, and 94 of 10,712 on-path ditch cells (0.9 %, down from 1,136) certified as ground. Some of them lie on a trench just beyond a crest (DEV seed 110), which the detector sees only at about 2.8 m (section 4.4).

### 4.3 Speed governor

The governor is designed so the vehicle can stop inside certified ground:

```
d_stop(v) = v²/(2a) + v·T_r + d_m  ≤  R_cert
v_cap     = a·( −T_r + sqrt(T_r² + 2·(R_cert − d_m)/a) )      (0 if R_cert ≤ d_m)
```

The terms:
- `a = min(1.5 m/s², μ·g)`, where μ is the lowest friction proxy on the probed path.
- `T_r` is the reaction time: measured compute latency (the 90th percentile of the last 25 ticks, clamped to [0.2, 0.6] s) plus one camera frame (0.2 s) plus actuator lag (0.2 s). A slower computer therefore lowers the speed cap, up to the 0.6 s clamp; beyond it the bound no longer holds. The bound is also only as good as the certification itself (section 4.4).
- `d_m = 0.5 m` is a margin.
- `R_cert` is the arc length along the MPPI nominal path, from the front bumper to the first non-certified cell. It is capped by `r_vis` and scaled by a health factor `min(1, q/0.7)`.

Two further caps apply. The static cap from design-ditch detectability requires the vehicle to be able to stop within `R_det` = 4.3 m. The 2.0 m/s platform cap applies last.

With `T_r` = 0.6 s (all Estimated, `results/theory.json`; flat ground, fixed deceleration):
- The stopping distance from 2 m/s is 3.03 m.
- The design-ditch cap is 2.60 m/s, above the platform cap.
- Over mast heights of 0.4-1.6 m and ditch widths of 0.2-1.2 m, the level-ground, small-angle model gives safe speeds from 1.61 m/s (registered) to 4.86 m/s (not in the ledger). On that grid it does not drop below the RSP's listed 1.0 m/s top speed [9]; sloped or crested ground is not covered.

MPPI uses the same stopping distance: a rollout is charged for every probe point, placed at fractions of its stopping distance, that is not certified.

### 4.4 What the rule does not guarantee

- **Moving objects.** Certification is a statement about ground *as it was seen*. It does not cover an object that walks into certified ground from outside the ±36° field of view. This may contribute to the F4 result in section 8.1, but that regression is undiagnosed.
- **Hidden trenches.** Certification also assumes that the ground continues as modelled just past the last observed row. A trench close behind a crest is therefore seen only at short range: about 2.8 m on DEV seed 110 (`results/perception_dev.json#notes`). Both of FULL's ditch entries on EVAL were of this hidden-trench type.

## 5. Localisation

**Stereo VO** (`localization/vo.py`). The pipeline runs frame to frame:
1. Shi-Tomasi corners [28] on an 8 × 5 bucketing grid.
2. Pyramidal KLT tracking [29], [30] with a forward-backward check under 1 px [31].
3. Back-projection with the previous frame's disparity.
4. PnP-RANSAC with AP3P [18], [32].
5. Levenberg-Marquardt refinement and a motion-sanity gate.

There is no bundle adjustment and no loop closure. The Mars rovers used this family of method for years, including to detect slip [33]. We did not use ORB-SLAM3 [34]: it is GPL-3.0, which a defence product is unlikely to accept; its main gain, loop closure, rarely applies to a one-way A-to-B mission; and loop closing costs compute that we need for perception [35].

**Depth odometry** (`localization/depth_odom.py`). Tier-0 frames carry no images, so VO cannot run. Instead we register consecutive disparity maps directly (cf. DIFODO [36]; ours is a Gauss-Newton alignment, not DIFODO's range-flow formulation).
- **Solver.** Gauss-Newton over `T ∈ SE(3)` minimises Huber-weighted residuals `r = D_cur(π(T·p)) − f·b/z(T·p)`, starting from the wheel + gyro prior. The covariance is `σ_r²·(JᵀWJ)⁻¹`, inflated because the residuals are spatially correlated.
- **Gates.** Yaw must agree with the gyro. Forward travel must not exceed wheel travel plus a margin: wheels over-count under slip but do not under-count.
- **Fallbacks.** On featureless ground there is no measurement, and the filter dead-reckons. In stereo mode VO stays the primary source.

**EKF** (`localization/ekf.py`). The state is planar SE(2), `[x, y, ψ]`, in the mission frame [37]. It starts at (0, 0, 0) with zero covariance, because the launch pose defines the frame. The prediction uses the skid-steer model with track `W` and an effective-track factor χ [38]:

```
d = r·(Δφ_L + Δφ_R)/2,     Δψ = r·(Δφ_R − Δφ_L)/(χ·W)   or, with the gyro,   Δψ = (ω_z − b̂)·Δt
```

- **Relative updates.** VO and depth-odometry increments are relative motions, so they are fused by stochastic cloning: `h = [R(ψ_c)ᵀ(p − p_c), ψ − ψ_c]`, taken against a clone of the pose at the previous frame.
- **Weighting and rejection.** The VO covariance is inflated by `1/inliers` and by `1/q`. VO is rejected when `q < 0.4`, and the filter then bridges on wheels and gyro (stereo mode only; never exercised in a scored run).
- **Online estimates.** χ is estimated online. A slip ratio `1 − Σd_VO/Σd_wheel` flags IMMOBILISED.

**Gyro bias.** A scalar Kalman filter (turn-on σ 1e-3 rad/s, random walk 1e-4 rad/s/√s) learns the bias from 1 s windows of standing still and from VO. It does not learn from depth-odometry yaw, which is biased on some terrain. This filter caused our one protocol deviation (section 7.5).

**Integrity monitor** (`localization/health.py`). VO can fail silently: it returns a confident but wrong pose. We borrow the idea behind GNSS RAIM [39], an independent check of whether the solution can be trusted. Our check is a learned classifier, not a residual test.
- **Features.** Twelve per frame: inlier count, inlier ratio, reprojection RMSE, feature coverage, track age, smallest pose-Hessian eigenvalue, blur, saturation, darkness, contrast, a dark-channel haze cue [40], and ground disparity density.
- **Model.** A logistic model, `p_fail = σ(w·z + b)`, fitted with scikit-learn [41]. `q = 1 − p_fail` is smoothed so it drops fast and recovers slowly.
- **Training data.** Real KITTI frames with synthetic degradations: gamma, exposure clipping, Koschmieder haze [42], sun flare, motion blur and smudge. We also evaluated a Platt-recalibrated candidate [43]. The onboard weights (`models/integrity.json`) are the KITTI-only model trained on degraded KITTI 05 and tested on 07 (AUROC 1.000 there, on only 20 positives with no silent failure; registered). The 07→05 model in section 8.4 is a separate cross-check trained the other way round, so its 0.957 is evidence about the method, not a score of the onboard weights.
- **In tier-0** there are no images, so the monitor is not called and `q` is fixed at 1.

## 6. Planning and safety

**Rolling map and costmap.** The rolling map is 80 m square at 0.2 m, in the mission frame. Each cell stores state, cost, certification age, ditch-confirmation counts and a lethal memory. A lethal cell persists until it is seen as ground again in 2 of the last 3 in-view frames. The costmap inflates lethal cells by the footprint circles plus a margin and exports a certified mask.

**Global planner.** A cost-to-go field is grown from the goal with an 8-connected Dijkstra [44] (`skimage.graph.MCP_Geometric` [45]) on a 0.4 m grid:

```
step = (1 + 10·c_obs) · (1 + 0.5·f_unseen + 1.0·f_suspicious)      (void cells ×6; lethal = wall)
```

Seen ground costs 1 per metre, unseen ground 1.5 and suspicious ground 2.0. The route prefers seen ground but may cross unknown ground at a cost, and MPPI then decides how fast to drive there. An earlier rule charged 9 per suspicious cell against 1.5 per unseen cell. It routed the vehicle around ground it had seen, so we changed it on DEV.

**MPPI** [46]. 512 unicycle rollouts × 30 steps × 0.1 s (a 3 s horizon), with skid-steer wheel limits applied to every sample.
- **Cost.** Map cost at 3 footprint circles + lethal penalty + certification term (section 4.3) + smoothness + speed tracking + terminal cost-to-go.
- **Update.** Weights are `w_k = exp(−(S_k − min S)/λ)` with λ = 3. The nominal is updated by the weighted noise and smoothed with a Savitzky-Golay filter [47]. Exploration noise is time-correlated.
- **Candidates.** Sample 0 is the shifted nominal and sample 1 is a full stop, so braking is always a candidate. Sampling-based MPC with sparse, indicator-like costs such as our lethal and certification terms is studied in [48].
- **Guided sampling.** A quarter of the samples are drawn around a pure-pursuit rollout of the global path, so MPPI can find routes that swing far from its previous plan. All samples are ranked by the same cost. This is the idea published as Biased-MPPI [49]. Our source docstring wrongly attributes it to [48]; we will correct that.

**Supervisor** (`safety/supervisor.py`).

| Mode | Entry | Effect |
|---|---|---|
| NOMINAL | q > 0.7 (upgrading needs q > 0.75 for 1 s) | full governor speed |
| CAUTION | 0.4 < q ≤ 0.7 | half speed, costs ×1.5 |
| DEGRADED | 0.2 < q ≤ 0.4 | quarter speed, in 2 s go / 1 s look hops |
| STOP_AND_LOOK | 3 s without 0.3 m of progress | turn in place +45°, −45° and back to certify nearby ground, then replan |
| DEAD_END (one tick) | still no progress after a look | remember the corner; the planner routes elsewhere |
| SAFE_STOP (latched) | q < 0.2 for 3 s, immobilised, 6 dead ends without progress, or ESTOP | zero command until the operator resumes |

In all scored (tier-0) runs q = 1, so the q-driven modes (CAUTION, DEGRADED, and SAFE_STOP on low q) never triggered; they are unit-tested only. The operator RESUME path is also unit-tested only.

**Dead-end memory.** This handles ditch lines that span the whole route, where the crossing gap may be 10 m or more off the straight line. The stall corner is remembered as a disc in the mission frame. Inside the disc, unknown cells become walls and seen cells get a penalty. The cost-to-go field then sends the vehicle the other way along the ditch. Discs expire, because the world may have been misread.

**Forward gate.** After MPPI, the chosen (v, ω) is simulated for 1 s. The vehicle brakes if its footprint would enter a lethal core, unless the motion increases clearance. In-place turns near a lethal core are checked against the swept body rectangle; this fixed a DEV case where a turn swung a body corner into a rock. Two watchdogs also stop the vehicle: a sensor-frame gap over 0.5 s, and, on the world side, 0.5 s without a new command.

**Mixer.** `v_L,R = v ∓ ω·χ·W/2` and `ω_L,R = v_L,R / r`, with limits on yaw rate, acceleration and jerk, and wheel saturation that preserves curvature.

**Operator link.** A packet carries pose, mode and a reason text; `v_cap`, `R_cert` and speed; up to 5 waypoints; a 64 × 64 costmap at 4 bits per cell covering 16 m; and health values. Packets are zlib-compressed with a CRC-32. The encoder targets 480 B, which is 80 % of the 600 B budget (9.6 kbit/s ÷ 8 ÷ 2 Hz). It steps down a fixed fidelity ladder until the packet fits: fewer ground levels, then 0.5 m cells, then fewer health fields, and finally no costmap.

**Operator console.** The console (`operator_ui/`) **replays** logged telemetry. It is not connected to a running vehicle, and its buttons are not wired to the autonomy.

**Figure 2** (`deck_assets/final/console_1.png`, `console_2.png`, `console_3.png`) shows replayed packets: 3.0 m of certified ground with a crest hiding the rest (EVAL seed 44, t = 22.8 s); a STOP_AND_LOOK in front of obstacles; and the new route afterwards, driven at 1.47 m/s (seed 48, t = 26.4 s) (†, `deck_assets/final/console.json`). These are selected successes. STOP_AND_LOOK happened in 8 of 60 FULL EVAL runs, and 2 of those 8 reached B (†, run logs via `deck_assets/final/console.json`).

## 7. Simulator and evaluation protocol

### 7.1 Worlds and families

**Worlds.** Each scenario is a pure function of its seed: a 64 m × 40 m world at 0.05 m resolution containing fBm hills with 0.3-1.2 m relief, a winding gravel trail, patches of grass, dirt, rock, mud and water, and rocks, trees, bushes and logs. A and B are 40-55 m apart. Grid A* on the ground-truth hazard raster must find a corridor at least 1.2 m wide; otherwise the world is regenerated. The family is `seed % 6`.

| Family | Content | Tests |
|---|---|---|
| F1 trail | rocks 0.2-1.0 m on and beside the trail, trees | positive obstacles |
| F2 ditch field | 1-3 trenches 0.3-1.2 m wide, 0.4-1.0 m deep, across the whole world, each with one 1.5-3 m gap | missing ground, finding the gap |
| F3 crest + ditch | a 0.4-0.8 m crest; on half the seeds a trench 1.2-2.5 m behind it, otherwise a safe crest (control) | crest shadow vs ditch, false stops |
| F4 sudden obstacle | a walker, box or boulder that starts behind a bush and crosses the route when the vehicle is 4-6 m away | dynamic obstacles |
| F5 lighting | low sun ahead, glare, dimming, dust | graceful degradation |
| F6 water / mud | water and mud on the route, with a dry way round | hazards that geometry sees as flat |

**Vehicle model.** The vehicle is a kinematic skid-steer model at 50 Hz. It has a 0.2 s wheel-rate lag and 3-8 % longitudinal slip (×2 on mud, ×2.5 on water). The true χ varies from 1.30 to 1.70 by material; the autonomy knows only the nominal χ = 1.4.

### 7.2 Sensor tiers

**Tier-0 synthetic depth sensor** (`sim/sensors.py`). It produces what a stereo matcher would output, without rendering images.
- **Geometry.** Terrain is rasterised into an inverse-depth z-buffer, so a ditch interior hidden behind its lip loses the depth test. Objects are ray-cast.
- **Noise model.** 0.25 px Gaussian disparity noise, 0.2 % gross mismatches, texture-dependent dropout, and lighting effects (a glare disc, dimming, dust dropout with phantom returns). The nominal dropout parameters are 0.60 on water and 0.12 on mud, but the measured water return on DEV drives was about 88 % of pixels (†, `docs/BUILD_LOG.md`); we have not yet reconciled the two.
- **Use.** It runs on any CPU and produced **all** closed-loop numbers in this report.
- **What it cannot test.** It has no images, so there is no VO, integrity monitor or segmenter in these runs. F5 lighting events act only as depth dropout and noise; this is not a camera-glare test.

**Rendered stereo** (`sim/render/`). A Three.js r186 [50] renderer driven through Playwright [51]. It renders procedural albedo, sky, sun shadows, fog, auto-exposure, sensor noise, vignetting, blur, lens flare and veiling glare.
- **Speed.** Browser-side render and readback take 31.6 ms per pair. Including transfer to Python it is 120.5 ms, which misses the 120 ms target we set (Simulated, `results/renderer_bench.json`).
- **Status.** SGBM and VO run on its images end to end. No closed-loop batch has been run in this mode yet.

### 7.3 Referee

The referee runs on the world side at 50 Hz. A run succeeds when the body origin comes within 2 m of the true B.

| Event | Rule |
|---|---|
| `collision` | the footprint overlaps a lethal or dynamic footprint |
| `ditch_entry` | a wheel contact is on a ditch cell, or more than 0.15 m below the local median |
| `water_entry` | a wheel contact is on water or mud |
| `tip_over` | the vehicle tips over |
| `stuck` | less than 0.5 m of progress in 20 s |
| `timeout` | the mission time limit is reached |
| `out_of_bounds` | the body leaves the terrain |

The results also use two derived counts:
- **`arrived_short`:** the stack declared arrival on its own pose estimate but ended outside the 2 m radius. This is an odometry error.
- **`false_stops`:** stops that ground truth does not justify, with no hazard in the forward corridor, no dynamic obstacle within 5 m, and no lighting event.

### 7.4 Ground-truth firewall, pre-registration and baseline

**Ground-truth firewall.** Six layers keep simulator ground truth out of the stack (`docs/ARCHITECTURE.md` §2): an AST import scan; a separate OS process; a pinned `SensorFrame` field set; a runtime audit-hook file guard that blocks every `open` outside an allow-list; seed pre-registration; and evaluation only after the run.

**Pre-registration** [52], [53].
- **Scenario set.** 90 scenarios: EVAL seeds 0-59 (ten per family) and DEV seeds 100-129 (five per family). Each is generated deterministically from its seed.
- **Hashing.** Each scenario is hashed with SHA-256 over its canonical JSON, and `load_scenario()` refuses a modified file.
- **Timing.** The manifest was committed in c14e23f (30 Sep 2026, 10:40 UTC), before any EVAL seed ran. On 1 Oct 2026, `results/scenario_manifest.json` was byte-identical to that commit (SHA-256 `ef0ae07f…`; checked with `git show c14e23f:metagross/results/scenario_manifest.json | sha256sum`; the repository was flattened to the root later, in fc44372).
- **Tuning.** All tuning used DEV seeds only, with one disclosed exception: the decision to fix the gyro-bias filter was triggered by EVAL run 1 (section 7.5), although the fix itself was developed and checked on DEV.

**Baseline.** TYPICAL is the same code with three switches flipped: unknown cells are free; the missing-ground detector is off; and the governor is off, with a fixed 1.5 m/s set-point. It keeps the same localiser, positive-obstacle detection, MPPI, supervisor and forward gate. The comparison therefore measures the three seen-ground parts as one bundle, including a different speed policy. It does not attribute an effect to any single part.

### 7.5 Disclosure: EVAL run 1

**Run 1.** The first EVAL run used the frozen commit 3cd62df and scored FULL 17/60, TYPICAL 15/60. Twenty-three FULL runs ended `arrived_short` [52].

**Cause.** We traced the cause on DEV only. After the last DEV closed-loop check, the gyro-bias prior had been widened from 1e-3 to 1e-2 rad/s to pass a unit test. With that prior, one brief stop mid-mission set the bias to a single ±4.5e-3 rad/s noise sample (`docs/BUILD_LOG.md`). That commit scores 6/30 on DEV [52].

**Fix and run 2.** The fix (45ec399) changes only the gyro-bias filter and scores 20/30 on DEV. We then ran EVAL again. Running EVAL a second time after seeing the first result is a protocol deviation. We disclose it here and in [52]. Run 1 is superseded and its files are archived in the repository; all EVAL numbers below are from run 2. We now re-run the DEV closed loop after every change and before freezing a commit for EVAL.

## 8. Results

### 8.1 EVAL closed loop, run 2 (Simulated, tier-0; second EVAL run after a disclosed protocol deviation, section 7.5)

Source: `results/closed_loop_eval.json` and `results/claims.csv` (`closed_loop_eval_tier0_*`). 60 seeds per configuration, on a Linux container with 4 vCPU running 4 episodes in parallel.

| Outcome (of 60) | FULL | TYPICAL |
|---|---|---|
| **Reached B** | **33** | **31** |
| Stopped short of B (arrived_short) | 4 | 4 |
| Stuck | 5 | 1 |
| Collision | 7 | 5 |
| Water entry | 9 | 10 |
| Ditch entry | 2 | 2 |
| **Out of bounds** | **0** | **7** |
| Stops not justified by ground truth (count) | 5 | 2 |
| Mean speed over the run | 1.29 m/s | 1.22 m/s |
| Compute per tick, p50 / p95 | 62.5 / 88.7 ms | 56.2 / 86.6 ms |

| Family | FULL reached B | TYPICAL reached B |
|---|---|---|
| F1 trail | 9/10 | 9/10 |
| F2 ditch field | 5/10 | 4/10 |
| F3 crest + ditch | 6/10 | 4/10 |
| F4 sudden obstacle | **2/10** | **5/10** |
| F5 lighting | 10/10 | 9/10 |
| F6 water / mud | 1/10 | 0/10 |
| F2 + F3 combined | 11/20 | 8/20 |

Figure 3 (`deck_assets/final/eval_by_family.png`) and Figure 4 (`deck_assets/final/eval_outcomes.png`) plot these tables.

**Reading.**

- **Success rates are not distinguishable.** Paired by seed, FULL reached B on 8 seeds where TYPICAL did not, and TYPICAL on 6 where FULL did not. An exact McNemar test gives p = 0.79 (†, computed by us from `results/runs_eval_tier0/summary.csv`). For F2 + F3 alone the split is 6 vs 3, p = 0.51 (†, same source). Repeated DEV batches also differ by about ±2 successes per 30, because latency is modelled from measured compute.
- **The out-of-bounds effect is clear in these worlds; other safety events are not better.** TYPICAL left the terrain 7 times, all in F2 and F3; FULL never did. Trenches cross the whole world, so a planner that treats unseen ground as free routes round the unseen end of a trench and off the map edge. Leaving the map is a simulator event; we read it only as a proxy for driving onto unobserved ground. On those 7 seeds FULL reached B twice, stopped short once and was **stuck** four times, so most of the gain became a stop, not a success.
- **Ditch entries are equal: 2 vs 2.** Both of FULL's (seeds 14 and 26) were F3 worlds with a trench hidden behind a crest (section 4.4). TYPICAL's were seeds 7 (F2 ditch field) and 38 (F3 crest + ditch) (registered). FULL's six extra F2 + F3 wins came from TYPICAL failures split evenly across three types (†): 2 out-of-bounds runs, 2 ditch entries and 2 stopped-short runs. By F3 sub-type, FULL reached B on 2 of 5 hidden-trench worlds (TYPICAL 1) and on 4 of 5 crest-only controls (TYPICAL 3) (†).
- **FULL regressed on sudden obstacles: 2/10 vs 5/10.** Both stacks collided on seeds 3, 9, 15 and 27. FULL also collided on 33, 45 and 57, where TYPICAL reached B; TYPICAL collided on 39, where FULL was scored stuck. All 7 of FULL's collisions were F4. We have not isolated the cause. The obstacle enters from outside the ±36° field of view (in the DEV F4 collision we traced, from 54-62° off the heading until impact; Simulated, DEV, †, `docs/BUILD_LOG.md`), and certification covers ground that was seen, not objects that enter it; but TYPICAL has the same field of view. On seeds 33, 45 and 57, FULL's run-mean speed was 1.35-1.49 m/s against TYPICAL's 1.26-1.34 m/s, over runs of different length (†).
- **Water is not handled in tier-0.** Without images there is no water cue: water still returns about 88 % of pixels, against about 96 % for ground (Simulated, DEV, †, `docs/BUILD_LOG.md`). FULL entered water in 9 of 10 F6 worlds.
- **FULL stops more without need:** 5 false stops against 2, and 5 stuck runs against 1. Refusing to drive on uncertified ground produces these stops; reducing them is DEV tuning work (section 9, item 5).
- **Speed.** FULL's mean speed was *higher* than the fixed-1.5 m/s baseline's (1.29 vs 1.22 m/s). The governor did not slow it on average in these worlds.
- **Final distance to B.** The median is 2.0 m for both stacks. This reflects the 2 m success rule, which ends a run at the radius; it says nothing about localisation accuracy.

**Worked examples.**
- **Figure 5** (`deck_assets/final/run_map_pair.png`) shows EVAL seed 38: a crest with a trench behind it (Simulated, registered, `results/example_runs_eval.json`). At 18.2 s, at the crest, FULL's map confirmed the trench 3.2 m ahead; all 44 confirmed cells lie within 1.0 m of the real trench. FULL turned hard for the gap (peak commanded yaw rate 1.20 rad/s), slowing to 0.7 m/s, and reached B at 33.0 s. TYPICAL drove into the trench at 22.3 s. The platform cap, not the governor's range terms, bound 15 of the 16 ticks after confirmation (†), so the slowdown came from the turn. This is one selected seed.
- **Figure 6** (`deck_assets/final/run_map_gallery.png`) shows one hand-picked FULL success per family. It shows breadth, not a success rate.
- **Backup figures** `deck_assets/final/run_map_pair_s007.png` and `run_map_pair_s037.png` (†, not registered) show that detection is noisy. On seed 7, only 9 of FULL's 53 confirmed ditch cells lie on the trench.

### 8.2 DEV closed loop, seeds 100-129 (Simulated, tier-0)

These are the tuning seeds, not a held-out test (`results/closed_loop_dev.json`).

| Outcome (of 30) | FULL | TYPICAL |
|---|---|---|
| **Reached B** | **20** | **16** |
| Out of bounds | 1 | 6 |
| Ditch entry | 0 | 0 |
| Collision | 1 | 2 |
| Water entry | 4 | 5 |
| Stuck | 3 | 0 |
| Stops not justified by ground truth (count) | 7 | 2 |
| Compute per tick, p50 / p95 | 58.5 / 81.4 ms | 51.3 / 71.0 ms |

DEV compute was measured on a shared 4-core laptop, EVAL compute on a 4-vCPU container; do not compare them.

- **Per family** (FULL/TYPICAL): F1 5/5, F2 2/1, F3 3/2, F4 4/3, F5 5/5, F6 1/0 (†, `results/closed_loop_dev.json#tier0.aggregate.*.by_family`).
- **Same pattern as EVAL:** fewer out-of-bounds runs, more stuck runs and more false stops.
- **F4 differs:** better than TYPICAL on DEV (4/5 vs 3/5), worse on EVAL (2/10 vs 5/10). With so few seeds we cannot tell noise from a real regression, and the EVAL result remains undiagnosed.

### 8.3 Visual odometry on real images (Tested)

Camera only, frame to frame, on KITTI odometry training sequences [54]. The metric is the mean relative error over all 100-800 m sub-segments, as defined by the KITTI benchmark [55], computed with a port of kitti-odom-eval [56].

| Sequence | Path | Translational error | Rotational error | Drift, 100 m segments | End-point error |
|---|---|---|---|---|---|
| 05 | 2,206 m | 1.53 % | 0.68°/100 m | 0.99 % | 52.3 m |
| 07 | 695 m | 2.05 % | 1.28°/100 m | 1.29 % | 12.1 m |
| 00 (frames 1101-4540) | 2,913 m | 2.11 % | 0.81°/100 m | 1.44 % | 27.4 m |
| Pooled, 4,525 segments | 5.81 km | **1.87 %** | 0.79°/100 m | — | — |

Sources: `results/kitti_vo_05.json`, `results/kitti_vo_07.json`, `results/kitti_vo_00.json`; pooled `results/kitti_summary.json`.

- **Failures.** VO did not fail on any of the 7,302 frames.
- **Figure 7** (`deck_assets/final/kitti_drift_plain.png`) shows the trajectories against GPS/INS ground truth, aligned at the start pose only.
- **Sequence 00.** Frames 0-1100 of the mirror we could download are a different drive, so we used frames 1101-4540 only (`results/kitti_summary.json#data_issues`).
- **Settings.** All table rows were measured with the previous VO settings (logged in each file; no photometric normalisation, 21 px KLT window). The current onboard settings were re-measured on 07 only: 2.04 % vs 2.05 % (`results/kitti_vo_07_check.json`); 00 and 05 were not re-run.
- **Leaderboard context** (Literature, `results/kitti_summary.json#literature_context`). The KITTI odometry leaderboard [55] lists VISO2-S [57], frame-to-frame stereo VO, at 2.44 %, and ORB-SLAM2 [58], with bundle adjustment and loop closure, at 1.15 %. Those values are on test sequences 11-21, while ours are on training sequences, so the comparison is only indicative.
- **Off-road transfer is unmeasured.** KITTI is a car on roads with a 54 cm baseline, and we have not measured VO on real off-road data. In rendered-stereo simulation, VO alone gives a median end-point drift of 0.62 % over 22 DEV drives totalling 1,071 m (Simulated, `results/vo_sim_dev.json`).

### 8.4 Integrity monitor (offline only)

**Real images** (Tested, `results/integrity_07to05.json`; cross-check model, not the onboard weights, which were trained on 05). Trained on degraded KITTI 07 (1,100 frames) and tested on degraded KITTI 05 (1,499 frames, 1.4 % failures):
- AUROC is **0.957** under the lenient label (no pose, or > 0.1 m or > 1.0° error per frame).
- On silent failures alone it is 0.849, but there are only 6 positives.
- Under a strict failure label (0.05 m / 0.25°, 19 positives) it is 0.73.
- At the q < 0.4 threshold the monitor rejected **no** frame on the 05 test, so gating changed nothing there. The AUROC shows the monitor ranks failures well; it does not show that the threshold is tuned.

**Rendered simulation** (Simulated, `results/integrity_sim.json`; onboard 05-trained weights). These are offline replays of scripted drives along the ground-truth path; q did not steer or gate the vehicle.
- Of 14 synthetically degraded segments, VO failed in 8, and all 8 drove q to CAUTION or below; the 6 without a VO failure were not flagged.
- 0.00 false DEGRADED onsets per 5 min over 14.3 min of nominal driving.
- All 3 simulated lighting events that caused a VO failure drove q to CAUTION (2 of 3 to DEGRADED).
- AUROC 0.996 on the F5 lighting family (DEV seeds never used for fitting).

### 8.5 Depth odometry (Simulated, DEV, †)

We used 30 recorded tier-0 DEV drives: 1,443 m in total, with 3-8 % wheel slip, driven open loop along the ground-truth path. We replayed the onboard localiser on each drive twice, once without and once with depth odometry. Nothing else changed between the two replays.

| Median over 30 drives | Wheel + gyro | + depth odometry |
|---|---|---|
| Along-track scale error | 6.3 % | 1.47 % |
| Final position error | 2.98 m | 0.83 m |

- **Result.** All 30 drives improved. Depth odometry took 9.4 ms per frame (median of per-drive medians) and accepted 93.5 % of frames. Figure 8 (`deck_assets/final/depth_odom.png`) shows each drive.
- **Reproduction.** Command: `python -m metagross.eval.odometry_dev analyse --tune 100 … 129` on `results/raw/odometry_dev/*.npz`. Output: `deck_assets/final/odometry_dev.json`.
- **Caveats.** The localiser was tuned on these same drives, so this is not a held-out test. It is tier-0 only, and not yet registered.

### 8.6 Terrain segmentation (Tested; built, not in the loop)

All three models were evaluated on real RUGD-5L images [59], [60] (733 test, 1,924 val). None is the deploy model, and none ran in the closed loop [61].

| Model | Test mIoU | Val mIoU | False-safe test / val | Latency (2 threads) | Source |
|---|---|---|---|---|---|
| SegFormer-B0 zero-shot, ADE20K → 5 classes [62], [63], [64] | 0.544 | 0.543 | 2.6 % / 16.9 % | 223 ms | `results/seg_zeroshot.json` |
| LR-ASPP SMOKE (300 CPU iterations, 800 images) | 0.633 | 0.551 | 4.3 % / 27.3 % | 31.1 ms | `results/seg_smoke.json` |
| LR-ASPP CPU-trained on RUGD-5L (interim; partial run, best checkpoint at iteration 2,500 of 2,750) | 0.728 | 0.577 | 2.7 % / 20.5 % | 63.5 ms | `results/seg_cpu.json` |

"False-safe" is the share of obstacle or water pixels predicted as traversable. The validation rates are high. Water is the interim model's weak class: IoU 0.41 on test (water is only 0.06 % of test pixels) and 0.045 on validation, where 88 % of water pixels were predicted traversable.

**Training recipe:** an ImageNet-initialised backbone [65]; cross-entropy with ENet class weights [66], plus 0.5 × soft Dice (after [67]); AdamW [68] and repeat-factor sampling [69]; photometric augmentation; all in PyTorch [70] and TorchVision [71].

The deploy run will train on OFFROAD5 [72], which we understand to combine RUGD [59], RELLIS-3D [73] and GOOSE [74] relabelled to 5 classes (the dataset card does not document its composition), with an optional DINOv2 [75] / FiT3D [76] teacher. It will run on the team laptop's RTX 3050 GPU (Proposed; not started).

### 8.7 Compute

Tier-0 EVAL, FULL stack, all 10,454 ticks of the 60 runs, on a Linux container with 4 vCPU, no GPU and 4 episodes in parallel. Figure 9 (`deck_assets/final/compute_budget.png`).

| Module | Median / p95 (ms) |
|---|---|
| Perception (ground, hazards, ditches) | 33.0 / 48.2 |
| Localisation (EKF + depth odometry) | 11.4 / 17.7 |
| Global planner | 5.9 / 13.2 |
| MPPI, 512 rollouts | 5.4 / 8.1 |
| Costmap | 2.6 / 4.0 |
| Rolling map | 2.3 / 3.5 |
| Governor, supervisor, gate + mixer | < 1 each |
| **Sensing to wheel command** | **62.5 / 88.7** (registered) |

- **Sources.** The per-module values are † (`results/runs_eval_tier0/FULL/*/autonomy/timings.csv`, via `deck_assets/final/compute_budget.json`) except the MPPI median (5.4 ms, ledger `closed_loop_eval_tier0_FULL_mppi_p50_ms`); the sensing-to-command row is registered.
- **Headroom.** At p95, 111 ms of the 200 ms tick is free. One tick of 10,454 went over, at 206 ms.
- **Stereo-mode stages, measured separately.** Tier-0 runs no SGBM, VO or segmenter. We measured them separately in stereo mode on a loaded, shared laptop (Simulated, rendered frames): perception with SGBM and the segmenter at one-third rate, 81.0 / 107.7 ms (`results/perception_dev.json`); localiser update with VO, 58.4 / 124.0 ms (`results/vo_sim_dev.json`).
- **Localiser alone.** Given disparity, the localiser runs at 31.5 Hz on KITTI frames (Tested, `results/localizer_timing.json`).
- **Caveats.** Different machines and loads, so do not add these to the tier-0 numbers. No full stereo + VO tick and nothing on a Jetson has been timed.

### 8.8 Telemetry link (Simulated)

- **Registered replay.** `results/link_budget.json#groups.holdout_verify_tier0` covers 4,437 packets from 60 DEV logs that were not used to tune the codec. The median packet fell from 993 B (codec v1) to 421 B (codec v2), with a maximum of 480 B and 0 % over the 600 B budget.
- **EVAL runs.** Their own 3,505 packets had a median of 423 B and a maximum of 480 B (†, Figure 10, `deck_assets/final/link_budget.png`). No packet dropped the costmap: 18 % carried it at 0.25 m and 82 % at 0.5 m. Packets came one every 0.6 s median (registered), not every 0.5 s.
- **Emulated link.** At 9.6 kbit/s with 20 % loss and 0.4 s latency, codec v2 delivered 3,578 of the 4,437 held-out DEV packets (p95 latency about 0.79 s, median over runs). With codec v1, packets queued and the p95 latency reached about 6.1 s (`results/link_budget.json#groups.holdout_verify_tier0`, not in the ledger).
- **Not tested.** The emulator and console are replay only and not in the closed loop; no radio has been tested.

## 9. Limitations and what we are doing next

1. **The success rate is not shown to improve.** 33/60 against 31/60 is within noise. Resolving a difference this small needs more held-out worlds. We will pre-register a second, larger EVAL set the same way, before any run (Proposed).
2. **Segmenter and water.** The deploy model (LR-ASPP on OFFROAD5 [72], clean and robust augmentation) is not trained yet. Training on the team laptop's GPU (RTX 3050 Laptop, 4 GB) with `scripts/train_seg_gpu.ps1` is the next step (Proposed). An earlier AWS route was dropped; `aws/` is kept for reference only. Until deploy weights exist, WATER never appears in the loop, and F6 fails in tier-0 by construction. When they arrive, we will report test metrics with the same false-safe measure and then wire the model into the stereo loop. Fine-tuning on Indian scenes such as IDD [77] (unstructured roads, not off-road) is Proposed.
3. **Obstacles entering from the side: an undiagnosed regression** (F4 2/10 vs 5/10; section 8.1). TYPICAL has the same field of view, so that alone does not explain the gap. We will first diagnose this on DEV, then test (all Proposed): expiring certification near occluders; capping speed near occluders by how far an object could move into the corridor within `T_r`; and a wider field of view or a second camera pair.
4. **Trench hidden behind a crest: a certification gap.** Certification assumes the ground continues past the last observed row, so a trench just behind a crest is seen only at about 2.8 m. Both FULL ditch entries on EVAL (seeds 14 and 26, both F3) were this case, and 25 hazard cells remain certified inside the stopping envelope on tier-0 DEV. We will make the governor treat the crest top as the end of certified ground on falling terrain, and measure the effect on F3 DEV worlds (Proposed).
5. **Collisions, stuck runs and needless stops.** FULL had 7 collisions against 5 (all F4), 5 stuck runs against 1 and 5 unjustified stops against 2 on EVAL; three F2 worlds ended stuck despite the dead-end memory. This is DEV tuning work.
6. **Rendered-stereo closed loop.** Every closed-loop number here is tier-0, with no VO, integrity monitor or segmenter in the loop. Next, we run the DEV and then EVAL batches in stereo mode on the team laptop's GPU. Software WebGL, at about 3 s per frame, is too slow for batches.
7. **Live console and link policy.** The console only replays logs. We will stream telemetry from a running episode and send operator commands back. We also need a link-loss policy and an independent hardware E-STOP (Proposed). A ROS 2 bridge [78] is Proposed; the repository has no ROS 2 code.
8. **Hardware.** Nothing has run on a vehicle, a real camera or a Jetson. Our field-test plan (`docs/QA.md` §29) has five stages, each with its pass criterion written before the test: record-and-replay of sensor streams through the unchanged stack; ditch detection range on dug trenches, measured against section 4.2; drift measured against RTK-GNSS, logged as ground truth only; closed-loop driving at 0.5 m/s with a safety operator; and degraded conditions. Visual teach-and-repeat [79] is a candidate for fixed patrol routes, and a TartanAir [80] VO stress test is planned (both Proposed).
9. **Not modelled.** Night, rain, lens droplets, deformable ground and tall passable grass. The vehicle model has no suspension or tyre forces. Night-time ditch detection in the literature relies on thermal imaging [27], [10].
10. **Licences.** Our runtime dependencies (NumPy [81], SciPy [82], OpenCV [15], scikit-image [45], scikit-learn [41], ONNX Runtime [21]) are BSD, MIT or Apache-2.0. KITTI, RUGD-5L, OFFROAD5 and RELLIS-3D are non-commercial, so weights trained on them are for research and evaluation only. A product model would be retrained on data BEL owns or licenses. The SegFormer-B0 baseline [64] is non-commercial and is never deployed.

## 10. Reproducibility

Run from the repository root [2] with Python 3.10 or 3.11. Component status is in `docs/PIPELINE_STATUS.md` [83]. The full command table is in `docs/REPRODUCE.md`, and `docs/RESULTS.md` is generated from the ledger.

```bash
pip install -e ".[dev]"
export OMP_NUM_THREADS=2
python -m pytest -q          # on 1 Oct 2026: 504 passed, 10 skipped (the skipped tests need Chrome for the renderer)

# Scenarios; compare the sha256 fields of data/scenarios/manifest.json with results/scenario_manifest.json
python scripts/gen_scenarios.py --split all --workers 4

# Closed loop, FULL vs TYPICAL, tier-0 (EVAL run 2 used OMP_NUM_THREADS=1 and 4 workers)
echo '[{"name":"FULL"},{"name":"TYPICAL","unknown_is_free":true,"use_negobs":false,"use_governor":false,"fixed_speed_mps":1.5}]' > runs/configs.json
python -m metagross.sim.batch --split eval --workers 4 --configs runs/configs.json --out runs/eval_tier0
python -m metagross.sim.closed_loop_summary --split eval --tier0 runs/eval_tier0 --out runs/closed_loop_eval.json
python -m metagross.sim.batch --split dev  --workers 4 --configs runs/configs.json --out runs/dev_tier0

# KITTI VO and integrity monitor (KITTI odometry data under data/kitti)
python scripts/download_kitti.py
python -m metagross.eval.kitti_vo --seqs 07 05 --threads 2
python -m metagross.eval.kitti_vo --seqs 00 --threads 2 --first-frame 1101
python -m metagross.eval.integrity_train collect --seq 05 --max-frames 1500
python -m metagross.eval.integrity_train collect --seq 07
python -m metagross.eval.integrity_train fit --train 07 --test 05 --tag 07to05

# Depth odometry on the recorded DEV drives (results/raw/odometry_dev/*.npz)
python -m metagross.eval.odometry_dev analyse --tune $(seq 100 129) --out runs/odometry_dev.json

# Analytic model, claims ledger, results page
python -m metagross.eval.theory
python -m metagross.eval.claims && python -m metagross.eval.results_md
```

Accuracy numbers are deterministic given the same data, code and seeds. Closed-loop outcomes can move by about ±2 per 30 seeds between batches, because measured compute latency enters the simulation; timings depend on the machine and its load.
- **Figures.** The figures in `deck_assets/final/` are drawn by the scripts in `deck_assets/final/src/`. Each has a JSON sidecar that lists its sources.

## 11. References

[1] Smart India Hackathon 2026, "Problem Statement SIH26126: Vision Based Autonomous Navigation for Unmanned Ground Vehicle for Outdoor environment," Bharat Electronics Limited (BEL), theme: Smart Automation, category: Software, 2026. [Online]. Available: [https://sih.gov.in/](https://sih.gov.in/) (accessed Sep. 30, 2026).

[2] Team CODETEYMONS, "METAGROSS: Seen-ground autonomy for GNSS-denied UGV navigation," source code repository (public), 2026. [Online]. Available: [https://github.com/shahanxd/metagross](https://github.com/shahanxd/metagross)

[3] METAGROSS, "Claims ledger," results/claims.csv (366 claims on 1 Oct 2026: 89 Tested, 271 Simulated, 5 Estimated, 1 Proposed), in [2], 2026. [Online]. Available: [https://github.com/shahanxd/metagross/blob/main/results/claims.csv](https://github.com/shahanxd/metagross/blob/main/results/claims.csv)

[4] Directorate General of Civil Aviation, Govt. of India, "ANSS Advisory Circular AC 1 of 2023: GNSS Interference in Airspace," Ref. DGCA-21040/1/2023-ANS, New Delhi, India, Nov. 24, 2023. [Online]. Available: [https://images.assettype.com/thefourthonline/2023-11/e7cf9f10-2b3b-44ed-ae92-bbf6223711de/DGCA_CIRCULAR.pdf](https://images.assettype.com/thefourthonline/2023-11/e7cf9f10-2b3b-44ed-ae92-bbf6223711de/DGCA_CIRCULAR.pdf) (copy hosted by The Fourth, not the official DGCA site; accessed Sep. 30, 2026).

[5] Ministry of Civil Aviation (M. Mohol, Minister of State), written reply in Lok Sabha on GPS interference and spoofing in the Amritsar and Jammu border region, Mar. 20, 2025. Reported in: PTI, "Several airlines operating aircraft in, around Amritsar report GPS interference: Govt," ETV Bharat, Mar. 20, 2025. [Online]. Available: [https://www.etvbharat.com/en/!bharat/several-airlines-operating-aircraft-in-around-amritsar-report-gps-interference-govt-enn25032004486](https://www.etvbharat.com/en/!bharat/several-airlines-operating-aircraft-in-around-amritsar-report-gps-interference-govt-enn25032004486)

[6] Ministry of Civil Aviation (M. Mohol, Minister of State), written reply in Lok Sabha on GPS interference reports, Mar. 12, 2026: 2,354 reports from Nov. 2023 to Dec. 2025 and 623 around Delhi airspace in Jan.–Feb. 2026. Reported in: "Airlines report 623 incidents of GPS spoofing in Delhi airspace during January–February," Zee News, Mar. 12, 2026. [Online]. Available: [https://zeenews.india.com/mobility/airlines-report-623-incidents-of-gps-spoofing-in-delhi-airspace-during-january-february-3026222.html](https://zeenews.india.com/mobility/airlines-report-623-incidents-of-gps-spoofing-in-delhi-airspace-during-january-february-3026222.html)

[7] All India Radio News, "Civil Aviation Minister informs Rajya Sabha of GPS spoofing near IGI Airport; DGCA issues SOP," newsonair.gov.in, Dec. 1, 2025. [Online]. Available: [https://www.newsonair.gov.in/civil-aviation-minister-informs-rajya-sabha-of-gps-spoofing-near-igi-airport-dgca-issues-sop](https://www.newsonair.gov.in/civil-aviation-minister-informs-rajya-sabha-of-gps-spoofing-near-igi-airport-dgca-issues-sop)

[8] T. Cozzens, "India's IRNSS-1F satellite fails after atomic clock malfunction," GPS World, Mar. 16, 2026 (quoting ISRO). [Online]. Available: [https://www.gpsworld.com/indias-irnss-1f-satellite-fails-after-atomic-clock-malfunction/](https://www.gpsworld.com/indias-irnss-1f-satellite-fails-after-atomic-clock-malfunction/)

[9] Bharat Electronics Ltd., "Robotic Surveillance Platform," product page. [Online]. Available: [https://bel-india.in/product/robotic-surveillance-platform/](https://bel-india.in/product/robotic-surveillance-platform/) (accessed Sep. 30, 2026).

[10] A. Rankin, A. Huertas, L. Matthies, M. Bajracharya, C. Assad, S. Brennan, P. Bellutta, and G. W. Sherwin, "Unmanned ground vehicle perception using thermal infrared cameras," in Proc. SPIE 8045, Unmanned Systems Technology XIII, 2011, Art. no. 804503, doi: [10.1117/12.884349](https://doi.org/10.1117/12.884349).

[11] Stereolabs Inc., "ZED 2i stereo camera," product page. [Online]. Available: [https://www.stereolabs.com/store/products/zed-2i](https://www.stereolabs.com/store/products/zed-2i) (accessed Sep. 30, 2026).

[12] Luxonis, "OAK-D Lite," hardware documentation. [Online]. Available: [https://docs.luxonis.com/hardware/products/OAK-D%20Lite](https://docs.luxonis.com/hardware/products/OAK-D%20Lite) (accessed Sep. 30, 2026).

[13] NVIDIA Corp., "Jetson Orin Nano Super Developer Kit," product page. [Online]. Available: [https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/) (accessed Sep. 30, 2026).

[14] K. Zuiderveld, "Contrast limited adaptive histogram equalization," in Graphics Gems IV, P. S. Heckbert, Ed. San Diego, CA, USA: Academic Press, 1994, pp. 474–485. [Online]. Available: [https://dl.acm.org/doi/10.5555/180895.180940](https://dl.acm.org/doi/10.5555/180895.180940)

[15] G. Bradski, "The OpenCV Library," Dr. Dobb's J. Softw. Tools, vol. 25, no. 11, pp. 120–125, Nov. 2000. (Used: opencv-python-headless 5.0.0.93, Apache-2.0.) [Online]. Available: [https://opencv.org/](https://opencv.org/)

[16] H. Hirschmüller, "Stereo processing by semiglobal matching and mutual information," IEEE Trans. Pattern Anal. Mach. Intell., vol. 30, no. 2, pp. 328–341, Feb. 2008, doi: [10.1109/TPAMI.2007.1166](https://doi.org/10.1109/TPAMI.2007.1166).

[17] R. Labayrade, D. Aubert, and J.-P. Tarel, "Real time obstacle detection in stereovision on non flat road geometry through 'v-disparity' representation," in Proc. IEEE Intell. Vehicle Symp. (IV), Versailles, France, 2002, vol. 2, pp. 646–651, doi: [10.1109/IVS.2002.1188024](https://doi.org/10.1109/IVS.2002.1188024).

[18] M. A. Fischler and R. C. Bolles, "Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography," Commun. ACM, vol. 24, no. 6, pp. 381–395, Jun. 1981, doi: [10.1145/358669.358692](https://doi.org/10.1145/358669.358692).

[19] K. Zhang, S.-C. Chen, D. Whitman, M.-L. Shyu, J. Yan, and C. Zhang, "A progressive morphological filter for removing nonground measurements from airborne LIDAR data," IEEE Trans. Geosci. Remote Sens., vol. 41, no. 4, pp. 872–882, Apr. 2003, doi: [10.1109/TGRS.2003.810682](https://doi.org/10.1109/TGRS.2003.810682).

[20] A. Howard et al., "Searching for MobileNetV3," in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Seoul, South Korea, 2019, pp. 1314–1324, doi: [10.1109/ICCV.2019.00140](https://doi.org/10.1109/ICCV.2019.00140).

[21] ONNX Runtime developers, "ONNX Runtime," 2021. [Online]. Available: [https://onnxruntime.ai/](https://onnxruntime.ai/) (used: v1.23.2, MIT).

[22] T. Guan, D. Kothandaraman, R. Chandra, A. J. Sathyamoorthy, K. Weerakoon, and D. Manocha, "GA-Nav: Efficient terrain segmentation for robot navigation in unstructured outdoor environments," IEEE Robot. Autom. Lett., vol. 7, no. 3, pp. 8138–8145, Jul. 2022, doi: [10.1109/LRA.2022.3187278](https://doi.org/10.1109/LRA.2022.3187278).

[23] J. Frey, M. Mattamala, N. Chebrolu, C. Cadena, M. Fallon, and M. Hutter, "Fast traversability estimation for wild visual navigation," in Proc. Robot.: Sci. Syst. (RSS XIX), Daegu, South Korea, 2023, doi: [10.15607/RSS.2023.XIX.054](https://doi.org/10.15607/RSS.2023.XIX.054).

[24] X. Meng et al., "TerrainNet: Visual modeling of complex terrain for high-speed, off-road navigation," in Proc. Robot.: Sci. Syst. (RSS XIX), Daegu, South Korea, 2023, doi: [10.15607/RSS.2023.XIX.103](https://doi.org/10.15607/RSS.2023.XIX.103).

[25] S. Triest, M. Sivaprakasam, S. Aich, D. Fan, W. Wang, and S. Scherer, "Velociraptor: Leveraging visual foundation models for label-free, risk-aware off-road navigation," in Proc. 8th Conf. Robot Learn. (CoRL), PMLR, vol. 270, 2025, pp. 4483–4494. [Online]. Available: [https://proceedings.mlr.press/v270/triest25a.html](https://proceedings.mlr.press/v270/triest25a.html)

[26] L. Matthies and A. Rankin, "Negative obstacle detection by thermal signature," in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), Las Vegas, NV, USA, 2003, vol. 1, pp. 906–913, doi: [10.1109/IROS.2003.1250744](https://doi.org/10.1109/IROS.2003.1250744).

[27] A. L. Rankin, A. Huertas, and L. H. Matthies, "Night-time negative obstacle detection for off-road autonomous navigation," in Proc. SPIE 6561, Unmanned Systems Technology IX, 2007, Art. no. 656103, doi: [10.1117/12.720513](https://doi.org/10.1117/12.720513).

[28] J. Shi and C. Tomasi, "Good features to track," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Seattle, WA, USA, 1994, pp. 593–600, doi: [10.1109/CVPR.1994.323794](https://doi.org/10.1109/CVPR.1994.323794).

[29] B. D. Lucas and T. Kanade, "An iterative image registration technique with an application to stereo vision," in Proc. 7th Int. Joint Conf. Artif. Intell. (IJCAI), Vancouver, BC, Canada, 1981, vol. 2, pp. 674–679. [Online]. Available: [https://www.ijcai.org/Proceedings/81-2/Papers/017.pdf](https://www.ijcai.org/Proceedings/81-2/Papers/017.pdf)

[30] J.-Y. Bouguet, "Pyramidal implementation of the Lucas Kanade feature tracker: Description of the algorithm," Intel Corp., Microprocessor Research Labs, Tech. Rep., 2000. [Online]. Available: [http://robots.stanford.edu/cs223b04/algo_tracking.pdf](http://robots.stanford.edu/cs223b04/algo_tracking.pdf)

[31] Z. Kalal, K. Mikolajczyk, and J. Matas, "Forward-backward error: Automatic detection of tracking failures," in Proc. 20th Int. Conf. Pattern Recognit. (ICPR), Istanbul, Turkey, 2010, pp. 2756–2759, doi: [10.1109/ICPR.2010.675](https://doi.org/10.1109/ICPR.2010.675).

[32] T. Ke and S. I. Roumeliotis, "An efficient algebraic solution to the perspective-three-point problem," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Honolulu, HI, USA, 2017, pp. 4618–4626, doi: [10.1109/CVPR.2017.491](https://doi.org/10.1109/CVPR.2017.491).

[33] M. Maimone, Y. Cheng, and L. Matthies, "Two years of visual odometry on the Mars Exploration Rovers," J. Field Robot., vol. 24, no. 3, pp. 169–186, Mar. 2007, doi: [10.1002/rob.20184](https://doi.org/10.1002/rob.20184).

[34] C. Campos, R. Elvira, J. J. Gómez Rodríguez, J. M. M. Montiel, and J. D. Tardós, "ORB-SLAM3: An accurate open-source library for visual, visual–inertial, and multimap SLAM," IEEE Trans. Robot., vol. 37, no. 6, pp. 1874–1890, Dec. 2021, doi: [10.1109/TRO.2021.3075644](https://doi.org/10.1109/TRO.2021.3075644).

[35] F. Schmidt, C. Blessing, M. Enzweiler, and A. Valada, "Visual-inertial SLAM for unstructured outdoor environments: Benchmarking the benefits and computational costs of loop closing," J. Field Robot., vol. 42, no. 7, pp. 3726–3747, 2025, doi: [10.1002/rob.22581](https://doi.org/10.1002/rob.22581).

[36] M. Jaimez and J. Gonzalez-Jimenez, "Fast visual odometry for 3-D range sensors," IEEE Trans. Robot., vol. 31, no. 4, pp. 809–822, Aug. 2015, doi: [10.1109/TRO.2015.2428512](https://doi.org/10.1109/TRO.2015.2428512).

[37] S. Thrun, W. Burgard, and D. Fox, Probabilistic Robotics. Cambridge, MA, USA: MIT Press, 2005, ISBN 978-0-262-20162-9. [Online]. Available: [https://mitpress.mit.edu/9780262201629/probabilistic-robotics/](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/)

[38] A. Mandow, J. L. Martínez, J. Morales, J. L. Blanco, A. García-Cerezo, and J. González, "Experimental kinematics for wheeled skid-steer mobile robots," in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), San Diego, CA, USA, 2007, pp. 1222–1227, doi: [10.1109/IROS.2007.4399139](https://doi.org/10.1109/IROS.2007.4399139).

[39] R. G. Brown, "A baseline GPS RAIM scheme and a note on the equivalence of three RAIM methods," NAVIGATION, J. Inst. Navig., vol. 39, no. 3, pp. 301–316, 1992, doi: [10.1002/j.2161-4296.1992.tb02278.x](https://doi.org/10.1002/j.2161-4296.1992.tb02278.x).

[40] K. He, J. Sun, and X. Tang, "Single image haze removal using dark channel prior," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Miami, FL, USA, 2009, pp. 1956–1963, doi: [10.1109/CVPR.2009.5206515](https://doi.org/10.1109/CVPR.2009.5206515).

[41] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011. [Online]. Available: [https://www.jmlr.org/papers/v12/pedregosa11a.html](https://www.jmlr.org/papers/v12/pedregosa11a.html)

[42] H. Koschmieder, "Theorie der horizontalen Sichtweite," Beiträge zur Physik der freien Atmosphäre, vol. 12, pp. 33–53, 1924.

[43] J. C. Platt, "Probabilities for SV machines," in Advances in Large-Margin Classifiers, A. J. Smola, P. L. Bartlett, B. Schölkopf, and D. Schuurmans, Eds. Cambridge, MA, USA: MIT Press, 2000, pp. 61–74, doi: [10.7551/mitpress/1113.003.0008](https://doi.org/10.7551/mitpress/1113.003.0008). (Widely cited under its preprint title, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," 1999.)

[44] E. W. Dijkstra, "A note on two problems in connexion with graphs," Numer. Math., vol. 1, pp. 269–271, 1959, doi: [10.1007/BF01386390](https://doi.org/10.1007/BF01386390).

[45] S. van der Walt et al., "scikit-image: Image processing in Python," PeerJ, vol. 2, Art. no. e453, 2014, doi: [10.7717/peerj.453](https://doi.org/10.7717/peerj.453).

[46] G. Williams, P. Drews, B. Goldfain, J. M. Rehg, and E. A. Theodorou, "Aggressive driving with model predictive path integral control," in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), Stockholm, Sweden, 2016, pp. 1433–1440, doi: [10.1109/ICRA.2016.7487277](https://doi.org/10.1109/ICRA.2016.7487277).

[47] A. Savitzky and M. J. E. Golay, "Smoothing and differentiation of data by simplified least squares procedures," Anal. Chem., vol. 36, no. 8, pp. 1627–1639, 1964, doi: [10.1021/ac60214a047](https://doi.org/10.1021/ac60214a047).

[48] G. Williams, B. Goldfain, P. Drews, K. Saigol, J. M. Rehg, and E. A. Theodorou, "Robust sampling based model predictive control with sparse objective information," in Proc. Robot.: Sci. Syst. (RSS XIV), Pittsburgh, PA, USA, 2018, doi: [10.15607/RSS.2018.XIV.042](https://doi.org/10.15607/RSS.2018.XIV.042).

[49] E. Trevisan and J. Alonso-Mora, "Biased-MPPI: Informing sampling-based model predictive control by fusing ancillary controllers," IEEE Robot. Autom. Lett., vol. 9, no. 6, pp. 5871–5878, Jun. 2024, doi: [10.1109/LRA.2024.3397083](https://doi.org/10.1109/LRA.2024.3397083).

[50] three.js authors, "three.js: JavaScript 3D library," release r186, 2026, MIT licence. [Online]. Available: [https://github.com/mrdoob/three.js/releases/tag/r186](https://github.com/mrdoob/three.js/releases/tag/r186)

[51] Microsoft, "Playwright for Python," GitHub repository, Apache-2.0. [Online]. Available: [https://github.com/microsoft/playwright-python](https://github.com/microsoft/playwright-python) (version not pinned; see pyproject.toml).

[52] METAGROSS, "EVAL pre-registration and record," docs/EVAL_PREREGISTRATION.md, in [2], 2026 (manifest committed in c14e23f before any EVAL run; run 1 at 3cd62df: FULL 17/60, TYPICAL 15/60, superseded; run 2 at stack commit 45ec399: FULL 33/60, TYPICAL 31/60; protocol deviation disclosed). [Online]. Available: [https://github.com/shahanxd/metagross/blob/main/docs/EVAL_PREREGISTRATION.md](https://github.com/shahanxd/metagross/blob/main/docs/EVAL_PREREGISTRATION.md)

[53] METAGROSS, "Scenario manifest (60 EVAL seeds 0–59, 30 DEV seeds 100–129, SHA-256 per scenario)," results/scenario_manifest.json, in [2], 2026. [Online]. Available: [https://github.com/shahanxd/metagross/blob/main/results/scenario_manifest.json](https://github.com/shahanxd/metagross/blob/main/results/scenario_manifest.json)

[54] A. Geiger, P. Lenz, and R. Urtasun, "Are we ready for autonomous driving? The KITTI vision benchmark suite," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Providence, RI, USA, 2012, pp. 3354–3361, doi: [10.1109/CVPR.2012.6248074](https://doi.org/10.1109/CVPR.2012.6248074).

[55] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, "KITTI Visual Odometry / SLAM Evaluation 2012," The KITTI Vision Benchmark Suite, licensed CC BY-NC-SA 3.0. [Online]. Available: [https://www.cvlibs.net/datasets/kitti/eval_odometry.php](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (accessed Sep. 30, 2026).

[56] H. Zhan, "kitti-odom-eval: KITTI odometry evaluation toolbox," GitHub repository, MIT licence. [Online]. Available: [https://github.com/Huangying-Zhan/kitti-odom-eval](https://github.com/Huangying-Zhan/kitti-odom-eval)

[57] A. Geiger, J. Ziegler, and C. Stiller, "StereoScan: Dense 3D reconstruction in real-time," in Proc. IEEE Intell. Vehicles Symp. (IV), Baden-Baden, Germany, 2011, pp. 963–968, doi: [10.1109/IVS.2011.5940405](https://doi.org/10.1109/IVS.2011.5940405).

[58] R. Mur-Artal and J. D. Tardós, "ORB-SLAM2: An open-source SLAM system for monocular, stereo, and RGB-D cameras," IEEE Trans. Robot., vol. 33, no. 5, pp. 1255–1262, Oct. 2017, doi: [10.1109/TRO.2017.2705103](https://doi.org/10.1109/TRO.2017.2705103).

[59] M. Wigness, S. Eum, J. G. Rogers, D. Han, and H. Kwon, "A RUGD dataset for autonomous navigation and visual perception in unstructured outdoor environments," in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), Macau, China, 2019, pp. 5000–5007, doi: [10.1109/IROS40897.2019.8968283](https://doi.org/10.1109/IROS40897.2019.8968283).

[60] GAIA-URJC, "RUGD-5Labels-resized," Hugging Face dataset, 2025, HF licence tag CC BY-NC-SA 3.0 (the uploader's tag; a relabelled mirror of RUGD [59]). [Online]. Available: [https://huggingface.co/datasets/GAIA-URJC/RUGD-5Labels-resized](https://huggingface.co/datasets/GAIA-URJC/RUGD-5Labels-resized)

[61] METAGROSS, "Model card: METAGROSS terrain segmenter," docs/MODEL_CARD.md, in [2], 2026. [Online]. Available: [https://github.com/shahanxd/metagross/blob/main/docs/MODEL_CARD.md](https://github.com/shahanxd/metagross/blob/main/docs/MODEL_CARD.md)

[62] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, "SegFormer: Simple and efficient design for semantic segmentation with transformers," in Advances in Neural Information Processing Systems (NeurIPS), vol. 34, 2021, pp. 12077–12090. [Online]. Available: [https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html)

[63] B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, and A. Torralba, "Scene parsing through ADE20K dataset," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Honolulu, HI, USA, 2017, pp. 5122–5130, doi: [10.1109/CVPR.2017.544](https://doi.org/10.1109/CVPR.2017.544).

[64] NVIDIA (weights; model card by Hugging Face), "nvidia/segformer-b0-finetuned-ade-512-512," Hugging Face model, licence: other (NVIDIA Source Code License for SegFormer, non-commercial). [Online]. Available: [https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) (accessed Sep. 30, 2026). ONNX export used: Xenova/segformer-b0-finetuned-ade-512-512.

[65] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Miami, FL, USA, 2009, pp. 248–255, doi: [10.1109/CVPR.2009.5206848](https://doi.org/10.1109/CVPR.2009.5206848).

[66] A. Paszke, A. Chaurasia, S. Kim, and E. Culurciello, "ENet: A deep neural network architecture for real-time semantic segmentation," arXiv:1606.02147, 2016. [Online]. Available: [https://arxiv.org/abs/1606.02147](https://arxiv.org/abs/1606.02147)

[67] F. Milletari, N. Navab, and S.-A. Ahmadi, "V-Net: Fully convolutional neural networks for volumetric medical image segmentation," in Proc. 4th Int. Conf. 3D Vis. (3DV), Stanford, CA, USA, 2016, pp. 565–571, doi: [10.1109/3DV.2016.79](https://doi.org/10.1109/3DV.2016.79).

[68] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in Proc. Int. Conf. Learn. Represent. (ICLR), New Orleans, LA, USA, 2019. [Online]. Available: [https://openreview.net/forum?id=Bkg6RiCqY7](https://openreview.net/forum?id=Bkg6RiCqY7)

[69] A. Gupta, P. Dollár, and R. Girshick, "LVIS: A dataset for large vocabulary instance segmentation," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Long Beach, CA, USA, 2019, pp. 5351–5359, doi: [10.1109/CVPR.2019.00550](https://doi.org/10.1109/CVPR.2019.00550). Repeat-factor sampling: extended version, arXiv:1908.03195, App. B.2.

[70] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019, pp. 8024–8035. [Online]. Available: [https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library](https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library)

[71] TorchVision maintainers and contributors, "TorchVision: PyTorch's computer vision library," GitHub repository, 2016. [Online]. Available: [https://github.com/pytorch/vision](https://github.com/pytorch/vision) (used: v0.29.0, BSD-3-Clause).

[72] GAIA-URJC, "OFFROAD5," Hugging Face dataset (composition undocumented on the card; per the uploader's related sets, RUGD, RELLIS-3D and GOOSE relabelled to 5 classes), 2025, HF licence tag CC BY-NC-SA 3.0 (the uploader's tag; upstream licences still apply). [Online]. Available: [https://huggingface.co/datasets/GAIA-URJC/OFFROAD5](https://huggingface.co/datasets/GAIA-URJC/OFFROAD5)

[73] P. Jiang, P. Osteen, M. Wigness, and S. Saripalli, "RELLIS-3D dataset: Data, benchmarks and analysis," in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), Xi'an, China, 2021, pp. 1110–1116, doi: [10.1109/ICRA48506.2021.9561251](https://doi.org/10.1109/ICRA48506.2021.9561251).

[74] P. Mortimer, R. Hagmanns, M. Granero, T. Luettel, J. Petereit, and H.-J. Wuensche, "The GOOSE dataset for perception in unstructured environments," in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), Yokohama, Japan, 2024, pp. 14838–14844, doi: [10.1109/ICRA57147.2024.10611298](https://doi.org/10.1109/ICRA57147.2024.10611298).

[75] M. Oquab et al., "DINOv2: Learning robust visual features without supervision," Trans. Mach. Learn. Res. (TMLR), Jan. 2024. [Online]. Available: [https://openreview.net/forum?id=a68SUt6zFt](https://openreview.net/forum?id=a68SUt6zFt)

[76] Y. Yue, A. Das, F. Engelmann, S. Tang, and J. E. Lenssen, "Improving 2D feature representations by 3D-aware fine-tuning," in Computer Vision – ECCV 2024 (Lecture Notes in Computer Science, vol. 15060). Cham, Switzerland: Springer, 2024, pp. 57–74, doi: [10.1007/978-3-031-72627-9_4](https://doi.org/10.1007/978-3-031-72627-9_4).

[77] G. Varma, A. Subramanian, A. Namboodiri, M. Chandraker, and C. V. Jawahar, "IDD: A dataset for exploring problems of autonomous navigation in unconstrained environments," in Proc. IEEE Winter Conf. Appl. Comput. Vis. (WACV), Waikoloa, HI, USA, 2019, pp. 1743–1751, doi: [10.1109/WACV.2019.00190](https://doi.org/10.1109/WACV.2019.00190).

[78] S. Macenski, T. Foote, B. Gerkey, C. Lalancette, and W. Woodall, "Robot Operating System 2: Design, architecture, and uses in the wild," Sci. Robot., vol. 7, no. 66, Art. no. eabm6074, May 2022, doi: [10.1126/scirobotics.abm6074](https://doi.org/10.1126/scirobotics.abm6074).

[79] P. Furgale and T. D. Barfoot, "Visual teach and repeat for long-range rover autonomy," J. Field Robot., vol. 27, no. 5, pp. 534–560, 2010, doi: [10.1002/rob.20342](https://doi.org/10.1002/rob.20342).

[80] W. Wang et al., "TartanAir: A dataset to push the limits of visual SLAM," in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), Las Vegas, NV, USA (virtual), 2020, pp. 4909–4916, doi: [10.1109/IROS45743.2020.9341801](https://doi.org/10.1109/IROS45743.2020.9341801).

[81] C. R. Harris et al., "Array programming with NumPy," Nature, vol. 585, no. 7825, pp. 357–362, Sep. 2020, doi: [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

[82] P. Virtanen et al., "SciPy 1.0: Fundamental algorithms for scientific computing in Python," Nat. Methods, vol. 17, no. 3, pp. 261–272, Mar. 2020, doi: [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).

[83] METAGROSS, "Pipeline status (audited status of every component)," docs/PIPELINE_STATUS.md, with generated results in docs/RESULTS.md, in [2], 2026. [Online]. Available: [https://github.com/shahanxd/metagross/blob/main/docs/PIPELINE_STATUS.md](https://github.com/shahanxd/metagross/blob/main/docs/PIPELINE_STATUS.md)
