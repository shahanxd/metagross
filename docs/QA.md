# Hard questions and honest answers

Answers cite the results file behind every number, with its honesty label: **Tested** (real data), **Simulated**,
**Estimated** (analytic model), **Proposed**, **Literature**. "Not measured" means exactly that. Closed-loop
simulation results are still being produced and are not quoted here; `docs/BUILD_LOG.md` records the engineering
state and `docs/RESULTS.md` lists every registered number.

## Sensing choices

**1. Why cameras only? Why not LiDAR?**
Three reasons. The problem statement asks for vision-based navigation. A military UGV often needs a passive,
non-emitting sensing mode (Rankin et al., SPIE 2011, in `docs/REFERENCES.md`); a LiDAR emits. A stereo camera is
cheaper and lighter than a 3-D LiDAR for a Scout-Mini-class vehicle. The cost is range on negative obstacles: from a
0.9 m mast a 0.3 m wide ditch is first resolvable at about 4.3 m (`results/theory.json`, Estimated). A LiDAR has the
same grazing-angle problem with ditches, but it works at night. We have not measured a LiDAR baseline.

**2. Why a 12 cm baseline (ZED 2i class) and not a cheaper 7.5 cm camera?**
Stereo depth error grows with range squared and falls with baseline. At 10 m the modelled depth sigma is 0.47 m
with 12 cm and 0.76 m with 7.5 cm (`results/theory.json#stereo`, Estimated, 0.25 px disparity noise). The 7.5 cm
option (OAK-D Lite class) is kept as a low-cost tier but is not the modelled sensor.

**3. How far ahead can you detect a ditch, and how does that depend on its width?**
Analytic first-resolution range from a 0.9 m mast with a 6-pixel criterion: 4.3 m for a 0.3 m ditch, 5.5 m for
0.5 m, 6.9 m for 0.8 m, 7.6 m for 1.0 m (`results/theory.json#ditch_detection`, Estimated). The detector on analytic
synthetic stereo scenes first flags a 0.6 m deep ditch at 4.45 m (0.3 m wide), 5.45 m (0.5 m) and 7.2 m (0.8 m)
camera range without noise (`results/perception_ditch_range.csv`, Simulated). A positive obstacle falls off as
1/R, not 1/R^2: a 0.3 m rock is resolvable at about 22 m (`results/theory.json#positive_detection`, Estimated),
beyond the 12 m range cap we use. Detection range on real ditches: not measured.

**4. Doesn't that short range force you to crawl?**
The speed governor answers this directly: the vehicle must stop inside ground it has certified. With 1.5 m/s^2
braking, 0.6 s reaction time and a 0.5 m margin, the stopping distance from 2 m/s is 3.03 m
(`results/theory.json#stopping`, Estimated), and the safe-speed envelope at our mast height and design ditch width is
2.60 m/s before the 2.0 m/s platform cap (`results/theory.json#envelope`, Estimated). For comparison, BEL's Robotic
Surveillance Platform lists a maximum of 3.6 km/h, 1.0 m/s (Literature, `docs/REFERENCES.md` [BEL-RSP]).

**5. What about dust, night and rain?**
Dust: modelled. Tier-0 adds range-dependent dropout and phantom returns; the renderer adds dust fog; scenario family
F5 schedules dust, glare and dimming events. The integrity monitor has contrast, dark-channel (haze), saturation and
darkness features. Closed-loop behaviour in dust: not measured yet. Night: not modelled. Passive stereo needs light;
the options are an IR illuminator (which emits) or a thermal camera; JPL's night-time ditch detection work used
thermal signatures (Matthies and Rankin 2003; Rankin et al. 2007, in `docs/REFERENCES.md`). Rain and lens droplets: not modelled, not measured.

**6. Can the stereo camera see water?**
Often not geometrically: calm water is flat and mostly gives no stereo match. In Tier-0 water has a 60 % dropout
probability per block (`metagross/sim/sensors.py`). The `WATER` state comes from terrain segmentation, which can only
raise cost, never clear a geometric hazard. So water avoidance depends on the segmenter, which is now trained but
not yet in the closed loop, and water is its weakest class (see question 17).

## Localisation

**7. How accurate is your visual odometry?**
On real KITTI drives, camera only, KITTI protocol (100-800 m segments): translation error 1.53 % on sequence 05,
2.05 % on 07, 2.11 % on 00 (frames 1101-4540), 1.87 % pooled; rotation error 0.79 deg/100 m pooled
(`results/kitti_vo_*.json`, `results/kitti_summary.json`, Tested). Mean end-point drift over 100 m segments is
0.99-1.44 % (same files). Loop closure and bundle adjustment are not used.

**8. KITTI is a car on roads. Why should that transfer to off-road terrain?**
It may not fully. KITTI has a 54 cm baseline, a camera about 1.65 m high, textured urban scenes and smooth motion. We
use it because it is real, public and has ground truth. Off-road VO on real data is not measured. The next real-data
checks are an off-road stereo sequence with ground truth and our own recorded drives (question 29).

**9. Why not ORB-SLAM3?**
Licence and fit. ORB-SLAM3 is GPL-3.0; linking it into a product would impose GPL terms on the combined work, which a
defence supplier is unlikely to accept (`LICENSES/THIRD_PARTY.md`). Its main advantage, loop closure, rarely helps on
a one-way A-to-B mission. Its full SLAM back end also costs compute we need for perception. For context, the KITTI
leaderboard lists ORB-SLAM2 stereo at 1.15 % on the held-out test sequences (Literature, recorded in
`results/kitti_summary.json#literature_context`); our 1.87 % is on training sequences, so the two are not directly
comparable. We have not run ORB-SLAM3 on the same data.

**10. What happens when VO fails?**
An integrity monitor scores every frame. Trained on degraded KITTI 07 and tested on degraded KITTI 05, it separates
failed from good VO frames with AUROC 0.957; for silent failures (VO returned a pose that was wrong) AUROC is 0.849,
but that subset has only 6 positives (`results/integrity_07to05.json`, Tested; real images, synthetic
degradations). When the smoothed score q drops below 0.4 the VO update is rejected and the EKF bridges on wheels and
gyro. The supervisor halves speed below q = 0.7, drops to quarter speed in go-look hops below 0.4, and stops (latched)
below 0.2 for 3 s. Old certified ground only counts while health is nominal. Closed-loop behaviour under VO failure:
in progress.

**11. How big is the position error at the goal after 50 m?**
Not measured in closed loop yet. As an extrapolation only: 1-2 % drift of distance is 0.5-1 m over 50 m, inside the
2 m success radius. Separately, the operator's launch-heading error rotates the goal: `mission.json` records
`goal_sigma_m = 0.5 m + |heading error in rad| x range` (`metagross/sim/runner.py`); a 1 deg error at 50 m adds about
0.87 m of lateral goal error (arithmetic, not a measurement).

**12. The gyro is z-only. What about pitch and roll?**
The contract carries a z-gyro and wheel encoders only; pitch and roll are not sensed, and the ground model absorbs
camera pitch through the v-disparity profile each frame. A full IMU would help VO on rough ground and allow tip-over
monitoring. Adding it is a documented, additive contract change (Proposed).

## Perception and hazards

**13. How do you avoid false ditches, for example on rolling ground or at crests?**
Candidates are not lethal at first. A `DITCH_CANDIDATE` must be flagged in 2 of 3 in-view frames to become lethal;
until then it is only uncertified and costed as suspicious: the speed governor's certified range ends at it, so
the vehicle slows as it approaches, but it does not trigger the lethal-cell brake. Crests are a separate
state (`CREST_SHADOW`, never lethal). Family F3 includes crest-only control scenarios to measure false stops, which
the referee counts (`false_stops` in `result.json`). False candidates on rolling terrain are a known open issue in the
DEV runs and the current engineering focus (`docs/BUILD_LOG.md`).

**14. What if the segmenter is wrong and calls water or a rock "ground"?**
Fusion is monotone: semantics can raise cost and mark water, but never clears a geometric obstacle or ditch. The
safety metric is the false-safe rate, the share of hazard pixels labelled traversable. Zero-shot SegFormer-B0:
2.6 % on RUGD-5L test, 16.9 % on validation. The CPU smoke model: 4.3 % test, 27.3 % validation
(`results/seg_zeroshot.json`, `results/seg_smoke.json`, Tested). The GPU-trained deploy model (clean aug): 2.3 % on
RUGD-5L test, 19.5 % on validation (`results/seg_lraspp_offroad5_clean_rugd5.json`, Tested).

**15. How do you handle people or vehicles that move into the path?**
A cell that becomes occupied where ground was seen recently becomes `DYNAMIC`: lethal, inflated by an extra 0.5 m,
held for at least 2 s. The final safety gate simulates the command for 1 s and brakes if the footprint would enter a
lethal cell. There is no motion prediction or tracking. Scenario family F4 tests a walker, box or boulder that
emerges from behind a bush 4-6 m ahead. Closed-loop results: in progress.

**16. Slopes, tip-over and getting stuck in mud?**
Slopes above 20 deg are lethal in perception; the referee declares tip-over at 25 deg roll or 30 deg pitch. Slip is
estimated from VO versus wheel travel; a sustained high slip ratio while driving latches SAFE_STOP (IMMOBILISED).
Mud and water are avoided through semantics. The simulator's vehicle is kinematic, so real sinkage and traction loss
are not modelled (`docs/SIMULATION.md`, section 6).

## Planning, control and safety

**17. Is the terrain model actually trained?**
Yes, offline. The deploy model (LR-ASPP MobileNetV3 on OFFROAD5 = RUGD + RELLIS-3D, clean and robust augmentation)
was trained on AWS SageMaker on 2026-10-01 (4x T4, 40 epochs, 1.05 billable hours). Test mIoU, clean aug: 0.825 on
OFFROAD5 (n=2405) and 0.768 on RUGD-5L (n=733), against 0.544 for zero-shot SegFormer-B0 on the same 733 images
(`results/seg_lraspp_offroad5_*.json`, Tested). Water stays weak (RUGD-5L val water IoU 0.02). It is not yet wired into
the closed loop and its CPU latency is not yet measured. The ONNX weights are git-ignored (see `docs/PIPELINE_STATUS.md`).

**18. Why MPPI and not DWA, TEB or a lattice planner?**
MPPI takes any cost function, so the certification term (probe points at fractions of each rollout's stopping
distance must lie on certified ground), lethal cells and skid-steer wheel limits go straight into one cost. It is a
batch of array operations, which suits numpy now and a GPU later. We always include a full-stop sample, so braking
is always a candidate. A cost-to-go field from the global planner provides the long-range guidance MPPI lacks.
Per-tick MPPI time in closed loop is logged in each run's `autonomy/timings.csv`. We have not benchmarked DWA or TEB.

**19. What happens if there is no certified route, for example the vehicle is boxed in?**
It does not drive into unknown ground. After 3 s without 0.3 m of progress the supervisor enters STOP_AND_LOOK and
rotates in place (+45, -45, back) to certify adjacent ground. If that still gives no progress it latches SAFE_STOP
and reports the reason to the operator, who can RESUME or send a new goal. Being too cautious (stopping or crawling)
is the expected failure mode of this design, and it is what the DEV runs are being tuned against.

**20. How do you know the stack is not secretly using simulator ground truth?**
Six layers (`docs/ARCHITECTURE.md`, section 2): an import scan test, a separate OS process, a pinned `SensorFrame`
field set, a runtime file guard that blocks any `open` outside an allow-list (so `gt/` and `data/scenarios/` are
unreadable), pre-registered seeds with hashed scenarios, and evaluation only after the run.

**21. How do you avoid tuning to your own simulator?**
Seeds are split before development: DEV 100-129 for tuning, EVAL 0-59 held out, fixed in the frozen contract.
Scenarios are generated from the seed and hashed; every result records the hash. The perception, VO and integrity
numbers that matter most are measured on real images, not in our simulator. This split is self-policed; to make it
independently checkable we propose to hand the EVAL scenario hashes to the evaluator before the final run.

**22. Why 5 Hz? Is the compute fast enough?**
The batch loop runs at 5 Hz, a 200 ms budget. On the development laptop the perception stage takes 123 ms median
at 640 x 400, of which SGBM is 59 ms (`results/perception_timing.json`); the localiser runs at 31.5 Hz given the shared
disparity, measured on 640-px-wide KITTI crops (`results/localizer_timing.json`, Tested); the smoke segmenter takes 31.1 ms median at 2 threads and runs
on every third frame (`results/seg_smoke.json`, Tested). Compute latency is measured every tick and enters the speed
governor's reaction time, so a slower computer lowers the speed cap instead of breaking the stopping guarantee.

**23. Will it run on a Jetson?**
Not measured; nothing has run on a Jetson. The target is a Jetson Orin Nano-class module (Proposed). The plan is to
move stereo matching to the GPU or the vision accelerator and the segmenter to TensorRT, and to keep the rest on the
CPU cores. The zero-shot SegFormer-B0 at 223 ms median on 2 laptop threads (`results/seg_zeroshot.json`, Tested)
shows why the deploy model is the smaller LR-ASPP.

## Operator link

**24. What does the operator see, and why no video?**
A 2 Hz packet: pose, drive mode and reason text, speed cap, certified range, speed, the next 5 waypoints, a 64 x 64
4-bit costmap (16 m square around the vehicle) and health values. Video does not fit a 9.6 kbit/s link. The packet is
designed to fit: a unit test requires every test packet to use under 75 % of the per-packet budget
(9.6 kbit/s / 8 / 2 Hz = 600 bytes) (`tests/test_plan_link.py::test_packet_size_fits_link_budget`). Packets carry a
CRC-32 and are rejected if corrupted.

**25. What happens on link loss?**
The autonomy does not depend on the link: it continues the mission on its own sensors and stops only for its own
safety reasons. There is no link-loss policy in the supervisor today (for example, hold after N seconds without an
operator heartbeat), and E-STOP travels over the same data link. For a fielded system we propose a configurable
link-loss behaviour and an independent hardware E-STOP radio (Proposed). The link emulator models 20 % loss and
0.4 s latency, but it has not been placed in the closed loop yet, and no real radio has been tested.

**26. Is the console connected to the running vehicle?**
Not yet. The console replays `telemetry.jsonl` from a run (drag and drop, or a URL) and emits operator commands to a
browser hook. The supervisor already handles GO, HOLD, RESUME, ESTOP and a new goal, and the runner sends GO at the
start of each episode. A live console-to-runner bridge is not built.

## Evidence, licences and next steps

**27. Which numbers in your deck are real?**
Only those labelled Tested come from real data: KITTI VO, the integrity monitor on degraded KITTI, segmentation on
RUGD, and laptop timings. Simulated numbers come from our simulator, Estimated numbers from closed-form models.
`docs/RESULTS.md` lists every registered number with its label and source file.

**28. What licences constrain a BEL product?**
Our runtime dependencies are permissive (BSD, MIT, Apache-2.0; see `LICENSES/THIRD_PARTY.md`). GPL/AGPL runtimes such
as ORB-SLAM3 and Ultralytics YOLO are deliberately not used. The training and evaluation datasets are the constraint:
KITTI, RUGD-5L, OFFROAD5 and RELLIS-3D are non-commercial (CC BY-NC-SA 3.0), so weights trained on them are for
research and evaluation. A product model would be retrained on data BEL owns or licenses. The SegFormer-B0
zero-shot baseline is under NVIDIA's non-commercial licence and is never deployed. The repository itself has no
licence file yet.

**29. How would you field-test this?**
Proposed, in stages, each with a pass criterion written before the test:
1. **Record and replay.** Mount the camera, encoders and gyro on the vehicle, record `SensorFrame` streams while
   driving manually, and replay them through the unchanged stack offline. The contract is the same as in simulation.
2. **Static perception.** Dig trenches of known width and depth; measure first-detection range against the analytic
   curve (question 3), at several times of day.
3. **Localisation.** Drive A-B loops with RTK-GNSS logged as ground truth only (never fed to the stack); measure
   drift per 100 m and goal error.
4. **Closed loop at low speed.** Speed cap 0.5 m/s, a safety operator with a hardware E-STOP, the same referee metrics
   as in simulation (success, collisions, ditch entries, false stops, SPL).
5. **Degraded conditions.** Low sun, dust, dusk, wet ground, link loss.

**30. What is not done yet?**
The trained deploy segmenter; closed-loop results on DEV and EVAL seeds; any hardware run; Jetson timing; a live
operator link and link-loss policy; night and rain; a project licence file. These are tracked in
`docs/BUILD_LOG.md` and marked Proposed in the README status table.
