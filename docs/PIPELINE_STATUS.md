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
| 4 | **Terrain segmenter (ML)** | **BUILT, NOT IN LOOP**: no trained weights exist, so the node runs without semantics | The train → ONNX export → `Segmenter` chain was verified on CPU with a toy dataset. The deploy model (OFFROAD5, GPU) was never trained. In tier-0 there are no images, so it cannot run there. |
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
| Segmenter mIoU (`seg_smoke`, `seg_zeroshot`, `seg_cpu`) | real RUGD images | Tested, but the models are smoke or zero-shot runs, not the deploy model |
| Ditch visibility range 4.3 m, speed envelope | analytic | Estimated |

## Gaps to close for a fully built end-to-end pipeline
1. **Train the terrain segmenter on a GPU** (`python aws/ec2_run.py launch --backend sagemaker --job seg`, see
   `aws/README.md`). This enables water and obstacle semantics in the loop. Blocked on AWS permissions (next section).
2. **Run the closed loop in stereo mode on the GPU box**, so that SGBM, VO, the integrity monitor and the segmenter
   all run on rendered images. The published numbers are tier-0 only.
3. **Connect the operator console live**, with a telemetry stream from the running episode and operator commands
   back to the autonomy.
4. Make the node fail loudly when the segmenter is missing, and record `impl` in `result.json`.

## Next steps (handoff, 2026-09-30)
- AWS credentials are set in the environment settings (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  `AWS_DEFAULT_REGION`), but they only reach a newly started session. Check them with
  `aws sts get-caller-identity` or `python aws/ec2_run.py check --backend sagemaker`.
- Approved quota, as seen in the Service Quotas console on 2026-10-01 (the team's main AWS account):
  **`ml.g6.16xlarge` for training job usage = 1 in us-east-1** (1x NVIDIA L4 24 GB, 64 vCPU). The same account has
  `ml.g6.24xlarge` for *notebook instance* usage = 1 in us-east-1; training jobs cannot use that one. The earlier
  "`ml.g6.24xlarge` training job, ap-south-1" note was wrong. The launcher now defaults to `ml.g6.16xlarge`.
  - Done: `aws/ec2_run.py --backend sagemaker` runs `aws/jobs/seg.sh` as a training job in the AWS PyTorch GPU
    container, with the code tarball on S3 and outputs to S3 (`aws/sagemaker_backend.py`, `aws/sagemaker_entry.sh`,
    usage in `aws/README.md`). It is unit-tested with a fake AWS session and a local run of the entry script
    (`tests/test_aws_sagemaker.py`); it has **not run on AWS yet**.
  - The EC2 G-instance quota is unchecked. The stereo-mode loop needs a GPU plus a browser, so check the EC2 quota
    first and fall back to the same SageMaker instance type.
- **Blocker found 2026-09-30.** The keys are valid: `sts get-caller-identity` succeeds as IAM user `metagross-bot`,
  which has `AdministratorAccess`. But the account is in an AWS Organization whose service control policy explicitly
  denies the calls this plan needs. Denied in real calls in ap-south-1: SageMaker `ListTrainingJobs`, `ListDomains`
  and `ListNotebookInstances`; S3 `ListAllMyBuckets`; EC2 `DescribeInstances` and `DescribeRegions`; SSM
  `GetParameter` (the GPU AMI); Service Quotas reads. Denied in the IAM policy simulator: `sagemaker:CreateTrainingJob`,
  `s3:CreateBucket`, `s3:PutObject`, `ec2:RunInstances`. IAM and STS calls work. Neither the SageMaker path nor the
  EC2 fallback can start until the organisation's management account allows these actions. IAM changes inside this
  account cannot override an SCP. Nothing was launched and no AWS resources were created. The full list of actions
  needed is in `aws/README.md` ("Permissions the account needs").
- **Resolution (2026-10-01):** the blocked keys belong to a separate, AWS-created "Proof of Concept" Free-plan account
  (created 2026-09-30) inside an organization the team does not administer, so its SCP cannot be changed by us. The
  quota is on the team's main account instead. Next: an IAM user and access key in that account, the environment's
  `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` replaced and `AWS_DEFAULT_REGION=us-east-1`, then a new session.
- Order, once allowed: (1) train the segmenter: `check --backend sagemaker`, then the 1.5 h dry run, then the full run
  (commands in `aws/README.md`); (2) fetch the ONNX files and wire them into the loop; (3) run the stereo-mode DEV/EVAL
  loop on the GPU; (4) connect the live console. After that, the video and slides.
