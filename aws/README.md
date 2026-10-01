# Terrain-segmenter training on an AWS GPU box

> **In use again (2026-10-01).** The full segmenter training runs as a SageMaker training job on `ml.g6.12xlarge`
> (section below). The laptop-GPU script `scripts/train_seg_gpu.ps1` is the fallback.

One command sets up a fresh Ubuntu 22.04/24.04 NVIDIA instance, downloads the data and trains two LR-ASPP models
in parallel: CLEAN vs ROBUST augmentation. It then exports ONNX and writes evaluation JSONs and figures.

**Fastest path (no SSH, no console clicking):** with `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and
`AWS_DEFAULT_REGION` in the environment, run `python aws/ec2_run.py check`, then
`python aws/ec2_run.py launch --job seg`. The instance uploads logs to S3 every 2 min, uploads the outputs, and
terminates itself. Use `status`, `fetch` and `cleanup` to follow it, collect the results and remove everything. The
manual SSH route below still works. `pip install -e ".[aws]"` installs boto3.

## SageMaker training job (`--backend sagemaker`)

The GPU quota is a SageMaker one: **`ml.g6.12xlarge` for training job usage** (4x NVIDIA L4 24 GB, 48 vCPU), the
launcher's default. Set `AWS_DEFAULT_REGION` to the region where that quota was granted; quotas are per region and per
use (a *notebook instance* or *processing* quota does not count). The 2-epoch dry run on 2026-09-30 used
`ml.g6.16xlarge` (1x L4) in another account and completed every stage. `--backend sagemaker` runs the same
`aws/jobs/seg.sh` as a SageMaker training job in the AWS PyTorch GPU container
(`pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker`; `--image` and `--type` override).
`check --backend sagemaker` also prints the account's Free/Paid plan and remaining credits: on the Free plan the
account stops, rather than bills, when the credits run out, so check they cover the run first.

```bash
python aws/ec2_run.py check --backend sagemaker            # identity, SageMaker API reachable?, training quota
# 1) dry run of the whole chain on the real instance (2 epochs on 512 images; hard cap 1.5 h)
python aws/ec2_run.py launch --backend sagemaker --job seg --max-hours 1.5 --env EPOCHS=2 TRAIN_SUBSET=512
python aws/ec2_run.py status --backend sagemaker --job seg  # job state + tail of logs/ (copied to S3 every 2 min)
python aws/ec2_run.py fetch  --backend sagemaker --job seg  # models/, results/, deck_assets/, runs/aws/logs/
# 2) the real run (40 epochs by default; training stops at a DEADLINE ~1 h before the cap, then export + eval)
python aws/ec2_run.py launch --backend sagemaker --job seg --max-hours 12
python aws/ec2_run.py terminate --backend sagemaker --job seg   # StopTrainingJob; outputs so far are kept
```

How it runs (`aws/sagemaker_backend.py`, `aws/sagemaker_entry.sh`):

- **Code** goes up as the same `git archive HEAD` tarball (commit first) and arrives as the `code` input channel. Shell
  scripts are archived with LF endings whatever the local `core.autocrlf` (`.gitattributes` pins `*.sh` to LF), and the
  launcher refuses a tarball with CRLF scripts. The container entrypoint unpacks it and runs
  `aws/sagemaker_entry.sh`, which runs `aws/jobs/seg.sh` unchanged with `SYSTEM_TORCH=1` (the container's CUDA torch
  is reused when it sees the GPU, so normally no torch download) and the datasets and venv on the instance's local
  NVMe.
- **GPUs**: on `ml.g6.12xlarge` (4x L4) `run_pipeline.sh` puts the clean run on GPU 0 and the robust run on GPU 1;
  `train_seg` has no multi-GPU mode, so GPUs 2 and 3 stay idle. On a one-GPU type (`--type ml.g6.16xlarge`) both
  runs share the GPU, as on any single-GPU box.
- **Time cap**: `--max-hours` becomes `MaxRuntimeInSeconds`. Unless `DEADLINE` is passed, the entry script sets one
  at max-hours minus a margin (runtime/8, clamped to 15-60 min), so training checkpoints and stops in time for the
  ONNX export and evaluation.
- **Logs**: `logs/` is copied to the job's checkpoint path on S3 every 2 min (`status` shows it). The job output,
  `logs/pipeline.log` and the tagged `logs/train_*.log` also stream to CloudWatch (`/aws/sagemaker/TrainingJobs`),
  where the job's metric definitions are set up to pick per-epoch val mIoU into `clean:val_miou` /
  `robust:val_miou` (the regexes are unit-tested on a sample log line, not yet seen on SageMaker).
- **Outputs**: the `JOB_OUTPUTS` globs are copied to `<checkpoints>/out/` (what `fetch` downloads) and to
  `/opt/ml/model` (`model.tar.gz`, the fallback once the job has ended). `fetch` writes only paths that match those
  globs (both backends), so a job cannot drop, say, `tests/conftest.py` into the local repo. A job that exits 0 without `models/lraspp_offroad5_*.onnx` is marked
  Failed (rc 4), and the failure reason carries the tail of the job log.
- **Resume**: `runs/` lives on the checkpoint path, so `launch ... --resume-from <earlier job name>` continues both
  runs from their `last.pt`. The launcher first checks that the earlier job left a `last.pt` on S3 (an empty channel
  would fail only after the instance is provisioned).
- **AWS resources**: bucket `metagross-<account>-<region>` (shared with EC2), prefix `jobs/seg/sagemaker/<job name>/`;
  execution role `metagross-sagemaker-role` (S3 `jobs/*/sagemaker/*` in that bucket and only while the bucket belongs
  to this account, the job's CloudWatch logs and metrics, pulls from the image's ECR repository only). Because the
  bucket name is predictable, both backends check its owner (`ExpectedBucketOwner`) before using it. `--env` refuses secret-looking names, because training-job environment
  variables are readable by anyone with `sagemaker:DescribeTrainingJob`. `cleanup` stops jobs and deletes the role.
- **Permissions the account needs**, for the calling user and, at run time, the execution role: `sagemaker`
  training-job create/describe/list/stop, S3 create-bucket/put/get/list on the bucket, `iam` create/put/get/pass on
  the role, CloudWatch Logs, ECR pull, and (for `check` only) Service Quotas read. An AWS Organizations service
  control policy that denies any of these blocks the job whatever the IAM policies say; `check --backend sagemaker`
  and `launch` report such a denial explicitly.

Not yet verified on AWS: this backend has only been unit-tested with a fake AWS session plus a local run of the
entry script (see `tests/test_aws_sagemaker.py`). The first launch should be the dry run above.

## 0. Instance

- **Instance and image.** Any NVIDIA GPU instance with the driver preinstalled works, e.g. an AWS "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)". The planned card has 96 GB of VRAM. LR-ASPP at batch 32 and 320x416 needs only a few GB per run, so both runs fit side by side. On this model the limit is CPU data loading, not the GPU. Choose ≥ 16 vCPUs if you can.
- **Disk.** ≥ 60 GB of free EBS: OFFROAD5 zips 8.5 GB plus the extracted copy, and RUGD-5L 2.1 GB plus the extracted copy.
- **Security group.** SSH (22) from your IP only.

## 1. Copy the repo

From the laptop, run one of these (Git Bash, WSL or PowerShell with OpenSSH):

```bash
# a) if the repo is on a git remote
ssh -i ~/.ssh/metagross.pem ubuntu@<ip> 'git clone <remote-url> ~/metagross'
# b) otherwise copy the working tree without venv/data/runs
tar --exclude=.venv --exclude=data --exclude=runs --exclude=.git -czf /tmp/metagross.tgz -C "/d/Downloads/sih again" metagross
scp -i ~/.ssh/metagross.pem /tmp/metagross.tgz ubuntu@<ip>:~ && ssh -i ~/.ssh/metagross.pem ubuntu@<ip> 'tar xzf metagross.tgz'
```

## 2. Run

```bash
ssh -i ~/.ssh/metagross.pem ubuntu@<ip>
cd ~/metagross && sed -i 's/\r$//' aws/*.sh      # Windows CRLF -> LF (harmless if already LF)
bash aws/setup_and_train.sh
tail -f logs/pipeline.log logs/train_lraspp_offroad5_*.log      # training progress (per-epoch val mIoU / false-safe)
```

`setup_and_train.sh` performs these steps:

1. Checks `nvidia-smi`.
2. Picks the torch wheel index from the GPU and driver: **cu128** for Blackwell GPUs (compute capability ≥ 10, e.g. the
   96 GB RTX PRO 6000 of G7e instances; needs driver ≥ 570) and for aarch64 hosts (GH200); **cu126** when the driver
   supports CUDA ≥ 12.6; **cu118** for a CUDA 11 driver; otherwise **cu124**. Set `TORCH_INDEX=...` to override. The
   selection logic was tested with a mocked `nvidia-smi` only; no real GPU host has run it yet. The older script always
   used cu124, whose wheels (torch ≤ 2.6) have no Blackwell kernels, so training would have failed with "no kernel image".
3. Creates `.venv-aws`. If `python3-venv` is missing (stock Ubuntu), it installs it with apt and retries.
4. Installs torch/torchvision and `aws/requirements-gpu.txt`.
5. Runs a CUDA sanity check: a conv forward and backward pass under autocast on the GPU.
6. Runs the segmentation unit tests. A failure prints a warning and the script continues; `STRICT_TESTS=1` aborts instead.
7. Downloads OFFROAD5 + RUGD-5L + the zero-shot SegFormer ONNX with `scripts/download_seg_data.py`. The download is
   resumable. The script then checks the layout, printing sample counts per split and per source
   (rugd / rellis / goose), and stops with a clear message if a split is empty.
8. Starts `aws/run_pipeline.sh` under `nohup`, so closing SSH does not stop training.

`run_pipeline.sh` then runs:

```
train  lraspp_offroad5_clean   (40 ep, bs 32, 320x416, AMP)  ─┐ in parallel
train  lraspp_offroad5_robust  (same, ROBUST augmentation)   ─┘
export models/lraspp_offroad5_{clean,robust}.onnx (+ .json sidecar, ORT-vs-torch check)
eval   results/seg_lraspp_offroad5_{clean,robust}_{offroad5,rugd5}.json + deck_assets/*.png
[opt.] RUN_TEACHER=1: results/seg_teacher_dinov2s.json (+ FIT3D_CKPT=... -> seg_teacher_fit3d_dinov2s.json)
```

Tunables are passed as environment variables, for example
`EPOCHS=40 BS=32 IMG=320x416 LR=6e-4 WORKERS=7 bash aws/setup_and_train.sh`. The table lists the useful ones.

| Command | What it does |
|---|---|
| `EPOCHS=2 TRAIN_SUBSET=512 bash aws/setup_and_train.sh` | 10-minute dry run of the whole chain |
| `SKIP_DATA=1 bash aws/setup_and_train.sh` | Reuses downloaded data |
| `SAMPLING=repeat bash aws/setup_and_train.sh` | Uses image-level repeat-factor sampling of rare classes (water/mud) in both runs. The default is `uniform` |
| `DEADLINE=2026-09-30T13:00 bash aws/setup_and_train.sh` | Wall-clock stop. Both runs validate, checkpoint and stop at that time, then export and evaluation run on `best.pt` |
| `bash aws/run_pipeline.sh` | Re-launches after an interruption. Training resumes from `runs/seg/<name>/last.pt` |
| `RUN_TEACHER=1 FIT3D_CKPT=/path/fit3d.pth bash aws/run_pipeline.sh` | Runs the optional DINOv2-S and FiT3D teacher rows. These need the FiT3D weights from https://github.com/ywyue/FiT3D |

## 3. Fetch results

```bash
bash aws/fetch_results.sh ubuntu@<ip> ~/.ssh/metagross.pem ~/metagross
```

This copies only what the GPU pipeline produces: `models/lraspp_offroad5_{clean,robust}.onnx/.json`,
`results/seg_lraspp_offroad5_*.json`, `results/seg_teacher*.json`, `deck_assets/seg_lraspp_offroad5_*.png`, and each
run's `metrics.json` and `train.log`. Checkpoints are copied only when `FETCH_CKPT=1` is set. The remote tree also holds
the laptop's `results/` and `models/` as they were at upload time, so broad globs such as `results/seg_*.json` would
overwrite newer local files like `seg_cpu.json` and `lraspp_offroad5_cpu.onnx`. The script does not use them.

The GPU box records accuracy only. Latency must be measured on the target CPU, so run this on the laptop:

```bash
python -m metagross.eval.seg_eval --model models/lraspp_offroad5_robust.onnx --data-root data/rugd5 \
    --splits val test --out results/seg_lraspp_offroad5_robust_laptop.json --fig-prefix deck_assets/seg_robust_laptop \
    --label "LR-ASPP robust (OFFROAD5)"
python -m metagross.train.model_card        # refresh docs/MODEL_CARD.md results table
```

The onboard `Segmenter` automatically prefers `models/lraspp_offroad5_robust.onnx`, then `models/lraspp_offroad5_clean.onnx`,
then `models/lraspp_smoke.onnx` (`DEFAULT_MODEL_CANDIDATES`). You can also pin a model with `seg_model_path` in the autonomy config.

## 4. Shut down

Stop or terminate the instance once `logs/pipeline.log` prints `pipeline done` and the results are fetched.
EBS volumes keep costing money until you delete them.

## No GPU yet: the laptop CPU run

`scripts/finish_seg_cpu.ps1 -Launch` starts a detached, resumable CPU run on RUGD-5L only. It uses robust augmentation,
repeat-factor sampling, 256x320 crops, batch 8, 3000 steps in 250-step epochs and a hard 09:25 stop, and writes to
`runs/seg_cpu/`. Running `scripts/finish_seg_cpu.ps1` with no switch then exports `models/lraspp_offroad5_cpu.onnx` and
evaluates it into `results/seg_cpu.json`, with a comparison against the smoke and zero-shot baselines. It works on a partial checkpoint.
The GPU pipeline above is unaffected: without `--epoch-iters` or `--sampling repeat`, `train_seg` keeps its original one-pass-per-epoch loop.

## Licence reminder

The OFFROAD5 and RUGD-5L mirrors are CC BY-NC-SA 3.0, so weights trained here are for research and evaluation.
See `docs/MODEL_CARD.md`.
