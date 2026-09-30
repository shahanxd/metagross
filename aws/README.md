# Terrain-segmenter training on an AWS GPU box

One command sets up a fresh Ubuntu 22.04/24.04 NVIDIA instance, downloads the data and trains two LR-ASPP models
in parallel: CLEAN vs ROBUST augmentation. It then exports ONNX and writes evaluation JSONs and figures.

**Fastest path (no SSH, no console clicking):** with `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and
`AWS_DEFAULT_REGION` in the environment, run `python aws/ec2_run.py check`, then
`python aws/ec2_run.py launch --job seg`. The instance uploads logs to S3 every 2 min, uploads the outputs, and
terminates itself. Use `status`, `fetch` and `cleanup` to follow it, collect the results and remove everything. The
manual SSH route below still works.

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
