#!/usr/bin/env bash
# METAGROSS terrain-segmentation training on a fresh Ubuntu 22.04 / 24.04 NVIDIA GPU box.
#
#   bash aws/setup_and_train.sh            # setup + data + launch both runs (detached)
#   SKIP_DATA=1 bash aws/setup_and_train.sh
#   EPOCHS=2 TRAIN_SUBSET=512 bash aws/setup_and_train.sh   # quick dry run of the whole chain
#
# Steps: nvidia-smi check -> python venv -> torch CUDA wheels (index chosen from the GPU
# architecture and driver) -> pip deps -> CUDA sanity (conv fwd/bwd on the GPU) -> unit
# tests -> download OFFROAD5 + RUGD-5L (huggingface_hub, resumable) -> data layout check
# -> launch aws/run_pipeline.sh under nohup (two parallel runs: clean vs robust
# augmentation, then ONNX export + evaluation JSONs). Safe to re-run: every step is
# idempotent and training resumes from runs/seg/<name>/last.pt.
#
# Env overrides: TORCH_INDEX (skip auto-selection), PYTHON, VENV, DATA_DIR, SKIP_DATA=1,
# STRICT_TESTS=1 (abort on a failing unit test; default: warn and continue), SYSTEM_TORCH=1
# (venv with --system-site-packages; reuse the interpreter's torch/torchvision when they
# see the GPU, as in the AWS PyTorch containers SageMaker runs), plus the run_pipeline.sh
# tunables (EPOCHS, BS, IMG, LR, WORKERS, TRAIN_SUBSET, SAMPLING, DEADLINE).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
# The repo is edited on Windows: strip CR line endings from the scripts this one calls
# (this file itself must already be LF - see aws/README.md step 2).
sed -i 's/\r$//' aws/run_pipeline.sh aws/fetch_results.sh

VENV="${VENV:-$REPO_DIR/.venv-aws}"
DATA_DIR="${DATA_DIR:-$REPO_DIR/data}"
PY="${PYTHON:-python3}"
SYSTEM_TORCH="${SYSTEM_TORCH:-0}"
SUDO="sudo"; [ "$(id -u)" = "0" ] && SUDO=""   # containers run as root, often without sudo
VENV_ARGS=(); [ "$SYSTEM_TORCH" = "1" ] && VENV_ARGS=(--system-site-packages)
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR" "$DATA_DIR" models results deck_assets runs/seg

say() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# ------------------------------------------------------------------ 1. GPU check
say "GPU check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found: use an NVIDIA GPU AMI (e.g. Deep Learning Base OSS Nvidia Driver, Ubuntu 22.04)" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi > "$LOG_DIR/nvidia-smi.txt"

# ------------------------------------------------------------------ 2. torch wheel index
# cu124 wheels (torch <= 2.6) have no Blackwell kernels (sm_100 / sm_120, e.g. the 96 GB
# RTX PRO 6000 of AWS G7e) and no aarch64 builds (GH200). Pick the index from the GPU
# compute capability, the driver's CUDA version and the CPU architecture.
DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)"
[[ "$COMPUTE_CAP" =~ ^[0-9]+\.[0-9]+$ ]] || COMPUTE_CAP=""   # old drivers lack the compute_cap field
[[ "$DRIVER_CUDA" =~ ^[0-9]+\.[0-9]+$ ]] || DRIVER_CUDA=""
ARCH="$(uname -m)"
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }   # ver_ge A B  <=>  A >= B
if [ -z "${TORCH_INDEX:-}" ]; then
  CC_MAJOR="${COMPUTE_CAP%%.*}"
  if [ "$ARCH" = "aarch64" ] || { [ -n "$CC_MAJOR" ] && [ "$CC_MAJOR" -ge 10 ] 2>/dev/null; }; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    if [ -n "$DRIVER_CUDA" ] && ! ver_ge "$DRIVER_CUDA" "12.8"; then
      echo "GPU ${COMPUTE_CAP:-?} / $ARCH needs cu128 wheels but the driver supports CUDA $DRIVER_CUDA (< 12.8): update the NVIDIA driver (>= 570)" >&2
      exit 1
    fi
  elif [ -n "$DRIVER_CUDA" ] && ver_ge "$DRIVER_CUDA" "12.6"; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu126"
  elif [ -n "$DRIVER_CUDA" ] && ! ver_ge "$DRIVER_CUDA" "12.0"; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"   # CUDA 11 driver
  else
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"   # runs on any CUDA 12 driver (>= 525)
  fi
fi
say "compute capability ${COMPUTE_CAP:-unknown}, driver CUDA ${DRIVER_CUDA:-unknown}, $ARCH -> torch index $TORCH_INDEX"

# ------------------------------------------------------------------ 3. python venv
# Note: `python3 -m venv --help` succeeds on Ubuntu even without python3-venv (ensurepip
# missing), so try to create the venv and install the package only if that fails.
say "Python venv at $VENV"
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  [ -f "$VENV/pyvenv.cfg" ] && rm -rf "$VENV"   # half-created venv from a failed earlier attempt
  if ! "$PY" -m venv ${VENV_ARGS[@]+"${VENV_ARGS[@]}"} "$VENV" 2>"$LOG_DIR/venv.err"; then
    cat "$LOG_DIR/venv.err" >&2
    PYVER="$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    $SUDO apt-get update -y
    $SUDO apt-get install -y python3-venv python3-pip "python${PYVER}-venv" || $SUDO apt-get install -y python3-venv python3-pip
    [ -f "$VENV/pyvenv.cfg" ] && rm -rf "$VENV"
    "$PY" -m venv ${VENV_ARGS[@]+"${VENV_ARGS[@]}"} "$VENV"
  fi
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null

# ------------------------------------------------------------------ 4. torch + deps
if [ "$SYSTEM_TORCH" = "1" ] && python -c "import torch, torchvision; assert torch.cuda.is_available()" 2>/dev/null; then
  say "SYSTEM_TORCH=1: using the interpreter's torch $(python -c 'import torch; print(torch.__version__)'); installing dependencies"
else
  say "Installing torch (CUDA wheels from $TORCH_INDEX) and dependencies"
  # With --system-site-packages pip would count the interpreter's (unusable) torch as installed: force the venv copy.
  TORCH_PIP_ARGS=(); [ "$SYSTEM_TORCH" = "1" ] && TORCH_PIP_ARGS=(--ignore-installed)
  python -m pip install ${TORCH_PIP_ARGS[@]+"${TORCH_PIP_ARGS[@]}"} torch torchvision --index-url "$TORCH_INDEX"
fi
python -m pip install -r aws/requirements-gpu.txt

# ------------------------------------------------------------------ 5. CUDA sanity
say "CUDA sanity"
python - <<'EOF'
import sys
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU"
p = torch.cuda.get_device_properties(0)
cap = torch.cuda.get_device_capability(0)
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | {p.name} | sm_{cap[0]}{cap[1]} | "
      f"{p.total_memory / 2**30:.1f} GiB | wheel archs {torch.cuda.get_arch_list()}")
amp = torch.bfloat16 if cap >= (8, 0) else torch.float16  # same rule as train_seg.amp_dtype_for_cuda
try:  # conv + depthwise conv (as in MobileNetV3) fwd/bwd under the training autocast dtype, cudnn.benchmark on
    torch.backends.cudnn.benchmark = True
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.BatchNorm2d(16), torch.nn.Hardswish(),
                              torch.nn.Conv2d(16, 16, 5, padding=2, groups=16), torch.nn.Conv2d(16, 16, 1)).cuda()
    x = torch.randn(4, 3, 64, 64, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=amp):
        y = net(x).float().mean()
    y.backward()
    torch.cuda.synchronize()
    print(f"conv fwd/bwd ok under {amp}", float(y.detach()))
except RuntimeError as exc:
    sys.exit(f"CUDA kernels failed on this GPU ({exc}); set TORCH_INDEX to a newer wheel index, e.g. .../whl/cu128")
EOF

# ------------------------------------------------------------------ 6. unit tests
say "Unit tests (segmentation pipeline)"
if ! OMP_NUM_THREADS=4 python -m pytest -q tests/test_seg_data.py tests/test_seg_metrics.py tests/test_seg_semantics.py \
  tests/test_seg_train.py tests/test_seg_teacher.py tests/test_train_sampling.py tests/test_train_seg_cpu_report.py; then
  if [ "${STRICT_TESTS:-0}" = "1" ]; then
    echo "unit tests failed (STRICT_TESTS=1): aborting" >&2
    exit 1
  fi
  echo "WARNING: unit tests failed; continuing (set STRICT_TESTS=1 to abort). Check the output above." >&2
fi

# ------------------------------------------------------------------ 7. data
if [ "${SKIP_DATA:-0}" != "1" ]; then
  say "Downloading OFFROAD5 (~8.5 GB) and RUGD-5L (~2.1 GB) into $DATA_DIR"
  export HF_HUB_DISABLE_PROGRESS_BARS=1
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
  python scripts/download_seg_data.py --dataset offroad5 --splits val test train --root "$DATA_DIR" 2>&1 | tee "$LOG_DIR/download_offroad5.log"
  python scripts/download_seg_data.py --dataset rugd5 --splits val test train --root "$DATA_DIR" 2>&1 | tee "$LOG_DIR/download_rugd5.log"
  python scripts/download_seg_data.py --models-only 2>&1 | tee -a "$LOG_DIR/download_rugd5.log"
fi
df -h "$DATA_DIR" | tail -1

say "Data layout check"
DATA_DIR="$DATA_DIR" python - <<'EOF'
import collections
import os
import sys
from pathlib import Path

from metagross.train.data_index import list_samples

root = Path(os.environ["DATA_DIR"])
bad = []
for ds in ("offroad5", "rugd5"):
    for split in ("train", "val", "test"):
        try:
            s = list_samples(root / ds, split)
        except FileNotFoundError as exc:
            bad.append(f"{ds}/{split}: {exc}")
            continue
        print(f"{ds:9s} {split:5s} {len(s):6d} samples  by source {dict(collections.Counter(x.source for x in s))}")
        if not s:
            bad.append(f"{ds}/{split}: 0 samples")
if bad:
    sys.exit("data layout problem:\n  " + "\n  ".join(bad))
EOF

# ------------------------------------------------------------------ 8. launch
say "Launching training pipeline (detached). Follow with: tail -f $LOG_DIR/pipeline.log"
export VENV DATA_DIR
nohup bash aws/run_pipeline.sh > "$LOG_DIR/pipeline.log" 2>&1 &
echo $! > "$LOG_DIR/pipeline.pid"
say "pipeline PID $(cat "$LOG_DIR/pipeline.pid"). Logs: $LOG_DIR/train_*.log, results in results/ and models/"
