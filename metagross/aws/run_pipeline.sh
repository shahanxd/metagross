#!/usr/bin/env bash
# Two parallel LR-ASPP runs on OFFROAD5 (clean vs robust augmentation), then ONNX export
# and evaluation. Normally started detached by aws/setup_and_train.sh; can be re-run
# alone (training resumes from last.pt, export/eval are recomputed).
#
# Tunables (env): EPOCHS=40 BS=32 IMG=320x416 LR=6e-4 WORKERS=<nproc/2-1> TRAIN_SUBSET=0
#                 SAMPLING=uniform (repeat = repeat-factor sampling of rare classes, e.g. water)
#                 DEADLINE= (optional local wall-clock stop, 'HH:MM' or 'YYYY-MM-DDTHH:MM'; the runs
#                            validate + checkpoint and stop, then export/eval proceed on best.pt)
#                 RUN_TEACHER=0 (1 = also run aws/teacher_dinov2.py after the main runs)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
VENV="${VENV:-$REPO_DIR/.venv-aws}"
DATA_DIR="${DATA_DIR:-$REPO_DIR/data}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

EPOCHS="${EPOCHS:-40}"
BS="${BS:-32}"
IMG="${IMG:-320x416}"
LR="${LR:-6e-4}"
NPROC="$(nproc)"
WORKERS="${WORKERS:-$(( NPROC / 2 > 2 ? NPROC / 2 - 1 : 2 ))}"
TRAIN_SUBSET="${TRAIN_SUBSET:-0}"
SAMPLING="${SAMPLING:-uniform}"
DEADLINE="${DEADLINE:-}"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

echo "[$(date)] EPOCHS=$EPOCHS BS=$BS IMG=$IMG LR=$LR WORKERS=$WORKERS (per run) TRAIN_SUBSET=$TRAIN_SUBSET SAMPLING=$SAMPLING DEADLINE=${DEADLINE:-none}"

train_run() {  # $1 = aug policy (clean | robust); everything else identical between the two runs
  local aug="$1" name="lraspp_offroad5_$1"
  local extra=()
  [ -n "$DEADLINE" ] && extra+=(--deadline "$DEADLINE")
  OMP_NUM_THREADS=2 python -m metagross.train.train_seg \
    --model lraspp --data offroad5 --data-root "$DATA_DIR/offroad5" --aug "$aug" --init imagenet \
    --epochs "$EPOCHS" --bs "$BS" --lr "$LR" --img "$IMG" --amp --workers "$WORKERS" \
    --train-subset "$TRAIN_SUBSET" --sampling "$SAMPLING" --out "runs/seg/$name" --resume "${extra[@]}" \
    > "$LOG_DIR/train_$name.log" 2>&1
}

train_run clean & PID_CLEAN=$!
train_run robust & PID_ROBUST=$!
echo "[$(date)] training PIDs clean=$PID_CLEAN robust=$PID_ROBUST"
wait $PID_CLEAN; RC_CLEAN=$?
wait $PID_ROBUST; RC_ROBUST=$?
echo "[$(date)] training finished rc clean=$RC_CLEAN robust=$RC_ROBUST"

for aug in clean robust; do
  name="lraspp_offroad5_$aug"
  if [ ! -f "runs/seg/$name/best.pt" ]; then
    echo "no checkpoint for $name, skipping export/eval"; continue
  fi
  python -m metagross.train.export_onnx --ckpt "runs/seg/$name/best.pt" --out "models/$name.onnx" \
    --data-root "$DATA_DIR/offroad5" --name "LR-ASPP MobileNetV3 OFFROAD5 ($aug aug)" || continue
  # Accuracy only on the GPU box (latency is measured on the target CPU after fetch_results.sh).
  python -m metagross.eval.seg_eval --model "models/$name.onnx" --data-root "$DATA_DIR/offroad5" \
    --splits val test --threads 8 --latency-threads --out "results/seg_${name}_offroad5.json" \
    --fig-prefix "deck_assets/seg_${name}_offroad5" --label "LR-ASPP ($aug aug), OFFROAD5" --tag "seg_${aug}_offroad5"
  python -m metagross.eval.seg_eval --model "models/$name.onnx" --data-root "$DATA_DIR/rugd5" \
    --splits val test --threads 8 --latency-threads --out "results/seg_${name}_rugd5.json" \
    --fig-prefix "deck_assets/seg_${name}_rugd5" --label "LR-ASPP ($aug aug), RUGD-5L" --tag "seg_${aug}_rugd5"
done

if [ "${RUN_TEACHER:-0}" = "1" ]; then
  echo "[$(date)] teacher experiment"
  python aws/teacher_dinov2.py --data-root "$DATA_DIR/offroad5" --out results/seg_teacher_dinov2s.json \
    > "$LOG_DIR/teacher_dinov2s.log" 2>&1 || echo "teacher run failed (optional)"
  if [ -n "${FIT3D_CKPT:-}" ]; then
    python aws/teacher_dinov2.py --data-root "$DATA_DIR/offroad5" --fit3d-ckpt "$FIT3D_CKPT" \
      --out results/seg_teacher_fit3d_dinov2s.json > "$LOG_DIR/teacher_fit3d.log" 2>&1 || echo "FiT3D run failed (optional)"
  fi
fi
echo "[$(date)] pipeline done"
