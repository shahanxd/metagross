#!/usr/bin/env bash
# Copy trained models, metrics and figures from the GPU box back to this repo.
#
#   bash aws/fetch_results.sh ubuntu@<public-ip> [~/.ssh/metagross.pem] [remote repo dir]
#
# Fetched: ONLY files the GPU pipeline produces - models/lraspp_offroad5_{clean,robust}.onnx
# + .json sidecars, results/seg_lraspp_offroad5_*.json, results/seg_teacher*.json,
# deck_assets/seg_lraspp_offroad5_*.png, runs/seg/*/{metrics.json,train.log} (NOT the .pt
# checkpoints; add them with FETCH_CKPT=1). The remote copy of the repo also carries the
# laptop's results/ and models/ from upload time (seg_smoke.json, seg_cpu.json,
# lraspp_offroad5_cpu.onnx ...); broad globs would overwrite newer local files with those
# stale copies, so they are deliberately not fetched.
# Afterwards measure latency on the target CPU:
#   python -m metagross.eval.seg_eval --model models/lraspp_offroad5_robust.onnx \
#     --data-root data/rugd5 --splits val --out results/seg_lraspp_offroad5_robust_laptop.json
set -euo pipefail
HOST="${1:?usage: fetch_results.sh user@host [key.pem] [remote_dir]}"
KEY="${2:-$HOME/.ssh/id_rsa}"
REMOTE="${3:-~/metagross}"
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new)
RUNS=(lraspp_offroad5_clean lraspp_offroad5_robust)

mkdir -p "$LOCAL/models" "$LOCAL/results" "$LOCAL/deck_assets" "$LOCAL/runs/seg"
for run in "${RUNS[@]}"; do
  scp "${SSH_OPTS[@]}" "$HOST:$REMOTE/models/$run.onnx" "$HOST:$REMOTE/models/$run.json" "$LOCAL/models/" \
    || echo "no exported model for $run yet"
done
scp "${SSH_OPTS[@]}" "$HOST:$REMOTE/results/seg_lraspp_offroad5_*.json" "$LOCAL/results/" || echo "no evaluation JSONs yet"
scp "${SSH_OPTS[@]}" "$HOST:$REMOTE/results/seg_teacher*.json" "$LOCAL/results/" 2>/dev/null || true
scp "${SSH_OPTS[@]}" "$HOST:$REMOTE/deck_assets/seg_lraspp_offroad5_*.png" "$LOCAL/deck_assets/" 2>/dev/null || true
for run in "${RUNS[@]}"; do
  mkdir -p "$LOCAL/runs/seg/$run"
  scp "${SSH_OPTS[@]}" "$HOST:$REMOTE/runs/seg/$run/metrics.json" "$HOST:$REMOTE/runs/seg/$run/train.log" "$LOCAL/runs/seg/$run/" || true
  if [ "${FETCH_CKPT:-0}" = "1" ]; then
    scp "${SSH_OPTS[@]}" "$HOST:$REMOTE/runs/seg/$run/best.pt" "$LOCAL/runs/seg/$run/" || true
  fi
done
mkdir -p "$LOCAL/runs/seg/aws_logs"
scp "${SSH_OPTS[@]}" "$HOST:$REMOTE/logs/*.log" "$HOST:$REMOTE/logs/nvidia-smi.txt" "$LOCAL/runs/seg/aws_logs/" 2>/dev/null || true
echo "fetched into $LOCAL (models/, results/, deck_assets/, runs/seg/)"
