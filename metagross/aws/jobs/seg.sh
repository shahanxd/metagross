#!/usr/bin/env bash
# Job "seg" for aws/ec2_run.py: set up the GPU box, download OFFROAD5 + RUGD-5L, train the two LR-ASPP
# terrain segmenters (clean / robust augmentation), export ONNX, evaluate. Blocks until the pipeline ends.
# Tunables are passed through: EPOCHS, BS, IMG, LR, WORKERS, TRAIN_SUBSET, SAMPLING, DEADLINE (see aws/README.md).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
nvidia-smi > logs/nvidia-smi.txt 2>&1 || true
bash aws/setup_and_train.sh || { echo "setup_and_train.sh failed rc=$?"; exit 1; }
PID="$(cat logs/pipeline.pid)"
echo "waiting for pipeline PID $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
grep -q "pipeline done" logs/pipeline.log && echo "pipeline done" || { echo "pipeline ended without 'pipeline done'"; exit 2; }
