#!/usr/bin/env bash
# Container entry point of a SageMaker training job launched by `python aws/ec2_run.py launch --backend sagemaker`.
#
# The job's ContainerEntrypoint (aws/sagemaker_backend.py: bootstrap_command) unpacks the code channel into
# /opt/ml/code/metagross and execs this script, which:
#   1. keeps runs/ on the checkpoint path (SageMaker syncs it to S3) and, when the optional "resume" channel is
#      present, restores an earlier job's runs/ so train_seg --resume continues from last.pt;
#   2. puts DATA_DIR and VENV on the instance ML volume, sets SYSTEM_TORCH=1 (use the container's CUDA torch) and,
#      unless DEADLINE is given, a DEADLINE that leaves time for ONNX export + evaluation before the hard cap
#      (MaxRuntimeInSeconds, passed as MG_MAX_RUNTIME_S);
#   3. runs aws/jobs/$MG_JOB.sh unchanged (its output also goes to logs/job.log), streams logs/pipeline.log and
#      logs/train_*.log (tagged per file) to stdout = CloudWatch, and copies logs/ to the checkpoint path every
#      MG_LOG_SYNC_S seconds;
#   4. copies the job outputs ($MG_OUTPUT_GLOBS, repo-relative globs) to <checkpoints>/out/ and /opt/ml/model/,
#      fails the job (rc 4) if $MG_REQUIRED_OUTPUTS matches nothing, writes /opt/ml/output/failure on error
#      (SageMaker's FailureReason) and exits with the job's return code. SIGTERM (StopTrainingJob or the hard cap;
#      SageMaker allows 120 s) saves logs and outputs the same way.
# MG_ML_ROOT overrides /opt/ml and MG_FOLLOW_LOGS=0 disables the log streaming (unit tests).
set -uo pipefail

ML_ROOT="${MG_ML_ROOT:-/opt/ml}"
JOB="${MG_JOB:?MG_JOB is not set}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$ML_ROOT/checkpoints"
SCRATCH="$ML_ROOT/input/data/scratch"   # on the instance ML volume (local NVMe on ml.g6): datasets + venv
LOG_SYNC_S="${MG_LOG_SYNC_S:-120}"       # logs/ -> checkpoint path (S3) cadence, as on the EC2 runner
MARGIN_MIN_S=900                         # DEADLINE margin before the hard cap: runtime/8, clamped to 15..60 min
MARGIN_MAX_S=3600
FAILURE_MAX_BYTES=1000                   # SageMaker keeps the first 1024 characters of /opt/ml/output/failure

cd "$REPO_DIR"
say() { printf '[%s] sagemaker_entry: %s\n' "$(date -u +%FT%TZ)" "$*"; }
[ -f "aws/jobs/$JOB.sh" ] || { say "no job script aws/jobs/$JOB.sh"; exit 3; }

# ------------------------------------------------------------------ 1. runs/ on the checkpoint path
mkdir -p "$CKPT/runs" "$ML_ROOT/model" "$ML_ROOT/output" "$SCRATCH" logs
if [ -d "$ML_ROOT/input/data/resume" ]; then
  say "restoring runs/ from the resume channel"
  cp -a "$ML_ROOT/input/data/resume/." "$CKPT/runs/"
fi
rm -rf runs && ln -s "$CKPT/runs" runs   # runs/ is git-ignored, so the code tarball never carries one

# ------------------------------------------------------------------ 2. environment for the job script
export DATA_DIR="${DATA_DIR:-$SCRATCH/data}" VENV="${VENV:-$SCRATCH/venv-aws}" SYSTEM_TORCH="${SYSTEM_TORCH:-1}"
if [ -z "${DEADLINE:-}" ] && [ -n "${MG_MAX_RUNTIME_S:-}" ]; then
  margin=$(( MG_MAX_RUNTIME_S / 8 ))
  [ "$margin" -lt "$MARGIN_MIN_S" ] && margin=$MARGIN_MIN_S
  [ "$margin" -gt "$MARGIN_MAX_S" ] && margin=$MARGIN_MAX_S
  DEADLINE="$(date -d "@$(( $(date +%s) + MG_MAX_RUNTIME_S - margin ))" +%Y-%m-%dT%H:%M)"   # container-local time
  export DEADLINE
fi
say "job=$JOB repo=$REPO_DIR DATA_DIR=$DATA_DIR VENV=$VENV SYSTEM_TORCH=$SYSTEM_TORCH DEADLINE=${DEADLINE:-none}"
{ nvidia-smi -L; echo "nproc $(nproc)"; free -g; df -h "$ML_ROOT" "$SCRATCH" /tmp /dev/shm; } 2>&1 | sed 's/^/  /'

# ------------------------------------------------------------------ 3. logs: S3 copy + CloudWatch stream
sync_logs() { mkdir -p "$CKPT/logs" && cp -u logs/* "$CKPT/logs/" 2>/dev/null || true; }
( while sleep "$LOG_SYNC_S"; do sync_logs; done ) &
SYNC_PID=$!
follow_logs() {   # tail each pipeline/training log once it appears, prefixed with its name (metric regexes use it)
  local f n
  local -A seen=()
  while true; do
    for f in logs/pipeline.log logs/train_*.log; do
      [ -f "$f" ] && [ -z "${seen[$f]:-}" ] || continue
      seen[$f]=1
      n="$(basename "$f" .log)"
      tail -n +1 -F "$f" 2>/dev/null | sed -u "s/^/[$n] /" &
    done
    sleep 20
  done
}
if [ "${MG_FOLLOW_LOGS:-1}" = "1" ]; then follow_logs & fi

# ------------------------------------------------------------------ 4. outputs + exit status
collect_outputs() {
  local f dest n=0
  for f in ${MG_OUTPUT_GLOBS:-}; do   # unquoted on purpose: split into globs, expanded against the repo root
    [ -f "$f" ] || continue
    for dest in "$CKPT/out" "$ML_ROOT/model"; do
      mkdir -p "$dest/$(dirname "$f")" && cp -f "$f" "$dest/$f"
    done
    n=$((n + 1))
  done
  say "collected $n output files into $CKPT/out and $ML_ROOT/model"
}
has_required() {
  local f
  for f in ${MG_REQUIRED_OUTPUTS:-}; do [ -f "$f" ] && return 0; done
  [ -z "${MG_REQUIRED_OUTPUTS:-}" ]
}
finish() {
  local rc="$1" why="${2:-}"
  kill "$SYNC_PID" 2>/dev/null
  collect_outputs
  if [ "$rc" -eq 0 ] && ! has_required; then
    rc=4 why="job exited 0 but produced none of: $MG_REQUIRED_OUTPUTS"
  fi
  if [ "$rc" -ne 0 ]; then
    { echo "job $JOB failed (rc=$rc) ${why}"; tail -n 6 logs/job.log 2>/dev/null; tail -n 8 logs/pipeline.log 2>/dev/null; } \
      | head -c "$FAILURE_MAX_BYTES" > "$ML_ROOT/output/failure"
  fi
  sync_logs
  say "done rc=$rc"
  exit "$rc"
}
trap 'say "SIGTERM (StopTrainingJob or MaxRuntimeInSeconds): saving logs and outputs"; finish 143 "stopped (SIGTERM)"' TERM

# Background + wait, so the TERM trap runs while the job is going; with pipefail, wait returns the job's status.
bash "aws/jobs/$JOB.sh" 2>&1 | tee -a logs/job.log &
wait $!
finish $?
