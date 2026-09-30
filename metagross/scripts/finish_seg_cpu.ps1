<#
.SYNOPSIS
  Launch (-Launch) or finish (default) the budgeted CPU training of the LR-ASPP terrain segmenter.

.DESCRIPTION
  -Launch   start (or resume) runs/seg_cpu training DETACHED (hidden window, survives the calling
            session); refuses if a run is already alive. PID -> runs/seg_cpu/train.pid. Also starts a
            hidden watcher (-WaitPid) that runs the finish step automatically when training exits.
  -WaitPid  wait until that process has exited, then run the finish step with its output in
            runs/seg_cpu/finish/auto_finish.{out,err}.log.
  default   finish: snapshot best.pt -> export models/lraspp_offroad5_cpu.onnx (+ JSON model card)
            -> metagross.eval.seg_eval on RUGD-5L val + test (mIoU, per-class IoU, false-safe rate)
            -> results/seg_cpu.json -> comparison + Tested claims vs results/seg_smoke.json and
            results/seg_zeroshot.json (metagross.train.seg_cpu_report). Works on partial checkpoints
            (any best.pt written after the first 250-step epoch), also while training is running.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\finish_seg_cpu.ps1 -Launch
  powershell -ExecutionPolicy Bypass -File scripts\finish_seg_cpu.ps1
  powershell -ExecutionPolicy Bypass -File scripts\finish_seg_cpu.ps1 -MaxImages 100 -Result results\seg_cpu_quick.json
#>
[CmdletBinding()]
param(
    [switch]$Launch,
    [int]$WaitPid = 0,
    [string]$RunDir = "runs/seg_cpu",
    [string]$Model = "models/lraspp_offroad5_cpu.onnx",
    [string]$Result = "results/seg_cpu.json",
    [string]$DataRoot = "data/rugd5",
    [int]$Threads = 2,
    [int]$MaxImages = 0
)
# "Continue", not "Stop": Windows PowerShell 5.1 turns the stderr lines of native commands (Python
# logging) into error records when its output is redirected; failures are detected via $LASTEXITCODE.
$ErrorActionPreference = "Continue"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
$env:OMP_NUM_THREADS = "$Threads"
$env:PYTHONIOENCODING = "utf-8"

# Single source of truth for the CPU training recipe (resume-safe: re-running continues from last.pt).
# 256x320 input = same as models/lraspp_smoke.onnx, so onboard latency is unchanged.
# Budget: 11 x 250 = 2750 steps of batch 8. Measured 1.8 s/step on an idle machine but 4.7-5.4 s/step
# on this shared laptop (2 threads, other agents' simulations running) + ~45 s per validation of 300
# images => ~4 h; the 09:25 deadline is a hard stop if the machine gets slower (validate + checkpoint
# + exit, resumable). Changing --epochs and relaunching resumes exactly (step-indexed sample stream).
$TrainArgs = @(
    "-m", "metagross.train.train_seg", "--model", "lraspp", "--data", "rugd5", "--aug", "robust", "--init", "imagenet",
    "--img", "256x320", "--bs", "8", "--lr", "6e-4", "--threads", "2", "--workers", "1",
    "--epochs", "11", "--epoch-iters", "250", "--sampling", "repeat", "--val-subset", "300",
    "--deadline", "2026-09-30T09:25", "--resume", "--out", $RunDir
)

function Get-TrainProcessById([int]$TrainPid) {
    # the PID must still belong to a train_seg process (Windows reuses PIDs)
    $p = Get-CimInstance Win32_Process -Filter "ProcessId=$TrainPid" -ErrorAction SilentlyContinue
    if ($p -and $p.CommandLine -match "train_seg") { return $p }
    return $null
}

function Get-TrainProcess {
    $pidFile = Join-Path $RunDir "train.pid"
    if (-not (Test-Path $pidFile)) { return $null }
    return Get-TrainProcessById ([int](Get-Content $pidFile -TotalCount 1))
}

# Same options for the watcher / the finish child as this invocation.
$PassArgs = "-RunDir `"$RunDir`" -Model `"$Model`" -Result `"$Result`" -DataRoot `"$DataRoot`" -Threads $Threads -MaxImages $MaxImages"

if ($Launch) {
    New-Item -ItemType Directory -Force $RunDir | Out-Null
    $alive = Get-TrainProcess
    if ($alive) { Write-Host "training already running: PID $($alive.ProcessId)"; exit 0 }
    $p = Start-Process -FilePath $Py -ArgumentList $TrainArgs -WorkingDirectory $Repo -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $RunDir "stdout.log") -RedirectStandardError (Join-Path $RunDir "stderr.log")
    Set-Content -Path (Join-Path $RunDir "train.pid") -Value $p.Id -Encoding ascii
    $w = Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -PassThru -ArgumentList (
        "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSCommandPath`" -WaitPid $($p.Id) $PassArgs")
    Write-Host "launched PID $($p.Id) (auto-finish watcher PID $($w.Id)); progress: Get-Content $RunDir\train.log -Tail 5"
    exit 0
}

if ($WaitPid -gt 0) {
    $finishDir = Join-Path $RunDir "finish"
    New-Item -ItemType Directory -Force $finishDir | Out-Null
    $pollS = 60  # seconds between liveness checks of the training process
    while (Get-TrainProcessById $WaitPid) { Start-Sleep -Seconds $pollS }
    $fin = Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -Wait -PassThru `
        -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" $PassArgs" `
        -RedirectStandardOutput (Join-Path $finishDir "auto_finish.out.log") -RedirectStandardError (Join-Path $finishDir "auto_finish.err.log")
    exit $fin.ExitCode
}

# ------------------------------------------------------------------ finish
$best = Join-Path $RunDir "best.pt"
if (-not (Test-Path $best)) { throw "no $best yet (first checkpoint is written after the first 250-step epoch)" }
$snapDir = Join-Path $RunDir "finish"
New-Item -ItemType Directory -Force $snapDir | Out-Null
$snap = Join-Path $snapDir "best_snapshot.pt"
Copy-Item $best $snap -Force -ErrorAction Stop   # training may overwrite best.pt while we work
$alive = Get-TrainProcess
Write-Host ("training process: " + $(if ($alive) { "running (PID $($alive.ProcessId)) - evaluating a PARTIAL checkpoint" } else { "not running" }))

& $Py -m metagross.train.export_onnx --ckpt $snap --out $Model --data-root $DataRoot `
    --name "LR-ASPP MobileNetV3, RUGD-5L (OFFROAD5 classes), CPU-trained, robust aug"
if ($LASTEXITCODE -ne 0) { throw "export failed" }

$label = & $Py -m metagross.train.seg_cpu_report --label-only --ckpt $snap --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { throw "label failed" }
$figDir = Join-Path $snapDir "figs"
& $Py -m metagross.eval.seg_eval --model $Model --data-root $DataRoot --splits val test --threads $Threads `
    --latency-threads 2 4 --max-images $MaxImages --out $Result --fig-prefix (Join-Path $figDir "seg_cpu") `
    --label "$label" --tag "seg_cpu"
if ($LASTEXITCODE -ne 0) { throw "seg_eval failed" }

& $Py -m metagross.train.seg_cpu_report --result $Result --ckpt $snap --run-dir $RunDir `
    --baselines results/seg_smoke.json results/seg_zeroshot.json
if ($LASTEXITCODE -ne 0) { throw "report failed" }
Write-Host "done: $Model, $Result, figures in $figDir"
