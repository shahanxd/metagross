<#
.SYNOPSIS
  Train the deploy terrain segmenter (LR-ASPP MobileNetV3-Large, OFFROAD5) on the laptop's NVIDIA GPU.

.DESCRIPTION
  One script, five modes (run from anywhere; it works from the repo root). Defaults are sized for a
  4 GB laptop GPU (RTX 3050): batch 8, 320x416 crops, mixed precision.

  -Setup    nvidia-smi check -> separate venv .venv-gpu (the CPU venv .venv is left untouched) -> torch
            CUDA wheels (index chosen from the driver's CUDA version) -> metagross + training deps ->
            CUDA sanity (conv fwd/bwd under autocast) -> segmentation unit tests -> download OFFROAD5
            (~8.5 GB) and RUGD-5L (~2.1 GB) into data/ unless present -> data layout check. Idempotent.
  -Bench    short timing run (800 training images = 100 steps, validation on 200 images) into
            runs/seg/bench_gpu; prints seconds per step and the projected wall time of the full run.
            Its numbers are timing only, never results.
  -Launch   start (or resume) training DETACHED (hidden window; survives closing the terminal).
            PID -> <RunDir>/train.pid. A hidden watcher keeps Windows from sleeping while training runs,
            then runs the finish step automatically when training exits.
  -Status   training process state + last log lines.
  default   finish: snapshot best.pt -> ONNX export models/<name>.onnx (+ JSON card, ORT-vs-torch check)
            -> metagross.eval.seg_eval on OFFROAD5 and RUGD-5L val + test (mIoU, per-class IoU,
            false-safe rate, CPU latency on this laptop) -> results/seg_<name>_{offroad5,rugd5}.json and
            deck_assets figures. Works on a partial checkpoint too.

  Training is resumable: an interrupted run (reboot, crash, deadline) continues from last.pt with the
  same command, at the exact step (step-indexed sampler, -EpochIters steps per epoch). Do not change
  -Epochs / -Bs / -EpochIters between launch and resume: the LR schedule depends on the total steps.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\train_seg_gpu.ps1 -Setup
  powershell -ExecutionPolicy Bypass -File scripts\train_seg_gpu.ps1 -Bench
  powershell -ExecutionPolicy Bypass -File scripts\train_seg_gpu.ps1 -Launch
  powershell -ExecutionPolicy Bypass -File scripts\train_seg_gpu.ps1 -Status
  powershell -ExecutionPolicy Bypass -File scripts\train_seg_gpu.ps1            # finish (export + eval)
  powershell -ExecutionPolicy Bypass -File scripts\train_seg_gpu.ps1 -Launch -Aug clean   # ablation run
#>
[CmdletBinding()]
param(
    [switch]$Setup,
    [switch]$Bench,
    [switch]$Launch,
    [switch]$Status,
    [int]$WaitPid = 0,
    [ValidateSet("robust", "clean")] [string]$Aug = "robust",
    [int]$Epochs = 25,          # 25 x 1000 steps x batch 8 = 200k images ~ 25 passes over the 8081 OFFROAD5 train images
    [int]$EpochIters = 1000,    # steps per epoch = validation + checkpoint period (~1 pass at batch 8)
    [int]$Bs = 8,               # fits 4 GB with --amp at 320x416; the AWS recipe used 32 on 24 GB
    [string]$Img = "320x416",   # same input size as the AWS recipe and the onboard Segmenter
    [int]$Workers = 4,          # DataLoader worker processes (decode + augmentation on the CPU)
    [string]$Deadline = "",     # optional local stop time 'HH:MM' or 'YYYY-MM-DDTHH:MM' (validate + checkpoint + exit)
    [int]$EvalThreads = 4,      # ONNX Runtime threads for the accuracy pass
    [string]$DataDir = "data"
)
# "Continue", not "Stop": Windows PowerShell 5.1 turns the stderr lines of native commands (Python
# logging) into error records when its output is redirected; failures are detected via $LASTEXITCODE.
$ErrorActionPreference = "Continue"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Venv = Join-Path $Repo ".venv-gpu"
$Py = Join-Path $Venv "Scripts\python.exe"
$Name = "lraspp_offroad5_$Aug"
$RunDir = "runs/seg/$Name"
$LogDir = "logs"
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "2"   # per process: the DataLoader workers do the CPU work
New-Item -ItemType Directory -Force $LogDir, "models", "results", "deck_assets", "runs/seg" | Out-Null

function Say([string]$msg) { Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg) }
function Check([string]$what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }

function Get-TorchIndex {
    # PyTorch CUDA wheels need a driver at least as new as their CUDA version; an RTX 30xx (sm_86) runs on all of them.
    $smi = (& nvidia-smi) -join "`n"
    if ($smi -notmatch "CUDA Version:\s*(\d+)\.(\d+)") { return "https://download.pytorch.org/whl/cu124" }
    $v = [int]$Matches[1] * 100 + [int]$Matches[2]
    if ($v -ge 1208) { return "https://download.pytorch.org/whl/cu128" }
    if ($v -ge 1206) { return "https://download.pytorch.org/whl/cu126" }
    if ($v -ge 1204) { return "https://download.pytorch.org/whl/cu124" }
    if ($v -ge 1201) { return "https://download.pytorch.org/whl/cu121" }
    return "https://download.pytorch.org/whl/cu118"
}

function Get-BasePython {
    # Same interpreter as the existing CPU venv (.venv\pyvenv.cfg 'home'), else the py launcher, else python on PATH.
    $cfg = Join-Path $Repo ".venv\pyvenv.cfg"
    if (Test-Path $cfg) {
        $home_ = (Get-Content $cfg | Where-Object { $_ -match "^home\s*=" }) -replace "^home\s*=\s*", ""
        $exe = Join-Path $home_.Trim() "python.exe"
        if (Test-Path $exe) { return @($exe) }
    }
    if (Get-Command py -ErrorAction SilentlyContinue) { return @("py", "-3") }
    return @("python")
}

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

function Get-TrainArgs([string]$out, [int]$epochs, [int]$epochIters, [int]$subset, [int]$valSubset) {
    $a = @("-m", "metagross.train.train_seg", "--model", "lraspp", "--data", "offroad5",
        "--data-root", (Join-Path $DataDir "offroad5"), "--aug", $Aug, "--init", "imagenet",
        "--epochs", "$epochs", "--epoch-iters", "$epochIters", "--bs", "$Bs", "--lr", "6e-4", "--img", $Img,
        "--amp", "--device", "cuda", "--workers", "$Workers", "--threads", "2",
        "--train-subset", "$subset", "--val-subset", "$valSubset", "--resume", "--out", $out)
    if ($Deadline) { $a += @("--deadline", $Deadline) }
    return $a
}

# ------------------------------------------------------------------ setup
if ($Setup) {
    Say "GPU check"
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw "nvidia-smi not found: install the NVIDIA driver" }
    & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    & nvidia-smi | Out-File -Encoding utf8 (Join-Path $LogDir "nvidia-smi.txt")

    if (-not (Test-Path $Py)) {
        $base = Get-BasePython
        Say "creating $Venv with $($base -join ' ')"
        if ($base.Count -gt 1) { & $base[0] $base[1..($base.Count - 1)] -m venv $Venv } else { & $base[0] -m venv $Venv }
        Check "venv creation"
    }
    & $Py -m pip install --upgrade pip wheel | Out-Null

    $index = Get-TorchIndex
    & $Py -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Say "installing torch + torchvision from $index"
        & $Py -m pip install torch torchvision --index-url $index
        Check "torch install"
    } else { Say "CUDA torch already installed" }
    Say "installing metagross + training dependencies"
    & $Py -m pip install -e ".[train,dev]" "numpy<2.3" onnx tqdm
    Check "pip install"

    Say "CUDA sanity"
    & $Py -c @"
import torch
assert torch.cuda.is_available(), 'torch cannot see the GPU'
p = torch.cuda.get_device_properties(0)
print(f'torch {torch.__version__} (CUDA {torch.version.cuda}) | {p.name} | {p.total_memory / 2**30:.1f} GiB | bf16={torch.cuda.is_bf16_supported()}')
net = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
x = torch.randn(4, 3, 64, 64, device='cuda', requires_grad=True)
with torch.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
    y = net(x).float().mean()
y.backward(); torch.cuda.synchronize(); print('conv fwd/bwd ok')
"@
    Check "CUDA sanity"

    Say "unit tests (segmentation pipeline)"
    & $Py -m pytest -q tests/test_seg_data.py tests/test_seg_metrics.py tests/test_seg_semantics.py tests/test_seg_train.py tests/test_train_sampling.py
    if ($LASTEXITCODE -ne 0) { Write-Warning "unit tests failed: check the output above before training" }

    $env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
    foreach ($ds in @("offroad5", "rugd5")) {
        if (Test-Path (Join-Path $DataDir "$ds\train")) { Say "$ds already in $DataDir (delete the folder to re-download)"; continue }
        Say "downloading $ds into $DataDir (resumable; re-run -Setup if it is interrupted)"
        & $Py scripts/download_seg_data.py --dataset $ds --splits val test train --root $DataDir 2>&1 |
            Tee-Object -FilePath (Join-Path $LogDir "download_$ds.log")
        Check "download $ds"
    }

    Say "data layout check"
    $env:MG_DATA_DIR = $DataDir
    & $Py -c @"
import collections, os, sys
from pathlib import Path
from metagross.train.data_index import list_samples
root, bad = Path(os.environ['MG_DATA_DIR']), []
for ds in ('offroad5', 'rugd5'):
    for split in ('train', 'val', 'test'):
        try:
            s = list_samples(root / ds, split)
        except FileNotFoundError as exc:
            bad.append(f'{ds}/{split}: {exc}'); continue
        print(f'{ds:9s} {split:5s} {len(s):6d} samples  by source {dict(collections.Counter(x.source for x in s))}')
        if not s: bad.append(f'{ds}/{split}: 0 samples')
sys.exit('data layout problem:\n  ' + '\n  '.join(bad) if bad else 0)
"@
    Check "data layout check"
    Say "setup done. Next: -Bench (5 min timing), then -Launch"
    exit 0
}

# ------------------------------------------------------------------ bench
if ($Bench) {
    $benchDir = "runs/seg/bench_gpu"
    if (Test-Path $benchDir) { Remove-Item -Recurse -Force $benchDir }   # always a fresh timing run
    $benchSteps = 100  # 800 images at batch 8; the rate is taken between logged steps 20 and 100 (after warm-up)
    Say "timing run: $benchSteps steps of batch $Bs at $Img (timing only, not a result)"
    $t0 = Get-Date
    & $Py @(Get-TrainArgs $benchDir 1 0 ($benchSteps * $Bs) 200)
    Check "bench"
    & $Py -c @"
import re, sys
from datetime import datetime
pts = [(datetime.strptime(t, '%Y-%m-%d %H:%M:%S,%f'), int(i)) for t, i in
       re.findall(r'^(\S+ \S+) INFO \S+: ep \d+ it (\d+)/', open('$benchDir/train.log', encoding='utf-8').read(), re.M)]
pts = [p for p in pts if p[1] >= 20]   # the logged 's/it' averages from the epoch start, including warm-up
if len(pts) < 2: sys.exit('not enough logged steps to time')
s = (pts[-1][0] - pts[0][0]).total_seconds() / (pts[-1][1] - pts[0][1])
total = $Epochs * $EpochIters
print(f'steady {s:.2f} s/step ({$Bs / s:.0f} img/s) -> {total} steps = {s * total / 3600:.1f} h of training, plus {$Epochs} validations of 2907 images')
"@
    Check "bench timing"
    Say "bench done in $([int]((Get-Date) - $t0).TotalSeconds) s (includes one validation of 200 images). If it ran out of GPU memory, retry with -Bs 4."
    exit 0
}

# ------------------------------------------------------------------ status
if ($Status) {
    $p = Get-TrainProcess
    Write-Host ("training ($Name): " + $(if ($p) { "RUNNING (PID $($p.ProcessId))" } else { "not running" }))
    $log = Join-Path $RunDir "train.log"
    if (Test-Path $log) { Get-Content $log -Tail 8 }
    if (Test-Path (Join-Path $RunDir "best.pt")) { Write-Host "best.pt present (finish step can run)" }
    exit 0
}

# ------------------------------------------------------------------ launch
if ($Launch) {
    if (-not (Test-Path $Py)) { throw "no ${Venv}: run -Setup first" }
    New-Item -ItemType Directory -Force $RunDir | Out-Null
    $alive = Get-TrainProcess
    if ($alive) { Write-Host "training already running: PID $($alive.ProcessId)"; exit 0 }
    $a = Get-TrainArgs $RunDir $Epochs $EpochIters 0 0
    $p = Start-Process -FilePath $Py -ArgumentList $a -WorkingDirectory $Repo -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $RunDir "stdout.log") -RedirectStandardError (Join-Path $RunDir "stderr.log")
    Set-Content -Path (Join-Path $RunDir "train.pid") -Value $p.Id -Encoding ascii
    $pass = "-Aug $Aug -EvalThreads $EvalThreads -DataDir `"$DataDir`""
    $w = Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -PassThru -ArgumentList (
        "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSCommandPath`" -WaitPid $($p.Id) $pass")
    Say "launched ${Name}: PID $($p.Id) ($Epochs x $EpochIters steps, batch $Bs, $Img); watcher PID $($w.Id) keeps the PC awake and finishes automatically"
    Say "progress: powershell -ExecutionPolicy Bypass -File scripts\train_seg_gpu.ps1 -Status   (or Get-Content $RunDir\train.log -Tail 5 -Wait)"
    exit 0
}

# ------------------------------------------------------------------ watcher
if ($WaitPid -gt 0) {
    # Keep Windows awake (not the display) while training runs: ES_CONTINUOUS | ES_SYSTEM_REQUIRED, released on exit.
    Add-Type -Namespace MG -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
    [void][MG.Power]::SetThreadExecutionState([uint32]2147483649)
    $finishDir = Join-Path $RunDir "finish"
    New-Item -ItemType Directory -Force $finishDir | Out-Null
    $pollS = 60  # seconds between liveness checks of the training process
    while (Get-TrainProcessById $WaitPid) { Start-Sleep -Seconds $pollS }
    [void][MG.Power]::SetThreadExecutionState([uint32]2147483648)
    $fin = Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -Wait -PassThru `
        -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Aug $Aug -EvalThreads $EvalThreads -DataDir `"$DataDir`"" `
        -RedirectStandardOutput (Join-Path $finishDir "auto_finish.out.log") -RedirectStandardError (Join-Path $finishDir "auto_finish.err.log")
    exit $fin.ExitCode
}

# ------------------------------------------------------------------ finish
$best = Join-Path $RunDir "best.pt"
if (-not (Test-Path $best)) { throw "no $best yet (the first checkpoint is written after the first $EpochIters-step epoch)" }
$snapDir = Join-Path $RunDir "finish"
New-Item -ItemType Directory -Force $snapDir | Out-Null
$snap = Join-Path $snapDir "best_snapshot.pt"
Copy-Item $best $snap -Force -ErrorAction Stop   # training may overwrite best.pt while we work
$alive = Get-TrainProcess
$partial = [bool]$alive
Say ("training process: " + $(if ($partial) { "running (PID $($alive.ProcessId)) - evaluating a PARTIAL checkpoint" } else { "not running" }))

$model = "models/$Name.onnx"
& $Py -m metagross.train.export_onnx --ckpt $snap --out $model --data-root (Join-Path $DataDir "offroad5") `
    --name "LR-ASPP MobileNetV3 OFFROAD5 ($Aug aug), laptop GPU"
Check "export"

foreach ($ds in @("offroad5", "rugd5")) {
    $dsLabel = if ($ds -eq "offroad5") { "OFFROAD5" } else { "RUGD-5L" }
    $label = "LR-ASPP ($Aug aug), $dsLabel" + $(if ($partial) { " [PARTIAL checkpoint]" } else { "" })
    & $Py -m metagross.eval.seg_eval --model $model --data-root (Join-Path $DataDir $ds) --splits val test `
        --threads $EvalThreads --latency-threads 2 4 --out "results/seg_${Name}_$ds.json" `
        --fig-prefix "deck_assets/seg_${Name}_$ds" --label $label --tag "seg_${Aug}_$ds"
    Check "seg_eval $ds"
}
Say "done: $model, results/seg_${Name}_offroad5.json, results/seg_${Name}_rugd5.json, figures in deck_assets/"
Say "claims rows are inside each JSON ('claims'); register them in results/claims.csv as Tested."
