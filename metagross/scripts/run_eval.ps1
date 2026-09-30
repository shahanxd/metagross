# Pre-registered EVAL run (see docs/EVAL_PREREGISTRATION.md). Run from the repo root in PowerShell
# on a quiet machine, with the code frozen at the commit recorded in the preregistration.
#
#   Stage A (headline, real sensor path = rendered stereo):
#     A1  FULL        on all 60 EVAL seeds (0-59), 2 workers
#     A2  T1_TYPICAL  on the 30 hazard seeds (F2 ditch field, F3 crest+ditch, F4 sudden obstacle), 1 worker
#   Stage B (optional, only if time remains): tier0 FULL / T1_TYPICAL / NO_NEG on all 60 seeds.
#
# Nothing here is tuned on EVAL: configs come from results/configs/eval_configs.json (DEV-tuned).
param(
    [ValidateSet("A", "B")] [string] $Stage = "A",
    [double] $MaxSimS = 150
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$py = Join-Path $repo ".venv\Scripts\python.exe"
$cfg = "results\configs\eval_configs.json"
$env:OMP_NUM_THREADS = "2"
New-Item -ItemType Directory -Force "results\runs_eval_logs" | Out-Null

$hazard = @(1, 7, 13, 19, 25, 31, 37, 43, 49, 55,     # F2_ditch_field
            2, 8, 14, 20, 26, 32, 38, 44, 50, 56,     # F3_crest_ditch
            3, 9, 15, 21, 27, 33, 39, 45, 51, 57)     # F4_sudden_obstacle

function Start-Batch([string] $name, [string[]] $batchArgs) {
    $log = "results\runs_eval_logs\$name.log"
    $err = "results\runs_eval_logs\$name.err.log"
    $p = Start-Process -FilePath $py -ArgumentList (@("-m", "metagross.sim.batch") + $batchArgs) `
        -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru
    "{0} started: PID {1} -> {2}" -f $name, $p.Id, $log
}

if ($Stage -eq "A") {
    Start-Batch "A1_stereo_FULL" @("--split", "eval", "--sensor-mode", "stereo", "--workers", "2", "--max-sim-s", "$MaxSimS",
        "--configs", $cfg, "--config-names", "FULL", "--out", "results\runs_eval_stereo")
    Start-Batch "A2_stereo_T1" (@("--split", "eval", "--sensor-mode", "stereo", "--workers", "1", "--max-sim-s", "$MaxSimS",
        "--configs", $cfg, "--config-names", "T1_TYPICAL", "--out", "results\runs_eval_stereo", "--seeds") + ($hazard | ForEach-Object { "$_" }))
} else {
    Start-Batch "B_tier0_all" @("--split", "eval", "--sensor-mode", "tier0", "--workers", "2", "--max-sim-s", "$MaxSimS",
        "--configs", $cfg, "--out", "results\runs_eval_tier0")
}
# Afterwards: python -m metagross.sim.closed_loop_summary --split eval --tier0 results/runs_eval_tier0 `
#   --stereo results/runs_eval_stereo --out results/closed_loop_eval.json ; python -m metagross.eval.claims
