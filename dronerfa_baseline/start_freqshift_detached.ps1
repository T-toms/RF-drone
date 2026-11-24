# Launch the frequency-shift Stage2 training as an independent Windows process,
# so it keeps running after the Claude Code session (or this terminal) closes.
# If a checkpoint from a finished epoch exists, it resumes from the next epoch.
#
# Usage (PowerShell, from anywhere):
#   powershell -ExecutionPolicy Bypass -File "<this file>"

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $here "..\.venv\Scripts\python.exe"
$out = Join-Path $here "outputs\stage2_freqshift_20261010\seed_2027"
$log = Join-Path $here "outputs\stage2_freqshift_20261010\training_console.log"
$err = Join-Path $here "outputs\stage2_freqshift_20261010\training_console.err.log"

$running = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
    Where-Object { $_.CommandLine -like "*run_stage2_12drones_freqshift.py*" }
if ($running) {
    Write-Host "Training is already running (PID $($running.ProcessId -join ', ')). Not starting another one."
    exit 1
}

$args = @("-u", "run_stage2_12drones_freqshift.py", "--output", $out)
if (Test-Path (Join-Path $out "last_training_state.pt")) {
    $args += "--resume"
    Write-Host "Found a saved epoch checkpoint: resuming."
} elseif (Test-Path $out) {
    # No finished epoch yet: a partial run leaves only config/cache manifests, start clean.
    Get-ChildItem $out -File | Where-Object { $_.Name -in @("run_config.json", "validation_cache_manifest.json") } | Remove-Item
    Write-Host "No finished epoch yet: starting from epoch 1."
}

$p = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $here `
    -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru
Write-Host "Started PID $($p.Id). Progress: $log"
