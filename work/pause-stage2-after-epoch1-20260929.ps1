$ErrorActionPreference = 'Stop'
$project = 'C:\Users\Administrator\Desktop\数据集处理'
$run = Join-Path $project 'dronerfa_baseline\outputs\stage2_12drones_20260927\seed_2027'
$progressPath = Join-Path $run 'progress.json'
$statePath = Join-Path $run 'last_training_state.pt'
$selectionPath = Join-Path $run 'selection.json'
$logPath = Join-Path $project 'work\pause-stage2-after-epoch1-20260929.json'
$targetPid = 17668
$expectedStart = [datetime]'2026-09-29T15:34:24'
$deadline = (Get-Date).AddHours(2)

function Save-Status([string]$status, [string]$detail) {
    [pscustomobject]@{
        status = $status
        detail = $detail
        targetPid = $targetPid
        checkedAt = (Get-Date).ToString('o')
        progressPath = $progressPath
        statePath = $statePath
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $logPath -Encoding utf8
}

Save-Status 'watching' 'Waiting for a complete epoch 1 progress record and saved training state.'
try {
    while ((Get-Date) -lt $deadline) {
        $process = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            Save-Status 'process_exited_before_pause' 'Training process exited before the watcher saw a complete epoch 1 checkpoint.'
            exit 2
        }
        if ($process.Path -ne 'D:\anaconda\python.exe' -or [math]::Abs(($process.StartTime - $expectedStart).TotalSeconds) -gt 2) {
            Save-Status 'identity_mismatch' 'The PID no longer belongs to the observed training process; no process was stopped.'
            exit 3
        }
        if ((Test-Path -LiteralPath $progressPath) -and (Test-Path -LiteralPath $statePath) -and (Test-Path -LiteralPath $selectionPath)) {
            try {
                $progress = Get-Content -Raw -LiteralPath $progressPath | ConvertFrom-Json
                $selection = Get-Content -Raw -LiteralPath $selectionPath | ConvertFrom-Json
                $stateSize = (Get-Item -LiteralPath $statePath).Length
                if ([int]$progress.epoch -eq 1 -and [int]$selection.selected_epoch -eq 1 -and $stateSize -gt 1000000) {
                    Stop-Process -Id $targetPid -Force -ErrorAction Stop
                    Save-Status 'paused_after_epoch1' ("Saved state bytes=$stateSize; progress epoch=1; stopped only the verified training PID.")
                    exit 0
                }
            } catch {
                # Progress may be between write operations; retry on the next poll.
            }
        }
        Start-Sleep -Milliseconds 200
    }
    Save-Status 'timeout' 'Epoch 1 was not observed within two hours; no process was stopped.'
    exit 4
} catch {
    Save-Status 'watcher_error' $_.Exception.Message
    exit 5
}
