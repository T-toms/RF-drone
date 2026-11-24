$ErrorActionPreference = 'Stop'
$project = 'C:\Users\Administrator\Desktop\数据集处理'
$run = Join-Path $project 'dronerfa_baseline\outputs\stage2_12drones_20260927\seed_2027'
$summaryPath = Join-Path $run 'summary.json'
$evidencePath = Join-Path $project 'work\verify-stage2-after-epoch5-20260929.json'
$verifierLog = Join-Path $project 'work\verify-stage2-after-epoch5-20260929.log'
$python = Join-Path $project '.venv\Scripts\python.exe'
$verifier = Join-Path $project 'dronerfa_baseline\verify_stage2_12drones_codex.py'
$targetPid = 10700
$expectedStart = [datetime]'2026-09-29T16:58:36'
$deadline = (Get-Date).AddHours(8)

function Save-Status([string]$status, [string]$detail) {
    [pscustomobject]@{
        status = $status
        detail = $detail
        targetPid = $targetPid
        checkedAt = (Get-Date).ToString('o')
        summaryPath = $summaryPath
        verifierLog = $verifierLog
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $evidencePath -Encoding utf8
}

Save-Status 'watching' 'One-shot wait for the five-epoch run; verify saved outputs after completion.'
try {
    while ((Get-Date) -lt $deadline) {
        $process = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
        if ($null -eq $process) { break }
        if ($process.Path -ne 'D:\anaconda\python.exe' -or [math]::Abs(($process.StartTime - $expectedStart).TotalSeconds) -gt 2) {
            Save-Status 'identity_mismatch' 'The PID no longer belongs to the observed training process; verifier was not run.'
            exit 3
        }
        Start-Sleep -Seconds 5
    }
    if ((Get-Date) -ge $deadline) {
        Save-Status 'timeout' 'Training did not finish within eight hours; verifier was not run.'
        exit 4
    }
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        Save-Status 'training_failed' 'Training process exited without summary.json; verifier was not run.'
        exit 2
    }
    $summary = Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
    if ($summary.status -ne 'completed') {
        Save-Status 'training_incomplete' 'Summary exists but status is not completed; verifier was not run.'
        exit 2
    }
    & $python $verifier 2>&1 | Set-Content -LiteralPath $verifierLog -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        Save-Status 'verification_failed' "Independent verifier exited $LASTEXITCODE; inspect verifier log."
        exit 5
    }
    Save-Status 'verified' 'Five epochs completed; independent verifier passed. Test IQ evaluation has not started.'
    exit 0
} catch {
    Save-Status 'monitor_error' $_.Exception.Message
    exit 6
}
