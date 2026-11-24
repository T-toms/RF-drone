$ErrorActionPreference = 'Stop'
$project = 'C:\Users\Administrator\Desktop\数据集处理'
$out = Join-Path $project 'dronerfa_baseline\outputs\cascade12_development_20260927'
$summaryPath = Join-Path $out 'summary.json'
$evidencePath = Join-Path $project 'work\verify-cascade12-after-evaluation-20260929.json'
$verifierLog = Join-Path $project 'work\verify-cascade12-after-evaluation-20260929.log'
$python = Join-Path $project '.venv\Scripts\python.exe'
$verifier = Join-Path $project 'dronerfa_baseline\verify_cascade12_codex.py'
$targetPid = 11560
$expectedStart = [datetime]'2026-09-29T20:21:54'
$deadline = (Get-Date).AddHours(4)

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

Save-Status 'watching' 'One-shot wait for the locked cascade development evaluation.'
try {
    while ((Get-Date) -lt $deadline) {
        $process = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
        if ($null -eq $process) { break }
        if ($process.Path -ne 'D:\anaconda\python.exe' -or [math]::Abs(($process.StartTime - $expectedStart).TotalSeconds) -gt 2) {
            Save-Status 'identity_mismatch' 'The PID no longer belongs to the observed evaluation process; verifier was not run.'
            exit 3
        }
        Start-Sleep -Seconds 5
    }
    if ((Get-Date) -ge $deadline) {
        Save-Status 'timeout' 'Evaluation did not finish within four hours; verifier was not run.'
        exit 4
    }
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        Save-Status 'evaluation_failed' 'Evaluation process exited without summary.json; verifier was not run.'
        exit 2
    }
    $summary = Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
    if ($summary.status -ne 'completed') {
        Save-Status 'evaluation_incomplete' 'Summary exists but status is not completed; verifier was not run.'
        exit 2
    }
    & $python $verifier 2>&1 | Set-Content -LiteralPath $verifierLog -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        Save-Status 'verification_failed' "Independent verifier exited $LASTEXITCODE; inspect verifier log."
        exit 5
    }
    Save-Status 'verified' 'Locked cascade evaluation completed and independent record-level verifier passed.'
    exit 0
} catch {
    Save-Status 'monitor_error' $_.Exception.Message
    exit 6
}
