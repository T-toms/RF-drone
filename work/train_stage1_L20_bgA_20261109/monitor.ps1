$ErrorActionPreference='Stop'
$outDir=Join-Path $PSScriptRoot 'memory_monitor.json'
$peak=0L
while (-not (Test-Path (Join-Path $PSScriptRoot 'training_complete.json'))) {
  $tasks=Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like '*train_stage1_L20_bgA_20261109*train.py*' }
  foreach ($task in $tasks) {
    $proc=Get-Process -Id $task.ProcessId -ErrorAction SilentlyContinue
    if ($proc -and $proc.PeakWorkingSet64 -gt $peak) { $peak=$proc.PeakWorkingSet64 }
  }
  @{peak_training_process_bytes=$peak;updated=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $outDir
  Start-Sleep -Seconds 5
}
