# Start batch-1 download as an independent Windows process so it survives
# closing Claude Code / the terminal. Rerun to resume (finished classes are
# skipped, partial downloads continue from their checkpoint).
#   powershell -ExecutionPolicy Bypass -File "<this file>"
$ErrorActionPreference = "Stop"
$running = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*download_batch1_20261014.sh*" -and $_.CommandLine -notlike "*Get-CimInstance*" }
if ($running) { Write-Host "Already running (PID $($running.ProcessId -join ', '))."; exit 1 }
$bash = "D:/git/Git/bin/bash.exe"
$script = "C:/Users/Administrator/Desktop/数据集处理/work/dl_bg/download_batch1_20261014.sh"
$log = "G:\DroneRFa数据集\download_batch1_20261014.log"
$err = "G:\DroneRFa数据集\download_batch1_20261014.err.log"
$p = Start-Process -FilePath $bash -ArgumentList @("-c", "`"bash '$script' >> '/g/DroneRFa数据集/download_batch1_20261014.log' 2>&1`"") `
    -WindowStyle Minimized -PassThru
Write-Host "Started PID $($p.Id). Log: $log"
