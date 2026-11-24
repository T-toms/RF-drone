# SELECT12-G2 launcher
# 在 G 盘启动「剩余 41 个目标 MAT」的可续传后台下载进程。
# 只读前置检查 + 隐藏窗口后台启动 + 启动记录落盘；不等待下载完成。
# 不使用 --restart-from-zero（保留既有断点）。
[CmdletBinding()]
param(
    [int]$Workers = 3,
    [int]$BlockBytes = 8388608,
    [int]$FileAttempts = 1,
    [int]$RetryRounds = 1
)

$ErrorActionPreference = 'Stop'

$root = 'C:\Users\Administrator\Desktop\数据集处理'
$py = Join-Path $root '.venv\Scripts\python.exe'
$script = Join-Path $root 'dronerfa_baseline\download_select12g.py'
$cfg = Join-Path $root 'work\sel12g\sel12g-config-20260924.json'
$rec = 'G:\DroneRFa数据集\download_records_12models_20260922'
$logDir = Join-Path $root 'tmp\sel12g-g2'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 每次启动用带时间戳的 stdout/stderr/启动记录，避免覆盖上一次运行的证据
# （G 盘 select12g-g2.log 由程序自身以追加方式写，跨次运行连续保留）
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$appLog = Join-Path $rec 'select12g-g2.log'
$outLog = Join-Path $logDir "stdout-$stamp.log"
$errLog = Join-Path $logDir "stderr-$stamp.log"
$launchRecord = Join-Path $logDir "launch-record-$stamp.json"
$lockPath = Join-Path $rec 'select12g.lock'

foreach ($required in @($py, $script, $cfg)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "缺少必要文件：$required" }
}
if (Test-Path -LiteralPath $lockPath) {
    $holderRaw = Get-Content -LiteralPath $lockPath -Raw -Encoding UTF8
    $holderPid = $null
    try { $holderPid = (ConvertFrom-Json $holderRaw).pid } catch { }
    if ($holderPid -and (Get-Process -Id $holderPid -ErrorAction SilentlyContinue)) {
        throw "已有下载实例在运行（锁持有者 PID $holderPid），按任务要求绝不启动第二个进程：$holderRaw"
    }
    # 与 download_select12g.py 的 SingleInstanceLock 语义一致：持有者已不存在则回收残留锁。
    # 显式检查可避免 PID 被复用导致的误判，也让「暂停后次日续传」不必手工删锁。
    Write-Output "发现残留锁（PID $holderPid 已不存在），按程序自身的回收语义清理后继续"
    Remove-Item -LiteralPath $lockPath -Force
}

$argList = @(
    $script,
    '--config', $cfg,
    '--workers', $Workers,
    '--block-bytes', $BlockBytes,
    '--file-attempts', $FileAttempts,
    '--retry-rounds', $RetryRounds,
    '--log-file', $appLog
)

$proc = Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $root `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput $outLog -RedirectStandardError $errLog

$record = [ordered]@{
    task        = 'SELECT12-G2'
    pid         = $proc.Id
    startedAt   = (Get-Date).ToString('o')
    interpreter = $py
    script      = $script
    config      = $cfg
    commandLine = ($py + ' ' + ($argList -join ' '))
    appLog      = $appLog
    stdoutLog   = $outLog
    stderrLog   = $errLog
    windowStyle = 'Hidden'
    note        = '后台隐藏窗口进程，独立于 DSH 会话继续运行；单实例锁位于 G 盘 records 目录'
}
$record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $launchRecord -Encoding UTF8
$record | ConvertTo-Json -Depth 4
