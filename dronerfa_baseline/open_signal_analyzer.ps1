# DroneRFa 任意IQ信号分析界面 —— 一键启动器（桌面快捷方式入口）
#
# 做什么：
#   1) 先用短超时检查 http://127.0.0.1:8765/api/health；
#   2) 若服务已就绪（ok=true 且 model.state=ready），直接用默认浏览器打开界面，
#      不会启动第二个服务；
#   3) 若服务未运行，隐藏启动 dronerfa_baseline\launch_signal_analyzer.ps1 -NoBrowser -Quiet，
#      等模型就绪（默认最长 90 秒）后自动打开界面；服务日志写到项目 dronerfa_baseline\scratch\。
#
# 桌面快捷方式由 Codex 在验收后创建，目标就是本脚本，例如：
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "<项目>\dronerfa_baseline\open_signal_analyzer.ps1"
#
# 测试用法（不打开浏览器、不弹窗、临时端口、临时日志目录）：
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File dronerfa_baseline\open_signal_analyzer.ps1 `
#       -Port 8899 -NoOpenBrowser -NoMessageBox -Quiet -ReadyTimeoutSeconds 120 `
#       -LogDirectory <临时目录> -PidFile <临时目录>\service.pid
#
# 退出码：
#   0 = 成功（服务就绪；浏览器已打开，或被 -NoOpenBrowser 抑制）
#   2 = 等待服务就绪超时
#   3 = 后台启动进程提前退出或无法启动（启动失败）
#   4 = 该端口上的分析服务已运行，但模型加载失败（state=failed）
#   5 = 项目环境缺失（.venv 解释器或 launch_signal_analyzer.ps1 不存在）
#   6 = 参数不合法，或端口被其他程序占用
#
# 边界：只访问 127.0.0.1；不修改防火墙；不安装依赖；不联网下载；
#       不结束任何已运行的服务或进程（只记录自己启动的子进程 PID 供排查/测试）。

[CmdletBinding()]
param(
    [int]$Port = 8765,
    [string]$Host_ = "127.0.0.1",
    [string]$OutputRoot = "F:/DroneRFa数据集/analysis_outputs",
    [string[]]$AllowedRoot = @("F:/DroneRFa数据集", "C:/Users/Administrator/Desktop/数据集处理"),
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [int]$ReadyTimeoutSeconds = 90,
    [int]$HealthTimeoutSeconds = 2,
    [int]$HealthAttempts = 3,
    [string]$LogDirectory = "",
    [string]$PidFile = "",
    [string]$LauncherScript = "",
    [ValidateSet("chrome", "default")]
    [string]$Browser = "chrome",
    [switch]$NoOpenBrowser,
    [switch]$NoMessageBox,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ExitOk = 0
$ExitTimeout = 2
$ExitChildExited = 3
$ExitModelFailed = 4
$ExitEnvironment = 5
$ExitUsage = 6

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

if (-not $LogDirectory) { $LogDirectory = Join-Path $ScriptDir "scratch" }
if (-not $PidFile) { $PidFile = Join-Path $LogDirectory "signal_analyzer_service.pid" }
if (-not $LauncherScript) { $LauncherScript = Join-Path $ScriptDir "launch_signal_analyzer.ps1" }

$StdoutLogPath = Join-Path $LogDirectory "signal_analyzer_service_stdout.log"
$StderrLogPath = Join-Path $LogDirectory "signal_analyzer_service_stderr.log"
$LauncherLogPath = Join-Path $LogDirectory "open_signal_analyzer.log"
$BaseUrl = "http://${Host_}:$Port"
$MutexName = "Local\DroneRFaSignalAnalyzerLauncher_$Port"
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PollMilliseconds = 500

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
function Write-LauncherLog {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    $line = "[{0}] [{1}] {2}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"), $Level, $Message
    if (-not $Quiet) { Write-Host $line }
    if (-not $LauncherLogPath) { return }
    try {
        $parent = Split-Path -Parent $LauncherLogPath
        if ($parent -and -not (Test-Path -LiteralPath $parent)) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        Add-Content -LiteralPath $LauncherLogPath -Value $line -Encoding UTF8
    } catch {
        # 日志写不进去不影响启动流程
    }
}

function Get-AnalyzerHealth {
    param(
        [string]$Url,
        [int]$TimeoutSeconds
    )
    $result = [pscustomobject]@{
        Reachable     = $false
        Ok            = $false
        State         = ""
        ModelError    = ""
        Service       = ""
        UptimeSeconds = -1.0
        Detail        = ""
    }
    try {
        $response = Invoke-WebRequest -Uri ($Url + "/api/health") -UseBasicParsing -TimeoutSec $TimeoutSeconds -ErrorAction Stop
        $result.Reachable = $true
    } catch {
        $webResponse = $null
        if ($_.Exception -and $_.Exception.Response) { $webResponse = $_.Exception.Response }
        if ($webResponse -ne $null) {
            $result.Reachable = $true
            $statusText = "未知"
            try { $statusText = [string][int]$webResponse.StatusCode } catch { $statusText = "未知" }
            $result.Detail = "HTTP 状态 " + $statusText
        } else {
            $result.Detail = $_.Exception.Message
        }
        return $result
    }
    $payload = $null
    try {
        $payload = $response.Content | ConvertFrom-Json -ErrorAction Stop
    } catch {
        $result.Detail = "健康检查响应不是合法 JSON"
        return $result
    }
    if ($null -eq $payload) {
        $result.Detail = "健康检查响应为空"
        return $result
    }
    if ($payload.ok -ne $true) {
        $result.Detail = "健康检查 ok 不为 true"
        return $result
    }
    $result.Ok = $true
    if ($payload.service) { $result.Service = [string]$payload.service }
    if ($null -ne $payload.uptime_seconds) {
        try { $result.UptimeSeconds = [double]$payload.uptime_seconds } catch { $result.UptimeSeconds = -1.0 }
    }
    if ($null -ne $payload.model) {
        $result.State = [string]$payload.model.state
        if ($payload.model.error) { $result.ModelError = [string]$payload.model.error }
    }
    return $result
}

function Quote-Single {
    param([string]$Value)
    if ($null -eq $Value) { return "''" }
    return "'" + ($Value -replace "'", "''") + "'"
}

function Get-PowerShellExecutable {
    $current = $null
    try { $current = (Get-Process -Id $PID).Path } catch { $current = $null }
    if ($current -and (Test-Path -LiteralPath $current)) { return $current }
    $fallback = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
    if (Test-Path -LiteralPath $fallback) { return $fallback }
    return "powershell.exe"
}

function Get-LogTail {
    param(
        [string]$Path,
        [int]$Lines = 12
    )
    if (-not $Path) { return "" }
    if (-not (Test-Path -LiteralPath $Path)) { return "" }
    try {
        return ((Get-Content -LiteralPath $Path -Tail $Lines -ErrorAction Stop) -join "`r`n")
    } catch {
        return ""
    }
}

# 端口上是否已有进程在监听（只做本机回环 TCP 连接；不启动、不结束任何进程）。
function Test-PortListening {
    param(
        [string]$TargetHost,
        [int]$TargetPort,
        [int]$TimeoutMilliseconds = 1500
    )
    $client = $null
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $async = $client.BeginConnect($TargetHost, $TargetPort, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne($TimeoutMilliseconds, $false)) { return $false }
        $client.EndConnect($async)
        return $true
    } catch {
        return $false
    } finally {
        if ($null -ne $client) { try { $client.Close() } catch { } }
    }
}

function Find-ChromeExecutable {
    $candidates = @(
        (Join-Path $env:ProgramFiles "Google\Chrome\Application\chrome.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Google\Chrome\Application\chrome.exe"),
        (Join-Path $env:LOCALAPPDATA "Google\Chrome\Application\chrome.exe")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) { return $candidate }
    }
    try {
        $command = Get-Command "chrome.exe" -ErrorAction Stop
        if ($command.Source) { return $command.Source }
    } catch { }
    return $null
}

function Open-AnalyzerPage {
    param([string]$Url)
    if ($NoOpenBrowser) {
        Write-LauncherLog -Message "按 -NoOpenBrowser 要求，未打开浏览器；请手动访问 $Url"
        return
    }
    try {
        if ($Browser -eq "chrome") {
            $chrome = Find-ChromeExecutable
            if ($chrome) {
                Start-Process -FilePath $chrome -ArgumentList @("--new-tab", $Url) | Out-Null
                Write-LauncherLog -Message "已用 Google Chrome 打开界面：$Url"
                return
            }
            Write-LauncherLog -Level "WARN" -Message "未找到 Google Chrome，改用系统默认浏览器。"
        }
        Start-Process $Url | Out-Null
        Write-LauncherLog -Message "已用系统默认浏览器打开界面：$Url"
    } catch {
        Write-LauncherLog -Level "WARN" -Message ("打开默认浏览器失败：{0}；请手动访问 {1}" -f $_.Exception.Message, $Url)
    }
}

function Show-FailureBanner {
    param(
        [int]$Code,
        [string]$Headline,
        [string]$Detail = "",
        [string]$LogHint = ""
    )
    $message = @"
DroneRFa 分析界面一键启动失败（退出码 $Code）。

$Headline
$Detail

服务地址：$BaseUrl/
$LogHint
手动排查：在项目目录执行
  .venv\Scripts\python.exe dronerfa_baseline\signal_analyzer_server.py --port $Port
"@
    $flatDetail = ($Detail -replace "\s+", " ").Trim()
    $flatHeadline = ($Headline -replace "\s+", " ").Trim()
    $flatHint = ($LogHint -replace "\s+", " ").Trim()
    Write-LauncherLog -Level "ERROR" -Message ("启动失败 exit={0}；{1}；{2}；{3}" -f $Code, $flatHeadline, $flatDetail, $flatHint)
    if (-not $Quiet) { Write-Host $message }
    if ($NoMessageBox) { return }
    try {
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
        [System.Windows.Forms.MessageBox]::Show(
            $message,
            "DroneRFa 分析界面启动失败",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Warning
        ) | Out-Null
    } catch {
        Write-Host $message
    }
}

function New-LauncherLogHint {
    return @"
启动日志（标准输出）：$StdoutLogPath
启动日志（标准错误）：$StderrLogPath
启动器日志：$LauncherLogPath
"@
}

function Release-StarterMutex {
    param(
        $Mutex,
        [bool]$Owned
    )
    if ($null -eq $Mutex) { return }
    if ($Owned) {
        try { $Mutex.ReleaseMutex() } catch { }
    }
    try { $Mutex.Dispose() } catch { }
}

# ---------------------------------------------------------------------------
# 参数与环境检查
# ---------------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $LogDirectory)) {
    try { New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null } catch { }
}

if ($Port -lt 1 -or $Port -gt 65535) {
    Write-Host "端口不合法：$Port（应为 1-65535）。"
    exit $ExitUsage
}
if ($ReadyTimeoutSeconds -lt 1) { $ReadyTimeoutSeconds = 1 }
if ($HealthTimeoutSeconds -lt 1) { $HealthTimeoutSeconds = 1 }

if ($Host_ -ne "127.0.0.1" -and $Host_ -ne "localhost") {
    Show-FailureBanner -Code $ExitUsage -Headline "本启动器只允许本机回环地址 127.0.0.1（收到 $Host_）。" -Detail "不会修改防火墙，也不会绑定 0.0.0.0。" -LogHint (New-LauncherLogHint)
    exit $ExitUsage
}

if (-not (Test-Path -LiteralPath $PythonPath)) {
    Show-FailureBanner -Code $ExitEnvironment -Headline "未找到项目虚拟环境解释器：$PythonPath" -Detail "本界面必须使用项目 .venv（CUDA PyTorch），不会回退到 anaconda 的 CPU PyTorch。" -LogHint (New-LauncherLogHint)
    exit $ExitEnvironment
}

if (-not (Test-Path -LiteralPath $LauncherScript)) {
    Show-FailureBanner -Code $ExitEnvironment -Headline "未找到启动脚本：$LauncherScript" -Detail "请确认项目目录完整（dronerfa_baseline\launch_signal_analyzer.ps1）。" -LogHint (New-LauncherLogHint)
    exit $ExitEnvironment
}

# ---------------------------------------------------------------------------
# 1) 先做短超时健康检查（失败重试，避免服务偶尔忙时误判成“没在运行”）
# ---------------------------------------------------------------------------
$startedAt = Get-Date
Write-LauncherLog -Message ("一键启动开始：url=$BaseUrl readyTimeout=${ReadyTimeoutSeconds}s healthTimeout=${HealthTimeoutSeconds}s healthAttempts=$HealthAttempts launcher=$LauncherScript")

$health = Get-AnalyzerHealth -Url $BaseUrl -TimeoutSeconds $HealthTimeoutSeconds
$attempts = 1
while ((0 -eq $attempts -or $attempts -lt $HealthAttempts) -and (-not $health.Reachable)) {
    Start-Sleep -Milliseconds 700
    $attempts++
    $health = Get-AnalyzerHealth -Url $BaseUrl -TimeoutSeconds $HealthTimeoutSeconds
}

if ($health.Reachable -and $health.Ok -and $health.State -eq "ready") {
    Write-LauncherLog -Message ("检测到服务已就绪（uptime={0}s，model.state=ready，第 {1} 次健康检查），不启动第二个服务。" -f $health.UptimeSeconds, $attempts)
    Open-AnalyzerPage -Url ($BaseUrl + "/")
    exit $ExitOk
}

if ($health.Reachable -and $health.Ok -and $health.State -eq "failed") {
    Show-FailureBanner -Code $ExitModelFailed -Headline "该端口上的分析服务正在运行，但模型加载失败（model.state=failed）。" -Detail ("服务报告的错误：" + $health.ModelError) -LogHint (New-LauncherLogHint)
    exit $ExitModelFailed
}

if ($health.Reachable -and (-not $health.Ok)) {
    Show-FailureBanner -Code $ExitUsage -Headline "端口 $Port 已被其他程序占用，且它不是 DroneRFa 分析界面。" -Detail ("健康检查结果：" + $health.Detail + "。请关闭占用该端口的程序，或改用 -Port 指定其他端口。") -LogHint (New-LauncherLogHint)
    exit $ExitUsage
}

# 走到这里只有两种可能：服务没起来（端口空闲），或服务已在监听但暂时没响应健康检查
# （模型还在加载，或正在忙）。后者绝不能当作“没运行”，否则会拉起第二个服务。
$portListening = $false
if (-not ($health.Reachable -and $health.Ok)) {
    $portListening = Test-PortListening -TargetHost $Host_ -TargetPort $Port
}
$alreadyLoading = ($health.Reachable -and $health.Ok) -or $portListening
if ($portListening -and -not $health.Reachable) {
    Write-LauncherLog -Message "端口 $Port 已有服务在监听，但健康检查暂时没有响应（第 $attempts 次）：只等待同一个服务。"
}

# ---------------------------------------------------------------------------
# 2) 未就绪：抢命名 mutex，决定谁来启动后台服务
# ---------------------------------------------------------------------------
$mutex = $null
$isStarter = $false
$childProcess = $null

if ($alreadyLoading) {
    if ($portListening -and -not $health.Reachable) {
        Write-LauncherLog -Message "检测到端口 $Port 上已有服务在监听但健康检查无响应，只等待同一个服务，不再启动第二个。"
    } else {
        Write-LauncherLog -Message "检测到服务已在本机运行但模型尚未就绪，只等待同一个服务，不再启动第二个。"
    }
} else {
    try {
        $mutex = New-Object System.Threading.Mutex($false, $MutexName)
        $isStarter = $mutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $isStarter = $true
    } catch {
        $isStarter = $true
        $mutex = $null
        Write-LauncherLog -Level "WARN" -Message ("创建命名 mutex 失败（{0}），按启动者继续。" -f $_.Exception.Message)
    }
}

if ((-not $alreadyLoading) -and (-not $isStarter)) {
    Write-LauncherLog -Message "另一个一键启动过程已经在启动服务，本进程只等待同一个服务就绪。"
}

if ((-not $alreadyLoading) -and $isStarter) {
    $powerShellExe = Get-PowerShellExecutable
    $childCommand = "& " + (Quote-Single $LauncherScript) +
        " -Port " + $Port +
        " -Host_ " + (Quote-Single $Host_) +
        " -OutputRoot " + (Quote-Single $OutputRoot) +
        " -Device " + $Device +
        " -NoBrowser -Quiet"
    if ($AllowedRoot -and $AllowedRoot.Count -gt 0) {
        $quotedRoots = @()
        foreach ($root in $AllowedRoot) { $quotedRoots += (Quote-Single $root) }
        $childCommand += " -AllowedRoot " + ($quotedRoots -join ",")
    }
    # powershell.exe -Command 在子脚本非零退出时只会给出 1，这里显式回传真实退出码。
    $childCommand += '; exit $LASTEXITCODE'
    $childArguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $childCommand)

    Write-LauncherLog -Message ("正在后台隐藏启动分析服务：{0}" -f $powerShellExe)
    Write-LauncherLog -Message ("子进程命令行：{0}" -f $childCommand)
    try {
        $childProcess = Start-Process -FilePath $powerShellExe -ArgumentList $childArguments -WindowStyle Hidden -RedirectStandardOutput $StdoutLogPath -RedirectStandardError $StderrLogPath -PassThru
    } catch {
        Show-FailureBanner -Code $ExitChildExited -Headline ("无法在后台启动分析服务：{0}" -f $_.Exception.Message) -Detail "请检查日志目录是否可写，或手动运行 dronerfa_baseline\launch_signal_analyzer.ps1 查看原因。" -LogHint (New-LauncherLogHint)
        Release-StarterMutex -Mutex $mutex -Owned $isStarter
        exit $ExitChildExited
    }

    Write-LauncherLog -Message ("后台分析服务已启动：pid={0}；输出日志={1}；错误日志={2}" -f $childProcess.Id, $StdoutLogPath, $StderrLogPath)
    try {
        Set-Content -LiteralPath $PidFile -Value ("{0}" -f $childProcess.Id) -Encoding ASCII
    } catch {
        Write-LauncherLog -Level "WARN" -Message ("写入 PID 文件失败（{0}）：{1}" -f $PidFile, $_.Exception.Message)
    }
}

# ---------------------------------------------------------------------------
# 3) 等待模型就绪
# ---------------------------------------------------------------------------
$deadline = $startedAt.AddSeconds($ReadyTimeoutSeconds)
$ready = $false
$modelFailed = $false
$childExited = $false
$lastDetail = $health.Detail

while ($true) {
    $health = Get-AnalyzerHealth -Url $BaseUrl -TimeoutSeconds $HealthTimeoutSeconds
    if ($health.Reachable -and $health.Ok -and $health.State -eq "ready") { $ready = $true; break }
    if ($health.Reachable -and $health.Ok -and $health.State -eq "failed") { $modelFailed = $true; break }
    if (-not $health.Reachable) { $lastDetail = $health.Detail }
    if ($null -ne $childProcess -and $childProcess.HasExited) { $childExited = $true; break }
    if ((Get-Date) -ge $deadline) { break }
    Start-Sleep -Milliseconds $PollMilliseconds
}

$elapsedSeconds = ((Get-Date) - $startedAt).TotalSeconds
$exitCode = $ExitOk

if ($ready) {
    Write-LauncherLog -Message ("服务已就绪（等待 {0:N1} 秒）：{1}/" -f $elapsedSeconds, $BaseUrl)
    Open-AnalyzerPage -Url ($BaseUrl + "/")
} elseif ($modelFailed) {
    Show-FailureBanner -Code $ExitModelFailed -Headline "分析服务已启动，但模型加载失败（model.state=failed）。" -Detail ("服务报告的错误：" + $health.ModelError) -LogHint (New-LauncherLogHint)
    $exitCode = $ExitModelFailed
} elseif ($childExited) {
    $childCode = "未知"
    try {
        if ($childProcess.HasExited) { $childProcess.WaitForExit() }
    } catch { }
    try {
        $exitValue = $childProcess.ExitCode
        if ($null -ne $exitValue -and ("{0}" -f $exitValue) -ne "") { $childCode = [string]$exitValue }
    } catch {
        $childCode = "未知"
    }
    $stderrTail = Get-LogTail -Path $StderrLogPath -Lines 12
    $detail = "后台启动进程已提前退出（退出码 $childCode）。"
    if ($stderrTail) { $detail += "`r`n错误日志末尾：`r`n$stderrTail" }
    Show-FailureBanner -Code $ExitChildExited -Headline $detail -Detail "" -LogHint (New-LauncherLogHint)
    $exitCode = $ExitChildExited
} else {
    $detail = "已等待 $ReadyTimeoutSeconds 秒，服务仍未就绪。"
    if ($portListening -and -not $health.Reachable) {
        $detail += "端口 $Port 上已有服务在监听（未再启动第二个服务），但健康检查一直没有响应，可能正在加载模型或正忙；请稍后刷新页面。"
    } elseif ($alreadyLoading) {
        $detail += "服务进程已在运行但模型还没加载完，请稍后刷新页面。"
    } elseif (-not $isStarter) {
        $detail += "另一个启动过程仍在启动服务，请稍后刷新页面。"
    } else {
        $detail += "后台服务仍在启动中（首次加载 PyTorch/CUDA 较慢），稍后可刷新页面；若长时间不成功请查看日志。"
    }
    if ($lastDetail) { $detail += "`r`n最后一次健康检查：" + $lastDetail }
    Show-FailureBanner -Code $ExitTimeout -Headline $detail -Detail "" -LogHint (New-LauncherLogHint)
    $exitCode = $ExitTimeout
}

Release-StarterMutex -Mutex $mutex -Owned $isStarter
exit $exitCode
