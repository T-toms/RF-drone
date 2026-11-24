# DroneRFa 任意IQ信号分析界面 —— 启动脚本
# 只监听 127.0.0.1，不修改防火墙，不绑定 0.0.0.0，不安装任何新依赖。
#
# 用法：
#   pwsh -File dronerfa_baseline\launch_signal_analyzer.ps1
#   pwsh -File dronerfa_baseline\launch_signal_analyzer.ps1 -Port 8899 -NoBrowser
#   pwsh -File dronerfa_baseline\launch_signal_analyzer.ps1 -OutputRoot "C:\Users\Administrator\Desktop\数据集处理\dronerfa_baseline\scratch\analysis_outputs"
[CmdletBinding()]
param(
    [int]$Port = 8765,
    [string]$Host_ = "127.0.0.1",
    [string]$OutputRoot = "F:/DroneRFa数据集/analysis_outputs",
    [string[]]$AllowedRoot = @("F:/DroneRFa数据集", "C:/Users/Administrator/Desktop/数据集处理"),
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [switch]$NoBrowser,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $projectDir
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "未找到项目虚拟环境解释器：$python`n本界面必须使用项目 .venv（CUDA PyTorch），不会回退到 anaconda 的 CPU PyTorch。"
    exit 2
}

if ($Host_ -ne "127.0.0.1" -and $Host_ -ne "localhost") {
    Write-Error "本界面只允许监听 127.0.0.1（拒绝 $Host_）；不会修改防火墙，也不会绑定 0.0.0.0。"
    exit 3
}

$serverScript = Join-Path $projectDir "signal_analyzer_server.py"
$arguments = @(
    $serverScript,
    "--host", $Host_,
    "--port", $Port,
    "--output-root", $OutputRoot,
    "--device", $Device
)
foreach ($root in $AllowedRoot) {
    $arguments += @("--allowed-root", $root)
}
if (-not $NoBrowser) { $arguments += "--open-browser" }
if ($Quiet) { $arguments += "--quiet" }

Write-Host "使用解释器：$python"
Write-Host "界面地址：http://${Host_}:$Port/"
Write-Host "输出根目录：$OutputRoot"
Write-Host "允许根目录：$($AllowedRoot -join '；')"
Write-Host "按 Ctrl+C 停止服务。"
Write-Host ""

& $python @arguments
exit $LASTEXITCODE
