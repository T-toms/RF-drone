# 测试用假启动脚本：只记录收到的参数并退出，不启动任何服务。
[CmdletBinding()]
param(
    [int]$Port = 0,
    [string]$Host_ = '',
    [string]$OutputRoot = '',
    [string[]]$AllowedRoot = @(),
    [string]$Device = '',
    [switch]$NoBrowser,
    [switch]$Quiet
)
Start-Sleep -Seconds 1
$payload = [pscustomobject]@{
    port = $Port
    host = $Host_
    output_root = $OutputRoot
    allowed_root = $AllowedRoot
    device = $Device
    no_browser = [bool]$NoBrowser
    quiet = [bool]$Quiet
}
$payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'stub_params.json') -Encoding UTF8
[Console]::Error.WriteLine('stub: 故意让后台启动失败')
exit 5
