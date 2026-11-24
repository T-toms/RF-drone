[CmdletBinding()]
param(
    [int]$Port = 8765,
    [string]$Host_ = "127.0.0.1",
    [string]$OutputRoot = "F:/DroneRFa数据集/analysis_outputs",
    [string[]]$AllowedRoot = @("F:/DroneRFa数据集", "C:/Users/Administrator/Desktop/数据集处理"),
    [ValidateSet("auto","cpu","cuda")][string]$Device = "auto",
    [int]$ExitCode = 0
)
"port=$Port host=$Host_ output=$OutputRoot device=$Device allowedCount=$($AllowedRoot.Count)"
foreach ($r in $AllowedRoot) { "allowed=[$r]" }
exit $ExitCode
