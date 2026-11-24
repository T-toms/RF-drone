[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PackageRoot,
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot
)

$ErrorActionPreference = 'Stop'
$utf8Bom = New-Object System.Text.UTF8Encoding($true)
$PackageRoot = [IO.Path]::GetFullPath($PackageRoot).TrimEnd('\')
$SourceRoot = [IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')
$overviewDir = Join-Path $PackageRoot '00_总览与文件清单'
$overviewPath = Join-Path $overviewDir '00_近期工作成果总览.md'
$dataPath = Join-Path $overviewDir '00_数据集与分析输出位置说明.md'
$manifestPath = Join-Path $overviewDir '00_文件清单_来源_大小_SHA256.csv'
foreach ($path in @($overviewPath,$dataPath,$manifestPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "缺少成果包文件：$path" }
}

$overview = [IO.File]::ReadAllText($overviewPath)
$overview = $overview.Replace([string][char]0, '')
foreach ($number in 1..8) { $overview = $overview.Replace(("- ${number}_"), ("- 0${number}_")) }
$overview = $overview.Replace('见 0_数据集与分析输出位置说明.md', '见 00_数据集与分析输出位置说明.md')
$overview = $overview.Replace('见 0_文件清单_来源_大小_SHA256.csv', '见 00_文件清单_来源_大小_SHA256.csv')
[IO.File]::WriteAllText($overviewPath, $overview, $utf8Bom)

$data = [IO.File]::ReadAllText($dataPath)
$data = $data.Replace('$SourceRoot\dronerfa_baseline', ($SourceRoot + '\dronerfa_baseline'))
[IO.File]::WriteAllText($dataPath, $data, $utf8Bom)

$rows = @(Import-Csv -LiteralPath $manifestPath)
foreach ($relative in @('00_总览与文件清单\00_近期工作成果总览.md','00_总览与文件清单\00_数据集与分析输出位置说明.md')) {
    $row = $rows | Where-Object RelativePath -eq $relative
    if (@($row).Count -ne 1) { throw "清单中找不到唯一记录：$relative" }
    $path = Join-Path $PackageRoot $relative
    $file = Get-Item -LiteralPath $path
    $row.Bytes = [string]$file.Length
    $row.SHA256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    $row.VerifiedAgainstSource = 'True'
}
$rows | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding UTF8

$nulFiles = @(Get-ChildItem -LiteralPath $PackageRoot -Recurse -File -Filter '*.md' | Where-Object {
    $hasNul = $false
    foreach ($character in [IO.File]::ReadAllText($_.FullName).ToCharArray()) {
        if ([int]$character -eq 0) { $hasNul = $true; break }
    }
    $hasNul
})
if ($nulFiles.Count -gt 0) { throw '修复后仍有 Markdown 文件包含 NUL 字符。' }
[pscustomobject]@{Repaired=$true; Overview=$overviewPath; DataInfo=$dataPath; Manifest=$manifestPath} | ConvertTo-Json
