[CmdletBinding()]
param([switch]$Execute)

$ErrorActionPreference = 'Stop'
$project = 'C:\Users\Administrator\Desktop\数据集处理'
$downloads = 'G:\DroneRFa数据集\download_records_12models_20260922\downloads'
$original = 'G:\DroneRFa数据集\original'
$manifest = 'G:\DroneRFa数据集\dataset_manifest.csv'
$statePath = 'G:\DroneRFa数据集\download_records_12models_20260922\select12g-state.json'
$hashPath = Join-Path $project 'work\sel12g\sel12g-mat-sha256-20260925.json'
$auditPath = Join-Path $project 'work\sel12g\restart-cleanup-audit-20260925.json'
$expectedManifestHash = '9765771D84C6DFC6B5744FD73AD548B1EAB5EB9E2A1AE35952D54C0CBE15644D'

if ((Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash -ne $expectedManifestHash) {
    throw 'G manifest changed; refusing cleanup'
}
if (Test-Path -LiteralPath 'G:\DroneRFa数据集\download_records_12models_20260922\select12g.lock') {
    throw 'Downloader lock exists; refusing cleanup'
}
$state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
$hashReport = Get-Content -LiteralPath $hashPath -Raw -Encoding UTF8 | ConvertFrom-Json
$hashes = @{}
foreach ($item in $hashReport.files) { $hashes[$item.filename] = $item }
$base = (Resolve-Path -LiteralPath $downloads).Path.TrimEnd('\')
$files = @(Get-ChildItem -LiteralPath $base -Filter '*.rar.part.restart' -File)
$records = @()
foreach ($file in $files) {
    $resolved = (Resolve-Path -LiteralPath $file.FullName).Path
    if (-not $resolved.StartsWith($base + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unexpected file location: $resolved"
    }
    $filename = $file.Name -replace '\.rar\.part\.restart$', ''
    $property = $state.files.PSObject.Properties[$filename]
    if (-not $property -or $property.Value.status -ne 'completed') {
        throw "Restart block is not completed: $filename"
    }
    if ([int64]$file.Length -ne [int64]$property.Value.archive_block_size) {
        throw "Restart block size mismatch: $filename"
    }
    if (-not $hashes.ContainsKey($filename)) { throw "MAT hash evidence missing: $filename" }
    $mat = Join-Path $original $filename
    if (-not (Test-Path -LiteralPath $mat)) { throw "MAT missing: $filename" }
    if ([int64](Get-Item -LiteralPath $mat).Length -ne [int64]$hashes[$filename].bytes) {
        throw "MAT size differs from hash report: $filename"
    }
    $records += [ordered]@{file=$resolved; bytes=[int64]$file.Length; mat=$mat; mat_sha256=$hashes[$filename].sha256}
}
$total = [int64]0
foreach ($record in $records) { $total += [int64]$record.bytes }
if ($records.Count -ne 42 -or $total -ne 42597539738) {
    throw "Expected exactly 42 completed blocks / 42597539738 bytes; found $($records.Count) / $total"
}
$audit = [ordered]@{
    checked_at = (Get-Date).ToString('o')
    manifest_sha256 = $expectedManifestHash.ToLowerInvariant()
    blocks = $records.Count
    bytes = $total
    action = if ($Execute) { 'delete_completed_restart_blocks' } else { 'dry_run' }
    old_partial_files_preserved = @(Get-ChildItem -LiteralPath $base -Filter '*.rar.part' -File).Count
    files = $records
}
$audit | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $auditPath -Encoding UTF8
if ($Execute) {
    foreach ($record in $records) { Remove-Item -LiteralPath $record.file -Force }
    $remaining = @(Get-ChildItem -LiteralPath $base -Filter '*.rar.part.restart' -File)
    if ($remaining.Count -ne 0) { throw "$($remaining.Count) completed restart blocks remain" }
    $audit.completed_at = (Get-Date).ToString('o')
    $audit.remaining_restart_blocks = 0
    $audit | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $auditPath -Encoding UTF8
}
Write-Output "blocks=$($records.Count) bytes=$total action=$($audit.action) remaining=$(@(Get-ChildItem -LiteralPath $base -Filter '*.rar.part.restart' -File).Count)"
