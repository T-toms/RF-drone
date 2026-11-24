$ErrorActionPreference = 'Stop'
$root = 'C:\Users\Administrator\Desktop\数据集处理'
$inputFile = Join-Path $root 'reports\DroneRFa进展汇报_导师版.docx'
$outputDir = Join-Path $root 'work\advisor_report_native_render'
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$pdf = Join-Path $outputDir 'report.pdf'
$app = $null
$document = $null
try {
    $app = New-Object -ComObject KWPS.Application
    $app.Visible = $false
    $app.DisplayAlerts = 0
    $document = $app.Documents.Open($inputFile, $false, $true)
    $document.Repaginate()
    $pages = $document.ComputeStatistics(2)
    $document.ExportAsFixedFormat($pdf, 17)
    [pscustomobject]@{pages=$pages;pdf=$pdf} | ConvertTo-Json
} finally {
    if ($null -ne $document) { $document.Close(0) }
    if ($null -ne $app) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app) }
}
