$dir = $PSScriptRoot
$result = Join-Path $dir "probe3_result.txt"
$lines = @()
try {
    $p = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", "echo innerok > `"$dir\inner.log`"") -WindowStyle Hidden -PassThru -Wait
    $lines += "started pid=$($p.Id) exit=[$($p.ExitCode)]"
} catch {
    $lines += "START-EXCEPTION: " + $_.Exception.Message
}
$inner = Join-Path $dir "inner.log"
if (Test-Path -LiteralPath $inner) { $lines += "inner.log=[" + (Get-Content -LiteralPath $inner -Raw) + "]" } else { $lines += "inner.log missing" }
try {
    $o = Join-Path $dir "probe3_out.log"
    $e = Join-Path $dir "probe3_err.log"
    $p2 = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-Command", "Write-Output 'redirect-ok'") -WindowStyle Hidden -RedirectStandardOutput $o -RedirectStandardError $e -PassThru -Wait
    $lines += "redirect started pid=$($p2.Id) exit=[$($p2.ExitCode)]"
    if (Test-Path -LiteralPath $o) { $lines += "probe3_out=[" + (Get-Content -LiteralPath $o -Raw) + "]" } else { $lines += "probe3_out missing" }
} catch {
    $lines += "REDIRECT-EXCEPTION: " + $_.Exception.Message
}
Set-Content -LiteralPath $result -Value $lines -Encoding UTF8
Get-Content -LiteralPath $result -Encoding UTF8
