try {
    $p = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', 'exit 0') -WindowStyle Hidden -PassThru -Wait
    if ($p.ExitCode -eq 0) {
        $text = 'ok 0'
    } else {
        $text = 'blocked 子进程退出码 ' + $p.ExitCode
    }
} catch {
    $text = 'blocked ' + $_.Exception.Message
}
Set-Content -LiteralPath (Join-Path $PSScriptRoot 'result.txt') -Value $text -Encoding UTF8
