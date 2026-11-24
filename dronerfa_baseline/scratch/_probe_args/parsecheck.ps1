param([string]$Target)
$tokens = $null
$errors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile($Target, [ref]$tokens, [ref]$errors)
"ps_version=" + $PSVersionTable.PSVersion.ToString()
"parse_errors=" + $errors.Count
foreach ($e in $errors) { "ERR line " + $e.Extent.StartLineNumber + ": " + $e.Message }
$raw = [System.IO.File]::ReadAllBytes($Target)
"has_bom=" + ($raw[0] -eq 0xEF -and $raw[1] -eq 0xBB -and $raw[2] -eq 0xBF)
$text = [System.IO.File]::ReadAllText($Target, (New-Object System.Text.UTF8Encoding($true)))
$needle = [string]([char]0x4E00) + [string]([char]0x952E)        # 一键
$needle2 = [string]([char]0x5931) + [string]([char]0x8D25)       # 失败
"cjk_ok=" + ($text.Contains($needle) -and $text.Contains($needle2))
"tokens=" + $tokens.Count
