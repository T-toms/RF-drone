param([string[]]$AllowedRoot = @("default"))
"count=$($AllowedRoot.Count)"
foreach ($r in $AllowedRoot) { "item=[$r]" }
