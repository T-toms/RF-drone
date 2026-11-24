"""One-off fixup: make launch_signal_analyzer.ps1 safe for Windows PowerShell 5.1.

Windows PowerShell reads .ps1 files without a BOM using the ANSI code page, which
corrupts the Chinese messages; PowerShell 7 additionally rejects ``"$Host_:..."``
because ``:`` cannot terminate a variable name.  This script rewrites the launcher as
UTF-8 with BOM and uses the braced variable form.  Kept as an audit record of the fix.

    .venv\\Scripts\\python.exe dronerfa_baseline\\_fix_launcher_encoding.py
"""
from __future__ import annotations

from pathlib import Path

LAUNCHER = Path(__file__).resolve().parent / "launch_signal_analyzer.ps1"
OLD = "http://$Host_:$Port/"
NEW = "http://${Host_}:$Port/"


def main() -> int:
    text = LAUNCHER.read_text(encoding="utf-8-sig")
    replacements = text.count(OLD)
    text = text.replace(OLD, NEW)
    LAUNCHER.write_text(text, encoding="utf-8-sig")
    raw = LAUNCHER.read_bytes()
    print(f"replacements={replacements}")
    print(f"bom={raw[:3]!r}")
    print(f"braced_variable_present={NEW in LAUNCHER.read_text(encoding='utf-8-sig')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
