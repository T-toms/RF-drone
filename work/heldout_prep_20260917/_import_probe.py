"""Diagnose why the wrapper cannot import download_dronerfa_subset."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET = SCRIPT_DIR.parent
print("script dir :", SCRIPT_DIR)
print("target root:", TARGET)
print("root exists:", TARGET.is_dir())
found = sorted(p.name for p in TARGET.glob("download_*.py"))
print("candidates in root:", found)
print("sys.path[0:5]:", sys.path[:5])
sys.path.insert(0, str(TARGET))
print("after insert, path[0:3]:", sys.path[:3])
try:
    import download_dronerfa_subset as base  # noqa: F401

    print("import OK:", base.__file__)
except Exception as exc:  # noqa: BLE001
    print("import FAILED:", type(exc).__name__, exc)
