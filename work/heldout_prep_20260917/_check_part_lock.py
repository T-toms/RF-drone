"""Check whether the .part breakpoint is currently held open by another process.

On Windows a process writing the file holds a handle; opening it for append
without FILE_SHARE_WRITE fails with a sharing violation.  That is a direct test
for "the old downloader is still alive", independent of process listing.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

PART = Path(r"F:\DroneRFa数据集\heldout_20260917\downloads\T0000_D00_S0100.mat.rar.part")
STATE = Path(r"F:\DroneRFa数据集\heldout_20260917\downloads\T0000_D00_S0100.mat.rar.state.json")

GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
CreateFileW = kernel32.CreateFileW
CreateFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
]
CreateFileW.restype = wintypes.HANDLE
CloseHandle = kernel32.CloseHandle


def try_open(path: Path, share_write: bool) -> tuple[bool, str]:
    share = 0x00000001 | (0x00000002 if share_write else 0)  # READ | WRITE
    handle = CreateFileW(
        str(path),
        GENERIC_READ | GENERIC_WRITE,
        share,
        None,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        None,
    )
    if handle == INVALID_HANDLE_VALUE:
        return False, f"WinError {ctypes.get_last_error()}"
    CloseHandle(handle)
    return True, "opened"


print("part exists:", PART.exists(), "size:", PART.stat().st_size if PART.exists() else None)
print("state exists:", STATE.exists())
if STATE.exists():
    print("state:", STATE.read_text(encoding="utf-8").strip())

# Exclusive open: fails if any other process holds the file.
ok, why = try_open(PART, share_write=False)
print(f"exclusive open of .part -> {'OK (no other writer)' if ok else 'BLOCKED: ' + why}")

# Append-mode open, which is what the downloader itself does.
ok2, why2 = try_open(PART, share_write=True)
print(f"shared open of .part  -> {'OK' if ok2 else 'BLOCKED: ' + why2}")

sys.exit(0 if ok and ok2 else 1)
