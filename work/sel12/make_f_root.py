#!/usr/bin/env python3
"""SELECT12-F1 step 3: create the protected root on the new F: volume.

Creates ``F:\\DroneRFa12机型`` with its sub-directories and a volume marker that
records the UUID plus the physical volume identity (label, filesystem, serial
number, capacity) of the drive that was mounted as ``F:`` when the root was made.

The marker is the guard every later operation re-checks: a mount change that
silently re-points ``F:`` at another volume makes the recorded serial number stop
matching, and every writer refuses to continue.

Refuses to touch an existing marker with a different UUID.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import uuid
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "work" / "sel12" / "sel12-config-20260924.json"

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_GetVolumeInformationW = _kernel32.GetVolumeInformationW
_GetVolumeInformationW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.LPWSTR,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    ctypes.POINTER(wintypes.DWORD),
    ctypes.POINTER(wintypes.DWORD),
    wintypes.LPWSTR,
    wintypes.DWORD,
]
_GetVolumeInformationW.restype = wintypes.BOOL

_GetDiskFreeSpaceExW = _kernel32.GetDiskFreeSpaceExW
_GetDiskFreeSpaceExW.argtypes = [
    wintypes.LPCWSTR,
    ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.POINTER(ctypes.c_ulonglong),
]
_GetDiskFreeSpaceExW.restype = wintypes.BOOL


def volume_identity(drive: str) -> dict:
    """Return the physical identity of a mounted volume (``drive`` like ``F:``)."""
    root = drive if drive.endswith("\\") else drive + "\\"
    label = ctypes.create_unicode_buffer(261)
    fs_name = ctypes.create_unicode_buffer(261)
    serial = wintypes.DWORD(0)
    max_len = wintypes.DWORD(0)
    flags = wintypes.DWORD(0)
    ok = _GetVolumeInformationW(
        root, label, 261, ctypes.byref(serial), ctypes.byref(max_len),
        ctypes.byref(flags), fs_name, 261,
    )
    if not ok:
        raise OSError(ctypes.get_last_error(), f"GetVolumeInformationW 失败：{root}")
    free_caller = ctypes.c_ulonglong(0)
    total = ctypes.c_ulonglong(0)
    total_free = ctypes.c_ulonglong(0)
    ok = _GetDiskFreeSpaceExW(root, ctypes.byref(free_caller), ctypes.byref(total),
                              ctypes.byref(total_free))
    if not ok:
        raise OSError(ctypes.get_last_error(), f"GetDiskFreeSpaceExW 失败：{root}")
    return {
        "driveLetter": drive,
        "volumeLabel": label.value,
        "fileSystem": fs_name.value,
        "serialNumber": f"{serial.value:08X}",
        "totalBytes": int(total.value),
        "freeBytes": int(total_free.value),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    root = Path(paths["f_root"])

    identity = volume_identity("F:")
    print(json.dumps({"detected_volume": identity}, ensure_ascii=False, indent=2))

    marker_path = Path(paths["f_marker"])
    if marker_path.exists():
        existing = json.loads(marker_path.read_text(encoding="utf-8"))
        print(json.dumps({"existing_marker": existing}, ensure_ascii=False, indent=2))
        if existing.get("serialNumber") != identity["serialNumber"]:
            print("F盘卷序列号与既有标记不符，拒绝继续。", file=sys.stderr)
            return 3
        if existing.get("totalBytes") != identity["totalBytes"]:
            print("F盘容量与既有标记不符，拒绝继续。", file=sys.stderr)
            return 3
        print("卷标记已存在且与当前F盘一致，无需重建。")
        return 0

    if args.dry_run:
        print("--dry-run：未创建任何目录或标记。")
        return 0

    for path in (root, Path(paths["f_original"]), Path(paths["f_records"]),
                 Path(paths["f_scratch"])):
        path.mkdir(parents=True, exist_ok=True)

    marker = {
        "schema": "dronerfa-select12-volume-marker/v1",
        "uuid": str(uuid.uuid4()),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "createdAtLocal": datetime.now().astimezone().isoformat(),
        "rootDirectory": str(root),
        "purpose": "SELECT12-F1：12机型（12种无人机+背景）DroneRFa数据的受保护目标目录",
        **identity,
    }
    temporary = marker_path.with_name(marker_path.name + ".tmp")
    temporary.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    with temporary.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, marker_path)
    print(json.dumps({"marker_written": str(marker_path), "marker": marker},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
