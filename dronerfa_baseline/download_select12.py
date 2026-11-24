#!/usr/bin/env python3
"""SELECT12-F1 step 4: resumable, range-exact downloader for the missing MAT files.

Why a new module instead of re-running ``download_heldout_files.py``
--------------------------------------------------------------------
The previous acceptances rejected the older stack for three specific reasons:

1. ``output.write(payload)`` return values were not checked (a short write could
   be silently treated as a complete block);
2. the per-block state was saved before the payload was durable (no ``fsync``),
   so a crash could leave a state file pointing past the real data;
3. disk space was only checked once, before the round, and the volume identity
   was never re-checked while running.

This module fixes all three and keeps the parts that were independently verified:
the RAR5 wrapper layout, ``UnRAR t`` / extraction / HDF5 checks, the archive index
semantics and the 206/Content-Range/ETag protocol rules.

Durability model
----------------
* The selected-entry RAR is written **strictly sequentially** into one contiguous
  ``<name>.rar.part`` (downloads run ahead in a pool, commits happen in order).
* Every block is written, ``flush``ed and ``fsync``ed, then the real on-disk size
  is re-``stat``ed and must equal the expected size.  Any mismatch is fatal for
  that file.
* Only after the data is durable is ``<name>.rar.state.json`` written atomically
  (tmp + ``fsync`` + ``os.replace``) and the directory entry synced.
* On resume the part file is truncated to the committed byte count, so an
  unrecorded tail can never be treated as valid data.

Protocol rules
--------------
Every response must be ``206`` with an exact ``Content-Range: bytes a-b/total``
matching the request, ``total`` equal to the frozen archive size, ``ETag`` equal
to the value captured while the run was healthy, and a body of exactly ``b-a+1``
bytes.  A changed ETag/size or a non-206 answer aborts the whole run.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import http.client
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "work" / "sel12" / "sel12-config-20260924.json"
DEFAULT_UNRAR = Path(r"C:\Program Files\WinRAR\UnRAR.exe")
REQUIRED_KEYS = ("RF0_I", "RF0_Q", "RF1_I", "RF1_Q")
RAR5_SIGNATURE = bytes.fromhex("526172211a070100")
HEAD_FILE = 2
HEAD_ENDARC = 5
RETRYABLE_HTTP = (408, 425, 429, 500, 502, 503, 504)
SPACE_MARGIN_BYTES = 5 * 1024**3

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_GetVolumeInformationW = _kernel32.GetVolumeInformationW
_GetVolumeInformationW.argtypes = [
    wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
    ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR,
    wintypes.DWORD,
]
_GetVolumeInformationW.restype = wintypes.BOOL
_GetDiskFreeSpaceExW = _kernel32.GetDiskFreeSpaceExW
_GetDiskFreeSpaceExW.argtypes = [
    wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong),
]
_GetDiskFreeSpaceExW.restype = wintypes.BOOL


class Select12Error(RuntimeError):
    """Expected, user-facing failure."""


class ProtocolViolation(Select12Error):
    """The server answered a range request in a way that cannot be trusted."""


class VolumeGuardError(Select12Error):
    """The target volume is not the volume this run was frozen against."""


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)


def sha256_file(path: Path, chunk: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def volume_identity(drive: str) -> dict:
    root = drive if drive.endswith("\\") else drive + "\\"
    label = ctypes.create_unicode_buffer(261)
    fs_name = ctypes.create_unicode_buffer(261)
    serial = wintypes.DWORD(0)
    max_len = wintypes.DWORD(0)
    flags = wintypes.DWORD(0)
    if not _GetVolumeInformationW(root, label, 261, ctypes.byref(serial),
                                  ctypes.byref(max_len), ctypes.byref(flags), fs_name, 261):
        raise OSError(ctypes.get_last_error(), f"GetVolumeInformationW 失败：{root}")
    free_caller = ctypes.c_ulonglong(0)
    total = ctypes.c_ulonglong(0)
    total_free = ctypes.c_ulonglong(0)
    if not _GetDiskFreeSpaceExW(root, ctypes.byref(free_caller), ctypes.byref(total),
                                ctypes.byref(total_free)):
        raise OSError(ctypes.get_last_error(), f"GetDiskFreeSpaceExW 失败：{root}")
    return {
        "driveLetter": drive,
        "volumeLabel": label.value,
        "fileSystem": fs_name.value,
        "serialNumber": f"{serial.value:08X}",
        "totalBytes": int(total.value),
        "freeBytes": int(total_free.value),
    }


def write_json_atomic(path: Path, data: object) -> None:
    """tmp -> flush -> fsync -> os.replace -> directory sync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def sync_directory(path: Path) -> None:
    """Best-effort directory metadata sync so a rename survives a crash."""
    try:
        handle = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "dronerfa-select12-plan/v1":
        raise Select12Error(f"不是SELECT12配置：{path}")
    return config


class VolumeGuard:
    """Re-checks the marker and the real volume identity at a fixed cadence."""

    def __init__(self, marker_path: Path, drive: str = "F:", interval_seconds: float = 300.0):
        self.marker_path = marker_path
        self.marker = json.loads(marker_path.read_text(encoding="utf-8"))
        self.drive = drive
        self.interval = interval_seconds
        self._last_check = 0.0
        self.checks = 0
        self.check(force=True)

    def check(self, *, force: bool = False) -> dict:
        now = time.monotonic()
        if not force and now - self._last_check < self.interval:
            return {}
        identity = volume_identity(self.drive)
        if identity["serialNumber"] != self.marker.get("serialNumber"):
            raise VolumeGuardError(
                f"{self.drive}卷序列号{identity['serialNumber']}与标记"
                f"{self.marker.get('serialNumber')}不符，立即停止"
            )
        if identity["totalBytes"] != self.marker.get("totalBytes"):
            raise VolumeGuardError(
                f"{self.drive}容量{identity['totalBytes']}与标记"
                f"{self.marker.get('totalBytes')}不符，立即停止"
            )
        self._last_check = now
        self.checks += 1
        return identity


class SingleInstanceLock:
    """O_CREAT|O_EXCL lock file; a stale lock whose PID is gone is reclaimed."""

    def __init__(self, path: Path):
        self.path = path

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                holder = self._read_holder()
                if attempt == 0 and not self._holder_alive(holder):
                    log(f"发现残留锁（PID {holder.get('pid')} 已不存在），回收后继续")
                    self.path.unlink(missing_ok=True)
                    continue
                raise Select12Error(f"已有实例在运行：{holder}") from None
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "started_at": utc_now()}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            return
        raise Select12Error("无法获得单实例锁")

    def _read_holder(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _holder_alive(holder: dict) -> bool:
        pid = holder.get("pid")
        if not isinstance(pid, int):
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        except PermissionError:
            return True
        return True

    def release(self) -> None:
        self.path.unlink(missing_ok=True)


def parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


class HttpRangeClient:
    """One HTTPS connection per request; strict 206 / Content-Range / ETag checks."""

    def __init__(self, url: str, expected_total: int, expected_etag: str,
                 *, timeout: float = 300.0, retries: int = 5):
        parts = urlsplit(url)
        self.host = parts.netloc
        self.target = parts.path + ("?" + parts.query if parts.query else "")
        self.url = url
        self.expected_total = expected_total
        self.expected_etag = expected_etag
        self.timeout = timeout
        self.retries = retries
        self.ssl_context = ssl.create_default_context()
        self.stats = {
            "range_requests": 0,
            "retries": 0,
            "bytes_transferred": 0,
            "protocol_violations": [],
            "http_status_counts": {},
        }

    def _fetch_once(self, start: int, end: int) -> tuple[bytes, dict, int]:
        connection = http.client.HTTPSConnection(
            self.host, timeout=self.timeout, context=self.ssl_context
        )
        try:
            connection.request(
                "GET",
                self.target,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                    "User-Agent": "DroneRFa-Select12-Downloader/1.0",
                    "Connection": "close",
                },
            )
            response = connection.getresponse()
            status = response.status
            headers = {
                "content-range": response.getheader("Content-Range"),
                "content-length": response.getheader("Content-Length"),
                "etag": response.getheader("ETag"),
                "content-encoding": response.getheader("Content-Encoding"),
                "accept-ranges": response.getheader("Accept-Ranges"),
            }
            body = response.read()
            return body, headers, status
        finally:
            connection.close()

    def _validate(self, start: int, end: int, body: bytes, headers: dict, status: int) -> None:
        expected = end - start + 1
        if status != 206:
            raise Select12Error(
                f"HTTP {status}（期望206），stderr/区间{start}-{end}"
            )
        encoding = (headers.get("content-encoding") or "identity").strip().lower()
        if encoding not in ("", "identity"):
            raise ProtocolViolation(f"意外的Content-Encoding：{encoding!r}")
        parsed = parse_content_range(headers.get("content-range"))
        if parsed is None:
            raise ProtocolViolation(f"Content-Range不可解析：{headers.get('content-range')!r}")
        actual_start, actual_end, total = parsed
        if (actual_start, actual_end) != (start, end):
            raise ProtocolViolation(
                f"Content-Range区间不符：请求{start}-{end}，返回{actual_start}-{actual_end}"
            )
        if total != self.expected_total:
            raise ProtocolViolation(f"归档总长{total}与冻结值{self.expected_total}不符")
        etag = headers.get("etag")
        if etag != self.expected_etag:
            raise ProtocolViolation(f"ETag变化：{etag!r} != {self.expected_etag!r}")
        declared = headers.get("content-length")
        if declared is not None and int(declared) != expected:
            raise ProtocolViolation(f"Content-Length {declared} 与请求长度 {expected} 不符")
        if len(body) != expected:
            raise ProtocolViolation(f"正文{len(body)}字节，期望{expected}")

    def fetch(self, start: int, end: int) -> bytes:
        expected = end - start + 1
        last = ""
        for attempt in range(1, self.retries + 1):
            self.stats["range_requests"] += 1
            try:
                body, headers, status = self._fetch_once(start, end)
                self.stats["http_status_counts"][str(status)] = (
                    self.stats["http_status_counts"].get(str(status), 0) + 1
                )
                if status in RETRYABLE_HTTP:
                    last = f"HTTP {status}"
                    raise Select12Error(last)
                self._validate(start, end, body, headers, status)
                self.stats["bytes_transferred"] += len(body)
                return body
            except ProtocolViolation:
                self.stats["protocol_violations"].append(f"{start}-{end}")
                raise
            except Select12Error as exc:
                last = str(exc)
                match = re.search(r"HTTP (\d{3})", last)
                status_code = int(match.group(1)) if match else -1
                if attempt == self.retries:
                    break
                self.stats["retries"] += 1
                delay = min(60 * attempt, 300) if status_code in RETRYABLE_HTTP else min(2**attempt, 15)
                log(f"  区间{start}-{end}失败（{last}），{delay}秒后重试 {attempt}/{self.retries}")
                time.sleep(delay)
            except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
                last = f"{type(exc).__name__}: {exc}"
                if attempt == self.retries:
                    break
                self.stats["retries"] += 1
                delay = min(2**attempt, 30)
                log(f"  区间{start}-{end}网络错误（{last}），{delay}秒后重试 {attempt}/{self.retries}")
                time.sleep(delay)
        raise Select12Error(f"多次重试后仍无法下载区间 {start}-{end}：{last}")

    def probe_identity(self) -> dict:
        body, headers, status = self._fetch_once(0, 7)
        self._validate(0, 7, body, headers, status)
        if body != RAR5_SIGNATURE:
            raise ProtocolViolation(f"远程文件不是RAR5压缩包：{body.hex()}")
        return {
            "status": status,
            "content_range": headers.get("content-range"),
            "etag": headers.get("etag"),
            "signature": body.hex(),
        }


def decode_vint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(10):
        if offset >= len(data):
            raise Select12Error("RAR头部不完整")
        current = data[offset]
        offset += 1
        value |= (current & 0x7F) << shift
        if current & 0x80 == 0:
            return value, offset
        shift += 7
    raise Select12Error("RAR头部包含无效的变长整数")


def parse_block(data: bytes, position: int, archive_size: int) -> dict:
    import zlib

    if position + 4 > archive_size:
        raise Select12Error("RAR块越过文件末尾")
    crc = int.from_bytes(data[position:position + 4], "little")
    header_size, offset = decode_vint(data, position + 4)
    header_start = position + 4
    header_end = header_start + header_size
    if header_end > archive_size:
        raise Select12Error("RAR块头长度越界")
    if zlib.crc32(data[header_start:header_end]) & 0xFFFFFFFF != crc:
        raise Select12Error("RAR块头CRC不匹配")
    cursor = offset
    header_type, cursor = decode_vint(data, cursor)
    header_flags, cursor = decode_vint(data, cursor)
    extra_size = 0
    data_size = 0
    if header_flags & 0x0001:
        extra_size, cursor = decode_vint(data, cursor)
    if header_flags & 0x0002:
        data_size, cursor = decode_vint(data, cursor)
    block: dict = {
        "header_type": header_type,
        "header_flags": header_flags,
        "header_total_size": header_end - position,
        "data_size": data_size,
        "extra_size": extra_size,
    }
    if header_type == HEAD_FILE:
        cursor = header_end
        file_flags, cursor = decode_vint(data, cursor)
        unpacked_size, cursor = decode_vint(data, cursor)
        attributes, cursor = decode_vint(data, cursor)
        if file_flags & 0x0002:
            _, cursor = decode_vint(data, cursor)
        if file_flags & 0x0004:
            _, cursor = decode_vint(data, cursor)
        compression_info, cursor = decode_vint(data, cursor)
        host_os, cursor = decode_vint(data, cursor)
        name_size, cursor = decode_vint(data, cursor)
        name = data[cursor:cursor + name_size].decode("utf-8", errors="replace")
        block.update(
            {
                "file_flags": file_flags,
                "unpacked_size": unpacked_size,
                "attributes": attributes,
                "compression_info": compression_info,
                "host_os": host_os,
                "name": name,
            }
        )
    return block


def verify_wrapper(archive: Path, entry: dict) -> dict:
    size = archive.stat().st_size
    data = archive.read_bytes()
    if data[:8] != RAR5_SIGNATURE:
        raise Select12Error(f"{archive.name}: 缺少RAR5签名")
    offset = 8
    blocks: list[dict] = []
    while offset < size:
        block = parse_block(data, offset, size)
        blocks.append(block)
        offset += int(block["header_total_size"]) + int(block["data_size"])
    if offset != size:
        raise Select12Error(f"{archive.name}: RAR块链未正好到达文件末尾（{offset} != {size}）")
    file_blocks = [block for block in blocks if int(block["header_type"]) == HEAD_FILE]
    if len(file_blocks) != 1:
        raise Select12Error(f"{archive.name}: 期望1个文件块，实际{len(file_blocks)}个")
    if int(blocks[-1]["header_type"]) != HEAD_ENDARC:
        raise Select12Error(f"{archive.name}: 缺少RAR结束块")
    short = entry["filename"]
    actual_name = str(file_blocks[0]["name"]).rsplit("/", 1)[-1]
    if actual_name != short:
        raise Select12Error(f"{archive.name}: 内嵌文件名{actual_name}，预期{short}")
    if int(file_blocks[0]["data_size"]) != entry["archive_packed_size"]:
        raise Select12Error(f"{archive.name}: 打包大小与清单不一致")
    if int(file_blocks[0]["unpacked_size"]) != entry["expected_mat_bytes"]:
        raise Select12Error(f"{archive.name}: 解压大小与清单不一致")
    if int(file_blocks[0]["compression_info"]) & 0x3FFF != 0:
        raise Select12Error(f"{archive.name}: 压缩方法不是stored，无法安全复用")
    return {
        "archive_size": size,
        "block_count": len(blocks),
        "block_types": [int(block["header_type"]) for block in blocks],
        "embedded_name": actual_name,
        "packed_size": int(file_blocks[0]["data_size"]),
        "unpacked_size": int(file_blocks[0]["unpacked_size"]),
        "compression_info": int(file_blocks[0]["compression_info"]),
    }


def verify_wrapper_prefix(part: Path, entry: dict) -> bool:
    """Check a partial archive's leading blocks without needing the whole body."""
    size = part.stat().st_size
    with part.open("rb") as handle:
        head = handle.read(8)
        if head != RAR5_SIGNATURE:
            return False
        prefix = handle.read(min(size - 8, 4096))
    if len(prefix) < 16:
        return False
    offset = 0
    try:
        while offset < len(prefix):
            block = parse_block(prefix, offset, len(prefix))
            offset += int(block["header_total_size"]) + int(block["data_size"])
            if int(block["header_type"]) == HEAD_FILE:
                return str(block["name"]).rsplit("/", 1)[-1] == entry["filename"]
    except Select12Error:
        return False
    return False


def run_unrar(unrar: Path, command: str, archive: Path, destination: Path | None) -> str:
    args = [str(unrar), command, "-idq", "-p-", str(archive)]
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        args.extend(["-o+", str(destination) + os.sep])
    result = subprocess.run(args, capture_output=True, check=False)
    text = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    if result.returncode != 0:
        action = "解压" if destination is not None else "完整性测试"
        raise Select12Error(f"UnRAR {action}失败，退出码 {result.returncode}：{text[:400]}")
    return text


def inspect_mat(path: Path) -> dict:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        keys = sorted(handle.keys())
        info: dict = {
            "keys": keys,
            "missing_required": [key for key in REQUIRED_KEYS if key not in handle],
        }
        if info["missing_required"]:
            info["ok"] = False
            return info
        lengths = {key: int(handle[key].shape[0]) for key in REQUIRED_KEYS}
        dtypes = {key: str(handle[key].dtype) for key in REQUIRED_KEYS}
        shapes = {key: list(handle[key].shape) for key in REQUIRED_KEYS}
        info.update(
            {
                "channel_lengths": lengths,
                "shapes": shapes,
                "dtypes": dtypes,
                "source_samples": lengths["RF0_I"],
                "all_float": all(value.startswith("float") for value in dtypes.values()),
                "lengths_consistent": len(set(lengths.values())) == 1,
            }
        )
        probe_start = max(info["source_samples"] // 2 - 500_000, 0)
        values = handle["RF0_I"][probe_start:probe_start + 1000]
        info["read_probe"] = {
            "start": probe_start,
            "samples": int(values.shape[0]),
            "finite": bool(np.isfinite(values).all()),
        }
        info["ok"] = bool(
            info["lengths_consistent"]
            and info["source_samples"] > 0
            and info["all_float"]
            and info["read_probe"]["samples"] == 1000
            and info["read_probe"]["finite"]
        )
    return info


def source_samples_from_index(config: dict) -> None:
    return None


class Downloader:
    def __init__(self, config: dict, *, workers: int, block_bytes: int,
                 unrar: Path, dry_run: bool = False, limit_files: int = 0,
                 only: list[str] | None = None):
        self.config = config
        self.paths = config["paths"]
        self.workers = workers
        self.block_bytes = block_bytes
        self.unrar = unrar
        self.dry_run = dry_run
        self.limit_files = limit_files
        self.only = set(only or [])
        self.archive = config["archive"]
        self.f_original = Path(self.paths["f_original"])
        self.f_records = Path(self.paths["f_records"])
        self.scratch = Path(self.paths["f_scratch"])
        self.state_path = Path(self.paths["f_state"])
        self.manifest_path = Path(self.paths["f_manifest_csv"])
        self.volume = VolumeGuard(Path(self.paths["f_marker"]))
        self.lock = SingleInstanceLock(Path(self.paths["f_lock"]))
        self.client = HttpRangeClient(
            self.archive["url"], self.archive["size"], self.archive["etag"]
        )
        self.state: dict = {}

    # ---------------------------------------------------------------- state
    def load_state(self) -> None:
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.state = {
                "schema": "dronerfa-select12-state/v1",
                "volume_uuid": self.volume.marker["uuid"],
                "files": {},
                "errors": [],
                "started_at": utc_now(),
            }
        if self.state.get("volume_uuid") != self.volume.marker["uuid"]:
            raise Select12Error("状态文件属于另一个卷标记，拒绝继续")

    def save_state(self) -> None:
        self.state["updated_at"] = utc_now()
        self.state["remote_identity"] = {
            "archive_size": self.archive["size"],
            "etag": self.archive["etag"],
        }
        self.state["stats"] = self.client.stats
        write_json_atomic(self.state_path, self.state)

    # ----------------------------------------------------------- disk space
    def check_space(self, needed: int) -> int:
        free = volume_identity("F:")["freeBytes"]
        if free < needed + SPACE_MARGIN_BYTES:
            raise Select12Error(
                f"F盘可用空间不足：需要 {needed} + 余量 {SPACE_MARGIN_BYTES}，实际 {free}"
            )
        return free

    # -------------------------------------------------------------- workers
    def _download_block(self, start: int, end: int) -> bytes:
        return self.client.fetch(start, end)

    def process_file(self, entry: dict, free_bytes: int) -> dict:
        filename = entry["filename"]
        block_size = entry["archive_block_size"]
        block_offset = entry["archive_block_offset"]
        part = self.scratch / f"{filename}.rar.part"
        final = self.scratch / f"{filename}.rar"
        state_path = self.scratch / f"{filename}.rar.state.json"
        target = self.f_original / filename

        record: dict = {
            "filename": filename,
            "target_id": entry["target_id"],
            "class_name": entry["class_name"],
            "origin": entry["origin"],
            "former_partition": entry["former_partition"],
            "split_status": entry["split_status"],
            "expected_mat_bytes": entry["expected_mat_bytes"],
            "archive_block_offset": block_offset,
            "archive_block_size": block_size,
            "archive_packed_size": entry["archive_packed_size"],
            "free_bytes_before": free_bytes,
            "started_at": utc_now(),
        }

        if target.exists() and target.stat().st_size == entry["expected_mat_bytes"]:
            info = inspect_mat(target)
            if info.get("ok") and info.get("source_samples"):
                record.update(
                    {
                        "status": "skipped_existing_valid",
                        "mat_sha256": sha256_file(target),
                        "hdf5": info,
                        "finished_at": utc_now(),
                    }
                )
                return record

        self.check_space(block_size + entry["expected_mat_bytes"])

        completed = 0
        if part.exists() and state_path.exists():
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            if saved.get("archive_block_offset") != block_offset:
                raise Select12Error(f"{filename}: 断点状态与目标条目不匹配")
            if not verify_wrapper_prefix(part, entry):
                raise Select12Error(f"{filename}: 断点文件头不是目标条目，拒绝续传")
            completed = int(saved["committed_bytes"])
            if not 0 <= completed <= block_size:
                raise Select12Error(f"{filename}: 断点字节数越界：{completed}")
            with part.open("r+b") as handle:
                handle.truncate(completed)
                handle.flush()
                os.fsync(handle.fileno())
            record["resumed_from_bytes"] = completed
            log(f"{filename}: 续传，已提交 {completed}/{block_size} 字节")
        else:
            part.unlink(missing_ok=True)
            state_path.unlink(missing_ok=True)
            with part.open("wb") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            self._commit_state(state_path, entry, 0)

        local_offset = 8 + completed
        total = block_size
        started = time.monotonic()
        session_start = completed

        with part.open("r+b") as output, ThreadPoolExecutor(max_workers=self.workers) as pool:
            output.seek(local_offset)
            pending: list[tuple[int, int, object]] = []
            next_local = local_offset

            def schedule() -> bool:
                nonlocal next_local
                if next_local >= 8 + total:
                    return False
                size = min(self.block_bytes, 8 + total - next_local)
                start = block_offset + (next_local - 8)
                end = start + size - 1
                pending.append((next_local, size, pool.submit(self._download_block, start, end)))
                next_local += size
                return True

            for _ in range(self.workers):
                if not schedule():
                    break

            while pending:
                position, size, future = pending.pop(0)
                payload = future.result()
                if position != 8 + completed:
                    raise Select12Error(f"{filename}: 提交顺序错乱 {position} != {8 + completed}")
                if len(payload) != size:
                    raise Select12Error(f"{filename}: 分块长度 {len(payload)} != {size}")
                written = output.write(payload)
                if written != size:
                    raise Select12Error(f"{filename}: 短写 {written} != {size}")
                output.flush()
                os.fsync(output.fileno())
                if output.tell() != 8 + completed + size:
                    raise Select12Error(f"{filename}: 落盘位置异常 {output.tell()}")
                completed += size
                # Durability barrier: data is on the platter before any state says so.
                self._commit_state(state_path, entry, completed)
                self.volume.check()
                elapsed = max(time.monotonic() - started, 1e-6)
                speed = (completed - session_start) / elapsed
                percent = completed / total * 100
                print(
                    f"\r{filename} {percent:6.2f}% "
                    f"{completed}/{total} 字节 {speed/1048576:.2f} MiB/s",
                    end="", flush=True,
                )
                schedule()
            print()

        if part.stat().st_size != 8 + total:
            raise Select12Error(
                f"{filename}: 压缩包长度 {part.stat().st_size} != {8 + total}"
            )
        record["download_seconds"] = round(time.monotonic() - started, 1)
        record["download_bytes"] = total

        os.replace(part, final)
        sync_directory(final.parent)
        state_path.unlink(missing_ok=True)

        record["checks"] = {}
        record["checks"]["wrapper"] = verify_wrapper(final, entry)
        run_unrar(self.unrar, "t", final, None)
        record["checks"]["unrar_test"] = True

        temp_extract = self.scratch / f"extract_{filename}"
        if temp_extract.exists():
            shutil.rmtree(temp_extract)
        run_unrar(self.unrar, "x", final, temp_extract)
        produced = sorted(temp_extract.rglob("*.mat"))
        if len(produced) != 1 or produced[0].name != filename:
            raise Select12Error(f"{filename}: 解压结果异常：{[p.name for p in produced]}")
        actual_size = produced[0].stat().st_size
        if actual_size != entry["expected_mat_bytes"]:
            raise Select12Error(
                f"{filename}: 解压大小 {actual_size} != 清单 {entry['expected_mat_bytes']}"
            )
        record["checks"]["extract"] = {"bytes": actual_size}
        os.replace(produced[0], target)
        sync_directory(target.parent)
        shutil.rmtree(temp_extract, ignore_errors=True)

        info = inspect_mat(target)
        record["hdf5"] = info
        if not info.get("ok"):
            raise Select12Error(f"{filename}: HDF5结构检查未通过：{info}")
        record["mat_sha256"] = sha256_file(target)
        record["mat_bytes"] = actual_size
        record["source_samples"] = info["source_samples"]
        record["windows_1M"] = info["source_samples"] // 1_000_000
        final.unlink(missing_ok=True)
        sync_directory(final.parent)
        record["status"] = "completed"
        record["finished_at"] = utc_now()
        record["free_bytes_after"] = volume_identity("F:")["freeBytes"]
        return record

    def _commit_state(self, state_path: Path, entry: dict, committed_bytes: int) -> None:
        write_json_atomic(
            state_path,
            {
                "filename": entry["filename"],
                "archive_block_offset": entry["archive_block_offset"],
                "archive_block_size": entry["archive_block_size"],
                "packed_size": entry["archive_packed_size"],
                "expected_mat_bytes": entry["expected_mat_bytes"],
                "expected_mat_sha256": entry.get("expected_mat_sha256"),
                "committed_bytes": committed_bytes,
                "updated_at": utc_now(),
            },
        )

    # ------------------------------------------------------------------ run
    def run(self) -> int:
        self.lock.acquire()
        try:
            self.load_state()
            pending = [
                item for item in self.config["files"]
                if item["origin"] == "to_download"
                and (not self.only or item["filename"] in self.only)
                and self.state.get("files", {}).get(item["filename"], {}).get("status")
                not in ("completed", "skipped_existing_valid")
            ]
            if self.limit_files:
                pending = pending[: self.limit_files]
            packed = sum(item["archive_block_size"] for item in pending)
            mats = sum(item["expected_mat_bytes"] for item in pending)
            log(f"待下载 {len(pending)} 个文件，压缩 {packed} 字节，解压 {mats} 字节")
            for path in (self.scratch, self.f_original):
                path.mkdir(parents=True, exist_ok=True)
            self.check_space(packed + mats if len(pending) <= 2 else
                             max(item["archive_block_size"] + item["expected_mat_bytes"]
                                 for item in pending) if pending else 0)
            log(f"远程确认：{self.client.probe_identity()}")
            failures = 0
            for index, entry in enumerate(pending, start=1):
                log(f"=== [{index}/{len(pending)}] {entry['filename']} ===")
                try:
                    record = self.process_file(entry, volume_identity("F:")["freeBytes"])
                    self.state.setdefault("files", {})[entry["filename"]] = record
                    log(f"{entry['filename']}: {record['status']}"
                        + (f" sha256={record.get('mat_sha256', '-')}" if record.get("status") != "failed" else ""))
                except (Select12Error, OSError) as exc:
                    failures += 1
                    record = {
                        "filename": entry["filename"],
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "failed_at": utc_now(),
                    }
                    self.state.setdefault("files", {})[entry["filename"]] = record
                    self.state.setdefault("errors", []).append(record)
                    log(f"{entry['filename']}: 失败 —— {exc}")
                    if isinstance(exc, (ProtocolViolation, VolumeGuardError)):
                        self.save_state()
                        log("协议或卷身份异常，立即停止整个运行")
                        return 3
                self.save_state()
            log(f"本轮结束：失败 {failures} 个")
            return 1 if failures else 0
        finally:
            try:
                self.save_state()
            except Exception:  # noqa: BLE001
                pass
            self.lock.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SELECT12-F1 断点续传下载器")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--block-mib", type=int, default=8)
    parser.add_argument("--unrar", type=Path, default=DEFAULT_UNRAR)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--only", default="", help="逗号分隔的文件名")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.workers <= 12:
        raise Select12Error("--workers 必须在1到12之间")
    if not 1 <= args.block_mib <= 64:
        raise Select12Error("--block-mib 必须在1到64之间")
    config = load_config(args.config)
    only = [token.strip() for token in args.only.split(",") if token.strip()]
    downloader = Downloader(
        config,
        workers=args.workers,
        block_bytes=args.block_mib * 1024**2,
        unrar=args.unrar,
        dry_run=args.dry_run,
        limit_files=args.limit_files,
        only=only,
    )
    if args.dry_run:
        pending = [item for item in config["files"] if item["origin"] == "to_download"]
        for item in pending[: (args.limit_files or len(pending))]:
            print(f"  {item['filename']} offset={item['archive_block_offset']} "
                  f"block={item['archive_block_size']} mat={item['expected_mat_bytes']}")
        log(f"--dry-run：将处理 {len(pending)} 个文件，未下载")
        return 0
    if not args.unrar.is_file():
        raise Select12Error(f"未找到UnRAR：{args.unrar}")
    return downloader.run()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中断，断点已保留。")
        raise SystemExit(130)
    except (Select12Error, OSError) as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
