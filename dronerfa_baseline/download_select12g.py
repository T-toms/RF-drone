#!/usr/bin/env python3
"""SELECT12-G1 step 3: resumable, range-exact downloader for the missing MAT files on G:.

Scope
-----
Fetches only the frozen per-entry ``Range`` regions of the official DroneRFa RAR archive
for the 42 (currently) missing MAT files of the frozen 104-file target table and lands
them in ``G:\\DroneRFa数据集\\original``.  It never touches the new F: volume, never
resumes the full-archive download and never deletes or moves pre-existing G: files.

Why this module instead of re-running the old per-file curl loop
---------------------------------------------------------------
The 2026-09-22/23 loop was rejected by Codex for three concrete reasons:

1. ``output.write(payload)`` return values were not checked (a short write could be
   silently treated as a complete block);
2. the per-block state was saved before the payload was durable (no ``fsync``), so a
   crash could leave a breakpoint pointing past the real data;
3. disk space was only checked once, before the round, and the volume identity was
   never re-checked while running.

It also left *complete but corrupt* archives in staging together with their
``.selection.json``, so a plain resume skipped and re-failed them forever
("skip-and-refail trap", 8 files / 8.09 GB).

Durability model
----------------
* Each selected entry is downloaded **strictly sequentially** into one contiguous
  ``<name>.mat.rar.part``; range fetches run ahead in a bounded pool but commits happen
  in order, so the part file is a single monotone prefix.
* Every block is written, ``flush``ed and ``fsync``ed, then the real on-disk size is
  re-``stat``ed and must equal the expected size.  Any mismatch is fatal for that file.
* Only after the data is durable is ``<name>.mat.rar.part.state.json`` written atomically
  (tmp + ``fsync`` + ``os.replace``) and the directory entry synced.
* On resume the part file is truncated to the committed byte count, so an unrecorded
  tail can never be treated as valid data.  A part file longer than its state is
  truncated; a part file *shorter* than its state is a hard stop, never a silent restart.

Protocol rules
--------------
Every response must be ``206`` with an exact ``Content-Range: bytes a-b/total`` matching
the request, ``total`` equal to the frozen archive size, ``ETag`` equal to the frozen
value, a matching ``Content-Length`` and a body of exactly ``b-a+1`` bytes.  A changed
ETag/archive size or a non-206 answer with an unexpected status is a hard stop for the
whole run; a transient status (408/425/429/5xx) or a transport error is retried with
bounded backoff.  A response whose ``Content-Range``/``Content-Length``/body length does
not match the request is retried (a single glitched response must not abort a multi-hour
run) and only aborts the file after the retry budget is exhausted.

Content verification (the part that matters most)
-------------------------------------------------
The old failures were mostly *correct-length but wrong bytes* -- HTTP 206 with a valid
Content-Range, so the client could not tell.  Therefore every completed file is verified
end to end, and the staging layout is used correctly:

* ``<name>.mat.rar.part.restart`` holds the **bare RAR5 file block** of one entry, exactly
  the frozen ``archive_block_size`` window that starts at ``archive_block_offset``.  It is
  *not* a RAR archive: byte 0 is the file-block header CRC, not the RAR5 signature.  The
  2026-09-24 ``缺少RAR5签名`` failures were a checker bug, not an endpoint fault; the header
  CRC, the block type, the extra-area size, the packed size, the entry tail and the entry
  name are all parsed from offset 0 **without** any signature.
* The payload of that entry is **compressed** (``data_size`` 1,057,934,310 vs. unpacked
  1,096,484,492 for ``T0011_D00_S0000.mat``), so the payload may never be copied out as if
  it were the MAT and a "packed SHA256 == MAT SHA256" test can never hold.
* Instead the verified block is wrapped into a **temporary single-file RAR**
  (``RAR5_SIGNATURE + MINIMAL_MAIN_HEADER + <original block> + MINIMAL_END_HEADER``, the
  layout already proven on 38 files by ``download_dronerfa_subset.py``), ``UnRAR t`` is run
  on it, and the member is extracted with ``UnRAR e`` into controlled scratch.  Only a MAT
  that matches the frozen ``expected_mat_bytes`` and passes the HDF5 four-channel probe is
  moved atomically into ``original/``.
* The bare block breakpoint is kept until the wrapped archive has verified, so a failure
  never destroys the evidence.
* A content failure does **not** silently re-download the whole file: the retry loop is
  closed unless ``--file-attempts`` is raised explicitly, so a repaired check can be tried
  on one real file before any bulk traffic is spent.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import http.client
import io
import json
import os
import re
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK = PROJECT_ROOT / "work" / "sel12g"
DEFAULT_CONFIG = WORK / "sel12g-config-20260924.json"
DEFAULT_UNRAR = Path(r"C:\Program Files\WinRAR\UnRAR.exe")
# Child stdio may not be piped here (the sandbox denies CreatePipe), so UnRAR output is
# redirected to a file; it must live somewhere writable, i.e. inside the workspace.
CHILD_LOG_DIR = PROJECT_ROOT / "tmp" / "sel12g-childlogs"

REQUIRED_KEYS = ("RF0_I", "RF0_Q", "RF1_I", "RF1_Q")
RAR5_SIGNATURE = bytes.fromhex("526172211a070100")
HEAD_FILE = 2
HEAD_ENDARC = 5
RETRYABLE_HTTP = (408, 425, 429, 500, 502, 503, 504)
TRANSIENT_RANGE_STATUS = RETRYABLE_HTTP
SPACE_MARGIN_BYTES = 5 * 1024**3
STATE_SCHEMA = "dronerfa-select12g-part-state/v1"

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


class Select12GError(RuntimeError):
    """Expected, user-facing failure."""


class ProtocolViolation(Select12GError):
    """The server answered in a way that cannot be trusted -> hard stop of the run."""


class RangeMismatch(Select12GError):
    """One bad response for one range -> retried / fails only that file."""


class VolumeGuardError(Select12GError):
    """The target volume is not the frozen volume -> hard stop."""


class ContentIntegrityError(Select12GError):
    """The fetched bytes do not form the expected MAT -> retry the whole file."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str, log_file: Path | None = None) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if log_file is not None:
        try:
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass


def sha256_file(path: Path, chunk: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_region(path: Path, offset: int, length: int, chunk: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining > 0:
            block = handle.read(min(chunk, remaining))
            if not block:
                raise ContentIntegrityError(f"{path.name}: 区段在 {length - remaining} 字节处提前结束")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


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


def remove_tree(path: Path) -> None:
    """Best-effort recursive removal of a scratch directory."""
    if not path.exists():
        return
    for child in path.rglob("*"):
        if child.is_file():
            child.unlink(missing_ok=True)
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_dir():
            try:
                child.rmdir()
            except OSError:
                pass
    try:
        path.rmdir()
    except OSError:
        pass


def assert_under(path: Path, root: Path, label: str) -> Path:
    """Absolute-path containment check: nothing may be written outside the G: tree."""
    resolved = Path(os.path.abspath(path))
    root_resolved = Path(os.path.abspath(root))
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise Select12GError(f"{label} 越界：{resolved} 不在 {root_resolved} 之内")
    return resolved


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "dronerfa-select12g-plan/v1":
        raise Select12GError(f"不是SELECT12-G1配置：{path}")
    if not config.get("files"):
        raise Select12GError("配置中没有目标文件")
    return config


class VolumeGuard:
    """Re-checks the frozen volume identity (serial + capacity) at a fixed cadence."""

    def __init__(self, drive: str, marker: dict, interval_seconds: float = 300.0):
        self.drive = drive
        self.marker = marker
        self.interval = interval_seconds
        self._last_check = 0.0
        self.checks = 0
        self.check(force=True)

    def check(self, *, force: bool = False) -> dict:
        now = time.monotonic()
        if not force and now - self._last_check < self.interval:
            return {}
        identity = volume_identity(self.drive)
        if self.marker.get("serialNumber") and identity["serialNumber"] != self.marker["serialNumber"]:
            raise VolumeGuardError(
                f"{self.drive} 卷序列号 {identity['serialNumber']} 与冻结标记 "
                f"{self.marker['serialNumber']} 不符，立即停止"
            )
        if self.marker.get("fileSystem") and identity["fileSystem"] != self.marker["fileSystem"]:
            raise VolumeGuardError(
                f"{self.drive} 文件系统 {identity['fileSystem']} 与冻结标记 "
                f"{self.marker['fileSystem']} 不符，立即停止"
            )
        if self.marker.get("totalBytes") and identity["totalBytes"] != self.marker["totalBytes"]:
            raise VolumeGuardError(
                f"{self.drive} 容量 {identity['totalBytes']} 与冻结标记 "
                f"{self.marker['totalBytes']} 不符，立即停止"
            )
        self._last_check = now
        self.checks += 1
        return identity

    def free_bytes(self) -> int:
        return volume_identity(self.drive)["freeBytes"]


class SingleInstanceLock:
    """O_CREAT|O_EXCL lock file; a stale lock whose PID is gone is reclaimed once."""

    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

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
                raise Select12GError(f"已有实例在运行：{holder}") from None
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "started_at": utc_now()}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            self.acquired = True
            return
        raise Select12GError("无法获得单实例锁")

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
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False


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
                 *, timeout: float = 300.0, retries: int = 6, host_override: str = "",
                 allow_http: bool = False, verbose: bool = True,
                 status_backoff: float = 60.0):
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        if scheme != "https" and not allow_http:
            raise Select12GError(f"只接受HTTPS端点：{url}")
        self.scheme = scheme
        self.host = parts.netloc
        self.connect_host = host_override or parts.netloc
        self.target = parts.path + ("?" + parts.query if parts.query else "")
        self.url = url
        self.expected_total = expected_total
        self.expected_etag = expected_etag
        self.timeout = timeout
        self.retries = retries
        self.verbose = verbose
        self.status_backoff = status_backoff
        self.ssl_context = ssl.create_default_context()
        self.stats = {
            "range_requests": 0,
            "retries": 0,
            "bytes_transferred": 0,
            "protocol_violations": [],
            "http_status_counts": {},
            "range_mismatches": [],
        }
        self._stats_lock = threading.Lock()

    # ------------------------------------------------------------------ fetch
    def _fetch_once(self, start: int, end: int) -> tuple[bytes, dict, int]:
        if self.scheme == "https":
            connection = http.client.HTTPSConnection(
                self.connect_host, timeout=self.timeout, context=self.ssl_context
            )
        else:
            connection = http.client.HTTPConnection(self.connect_host, timeout=self.timeout)
        try:
            connection.request(
                "GET",
                self.target,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                    "User-Agent": "DroneRFa-Select12G-Downloader/1.0",
                    "Connection": "close",
                },
            )
            response = connection.getresponse()
            headers = {
                "content-range": response.getheader("Content-Range"),
                "content-length": response.getheader("Content-Length"),
                "etag": response.getheader("ETag"),
                "content-encoding": response.getheader("Content-Encoding"),
                "accept-ranges": response.getheader("Accept-Ranges"),
            }
            status = response.status
            body = response.read()
            return body, headers, status
        finally:
            connection.close()

    def _validate(self, start: int, end: int, body: bytes, headers: dict, status: int) -> None:
        expected = end - start + 1
        if status != 206:
            # retryable statuses are handled by fetch(); anything else is a hard stop
            raise ProtocolViolation(f"HTTP {status}（期望206），区间 {start}-{end}")
        encoding = (headers.get("content-encoding") or "identity").strip().lower()
        if encoding not in ("", "identity"):
            raise ProtocolViolation(f"意外的Content-Encoding：{encoding!r}")
        parsed = parse_content_range(headers.get("content-range"))
        if parsed is None:
            raise RangeMismatch(f"Content-Range不可解析：{headers.get('content-range')!r}")
        actual_start, actual_end, total = parsed
        if total != self.expected_total:
            raise ProtocolViolation(f"归档总长 {total} 与冻结值 {self.expected_total} 不符")
        if (actual_start, actual_end) != (start, end):
            raise RangeMismatch(
                f"Content-Range区间不符：请求 {start}-{end}，返回 {actual_start}-{actual_end}"
            )
        etag = headers.get("etag")
        if etag != self.expected_etag:
            raise ProtocolViolation(f"ETag变化：{etag!r} != {self.expected_etag!r}")
        declared = headers.get("content-length")
        if declared is not None and int(declared) != expected:
            raise RangeMismatch(f"Content-Length {declared} 与请求长度 {expected} 不符")
        if len(body) != expected:
            raise RangeMismatch(f"正文 {len(body)} 字节，期望 {expected}")

    def fetch(self, start: int, end: int) -> bytes:
        last = ""
        for attempt in range(1, self.retries + 1):
            with self._stats_lock:
                self.stats["range_requests"] += 1
            try:
                body, headers, status = self._fetch_once(start, end)
                with self._stats_lock:
                    key = str(status)
                    self.stats["http_status_counts"][key] = (
                        self.stats["http_status_counts"].get(key, 0) + 1
                    )
                if status in RETRYABLE_HTTP:
                    last = f"HTTP {status}"
                    raise RangeMismatch(last)
                self._validate(start, end, body, headers, status)
                with self._stats_lock:
                    self.stats["bytes_transferred"] += len(body)
                return body
            except ProtocolViolation as exc:
                with self._stats_lock:
                    self.stats["protocol_violations"].append(f"{start}-{end}: {exc}")
                raise
            except RangeMismatch as exc:
                last = str(exc)
                with self._stats_lock:
                    self.stats["range_mismatches"].append(f"{start}-{end}: {exc}")
            except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            if attempt == self.retries:
                break
            with self._stats_lock:
                self.stats["retries"] += 1
            delay = (min(self.status_backoff * attempt, 300)
                     if last.startswith("HTTP ") else min(2 ** attempt, 30))
            if self.verbose:
                log(f"  区间 {start}-{end} 失败（{last}），{delay} 秒后重试 {attempt}/{self.retries}")
            time.sleep(delay)
        raise Select12GError(f"多次重试后仍无法下载区间 {start}-{end}：{last}")

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


# --------------------------------------------------------------------- RAR5
# RAR5 block layout used here, measured on four real archives (the official DroneRFa
# archive's live file block, ``DroneRFa_pilot.rar``, ``figure.rar`` and this repo's own
# builder) rather than taken on faith from the specification:
#
#   [0:4]   header CRC32 (little endian)
#   [4:..]  size vint, then the header body
#   [..]    payload of ``data_size`` bytes (compressed -- never copy it out as the MAT)
#
# Two details were measured, not assumed:
#
# * ``crc_end = pos + 3 + header_size`` for every block of every real archive above.  The
#   official file block has ``header_size == 66`` and its stored CRC matches
#   ``zlib.crc32(bytes[4:71])``; a spec-only reading would put the end at 70 or 72 and
#   fail 100% of the time.  Because that offset is load bearing, ``parse_block`` also
#   verifies the block against the frozen index geometry and falls back to the
#   ``pos + 2 + header_size`` reading if -- and only if -- the first one cannot describe
#   the frozen ``archive_block_size``.
# * the entry name sits at the very end of the header body (``length vint + UTF-8``).
#   Measured on the official block: the header body ends at byte 71 and the tail is
#   ``1c`` + ``DroneRFa/T0011_D00_S0000.mat`` (28 bytes).

HEAD_MAIN = 1
HEAD_FILE = 2
HEAD_ENDARC = 5


def decode_vint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(10):
        if offset < 0 or offset >= len(data):
            raise Select12GError("RAR头部不完整")
        current = data[offset]
        offset += 1
        value |= (current & 0x7F) << shift
        if current & 0x80 == 0:
            return value, offset
        shift += 7
    raise Select12GError("RAR头部包含无效的变长整数")


def encode_vint(value: int) -> bytes:
    if value < 0:
        raise Select12GError("变长整数不能为负")
    out = bytearray()
    while True:
        current = value & 0x7F
        value >>= 7
        if value:
            out.append(current | 0x80)
        else:
            out.append(current)
            return bytes(out)


def make_rar5_block(header_type: int, extra_area: bytes = b"") -> bytes:
    """A header-only RAR5 block (main header / end-of-archive header)."""
    flags = 0x0001 if extra_area else 0
    body = encode_vint(header_type) + encode_vint(flags)
    if extra_area:
        body += encode_vint(len(extra_area)) + extra_area
    crc = zlib.crc32(encode_vint(len(body)) + body) & 0xFFFFFFFF
    return crc.to_bytes(4, "little") + encode_vint(len(body)) + body


MINIMAL_MAIN_HEADER = make_rar5_block(HEAD_MAIN)
MINIMAL_END_HEADER = make_rar5_block(HEAD_ENDARC)


def _entry_name_at(data: bytes, name_start: int) -> str:
    """Decode ``length vint + UTF-8`` starting at ``name_start`` (``""`` when invalid)."""
    if name_start < 1:
        return ""
    try:
        length, after = decode_vint(data, name_start)
    except Select12GError:
        return ""
    if length <= 0 or length > 4096 or after + length > len(data):
        return ""
    try:
        name = data[after:after + length].decode("utf-8")
    except UnicodeDecodeError:
        return ""
    if not name or any(character < " " for character in name):
        return ""
    return name


def _find_entry_name(data: bytes, span_start: int, span_end: int,
                     expected_name: str) -> tuple[str, int]:
    """Locate ``length vint + UTF-8 name`` inside the CRC-covered header.

    The RAR5 file-header field order differs between the archivers involved -- the official
    DroneRFa block stores the name length immediately before the name (byte 31 of a 71-byte
    block), while this repo's own builder appends it after the packed-size vint -- so the
    name is located structurally instead of at a fixed cursor: the length vint must decode
    to exactly the number of bytes that follow it, and the name must be a printable relative
    path that fits inside the CRC-covered span.  When ``expected_name`` is supplied (always
    the case for real work, where it comes from the frozen index) the located name must
    equal it, so a block can never be accepted for the wrong entry.
    """
    if span_start < 4 or span_end > len(data) or span_start >= span_end:
        return "", -1
    needle = expected_name.encode("utf-8") if expected_name else b""
    region = data[span_start:span_end]
    if needle:
        found = region.find(needle)
        if found < 0:
            return "", -1
        name_start = span_start + found
        name = _entry_name_at(data, name_start - 1)
        if name != expected_name:
            return "", -1
        return name, name_start
    # No hint (fixture inspection / diagnostics): walk the header for a name-shaped tail.
    for name_start in range(span_start + 1, span_end):
        name = _entry_name_at(data, name_start - 1)
        if name and "/" in name and name_start + len(name.encode("utf-8")) <= span_end:
            return name, name_start
    return "", -1


def _is_compressed(data_size: int, expected_unpacked: int | None) -> bool:
    """Decide whether the payload must be unpacked.

    The evidence is the CRC-verified ``data_size`` of the block against the frozen
    ``expected_mat_bytes`` of the entry:

    * ``T0011_D00_S0000.mat`` declares 1,057,934,310 packed bytes but the entry unpacks to
      1,096,484,492 bytes, so the payload is deflated and cannot be copied out as the MAT;
    * a *stored* entry has ``data_size == expected_mat_bytes`` (``figure.rar`` members do),
      so nothing needs unpacking.

    The header's own unpacked-size field is deliberately *not* decoded: the official archiver
    and the WinRAR CLI lay the file fields out differently (measured on both), and a wrong
    guess there would either skip a needed UnRAR step or add a pointless one.  The comparison
    above cannot be wrong in the dangerous direction -- a compressed payload is always
    smaller than the MAT it unpacks to.
    """
    if not expected_unpacked:
        return False
    return data_size < int(expected_unpacked)


def parse_block(data: bytes, position: int, archive_size: int, entry: dict | None = None,
                *, expected_data_size: int | None = None,
                expected_block_size: int | None = None,
                expected_name: str = "",
                expected_mat_bytes: int | None = None) -> dict:
    """Parse and CRC-verify one RAR5 block.

    ``data`` holds at least the block header; ``archive_size`` is how many of its bytes are
    real (i.e. how much of the staging file exists right now), so the same function works
    for a complete file block and for a partial one.  ``entry``/``expected_*`` are optional
    cross-checks against the frozen index and are only applied when supplied.
    """
    if position < 0 or position + 6 > archive_size:
        raise Select12GError("RAR块越过文件末尾")
    stored_crc = int.from_bytes(data[position:position + 4], "little")
    header_size, vint_end = decode_vint(data, position + 4)
    vint_bytes = vint_end - (position + 4)

    # Candidate header ends.  Rule A is the measured convention (see the section comment);
    # rule B is the plain spec reading and is only accepted when rule A cannot describe the
    # frozen geometry, which keeps a future archiver change detectable instead of silent.
    candidates: list[tuple[str, int]] = [
        ("crc_end=size_vint_start+header_size", vint_end + header_size)
    ]
    spec_end = vint_end - 1 + header_size
    if spec_end != vint_end + header_size:
        candidates.append(("crc_end=size_vint_end+header_size", spec_end))

    chosen: tuple[str, int, int] | None = None
    observed: list[dict] = []
    for label, header_end in candidates:
        span_start = vint_end - 1
        record = {"convention": label, "header_end": header_end}
        if span_start <= position or header_end <= span_start or header_end > archive_size:
            record["error"] = "头部越界"
            observed.append(record)
            continue
        stored_span = zlib.crc32(data[span_start:header_end]) & 0xFFFFFFFF
        record["stored_span_crc32"] = f"{stored_span:08X}"
        record["stored_span_length"] = header_end - span_start
        if stored_span != stored_crc:
            record["error"] = "块头CRC不匹配"
            observed.append(record)
            continue
        try:
            block_type, cursor = decode_vint(data, vint_end)
            header_flags, cursor = decode_vint(data, cursor)
            extra_size = 0
            data_size = 0
            if header_flags & 0x0001:
                extra_size, cursor = decode_vint(data, cursor)
            if header_flags & 0x0002:
                data_size, cursor = decode_vint(data, cursor)
        except Select12GError as exc:
            record["error"] = f"块头字段不可解析：{exc}"
            observed.append(record)
            continue
        record.update({"type": block_type, "flags": header_flags,
                       "extra_area_size": extra_size, "data_size": data_size})
        if expected_data_size is not None and block_type == HEAD_FILE \
                and data_size != expected_data_size:
            record["error"] = (f"打包大小 {data_size} != 清单 {expected_data_size}")
            observed.append(record)
            continue
        if expected_block_size is not None and block_type == HEAD_FILE:
            block_bytes = header_end - position + data_size
            if block_bytes != expected_block_size:
                record["error"] = (
                    f"块长 {block_bytes} != 冻结块长 {expected_block_size}"
                )
                observed.append(record)
                continue
        name = ""
        if block_type == HEAD_FILE:
            name, name_offset = _find_entry_name(data, vint_end - 1 - position, header_end,
                                                expected_name)
            if name_offset < 0:
                record["error"] = "块头中未找到目标条目名"
                observed.append(record)
                continue
            record["name_offset"] = position + name_offset
            reference = expected_mat_bytes
            if reference is None and isinstance(entry, dict):
                reference = entry.get("expected_mat_bytes")
            record["compressed_payload"] = _is_compressed(data_size, reference)
        chosen = (label, header_end, data_size)
        record.update({"crc_ok": True, "name": name, "extra_area_size": extra_size,
                       "header_bytes_from_size_vint": header_end - (position + 4)})
        observed.append(record)
        break

    if chosen is None:
        raise ContentIntegrityError(
            f"RAR块校验失败（候选 {len(observed)} 个）：{json.dumps(observed, ensure_ascii=False)}"
        )
    convention, header_end, data_size = chosen
    block = observed[-1]
    block["header_type"] = int(block["type"])
    # ``header_end`` and ``payload_offset`` are ABSOLUTE offsets into ``data`` (callers use
    # them to walk the chain), while ``crc_span``/``size_vint_bytes`` are block-relative.
    # Mixing the two up here previously mis-placed the payload by ``position`` bytes whenever
    # a block did not start at 0 -- exactly the case for the wrapped archive, whose file
    # block starts after the signature plus the main header.
    block["crc_span_start"] = vint_end - 1
    block["header_total_size"] = header_end - position
    block["header_end"] = header_end
    block["payload_offset"] = header_end
    block["payload_size"] = data_size
    block["block_size"] = (header_end - position) + data_size
    block["crc_span"] = f"[{vint_end - 1 - position}:{header_end - position})"
    block["size_vint_bytes"] = vint_bytes
    block["convention"] = convention
    block["candidates"] = [dict(item) for item in observed]
    return block


def verify_file_block_prefix(part: Path, entry: dict) -> dict:
    """Verify the *bare* file block stored in ``<name>.mat.rar.part.restart``.

    The staging file is a raw RAR5 file block, not an archive: its first four bytes are the
    header CRC and **not** the RAR5 signature, so requiring a signature here is what made
    every resume and every finished download fail on 2026-09-24.  The block type, the
    header CRC, the packed size, the entry name and the frozen geometry are checked
    instead; a block header that is not yet fully on disk is reported as ``complete=False``
    rather than as corruption.
    """
    size = part.stat().st_size
    with part.open("rb") as handle:
        head = handle.read(min(size, 256 * 1024))
    if len(head) < 12:
        return {"complete": False, "header_complete": False, "available_bytes": len(head),
                "reason": "块头尚未下载完整"}
    try:
        header_size, _ = decode_vint(head, 4)
    except Select12GError as exc:
        return {"complete": False, "header_complete": False, "available_bytes": len(head),
                "reason": str(exc)}
    needed = 4 + 2 + header_size
    if len(head) < needed:
        return {"complete": False, "header_complete": False, "available_bytes": len(head),
                "needed_for_header": needed, "reason": "块头尚未下载完整"}
    block = parse_block(
        head, 0, len(head), entry,
        expected_data_size=int(entry["archive_packed_size"]),
        expected_block_size=int(entry["archive_block_size"]),
        expected_name=str(entry.get("archive_name") or entry["filename"]),
        expected_mat_bytes=int(entry["expected_mat_bytes"]),
    )
    block["complete"] = size >= int(block["block_size"])
    block["header_complete"] = True
    block["available_bytes"] = len(head)
    return block


def build_single_file_rar(block_path: Path, destination: Path) -> dict:
    """Wrap one bare RAR5 file block into a temporary single-file RAR archive.

    ``RAR5_SIGNATURE + minimal main header + the untouched original block + minimal
    end-of-archive header`` -- the same wrapper this project already used successfully for
    38 landed MAT files (``download_dronerfa_subset.py::stream_selected_archive``).  Every
    write is length-checked and the result is flushed and ``fsync``ed before it is handed
    back.  The original block is never modified, moved or deleted here.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = len(RAR5_SIGNATURE) + len(MINIMAL_MAIN_HEADER) + block_path.stat().st_size \
        + len(MINIMAL_END_HEADER)
    written = 0
    with block_path.open("rb") as source, destination.open("wb") as output:
        for piece in (RAR5_SIGNATURE, MINIMAL_MAIN_HEADER):
            if output.write(piece) != len(piece):
                raise Select12GError("包装写入短写（头部）")
            written += len(piece)
        while True:
            chunk = source.read(8 * 1024**2)
            if not chunk:
                break
            if output.write(chunk) != len(chunk):
                raise Select12GError("包装写入短写（原始文件块）")
            written += len(chunk)
        if output.write(MINIMAL_END_HEADER) != len(MINIMAL_END_HEADER):
            raise Select12GError("包装写入短写（尾部）")
        written += len(MINIMAL_END_HEADER)
        output.flush()
        os.fsync(output.fileno())
    on_disk = destination.stat().st_size
    if written != expected or on_disk != expected:
        raise ContentIntegrityError(
            f"包装后长度 {on_disk}（写入 {written}）!= 预期 {expected}"
        )
    return {
        "archive": str(destination),
        "archive_bytes": on_disk,
        "header_bytes": len(RAR5_SIGNATURE) + len(MINIMAL_MAIN_HEADER) + len(MINIMAL_END_HEADER),
        "source_block_bytes": block_path.stat().st_size,
        "layout": "RAR5签名 + 最小主头 + 原始文件块 + 最小结束块",
    }


def verify_wrapped_archive(archive: Path, entry: dict) -> dict:
    """Parse a wrapped single-file RAR and check it against the frozen index."""
    size = archive.stat().st_size
    with archive.open("rb") as handle:
        # 256 KiB covers the main header, the file header and any plausible extra area; the
        # payload itself is never parsed, only its declared extent is used.
        blob = handle.read(min(size, 256 * 1024))
    if blob[:8] != RAR5_SIGNATURE:
        raise ContentIntegrityError(f"{archive.name}: 包装后缺少RAR5签名")
    main = parse_block(blob, len(RAR5_SIGNATURE), size)
    file_offset = int(main["header_end"]) + int(main["data_size"])
    if file_offset >= len(blob):
        raise ContentIntegrityError(f"{archive.name}: 文件块头超过预读范围")
    file_block = parse_block(blob, file_offset, size, entry,
                             expected_data_size=int(entry["archive_packed_size"]),
                             expected_name=str(entry.get("archive_name") or entry["filename"]),
                             expected_mat_bytes=int(entry["expected_mat_bytes"]))
    if int(file_block["header_total_size"]) + int(file_block["data_size"]) != int(entry["archive_block_size"]):
        raise ContentIntegrityError(f"{archive.name}: 文件块长度与清单不符")
    end_offset = int(file_block["header_end"]) + int(file_block["data_size"])
    if end_offset < file_offset or end_offset + len(MINIMAL_END_HEADER) != size:
        raise ContentIntegrityError(f"{archive.name}: 文件块末端与包装长度不符")
    with archive.open("rb") as handle:
        handle.seek(end_offset)
        tail = handle.read(len(MINIMAL_END_HEADER))
    end_block = parse_block(tail, 0, len(tail))
    end_block["header_end"] = end_offset + int(end_block["header_end"])
    blocks = [main, file_block, end_block]
    types = [int(item["header_type"]) for item in blocks]
    if types[:1] != [HEAD_MAIN] or types[-1:] != [HEAD_ENDARC]:
        raise ContentIntegrityError(f"{archive.name}: 包装块类型异常 {types}")
    file_blocks = [item for item in blocks if int(item["header_type"]) == HEAD_FILE]
    if len(file_blocks) != 1:
        raise ContentIntegrityError(f"{archive.name}: 期望1个文件块，实际 {len(file_blocks)} 个")
    block = file_blocks[0]
    expected_name = str(entry.get("archive_name") or entry["filename"])
    if block["name"] != expected_name:
        raise ContentIntegrityError(f"{archive.name}: 条目名 {block['name']!r} != {expected_name!r}")
    if int(block["data_size"]) != int(entry["archive_packed_size"]):
        raise ContentIntegrityError(
            f"{archive.name}: 打包大小 {block['data_size']} != 清单 {entry['archive_packed_size']}"
        )
    payload_offset = int(block["header_end"])
    payload_size = int(block["data_size"])
    # The chain must end exactly at the file end: the end-of-archive block is the last
    # thing in the wrapper, so a truncated or over-long wrapper is rejected here.
    if int(end_block["header_end"]) != size:
        raise ContentIntegrityError(
            f"{archive.name}: 结束块末端 {end_block['header_end']} != 文件长度 {size}"
        )
    return {
        "archive_bytes": size,
        "block_types": types,
        "block_count": len(blocks),
        "header_crc_all_ok": all(item.get("crc_ok") for item in blocks),
        "entry_path": expected_name,
        "entry_name": block["name"],
        "entry_name_bytes": len(expected_name.encode("utf-8")),
        "packed_size": payload_size,
        "payload_offset": payload_offset,
        "payload_size": payload_size,
        "end_header_offset": int(end_block["header_end"]) - len(MINIMAL_END_HEADER),
        "header_convention": block["convention"],
        "header_crc_span": block["crc_span"],
        "header_bytes": int(block["header_total_size"]),
        "extra_area_size": int(block["extra_area_size"]),
    }


def compressed_payload_stats(block_path: Path, block: dict) -> dict:
    """Record the compressed payload length and SHA256 for provenance.

    Deliberately *not* a pass/fail check: the payload is compressed, so its hash can never
    equal the extracted MAT hash.  The authoritative content checks are ``UnRAR t``, the
    extracted byte count and the HDF5 probe.
    """
    length = int(block["payload_size"])
    offset = int(block["payload_offset"])
    if offset + length > block_path.stat().st_size:
        raise ContentIntegrityError(f"{block_path.name}: 压缩载荷超出断点文件长度")
    digest = hashlib.sha256()
    remaining = length
    with block_path.open("rb") as handle:
        handle.seek(offset)
        while remaining > 0:
            chunk = handle.read(min(8 * 1024**2, remaining))
            if not chunk:
                raise ContentIntegrityError(f"{block_path.name}: 压缩载荷提前结束")
            digest.update(chunk)
            remaining -= len(chunk)
    return {"compressed_bytes": length, "sha256": digest.hexdigest(),
            "note": "压缩载荷哈希，仅作溯源；不等于解压MAT哈希"}


# ---------------------------------------------------------------- extraction
# REMOVED: ``extract_stored_payload``.  It copied the RAR payload region straight into the
# MAT file and then compared the two SHA256 values, which can only work for a *stored*
# entry.  The official DroneRFa archive compresses every target entry
# (``T0011_D00_S0000.mat``: packed 1,057,934,310 vs. unpacked 1,096,484,492 bytes), so that
# function could never validate a real file and its use is now forbidden: extraction is
# done by UnRAR from the wrapped single-file archive (see ``Downloader._attempt_file``).


def run_unrar(unrar: Path, command: str, archive: Path, destination: Path | None,
              log_dir: Path) -> tuple[int, str]:
    """Run UnRAR with **file-redirected** stdio.

    The DSH sandbox denies creating child stdio pipes (PermissionError WinError 5 on
    CreatePipe), but file redirection is allowed, so the output is captured to a file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"unrar-{command}-{os.getpid()}-{threading.get_ident()}.log"
    args = [str(unrar), command, "-idq", "-p-", str(archive)]
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        args.extend(["-o+", str(destination) + os.sep])
    with out_path.open("wb") as handle:
        result = subprocess.run(args, stdout=handle, stderr=subprocess.STDOUT, check=False)
    raw = out_path.read_bytes()
    text = ""
    for encoding in ("gbk", "utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    out_path.unlink(missing_ok=True)
    return result.returncode, text


def find_unrar(explicit: Path | None) -> Path | None:
    candidates = [explicit] if explicit else []
    candidates += [Path(r"C:\Program Files\WinRAR\UnRAR.exe"),
                   Path(r"C:\Program Files (x86)\WinRAR\UnRAR.exe")]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    return None


def inspect_mat(path: Path) -> dict:
    import h5py
    import numpy as np

    info: dict = {"path": str(path)}
    with h5py.File(path, "r") as handle:
        keys = sorted(handle.keys())
        info["keys"] = keys
        info["missing_required"] = [key for key in REQUIRED_KEYS if key not in handle]
        if info["missing_required"]:
            info["ok"] = False
            return info
        shapes: dict[str, list[int]] = {}
        dtypes: dict[str, str] = {}
        lengths: dict[str, int] = {}
        for key in REQUIRED_KEYS:
            dataset = handle[key]
            shapes[key] = list(dataset.shape)
            dtypes[key] = str(dataset.dtype)
            lengths[key] = int(np.prod(dataset.shape))
        info["shapes"] = shapes
        info["dtypes"] = dtypes
        info["channel_lengths"] = lengths
        info["channel_ndim"] = {key: len(shapes[key]) for key in REQUIRED_KEYS}
        info["lengths_consistent"] = len(set(lengths.values())) == 1
        info["source_samples"] = next(iter(lengths.values()))
        info["all_float"] = all(value.startswith("float") for value in dtypes.values())
        probe_start = max(info["source_samples"] // 2 - 500_000, 0)
        probe_end = min(probe_start + 1000, info["source_samples"])
        probe = handle["RF0_I"]
        # DroneRFa channels are stored as (1, N); be robust to a flat (N,) layout.
        if probe.ndim == 1:
            values = probe[probe_start:probe_end]
        elif probe.ndim == 2 and probe.shape[0] == 1:
            values = probe[0, probe_start:probe_end]
        elif probe.ndim == 2 and probe.shape[1] == 1:
            values = probe[probe_start:probe_end, 0]
        else:
            raise ContentIntegrityError(
                f"{path.name}: 不支持的RF0_I形状 {tuple(probe.shape)}"
            )
        info["read_probe"] = {
            "start": probe_start,
            "samples": int(values.shape[0]),
            "finite": bool(np.isfinite(values).all()),
        }
        info["complete_1M_windows"] = info["source_samples"] // 1_000_000
        info["ok"] = bool(
            info["lengths_consistent"]
            and info["source_samples"] > 0
            and info["all_float"]
            and info["read_probe"]["samples"] == probe_end - probe_start
            and info["read_probe"]["finite"]
        )
    return info


class Downloader:
    def __init__(self, config: dict, *, workers: int, block_bytes: int,
                 unrar: Path | None, file_attempts: int, limit_files: int = 0,
                 only: list[str] | None = None, http_retries: int = 6,
                 host_override: str = "", allow_http: bool = False,
                 log_file: Path | None = None, anchor: Path | None = None,
                 progress: bool = True, status_backoff: float = 60.0,
                 max_seconds: float = 0.0, retry_rounds: int = 3,
                 retry_round_pause: float = 600.0, restart_from_zero: bool = False):
        self.config = config
        self.paths = config["paths"]
        self.workers = workers
        self.block_bytes = block_bytes
        self.unrar = unrar
        self.file_attempts = file_attempts
        self.limit_files = limit_files
        self.only = set(only or [])
        self.log_file = log_file
        self.progress = progress
        self.max_seconds = max_seconds
        self.retry_rounds = max(retry_rounds, 1)
        self.retry_round_pause = retry_round_pause
        self.restart_from_zero = restart_from_zero
        self.archive = config["archive"]

        self.g_original = Path(self.paths["g_original"])
        self.g_records = Path(self.paths["g_records"])
        self.g_downloads = Path(self.paths["g_downloads"])
        self.scratch = Path(self.paths["g_scratch"])
        self.state_path = Path(self.paths["g_state"])
        self.lock_path = Path(self.paths["g_lock"])

        # Every write target must live inside the authorized G: dataset root; the check is
        # on absolute paths so a drive remap cannot silently redirect a write to F:.
        self.anchor = anchor or Path(self.paths["g_root"])
        for label, path in (("g_original", self.g_original), ("g_records", self.g_records),
                            ("g_scratch", self.scratch), ("g_state", self.state_path),
                            ("g_lock", self.lock_path)):
            assert_under(path, self.anchor, label)
        if str(self.anchor).upper().startswith("F:"):
            raise Select12GError(f"拒绝在F盘写入：{self.anchor}")

        marker = dict(config.get("volume") or {})
        self.volume = VolumeGuard(marker.get("driveLetter", "G:"), marker)
        self.lock = SingleInstanceLock(self.lock_path)
        self.client = HttpRangeClient(
            self.archive["url"], self.archive["size"], self.archive["etag"],
            retries=http_retries, host_override=host_override, allow_http=allow_http,
            verbose=progress, status_backoff=status_backoff,
        )
        self.state: dict = {}
        self.failures: list[dict] = []

    # ---------------------------------------------------------------- state
    def load_state(self) -> None:
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("volume_serial") != self.config["volume"]["serialNumber"]:
                raise Select12GError("状态文件属于另一个卷标记，拒绝继续")
            if self.state.get("archive_etag") != self.archive["etag"]:
                raise Select12GError("状态文件的归档ETag与冻结配置不符，拒绝继续")
        else:
            self.state = {
                "schema": STATE_SCHEMA,
                "volume_serial": self.config["volume"]["serialNumber"],
                "archive_etag": self.archive["etag"],
                "archive_size": self.archive["size"],
                "files": {},
                "errors": [],
                "runs": [],
                "started_at": utc_now(),
            }

    def save_state(self) -> None:
        self.state["updated_at"] = utc_now()
        self.state["stats"] = self.client.stats
        write_json_atomic(self.state_path, self.state)

    def _part_paths(self, filename: str) -> tuple[Path, Path, Path]:
        """Active staging paths.

        Deliberately *different* names from the 2026-09-22/23 staging files
        (``<name>.mat.rar.part`` / ``.part.state.json`` / ``.rar``): the task requires
        that no pre-existing G: file is moved, deleted or overwritten, so the old
        evidence is kept byte-for-byte and the new attempt writes its own files.
        """
        part = self.g_downloads / f"{filename}.rar.part.restart"
        state = self.g_downloads / f"{filename}.rar.part.restart.state.json"
        final_rar = self.scratch / f"{filename}.rar.part.restart.rar"
        return part, state, final_rar

    def _commit_part_state(self, state_path: Path, entry: dict, committed_bytes: int) -> None:
        write_json_atomic(
            state_path,
            {
                "schema": STATE_SCHEMA,
                "filename": entry["filename"],
                "archive_block_offset": entry["archive_block_offset"],
                "archive_block_size": entry["archive_block_size"],
                "archive_packed_size": entry["archive_packed_size"],
                "expected_mat_bytes": entry["expected_mat_bytes"],
                "archive_etag": self.archive["etag"],
                "archive_size": self.archive["size"],
                "committed_bytes": committed_bytes,
                "updated_at": utc_now(),
            },
        )

    def check_space(self, needed: int) -> int:
        free = self.volume.free_bytes()
        if free < needed + SPACE_MARGIN_BYTES:
            raise Select12GError(
                f"G盘可用空间不足：需要 {needed} + 余量 {SPACE_MARGIN_BYTES}，实际 {free}"
            )
        return free

    # ------------------------------------------------------------- one file
    def process_file(self, entry: dict) -> dict:
        filename = entry["filename"]
        offset = int(entry["archive_block_offset"])
        block_size = int(entry["archive_block_size"])
        part, part_state, final_rar = self._part_paths(filename)
        target = self.g_original / filename
        extract_dir = self.scratch / f"extract_{filename}"
        legacy_part = self.g_downloads / f"{filename}.rar.part"
        legacy_state = self.g_downloads / f"{filename}.rar.part.state.json"
        legacy_rar = self.g_downloads / f"{filename}.rar"

        record: dict = {
            "filename": filename,
            "target_id": entry["target_id"],
            "class_name": entry["class_name"],
            "origin": entry["origin"],
            "former_partition": entry["former_partition"],
            "split_status": entry["split_status"],
            "expected_mat_bytes": entry["expected_mat_bytes"],
            "archive_block_offset": offset,
            "archive_block_size": block_size,
            "archive_packed_size": entry["archive_packed_size"],
            "staging_part": str(part),
            "started_at": utc_now(),
        }
        # Record - but never touch - the pre-existing 2026-09-22/23 staging files.
        record["preexisting_staging"] = {
            "part": str(legacy_part) if legacy_part.exists() else None,
            "part_bytes": legacy_part.stat().st_size if legacy_part.exists() else None,
            "part_state": str(legacy_state) if legacy_state.exists() else None,
            "complete_rar": str(legacy_rar) if legacy_rar.exists() else None,
            "rar_bytes": legacy_rar.stat().st_size if legacy_rar.exists() else None,
            "action": "保留原样，不移动不删除；本次下载写入 .part.restart 独立文件",
        }
        record["free_bytes_before"] = self.volume.free_bytes()
        if self.restart_from_zero:
            record["restart_from_zero"] = {
                "part_bytes": part.stat().st_size if part.exists() else None,
                "note": "显式要求从头重下，丢弃现有断点",
            }
            self._reset_staging(part, part_state, final_rar, extract_dir)

        last_error = ""
        for attempt in range(1, self.file_attempts + 1):
            record["attempt"] = attempt
            try:
                self._attempt_file(entry, part, part_state, final_rar, target,
                                   extract_dir, record)
                break
            except ContentIntegrityError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log(f"{filename}: 内容校验失败（第{attempt}次）—— {exc}", self.log_file)
                # The breakpoint is evidence: it is kept, not wiped.  Nothing here deletes
                # a fetched block, so a repaired check can be re-run on the same bytes and
                # Codex can inspect exactly what the endpoint returned.
                record["preserved_breakpoint"] = {
                    "part": str(part) if part.exists() else None,
                    "part_bytes": part.stat().st_size if part.exists() else None,
                    "part_state": str(part_state) if part_state.exists() else None,
                    "note": "内容校验失败，断点保留作为证据；重跑默认从该断点续传，可用 --restart-from-zero 从头重下",
                }
                if attempt == self.file_attempts:
                    raise
            except (Select12GError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, (ProtocolViolation, VolumeGuardError)):
                    raise
                if attempt == self.file_attempts:
                    raise
                log(f"{filename}: 第{attempt}次尝试出错，重来 —— {exc}", self.log_file)
        else:  # pragma: no cover - loop always breaks or raises
            raise Select12GError(f"{filename}: 未完成，最后错误 {last_error}")
        return record

    @staticmethod
    def _reset_staging(part: Path, part_state: Path, final_rar: Path,
                       extract_dir: Path) -> None:
        """Remove this module's own staging artefacts (explicit opt-in only).

        Called only when ``--restart-from-zero`` is requested, or when a caller needs a
        clean slate.  The default failure path deliberately keeps the block + state so the
        fetched bytes stay inspectable.  The pre-existing 2026-09-22/23 ``.rar`` / ``.part``
        evidence and every landed MAT are out of scope by name.
        """
        for path in (final_rar, part_state, part):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        remove_tree(extract_dir)

    def _attempt_file(self, entry: dict, part: Path, part_state: Path, final_rar: Path,
                      target: Path, extract_dir: Path, record: dict) -> None:
        filename = entry["filename"]
        offset = int(entry["archive_block_offset"])
        block_size = int(entry["archive_block_size"])
        part.parent.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)

        # 1. already-landed valid MAT -> skip without touching anything
        if target.exists() and target.stat().st_size == entry["expected_mat_bytes"]:
            info = inspect_mat(target)
            if info.get("ok"):
                record.update({
                    "status": "skipped_existing_valid",
                    "mat_sha256": sha256_file(target),
                    "hdf5": info,
                    "finished_at": utc_now(),
                })
                return

        self.check_space(block_size + entry["expected_mat_bytes"])

        # 2. breakpoint bookkeeping
        completed = 0
        if part.exists():
            if part_state.exists():
                saved = json.loads(part_state.read_text(encoding="utf-8"))
                if saved.get("schema") != STATE_SCHEMA:
                    raise Select12GError(f"{filename}: 断点状态schema不符，拒绝续传")
                if int(saved.get("archive_block_offset", -1)) != offset:
                    raise Select12GError(f"{filename}: 断点状态与目标条目偏移不匹配")
                completed = int(saved["committed_bytes"])
                if not 0 <= completed <= block_size:
                    raise Select12GError(f"{filename}: 断点字节数越界：{completed}")
                on_disk = part.stat().st_size
                if on_disk < completed:
                    raise Select12GError(
                        f"{filename}: 断点文件{on_disk}字节短于已提交的{completed}字节，拒绝续传"
                    )
                # The leading bytes of the staging file are the *bare RAR5 file block*, so
                # they are verified as such: header CRC, block type, packed size and the
                # entry tail.  No RAR5 signature is expected at offset 0, and a header that
                # is not complete yet is a hard stop rather than a silent restart.
                prefix = verify_file_block_prefix(part, entry)
                record["resume_header"] = {
                    key: prefix.get(key) for key in
                    ("complete", "header_complete", "header_total_size", "payload_size",
                     "name", "convention", "crc_span", "reason")
                }
                if not prefix.get("header_complete"):
                    raise Select12GError(
                        f"{filename}: 断点文件头部尚未完整（{prefix.get('reason')}），拒绝续传"
                    )
                if int(prefix["header_total_size"]) + int(prefix["payload_size"]) != block_size:
                    raise Select12GError(
                        f"{filename}: 断点文件头声明的块长 "
                        f"{prefix['header_total_size'] + prefix['payload_size']} != {block_size}"
                    )
                with part.open("r+b") as handle:
                    handle.truncate(completed)
                    handle.flush()
                    os.fsync(handle.fileno())
                record["resumed_from_bytes"] = completed
                record["truncated_from_bytes"] = on_disk
                log(f"{filename}: 续传，已提交 {completed}/{block_size} 字节"
                    f"（截去未记账尾部 {on_disk - completed} 字节）", self.log_file)
            else:
                # A part file without our state file has no provable byte accounting
                # (legacy tooling, crash inside the state write, foreign tool).  It is
                # never adopted: this entry restarts from byte 0 into our own file.
                record["unproven_part_bytes"] = part.stat().st_size
                log(f"{filename}: 发现无状态断点文件 {record['unproven_part_bytes']} 字节，"
                    f"身份/连续性无法证明，从条目起点重新下载", self.log_file)
                with part.open("r+b") as handle:
                    handle.truncate(0)
                    handle.flush()
                    os.fsync(handle.fileno())
                self._commit_part_state(part_state, entry, 0)
        else:
            with part.open("wb") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            self._commit_part_state(part_state, entry, 0)

        # The durable zero-byte breakpoint exists before any byte is fetched, so an
        # interrupted first attempt still leaves a readable (filename, size, 0) record.
        self._commit_part_state(part_state, entry, completed)

        # 3. sequential commit of out-of-order parallel range fetches
        started = time.monotonic()
        session_start = completed
        expected_total = block_size
        deadline = (started + self.max_seconds) if self.max_seconds else 0.0
        bounded_stop = False
        with part.open("r+b") as output, ThreadPoolExecutor(max_workers=self.workers) as pool:
            output.seek(completed)
            pending: list[tuple[int, int, object]] = []
            next_local = completed

            def schedule() -> bool:
                nonlocal next_local
                if next_local >= expected_total:
                    return False
                size = min(self.block_bytes, expected_total - next_local)
                start = offset + next_local
                end = start + size - 1
                pending.append((next_local, size, pool.submit(self.client.fetch, start, end)))
                next_local += size
                return True

            for _ in range(self.workers):
                if not schedule():
                    break

            while pending:
                position, size, future = pending.pop(0)
                payload = future.result()
                if position != completed:
                    raise Select12GError(f"{filename}: 提交顺序错乱 {position} != {completed}")
                if len(payload) != size:
                    raise Select12GError(f"{filename}: 分块长度 {len(payload)} != {size}")
                written = output.write(payload)
                if written != size:
                    raise Select12GError(f"{filename}: 短写 {written} != {size}")
                output.flush()
                os.fsync(output.fileno())
                if output.tell() != completed + size:
                    raise Select12GError(f"{filename}: 落盘位置异常 {output.tell()}")
                completed += size
                # Durability barrier: the bytes are on the platter before any state says so.
                self._commit_part_state(part_state, entry, completed)
                on_disk = part.stat().st_size
                if on_disk != completed:
                    raise Select12GError(
                        f"{filename}: 落盘后实际长度 {on_disk} != 已提交 {completed}"
                    )
                self.volume.check()
                elapsed = max(time.monotonic() - started, 1e-6)
                speed = (completed - session_start) / elapsed
                if self.progress:
                    print(
                        f"\r{filename} {completed / expected_total * 100:6.2f}% "
                        f"{completed}/{expected_total} 字节 {speed / 1048576:.2f} MiB/s",
                        end="", flush=True,
                    )
                if deadline and time.monotonic() >= deadline:
                    bounded_stop = True
                    log(f"{filename}: 达到 --max-seconds 上限，断点停在 "
                        f"{completed}/{expected_total} 字节", self.log_file)
                    for _, _, pending_future in pending:
                        pending_future.cancel()
                    pending = []
                    break
                schedule()
            if self.progress:
                print()

        if bounded_stop:
            record["status"] = "bounded_stop"
            record["committed_bytes"] = completed
            record["finished_at"] = utc_now()
            self._commit_part_state(part_state, entry, completed)
            return

        if part.stat().st_size != block_size:
            raise Select12GError(
                f"{filename}: 压缩块长度 {part.stat().st_size} != {block_size}"
            )
        record["download_seconds"] = round(time.monotonic() - started, 1)
        record["downloaded_bytes"] = completed - session_start
        record["committed_bytes"] = completed
        self._finish_file(entry, part, part_state, final_rar, target, extract_dir, record)

    # ------------------------------------------------------------- completion
    def _finish_file(self, entry: dict, part: Path, part_state: Path, final_rar: Path,
                     target: Path, extract_dir: Path, record: dict) -> None:
        """Verify a complete raw block, turn it into a MAT and land it atomically.

        The bare block is never renamed into ``.rar``: it is *wrapped* into a temporary
        single-file RAR, validated, tested with ``UnRAR t``, extracted with ``UnRAR e``,
        and only then is the extracted MAT allowed into ``original/``.  The breakpoint is
        kept until the wrapped archive has verified.
        """
        filename = entry["filename"]
        record["checks"] = {}

        # 4a. the staging file *is* the block, verify it as a block
        block = verify_file_block_prefix(part, entry)
        if not block.get("complete"):
            raise ContentIntegrityError(f"{filename}: 断点文件不是完整的原始文件块")
        expected_block = int(block["header_total_size"]) + int(block["payload_size"])
        if expected_block != int(entry["archive_block_size"]):
            raise ContentIntegrityError(
                f"{filename}: 块长核对 {expected_block} != 清单 {entry['archive_block_size']}"
            )
        record["checks"]["block"] = {
            "header_total_size": block["header_total_size"],
            "payload_size": block["payload_size"],
            "block_size": expected_block,
            "entry_name": block["name"],
            "extra_area_size": block["extra_area_size"],
            "compressed_payload": bool(block.get("compressed_payload")),
            "header_crc_span": block["crc_span"],
            "header_convention": block["convention"],
            "size_vint_bytes": block["size_vint_bytes"],
        }

        # 4b. wrap the untouched block into a temporary single-file RAR.
        # The wrapper is built *and verified* under a temporary name; it only becomes the
        # staging artefact once it has passed, so a rejected archive is never left behind.
        remove_tree(extract_dir)
        final_rar.unlink(missing_ok=True)
        staging_rar = final_rar.with_name(final_rar.name + ".building")
        staging_rar.unlink(missing_ok=True)
        try:
            record["checks"]["wrapped_archive"] = build_single_file_rar(part, staging_rar)
            archive = verify_wrapped_archive(staging_rar, entry)
            record["checks"]["archive"] = archive
            record["checks"]["compressed_payload"] = compressed_payload_stats(part, block)
            # 4c-4e: unpack with UnRAR and land the MAT.  The bare block and its state are
            # deliberately left untouched by this helper on failure, so the fetched bytes
            # stay inspectable and a repaired check can be re-run without new traffic.
            self._extract_and_land(entry, final_rar, staging_rar, extract_dir, target, record)
        finally:
            # The ~1 GB temporary wrapper is scratch, not evidence: the bare block in the
            # staging file is the durable record, and it can be re-wrapped at any time.
            staging_rar.unlink(missing_ok=True)
            if record.get("status") != "completed":
                final_rar.unlink(missing_ok=True)
                remove_tree(extract_dir)

    # ------------------------------------------------------------- extraction
    def _extract_and_land(self, entry: dict, final_rar: Path, staging_rar: Path,
                          extract_dir: Path, target: Path, record: dict) -> None:
        """``UnRAR t`` -> ``UnRAR e`` -> size/HDF5 gate -> atomic placement."""
        filename = entry["filename"]
        if self.unrar is None:
            raise ContentIntegrityError(
                f"{filename}: 条目是压缩载荷，必须用UnRAR解压；未找到UnRAR可执行文件，拒绝继续"
            )
        code, text = run_unrar(self.unrar, "t", staging_rar, None,
                               self.scratch / "unrarlogs")
        record["checks"]["unrar_test"] = {"exit_code": code, "output": text.strip()[:400]}
        if code != 0:
            raise ContentIntegrityError(
                f"{filename}: UnRAR 完整性测试失败，退出码 {code}：{text[:200]}"
            )
        code, text = run_unrar(self.unrar, "e", staging_rar, extract_dir,
                               self.scratch / "unrarlogs")
        record["checks"]["unrar_extract"] = {"exit_code": code, "output": text.strip()[:400]}
        if code != 0:
            raise ContentIntegrityError(
                f"{filename}: UnRAR 解压失败，退出码 {code}：{text[:200]}"
            )

        members = [item for item in extract_dir.rglob("*") if item.is_file()]
        record["checks"]["extracted_members"] = [
            {"name": item.name, "bytes": item.stat().st_size} for item in members
        ]
        if len(members) != 1:
            raise ContentIntegrityError(f"{filename}: 期望解压出1个文件，实际 {len(members)} 个")
        produced = members[0]
        if produced.name != filename:
            raise ContentIntegrityError(
                f"{filename}: 解压文件名 {produced.name} 与目标不符"
            )
        if produced.stat().st_size != int(entry["expected_mat_bytes"]):
            raise ContentIntegrityError(
                f"{filename}: 解包大小 {produced.stat().st_size} != 清单 {entry['expected_mat_bytes']}"
            )

        # 5. verify the extracted MAT *before* it is allowed into original/
        info = inspect_mat(produced)
        record["hdf5"] = info
        if not info.get("ok"):
            raise ContentIntegrityError(f"{filename}: HDF5结构检查未通过：{info}")

        if target.exists():
            raise ContentIntegrityError(f"{filename}: 目标MAT已存在，拒绝覆盖")
        os.replace(produced, target)
        sync_directory(target.parent)
        os.replace(staging_rar, final_rar)
        sync_directory(final_rar.parent)

        record["mat_sha256"] = sha256_file(target)
        record["mat_bytes"] = target.stat().st_size
        record["source_samples"] = info["source_samples"]
        record["windows_1M"] = info["source_samples"] // 1_000_000

        # Only now -- after the MAT is durably in place -- is the breakpoint retired.
        part_state = self._part_paths(filename)[1]
        part_state.unlink(missing_ok=True)
        sync_directory(part_state.parent)
        final_rar.unlink(missing_ok=True)
        sync_directory(final_rar.parent)
        try:
            extract_dir.rmdir()
        except OSError:
            pass
        record["status"] = "completed"
        record["finished_at"] = utc_now()
        record["free_bytes_after"] = self.volume.free_bytes()

    # ------------------------------------------------------------------- run
    def select_pending(self) -> list[dict]:
        files = self.config["files"]
        pending = []
        for item in files:
            if item["origin"] != "to_download":
                continue
            if self.only and item["filename"] not in self.only:
                continue
            landed = self.g_original / item["filename"]
            if landed.exists() and landed.stat().st_size == item["expected_mat_bytes"]:
                continue
            pending.append(item)
        pending.sort(key=lambda item: int(item["archive_block_offset"]))
        if self.only:
            order = {name: index for index, name in enumerate(
                token.strip() for token in self.config.get("_only_order", []))}
            if order:
                pending.sort(key=lambda item: order.get(item["filename"], 10**6))
        if self.limit_files:
            pending = pending[: self.limit_files]
        return pending

    def run(self) -> int:
        self.lock.acquire()
        try:
            self.load_state()
            pending = self.select_pending()
            packed = sum(int(item["archive_block_size"]) for item in pending)
            mats = sum(int(item["expected_mat_bytes"]) for item in pending)
            log(f"待下载 {len(pending)} 个文件，压缩块 {packed} 字节，解压 {mats} 字节", self.log_file)
            log(f"卷身份：{self.volume.check(force=True)}", self.log_file)
            identity = self.client.probe_identity()
            log(f"远程确认：{identity}", self.log_file)
            run_record = {
                "pid": os.getpid(),
                "started_at": utc_now(),
                "pending": len(pending),
                "retry_rounds": self.retry_rounds,
                "volume": volume_identity(self.config["volume"]["driveLetter"]),
                "remote_identity": identity,
            }
            self.state.setdefault("runs", []).append(run_record)
            self.save_state()

            failures = 0
            queue = list(pending)
            for round_index in range(1, self.retry_rounds + 1):
                if not queue:
                    break
                if round_index > 1:
                    log(f"=== 重试轮 {round_index}/{self.retry_rounds}：{len(queue)} 个此前失败的文件"
                        f"（该端点会间歇性返回内容错误的字节，隔一段时间换一份副本重试）===",
                        self.log_file)
                retry_queue: list[dict] = []
                for index, entry in enumerate(queue, start=1):
                    log(f"=== [轮{round_index} {index}/{len(queue)}] {entry['filename']} ===",
                        self.log_file)
                    record: dict
                    try:
                        record = self.process_file(entry)
                        self.state.setdefault("files", {})[entry["filename"]] = record
                        log(f"{entry['filename']}: {record['status']}"
                            + (f" sha256={record.get('mat_sha256', '-')}"
                               if record.get("status") != "failed" else ""), self.log_file)
                    except (ProtocolViolation, VolumeGuardError) as exc:
                        record = {
                            "filename": entry["filename"],
                            "status": "hard_stop",
                            "error": f"{type(exc).__name__}: {exc}",
                            "failed_at": utc_now(),
                        }
                        self.state.setdefault("files", {})[entry["filename"]] = record
                        self.state.setdefault("errors", []).append(record)
                        self.save_state()
                        log(f"{entry['filename']}: 协议或卷身份异常，立即停止整个运行 —— {exc}",
                            self.log_file)
                        return 3
                    except (Select12GError, OSError) as exc:
                        record = {
                            "filename": entry["filename"],
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "failed_at": utc_now(),
                        }
                        self.state.setdefault("files", {})[entry["filename"]] = record
                        self.state.setdefault("errors", []).append(record)
                        retry_queue.append(entry)
                        log(f"{entry['filename']}: 失败 —— {exc}", self.log_file)
                    self.save_state()
                failures = len(retry_queue)
                queue = retry_queue
                if queue and round_index < self.retry_rounds:
                    delay = self.retry_round_pause
                    log(f"本轮失败 {len(queue)} 个，等待 {delay:.0f} 秒后进入下一轮重试",
                        self.log_file)
                    time.sleep(delay)
            run_record["finished_at"] = utc_now()
            run_record["failures"] = failures
            run_record["pending_this_run"] = len(pending)
            log(f"本轮结束：处理 {len(pending)} 个，最终失败 {failures} 个", self.log_file)
            self.save_state()
            return 1 if failures else 0
        finally:
            try:
                self.save_state()
            except Exception:  # noqa: BLE001
                pass
            self.lock.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SELECT12-G1 G盘断点续传下载器")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--block-bytes", type=int, default=8 * 1024**2)
    parser.add_argument("--file-attempts", type=int, default=1,
                        help="内容校验失败后整文件重试次数；默认1（不重试，避免重复消耗流量）")
    parser.add_argument("--http-retries", type=int, default=6)
    parser.add_argument("--unrar", type=Path, default=None,
                        help="UnRAR路径；默认自动探测。解压必须使用UnRAR，--no-unrar 会在解压阶段硬停")
    parser.add_argument("--no-unrar", action="store_true")
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=0.0,
                        help="有界运行：到点后把断点留在当前字节并正常退出")
    parser.add_argument("--retry-rounds", type=int, default=1,
                        help="失败文件在整轮跑完后重新排队的轮数；默认1（关闭整轮重试）")
    parser.add_argument("--retry-round-pause", type=float, default=600.0,
                        help="两轮之间的等待秒数")
    parser.add_argument("--only", default="", help="逗号分隔的文件名，按给定顺序下载")
    parser.add_argument("--restart-from-zero", action="store_true",
                        help="丢弃现有断点，从头重新下载（默认保留断点并续传）")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--host-override", default="", help=argparse.SUPPRESS)
    parser.add_argument("--allow-http", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--anchor", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def show_status(config: dict) -> int:
    path = Path(config["paths"]["g_state"])
    if not path.exists():
        print(f"尚无状态文件：{path}")
        return 0
    state = json.loads(path.read_text(encoding="utf-8"))
    files = state.get("files", {})
    counts: dict[str, int] = {}
    for item in files.values():
        key = item.get("status", "?")
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({
        "updated_at": state.get("updated_at"),
        "counts": counts,
        "stats": state.get("stats"),
        "last_runs": state.get("runs", [])[-2:],
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.workers <= 16:
        raise Select12GError("--workers 必须在1到16之间")
    if not 4096 <= args.block_bytes <= 64 * 1024**2:
        raise Select12GError("--block-bytes 必须在4 KiB到64 MiB之间")
    config = load_config(args.config)
    only = [token.strip() for token in args.only.split(",") if token.strip()]
    if args.status:
        return show_status(config)
    if args.dry_run:
        pending = [item for item in config["files"] if item["origin"] == "to_download"]
        for item in pending[: (args.limit_files or len(pending))]:
            print(f"  {item['filename']} offset={item['archive_block_offset']} "
                  f"block={item['archive_block_size']} mat={item['expected_mat_bytes']}")
        print(json.dumps(config["totals"], ensure_ascii=False, indent=2))
        log(f"--dry-run：将处理 {len(pending)} 个文件，未下载")
        return 0
    unrar = None if args.no_unrar else find_unrar(args.unrar)
    config = dict(config)
    config["_only_order"] = only
    downloader = Downloader(
        config,
        workers=args.workers,
        block_bytes=args.block_bytes,
        unrar=unrar,
        file_attempts=args.file_attempts,
        limit_files=args.limit_files,
        only=only,
        http_retries=args.http_retries,
        host_override=args.host_override,
        allow_http=args.allow_http,
        log_file=args.log_file,
        anchor=args.anchor,
        max_seconds=args.max_seconds,
        retry_rounds=args.retry_rounds,
        retry_round_pause=args.retry_round_pause,
        restart_from_zero=args.restart_from_zero,
    )
    if unrar is None:
        log("警告：未找到UnRAR，仅使用内置RAR5(stored)解析 + 载荷哈希 + HDF5 校验")
    else:
        log(f"UnRAR：{unrar}")
    return downloader.run()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中断，断点已保留。")
        raise SystemExit(130)
    except (Select12GError, OSError) as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
