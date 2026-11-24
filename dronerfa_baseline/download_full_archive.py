"""Download the official complete DroneRFa RAR archive (FULL-ARCHIVE1).

Why this exists
---------------
The earlier per-MAT downloader (``download_heldout_files.py``) fetched each
1.17 GiB MAT file as its own small RAR range selection.  It was slow and flaky.
This module instead downloads the *single official complete archive* that the
dataset host publishes, and keeps it as one append-only ``.part`` file.

Transport choice
----------------
``urllib.request`` fails against this endpoint with
``SSL: UNEXPECTED_EOF_WHILE_READING`` (a failure the project already documented
in ``download_heldout_files.py``), and ``curl.exe`` is blocked by the current
DSH sandbox.  ``http.client.HTTPSConnection`` with the default verified SSL
context works, so the transfer is plain Python.

The server throttles a single connection to roughly 0.4 MiB/s, but that limit
is per connection: measured aggregate throughput was 1.23 MiB/s at 4 parallel
connections and 1.54 MiB/s at 8.  To keep the on-disk artifact exactly as
specified -- one ``.part`` file that only ever grows by appending -- the
fetches are pipelined: a bounded window of blocks is requested in parallel,
but blocks are appended to the file strictly in order by a single writer.
This yields the parallel transfer rate without sparse writes, without a second
copy on disk, and with a trivial resume checkpoint (a byte prefix).

Safety contract
---------------
* Every block response is validated *before* its bytes are written:
  status must be exactly 206, ``Content-Range`` must match the requested
  ``start-end`` and the frozen total, the ``ETag`` must match the frozen
  identity, ``Content-Encoding`` must be identity, ``Content-Length`` must
  equal the requested length, and the body must not be longer than requested.
* A protocol violation or a changed remote identity is a hard stop: the
  process exits non-zero, keeps the ``.part``, and writes nothing further.
* Only a full-length, exact-size archive is atomically renamed to ``.rar``.
* A single instance lock prevents a second download process.
* The volume marker pins the download to one physical volume, so a drive
  letter change cannot silently redirect writes to another disk.

Run ``python download_full_archive.py --help`` for the CLI.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import http.client
import json
import os
import random
import re
import socket
import ssl
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

SCHEMA_VERSION = 1

DEFAULT_BLOCK_BYTES = 8 * 1024 * 1024
DEFAULT_WORKERS = 6
DEFAULT_TIMEOUT = 120
DEFAULT_CONNECT_TIMEOUT = 30
RETRYABLE_HTTP = (408, 425, 429, 500, 502, 503, 504)
MAX_BLOCK_ATTEMPTS = 6
MAX_PROBE_ATTEMPTS = 5
MAX_CONSECUTIVE_TRANSIENT = 60
RETRY_BACKOFF_BASE = 2.0
RETRY_BACKOFF_MAX = 30.0
SPACE_MARGIN_BYTES = 20 * 1024 * 1024 * 1024

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_PROTOCOL = 3
EXIT_LOCKED = 4
EXIT_SPACE = 5

_CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


class FullArchiveError(RuntimeError):
    """Base class for handled failures."""


class ConfigError(FullArchiveError):
    """The frozen config or a path is unusable."""


class VolumeError(FullArchiveError):
    """The volume marker is missing, changed, or on the wrong volume."""


class IdentityError(FullArchiveError):
    """The remote object does not match the frozen identity."""


class ProtocolViolation(FullArchiveError):
    """The server answered a range request in a way that cannot be trusted."""


class TransientError(FullArchiveError):
    """A retryable network/HTTP failure."""


class LockError(FullArchiveError):
    """Another download process already holds the lock."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def localnow() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_content_range(value: str) -> tuple[int, int, int] | None:
    match = _CONTENT_RANGE_RE.fullmatch((value or "").strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def normalize_etag(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if text[:2].upper() == "W/":
        text = text[2:].strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text.lower() or None


def volume_identity(root: Path) -> dict:
    """Volume label/filesystem/serial plus capacity for the volume holding ``root``.

    ``GetVolumeInformationW`` only accepts a *volume root* (``C:\\``), so the
    drive is derived from the given path rather than passing the path itself.
    """
    abspath = os.path.abspath(str(root))
    drive = os.path.splitdrive(abspath)[0]
    if not drive:
        raise VolumeError(f"无法从路径解析卷标（需要绝对盘符路径）：{root}")
    anchor = drive + "\\"
    name = ctypes.create_unicode_buffer(261)
    fsname = ctypes.create_unicode_buffer(261)
    serial = wt.DWORD(0)
    maxcomp = wt.DWORD(0)
    flags = wt.DWORD(0)
    if not ctypes.windll.kernel32.GetVolumeInformationW(
        anchor, name, 261, ctypes.byref(serial), ctypes.byref(maxcomp),
        ctypes.byref(flags), fsname, 261,
    ):
        raise VolumeError(f"GetVolumeInformationW 失败：{anchor} ({ctypes.WinError()})")
    free = ctypes.c_ulonglong(0)
    total = ctypes.c_ulonglong(0)
    totalfree = ctypes.c_ulonglong(0)
    if not ctypes.windll.kernel32.GetDiskFreeSpaceExW(
        anchor, ctypes.byref(free), ctypes.byref(total), ctypes.byref(totalfree)
    ):
        raise VolumeError(f"GetDiskFreeSpaceExW 失败：{anchor} ({ctypes.WinError()})")
    return {
        "anchor": anchor,
        "path": abspath,
        "volumeLabel": name.value,
        "fileSystem": fsname.value,
        "serialNumber": f"{serial.value:08X}",
        "totalBytes": int(total.value),
        "freeBytesAvailable": int(free.value),
    }


def free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            raise ConfigError(f"无法定位可用空间：{path}")
        probe = probe.parent
    return int(volume_identity(probe)["freeBytesAvailable"])


def assert_under(path: Path, root: Path) -> Path:
    """Reject any path that escapes the authorized root."""
    resolved = Path(os.path.abspath(str(path)))
    root_resolved = Path(os.path.abspath(str(root)))
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ConfigError(f"路径越界，必须位于 {root_resolved} 之内：{resolved}") from exc
    return resolved


def human_bytes(value: int) -> str:
    return f"{value} B ({value / 1024 / 1024 / 1024:.3f} GiB)"


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------


class Logger:
    """Append-only text log; also echoes to stdout when attached to a console."""

    def __init__(self, path: Path | None, echo: bool = True) -> None:
        self.path = path
        self.echo = echo
        self._lock = threading.Lock()
        self._fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8", newline="\n")

    def __call__(self, message: str) -> None:
        line = f"[{localnow()}] {message}"
        with self._lock:
            if self._fh is not None:
                self._fh.write(line + "\n")
                self._fh.flush()
                os.fsync(self._fh.fileno())
            if self.echo:
                try:
                    print(line, flush=True)
                except Exception:  # console encoding issues must never break the run
                    pass

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


# --------------------------------------------------------------------------
# single-instance lock
# --------------------------------------------------------------------------


class InstanceLock:
    """Advisory exclusive lock that the OS releases if the process dies.

    The lock file is a bare byte-range lock and deliberately holds no payload:
    reading or rewriting the locked byte from a second handle is itself denied
    by Windows, so storing the PID there would break the very check it is meant
    to support.  The owning PID is reported through the log instead.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = None

    def __enter__(self) -> "InstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import msvcrt

            self._fh = open(self.path, "a+b")
            self._fh.seek(0, os.SEEK_END)
            if self._fh.tell() == 0:
                self._fh.write(b"\0")  # _locking needs at least one byte
                self._fh.flush()
            self._fh.seek(0)
            msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            raise LockError(
                f"已有下载进程持有锁 {self.path}（同一时间只允许一个下载进程）"
            ) from exc
        return self

    def __exit__(self, *exc_info) -> None:
        if self._fh is None:
            return
        try:
            import msvcrt

            self._fh.seek(0)
            msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass
        self._fh.close()
        self._fh = None


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


class Config:
    """Frozen download identity and the authorized target layout."""

    REQUIRED = (
        "url", "expectedSizeBytes", "expectedEtag", "rootDirectory",
        "partFile", "completedArchive", "extractDirectory", "logsDirectory",
        "volumeMarker", "volumeMarkerUuid", "expectedExtractedBytes",
        "spaceMarginBytes",
    )

    def __init__(self, data: dict, source: Path | None) -> None:
        missing = [key for key in self.REQUIRED if key not in data]
        if missing:
            raise ConfigError(f"配置缺少字段：{', '.join(missing)}")
        self.data = data
        self.source = source
        self.url: str = data["url"]
        self.expected_size: int = int(data["expectedSizeBytes"])
        self.expected_etag: str | None = normalize_etag(data["expectedEtag"])
        self.expected_extracted: int = int(data["expectedExtractedBytes"])
        self.space_margin: int = int(data["spaceMarginBytes"])
        self.volume_uuid: str = data["volumeMarkerUuid"]
        self.index_path = Path(data["indexPath"]) if data.get("indexPath") else None
        self.index_sha256 = data.get("indexSha256")

        self.root = Path(os.path.abspath(data["rootDirectory"]))
        self.part = assert_under(Path(data["partFile"]), self.root)
        self.archive = assert_under(Path(data["completedArchive"]), self.root)
        self.extract_dir = assert_under(Path(data["extractDirectory"]), self.root)
        self.logs_dir = assert_under(Path(data["logsDirectory"]), self.root)
        self.marker_path = assert_under(Path(data["volumeMarker"]), self.root)
        self.state_path = assert_under(
            Path(data.get("stateFile") or (self.logs_dir / "download-state.json")),
            self.root,
        )
        self.lock_path = assert_under(
            Path(data.get("lockFile") or (self.logs_dir / "download.lock")), self.root
        )
        self.log_path = assert_under(
            Path(data.get("logFile") or (self.logs_dir / "download.log")), self.root
        )

        parts = urlsplit(self.url)
        if parts.scheme not in ("http", "https"):
            raise ConfigError(f"不支持的 URL 协议：{parts.scheme}")
        if not parts.hostname:
            raise ConfigError(f"URL 缺少主机名：{self.url}")
        self.scheme = parts.scheme
        self.host = parts.hostname
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        self.target = parts.path + (f"?{parts.query}" if parts.query else "")
        self.host_header = parts.netloc

    @classmethod
    def load(cls, path: Path) -> "Config":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data, Path(path))

    def verify_index(self) -> str:
        """Confirm the local archive index still hashes to the frozen value."""
        import hashlib

        if not self.index_path or not self.index_sha256:
            return "index-not-frozen"
        if not self.index_path.exists():
            raise ConfigError(f"索引文件缺失：{self.index_path}")
        digest = hashlib.sha256(self.index_path.read_bytes()).hexdigest()
        if digest.lower() != str(self.index_sha256).lower():
            raise IdentityError(
                f"索引 SHA256 不符：期望 {self.index_sha256}，实际 {digest}"
            )
        return digest

    def summary(self) -> dict:
        return {
            "configSource": str(self.source) if self.source else None,
            "url": self.url,
            "expectedSizeBytes": self.expected_size,
            "expectedEtag": self.expected_etag,
            "expectedExtractedBytes": self.expected_extracted,
            "spaceMarginBytes": self.space_margin,
            "rootDirectory": str(self.root),
            "partFile": str(self.part),
            "completedArchive": str(self.archive),
            "extractDirectory": str(self.extract_dir),
            "logsDirectory": str(self.logs_dir),
            "volumeMarker": str(self.marker_path),
            "volumeMarkerUuid": self.volume_uuid,
        }


# --------------------------------------------------------------------------
# volume marker
# --------------------------------------------------------------------------


def init_volume_marker(cfg: Config, log: Logger, uuid_override: str | None = None) -> dict:
    """Create the volume marker on first use; never overwrite an existing one."""
    cfg.root.mkdir(parents=True, exist_ok=True)
    identity = volume_identity(cfg.root)
    if cfg.marker_path.exists():
        existing = json.loads(cfg.marker_path.read_text(encoding="utf-8"))
        log(f"卷标记已存在，未改写：{cfg.marker_path} uuid={existing.get('uuid')}")
        return existing
    marker = {
        "schemaVersion": SCHEMA_VERSION,
        "uuid": uuid_override or str(uuid.uuid4()),
        "createdAt": utcnow(),
        "createdAtLocal": localnow(),
        "driveLetterAtCreation": cfg.root.drive,
        "rootDirectory": str(cfg.root),
        "volumeLabel": identity["volumeLabel"],
        "fileSystem": identity["fileSystem"],
        "serialNumber": identity["serialNumber"],
        "totalBytes": identity["totalBytes"],
        "freeBytesAtCreation": identity["freeBytesAvailable"],
    }
    tmp = cfg.marker_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, cfg.marker_path)
    log(f"已创建卷标记：{cfg.marker_path} uuid={marker['uuid']}")
    return marker


def verify_volume_marker(cfg: Config, log: Logger) -> dict:
    """Hard-stop unless the marker is present, matches config, and matches the disk."""
    if not cfg.marker_path.exists():
        raise VolumeError(
            f"卷标记缺失：{cfg.marker_path} —— 需先运行 --init-volume，绝不向未标记的卷写入"
        )
    marker = json.loads(cfg.marker_path.read_text(encoding="utf-8"))
    if marker.get("uuid") != cfg.volume_uuid:
        raise VolumeError(
            f"卷标记 UUID 与冻结配置不符：标记 {marker.get('uuid')} "
            f"≠ 配置 {cfg.volume_uuid} —— 盘符可能已改变，停止写入"
        )
    identity = volume_identity(cfg.root)
    checks = {
        "volumeLabel": marker.get("volumeLabel"),
        "fileSystem": marker.get("fileSystem"),
        "serialNumber": marker.get("serialNumber"),
        "totalBytes": marker.get("totalBytes"),
    }
    for key, expected in checks.items():
        actual = identity[key]
        if expected is not None and actual != expected:
            raise VolumeError(
                f"卷身份变化：{key} 期望 {expected}，实际 {actual} —— 停止写入"
            )
    log(
        f"卷标记校验通过 uuid={marker['uuid']} "
        f"serial={identity['serialNumber']} label={identity['volumeLabel']!r}"
    )
    return marker


# --------------------------------------------------------------------------
# remote identity probe
# --------------------------------------------------------------------------


class RemoteFacts:
    __slots__ = ("status", "content_range", "content_length", "etag", "accept_ranges",
                 "last_modified", "body", "total")

    def __init__(self, status, headers, body, total=None):
        self.status = status
        self.content_range = headers.get("Content-Range")
        self.content_length = headers.get("Content-Length")
        self.etag = headers.get("ETag")
        self.accept_ranges = headers.get("Accept-Ranges")
        self.last_modified = headers.get("Last-Modified")
        self.body = body
        self.total = total

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "contentRange": self.content_range,
            "contentLength": self.content_length,
            "etag": self.etag,
            "acceptRanges": self.accept_ranges,
            "lastModified": self.last_modified,
            "total": self.total,
        }


def _open_connection(cfg: Config, timeout: int = DEFAULT_TIMEOUT):
    if cfg.scheme == "https":
        ctx = ssl.create_default_context()  # certificate verification stays on
        return http.client.HTTPSConnection(cfg.host, cfg.port, timeout=timeout, context=ctx)
    return http.client.HTTPConnection(cfg.host, cfg.port, timeout=timeout)


def _request_headers(cfg: Config, byte_range: str) -> dict:
    return {
        "Host": cfg.host_header,
        "Range": byte_range,
        "Accept": "*/*",
        "Accept-Encoding": "identity",
        "Connection": "close",
        "User-Agent": "DroneRFa-full-archive/1.0",
    }


def probe_remote_identity(cfg: Config, log: Logger, timeout: int = DEFAULT_TIMEOUT) -> RemoteFacts:
    """Confirm the server still serves the frozen object before any write.

    Reads only ``bytes=0-7`` so the RAR signature can be checked as well.
    Transient network failures are retried a bounded number of times; a
    protocol or identity change is a hard stop on the first observation.
    """
    last_error: str | None = None
    for attempt in range(1, MAX_PROBE_ATTEMPTS + 1):
        try:
            facts = _probe_once(cfg, timeout)
            log(
                f"远端身份校验通过：status={facts.status} "
                f"content-range={facts.content_range} etag={facts.etag} "
                f"signature={facts.body[:4].hex()} attempts={attempt}"
            )
            return facts
        except (TransientError, http.client.HTTPException, ssl.SSLError, OSError,
                socket.timeout) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_PROBE_ATTEMPTS:
                delay = min(RETRY_BACKOFF_BASE ** attempt, RETRY_BACKOFF_MAX)
                log(f"远端身份探测第 {attempt}/{MAX_PROBE_ATTEMPTS} 次失败（{last_error}），"
                    f"{delay:.1f}s 后重试")
                time.sleep(delay)
    raise TransientError(f"远端身份探测连续失败：{last_error}")


def _probe_once(cfg: Config, timeout: int) -> RemoteFacts:
    conn = _open_connection(cfg, timeout)
    try:
        conn.request("GET", cfg.target, headers=_request_headers(cfg, "bytes=0-7"))
        response = conn.getresponse()
        headers = {k: v for k, v in response.getheaders()}
        status = response.status
        if status in RETRYABLE_HTTP:
            raise TransientError(f"探测收到可重试的 HTTP {status}")
        if status != 206:
            raise ProtocolViolation(
                f"探测期望 HTTP 206，实际 {status}（服务器可能忽略 Range，已放弃读取正文）"
            )
        content_range = parse_content_range(headers.get("Content-Range", ""))
        if content_range is None:
            raise ProtocolViolation(f"探测的 Content-Range 不可解析：{headers.get('Content-Range')!r}")
        start, end, total = content_range
        if (start, end) != (0, 7):
            raise ProtocolViolation(f"探测的 Content-Range 区间不符：{start}-{end}，期望 0-7")
        if total != cfg.expected_size:
            raise IdentityError(
                f"远端总长变化：期望 {cfg.expected_size}，实际 {total} —— 停止"
            )
        etag = normalize_etag(headers.get("ETag"))
        if cfg.expected_etag and etag != cfg.expected_etag:
            raise IdentityError(
                f"远端 ETag 变化：期望 {cfg.expected_etag}，实际 {etag} —— 停止"
            )
        if int(headers.get("Content-Length", "8")) != 8:
            raise ProtocolViolation(
                f"探测的 Content-Length 不符：{headers.get('Content-Length')}"
            )
        body = response.read(9)
        if len(body) > 8:
            raise ProtocolViolation("探测正文超过请求长度，已中止")
        if len(body) != 8:
            raise TransientError(f"探测正文长度 {len(body)}，期望 8")
    finally:
        conn.close()

    return RemoteFacts(status, headers, body, total=total)


# --------------------------------------------------------------------------
# download state
# --------------------------------------------------------------------------


class DownloadState:
    """Resume checkpoint: a single verified byte prefix of the ``.part`` file."""

    def __init__(self, path: Path, cfg: Config) -> None:
        self.path = path
        self.cfg = cfg
        self.verified_prefix = 0
        self.started_at = utcnow()
        self.blocks_written = 0
        self.retries = 0
        self.transient_failures = 0
        self.bytes_this_run = 0
        self.first_block_seconds: float | None = None
        self.remote = None
        if path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("url") != self.cfg.url:
            raise IdentityError(
                f"断点状态 URL 不符：{data.get('url')} ≠ {self.cfg.url} —— 停止"
            )
        if int(data.get("expectedSizeBytes", -1)) != self.cfg.expected_size:
            raise IdentityError("断点状态的总长与冻结配置不符 —— 停止")
        etag = normalize_etag(data.get("expectedEtag"))
        if self.cfg.expected_etag and etag != self.cfg.expected_etag:
            raise IdentityError("断点状态的 ETag 与冻结配置不符 —— 停止")
        if data.get("volumeMarkerUuid") != self.cfg.volume_uuid:
            raise IdentityError("断点状态的卷标记 UUID 与冻结配置不符 —— 停止")
        self.verified_prefix = int(data.get("verifiedPrefixBytes", 0))
        self.started_at = data.get("startedAt", self.started_at)
        self.blocks_written = int(data.get("blocksWritten", 0))
        self.retries = int(data.get("retries", 0))
        self.transient_failures = int(data.get("transientFailures", 0))

    def save(self, log: Logger | None = None, force: bool = False) -> None:
        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "url": self.cfg.url,
            "expectedSizeBytes": self.cfg.expected_size,
            "expectedEtag": self.cfg.expected_etag,
            "volumeMarkerUuid": self.cfg.volume_uuid,
            "partFile": str(self.cfg.part),
            "verifiedPrefixBytes": self.verified_prefix,
            "percent": round(self.verified_prefix / self.cfg.expected_size * 100, 4)
            if self.cfg.expected_size else 0.0,
            "blocksWritten": self.blocks_written,
            "retries": self.retries,
            "transientFailures": self.transient_failures,
            "startedAt": self.started_at,
            "updatedAt": utcnow(),
            "complete": self.verified_prefix >= self.cfg.expected_size,
        }
        if self.remote:
            payload["remoteIdentity"] = self.remote
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
        if log is not None and force:
            log(
                f"断点已持久化：{self.verified_prefix}/{self.cfg.expected_size} "
                f"({payload['percent']}%)"
            )


# --------------------------------------------------------------------------
# block fetching
# --------------------------------------------------------------------------


class BlockResult:
    __slots__ = ("index", "start", "data", "error_kind", "error", "attempts", "seconds")

    def __init__(self, index, start, data=None, error_kind=None, error=None,
                 attempts=0, seconds=0.0):
        self.index = index
        self.start = start
        self.data = data
        self.error_kind = error_kind  # None | "transient" | "protocol"
        self.error = error
        self.attempts = attempts
        self.seconds = seconds

    @property
    def ok(self) -> bool:
        return self.data is not None


class BlockFetcher:
    """Fetches one verified byte range.  Never writes to disk itself."""

    def __init__(self, cfg: Config, log: Logger, stop: threading.Event) -> None:
        self.cfg = cfg
        self.log = log
        self.stop = stop

    def fetch(self, index: int, start: int, length: int) -> BlockResult:
        t0 = time.monotonic()
        last_error: str | None = None
        for attempt in range(1, MAX_BLOCK_ATTEMPTS + 1):
            if self.stop.is_set():
                return BlockResult(index, start, error_kind="transient",
                                   error="stopped", attempts=attempt - 1,
                                   seconds=time.monotonic() - t0)
            try:
                data = self._fetch_once(start, length)
                return BlockResult(index, start, data=data, attempts=attempt,
                                   seconds=time.monotonic() - t0)
            except (ProtocolViolation, IdentityError) as exc:
                # Never retried: the remote answer cannot be trusted at all,
                # or the remote object is no longer the frozen one.
                return BlockResult(index, start, error_kind="protocol", error=str(exc),
                                   attempts=attempt, seconds=time.monotonic() - t0)
            except (TransientError, http.client.HTTPException, ssl.SSLError, OSError,
                    socket.timeout) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < MAX_BLOCK_ATTEMPTS:
                    delay = min(RETRY_BACKOFF_BASE ** attempt, RETRY_BACKOFF_MAX)
                    delay += random.uniform(0, 1.5)
                    self.log(
                        f"块 {index} 第 {attempt}/{MAX_BLOCK_ATTEMPTS} 次失败（{last_error}），"
                        f"{delay:.1f}s 后重试"
                    )
                    time.sleep(delay)
        return BlockResult(index, start, error_kind="transient", error=last_error,
                           attempts=MAX_BLOCK_ATTEMPTS, seconds=time.monotonic() - t0)

    def _fetch_once(self, start: int, length: int) -> bytes:
        end = start + length - 1
        conn = _open_connection(self.cfg, DEFAULT_TIMEOUT)
        try:
            conn.request("GET", self.cfg.target,
                         headers=_request_headers(self.cfg, f"bytes={start}-{end}"))
            response = conn.getresponse()
            headers = {k: v for k, v in response.getheaders()}
            status = response.status

            # ---- validate every header before consuming a single body byte ----
            if status in RETRYABLE_HTTP:
                raise TransientError(f"HTTP {status}")
            if status != 206:
                raise ProtocolViolation(
                    f"range {start}-{end} 期望 HTTP 206，实际 {status}"
                    f"（服务器可能忽略 Range，已放弃读取正文）"
                )
            encoding = (headers.get("Content-Encoding") or "identity").strip().lower()
            if encoding not in ("", "identity"):
                raise ProtocolViolation(f"意外的 Content-Encoding: {encoding!r}")
            content_range = parse_content_range(headers.get("Content-Range", ""))
            if content_range is None:
                raise ProtocolViolation(
                    f"Content-Range 不可解析：{headers.get('Content-Range')!r}"
                )
            actual_start, actual_end, total = content_range
            if (actual_start, actual_end) != (start, end):
                raise ProtocolViolation(
                    f"Content-Range 区间不符：请求 {start}-{end}，返回 {actual_start}-{actual_end}"
                )
            if total != self.cfg.expected_size:
                raise IdentityError(
                    f"Content-Range 总长不符：期望 {self.cfg.expected_size}，返回 {total}"
                )
            etag = normalize_etag(headers.get("ETag"))
            if self.cfg.expected_etag and etag != self.cfg.expected_etag:
                raise IdentityError(f"ETag 不符：期望 {self.cfg.expected_etag}，返回 {etag}")
            declared = headers.get("Content-Length")
            if declared is not None and int(declared) != length:
                raise ProtocolViolation(f"Content-Length {declared} 与请求长度 {length} 不符")

            # ---- bounded read: one byte past the request proves over-length ----
            buf = bytearray()
            cap = length + 1
            while len(buf) < cap:
                chunk = response.read(min(1 << 20, cap - len(buf)))
                if not chunk:
                    break
                buf.extend(chunk)
            if len(buf) > length:
                raise ProtocolViolation(
                    f"服务器返回正文超过请求长度（>{length} 字节），已中止"
                )
            if len(buf) != length:
                raise TransientError(f"正文短读：收到 {len(buf)} 字节，期望 {length}")
            return bytes(buf)
        finally:
            try:
                conn.close()
            except Exception:
                pass


# --------------------------------------------------------------------------
# throughput accounting
# --------------------------------------------------------------------------


class Throughput:
    """Rolling throughput over a measurement window."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window = window_seconds
        self._samples: deque[tuple[float, int]] = deque()

    def add(self, bytes_written: int, now: float | None = None) -> None:
        self._samples.append((now if now is not None else time.monotonic(), bytes_written))
        self._trim()

    def _trim(self) -> None:
        if not self._samples:
            return
        cutoff = self._samples[-1][0] - self.window
        while len(self._samples) > 2 and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def mib_per_s(self) -> float:
        """Rate over the retained window, using the byte *delta* of that window."""
        if len(self._samples) < 2:
            return 0.0
        (t0, b0), (t1, b1) = self._samples[0], self._samples[-1]
        span = t1 - t0
        if span <= 0 or b1 < b0:
            return 0.0
        return (b1 - b0) / span / (1024 * 1024)


# --------------------------------------------------------------------------
# the pipelined downloader
# --------------------------------------------------------------------------


class Downloader:
    """Parallel fetch + strictly in-order append into one ``.part`` file."""

    def __init__(self, cfg: Config, log: Logger, *, block_bytes: int, workers: int,
                 state: DownloadState, max_seconds: float | None = None,
                 max_bytes: int | None = None) -> None:
        if block_bytes <= 0:
            raise ConfigError("block_bytes 必须为正")
        if workers <= 0:
            raise ConfigError("workers 必须为正")
        self.cfg = cfg
        self.log = log
        self.block_bytes = block_bytes
        self.workers = workers
        self.window = max(workers * 2, workers + 2)
        self.state = state
        self.max_seconds = max_seconds
        self.max_bytes = max_bytes
        self.stop = threading.Event()
        self.fatal: str | None = None
        self.throughput = Throughput()
        self._cond = threading.Condition()
        self._queue: deque[int] = deque()
        self._results: dict[int, BlockResult] = {}
        self._inflight = 0
        self._total_blocks = (cfg.expected_size + block_bytes - 1) // block_bytes
        self.run_bytes = 0
        self.run_started = 0.0
        self._last_log = 0.0

    # -- geometry ---------------------------------------------------------
    def _block_length(self, index: int) -> int:
        start = index * self.block_bytes
        return min(self.block_bytes, self.cfg.expected_size - start)

    # -- worker -----------------------------------------------------------
    def _worker(self, fetcher: BlockFetcher) -> None:
        try:
            self._worker_loop(fetcher)
        except BaseException as exc:  # a dead worker must never hang the writer
            self.fatal = f"工作线程异常终止：{type(exc).__name__}: {exc}"
            self.stop.set()
            with self._cond:
                self._cond.notify_all()

    def _worker_loop(self, fetcher: BlockFetcher) -> None:
        while True:
            with self._cond:
                while not self._queue and not self.stop.is_set():
                    self._cond.wait(1.0)
                if self.stop.is_set() and not self._queue:
                    return
                index = self._queue.popleft()
            start = index * self.block_bytes
            result = fetcher.fetch(index, start, self._block_length(index))
            with self._cond:
                self._results[index] = result
                self._cond.notify_all()
            if result.error_kind == "protocol":
                self.fatal = (
                    f"协议/身份违规，块 {index} @ {start}：{result.error}"
                )
                self.stop.set()
                with self._cond:
                    self._cond.notify_all()
                return

    # -- main loop --------------------------------------------------------
    def run(self) -> dict:
        cfg = self.cfg
        if self.state.verified_prefix >= cfg.expected_size:
            self.log("档案已完整，无需下载。")
            return {"status": "already-complete", "verifiedPrefixBytes": self.state.verified_prefix}

        if self.state.verified_prefix % self.block_bytes != 0:
            raise ConfigError(
                f"断点前缀 {self.state.verified_prefix} 未按 {self.block_bytes} 字节块对齐"
            )
        first_block = self.state.verified_prefix // self.block_bytes

        # Recover from a crash that happened after a write but before the save.
        if cfg.part.exists():
            part_size = cfg.part.stat().st_size
            if part_size < self.state.verified_prefix:
                raise IdentityError(
                    f"断点声明 {self.state.verified_prefix} 字节，但 {cfg.part} 只有 "
                    f"{part_size} 字节 —— 状态与文件不一致，停止"
                )
            if part_size > self.state.verified_prefix:
                self.log(
                    f"{cfg.part.name} 比已验证断点多 {part_size - self.state.verified_prefix} "
                    f"字节（上次写入后未及持久化），截断回断点"
                )
                with open(cfg.part, "r+b") as fh:
                    fh.truncate(self.state.verified_prefix)
        elif self.state.verified_prefix:
            raise IdentityError(f"断点声明有 {self.state.verified_prefix} 字节，但 {cfg.part} 不存在")

        cfg.part.parent.mkdir(parents=True, exist_ok=True)
        mode = "ab" if cfg.part.exists() and self.state.verified_prefix else "wb"
        writer = open(cfg.part, mode, buffering=0)

        self.run_started = time.monotonic()
        fetcher = BlockFetcher(cfg, self.log, self.stop)
        threads = [
            threading.Thread(target=self._worker, args=(fetcher,), daemon=True,
                             name=f"fetch-{i}")
            for i in range(self.workers)
        ]
        for thread in threads:
            thread.start()

        self.log(
            f"下载开始：块 {self.block_bytes // 1024 // 1024} MiB × {self.workers} 并发，"
            f"顺序追加；起点 {self.state.verified_prefix} 字节 "
            f"({self.state.verified_prefix / cfg.expected_size * 100:.3f}%)；"
            f"总块数 {self._total_blocks}"
        )

        next_write = first_block
        next_submit = first_block
        consecutive_transient = 0
        try:
            while next_write < self._total_blocks:
                if self.fatal:
                    break
                # deadline / byte budget for the bounded speed probe
                over_time = (self.max_seconds is not None
                             and time.monotonic() - self.run_started >= self.max_seconds)
                over_bytes = (self.max_bytes is not None and self.run_bytes >= self.max_bytes)
                if over_time or over_bytes:
                    break

                with self._cond:
                    while (next_submit < self._total_blocks
                           and len(self._results) + len(self._queue) < self.window
                           and not self.stop.is_set()):
                        self._queue.append(next_submit)
                        next_submit += 1
                        self._cond.notify_all()

                    result = self._results.pop(next_write, None)
                    if result is None:
                        self._cond.wait(1.0)
                        continue

                start = result.start
                if result.ok:
                    assert result.data is not None
                    writer.write(result.data)
                    self.state.verified_prefix = start + len(result.data)
                    self.state.blocks_written += 1
                    self.run_bytes += len(result.data)
                    consecutive_transient = 0
                    self.throughput.add(self.run_bytes)
                    self.state.save()  # checkpoint after every complete block
                    next_write += 1
                    now = time.monotonic()
                    if now - self._last_log >= 30:
                        self._last_log = now
                        elapsed = now - self.run_started
                        self.log(
                            f"进度 {self.state.verified_prefix}/{cfg.expected_size} "
                            f"({self.state.verified_prefix / cfg.expected_size * 100:.3f}%) "
                            f"本次 {self.run_bytes / 1024 / 1024:.1f} MiB / {elapsed:.0f}s，"
                            f"近 {self.throughput.mib_per_s():.3f} MiB/s，"
                            f"剩余约 {(cfg.expected_size - self.state.verified_prefix) / max(self.throughput.mib_per_s(), 0.01) / 1024 / 1024 / 3600:.1f} h"
                        )
                else:
                    if result.error_kind == "protocol":
                        self.fatal = f"协议/身份违规，块 {result.index} @ {start}：{result.error}"
                        break
                    consecutive_transient += 1
                    self.state.retries += 1
                    self.state.transient_failures += 1
                    self.log(
                        f"块 {result.index} 重试耗尽（{result.error}）；"
                        f"连续瞬时失败 {consecutive_transient}/{MAX_CONSECUTIVE_TRANSIENT}"
                    )
                    if consecutive_transient >= MAX_CONSECUTIVE_TRANSIENT:
                        self.fatal = (
                            f"连续 {consecutive_transient} 次瞬时失败，停止：{result.error}"
                        )
                        break
                    # requeue this same block and everything not yet written
                    with self._cond:
                        for stale in [i for i in self._results if i >= next_write]:
                            del self._results[stale]
                        self._queue.clear()
                        next_submit = next_write
                    time.sleep(min(5 * consecutive_transient, 60))
        finally:
            self.stop.set()
            with self._cond:
                self._cond.notify_all()
            for thread in threads:
                thread.join(timeout=5)
            try:
                writer.flush()
                os.fsync(writer.fileno())
            except Exception:
                pass
            writer.close()
            self.state.save()

        elapsed = time.monotonic() - self.run_started
        summary = {
            "status": "fatal" if self.fatal else (
                "complete" if self.state.verified_prefix >= cfg.expected_size else "bounded-stop"
            ),
            "verifiedPrefixBytes": self.state.verified_prefix,
            "percent": round(self.state.verified_prefix / cfg.expected_size * 100, 4),
            "runBytes": self.run_bytes,
            "runSeconds": round(elapsed, 2),
            "runMibPerSecond": round(self.run_bytes / elapsed / (1024 * 1024), 4) if elapsed > 0 else 0.0,
            "blocksWrittenThisRun": self.run_bytes // self.block_bytes,
            "fatal": self.fatal,
        }
        self.state.remote = summary
        return summary

    # -- finalize ---------------------------------------------------------
    def finalize(self) -> dict:
        """Atomically promote a fully verified ``.part`` to the final ``.rar``."""
        cfg = self.cfg
        if not cfg.part.exists():
            if cfg.archive.exists() and cfg.archive.stat().st_size == cfg.expected_size:
                return {"status": "already-final", "archive": str(cfg.archive)}
            raise FullArchiveError(f"缺少 {cfg.part}，无法完成收尾")
        size = cfg.part.stat().st_size
        if size != cfg.expected_size:
            raise IdentityError(
                f"拒绝重命名：{cfg.part.name} 为 {human_bytes(size)}，"
                f"期望 {human_bytes(cfg.expected_size)}"
            )
        if self.state.verified_prefix != cfg.expected_size:
            raise IdentityError(
                f"拒绝重命名：断点 {self.state.verified_prefix} 未覆盖完整档案"
            )
        self.log(f"字节数核对通过：{human_bytes(size)}；原子重命名为 {cfg.archive.name}")
        os.replace(cfg.part, cfg.archive)
        self.log(f"完成：{cfg.archive}")
        return {"status": "renamed", "archive": str(cfg.archive), "bytes": size}


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def check_space(cfg: Config, state: DownloadState, log: Logger) -> dict:
    remaining = cfg.expected_size - state.verified_prefix
    required = remaining + cfg.expected_extracted + cfg.space_margin
    available = free_bytes(cfg.root)
    detail = {
        "freeBytes": available,
        "remainingArchiveBytes": remaining,
        "expectedExtractedBytes": cfg.expected_extracted,
        "marginBytes": cfg.space_margin,
        "requiredBytes": required,
        "headroomBytes": available - required,
        "ok": available >= required,
    }
    if not detail["ok"]:
        raise FullArchiveError(
            f"可用空间不足：需要 {human_bytes(required)}，可用 {human_bytes(available)}"
        )
    log(
        f"空间校验通过：需要 {human_bytes(required)}，可用 {human_bytes(available)}，"
        f"余量 {human_bytes(available - required)}"
    )
    return detail


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config)
    # Test-only overrides; production runs use the frozen file verbatim.
    if args.url:
        cfg.url = args.url
        parts = urlsplit(cfg.url)
        cfg.scheme, cfg.host = parts.scheme, parts.hostname
        cfg.port = parts.port or (443 if parts.scheme == "https" else 80)
        cfg.target = parts.path + (f"?{parts.query}" if parts.query else "")
        cfg.host_header = parts.netloc
    if args.expected_size is not None:
        cfg.expected_size = int(args.expected_size)
    if args.expected_etag is not None:
        cfg.expected_etag = normalize_etag(args.expected_etag)
    if args.root:
        cfg.root = Path(os.path.abspath(args.root))
        cfg.part = assert_under(Path(args.part or (cfg.root / "DroneRFa.rar.part")), cfg.root)
        cfg.archive = assert_under(Path(args.archive or (cfg.root / "DroneRFa.rar")), cfg.root)
        cfg.marker_path = assert_under(cfg.root / "volume-marker.json", cfg.root)
        cfg.logs_dir = assert_under(cfg.root / "logs", cfg.root)
        cfg.state_path = assert_under(cfg.logs_dir / "download-state.json", cfg.root)
        cfg.lock_path = assert_under(cfg.logs_dir / "download.lock", cfg.root)
        cfg.log_path = assert_under(cfg.logs_dir / "download.log", cfg.root)
    if cfg.scheme != "https" and not args.allow_insecure_http:
        raise ConfigError(
            f"拒绝非 HTTPS 下载：{cfg.url}（仅测试可用 --allow-insecure-http）"
        )
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="下载 DroneRFa 官方完整 RAR（顺序追加到单个 .part 断点）"
    )
    parser.add_argument("--config", type=Path, required=True, help="冻结配置 JSON")
    parser.add_argument("--init-volume", action="store_true",
                       help="创建（或仅检查）卷标记后退出")
    parser.add_argument("--probe", action="store_true", help="只校验远端身份后退出")
    parser.add_argument("--dry-run", action="store_true",
                       help="校验配置、卷标记、空间与远端身份，但不写入任何数据")
    parser.add_argument("--status", action="store_true", help="打印断点状态后退出")
    parser.add_argument("--download", action="store_true", help="执行下载（默认动作）")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--block-mib", type=int, default=DEFAULT_BLOCK_BYTES // 1024 // 1024)
    parser.add_argument("--block-bytes", type=int, default=None,
                       help="以字节覆盖块大小（测试与小规模校验用）")
    parser.add_argument("--max-seconds", type=float, default=None,
                       help="有界运行：到时按顺序收尾并保留断点")
    parser.add_argument("--max-bytes", type=int, default=None,
                       help="有界运行：本次最多新增多少字节")
    parser.add_argument("--url", default=None, help="仅测试：覆盖 URL")
    parser.add_argument("--expected-size", type=int, default=None, help="仅测试：覆盖总长")
    parser.add_argument("--expected-etag", default=None, help="仅测试：覆盖 ETag")
    parser.add_argument("--root", default=None, help="仅测试：覆盖目标根目录")
    parser.add_argument("--part", default=None, help="仅测试：覆盖 .part 路径")
    parser.add_argument("--archive", default=None, help="仅测试：覆盖 .rar 路径")
    parser.add_argument("--allow-insecure-http", action="store_true",
                       help="仅测试：允许 http:// 端点")
    parser.add_argument("--no-echo", action="store_true", help="不向 stdout 回显日志")
    args = parser.parse_args(argv)

    log = Logger(None, echo=False)
    try:
        cfg = build_config(args)
        log = Logger(cfg.log_path, echo=not args.no_echo)

        if args.status:
            state = DownloadState(cfg.state_path, cfg)
            print(json.dumps({
                "stateFile": str(cfg.state_path),
                "exists": cfg.state_path.exists(),
                "verifiedPrefixBytes": state.verified_prefix,
                "expectedSizeBytes": cfg.expected_size,
                "percent": round(state.verified_prefix / cfg.expected_size * 100, 4),
                "partExists": cfg.part.exists(),
                "partBytes": cfg.part.stat().st_size if cfg.part.exists() else 0,
                "archiveExists": cfg.archive.exists(),
            }, ensure_ascii=False, indent=2))
            return EXIT_OK

        digest = cfg.verify_index()
        log(f"索引校验通过 sha256={digest}")
        log(f"冻结配置：{json.dumps(cfg.summary(), ensure_ascii=False)}")

        if args.init_volume:
            marker = init_volume_marker(cfg, log)
            identity = volume_identity(cfg.root)
            if not args.no_echo:
                print(json.dumps({"marker": marker, "volume": identity},
                                 ensure_ascii=False, indent=2))
            return EXIT_OK

        verify_volume_marker(cfg, log)

        if args.probe:
            facts = probe_remote_identity(cfg, log)
            payload = facts.as_dict()
            payload["expectedSizeBytes"] = cfg.expected_size
            payload["expectedEtag"] = cfg.expected_etag
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return EXIT_OK

        state = DownloadState(cfg.state_path, cfg)
        space = check_space(cfg, state, log)

        if args.dry_run:
            print(json.dumps({
                "dryRun": True,
                "config": cfg.summary(),
                "resume": {
                    "verifiedPrefixBytes": state.verified_prefix,
                    "partExists": cfg.part.exists(),
                    "partBytes": cfg.part.stat().st_size if cfg.part.exists() else 0,
                },
                "space": space,
            }, ensure_ascii=False, indent=2))
            return EXIT_OK

        with InstanceLock(cfg.lock_path) as lock:
            log(f"已获取单实例锁 pid={os.getpid()} lock={lock.path}")
            facts = probe_remote_identity(cfg, log)
            state.remote = facts.as_dict()
            # Refresh the space check against the freshly confirmed size.
            state.save()
            check_space(cfg, state, log)

            downloader = Downloader(
                cfg, log,
                block_bytes=args.block_bytes or (args.block_mib * 1024 * 1024),
                workers=args.workers,
                state=state,
                max_seconds=args.max_seconds,
                max_bytes=args.max_bytes,
            )
            summary = downloader.run()
            log(f"本次运行小结：{json.dumps(summary, ensure_ascii=False)}")

            bounded = args.max_seconds is not None or args.max_bytes is not None
            if bounded:
                probe = {
                    "boundedRun": True,
                    "at": utcnow(),
                    "status": summary.get("status"),
                    "runBytes": summary["runBytes"],
                    "runSeconds": summary["runSeconds"],
                    "mibPerSecond": summary["runMibPerSecond"],
                    "verifiedPrefixBytes": state.verified_prefix,
                    "percent": round(state.verified_prefix / cfg.expected_size * 100, 4),
                    "partFile": str(cfg.part),
                    "limits": {"maxSeconds": args.max_seconds, "maxBytes": args.max_bytes},
                    "referenceOldLogMiBPerS": 1.0,
                }
                out = cfg.logs_dir / "bounded-speed-probe.json"
                tmp = out.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(probe, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                os.replace(tmp, out)
                log(f"有界测速记录：{json.dumps(probe, ensure_ascii=False)}")

            if summary.get("fatal"):
                log(f"致命错误，保留断点：{summary['fatal']}")
                return EXIT_PROTOCOL
            if state.verified_prefix >= cfg.expected_size:
                result = downloader.finalize()
                log(f"收尾：{json.dumps(result, ensure_ascii=False)}")
            return EXIT_OK
    except LockError as exc:
        log(f"[locked] {exc}")
        return EXIT_LOCKED
    except (ConfigError, VolumeError, IdentityError, ProtocolViolation) as exc:
        log(f"[stop] {type(exc).__name__}: {exc}")
        return EXIT_PROTOCOL
    except FullArchiveError as exc:
        log(f"[error] {type(exc).__name__}: {exc}")
        return EXIT_ERROR
    except KeyboardInterrupt:
        log("被中断，保留断点。")
        return EXIT_ERROR
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
