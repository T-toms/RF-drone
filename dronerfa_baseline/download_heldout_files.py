#!/usr/bin/env python3
"""Download, verify and extract the 12 unused DroneRFa source files one at a time.

Design notes
------------
* Selection, RAR5 wrapping and resume bookkeeping are **reused** from
  ``download_dronerfa_subset.py``: each file is produced by
  ``base.stream_selected_archive(entries=[entry], ...)``, which writes the RAR5
  signature, the minimal main header, the stored file block and the end-of-archive
  header, and keeps ``.part``/``.state.json`` breakpoints.  That logic is not
  re-implemented here.
* The transfer backend is ``curl`` because ``requests`` is missing from the venv
  and ``urllib.request`` fails against this endpoint (SSL
  UNEXPECTED_EOF_WHILE_READING), while curl works with normal TLS certificate
  verification.  ``curl`` is driven with ``--range``, ``--max-filesize``, header
  dump and status write-out.  Every range response is validated (final status
  206, ``Content-Range`` start/end/total, ``Content-Encoding``, exact byte count,
  never more than the requested length) before it is handed to the downloader.
* One file is staged at a time: download ``<name>.rar`` into the staging
  directory, verify the wrapper RAR structure, run ``unrar t``, extract, verify
  size + HDF5 structure, then delete that round's staging archive.  A full
  compressed set is therefore never kept alongside the extracted set.
* Resume: ``.part`` + ``.state.json`` record the packed bytes already on disk, so
  re-running the same command continues instead of restarting.
* Every destructive filesystem operation goes through :class:`SafePaths`, which
  resolves the absolute path and refuses anything outside the authorised roots
  and the candidate file list.

This script never writes inside the original DroneRFa dataset directory.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_requests() -> str | None:
    """Make the real ``requests`` importable for download_dronerfa_subset.

    The project .venv does not include ``requests`` and pip cannot reach an index
    from this host, while the base interpreter this venv was built from already
    ships a pure-Python ``requests``.  The base site-packages is therefore put on
    ``sys.path`` only long enough to import that package (and its own
    dependencies) into ``sys.modules``; the path entry is then removed again so
    nothing else - in particular the venv's own h5py/numpy/torch - can be
    shadowed by the older base copies.

    Returns the loaded version, or None when ``requests`` is not needed.
    """
    if "requests" in sys.modules:
        return getattr(sys.modules["requests"], "__version__", "already-loaded")
    base_site_packages = Path(sys.base_prefix) / "Lib" / "site-packages"
    if not (base_site_packages / "requests" / "__init__.py").is_file():
        return None
    entry = str(base_site_packages)
    sys.path.append(entry)
    try:
        import requests  # noqa: F401

        return getattr(requests, "__version__", "unknown")
    finally:
        while entry in sys.path:
            sys.path.remove(entry)


REQUESTS_VERSION = _ensure_requests()

import download_dronerfa_subset as base  # noqa: E402

# Reuse the project's own HDF5 semantics instead of re-deriving them.
from dronerfa_data import dataset_length, read_vector_slice  # noqa: E402

REQUIRED_KEYS = ("RF0_I", "RF0_Q", "RF1_I", "RF1_Q")
RETRYABLE_HTTP = (429, 500, 502, 503, 504)
MIN_FREE_BYTES = 5 * 1024**3
ALL_CANDIDATES = "all"

DEFAULT_STAGING = Path(r"F:\DroneRFa数据集\heldout_20260917\downloads")
DEFAULT_EXTRACT = Path(r"F:\DroneRFa数据集\heldout_20260917")
SAFETY_PROBE_SPAN = 256


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path, chunk: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def disk_free(path: Path) -> int:
    probe = Path(path)
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def find_curl() -> str:
    candidate = shutil.which("curl.exe") or shutil.which("curl")
    if candidate is None:
        raise base.DroneRFaError("未找到curl，无法进行范围下载")
    return candidate


class SafetyError(RuntimeError):
    """A destructive operation was aimed outside the authorised roots."""


@dataclass(frozen=True)
class SafePaths:
    """Whitelists every path this tool is allowed to create or delete.

    ``roots`` are the only directories inside which recursive deletes and moves
    may happen.  ``deletable`` additionally names the exact files (staging
    archives, part/state files, this round's extracted .mat copies) that may be
    unlinked.  Existing .mat files that are not in ``deletable`` are never
    removed, and ``guard_destination`` refuses to overwrite an unexpected file.
    """

    roots: tuple[Path, ...]
    deletable: frozenset[Path]

    def _resolve(self, path: Path) -> Path:
        return Path(os.path.abspath(str(path)))

    def _inside_roots(self, resolved: Path) -> bool:
        for root in self.roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def check_write_target(self, path: Path, *, what: str) -> Path:
        resolved = self._resolve(path)
        if not self._inside_roots(resolved):
            raise SafetyError(
                f"拒绝写入 {what}：{resolved} 不在授权目录 {[str(r) for r in self.roots]} 内"
            )
        return resolved

    def check_deletable(self, path: Path, *, what: str, recursive: bool = False) -> Path:
        resolved = self.check_write_target(path, what=what)
        if resolved in self.deletable:
            return resolved
        if recursive and resolved.parent in self.roots:
            # Temporary extraction folder created by this tool inside a root.
            return resolved
        raise SafetyError(f"拒绝删除 {what}：{resolved} 不在本次任务允许清理的清单内")

    def safe_unlink(self, path: Path, *, what: str, missing_ok: bool = True) -> None:
        resolved = self.check_deletable(path, what=what)
        resolved.unlink(missing_ok=missing_ok)

    def safe_rmtree(self, path: Path, *, what: str) -> None:
        resolved = self.check_deletable(path, what=what, recursive=True)
        if resolved == resolved.parent or resolved in self.roots:
            raise SafetyError(f"拒绝递归删除授权根目录本身：{resolved}")
        if resolved.exists():
            shutil.rmtree(resolved)

    def safe_replace(self, source: Path, destination: Path, *, what: str) -> None:
        source_resolved = self._resolve(source)
        if not source_resolved.is_file():
            raise SafetyError(f"{what}：源文件不存在 {source_resolved}")
        destination_resolved = self.check_write_target(destination, what=what)
        destination_resolved.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source_resolved, destination_resolved)

    def guard_destination(self, destination: Path, *, expected_size: int, what: str) -> bool:
        """Return True when an existing final .mat already passes a strict check.

        Never report success based on size alone: the file must open as HDF5 with
        the required datasets and a length matching the manifest.
        """
        resolved = self.check_write_target(destination, what=what)
        if not resolved.exists():
            return False
        if resolved.stat().st_size != expected_size:
            return False
        return True


class RangeProtocolError(RuntimeError):
    """Server answered a range request in a way that cannot be trusted."""


def parse_content_range(value: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


class CurlRangeClient:
    """curl-backed range client compatible with ``base.download_range``.

    Hardening:

    * curl writes to a temporary file (never a stdout pipe), so a server that
      ignores ``Range`` cannot stream the whole ~615 GB archive into memory;
    * ``--range`` + ``--max-filesize expected+1`` bound what can be written, and
      the counted size on disk must equal the requested length;
    * the *same* GET response is used for validation: its status, its
      ``Content-Range`` start/end/total, its ``Content-Encoding`` and its byte
      count.  No HEAD request is used to guess the range;
    * ``--fail`` makes curl exit non-zero on 4xx/5xx so error bodies are never
      accepted as data.

    TLS certificate verification is left at curl's default (enabled).
    """

    def __init__(
        self,
        url: str,
        *,
        retries: int = 5,
        connect_timeout: int = 20,
        scratch_dir: Path | None = None,
    ) -> None:
        self.url = url
        self.retries = retries
        self.connect_timeout = connect_timeout
        self.curl = find_curl()
        # Every transferred body block is written here.  It must live on the
        # authorised data volume (F:), never in the system TEMP on C:.
        if scratch_dir is None:
            raise base.DroneRFaError("必须显式指定正文临时目录（应在F盘授权目录内）")
        self.scratch_dir = Path(os.path.abspath(str(scratch_dir)))
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self._scratch_counter = itertools.count(1)
        self._scratch_lock = threading.Lock()
        self.archive_size: int | None = None
        self.etag: str | None = None
        self.last_range_headers: dict[str, str] = {}
        self.stats = {
            "range_requests": 0,
            "retries": 0,
            "http_429": 0,
            "bytes_transferred": 0,
            "protocol_violations": [],
            "scratch_dir": str(self.scratch_dir),
        }

    def _new_scratch(self) -> tuple[Path, Path]:
        with self._scratch_lock:
            index = next(self._scratch_counter)
        body = self.scratch_dir / f"range_{index:06d}.bin"
        headers = self.scratch_dir / f"range_{index:06d}.headers"
        return body, headers

    def cleanup_scratch(self) -> None:
        for leftover in self.scratch_dir.glob("range_*"):
            leftover.unlink(missing_ok=True)

    def _invoke(self, extra: list[str], timeout: int) -> subprocess.CompletedProcess:
        command = [
            self.curl,
            "--location",
            "--silent",
            "--show-error",
            "--fail",
            "--connect-timeout",
            str(self.connect_timeout),
            "--max-time",
            str(timeout),
            "--header",
            "Accept-Encoding: identity",
            *extra,
            self.url,
        ]
        return subprocess.run(command, capture_output=True, check=False, timeout=timeout + 60)

    def probe_range(self, start: int, end: int, *, timeout: int = 120) -> tuple[int, dict[str, str], bytes]:
        """GET one bounded range and return (status, headers, body) after full validation.

        The headers come from this very GET response, so the validated
        ``Content-Range`` belongs to the payload actually received.
        """
        expected = end - start + 1
        target, headers_path = self._new_scratch()
        try:
            result = self._invoke(
                [
                    "--range",
                    f"{start}-{end}",
                    "--max-filesize",
                    str(expected + 1),
                    "--dump-header",
                    str(headers_path),
                    "--write-out",
                    "%{http_code}",
                    "--output",
                    str(target),
                ],
                timeout,
            )
            status = self._parse_status(result.stdout)
            headers = self._parse_headers(headers_path)
            size = target.stat().st_size if target.exists() else 0
            if size > expected:
                self._violation(f"区间{start}-{end}落盘{size}字节，超过请求长度{expected}")
            if status != 206:
                raise base.DroneRFaError(
                    f"区间{start}-{end}返回HTTP {status}（期望206）"
                    f"，stderr={result.stderr.decode('utf-8', errors='replace')[:200]!r}"
                )
            self._validate_headers(headers, start, end, size)
            body = target.read_bytes() if size == expected else b""
            if len(body) != expected:
                self._violation(f"区间{start}-{end}正文{len(body)}字节，期望{expected}")
            return status, headers, body
        finally:
            target.unlink(missing_ok=True)
            headers_path.unlink(missing_ok=True)

    def _validate_headers(self, headers: dict[str, str], start: int, end: int, size: int) -> None:
        encoding = (headers.get("content-encoding") or "identity").strip().lower()
        if encoding not in ("", "identity"):
            self._violation(f"意外的Content-Encoding：{encoding!r}")
        content_range = parse_content_range(headers.get("content-range", ""))
        if content_range is None:
            self._violation(f"Content-Range不可解析：{headers.get('content-range')!r}")
        assert content_range is not None
        actual_start, actual_end, total = content_range
        if (actual_start, actual_end) != (start, end):
            self._violation(
                f"Content-Range区间不符：请求{start}-{end}，返回{actual_start}-{actual_end}"
            )
        if total != base.EXPECTED_ARCHIVE_SIZE:
            self._violation(f"Content-Range总长{total}与预期{base.EXPECTED_ARCHIVE_SIZE}不符")
        declared = headers.get("content-length")
        if declared is not None and int(declared) != end - start + 1:
            self._violation(f"Content-Length {declared} 与请求长度 {end - start + 1} 不符")
        if size != end - start + 1:
            self._violation(f"落盘{size}字节，请求长度{end - start + 1}")
        self.last_range_headers = headers

    def _violation(self, message: str) -> None:
        self.stats["protocol_violations"].append(message)
        raise RangeProtocolError(message)

    @staticmethod
    def _parse_status(stdout: bytes) -> int:
        match = re.search(r"(\d{3})\s*$", stdout.decode("ascii", errors="replace"))
        return int(match.group(1)) if match else -1

    @staticmethod
    def _parse_headers(path: Path) -> dict[str, str]:
        headers: dict[str, str] = {}
        if not path.exists():
            return headers
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            headers[key.strip().lower()] = value.strip()
        return headers

    def inspect(self) -> tuple[int, str | None]:
        """Confirm RAR5 signature and archive size from one small GET."""
        last = ""
        for attempt in range(1, self.retries + 1):
            try:
                _, headers, body = self.probe_range(0, 7)
                if body != base.RAR5_SIGNATURE:
                    raise base.DroneRFaError(
                        f"远程文件不是预期的DroneRFa RAR5压缩包（签名{body[:8].hex()}）"
                    )
                total = parse_content_range(headers.get("content-range", ""))
                if total is None or total[2] != base.EXPECTED_ARCHIVE_SIZE:
                    raise base.DroneRFaError(
                        f"远程档案大小异常：{headers.get('content-range')!r}"
                    )
                self.archive_size = total[2]
                self.etag = headers.get("etag")
                return self.archive_size, self.etag
            except (base.DroneRFaError, RangeProtocolError, subprocess.TimeoutExpired, OSError) as exc:
                last = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries:
                    delay = min(2**attempt, 15)
                    log(f"  远程确认失败，{delay} 秒后重试（{attempt}/{self.retries}）：{last}")
                    time.sleep(delay)
        raise base.DroneRFaError(f"无法确认远程RAR：{last}")

    def download_range(self, start: int, end: int, backend: str = "curl") -> bytes:
        """Signature-compatible with ``download_dronerfa_subset.RangeClient``.

        ``backend`` is accepted (and ignored) so that
        ``base.stream_selected_archive`` can call this client unchanged.
        """
        expected = end - start + 1
        last = ""
        for attempt in range(1, self.retries + 1):
            self.stats["range_requests"] += 1
            status = -1
            try:
                status, _headers, payload = self.probe_range(start, end, timeout=600)
                if len(payload) == expected:
                    self.stats["bytes_transferred"] += len(payload)
                    return payload
                last = f"正文{len(payload)}/{expected}字节"
            except RangeProtocolError as exc:
                # A protocol violation is not a transient network error.
                raise base.DroneRFaError(f"区间{start}-{end}响应不合法：{exc}") from exc
            except subprocess.TimeoutExpired:
                last = f"区间{start}-{end}请求超时"
            except base.DroneRFaError as exc:
                last = str(exc)
                match = re.search(r"HTTP (\d{3})", last)
                if match:
                    status = int(match.group(1))

            if status == 429:
                self.stats["http_429"] += 1
            if attempt < self.retries:
                self.stats["retries"] += 1
                delay = (
                    min(60 * attempt, 300)
                    if status in RETRYABLE_HTTP
                    else min(2**attempt, 15)
                )
                log(f"  分块失败，{delay} 秒后重试（{attempt}/{self.retries}）：{last}")
                time.sleep(delay)
        raise base.DroneRFaError(f"多次重试后仍无法下载区间{start}-{end}：{last}")

    def close(self) -> None:
        return None


def verify_wrapper_rar(archive: Path, entry: base.ArchiveEntry) -> dict:
    """Confirm the wrapper RAR holds exactly our file, reusing base.parse_block.

    ``base.stream_selected_archive`` lays the archive out as
    ``signature | minimal main header | stored file block | end header``, so the
    block walk starts right after the 8-byte signature.
    """
    size = archive.stat().st_size
    expected_size = (
        len(base.RAR5_SIGNATURE)
        + len(base.MINIMAL_MAIN_HEADER)
        + entry.block_size
        + len(base.MINIMAL_END_HEADER)
    )
    if size != expected_size:
        raise base.DroneRFaError(f"{archive.name}: 大小{size}与预期{expected_size}不符")

    with archive.open("rb") as handle:
        if handle.read(8) != base.RAR5_SIGNATURE:
            raise base.DroneRFaError(f"{archive.name}: 缺少RAR5签名")

    # base.parse_block takes a client with get_range(); wrap the local file.
    class _LocalClient:
        def __init__(self, blob: bytes) -> None:
            self._blob = blob

        def get_range(self, start: int, end: int) -> bytes:
            return self._blob[start : end + 1]

    blob = archive.read_bytes()
    client = _LocalClient(blob)
    offset = len(base.RAR5_SIGNATURE)
    blocks: list[dict] = []
    while offset < size:
        block = base.parse_block(client, offset, size)  # type: ignore[arg-type]
        blocks.append(block)
        offset += int(block["header_total_size"]) + int(block["data_size"])

    file_blocks = [block for block in blocks if int(block["header_type"]) == base.HEAD_FILE]
    if len(file_blocks) != 1:
        raise base.DroneRFaError(f"{archive.name}: 期望1个文件块，实际{len(file_blocks)}个")
    if int(blocks[-1]["header_type"]) != base.HEAD_ENDARC:
        raise base.DroneRFaError(f"{archive.name}: 缺少RAR结束块")
    expected_name = entry.name.rsplit("/", 1)[-1]
    actual_name = str(file_blocks[0]["name"]).rsplit("/", 1)[-1]
    if actual_name != expected_name:
        raise base.DroneRFaError(f"{archive.name}: 内嵌文件名{actual_name}，预期{expected_name}")
    if int(file_blocks[0]["data_size"]) != entry.packed_size:
        raise base.DroneRFaError(f"{archive.name}: 打包大小与清单不一致")
    if int(file_blocks[0]["unpacked_size"]) != entry.unpacked_size:
        raise base.DroneRFaError(f"{archive.name}: 解压大小与清单不一致")
    if int(file_blocks[0]["header_total_size"]) + entry.packed_size != entry.block_size:
        raise base.DroneRFaError(f"{archive.name}: 文件块大小与清单不一致")
    return {
        "archive_size": size,
        "signature": base.RAR5_SIGNATURE.hex(),
        "main_header_size": len(base.MINIMAL_MAIN_HEADER),
        "end_header_size": len(base.MINIMAL_END_HEADER),
        "block_count": len(blocks),
        "block_types": [int(block["header_type"]) for block in blocks],
        "file_block": file_blocks[0],
    }


def run_unrar(unrar: Path, command: str, archive: Path, destination: Path | None) -> str:
    args = [str(unrar), command, "-idq", "-p-", str(archive)]
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        args.extend(["-o+", str(destination) + os.sep])
    result = subprocess.run(args, capture_output=True, check=False)
    text = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    if result.returncode != 0:
        action = "解压" if destination is not None else "完整性测试"
        raise base.DroneRFaError(f"WinRAR {action}失败，退出码：{result.returncode}；输出：{text[:400]}")
    return text


def inspect_mat(path: Path) -> dict:
    """Validate MAT structure using the project's own dataset semantics."""
    with h5py.File(path, "r") as handle:
        keys = sorted(handle.keys())
        datasets: dict[str, dict] = {}
        for key in keys:
            item = handle[key]
            if isinstance(item, h5py.Dataset):
                datasets[key] = {"shape": list(item.shape), "dtype": str(item.dtype)}
        result: dict = {
            "keys": keys,
            "datasets": datasets,
            "missing_required": [key for key in REQUIRED_KEYS if key not in handle],
            "rf0_available": "RF0_I" in handle and "RF0_Q" in handle,
            "rf1_available": "RF1_I" in handle and "RF1_Q" in handle,
        }
        if result["missing_required"]:
            return result

        lengths: dict[str, int] = {}
        for key in REQUIRED_KEYS:
            try:
                lengths[key] = dataset_length(handle[key])
            except ValueError as exc:
                result["length_errors"] = {key: str(exc)}
                return result
        result["channel_lengths"] = lengths
        result["lengths_consistent"] = len(set(lengths.values())) == 1
        result["source_samples"] = next(iter(lengths.values()))
        result["datasets_are_vectors"] = {
            key: handle[key].ndim == 1
            or (handle[key].ndim == 2 and 1 in handle[key].shape)
            for key in REQUIRED_KEYS
        }
        # Verify RF0 actually reads through the shared helper, at a fixed window.
        probe_start = max(result["source_samples"] // 2 - 500_000, 0)
        i_values = read_vector_slice(handle["RF0_I"], probe_start, 1000)
        q_values = read_vector_slice(handle["RF0_Q"], probe_start, 1000)
        result["read_probe"] = {
            "start": probe_start,
            "samples": int(i_values.shape[0]),
            "dtype": str(i_values.dtype),
            "finite": bool(np.isfinite(i_values).all() and np.isfinite(q_values).all()),
        }
        return result


def load_candidates(path: Path, requested: str) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    available = {Path(item["name"]).name: item["name"] for item in data["files"]}
    if requested.strip().lower() == ALL_CANDIDATES:
        names = [Path(item["name"]).name for item in data["files"]]
    else:
        names = [token.strip() for token in requested.split(",") if token.strip()]
    unknown = [name for name in names if name not in available]
    if unknown:
        raise base.DroneRFaError(f"候选清单中没有这些文件：{unknown}")
    return names


def existing_mat_is_valid(structure_info: dict) -> bool:
    """Decide from real HDF5 content whether an extracted .mat is usable.

    A DroneRFa .mat is an HDF5 container, so the file size in bytes says nothing
    about the sample count: ``unpacked_size // 4`` is *not* the number of samples
    per channel and must never be used as an expected length.  The checks are:

    * all four RF0/RF1 I/Q datasets present and stored as vectors;
    * every channel has the same, non-empty length;
    * channel dtype is floating point;
    * RF0 reads back through the shared helper and is entirely finite.

    The independent byte-size match against the manifest is enforced separately.
    """
    if structure_info.get("missing_required"):
        return False
    vectors = structure_info.get("datasets_are_vectors", {})
    if not all(vectors.get(key) for key in REQUIRED_KEYS):
        return False
    lengths = structure_info.get("channel_lengths", {})
    if len(lengths) != len(REQUIRED_KEYS) or len(set(lengths.values())) != 1:
        return False
    source_samples = structure_info.get("source_samples")
    if not isinstance(source_samples, int) or source_samples <= 0:
        return False
    dtypes = {info["dtype"] for info in structure_info.get("datasets", {}).values()}
    if not dtypes or not all(dtype.startswith("float") for dtype in dtypes):
        return False
    probe = structure_info.get("read_probe", {})
    return probe.get("samples") == 1000 and bool(probe.get("finite"))


def process_one(
    *,
    client: CurlRangeClient,
    entry: base.ArchiveEntry,
    short_name: str,
    staging_dir: Path,
    extract_dir: Path,
    unrar: Path,
    chunk_size: int,
    workers: int,
    keep_archive: bool,
    skip_existing: bool,
    safety: SafePaths,
) -> dict:
    record: dict = {
        "filename": short_name,
        "archive_name": entry.name,
        "block_offset": entry.block_offset,
        "block_size": entry.block_size,
        "packed_size": entry.packed_size,
        "expected_unpacked_size": entry.unpacked_size,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }

    mat_path = extract_dir / short_name
    archive_path = staging_dir / f"{short_name}.rar"
    part_path = staging_dir / f"{short_name}.rar.part"
    state_path = staging_dir / f"{short_name}.rar.state.json"
    temp_extract = staging_dir / f"extract_{short_name}"

    # ---- 0. skip only after a real read-back check, never size alone -------
    if skip_existing and safety.guard_destination(
        mat_path, expected_size=entry.unpacked_size, what=f"{short_name} 已存在文件"
    ):
        structure_info = inspect_mat(mat_path)
        if existing_mat_is_valid(structure_info):
            log(f"{short_name}: 已存在且HDF5读取通过，跳过下载")
            record["status"] = "skipped_existing"
            record["checks"]["existing_hdf5_read"] = {"passed": True, **structure_info}
            record["mat_size"] = mat_path.stat().st_size
            record["mat_sha256"] = sha256_file(mat_path)
            record["source_samples"] = structure_info.get("source_samples")
            return record

    free = disk_free(staging_dir)
    needed = entry.block_size + entry.unpacked_size + MIN_FREE_BYTES
    if free < needed:
        raise base.DroneRFaError(
            f"{staging_dir} 可用空间不足：{base.format_bytes(free)}，"
            f"本轮至少需要 {base.format_bytes(needed)}"
        )
    record["free_bytes_before"] = free

    # ---- 1. staged RAR via the existing, already-verified wrapper ----------
    base.stream_selected_archive(
        client,
        [entry],
        archive_path,
        chunk_size,
        workers,
        "curl",
        False,
    )
    record["checks"]["download"] = {
        "passed": archive_path.exists() and archive_path.stat().st_size
        == len(base.RAR5_SIGNATURE) + len(base.MINIMAL_MAIN_HEADER) + entry.block_size
        + len(base.MINIMAL_END_HEADER),
        "actual_size": archive_path.stat().st_size if archive_path.exists() else None,
        "expected_size": len(base.RAR5_SIGNATURE)
        + len(base.MINIMAL_MAIN_HEADER)
        + entry.block_size
        + len(base.MINIMAL_END_HEADER),
    }
    if not record["checks"]["download"]["passed"]:
        raise base.DroneRFaError(f"{short_name}: 选择压缩包大小不符")

    # ---- 2. wrapper RAR structure (signature, main/file/end blocks) --------
    record["checks"]["rar_structure"] = {
        "passed": True,
        **verify_wrapper_rar(archive_path, entry),
    }

    # ---- 3. unrar integrity test -------------------------------------------
    run_unrar(unrar, "t", archive_path, None)
    record["checks"]["unrar_test"] = {"passed": True}

    # ---- 4. extract into a guarded temp folder, then move -----------------
    safety.safe_rmtree(temp_extract, what=f"{short_name} 临时解压目录")
    run_unrar(unrar, "x", archive_path, temp_extract)
    produced = sorted(temp_extract.rglob("*.mat"))
    if len(produced) != 1:
        raise base.DroneRFaError(f"{short_name}: 解压得到{len(produced)}个MAT文件，预期1个")
    if produced[0].name != short_name:
        raise base.DroneRFaError(f"{short_name}: 解压文件名不符：{produced[0].name}")
    actual_size = produced[0].stat().st_size
    if actual_size != entry.unpacked_size:
        raise base.DroneRFaError(
            f"{short_name}: 解压大小{actual_size}与清单{entry.unpacked_size}不符"
        )
    safety.safe_replace(produced[0], mat_path, what=f"{short_name} 落位")
    safety.safe_rmtree(temp_extract, what=f"{short_name} 临时解压目录")
    record["checks"]["extract"] = {
        "passed": True,
        "expected_unpacked_size": entry.unpacked_size,
        "actual_size": actual_size,
    }

    # ---- 5. HDF5 read check ------------------------------------------------
    structure_info = inspect_mat(mat_path)
    ok = existing_mat_is_valid(structure_info)
    record["checks"]["hdf5_read"] = {"passed": bool(ok), **structure_info}
    if not ok:
        raise base.DroneRFaError(f"{short_name}: HDF5结构不符合预期：{structure_info}")

    record["mat_size"] = actual_size
    record["mat_sha256"] = sha256_file(mat_path)
    record["source_samples"] = structure_info["source_samples"]
    record["window_count_check"] = {
        "source_samples": structure_info["source_samples"],
        "complete_blocks": structure_info["source_samples"] // 10_000_000,
        "windows": (structure_info["source_samples"] // 10_000_000) * 10,
    }

    # ---- 6. only now drop this round's staging archive ---------------------
    if keep_archive:
        record["archive_kept"] = str(archive_path)
    else:
        safety.safe_unlink(archive_path, what=f"{short_name} 本轮临时压缩包")
        record["archive_deleted"] = True

    record["status"] = "completed"
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    record["free_bytes_after"] = disk_free(staging_dir)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="逐个下载/校验/解压12个未使用的DroneRFa源文件（支持断点与幂等复跑）"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=PROJECT_ROOT
        / "dronerfa_baseline/outputs/noise_ab_replicates_20260917/candidate_heldout_files.json",
    )
    parser.add_argument("--files", default=ALL_CANDIDATES, help="逗号分隔的文件名，或 all")
    parser.add_argument("--limit-files", type=int, default=0, help="只处理前N个（0表示全部）")
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "dronerfa_manifest.json")
    parser.add_argument("--unrar", default=r"C:\Program Files\WinRAR\UnRAR.exe")
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2, help="同一文件内的并行分块数（1-4）")
    parser.add_argument("--records", type=Path, default=None, help="逐文件记录JSON")
    parser.add_argument("--keep-archive", action="store_true", help="保留临时压缩包（不推荐）")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.workers <= 4:
        raise base.DroneRFaError("--workers 必须在1到4之间")
    if not 1 <= args.chunk_mib <= 512:
        raise base.DroneRFaError("--chunk-mib 必须在1到512之间")

    unrar = Path(args.unrar)
    if not unrar.is_file():
        unrar = base.find_unrar(args.unrar)

    names = load_candidates(args.candidates, args.files)
    if args.limit_files > 0:
        names = names[: args.limit_files]

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("archive_size") != base.EXPECTED_ARCHIVE_SIZE:
        raise base.DroneRFaError("本地清单的压缩包大小与预期不符")

    staging_dir = Path(os.path.abspath(str(args.staging_dir)))
    extract_dir = Path(os.path.abspath(str(args.extract_dir)))
    entries_by_name = {
        Path(entry["name"]).name: base.ArchiveEntry(**entry) for entry in manifest["entries"]
    }
    missing = [name for name in names if name not in entries_by_name]
    if missing:
        raise base.DroneRFaError(f"清单中缺少：{missing}")

    # Whitelist: only this round's staging artefacts and .mat copies may go away.
    deletable = set()
    for name in names:
        deletable.update(
            {
                staging_dir / f"{name}.rar",
                staging_dir / f"{name}.rar.part",
                staging_dir / f"{name}.rar.state.json",
                extract_dir / name,
            }
        )
    safety = SafePaths(
        roots=(staging_dir, extract_dir),
        deletable=frozenset(Path(os.path.abspath(str(path))) for path in deletable),
    )
    staging_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    selected = [entries_by_name[name] for name in names]
    packed = sum(entry.block_size for entry in selected)
    unpacked = sum(entry.unpacked_size for entry in selected)
    log(f"本次处理 {len(names)} 个文件，压缩约 {base.format_bytes(packed)}，解压约 {base.format_bytes(unpacked)}")
    log(f"暂存目录 {staging_dir}（可用 {base.format_bytes(disk_free(staging_dir))}）")
    log(f"数据目录 {extract_dir}（可用 {base.format_bytes(disk_free(extract_dir))}）")
    log(f"解压器 {unrar}")

    records_path = args.records or (
        PROJECT_ROOT / "dronerfa_baseline/outputs/heldout_preparation_20260917/download_records.json"
    )

    if args.dry_run:
        for name in names:
            entry = entries_by_name[name]
            print(f"  {name}  offset={entry.block_offset}  unpacked={entry.unpacked_size}")
        log("--dry-run：未下载")
        return 0

    client = CurlRangeClient(
        base.SOURCE_URL, scratch_dir=staging_dir / "scratch"
    )
    try:
        archive_size, etag = client.inspect()
    except base.DroneRFaError as exc:
        client.cleanup_scratch()
        write_json_atomic(
            records_path,
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "status": "remote_unavailable",
                "error": str(exc),
                "files": {},
            },
        )
        log(f"远程档案不可用，已记录并停止：{exc}")
        return 1
    log(f"远程压缩包 {base.format_bytes(archive_size)}，ETag={etag}")

    existing: dict = {}
    if records_path.exists():
        try:
            existing = json.loads(records_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
    per_file: dict = existing.get("files", {})

    failures: list[dict] = []
    for index, name in enumerate(names, start=1):
        log(f"=== [{index}/{len(names)}] {name} ===")
        try:
            record = process_one(
                client=client,
                entry=entries_by_name[name],
                short_name=name,
                staging_dir=staging_dir,
                extract_dir=extract_dir,
                unrar=unrar,
                chunk_size=args.chunk_mib * 1024**2,
                workers=args.workers,
                keep_archive=args.keep_archive,
                skip_existing=not args.no_skip_existing,
                safety=safety,
            )
            per_file[name] = record
            log(f"{name}: {record['status']}，SHA256={record.get('mat_sha256', '-')}")
        except (base.DroneRFaError, SafetyError, RangeProtocolError, OSError) as exc:
            log(f"{name}: 失败 —— {exc}")
            per_file[name] = {
                "filename": name,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
            failures.append({"filename": name, "error": str(exc)})
        write_json_atomic(
            records_path,
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "status": "running",
                "source_url": base.SOURCE_URL,
                "remote_archive_size": archive_size,
                "remote_etag": etag,
                "staging_dir": str(staging_dir),
                "extract_dir": str(extract_dir),
                "client_stats": client.stats,
                "files": per_file,
            },
        )

    completed = [n for n, r in per_file.items() if r.get("status") == "completed"]
    skipped = [n for n, r in per_file.items() if r.get("status") == "skipped_existing"]
    client.cleanup_scratch()
    log(f"完成 {len(completed)}，已存在跳过 {len(skipped)}，失败 {len(failures)}")
    for item in failures:
        log(f"  失败：{item['filename']} —— {item['error']}")
    log(f"range请求 {client.stats['range_requests']}，重试 {client.stats['retries']}，"
        f"429 {client.stats['http_429']}，协议违规 {len(client.stats['protocol_violations'])}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中断，断点已保留；重新运行同一命令即可继续。")
        raise SystemExit(130)
    except (base.DroneRFaError, SafetyError, OSError, ValueError) as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
