#!/usr/bin/env python3
"""Selectively download files from the remote non-solid DroneRFa RAR archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import zlib
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterable

import requests


SOURCE_URL = (
    "https://china.scidb.cn/download?"
    "fileId=c403fc76444e4b9989e4f3ff570f3b3d"
)
RAR5_SIGNATURE = bytes.fromhex("526172211a070100")
EXPECTED_ARCHIVE_SIZE = 615_955_805_466
MANIFEST_SCHEMA = 1

DEFAULT_TARGETS = ("T0000", "T0010", "T0101")
DEFAULT_DISTANCES = ("D00",)
DEFAULT_SEGMENTS = ("S0000", "S0001", "S0010", "S0011")

TARGET_LABELS = {
    "T0000": "背景（无无人机）",
    "T0010": "DJI Phantom 4 Pro",
    "T0101": "DJI Air 2S",
}

HFL_EXTRA = 0x0001
HFL_DATA = 0x0002
HEAD_FILE = 2
HEAD_ENDARC = 5
FHFL_UTIME = 0x0002
FHFL_CRC32 = 0x0004


class DroneRFaError(RuntimeError):
    """Expected error with a user-facing message."""


@dataclass(frozen=True)
class ArchiveEntry:
    name: str
    block_offset: int
    block_size: int
    packed_size: int
    unpacked_size: int


def format_bytes(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def parse_csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("列表不能为空")
    return items


def read_vint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(10):
        if offset >= len(data):
            raise DroneRFaError("RAR头部不完整，无法读取变长整数")
        current = data[offset]
        offset += 1
        value |= (current & 0x7F) << shift
        if current & 0x80 == 0:
            return value, offset
        shift += 7
    raise DroneRFaError("RAR头部包含无效的变长整数")


def encode_vint(value: int) -> bytes:
    if value < 0:
        raise ValueError("vint cannot be negative")
    result = bytearray()
    while True:
        current = value & 0x7F
        value >>= 7
        if value:
            result.append(current | 0x80)
        else:
            result.append(current)
            return bytes(result)


def make_rar5_block(header_type: int, specific_data: bytes = b"\x00") -> bytes:
    header_data = encode_vint(header_type) + encode_vint(0) + specific_data
    crc_payload = encode_vint(len(header_data)) + header_data
    crc = zlib.crc32(crc_payload) & 0xFFFFFFFF
    return struct.pack("<I", crc) + crc_payload


MINIMAL_MAIN_HEADER = make_rar5_block(1)
MINIMAL_END_HEADER = make_rar5_block(HEAD_ENDARC)


class RangeClient:
    def __init__(
        self,
        url: str,
        *,
        timeout: tuple[int, int] = (20, 180),
        retries: int = 5,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self._local = threading.local()
        self._sessions: list[requests.Session] = []
        self._sessions_lock = threading.Lock()
        self.archive_size: int | None = None
        self.etag: str | None = None

    def _get_session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "Accept-Encoding": "identity",
                    "User-Agent": "DroneRFa-Selective-Downloader/1.0",
                }
            )
            self._local.session = session
            with self._sessions_lock:
                self._sessions.append(session)
        return session

    def close(self) -> None:
        with self._sessions_lock:
            sessions = list(self._sessions)
            self._sessions.clear()
        for session in sessions:
            session.close()

    def __enter__(self) -> "RangeClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def inspect(self) -> tuple[int, str | None]:
        response = None
        try:
            response = self._get_session().get(
                self.url,
                headers={"Range": "bytes=0-7"},
                stream=True,
                timeout=self.timeout,
                allow_redirects=True,
            )
            if response.status_code != 206:
                raise DroneRFaError(
                    f"服务器未响应分段下载请求（HTTP {response.status_code}）"
                )
            start, end, total = self._parse_content_range(response)
            signature = response.raw.read(8)
            if (start, end) != (0, 7) or signature != RAR5_SIGNATURE:
                raise DroneRFaError("远程文件不是预期的DroneRFa RAR5压缩包")
            self.archive_size = total
            self.etag = response.headers.get("ETag")
            return total, self.etag
        except requests.RequestException as exc:
            raise DroneRFaError(f"无法连接ScienceDB：{exc}") from exc
        finally:
            if response is not None:
                response.close()

    @staticmethod
    def _parse_content_range(response: requests.Response) -> tuple[int, int, int]:
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
        if not match:
            raise DroneRFaError(
                f"服务器返回了无效的Content-Range：{content_range!r}"
            )
        return tuple(int(part) for part in match.groups())  # type: ignore[return-value]

    def get_range(self, start: int, end: int) -> bytes:
        if start < 0 or end < start:
            raise ValueError(f"invalid byte range: {start}-{end}")
        expected = end - start + 1
        last_error: Exception | None = None

        for attempt in range(1, self.retries + 1):
            response = None
            try:
                response = self._get_session().get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=self.timeout,
                    allow_redirects=True,
                )
                if response.status_code != 206:
                    raise DroneRFaError(
                        f"Range {start}-{end} 返回HTTP {response.status_code}"
                    )
                actual_start, actual_end, total = self._parse_content_range(response)
                if (actual_start, actual_end) != (start, end):
                    raise DroneRFaError(
                        f"服务器返回了错误区间 {actual_start}-{actual_end}"
                    )
                payload = response.content
                if len(payload) != expected:
                    raise DroneRFaError(
                        f"区间应为{expected}字节，实际收到{len(payload)}字节"
                    )
                if self.archive_size is not None and total != self.archive_size:
                    raise DroneRFaError("下载过程中远程压缩包大小发生变化")
                return payload
            except (requests.RequestException, DroneRFaError) as exc:
                last_error = exc
                if attempt < self.retries:
                    delay = min(2**attempt, 15)
                    print(
                        f"\n网络请求失败，{delay}秒后重试"
                        f"（{attempt}/{self.retries}）：{exc}",
                        flush=True,
                    )
                    time.sleep(delay)
            finally:
                if response is not None:
                    response.close()

        raise DroneRFaError(f"多次重试后仍无法下载区间{start}-{end}：{last_error}")

    def download_range(
        self,
        start: int,
        end: int,
        backend: str,
    ) -> bytes:
        curl_path = shutil.which("curl.exe") or shutil.which("curl")
        if backend == "requests" or (backend == "auto" and curl_path is None):
            return self.get_range(start, end)
        if curl_path is None:
            raise DroneRFaError("未找到curl，无法使用curl下载后端")

        expected = end - start + 1
        last_error = ""
        for attempt in range(1, self.retries + 1):
            try:
                result = subprocess.run(
                    [
                        curl_path,
                        "--location",
                        "--fail",
                        "--silent",
                        "--show-error",
                        "--connect-timeout",
                        str(self.timeout[0]),
                        "--max-time",
                        "300",
                        "--range",
                        f"{start}-{end}",
                        "--output",
                        "-",
                        self.url,
                    ],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=330,
                )
                if result.returncode == 0 and len(result.stdout) == expected:
                    return result.stdout
                error_text = result.stderr.decode(
                    encoding="utf-8",
                    errors="replace",
                ).strip()
                last_error = (
                    f"curl退出码{result.returncode}，收到{len(result.stdout)}/"
                    f"{expected}字节：{error_text}"
                )
            except subprocess.TimeoutExpired:
                last_error = "curl请求超过330秒"

            if attempt < self.retries:
                if "returned error: 429" in last_error:
                    delay = min(60 * attempt, 300)
                else:
                    delay = min(2**attempt, 15)
                print(
                    f"\ncurl分块失败，{delay}秒后重试"
                    f"（{attempt}/{self.retries}）：{last_error}",
                    flush=True,
                )
                time.sleep(delay)

        raise DroneRFaError(
            f"多次重试后仍无法用curl下载区间{start}-{end}：{last_error}"
        )


def parse_block(client: RangeClient, position: int, archive_size: int) -> dict[str, int | str]:
    probe_end = min(position + 4095, archive_size - 1)
    data = client.get_range(position, probe_end)
    if len(data) < 8:
        raise DroneRFaError(f"RAR块头部过短，位置：{position}")

    stored_crc = struct.unpack_from("<I", data, 0)[0]
    header_size, cursor = read_vint(data, 4)
    header_end = cursor + header_size
    if header_end > len(data):
        data = client.get_range(position, position + header_end - 1)

    calculated_crc = zlib.crc32(data[4:header_end]) & 0xFFFFFFFF
    if stored_crc != calculated_crc:
        raise DroneRFaError(f"RAR头部CRC错误，位置：{position}")

    header_type, cursor = read_vint(data, cursor)
    header_flags, cursor = read_vint(data, cursor)
    extra_size = 0
    data_size = 0
    if header_flags & HFL_EXTRA:
        extra_size, cursor = read_vint(data, cursor)
    if header_flags & HFL_DATA:
        data_size, cursor = read_vint(data, cursor)

    result: dict[str, int | str] = {
        "header_type": header_type,
        "header_flags": header_flags,
        "header_total_size": header_end,
        "data_size": data_size,
    }

    if header_type == HEAD_FILE:
        file_flags, cursor = read_vint(data, cursor)
        unpacked_size, cursor = read_vint(data, cursor)
        _, cursor = read_vint(data, cursor)  # file attributes
        if file_flags & FHFL_UTIME:
            cursor += 4
        if file_flags & FHFL_CRC32:
            cursor += 4
        _, cursor = read_vint(data, cursor)  # compression information
        _, cursor = read_vint(data, cursor)  # host OS
        name_size, cursor = read_vint(data, cursor)

        basic_header_end = header_end - extra_size
        if cursor + name_size > basic_header_end:
            raise DroneRFaError(f"文件名超出RAR头部边界，位置：{position}")
        try:
            name = data[cursor : cursor + name_size].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DroneRFaError(f"RAR文件名不是有效UTF-8，位置：{position}") from exc
        result["name"] = name
        result["unpacked_size"] = unpacked_size

    return result


def build_manifest(
    client: RangeClient,
    manifest_path: Path,
    archive_size: int,
    etag: str | None,
) -> list[ArchiveEntry]:
    print(
        "首次运行需要读取远程RAR目录，约需3-6分钟；"
        "只读取约40 MiB头部数据，不会下载整包。",
        flush=True,
    )
    entries: list[ArchiveEntry] = []
    position = len(RAR5_SIGNATURE)
    block_count = 0
    started = time.monotonic()

    while position < archive_size:
        block = parse_block(client, position, archive_size)
        block_count += 1
        header_type = int(block["header_type"])
        header_total_size = int(block["header_total_size"])
        data_size = int(block["data_size"])
        total_block_size = header_total_size + data_size

        if total_block_size <= 0 or position + total_block_size > archive_size:
            raise DroneRFaError(f"RAR块大小无效，位置：{position}")

        if header_type == HEAD_FILE:
            entries.append(
                ArchiveEntry(
                    name=str(block["name"]),
                    block_offset=position,
                    block_size=total_block_size,
                    packed_size=data_size,
                    unpacked_size=int(block["unpacked_size"]),
                )
            )
            if len(entries) % 20 == 0:
                elapsed = time.monotonic() - started
                print(
                    f"\r已索引 {len(entries)} 个文件，耗时 {elapsed:.0f} 秒",
                    end="",
                    flush=True,
                )

        position += total_block_size
        if header_type == HEAD_ENDARC:
            break

    print()
    if not entries:
        raise DroneRFaError("未在远程RAR中找到任何文件")
    if header_type != HEAD_ENDARC:
        raise DroneRFaError("未找到RAR结束块，远程目录可能不完整")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source_url": client.url,
        "archive_size": archive_size,
        "etag": etag,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "entries": [asdict(entry) for entry in entries],
    }
    write_json_atomic(manifest_path, manifest)
    print(f"目录索引已保存：{manifest_path}（{len(entries)}个文件）")
    return entries


def load_manifest(
    manifest_path: Path,
    source_url: str,
    archive_size: int,
) -> list[ArchiveEntry] | None:
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            data.get("schema") != MANIFEST_SCHEMA
            or data.get("source_url") != source_url
            or data.get("archive_size") != archive_size
        ):
            print("本地目录索引与远程文件不匹配，将重新索引。")
            return None
        return [ArchiveEntry(**entry) for entry in data["entries"]]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"本地目录索引无法读取，将重新索引：{exc}")
        return None


def write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def has_token(name: str, token: str) -> bool:
    return re.search(
        rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])",
        name,
        flags=re.IGNORECASE,
    ) is not None


def select_entries(
    entries: Iterable[ArchiveEntry],
    targets: tuple[str, ...],
    distances: tuple[str, ...],
    segments: tuple[str, ...],
) -> list[ArchiveEntry]:
    mat_entries = [entry for entry in entries if entry.name.lower().endswith(".mat")]
    selected: list[ArchiveEntry] = []

    for target in targets:
        for distance in distances:
            for segment in segments:
                matches = [
                    entry
                    for entry in mat_entries
                    if has_token(entry.name, target)
                    and has_token(entry.name, distance)
                    and has_token(entry.name, segment)
                ]
                if len(matches) != 1:
                    raise DroneRFaError(
                        f"{target}/{distance}/{segment} 应匹配1个文件，"
                        f"实际匹配{len(matches)}个"
                    )
                selected.append(matches[0])
    return selected


def print_selection(entries: list[ArchiveEntry]) -> None:
    print("\n选择结果：")
    print("-" * 78)
    for index, entry in enumerate(entries, start=1):
        print(
            f"{index:>2}. {entry.name}\n"
            f"    压缩 {format_bytes(entry.packed_size)}，"
            f"解压 {format_bytes(entry.unpacked_size)}"
        )
    packed = sum(entry.block_size for entry in entries)
    unpacked = sum(entry.unpacked_size for entry in entries)
    print("-" * 78)
    print(
        f"共 {len(entries)} 个文件；下载约 {format_bytes(packed)}；"
        f"解压约 {format_bytes(unpacked)}"
    )


def print_download_status(
    manifest_path: Path,
    output_archive: Path,
    targets: tuple[str, ...],
    distances: tuple[str, ...],
    segments: tuple[str, ...],
) -> int:
    if not manifest_path.exists():
        raise DroneRFaError("尚未建立dronerfa_manifest.json，无法计算进度")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = [ArchiveEntry(**entry) for entry in manifest["entries"]]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise DroneRFaError(f"目录索引无法读取：{exc}") from exc

    selected = select_entries(entries, targets, distances, segments)
    total = sum(entry.block_size for entry in selected)
    selection_path = output_archive.with_name(
        output_archive.name + ".selection.json"
    )
    part_path = output_archive.with_name(output_archive.name + ".part")
    state_path = output_archive.with_name(output_archive.name + ".state.json")

    if output_archive.exists() and selection_path.exists():
        print("状态：下载完成")
        print(f"选择压缩包：{output_archive}")
        print(f"压缩包大小：{format_bytes(output_archive.stat().st_size)}")
        return 0

    if part_path.exists() and state_path.exists():
        downloaded = max(
            part_path.stat().st_size
            - len(RAR5_SIGNATURE)
            - len(MINIMAL_MAIN_HEADER),
            0,
        )
        percent = min(downloaded / total * 100, 100)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        current_file = min(int(state.get("next_entry", 0)) + 1, len(selected))
        print("状态：下载中或已暂停")
        print(
            f"进度：{percent:.2f}%  "
            f"{format_bytes(downloaded)}/{format_bytes(total)}"
        )
        print(f"当前文件：{current_file}/{len(selected)}")
        print(f"断点文件：{part_path}")
        return 0

    print("状态：尚未开始下载")
    return 0


def selection_fingerprint(entries: list[ArchiveEntry], source_url: str) -> str:
    payload = {
        "source_url": source_url,
        "entries": [asdict(entry) for entry in entries],
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def check_disk_space(
    output_archive: Path,
    extract_dir: Path | None,
    download_size: int,
    unpacked_size: int,
    partial_size: int,
) -> None:
    reserve = 2 * 1024**3
    output_free = shutil.disk_usage(output_archive.parent).free
    remaining_download = max(download_size - partial_size, 0)

    if extract_dir is None:
        required = remaining_download + reserve
        if output_free < required:
            raise DroneRFaError(
                f"磁盘空间不足：当前可用{format_bytes(output_free)}，"
                f"至少需要{format_bytes(required)}"
            )
        return

    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    same_drive = (
        os.path.splitdrive(output_archive.resolve())[0].lower()
        == os.path.splitdrive(extract_dir.resolve())[0].lower()
    )
    if same_drive:
        required = remaining_download + unpacked_size + reserve
        if output_free < required:
            raise DroneRFaError(
                f"下载和解压位于同一磁盘，当前可用{format_bytes(output_free)}，"
                f"至少需要{format_bytes(required)}"
            )
    else:
        if output_free < remaining_download + reserve:
            raise DroneRFaError("压缩包所在磁盘空间不足")
        extract_free = shutil.disk_usage(extract_dir.parent).free
        if extract_free < unpacked_size + reserve:
            raise DroneRFaError("解压目录所在磁盘空间不足")


def stream_selected_archive(
    client: RangeClient,
    entries: list[ArchiveEntry],
    output_archive: Path,
    chunk_size: int,
    workers: int,
    backend: str,
    overwrite: bool,
) -> None:
    part_path = output_archive.with_name(output_archive.name + ".part")
    state_path = output_archive.with_name(output_archive.name + ".state.json")
    selection_path = output_archive.with_name(
        output_archive.name + ".selection.json"
    )
    fingerprint = selection_fingerprint(entries, client.url)

    if output_archive.exists():
        if overwrite:
            output_archive.unlink()
            selection_path.unlink(missing_ok=True)
        elif selection_path.exists():
            try:
                completed = json.loads(selection_path.read_text(encoding="utf-8"))
                if completed.get("fingerprint") == fingerprint:
                    print(f"选择压缩包已经完整存在，跳过下载：{output_archive}")
                    return
            except (OSError, ValueError):
                pass
            raise DroneRFaError(
                f"已有压缩包的选择记录不匹配：{output_archive}。"
                "确认需要重下时使用 --overwrite"
            )
        else:
            raise DroneRFaError(
                f"目标文件已存在但缺少选择记录：{output_archive}。"
                "确认需要重下时使用 --overwrite"
            )

    if overwrite:
        part_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        selection_path.unlink(missing_ok=True)

    completed_bytes = 0
    next_entry = 0
    current_block_bytes = 0
    expected_part_size = len(RAR5_SIGNATURE) + len(MINIMAL_MAIN_HEADER)

    if part_path.exists() and state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["fingerprint"] != fingerprint:
                raise DroneRFaError(
                    "断点文件属于另一组数据，请删除.part/.state文件或使用--overwrite"
                )
            next_entry = int(state["next_entry"])
            current_block_bytes = int(state["current_block_bytes"])
            completed_bytes = sum(
                entry.block_size for entry in entries[:next_entry]
            ) + current_block_bytes
            expected_part_size += completed_bytes
            actual_size = part_path.stat().st_size
            if actual_size > expected_part_size:
                with part_path.open("r+b") as output:
                    output.truncate(expected_part_size)
                print(
                    "检测到一个尚未登记的尾部分块，"
                    "已回退到最后一个有效断点。"
                )
            elif actual_size < expected_part_size:
                raise DroneRFaError(
                    f"断点文件大小异常：应为{expected_part_size}，实际{actual_size}"
                )
            print(
                f"发现有效断点，已完成 {format_bytes(completed_bytes)}，继续下载。"
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise DroneRFaError(f"断点状态无法读取：{exc}") from exc
    elif part_path.exists() or state_path.exists():
        raise DroneRFaError(
            "只找到一个断点文件，状态不完整；请使用--overwrite重新开始"
        )
    else:
        output_archive.parent.mkdir(parents=True, exist_ok=True)
        with part_path.open("wb") as output:
            output.write(RAR5_SIGNATURE)
            output.write(MINIMAL_MAIN_HEADER)
        write_json_atomic(
            state_path,
            {
                "fingerprint": fingerprint,
                "next_entry": 0,
                "current_block_bytes": 0,
            },
        )

    total_bytes = sum(entry.block_size for entry in entries)
    started = time.monotonic()
    session_start_bytes = completed_bytes

    with (
        part_path.open("ab") as output,
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        for entry_index in range(next_entry, len(entries)):
            entry = entries[entry_index]
            block_downloaded = current_block_bytes if entry_index == next_entry else 0
            scheduled_bytes = block_downloaded
            pending: deque[tuple[int, Future[bytes]]] = deque()

            def schedule_chunk() -> bool:
                nonlocal scheduled_bytes
                if scheduled_bytes >= entry.block_size:
                    return False
                request_size = min(
                    chunk_size,
                    entry.block_size - scheduled_bytes,
                )
                start = entry.block_offset + scheduled_bytes
                end = start + request_size - 1
                pending.append(
                    (
                        request_size,
                        executor.submit(
                            client.download_range,
                            start,
                            end,
                            backend,
                        ),
                    )
                )
                scheduled_bytes += request_size
                return True

            for _ in range(workers):
                if not schedule_chunk():
                    break

            while pending:
                expected_size, future = pending.popleft()
                payload = future.result()
                if len(payload) != expected_size:
                    raise DroneRFaError(
                        f"并行分块大小错误：应为{expected_size}，实际{len(payload)}"
                    )
                output.write(payload)
                output.flush()
                block_downloaded += len(payload)
                completed_bytes += len(payload)

                elapsed = max(time.monotonic() - started, 0.001)
                speed = (completed_bytes - session_start_bytes) / elapsed
                percent = completed_bytes / total_bytes * 100
                print(
                    f"\r下载 {percent:6.2f}%  "
                    f"{format_bytes(completed_bytes)}/{format_bytes(total_bytes)}  "
                    f"{format_bytes(int(speed))}/s  "
                    f"[{entry_index + 1}/{len(entries)}，{workers}路]",
                    end="",
                    flush=True,
                )
                write_json_atomic(
                    state_path,
                    {
                        "fingerprint": fingerprint,
                        "next_entry": entry_index,
                        "current_block_bytes": block_downloaded,
                    },
                )
                schedule_chunk()

            current_block_bytes = 0
            write_json_atomic(
                state_path,
                {
                    "fingerprint": fingerprint,
                    "next_entry": entry_index + 1,
                    "current_block_bytes": 0,
                },
            )

        output.write(MINIMAL_END_HEADER)
        output.flush()
        os.fsync(output.fileno())

    print()
    os.replace(part_path, output_archive)
    state_path.unlink(missing_ok=True)
    write_json_atomic(
        selection_path,
        {
            "fingerprint": fingerprint,
            "source_url": client.url,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "archive_size": output_archive.stat().st_size,
            "entries": [asdict(entry) for entry in entries],
        },
    )
    print(f"选择压缩包已生成：{output_archive}")


def find_unrar(explicit_path: str | None) -> Path:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    path_candidate = shutil.which("UnRAR.exe") or shutil.which("unrar")
    if path_candidate:
        candidates.append(Path(path_candidate))
    candidates.extend(
        [
            Path(r"C:\Program Files\WinRAR\UnRAR.exe"),
            Path(r"C:\Program Files (x86)\WinRAR\UnRAR.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise DroneRFaError(
        "未找到UnRAR.exe。请安装WinRAR，或使用--unrar指定程序路径"
    )


def run_unrar(unrar: Path, command: str, archive: Path, destination: Path | None) -> None:
    args = [str(unrar), command, "-idq", "-p-", str(archive)]
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        args.extend(["-o-", str(destination) + os.sep])
    result = subprocess.run(args, check=False)
    if result.returncode != 0:
        action = "解压" if destination is not None else "完整性测试"
        raise DroneRFaError(f"WinRAR {action}失败，退出码：{result.returncode}")


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "从ScienceDB的DroneRFa.rar中选择性下载文件，"
            "无需下载完整573.65 GiB压缩包。"
        )
    )
    parser.add_argument("--url", default=SOURCE_URL, help=argparse.SUPPRESS)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=script_dir / "dronerfa_manifest.json",
        help="RAR目录缓存文件",
    )
    parser.add_argument(
        "--targets",
        type=parse_csv,
        default=DEFAULT_TARGETS,
        help="目标编号，逗号分隔（默认：T0000,T0010,T0101）",
    )
    parser.add_argument(
        "--distances",
        type=parse_csv,
        default=DEFAULT_DISTANCES,
        help="距离编号，逗号分隔（默认：D00）",
    )
    parser.add_argument(
        "--segments",
        type=parse_csv,
        default=DEFAULT_SEGMENTS,
        help="片段编号，逗号分隔（默认：S0000,S0001,S0010,S0011）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("DroneRFa_pilot.rar"),
        help="输出选择压缩包（默认：DroneRFa_pilot.rar）",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("DroneRFa_pilot"),
        help="解压目录（默认：DroneRFa_pilot）",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="只生成并验证RAR，不自动解压",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将下载的文件和空间需求",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="不联网，只显示默认下载任务的本地进度",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="只建立远程RAR目录索引",
    )
    parser.add_argument(
        "--refresh-index",
        action="store_true",
        help="忽略本地目录缓存并重新索引",
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=16,
        help="每次Range请求大小（MiB，默认：16）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并行Range请求数（默认：4，范围：1-8）",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "curl", "requests"),
        default="auto",
        help="大文件分块下载后端（默认：auto，Windows优先curl）",
    )
    parser.add_argument(
        "--unrar",
        help="UnRAR.exe路径",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="删除已有输出和断点，从头重新下载",
    )
    parser.add_argument(
        "--ignore-space-check",
        action="store_true",
        help="忽略磁盘空间检查（不建议）",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.chunk_mib < 1 or args.chunk_mib > 512:
        raise DroneRFaError("--chunk-mib必须在1到512之间")
    if args.workers < 1 or args.workers > 8:
        raise DroneRFaError("--workers必须在1到8之间")

    output_archive = args.output.resolve()
    manifest_path = args.manifest.resolve()
    extract_dir = None if args.no_extract else args.extract_dir.resolve()

    if args.status:
        return print_download_status(
            manifest_path,
            output_archive,
            args.targets,
            args.distances,
            args.segments,
        )

    print("连接ScienceDB并检查DroneRFa压缩包……", flush=True)
    with RangeClient(args.url) as client:
        archive_size, etag = client.inspect()
        print(f"远程压缩包：{format_bytes(archive_size)}")
        if args.url == SOURCE_URL and archive_size != EXPECTED_ARCHIVE_SIZE:
            raise DroneRFaError(
                f"远程文件大小发生变化：预期{EXPECTED_ARCHIVE_SIZE}，"
                f"实际{archive_size}"
            )

        entries = None
        if not args.refresh_index:
            entries = load_manifest(manifest_path, args.url, archive_size)
        if entries is None:
            entries = build_manifest(
                client,
                manifest_path,
                archive_size,
                etag,
            )
        else:
            print(f"使用本地目录索引：{manifest_path}（{len(entries)}个文件）")

        if args.index_only:
            return 0

        selected = select_entries(
            entries,
            args.targets,
            args.distances,
            args.segments,
        )
        print_selection(selected)
        for target in args.targets:
            print(f"  {target}: {TARGET_LABELS.get(target, '未登记机型')}")

        if args.dry_run:
            print("\n当前为--dry-run，未下载数据。")
            return 0

        download_size = sum(entry.block_size for entry in selected)
        unpacked_size = sum(entry.unpacked_size for entry in selected)
        part_path = output_archive.with_name(output_archive.name + ".part")
        partial_size = part_path.stat().st_size if part_path.exists() else 0
        if not args.ignore_space_check:
            check_disk_space(
                output_archive,
                extract_dir,
                download_size,
                unpacked_size,
                partial_size,
            )

        stream_selected_archive(
            client,
            selected,
            output_archive,
            args.chunk_mib * 1024**2,
            args.workers,
            args.backend,
            args.overwrite,
        )

    unrar = find_unrar(args.unrar)
    print("使用WinRAR检查压缩包完整性……")
    run_unrar(unrar, "t", output_archive, None)
    print("压缩包完整性测试通过。")

    if extract_dir is not None:
        print(f"解压到：{extract_dir}")
        run_unrar(unrar, "x", output_archive, extract_dir)
        print("解压完成。")

    print("\n完成。中断时重新运行相同命令即可从断点继续。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中断，断点已经保留。重新运行相同命令即可继续。")
        raise SystemExit(130)
    except DroneRFaError as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
