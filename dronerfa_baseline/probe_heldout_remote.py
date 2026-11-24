"""Remote availability probe for the 12 candidate DroneRFa files (read-only, tiny ranges).

Safety contract
---------------
A range request is only accepted after *all* of these hold, and none of them is
evaluated after the body has been consumed:

  * HTTP status is exactly 206 (a 200 means the server ignored Range, so the
    body would be the whole ~615 GB archive -> aborted without reading);
  * ``Content-Range`` parses as ``bytes <start>-<end>/<total>`` with the exact
    requested start/end and a total equal to the expected archive size;
  * ``Content-Encoding`` is absent or ``identity`` (otherwise byte counts and
    offsets are meaningless);
  * ``Content-Length``, when present, equals the requested length.

The body is then read with a hard cap of ``requested length + 1`` bytes so a
misbehaving server can never stream the full archive into memory.

Failure handling
----------------
A wrong signature or a wrong archive size is a hard stop.  The report is always
written (success or failure) before exiting.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import socket
import ssl
import struct
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

URL = "https://china.scidb.cn/download?fileId=c403fc76444e4b9989e4f3ff570f3b3d"
RAR5_SIGNATURE = bytes.fromhex("526172211a070100")
EXPECTED_ARCHIVE_SIZE = 615_955_805_466
HEAD_FILE = 2

HFL_EXTRA = 0x0001
HFL_DATA = 0x0002
FHFL_UTIME = 0x0002
FHFL_CRC32 = 0x0004

ROOT = Path(r"C:\Users\Administrator\Desktop\数据集处理")
CANDIDATES = (
    ROOT / "dronerfa_baseline/outputs/noise_ab_replicates_20260917/candidate_heldout_files.json"
)
DEFAULT_REPORT = (
    ROOT / "dronerfa_baseline/outputs/heldout_preparation_20260917/remote_probe.json"
)

RETRYABLE_HTTP = (429, 500, 502, 503, 504)


class ProtocolViolation(RuntimeError):
    """The server answered a range request in a way that cannot be trusted."""


class RangeResult:
    __slots__ = ("status", "headers", "body", "attempts", "content_range")

    def __init__(
        self,
        status: int,
        headers: dict[str, str],
        body: bytes,
        attempts: int,
        content_range: tuple[int, int, int] | None,
    ) -> None:
        self.status = status
        self.headers = headers
        self.body = body
        self.attempts = attempts
        self.content_range = content_range


def parse_content_range(value: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _response_facts(status: int, headers: dict[str, str]) -> dict:
    return {
        "http_status": status,
        "content_range": headers.get("Content-Range"),
        "content_length": headers.get("Content-Length"),
        "content_encoding": headers.get("Content-Encoding"),
        "accept_ranges": headers.get("Accept-Ranges"),
        "transfer_encoding": headers.get("Transfer-Encoding"),
        "etag": headers.get("ETag"),
        "last_modified": headers.get("Last-Modified"),
    }


def http_get_range(
    start: int,
    end: int,
    *,
    retries: int = 4,
    timeout: int = 60,
    expected_total: int | None = EXPECTED_ARCHIVE_SIZE,
) -> RangeResult:
    """Fetch ``bytes=start-end`` with full header validation before any body read."""
    parts = urlsplit(URL)
    if parts.scheme != "https":
        raise ValueError(f"unexpected URL scheme: {parts.scheme}")
    target = parts.path + (f"?{parts.query}" if parts.query else "")
    length = end - start + 1

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        connection = None
        try:
            context = ssl.create_default_context()  # certificate verification stays on
            connection = http.client.HTTPSConnection(
                parts.hostname, parts.port or 443, timeout=timeout, context=context
            )
            connection.request(
                "GET",
                target,
                headers={
                    "Host": parts.netloc,
                    "Range": f"bytes={start}-{end}",
                    "Accept": "*/*",
                    "Accept-Encoding": "identity",
                    "Connection": "close",
                    "User-Agent": "DroneRFa-Heldout-Probe/1.0",
                },
            )
            response = connection.getresponse()
            headers = {key: value for key, value in response.getheaders()}
            facts = _response_facts(response.status, headers)

            # ---- validate BEFORE consuming any body -------------------------
            if response.status in RETRYABLE_HTTP and response.status != 206:
                try:
                    response.read(0)
                finally:
                    connection.close()
                last_error = ProtocolViolation(f"HTTP {response.status} for range {start}-{end}")
                if attempt < retries:
                    delay = min(60 * attempt, 300)
                    print(f"  收到 HTTP {response.status}，{delay} 秒后重试（{attempt}/{retries}）", flush=True)
                    time.sleep(delay)
                continue
            if response.status != 206:
                raise ProtocolViolation(
                    f"range {start}-{end} 期望HTTP 206，实际 {response.status}"
                    f"（服务器可能忽略Range，已放弃读取正文）"
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
            if expected_total is not None and total != expected_total:
                raise ProtocolViolation(
                    f"Content-Range 总长不符：期望 {expected_total}，返回 {total}"
                )
            declared = headers.get("Content-Length")
            if declared is not None and int(declared) != length:
                raise ProtocolViolation(
                    f"Content-Length {declared} 与请求长度 {length} 不符"
                )

            # ---- bounded read: never more than requested length + 1 ---------
            body = _read_bounded(response, length)
            if len(body) != length:
                raise ProtocolViolation(f"收到 {len(body)} 字节，期望 {length} 字节")

            return RangeResult(response.status, headers, body, attempt, content_range)
        except (ProtocolViolation, http.client.HTTPException, ssl.SSLError, OSError, socket.timeout, ValueError) as exc:
            last_error = exc
            if isinstance(exc, ProtocolViolation):
                raise
            if attempt < retries:
                time.sleep(min(2**attempt, 15))
        finally:
            if connection is not None:
                connection.close()

    raise RuntimeError(f"range {start}-{end} failed after {retries} attempts: {last_error}")


def _read_bounded(response: http.client.HTTPResponse, length: int) -> bytes:
    """Read at most ``length + 1`` bytes; a longer body is itself a failure."""
    cap = length + 1
    chunks: list[bytes] = []
    total = 0
    while total < cap:
        block = response.read(min(65536, cap - total))
        if not block:
            break
        chunks.append(block)
        total += len(block)
    if total > length:
        raise ProtocolViolation(
            f"服务器返回的正文超过请求长度（>{length}字节），已中止"
        )
    return b"".join(chunks)


def read_vint(data: bytes, offset: int):
    value = 0
    shift = 0
    for _ in range(10):
        if offset >= len(data):
            raise ValueError("truncated vint")
        current = data[offset]
        offset += 1
        value |= (current & 0x7F) << shift
        if current & 0x80 == 0:
            return value, offset
        shift += 7
    raise ValueError("invalid vint")


def parse_rar5_header(data: bytes) -> dict:
    stored_crc = struct.unpack_from("<I", data, 0)[0]
    header_size, cursor = read_vint(data, 4)
    header_end = cursor + header_size
    if header_end > len(data):
        raise ValueError("header longer than probe window")
    if zlib.crc32(data[4:header_end]) & 0xFFFFFFFF != stored_crc:
        raise ValueError("RAR header CRC mismatch")

    header_type, cursor = read_vint(data, cursor)
    header_flags, cursor = read_vint(data, cursor)
    extra_size = 0
    data_size = 0
    if header_flags & HFL_EXTRA:
        extra_size, cursor = read_vint(data, cursor)
    if header_flags & HFL_DATA:
        data_size, cursor = read_vint(data, cursor)

    info: dict = {
        "header_type": header_type,
        "header_flags": header_flags,
        "header_total_size": header_end,
        "data_size": data_size,
    }
    if header_type == HEAD_FILE:
        file_flags, cursor = read_vint(data, cursor)
        unpacked_size, cursor = read_vint(data, cursor)
        _, cursor = read_vint(data, cursor)
        if file_flags & FHFL_UTIME:
            cursor += 4
        if file_flags & FHFL_CRC32:
            cursor += 4
        _, cursor = read_vint(data, cursor)
        _, cursor = read_vint(data, cursor)
        name_size, cursor = read_vint(data, cursor)
        basic_end = header_end - extra_size
        if cursor + name_size > basic_end:
            raise ValueError("name crosses header boundary")
        info["name"] = data[cursor : cursor + name_size].decode("utf-8")
        info["unpacked_size"] = unpacked_size
    return info


def curl_signature_check() -> dict:
    """Independent confirmation of signature/Content-Range via curl as a backend."""
    curl = "curl.exe"
    result = subprocess.run(
        [
            curl,
            "--location",
            "--silent",
            "--show-error",
            "--head",
            "--connect-timeout",
            "20",
            "--max-time",
            "120",
            URL,
        ],
        capture_output=True,
        check=False,
        timeout=180,
    )
    text = result.stdout.decode("utf-8", errors="replace")
    match = re.search(r"Content-Range:\s*bytes (\d+)-(\d+)/(\d+)", text, re.IGNORECASE)
    return {
        "curl_returncode": result.returncode,
        "content_range": match.group(0) if match else None,
        "total": int(match.group(3)) if match else None,
        "stderr": result.stderr.decode("utf-8", errors="replace")[:200],
    }


def run(args: argparse.Namespace) -> tuple[dict, int]:
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))["files"]
    report: dict = {
        "probed_at": datetime.now(timezone.utc).isoformat(),
        "url": URL,
        "expected_archive_size": EXPECTED_ARCHIVE_SIZE,
        "probe_span_bytes": 4096,
        "checks": {},
        "entries": [],
        "summary": {},
    }
    exit_code = 1

    # ---- probe 1: signature + archive size --------------------------------
    print("== 探测1：RAR5签名与档案总长 ==", flush=True)
    try:
        first = http_get_range(0, 7)
    except (ProtocolViolation, RuntimeError) as exc:
        report["checks"]["range_0_7"] = {"passed": False, "error": str(exc)}
        report["summary"] = {"stopped_at": "signature_probe", "reason": str(exc)}
        print(f"签名探测失败：{exc}", flush=True)
        return report, 1

    signature_ok = first.body == RAR5_SIGNATURE
    total = first.content_range[2] if first.content_range else None
    size_ok = total == EXPECTED_ARCHIVE_SIZE
    report["checks"]["range_0_7"] = {
        **_response_facts(first.status, first.headers),
        "requested": [0, 7],
        "bytes_received": len(first.body),
        "signature_hex": first.body.hex(),
        "signature_ok": signature_ok,
        "archive_size": total,
        "size_matches_expected": size_ok,
        "attempts": first.attempts,
        "passed": signature_ok and size_ok,
    }
    print(
        f"status={first.status} content-range={first.headers.get('Content-Range')} "
        f"sig_ok={signature_ok} size_ok={size_ok}",
        flush=True,
    )
    if not signature_ok or not size_ok:
        report["summary"] = {
            "stopped_at": "signature_probe",
            "reason": f"signature_ok={signature_ok} size_ok={size_ok}",
            "signature_ok": signature_ok,
            "size_matches_expected": size_ok,
        }
        print("签名或档案大小不符 —— 立即停止，不进行大文件下载。", flush=True)
        return report, 1

    # ---- probe 2: interior range correctness ------------------------------
    print("== 探测2：内部区间的Content-Range与长度 ==", flush=True)
    probe_start, probe_len = 1024, 512
    try:
        second = http_get_range(probe_start, probe_start + probe_len - 1)
        report["checks"]["interior_range"] = {
            **_response_facts(second.status, second.headers),
            "requested": [probe_start, probe_start + probe_len - 1],
            "bytes_received": len(second.body),
            "sha256": hashlib.sha256(second.body).hexdigest(),
            "attempts": second.attempts,
            "passed": len(second.body) == probe_len,
        }
        print(
            f"status={second.status} content-range={second.headers.get('Content-Range')} "
            f"len={len(second.body)}",
            flush=True,
        )
    except (ProtocolViolation, RuntimeError) as exc:
        report["checks"]["interior_range"] = {"passed": False, "error": str(exc)}
        report["summary"] = {"stopped_at": "interior_range", "reason": str(exc)}
        print(f"内部区间探测失败：{exc}", flush=True)
        return report, 1

    # ---- probe 3: candidate block headers ---------------------------------
    print("== 探测3：12个候选文件块头 ==", flush=True)
    for item in candidates:
        name = item["name"]
        offset = item["block_offset"]
        packet: dict = {
            "name": name,
            "manifest_block_offset": offset,
            "manifest_block_size": item["block_size"],
            "manifest_packed_size": item["packed_size"],
            "manifest_unpacked_size": item["unpacked_size"],
        }
        try:
            result = http_get_range(offset, offset + args.probe_span - 1)
            packet.update(_response_facts(result.status, result.headers))
            packet["bytes_received"] = len(result.body)
            packet["attempts"] = result.attempts
            if result.content_range is None:
                raise ProtocolViolation("缺少可解析的Content-Range")
            packet["requested_start"] = offset
            packet["content_range_start_matches_request"] = result.content_range[0] == offset
            packet["content_range_end_matches_request"] = (
                result.content_range[1] == offset + args.probe_span - 1
            )
            packet["content_range_total_matches_archive"] = (
                result.content_range[2] == EXPECTED_ARCHIVE_SIZE
            )

            header = parse_rar5_header(result.body)
            packet["header"] = header
            packet["name_matches"] = header.get("name") == name
            packet["header_type_is_file"] = header.get("header_type") == HEAD_FILE
            packet["packed_size_matches_header_data_size"] = (
                header["data_size"] == item["packed_size"]
            )
            packet["block_size_matches"] = (
                header["header_total_size"] + header["data_size"] == item["block_size"]
            )
            packet["unpacked_size_matches"] = (
                header.get("unpacked_size") == item["unpacked_size"]
            )

            packet["passed"] = all(
                [
                    result.status == 206,
                    packet["content_range_start_matches_request"],
                    packet["content_range_end_matches_request"],
                    packet["content_range_total_matches_archive"],
                    len(result.body) == args.probe_span,
                    packet["name_matches"],
                    packet["header_type_is_file"],
                    packet["packed_size_matches_header_data_size"],
                    packet["block_size_matches"],
                    packet["unpacked_size_matches"],
                ]
            )
        except (ProtocolViolation, RuntimeError, ValueError) as exc:
            packet["passed"] = False
            packet["error"] = f"{type(exc).__name__}: {exc}"
        report["entries"].append(packet)
        print(
            f"{Path(name).name}: passed={packet.get('passed')} "
            f"name={packet.get('name_matches')} packed={packet.get('packed_size_matches_header_data_size')} "
            f"block={packet.get('block_size_matches')} unpacked={packet.get('unpacked_size_matches')} "
            f"err={packet.get('error', '-')}",
            flush=True,
        )

    verified = sum(1 for entry in report["entries"] if entry.get("passed"))
    report["checks"]["curl_cross_check"] = curl_signature_check()
    report["checks"]["curl_cross_check"]["passed"] = (
        report["checks"]["curl_cross_check"]["total"] == EXPECTED_ARCHIVE_SIZE
    )
    report["summary"] = {
        "signature_ok": signature_ok,
        "size_matches_expected": size_ok,
        "range_supported": first.status == 206 and second.status == 206,
        "candidate_count": len(candidates),
        "candidates_verified": verified,
        "candidates_failed": len(candidates) - verified,
        "all_candidates_verified": verified == len(candidates),
    }
    exit_code = 0 if report["summary"]["all_candidates_verified"] else 1
    return report, exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--probe-span", type=int, default=4096)
    args = parser.parse_args()
    if args.probe_span < 512:
        raise SystemExit("--probe-span 至少为512")

    report: dict = {}
    exit_code = 1
    try:
        report, exit_code = run(args)
    except BaseException as exc:  # noqa: BLE001 - report must always be written
        report = {
            "probed_at": datetime.now(timezone.utc).isoformat(),
            "url": URL,
            "checks": {},
            "entries": [],
            "summary": {"stopped_at": "unhandled", "reason": f"{type(exc).__name__}: {exc}"},
        }
        print(f"探测未预期终止：{type(exc).__name__}: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nsummary:", json.dumps(report.get("summary", {}), ensure_ascii=False))
        print("报告已写入:", args.out)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
