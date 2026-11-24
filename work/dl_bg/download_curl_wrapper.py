"""Run download_dronerfa_subset.py with every HTTP request going through curl.

Python's ssl fails the TLS handshake with china.scidb.cn on this machine
(SSLEOFError), while curl.exe works. This wrapper replaces the two
requests-based methods of RangeClient (inspect, get_range) with curl calls and
then runs the original main() unchanged. All other arguments pass through.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import download_dronerfa_subset as dl  # noqa: E402

CURL = shutil.which("curl.exe") or shutil.which("curl")


def _curl_range(url: str, start: int, end: int) -> tuple[bytes, str]:
    result = subprocess.run(
        [
            CURL, "-sS", "-L", "--fail", "--max-time", "330",
            "-H", "Accept-Encoding: identity",
            "-r", f"{start}-{end}",
            "-D", "-",  # headers to stdout before body
            url,
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise dl.DroneRFaError(
            f"curl退出码{result.returncode}：{result.stderr.decode(errors='replace')}"
        )
    raw = result.stdout
    # With -L there may be several header blocks; the body follows the last one.
    headers_text = ""
    body = raw
    while body[:5] == b"HTTP/":
        cut = body.find(b"\r\n\r\n")
        if cut < 0:
            break
        headers_text = body[:cut].decode("latin-1")
        body = body[cut + 4:]
    return body, headers_text


def _content_range(headers_text: str) -> tuple[int, int, int]:
    match = re.search(r"(?im)^content-range:\s*bytes (\d+)-(\d+)/(\d+)", headers_text)
    if not match:
        raise dl.DroneRFaError(f"服务器返回了无效的Content-Range：{headers_text!r}")
    return tuple(int(x) for x in match.groups())  # type: ignore[return-value]


def inspect(self):
    body, headers = _curl_range(self.url, 0, 7)
    start, end, total = _content_range(headers)
    if (start, end) != (0, 7) or body[:8] != dl.RAR5_SIGNATURE:
        raise dl.DroneRFaError("远程文件不是预期的DroneRFa RAR5压缩包")
    etag = re.search(r'(?im)^etag:\s*(.+?)\s*$', headers)
    self.archive_size = total
    self.etag = etag.group(1) if etag else None
    return total, self.etag


def get_range(self, start: int, end: int) -> bytes:
    expected = end - start + 1
    last_error: Exception | None = None
    for attempt in range(1, self.retries + 1):
        try:
            body, headers = _curl_range(self.url, start, end)
            actual_start, actual_end, total = _content_range(headers)
            if (actual_start, actual_end) != (start, end):
                raise dl.DroneRFaError(f"服务器返回了错误区间 {actual_start}-{actual_end}")
            if len(body) != expected:
                raise dl.DroneRFaError(f"区间应为{expected}字节，实际收到{len(body)}字节")
            if self.archive_size is not None and total != self.archive_size:
                raise dl.DroneRFaError("下载过程中远程压缩包大小发生变化")
            return body
        except dl.DroneRFaError as exc:
            last_error = exc
            if attempt < self.retries:
                delay = min(2**attempt, 15)
                print(f"\n网络请求失败，{delay}秒后重试（{attempt}/{self.retries}）：{exc}", flush=True)
                time.sleep(delay)
    raise dl.DroneRFaError(f"多次重试后仍无法下载区间{start}-{end}：{last_error}")


ZERO_PAGE = 4096


def has_zero_hole(payload: bytes) -> bool:
    """True if any 4 KiB-aligned page is all zero.

    The server sometimes returns a full-length 206 response with a ~1 MiB run
    of zero bytes inside (2026-10-14, found by comparing a corrupt T0011 block
    with a fresh download). Genuine data can also contain zero pages, so a
    chunk with zero pages is only accepted once two downloads agree.
    """
    import numpy as np

    n = len(payload) // ZERO_PAGE * ZERO_PAGE
    if n == 0:
        return False
    pages = np.frombuffer(payload, dtype=np.uint8, count=n).reshape(-1, ZERO_PAGE)
    return bool((~pages.any(axis=1)).any())


_original_download_range = dl.RangeClient.download_range


def download_range(self, start: int, end: int, backend: str) -> bytes:
    payload = _original_download_range(self, start, end, backend)
    if not has_zero_hole(payload):
        return payload
    seen = [payload]
    for attempt in range(1, 9):
        print(f"\n区间{start}-{end}含全零数据页，重新下载核对（{attempt}/8）", flush=True)
        payload = _original_download_range(self, start, end, backend)
        if any(payload == s for s in seen):
            return payload
        seen.append(payload)
    raise dl.DroneRFaError(f"区间{start}-{end}多次下载结果不一致")


dl.RangeClient.inspect = inspect
dl.RangeClient.get_range = get_range
dl.RangeClient.download_range = download_range

if __name__ == "__main__":
    if "--backend" not in sys.argv:
        sys.argv += ["--backend", "curl"]
    try:
        sys.exit(dl.main())
    except dl.DroneRFaError as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        sys.exit(1)
