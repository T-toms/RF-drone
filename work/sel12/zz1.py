#!/usr/bin/env python3
"""Read-only remote identity + throughput probe for the DroneRFa archive.

Never writes anything; used to confirm URL/ETag/size/206 semantics and to measure
real single-stream and multi-stream throughput before freezing download settings.
"""

from __future__ import annotations

import argparse
import http.client
import json
import ssl
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

URL = "https://china.scidb.cn/download?fileId=c403fc76444e4b9989e4f3ff570f3b3d"
EXPECTED_SIZE = 615_955_805_466
EXPECTED_ETAG = '"6774350c-8f69d3b51a"'


def open_conn(host: str, timeout: float) -> http.client.HTTPSConnection:
    ctx = ssl.create_default_context()
    return http.client.HTTPSConnection(host, timeout=timeout, context=ctx)


def fetch(host: str, path: str, start: int, end: int, timeout: float) -> dict:
    conn = open_conn(host, timeout)
    try:
        conn.request(
            "GET",
            path,
            headers={
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
                "User-Agent": "DroneRFa-Select12-Probe/1.0",
                "Connection": "close",
            },
        )
        resp = conn.getresponse()
        body = resp.read()
        return {
            "status": resp.status,
            "reason": resp.reason,
            "content_range": resp.getheader("Content-Range"),
            "content_length": resp.getheader("Content-Length"),
            "etag": resp.getheader("ETag"),
            "accept_ranges": resp.getheader("Accept-Ranges"),
            "content_encoding": resp.getheader("Content-Encoding"),
            "last_modified": resp.getheader("Last-Modified"),
            "bytes": len(body),
            "first8": body[:8].hex(),
            "start": start,
            "end": end,
        }
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-mib", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--base-offset", type=int, default=29_000_000_000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    parts = urlsplit(URL)
    host, path = parts.netloc, parts.path + "?" + parts.query
    block = args.block_mib * 1024 * 1024

    identity = fetch(host, path, 0, 7, 60.0)
    identity_ok = (
        identity["status"] == 206
        and identity["content_range"] == f"bytes 0-7/{EXPECTED_SIZE}"
        and identity["etag"] == EXPECTED_ETAG
        and identity["bytes"] == 8
        and identity["first8"] == "526172211a070100"
    )

    results: list[dict] = []
    notes: list[str] = []

    def one(i: int) -> dict:
        start = args.base_offset + i * block
        end = start + block - 1
        t0 = time.monotonic()
        last = ""
        for attempt in range(3):
            try:
                r = fetch(host, path, start, end, 600.0)
                if (
                    r["status"] == 206
                    and r["content_range"] == f"bytes {start}-{end}/{EXPECTED_SIZE}"
                    and r["etag"] == EXPECTED_ETAG
                    and r["bytes"] == block
                ):
                    r["seconds"] = time.monotonic() - t0
                    r["ok"] = True
                    return r
                last = json.dumps(r, ensure_ascii=False)
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0 + attempt)
        return {"ok": False, "start": start, "end": end, "error": last}

    t0 = time.monotonic()
    single = one(0)
    single_seconds = time.monotonic() - t0

    t1 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, range(1, args.workers + 1)))
    parallel_seconds = time.monotonic() - t1

    good = [r for r in results if r.get("ok")]
    total_bytes = sum(r["bytes"] for r in good)
    report = {
        "url": URL,
        "identity": identity,
        "identity_ok": identity_ok,
        "block_bytes": block,
        "single_stream": {
            "seconds": single_seconds,
            "bytes": single.get("bytes"),
            "ok": single.get("ok"),
            "MiB_per_s": round(single.get("bytes", 0) / 1048576 / max(single_seconds, 1e-6), 4),
        },
        "parallel": {
            "workers": args.workers,
            "seconds": parallel_seconds,
            "ok_count": len(good),
            "bytes": total_bytes,
            "MiB_per_s": round(total_bytes / 1048576 / max(parallel_seconds, 1e-6), 4),
        },
        "per_stream": [
            {
                "ok": r.get("ok"),
                "seconds": round(r.get("seconds", 0), 3),
                "MiB_per_s": round(r.get("bytes", 0) / 1048576 / max(r.get("seconds", 1e-6), 1e-6), 4)
                if r.get("ok")
                else None,
                "error": r.get("error"),
            }
            for r in results
        ],
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    return 0 if (identity_ok and single.get("ok") and good) else 1


if __name__ == "__main__":
    sys.exit(main())
