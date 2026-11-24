# -*- coding: utf-8 -*-
"""META-HEADER1 / step 2 (network, strictly header-only).

For each of the 88 un-downloaded members of the three legacy classes
(T0000 Background, T0010 Phantom 4 Pro, T0101 Air 2S) this reads ONLY the RAR5
FILE block header that sits at the frozen ``block_offset`` of the official
DroneRFa archive:

  stage 1  Range: bytes=<off>-<off+7>                     -> decode HEAD_SIZE vint
  stage 2  Range: bytes=<off>-<off+header_span-1>         -> full header, no payload

Hard limits, enforced in code:
  * ``header_span`` must be <= 128 bytes (rar5_header.MAX_HEADER_SPAN_BYTES);
    a larger header is reported as an anomaly and its stage-2 read is skipped.
  * a response is only read after the status/Content-Range/total/ETag checks
    pass; a non-206 answer is closed WITHOUT reading a body, so the client can
    never fall back to pulling the 615 GB object.
  * HTTP 429 gets a bounded backoff (3 attempts); persistent failure keeps the
    error and moves on.  A run-wide 429 circuit breaker stops the sweep.

No payload byte is ever requested, nothing is written outside work/meta_header1/.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import http.client
import json
import os
import re
import ssl
import sys
import time
from urllib.parse import urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import rar5_header as R  # noqa: E402

CONFIG = os.path.join(WORKSPACE, "work", "sel12g", "sel12g-config-20260924.json")
CANDIDATES = os.path.join(WORKSPACE, "work", "meta_split1",
                          "archive_unused_for_legacy3_20260925.csv")
INDEX = os.path.join(WORKSPACE, "dronerfa_manifest.json")
PREREG = os.path.join(HERE, "htime-decode-preregistration-20260925.json")
OUT_JSON = os.path.join(HERE, "member_header_probe_88.json")
OUT_CSV = os.path.join(HERE, "member_header_probe_88.csv")
OUT_HTTP = os.path.join(HERE, "http_request_log.json")

RETRYABLE_HTTP = (408, 425, 429, 500, 502, 503, 504)
BACKOFF_429 = (2.0, 5.0, 15.0)      # bounded: at most 3 attempts on a 429
BACKOFF_OTHER = (1.0, 3.0)
POLITE_DELAY = 0.15
MAX_CONSECUTIVE_429 = 8
MAX_TOTAL_429 = 30
CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$", re.I)


class ProtocolViolation(RuntimeError):
    pass


class HeaderRangeClient:
    """Bounded Range client.  One HTTPS connection per request."""

    def __init__(self, url: str, expected_total: int, expected_etag: str,
                 timeout: float = 60.0):
        parts = urlsplit(url)
        if parts.scheme.lower() != "https":
            raise ProtocolViolation(f"only https endpoints are accepted: {url}")
        self.host = parts.netloc
        self.target = parts.path + ("?" + parts.query if parts.query else "")
        self.expected_total = expected_total
        self.expected_etag = expected_etag
        self.timeout = timeout
        self.ssl_context = ssl.create_default_context()
        self.log: list = []
        self.bytes_transferred = 0
        self.requests = 0

    def _fetch_once(self, start: int, end: int):
        expected = end - start + 1
        connection = http.client.HTTPSConnection(
            self.host, timeout=self.timeout, context=self.ssl_context)
        try:
            connection.request("GET", self.target, headers={
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
                "User-Agent": "DroneRFa-MetaHeader1-HeaderProbe/1.0",
                "Connection": "close",
            })
            response = connection.getresponse()
            status = response.status
            headers = {
                "content-range": response.getheader("Content-Range"),
                "content-length": response.getheader("Content-Length"),
                "etag": response.getheader("ETag"),
                "content-encoding": response.getheader("Content-Encoding"),
                "accept-ranges": response.getheader("Accept-Ranges"),
            }
            # ---- validate BEFORE touching the body -------------------------
            if status != 206:
                response.close()
                return None, headers, status, f"HTTP {status} (206 expected)"
            encoding = (headers.get("content-encoding") or "identity").strip().lower()
            if encoding not in ("", "identity"):
                response.close()
                return None, headers, status, f"unexpected Content-Encoding {encoding!r}"
            match = CONTENT_RANGE_RE.match(headers.get("content-range") or "")
            if not match:
                response.close()
                return None, headers, status, (
                    f"unparsable Content-Range {headers.get('content-range')!r}")
            actual_start, actual_end, total = (int(match.group(1)),
                                               int(match.group(2)),
                                               int(match.group(3)))
            if total != self.expected_total:
                response.close()
                return None, headers, status, (
                    f"archive total {total} != frozen {self.expected_total}")
            if (actual_start, actual_end) != (start, end):
                response.close()
                return None, headers, status, (
                    f"Content-Range {actual_start}-{actual_end} != requested {start}-{end}")
            if headers.get("etag") != self.expected_etag:
                response.close()
                return None, headers, status, (
                    f"ETag {headers.get('etag')!r} != frozen {self.expected_etag!r}")
            declared = headers.get("content-length")
            if declared is not None and int(declared) != expected:
                response.close()
                return None, headers, status, (
                    f"Content-Length {declared} != requested {expected}")
            # ---- only now read, and only the requested window ---------------
            body = response.read(expected)
            return body, headers, status, None
        finally:
            connection.close()

    def fetch(self, start: int, end: int, label: str) -> tuple:
        expected = end - start + 1
        attempts = 0
        last_error = None
        while True:
            attempts += 1
            self.requests += 1
            entry = {"label": label, "range": f"{start}-{end}", "attempt": attempts,
                     "requested_bytes": expected,
                     "at": dt.datetime.now(dt.timezone.utc).isoformat()}
            try:
                body, headers, status, error = self._fetch_once(start, end)
            except Exception as exc:                       # noqa: BLE001
                body, headers, status, error = None, {}, None, f"{type(exc).__name__}: {exc}"
            entry["status"] = status
            entry["etag"] = headers.get("etag")
            entry["content_range"] = headers.get("content-range")
            entry["error"] = error
            entry["body_bytes"] = len(body) if body is not None else 0
            self.bytes_transferred += entry["body_bytes"]
            self.log.append(entry)
            if body is not None and len(body) == expected:
                return body, {"attempts": attempts, "status": status,
                              "etag": headers.get("etag"),
                              "content_range": headers.get("content-range")}
            if body is not None:
                error = f"body {len(body)} bytes, expected {expected}"
                entry["error"] = error
            last_error = error
            retryable = (status in RETRYABLE_HTTP) or (status is None)
            if not retryable:
                raise ProtocolViolation(f"{label}: {error}")
            table = BACKOFF_429 if status == 429 else BACKOFF_OTHER
            if attempts > len(table):
                raise TimeoutError(f"{label}: giving up after {attempts} attempts ({error})")
            delay = table[attempts - 1]
            print(f"    backoff {delay}s before attempt {attempts + 1} ({error})",
                  flush=True)
            time.sleep(delay)
        raise AssertionError("unreachable", last_error)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_inputs() -> dict:
    config = json.load(open(CONFIG, encoding="utf-8"))
    index = json.load(open(INDEX, encoding="utf-8"))
    index_by_name = {e["name"]: e for e in index["entries"]}
    with open(CANDIDATES, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        archive_name = "DroneRFa/" + row["file_name"]
        entry = index_by_name.get(archive_name)
        row["archive_name"] = archive_name
        row["packed_size"] = int(row["packed_size"])
        row["unpacked_size"] = int(row["unpacked_size"])
        row["block_offset"] = int(row["block_offset"])
        row["index_entry"] = entry
    return {"config": config, "index": index, "index_by_name": index_by_name,
            "rows": rows}


def gate_checks(frozen: dict, prereg: dict) -> dict:
    config, index, rows = frozen["config"], frozen["index"], frozen["rows"]
    archive = config["archive"]
    missing = [r["file_name"] for r in rows if r["index_entry"] is None]
    mismatched = []
    for row in rows:
        entry = row["index_entry"]
        if entry is None:
            continue
        problems = []
        if entry["block_offset"] != row["block_offset"]:
            problems.append(f"block_offset {entry['block_offset']} != {row['block_offset']}")
        if entry["packed_size"] != row["packed_size"]:
            problems.append(f"packed_size {entry['packed_size']} != {row['packed_size']}")
        if entry["unpacked_size"] != row["unpacked_size"]:
            problems.append(f"unpacked_size {entry['unpacked_size']} != {row['unpacked_size']}")
        if entry.get("name") != row["archive_name"]:
            problems.append(f"name {entry.get('name')!r} != {row['archive_name']!r}")
        if problems:
            mismatched.append({"file_name": row["file_name"], "problems": problems})
    index_sha = sha256_file(INDEX)
    rule_sha = sha256_file(os.path.join(HERE, "rar5_header.py"))
    gates = {
        "index_path": INDEX,
        "index_entries": len(index["entries"]),
        "index_sha256": index_sha,
        "index_sha256_matches_frozen_config": (
            index_sha == archive.get("index_sha256")),
        "index_archive_size": index.get("archive_size"),
        "index_archive_size_matches_config": (
            index.get("archive_size") == archive["size"]),
        "index_etag": index.get("etag"),
        "index_etag_matches_config": (index.get("etag") == archive["etag"]),
        "candidate_rows": len(rows),
        "candidates_missing_from_index": missing,
        "candidates_mismatching_index": mismatched,
        "rule_module_sha256": rule_sha,
        "rule_module_sha256_matches_preregistration": (
            rule_sha == prereg["rule_module"]["sha256"]),
        "preregistration_pass": prereg["checks"]["preregistration_pass"],
        "max_header_span_allowed": R.MAX_HEADER_SPAN_BYTES,
        "prefix_bytes": R.PREFIX_BYTES,
    }
    gates["candidate_set_matches_index"] = not missing and not mismatched
    gates["endpoint_identity_frozen"] = bool(
        gates["index_archive_size_matches_config"]
        and gates["index_etag_matches_config"]
        and gates["index_sha256_matches_frozen_config"])
    gates["all_preflight_gates_pass"] = bool(
        gates["candidate_set_matches_index"]
        and gates["endpoint_identity_frozen"]
        and gates["rule_module_sha256_matches_preregistration"]
        and gates["preregistration_pass"]
        and len(rows) == 88)
    return gates


def probe_candidate(client: HeaderRangeClient, row: dict) -> dict:
    off = row["block_offset"]
    record = {
        "file_name": row["file_name"],
        "archive_name": row["archive_name"],
        "class_name": row["class_name"],
        # frozen candidate CSV carries the bare 4-digit code in column "t"; the
        # canonical class id used everywhere else in the project is "T<code>".
        "t_code": "T" + row["t"],
        "archive_t_field": row["t"],
        "d_code": row["d"],
        "s_code": row["s"],
        "distance_label": row["distance_label"],
        "s_band_group": row["s_band_group"],
        "frozen_block_offset": off,
        "frozen_packed_size": row["packed_size"],
        "frozen_unpacked_size": row["unpacked_size"],
        "frozen_block_size": (row["index_entry"] or {}).get("block_size"),
        "http_requests": 0,
        "http_bytes": 0,
        "status": None,
        "error": None,
    }
    try:
        prefix, meta1 = client.fetch(off, off + R.PREFIX_BYTES - 1,
                                     f"{row['file_name']}:prefix")
        record["http_requests"] += meta1["attempts"]
        record["http_bytes"] += R.PREFIX_BYTES
        record["prefix_hex"] = prefix.hex()
        stage1 = R.header_span_from_prefix(prefix)
        record.update({
            "head_crc32_stored": stage1["head_crc32"],
            "head_size": stage1["head_size"],
            "head_size_field_bytes": stage1["head_size_field_bytes"],
            "header_span_bytes": stage1["header_span"],
            "head_type": stage1["head_type"],
            "head_type_name": stage1["head_type_name"],
        })
        if stage1["head_type"] != 2:
            record["status"] = "anomaly_not_file_block"
            record["error"] = (f"block at frozen offset is "
                               f"{stage1['head_type_name']}, not FILE")
            return record
        if stage1["header_span"] > R.MAX_HEADER_SPAN_BYTES:
            record["status"] = "anomaly_header_too_large"
            record["error"] = (f"header span {stage1['header_span']} exceeds the "
                               f"{R.MAX_HEADER_SPAN_BYTES}-byte cap; stage-2 read skipped")
            return record

        blob, meta2 = client.fetch(off, off + stage1["header_span"] - 1,
                                   f"{row['file_name']}:header")
        record["http_requests"] += meta2["attempts"]
        record["http_bytes"] += stage1["header_span"]
        record["header_bytes_received"] = len(blob)
        parsed = R.parse_block_header(blob, block_offset=off)
        record["head_crc32_computed"] = parsed["head_crc32_computed"]
        record["head_crc_ok"] = parsed["head_crc_ok"]
        record["header_span_recomputed"] = parsed["header_span_bytes"]
        record["header_flags"] = parsed["header_flags"]
        record["extra_size"] = parsed["extra_size"]
        record["data_size"] = parsed["data_size"]
        record["block_total_bytes"] = parsed["block_total_bytes"]
        record["file_flags"] = parsed["file_flags"]
        record["unpacked_size"] = parsed["unpacked_size"]
        record["data_crc32"] = parsed["data_crc32"]
        record["compression_info"] = parsed["compression_info"]
        record["compression_method"] = parsed["compression_method"]
        record["host_os"] = parsed["host_os"]
        record["header_file_name"] = parsed["file_name"]
        record["extra_area_hex"] = parsed["extra_area_hex"]
        record["extra_records"] = parsed["extra_records"]
        record["extra_area_fully_consumed"] = parsed["extra_area_fully_consumed"]

        mtime = R.resolve_file_mtime(parsed)
        proxy = R.time_proxy_fields(mtime["mtime_epoch"])
        record.update({
            "time_source": mtime["source"],
            "htime_flags": mtime["flags"],
            "htime_raw_hex": mtime["raw_hex"],
            "htime_encoding": mtime["encoding"],
            "htime_raw_value": mtime["raw_value"],
            "time_decode_error": mtime["error"],
            "mtime_utc": proxy["mtime_utc"],
            "file_mtime_proxy_local": proxy["file_mtime_proxy_local"],
            "file_mtime_proxy_date": proxy["file_mtime_proxy_date"],
            "night_key": proxy["night_key"],
        })

        checks = {
            "head_crc_ok": parsed["head_crc_ok"],
            "head_type_is_file": parsed["head_type"] == 2,
            "name_matches_frozen": parsed["file_name"] == row["archive_name"],
            "packed_size_matches_frozen": parsed["data_size"] == row["packed_size"],
            "unpacked_size_matches_frozen": parsed["unpacked_size"] == row["unpacked_size"],
            "block_size_matches_frozen": (
                parsed["block_total_bytes"] == record["frozen_block_size"]),
            "header_span_recomputed_matches": (
                parsed["header_span_bytes"] == stage1["header_span"]),
            "extra_area_fully_consumed": parsed["extra_area_fully_consumed"],
            "time_decoded": proxy["file_mtime_proxy_date"] is not None,
        }
        record["verification"] = checks
        failed = [k for k, v in checks.items() if not v]
        if failed:
            record["status"] = "verification_failed"
            record["error"] = "failed checks: " + ", ".join(failed)
        else:
            record["status"] = "ok"
    except (ProtocolViolation, TimeoutError) as exc:
        record["status"] = "http_error"
        record["error"] = str(exc)
    except R.Rar5ParseError as exc:
        record["status"] = "anomaly_parse_error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:                                # noqa: BLE001
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


CSV_FIELDS = [
    "file_name", "class_name", "t_code", "d_code", "s_code", "distance_label",
    "s_band_group", "frozen_block_offset", "frozen_packed_size",
    "frozen_unpacked_size", "frozen_block_size", "header_span_bytes", "head_size",
    "head_type_name", "head_crc32_stored", "head_crc32_computed", "head_crc_ok",
    "data_size", "unpacked_size", "block_total_bytes", "header_file_name",
    "extra_size", "extra_area_hex", "time_source", "htime_flags", "htime_raw_hex",
    "htime_encoding", "mtime_utc", "file_mtime_proxy_local",
    "file_mtime_proxy_date", "night_key", "http_requests", "http_bytes",
    "status", "error",
]


def write_outputs(result: dict, out_json: str, out_csv: str, partial: bool) -> None:
    """Checkpoint after every candidate so an interrupted sweep keeps its work."""
    blob = dict(result)
    if partial:
        blob["partial"] = True
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, ensure_ascii=False, indent=1)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in result["candidates"]:
            writer.writerow(record)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="probe only the first N candidates (smoke test)")
    parser.add_argument("--out-suffix", default="")
    parser.add_argument("--resume", action="store_true",
                        help="reuse candidates already present in the output JSON")
    args = parser.parse_args()

    out_json = OUT_JSON if not args.out_suffix else OUT_JSON.replace(
        ".json", f"{args.out_suffix}.json")
    out_csv = OUT_CSV if not args.out_suffix else OUT_CSV.replace(
        ".csv", f"{args.out_suffix}.csv")

    prereg = json.load(open(PREREG, encoding="utf-8"))
    frozen = load_frozen_inputs()
    gates = gate_checks(frozen, prereg)
    print(json.dumps(gates, ensure_ascii=False, indent=1), flush=True)

    started = dt.datetime.now(dt.timezone.utc)
    result = {
        "schema": "dronerfa-meta-header1-member-header-probe/v1",
        "task": "META-HEADER1",
        "generated_at": started.isoformat(),
        "scope": ("RAR5 FILE block headers only; no packed payload is requested or read. "
                  "Per-member read budget: 8-byte prefix + header_span <= 128 bytes."),
        "preregistration": {
            "path": PREREG,
            "sha256": sha256_file(PREREG),
            "rule_module_sha256": prereg["rule_module"]["sha256"],
        },
        "preflight_gates": gates,
        "candidates": [],
        "http_log": [],
        "http_stats": {},
    }
    if not gates["all_preflight_gates_pass"]:
        result["aborted"] = "preflight gates failed; no HTTP request was issued"
        write_outputs(result, out_json, out_csv, partial=False)
        print("PREFLIGHT FAILED - aborting before any network call")
        return 3

    archive = frozen["config"]["archive"]
    client = HeaderRangeClient(archive["url"], archive["size"], archive["etag"])

    resumed = {}
    if args.resume and os.path.exists(out_json):
        previous = json.load(open(out_json, encoding="utf-8"))
        for record in previous.get("candidates", []):
            if record.get("status"):
                resumed[record["file_name"]] = record
        print(f"resume: {len(resumed)} candidates already recorded", flush=True)

    # --- endpoint identity + signature, 8 bytes ---------------------------------
    try:
        signature, meta = client.fetch(0, 7, "archive-signature")
        result["preflight_gates"]["signature_probe"] = {
            "status": meta["status"],
            "etag": meta["etag"],
            "content_range": meta["content_range"],
            "bytes": signature.hex(),
            "is_rar5_signature": signature == R.RAR5_SIGNATURE,
        }
    except Exception as exc:                                # noqa: BLE001
        result["preflight_gates"]["signature_probe"] = {"error": str(exc)}
        result["aborted"] = (f"endpoint identity probe failed ({exc}); "
                             "no candidate header was read")
        write_outputs(result, out_json, out_csv, partial=False)
        print("SIGNATURE PROBE FAILED - aborting")
        return 4
    if not result["preflight_gates"]["signature_probe"]["is_rar5_signature"]:
        result["aborted"] = "remote prefix is not the RAR5 signature; endpoint identity changed"
        write_outputs(result, out_json, out_csv, partial=False)
        print("SIGNATURE MISMATCH - aborting")
        return 5

    rows = frozen["rows"]
    if args.limit:
        rows = rows[:args.limit]
    consecutive_429 = 0
    total_429 = 0
    aborted = None
    for number, row in enumerate(rows, 1):
        if row["file_name"] in resumed:
            print(f"[{number}/{len(rows)}] {row['file_name']} (resumed, no request)",
                  flush=True)
            result["candidates"].append(resumed[row["file_name"]])
            continue
        print(f"[{number}/{len(rows)}] {row['file_name']} @ {row['block_offset']}",
              flush=True)
        record = probe_candidate(client, row)
        result["candidates"].append(record)
        status_429 = [e for e in client.log
                      if e.get("status") == 429 and e["label"].startswith(row["file_name"])]
        if status_429:
            consecutive_429 += 1
            total_429 += len(status_429)
        else:
            consecutive_429 = 0
        print(f"    -> {record['status']} {record.get('file_mtime_proxy_date')} "
              f"src={record.get('time_source')} span={record.get('header_span_bytes')} "
              f"{record.get('error') or ''}", flush=True)
        result["http_log"] = client.log
        result["totals"] = {
            "candidates_recorded": len(result["candidates"]),
            "http_bytes": client.bytes_transferred,
        }
        write_outputs(result, out_json, out_csv, partial=True)
        if consecutive_429 >= MAX_CONSECUTIVE_429 or total_429 > MAX_TOTAL_429:
            aborted = (f"aborted after {total_429} HTTP 429 responses "
                       f"(consecutive={consecutive_429}); remaining candidates not probed")
            print("    " + aborted, flush=True)
            break
        time.sleep(POLITE_DELAY)

    finished = dt.datetime.now(dt.timezone.utc)
    result.pop("partial", None)
    result["aborted"] = aborted
    result["http_log"] = client.log
    result["http_stats"] = {
        "requests": client.requests,
        "bytes_transferred": client.bytes_transferred,
        "distinct_statuses": sorted({e.get("status") for e in client.log},
                                    key=lambda x: (x is None, x)),
        "status_counts": {str(s): sum(1 for e in client.log if e.get("status") == s)
                          for s in sorted({e.get("status") for e in client.log},
                                          key=lambda x: (x is None, x))},
        "retries": sum(1 for e in client.log if e["attempt"] > 1),
        "elapsed_seconds": (finished - started).total_seconds(),
        "payload_bytes_read": 0,
    }
    result["http_stats"]["bytes_per_candidate_avg"] = (
        round(client.bytes_transferred / max(len(result["candidates"]), 1), 1))
    result["totals"] = {
        "candidates_probed": len(result["candidates"]),
        "candidates_freshly_probed": len(result["candidates"]) - len(resumed),
        "status_counts": {},
        "http_bytes": client.bytes_transferred,
    }
    for record in result["candidates"]:
        key = record["status"]
        result["totals"]["status_counts"][key] = (
            result["totals"]["status_counts"].get(key, 0) + 1)

    written = [out_json, out_csv]
    write_outputs(result, out_json, out_csv, partial=False)
    if not args.out_suffix:
        written.append(OUT_HTTP)
        with open(OUT_HTTP, "w", encoding="utf-8") as fh:
            json.dump({"schema": "dronerfa-meta-header1-http-log/v1",
                       "generated_at": finished.isoformat(),
                       "archive_url": archive["url"],
                       "expected_total": archive["size"],
                       "expected_etag": archive["etag"],
                       "stats": result["http_stats"],
                       "requests": client.log}, fh, ensure_ascii=False, indent=1)

    print(json.dumps({"totals": result["totals"], "http_stats": result["http_stats"],
                      "aborted": aborted, "wrote": written},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
