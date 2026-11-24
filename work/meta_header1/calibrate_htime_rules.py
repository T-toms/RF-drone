# -*- coding: utf-8 -*-
"""META-HEADER1 / step 1 (offline): lock the RAR5 header + HTIME decoding rules
against the 14 local members whose MATLAB ``Created on`` stamp is already known.

No network access.  Reads only:
  DroneRFa_pilot.rar                                                  12 members
  G:\\...\\downloads\\T0011_D00_S0111.mat.rar.part                     1 member
  G:\\...\\downloads\\T1000_D00_S0111.mat.rar.part                     1 member
and the already-produced work/meta_split1/mat_headers_104.json.

Writes work/meta_header1/htime-decode-preregistration-20260925.json, which is
the frozen statement of the rules that probe_member_headers.py must obey.  It
is produced BEFORE any HTTP request, so the candidate dates cannot influence
the decoder.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import rar5_header as R  # noqa: E402

PILOT = os.path.join(WORKSPACE, "DroneRFa_pilot.rar")
G_DOWNLOADS = r"G:\DroneRFa数据集\download_records_12models_20260922\downloads"
PARTS = [
    os.path.join(G_DOWNLOADS, "T0011_D00_S0111.mat.rar.part"),
    os.path.join(G_DOWNLOADS, "T1000_D00_S0111.mat.rar.part"),
]
MAT_HEADERS = os.path.join(WORKSPACE, "work", "meta_split1", "mat_headers_104.json")
OUT = os.path.join(HERE, "htime-decode-preregistration-20260925.json")

TOLERANCE_SECONDS = 2.0


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_local_block(path: str, offset: int) -> dict:
    """Same two-stage read the network probe performs, but from a local file."""
    with open(path, "rb") as fh:
        fh.seek(offset)
        prefix = fh.read(R.PREFIX_BYTES)
    stage1 = R.header_span_from_prefix(prefix)
    if stage1["header_span"] > R.MAX_HEADER_SPAN_BYTES:
        raise R.Rar5ParseError(f"header span {stage1['header_span']} > cap")
    with open(path, "rb") as fh:
        fh.seek(offset)
        blob = fh.read(stage1["header_span"])
    parsed = R.parse_block_header(blob, block_offset=offset)
    parsed["read_bytes"] = len(prefix) + len(blob)
    return parsed


def local_members() -> list:
    """Walk the local archives and return (source, block_offset) for every FILE block."""
    found = []
    pilot_path = PILOT
    with open(pilot_path, "rb") as fh:
        if fh.read(8) != R.RAR5_SIGNATURE:
            raise SystemExit("pilot archive is not RAR5")
    # pilot: walk block chain from offset 8 (signature already consumed)
    offset = 8
    size = os.path.getsize(pilot_path)
    while offset + 8 < size:
        block = read_local_block(pilot_path, offset)
        if block["head_type"] == 2:
            found.append(("DroneRFa_pilot.rar", offset, block))
        if block["head_type"] == 5:
            break
        offset += block["block_total_bytes"]
    for part in PARTS:
        if not os.path.exists(part):
            found.append(("MISSING:" + part, None, None))
            continue
        # the .part file itself starts with the RAR5 signature; the FILE block
        # (or a MAIN block, then the FILE block) follows.
        with open(part, "rb") as fh:
            head = fh.read(16)
        pos = 8 if head[:8] == R.RAR5_SIGNATURE else 0
        while pos < os.path.getsize(part):
            block = read_local_block(part, pos)
            if block["head_type"] == 2:
                found.append((os.path.basename(part), pos, block))
                break
            if block["head_type"] == 5:
                break
            pos += block["block_total_bytes"]
    return found


def main() -> int:
    heads = json.load(open(MAT_HEADERS, encoding="utf-8"))
    by_name = {rec["file_name"]: rec for rec in heads["records"]}

    members = local_members()
    rows = []
    for source, offset, block in members:
        if block is None:
            rows.append({"source": source, "error": "local archive not found"})
            continue
        short = os.path.basename(block.get("file_name", ""))
        mat = by_name.get(short)
        mtime = R.resolve_file_mtime(block)
        proxy = R.time_proxy_fields(mtime["mtime_epoch"])
        mat_created = mat.get("created_iso") if mat else None
        residual = None
        if mat_created and proxy["file_mtime_proxy_local_wallclock"]:
            residual = (dt.datetime.fromisoformat(mat_created.replace(" ", "T"))
                        - dt.datetime.fromisoformat(
                            proxy["file_mtime_proxy_local_wallclock"])
                        ).total_seconds()
        rows.append({
            "source": source,
            "block_offset": offset,
            "file_name": short,
            "head_type_name": block["head_type_name"],
            "head_size": block["head_size"],
            "header_span_bytes": block["header_span_bytes"],
            "head_crc32_stored": block["head_crc32_stored"],
            "head_crc32_computed": block["head_crc32_computed"],
            "head_crc_ok": block["head_crc_ok"],
            "file_flags": block.get("file_flags"),
            "file_head_mtime_set": bool((block.get("file_flags") or 0) & R.FILE_FLAG_MTIME),
            "extra_size": block.get("extra_size"),
            "extra_records": block.get("extra_records"),
            "data_size": block.get("data_size"),
            "unpacked_size": block.get("unpacked_size"),
            "resolved_time_source": mtime["source"],
            "resolved_time_flags": mtime["flags"],
            "resolved_time_encoding": mtime["encoding"],
            "resolved_time_raw_hex": mtime["raw_hex"],
            "resolved_time_error": mtime["error"],
            "mtime_utc": proxy["mtime_utc"],
            "file_mtime_proxy_local": proxy["file_mtime_proxy_local"],
            "file_mtime_proxy_date": proxy["file_mtime_proxy_date"],
            "night_key": proxy["night_key"],
            "extra_area_fully_consumed": block.get("extra_area_fully_consumed"),
            "mat_created_local": mat_created,
            "residual_seconds": residual,
            "within_tolerance": (residual is not None
                                 and abs(residual) <= TOLERANCE_SECONDS),
        })

    usable = [r for r in rows if r.get("residual_seconds") is not None]
    crc_ok = [r for r in rows if r.get("head_crc_ok")]
    residuals = sorted({r["residual_seconds"] for r in usable})
    payload = {
        "schema": "dronerfa-meta-header1-htime-preregistration/v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": ("Freeze the RAR5 header and FHEXTRA_HTIME decoding rules on the 14 "
                    "local members with known MATLAB 'Created on' stamps, BEFORE issuing "
                    "any HTTP request for the 88 un-downloaded candidates."),
        "network_access": "none",
        "rule_module": {
            "path": os.path.join(HERE, "rar5_header.py"),
            "sha256": sha256_file(os.path.join(HERE, "rar5_header.py")),
        },
        "rules": {
            "HDR-1": ("HEAD_CRC (4 bytes LE) equals CRC32 over blob[4:4+len(HEAD_SIZE vint)+HEAD_SIZE]."),
            "HDR-2": ("header_span = 4 + len(HEAD_SIZE vint) + HEAD_SIZE; the frozen "
                      "block_offset must point at a FILE block (HEAD_TYPE == 2)."),
            "HTIME-1": "Use the FHEXTRA_HTIME (extra type 3) record(s) inside the FILE header extra area.",
            "HTIME-2": ("Prefer the first HTIME record whose flags byte has bit1 (0x02, mtime) set; "
                        "otherwise the first HTIME record."),
            "HTIME-3": ("Encoding is chosen by bit0 (0x01) ALONE: bit0 set -> uint32 LE unix seconds "
                        "at data[1:5]; bit0 clear -> uint64 LE Windows FILETIME at data[1:9], "
                        "epoch = value/1e7 - 11644473600."),
            "HTIME-4": "A record too short for its declared encoding is an error; no alternative interpretation.",
            "HTIME-5": ("If no HTIME record exists, fall back to the FILE-header MTIME field "
                        "(unix uint32) when FILE_FLAGS bit1 (0x0002) is set; otherwise report null."),
            "TZ-1": ("The stored instant is UTC. file_mtime_proxy_local = instant + 8 h "
                     "(Asia/Shanghai), always, with no per-candidate adjustment."),
            "TZ-2": ("file_mtime_proxy_date = date of the local proxy time; night_key = "
                     "date of (local proxy time - 6 h), matching META-SPLIT-1."),
            "BUDGET-1": ("Network probe may read at most 128 header bytes per member "
                         f"(MAX_HEADER_SPAN_BYTES={R.MAX_HEADER_SPAN_BYTES}) plus an "
                         f"{R.PREFIX_BYTES}-byte repeated prefix, and must never read a byte at or "
                         "beyond block_offset + header_span."),
        },
        "tolerance_seconds": TOLERANCE_SECONDS,
        "calibration_members": rows,
        "checks": {
            "members_parsed": len(rows),
            "head_crc_ok_all": len(crc_ok) == len(rows) and bool(rows),
            "mat_header_matched": len(usable),
            "within_tolerance": sum(1 for r in usable if r["within_tolerance"]),
            "residual_seconds_set": residuals,
            "time_source_set": sorted({r.get("resolved_time_source") for r in rows}),
            "htime_flags_set": sorted({r.get("resolved_time_flags") for r in rows},
                                      key=lambda x: (x is None, x)),
            "time_encoding_set": sorted({str(r.get("resolved_time_encoding")) for r in rows}),
            "header_span_max": max(r["header_span_bytes"] for r in rows),
        },
    }
    payload["checks"]["preregistration_pass"] = (
        payload["checks"]["head_crc_ok_all"]
        and payload["checks"]["mat_header_matched"] == len(rows)
        and payload["checks"]["within_tolerance"] == len(usable)
    )

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    summary = {
        "members": len(rows),
        "checks": payload["checks"],
        "wrote": OUT,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if payload["checks"]["preregistration_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
