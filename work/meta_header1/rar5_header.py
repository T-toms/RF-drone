# -*- coding: utf-8 -*-
"""META-HEADER1: shared, offline RAR5 header parser plus the *pre-registered*
time-decoding rules.

This module contains no network code.  It is imported by
``calibrate_htime_rules.py`` (offline, locks the rules against the 14 local
members whose MATLAB ``Created on`` stamp is already known) and by
``probe_member_headers.py`` (network, applies the very same functions to the
88 un-downloaded candidates).

Nothing in here may be tuned per candidate.  The rules below were frozen from
the 14 local members before the first HTTP request was issued; see
``htime-decode-preregistration-20260925.json``.
"""

from __future__ import annotations

import datetime as dt
import zlib

RAR5_SIGNATURE = bytes.fromhex("526172211a070100")

BLOCK_TYPES = {1: "MAIN", 2: "FILE", 3: "SERVICE", 4: "ENCRYPTION", 5: "ENDARC"}

HEAD_FLAG_EXTRA = 0x0001
HEAD_FLAG_DATA = 0x0002

FILE_FLAG_MTIME = 0x0002
FILE_FLAG_CRC32 = 0x0004

FHEXTRA_HTIME = 3
EXTRA_TYPES = {1: "CRYPT", 2: "HASH", 3: "HTIME", 4: "VERSION", 5: "REDIR",
               6: "UOWNER", 7: "SUBDATA"}

# --- pre-registered FHEXTRA_HTIME flag bits (unrar arcread.cpp / unpack.hpp) --
HTIME_UNIXTIME = 0x01   # bit0: 1 => uint32 unix seconds, 0 => uint64 FILETIME
HTIME_MTIME = 0x02      # bit1: mtime present
HTIME_CTIME = 0x04      # bit2: ctime present
HTIME_ATIME = 0x08      # bit3: atime present
HTIME_UNIX_NS = 0x10    # bit4: 4-byte nanosecond remainder follows

FILETIME_EPOCH_OFFSET = 11644473600  # seconds between 1601-01-01 and 1970-01-01

# --- pre-registered timezone rule -------------------------------------------
# RULE TZ-1: the stored instant is UTC.  On the 14 local calibration members the
# MATLAB v7.3 header string "Created on: ..." equals decoded_instant + 8 h
# (Asia/Shanghai, UTC+8) with a residual of -2..-1 s.  Therefore the proxy local
# time is ALWAYS decoded_instant + 8 h -- no per-candidate offset is allowed.
LOCAL_UTC_OFFSET_SECONDS = 8 * 3600
LOCAL_TZ = dt.timezone(dt.timedelta(seconds=LOCAL_UTC_OFFSET_SECONDS))

# --- pre-registered night-key rule (identical to META-SPLIT-1 section 3.3) ---
NIGHT_BOUNDARY_HOURS = 6

# --- pre-registered read budget ---------------------------------------------
MAX_HEADER_SPAN_BYTES = 128     # 4 + len(HEAD_SIZE vint) + HEAD_SIZE
PREFIX_BYTES = 8                # enough for HEAD_CRC + a 4-byte HEAD_SIZE vint


class Rar5ParseError(ValueError):
    pass


def decode_vint(buf: bytes, pos: int) -> tuple[int, int]:
    """RAR5 variable-length integer.  Returns (value, new_pos)."""
    value = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise Rar5ParseError("vint runs past the end of the buffer")
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            break
        shift += 7
        if shift > 70:
            raise Rar5ParseError("vint longer than 10 bytes")
    return value, pos


def header_span_from_prefix(prefix: bytes) -> dict:
    """Step 1 of the two-stage read: HEAD_CRC + HEAD_SIZE vint + HEAD_TYPE.

    ``prefix`` must start at the block start (the frozen ``block_offset``).
    Returns the exact number of bytes (``header_span``) that step 2 must read.
    """
    if len(prefix) < 5:
        raise Rar5ParseError(f"prefix too short for HEAD_CRC + HEAD_SIZE: {len(prefix)}")
    head_crc = int.from_bytes(prefix[0:4], "little")
    head_size, pos = decode_vint(prefix, 4)
    head_size_len = pos - 4
    head_type, pos = decode_vint(prefix, pos)
    return {
        "head_crc32": format(head_crc, "08x"),
        "head_size": head_size,
        "head_size_field_bytes": head_size_len,
        # RAR5 block total = HEAD_CRC(4) + len(HEAD_SIZE vint) + HEAD_SIZE + DATA_SIZE
        "header_span": 4 + head_size_len + head_size,
        "head_type": head_type,
        "head_type_name": BLOCK_TYPES.get(head_type, "UNKNOWN"),
    }


def parse_block_header(blob: bytes, block_offset: int = 0) -> dict:
    """Full RAR5 block-header parse of ``blob`` (which must begin at the block).

    RULE HDR-1: HEAD_CRC is CRC32 over the bytes from the first HEAD_SIZE byte
    through the end of the header (i.e. blob[4 : header_span]).
    """
    if len(blob) < 5:
        raise Rar5ParseError(f"block buffer too short: {len(blob)}")
    head_crc = int.from_bytes(blob[0:4], "little")
    pos = 4
    head_size, pos = decode_vint(blob, pos)
    head_size_len = pos - 4
    header_span = 4 + head_size_len + head_size
    if len(blob) < header_span:
        raise Rar5ParseError(
            f"need {header_span} header bytes, only {len(blob)} available")
    computed_crc = zlib.crc32(blob[4:header_span]) & 0xFFFFFFFF
    info = {
        "block_offset": block_offset,
        "head_crc32_stored": format(head_crc, "08x"),
        "head_crc32_computed": format(computed_crc, "08x"),
        "head_crc_ok": computed_crc == head_crc,
        "head_size": head_size,
        "head_size_field_bytes": head_size_len,
        "header_span_bytes": header_span,
    }
    head_type, pos = decode_vint(blob, pos)
    info["head_type"] = head_type
    info["head_type_name"] = BLOCK_TYPES.get(head_type, "UNKNOWN")

    head_flags, pos = decode_vint(blob, pos)
    info["header_flags"] = head_flags
    extra_size = 0
    data_size = 0
    if head_flags & HEAD_FLAG_EXTRA:
        extra_size, pos = decode_vint(blob, pos)
    if head_flags & HEAD_FLAG_DATA:
        data_size, pos = decode_vint(blob, pos)
    info["extra_size"] = extra_size
    info["data_size"] = data_size
    info["block_total_bytes"] = header_span + data_size

    if head_type != 2:
        # MAIN / SERVICE / ENDARC: no file record to decode.
        info["is_file_block"] = False
        return info
    info["is_file_block"] = True

    file_flags, pos = decode_vint(blob, pos)
    unpacked_size, pos = decode_vint(blob, pos)
    attributes, pos = decode_vint(blob, pos)
    info["file_flags"] = file_flags
    info["unpacked_size"] = unpacked_size
    info["attributes"] = attributes
    info["file_head_mtime_epoch"] = None
    if file_flags & FILE_FLAG_MTIME:
        info["file_head_mtime_epoch"] = int.from_bytes(blob[pos:pos + 4], "little")
        pos += 4
    info["data_crc32"] = None
    if file_flags & FILE_FLAG_CRC32:
        info["data_crc32"] = format(int.from_bytes(blob[pos:pos + 4], "little"), "08x")
        pos += 4
    compression_info, pos = decode_vint(blob, pos)
    host_os, pos = decode_vint(blob, pos)
    name_len, pos = decode_vint(blob, pos)
    info["compression_info"] = compression_info
    info["compression_method"] = compression_info & 0x3F
    info["host_os"] = host_os
    info["name_len"] = name_len
    info["file_name"] = blob[pos:pos + name_len].decode("utf-8", "replace")
    pos += name_len
    extra_area = blob[pos:pos + extra_size]
    info["extra_area_hex"] = extra_area.hex()
    info["extra_records"] = parse_extra_area(extra_area)
    info["extra_area_fully_consumed"] = (
        sum(1 for r in info["extra_records"] if "error" not in r) > 0
        and all("error" not in r for r in info["extra_records"])
        and sum(_extra_record_total(r, extra_area) for r in info["extra_records"]) == len(extra_area)
    ) if extra_size else True
    info["header_consumed_bytes"] = pos + extra_size
    return info


def _extra_record_total(record: dict, extra: bytes) -> int:
    """Byte length of one parsed extra record, including its size vint."""
    size_len = 1
    value = record["record_size"]
    while value >= 0x80:
        value >>= 7
        size_len += 1
    return size_len + record["record_size"]


def parse_extra_area(extra: bytes) -> list:
    """Decode the RAR5 extra area into records (raw, undecided)."""
    out = []
    pos = 0
    while pos < len(extra):
        start = pos
        try:
            record_size, p = decode_vint(extra, pos)
        except Rar5ParseError as exc:
            out.append({"error": repr(exc), "at": pos})
            break
        size_len = p - pos
        # RAR5: the record size field counts the type field and the payload, but
        # NOT the size field itself, so the record ends at start+size_len+size.
        record_end = start + size_len + record_size
        if record_size < 1 or record_end > len(extra):
            out.append({"error": "extra record size out of range",
                        "at": pos, "declared_size": record_size})
            break
        try:
            record_type, data_start = decode_vint(extra, p)
        except Rar5ParseError as exc:
            out.append({"error": repr(exc), "at": pos})
            break
        data = extra[data_start:record_end]
        out.append({
            "type": record_type,
            "type_name": EXTRA_TYPES.get(record_type, "UNKNOWN"),
            "record_size": record_size,
            "data_hex": data.hex(),
        })
        pos = record_end
    return out


def decode_htime_data(data: bytes) -> dict:
    """RULE HTIME-3 / HTIME-4: decode one FHEXTRA_HTIME payload.

    ``data[0]`` is the flags byte.  Bit0 alone selects the encoding:
        bit0 == 1 -> uint32 LE unix seconds at data[1:5]
        bit0 == 0 -> uint64 LE Windows FILETIME at data[1:9]
    No fallback interpretation is attempted: a short record is an error.
    """
    if not data:
        return {"ok": False, "error": "empty HTIME payload"}
    flags = data[0]
    rec = {"ok": False, "flags": flags, "raw_hex": data.hex(),
           "unix_time_format": bool(flags & HTIME_UNIXTIME),
           "mtime_present": bool(flags & HTIME_MTIME),
           "ctime_present": bool(flags & HTIME_CTIME),
           "atime_present": bool(flags & HTIME_ATIME),
           "nano_present": bool(flags & HTIME_UNIX_NS)}
    if flags & HTIME_UNIXTIME:
        if len(data) < 5:
            rec["error"] = f"unix-format HTIME needs 5 bytes, has {len(data)}"
            return rec
        rec["ok"] = True
        rec["encoding"] = "unix_uint32"
        rec["raw_value"] = int.from_bytes(data[1:5], "little")
        rec["mtime_epoch"] = rec["raw_value"]
    else:
        if len(data) < 9:
            rec["error"] = f"FILETIME HTIME needs 9 bytes, has {len(data)}"
            return rec
        rec["ok"] = True
        rec["encoding"] = "windows_filetime_uint64"
        rec["raw_value"] = int.from_bytes(data[1:9], "little")
        rec["mtime_epoch"] = rec["raw_value"] / 10_000_000 - FILETIME_EPOCH_OFFSET
    return rec


def resolve_file_mtime(parsed: dict) -> dict:
    """RULE HTIME-1 / HTIME-2 / HTIME-5: pick the one authoritative time.

    Priority: first HTIME extra record whose mtime bit (0x02) is set; else the
    first HTIME record at all; else the FILE-header MTIME field; else nothing.
    """
    out = {"source": None, "record_index": None, "flags": None,
           "raw_hex": None, "encoding": None, "raw_value": None,
           "mtime_epoch": None, "error": None}
    records = [r for r in parsed.get("extra_records", []) if r.get("type") == FHEXTRA_HTIME]
    chosen = None
    chosen_index = None
    for index, rec in enumerate(records):
        data = bytes.fromhex(rec["data_hex"])
        if data and (data[0] & HTIME_MTIME):
            chosen, chosen_index = rec, index
            break
    if chosen is None and records:
        chosen, chosen_index = records[0], 0
    if chosen is not None:
        decoded = decode_htime_data(bytes.fromhex(chosen["data_hex"]))
        out.update({
            "source": "fextra_htime",
            "record_index": chosen_index,
            "flags": decoded.get("flags"),
            "raw_hex": chosen["data_hex"],
            "encoding": decoded.get("encoding"),
            "raw_value": decoded.get("raw_value"),
            "mtime_epoch": decoded.get("mtime_epoch"),
            "error": decoded.get("error"),
            "mtime_bit_set": decoded.get("mtime_present"),
        })
        return out
    if parsed.get("file_head_mtime_epoch") is not None:
        out.update({"source": "file_head_mtime_field",
                    "record_index": None,
                    "flags": None,
                    "raw_hex": None,
                    "encoding": "unix_uint32",
                    "raw_value": parsed["file_head_mtime_epoch"],
                    "mtime_epoch": float(parsed["file_head_mtime_epoch"])})
        return out
    out["error"] = "no FHEXTRA_HTIME record and FILE_FLAG_MTIME not set"
    return out


def time_proxy_fields(mtime_epoch) -> dict:
    """RULE TZ-1 / TZ-2: UTC instant -> pre-registered local proxy fields."""
    if mtime_epoch is None:
        return {"mtime_utc": None, "file_mtime_proxy_local": None,
                "file_mtime_proxy_date": None, "night_key": None}
    utc = dt.datetime.fromtimestamp(mtime_epoch, dt.timezone.utc)
    local = utc.astimezone(LOCAL_TZ)
    night = (local - dt.timedelta(hours=NIGHT_BOUNDARY_HOURS)).date()
    return {
        "mtime_utc": utc.replace(microsecond=0).isoformat(),
        "file_mtime_proxy_local": local.replace(microsecond=0).isoformat(),
        "file_mtime_proxy_local_wallclock": local.replace(
            microsecond=0, tzinfo=None).isoformat(),
        "file_mtime_proxy_date": local.date().isoformat(),
        "night_key": "N" + night.strftime("%Y%m%d"),
    }
