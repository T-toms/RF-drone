# -*- coding: utf-8 -*-
"""META-SPLIT-1 / step 3b: recover per-member acquisition timestamps from local
RAR5 headers that are already on disk.  No network access.

Why this matters: the official DroneRFa archive stores each .mat as an ordinary
RAR5 file member.  RAR5 keeps the file time either in the FILE_HEAD flags or in
the FHEXTRA_HTIME extra-area record.  If that timestamp equals the MATLAB
"Created on" value of the same file, then the timestamp is a real per-file
acquisition-time key -- and it is present in the *header* of members that have
never been downloaded, which makes it the cheapest possible supplementary
evidence for the session-independence question.

Sources read (all local, all pre-existing):
  DroneRFa_pilot.rar                                 full RAR5 archive, 12 members
  G:\\...\\downloads\\T0011_D00_S0111.mat.rar.part   leftover partial download
  G:\\...\\downloads\\T1000_D00_S0111.mat.rar.part   leftover partial download

Output: work/meta_split1/rar_header_times.json
"""

from __future__ import annotations

import datetime as dt
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
RAR5_SIG = bytes([0x52, 0x61, 0x72, 0x21, 0x1A, 0x07, 0x01, 0x00])

BLOCK_TYPES = {1: "MAIN", 2: "FILE", 3: "SERVICE", 4: "ENCRYPTION", 5: "ENDARC"}
EXTRA_TYPES = {1: "CRYPT", 2: "HASH", 3: "HTIME", 4: "VERSION", 5: "REDIR",
               6: "UOWNER", 7: "SUBDATA"}
FILE_FLAG_MTIME = 0x0002
FILE_FLAG_CRC32 = 0x0004
HEAD_FLAG_EXTRA = 0x0001
HEAD_FLAG_DATA = 0x0002


def vint(buf: bytes, pos: int):
    value = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
        if shift > 70:
            raise ValueError("vint too long")
    return value, pos


def parse_extra(extra: bytes) -> list:
    """Decode the RAR5 extra area.  FHEXTRA_HTIME flag bits follow unrar's
    unpack.hpp: 0x01 unix-time format, 0x02 mtime, 0x04 ctime, 0x08 atime,
    0x10 nanosecond part present."""
    out = []
    pos = 0
    while pos < len(extra):
        try:
            size, p = vint(extra, pos)
            size_len = p - pos
            rtype, p2 = vint(extra, p)
            data = extra[p2:pos + size_len + size]
            rec = {"type": rtype, "type_name": EXTRA_TYPES.get(rtype, "UNKNOWN"),
                   "field_size": size, "raw_hex": data.hex()}
            if rtype == 3 and len(data) >= 1:      # FHEXTRA_HTIME
                flags = data[0]
                rec["htime_flags"] = flags
                # Verified empirically against the MATLAB "Created on" strings of the
                # 12 pilot members: bit0 clear => Windows FILETIME (100 ns since
                # 1601-01-01 UTC), and the stored value equals the MATLAB local
                # timestamp minus 8 h (Asia/Shanghai) to within 1-2 seconds.
                unix_fmt = bool(flags & 0x01)
                if unix_fmt:
                    secs = struct.unpack_from("<I", data, 1)[0]
                    rec["mtime_epoch"] = secs
                    rec["mtime_utc"] = dt.datetime.fromtimestamp(
                        secs, dt.timezone.utc).isoformat()
                elif len(data) >= 9:
                    ft = struct.unpack_from("<Q", data, 1)[0]
                    secs = ft / 10_000_000 - 11644473600
                    rec["mtime_filetime"] = ft
                    rec["mtime_epoch"] = secs
                    rec["mtime_utc"] = dt.datetime.fromtimestamp(
                        secs, dt.timezone.utc).isoformat()
                    rec["mtime_local_utc8"] = dt.datetime.fromtimestamp(
                        secs + 8 * 3600, dt.timezone.utc).isoformat()
            out.append(rec)
            pos += size_len + size
        except Exception as exc:
            out.append({"error": repr(exc), "at": pos})
            break
    return out


def read_block_header(fh, offset: int) -> dict:
    fh.seek(offset)
    buf = fh.read(8192)
    head_crc = struct.unpack_from("<I", buf, 0)[0]
    pos = 4
    hsz_start = pos
    head_size, pos = vint(buf, pos)
    hsz_len = pos - hsz_start
    head_start = pos
    btype, p = vint(buf, pos)
    info = {"block_offset": offset, "head_crc32": format(head_crc, "08x"),
            "head_size": head_size, "head_size_field_bytes": hsz_len,
            "head_type": btype,
            "head_type_name": BLOCK_TYPES.get(btype, "UNKNOWN")}
    hflags, p2 = vint(buf, p)
    info["header_flags"] = hflags
    if hflags & HEAD_FLAG_EXTRA:
        extra_size, p2 = vint(buf, p2)
        info["extra_size"] = extra_size
    else:
        extra_size = 0
    if hflags & HEAD_FLAG_DATA:
        data_size, p2 = vint(buf, p2)
        info["data_size"] = data_size
    else:
        data_size = 0
    if btype == 2:
        file_flags, q = vint(buf, p2)
        unp_size, q = vint(buf, q)
        attrs, q = vint(buf, q)
        info["file_flags"] = file_flags
        info["unpacked_size"] = unp_size
        info["attributes"] = attrs
        if file_flags & FILE_FLAG_MTIME:
            secs = struct.unpack_from("<I", buf, q)[0]
            q += 4
            info["mtime_epoch"] = secs
            info["mtime_utc"] = dt.datetime.fromtimestamp(secs, dt.timezone.utc).isoformat()
        if file_flags & FILE_FLAG_CRC32:
            info["data_crc32"] = format(struct.unpack_from("<I", buf, q)[0], "08x")
            q += 4
        comp_info, q = vint(buf, q)
        host_os, q = vint(buf, q)
        name_len, q = vint(buf, q)
        info["compression_info"] = comp_info
        info["compression_method"] = comp_info & 0x3F
        info["host_os"] = host_os
        info["file_name"] = buf[q:q + name_len].decode("utf-8", "replace")
        if extra_size:
            info["extra_records"] = parse_extra(buf[q + name_len:q + name_len + extra_size])
    # RAR5: block total = 4 (crc) + len(head_size vint) + head_size + data_size
    info["block_total_bytes"] = 4 + hsz_len + head_size + data_size
    return info


def walk_archive(path: str, limit: int = 40) -> dict:
    out = {"path": path, "exists": os.path.exists(path)}
    if not out["exists"]:
        return out
    out["size_bytes"] = os.path.getsize(path)
    with open(path, "rb") as fh:
        if fh.read(8) != RAR5_SIG:
            out["has_rar5_signature"] = False
            return out
        out["has_rar5_signature"] = True
        offset = 8
        members = []
        blocks = []
        while len(blocks) < limit and offset + 8 < out["size_bytes"]:
            info = read_block_header(fh, offset)
            blocks.append(info)
            if info["head_type"] == 2:
                members.append(info)
            if info["head_type"] == 5:
                break
            offset += info["block_total_bytes"]
        out["blocks_scanned"] = len(blocks)
        out["blocks"] = blocks
        out["members"] = members
    return out


def dump_partial(path: str) -> dict:
    out = {"path": path, "exists": os.path.exists(path)}
    if not out["exists"]:
        return out
    out["size_bytes"] = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(64)
    out["first_64_bytes_hex"] = head.hex()
    out["has_rar5_signature_at_0"] = head[:8] == RAR5_SIG
    sig_at = head.find(RAR5_SIG)
    out["rar5_signature_offset_in_first_64"] = sig_at
    # Is it a bare FILE block?  Then byte 4 starts a vint head_size and the
    # header should parse cleanly.
    try:
        head_crc = struct.unpack_from("<I", head, 0)[0]
        pos = 4
        head_size, pos = vint(head, pos)
        btype, p = vint(head, pos)
        out["block0_head_crc32"] = format(head_crc, "08x")
        out["block0_head_size"] = head_size
        out["block0_head_type"] = btype
        out["block0_head_type_name"] = BLOCK_TYPES.get(btype, "UNKNOWN")
        if btype == 2:
            hflags, p2 = vint(head, p)
            extra_size = 0
            if hflags & HEAD_FLAG_EXTRA:
                extra_size, p2 = vint(head, p2)
            if hflags & HEAD_FLAG_DATA:
                _data_size, p2 = vint(head, p2)
            file_flags, q = vint(head, p2)
            unp, q = vint(head, q)
            attrs, q = vint(head, q)
            out["block0_file_flags"] = file_flags
            out["block0_unpacked_size"] = unp
            if file_flags & FILE_FLAG_CRC32:
                q += 4
            ci, q = vint(head, q)
            host, q = vint(head, q)
            nl, q = vint(head, q)
            out["block0_file_name"] = head[q:q + nl].decode("utf-8", "replace")
            out["block0_extra_size"] = extra_size
    except Exception as exc:
        out["block0_parse_error"] = repr(exc)
    return out


def main() -> int:
    g_dl = r"G:\DroneRFa数据集\download_records_12models_20260922\downloads"
    pilot = walk_archive(os.path.join(WORKSPACE, "DroneRFa_pilot.rar"))
    parts = [walk_archive(os.path.join(g_dl, "T0011_D00_S0111.mat.rar.part"), limit=3),
             walk_archive(os.path.join(g_dl, "T1000_D00_S0111.mat.rar.part"), limit=3)]

    heads = json.load(open(os.path.join(HERE, "mat_headers_104.json"), encoding="utf-8"))
    by_name = {r["file_name"]: r for r in heads["records"]}
    all_members = [("pilot_archive", m) for m in pilot.get("members", [])]
    for part in parts:
        for m in part.get("members", []):
            all_members.append(("partial_download", m))
    checks = []
    for source, m in all_members:
        short = os.path.basename(m.get("file_name", ""))
        rec = by_name.get(short)
        hrec = next((r for r in m.get("extra_records", []) if r.get("type_name") == "HTIME"), None)
        rar_local = (hrec or {}).get("mtime_local_utc8")
        mat_created = rec.get("created_iso") if rec else None
        delta = None
        if rar_local and mat_created:
            delta = (dt.datetime.fromisoformat(mat_created).replace(tzinfo=dt.timezone.utc)
                     - dt.datetime.fromisoformat(rar_local)).total_seconds()
        checks.append({
            "source": source,
            "file_name": short,
            "rar_file_flags": m.get("file_flags"),
            "rar_htime_flags": (hrec or {}).get("htime_flags"),
            "rar_mtime_utc": (hrec or {}).get("mtime_utc"),
            "rar_mtime_local_utc8": rar_local,
            "mat_created_iso": mat_created,
            "delta_seconds": delta,
            "match_within_2s": (delta is not None and abs(delta) <= 2),
        })

    result = {
        "schema": "dronerfa-meta-split-rar-header-times/v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "network_access": "none",
        "pilot_archive": {k: v for k, v in pilot.items() if k not in ("members", "blocks")},
        "pilot_members": pilot.get("members", []),
        "partial_downloads": parts,
        "mtime_vs_mat_header_check": checks,
        "check_summary": {
            "members_checked": len(checks),
            "match_within_2s": sum(1 for c in checks if c["match_within_2s"]),
            "delta_seconds_range": [min((c["delta_seconds"] for c in checks
                                         if c["delta_seconds"] is not None), default=None),
                                    max((c["delta_seconds"] for c in checks
                                         if c["delta_seconds"] is not None), default=None)],
        },
    }
    out_path = os.path.join(HERE, "rar_header_times.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"pilot": result["pilot_archive"], "members": len(pilot.get("members", [])),
                      "check_summary": result["check_summary"], "checks": checks},
                     ensure_ascii=False, indent=1))
    print(json.dumps(parts, ensure_ascii=False, indent=1))
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
