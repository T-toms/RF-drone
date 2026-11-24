# -*- coding: utf-8 -*-
"""META-SPLIT-1: decode the FHEXTRA_HTIME record of the pilot archive members
and check it against the MATLAB 'Created on' values that are already known.

This is a small numeric probe: it prints the raw HTIME bytes and every plausible
interpretation, then states which one reproduces the known timestamps.
"""
import datetime as dt
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scan_rar_headers import WORKSPACE, walk_archive  # noqa: E402

pilot = walk_archive(os.path.join(WORKSPACE, "DroneRFa_pilot.rar"))
heads = json.load(open(os.path.join(HERE, "mat_headers_104.json"), encoding="utf-8"))
by_name = {r["file_name"]: r for r in heads["records"]}

rows = []
for m in pilot["members"]:
    short = os.path.basename(m["file_name"])
    hrec = next((r for r in m.get("extra_records", []) if r.get("type_name") == "HTIME"), None)
    raw = bytes.fromhex(hrec["raw_hex"]) if hrec and "raw_hex" in hrec else b""
    interp = {}
    if len(raw) >= 9:
        flags = raw[0]
        interp["flags"] = flags
        # interpretation A: flags bit0 = unix format
        if flags & 0x01:
            interp["A_unix_mtime_utc"] = dt.datetime.fromtimestamp(
                struct.unpack_from("<I", raw, 1)[0], dt.timezone.utc).isoformat()
        else:
            ft = struct.unpack_from("<Q", raw, 1)[0]
            secs = ft / 10_000_000 - 11644473600
            interp["A_filetime_utc"] = dt.datetime.fromtimestamp(secs, dt.timezone.utc).isoformat()
        # interpretation B: last 4 bytes are a unix time
        interp["B_tail4_unix_utc"] = dt.datetime.fromtimestamp(
            struct.unpack_from("<I", raw, len(raw) - 4)[0], dt.timezone.utc).isoformat()
        # interpretation C: bytes 1..5 as unix
        interp["C_bytes1_5_unix_utc"] = dt.datetime.fromtimestamp(
            struct.unpack_from("<I", raw, 1)[0], dt.timezone.utc).isoformat()
        # interpretation D: FILETIME at bytes 1..9 (needs 9 bytes)
        if len(raw) >= 9:
            ft = struct.unpack_from("<Q", raw, 1)[0]
            secs = ft / 10_000_000 - 11644473600
            interp["D_filetime_utc"] = dt.datetime.fromtimestamp(secs, dt.timezone.utc).isoformat()
    rows.append({"file_name": short, "mat_created_local": by_name[short]["created_iso"],
                 "htime_raw_hex": raw.hex(), "n_bytes": len(raw), "interp": interp})

print(json.dumps(rows[:4], ensure_ascii=False, indent=1))

# Which interpretation reproduces the known local timestamps with a fixed offset?
cands = ["A_unix_mtime_utc", "A_filetime_utc", "B_tail4_unix_utc", "C_bytes1_5_unix_utc",
         "D_filetime_utc"]
summary = {}
for c in cands:
    deltas = []
    for r in rows:
        v = r["interp"].get(c)
        if not v:
            continue
        got = dt.datetime.fromisoformat(v)
        want = dt.datetime.fromisoformat(r["mat_created_local"]).replace(tzinfo=dt.timezone.utc)
        deltas.append((want - got).total_seconds())
    if deltas:
        summary[c] = {"n": len(deltas), "unique_deltas_seconds": sorted(set(deltas))[:5],
                      "all_equal": len(set(deltas)) == 1}
print(json.dumps(summary, ensure_ascii=False, indent=1))
