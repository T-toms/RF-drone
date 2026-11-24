# -*- coding: utf-8 -*-
"""META-SPLIT-1 / step 1a: read the MATLAB v7.3 116-byte file header of every
manifest MAT file.

The MATLAB v7.3 (HDF5) header is a fixed 116-byte text block at offset 0 that
ends with the creation timestamp of the file, e.g.

    MATLAB 7.3 MAT-file, Platform: PCWIN64, Created on: Sun Nov 27 18:36:04 2022 HDF5 schema 1.00 .

Reading it costs 512 bytes per file and never touches the waveform payload.

Outputs (work/meta_split1/):
  mat_headers_104.json   full parse result, one record per manifest row
  mat_headers_104.csv    flat table for inspection
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys

HEADER_BYTES = 512
HDF5_SIG = b"\x89HDF\r\n\x1a\n"

# "MATLAB 7.3 MAT-file, Platform: PCWIN64, Created on: Sun Nov 27 18:36:04 2022 HDF5 schema 1.00 ."
CREATED_RE = re.compile(r"Created on:\s*([A-Za-z]{3} [A-Za-z]{3}\s+\d{1,2} \d{2}:\d{2}:\d{2} \d{4})")
PLATFORM_RE = re.compile(r"Platform:\s*([^,]+),")


def parse_header(raw: bytes) -> dict:
    text = raw[:116].decode("latin-1")
    out = {"header_text": text.rstrip()}
    m = PLATFORM_RE.search(text)
    out["platform"] = m.group(1).strip() if m else None
    m = CREATED_RE.search(text)
    out["created_raw"] = m.group(1) if m else None
    if m:
        # Collapse the double space that MATLAB emits for single-digit days.
        norm = re.sub(r"\s+", " ", m.group(1))
        try:
            stamp = dt.datetime.strptime(norm, "%a %b %d %H:%M:%S %Y")
            out["created_iso"] = stamp.isoformat()
            out["created_epoch"] = int(stamp.replace(tzinfo=dt.timezone.utc).timestamp())
            out["created_day"] = stamp.strftime("%Y-%m-%d")
            out["created_minute"] = stamp.strftime("%Y-%m-%dT%H:%M")
        except ValueError as exc:  # pragma: no cover - defensive
            out["created_iso"] = None
            out["parse_error"] = str(exc)
    else:
        out["created_iso"] = None
    out["hdf5_signature_offset"] = raw.find(HDF5_SIG)
    return out


def split_filename(name: str) -> dict:
    """T####_D##_S####.mat -> structured fields.  Pure string parsing; the
    meaning of each field is documented in the DroneRFa JEIT 2024 paper, but no
    field is interpreted here as an acquisition session."""
    stem = name[:-4] if name.lower().endswith(".mat") else name
    parts = stem.split("_")
    out = {"stem": stem, "n_fields": len(parts), "t_code": None, "d_code": None, "s_code": None,
           "parse_ok": False}
    for p in parts:
        if p.startswith("T") and len(p) > 1:
            out["t_code"] = p[1:]
        elif p.startswith("D") and len(p) > 1:
            out["d_code"] = p[1:]
        elif p.startswith("S") and len(p) > 1:
            out["s_code"] = p[1:]
    out["parse_ok"] = bool(out["t_code"] and out["d_code"] and out["s_code"]) and len(parts) == 3
    if out["s_code"]:
        out["s_bits"] = len(out["s_code"])
        out["s_band_group_bit"] = out["s_code"][0]          # 0 = initial band, 1 = switched band
        out["s_segment_bits"] = out["s_code"][1:]           # low bits = segment index field
        out["s_band_group"] = "initial_comm_band" if out["s_code"][0] == "0" else "switched_band"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=r"G:\DroneRFa数据集\dataset_manifest.csv")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    with open(args.manifest, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    records = []
    problems = []
    for row in rows:
        name = row["file_name"]
        path = row["unified_path"]
        rec = {
            "file_name": name,
            "class_id": row.get("class_id"),
            "class_name": row.get("class_name"),
            "former_partition": row.get("former_partition"),
            "split_status": row.get("split_status"),
            "evaluation_role": row.get("evaluation_role"),
            "origin": row.get("origin"),
            "manifest_size_bytes": int(row["size_bytes"]) if row.get("size_bytes") else None,
            "manifest_path": path,
        }
        rec.update(split_filename(name))
        if not os.path.exists(path):
            problems.append({"file_name": name, "problem": "path_missing", "path": path})
            rec["header_read_ok"] = False
            records.append(rec)
            continue
        rec["actual_size_bytes"] = os.path.getsize(path)
        rec["size_matches_manifest"] = rec["actual_size_bytes"] == rec["manifest_size_bytes"]
        with open(path, "rb") as fh:
            raw = fh.read(HEADER_BYTES)
        rec.update(parse_header(raw))
        rec["header_read_ok"] = rec.get("hdf5_signature_offset") == 512
        records.append(rec)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, "mat_headers_104.json")
    csv_path = os.path.join(outdir, "mat_headers_104.csv")

    payload = {
        "schema": "dronerfa-meta-split-mat-headers/v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": args.manifest,
        "manifest_rows": len(rows),
        "records": records,
        "problems": problems,
        "header_parse_failures": [r["file_name"] for r in records if not r.get("created_iso")],
        "notes": [
            "created_iso is the MATLAB v7.3 'Created on' string from the 116-byte MAT header.",
            "It records when MATLAB wrote the .mat container, which per the DroneRFa acquisition "
            "workflow happens during the acquisition run; it is NOT proof of an independent flight session.",
        ],
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    cols = ["file_name", "class_id", "class_name", "t_code", "d_code", "s_code", "s_band_group",
            "former_partition", "split_status", "evaluation_role", "origin",
            "manifest_size_bytes", "actual_size_bytes", "size_matches_manifest",
            "created_raw", "created_iso", "created_day", "created_minute", "platform",
            "hdf5_signature_offset", "header_read_ok"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            w.writerow(rec)

    print(json.dumps({
        "rows": len(rows),
        "records": len(records),
        "problems": problems,
        "created_iso_missing": payload["header_parse_failures"],
        "distinct_days": sorted({r["created_day"] for r in records if r.get("created_day")}),
        "distinct_classes": len({r["class_id"] for r in records}),
        "wrote": [json_path, csv_path],
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
