# -*- coding: utf-8 -*-
"""META-SPLIT-1 / step 3a: read-only inventory of the official DroneRFa archive
index (the 632-entry RAR manifest `dronerfa_manifest.json`).

Nothing is downloaded and no network call is made: the index was produced when
the archive was first fingerprinted and is already in the project.

The goal is to answer, from metadata alone:
  * which (T, D, S) cells exist in the official archive,
  * which of them are already on G:,
  * which additional files exist for the three legacy classes
    (T0000 Background, T0010 Phantom 4 Pro, T0101 Air 2S) and their packed /
    unpacked size, without acquiring them.

Outputs (work/meta_split1/):
  archive_index_cells_20260925.json
  archive_unused_for_legacy3_20260925.csv
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
ARCHIVE_INDEX = os.path.join(WORKSPACE, "dronerfa_manifest.json")
MANIFEST = r"G:\DroneRFa数据集\dataset_manifest.csv"

# Class code -> model name, transcribed from DroneRFa JEIT 2024 Table 3
# (dronerfa_baseline/outputs/literature_review/text/01_DroneRFa_Dataset_JEIT2024.txt
#  lines 164-190).  Codes above 4 bits are controller classes.
T_TABLE = {
    "0000": ("Background", "背景(含蓝牙、WiFi)"),
    "0001": ("DJI Phantom 3", "无人机"),
    "0010": ("DJI Phantom 4 Pro", "无人机"),
    "0011": ("DJI MATRICE 200", "无人机"),
    "0100": ("DJI MATRICE 100", "无人机"),
    "0101": ("DJI Air 2S", "无人机"),
    "0110": ("DJI Mini 3 Pro", "无人机"),
    "0111": ("DJI Inspire 2", "无人机"),
    "1000": ("DJI Mavic Pro", "无人机"),
    "1001": ("DJI Mini 2", "无人机"),
    "1010": ("DJI Mavic 3", "无人机"),
    "1011": ("DJI MATRICE 300", "无人机"),
    "1100": ("DJI Phantom 4 Pro RTK", "无人机"),
    "1101": ("DJI MATRICE 30T", "无人机"),
    "1110": ("DJI AVATA", "无人机"),
    "1111": ("DJI 通信模块自组机", "无人机"),
    "10000": ("DJI MATRICE 600 Pro", "无人机"),
    "10001": ("VBar 飞控器", "飞控器"),
    "10010": ("FrSky X20 飞控器", "飞控器"),
    "10011": ("Futaba T6IZ 飞控器", "飞控器"),
    "10100": ("Taranis Plus 飞控器", "飞控器"),
    "10101": ("RadioLink AT9S 飞控器", "飞控器"),
    "10110": ("Futaba T14SG 飞控器", "飞控器"),
    "10111": ("云卓 T12 飞控器", "飞控器"),
    "11000": ("云卓 T10 飞控器", "飞控器"),
}
D_TABLE = {"00": "20-40 m", "01": "40-80 m", "10": "80-150 m"}

LEGACY3 = {"0000", "0010", "0101"}  # T0000 Background, T0010 Phantom 4 Pro, T0101 Air 2S


def parse(name: str) -> dict:
    stem = name.split("/")[-1]
    if stem.lower().endswith(".mat"):
        stem = stem[:-4]
    parts = stem.split("_")
    out = {"stem": stem, "t": None, "d": None, "s": None}
    for p in parts:
        if p.startswith("T"):
            out["t"] = p[1:]
        elif p.startswith("D"):
            out["d"] = p[1:]
        elif p.startswith("S"):
            out["s"] = p[1:]
    return out


def main() -> int:
    idx = json.load(open(ARCHIVE_INDEX, encoding="utf-8"))
    mats = [e for e in idx["entries"] if e["name"].lower().endswith(".mat")]

    with open(MANIFEST, "r", encoding="utf-8-sig", newline="") as fh:
        local = {r["file_name"] for r in csv.DictReader(fh)}

    cells = {}
    for e in mats:
        p = parse(e["name"])
        key = (p["t"], p["d"], p["s"])
        cells[key] = {
            "file_name": p["stem"] + ".mat",
            "t": p["t"], "d": p["d"], "s": p["s"],
            "block_offset": e["block_offset"],
            "block_size": e["block_size"],
            "packed_size": e["packed_size"],
            "unpacked_size": e["unpacked_size"],
            "on_g_now": (p["stem"] + ".mat") in local,
            "class_name": T_TABLE.get(p["t"], ("UNMAPPED", "?"))[0],
            "distance_label": D_TABLE.get(p["d"], "no D field in archive name"),
            "s_band_group": ("initial_comm_band" if p["s"] and p["s"][0] == "0"
                             else "switched_band" if p["s"] else None),
        }

    by_t = {}
    for (t, d, s), v in sorted(cells.items()):
        by_t.setdefault(t, {"n": 0, "n_on_g": 0, "d_values": set(), "s_values": set(),
                            "unpacked_bytes": 0, "packed_bytes": 0,
                            "class_name": v["class_name"]})
        b = by_t[t]
        b["n"] += 1
        b["n_on_g"] += 1 if v["on_g_now"] else 0
        b["d_values"].add(d)
        b["s_values"].add(s)
        b["unpacked_bytes"] += v["unpacked_size"]
        b["packed_bytes"] += v["packed_size"]

    by_t_out = {}
    for t, b in by_t.items():
        by_t_out[t] = {
            "class_name": b["class_name"],
            "archive_files": b["n"],
            "archive_files_on_g": b["n_on_g"],
            "archive_files_not_on_g": b["n"] - b["n_on_g"],
            "d_values": sorted(x for x in b["d_values"] if x),
            "d_field_present": any(x for x in b["d_values"]),
            "s_values": sorted(x for x in b["s_values"] if x),
            "unpacked_bytes_total": b["unpacked_bytes"],
            "packed_bytes_total": b["packed_bytes"],
            "unpacked_bytes_not_on_g": sum(v["unpacked_size"] for k, v in cells.items()
                                           if k[0] == t and not v["on_g_now"]),
        }

    legacy_rows = [v for k, v in sorted(cells.items()) if k[0] in LEGACY3]
    unused_legacy = [v for v in legacy_rows if not v["on_g_now"]]

    payload = {
        "schema": "dronerfa-meta-split-archive-cells/v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "archive_index_path": ARCHIVE_INDEX,
        "archive_index_entries": len(idx["entries"]),
        "archive_index_mat_entries": len(mats),
        "archive_size_bytes": idx.get("archive_size"),
        "local_manifest": MANIFEST,
        "local_manifest_rows": len(local),
        "t_table_source": "DroneRFa JEIT 2024 Table 3 (local extracted text)",
        "per_class": by_t_out,
        "legacy3_cells": legacy_rows,
        "legacy3_unused_cells": unused_legacy,
        "legacy3_unused_count": len(unused_legacy),
        "legacy3_unused_unpacked_bytes": sum(v["unpacked_size"] for v in unused_legacy),
        "legacy3_unused_packed_bytes": sum(v["packed_size"] for v in unused_legacy),
        "notes": [
            "Read-only inventory of an index that is already in the project; no network access, no download.",
            "The archive index carries no file timestamps, only name/offset/sizes, so the acquisition date of "
            "an un-downloaded file cannot be established from it.",
            "S 0000-0111 = initial communication band group, S 1000-1111 = switched band group (JEIT 2024 Table 3).",
        ],
    }
    out_json = os.path.join(HERE, "archive_index_cells_20260925.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    out_csv = os.path.join(HERE, "archive_unused_for_legacy3_20260925.csv")
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file_name", "class_name", "t", "d", "s", "distance_label", "s_band_group",
                    "packed_size", "unpacked_size", "block_offset"])
        for v in sorted(unused_legacy, key=lambda x: x["file_name"]):
            w.writerow([v["file_name"], v["class_name"], v["t"], v["d"], v["s"],
                        v["distance_label"], v["s_band_group"], v["packed_size"],
                        v["unpacked_size"], v["block_offset"]])

    print(json.dumps({
        "archive_mat_entries": len(mats),
        "distinct_classes_in_archive": len(by_t_out),
        "per_class": {k: {"name": v["class_name"], "n": v["archive_files"],
                          "on_g": v["archive_files_on_g"],
                          "d_values": v["d_values"], "s_values": v["s_values"]}
                      for k, v in sorted(by_t_out.items())},
        "legacy3_unused_count": len(unused_legacy),
        "legacy3_unused_unpacked_bytes": payload["legacy3_unused_unpacked_bytes"],
        "wrote": [out_json, out_csv],
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
