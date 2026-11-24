# -*- coding: utf-8 -*-
"""Ad-hoc: how many un-downloaded archive members do the other ten classes have,
and what would extending the header probe to them cost?"""
import json
import os

ROOT = r"C:\Users\Administrator\Desktop\数据集处理"
cells = json.load(open(os.path.join(ROOT, "work", "meta_split1",
                                    "archive_index_cells_20260925.json"),
                       encoding="utf-8"))
LEGACY3 = {"0000", "0010", "0101"}
# measured in META-HEADER1: 78 bytes and ~24 s per member (2 requests each)
BYTES_PER_MEMBER = 80
SECONDS_PER_REQUEST = 24.0

rows = []
tot_members = tot_bytes = 0
for t, info in sorted(cells["per_class"].items()):
    if t in LEGACY3:
        continue
    missing = info["archive_files_not_on_g"]
    if missing == 0:
        continue
    bytes_missing = info["unpacked_bytes_not_on_g"]
    rows.append((t, info["class_name"], info["archive_files"],
                 info["archive_files_on_g"], missing, bytes_missing))
    tot_members += missing
    tot_bytes += bytes_missing

print("%-7s %-24s %6s %6s %8s %18s" % ("T", "class", "arch", "onG", "missing", "unpacked_bytes"))
for t, name, arch, on_g, missing, bytes_missing in rows:
    print("%-7s %-24s %6d %6d %8d %18d" % ("T" + t, name, arch, on_g, missing, bytes_missing))
print()
print("classes needing a probe :", len(rows))
print("un-downloaded members   :", tot_members)
print("their unpacked payload  :", tot_bytes, "(%.1f GiB)" % (tot_bytes / 2**30))
print()
print("header-probe cost if extended:")
print("  transfer  ~ %d bytes (%.1f KiB)" % (tot_members * BYTES_PER_MEMBER,
                                             tot_members * BYTES_PER_MEMBER / 1024))
print("  requests  ~ %d" % (tot_members * 2 + 1))
print("  wall time ~ %.1f h at %.0f s/request" % (
    (tot_members * 2 + 1) * SECONDS_PER_REQUEST / 3600, SECONDS_PER_REQUEST))
