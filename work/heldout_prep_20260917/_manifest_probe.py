"""Inspect cached manifest + candidate list; report disk staging math (read-only)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\Administrator\Desktop\数据集处理")
manifest = json.loads((ROOT / "dronerfa_manifest.json").read_text(encoding="utf-8"))
cand = json.loads(
    (
        ROOT
        / "dronerfa_baseline/outputs/noise_ab_replicates_20260917/candidate_heldout_files.json"
    ).read_text(encoding="utf-8")
)

print("manifest keys:", sorted(manifest.keys()))
print("archive_size:", manifest.get("archive_size"))
entries = {e["name"]: e for e in manifest["entries"]}
print("manifest entries:", len(entries))

names = {Path(c["name"]).name for c in cand["files"]}
print("candidate count:", len(cand["files"]))
total_packed = 0
total_unpacked = 0
for item in cand["files"]:
    short = Path(item["name"]).name
    entry = entries.get(item["name"])
    if entry is None:
        print(f"MISSING IN MANIFEST: {item['name']}")
        continue
    same = all(
        entry[k] == item[k] for k in ("block_offset", "block_size", "packed_size", "unpacked_size")
    )
    total_packed += entry["block_size"]
    total_unpacked += entry["unpacked_size"]
    print(
        f"{short}  consistent={same}  off={entry['block_offset']}  "
        f"packed={entry['packed_size']}  unpacked={entry['unpacked_size']}"
    )

print("candidate packed (block_size) total:", total_packed)
print("candidate unpacked total:", total_unpacked)
print("cand json compressed_bytes:", cand["compressed_bytes"])
print("cand json uncompressed_bytes:", cand["uncompressed_bytes"])

free_c = shutil.disk_usage("C:\\").free
free_d = shutil.disk_usage("D:\\").free
print(f"C: free {free_c / 1024**3:.2f} GiB   D: free {free_d / 1024**3:.2f} GiB")
print(
    "staging check -> one temp rar + all new mat on C:",
    f"{(total_unpacked + max(e['block_size'] for e in entries.values() if Path(e['name']).name in names)) / 1024**3:.2f} GiB",
)

# What do the sibling (already-used) entries look like, for cross-check of semantics?
used = ["S0000", "S0001", "S0010", "S0011"]
for tag in ("T0000", "T0010", "T0101"):
    for seg in used:
        nm = f"DroneRFa/{tag}_D00_{seg}.mat"
        e = entries.get(nm)
        print(
            f"used {nm}: {'FOUND' if e else 'absent'} "
            + (
                f"off={e['block_offset']} unpacked={e['unpacked_size']}"
                if e
                else ""
            )
        )
