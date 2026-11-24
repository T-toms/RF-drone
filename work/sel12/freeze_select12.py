#!/usr/bin/env python3
"""SELECT12-F1 step 1: freeze the 104-file target table for the 12-model dataset.

Reads (read-only):
  * ``dronerfa_manifest.json``                       - local archive index (630 .mat entries)
  * ``work/dronerfa-12model-download-candidates-20260922.json`` - the frozen 80 new targets
  * ``G:\\DroneRFa数据集\\dataset_manifest.csv``     - former_partition of the existing 24 files

Writes:
  * ``work/sel12/sel12-config-20260924.json``        - frozen plan (machine readable)
  * ``work/sel12/sel12-target-table.csv``            - flat 104-row table for review

The script is strictly read-only with respect to G: and F: and writes nothing outside
the project workspace.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK = PROJECT_ROOT / "work" / "sel12"
MANIFEST = PROJECT_ROOT / "dronerfa_manifest.json"
CANDIDATES = PROJECT_ROOT / "work" / "dronerfa-12model-download-candidates-20260922.json"
G_ROOT = Path("G:/DroneRFa数据集")
G_MANIFEST_CSV = G_ROOT / "dataset_manifest.csv"

F_ROOT = Path("F:/DroneRFa12机型")
F_ORIGINAL = F_ROOT / "original"
F_RECORDS = F_ROOT / "download_records"

ARCHIVE_URL = "https://china.scidb.cn/download?fileId=c403fc76444e4b9989e4f3ff570f3b3d"
ARCHIVE_SIZE = 615_955_805_466
ARCHIVE_ETAG = '"6774350c-8f69d3b51a"'

# 12 UAV models + background.  Labels follow the frozen candidate list and the
# dataset_manifest.csv naming already used by the project.
MODEL_LABELS = {
    "T0000": "Background",
    "T0001": "DJI Phantom 3",
    "T0010": "DJI Phantom 4 Pro",
    "T0011": "DJI Matrice 200",
    "T0100": "DJI Matrice 100",
    "T0101": "DJI Air 2S",
    "T0110": "DJI Mini 3 Pro",
    "T0111": "DJI Inspire 2",
    "T1000": "DJI Mavic Pro",
    "T1001": "DJI Mini 2",
    "T1010": "DJI Mavic 3",
    "T1011": "DJI Matrice 300",
    "T1100": "DJI Phantom 4 Pro RTK",
}
CLASS_INDEX = {
    "T0000": 0,
    "T0001": 1,
    "T0010": 10,
    "T0011": 11,
    "T0100": 100,
    "T0101": 101,
    "T0110": 110,
    "T0111": 111,
    "T1000": 1000,
    "T1001": 1001,
    "T1010": 1010,
    "T1011": 1011,
    "T1100": 1100,
}

# The eight leading S segments frozen by the candidate list.
SEGMENTS = ("S0000", "S0001", "S0010", "S0011", "S0100", "S0101", "S0110", "S0111")

NAME_RE = re.compile(r"^(T[01]{4})(?:_(D\d{2}))?_(S[01]{4})\.mat$")

# The three targets whose official archive filenames carry no distance field.
NO_DISTANCE_TARGETS = ("T1010", "T1011", "T1100")


def sha256_file(path: Path, chunk: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_name(name: str) -> tuple[str, str | None, str]:
    match = NAME_RE.match(name)
    if not match:
        raise ValueError(f"文件名不符合T*/D*/S*规则：{name}")
    return match.group(1), match.group(2), match.group(3)


def read_g_former_partitions() -> tuple[dict[str, dict], list[str]]:
    """Read the already-verified G drive manifest (UTF-8 with BOM)."""
    raw = G_MANIFEST_CSV.read_bytes()
    text = raw.decode("utf-8-sig")
    table: dict[str, dict] = {}
    for row in csv.DictReader(io.StringIO(text)):
        table[row["file_name"]] = {
            "former_partition": row["former_partition"],
            "class_name": row["class_name"],
            "class_index": int(row["class_index"]),
            "size_bytes": int(row["size_bytes"]),
            "manifest_path_at_creation": row["unified_path"],
        }
    return table, list(table)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)

    manifest_raw = MANIFEST.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    if manifest["archive_size"] != ARCHIVE_SIZE:
        raise SystemExit("本地归档索引的档案大小与冻结配置不符")
    if manifest.get("etag") != ARCHIVE_ETAG:
        raise SystemExit("本地归档索引的ETag与冻结配置不符")

    by_basename: dict[str, dict] = {}
    for entry in manifest["entries"]:
        base_name = entry["name"].rsplit("/", 1)[-1]
        if base_name.endswith(".mat"):
            if base_name in by_basename:
                raise SystemExit(f"归档索引出现重复文件名：{base_name}")
            by_basename[base_name] = entry

    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    added_names = [Path(item["name"]).name for item in candidates["files"]]
    if len(added_names) != 80 or len(set(added_names)) != 80:
        raise SystemExit("冻结候选清单不是80个唯一文件名")

    former, manifest_names = read_g_former_partitions()
    if len(manifest_names) != 24:
        raise SystemExit(f"G盘dataset_manifest.csv不是24行：{len(manifest_names)}")

    existing_names: list[str] = []
    for target_id in ("T0000", "T0010", "T0101"):
        for segment in SEGMENTS:
            existing_names.append(f"{target_id}_D00_{segment}.mat")
    if sorted(existing_names) != sorted(manifest_names):
        raise SystemExit("由既有12机型规则推出的24个文件名与G盘manifest不一致")

    all_names = existing_names + added_names
    if len(all_names) != 104 or len(set(all_names)) != 104:
        raise SystemExit("目标表不是104个唯一文件名")

    files: list[dict] = []
    problems: list[str] = []
    for name in all_names:
        target_id, distance_code, segment = parse_name(name)
        entry = by_basename.get(name)
        if entry is None:
            problems.append(f"归档索引中缺少条目：{name}")
            continue
        if target_id in NO_DISTANCE_TARGETS and distance_code is not None:
            problems.append(f"{name}: 该机型在官方归档中不带距离字段")
        if target_id not in NO_DISTANCE_TARGETS and distance_code is None:
            problems.append(f"{name}: 该机型在官方归档中应带距离字段")
        is_existing = name in former
        expected_unpacked = entry["unpacked_size"]
        record = {
            "filename": name,
            "target_id": target_id,
            "class_name": MODEL_LABELS[target_id],
            "class_index": CLASS_INDEX[target_id],
            "segment": segment,
            "distance_code": distance_code,
            "archive_name": entry["name"],
            "archive_block_offset": entry["block_offset"],
            "archive_block_size": entry["block_size"],
            "archive_packed_size": entry["packed_size"],
            "expected_mat_bytes": expected_unpacked,
            "origin": "g_existing" if is_existing else "to_download",
            "former_partition": former[name]["former_partition"] if is_existing else None,
            "split_status": "preserve_former_partition" if is_existing else "unassigned",
            "evaluation_role": (
                "already_used_in_accepted_evaluation"
                if is_existing and former[name]["former_partition"] == "evaluated_holdout"
                else ("development_only" if is_existing else "unassigned")
            ),
            "projected_path": str(F_ORIGINAL / name),
        }
        if is_existing and former[name]["size_bytes"] != expected_unpacked:
            problems.append(
                f"{name}: G盘manifest记录大小{former[name]['size_bytes']}与归档索引{expected_unpacked}不符"
            )
        files.append(record)

    if problems:
        print(json.dumps({"problems": problems}, ensure_ascii=False, indent=2))
        return 2

    to_copy = [item for item in files if item["origin"] == "g_existing"]
    to_download = [item for item in files if item["origin"] == "to_download"]
    totals = {
        "target_files": len(files),
        "existing_on_g": len(to_copy),
        "to_download": len(to_download),
        "existing_bytes": sum(item["expected_mat_bytes"] for item in to_copy),
        "to_download_mat_bytes": sum(item["expected_mat_bytes"] for item in to_download),
        "to_download_packed_bytes": sum(item["archive_packed_size"] for item in to_download),
        "to_download_block_bytes": sum(item["archive_block_size"] for item in to_download),
        "total_mat_bytes": sum(item["expected_mat_bytes"] for item in files),
    }

    per_target: dict[str, dict] = {}
    for item in files:
        bucket = per_target.setdefault(
            item["target_id"],
            {
                "class_name": item["class_name"],
                "class_index": item["class_index"],
                "files": 0,
                "existing_on_g": 0,
                "to_download": 0,
            },
        )
        bucket["files"] += 1
        bucket["existing_on_g" if item["origin"] == "g_existing" else "to_download"] += 1

    config = {
        "schema": "dronerfa-select12-plan/v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "task": "SELECT12-F1",
        "scope_note": (
            "104个目标文件 = 12个无人机机型 + 背景(T0000)，每个机型只取冻结清单的8个S段。"
            "不扩大到整个归档、不取其它距离条件。"
        ),
        "archive": {
            "url": ARCHIVE_URL,
            "size": ARCHIVE_SIZE,
            "etag": ARCHIVE_ETAG,
            "index_path": str(MANIFEST),
            "index_sha256": sha256_file(MANIFEST),
            "index_entries": len(manifest["entries"]),
            "index_mat_entries": len(by_basename),
            "range_probe": {
                "checked_at": "2026-09-24T14:21:04+08:00",
                "status": 206,
                "content_range": f"bytes 0-7/{ARCHIVE_SIZE}",
                "etag": ARCHIVE_ETAG,
                "signature": "526172211a070100",
                "single_stream_MiB_per_s": 0.1848,
                "six_stream_MiB_per_s": 0.9039,
                "probe_record": "work/sel12/remote-probe-20260924.json",
            },
        },
        "paths": {
            "workspace_root": str(PROJECT_ROOT),
            "g_root": str(G_ROOT),
            "g_original": str(G_ROOT / "original"),
            "f_root": str(F_ROOT),
            "f_original": str(F_ORIGINAL),
            "f_records": str(F_RECORDS),
            "f_marker": str(F_ROOT / "volume-marker.json"),
            "f_manifest_csv": str(F_ROOT / "dataset_manifest.csv"),
            "f_state": str(F_RECORDS / "select12-state.json"),
            "f_lock": str(F_RECORDS / "select12.lock"),
            "f_log": str(F_RECORDS / "select12.log"),
            "f_scratch": str(F_RECORDS / "scratch"),
        },
        "target_labels": MODEL_LABELS,
        "segments": list(SEGMENTS),
        "no_distance_targets": list(NO_DISTANCE_TARGETS),
        "totals": totals,
        "per_target": per_target,
        "files": files,
    }

    config_path = WORK / "sel12-config-20260924.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = WORK / "sel12-target-table.csv"
    fields = [
        "filename",
        "target_id",
        "class_name",
        "segment",
        "distance_code",
        "origin",
        "former_partition",
        "split_status",
        "evaluation_role",
        "expected_mat_bytes",
        "archive_block_offset",
        "archive_block_size",
        "archive_packed_size",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in files:
            writer.writerow({key: item[key] for key in fields})

    print(json.dumps({"config": str(config_path), "csv": str(csv_path), "totals": totals},
                     ensure_ascii=False, indent=2))
    print(json.dumps({"per_target": per_target}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
