#!/usr/bin/env python3
"""SELECT12-G1 step 1: freeze the 104-file target table for the 12-model dataset **on G:**.

This is the G-drive successor of ``work/sel12/freeze_select12.py`` (that one was frozen
for ``F:\\DroneRFa12机型``; the user cancelled that plan and required everything to stay
on ``G:\\DroneRFa数据集``).  The target *selection* is unchanged and is still exactly the
frozen 104 files: 12 UAV models + Background, 8 leading S segments each.

Reads (read-only):
  * ``dronerfa_manifest.json``                                    - local archive index
  * ``work/dronerfa-12model-download-candidates-20260922.json``   - the frozen 80 new targets
  * ``G:\\DroneRFa数据集\\dataset_manifest.csv``                  - former_partition of the 24 baseline files
  * ``G:\\DroneRFa数据集\\original``                              - to cross-check already-landed MAT sizes

Writes (workspace + authorized G record tree only):
  * ``work/sel12g/sel12g-config-20260924.json``
  * ``work/sel12g/sel12g-target-table.csv``

No MAT / .part / .rar / log on G: is written, moved or deleted by this script.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK = PROJECT_ROOT / "work" / "sel12g"
MANIFEST = PROJECT_ROOT / "dronerfa_manifest.json"
CANDIDATES = PROJECT_ROOT / "work" / "dronerfa-12model-download-candidates-20260922.json"

G_ROOT = Path("G:/DroneRFa数据集")
G_ORIGINAL = G_ROOT / "original"
G_RECORDS = G_ROOT / "download_records_12models_20260922"
G_DOWNLOADS = G_RECORDS / "downloads"
G_MANIFEST_CSV = G_ROOT / "dataset_manifest.csv"

ARCHIVE_URL = "https://china.scidb.cn/download?fileId=c403fc76444e4b9989e4f3ff570f3b3d"
ARCHIVE_SIZE = 615_955_805_466
ARCHIVE_ETAG = '"6774350c-8f69d3b51a"'

# Expected G: volume identity.  exFAT volume that used to be F: before the 2026-09-24 remap.
G_VOLUME = {
    "driveLetter": "G:",
    "fileSystem": "exFAT",
    "serialNumber": "24FF42D7",
    "source": "recorded in work/deepseek-context.md and the 2026-09-24 incident record",
}

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

SEGMENTS = ("S0000", "S0001", "S0010", "S0011", "S0100", "S0101", "S0110", "S0111")
NAME_RE = re.compile(r"^(T[01]{4})(?:_(D\d{2}))?_(S[01]{4})\.mat$")
NO_DISTANCE_TARGETS = ("T1010", "T1011", "T1100")


class FreezeError(RuntimeError):
    pass


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
        raise FreezeError(f"文件名不符合T*/D*/S*规则：{name}")
    return match.group(1), match.group(2), match.group(3)


def assert_under(path: Path, root: Path, label: str) -> Path:
    """Reject any path that escapes the authorized G: root (absolute check)."""
    resolved = Path(os.path.abspath(path))
    root_resolved = Path(os.path.abspath(root))
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise FreezeError(f"{label} 越界：{resolved} 不在 {root_resolved} 之内")
    return resolved


def read_g_former_partitions() -> dict[str, dict]:
    raw = G_MANIFEST_CSV.read_bytes()
    text = raw.decode("utf-8-sig")
    table: dict[str, dict] = {}
    for row in csv.DictReader(io.StringIO(text)):
        table[row["file_name"]] = {
            "former_partition": row["former_partition"],
            "class_name": row["class_name"],
            "class_index": int(row["class_index"]),
            "size_bytes": int(row["size_bytes"]),
        }
    return table


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)

    if not G_ROOT.is_dir():
        raise FreezeError(f"G盘数据集根不存在：{G_ROOT}")

    manifest_raw = MANIFEST.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    if manifest["archive_size"] != ARCHIVE_SIZE:
        raise FreezeError("本地归档索引的档案大小与冻结配置不符")
    if manifest.get("etag") != ARCHIVE_ETAG:
        raise FreezeError("本地归档索引的ETag与冻结配置不符")

    by_basename: dict[str, dict] = {}
    for entry in manifest["entries"]:
        base_name = entry["name"].rsplit("/", 1)[-1]
        if base_name.endswith(".mat"):
            if base_name in by_basename:
                raise FreezeError(f"归档索引出现重复文件名：{base_name}")
            by_basename[base_name] = entry

    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    added_names = [Path(item["name"]).name for item in candidates["files"]]
    if len(added_names) != 80 or len(set(added_names)) != 80:
        raise FreezeError("冻结候选清单不是80个唯一文件名")

    former = read_g_former_partitions()
    if len(former) != 24:
        raise FreezeError(f"G盘dataset_manifest.csv不是24行：{len(former)}")

    existing_names: list[str] = []
    for target_id in ("T0000", "T0010", "T0101"):
        for segment in SEGMENTS:
            existing_names.append(f"{target_id}_D00_{segment}.mat")
    if sorted(existing_names) != sorted(former):
        raise FreezeError("由既有12机型规则推出的24个文件名与G盘manifest不一致")

    all_names = existing_names + added_names
    if len(all_names) != 104 or len(set(all_names)) != 104:
        raise FreezeError("目标表不是104个唯一文件名")

    on_disk = {path.name: path.stat().st_size for path in G_ORIGINAL.glob("*.mat")}

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
        is_baseline = name in former
        expected_unpacked = entry["unpacked_size"]
        actual_bytes = on_disk.get(name)
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
            "origin": "baseline_24" if is_baseline else "to_download",
            "former_partition": former[name]["former_partition"] if is_baseline else None,
            "split_status": "preserve_former_partition" if is_baseline else "unassigned",
            "evaluation_role": (
                "already_used_in_accepted_evaluation"
                if is_baseline and former[name]["former_partition"] == "evaluated_holdout"
                else ("development_only" if is_baseline else "unassigned")
            ),
            "present_on_g": actual_bytes is not None,
            "present_bytes": actual_bytes,
            "target_path": str(G_ORIGINAL / name),
        }
        if is_baseline and former[name]["size_bytes"] != expected_unpacked:
            problems.append(
                f"{name}: G盘manifest记录大小{former[name]['size_bytes']}与归档索引{expected_unpacked}不符"
            )
        files.append(record)

    if problems:
        print(json.dumps({"problems": problems}, ensure_ascii=False, indent=2))
        return 2

    # Any path this plan can write must live on G:, never on F:.
    for item in files:
        assert_under(Path(item["target_path"]), G_ORIGINAL, item["filename"])
    for scratch_name in ("scratch",):
        assert_under(G_DOWNLOADS / scratch_name, G_RECORDS, scratch_name)

    baseline = [item for item in files if item["origin"] == "baseline_24"]
    to_download = [item for item in files if item["origin"] == "to_download"]
    landed = [item for item in to_download if item["present_on_g"]]
    missing = [item for item in to_download if not item["present_on_g"]]
    size_mismatch = [
        item["filename"] for item in to_download
        if item["present_on_g"] and item["present_bytes"] != item["expected_mat_bytes"]
    ]

    totals = {
        "target_files": len(files),
        "baseline_24": len(baseline),
        "baseline_bytes": sum(item["expected_mat_bytes"] for item in baseline),
        "to_download_80": len(to_download),
        "to_download_mat_bytes": sum(item["expected_mat_bytes"] for item in to_download),
        "to_download_packed_bytes": sum(item["archive_packed_size"] for item in to_download),
        "to_download_block_bytes": sum(item["archive_block_size"] for item in to_download),
        "already_landed_on_g": len(landed),
        "already_landed_mat_bytes": sum(item["expected_mat_bytes"] for item in landed),
        "still_missing_on_g": len(missing),
        "still_missing_mat_bytes": sum(item["expected_mat_bytes"] for item in missing),
        "still_missing_packed_bytes": sum(item["archive_packed_size"] for item in missing),
        "still_missing_block_bytes": sum(item["archive_block_size"] for item in missing),
        "landed_size_mismatch": size_mismatch,
        "total_mat_bytes": sum(item["expected_mat_bytes"] for item in files),
        "mat_on_g_now": len(on_disk),
        "mat_bytes_on_g_now": sum(on_disk.values()),
    }

    per_target: dict[str, dict] = {}
    for item in files:
        bucket = per_target.setdefault(
            item["target_id"],
            {
                "class_name": item["class_name"],
                "class_index": item["class_index"],
                "target_files": 0,
                "baseline_24": 0,
                "landed_on_g": 0,
                "missing_on_g": 0,
            },
        )
        bucket["target_files"] += 1
        if item["origin"] == "baseline_24":
            bucket["baseline_24"] += 1
        if item["present_on_g"]:
            bucket["landed_on_g"] += 1
        else:
            bucket["missing_on_g"] += 1

    config = {
        "schema": "dronerfa-select12g-plan/v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "task": "SELECT12-G1",
        "scope_note": (
            "104个目标文件 = 12个无人机机型 + 背景(T0000)，每个机型只取冻结清单的8个S段。"
            "全部留在G盘原位；不建立新F盘目录、不迁移G盘文件、不续传完整RAR归档。"
        ),
        "authorization": {
            "user_decision": "仍下载到 G 盘，不转移到新的 F 盘",
            "superseded_task": "SELECT12-F1（针对 F:\\DroneRFa12机型，已被用户终止，本配置不引用其路径）",
        },
        "volume": G_VOLUME,
        "archive": {
            "url": ARCHIVE_URL,
            "size": ARCHIVE_SIZE,
            "etag": ARCHIVE_ETAG,
            "index_path": str(MANIFEST),
            "index_sha256": sha256_file(MANIFEST),
            "index_entries": len(manifest["entries"]),
            "index_mat_entries": len(by_basename),
        },
        "paths": {
            "workspace_root": str(PROJECT_ROOT),
            "g_root": str(G_ROOT),
            "g_original": str(G_ORIGINAL),
            "g_records": str(G_RECORDS),
            "g_downloads": str(G_DOWNLOADS),
            "g_scratch": str(G_DOWNLOADS / "scratch"),
            "g_state": str(G_RECORDS / "select12g-state.json"),
            "g_lock": str(G_RECORDS / "select12g.lock"),
            "g_log": str(G_RECORDS / "select12g.log"),
            "g_baseline_manifest_csv": str(G_MANIFEST_CSV),
            "g_baseline_manifest_sha256": sha256_file(G_MANIFEST_CSV),
        },
        "target_labels": MODEL_LABELS,
        "segments": list(SEGMENTS),
        "no_distance_targets": list(NO_DISTANCE_TARGETS),
        "totals": totals,
        "per_target": per_target,
        "files": files,
    }

    config_path = WORK / "sel12g-config-20260924.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = WORK / "sel12g-target-table.csv"
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
        "present_on_g",
        "present_bytes",
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
    try:
        raise SystemExit(main())
    except FreezeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
