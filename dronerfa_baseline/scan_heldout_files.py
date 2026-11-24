#!/usr/bin/env python3
"""Structural scan + integrity check + frozen heldout window index.

Reads the 12 downloaded heldout source files once, and for every file:

* verifies the four RF0/RF1 I/Q datasets are stored as ``(1, N)`` float vectors of
  equal length, using the project's own ``dataset_length``/``read_vector_slice``;
* scans **every** stored sample of every channel in bounded chunks, checking for
  non-finite values and collecting raw IQ statistics (so nothing is concluded
  from a sample of the data);
* reports acquisition blocks and the usable 10 ms window count under the frozen
  10,000,000-sample block / 1,000,000-sample window rule with no cross-block and
  no overlapping windows;
* records any acquisition metadata actually present in the MAT container.

It then writes ``heldout_window_index.csv`` (split=heldout, field layout
identical to the existing index), ``heldout_dataset_summary.json`` and
``verification.json``.

Reading is parallel across files but bounded in memory; whole arrays are never
loaded at once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dronerfa_data import TARGET_NAMES, dataset_length, read_vector_slice  # noqa: E402

REQUIRED_KEYS = ("RF0_I", "RF0_Q", "RF1_I", "RF1_Q")
MAT_KEYS = ("target_code", "distance_code", "segment_code")
HELDOUT_ROOT = Path(r"F:\DroneRFa数据集\heldout_20260917")
ORIGINAL_ROOT = Path(r"C:\Users\Administrator\Documents\DroneRFa数据集")
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "heldout_preparation_20260917"
CANDIDATES = (
    SCRIPT_DIR / "outputs" / "noise_ab_replicates_20260917" / "candidate_heldout_files.json"
)
OLD_INDEX = SCRIPT_DIR / "outputs" / "window_index.csv"

WINDOW_SIZE = 1_000_000
BLOCK_SIZE = 10_000_000
SAMPLE_RATE_HZ = 100_000_000
LABELS = {"0000": 0, "0010": 1, "0101": 1}
CSV_FIELDS = (
    "file_path",
    "filename",
    "target_code",
    "target_name",
    "distance_code",
    "segment_code",
    "split",
    "label",
    "block_index",
    "window_index",
    "start",
    "length",
    "source_samples",
)

_print_lock = threading.Lock()


def log(message: str) -> None:
    with _print_lock:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path, chunk: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_name(filename: str) -> dict[str, str]:
    stem = filename[:-4] if filename.lower().endswith(".mat") else filename
    parts = stem.split("_")
    if len(parts) != 3 or not all(part[0] in "TDS" for part in parts):
        raise ValueError(f"无法解析文件名：{filename}")
    return {
        "target_code": parts[0][1:],
        "distance_code": parts[1][1:],
        "segment_code": parts[2][1:],
    }


def collect_metadata(handle: h5py.File) -> dict:
    """Record whatever *acquisition* metadata the container actually carries.

    ``MATLAB_class`` is an HDF5/MAT v7.3 format marker written for every array, so
    it is not acquisition metadata and must not make the check look positive.
    """
    root_attrs = {key: str(value) for key, value in handle.attrs.items()}
    dataset_attrs: dict[str, dict[str, str]] = {}
    for key in handle.keys():
        item = handle[key]
        if isinstance(item, h5py.Dataset):
            meaningful = {
                k: str(v) for k, v in item.attrs.items() if not k.startswith("MATLAB_")
            }
            if meaningful:
                dataset_attrs[key] = meaningful
    other_datasets: dict[str, str] = {}
    for key in handle.keys():
        if key in REQUIRED_KEYS:
            continue
        item = handle[key]
        if isinstance(item, h5py.Dataset):
            other_datasets[key] = f"shape={list(item.shape)} dtype={item.dtype}"
    return {
        "root_attrs": root_attrs,
        "dataset_attrs": dataset_attrs,
        "other_datasets": other_datasets,
        "has_any_metadata": bool(root_attrs or dataset_attrs or other_datasets),
        "matlab_class_markers": {
            key: str(handle[key].attrs.get("MATLAB_class"))
            for key in handle.keys()
            if isinstance(handle[key], h5py.Dataset)
            and "MATLAB_class" in handle[key].attrs
        },
    }


def scan_channel(dataset: h5py.Dataset, chunk: int) -> dict:
    """Fully scan one channel in bounded chunks for non-finite values and stats."""
    total = dataset_length(dataset)
    finite = 0
    non_finite = 0
    minimum = np.inf
    maximum = -np.inf
    sum_sq = 0.0
    sum_value = 0.0
    position = 0
    while position < total:
        length = min(chunk, total - position)
        values = read_vector_slice(dataset, position, length)
        finite_mask = np.isfinite(values)
        finite_count = int(finite_mask.sum())
        finite += finite_count
        non_finite += int(values.size - finite_count)
        if finite_count:
            good = values[finite_mask]
            minimum = min(minimum, float(good.min()))
            maximum = max(maximum, float(good.max()))
            sum_sq += float(np.sum(good.astype(np.float64) ** 2))
            sum_value += float(np.sum(good.astype(np.float64)))
        position += length
    scan = {
        "samples": total,
        "finite_samples": finite,
        "non_finite_samples": non_finite,
        "all_finite": non_finite == 0,
        "min": minimum if finite else None,
        "max": maximum if finite else None,
        "mean": (sum_value / finite) if finite else None,
        "mean_power": (sum_sq / finite) if finite else None,
    }
    if dataset.dtype.kind == "f" and int(dataset.dtype.itemsize) > 1:
        scan["byte_order"] = "little" if dataset.dtype.byteorder in ("<", "=", "|") else "big"
    return scan


def scan_file(path: Path, chunk: int) -> dict:
    log(f"扫描 {path.name} …")
    record: dict = {
        "filename": path.name,
        "file_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "checks": {},
    }
    with h5py.File(path, "r") as handle:
        keys = sorted(handle.keys())
        record["keys"] = keys
        record["metadata"] = collect_metadata(handle)
        missing = [key for key in REQUIRED_KEYS if key not in handle]
        record["missing_required"] = missing
        record["datasets"] = {
            key: {
                "shape": list(handle[key].shape),
                "dtype": str(handle[key].dtype),
                "ndim": handle[key].ndim,
                "chunks": list(handle[key].chunks) if handle[key].chunks else None,
                "compression": handle[key].compression,
            }
            for key in keys
            if isinstance(handle[key], h5py.Dataset)
        }
        if missing:
            raise RuntimeError(f"{path.name}: 缺少数据集 {missing}")

        lengths = {key: dataset_length(handle[key]) for key in REQUIRED_KEYS}
        record["channel_lengths"] = lengths
        record["source_samples"] = next(iter(lengths.values()))
        record["lengths_consistent"] = len(set(lengths.values())) == 1
        record["channel_scans"] = {
            key: scan_channel(handle[key], chunk) for key in REQUIRED_KEYS
        }
        record["file_attrs_read"] = True

    record["complete_blocks"] = record["source_samples"] // BLOCK_SIZE
    record["ignored_tail_samples"] = record["source_samples"] % BLOCK_SIZE
    record["windows_per_block"] = BLOCK_SIZE // WINDOW_SIZE
    record["indexed_windows"] = record["complete_blocks"] * record["windows_per_block"]
    record["non_finite_total"] = sum(
        item["non_finite_samples"] for item in record["channel_scans"].values()
    )
    record["checks"] = {
        "four_channels_present": True,
        "lengths_consistent": record["lengths_consistent"],
        "all_channels_all_finite": all(
            item["all_finite"] for item in record["channel_scans"].values()
        ),
        "file_size_matches_manifest": None,  # filled by caller
    }
    log(
        f"{path.name}: {record['source_samples']:,} 样本/通道，"
        f"{record['complete_blocks']} 块，{record['indexed_windows']} 窗，"
        f"非有限值 {record['non_finite_total']}"
    )
    return record


def build_rows(files: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in files:
        codes = parse_name(record["filename"])
        label = LABELS[codes["target_code"]]
        for block_index in range(record["complete_blocks"]):
            block_start = block_index * BLOCK_SIZE
            for window_in_block in range(record["windows_per_block"]):
                start = block_start + window_in_block * WINDOW_SIZE
                rows.append(
                    {
                        "file_path": record["file_path"],
                        "filename": record["filename"],
                        "target_code": codes["target_code"],
                        "target_name": TARGET_NAMES[codes["target_code"]],
                        "distance_code": codes["distance_code"],
                        "segment_code": codes["segment_code"],
                        "split": "heldout",
                        "label": label,
                        "block_index": block_index,
                        "window_index": window_in_block,
                        "start": start,
                        "length": WINDOW_SIZE,
                        "source_samples": record["source_samples"],
                    }
                )
    return rows


def audit_rows(rows: list[dict]) -> dict:
    seen: set[tuple[str, int, int]] = set()
    duplicates: list[str] = []
    boundary_errors: list[str] = []
    for row in rows:
        start = int(row["start"])
        length = int(row["length"])
        if start // BLOCK_SIZE != (start + length - 1) // BLOCK_SIZE:
            boundary_errors.append(f"{row['filename']}:{start}+{length} 跨块")
        if start % WINDOW_SIZE != 0:
            boundary_errors.append(f"{row['filename']}:{start} 未对齐窗口")
        key = (row["filename"], start, length)
        if key in seen:
            duplicates.append(f"{row['filename']}:{start}+{length}")
        seen.add(key)
    return {
        "windows": len(rows),
        "unique_windows": len(seen),
        "duplicate_windows": duplicates[:20],
        "duplicate_count": len(duplicates),
        "boundary_errors": boundary_errors[:20],
        "boundary_error_count": len(boundary_errors),
        "passed": not duplicates and not boundary_errors,
    }


def cross_check_old_index(rows: list[dict]) -> dict:
    old_files: set[str] = set()
    old_paths: set[str] = set()
    old_windows = 0
    if OLD_INDEX.is_file():
        import csv as _csv

        with OLD_INDEX.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in _csv.DictReader(handle):
                old_files.add(row["filename"])
                old_paths.add(str(Path(row["file_path"]).resolve()))
                old_windows += 1
    new_files = {row["filename"] for row in rows}
    new_paths = {str(Path(row["file_path"]).resolve()) for row in rows}
    return {
        "old_index_path": str(OLD_INDEX),
        "old_index_present": OLD_INDEX.is_file(),
        "old_index_windows": old_windows,
        "heldout_windows": len(rows),
        "filename_overlap": sorted(old_files & new_files),
        "file_path_overlap": sorted(old_paths & new_paths),
        "disjoint_files": not (old_files & new_files),
        "disjoint_paths": not (old_paths & new_paths),
        "old_segments": sorted(
            {name[:-4].split("_")[2] for name in old_files if name.lower().endswith(".mat")}
        ),
        "heldout_segments": sorted({"S" + row["segment_code"] for row in rows}),
    }


def frozen_evaluation_config() -> dict:
    return {
        "status": "frozen_before_reading_candidate_signals_or_model_scores",
        "source_document": "dronerfa_baseline/outputs/noise_ab_replicates_20260917/后续留出数据方案.md",
        "models_to_evaluate": [
            "training seed 2026 arm A selected checkpoint",
            "training seed 2026 arm B selected checkpoint",
            "training seed 2027 arm A selected checkpoint",
            "training seed 2027 arm B selected checkpoint",
            "training seed 2028 arm A selected checkpoint",
            "training seed 2028 arm B selected checkpoint",
        ],
        "frontend": "RF0 I/Q -> 10ms window -> mean removal -> RMS normalize -> complex AWGN "
        "-> Hamming STFT n_fft=1024 hop=512 -> log power -> fftshift -> per-sample normalize "
        "-> 1x256x256",
        "decision_rule": "raw logit difference > 0 => Drone (default threshold, unchanged)",
        "conditions_db": ["clean", -10, -12, -15],
        "noise_seed": 22026,
        "noise_application": "independent complex AWGN with seed 22026, identical window set and "
        "identical noise implementation shared by all six models",
        "primary_aggregation": "per airframe, equal weight per source file, counts retained",
        "secondary_aggregation": "pooled over all windows",
        "constraints": [
            "no threshold tuning on heldout data",
            "no epoch or training-noise-range re-selection on heldout data",
            "no model selection based on heldout results",
            "different source files are not by themselves evidence of independent capture sessions",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=HELDOUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--chunk", type=int, default=2_000_000, help="每通道每次读取样本数")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))["files"]
    expected = {Path(item["name"]).name: item for item in candidates}
    paths = sorted(args.data_root.glob("*.mat"))
    if len(paths) != len(expected):
        raise SystemExit(f"预期 {len(expected)} 个文件，实际找到 {len(paths)} 个")
    unknown = [path.name for path in paths if path.name not in expected]
    if unknown:
        raise SystemExit(f"目录中存在候选清单之外的文件：{unknown}")

    log(f"扫描 {len(paths)} 个文件，chunk={args.chunk:,}，workers={args.workers}")
    started = datetime.now(timezone.utc)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(lambda path: scan_file(path, args.chunk), paths))
    files.sort(key=lambda item: item["filename"])

    manifest_match = 0
    for record in files:
        item = expected[record["filename"]]
        matched = record["file_size_bytes"] == item["unpacked_size"]
        record["checks"]["file_size_matches_manifest"] = matched
        record["manifest_unpacked_size"] = item["unpacked_size"]
        record["manifest_block_offset"] = item["block_offset"]
        record["manifest_packed_size"] = item["packed_size"]
        manifest_match += int(matched)

    rows = build_rows(files)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "heldout_window_index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    audit = audit_rows(rows)
    cross = cross_check_old_index(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["target_code"]] = counts.get(row["target_code"], 0) + 1
    label_counts: dict[str, int] = {}
    for row in rows:
        key = f"label_{row['label']}"
        label_counts[key] = label_counts.get(key, 0) + 1
    files_per_target: dict[str, int] = {}
    for record in files:
        code = parse_name(record["filename"])["target_code"]
        files_per_target[code] = files_per_target.get(code, 0) + 1

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started.isoformat(),
        "data_root": str(args.data_root),
        "index_path": str(index_path),
        "window_size_samples": WINDOW_SIZE,
        "window_duration_ms_at_100MSps": WINDOW_SIZE / 100_000.0,
        "acquisition_block_size_samples": BLOCK_SIZE,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "split": "heldout",
        "total_files": len(files),
        "total_windows": len(rows),
        "windows_per_file": files[0]["indexed_windows"] if files else 0,
        "window_counts_by_target": counts,
        "window_counts_by_label": label_counts,
        "files_by_target": files_per_target,
        "manifest_size_matches": f"{manifest_match}/{len(files)}",
        "index_audit": audit,
        "cross_check_with_existing_index": cross,
        "files": files,
    }
    summary_path = args.output_dir / "heldout_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    records_path = args.output_dir / "download_records.json"
    download_records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.is_file() else {}
    all_metadata_absent = all(not record["metadata"]["has_any_metadata"] for record in files)
    files_with_metadata = [
        record["filename"] for record in files if record["metadata"]["has_any_metadata"]
    ]
    verification = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "prepare and verify the 12 unused DroneRFa source files",
        "data_root": str(args.data_root),
        "file_count": len(files),
        "total_bytes": sum(record["file_size_bytes"] for record in files),
        "file_sha256": {record["filename"]: record["file_sha256"] for record in files},
        "index_sha256": sha256_file(index_path),
        "index_path": str(index_path),
        "summary_sha256": sha256_file(summary_path),
        "window_count": len(rows),
        "window_counts_by_target": counts,
        "window_counts_by_label": label_counts,
        "acquisition_metadata": {
            "present_in_mat": not all_metadata_absent,
            "files_with_acquisition_metadata": files_with_metadata,
            "checked": True,
            "checked_scope": "HDF5 root attrs, per-dataset attrs (excluding the MATLAB_class "
            "format marker), and any datasets beyond RF0/RF1 I/Q",
            "note": "MAT容器内仅有 RF0_I/RF0_Q/RF1_I/RF1_Q 四个数据集，无采集中心频率、"
            "增益或天线等元数据；因此任何频率表述只能基于采样率与固定信号统计，"
            "不虚构绝对频段",
        },
        "checks": {
            "manifest_size_match": {
                "passed": manifest_match == len(files),
                "detail": f"{manifest_match}/{len(files)}",
            },
            "four_channels_present": {
                "passed": all(not record["missing_required"] for record in files),
                "detail": "RF0_I/RF0_Q/RF1_I/RF1_Q 全部存在",
            },
            "lengths_consistent": {
                "passed": all(record["lengths_consistent"] for record in files),
                "detail": {record["filename"]: record["source_samples"] for record in files},
            },
            "all_samples_finite": {
                "passed": all(record["non_finite_total"] == 0 for record in files),
                "detail": {
                    record["filename"]: record["non_finite_total"] for record in files
                },
                "scanned_samples_per_channel": files[0]["source_samples"] if files else 0,
                "note": "逐文件、逐通道完整扫描全部存储样本，非抽样",
            },
            "complete_blocks_and_windows": {
                "passed": all(
                    # Windows must equal complete_blocks * windows_per_block, derived
                    # from each file's own stored length.  Files legitimately differ
                    # (140M/150M/160M/180M samples), so no fixed count is assumed.
                    record["source_samples"] % BLOCK_SIZE == 0
                    and record["indexed_windows"]
                    == (record["source_samples"] // BLOCK_SIZE)
                    * (BLOCK_SIZE // WINDOW_SIZE)
                    and record["ignored_tail_samples"] == 0
                    for record in files
                ),
                "detail": {
                    record["filename"]: {
                        "complete_blocks": record["complete_blocks"],
                        "ignored_tail_samples": record["ignored_tail_samples"],
                        "indexed_windows": record["indexed_windows"],
                    }
                    for record in files
                },
                "rule": "每个 10,000,000 点块内切 10 个 1,000,000 点窗，不跨块、不重叠；"
                "窗数由各文件实际样本数推导",
            },
            "index_block_boundaries": {
                "passed": audit["passed"],
                "detail": audit,
            },
            "index_disjoint_from_existing": {
                "passed": cross["disjoint_files"] and cross["disjoint_paths"],
                "detail": cross,
            },
            "download_records_consistent": {
                "passed": bool(download_records)
                and len(download_records.get("files", {})) >= len(files),
                "detail": {
                    "records": len(download_records.get("files", {})),
                    "statuses": _status_counts(download_records),
                },
            },
        },
        "not_checked": [
            "采集会话/日期/地点独立性：元数据缺失，无法证明",
            "接收机设置详情（增益、天线）：MAT内无元数据",
            "跨文件为独立采集而非同批录制：无法证明",
        ],
        "frozen_evaluation_config": frozen_evaluation_config(),
    }
    verification_path = args.output_dir / "verification.json"
    verification_path.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log(f"索引：{index_path}（{len(rows)} 窗）")
    log(f"汇总：{summary_path}")
    log(f"核查：{verification_path}")
    log(f"大小匹配 {manifest_match}/{len(files)}；索引审计通过={audit['passed']}；"
        f"与旧索引文件交集={len(cross['filename_overlap'])}")
    return 0


def _status_counts(records: dict) -> dict:
    counts: dict[str, int] = {}
    for record in records.get("files", {}).values():
        status = record.get("status", "?")
        counts[status] = counts.get(status, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
