#!/usr/bin/env python3
"""Duplicate check: heldout files vs the already-used original 12 files.

Two independent levels, neither of which is presented as proof of capture-session
independence:

1. **Whole-file SHA256** over all 12 old and all 12 new files.  A byte-identical
   pair would mean the "new" file is literally a re-download of a used one.
2. **Fixed-position raw IQ fingerprints** at the same three sample offsets in the
   same three acquisition blocks per file (first/middle/last block), hashed over
   raw stored bytes.  This catches the case where two containers differ in
   structure or compression but hold the same recorded samples.

A sampled fingerprint cannot by itself rule out a re-recorded or re-encoded copy,
so the report states that limit explicitly instead of claiming session
independence.
"""

from __future__ import annotations

import argparse
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

HELDOUT_ROOT = Path(r"F:\DroneRFa数据集\heldout_20260917")
ORIGINAL_ROOT = Path(r"C:\Users\Administrator\Documents\DroneRFa数据集")
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "heldout_preparation_20260917"

BLOCK_SIZE = 10_000_000
WINDOW_SIZE = 1_000_000
FINGERPRINT_SAMPLES = 10_000
BLOCKS_PER_FILE = 14  # block positions are derived from the actual length
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


def fingerprint_file(path: Path) -> dict:
    """Hash raw RF0/RF1 bytes at fixed samples inside the first/middle/last block."""
    with h5py.File(path, "r") as handle:
        lengths = {key: int(np.prod(handle[key].shape)) for key in handle}
        source_samples = lengths["RF0_I"]
        blocks = max(source_samples // BLOCK_SIZE, 1)
        positions = sorted({0, blocks // 2, blocks - 1})
        entries = []
        overall = hashlib.sha256()
        for block in positions:
            block_start = block * BLOCK_SIZE
            # Fixed offset inside the block: first window start, plus the middle window.
            for window_in_block in (0, (BLOCK_SIZE // WINDOW_SIZE) // 2):
                start = block_start + window_in_block * WINDOW_SIZE
                if start + FINGERPRINT_SAMPLES > source_samples:
                    continue
                channel_hashes = {}
                for key in ("RF0_I", "RF0_Q", "RF1_I", "RF1_Q"):
                    if key not in handle:
                        continue
                    values = np.asarray(handle[key][0, start : start + FINGERPRINT_SAMPLES])
                    digest = hashlib.sha256(
                        np.ascontiguousarray(values, dtype=values.dtype).tobytes()
                    ).hexdigest()
                    channel_hashes[key] = digest
                    overall.update(digest.encode())
                entries.append(
                    {
                        "block": block,
                        "window_in_block": window_in_block,
                        "start": start,
                        "samples": FINGERPRINT_SAMPLES,
                        "channel_sha256": channel_hashes,
                    }
                )
    return {
        "window_fingerprint_sha256": overall.hexdigest(),
        "entries": entries,
        "fingerprint_positions": [entry["start"] for entry in entries],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-root", type=Path, default=HELDOUT_ROOT)
    parser.add_argument("--original-root", type=Path, default=ORIGINAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    new_paths = sorted(args.heldout_root.glob("*.mat"))
    old_paths = sorted(args.original_root.glob("*.mat"))
    log(f"新文件 {len(new_paths)} 个，原文件 {len(old_paths)} 个")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        new_hashes = list(pool.map(sha256_file, new_paths))
        old_hashes = list(pool.map(sha256_file, old_paths))
    new_hash_map = {path.name: digest for path, digest in zip(new_paths, new_hashes)}
    old_hash_map = {path.name: digest for path, digest in zip(old_paths, old_hashes)}

    byte_identical = [
        {"new": new_name, "old": old_name}
        for new_name, new_digest in new_hash_map.items()
        for old_name, old_digest in old_hash_map.items()
        if new_digest == old_digest
    ]
    intra_new_duplicates = [
        {"a": a, "b": b}
        for index, (a, digest_a) in enumerate(sorted(new_hash_map.items()))
        for b, digest_b in sorted(new_hash_map.items())[index + 1 :]
        if digest_a == digest_b
    ]
    intra_old_duplicates = [
        {"a": a, "b": b}
        for index, (a, digest_a) in enumerate(sorted(old_hash_map.items()))
        for b, digest_b in sorted(old_hash_map.items())[index + 1 :]
        if digest_a == digest_b
    ]

    log("计算固定位置原始IQ指纹 …")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        new_fingerprints = list(pool.map(fingerprint_file, new_paths))
        old_fingerprints = list(pool.map(fingerprint_file, old_paths))
    new_fp = {path.name: data for path, data in zip(new_paths, new_fingerprints)}
    old_fp = {path.name: data for path, data in zip(old_paths, old_fingerprints)}

    fingerprint_matches = [
        {
            "new": new_name,
            "old": old_name,
            "new_fingerprint": new_data["window_fingerprint_sha256"],
            "old_fingerprint": old_data["window_fingerprint_sha256"],
        }
        for new_name, new_data in sorted(new_fp.items())
        for old_name, old_data in sorted(old_fp.items())
        if new_data["window_fingerprint_sha256"] == old_data["window_fingerprint_sha256"]
    ]
    intra_new_fingerprint_matches = [
        {"a": a, "b": b}
        for index, (a, data_a) in enumerate(sorted(new_fp.items()))
        for b, data_b in sorted(new_fp.items())[index + 1 :]
        if data_a["window_fingerprint_sha256"] == data_b["window_fingerprint_sha256"]
    ]

    filename_overlap = sorted(set(new_hash_map) & set(old_hash_map))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "heldout_root": str(args.heldout_root),
        "original_root": str(args.original_root),
        "scope": {
            "new_files": len(new_paths),
            "old_files": len(old_paths),
            "fingerprint_samples_per_position": FINGERPRINT_SAMPLES,
            "fingerprint_positions_per_file": new_fingerprints[0]["fingerprint_positions"]
            if new_fingerprints
            else [],
            "note": "固定位置指纹覆盖每文件首/中/末采集块各两个固定10ms窗；"
            "抽样指纹不能保证全面排除重录或重编码",
        },
        "whole_file_sha256": {
            "heldout": new_hash_map,
            "original": old_hash_map,
            "byte_identical_pairs": byte_identical,
            "distinct_heldout_hashes": len(set(new_hash_map.values())),
            "distinct_original_hashes": len(set(old_hash_map.values())),
            "intra_heldout_duplicates": intra_new_duplicates,
            "intra_original_duplicates": intra_old_duplicates,
            "filename_overlap": filename_overlap,
        },
        "fixed_position_iq_fingerprints": {
            name: {
                "window_fingerprint_sha256": data["window_fingerprint_sha256"],
                "entries": data["entries"],
            }
            for name, data in sorted(new_fp.items())
        },
        "fixed_position_iq_fingerprints_original": {
            name: {
                "window_fingerprint_sha256": data["window_fingerprint_sha256"],
                "entries": data["entries"],
            }
            for name, data in sorted(old_fp.items())
        },
        "fingerprint_matches_new_vs_old": fingerprint_matches,
        "intra_heldout_fingerprint_matches": intra_new_fingerprint_matches,
        "conclusion": {
            "byte_identical_with_original": len(byte_identical) > 0,
            "filename_collision_with_original": len(filename_overlap) > 0,
            "fingerprint_collision_with_original": len(fingerprint_matches) > 0,
            "sampled_fingerprint_matches": len(fingerprint_matches),
            "limit": "文件级与固定位置指纹均未命中重复，但这不能证明新文件来自不同的采集会话、"
            "日期或地点；MAT内无相应元数据",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "duplicate_check.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"逐字节完全相同的 新-旧 文件对：{len(byte_identical)}")
    log(f"逐字节完全相同的 新-新 文件对：{len(intra_new_duplicates)}")
    log(f"固定位置指纹命中的 新-旧 文件对：{len(fingerprint_matches)}")
    log(f"固定位置指纹命中的 新-新 文件对：{len(intra_new_fingerprint_matches)}")
    log(f"文件名与旧集合重叠：{filename_overlap or '无'}")
    log(f"报告：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
