from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import h5py

from dronerfa_data import SEGMENT_SPLITS, TARGET_NAMES, dataset_length


FILE_PATTERN = re.compile(
    r"^T(?P<target>\d{4})_D(?P<distance>\d{2})_S(?P<segment>\d{4})\.mat$",
    re.IGNORECASE,
)
REQUIRED_KEYS = ("RF0_I", "RF0_Q", "RF1_I", "RF1_Q")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a leakage-resistant virtual-window index for DroneRFa."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\Administrator\Documents\DroneRFa数据集"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "window_index.csv",
    )
    parser.add_argument("--window-size", type=int, default=1_000_000)
    parser.add_argument("--block-size", type=int, default=10_000_000)
    return parser.parse_args()


def inspect_file(path: Path) -> tuple[int, dict[str, list[int]]]:
    with h5py.File(path, "r") as handle:
        missing = [key for key in REQUIRED_KEYS if key not in handle]
        if missing:
            raise KeyError(f"{path.name}: missing datasets {missing}")
        shapes = {key: list(handle[key].shape) for key in REQUIRED_KEYS}
        lengths = {key: dataset_length(handle[key]) for key in REQUIRED_KEYS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{path.name}: channel lengths do not match: {lengths}")
        return next(iter(lengths.values())), shapes


def main() -> None:
    args = parse_args()
    if args.window_size <= 0 or args.block_size <= 0:
        raise ValueError("window-size and block-size must be positive")
    if args.block_size % args.window_size != 0:
        raise ValueError("block-size must be divisible by window-size")
    if not args.data_root.exists():
        raise FileNotFoundError(args.data_root)

    rows: list[dict[str, object]] = []
    files_summary: list[dict[str, object]] = []
    ignored: list[str] = []

    for path in sorted(args.data_root.glob("*.mat")):
        match = FILE_PATTERN.match(path.name)
        if match is None:
            ignored.append(path.name)
            continue

        target = match.group("target")
        distance = match.group("distance")
        segment = match.group("segment")
        if target not in TARGET_NAMES or segment not in SEGMENT_SPLITS:
            ignored.append(path.name)
            continue

        source_samples, shapes = inspect_file(path)
        complete_blocks = source_samples // args.block_size
        windows_per_block = args.block_size // args.window_size
        split = SEGMENT_SPLITS[segment]
        label = 0 if target == "0000" else 1

        for block_index in range(complete_blocks):
            block_start = block_index * args.block_size
            for window_in_block in range(windows_per_block):
                start = block_start + window_in_block * args.window_size
                rows.append(
                    {
                        "file_path": str(path.resolve()),
                        "filename": path.name,
                        "target_code": target,
                        "target_name": TARGET_NAMES[target],
                        "distance_code": distance,
                        "segment_code": segment,
                        "split": split,
                        "label": label,
                        "block_index": block_index,
                        "window_index": window_in_block,
                        "start": start,
                        "length": args.window_size,
                        "source_samples": source_samples,
                    }
                )

        files_summary.append(
            {
                "filename": path.name,
                "target_code": target,
                "target_name": TARGET_NAMES[target],
                "distance_code": distance,
                "segment_code": segment,
                "split": split,
                "label": label,
                "source_samples": source_samples,
                "complete_blocks": complete_blocks,
                "indexed_windows": complete_blocks * windows_per_block,
                "ignored_tail_samples": source_samples
                - complete_blocks * args.block_size,
                "datasets": shapes,
            }
        )

    if not rows:
        raise RuntimeError(f"No supported .mat files found under {args.data_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    split_counts = Counter(str(row["split"]) for row in rows)
    label_counts = Counter(
        f"{row['split']}:label_{row['label']}" for row in rows
    )
    summary = {
        "data_root": str(args.data_root.resolve()),
        "index_path": str(args.output.resolve()),
        "window_size_samples": args.window_size,
        "window_duration_ms_at_100MSps": args.window_size / 100_000.0,
        "acquisition_block_size_samples": args.block_size,
        "total_files": len(files_summary),
        "total_windows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "split_label_counts": dict(sorted(label_counts.items())),
        "ignored_files": ignored,
        "files": files_summary,
    }
    summary_path = args.output.with_name("dataset_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Index: {args.output.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Files: {len(files_summary)}, windows: {len(rows)}")
    print(f"Splits: {dict(sorted(split_counts.items()))}")
    print(f"Split/labels: {dict(sorted(label_counts.items()))}")


if __name__ == "__main__":
    main()
