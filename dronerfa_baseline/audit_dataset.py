from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from dronerfa_data import load_index, read_vector_slice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the DroneRFa virtual index.")
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "window_index.csv",
    )
    parser.add_argument("--block-size", type=int, default=10_000_000)
    parser.add_argument("--checked-windows", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_index(args.index)
    if not rows:
        raise RuntimeError("Index is empty")

    unique_windows: set[tuple[str, int, int]] = set()
    file_splits: dict[str, set[str]] = defaultdict(set)
    boundary_errors: list[str] = []
    duplicate_errors: list[str] = []

    for row in rows:
        start = int(row["start"])
        length = int(row["length"])
        first_block = start // args.block_size
        last_block = (start + length - 1) // args.block_size
        if first_block != last_block:
            boundary_errors.append(
                f"{row['filename']}:{start}+{length} crosses a block boundary"
            )
        key = (str(row["filename"]), start, length)
        if key in unique_windows:
            duplicate_errors.append(f"{row['filename']}:{start}+{length}")
        unique_windows.add(key)
        file_splits[str(row["filename"])].add(str(row["split"]))

    split_leakage = {
        filename: sorted(splits)
        for filename, splits in file_splits.items()
        if len(splits) != 1
    }

    positions = np.linspace(
        0, len(rows) - 1, min(args.checked_windows, len(rows)), dtype=int
    )
    read_checks: list[dict[str, object]] = []
    for position in positions:
        row = rows[int(position)]
        with h5py.File(str(row["file_path"]), "r") as handle:
            start = int(row["start"])
            length = int(row["length"])
            i_values = read_vector_slice(handle["RF0_I"], start, length)
            q_values = read_vector_slice(handle["RF0_Q"], start, length)
        iq = i_values + 1j * q_values
        read_checks.append(
            {
                "filename": row["filename"],
                "start": start,
                "samples": len(iq),
                "finite": bool(np.isfinite(iq).all()),
                "mean_power": float(np.mean(np.abs(iq) ** 2, dtype=np.float64)),
            }
        )

    invalid_reads = [
        item
        for item in read_checks
        if not item["finite"] or int(item["samples"]) != int(rows[0]["length"])
    ]
    report = {
        "index": str(args.index.resolve()),
        "windows": len(rows),
        "source_files": len(file_splits),
        "unique_windows": len(unique_windows),
        "boundary_error_count": len(boundary_errors),
        "duplicate_error_count": len(duplicate_errors),
        "split_leakage": split_leakage,
        "checked_windows": len(read_checks),
        "invalid_read_count": len(invalid_reads),
        "passed": not (
            boundary_errors or duplicate_errors or split_leakage or invalid_reads
        ),
        "boundary_errors": boundary_errors[:10],
        "duplicate_errors": duplicate_errors[:10],
        "invalid_reads": invalid_reads[:10],
        "read_checks": read_checks,
    }
    output_path = args.index.with_name("audit_report.json")
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "read_checks"}, ensure_ascii=False, indent=2))
    print(f"Report: {output_path.resolve()}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
