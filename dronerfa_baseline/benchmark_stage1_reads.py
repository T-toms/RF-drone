#!/usr/bin/env python3
"""Bounded RF0 read benchmark over the frozen split (diagnostic, read-only).

Measures the per-window read cost of the frozen reader over the actual train and
val files: one warm pass over a single file, then one first-touch pass per file,
with a hard time budget so the probe never runs long.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INDEX = SCRIPT_DIR / "outputs" / "stage1_12drones_20260926" / "window_index_stage1_12drones.csv"


def read_channel(dataset: h5py.Dataset, start: int, length: int) -> np.ndarray:
    if dataset.ndim == 1:
        values = dataset[start : start + length]
    elif dataset.ndim == 2 and dataset.shape[0] == 1:
        values = dataset[0, start : start + length]
    else:
        values = dataset[start : start + length, 0]
    return np.asarray(values, dtype=np.float32)


def timed_read(path: str, start: int) -> dict:
    with h5py.File(path, "r") as handle:
        begin = time.perf_counter()
        real = read_channel(handle["RF0_I"], start, 1_000_000)
        imag = read_channel(handle["RF0_Q"], start, 1_000_000)
        elapsed = time.perf_counter() - begin
    return {
        "seconds": elapsed,
        "checksum": float(real.sum(dtype=np.float64)) + float(imag.sum(dtype=np.float64)),
    }


def main() -> int:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    rows = list(csv.DictReader(INDEX.open(encoding="utf-8-sig")))
    by_role: dict[str, dict[str, list[dict[str, str]]]] = {"train": {}, "val": {}}
    for row in rows:
        role = str(row["source_role"])
        if role in by_role:
            by_role[role].setdefault(str(row["filename"]), []).append(row)

    report: dict[str, object] = {"budget_seconds": budget, "roles": {}}
    started = time.perf_counter()
    for role, files in by_role.items():
        per_file: list[dict[str, object]] = []
        for filename, file_rows in files.items():
            if time.perf_counter() - started > budget:
                break
            first = timed_read(str(file_rows[0]["file_path"]), 0)
            middle_row = file_rows[len(file_rows) // 2]
            second = timed_read(str(middle_row["file_path"]), int(middle_row["start"]))
            per_file.append(
                {
                    "filename": filename,
                    "first_seconds": round(first["seconds"], 4),
                    "second_seconds": round(second["seconds"], 4),
                }
            )
        seconds = [row["first_seconds"] for row in per_file] + [
            row["second_seconds"] for row in per_file
        ]
        report["roles"][role] = {
            "files_measured": len(per_file),
            "mean_seconds": round(float(np.mean(seconds)), 4) if seconds else None,
            "median_seconds": round(float(np.median(seconds)), 4) if seconds else None,
            "max_seconds": round(float(np.max(seconds)), 4) if seconds else None,
            "per_file": per_file,
        }
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
