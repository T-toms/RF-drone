#!/usr/bin/env python3
"""Read-only RF0/RF1 amplitude audit for the T1011 confusion cluster.

The audit samples fixed positions in each source MAT. It never changes labels,
split assignments or model outputs. Development-test samples are diagnostic.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "outputs/stage1_12drones_20260926/window_index_stage1_12drones.csv"
OUT = ROOT / "outputs/qa_t1011_20260930"
CODES = {"T1010", "T1011", "T1100"}
SNIPPET = 65536
POSITIONS = 12


def signal(handle: h5py.File, key: str, start: int) -> np.ndarray:
    ds = handle[key]
    if ds.ndim == 2 and ds.shape[0] == 1:
        return np.asarray(ds[0, start:start + SNIPPET], dtype=np.float64)
    if ds.ndim == 1:
        return np.asarray(ds[start:start + SNIPPET], dtype=np.float64)
    raise ValueError(f"Unexpected shape for {key}: {ds.shape}")


def main() -> None:
    files = {}
    with INDEX.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["target_code"] in CODES:
                files[row["filename"]] = {
                    "filename": row["filename"], "target_code": row["target_code"],
                    "split": row["split"], "path": row["file_path"],
                    "index_sha256": row["file_sha256"],
                }
    if len(files) != 24:
        raise ValueError(f"Expected 24 source files, got {len(files)}")
    samples = []
    summary = []
    for file in sorted(files.values(), key=lambda x: x["filename"]):
        with h5py.File(file["path"], "r") as handle:
            length = min(int(handle[k].shape[-1]) for k in ("RF0_I", "RF0_Q", "RF1_I", "RF1_Q"))
            starts = np.linspace(0, length - SNIPPET, POSITIONS, dtype=np.int64)
            for position, start in enumerate(starts):
                rms = {}
                for channel in ("RF0", "RF1"):
                    i = signal(handle, channel + "_I", int(start))
                    q = signal(handle, channel + "_Q", int(start))
                    if len(i) != SNIPPET or len(q) != SNIPPET or not np.isfinite(i).all() or not np.isfinite(q).all():
                        raise ValueError(f"Invalid IQ: {file['filename']} {channel} {start}")
                    rms[channel] = float(np.sqrt(np.mean(i*i + q*q)))
                samples.append({
                    "filename": file["filename"], "target_code": file["target_code"],
                    "split": file["split"], "position": position, "start": int(start),
                    "rf0_rms": rms["RF0"], "rf1_rms": rms["RF1"],
                    "rf0_rf1_rms_ratio": rms["RF0"] / rms["RF1"] if rms["RF1"] else None,
                })
        group = [s for s in samples if s["filename"] == file["filename"]]
        summary.append({
            "filename": file["filename"], "target_code": file["target_code"],
            "split": file["split"], "source_samples": length,
            "index_sha256": file["index_sha256"],
            "rf0_rms_median": float(np.median([s["rf0_rms"] for s in group])),
            "rf1_rms_median": float(np.median([s["rf1_rms"] for s in group])),
            "rf0_rf1_ratio_median": float(np.median([s["rf0_rf1_rms_ratio"] for s in group])),
            "rf0_rms_min": min(s["rf0_rms"] for s in group),
            "rf0_rms_max": max(s["rf0_rms"] for s in group),
        })
        print(f"{file['filename']} {file['split']} RF0={summary[-1]['rf0_rms_median']:.6g} ratio={summary[-1]['rf0_rf1_ratio_median']:.3f}", flush=True)
    train = [r for r in summary if r["target_code"] == "T1011" and r["split"] == "train"]
    reference = {}
    for feature in ("rf0_rms_median", "rf1_rms_median", "rf0_rf1_ratio_median"):
        values = np.log([r[feature] for r in train])
        reference[feature] = {"train_log_median": float(np.median(values)),
                              "train_log_mad": float(np.median(np.abs(values - np.median(values))))}
        for row in summary:
            if row["target_code"] == "T1011":
                mad = reference[feature]["train_log_mad"]
                row[feature + "_relative_to_t1011_train_robust_z"] = (
                    float((np.log(row[feature]) - reference[feature]["train_log_median"]) / (1.4826 * mad))
                    if mad > 0 else None
                )
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("sampled_snippets.csv", samples), ("file_summary.csv", summary)):
        with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    (OUT / "method.json").write_text(json.dumps({
        "sample_count_per_file": POSITIONS, "snippet_samples": SNIPPET,
        "sample_start_rule": "12 evenly spaced starts across full recording, inclusive endpoints",
        "features": "complex RMS=sqrt(mean(I^2+Q^2)); median over snippets; RF0/RF1 ratio is not physical SNR",
        "t1011_reference": "five train source files only; log median and MAD; robust z descriptive due small n",
        "test_use": "diagnostic only; no label or checkpoint selection",
        "reference": reference,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
