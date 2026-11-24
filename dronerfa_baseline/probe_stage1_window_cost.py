#!/usr/bin/env python3
"""Break the stage-1 window pipeline into its parts and time each one.

``benchmark_stage1_io.py`` answers "which worker count is fastest".  It does not
answer *why*, and the answer matters here: the frozen ``RF0_I``/``RF0_Q`` datasets
are ``float64`` with ``chunks=(1, 8191)`` and ``gzip`` level 3, so one 1,000,000
sample window is 16 MB of raw float64 spread over ~122 independently gzipped chunks
per channel.  Decompression happens on the CPU inside HDF5, which is exactly the
work a reader thread cannot overlap with GPU compute.

This probe measures, on the same frozen index and the same window order:

* ``hdf5_only``   -- read the two channels, no arithmetic (decompression + copy);
* ``rms``         -- + mean removal and RMS normalisation;
* ``rms_awgn``    -- + the frozen complex-AWGN draw (what the loader returns);
* ``frontend``    -- + the Hamming STFT log-power image;
* ``model``       -- + a ResNet18 forward under fp16 autocast.

and it also contrasts a scattered window order with a contiguous one, because a
shuffled epoch jumps ~150 MB through a gzip stream per window while a sequential
read walks the chunks in order.

Read-only, bounded, and it never starts a training run.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / "outputs" / ".matplotlib"))

import h5py
import numpy as np
import torch

from dronerfa_data import (
    DroneRFaWindowDataset,
    add_complex_awgn,
    parse_snr_list,
    read_vector_slice,
    rms_normalize,
)
from model import ResNet18, SpectrogramFrontEnd

OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "stage1_12drones_20260926"
INDEX = OUTPUT_ROOT / "window_index_stage1_12drones.csv"
IO_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "stage1_io_optimization_20260927"
TRAIN_ROLE = "train"
SEED = 2027
TRAIN_SNRS = parse_snr_list("clean,10,5,0,-5,-10,-12")
WINDOW_SAMPLES = 1_000_000
N_FFT, HOP_LENGTH, IMAGE_SIZE = 512, 256, 256
STAGES = ("hdf5_only", "rms", "rms_awgn", "frontend", "model")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dataset_layout(path: Path) -> dict[str, Any]:
    """Record the on-disk layout that drives the cost."""

    with h5py.File(path, "r") as handle:
        channels = {}
        for name in handle:
            item = handle[name]
            channels[name] = {
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "chunks": list(item.chunks) if item.chunks else None,
                "compression": item.compression,
                "compression_opts": item.compression_opts,
                "chunks_per_window": (
                    int(np.ceil(WINDOW_SAMPLES / item.chunks[1])) if item.chunks else None
                ),
            }
    return channels


def pick_windows(rows: Sequence[dict[str, Any]], count: int, layout: str) -> list[int]:
    """Positions to read: spread across files, or a contiguous run inside one file."""

    if layout == "contiguous":
        by_file: dict[str, list[int]] = {}
        for position, row in enumerate(rows):
            by_file.setdefault(str(row["file_path"]), []).append(position)
        # The file with the most windows, read from its start.
        best = max(by_file.values(), key=len)
        return best[:count]
    stride = max(1, len(rows) // count)
    return [min(index * stride, len(rows) - 1) for index in range(count)]


def measure(
    rows: Sequence[dict[str, Any]],
    positions: Sequence[int],
    awgn_db: float | None,
    seed: int,
    n_threads: int,
) -> dict[str, Any]:
    """Run every stage over the same windows and return per-stage seconds."""

    torch.set_num_threads(n_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frontend = SpectrogramFrontEnd(n_fft=N_FFT, hop_length=HOP_LENGTH, image_size=IMAGE_SIZE).to(device)
    frontend.eval()
    torch.manual_seed(seed)
    model = ResNet18(input_channels=1, num_classes=2).to(device)
    model.eval()

    totals: dict[str, float] = {stage: 0.0 for stage in STAGES}
    handles: dict[str, h5py.File] = {}
    bytes_read = 0
    try:
        with torch.inference_mode():
            for order_index, position in enumerate(positions):
                row = rows[position]
                path = str(row["file_path"])
                if path not in handles:
                    handles[path] = h5py.File(path, "r")
                handle = handles[path]
                start = int(row["start"])
                length = int(row["length"])

                mark = time.perf_counter()
                real = read_vector_slice(handle["RF0_I"], start, length)
                imag = read_vector_slice(handle["RF0_Q"], start, length)
                iq = (real + 1j * imag).astype(np.complex64)
                totals["hdf5_only"] += time.perf_counter() - mark
                bytes_read += int(real.nbytes + imag.nbytes)

                mark = time.perf_counter()
                iq = rms_normalize(iq)
                totals["rms"] += time.perf_counter() - mark

                mark = time.perf_counter()
                if awgn_db is not None:
                    rng = np.random.default_rng(seed + 1 * len(rows) + position)
                    selection = TRAIN_SNRS[int(rng.integers(0, len(TRAIN_SNRS)))]
                    if selection is not None:
                        iq = add_complex_awgn(iq, selection, rng)
                elif order_index % 4 == 0:
                    rng = np.random.default_rng(seed + position)
                    iq = add_complex_awgn(iq, -10.0, rng)
                totals["rms_awgn"] += time.perf_counter() - mark

                mark = time.perf_counter()
                tensor = torch.from_numpy(np.stack((iq.real, iq.imag), axis=0)).unsqueeze(0)
                features = frontend(tensor.to(device))
                if device.type == "cuda":
                    torch.cuda.synchronize()
                totals["frontend"] += time.perf_counter() - mark

                mark = time.perf_counter()
                with torch.amp.autocast(
                    device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16
                ):
                    logits = model(features)
                float(logits.float().sum())
                if device.type == "cuda":
                    torch.cuda.synchronize()
                totals["model"] += time.perf_counter() - mark
    finally:
        for handle in handles.values():
            handle.close()
    del frontend, model

    windows = len(positions)
    cumulative = 0.0
    stages: dict[str, Any] = {}
    for stage in STAGES:
        cumulative += totals[stage]
        stages[stage] = {
            "seconds": totals[stage],
            "ms_per_window": totals[stage] / windows * 1000.0,
            "cumulative_ms_per_window": cumulative / windows * 1000.0,
        }
    return {
        "windows": windows,
        "torch_threads": n_threads,
        "device": str(device),
        "host_bytes_read": bytes_read,
        "host_MB_per_second": bytes_read / (1 << 20) / max(cumulative, 1e-9),
        "stages": stages,
        "total_seconds": cumulative,
        "windows_per_second_full": windows / max(cumulative, 1e-9),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-stage cost of one stage-1 window.")
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument(
        "--layout",
        choices=("spread", "contiguous", "both"),
        default="both",
        help="spread = epoch-like jumps across files; contiguous = sequential in one file",
    )
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--output-dir", type=Path, default=IO_OUTPUT_ROOT)
    parser.add_argument(
        "--layer0",
        default=None,
        help="MAT path to report the HDF5 layout of (default: the first train file)",
    )
    args = parser.parse_args(argv)

    dataset = DroneRFaWindowDataset(INDEX, split=TRAIN_ROLE, train_snrs=TRAIN_SNRS, seed=SEED)
    rows = dataset.rows
    dataset.close()
    if args.layer0 is None:
        first = rows[0]
        layer0 = dataset_layout(Path(str(first["file_path"])))
    else:
        layer0 = dataset_layout(Path(args.layer0))

    threads = args.threads
    layouts = ("spread", "contiguous") if args.layout == "both" else (args.layout,)
    results: dict[str, Any] = {}
    for layout in layouts:
        positions = pick_windows(rows, args.windows, layout)
        if threads is None:
            entry = measure(rows, positions, awgn_db=None, seed=SEED, n_threads=torch.get_num_threads())
        else:
            entry = measure(rows, positions, awgn_db=None, seed=SEED, n_threads=threads)
        entry["positions"] = positions
        entry["files"] = sorted({str(rows[p]["filename"]) for p in positions})
        results[layout] = entry
        print(
            f"[{layout}] {entry['windows']} windows: "
            + " ".join(
                f"{stage}={entry['stages'][stage]['ms_per_window']:.1f}ms"
                for stage in STAGES
            )
            + f" | total={entry['windows_per_second_full']:.2f} win/s",
            flush=True,
        )

    payload = {
        "schema": "dronerfa-stage1-window-cost-breakdown/v1",
        "generated_at": utc_now(),
        "index": str(INDEX),
        "windows_requested": args.windows,
        "torch_threads": torch.get_num_threads() if threads is None else threads,
        "hdf5_layout": layer0,
        "layout_note": (
            "RF0_I/RF0_Q are float64 with chunks=(1, 8191) and gzip level 3, so one "
            "1,000,000-sample window is ~122 independently gzipped chunks per channel "
            "and every read pays CPU-side zlib decompression"
        ),
        "results": results,
    }
    write_json(args.output_dir / "window_cost_breakdown.json", payload)
    print(json.dumps({k: v["stages"] for k, v in results.items()}, ensure_ascii=False, indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
