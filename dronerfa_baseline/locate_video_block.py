#!/usr/bin/env python3
"""Read-only: per 1M-sample window, locate the wideband (video/OFDM) block in RF0 and RF1.

Uses the full 10 ms window (unlike audit_band_switch.py, which reads 0.66 ms).
For each frequency bin the duty cycle is the fraction of STFT frames more than
THRESH_DB above that bin's estimated noise mean. After smoothing over SMOOTH_MHZ, contiguous runs
with duty > MIN_DUTY and width >= MIN_WIDTH_MHZ are candidates; the run with the
largest duty x width is reported. Narrow control hops (~1-2 MHz, ~0.5 ms) fall
below these limits by construction.

Optionally joins the frozen cascade12 per-window predictions (clean condition)
for development-test files. Diagnostic only: nothing here selects a model.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
PRED = ROOT / "outputs/cascade12_development_20260927/window_predictions.csv"
WINDOW = 1_000_000
NFFT = 1024
FS_MHZ = 100.0
CENTRE = {"RF0": 2440.0, "RF1": 5800.0}
THRESH_DB = 5.0
SMOOTH_MHZ = 3.0
MIN_DUTY = 0.15
MIN_WIDTH_MHZ = 5.0
FLOOR_MHZ = 41.0
NOISE_P10_DB = 9.8         # 10*log10(1/-ln(0.9))
STAGE2_CODES = ("T0001", "T0010", "T0011", "T0100", "T0101", "T0110", "T0111",
                "T1000", "T1001", "T1010", "T1011", "T1100")


def read(handle: h5py.File, key: str, start: int) -> np.ndarray:
    ds = handle[key]
    return np.asarray(ds[0, start:start + WINDOW] if ds.ndim == 2 else ds[start:start + WINDOW], dtype=np.float32)


def locate(handle: h5py.File, ch: str, start: int) -> dict:
    x = read(handle, ch + "_I", start) + 1j * read(handle, ch + "_Q", start)
    frames = x[: (len(x) // NFFT) * NFFT].reshape(-1, NFFT) * np.hamming(NFFT).astype(np.float32)
    db = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)) ** 2 + 1e-20)
    # The receiver floor is not flat (RF1 is ~3 dB higher mid-band), so estimate it per
    # bin: 10th percentile over frames, median-filtered over FLOOR_MHZ so that a block
    # occupying a bin in >90% of frames does not lift its own floor. For exponential
    # noise power the 10th percentile sits NOISE_P10_DB below the mean.
    p10 = np.percentile(db, 10, axis=0)
    half = int(FLOOR_MHZ / FS_MHZ * NFFT) // 2
    padded = np.pad(p10, half, mode="edge")
    floor = np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1), axis=1)
    duty = (db > floor + NOISE_P10_DB + THRESH_DB).mean(axis=0)
    k = max(1, int(SMOOTH_MHZ / FS_MHZ * NFFT))
    duty = np.convolve(duty, np.ones(k) / k, mode="same")
    freqs = CENTRE[ch] + (np.arange(NFFT) - NFFT / 2) * FS_MHZ / NFFT
    edge = int(3 / FS_MHZ * NFFT)          # ignore the anti-alias roll-off at both band edges
    mask = duty > MIN_DUTY
    mask[:edge] = mask[-edge:] = False
    best = {f"{ch}_block_centre": "", f"{ch}_block_width": 0.0, f"{ch}_block_duty": 0.0}
    score, i = 0.0, 0
    while i < NFFT:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < NFFT and mask[j]:
            j += 1
        width = (j - i) * FS_MHZ / NFFT
        d = float(duty[i:j].mean())
        if width >= MIN_WIDTH_MHZ and width * d > score:
            score = width * d
            best = {f"{ch}_block_centre": round(float(freqs[(i + j) // 2]), 1),
                    f"{ch}_block_width": round(width, 1), f"{ch}_block_duty": round(d, 3)}
        i = j
    return best


def load_predictions() -> dict:
    out = {}
    if not PRED.exists():
        return out
    with PRED.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["condition"] == "clean":
                code = STAGE2_CODES[int(r["stage2_prediction"])] if r["stage2_prediction"] else ""
                out[(r["filename"], int(r["window_index"]))] = {
                    "stage1_pred": r["stage1_prediction"], "stage2_pred": code,
                    "stage2_correct": int(code == r["target_code"]) if code else ""}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--files", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    preds = load_predictions()
    rows = []
    for fn in [f.strip() for f in args.files.split(",") if f.strip()]:
        with h5py.File(args.data_root / fn, "r") as h:
            n = min(int(h[k].shape[-1]) for k in ("RF0_I", "RF0_Q", "RF1_I", "RF1_Q"))
            for w in range(n // WINDOW):
                row = {"file": fn, "window": w}
                for ch in ("RF0", "RF1"):
                    row.update(locate(h, ch, w * WINDOW))
                row.update(preds.get((fn, w), {"stage1_pred": "", "stage2_pred": "", "stage2_correct": ""}))
                rows.append(row)
        fr = [r for r in rows if r["file"] == fn]
        rf0 = [r["RF0_block_centre"] for r in fr if r["RF0_block_centre"] != ""]
        rf1 = [r["RF1_block_centre"] for r in fr if r["RF1_block_centre"] != ""]
        print(f"{fn}: {len(fr)} win | RF0 block in {len(rf0)} win, median centre "
              f"{np.median(rf0) if rf0 else '-'} | RF1 block in {len(rf1)} win, median centre "
              f"{np.median(rf1) if rf1 else '-'}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("written", args.output)


if __name__ == "__main__":
    main()
