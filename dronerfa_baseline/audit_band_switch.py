#!/usr/bin/env python3
"""Read-only band-occupancy audit: is the T1011 S0111 failure a 2.4 -> 5.8 GHz switch?

DroneRFa paper: RF0 is centred at 2440 MHz, RF1 at 5800 MHz (100 MS/s each);
segments S0000-S0111 start on 2.4 GHz, S1000-S1111 switch band where supported.
The models only read RF0, so a drone moving its link to 5.8 GHz becomes invisible.

For every 1M-sample window (same grid as the training index) this measures, per
channel, RMS and the fraction of 2048-point FFT bins above the noise floor + 6 dB.
Nothing is written outside --output. Development-test files are diagnostic only.

Usage:
  .venv\\Scripts\\python.exe dronerfa_baseline\\audit_band_switch.py --data-root <dir with MATs>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_FILES = [f"{t}_{s}.mat" for t in ("T1010", "T1011", "T1100")
                 for s in ("S0000", "S0100", "S0101", "S0110", "S0111", "S1000", "S1001")]
WINDOW = 1_000_000
SNIP = 2048 * 32          # samples analysed at the start of each window
NFFT = 2048
FS_MHZ = 100.0
CENTRE = {"RF0": 2440.0, "RF1": 5800.0}


def read(handle: h5py.File, key: str, start: int, n: int) -> np.ndarray:
    ds = handle[key]
    return np.asarray(ds[0, start:start + n] if ds.ndim == 2 else ds[start:start + n], dtype=np.float64)


def window_stats(handle: h5py.File, ch: str, start: int) -> tuple[float, np.ndarray]:
    x = read(handle, ch + "_I", start, SNIP) + 1j * read(handle, ch + "_Q", start, SNIP)
    frames = x[: (len(x) // NFFT) * NFFT].reshape(-1, NFFT) * np.hamming(NFFT)
    psd = np.fft.fftshift(np.mean(np.abs(np.fft.fft(frames, axis=1)) ** 2, axis=0))
    return float(np.sqrt(np.mean(np.abs(x) ** 2))), 10 * np.log10(psd + 1e-30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--files", default=",".join(DEFAULT_FILES))
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/band_switch_audit_20261010")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows, psds, missing = [], {}, []
    for fn in [f.strip() for f in args.files.split(",") if f.strip()]:
        path = args.data_root / fn
        if not path.exists():
            missing.append(fn)
            print("missing", fn)
            continue
        with h5py.File(path, "r") as h:
            n = min(int(h[k].shape[-1]) for k in ("RF0_I", "RF0_Q", "RF1_I", "RF1_Q"))
            acc = {}
            for w in range(n // WINDOW):
                row = {"file": fn, "window": w}
                for ch in ("RF0", "RF1"):
                    rms, db = window_stats(h, ch, w * WINDOW)
                    floor = float(np.median(db))
                    row[f"{ch}_rms"] = rms
                    row[f"{ch}_occ6"] = float(np.mean(db > floor + 6))
                    row[f"{ch}_peak_db"] = float(db.max() - floor)
                    acc.setdefault(ch, []).append(db)
                rows.append(row)
            psds[fn] = {ch: np.mean(v, axis=0) for ch, v in acc.items()}
        fr = [r for r in rows if r["file"] == fn]
        print(f"{fn}: {len(fr)} win | RF0 rms {np.median([r['RF0_rms'] for r in fr]):.4f} "
              f"occ {np.median([r['RF0_occ6'] for r in fr]):.3f} | RF1 rms "
              f"{np.median([r['RF1_rms'] for r in fr]):.4f} occ {np.median([r['RF1_occ6'] for r in fr]):.3f}",
              flush=True)

    if not rows:
        raise SystemExit("no files found under --data-root")
    with (args.output / "per_window.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for fn in psds:
        fr = [r for r in rows if r["file"] == fn]
        summary[fn] = {k: float(np.median([r[k] for r in fr])) for k in rows[0] if k not in ("file", "window")}
        summary[fn]["windows"] = len(fr)
    json.dump({"summary": summary, "missing": missing, "window": WINDOW, "snippet": SNIP,
               "nfft": NFFT, "occ6": "fraction of bins > median(PSD dB) + 6 dB"},
              open(args.output / "summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(psds)
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 1.8 * len(names)), sharex="col", squeeze=False)
    for i, fn in enumerate(names):
        for j, ch in enumerate(("RF0", "RF1")):
            f = CENTRE[ch] + np.linspace(-FS_MHZ / 2, FS_MHZ / 2, NFFT, endpoint=False)
            axes[i, j].plot(f, psds[fn][ch], lw=0.6)
            axes[i, j].set_title(f"{fn} {ch}", fontsize=8)
    for j in range(2):
        axes[-1, j].set_xlabel("MHz")
    fig.tight_layout()
    fig.savefig(args.output / "mean_psd.png", dpi=110)

    fig, axes = plt.subplots(len(names), 1, figsize=(10, 1.6 * len(names)), squeeze=False)
    for i, fn in enumerate(names):
        fr = [r for r in rows if r["file"] == fn]
        axes[i, 0].plot([r["RF0_occ6"] for r in fr], label="RF0 2.4G occ")
        axes[i, 0].plot([r["RF1_occ6"] for r in fr], label="RF1 5.8G occ")
        axes[i, 0].set_title(fn, fontsize=8)
    axes[0, 0].legend(fontsize=7)
    axes[-1, 0].set_xlabel("1M-sample window")
    fig.tight_layout()
    fig.savefig(args.output / "occupancy_over_time.png", dpi=110)
    print("written", args.output)


if __name__ == "__main__":
    main()
