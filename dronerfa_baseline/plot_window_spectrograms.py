#!/usr/bin/env python3
"""Read-only: full 1M-sample (10 ms) spectrograms of RF0/RF1 for chosen windows.

Complements audit_band_switch.py, which only looks at the first 65,536 samples
of each window and averages the PSD, so short bursts and weak wideband links can
be missed. Development-test files are diagnostic only.

Usage:
  .venv\\Scripts\\python.exe dronerfa_baseline\\plot_window_spectrograms.py --data-root <dir> \\
      --picks T1011_S0101.mat:50,T1011_S0111.mat:60 --output <png>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

WINDOW = 1_000_000
NFFT = 1024
FS_MHZ = 100.0
CENTRE = {"RF0": 2440.0, "RF1": 5800.0}


def spectrogram(handle: h5py.File, ch: str, start: int) -> np.ndarray:
    def read(key):
        ds = handle[key]
        return np.asarray(ds[0, start:start + WINDOW] if ds.ndim == 2 else ds[start:start + WINDOW], dtype=np.float32)
    x = read(ch + "_I") + 1j * read(ch + "_Q")
    frames = x[: (len(x) // NFFT) * NFFT].reshape(-1, NFFT) * np.hamming(NFFT).astype(np.float32)
    power = np.abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)) ** 2
    return 10 * np.log10(power + 1e-20).T          # [freq, time]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--picks", required=True, help="file:window,file:window,...")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    picks = [(p.split(":")[0], int(p.split(":")[1])) for p in args.picks.split(",")]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(picks), 2, figsize=(11, 2.1 * len(picks)), squeeze=False)
    for i, (fn, w) in enumerate(picks):
        with h5py.File(args.data_root / fn, "r") as h:
            for j, ch in enumerate(("RF0", "RF1")):
                s = spectrogram(h, ch, w * WINDOW)
                floor = np.median(s)
                f0 = CENTRE[ch] - FS_MHZ / 2
                axes[i, j].imshow(s, aspect="auto", origin="lower", cmap="viridis",
                                  vmin=floor - 3, vmax=floor + 25,
                                  extent=(0, WINDOW / FS_MHZ / 1000, f0, f0 + FS_MHZ))
                axes[i, j].set_title(f"{fn} win {w} {ch}", fontsize=8)
                axes[i, j].tick_params(labelsize=7)
    for j in range(2):
        axes[-1, j].set_xlabel("ms")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=90)
    print("written", args.output)


if __name__ == "__main__":
    main()
