"""Probe the lab .mat segments: peak-hold spectrum profile + time envelope.

Goal: infer the receiver sample rate (not recorded in the files) by comparing the
occupied bandwidth with the bandwidth label in the directory name, and look at the
burst structure in time.

Usage: python -I probe_newdata_fs_20261022.py <file.mat> [...]
"""
import os
import sys

import numpy as np
from scipy.io import loadmat

BARS = " .:-=+*#%@"


def profile(vals, lo, hi):
    """Render a vector as an ASCII bar string scaled between lo and hi dB."""
    idx = np.clip(((vals - lo) / max(hi - lo, 1e-9) * (len(BARS) - 1)).astype(int), 0, len(BARS) - 1)
    return "".join(BARS[i] for i in idx)


def spectrum(x, nfft=2048, nbins=64):
    n = (len(x) // nfft) * nfft
    seg = x[:n].reshape(-1, nfft)
    seg = seg[:: max(1, len(seg) // 600)]
    win = np.hanning(nfft).astype(np.float32)
    spec = np.fft.fftshift(np.abs(np.fft.fft(seg * win, axis=1)) ** 2, axes=1)
    mean_db = 10 * np.log10(spec.mean(axis=0) + 1e-12)
    peak_db = 10 * np.log10(spec.max(axis=0) + 1e-12)
    # fold to nbins
    f = nfft // nbins
    mean_db = mean_db.reshape(nbins, f).mean(axis=1)
    peak_db = peak_db.reshape(nbins, f).max(axis=1)
    return mean_db, peak_db


def envelope(x, hop=2048, nshow=96):
    n = (len(x) // hop) * hop
    env = 20 * np.log10(np.abs(x[:n].reshape(-1, hop)).mean(axis=1) + 1e-9)
    f = max(1, len(env) // nshow)
    env = env[: (len(env) // f) * f].reshape(-1, f).max(axis=1)
    return env


def main(paths):
    for p in paths:
        x = loadmat(p)["data"].ravel()
        mean_db, peak_db = spectrum(x)
        env = envelope(x)
        name = os.path.basename(os.path.dirname(p))
        base = np.percentile(mean_db, 10)
        top = np.max(peak_db)
        print("=" * 104)
        print(f"{name}   N={len(x)}  noise~{base:.1f}dB  peak~{top:.1f}dB")
        print("  spec mean |" + profile(mean_db, base, np.max(mean_db)) + "|  (left=-fs/2, right=+fs/2)")
        print("  spec peak |" + profile(peak_db, base, top) + "|")
        # occupied bandwidth from the peak-hold curve, 20 dB down from its max
        occ = (peak_db > top - 20).mean()
        print(f"  occupied(peak-hold, -20dB) = {occ * 100:.1f}% of span")
        print("  time env  |" + profile(env, np.percentile(env, 5), np.max(env)) + "|  (whole segment)")


if __name__ == "__main__":
    main(sys.argv[1:])
