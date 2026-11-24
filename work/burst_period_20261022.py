"""Measure burst on/off timing of a capture, in samples.

Used to cross-calibrate the unknown sample rate of the lab .mat segments against the
July .bin captures, whose sample rate is recorded in their JSON sidecar.

Usage: python -I burst_period_20261022.py <file.mat|file.bin> [...]
  .bin is read as interleaved int16 (ci16_le); .mat as the complex 'data' variable.
"""
import os
import sys

import numpy as np
from scipy.io import loadmat

HOP = 128


def load(path, max_samples=12_000_000):
    if path.endswith(".mat"):
        return loadmat(path)["data"].ravel()[:max_samples]
    raw = np.fromfile(path, dtype="<i2", count=2 * max_samples)
    return raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)


def burst_timing(x, hop=HOP):
    n = (len(x) // hop) * hop
    env = 20 * np.log10(np.abs(x[:n].reshape(-1, hop)).mean(axis=1) + 1e-9)
    noise = np.percentile(env, 20)
    top = np.percentile(env, 99.5)
    if top - noise < 6:
        return None
    thr = noise + 0.5 * (top - noise)
    on = env > thr
    edges = np.diff(on.astype(np.int8))
    rise = np.nonzero(edges == 1)[0] + 1
    fall = np.nonzero(edges == -1)[0] + 1
    if len(rise) < 3:
        return None
    gaps = np.diff(rise) * hop
    # burst widths (pair each rise with the next fall)
    widths = []
    for r in rise:
        f = fall[fall > r]
        if len(f):
            widths.append((f[0] - r) * hop)
    return {
        "nburst": len(rise),
        "duty": float(on.mean()),
        "dyn_db": top - noise,
        "gap_med": float(np.median(gaps)),
        "gap_p10": float(np.percentile(gaps, 10)),
        "gap_p90": float(np.percentile(gaps, 90)),
        "width_med": float(np.median(widths)) if widths else 0.0,
    }


def main(paths):
    print(
        f"{'file':<54} {'samples':>9} {'dyn':>5} {'duty':>5} {'nb':>4} "
        f"{'gap_med':>8} {'gap_p10':>8} {'gap_p90':>8} {'width':>7}"
    )
    for p in paths:
        x = load(p)
        r = burst_timing(x)
        tag = os.path.basename(p)[:52]
        if r is None:
            print(f"{tag:<54} {len(x):>9}   (no clear bursts / continuous)")
            continue
        print(
            f"{tag:<54} {len(x):>9} {r['dyn_db']:>5.1f} {r['duty']:>5.2f} {r['nburst']:>4} "
            f"{r['gap_med']:>8.0f} {r['gap_p10']:>8.0f} {r['gap_p90']:>8.0f} {r['width_med']:>7.0f}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
