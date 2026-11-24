# -*- coding: utf-8 -*-
"""Is any data/ file the same capture as a datasets_v3 file? Waveform correlation test.
data/ 007 (mini5 pro) 5.8GHz vs datasets_v3 consumer_drone/mini5pro 5800M files.
data/ assumed 122.88 MS/s -> resampled to 100 MS/s (625/768). Read-only."""
import glob, os
import numpy as np, scipy.io as sio
from scipy.signal import resample_poly, fftconvolve

ROOT = r"G:/drone"
dfiles = sorted(glob.glob(os.path.join(ROOT, "data", "*", "*", "007_5.8GHz*", "*.mat")))
dfiles = [p for p in dfiles if "(1)" not in p]
vfiles = sorted(glob.glob(os.path.join(ROOT, "datasets_v3", "consumer_drone", "mini5pro", "*5800M*.mat")))
print("data/ 007 5.8G files:", len(dfiles), " v3 mini5pro 5800 files:", len(vfiles), flush=True)

def ncorr(y, probe):
    c = fftconvolve(y, np.conj(probe[::-1]), mode="valid")
    e = np.concatenate([[0], np.cumsum(np.abs(y.astype(np.complex128)) ** 2)])
    L = len(probe); win = (e[L:] - e[:-L])[:len(c)]
    nc = np.abs(c) / (np.sqrt(win) * np.linalg.norm(probe) + 1e-30)
    return float(nc.max())

probes = []
for p in [dfiles[0], dfiles[len(dfiles) // 2]]:
    x = sio.loadmat(p, variable_names=["data"])["data"].ravel().astype(np.complex64)
    x = x[np.abs(x) > 0] if np.count_nonzero(x) > 200_000 else x
    probes.append((os.path.basename(p), resample_poly(x, 625, 768).astype(np.complex64)[:100_000]))
best = 0
for vp in vfiles:
    y = sio.loadmat(vp, variable_names=["IQ"])["IQ"].ravel().astype(np.complex64)
    for nm, pr in probes:
        r = ncorr(y, pr); best = max(best, r)
print("max normalized correlation over all pairs:", round(best, 4), "(same capture would give ~1.0)", flush=True)
