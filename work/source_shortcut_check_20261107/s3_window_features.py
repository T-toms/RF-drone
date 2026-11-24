# -*- coding: utf-8 -*-
"""Step 3: per 10 ms training window (index_L10.csv, Task F split), compute "nuisance" features
that carry NO drone information:
  5.8G receiver fingerprint: left/right band-edge roll-off and tilt of the noise floor
  2.4G environment fingerprint: WiFi ch1 / ch6 floor bump (2412 / 2437 MHz)
Per window: per-bin MIN over its 10 cells of the cell-mean PSD (dB) -> removes intermittent bursts.
Each feature = median(dB in feature range) - median(dB in a reference range of the same window),
so the absolute gain cancels.
"""
import csv, json, collections
from pathlib import Path
import numpy as np

T = Path(r"C:/Users/Administrator/Desktop/数据集处理/work/crop_slice_20261107")
O = Path(__file__).resolve().parent
FS = {"2.4GHz": 61440000, "5.8GHz": 100000000}
FC = {"2.4GHz": 2430.0, "5.8GHz": 5800.0}
L = 10
FEAT = {
    "5.8GHz": dict(ref=(5762, 5790),
                   edge_left=(5755.3, 5757.5), edge_right=(5842.5, 5844.7), right_mid=(5830, 5840)),
    "2.4GHz": dict(ref=(2419, 2426),
                   wifi_ch1=(2409, 2416), wifi_ch6=(2432, 2439.5), edge_left=(2405.3, 2407.5)),
}

rows = list(csv.DictReader(open(T / f"index_L{L}.csv", encoding="utf-8-sig")))
byblock = collections.defaultdict(list)
for r in rows:
    byblock[r["block_file"]].append(r)
out = []
for i, (bf, rr) in enumerate(byblock.items()):
    band = rr[0]["band"]
    nfft = 512 if band == "2.4GHz" else 1024
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / FS[band])) / 1e6 + FC[band]
    M = np.load(T / "cells" / Path(bf).stem / "mean.npy")
    cs = FS[band] // 1000
    rng = {k: (f >= a) & (f <= b) for k, (a, b) in FEAT[band].items()}
    for r in rr:
        c0 = int(r["start"]) // cs
        w = M[c0:c0 + L].min(0)
        ref = np.median(w[rng["ref"]])
        rec = {k: r[k] for k in ["sample_id", "dataset", "band", "label", "model", "state", "group_id", "split", "keep"]}
        for k in FEAT[band]:
            if k != "ref":
                rec[k] = float(np.median(w[rng[k]]) - ref)
        out.append(rec)
    if i % 50 == 0:
        print(i, len(byblock), flush=True)
for band in FEAT:
    sel = [r for r in out if r["band"] == band]
    with open(O / f"window_features_L{L}_{band}.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sel[0].keys())); w.writeheader(); w.writerows(sel)
    print(band, len(sel))
