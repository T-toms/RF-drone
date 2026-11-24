# -*- coding: utf-8 -*-
"""Step 10: what a NORMAL spectrogram of each 10 ms training window looks like, in two numbers.
For every keep=True window of Task F index_L10 (the actual training samples):
  peak_db  = 99th percentile over (10 cells x in-band bins) of cell-mean PSD minus the recording's floor level
             -> "how strong is the strongest signal in this window"
  occ      = fraction of (cell, bin) with cell-mean PSD > floor + 6 dB
             -> "how much of the time-frequency picture is filled with signal"
Both are relative to the recording's own floor, so they do not depend on absolute gain / normalisation.
"""
import csv, collections, json
from pathlib import Path
import numpy as np

T = Path(r"C:/Users/Administrator/Desktop/数据集处理/work/crop_slice_20261107")
O = Path(__file__).resolve().parent
FS = {"2.4GHz": 61440000, "5.8GHz": 100000000}
FC = {"2.4GHz": 2430.0, "5.8GHz": 5800.0}
W = {"2.4GHz": 25.0, "5.8GHz": 45.0}
cfg = json.loads((T / "config.json").read_text(encoding="utf-8"))
lvl = {r["group_id"]: float(r["floor_level_db"]) for r in csv.DictReader(open(O / "floor_per_recording.csv", encoding="utf-8"))}
rows = [r for r in csv.DictReader(open(T / "index_L10.csv", encoding="utf-8-sig")) if r["keep"] == "True"]
by = collections.defaultdict(list)
for r in rows:
    by[r["block_file"]].append(r)
out = []
for bf, rr in by.items():
    band = rr[0]["band"]; nfft = 512 if band == "2.4GHz" else 1024
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / FS[band])) / 1e6 + FC[band]
    m = np.abs(f - FC[band]) <= W[band]
    for n in cfg["notches"]:
        if n["band"] == band:
            m &= np.abs(f - n["freq_mhz"]) > max(n["half_width_khz"] / 1000, 0.12)
    M = np.load(T / "cells" / Path(bf).stem / "mean.npy"); cs = FS[band] // 1000
    fl = lvl[rr[0]["group_id"]]
    for r in rr:
        c0 = int(r["start"]) // cs
        w = M[c0:c0 + 10][:, m] - fl
        out.append(dict(band=band, dataset=r["dataset"], label=r["label"], model=r["model"], state=r["state"],
                        group_id=r["group_id"], split=r["split"],
                        peak_db=float(np.percentile(w, 99)), occ=float((w > 6).mean())))
with open(O / "window_metrics_L10.csv", "w", encoding="utf-8", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(out[0].keys())); wr.writeheader(); wr.writerows(out)
for band in FS:
    for key in [("data", "drone"), ("v3", "drone"), ("v3", "background")]:
        s = [x for x in out if x["band"] == band and (x["dataset"], x["label"]) == key]
        p = np.array([x["peak_db"] for x in s]); o = np.array([x["occ"] for x in s])
        print(band, key, len(s), "peak dB p10/p50/p90 = %.1f/%.1f/%.1f" % tuple(np.percentile(p, [10, 50, 90])),
              "| occ p10/p50/p90 = %.3f/%.3f/%.3f" % tuple(np.percentile(o, [10, 50, 90])))
