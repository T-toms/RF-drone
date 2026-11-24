# -*- coding: utf-8 -*-
"""Step 1: per-recording noise-floor curve from Task F 1 ms cells (no IQ re-read).

For every recording (group_id) and every frequency bin: 10th percentile over all 1 ms cells of
the cell-mean PSD (dB).  A bin that is empty >=10 % of the time gives the noise floor there;
intermittent bursts (WiFi, hopping) are excluded per bin.  Level = median over in-band,
non-notch bins; shape = curve - level.  Also records per-bin occupancy (> floor + 6 dB).
"""
import csv, re, collections, json
from pathlib import Path
import numpy as np

T = Path(r"C:/Users/Administrator/Desktop/数据集处理/work/crop_slice_20261107")
O = Path(__file__).resolve().parent
FS = {"2.4GHz": 61440000, "5.8GHz": 100000000}
FC = {"2.4GHz": 2430.0, "5.8GHz": 5800.0}
WIDTH = {"2.4GHz": 25.0, "5.8GHz": 45.0}
cfg = json.loads((T / "config.json").read_text(encoding="utf-8"))
NOTCH = cfg["notches"]

rows = list(csv.DictReader(open(T / "blocks.csv", encoding="utf-8-sig")))
by = collections.defaultdict(list)
for r in rows:
    by[r["group_id"]].append(r)


def meta(r):
    p = r["source_paths"].split(";")[0]
    if r["dataset"] == "data":
        return dict(campaign="data(2026-09)", gain="?", fc_in=2400.0 if r["band"] == "2.4GHz" else 5800.0)
    m = re.search(r"_(\d+)M_(\d+)dB_", p)
    camp = "noise_adrv" if "noise_adrv" in p else (re.search(r"(2026-07-\d\d_[a-z]+)", p).group(1) if re.search(r"(2026-07-\d\d_[a-z]+)", p) else "?")
    return dict(campaign=camp, gain=m.group(2) if m else "?", fc_in=float(m.group(1)) if m else float("nan"))


out = []
curves = {}
for gid, bl in by.items():
    band = bl[0]["band"]
    nfft = 512 if band == "2.4GHz" else 1024
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / FS[band])) / 1e6 + FC[band]
    mask = np.abs(f - FC[band]) <= WIDTH[band]
    for n in NOTCH:
        if n["band"] == band:
            mask &= np.abs(f - n["freq_mhz"]) > max(n["half_width_khz"] / 1000, 0.12)
    mean = np.concatenate([np.load(T / "cells" / Path(b["block_file"]).stem / "mean.npy") for b in bl])
    p10 = np.percentile(mean, 10, axis=0)
    p2 = np.percentile(mean, 2, axis=0)
    level = float(np.median(p10[mask]))
    occ = (mean > p10 + 6).mean(0)
    md = meta(bl[0])
    rec = dict(group_id=gid, band=band, dataset=bl[0]["dataset"], label=bl[0]["label"], model=bl[0]["model"],
               state=bl[0]["state"], n_cells=int(len(mean)), floor_level_db=level, **md)
    out.append(rec)
    curves[gid] = dict(f=f, mask=mask, p10=p10, p2=p2, occ=occ)
    print(gid, band, rec["dataset"], rec["label"], md["campaign"], md["gain"], len(mean), "%.1f" % level, flush=True)

with open(O / "floor_per_recording.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
np.savez_compressed(O / "floor_curves.npz",
                    **{f"{k}|{kk}": vv for k, v in curves.items() for kk, vv in v.items()})
print("DONE", len(out))
