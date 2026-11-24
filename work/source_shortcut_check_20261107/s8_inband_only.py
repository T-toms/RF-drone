# -*- coding: utf-8 -*-
"""Step 8: does the shortcut survive if we stay away from the crop edges?
Features only from the middle of the cropped band (no band-edge bins), plus a level-only test.
Same windows / split as step 4 (Task F index_L10, keep=True).
  5.8G: tilt_left = 5762-5772 vs ref 5785-5795 ; tilt_right = 5828-5838 vs ref   (all >= 7 MHz from crop edges)
  2.4G: wifi_ch1 2409-2416, wifi_ch6 2432-2439.5 vs ref 2419-2426      (no edge feature)
  level: absolute floor level (median dB of per-bin window-min over ref range) -- Task F has no normalization
"""
import csv, collections, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

T = Path(r"C:/Users/Administrator/Desktop/数据集处理/work/crop_slice_20261107")
O = Path(__file__).resolve().parent
FS = {"2.4GHz": 61440000, "5.8GHz": 100000000}
FC = {"2.4GHz": 2430.0, "5.8GHz": 5800.0}
FEAT = {"5.8GHz": dict(ref=(5785, 5795), tilt_left=(5762, 5772), tilt_right=(5828, 5838)),
        "2.4GHz": dict(ref=(2419, 2426), wifi_ch1=(2409, 2416), wifi_ch6=(2432, 2439.5))}
rows = [r for r in csv.DictReader(open(T / "index_L10.csv", encoding="utf-8-sig")) if r["keep"] == "True"]
byblock = collections.defaultdict(list)
for r in rows:
    byblock[r["block_file"]].append(r)
data = collections.defaultdict(list)
for bf, rr in byblock.items():
    band = rr[0]["band"]; nfft = 512 if band == "2.4GHz" else 1024
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / FS[band])) / 1e6 + FC[band]
    M = np.load(T / "cells" / Path(bf).stem / "mean.npy"); cs = FS[band] // 1000
    rg = {k: (f >= a) & (f <= b) for k, (a, b) in FEAT[band].items()}
    for r in rr:
        w = M[int(r["start"]) // cs:int(r["start"]) // cs + 10].min(0)
        ref = float(np.median(w[rg["ref"]]))
        shp = [float(np.median(w[rg[k]]) - ref) for k in FEAT[band] if k != "ref"]
        data[band].append((shp, ref, r["label"] == "drone", r["dataset"], r["split"]))


def run(X, y, src, sp):
    tr, te = sp == "train", sp == "test"
    c = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)).fit(X[tr], y[tr])
    p = c.predict(X[te]); yt, st = y[te], src[te]
    return dict(bg_false_alarm=round(float((p[yt == 0] == 1).mean()), 3),
                recall_data_drone=round(float((p[(yt == 1) & (st == "data")] == 1).mean()), 3),
                recall_v3_drone=round(float((p[(yt == 1) & (st == "v3")] == 1).mean()), 3))


res = {}
for band, d in data.items():
    S = np.array([x[0] for x in d]); Lv = np.array([[x[1]] for x in d]); y = np.array([x[2] for x in d], int)
    src = np.array([x[3] for x in d]); sp = np.array([x[4] for x in d])
    res[band] = {"mid_band_shape_only": run(S, y, src, sp), "floor_level_only": run(Lv, y, src, sp),
                 "floor_level_median_dB": {k: round(float(np.median(Lv[m, 0])), 2) for k, m in
                                           [("data_drone", (y == 1) & (src == "data")), ("v3_drone", (y == 1) & (src == "v3")), ("v3_bg", y == 0)]}}
    print(band, json.dumps(res[band], ensure_ascii=False))
(O / "inband_only_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
