# -*- coding: utf-8 -*-
"""Step 5: controls for the shortcut test.
(a) same features, v3 only (one receiver, both labels): if ~chance -> features carry receiver/env info, not drone info.
(b) which background recordings (campaign) sit in each split; seconds per split.
(c) per-test-group predictions of the step-4 classifier.
"""
import csv, json, collections, re
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score

O = Path(__file__).resolve().parent
T = Path(r"C:/Users/Administrator/Desktop/数据集处理/work/crop_slice_20261107")
FEATS = {"5.8GHz": ["edge_left", "edge_right", "right_mid"], "2.4GHz": ["wifi_ch1", "wifi_ch6", "edge_left"]}
camp = {r["group_id"]: r["campaign"] for r in csv.DictReader(open(O / "floor_per_recording.csv", encoding="utf-8"))}
out = {}
for band, fe in FEATS.items():
    rs = [r for r in csv.DictReader(open(O / f"window_features_L10_{band}.csv", encoding="utf-8")) if r["keep"] == "True"]
    X = np.array([[float(r[k]) for k in fe] for r in rs]); y = np.array([r["label"] == "drone" for r in rs], int)
    src = np.array([r["dataset"] for r in rs]); sp = np.array([r["split"] for r in rs]); g = np.array([r["group_id"] for r in rs])
    o = {}
    # (a) v3-only control
    v = src == "v3"
    c = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)).fit(X[v & (sp == "train")], y[v & (sp == "train")])
    pv = c.predict(X[v & (sp == "test")])
    o["control_v3_only_balanced_acc"] = float(balanced_accuracy_score(y[v & (sp == "test")], pv))
    # urban-only control (same receiver, same day, same gain, same place)
    u = v & np.array([camp[x] == "2026-07-09_urban" for x in g])
    c = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)).fit(X[u & (sp == "train")], y[u & (sp == "train")])
    m = u & (sp == "test")
    o["control_v3_urban_only_balanced_acc"] = float(balanced_accuracy_score(y[m], c.predict(X[m]))) if len(set(y[m])) == 2 else None
    o["urban_test_counts"] = dict(collections.Counter(y[m].tolist()))
    # main classifier (same as step 4) -> per test group
    c = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)).fit(X[sp == "train"], y[sp == "train"])
    te = sp == "test"; p = c.predict(X)
    per = []
    for gid in sorted(set(g[te])):
        mm = te & (g == gid)
        per.append(dict(group=gid, label="drone" if y[mm][0] else "background", campaign=camp[gid], n=int(mm.sum()),
                        predicted_drone=float(p[mm].mean())))
    o["per_test_group"] = per
    # (b) background recordings per split
    bgs = collections.defaultdict(lambda: collections.Counter())
    for r in rs:
        if r["label"] == "background":
            bgs[r["split"]][camp[r["group_id"]]] += 1
    o["bg_windows_by_split_campaign"] = {k: dict(v) for k, v in bgs.items()}
    out[band] = o
    print(band, "v3-only balanced acc %.3f | urban-only %s" % (o["control_v3_only_balanced_acc"], o["control_v3_urban_only_balanced_acc"]))
    print("  bg windows by split/campaign:", o["bg_windows_by_split_campaign"])
    for q in per:
        if q["label"] == "background" or q["group"].startswith("v3"):
            print("   ", q)
    dpd = [q["predicted_drone"] for q in per if q["group"].startswith("data")]
    print("   data/ drone test groups: n=%d, predicted-drone fraction min %.2f median %.2f" % (len(dpd), min(dpd), np.median(dpd)))
(O / "shortcut_controls.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
