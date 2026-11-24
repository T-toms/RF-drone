# -*- coding: utf-8 -*-
"""Step 4: shortcut test.
A logistic regression that sees ONLY the nuisance features (receiver edge shape / WiFi bump),
i.e. nothing about the drone signal, is trained bg-vs-drone on the Task F train split and
evaluated on the Task F test split (keep=True windows, exactly what training would use).
If it scores high, the label is predictable from recording conditions -> shortcut exists.
"""
import csv, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

O = Path(__file__).resolve().parent
FEATS = {"5.8GHz": ["edge_left", "edge_right", "right_mid"], "2.4GHz": ["wifi_ch1", "wifi_ch6", "edge_left"]}
res = {}
for band, fe in FEATS.items():
    rs = [r for r in csv.DictReader(open(O / f"window_features_L10_{band}.csv", encoding="utf-8")) if r["keep"] == "True"]
    X = np.array([[float(r[k]) for k in fe] for r in rs])
    y = np.array([r["label"] == "drone" for r in rs], int)
    src = np.array([r["dataset"] for r in rs])
    sp = np.array([r["split"] for r in rs])
    tr, te = sp == "train", sp == "test"
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    clf.fit(X[tr], y[tr])
    p = clf.predict(X[te])
    yt, st = y[te], src[te]
    bg = yt == 0
    dd = (yt == 1) & (st == "data")
    vd = (yt == 1) & (st == "v3")
    r = dict(
        n_test=int(te.sum()), n_bg=int(bg.sum()), n_data_drone=int(dd.sum()), n_v3_drone=int(vd.sum()),
        accuracy=float((p == yt).mean()),
        always_drone_accuracy=float(yt.mean()),
        bg_false_alarm=float((p[bg] == 1).mean()),
        recall_data_drone=float((p[dd] == 1).mean()),
        recall_v3_drone=float((p[vd] == 1).mean()),
    )
    # source identifiability among drone windows only (data vs v3), same split
    m_tr = tr & (y == 1); m_te = te & (y == 1)
    c2 = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    c2.fit(X[m_tr], src[m_tr] == "data")
    r["source_id_accuracy_drone_windows"] = float((c2.predict(X[m_te]) == (src[m_te] == "data")).mean())
    # per-group-feature summary (median per source/label)
    summ = {}
    for key, msk in [("data_drone", (y == 1) & (src == "data")), ("v3_drone", (y == 1) & (src == "v3")), ("v3_bg", y == 0)]:
        summ[key] = {k: [float(np.percentile(X[msk, j], q)) for q in (5, 50, 95)] for j, k in enumerate(fe)}
    r["feature_p5_p50_p95"] = summ
    res[band] = r
    print(band, json.dumps(r, ensure_ascii=False, indent=1))
(O / "shortcut_test_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
