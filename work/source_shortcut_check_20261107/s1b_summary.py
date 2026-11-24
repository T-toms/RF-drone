# -*- coding: utf-8 -*-
import csv, collections, numpy as np
from pathlib import Path
O = Path(__file__).resolve().parent
rs = list(csv.DictReader(open(O / "floor_per_recording.csv", encoding="utf-8")))
d = collections.defaultdict(list)
for r in rs:
    d[(r["band"], r["dataset"], r["label"], r["campaign"], r["gain"])].append(float(r["floor_level_db"]))
print("band, dataset, label, campaign, gain -> n recordings, floor level dB (min/median/max)")
for k, v in sorted(d.items()):
    print(k, len(v), "%.1f / %.1f / %.1f" % (min(v), np.median(v), max(v)))
