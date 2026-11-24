# -*- coding: utf-8 -*-
"""Step 6: how much background exists in raw/ (original .bin) vs how much datasets_v3 kept."""
import csv, collections
rs = list(csv.DictReader(open(r"C:/Users/Administrator/Desktop/数据集处理/work/verify_v3_20261104/raw_inventory.csv", encoding="utf-8")))
uniq = {}
for r in rs:
    uniq.setdefault(r["sha"], r)
print("raw json rows", len(rs), "unique captures", len(uniq))
agg = collections.defaultdict(lambda: [0, 0.0, set()])
for r in uniq.values():
    fc = float(r["fc"]) / 1e6
    if fc not in (2440, 2450, 5800):
        continue
    dur = float(r["n"]) / float(r["fs"])
    k = (r["cls"], fc, float(r["fs"]) / 1e6, r["campaign"], r["bin_exists"])
    agg[k][0] += 1; agg[k][1] += dur; agg[k][2].add(round(dur, 4))
print("(class, fc MHz, fs MHz, campaign, bin_exists) -> captures, total seconds, distinct capture durations (s)")
for k, v in sorted(agg.items(), key=str):
    print(k, v[0], round(v[1], 3), sorted(v[2])[:5])
