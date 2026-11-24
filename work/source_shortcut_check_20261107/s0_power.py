# -*- coding: utf-8 -*-
"""Step 0: absolute power per source/label (Task F output is NOT normalized) + v3 campaign breakdown."""
import csv, collections, numpy as np
F = r"C:/Users/Administrator/Desktop/数据集处理/work/crop_slice_20261107/blocks.csv"
rs = list(csv.DictReader(open(F, encoding="utf-8-sig")))
d = collections.defaultdict(list)
for r in rs:
    d[(r["band"], r["dataset"], r["label"])].append(float(r["power_db_median"]))
for k, v in sorted(d.items()):
    v = np.array(v)
    print(k, len(v), "median %.1f  min %.1f  max %.1f" % (np.median(v), v.min(), v.max()))
camp = collections.Counter()
for r in rs:
    if r["dataset"] == "v3":
        p = r["source_paths"].split(";")[0]
        camp[(r["band"], r["label"], "/".join(p.split("/")[:3]))] += 1
for k, v in sorted(camp.items()):
    print(k, v)
print(rs[-1]["group_id"], rs[-1]["source_paths"][:200])
