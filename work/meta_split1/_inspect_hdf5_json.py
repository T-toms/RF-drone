# -*- coding: utf-8 -*-
"""META-SPLIT-1 quick inspection of the collected HDF5 metadata JSON."""
import json
import os
import sys

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hdf5_metadata_104.json")
d = json.load(open(p, encoding="utf-8"))
recs = d["records"]
print("records", len(recs))
seen = {}
for r in recs:
    for k, v in (r.get("datasets") or {}).items():
        seen.setdefault(k, set()).add((tuple(v.get("shape") or []), v.get("dtype"),
                                       v.get("chunks") and tuple(v["chunks"]), v.get("compression"),
                                       json.dumps(v.get("attrs"), ensure_ascii=False)))
for k in sorted(seen):
    print("dataset", k)
    for s in sorted(seen[k], key=str):
        print("   ", s)
print("root attrs sample:", json.dumps(recs[0].get("root_attrs"), ensure_ascii=False))
print("dataset attrs sample:", json.dumps(recs[0]["datasets"]["RF0_I"].get("attrs"), ensure_ascii=False))
probe = [(r["file_name"], r["datasets"]["RF0_I"]["probe_n"], r["datasets"]["RF0_I"]["probe_finite"]) for r in recs[:3]]
print("probe sample:", probe)
print("root_groups sample:", recs[0].get("root_groups"))
