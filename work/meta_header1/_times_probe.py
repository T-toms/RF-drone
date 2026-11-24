# -*- coding: utf-8 -*-
"""Ad-hoc helper: print per-class time-of-day spread of the probe results."""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "member_header_probe_88.json")
data = json.load(open(path, encoding="utf-8"))
rows = [c for c in data["candidates"] if c["status"] == "ok"]
by = collections.defaultdict(list)
for c in rows:
    by[c["t_code"]].append((c["file_mtime_proxy_local"], c["file_name"],
                            "D%s/%s" % (c["d_code"], c["s_band_group"]),
                            "S" + c["s_code"]))
print("candidates:", len(data["candidates"]), "ok:", len(rows))
for t in sorted(by):
    v = sorted(by[t])
    hhmm = sorted({x[0][11:16] for x in v})
    print("==", t, "n =", len(v))
    print("   first:", v[0][0], v[0][1], v[0][2])
    print("   last :", v[-1][0], v[-1][1], v[-1][2])
    print("   distinct HH:MM:", len(hhmm), "range", hhmm[0], "-", hhmm[-1])
    groups = collections.defaultdict(list)
    for x in v:
        groups[(x[0][:10], x[2])].append(x[3])
    for key in sorted(groups):
        print("     ", key[0], key[1], "n=%d" % len(groups[key]),
              "S:", ",".join(sorted(groups[key])))
