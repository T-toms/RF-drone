# -*- coding: utf-8 -*-
"""Ad-hoc helper: dump continuity details for the report."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
c = json.load(open(os.path.join(HERE, "campaign_continuity_check.json"),
                   encoding="utf-8"))
for t in ("T0000", "T0010", "T0101"):
    info = c["per_class"][t]
    print("==", t)
    print("  span_minutes", info["span_minutes"],
          "gaps min/max", info["gap_seconds_min"], info["gap_seconds_max"])
    print("  gaps sorted:", info["gap_seconds_sorted"])
    print("  outside reference range:", info["gaps_outside_reference_range"])
    print("  night transition files:",
          json.dumps(info["un_downloaded_night_transition_files"], ensure_ascii=False))
    tl = info["timeline"]
    print("  first 3:", [(r["file_name"], r["local_time"]) for r in tl[:3]])
    print("  last 3:", [(r["file_name"], r["local_time"]) for r in tl[-3:]])
    if t != "T0000":
        boundary = [i for i, r in enumerate(tl) if r["file_name"].endswith("S0111.mat")
                    or r["file_name"].endswith("S1111.mat")]
        print("  positions of S0111:", boundary)
