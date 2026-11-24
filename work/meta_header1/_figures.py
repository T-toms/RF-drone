# -*- coding: utf-8 -*-
"""Ad-hoc helper: dump the exact figures the report will quote."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
s = json.load(open(os.path.join(HERE, "availability_summary_by_class_date.json"),
                   encoding="utf-8"))
print("probe_status_counts:", s["probe_status_counts"])
print("probe_http_stats:", json.dumps(s["probe_http_stats"], ensure_ascii=False))
print("anomalies:", s["anomalies"])
print("frozen volume:", json.dumps(s["remaining_frozen_candidate_volume_for_legacy3"],
                                   ensure_ascii=False))
tot = {"files": 0, "packed": 0, "unpacked": 0}
for t in ("T0000", "T0010", "T0101"):
    info = s["per_class"][t]
    print("==", t, info["class_name"])
    print("   existing:", info["existing_on_g"]["proxy_dates"],
          info["existing_on_g"]["configurations"],
          "files", info["existing_on_g"]["files"],
          "all_exposed", info["existing_on_g"]["all_exposed_to_prior_selection"])
    for cfg in info["candidate_configuration_groups"]:
        print("   cand %s %s %s n=%d packed=%d unpacked=%d" % (
            cfg["proxy_date"], cfg["d_code"], cfg["s_band_group"], cfg["files"],
            cfg["packed_bytes"], cfg["unpacked_bytes"]))
        tot["files"] += cfg["files"]
        tot["packed"] += cfg["packed_bytes"]
        tot["unpacked"] += cfg["unpacked_bytes"]
    print("   new_date_files:", info["new_date_file_count"],
          "new_date packed:", info["new_date_packed_bytes"],
          "unpacked:", info["new_date_unpacked_bytes"])
print("TOTAL candidates files/packed/unpacked:", tot)
print()
for key in sorted(s["download_volume_options"]):
    option = s["download_volume_options"][key]
    print("%-52s feasible=%s packed=%d unpacked=%d" % (
        key, option["all_three_classes_feasible"], option["total_packed_bytes"],
        option["total_unpacked_bytes"]))
    for t, v in option["per_class"].items():
        if v["feasible"]:
            print("      %s %s -> %s" % (t, v["picked_files"], v["picked_configurations"]))
