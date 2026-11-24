# -*- coding: utf-8 -*-
"""META-HEADER1 / step 3b: are the un-downloaded candidates a *new* acquisition
batch, or a continuation of the run the G-drive files already came from?

Merges, per legacy class, the MATLAB ``Created on`` times of the eight local
files with the RAR-derived file-time proxies of the un-downloaded candidates,
sorts them, and reports the gap sequence.  META-SPLIT-1 measured intra-class
gaps of 81-159 s inside a single continuous run, so a candidate that follows the
local files with a gap in that range is a continuation, not a new batch.

Caveat carried into the output: the local times are MATLAB header strings that
run 1-2 s behind the RAR HTIME of the same file (measured on 14 members in
htime-decode-preregistration-20260925.json), so gaps spanning the local/remote
boundary carry a <=2 s systematic offset.

Output: work/meta_header1/campaign_continuity_check.json
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
PROBE = os.path.join(HERE, "member_header_probe_88.json")
MAT_HEADERS = os.path.join(WORKSPACE, "work", "meta_split1", "mat_headers_104.json")
OUT = os.path.join(HERE, "campaign_continuity_check.json")

LEGACY3 = ["T0000", "T0010", "T0101"]
NIGHT_BOUNDARY_HOURS = 6
REFERENCE_INTRA_CLASS_GAP = [81, 159]     # seconds, from META-SPLIT-1 section 3.4


def main() -> int:
    probe = json.load(open(PROBE, encoding="utf-8"))
    heads = json.load(open(MAT_HEADERS, encoding="utf-8"))

    local = {}
    for record in heads["records"]:
        if record["class_id"] in LEGACY3 and record.get("created_iso"):
            local.setdefault(record["class_id"], []).append({
                "file_name": record["file_name"],
                "local_time": record["created_iso"],
                "source": "matlab_header_on_g_drive",
                "on_g": True,
            })
    remote = {}
    for record in probe["candidates"]:
        if record["status"] != "ok":
            continue
        remote.setdefault(record["t_code"], []).append({
            "file_name": record["file_name"],
            "local_time": record["file_mtime_proxy_local"],
            "source": "rar5_htime_proxy",
            "on_g": False,
            "d_code": record["d_code"],
            "s_band_group": record["s_band_group"],
        })

    per_class = {}
    for t in LEGACY3:
        merged = local.get(t, []) + remote.get(t, [])
        for row in merged:
            row["epoch"] = dt.datetime.fromisoformat(
                row["local_time"].replace(" ", "T")).timestamp()
        merged.sort(key=lambda r: r["epoch"])
        gaps = []
        for previous, current in zip(merged, merged[1:]):
            gaps.append({
                "after": previous["file_name"],
                "before": current["file_name"],
                "gap_seconds": round(current["epoch"] - previous["epoch"], 1),
                "crosses_local_remote_boundary":
                    previous["on_g"] != current["on_g"],
                "crosses_night_boundary":
                    previous["local_time"][:10] != current["local_time"][:10]
                    or (dt.datetime.fromisoformat(previous["local_time"]) -
                        dt.timedelta(hours=NIGHT_BOUNDARY_HOURS)).date()
                    != (dt.datetime.fromisoformat(current["local_time"]) -
                        dt.timedelta(hours=NIGHT_BOUNDARY_HOURS)).date(),
            })
        g = [row["gap_seconds"] for row in gaps]
        night_of = lambda row: "N" + (dt.datetime.fromisoformat(row["local_time"])
                                      - dt.timedelta(hours=NIGHT_BOUNDARY_HOURS)
                                      ).strftime("%Y%m%d")
        # a new batch would show one gap far larger than the intra-run gaps
        largest = max(gaps, key=lambda r: r["gap_seconds"])
        per_class[t] = {
            "class_name": merged[0]["file_name"].split("_")[0] if merged else None,
            "files_total": len(merged),
            "files_on_g": sum(1 for r in merged if r["on_g"]),
            "files_un_downloaded": sum(1 for r in merged if not r["on_g"]),
            "first_local_time": merged[0]["local_time"],
            "first_file": merged[0]["file_name"],
            "last_local_time": merged[-1]["local_time"],
            "last_file": merged[-1]["file_name"],
            "span_minutes": round((merged[-1]["epoch"] - merged[0]["epoch"]) / 60, 1),
            "gap_seconds_min": min(g),
            "gap_seconds_max": max(g),
            "gap_seconds_sorted": sorted(g),
            "largest_gap": largest,
            "gaps_all_within_reference_intra_class_range": all(
                REFERENCE_INTRA_CLASS_GAP[0] <= x <= REFERENCE_INTRA_CLASS_GAP[1] for x in g),
            "gaps_outside_reference_range": [row for row in gaps
                                             if not (REFERENCE_INTRA_CLASS_GAP[0] <=
                                                     row["gap_seconds"] <=
                                                     REFERENCE_INTRA_CLASS_GAP[1])],
            "boundary_gap_local_to_remote": next(
                (row for row in gaps if row["crosses_local_remote_boundary"]), None),
            "night_keys_present": sorted({night_of(r) for r in merged}),
            "night_keys_of_on_g_files": sorted({night_of(r) for r in merged if r["on_g"]}),
            "night_keys_of_un_downloaded_files": sorted(
                {night_of(r) for r in merged if not r["on_g"]}),
            "un_downloaded_night_transition_files": [
                {"file_name": r["file_name"], "local_time": r["local_time"],
                 "night_key": night_of(r)}
                for r in merged if not r["on_g"] and night_of(r) not in
                {night_of(x) for x in merged if x["on_g"]}],
            "timeline": merged,
        }

    payload = {
        "schema": "dronerfa-meta-header1-campaign-continuity/v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": ("merge MATLAB header times of the eight G-drive files with the RAR5 "
                   "HTIME proxies of the un-downloaded candidates, sort, report gaps"),
        "reference_intra_class_gap_seconds": REFERENCE_INTRA_CLASS_GAP,
        "reference_night_boundary_hours": NIGHT_BOUNDARY_HOURS,
        "caveats": [
            "Local times are MATLAB 'Created on' strings that lag the RAR HTIME by 1-2 s "
            "(14/14 calibration), so a gap that straddles the local/remote boundary carries "
            "a <=2 s systematic offset.",
            "A single continuous gap sequence does NOT prove one radio session; it shows the "
            "file-time proxies are consistent with one continuous acquisition run.",
            "The 06:00 night boundary is an arbitrary cut adopted by META-SPLIT-1; a run that "
            "crosses 06:00 produces two night keys without any real shift change.",
        ],
        "per_class": per_class,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    print(json.dumps({t: {k: per_class[t][k] for k in
                          ("files_total", "files_on_g", "files_un_downloaded",
                           "first_local_time", "last_local_time", "span_minutes",
                           "gap_seconds_min", "gap_seconds_max",
                           "gaps_all_within_reference_intra_class_range",
                           "boundary_gap_local_to_remote", "night_keys_present",
                           "night_keys_of_on_g_files",
                           "night_keys_of_un_downloaded_files")}
                      for t in LEGACY3}, ensure_ascii=False, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
