# -*- coding: utf-8 -*-
"""META-HEADER1 / step 3: turn the per-member header probe into a by-class,
by-date availability table with download-volume estimates.

Reads only:
  work/meta_header1/member_header_probe_88.json   (this task's probe output)
  work/meta_split1/mat_headers_104.json           (existing 104 files' file times)
  work/meta_split1/archive_unused_for_legacy3_20260925.csv (frozen candidate sizes)
  work/meta_split1/campaign_grouping_104.json     (night keys of the existing corpus)

Writes:
  work/meta_header1/availability_summary_by_class_date.json
  work/meta_header1/availability_summary_by_class_date.csv

Terminology is deliberately bounded: a file time is a *file-time proxy*; a
different proxy date is NOT an independent acquisition session.  A different
`night_key` is the strongest statement this evidence supports, and it is still
only a "different acquisition-night proxy".

The summary also cross-tabulates distance code (D) and band group (S) because a
candidate on a new date may also differ in acquisition configuration, which
would confound a plain "new date == held-out" reading.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))

PROBE = os.path.join(HERE, "member_header_probe_88.json")
MAT_HEADERS = os.path.join(WORKSPACE, "work", "meta_split1", "mat_headers_104.json")
CANDIDATES = os.path.join(WORKSPACE, "work", "meta_split1",
                          "archive_unused_for_legacy3_20260925.csv")
GROUPS = os.path.join(WORKSPACE, "work", "meta_split1", "campaign_grouping_104.json")
OUT_JSON = os.path.join(HERE, "availability_summary_by_class_date.json")
OUT_CSV = os.path.join(HERE, "availability_summary_by_class_date.csv")

LEGACY3 = ["T0000", "T0010", "T0101"]
ALL_CLASSES = ["T0000", "T0001", "T0010", "T0011", "T0100", "T0101", "T0110",
               "T0111", "T1000", "T1001", "T1010", "T1011", "T1100"]
NIGHT_BOUNDARY_HOURS = 6
D_LABELS = {"00": "20-40 m", "01": "40-80 m", "10": "80-150 m",
            None: "no D field in archive name"}


def night_key_of_iso(iso: str) -> str:
    moment = dt.datetime.fromisoformat(iso.replace(" ", "T"))
    return "N" + (moment - dt.timedelta(hours=NIGHT_BOUNDARY_HOURS)).strftime("%Y%m%d")


def date_of_iso(iso: str) -> str:
    return iso.replace(" ", "T")[:10]


def configuration_of(row: dict) -> dict:
    """Acquisition-configuration axes carried by the archive file name."""
    return {"d_code": row["d_code"], "distance_label": row["distance_label"],
            "s_band_group": row["s_band_group"]}


def main() -> int:
    probe_path = PROBE
    out_json, out_csv = OUT_JSON, OUT_CSV
    if len(sys.argv) > 1:
        probe_path = sys.argv[1]
    if len(sys.argv) > 3:
        out_json, out_csv = sys.argv[2], sys.argv[3]
    probe = json.load(open(probe_path, encoding="utf-8"))
    heads = json.load(open(MAT_HEADERS, encoding="utf-8"))
    grouping = json.load(open(GROUPS, encoding="utf-8"))

    existing = {}
    config_by_stem = {}
    for record in heads["records"]:
        t = record["class_id"]
        entry = {
            "file_name": record["file_name"],
            "class_name": record["class_name"],
            "d_code": record.get("d_code"),
            "distance_label": D_LABELS.get(record.get("d_code"), "?"),
            "s_band_group": record.get("s_band_group"),
            "proxy_date": date_of_iso(record["created_iso"]) if record.get("created_iso") else None,
            "proxy_local_time": record.get("created_iso"),
            "night_key": night_key_of_iso(record["created_iso"]) if record.get("created_iso") else None,
            "evaluation_role": record.get("evaluation_role"),
            "former_partition": record.get("former_partition"),
            "exposed_to_prior_selection": record.get("former_partition") is not None,
        }
        existing.setdefault(t, []).append(entry)
        config_by_stem[record["file_name"]] = entry

    candidates_by_class = {}
    anomalies = []
    status_counts = {}
    for record in probe["candidates"]:
        t = record["t_code"]
        status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1
        entry = {
            "file_name": record["file_name"],
            "class_name": record["class_name"],
            "d_code": record["d_code"],
            "s_code": record["s_code"],
            "distance_label": record["distance_label"],
            "s_band_group": record["s_band_group"],
            "status": record["status"],
            "error": record.get("error"),
            "block_offset": record["frozen_block_offset"],
            "packed_size": record["frozen_packed_size"],
            "unpacked_size": record["frozen_unpacked_size"],
            "proxy_date": record.get("file_mtime_proxy_date"),
            "proxy_local_time": record.get("file_mtime_proxy_local"),
            "night_key": record.get("night_key"),
            "htime_flags": record.get("htime_flags"),
            "htime_raw_hex": record.get("htime_raw_hex"),
            "htime_encoding": record.get("htime_encoding"),
            "header_span_bytes": record.get("header_span_bytes"),
            "http_requests": record.get("http_requests"),
            "http_bytes": record.get("http_bytes"),
            "verification": record.get("verification"),
        }
        if record["status"] != "ok":
            anomalies.append({k: entry[k] for k in
                              ("file_name", "class_name", "status", "error")})
        candidates_by_class.setdefault(t, []).append(entry)

    per_class = {}
    for t in LEGACY3:
        rows = candidates_by_class.get(t, [])
        ex = sorted(existing.get(t, []), key=lambda r: r["file_name"])
        ex_dates = sorted({r["proxy_date"] for r in ex if r["proxy_date"]})
        ex_nights = sorted({r["night_key"] for r in ex if r["night_key"]})
        ex_configs = sorted({(r["d_code"], r["s_band_group"]) for r in ex})
        ok_rows = [r for r in rows if r["status"] == "ok" and r["proxy_date"]]

        date_groups = {}
        config_groups = {}
        for row in ok_rows:
            bucket = date_groups.setdefault(row["proxy_date"], {
                "proxy_date": row["proxy_date"], "night_key": row["night_key"],
                "files": 0, "packed_bytes": 0, "unpacked_bytes": 0,
                "file_names": [], "s_codes": [], "d_codes": [],
                "configurations": [],
            })
            bucket["files"] += 1
            bucket["packed_bytes"] += row["packed_size"]
            bucket["unpacked_bytes"] += row["unpacked_size"]
            bucket["file_names"].append(row["file_name"])
            bucket["s_codes"].append(row["s_code"])
            bucket["d_codes"].append(row["d_code"])
            bucket["configurations"].append(f"D{row['d_code']}/{row['s_band_group']}")
            cfg = config_groups.setdefault(
                (row["proxy_date"], row["d_code"], row["s_band_group"]), {
                    "proxy_date": row["proxy_date"], "night_key": row["night_key"],
                    "d_code": row["d_code"],
                    "distance_label": D_LABELS.get(row["d_code"], "?"),
                    "s_band_group": row["s_band_group"],
                    "files": 0, "packed_bytes": 0, "unpacked_bytes": 0, "file_names": []})
            cfg["files"] += 1
            cfg["packed_bytes"] += row["packed_size"]
            cfg["unpacked_bytes"] += row["unpacked_size"]
            cfg["file_names"].append(row["file_name"])
        for row in ok_rows:
            row["same_proxy_date_as_some_existing_file"] = row["proxy_date"] in ex_dates
            row["date_is_new_for_class"] = row["proxy_date"] not in ex_dates
            row["night_is_new_for_class"] = row["night_key"] not in ex_nights
            row["configuration_matches_existing"] = (
                (row["d_code"], row["s_band_group"]) in ex_configs)
            row["shares_distance_axis_with_existing"] = any(
                r["d_code"] == row["d_code"] for r in ex)
            row["shares_band_axis_with_existing"] = any(
                r["s_band_group"] == row["s_band_group"] for r in ex)
        new_date_rows = [r for r in ok_rows if r["date_is_new_for_class"]]
        new_night_rows = [r for r in ok_rows if r["night_is_new_for_class"]]
        for bucket in config_groups.values():
            bucket["file_names"].sort()
        per_class[t] = {
            "class_name": ex[0]["class_name"] if ex else (rows[0]["class_name"] if rows else None),
            "existing_on_g": {
                "files": len(ex),
                "proxy_dates": ex_dates,
                "night_keys": ex_nights,
                "configurations": [f"D{d}/{b}" for d, b in ex_configs],
                "all_exposed_to_prior_selection": all(r["exposed_to_prior_selection"] for r in ex),
                "files_detail": ex,
            },
            "candidates_probed": len(rows),
            "candidates_ok": len(ok_rows),
            "candidates_anomalous": len(rows) - len(ok_rows),
            "candidate_dates_all": sorted(date_groups),
            "candidate_nights_all": sorted({r["night_key"] for r in ok_rows}),
            "candidate_date_groups": {d: date_groups[d] for d in sorted(date_groups)},
            "candidate_configuration_groups": [
                config_groups[k] for k in sorted(config_groups,
                                                 key=lambda x: (x[0], x[1], str(x[2])))],
            "new_date_groups": {d: v for d, v in sorted(date_groups.items())
                                if d not in ex_dates},
            "new_date_file_count": len(new_date_rows),
            "new_date_packed_bytes": sum(r["packed_size"] for r in new_date_rows),
            "new_date_unpacked_bytes": sum(r["unpacked_size"] for r in new_date_rows),
            "new_date_file_names": sorted(r["file_name"] for r in new_date_rows),
            "new_night_file_count": len(new_night_rows),
            "new_night_file_names": sorted(r["file_name"] for r in new_night_rows),
            "new_date_and_same_configuration_file_names": sorted(
                r["file_name"] for r in new_date_rows if r["configuration_matches_existing"]),
            "new_date_and_same_distance_file_names": sorted(
                r["file_name"] for r in new_date_rows if r["shares_distance_axis_with_existing"]),
            "new_date_and_same_band_file_names": sorted(
                r["file_name"] for r in new_date_rows if r["shares_band_axis_with_existing"]),
            "has_candidate_on_a_new_proxy_date": bool(new_date_rows),
            "has_candidate_in_a_new_night_proxy": bool(new_night_rows),
            "has_exact_configuration_match_on_a_new_date": bool(
                any(r["configuration_matches_existing"] for r in new_date_rows)),
            "candidates_detail": sorted(rows, key=lambda r: r["file_name"]),
        }

    # --- estimated volumes for the minimal acquisition options -----------------
    def option(t: str, k: int, selector) -> dict:
        rows = [r for r in candidates_by_class.get(t, [])
                if r["status"] == "ok" and r["date_is_new_for_class"] and selector(r)]
        # cheapest-first, but keep one proxy date per pick where possible
        rows.sort(key=lambda r: (r["proxy_date"], r["packed_size"]))
        chosen = rows[:k]
        return {
            "class": t, "files_per_class": k,
            "available_files_matching_selector": len(rows),
            "feasible": len(rows) >= k,
            "picked_files": [r["file_name"] for r in chosen],
            "picked_proxy_dates": [r["proxy_date"] for r in chosen],
            "picked_configurations": [f"D{r['d_code']}/{r['s_band_group']}" for r in chosen],
            "packed_bytes": sum(r["packed_size"] for r in chosen),
            "unpacked_bytes": sum(r["unpacked_size"] for r in chosen),
        }

    selectors = {
        "exact_configuration_match": lambda r: r["configuration_matches_existing"],
        "same_distance_axis": lambda r: r["shares_distance_axis_with_existing"],
        "same_band_axis": lambda r: r["shares_band_axis_with_existing"],
        "any_new_date_candidate": lambda r: True,
    }
    volume_options = {}
    for name, selector in selectors.items():
        for k in (1, 2, 3, 8):
            per_class_option = {t: option(t, k, selector) for t in LEGACY3}
            volume_options[f"{name}__{k}_per_class"] = {
                "selector": name, "files_per_class": k,
                "per_class": per_class_option,
                "all_three_classes_feasible": all(
                    v["feasible"] for v in per_class_option.values()),
                "total_packed_bytes": sum(v["packed_bytes"] for v in per_class_option.values()),
                "total_unpacked_bytes": sum(v["unpacked_bytes"] for v in per_class_option.values()),
            }

    with open(CANDIDATES, "r", encoding="utf-8-sig", newline="") as fh:
        frozen_rows = list(csv.DictReader(fh))
    frozen_totals = {}
    for row in frozen_rows:
        bucket = frozen_totals.setdefault(row["t"], {"files": 0, "packed_bytes": 0,
                                                     "unpacked_bytes": 0})
        bucket["files"] += 1
        bucket["packed_bytes"] += int(row["packed_size"])
        bucket["unpacked_bytes"] += int(row["unpacked_size"])

    q1_by_class = {
        t: {
            "existing_proxy_dates": per_class[t]["existing_on_g"]["proxy_dates"],
            "candidate_proxy_dates": per_class[t]["candidate_dates_all"],
            "new_proxy_date_files": per_class[t]["new_date_file_count"],
            "new_proxy_dates": sorted(per_class[t]["new_date_groups"]),
            "new_night_proxy_files": per_class[t]["new_night_file_count"],
            "exact_configuration_match_on_a_new_date": (
                per_class[t]["has_exact_configuration_match_on_a_new_date"]),
        } for t in LEGACY3}

    answers = {
        "Q1_each_legacy_class_has_a_new_proxy_date_candidate": {
            t: per_class[t]["has_candidate_on_a_new_proxy_date"] for t in LEGACY3},
        "Q1_by_class": q1_by_class,
        "Q1_verdict": ("yes - all three legacy classes have un-downloaded archive members whose "
                       "file-time proxy falls on a calendar date that none of their existing "
                       "eight G-drive files uses"
                       if all(per_class[t]["has_candidate_on_a_new_proxy_date"] for t in LEGACY3)
                       else "no - at least one legacy class has no candidate on a new date"),
        "Q1_terminology_guard": ("A different file-time proxy date is a different *file-time "
                                 "proxy*, not a measured independent acquisition session; the "
                                 "RAR HTIME can be inherited from the source file's "
                                 "modification time."),
        "Q2_can_candidates_support_a_file_level_test_not_used_in_selection": {
            "mechanical_availability": ("yes - the candidates are absent from the 104-file G "
                                        "manifest and from every prior project artefact, so a "
                                        "file drawn from them has never taken part in this "
                                        "project's fitting or selection"),
            "claim_that_is_supported": ("file-level held-out evaluation with a different "
                                        "file-time-proxy date for the class"),
            "claims_that_are_NOT_supported": [
                "cross-session evaluation",
                "independent acquisition session",
                "like-for-like comparison with the existing eight files",
                "13-class closed-set metric that differs only in 'held-out' status",
            ],
            "confound": ("every new-date candidate for the three legacy classes also differs "
                         "from the existing eight files on at least one acquisition-"
                         "configuration axis (distance code D and/or band group S); see "
                         "per_class.*.has_exact_configuration_match_on_a_new_date"),
        },
        "Q2_per_class_support": {
            t: {
                "supports_file_level_holdout_on_a_new_proxy_date": (
                    per_class[t]["has_candidate_on_a_new_proxy_date"]),
                "supports_new_night_proxy": per_class[t]["has_candidate_in_a_new_night_proxy"],
                "new_date_files": per_class[t]["new_date_file_count"],
                "new_night_files": per_class[t]["new_night_file_count"],
                "exact_configuration_match_on_a_new_date": (
                    per_class[t]["has_exact_configuration_match_on_a_new_date"]),
            } for t in LEGACY3},
        "Q3_classes_still_missing_a_cross_batch_proxy": {
            "probed_in_this_task": LEGACY3,
            "not_probed_in_this_task": [t for t in ALL_CLASSES if t not in LEGACY3],
            "note": ("META-HEADER1 was authorised only for the 88 un-downloaded members of "
                     "T0000/T0010/T0101. The other ten selected classes still have no measured "
                     "file-time proxy for any of their un-downloaded archive members, so for "
                     "those classes nothing changed. Even for the three probed classes, a "
                     "different file-time proxy is at most a 'different acquisition-night "
                     "proxy', never a proven independent session."),
            "classes_with_no_cross_batch_proxy_at_all": [t for t in ALL_CLASSES],
            "classes_with_a_candidate_outside_their_existing_proxy_dates": [
                t for t in LEGACY3 if per_class[t]["has_candidate_on_a_new_proxy_date"]],
        },
    }

    payload = {
        "schema": "dronerfa-meta-header1-availability/v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": {
            "probe": probe_path,
            "probe_generated_at": probe.get("generated_at"),
            "mat_headers": MAT_HEADERS,
            "frozen_candidates": CANDIDATES,
            "campaign_grouping": GROUPS,
        },
        "existing_corpus": {
            "files": len(heads["records"]),
            "classes": len({r["class_id"] for r in heads["records"]}),
            "campaign_count": grouping.get("campaign_count"),
            "night_count": grouping.get("night_count"),
            "night_keys": sorted(grouping.get("nights", {}).keys()),
            "classes_all_on_one_day": grouping.get("classes_all_on_one_day"),
        },
        "terminology": {
            "file_time_proxy": ("MATLAB v7.3 'Created on' and RAR5 FHEXTRA_HTIME are both file "
                                "times; the RAR value can be inherited from the source file's "
                                "modification time, so agreement between them is not two "
                                "independent acquisition logs."),
            "proxy_date": "calendar date of the +08:00 file-time proxy",
            "night_key": "date of (proxy local time - 6 h); an acquisition-shift proxy only",
            "forbidden_claims": [
                "different file-time proxy == independent acquisition session",
                "different night_key == proven cross-session",
                "13 classes can be given a cross-session final test",
            ],
        },
        "per_class": per_class,
        "anomalies": anomalies,
        "remaining_frozen_candidate_volume_for_legacy3": frozen_totals,
        "download_volume_options": volume_options,
        "answers": answers,
        "probe_status_counts": status_counts,
        "probe_http_stats": probe.get("http_stats") or {},
        "probe_aborted": probe.get("aborted"),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class_name", "t_code", "provenance", "proxy_date", "night_key",
                         "d_code", "distance_label", "s_band_group", "files",
                         "packed_bytes", "unpacked_bytes", "file_names"])
        for t in LEGACY3:
            info = per_class[t]
            detail = info["existing_on_g"]["files_detail"]
            for date in info["existing_on_g"]["proxy_dates"]:
                subset = [r for r in detail if r["proxy_date"] == date]
                for (d_code, band) in sorted({(r["d_code"], r["s_band_group"]) for r in subset}):
                    names = sorted(r["file_name"] for r in subset
                                   if r["d_code"] == d_code and r["s_band_group"] == band)
                    writer.writerow([info["class_name"], t, "existing_on_g", date,
                                     subset[0]["night_key"], d_code,
                                     D_LABELS.get(d_code, "?"), band, len(names),
                                     "", "", ";".join(names)])
            for cfg in info["candidate_configuration_groups"]:
                writer.writerow([info["class_name"], t, "archive_candidate",
                                 cfg["proxy_date"], cfg["night_key"], cfg["d_code"],
                                 cfg["distance_label"], cfg["s_band_group"], cfg["files"],
                                 cfg["packed_bytes"], cfg["unpacked_bytes"],
                                 ";".join(cfg["file_names"])])

    print(json.dumps({
        "answers_Q1": answers["Q1_each_legacy_class_has_a_new_proxy_date_candidate"],
        "per_class_summary": {
            t: {"existing_dates": per_class[t]["existing_on_g"]["proxy_dates"],
                "existing_configs": per_class[t]["existing_on_g"]["configurations"],
                "candidate_dates": per_class[t]["candidate_dates_all"],
                "new_date_files": per_class[t]["new_date_file_count"],
                "new_night_files": per_class[t]["new_night_file_count"],
                "exact_config_match_on_new_date": (
                    per_class[t]["has_exact_configuration_match_on_a_new_date"]),
                "anomalies": per_class[t]["candidates_anomalous"]}
            for t in LEGACY3},
        "volume_any_new_date_2_per_class": volume_options["any_new_date_candidate__2_per_class"],
        "anomalies_total": len(anomalies),
        "wrote": [out_json, out_csv],
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
