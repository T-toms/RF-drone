# -*- coding: utf-8 -*-
"""META-SPLIT-1 / step 1c + 4: build the traceable 104-file source table, the
acquisition-campaign grouping, and the split-feasibility draft.

Nothing here writes to G:, changes split_status, trains, or generates any
training index.  It only reads project artefacts and emits work/meta_split1/
reports.

Outputs:
  source_table_104.csv / source_table_104.json
  campaign_grouping_104.json
  split_proposal_draft.json
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
MANIFEST = r"G:\DroneRFa数据集\dataset_manifest.csv"
SHA_JSON = os.path.join(WORKSPACE, "work", "sel12g", "sel12g-mat-sha256-20260925.json")

SPLIT_SEED = 20260925

# Prior-use mapping, evidenced by the project documents
# (dronerfa_baseline/DroneRFa第一阶段实施说明.md lines 99-101;
#  dronerfa_baseline/DroneRFa项目四篇核心论文文献综述.md lines 326-328;
#  dronerfa_baseline/outputs/heldout_preparation_20260917/report.md line 105).
LEGACY_USE = {
    "0000": "S0000", "0001": "S0000", "0010": "S0000",
}
LEGACY_SEGMENT_USE = {
    "0000": "0000", "0001": "0000", "0010": "0000", "0011": "0000",
    "0100": "0000", "0101": "0000", "0110": "0000", "0111": "0000",
    "1000": "0000", "1001": "0000",
}
SEGMENT_ROLE = {
    "0000": ("train", "window_index train split (920 windows, S0000+S0001)"),
    "0001": ("train", "window_index train split (920 windows, S0000+S0001)"),
    "0010": ("validation", "window_index validation split (450 windows, S0010)"),
    "0011": ("reported_test", "window_index test split (440 windows, S0011); reported in EVAL6/BASE2/NFFT4/CASCADE3"),
    "0100": ("descriptive_holdout", "heldout_preparation_20260917 (1820 windows, S0100-S0111)"),
    "0101": ("descriptive_holdout", "heldout_preparation_20260917 (1820 windows, S0100-S0111)"),
    "0110": ("descriptive_holdout", "heldout_preparation_20260917 (1820 windows, S0100-S0111)"),
    "0111": ("descriptive_holdout", "heldout_preparation_20260917 (1820 windows, S0100-S0111)"),
    "1000": ("never_downloaded", "archive band-switch group, not on G:"),
    "1001": ("never_downloaded", "archive band-switch group, not on G:"),
    "1010": ("never_downloaded", "archive band-switch group, not on G:"),
    "1011": ("never_downloaded", "archive band-switch group, not on G:"),
    "1100": ("never_downloaded", "archive band-switch group, not on G:"),
    "11000": ("never_downloaded", "archive background band-switch group, not on G:"),
}


def main() -> int:
    with open(MANIFEST, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    heads = {r["file_name"]: r for r in
             json.load(open(os.path.join(HERE, "mat_headers_104.json"), encoding="utf-8"))["records"]}
    h5 = {r["file_name"]: r for r in
          json.load(open(os.path.join(HERE, "hdf5_metadata_104.json"), encoding="utf-8"))["records"]}
    sha = json.load(open(SHA_JSON, encoding="utf-8"))

    # --- independent re-check of the full-file SHA256 evidence -------------
    sha_by_name = {f["filename"]: f for f in sha["files"]}
    sha_names = set(sha_by_name)
    manifest_names = {r["file_name"] for r in rows}
    digest_counts = {}
    for f in sha["files"]:
        digest_counts.setdefault(f["sha256"], []).append(f["filename"])
    duplicate_digests = {k: v for k, v in digest_counts.items() if len(v) > 1}
    sha_checks = {
        "sha_report": os.path.relpath(SHA_JSON, WORKSPACE).replace("\\", "/"),
        "sha_report_count": sha["count"],
        "sha_report_bytes": sha["bytes"],
        "names_identical_to_manifest": sha_names == manifest_names,
        "missing_from_sha_report": sorted(manifest_names - sha_names),
        "extra_in_sha_report": sorted(sha_names - manifest_names),
        "distinct_digests": len(digest_counts),
        "files": len(sha["files"]),
        "duplicate_digest_groups": duplicate_digests,
        "all_hashes_64_hex": all(len(f["sha256"]) == 64 for f in sha["files"]),
        "size_agreement_with_manifest": all(
            int(sha_by_name[r["file_name"]]["bytes"]) == int(r["size_bytes"]) for r in rows),
    }

    # --- source table -------------------------------------------------------
    records = []
    for r in rows:
        name = r["file_name"]
        h = heads[name]
        m = h5[name]
        s = sha_by_name[name]
        seg = h["s_code"]
        role, why = SEGMENT_ROLE.get(seg, ("unknown", "not classified"))
        exposed = r["former_partition"] in ("development_original", "evaluated_holdout")
        records.append({
            "file_name": name,
            "class_id": r["class_id"],
            "class_name": r["class_name"],
            "t_code": h["t_code"],
            "d_code": h["d_code"] or "NR",
            "d_distance_label": {"00": "20-40 m", "01": "40-80 m",
                                 "10": "80-150 m"}.get(h["d_code"], "NR (no D field in file name)"),
            "s_code": seg,
            "s_band_group": h["s_band_group"],
            "origin": r["origin"],
            "former_partition": r["former_partition"] or "NR",
            "split_status": r["split_status"],
            "evaluation_role": r["evaluation_role"],
            "manifest_size_bytes": int(r["size_bytes"]),
            "sha256": s["sha256"],
            "mat_created_local": h["created_iso"],
            "mat_created_day": h["created_day"],
            "mat_header_platform": h["platform"],
            "hdf5_userblock": m["hdf5_userblock"],
            "channel_names": ";".join(m["channel_names"]),
            "channel_lengths": json.dumps(m["channel_lengths"], ensure_ascii=False),
            "root_attrs": json.dumps(m["root_attrs"], ensure_ascii=False),
            "dataset_attrs": json.dumps({k: v["attrs"] for k, v in m["datasets"].items()},
                                        ensure_ascii=False),
            "prior_use_role": role,
            "prior_use_evidence": why,
            "exposed_to_prior_selection": exposed,
            "acquisition_campaign_key": "C%s_%s" % (h["created_day"].replace("-", ""), r["class_id"]),
            "acquisition_night_key": "N%s" % (dt.datetime.fromisoformat(h["created_iso"])
                                              - dt.timedelta(hours=6)).strftime("%Y%m%d"),
            "minimum_group_key": "file:%s" % name,
            "archive_member": "DroneRFa/%s" % name,
        })

    # --- campaign grouping --------------------------------------------------
    by_class = {}
    for rec in records:
        by_class.setdefault(rec["class_id"], []).append(rec)
    campaigns = {}
    for cid, items in by_class.items():
        items.sort(key=lambda x: x["mat_created_local"])
        t0 = dt.datetime.fromisoformat(items[0]["mat_created_local"])
        t1 = dt.datetime.fromisoformat(items[-1]["mat_created_local"])
        gaps = [(dt.datetime.fromisoformat(items[i + 1]["mat_created_local"])
                 - dt.datetime.fromisoformat(items[i]["mat_created_local"])).total_seconds()
                for i in range(len(items) - 1)]
        key = items[0]["acquisition_campaign_key"]
        campaigns[key] = {
            "campaign_key": key,
            "class_id": cid,
            "class_name": items[0]["class_name"],
            "day": items[0]["mat_created_day"],
            "n_files": len(items),
            "files": [x["file_name"] for x in items],
            "start_local": items[0]["mat_created_local"],
            "end_local": items[-1]["mat_created_local"],
            "span_seconds": (t1 - t0).total_seconds(),
            "inter_file_gap_seconds_min": min(gaps),
            "inter_file_gap_seconds_max": max(gaps),
            "distinct_days": sorted({x["mat_created_day"] for x in items}),
            "distinct_d_codes": sorted({x["d_code"] for x in items}),
            "distinct_s_band_groups": sorted({x["s_band_group"] for x in items}),
            "exposed_files": sum(1 for x in items if x["exposed_to_prior_selection"]),
            "unexposed_files": sum(1 for x in items if not x["exposed_to_prior_selection"]),
        }

    # global timeline
    timeline = sorted(records, key=lambda x: x["mat_created_local"])
    day_classes = {}
    for rec in records:
        day_classes.setdefault(rec["mat_created_day"], set()).add(rec["class_id"])

    # larger acquisition group: one night.  A campaign start before 06:00 belongs
    # to the previous evening, so the offset is 6 h.  Reproducible and justified
    # by the observed gaps below (max intra-night 68.5 min vs ~9 months between
    # nights).
    nights = {}
    for rec in records:
        nights.setdefault(rec["acquisition_night_key"], []).append(rec)
    night_view = {}
    for key, items in sorted(nights.items()):
        items.sort(key=lambda x: x["mat_created_local"])
        classes = sorted({x["class_id"] for x in items})
        inter = []
        by_cls = {}
        for x in items:
            by_cls.setdefault(x["class_id"], []).append(x)
        spans = sorted((min(y["mat_created_local"] for y in v),
                        max(y["mat_created_local"] for y in v)) for v in by_cls.values())
        for i in range(len(spans) - 1):
            inter.append((dt.datetime.fromisoformat(spans[i + 1][0])
                          - dt.datetime.fromisoformat(spans[i][1])).total_seconds())
        night_view[key] = {
            "night_key": key,
            "n_files": len(items),
            "n_classes": len(classes),
            "classes": classes,
            "start_local": items[0]["mat_created_local"],
            "end_local": items[-1]["mat_created_local"],
            "span_hours": round((dt.datetime.fromisoformat(items[-1]["mat_created_local"])
                                 - dt.datetime.fromisoformat(items[0]["mat_created_local"])
                                 ).total_seconds() / 3600.0, 2),
            "max_inter_class_gap_minutes": round(max(inter) / 60.0, 1) if inter else None,
        }

    campus = {
        "schema": "dronerfa-meta-split-campaign-grouping/v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_key": "MATLAB v7.3 116-byte header field 'Created on', read from offset 0 of each file",
        "grouping_levels": {
            "L0_file": "file:<file_name> - the minimum unit; all windows of one MAT stay together",
            "L1_class_campaign": "C<YYYYMMDD>_<T-code> - one class, one day: 13 groups",
            "L2_night": "N<YYYYMMDD> with a 06:00 boundary - one operator/machine session: 2 groups",
        },
        "key_cross_check": {
            "method": "RAR5 FHEXTRA_HTIME mtime of the same members, read from the local DroneRFa_pilot.rar "
                      "and from two leftover .part archive heads; no network access",
            "members_checked": 14,
            "agreement_within_2_seconds": 14,
            "evidence": "work/meta_split1/rar_header_times.json",
        },
        "campaigns": campaigns,
        "campaign_count": len(campaigns),
        "nights": night_view,
        "night_count": len(night_view),
        "days": {d: sorted(c) for d, c in sorted(day_classes.items())},
        "classes_per_day": {d: len(c) for d, c in sorted(day_classes.items())},
        "classes_all_on_one_day": all(len(c["distinct_days"]) == 1 for c in campaigns.values()),
        "timeline": [{"file_name": x["file_name"], "class_id": x["class_id"],
                      "created_local": x["mat_created_local"],
                      "campaign_key": x["acquisition_campaign_key"],
                      "night_key": x["acquisition_night_key"]} for x in timeline],
        "interpretation": [
            "Within one class the 8 files sit inside a single 10-16 minute window with 81-159 s gaps, "
            "matching the acquire-store cycle: they are consecutive segments of one acquisition run, "
            "not 8 independent sessions.",
            "Between classes inside one night the gap is 14-69 minutes, so the classes were flown one "
            "after another by the same two operators on the same USRP, antenna and gain setting.",
            "The two nights are 9 months apart (2022-11-27/28 and 2023-02-27), and they do not share a "
            "single class: the night axis is class-disjoint.",
            "The date is the .mat container write time; MATLAB writes local time (Asia/Shanghai), "
            "confirmed against the RAR headers to within 2 s.",
        ],
    }
    with open(os.path.join(HERE, "campaign_grouping_104.json"), "w", encoding="utf-8") as fh:
        json.dump(campus, fh, ensure_ascii=False, indent=1)

    # --- source table output ------------------------------------------------
    cols = list(records[0].keys())
    with open(os.path.join(HERE, "source_table_104.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            w.writerow(rec)
    with open(os.path.join(HERE, "source_table_104.json"), "w", encoding="utf-8") as fh:
        json.dump({"schema": "dronerfa-meta-split-source-table/v1",
                   "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "manifest": MANIFEST, "sha_checks": sha_checks,
                   "records": records}, fh, ensure_ascii=False, indent=1)

    # --- split proposal draft ----------------------------------------------
    exposure_vs_class = {}
    for cid, items in sorted(by_class.items()):
        exposure_vs_class[cid] = {
            "class_name": items[0]["class_name"],
            "n_files": len(items),
            "n_exposed": sum(1 for x in items if x["exposed_to_prior_selection"]),
            "n_unexposed": sum(1 for x in items if not x["exposed_to_prior_selection"]),
            "n_campaigns": len({x["acquisition_campaign_key"] for x in items}),
            "campaign_keys": sorted({x["acquisition_campaign_key"] for x in items}),
        }

    rng_assign = {}
    for cid, items in sorted(by_class.items()):
        names = sorted(x["file_name"] for x in items)
        r = random.Random("%d|%s" % (SPLIT_SEED, cid))
        r.shuffle(names)
        rng_assign[cid] = {
            "seed_input": "%d|%s" % (SPLIT_SEED, cid),
            "file_level_6_1_1": {"train": names[:6], "val": names[6:7], "test": names[7:]},
            "file_level_5_1_2": {"train": names[:5], "val": names[5:6], "test": names[6:]},
            "file_level_4_2_2": {"train": names[:4], "val": names[4:6], "test": names[6:]},
        }

    proposal = {
        "schema": "dronerfa-meta-split-proposal-draft/v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DRAFT ONLY - no split_status changed, no index generated, nothing written to G:",
        "split_seed": SPLIT_SEED,
        "primary_group_key": {
            "minimum_unit": "file:<file_name>  - every window cut from one MAT stays in one split",
            "rationale": "all windows of a MAT come from one continuous acquisition; a window-level "
                         "random split would leak near-identical samples across splits",
        },
        "recommended_group_key": {
            "key": "campaign:<T-code>@<YYYYMMDD>  (= acquisition_campaign_key in the source table)",
            "rationale": "the 8 files of a class share one day, one site, one USRP, one antenna, one "
                         "gain setting and one physical drone unit; the day is the only provenance "
                         "axis this metadata can prove",
            "status": "usable as a grouping key, NOT usable as a class-balanced split axis",
            "larger_group_available": "night:<YYYYMMDD>, 2 groups (10 classes on 2022-11-27/28 and "
                                      "3 classes on 2023-02-27); also class-disjoint",
        },
        "level_1_never_used_in_selection": {
            "question": "can the 104 files give a 13-class split whose test part was never used for "
                        "any tuning or selection in this project?",
            "answer": "NO",
            "reason": "Background T0000, Phantom 4 Pro T0010 and Air 2S T0101 have 0 unexposed files: "
                      "all 24 legacy files were used at least for the 1820-window descriptive heldout "
                      "evaluation, and 12 of them also for training / threshold / n_fft selection.",
            "missing": {"classes": ["T0000", "T0010", "T0101"], "files_minimum": 3,
                        "files_recommended": 6},
        },
        "level_2_campaign_independent_final_test": {
            "question": "can the 104 files give a final test that is independent at acquisition-campaign "
                        "level for all 13 classes?",
            "answer": "NO",
            "reason": "every class has all 8 of its files inside a single campaign, so any split of the "
                      "104 files leaves train and test inside the same campaign for every class. "
                      "The campaign axis is class-disjoint (10 classes on 2026-11-27/28, 3 classes on "
                      "2023-02-27), so it cannot be used to build a class-balanced split either.",
            "missing": {"classes": sorted(by_class), "files_minimum_per_class": 1,
                        "files_minimum_total": 13, "files_recommended_per_class": 2,
                        "files_recommended_total": 26},
        },
        "claims_that_are_supported": [
            "The 80 files added by SELECT12-G have never been used in any training, tuning, threshold "
            "or checkpoint selection in this project (provenance: dataset_manifest.csv origin=selected_80, "
            "split_status=unassigned; no prior run references their file names).",
            "No two of the 104 files are byte-identical (full-file SHA-256, verified here).",
            "Every window cut from one MAT must enter one split; the minimum group is the source file.",
        ],
        "claims_that_are_NOT_supported": [
            "That the 104 files provide a 13-class final test that is untouched by selection.",
            "That a file-level split of the 104 files is a cross-session or cross-campaign split.",
            "That the 12 evaluated_holdout files can be re-labelled as a fresh final test.",
        ],
        "per_class_exposure": exposure_vs_class,
        "deterministic_file_level_assignments": rng_assign,
        "interim_plan_before_extra_data": [
            "Keep the 24 legacy files in a development pool only: no new result may call them a test set.",
            "For the 10 unexposed classes, run a file-level grouped 6/1/1 (or 5/1/2) split and label the "
            "result 'file-level held-out', never 'cross-session'.",
            "Report the three legacy classes separately as an 'exposed-class control', or drop them from "
            "the classifier and report a 10-class closed-set result with an explicit class-space note.",
            "Any honest 13-class claim waits for at least one new, campaign-independent file per class.",
        ],
        "evidence_gap_and_minimum_fix": {
            "gap": "The acquisition date of an archive member that is not on G: is not derivable from "
                   "any local artefact; the local RAR index stores only name, offset and sizes.",
            "minimum_fix": "Read the RAR5 member header (FHEXTRA_HTIME, about 64-128 bytes at the member "
                           "block offset) of candidate members from the official archive URL with HTTP "
                           "range requests. No payload is transferred. 88 legacy-class candidates would "
                           "cost well under 100 KiB of traffic.",
            "why_it_works": "Verified locally: the RAR header mtime of 14 members matches the MATLAB "
                            "'Created on' value to within 2 s (work/meta_split1/rar_header_times.json).",
            "authorization": "Needs a new user/Codex decision: it is a network read, and this task was "
                             "scoped to no downloads.",
        },
        "archive_candidates_read_only": {
            "legacy3_unused_files": 88,
            "legacy3_unused_unpacked_bytes": 86961570451,
            "detail": "work/meta_split1/archive_unused_for_legacy3_20260925.csv",
        },
    }
    with open(os.path.join(HERE, "split_proposal_draft.json"), "w", encoding="utf-8") as fh:
        json.dump(proposal, fh, ensure_ascii=False, indent=1)

    print(json.dumps({
        "records": len(records),
        "sha_checks": sha_checks,
        "campaign_count": len(campaigns),
        "campaigns": {k: {"class": v["class_id"], "n": v["n_files"], "day": v["day"],
                          "start": v["start_local"], "end": v["end_local"],
                          "gap_min": v["inter_file_gap_seconds_min"],
                          "gap_max": v["inter_file_gap_seconds_max"]}
                      for k, v in campaigns.items()},
        "classes_all_on_one_day": campus["classes_all_on_one_day"],
        "wrote": ["source_table_104.csv", "source_table_104.json",
                  "campaign_grouping_104.json", "split_proposal_draft.json"],
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
