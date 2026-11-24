#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""META-SPLIT2: build the 13-class development-use source-file split CANDIDATE.

Deterministic builder. Reads only the three frozen inputs:

  1. G:\\DroneRFa数据集\\dataset_manifest.csv                 (104 rows, SHA256 frozen)
  2. work/meta_split1/source_table_104.csv                  (MAT file-header times)
  3. work/sel12g/sel12g-mat-sha256-20260925.json            (SELECT12-G3 hashes)

Writes work/meta_split2/split_candidate_104_20260926.{csv,json}.

It never touches G:\\DroneRFa数据集\\dataset_manifest.csv, never reads MAT payloads,
never trains, never extracts windows and never downloads anything.

Rule (frozen by Codex, see work/deepseek-tasks/20260926-meta-split2-dev-files.md):

  new 10 classes (origin=selected_80, 8 files each)
      sort by MAT file-header time ascending, ties by file name ascending
      rank 1-5 -> train, rank 6 -> val, rank 7-8 -> development_test

  legacy 3 classes T0000/T0010/T0101
      former_partition=development_original  (4 files) -> train
      former_partition=evaluated_holdout     (4 files) -> sorted by time ascending
          first 2 -> val, last 2 -> development_test_exposed

Expected totals: 62 train / 16 val / 26 test (20 new-class + 6 legacy exposed).
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# frozen constants
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "work" / "meta_split2"

MANIFEST = Path(r"G:\DroneRFa数据集\dataset_manifest.csv")
SOURCE_TABLE = ROOT / "work" / "meta_split1" / "source_table_104.csv"
SHA_JSON = ROOT / "work" / "sel12g" / "sel12g-mat-sha256-20260925.json"
PRIOR_USE_SCAN = ROOT / "work" / "meta_split1" / "prior_use_scan.json"

MANIFEST_SHA256 = "9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d"

NEW_ORIGIN = "selected_80"
LEGACY_ORIGIN = "baseline_24"
LEGACY_CLASSES = ("T0000", "T0010", "T0101")
NEW_CLASSES = (
    "T0001", "T0011", "T0100", "T0110", "T0111",
    "T1000", "T1001", "T1010", "T1011", "T1100",
)

ROLE_TRAIN = "train"
ROLE_VAL = "val"
ROLE_TEST_NEW = "development_test"
ROLE_TEST_LEGACY = "development_test_exposed"

EXPECTED_TOTAL = 104
EXPECTED_TRAIN = 62
EXPECTED_VAL = 16
EXPECTED_TEST_NEW = 20
EXPECTED_TEST_LEGACY = 6
EXPECTED_TEST = EXPECTED_TEST_NEW + EXPECTED_TEST_LEGACY

# naming gate: this candidate must never claim a fresh / independent / final test
FORBIDDEN_ROLE_TOKENS = ("final_test", "fresh", "cross_session", "cross-session",
                         "independent_test", "test_final")

CSV_FIELDS = [
    "file_name",
    "class_id",
    "class_name",
    "role",
    "role_rank_in_class",
    "role_purpose",
    "metrics_scope",
    "prior_use_role",
    "prior_use_evidence",
    "archive_segment_role",
    "archive_segment_evidence",
    "archive_s_code",
    "former_partition",
    "origin",
    "exposed_to_prior_selection",
    "file_time_proxy",
    "campaign_proxy",
    "night_proxy",
    "mat_sha256",
    "g_path",
    "size_bytes",
    "class_index",
    "minimum_group_key",
    "window_rule",
]

PURPOSE = {
    ROLE_TRAIN: "model fitting",
    ROLE_VAL: "model selection / threshold and hyper-parameter choice",
    ROLE_TEST_NEW: "file-level time-blocked holdout for the new 10 classes; "
                   "same file-time cluster, NOT an independent acquisition session",
    ROLE_TEST_LEGACY: "already-exposed legacy files; descriptive control only, "
                      "never a fresh or final test",
}

METRICS_SCOPE_DEV = "development"
METRICS_SCOPE_NEW10 = "new10_file_level_test_primary"
METRICS_SCOPE_LEGACY = "legacy_exposed_control"

WINDOW_RULE = ("any future 1M-sample window inherits the role of its source MAT; "
               "a source MAT must never be split across roles")


# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path):
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def fail(msg: str) -> None:
    """Any input-identity or time-field doubt is a hard stop (report needs_codex)."""
    sys.stderr.write("META-SPLIT2 STOP: %s\n" % msg)
    raise SystemExit(2)


# --------------------------------------------------------------------------
def main() -> int:
    # ---- input identity gate ---------------------------------------------
    if not MANIFEST.is_file():
        fail("G manifest missing: %s" % MANIFEST)
    manifest_sha = sha256_file(MANIFEST)
    if manifest_sha != MANIFEST_SHA256:
        fail("G manifest SHA256 mismatch: got %s expected %s"
             % (manifest_sha, MANIFEST_SHA256))
    source_sha = sha256_file(SOURCE_TABLE)
    sha_json_sha = sha256_file(SHA_JSON)

    manifest = load_csv(MANIFEST)
    source_rows = load_csv(SOURCE_TABLE)
    sha_files = json.loads(SHA_JSON.read_text(encoding="utf-8"))["files"]

    if len(manifest) != EXPECTED_TOTAL:
        fail("G manifest has %d rows, expected %d" % (len(manifest), EXPECTED_TOTAL))

    source = {r["file_name"]: r for r in source_rows}
    shas = {r["filename"]: r for r in sha_files}
    man = {r["file_name"]: r for r in manifest}
    if not (set(source) == set(shas) == set(man)):
        fail("the three frozen inputs do not describe the same 104 file names")

    # ---- per-file consistency + time-field sanity ------------------------
    records = []
    for row in manifest:
        name = row["file_name"]
        s = source[name]
        h = shas[name]

        if int(row["size_bytes"]) != int(s["manifest_size_bytes"]):
            fail("size disagreement (manifest vs source) for %s" % name)
        if int(row["size_bytes"]) != int(h["bytes"]):
            fail("size disagreement (manifest vs sha json) for %s" % name)
        if s["sha256"] != h["sha256"]:
            fail("sha256 disagreement (source vs sha json) for %s" % name)
        if row["class_id"] != s["class_id"] or row["origin"] != s["origin"]:
            fail("class/origin disagreement for %s" % name)
        if h["path"] != row["unified_path"]:
            fail("G path disagreement for %s" % name)
        if os.path.basename(row["unified_path"]) != name:
            fail("G path basename mismatch for %s" % name)
        if not row["unified_path"].upper().startswith("G:"):
            fail("non-G path for %s" % name)
        # the sha record uses a different origin vocabulary ("to_download"); the
        # partition must nevertheless be identical
        if (row["origin"] == LEGACY_ORIGIN) != (h["origin"] == LEGACY_ORIGIN):
            fail("origin vocabulary partition differs for %s" % name)

        # time field must parse and agree with its own derived day/night keys
        try:
            t = dt.datetime.fromisoformat(s["mat_created_local"])
        except ValueError:
            fail("unparsable MAT header time for %s: %r" % (name, s["mat_created_local"]))
        if t.strftime("%Y-%m-%d") != s["mat_created_day"]:
            fail("mat_created_day inconsistent with mat_created_local for %s" % name)
        if "N%s" % (t - dt.timedelta(hours=6)).strftime("%Y%m%d") != s["acquisition_night_key"]:
            fail("night key not reproducible from MAT time for %s" % name)
        if "C%s_%s" % (t.strftime("%Y%m%d"), row["class_id"]) != s["acquisition_campaign_key"]:
            fail("campaign key not reproducible from MAT time for %s" % name)

        records.append({
            "file_name": name,
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "class_index": int(row["class_index"]),
            "origin": row["origin"],
            "former_partition": row["former_partition"] or "NR",
            "g_path": row["unified_path"],
            "size_bytes": int(row["size_bytes"]),
            "mat_sha256": s["sha256"],
            "file_time_proxy": s["mat_created_local"],
            "_time": t,
            "campaign_proxy": s["acquisition_campaign_key"],
            "night_proxy": s["acquisition_night_key"],
            "archive_segment_role": s["prior_use_role"],
            "archive_segment_evidence": s["prior_use_evidence"],
            "archive_s_code": s["s_code"],
            "exposed_to_prior_selection": s["exposed_to_prior_selection"] == "True",
            "minimum_group_key": s["minimum_group_key"],
            "sha_record_origin": h["origin"],
        })

    if len(records) != EXPECTED_TOTAL:
        fail("record count %d" % len(records))

    # ---- class composition ------------------------------------------------
    by_class = {}
    for r in records:
        by_class.setdefault(r["class_id"], []).append(r)

    if set(by_class) != set(NEW_CLASSES) | set(LEGACY_CLASSES):
        fail("class set is not the expected 13 classes: %s" % sorted(by_class))
    for cid, items in sorted(by_class.items()):
        if len(items) != 8:
            fail("class %s has %d files, expected 8" % (cid, len(items)))
        origins = {x["origin"] for x in items}
        if cid in LEGACY_CLASSES:
            if origins != {LEGACY_ORIGIN}:
                fail("legacy class %s has unexpected origins %s" % (cid, origins))
        else:
            if origins != {NEW_ORIGIN}:
                fail("new class %s has unexpected origins %s" % (cid, origins))

    # a class must not straddle two campaign proxies (else the "same cluster"
    # wording below would be wrong)
    for cid, items in sorted(by_class.items()):
        if len({x["campaign_proxy"] for x in items}) != 1:
            fail("class %s spans several campaign proxies" % cid)

    # the frozen tie-break ("same time -> file name ascending") is only defined
    # behaviourally if some class actually contains a repeated file-header time
    tie_break_exercised = any(
        len({x["file_time_proxy"] for x in items}) != len(items)
        for items in by_class.values()
    )
    time_collisions = sorted(
        {x["file_time_proxy"] for items in by_class.values() for x in items
         if sum(1 for y in items if y["file_time_proxy"] == x["file_time_proxy"]) > 1}
    )

    # ---- assign roles -----------------------------------------------------
    for cid, items in sorted(by_class.items()):
        items.sort(key=lambda x: (x["file_time_proxy"], x["file_name"]))
        for i, x in enumerate(items, start=1):
            x["role_rank_in_class"] = i

        if cid in NEW_CLASSES:
            for i, x in enumerate(items, start=1):
                if i <= 5:
                    x["role"] = ROLE_TRAIN
                elif i == 6:
                    x["role"] = ROLE_VAL
                else:
                    x["role"] = ROLE_TEST_NEW
        else:
            dev = [x for x in items if x["former_partition"] == "development_original"]
            hold = [x for x in items if x["former_partition"] == "evaluated_holdout"]
            if len(dev) != 4 or len(hold) != 4:
                fail("legacy class %s is not 4 development_original + 4 evaluated_holdout"
                     % cid)
            for x in dev:
                x["role"] = ROLE_TRAIN
            for i, x in enumerate(hold, start=1):
                x["role"] = ROLE_VAL if i <= 2 else ROLE_TEST_LEGACY

    # ---- prior-use corroboration (small JSON, no MAT access) --------------
    prior_use_evidence = None
    if PRIOR_USE_SCAN.is_file():
        scan = json.loads(PRIOR_USE_SCAN.read_text(encoding="utf-8"))
        prior_use_evidence = {
            "path": str(PRIOR_USE_SCAN.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(PRIOR_USE_SCAN),
            "files_scanned": scan["files_scanned"],
            "selected80_used_in_any_data_or_model_artifact":
                scan["verdict"]["selected80_used_in_any_data_or_model_artifact"],
            "legacy24_used_in_data_or_model_artifacts":
                scan["verdict"]["legacy24_used_in_data_or_model_artifacts"],
            "selected80_appearing_in_data_or_model_artifacts":
                scan["selected80_appearing_in_data_or_model_artifacts"],
        }

    # ---- prior-use role, purpose, metrics scope ---------------------------
    # work/meta_split1/source_table_104.csv labels every row with an
    # "archive_segment_role" derived purely from the DroneRFa S-code (S0000/S0001
    # = the upstream archive's own train naming, ...).  For the 24 legacy files
    # that coincides with this project's prior use; for the 80 new files it does
    # NOT, so prior_use_role is derived from origin/exposure instead.
    if prior_use_evidence is None:
        new_evidence = ("prior_use_scan.json unavailable; derived from "
                        "origin=selected_80 and exposed_to_prior_selection=False")
    else:
        new_evidence = ("not referenced by any data_or_model_artifact in "
                        "work/meta_split1/prior_use_scan.json (%d artifacts scanned, "
                        "verdict selected80_used_in_any_data_or_model_artifact=false)"
                        % prior_use_evidence["files_scanned"])
    for r in records:
        if r["origin"] == NEW_ORIGIN:
            r["prior_use_role"] = "not_previously_used_in_project"
            r["prior_use_evidence"] = new_evidence
        else:
            r["prior_use_role"] = r["archive_segment_role"]
            r["prior_use_evidence"] = r["archive_segment_evidence"]
        r["role_purpose"] = PURPOSE[r["role"]]
        if r["role"] == ROLE_TEST_NEW:
            r["metrics_scope"] = METRICS_SCOPE_NEW10
        elif r["role"] == ROLE_TEST_LEGACY:
            r["metrics_scope"] = METRICS_SCOPE_LEGACY
        else:
            r["metrics_scope"] = METRICS_SCOPE_DEV
        r["window_rule"] = WINDOW_RULE

    # ---- naming gate ------------------------------------------------------
    for r in records:
        low = r["role"].lower()
        for tok in FORBIDDEN_ROLE_TOKENS:
            if tok in low:
                fail("forbidden role name %r on %s" % (r["role"], r["file_name"]))

    # ---- expected counts --------------------------------------------------
    counts = {
        ROLE_TRAIN: sum(1 for r in records if r["role"] == ROLE_TRAIN),
        ROLE_VAL: sum(1 for r in records if r["role"] == ROLE_VAL),
        ROLE_TEST_NEW: sum(1 for r in records if r["role"] == ROLE_TEST_NEW),
        ROLE_TEST_LEGACY: sum(1 for r in records if r["role"] == ROLE_TEST_LEGACY),
    }
    if counts[ROLE_TRAIN] != EXPECTED_TRAIN or counts[ROLE_VAL] != EXPECTED_VAL:
        fail("unexpected train/val counts: %s" % counts)
    if counts[ROLE_TEST_NEW] != EXPECTED_TEST_NEW or counts[ROLE_TEST_LEGACY] != EXPECTED_TEST_LEGACY:
        fail("unexpected test counts: %s" % counts)
    if sum(counts.values()) != EXPECTED_TOTAL:
        fail("role counts do not sum to 104: %s" % counts)

    # ---- emit -------------------------------------------------------------
    records.sort(key=lambda r: (r["class_index"], r["role_rank_in_class"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "split_candidate_104_20260926.csv"
    json_path = OUT_DIR / "split_candidate_104_20260926.json"

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    per_class = {}
    for cid in sorted(by_class, key=lambda c: by_class[c][0]["class_index"]):
        items = sorted(by_class[cid], key=lambda x: x["role_rank_in_class"])
        per_class[cid] = {
            "class_name": items[0]["class_name"],
            "class_index": items[0]["class_index"],
            "origin": items[0]["origin"],
            "campaign_proxy": items[0]["campaign_proxy"],
            "night_proxy": items[0]["night_proxy"],
            "file_time_span": [items[0]["file_time_proxy"], items[-1]["file_time_proxy"]],
            "train": [x["file_name"] for x in items if x["role"] == ROLE_TRAIN],
            "val": [x["file_name"] for x in items if x["role"] == ROLE_VAL],
            "test": [x["file_name"] for x in items
                     if x["role"] in (ROLE_TEST_NEW, ROLE_TEST_LEGACY)],
            "test_role": (ROLE_TEST_NEW if items[0]["origin"] == NEW_ORIGIN
                          else ROLE_TEST_LEGACY),
        }

    doc = {
        "schema": "dronerfa-meta-split2-split-candidate/v1",
        "task": "META-SPLIT2",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "candidate_pending_codex_review_not_frozen",
        "builder": "work/meta_split2/build_split_candidate.py",
        "inputs": {
            "g_manifest": {
                "path": str(MANIFEST),
                "sha256": manifest_sha,
                "sha256_expected": MANIFEST_SHA256,
                "sha256_match": manifest_sha == MANIFEST_SHA256,
                "rows": len(manifest),
            },
            "source_table_104": {
                "path": "work/meta_split1/source_table_104.csv",
                "sha256": source_sha,
                "rows": len(source_rows),
                "time_field_used": "mat_created_local",
                "time_field_meaning": "MATLAB 'Created on' file-header time "
                                      "(file-time proxy, NOT proven radio acquisition time)",
            },
            "mat_sha256": {
                "path": "work/sel12g/sel12g-mat-sha256-20260925.json",
                "sha256": sha_json_sha,
                "rows": len(sha_files),
                "origin_vocabulary_note": "this record labels the 80 new files "
                                          "origin='to_download' while the G manifest labels "
                                          "them origin='selected_80'; the two labels "
                                          "partition the same 104 files identically "
                                          "(24 baseline_24 + 80 new), verified per file",
                "origin_vocabulary_partition_identical": True,
            },
            "prior_use_corroboration": prior_use_evidence,
        },
        "column_semantics": {
            "role": "the META-SPLIT2 development role assigned to this source MAT",
            "prior_use_role": "this project's PRIOR use of the file, not the archive's "
                              "S-code segment semantics: the 24 legacy files carry their "
                              "measured legacy role, the 80 new files carry "
                              "not_previously_used_in_project",
            "archive_segment_role": "the role implied by the DroneRFa file-name S-code "
                                    "(S0000/S0001/S0010/S0011/S01xx). For the 24 legacy "
                                    "files this coincides with prior_use_role; for the 80 "
                                    "new files it is an upstream naming convention only "
                                    "and is NOT evidence of prior use. Kept so nothing "
                                    "from the frozen inputs is silently dropped.",
            "former_partition": "G manifest value; the 80 new files are blank there and are "
                                "normalised to 'NR' exactly as "
                                "work/meta_split1/source_table_104.csv does",
            "file_time_proxy": "MATLAB 'Created on' header time from "
                               "work/meta_split1/source_table_104.csv; a FILE-time proxy, "
                               "never a proven radio acquisition time and never a session id",
            "campaign_proxy": "per-class file-time cluster key C<YYYYMMDD>_<class_id>; "
                              "each class sits in exactly one such cluster",
            "night_proxy": "N<YYYYMMDD> derived as file time minus 6 h; two nights exist "
                           "across the 13 classes (N20221127, N20230227)",
            "mat_sha256": "SHA256 recomputed by SELECT12-G3; NOT recomputed here",
            "grouping_unit": "source MAT file (minimum_group_key == 'file:<name>')",
        },
        "rule": {
            "new_classes": list(NEW_CLASSES),
            "legacy_classes": list(LEGACY_CLASSES),
            "new_class_order": "MAT file-header time ascending, ties by file name ascending",
            "new_class_assignment": "rank 1-5 train, rank 6 val, rank 7-8 development_test",
            "legacy_rule": "development_original (4 files) -> train; evaluated_holdout "
                           "(4 files) sorted by time ascending -> rank 1-2 val, rank 3-4 "
                           "development_test_exposed",
            "grouping_unit": "one source MAT file; a source MAT is never split across roles",
            "tie_break_exercised": tie_break_exercised,
            "tie_break_note": ("no two files of the same class share a MAT file-header "
                               "second, so the tie-break never fires; it is implemented "
                               "defensively only"
                               if not tie_break_exercised else
                               "tie-break fired on: %s" % time_collisions),
            "rule_is_frozen": True,
            "frozen_by": "Codex, work/deepseek-tasks/20260926-meta-split2-dev-files.md",
            "may_not_be_changed_by_model_accuracy": True,
        },
        "counts": {
            "total": len(records),
            "train": counts[ROLE_TRAIN],
            "val": counts[ROLE_VAL],
            "development_test": counts[ROLE_TEST_NEW],
            "development_test_exposed": counts[ROLE_TEST_LEGACY],
            "test_total": counts[ROLE_TEST_NEW] + counts[ROLE_TEST_LEGACY],
            "expected": {
                "total": EXPECTED_TOTAL, "train": EXPECTED_TRAIN, "val": EXPECTED_VAL,
                "development_test": EXPECTED_TEST_NEW,
                "development_test_exposed": EXPECTED_TEST_LEGACY,
                "test_total": EXPECTED_TEST,
            },
        },
        "per_class": per_class,
        "metric_boundaries": {
            "thirteen_class_overall": "development evaluation only; must not be reported "
                                      "as field accuracy, cross-session or open-set performance",
            "new10_file_level_test": "may be reported as a separate main indicator, but it "
                                     "is a within-same-file-time-cluster time-blocked file "
                                     "holdout, not an independent acquisition session",
            "legacy3_test": "descriptive, already-exposed control only; excluded from the "
                            "'not used in prior selection' main indicator; the former "
                            "evaluated_holdout must never be renamed fresh final test",
            "forbidden_claims": [
                "fresh final test", "independent acquisition session", "cross-session",
                "open-set generalisation", "field/deployment accuracy",
            ],
        },
        "files": [{k: v for k, v in r.items() if not k.startswith("_") and k != "sha_record_origin"}
                  for r in records],
    }

    json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    print("candidate_rows=%d" % len(records))
    print("counts=%s" % json.dumps(counts, sort_keys=True))
    print("csv=%s" % csv_path)
    print("csv_sha256=%s" % sha256_file(csv_path))
    print("json=%s" % json_path)
    print("json_sha256=%s" % sha256_file(json_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
