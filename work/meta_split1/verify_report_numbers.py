# -*- coding: utf-8 -*-
"""META-SPLIT-1 / step 5: re-derive every headline number in
META-SPLIT-1-report.md from the produced artefacts and fail loudly on any
mismatch.  Read-only."""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
fails = []
checks = []


def load(name):
    return json.load(open(os.path.join(HERE, name), encoding="utf-8"))


def check(label, got, want):
    ok = got == want
    checks.append({"check": label, "got": got, "expected": want, "ok": ok})
    if not ok:
        fails.append(label)


with open(r"G:\DroneRFa数据集\dataset_manifest.csv", "r", encoding="utf-8-sig", newline="") as fh:
    man = list(csv.DictReader(fh))

src = load("source_table_104.json")
camp = load("campaign_grouping_104.json")
nd = load("near_duplicate_screen.json")
ch = load("fingerprint_screen_characterisation.json")
rar = load("rar_header_times.json")
arc = load("archive_index_cells_20260925.json")
prior = load("prior_use_scan.json")
prop = load("split_proposal_draft.json")
h5 = load("hdf5_metadata_104.json")
hd = load("mat_headers_104.json")

check("manifest rows", len(man), 104)
check("source table records", len(src["records"]), 104)
check("distinct classes", len({r["class_id"] for r in src["records"]}), 13)
check("files per class", sorted({sum(1 for r in src["records"] if r["class_id"] == c)
                                 for c in {r["class_id"] for r in src["records"]}}), [8])
check("total bytes", sum(int(r["size_bytes"]) for r in man), 107450183727)
check("legacy rows", sum(1 for r in man if r["origin"] == "baseline_24"), 24)
check("new rows", sum(1 for r in man if r["origin"] == "selected_80"), 80)
check("development_original", sum(1 for r in man if r["former_partition"] == "development_original"), 12)
check("evaluated_holdout", sum(1 for r in man if r["former_partition"] == "evaluated_holdout"), 12)

check("sha names == manifest", src["sha_checks"]["names_identical_to_manifest"], True)
check("sha distinct digests", src["sha_checks"]["distinct_digests"], 104)
check("sha duplicate groups", src["sha_checks"]["duplicate_digest_groups"], {})
check("sha size agreement", src["sha_checks"]["size_agreement_with_manifest"], True)

check("mat header parse failures", hd["header_parse_failures"], [])
check("distinct created days", len({r["created_day"] for r in hd["records"]}), 3)
check("hdf5 ok", sum(1 for r in h5["records"] if r["hdf5_read_ok"]), 104)
check("hdf5 problems", h5["problems"], [])
check("hdf5 root attrs total", sum(len(r["root_attrs"]) for r in h5["records"]), 0)

check("campaigns", camp["campaign_count"], 13)
check("nights", camp["night_count"], 2)
check("classes all on one day", camp["classes_all_on_one_day"], True)
check("night1 classes", camp["nights"]["N20221127"]["n_classes"], 10)
check("night1 files", camp["nights"]["N20221127"]["n_files"], 80)
check("night2 classes", camp["nights"]["N20230227"]["n_classes"], 3)
check("night2 files", camp["nights"]["N20230227"]["n_files"], 24)
gaps = [(v["inter_file_gap_seconds_min"], v["inter_file_gap_seconds_max"])
        for v in camp["campaigns"].values()]
check("min intra-class gap s", min(g[0] for g in gaps), 81.0)
check("max intra-class gap s", max(g[1] for g in gaps), 159.0)
check("all campaigns class-unique",
      sorted(v["class_id"] for v in camp["campaigns"].values()) ==
      sorted({r["class_id"] for r in src["records"]}), True)

check("fragments", nd["fragments_compared"], 1664)
check("exact duplicate groups", nd["exact_duplicate_group_count"], 0)
check("float32 duplicate groups", nd["float32_cast_duplicate_group_count"], 0)
check("exact duplicate fragment groups", nd["exact_duplicate_fragment_groups"], {})
# The frozen output folds the (non-discriminative) L1<0.05 flags into the same
# "candidate pair" list; report both numbers explicitly rather than one.
check("candidate pairs from L1 flag (non-discriminative, not evidence)",
      len(nd["candidate_near_duplicate_file_pairs"]), 5217)
check("L1 flagged fragment pairs", len(nd["spectral_flag_l1_pairs"]), 36548)
check("spectral pairs total", ch["spectral_screen_characterisation"]["pairs_total"], 85696)
check("L1<0.05 fraction", ch["spectral_screen_characterisation"]["overall"]["frac_below_preregistered_0.05"],
      0.4265)
check("min L1", ch["spectral_screen_characterisation"]["overall"]["min"], 0.016703)

check("rar members checked", rar["check_summary"]["members_checked"], 14)
check("rar matches within 2s", rar["check_summary"]["match_within_2s"], 14)

check("archive mat entries", arc["archive_index_mat_entries"], 630)
check("archive classes", len(arc["per_class"]), 25)
check("legacy3 unused", arc["legacy3_unused_count"], 88)
check("legacy3 unused bytes", arc["legacy3_unused_unpacked_bytes"], 86961570451)
by_t = {}
for v in arc["legacy3_unused_cells"]:
    by_t.setdefault(v["t"], [0, 0])
    by_t[v["t"]][0] += 1
    by_t[v["t"]][1] += v["unpacked_size"]
check("unused T0000", by_t["0000"], [8, 7745532942])
check("unused T0010", by_t["0010"], [40, 38339965918])
check("unused T0101", by_t["0101"], [40, 40876071591])

check("prior-use files scanned", prior["files_scanned"], 1596)
check("selected80 in data/model artifacts", prior["selected80_appearing_in_data_or_model_artifacts"], [])
check("legacy files in data/model artifacts",
      len(prior["files_appearing_in_data_or_model_artifacts"]), 24)

check("proposal seed", prop["split_seed"], 20260925)
check("level1 answer", prop["level_1_never_used_in_selection"]["answer"], "NO")
check("level2 answer", prop["level_2_campaign_independent_final_test"]["answer"], "NO")
check("level1 missing classes",
      prop["level_1_never_used_in_selection"]["missing"]["classes"], ["T0000", "T0010", "T0101"])
check("level2 missing total",
      prop["level_2_campaign_independent_final_test"]["missing"]["files_minimum_total"], 13)
check("assignments cover 13 classes", len(prop["deterministic_file_level_assignments"]), 13)
for cid, a in prop["deterministic_file_level_assignments"].items():
    s = a["file_level_5_1_2"]
    if len(s["train"]) != 5 or len(s["val"]) != 1 or len(s["test"]) != 2:
        fails.append("assignment sizes %s" % cid)
    if len(set(s["train"] + s["val"] + s["test"])) != 8:
        fails.append("assignment not a partition %s" % cid)

out = {"checks": checks, "failures": fails, "n_checks": len(checks)}
with open(os.path.join(HERE, "report_number_verification.json"), "w", encoding="utf-8") as fh:
    json.dump({"schema": "dronerfa-meta-split-report-number-verification/v1",
               "generated_at": __import__("datetime").datetime.now(
                   __import__("datetime").timezone.utc).isoformat(), **out},
              fh, ensure_ascii=False, indent=1)
print(json.dumps({"n_checks": len(checks), "failures": fails,
                  "failed_detail": [c for c in checks if not c["ok"]]},
                 ensure_ascii=False, indent=1))
sys.exit(1 if fails else 0)
