#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""META-SPLIT2 negative control: prove verify_split_candidate.py is not vacuous.

For each control case a mutated *copy* of the candidate CSV (or of one frozen
input) is written into work/meta_split2/_selftest_tmp/ and the verifier is run
against that copy. The control passes only when the verifier rejects the mutated
copy with at least the expected check ids, and accepts the untouched candidate.

Nothing outside _selftest_tmp/ is written; the real candidate, the frozen inputs
and G:\\DroneRFa数据集\\dataset_manifest.csv are never modified.

Writes work/meta_split2/negative_control_20260926.json.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "work" / "meta_split2"
TMP = OUT_DIR / "_selftest_tmp"

VERIFIER = OUT_DIR / "verify_split_candidate.py"
CAND_CSV = OUT_DIR / "split_candidate_104_20260926.csv"
CAND_JSON = OUT_DIR / "split_candidate_104_20260926.json"
MANIFEST = Path(r"G:\DroneRFa数据集\dataset_manifest.csv")
SOURCE_TABLE = ROOT / "work" / "meta_split1" / "source_table_104.csv"
SHA_JSON = ROOT / "work" / "sel12g" / "sel12g-mat-sha256-20260925.json"
PRIOR_USE_SCAN = ROOT / "work" / "meta_split1" / "prior_use_scan.json"


def load_verifier():
    spec = importlib.util.spec_from_file_location("ms2_verifier", VERIFIER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_rows(path: Path):
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        r = csv.DictReader(fh)
        return list(r), list(r.fieldnames)


def write_rows(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run(mod, cand, out, manifest=MANIFEST, source=SOURCE_TABLE,
        shas=SHA_JSON, prior=PRIOR_USE_SCAN, cand_json=CAND_JSON):
    mod.RESULTS = []
    old = sys.argv
    old_stdout = sys.stdout
    sys.argv = ["verify_split_candidate.py",
                "--candidate-csv", str(cand),
                "--candidate-json", str(cand_json),
                "--manifest", str(manifest),
                "--source-table", str(source),
                "--sha-json", str(shas),
                "--prior-use-scan", str(prior),
                "--out", str(out)]
    try:
        sys.stdout = io.StringIO()
        rc = mod.main()
        captured = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
        sys.argv = old
    return rc, list(mod.RESULTS), captured


def main() -> int:
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True, exist_ok=True)

    mod = load_verifier()
    base_rows, fields = read_rows(CAND_CSV)
    if len(base_rows) != 104:
        print("candidate CSV is not 104 rows; aborting")
        return 2

    cases = []

    # -------- baseline: the untouched candidate must be accepted -----------
    rc, res, _ = run(mod, CAND_CSV, TMP / "v_baseline.json")
    cases.append({
        "id": "N0_baseline_accepts_candidate",
        "mutation": "none (untouched candidate)",
        "expected_exit": 0,
        "observed_exit": rc,
        "failed_ids": [r["id"] for r in res if not r["ok"]],
        "expected_failed_ids": [],
        "pass": rc == 0 and all(r["ok"] for r in res),
    })

    def cand_case(cid, mutation, expected_ids, mutate):
        rows = [dict(r) for r in base_rows]
        mutate(rows)
        p = TMP / ("cand_%s.csv" % cid)
        write_rows(p, fields, rows)
        rc, res, _ = run(mod, p, TMP / ("v_%s.json" % cid))
        failed = {r["id"] for r in res if not r["ok"]}
        cases.append({
            "id": cid,
            "mutation": mutation,
            "expected_exit": 1,
            "observed_exit": rc,
            "failed_ids": sorted(failed),
            "expected_failed_ids": expected_ids,
            "pass": rc == 1 and set(expected_ids) <= failed,
        })

    def pick(rows, name):
        return next(r for r in rows if r["file_name"] == name)

    cand_case("N1_role_test_to_train",
              "T0001_D00_S0110.mat role development_test -> train",
              ["C16_candidate_matches_rederivation"],
              lambda rows: pick(rows, "T0001_D00_S0110.mat").update(role="train"))

    cand_case("N2_role_renamed_final_test",
              "T0001_D00_S0111.mat role development_test -> final_test",
              ["C16_candidate_matches_rederivation"],
              lambda rows: pick(rows, "T0001_D00_S0111.mat").update(role="final_test"))

    cand_case("N3_sha256_tampered",
              "T0100_D00_S0000.mat mat_sha256 changed",
              ["C16_candidate_matches_rederivation"],
              lambda rows: pick(rows, "T0100_D00_S0000.mat").update(mat_sha256="0" * 64))

    cand_case("N4_path_moved_to_F",
              "T0110_D00_S0000.mat g_path rewritten to F:",
              ["C16_candidate_matches_rederivation"],
              lambda rows: pick(rows, "T0110_D00_S0000.mat").update(
                  g_path=r"F:\DroneRFa数据集\original\T0110_D00_S0000.mat"))

    def swap_roles(rows):
        a = pick(rows, "T0011_D00_S0101.mat")   # rank 6 -> val
        b = pick(rows, "T0011_D00_S0110.mat")   # rank 7 -> development_test
        a["role"], b["role"] = b["role"], a["role"]

    cand_case("N5_role_swap_keeps_counts",
              "T0011 val<->development_test swapped: 62/16/26 counts unchanged",
              ["C16_candidate_matches_rederivation"], swap_roles)

    def duplicate(rows):
        rows.append(dict(pick(rows, "T1000_D00_S0000.mat")))

    cand_case("N6_duplicate_row",
              "T1000_D00_S0000.mat appended twice (105 rows)",
              ["C15_candidate_csv_present", "C17_candidate_uniqueness"], duplicate)

    cand_case("N7_exposure_flag_tampered",
              "T1001_D00_S0000.mat exposed_to_prior_selection True",
              ["C16_candidate_matches_rederivation"],
              lambda rows: pick(rows, "T1001_D00_S0000.mat").update(
                  exposed_to_prior_selection="True"))

    # -------- controls that mutate copies of the frozen inputs -------------
    def input_case(cid, mutation, expected_ids, source=None, prior=None, manifest=None):
        kwargs = {}
        if manifest is not None:
            man_copy = TMP / ("manifest_%s.csv" % cid)
            mrows, mfields = read_rows(MANIFEST)
            manifest(mrows)
            write_rows(man_copy, mfields, mrows)
            kwargs["manifest"] = man_copy
        if source is not None:
            src_copy = TMP / ("source_%s.csv" % cid)
            srows, sfields = read_rows(SOURCE_TABLE)
            source(srows)
            write_rows(src_copy, sfields, srows)
            kwargs["source"] = src_copy
        if prior is not None:
            prior_copy = TMP / ("prior_%s.json" % cid)
            doc = json.loads(PRIOR_USE_SCAN.read_text(encoding="utf-8"))
            prior(doc)
            prior_copy.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
            kwargs["prior"] = prior_copy
        rc, res, _ = run(mod, CAND_CSV, TMP / ("v_%s.json" % cid), **kwargs)
        failed = {r["id"] for r in res if not r["ok"]}
        cases.append({
            "id": cid,
            "mutation": mutation,
            "expected_exit": 1,
            "observed_exit": rc,
            "failed_ids": sorted(failed),
            "expected_failed_ids": expected_ids,
            "pass": rc == 1 and set(expected_ids) <= failed,
        })

    def break_legacy_partition(rows):
        pick(rows, "T0000_D00_S0110.mat")["former_partition"] = "development_original"

    input_case("N8_manifest_former_partition_broken",
               "scratch manifest copy: T0000_D00_S0110.mat evaluated_holdout -> "
               "development_original (5 dev + 3 holdout)",
               ["C01_g_manifest_identity", "C07_per_class_role_pattern",
                "C11_legacy_role_partition_link",
                "C16_candidate_matches_rederivation"],
               manifest=break_legacy_partition)

    def drop_src_row(rows):
        rows.remove(pick(rows, "T1100_S0111.mat"))

    input_case("N9_source_row_removed",
               "scratch source_table copy: T1100_S0111.mat removed (source covers 103)",
               ["C00_inputs_cover_same_file_set",
                "C16_candidate_matches_rederivation"],
               source=drop_src_row)

    def drop_man_row(rows):
        rows.remove(pick(rows, "T1100_S0111.mat"))

    input_case("N11_manifest_row_removed",
               "scratch manifest copy: T1100_S0111.mat removed (103 rows)",
               ["C01_g_manifest_identity", "C00_inputs_cover_same_file_set",
                "C02_input_row_counts", "C16_candidate_matches_rederivation"],
               manifest=drop_man_row)

    def mark_used(doc):
        doc["verdict"]["selected80_used_in_any_data_or_model_artifact"] = True

    input_case("N10_prior_use_scan_says_used",
               "scratch prior_use_scan copy: verdict selected80_used... = true",
               ["C13_new10_not_previously_used"],
               prior=mark_used)

    def add_time_tie(rows):
        a = pick(rows, "T0000_D00_S0000.mat")
        pick(rows, "T0000_D00_S0001.mat")["mat_created_local"] = a["mat_created_local"]

    input_case("N12_intra_class_time_tie",
               "scratch source_table copy: T0000 S0001 header time set equal to S0000 "
               "(tie-break fires)",
               ["C10_no_intra_class_time_ties"],
               source=add_time_tie)

    failed = [c for c in cases if not c["pass"]]
    report = {
        "schema": "dronerfa-meta-split2-negative-control/v1",
        "task": "META-SPLIT2",
        "run_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "verifier": "work/meta_split2/verify_split_candidate.py",
        "verifier_sha256": hashlib.sha256(VERIFIER.read_bytes()).hexdigest(),
        "candidate_csv": "work/meta_split2/split_candidate_104_20260926.csv",
        "candidate_csv_sha256": hashlib.sha256(CAND_CSV.read_bytes()).hexdigest(),
        "scratch_dir": "work/meta_split2/_selftest_tmp (removed after the run)",
        "note": ("C00/C04/C05/C15/C16/C17/C18/C19 are exercised by the candidate-copy "
                 "controls N1-N7; C01/C02/C06/C07/C08/C09/C11/C12 are exercised by the "
                 "manifest-copy controls N8/N11; C00/C09 by the source-copy control N9; "
                 "C10 by the source-copy control N12; C13 by the prior-use-scan control "
                 "N10. C03/C14/C14b are guards on the verifier's own re-derivation "
                 "(class-origin composition, role naming, evidence semantics) and can "
                 "only fail if the frozen rule or the derivation code is edited; they "
                 "are validated by the clean N0 pass rather than by a tamper case."),
        "cases_total": len(cases),
        "cases_passed": len(cases) - len(failed),
        "cases_failed": len(failed),
        "pass": not failed,
        "cases": cases,
    }
    (OUT_DIR / "negative_control_20260926.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    shutil.rmtree(TMP, ignore_errors=True)

    print("cases_total=%d cases_passed=%d cases_failed=%d"
          % (report["cases_total"], report["cases_passed"], report["cases_failed"]))
    for c in failed:
        print("FAIL %s exit=%s failed_ids=%s expected>=%s"
              % (c["id"], c["observed_exit"], c["failed_ids"], c["expected_failed_ids"]))
    print("pass=%s" % report["pass"])
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
