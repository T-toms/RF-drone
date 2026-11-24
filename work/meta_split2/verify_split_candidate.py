#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""META-SPLIT2 independent verifier.

Re-derives the whole 104-row source-file split from the three frozen inputs on its
own (it does not trust the builder's JSON) and then checks the candidate CSV/JSON
against that re-derivation plus a set of structural, prior-use and naming rules.

Read-only. No MAT payload is read and no 107 GB hash is recomputed; the only file
hashing done here is on the small text inputs and on the candidate artifacts.
Writes work/meta_split2/verification_meta_split2.json and exits non-zero on any
failed check.

Usage:
    python verify_split_candidate.py [--candidate-csv PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

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
NEW_CLASSES = ("T0001", "T0011", "T0100", "T0110", "T0111",
               "T1000", "T1001", "T1010", "T1011", "T1100")

ROLE_TRAIN = "train"
ROLE_VAL = "val"
ROLE_TEST_NEW = "development_test"
ROLE_TEST_LEGACY = "development_test_exposed"

REQUIRED_COLUMNS = [
    "file_name", "class_id", "class_name", "role", "prior_use_role",
    "former_partition", "origin", "file_time_proxy", "campaign_proxy",
    "mat_sha256", "g_path",
]

FORBIDDEN_ROLE_TOKENS = ("final_test", "fresh", "cross_session", "cross-session",
                         "independent_test", "test_final")

RESULTS = []


def check(cid: str, ok: bool, detail) -> bool:
    RESULTS.append({"id": cid, "ok": bool(ok), "detail": detail})
    return bool(ok)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path):
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def derive(manifest_path: Path = MANIFEST, source_path: Path = SOURCE_TABLE,
           sha_path: Path = SHA_JSON):
    """Independently re-derive the candidate split from the frozen inputs.

    The paths are parameters so the negative-control self-test can point the
    verifier at mutated *copies* in scratch space; the real frozen inputs are
    never written.
    """
    manifest = load_csv(manifest_path)
    source = {r["file_name"]: r for r in load_csv(source_path)}
    shas = {r["filename"]: r for r in
            json.loads(sha_path.read_text(encoding="utf-8"))["files"]}

    placeholder = {
        "mat_created_local": "", "acquisition_campaign_key": "",
        "acquisition_night_key": "", "prior_use_role": "",
        "prior_use_evidence": "", "s_code": "", "sha256": "",
        "exposed_to_prior_selection": "False",
    }
    rows = []
    for m in manifest:
        name = m["file_name"]
        s = source.get(name, placeholder)
        h = shas.get(name, {"bytes": m["size_bytes"], "origin": "",
                            "path": m["unified_path"]})
        rows.append({
            "file_name": m["file_name"],
            "class_id": m["class_id"],
            "class_name": m["class_name"],
            "class_index": int(m["class_index"]),
            "origin": m["origin"],
            "former_partition": m["former_partition"] or "NR",
            "g_path": m["unified_path"],
            "size_bytes": int(m["size_bytes"]),
            "mat_sha256": s["sha256"],
            "file_time_proxy": s["mat_created_local"],
            "campaign_proxy": s["acquisition_campaign_key"],
            "night_proxy": s["acquisition_night_key"],
            "archive_segment_role": s["prior_use_role"],
            "archive_segment_evidence": s["prior_use_evidence"],
            "archive_s_code": s["s_code"],
            "exposed": s["exposed_to_prior_selection"] == "True",
            "sha_bytes": int(h["bytes"]),
            "sha_origin": h["origin"],
            "sha_path": h["path"],
        })

    by_class = {}
    for r in rows:
        by_class.setdefault(r["class_id"], []).append(r)

    for cid, items in by_class.items():
        items.sort(key=lambda x: (x["file_time_proxy"], x["file_name"]))
        for i, x in enumerate(items, 1):
            x["rank"] = i
        if cid in NEW_CLASSES:
            for x in items:
                x["role"] = (ROLE_TRAIN if x["rank"] <= 5 else
                             ROLE_VAL if x["rank"] == 6 else ROLE_TEST_NEW)
        else:
            dev = [x for x in items if x["former_partition"] == "development_original"]
            hold = sorted([x for x in items if x["former_partition"] == "evaluated_holdout"],
                          key=lambda x: (x["file_time_proxy"], x["file_name"]))
            rest = [x for x in items if x not in dev and x not in hold]
            for x in dev + rest:
                x["role"] = ROLE_TRAIN
            for i, x in enumerate(hold, 1):
                x["role"] = ROLE_VAL if i <= 2 else ROLE_TEST_LEGACY
    for r in rows:
        r["prior_use_role"] = ("not_previously_used_in_project"
                               if r["origin"] == NEW_ORIGIN
                               else r["archive_segment_role"])
    return rows, by_class

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-csv", default=str(OUT_DIR / "split_candidate_104_20260926.csv"))
    ap.add_argument("--candidate-json", default=str(OUT_DIR / "split_candidate_104_20260926.json"))
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--source-table", default=str(SOURCE_TABLE))
    ap.add_argument("--sha-json", default=str(SHA_JSON))
    ap.add_argument("--prior-use-scan", default=str(PRIOR_USE_SCAN))
    ap.add_argument("--out", default=str(OUT_DIR / "verification_meta_split2.json"))
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    source_path = Path(args.source_table)
    sha_path = Path(args.sha_json)
    prior_scan_path = Path(args.prior_use_scan)

    cand_path = Path(args.candidate_csv)
    cand = load_csv(cand_path) if cand_path.is_file() else None
    derived, by_class = derive(manifest_path, source_path, sha_path)

    # ---------------- C00 the frozen inputs must cover the same file set ------
    man_names = {r["file_name"] for r in load_csv(manifest_path)}
    src_names = {r["file_name"] for r in load_csv(source_path)}
    sha_names = {r["filename"] for r in
                 json.loads(sha_path.read_text(encoding="utf-8"))["files"]}
    check("C00_inputs_cover_same_file_set",
          man_names == src_names == sha_names,
          {"missing_from_source": sorted(man_names - src_names),
           "missing_from_sha_json": sorted(man_names - sha_names),
           "extra_in_source": sorted(src_names - man_names)})

    # ---------------- C01 input identity ----------------
    man_sha = sha256_file(manifest_path) if manifest_path.is_file() else None
    check("C01_g_manifest_identity",
          man_sha == MANIFEST_SHA256,
          {"path": str(manifest_path), "sha256": man_sha, "expected": MANIFEST_SHA256})

    check("C02_input_row_counts",
          len(derived) == 104 and len(by_class) == 13
          and all(len(v) == 8 for v in by_class.values()),
          {"files": len(derived), "classes": len(by_class),
           "per_class": sorted({len(v) for v in by_class.values()})})

    # ---------------- C03 class/origin composition ----------------
    comp = {cid: sorted({x["origin"] for x in v}) for cid, v in by_class.items()}
    ok = (set(by_class) == set(NEW_CLASSES) | set(LEGACY_CLASSES)
          and all(comp[c] == [LEGACY_ORIGIN] for c in LEGACY_CLASSES)
          and all(comp[c] == [NEW_ORIGIN] for c in NEW_CLASSES))
    check("C03_class_origin_composition", ok, comp)

    # ---------------- C04 per-file identity across the 3 inputs ----------------
    bad = []
    for r in derived:
        if r["size_bytes"] != r["sha_bytes"]:
            bad.append([r["file_name"], "size"])
        if r["sha_path"] != r["g_path"]:
            bad.append([r["file_name"], "path"])
        if os.path.basename(r["g_path"]) != r["file_name"]:
            bad.append([r["file_name"], "basename"])
        if not r["g_path"].upper().startswith("G:"):
            bad.append([r["file_name"], "drive"])
        if (r["origin"] == LEGACY_ORIGIN) != (r["sha_origin"] == LEGACY_ORIGIN):
            bad.append([r["file_name"], "origin_partition"])
    check("C04_three_input_agreement", not bad, {"mismatches": bad[:20], "n": len(bad)})

    # ---------------- C05 on-disk presence and size (no payload hashing) ------
    miss, wrong = [], []
    for r in derived:
        if not os.path.exists(r["g_path"]):
            miss.append(r["file_name"])
        elif os.path.getsize(r["g_path"]) != r["size_bytes"]:
            wrong.append(r["file_name"])
    check("C05_on_disk_stat_104", not miss and not wrong,
          {"missing": miss, "size_mismatch": wrong, "checked": len(derived),
           "total_bytes": sum(r["size_bytes"] for r in derived)})

    # ---------------- C06 role counts ----------------
    dc = {role: sum(1 for r in derived if r["role"] == role)
          for role in (ROLE_TRAIN, ROLE_VAL, ROLE_TEST_NEW, ROLE_TEST_LEGACY)}
    check("C06_role_counts_62_16_20_6",
          dc[ROLE_TRAIN] == 62 and dc[ROLE_VAL] == 16
          and dc[ROLE_TEST_NEW] == 20 and dc[ROLE_TEST_LEGACY] == 6
          and sum(dc.values()) == 104,
          dc)

    # ---------------- C07 per-class role pattern ----------------
    pat = {}
    for cid, items in by_class.items():
        pat[cid] = sorted(x["role"] for x in items)
    exp_new = sorted([ROLE_TRAIN] * 5 + [ROLE_VAL] + [ROLE_TEST_NEW] * 2)
    exp_old = sorted([ROLE_TRAIN] * 4 + [ROLE_VAL] * 2 + [ROLE_TEST_LEGACY] * 2)
    bad = [c for c in NEW_CLASSES if pat.get(c) != exp_new]
    bad += [c for c in LEGACY_CLASSES if pat.get(c) != exp_old]
    check("C07_per_class_role_pattern", not bad, {"bad": bad})

    # ---------------- C08 zero cross-split / exactly-once ----------------
    chars = {}
    for r in derived:
        chars.setdefault(r["file_name"], set()).add(r["role"])
    dupes = {k: sorted(v) for k, v in chars.items() if len(v) != 1}
    check("C08_each_file_exactly_one_role",
          len(derived) == 104 and len(chars) == 104 and not dupes,
          {"unique_files": len(chars), "multi_role": dupes})

    # ---------------- C09 split is time-blocked inside each class ----------------
    # train block must precede val block must precede test block, and every file
    # must belong to exactly one class-level campaign proxy
    order = {ROLE_TRAIN: 0, ROLE_VAL: 1, ROLE_TEST_NEW: 2, ROLE_TEST_LEGACY: 2}
    viol = []
    for cid, items in by_class.items():
        seq = [order[x["role"]] for x in items]  # items already time-sorted
        if seq != sorted(seq):
            viol.append([cid, "role_block_order"])
        if len({x["campaign_proxy"] for x in items}) != 1:
            viol.append([cid, "campaign_proxy"])
    check("C09_within_class_time_blocked", not viol, {"violations": viol})

    # tie-break never needed: no repeated header second inside a class
    coll = [[cid, t] for cid, items in by_class.items()
            for t in {x["file_time_proxy"] for x in items}
            if sum(1 for y in items if y["file_time_proxy"] == t) > 1]
    check("C10_no_intra_class_time_ties", not coll, {"collisions": coll})

    # ---------------- C11 legacy train only from development_original ----------
    bad = [r["file_name"] for r in derived
           if r["class_id"] in LEGACY_CLASSES and r["role"] == ROLE_TRAIN
           and r["former_partition"] != "development_original"]
    n_ok = sum(1 for r in derived
               if r["class_id"] in LEGACY_CLASSES and r["role"] == ROLE_TRAIN)
    bad += [r["file_name"] for r in derived
            if r["class_id"] in LEGACY_CLASSES and r["role"] in (ROLE_VAL, ROLE_TEST_LEGACY)
            and r["former_partition"] != "evaluated_holdout"]
    check("C11_legacy_role_partition_link", not bad and n_ok == 12,
          {"bad": bad, "legacy_train_files": n_ok})

    # ---------------- C12 legacy val/test are historically exposed -------------
    bad = [r["file_name"] for r in derived
           if r["class_id"] in LEGACY_CLASSES
           and r["role"] in (ROLE_VAL, ROLE_TEST_LEGACY) and not r["exposed"]]
    n = sum(1 for r in derived if r["class_id"] in LEGACY_CLASSES
            and r["role"] in (ROLE_VAL, ROLE_TEST_LEGACY))
    check("C12_legacy_val_test_exposed", not bad and n == 12,
          {"unexposed": bad, "legacy_val_test_files": n})

    # ---------------- C13 new-10 never used in prior selection ----------------
    bad = [r["file_name"] for r in derived
           if r["class_id"] in NEW_CLASSES
           and (r["exposed"] or r["origin"] != NEW_ORIGIN)]
    scan_evidence = None
    if prior_scan_path.is_file():
        scan = json.loads(prior_scan_path.read_text(encoding="utf-8"))
        listed = set(scan["selected80_appearing_in_data_or_model_artifacts"])
        new10_names = {r["file_name"] for r in derived if r["class_id"] in NEW_CLASSES}
        scan_ok = (scan["verdict"]["selected80_used_in_any_data_or_model_artifact"] is False
                   and not (listed & new10_names))
        scan_evidence = {
            "path": str(prior_scan_path),
            "sha256": sha256_file(prior_scan_path),
            "files_scanned": scan["files_scanned"],
            "selected80_used_in_any_data_or_model_artifact":
                scan["verdict"]["selected80_used_in_any_data_or_model_artifact"],
            "intersection_with_new10": sorted(listed & new10_names),
        }
    else:
        scan_ok = False
        scan_evidence = {"error": "prior_use_scan.json not found"}
    check("C13_new10_not_previously_used", not bad and scan_ok,
          {"flagged": bad, "scan": scan_evidence})

    # ---------------- C14 naming gate ----------------
    off = sorted({r["role"] for r in derived
                  if any(t in r["role"].lower() for t in FORBIDDEN_ROLE_TOKENS)})
    check("C14_naming_gate", not off, {"forbidden_roles": off})

    # ---------------- C14b prior-use evidence semantics -----------------------
    if cand is None:
        check("C14b_prior_use_evidence_semantics", False, "candidate CSV missing")
    else:
        bad = []
        for a in cand:
            is_new = a.get("origin") == NEW_ORIGIN
            role = a.get("prior_use_role", "")
            ev = a.get("prior_use_evidence", "")
            aseg = a.get("archive_segment_evidence", "")
            if is_new:
                if role != "not_previously_used_in_project" or "prior_use_scan" not in ev:
                    bad.append([a["file_name"], "new", role, ev[:60]])
            else:
                if role != a.get("archive_segment_role") or ev != aseg:
                    bad.append([a["file_name"], "legacy", role, ev[:60]])
        check("C14b_prior_use_evidence_semantics", not bad,
              {"bad": bad[:10], "n": len(bad)})

    # ---------------- C15 candidate file exists and has the required columns ---
    if cand is None:
        check("C15_candidate_csv_present", False, {"path": str(cand_path)})
    else:
        missing = [c for c in REQUIRED_COLUMNS if c not in (cand[0].keys() if cand else [])]
        check("C15_candidate_csv_present",
              bool(cand) and len(cand) == 104 and not missing,
              {"path": str(cand_path), "rows": len(cand) if cand else 0,
               "missing_columns": missing,
               "sha256": sha256_file(cand_path)})

    # ---------------- C16 candidate matches the independent re-derivation ------
    compare_cols = ("role", "class_id", "class_name", "origin", "former_partition",
                    "prior_use_role", "file_time_proxy", "campaign_proxy",
                    "mat_sha256", "g_path", "size_bytes",
                    "exposed_to_prior_selection", "archive_segment_role")
    if cand is None:
        check("C16_candidate_matches_rederivation", False, "candidate CSV missing")
    else:
        c = {r["file_name"]: r for r in cand}
        exp = {r["file_name"]: r for r in derived}
        diffs = []
        if set(c) != set(exp):
            diffs.append(["file_set", len(c), len(exp)])
        else:
            for name, e in exp.items():
                a = c[name]
                for col in compare_cols:
                    want = ("True" if e["exposed"] else "False") \
                        if col == "exposed_to_prior_selection" else e[col]
                    if str(a.get(col)) != str(want):
                        diffs.append([name, col, a.get(col), want])
        check("C16_candidate_matches_rederivation", not diffs,
              {"diffs": diffs[:20], "n_diffs": len(diffs), "columns": list(compare_cols)})

    # ---------------- C17 no duplicate file rows / no duplicate hashes ---------
    if cand is None:
        check("C17_candidate_uniqueness", False, "candidate CSV missing")
    else:
        names = [r["file_name"] for r in cand]
        hass = [r["mat_sha256"] for r in cand]
        check("C17_candidate_uniqueness",
              len(names) == 104 and len(set(names)) == 104 and len(set(hass)) == 104,
              {"rows": len(names), "unique_names": len(set(names)),
               "unique_hashes": len(set(hass))})

    # ---------------- C18 frozen G manifest untouched, G paths point to G ------
    still = sha256_file(manifest_path) if manifest_path.is_file() else None
    check("C18_g_manifest_untouched", still == MANIFEST_SHA256, {"sha256": still})

    # ---------------- C19 candidate JSON agrees with the candidate CSV ---------
    json_path = Path(args.candidate_json)
    if json_path.is_file():
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        jf = {r["file_name"]: r["role"] for r in doc["files"]}
        cf = {r["file_name"]: r["role"] for r in (cand or [])}
        cnt = doc["counts"]
        ok = (jf == cf and cnt["train"] == 62 and cnt["val"] == 16
              and cnt["development_test"] == 20
              and cnt["development_test_exposed"] == 6 and cnt["total"] == 104)
        check("C19_candidate_json_consistent", ok,
              {"files_match_csv": jf == cf, "counts": cnt})
    else:
        check("C19_candidate_json_consistent", False, "candidate JSON missing")

    failed = [r for r in RESULTS if not r["ok"]]
    report = {
        "schema": "dronerfa-meta-split2-verification/v1",
        "task": "META-SPLIT2",
        "verified_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidate_csv": str(cand_path),
        "candidate_csv_sha256": sha256_file(cand_path) if cand_path.is_file() else None,
        "verifier": "work/meta_split2/verify_split_candidate.py",
        "checks_total": len(RESULTS),
        "checks_passed": len(RESULTS) - len(failed),
        "checks_failed": len(failed),
        "failed_ids": [r["id"] for r in failed],
        "pass": not failed,
        "checks": RESULTS,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print("checks_total=%d checks_passed=%d checks_failed=%d"
          % (report["checks_total"], report["checks_passed"], report["checks_failed"]))
    for r in failed:
        print("FAIL %s: %s" % (r["id"], json.dumps(r["detail"], ensure_ascii=False)[:400]))
    print("pass=%s" % report["pass"])
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
