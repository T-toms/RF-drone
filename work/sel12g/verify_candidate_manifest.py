#!/usr/bin/env python3
"""SELECT12-G3 step 4: independent re-verification of the candidate manifest file.

``build_g_manifest_candidate.py`` checks its own in-memory rows.  This script starts again
from the bytes on disk -- the candidate CSV, the legacy G manifest, the frozen target table
and the SHA256 report -- and re-derives every published claim with separate code:

  * header order and row count;
  * 13 ``class_id`` values x 8 files, unique file names;
  * no ``F:`` anywhere in the file text, every ``unified_path`` an existing file whose real
    size equals the declared ``size_bytes``;
  * the 24 legacy rows reproduce the legacy manifest on the seven legacy columns, with only
    ``unified_path`` repointed at the real G file;
  * the 80 new rows are ``former_partition`` empty, ``unassigned``, ``origin=selected_80``;
  * no forbidden ``evaluation_role`` value;
  * every row carries a 64-hex SHA256 that agrees with the hash report -- and, for a small
    sample of files, with a freshly computed digest of the file itself.

Read-only with respect to ``G:``.

Run: .venv\\Scripts\\python.exe work\\sel12g\\verify_candidate_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK = PROJECT_ROOT / "work" / "sel12g"
CONFIG = WORK / "sel12g-config-20260924.json"
CANDIDATE = WORK / "dataset_manifest_candidate_104_20260925.csv"
SHA256_REPORT = WORK / "sel12g-mat-sha256-20260925.json"
LEGACY_MANIFEST = Path("G:/DroneRFa数据集/dataset_manifest.csv")

EXPECTED_COLUMNS = [
    "file_name", "class_id", "class_name", "former_partition", "unified_path",
    "size_bytes", "class_index", "split_status", "evaluation_role", "origin",
]
LEGACY_COLUMNS = EXPECTED_COLUMNS[:7]
FORBIDDEN_ROLES = {"final_test", "heldout", "holdout", "test"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
# One legacy + two new files: enough to tie the hash column back to real bytes at ~3 GiB.
RECHECK_SAMPLE = ["T0000_D00_S0000.mat", "T0001_D00_S0000.mat", "T1100_S0111.mat"]


def sha256_file(path: Path, chunk: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--out", type=Path,
                        default=WORK / "candidate-manifest-verification-20260925.json")
    parser.add_argument("--no-rehash", action="store_true")
    args = parser.parse_args()

    checks: list[dict] = []

    def check(name: str, ok: bool, detail) -> bool:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    raw = args.candidate.read_bytes()
    text = raw.decode("utf-8")
    reader = csv.DictReader(text.splitlines())
    rows = [dict(row) for row in reader]
    header = list(reader.fieldnames or [])

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    targets = {item["filename"]: item for item in config["files"]}
    hashes = {item["filename"]: item for item in
              json.loads(SHA256_REPORT.read_text(encoding="utf-8"))["files"]}
    legacy_rows = [dict(row) for row in csv.DictReader(
        LEGACY_MANIFEST.read_text(encoding="utf-8-sig").splitlines())]

    check("header_matches_contract", header == EXPECTED_COLUMNS, header)
    check("row_count_is_104", len(rows) == 104, {"rows": len(rows)})
    check("no_f_drive_reference_in_file", "F:" not in text and "f:" not in text,
          {"f_occurrences": text.count("F:") + text.count("f:")})
    check("runtime_newline_is_crlf", raw.count(b"\r\n") == len(rows) + 1,
          {"crlf": raw.count(b"\r\n"), "lines": len(rows) + 1})
    check("utf8_no_bom", not raw.startswith(b"\xef\xbb\xbf"), {"first_bytes": list(raw[:3])})

    names = [row["file_name"] for row in rows]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    check("file_names_unique", not duplicates, {"duplicates": duplicates})
    check("names_match_frozen_targets", set(names) == set(targets),
          {"unknown": sorted(set(names) - set(targets)),
           "missing": sorted(set(targets) - set(names))})

    per_class = defaultdict(list)
    for row in rows:
        per_class[row["class_id"]].append(row["file_name"])
    check("thirteen_classes_eight_files_each",
          len(per_class) == 13 and all(len(value) == 8 for value in per_class.values()),
          {key: len(value) for key, value in sorted(per_class.items())})

    path_problems = []
    for row in rows:
        path = Path(row["unified_path"])
        target = targets[row["file_name"]]
        if not path.is_file():
            path_problems.append(f"{row['file_name']}: 文件不存在 {path}")
            continue
        if path.stat().st_size != int(row["size_bytes"]):
            path_problems.append(f"{row['file_name']}: 声明 {row['size_bytes']} "
                                 f"实际 {path.stat().st_size}")
        if path.stat().st_size != target["expected_mat_bytes"]:
            path_problems.append(f"{row['file_name']}: 与冻结目标表字节数不一致")
        if row["class_id"] != target["target_id"] or row["class_name"] != target["class_name"] \
                or row["class_index"] != str(target["class_index"]):
            path_problems.append(f"{row['file_name']}: class 字段与冻结目标表不一致")
    check("sizes_paths_and_classes_verified", not path_problems, path_problems)

    rows_by_name = {row["file_name"]: row for row in rows}
    legacy_problems = []
    for legacy in legacy_rows:
        row = rows_by_name.get(legacy["file_name"])
        if row is None:
            legacy_problems.append(f"{legacy['file_name']}: 候选清单缺失该旧行")
            continue
        for column in LEGACY_COLUMNS:
            if column == "unified_path":
                continue
            if row[column] != legacy[column]:
                legacy_problems.append(
                    f"{legacy['file_name']}.{column}: 候选={row[column]!r} "
                    f"旧清单={legacy[column]!r}")
        if legacy["former_partition"] == "development_original" \
                and row["evaluation_role"] != "legacy_development":
            legacy_problems.append(f"{legacy['file_name']}: 角色应为 legacy_development")
        if legacy["former_partition"] == "evaluated_holdout" \
                and row["evaluation_role"] != "previously_evaluated":
            legacy_problems.append(f"{legacy['file_name']}: 角色应为 previously_evaluated")
    check("legacy_24_rows_preserved", not legacy_problems, legacy_problems)

    new_rows = [row for row in rows if row["file_name"] not in
                {item["file_name"] for item in legacy_rows}]
    new_problems = [
        f"{row['file_name']}: former_partition={row['former_partition']!r} "
        f"split_status={row['split_status']!r} role={row['evaluation_role']!r} "
        f"origin={row['origin']!r}"
        for row in new_rows
        if row["origin"] != "selected_80" or row["former_partition"] != ""
        or row["split_status"] != "unassigned" or row["evaluation_role"] != "unassigned"
    ]
    check("new_80_rows_unassigned", len(new_rows) == 80 and not new_problems,
          {"new_rows": len(new_rows), "problems": new_problems})

    roles = Counter(row["evaluation_role"] for row in rows)
    check("no_forbidden_role", not (set(roles) & FORBIDDEN_ROLES), dict(roles))
    check("roles_are_the_expected_three",
          set(roles) == {"legacy_development", "previously_evaluated", "unassigned"},
          dict(roles))

    hash_problems = [row["file_name"] for row in rows
                     if row["file_name"] not in hashes
                     or not HEX64.match(hashes[row["file_name"]]["sha256"])]
    check("hash_evidence_present_for_all_rows", not hash_problems, hash_problems)

    rehash_results = []
    if not args.no_rehash:
        for name in RECHECK_SAMPLE:
            path = Path(rows_by_name[name]["unified_path"])
            recomputed = sha256_file(path)
            rehash_results.append({
                "file_name": name,
                "recomputed_sha256": recomputed,
                "reported_sha256": hashes[name]["sha256"],
                "matches": recomputed == hashes[name]["sha256"],
            })
        check("spot_rehash_matches_report",
              all(item["matches"] for item in rehash_results), rehash_results)

    failed = [item["check"] for item in checks if not item["ok"]]
    report = {
        "schema": "dronerfa-select12g-candidate-verification/v1",
        "task": "SELECT12-G3",
        "candidate_csv": str(args.candidate),
        "candidate_sha256": hashlib.sha256(raw).hexdigest(),
        "checked_files": len(rows),
        "classes": len(per_class),
        "checks": checks,
        "spot_rehash": rehash_results,
        "failed_checks": failed,
        "ok": not failed,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "failed_checks": failed,
                      "candidate_sha256": report["candidate_sha256"],
                      "spot_rehash": [{k: v for k, v in item.items()
                                       if k != "recomputed_sha256"}
                                      for item in rehash_results]},
                     ensure_ascii=False, indent=2))
    print(f"report -> {args.out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
