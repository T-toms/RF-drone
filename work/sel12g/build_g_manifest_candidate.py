#!/usr/bin/env python3
"""SELECT12-G3 step 3: build the **candidate** 104-row G manifest and its diff report.

Nothing on ``G:`` is written.  The frozen G manifest (``G:\\DroneRFa数据集\\dataset_manifest.csv``,
24 rows, still pointing at the retired ``F:`` layout) is only read, and the candidate is
published next to this script:

  * ``work/sel12g/dataset_manifest_candidate_104_20260925.csv``
  * ``work/sel12g/manifest-candidate-diff-20260925.json``
  * ``work/sel12g/manifest-change-report-20260925.md``

Column semantics
----------------
The first seven columns keep the legacy meaning -- ``file_name``, ``class_id``,
``class_name``, ``former_partition``, ``unified_path``, ``size_bytes``, ``class_index`` --
so an existing reader does not have to change.  Three columns are appended:

``split_status``
    ``preserve_former_partition`` for the 24 legacy rows, ``unassigned`` for the 80 new
    ones.  No train/val/test split is frozen here.
``evaluation_role``
    ``legacy_development`` / ``previously_evaluated`` for the 24 legacy rows,
    ``unassigned`` for the new 80.  ``final_test`` and ``heldout`` are forbidden values:
    the 12 legacy holdout files already took part in model selection and the 80 new files
    cannot be called an untouched final test before the META-SPLIT session-group check.
``origin``
    ``baseline_24`` (the pre-existing DroneRFa files) or ``selected_80`` (this round's
    frozen 12-model selection).

Evidence rules enforced here (any failure ends with exit code 1 and ``ok: false``):

  * the candidate has exactly 104 rows, 13 ``class_id`` values with 8 files each;
  * file names are unique, every ``unified_path`` is a real file under the G original
    directory, and no path mentions ``F:``;
  * every row carries the on-disk size, the frozen target size and a SHA256 that matches
    the independent hash report;
  * the 24 legacy rows reproduce the legacy manifest byte-for-byte on the seven legacy
    columns except for the ``unified_path``, which now points at the real G file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK = PROJECT_ROOT / "work" / "sel12g"
CONFIG = WORK / "sel12g-config-20260924.json"
LEGACY_MANIFEST = Path("G:/DroneRFa数据集/dataset_manifest.csv")
AUDIT = WORK / "sel12g-mat-audit-20260925.json"
SHA256_REPORT = WORK / "sel12g-mat-sha256-20260925.json"
STATE = Path("G:/DroneRFa数据集/download_records_12models_20260922/select12g-state.json")

LEGACY_COLUMNS = [
    "file_name",
    "class_id",
    "class_name",
    "former_partition",
    "unified_path",
    "size_bytes",
    "class_index",
]
APPENDED_COLUMNS = ["split_status", "evaluation_role", "origin"]
CANDIDATE_COLUMNS = LEGACY_COLUMNS + APPENDED_COLUMNS

# Legacy ``former_partition`` -> the role carried into the G manifest.  The 12
# ``evaluated_holdout`` files were already used for model/observable selection, so they
# must never be relabelled as an untouched final test.
ROLE_BY_FORMER_PARTITION = {
    "development_original": "legacy_development",
    "evaluated_holdout": "previously_evaluated",
}
FORBIDDEN_ROLES = {"final_test", "heldout", "holdout", "test"}
NEW_ORIGIN = "selected_80"
LEGACY_ORIGIN = "baseline_24"
NEW_SPLIT_STATUS = "unassigned"
LEGACY_SPLIT_STATUS = "preserve_former_partition"
EXPECTED_ROWS = 104
EXPECTED_LEGACY_ROWS = 24
EXPECTED_NEW_ROWS = 80
EXPECTED_CLASSES = 13
EXPECTED_FILES_PER_CLASS = 8


def sha256_file(path: Path, chunk: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_legacy_manifest(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    return [dict(row) for row in csv.DictReader(io.StringIO(text))]


class CheckLog:
    """Collects explicit pass/fail evidence instead of asserting silently."""

    def __init__(self) -> None:
        self.checks: list[dict] = []

    def check(self, name: str, ok: bool, detail) -> bool:
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def failed(self) -> list[dict]:
        return [item for item in self.checks if not item["ok"]]


def build() -> tuple[dict, list[dict]]:
    log = CheckLog()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    targets = {item["filename"]: item for item in config["files"]}
    g_original = Path(config["paths"]["g_original"])
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    hashes = json.loads(SHA256_REPORT.read_text(encoding="utf-8"))
    legacy_rows = read_legacy_manifest(LEGACY_MANIFEST)

    audit_by_name = {item["filename"]: item for item in audit["files"]}
    hash_by_name = {item["filename"]: item for item in hashes["files"]}

    log.check("audit_covers_all_targets",
              audit["counts"]["mat_on_g"] == len(targets) == EXPECTED_ROWS
              and audit["counts"]["target_missing_on_g"] == 0,
              {"audit_counts": audit["counts"], "targets": len(targets)})
    log.check("audit_all_valid", audit["counts"]["invalid"] == 0 and not audit["problems"],
              {"invalid": audit["counts"]["invalid"], "problems": audit["problems"]})
    log.check("hash_report_covers_all_targets",
              hashes["count"] == EXPECTED_ROWS
              and set(hash_by_name) == set(targets)
              and all(len(item["sha256"]) == 64 for item in hashes["files"]),
              {"hash_count": hashes["count"], "targets": len(targets)})

    # ---------------------------------------------------------------- legacy rows
    legacy_issues: list[str] = []
    legacy_by_name = {row["file_name"]: row for row in legacy_rows}
    log.check("legacy_manifest_row_count", len(legacy_rows) == EXPECTED_LEGACY_ROWS,
              {"rows": len(legacy_rows)})
    log.check("legacy_file_names_unique",
              len(legacy_by_name) == len(legacy_rows),
              {"unique": len(legacy_by_name)})
    legacy_preserved: list[dict] = []
    for row in legacy_rows:
        name = row["file_name"]
        target = targets.get(name)
        problem = None
        if target is None:
            problem = "不在冻结的104文件目标表内"
        elif target["origin"] != LEGACY_ORIGIN:
            problem = f"目标表 origin={target['origin']}，不是 {LEGACY_ORIGIN}"
        elif target["former_partition"] != row["former_partition"]:
            problem = (f"former_partition 目标表={target['former_partition']} "
                       f"旧清单={row['former_partition']}")
        elif str(target["expected_mat_bytes"]) != row["size_bytes"]:
            problem = (f"size_bytes 目标表={target['expected_mat_bytes']} "
                       f"旧清单={row['size_bytes']}")
        elif target["target_id"] != row["class_id"]:
            problem = f"class_id 目标表={target['target_id']} 旧清单={row['class_id']}"
        elif target["class_name"] != row["class_name"]:
            problem = f"class_name 目标表={target['class_name']} 旧清单={row['class_name']}"
        elif str(target["class_index"]) != row["class_index"]:
            problem = f"class_index 目标表={target['class_index']} 旧清单={row['class_index']}"
        if problem:
            legacy_issues.append(f"{name}: {problem}")
        role = ROLE_BY_FORMER_PARTITION.get(row["former_partition"])
        if role is None:
            legacy_issues.append(f"{name}: 未知 former_partition={row['former_partition']}")
            role = "unassigned"
        legacy_preserved.append({
            "file_name": name,
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "class_index": row["class_index"],
            "size_bytes": row["size_bytes"],
            "former_partition": row["former_partition"],
            "evaluation_role": role,
            "split_status": LEGACY_SPLIT_STATUS,
            "legacy_unified_path": row["unified_path"],
        })
    log.check("legacy_rows_match_frozen_targets", not legacy_issues, legacy_issues)
    partition_counts = Counter(row["former_partition"] for row in legacy_preserved)
    log.check("legacy_partition_roles_known",
              bool(partition_counts) and set(partition_counts) <= set(ROLE_BY_FORMER_PARTITION),
              dict(partition_counts))

    # ---------------------------------------------------------------- candidate rows
    candidate: list[dict] = []
    issues: list[str] = []
    new_rows: list[dict] = []

    def emit(name: str, *, legacy: dict | None) -> None:
        target = targets[name]
        audit_row = audit_by_name.get(name)
        hash_row = hash_by_name.get(name)
        if audit_row is None or hash_row is None:
            issues.append(f"{name}: 缺少审计或哈希证据")
            return
        actual = audit_row["actual_bytes"]
        expected = target["expected_mat_bytes"]
        if actual != expected or int(hash_row["bytes"]) != expected:
            issues.append(f"{name}: 大小不一致 actual={actual} expected={expected} "
                          f"hash_bytes={hash_row['bytes']}")
        path = g_original / name
        if not path.exists() or path.stat().st_size != actual:
            issues.append(f"{name}: G 盘实际文件缺失或大小变化：{path}")
        row = {
            "file_name": name,
            "class_id": target["target_id"],
            "class_name": target["class_name"],
            "former_partition": legacy["former_partition"] if legacy else "",
            "unified_path": str(path),
            "size_bytes": str(expected),
            "class_index": str(target["class_index"]),
            "split_status": (LEGACY_SPLIT_STATUS if legacy else NEW_SPLIT_STATUS),
            "evaluation_role": (legacy["evaluation_role"] if legacy else "unassigned"),
            "origin": (LEGACY_ORIGIN if legacy else NEW_ORIGIN),
            "_sha256": hash_row["sha256"],
        }
        candidate.append(row)
        if not legacy:
            new_rows.append(row)

    legacy_roles = {row["file_name"]: row for row in legacy_preserved}
    for row in legacy_rows:  # legacy order first: keeps the diff against the old file trivial
        name = row["file_name"]
        if name not in targets:
            # Already recorded in legacy_issues: a row outside the frozen table cannot
            # produce a candidate row, and it must not abort the whole report.
            continue
        emit(name, legacy=legacy_roles[name])
    for name in sorted(name for name, item in targets.items()
                       if item["origin"] != LEGACY_ORIGIN):
        emit(name, legacy=None)

    log.check("candidate_row_count", len(candidate) == EXPECTED_ROWS, {"rows": len(candidate)})
    log.check("new_row_count", len(new_rows) == EXPECTED_NEW_ROWS, {"new_rows": len(new_rows)})
    names = [row["file_name"] for row in candidate]
    log.check("file_names_unique", len(set(names)) == len(names),
              {"duplicates": [name for name, count in Counter(names).items() if count > 1]})

    per_class = defaultdict(list)
    for row in candidate:
        per_class[row["class_id"]].append(row)
    class_counts = {key: len(value) for key, value in sorted(per_class.items())}
    log.check("classes_and_files_per_class",
              len(per_class) == EXPECTED_CLASSES
              and all(count == EXPECTED_FILES_PER_CLASS for count in class_counts.values()),
              {"classes": len(per_class), "counts": class_counts})

    bad_paths = [row["file_name"] for row in candidate
                 if "F:" in row["unified_path"]
                 or not row["unified_path"].lower().startswith(str(g_original).lower())
                 or not row["unified_path"].endswith(row["file_name"])]
    log.check("all_paths_point_to_g_original", not bad_paths, {"bad_paths": bad_paths})

    bad_roles = sorted({row["evaluation_role"] for row in candidate} & FORBIDDEN_ROLES)
    log.check("no_forbidden_evaluation_role", not bad_roles, {"forbidden_roles": bad_roles})
    log.check("new_rows_are_unassigned",
              all(row["former_partition"] == "" and row["split_status"] == NEW_SPLIT_STATUS
                  and row["origin"] == NEW_ORIGIN and row["evaluation_role"] == "unassigned"
                  for row in new_rows),
              {"new_rows": len(new_rows)})

    new_target_problems = [
        f"{name}: former_partition={targets[name]['former_partition']!r} "
        f"split_status={targets[name]['split_status']!r}"
        for name in targets
        if targets[name]["origin"] != LEGACY_ORIGIN
        and (targets[name]["former_partition"] is not None
             or targets[name]["split_status"] != NEW_SPLIT_STATUS)
    ]
    log.check("new_targets_have_no_former_partition", not new_target_problems,
              new_target_problems)

    missing_hashes = [row["file_name"] for row in candidate if len(row["_sha256"]) != 64]
    log.check("hash_evidence_complete", not missing_hashes, {"missing": missing_hashes})
    log.check("no_outstanding_issues", not issues, issues)

    # ---------------------------------------------------------------- extra evidence
    digest_groups = defaultdict(list)
    for row in candidate:
        digest_groups[row["_sha256"]].append(row["file_name"])
    duplicate_groups = sorted((sorted(files) for files in digest_groups.values()
                               if len(files) > 1))

    legacy_state = None
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
        state_files = state.get("files", {})
        new_target_names = [name for name in targets
                            if targets[name]["origin"] != LEGACY_ORIGIN]
        covered = [name for name in new_target_names if name in state_files]
        landed_earlier = len(new_target_names) - len(covered)
        legacy_state = {
            "path": str(STATE),
            "state_entries": len(state_files),
            "state_entries_completed": sum(
                1 for item in state_files.values() if item.get("status") == "completed"),
            "new_targets": len(new_target_names),
            "new_targets_covered_by_this_state_file": len(covered),
            "covered_targets_all_completed": all(
                state_files[name].get("status") == "completed" for name in covered),
            "new_targets_landed_before_this_state_round": landed_earlier,
            "frozen_totals_already_landed_on_g": config["totals"].get("already_landed_on_g"),
            "frozen_totals_still_missing_on_g": config["totals"].get("still_missing_on_g"),
            "historical_errors": len(state.get("errors", [])),
        }
        log.check("download_state_matches_frozen_totals",
                  len(covered) == config["totals"].get("still_missing_on_g")
                  and landed_earlier == config["totals"].get("already_landed_on_g")
                  and legacy_state["covered_targets_all_completed"],
                  legacy_state)

    report = {
        "schema": "dronerfa-select12g-manifest-candidate/v1",
        "task": "SELECT12-G3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope_note": (
            "候选清单，只写在项目 work/sel12g/ 下；未修改 G:\\DroneRFa数据集\\dataset_manifest.csv。"
            "不冻结 train/val/test，不重新标定 final test。"
        ),
        "inputs": {
            "frozen_target_table": str(CONFIG),
            "frozen_target_table_files": len(targets),
            "legacy_manifest": str(LEGACY_MANIFEST),
            "legacy_manifest_rows": len(legacy_rows),
            "legacy_manifest_f_paths": sum(1 for row in legacy_rows
                                           if row["unified_path"].startswith("F:")),
            "audit_report": str(AUDIT),
            "audit_checked_at": audit["checked_at"],
            "sha256_report": str(SHA256_REPORT),
            "sha256_computed_at": hashes["computed_at"],
            "download_state": legacy_state,
        },
        "outputs": {
            "candidate_csv": str(WORK / "dataset_manifest_candidate_104_20260925.csv"),
            "candidate_columns": CANDIDATE_COLUMNS,
        },
        "counts": {
            "candidate_rows": len(candidate),
            "legacy_rows_preserved": len(legacy_preserved),
            "new_rows": len(new_rows),
            "classes": len(per_class),
            "bytes_total": sum(int(row["size_bytes"]) for row in candidate),
            "bytes_new_rows": sum(int(row["size_bytes"]) for row in new_rows),
        },
        "legacy_rows_preserved": legacy_preserved,
        "new_rows": [
            {
                "file_name": row["file_name"],
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "class_index": row["class_index"],
                "size_bytes": int(row["size_bytes"]),
                "sha256": row["_sha256"],
                "former_partition": None,
                "split_status": row["split_status"],
                "evaluation_role": row["evaluation_role"],
                "origin": row["origin"],
                "unified_path": row["unified_path"],
            }
            for row in new_rows
        ],
        "per_class": {
            key: {
                "class_name": value[0]["class_name"],
                "files": len(value),
                "bytes": sum(int(row["size_bytes"]) for row in value),
                "legacy": sum(1 for row in value if row["origin"] == LEGACY_ORIGIN),
                "new": sum(1 for row in value if row["origin"] == NEW_ORIGIN),
            }
            for key, value in sorted(per_class.items())
        },
        "duplicate_content_groups": duplicate_groups,
        "anomalies": issues + legacy_issues,
        "checks": log.checks,
        "ok": not log.failed,
    }
    return report, candidate


def write_candidate_csv(path: Path, rows: list[dict]) -> str:
    """Write the candidate as UTF-8 / CRLF / all-quoted, like the legacy manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS,
                                quoting=csv.QUOTE_ALL, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return sha256_file(path)


def write_markdown(path: Path, report: dict) -> None:
    counts = report["counts"]
    legacy = report["legacy_rows_preserved"]
    lines = [
        "# SELECT12-G3 G 盘候选清单变更报告",
        "",
        f"生成时间（UTC）：{report['generated_at']}",
        f"候选 CSV：`{report['outputs']['candidate_csv']}`（{counts['candidate_rows']} 行，"
        f"{counts['bytes_total']} 字节）",
        "",
        "本文件是候选方案，**没有**改写 `G:\\DroneRFa数据集\\dataset_manifest.csv`。",
        "",
        "## 旧 24 行如何保留",
        "",
        "- 七个旧列 `file_name,class_id,class_name,former_partition,unified_path,size_bytes,class_index` "
        "逐行保留原 class/大小/former_partition 语义（已核对与冻结目标表一致）。",
        "- `unified_path` 由旧 F 盘路径改写为实际 G 盘文件路径；旧路径原值保留在差异报告的 "
        "`legacy_unified_path` 字段。",
        "- 旧 `development_original` 12 行 → `evaluation_role=legacy_development`；"
        "旧 `evaluated_holdout` 12 行 → `evaluation_role=previously_evaluated`；两者 "
        "`split_status=preserve_former_partition`。",
        "- 这 12 个已评估留出文件已经参与过模型/n_fft 选型，**不能**重新标成全新 final test。",
        "",
        "## 新增 80 行",
        "",
        f"- `former_partition` 为空、`split_status={NEW_SPLIT_STATUS}`、"
        f"`evaluation_role=unassigned`、`origin={NEW_ORIGIN}`。",
        f"- 覆盖 10 个新机型 × 8 个 S 段；加上旧 3 类，共 {counts['classes']} 类，每类 8 文件。",
        "- 每行都带实到字节数（与冻结归档索引一致）和独立 SHA256 证据。",
        "",
        "## 每类文件数",
        "",
        "| class_id | class_name | 文件数 | 旧 24 | 新增 80 | 字节 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key, item in report["per_class"].items():
        lines.append(f"| {key} | {item['class_name']} | {item['files']} | {item['legacy']} | "
                     f"{item['new']} | {item['bytes']} |")
    lines += [
        "",
        "## 异常",
        "",
    ]
    anomalies = report["anomalies"]
    if anomalies:
        lines += [f"- {item}" for item in anomalies]
    else:
        lines.append("- 无。")
    groups = report["duplicate_content_groups"]
    lines += ["", "## 内容重复组（SHA256 相同，供后续 META-SPLIT 防泄漏核查）", ""]
    if groups:
        for group in groups:
            lines.append(f"- {', '.join(group)}")
    else:
        lines.append("- 未发现 104 个文件之间的 SHA256 重复。")
    lines += ["", "## 检查项", ""]
    for item in report["checks"]:
        mark = "PASS" if item["ok"] else "FAIL"
        lines.append(f"- [{mark}] {item['check']}: `{json.dumps(item['detail'], ensure_ascii=False)}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path,
                        default=WORK / "dataset_manifest_candidate_104_20260925.csv")
    parser.add_argument("--report", type=Path,
                        default=WORK / "manifest-candidate-diff-20260925.json")
    parser.add_argument("--markdown", type=Path,
                        default=WORK / "manifest-change-report-20260925.md")
    args = parser.parse_args()

    report, rows = build()
    report["outputs"]["candidate_csv"] = str(args.csv)
    report["outputs"]["candidate_sha256"] = write_candidate_csv(args.csv, rows)
    report["outputs"]["candidate_bytes"] = args.csv.stat().st_size
    report["inputs"]["legacy_manifest_sha256"] = sha256_file(LEGACY_MANIFEST)
    write_markdown(args.markdown, report)
    report["outputs"]["diff_report"] = str(args.report)
    report["outputs"]["markdown_report"] = str(args.markdown)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "ok": report["ok"],
        "counts": report["counts"],
        "anomalies": report["anomalies"],
        "failed_checks": [item["check"] for item in report["checks"] if not item["ok"]],
        "candidate_sha256": report["outputs"]["candidate_sha256"],
        "duplicate_content_groups": report["duplicate_content_groups"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
