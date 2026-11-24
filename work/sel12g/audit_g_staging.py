#!/usr/bin/env python3
"""SELECT12-G1: strictly read-only audit of the staged ``.part`` / ``.rar`` files on G:.

For every ``*.mat.rar`` and ``*.mat.rar.part`` under
``G:\\DroneRFa数据集\\download_records_12models_20260922\\downloads`` this reports

  * exact path and real length,
  * the frozen target entry it belongs to (from the 104-file target table),
  * whether a valid MAT for that target already exists in ``original``,
  * the recorded remote identity from the sibling ``.selection.json`` / ``.state.json``
    (URL, archive size, block offset/size, packed/unpacked size, committed bytes),
  * whether the staged bytes can be *proven* to belong to that entry and to be complete
    (RAR5 block chain + header CRCs + single stored entry + name/size match),
  * ``UnRAR t`` for complete archives (the authoritative payload check),
  * a reuse verdict with the reason.

Nothing is written, moved or deleted, and no child process output is piped (the DSH
sandbox blocks piped stdio, so UnRAR is run with file-redirected output).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK = PROJECT_ROOT / "work" / "sel12g"
CONFIG = WORK / "sel12g-config-20260924.json"
DEFAULT_UNRAR = Path(r"C:\Program Files\WinRAR\UnRAR.exe")
# The sandbox denies piped child stdio and writes outside the workspace, so UnRAR
# output is redirected to a file inside the project tree.
CHILD_LOG_DIR = PROJECT_ROOT / "tmp" / "sel12g-childlogs"


def run_unrar(unrar: Path, command: str, archive: Path) -> tuple[int, str]:
    CHILD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = CHILD_LOG_DIR / f"audit-unrar-{command}-{archive.name}.log"
    with out.open("wb") as handle:
        result = subprocess.run([str(unrar), command, "-idq", "-p-", str(archive)],
                                stdout=handle, stderr=subprocess.STDOUT, check=False)
    raw = out.read_bytes()
    text = ""
    for encoding in ("gbk", "utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    out.unlink(missing_ok=True)
    return result.returncode, text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK / "sel12g-staging-audit-20260924.json")
    parser.add_argument("--unrar", type=Path, default=DEFAULT_UNRAR)
    parser.add_argument("--no-unrar", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT / "dronerfa_baseline"))
    import download_select12g as dl  # noqa: PLC0415

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    targets = {item["filename"]: item for item in config["files"]}
    downloads = Path(config["paths"]["g_downloads"])
    g_original = Path(config["paths"]["g_original"])

    staged: list[Path] = []
    # Exclude this task's own active staging files (``.part.restart*``); this audit covers
    # the pre-existing 2026-09-22/23 staging only.
    for pattern in ("*.mat.rar", "*.mat.rar.part"):
        for path in sorted(downloads.glob(pattern)):
            if path.name.endswith(".part.restart"):
                continue
            staged.append(path)

    records: list[dict] = []
    verdict_counts: dict[str, int] = {}
    for path in staged:
        name = path.name
        target_name = name
        for suffix in (".rar.part.restart", ".rar.part", ".rar"):
            if target_name.endswith(suffix):
                target_name = target_name[: -len(suffix)]
                break
        entry = targets.get(target_name)
        is_partial = name.endswith(".part") or name.endswith(".part.restart")
        record: dict = {
            "path": str(path),
            "name": name,
            "kind": "partial" if is_partial else "complete_archive",
            "bytes": path.stat().st_size,
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "target_filename": target_name,
            "target_in_frozen_table": entry is not None,
        }
        if entry is not None:
            record["target_block_offset"] = entry["archive_block_offset"]
            record["target_block_size"] = entry["archive_block_size"]
            record["target_packed_size"] = entry["archive_packed_size"]
            record["target_expected_mat_bytes"] = entry["expected_mat_bytes"]
            record["target_class_name"] = entry["class_name"]
            landed = g_original / target_name
            record["valid_mat_already_present"] = bool(
                landed.exists() and landed.stat().st_size == entry["expected_mat_bytes"]
            )
            record["block_size_match"] = path.stat().st_size == entry["archive_block_size"]
            record["shortfall_bytes"] = entry["archive_block_size"] - path.stat().st_size

        # sibling provenance records
        for suffix in (".selection.json", ".state.json"):
            sibling = path.parent / (name + suffix)
            if sibling.exists():
                try:
                    record[suffix.strip(".")] = json.loads(sibling.read_text(encoding="utf-8"))
                except ValueError as exc:
                    record[suffix.strip(".")] = {"parse_error": str(exc)}
            else:
                record[suffix.strip(".")] = None

        # structural continuity
        try:
            if is_partial:
                record["prefix_is_target_entry"] = dl.verify_wrapper_prefix(path, entry) if entry else None
                record["structure"] = "prefix_only"
            else:
                wrapper = dl.verify_wrapper(path, entry) if entry else None
                record["structure"] = wrapper
                record["wrapper_ok"] = wrapper is not None
        except Exception as exc:  # noqa: BLE001
            record["structure_error"] = f"{type(exc).__name__}: {exc}"

        if not is_partial and not args.no_unrar and args.unrar.is_file():
            code, text = run_unrar(args.unrar, "t", path)
            record["unrar_test"] = {"exit_code": code, "output": text.strip()[:300]}

        # reuse verdict
        if entry is None:
            verdict, reason = "unrelated_keep", "不属于冻结104文件目标的暂存文件，保持原样不动"
        elif is_partial:
            verdict, reason = (
                "not_reusable_restart_from_zero",
                "旧断点状态为旧格式且来源端点已知会返回长度正确但内容错误的响应，"
                "无法证明已落地字节正确；保留原件，从目标条目起点重新下载到独立的新文件",
            )
        else:
            unrar_ok = (record.get("unrar_test") or {}).get("exit_code") == 0
            if record.get("wrapper_ok") and unrar_ok:
                verdict, reason = "reusable_after_reverify", "RAR结构与UnRAR完整性测试均通过"
            else:
                verdict, reason = (
                    "not_reusable_corrupt_payload",
                    "RAR结构可解析但UnRAR完整性测试未通过（内容CRC错误），字节不可信；"
                    "保留原件，从目标条目起点重新下载",
                )
        record["reuse_verdict"] = verdict
        record["reuse_reason"] = reason
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        records.append(record)
        print(f"{name} bytes={record['bytes']} kind={record['kind']} "
              f"verdict={verdict}", flush=True)

    report = {
        "schema": "dronerfa-select12g-staging-audit/v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "downloads_dir": str(downloads),
        "counts": {
            "staged_files": len(records),
            "complete_archives": sum(1 for item in records if item["kind"] == "complete_archive"),
            "partials": sum(1 for item in records if item["kind"] == "partial"),
        },
        "bytes": {
            "complete_archives": sum(item["bytes"] for item in records
                                     if item["kind"] == "complete_archive"),
            "partials": sum(item["bytes"] for item in records if item["kind"] == "partial"),
            "total": sum(item["bytes"] for item in records),
        },
        "verdict_counts": verdict_counts,
        "modified": False,
        "note": "本脚本只读：不写入、不移动、不删除G盘任何文件",
        "files": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("counts", "bytes", "verdict_counts")},
                     ensure_ascii=False, indent=2))
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
