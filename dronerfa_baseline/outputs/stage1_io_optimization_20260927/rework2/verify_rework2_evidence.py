#!/usr/bin/env python3
"""Independent consistency check of the STAGE1-IO-OPT-R2 evidence (read-only).

Checks, from the artifacts themselves:

1. the R2 run's JSON and CSV agree field by field (throughput, audit, memory);
2. every memory sample is ok, finite and positive, ``peak >= working set``, and the
   per-configuration memory numbers are exactly the maxima over that run's samples;
3. the JSON states the memory scope as parent-process-only and says worker memory was
   not measured (the R2 scope decision);
4. the audit closes: reads == completed == windows, zero unconsumed reads, and the
   per-batch input digests are identical across the 0/2/4-worker columns;
5. ``source_sha256["benchmark_stage1_io.py"]`` in the JSON equals the current file, and
   the frozen window index hash is the accepted one;
6. the R1 (rework1) artifacts were not regenerated: their embedded
   ``source_sha256`` still records the pre-R2 source hash, and their mtime is older
   than this R2 run;
7. the two formal seed runs and the accepted verification evidence still hash to the
   values recorded by Codex.

Writes only ``rework2_evidence_check.json`` next to itself.  No file is modified.

Run: ``python verify_rework2_evidence.py`` (from ``dronerfa_baseline``).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE = SCRIPT_DIR.parents[2]
OUT_ROOT = BASELINE / "outputs" / "stage1_io_optimization_20260927"
REWORK1 = OUT_ROOT / "rework1"
REWORK2 = OUT_ROOT / "rework2"
FORMAL = BASELINE / "outputs" / "stage1_12drones_20260926"

R2_JSON = REWORK2 / "benchmark" / "benchmark_latest.json"
R2_CSV = REWORK2 / "benchmark" / "benchmark_summary.csv"
R1_JSON = REWORK1 / "benchmark" / "benchmark_latest.json"
R1_SOURCE_SHA = "129d26abb387b50daf0752a2a677c5989833a8e3d2b3469422bbca4f2189273d"
INDEX_SHA = "5973a2fa5b8476a507fcc7a665365daf8d572f569ce5908d54f5c4bc84c48bd6"
EXPECTED_FORMAL = {
    "outputs/stage1_12drones_20260926/seed_2026/best_model.pt": (
        "37dbb815450b8ec0f798a113a7f922858fcb2be604a863eb7594eb5c52ff814b"
    ),
    "outputs/stage1_12drones_20260926/seed_2027/best_model.pt": (
        "63e611e797d5e189662bbcb884fa9edc040692b699092e97ba04845e240cfbcc"
    ),
    "outputs/stage1_12drones_20260926/verification.json": (
        "9a921616bc720ca10bd6d17071253a5a6f7a198644360dd1ef419c4e452f36ae"
    ),
    "outputs/stage1_12drones_20260926/verification_seed_2027.json": (
        "b282211302a3582aec7182e6eccbcf87c43ddf415fe148baac140d68c17ecb4e"
    ),
}
R1_MUST_BE_UNCHANGED = (
    "smoke_serial/run_config.json",
    "smoke_serial/loader_read_audit.json",
    "smoke_parallel2/run_config.json",
    "smoke_parallel2/loader_read_audit.json",
    "equivalence/equivalence_latest.json",
    "verification/smoke_parallel2_loader_audit.json",
    "verification/smoke_serial_loader_audit.json",
    "benchmark/benchmark_latest.json",
)

checks: list[dict[str, Any]] = []


def check(name: str, passed: bool, evidence: Any) -> None:
    checks.append({"check": name, "passed": bool(passed), "evidence": evidence})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    payload = json.loads(R2_JSON.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(R2_CSV.read_text(encoding="utf-8-sig").splitlines()))
    r1 = json.loads(R1_JSON.read_text(encoding="utf-8"))
    run_id = payload["run_id"]

    check(
        "csv_and_json_are_the_same_run",
        len(rows) == len(payload["configurations"])
        and all(row["run_id"] == run_id for row in rows)
        and R2_JSON.read_bytes() == (REWORK2 / "benchmark" / f"benchmark_{run_id}.json").read_bytes(),
        {
            "run_id": run_id,
            "csv_rows": len(rows),
            "json_configurations": sorted(payload["configurations"]),
            "raw_equals_latest": True,
        },
    )

    for row in rows:
        workers = row["num_workers"]
        entry = payload["configurations"][workers]
        memory = entry["memory"]
        same = {
            "windows_per_second_mean": abs(
                float(row["windows_per_second_mean"]) - entry["windows_per_second_mean"]
            )
            < 1e-12,
            "speedup": abs(
                float(row["speedup_vs_first_configuration"])
                - entry["speedup_vs_first_configuration"]
            )
            < 1e-12,
            "windows_read": int(row["windows_read"]) == entry["audit"]["reads"],
            "windows_completed": int(row["windows_completed"]) == entry["audit"]["completed"],
            "unconsumed_reads": int(row["unconsumed_reads"]) == entry["audit"]["unconsumed_reads"],
            "parent_windows_read": int(row["parent_windows_read"])
            == entry["audit"]["parent_windows_read"],
            "parent_working_set_mb": abs(
                float(row["parent_working_set_mb"]) - memory["parent_working_set_mb_max"]
            )
            < 1e-9,
            "parent_peak_working_set_mb": abs(
                float(row["parent_peak_working_set_mb"])
                - memory["parent_peak_working_set_mb_max"]
            )
            < 1e-9,
            "memory_samples": int(row["memory_samples"]) == memory["sample_count"],
            "memory_sampling_ok": (row["memory_sampling_ok"] == "True") is memory["ok"],
            "memory_scope": row["memory_scope"] == memory["scope"],
        }
        check(f"csv_matches_json[workers={workers}]", all(same.values()), same)

        samples = memory["samples"]
        finite_positive = all(
            math.isfinite(float(sample["parent_working_set_mb"]))
            and math.isfinite(float(sample["parent_peak_working_set_mb"]))
            and float(sample["parent_working_set_mb"]) > 0.0
            and float(sample["parent_peak_working_set_mb"]) > 0.0
            for sample in samples
        )
        peak_covers = all(
            float(sample["parent_peak_working_set_mb"])
            >= float(sample["parent_working_set_mb"]) - 1e-9
            for sample in samples
        )
        maxima_match = (
            abs(
                max(float(s["parent_working_set_mb"]) for s in samples)
                - float(memory["parent_working_set_mb_max"])
            )
            < 1e-9
            and abs(
                max(float(s["parent_peak_working_set_mb"]) for s in samples)
                - float(memory["parent_peak_working_set_mb_max"])
            )
            < 1e-9
        )
        check(
            f"memory_samples_are_valid[workers={workers}]",
            bool(samples)
            and all(sample["ok"] for sample in samples)
            and finite_positive
            and peak_covers
            and maxima_match
            and memory["errors"] == [],
            {
                "samples": len(samples),
                "labels": sorted({sample["label"] for sample in samples}),
                "working_set_mb_max": memory["parent_working_set_mb_max"],
                "peak_working_set_mb_max": memory["parent_peak_working_set_mb_max"],
                "finite_positive": finite_positive,
                "peak_ge_working_set": peak_covers,
                "maxima_match_samples": maxima_match,
                "sample_seconds_total": memory["sample_seconds_total"],
            },
        )
        check(
            f"audit_closes[workers={workers}]",
            entry["audit"]["reads"] == entry["audit"]["completed"] == entry["windows_total"]
            and entry["audit"]["unconsumed_reads"] == 0
            and entry["audit"]["unread_consumed"] == 0,
            entry["audit"],
        )
        check(
            f"inputs_identical_to_reference[workers={workers}]",
            payload["equivalence"][workers]["digest_lists_identical"] is True,
            payload["equivalence"][workers],
        )

    sampling = payload["memory_sampling"]
    check(
        "memory_sampling_block_is_honest",
        sampling["ok"] is True
        and sampling["scope"] == "parent process only (the training-loop interpreter)"
        and set(sampling["fields"]) == {"parent_working_set_mb", "parent_peak_working_set_mb"}
        and "worker" in sampling["not_measured"].lower()
        and "must not be quoted" in sampling["not_measured"]
        and sampling["failures"] == []
        and sampling["api"]["api"].endswith("GetProcessMemoryInfo")
        and sampling["api"]["process_memory_counters_bytes"] == 72,
        {
            "ok": sampling["ok"],
            "scope": sampling["scope"],
            "fields": sampling["fields"],
            "not_measured": sampling["not_measured"],
            "api": sampling["api"],
            "failures": sampling["failures"],
            "sampling_points": sampling["sampling_points"],
        },
    )

    current_source_sha = sha256_file(BASELINE / "benchmark_stage1_io.py")
    check(
        "source_and_index_hashes",
        payload["source_sha256"]["benchmark_stage1_io.py"] == current_source_sha
        and current_source_sha != R1_SOURCE_SHA
        and payload["index"]["sha256"] == INDEX_SHA,
        {
            "benchmark_source_sha256_recorded": payload["source_sha256"]["benchmark_stage1_io.py"],
            "benchmark_source_sha256_current": current_source_sha,
            "r1_source_sha256": R1_SOURCE_SHA,
            "index_sha256": payload["index"]["sha256"],
            "index_sha256_expected": INDEX_SHA,
        },
    )

    r2_source_files = payload["source_sha256"]
    loader_unchanged = (
        r2_source_files["stage1_loader_optimization.py"]
        == sha256_file(BASELINE / "stage1_loader_optimization.py")
        and r2_source_files["run_stage1_12drones_training.py"]
        == sha256_file(BASELINE / "run_stage1_12drones_training.py")
    )
    check(
        "r2_changed_only_the_benchmark_script",
        loader_unchanged,
        {
            "stage1_loader_optimization.py": r2_source_files["stage1_loader_optimization.py"],
            "run_stage1_12drones_training.py": r2_source_files["run_stage1_12drones_training.py"],
        },
    )

    r2_started = datetime.fromisoformat(payload["started_at"]).timestamp()
    r1_state = []
    for relative in R1_MUST_BE_UNCHANGED:
        path = REWORK1 / relative
        entry = json.loads(path.read_text(encoding="utf-8"))
        recorded = entry.get("source_sha256") if isinstance(entry, dict) else None
        if recorded is None and isinstance(entry, dict):
            recorded = (entry.get("run_config") or {}).get("source_sha256")
        recorded_matches_current = None
        if recorded:
            recorded_matches_current = {
                name: value == sha256_file(BASELINE / name)
                for name, value in recorded.items()
                if (BASELINE / name).is_file()
            }
        r1_state.append(
            {
                "path": f"rework1/{relative}",
                "sha256": sha256_file(path),
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                "mtime_before_r2_run": path.stat().st_mtime < r2_started,
                # Self-timestamped by the R1 run itself: a regenerated artifact would carry
                # a 2026-09-28 timestamp, so this date is independent of the filesystem.
                "artifact_self_timestamp": (
                    entry.get("generated_at") or entry.get("completed_at")
                    if isinstance(entry, dict)
                    else None
                ),
                "recorded_source_sha256": recorded,
                "recorded_source_sha256_matches_current_sources": recorded_matches_current,
            }
        )
    # (a) nothing was rewritten during this run, and (b) every artifact that carries its
    # own timestamp still says 2026-09-27 (a regenerated one would say 2026-09-28).
    self_timestamped = [
        entry for entry in r1_state if entry["artifact_self_timestamp"] is not None
    ]
    check(
        "r1_artifacts_not_rewritten_in_r2",
        len(self_timestamped) >= 6
        and all(
            entry["mtime_before_r2_run"]
            and (
                entry["artifact_self_timestamp"] is None
                or str(entry["artifact_self_timestamp"]).startswith("2026-09-27")
            )
            for entry in r1_state
        ),
        {
            "r2_started_at": payload["started_at"],
            "files_with_self_timestamp": len(self_timestamped),
            "files": [
                {
                    "path": entry["path"],
                    "sha256": entry["sha256"],
                    "mtime": entry["mtime"],
                    "artifact_self_timestamp": entry["artifact_self_timestamp"],
                }
                for entry in r1_state
            ],
        },
    )
    # Every recorded source must still hash to what the artifact recorded, except
    # ``benchmark_stage1_io.py`` itself: R2 edited that file on purpose, which is
    # exactly why the R1 benchmark JSON must still carry the *old* hash (otherwise the
    # R1 numbers would have been regenerated with the R2 script).
    recorded_ok = True
    for entry in r1_state:
        mapping = entry["recorded_source_sha256_matches_current_sources"]
        if not mapping:
            continue
        for name, matches in mapping.items():
            if name == "benchmark_stage1_io.py":
                continue
            recorded_ok = recorded_ok and bool(matches)
    r1_benchmark_sha = json.loads(R1_JSON.read_text(encoding="utf-8"))["source_sha256"][
        "benchmark_stage1_io.py"
    ]
    current_benchmark_sha = sha256_file(BASELINE / "benchmark_stage1_io.py")
    check(
        "r1_artifacts_still_describe_the_r1_sources",
        recorded_ok
        and r1_benchmark_sha == R1_SOURCE_SHA
        and current_benchmark_sha != R1_SOURCE_SHA,
        {
            "r1_benchmark_source_sha256": r1_benchmark_sha,
            "expected_r1_source_sha256": R1_SOURCE_SHA,
            "current_benchmark_source_sha256": current_benchmark_sha,
            "note": (
                "benchmark_stage1_io.py differs from the R1 recording by design (the R2 "
                "fix); every other recorded source still matches the file on disk"
            ),
            "recorded_sources_all_current_except_benchmark": recorded_ok,
            "files": r1_state,
        },
    )

    formal = []
    for relative, expected in EXPECTED_FORMAL.items():
        actual = sha256_file(BASELINE / relative)
        formal.append(
            {"path": relative, "expected": expected, "actual": actual, "match": actual == expected}
        )
    check("formal_seed_runs_unchanged", all(entry["match"] for entry in formal), formal)

    comparison = {}
    for workers in sorted(payload["configurations"], key=int):
        new = payload["configurations"][workers]
        old = r1["configurations"][workers]
        comparison[workers] = {
            "r1_windows_per_second_mean": old["windows_per_second_mean"],
            "r2_windows_per_second_mean": new["windows_per_second_mean"],
            "r1_speedup_vs_first": old["speedup_vs_first_configuration"],
            "r2_speedup_vs_first": new["speedup_vs_first_configuration"],
            "r1_loader_wait_seconds": old["region_seconds"]["seconds_loader_wait"],
            "r2_loader_wait_seconds": new["region_seconds"]["seconds_loader_wait"],
            "r1_memory": old["memory"],
            "r2_parent_working_set_mb_max": new["memory"]["parent_working_set_mb_max"],
        }
    same_conclusion = (
        max(comparison, key=lambda w: comparison[w]["r2_windows_per_second_mean"]) == "0"
        and comparison["2"]["r2_speedup_vs_first"] < 1.0
        and comparison["4"]["r2_speedup_vs_first"] < comparison["2"]["r2_speedup_vs_first"]
    )
    check(
        "r1_run_size_and_memory_columns",
        len(rows[0]) > 25 and "parent_working_set_mb" in rows[0] and "memory_sampling_ok" in rows[0],
        {"csv_columns": len(rows[0]), "names": list(rows[0])},
    )
    check(
        "conclusion_unchanged_and_memory_now_recorded",
        same_conclusion
        and all(
            isinstance(entry["r1_memory"], dict) and entry["r1_memory"] == {}
            for entry in comparison.values()
        )
        and all(
            float(comparison[w]["r2_parent_working_set_mb_max"]) > 0.0
            for w in comparison
        ),
        {"comparison": comparison, "same_conclusion": same_conclusion},
    )

    result = {
        "schema": "dronerfa-stage1-io-rework2-evidence-check/v1",
        "checked_at": datetime.now().isoformat(),
        "run_id": run_id,
        "passed": all(entry["passed"] for entry in checks),
        "failed": [entry["check"] for entry in checks if not entry["passed"]],
        "checks": checks,
    }
    (SCRIPT_DIR / "rework2_evidence_check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"passed={result['passed']} checks={len(checks)} failed={result['failed']}")
    print(f"comparison={json.dumps(comparison, ensure_ascii=False)}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
