#!/usr/bin/env python3
"""Tests for the checklist in ``build_g_manifest_candidate.py``.

The builder's value is its checklist: "104 rows, 13x8 classes, no F paths, hash evidence
for every file, legacy rows preserved".  A checklist that cannot fail proves nothing, so
this suite drives the real :func:`build_g_manifest_candidate.build` against small synthetic
inputs (2 classes x 8 files) and asserts both directions:

* a clean synthetic input makes every check pass;
* each seeded defect makes the intended check fail.

Nothing here touches ``G:``: the synthetic config carries its own ``g_original`` directory
and the download-state path is redirected to a non-existent file.

Run: .venv\\Scripts\\python.exe work\\sel12g\\test_build_g_manifest_candidate.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))

import build_g_manifest_candidate as builder  # noqa: E402

TEST_ROOT = WORK / "_manifest_tests"
CLASSES = {
    "T0000": {"class_name": "Background", "class_index": 0},
    "T0001": {"class_name": "DJI Phantom 3", "class_index": 1},
}
SEGMENTS = [f"S{index:04d}" for index in range(8)]
LEGACY_FILES = [f"T0000_D00_{segment}.mat" for segment in SEGMENTS]
NEW_FILES = [f"T0001_D00_{segment}.mat" for segment in SEGMENTS]
FILE_BYTES = 64


class Fixture:
    """A tiny, self-consistent stand-in for the real G inputs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.original = root / "original"
        self.original.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict] = {}
        for index, name in enumerate(LEGACY_FILES + NEW_FILES):
            payload = bytes([index]) * FILE_BYTES
            (self.original / name).write_bytes(payload)
            self.records[name] = {
                "bytes": FILE_BYTES,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "present": True,
            }
        self.config_path = root / "config.json"
        self.audit_path = root / "audit.json"
        self.sha256_path = root / "sha256.json"
        self.legacy_path = root / "legacy-manifest.csv"
        self.legacy_sizes = {name: record["bytes"] for name, record in self.records.items()}
        self.audit_names = list(self.records)
        self.audit_bytes: dict[str, int] = {}
        self.hash_names = list(self.records)
        self.write()

    # ---------------------------------------------------------------- writers
    def write(self) -> None:
        files = []
        for name, record in self.records.items():
            class_id = name.split("_")[0]
            legacy = name in LEGACY_FILES
            files.append({
                "filename": name,
                "target_id": class_id,
                "class_name": CLASSES[class_id]["class_name"],
                "class_index": CLASSES[class_id]["class_index"],
                "origin": "baseline_24" if legacy else "to_download",
                "former_partition": (
                    "development_original" if name in LEGACY_FILES[:4]
                    else "evaluated_holdout" if legacy else None
                ),
                "split_status": (
                    "preserve_former_partition" if legacy else "unassigned"
                ),
                "expected_mat_bytes": record["bytes"],
            })
        self.config_path.write_text(json.dumps({
            "paths": {"g_original": str(self.original)},
            "files": files,
        }, ensure_ascii=False), encoding="utf-8")
        self.audit_path.write_text(json.dumps({
            "checked_at": "synthetic",
            "counts": {"mat_on_g": len(self.audit_names), "invalid": 0,
                       "target_missing_on_g": 0},
            "problems": [],
            "files": [{"filename": name,
                       "actual_bytes": self.audit_bytes.get(name, self.records[name]["bytes"])}
                      for name in self.audit_names],
        }), encoding="utf-8")
        self.sha256_path.write_text(json.dumps({
            "computed_at": "synthetic",
            "count": len(self.hash_names),
            "files": [{"filename": name, "bytes": self.records[name]["bytes"],
                       "sha256": self.records[name]["sha256"]}
                      for name in self.hash_names],
        }), encoding="utf-8")
        with self.legacy_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
            writer.writerow(builder.LEGACY_COLUMNS)
            for index, name in enumerate(LEGACY_FILES):
                class_id = name.split("_")[0]
                partition = ("development_original" if index < 4 else "evaluated_holdout")
                writer.writerow([name, class_id, CLASSES[class_id]["class_name"], partition,
                                 f"F:\\DroneRFa数据集\\original\\{name}",
                                 self.legacy_sizes[name],
                                 CLASSES[class_id]["class_index"]])

    def edit_legacy(self, mutate) -> None:
        rows = list(csv.reader(
            self.legacy_path.read_text(encoding="utf-8").splitlines()))
        mutate(rows)
        with self.legacy_path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle, quoting=csv.QUOTE_ALL).writerows(rows)

    def edit_config(self, mutate) -> None:
        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        mutate(payload)
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False),
                                    encoding="utf-8")

    def apply(self) -> None:
        """Point the builder at this fixture with synthetic expectations."""
        builder.CONFIG = self.config_path
        builder.AUDIT = self.audit_path
        builder.SHA256_REPORT = self.sha256_path
        builder.LEGACY_MANIFEST = self.legacy_path
        builder.STATE = self.root / "no-such-state.json"
        builder.EXPECTED_ROWS = len(LEGACY_FILES) + len(NEW_FILES)
        builder.EXPECTED_LEGACY_ROWS = len(LEGACY_FILES)
        builder.EXPECTED_NEW_ROWS = len(NEW_FILES)
        builder.EXPECTED_CLASSES = len(CLASSES)
        builder.EXPECTED_FILES_PER_CLASS = len(SEGMENTS)


class ManifestBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = TEST_ROOT / self._testMethodName
        shutil.rmtree(self.root, ignore_errors=True)
        self.fixture = Fixture(self.root)
        self.fixture.apply()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def run_build(self) -> dict:
        report, rows = builder.build()
        report["_rows"] = rows
        return report

    def failed_checks(self, report: dict) -> list[str]:
        return [item["check"] for item in report["checks"] if not item["ok"]]

    # ---------------------------------------------------------------- controls
    def test_clean_fixture_passes_every_check(self) -> None:
        report = self.run_build()
        self.assertEqual(self.failed_checks(report), [])
        self.assertTrue(report["ok"])
        self.assertEqual(report["counts"]["candidate_rows"], 16)
        self.assertEqual(report["counts"]["legacy_rows_preserved"], 8)
        self.assertEqual(report["counts"]["new_rows"], 8)

    def test_candidate_rows_carry_the_required_semantics(self) -> None:
        report = self.run_build()
        rows = {row["file_name"]: row for row in report["_rows"]}
        for name in LEGACY_FILES[:4]:
            self.assertEqual(rows[name]["former_partition"], "development_original")
            self.assertEqual(rows[name]["evaluation_role"], "legacy_development")
            self.assertEqual(rows[name]["split_status"], "preserve_former_partition")
            self.assertEqual(rows[name]["origin"], "baseline_24")
        for name in LEGACY_FILES[4:]:
            self.assertEqual(rows[name]["former_partition"], "evaluated_holdout")
            self.assertEqual(rows[name]["evaluation_role"], "previously_evaluated")
        for name in NEW_FILES:
            self.assertEqual(rows[name]["former_partition"], "")
            self.assertEqual(rows[name]["evaluation_role"], "unassigned")
            self.assertEqual(rows[name]["split_status"], "unassigned")
            self.assertEqual(rows[name]["origin"], "selected_80")
            self.assertTrue(rows[name]["unified_path"].startswith(str(self.fixture.original)))
            self.assertNotIn("F:", rows[name]["unified_path"])

    def test_legacy_order_is_kept_and_new_rows_are_appended(self) -> None:
        report = self.run_build()
        self.assertEqual([row["file_name"] for row in report["_rows"]][:len(LEGACY_FILES)],
                         LEGACY_FILES)

    # ---------------------------------------------------------------- defects
    def test_wrong_file_size_is_caught(self) -> None:
        """On-disk/audit size differs from the frozen target size."""
        name = NEW_FILES[0]
        self.fixture.audit_bytes[name] = FILE_BYTES + 1
        (self.fixture.original / name).write_bytes(b"\0" * (FILE_BYTES + 1))
        self.fixture.write()
        report = self.run_build()
        self.assertFalse(report["ok"])
        self.assertIn("no_outstanding_issues", self.failed_checks(report))
        self.assertTrue(any(name in item for item in report["anomalies"]))

    def test_size_evidence_that_disagrees_with_disk_is_caught(self) -> None:
        """The audit claims a size the real file does not have."""
        name = NEW_FILES[2]
        self.fixture.audit_bytes[name] = FILE_BYTES + 5
        self.fixture.write()
        report = self.run_build()
        self.assertFalse(report["ok"])
        self.assertIn("no_outstanding_issues", self.failed_checks(report))

    def test_hash_evidence_with_the_wrong_byte_count_is_caught(self) -> None:
        name = NEW_FILES[3]
        payload = json.loads(self.fixture.sha256_path.read_text(encoding="utf-8"))
        for item in payload["files"]:
            if item["filename"] == name:
                item["bytes"] = FILE_BYTES + 3
        self.fixture.sha256_path.write_text(json.dumps(payload), encoding="utf-8")
        report = self.run_build()
        self.assertFalse(report["ok"])
        self.assertIn("no_outstanding_issues", self.failed_checks(report))

    def test_missing_file_on_g_is_caught(self) -> None:
        (self.fixture.original / NEW_FILES[1]).unlink()
        report = self.run_build()
        self.assertFalse(report["ok"])
        self.assertIn("no_outstanding_issues", self.failed_checks(report))

    def test_missing_hash_evidence_is_caught(self) -> None:
        self.fixture.hash_names = self.fixture.hash_names[:-1]
        self.fixture.write()
        report = self.run_build()
        self.assertFalse(report["ok"])
        self.assertIn("hash_report_covers_all_targets", self.failed_checks(report))

    def test_missing_audit_evidence_is_caught(self) -> None:
        self.fixture.audit_names = self.fixture.audit_names[:-1]
        self.fixture.write()
        report = self.run_build()
        self.assertFalse(report["ok"])
        self.assertIn("no_outstanding_issues", self.failed_checks(report))

    def test_legacy_row_that_disagrees_with_target_table_is_caught(self) -> None:
        self.fixture.edit_legacy(lambda rows: rows[1].__setitem__(
            5, str(int(rows[1][5]) + 7)))
        report = self.run_build()
        self.assertFalse(report["ok"])
        self.assertIn("legacy_rows_match_frozen_targets", self.failed_checks(report))

    def test_legacy_row_outside_the_frozen_table_is_caught(self) -> None:
        self.fixture.edit_legacy(lambda rows: rows[1].__setitem__(
            0, "T9999_D00_S0000.mat"))
        report = self.run_build()
        self.assertFalse(report["ok"])
        self.assertIn("legacy_rows_match_frozen_targets", self.failed_checks(report))

    def test_legacy_partition_changed_by_the_target_table_is_caught(self) -> None:
        def mutate(payload: dict) -> None:
            for item in payload["files"]:
                if item["filename"] == LEGACY_FILES[0]:
                    item["former_partition"] = "evaluated_holdout"
        self.fixture.edit_config(mutate)
        report = self.run_build()
        self.assertFalse(report["ok"])
        self.assertIn("legacy_rows_match_frozen_targets", self.failed_checks(report))

    def test_forbidden_evaluation_role_is_caught(self) -> None:
        original = dict(builder.ROLE_BY_FORMER_PARTITION)
        builder.ROLE_BY_FORMER_PARTITION["development_original"] = "final_test"
        self.addCleanup(builder.ROLE_BY_FORMER_PARTITION.update, original)
        report = self.run_build()
        self.assertFalse(report["ok"])
        self.assertIn("no_forbidden_evaluation_role", self.failed_checks(report))

    def test_new_target_with_a_former_partition_is_caught(self) -> None:
        def mutate(payload: dict) -> None:
            for item in payload["files"]:
                if item["origin"] == "to_download":
                    item["former_partition"] = "development_original"
                    return
        self.fixture.edit_config(mutate)
        report = self.run_build()
        self.assertFalse(report["ok"])
        self.assertIn("new_targets_have_no_former_partition", self.failed_checks(report))

    def test_class_count_mismatch_is_caught(self) -> None:
        builder.EXPECTED_FILES_PER_CLASS = 7
        report = self.run_build()
        self.assertFalse(report["ok"])
        self.assertIn("classes_and_files_per_class", self.failed_checks(report))

    def test_candidate_row_count_mismatch_is_caught(self) -> None:
        builder.EXPECTED_ROWS = 15
        report = self.run_build()
        self.assertFalse(report["ok"])
        self.assertIn("candidate_row_count", self.failed_checks(report))

    def test_duplicate_content_is_reported(self) -> None:
        for name in NEW_FILES:
            self.fixture.records[name]["sha256"] = self.fixture.records[LEGACY_FILES[0]]["sha256"]
        self.fixture.write()
        report = self.run_build()
        self.assertTrue(report["ok"], self.failed_checks(report))
        self.assertEqual(report["duplicate_content_groups"],
                         [sorted([LEGACY_FILES[0]] + NEW_FILES)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
