#!/usr/bin/env python3
"""Tests for the CASCADE3-NEWSPLIT-A window index and its verification.

Three risk areas are covered, in the order they can invalidate the split:

1. ``WindowRuleTests`` – the window rule must produce *every* complete 1M window,
   non-overlapping and 1M aligned, and must drop only the sub-1M tail.  A silent
   change here (for example a block-quantised count) would shrink or overlap the
   training set without any visible error.
2. ``SyntheticEndToEndTests`` – a miniature authority tree (frozen CSV + freeze
   JSON + small HDF5 MATs) is pushed through the real builder functions and then
   through the real verifier check groups, so index/CSV/labels/summary wiring is
   exercised without touching the G: drive.
3. ``RealArtifactTests`` – skipped unless the published artifacts exist: the
   index must match its own summary hash, contain no legacy path, and the first,
   middle and last window of a real source MAT must load finite through
   ``dronerfa_data.read_vector_slice``.

Run with::

    python -m unittest dronerfa_baseline.test_newsplit_index -v
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_newsplit_index as builder  # noqa: E402
import verify_newsplit_index as verifier  # noqa: E402
from dronerfa_data import read_vector_slice  # noqa: E402

REAL_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "cascade3_newsplit_20260926"
#: Synthetic trees live inside the workspace scratch directory; the DSH file
#: sandbox may not grant directory creation in the platform temp area.
SCRATCH_ROOT = SCRIPT_DIR / "scratch" / "test_newsplit_index"
SYNTHETIC_ROLE_FILES = {"train": 1, "val": 1, "development_test_exposed": 1}
SYNTHETIC_SELECTION_FILES = len(builder.TARGET_CODES) * len(builder.ROLE_ORDER)
SYNTHETIC_SAMPLES = {
    # deliberately not a multiple of 1M: the tail must be dropped
    "T0000": 2_500_000,
    "T0010": 1_000_000,
    "T0101": 3_200_000,
}


# ---------------------------------------------------------------------------
# 1. pure window rule
# ---------------------------------------------------------------------------
class WindowRuleTests(unittest.TestCase):
    def test_counts_all_complete_windows_and_drops_the_tail(self) -> None:
        cases = {0: 0, 1: 0, 999_999: 0, 1_000_000: 1, 2_500_000: 2, 24_000_000: 24}
        for samples, expected in cases.items():
            with self.subTest(samples=samples):
                self.assertEqual(builder.window_count(samples), expected)

    def test_planned_windows_tile_the_record_without_gap_or_overlap(self) -> None:
        source = builder.FrozenSource(
            filename="T0010_D00_S0000.mat",
            file_path=Path("X:/nonexistent.mat"),
            target_code="T0010",
            role="train",
            size_bytes=0,
            class_id="T0010",
            class_name="DJI Phantom 4 Pro",
            segment_code="0000",
            distance_code="00",
            metrics_scope="development",
            role_purpose="model fitting",
            prior_use_role="train",
            archive_segment_role="train",
            exposed_to_prior_selection="True",
            expected_sha256="0" * 64,
        )
        rows = builder.plan_file_windows(source, 2_500_000, source.expected_sha256)
        self.assertEqual(len(rows), 2)
        spans = [(row["start"], row["start"] + row["length"]) for row in rows]
        self.assertEqual(spans, [(0, 1_000_000), (1_000_000, 2_000_000)])
        self.assertEqual([row["window_index"] for row in rows], [0, 1])
        self.assertEqual([row["block_index"] for row in rows], [0, 0])
        self.assertEqual([row["window_in_block"] for row in rows], [0, 1])
        self.assertTrue(all(row["length"] == 1_000_000 for row in rows))
        self.assertTrue(all(row["stage1_label"] == 1 for row in rows))
        self.assertTrue(all(row["class_index"] == 1 for row in rows))
        self.assertTrue(all(row["stage2_label"] == "0" for row in rows))
        # 24 windows of a 24M file must never cross a 10M acquisition block
        twenty_four = builder.plan_file_windows(source, 24_000_000)
        self.assertTrue(
            all(
                row["start"] // builder.BLOCK_SIZE
                == (row["start"] + row["length"] - 1) // builder.BLOCK_SIZE
                for row in twenty_four
            )
        )
        self.assertEqual(twenty_four[-1]["start"] + twenty_four[-1]["length"], 24_000_000)

    def test_background_rows_have_no_stage2_label(self) -> None:
        source = builder.FrozenSource(
            filename="T0000_D00_S0000.mat",
            file_path=Path("X:/nonexistent.mat"),
            target_code="T0000",
            role="val",
            size_bytes=0,
            class_id="T0000",
            class_name="Background",
            segment_code="0000",
            distance_code="00",
            metrics_scope="development",
            role_purpose="model selection",
            prior_use_role="validation",
            archive_segment_role="validation",
            exposed_to_prior_selection="True",
            expected_sha256="0" * 64,
        )
        rows = builder.plan_file_windows(source, 1_000_000)
        self.assertEqual(rows[0]["stage1_label"], 0)
        self.assertEqual(rows[0]["label"], 0)
        self.assertEqual(rows[0]["stage2_eligible"], 0)
        self.assertEqual(rows[0]["stage2_label"], "")
        self.assertEqual(builder.stage2_view(rows), [])

    def test_stage2_view_keeps_every_drone_window_with_stable_positions(self) -> None:
        def row(row_index: int, eligible: int, role: str, code: str, stage2: str) -> dict:
            return {
                "row_index": row_index,
                "stage2_eligible": eligible,
                "source_role": role,
                "target_code": code,
                "stage2_label": stage2,
                "file_path": f"G:/x/{code}.mat",
                "filename": f"{code}_D00_S0000.mat",
                "target_name": code,
                "metrics_scope": "development",
                "class_index": builder.CLASS_INDEX[code],
                "class_name": builder.CLASS_NAME[code],
                "window_index": 0,
                "start": 0,
                "length": builder.WINDOW_SIZE,
                "source_samples": builder.WINDOW_SIZE,
                "file_sha256": "0" * 64,
            }

        rows = [
            row(0, 0, "train", "T0000", ""),
            row(1, 1, "train", "T0010", "0"),
            row(2, 1, "val", "T0101", "1"),
            row(3, 1, "train", "T0101", "1"),
        ]
        view = builder.stage2_view(rows)
        self.assertEqual([item["row_index"] for item in view], [1, 2, 3])
        self.assertEqual([item["stage2_position"] for item in view], [0, 1, 2])
        self.assertEqual([item["stage2_position_in_split"] for item in view], [0, 0, 1])
        self.assertEqual([item["stage2_label"] for item in view], [0, 1, 1])


# ---------------------------------------------------------------------------
# 2. synthetic authority tree -> builder -> verifier
# ---------------------------------------------------------------------------
class SyntheticEndToEndTests(unittest.TestCase):
    """Build a 3-file (one per class) miniature authority and verify it."""

    @classmethod
    def setUpClass(cls) -> None:
        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        cls.root = SCRATCH_ROOT / f"case_{uuid.uuid4().hex[:8]}"
        # registered first: class cleanups run even when setUpClass fails later
        cls.addClassCleanup(shutil.rmtree, cls.root, ignore_errors=True)
        cls.data_root = cls.root / "original"
        cls.data_root.mkdir(parents=True, exist_ok=True)
        cls.output_dir = cls.root / "out"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, str]] = []
        for code, name in builder.CLASS_NAME.items():
            for role_index, role in enumerate(builder.ROLE_ORDER):
                if SYNTHETIC_ROLE_FILES[role] != 1:
                    continue
                segment = {
                    "train": "0000",
                    "val": "0100",
                    "development_test_exposed": "0110",
                }[role]
                filename = f"{code}_D00_S{segment}.mat"
                path = cls.data_root / filename
                samples = SYNTHETIC_SAMPLES[code]
                # every source file must be distinguishable by content
                payload = np.arange(samples, dtype=np.float32) + role_index
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("RF0_I", data=payload)
                    handle.create_dataset("RF0_Q", data=np.zeros(samples, dtype=np.float32))
                rows.append(
                    {
                        "file_name": filename,
                        "class_id": code,
                        "class_name": name,
                        "role": role,
                        "role_rank_in_class": "1",
                        "role_purpose": {
                            "train": "model fitting",
                            "val": "model selection",
                            "development_test_exposed": "exposed control",
                        }[role],
                        "metrics_scope": (
                            "legacy_exposed_control"
                            if role == "development_test_exposed"
                            else "development"
                        ),
                        "prior_use_role": "descriptive_holdout" if role != "train" else "train",
                        "archive_segment_role": "train",
                        "archive_s_code": segment,
                        "exposed_to_prior_selection": "True",
                        "mat_sha256": verifier.sha256_file(path),
                        "g_path": str(path),
                        "size_bytes": str(path.stat().st_size),
                    }
                )

        cls.csv_path = cls.root / "synthetic_split.csv"
        fieldnames = list(rows[0].keys())
        with cls.csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        cls.manifest_path = cls.root / "synthetic_manifest.csv"
        cls.manifest_path.write_text("file_name,size_bytes\n", encoding="utf-8")
        cls.freeze_path = cls.root / "synthetic_freeze.json"
        cls.freeze_path.write_text(
            json.dumps(
                {
                    "schema": "dronerfa-meta-split2-codex-freeze/v1",
                    "decision": "accepted_frozen_development_split",
                    "frozen_split_csv": str(cls.csv_path),
                    "frozen_split_csv_sha256": verifier.sha256_file(cls.csv_path),
                    "source_manifest": str(cls.manifest_path),
                    "source_manifest_sha256": verifier.sha256_file(cls.manifest_path),
                    "counts": {
                        "total_sources": 3,
                        "train": 3,
                        "val": 3,
                        "development_test": 0,
                        "development_test_exposed": 3,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        cls.freeze, cls.csv_resolved, cls.manifest_sha = builder.load_freeze(cls.freeze_path)
        cls.sources = builder.load_frozen_sources(
            cls.csv_resolved,
            builder.TARGET_CODES,
            role_files_per_class=SYNTHETIC_ROLE_FILES,
            expected_total=SYNTHETIC_SELECTION_FILES,
        )
        cls.context = cls._build(cls.output_dir)

    @classmethod
    def _build(cls, output_dir: Path) -> verifier.Context:
        """Run the real builder functions into ``output_dir``."""
        output_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        identities: list[dict[str, object]] = []
        for source in sorted(cls.sources, key=lambda item: item.filename):
            identity = builder.check_source_identity(
                source, cls.data_root, "compute", progress=False
            )
            identities.append(identity)
            rows.extend(
                builder.plan_file_windows(
                    source, int(identity["source_samples"]), str(identity["sha256"])
                )
            )
        for position, row in enumerate(rows):
            row["row_index"] = position
        view = builder.stage2_view(rows)
        index_path = output_dir / "window_index_newsplit3.csv"
        stage2_path = output_dir / "stage2_view_newsplit3.csv"
        summary_path = output_dir / "newsplit3_dataset_summary.json"
        identity_path = output_dir / "newsplit3_source_identity.json"
        scheme_path = output_dir / "newsplit3_scheme.md"
        builder.write_csv(index_path, builder.CSV_FIELDS, rows)
        builder.write_csv(stage2_path, builder.STAGE2_FIELDS, view)
        counts = builder.count_table(rows)
        stage2_counts = builder.count_table(view)
        builder.write_json(identity_path, {"files": identities})
        builder.write_json(
            summary_path,
            {
                "counts": counts,
                "source_file_count": len(cls.sources),
                "index": {
                    "sha256": verifier.sha256_file(index_path),
                    "rows": len(rows),
                    "fields": list(builder.CSV_FIELDS),
                },
                "stage2_view": {"sha256": verifier.sha256_file(stage2_path), "counts": stage2_counts},
                "files": [
                    {
                        "filename": identity["filename"],
                        "target_code": next(
                            item.target_code
                            for item in cls.sources
                            if item.filename == identity["filename"]
                        ),
                        "source_role": next(
                            item.role
                            for item in cls.sources
                            if item.filename == identity["filename"]
                        ),
                        "windows": identity["complete_windows"],
                        "source_samples": identity["source_samples"],
                        "sha256": identity["sha256"],
                        "sha256_source": identity["sha256_source"],
                        "g_path": identity["g_path"],
                        "dropped_tail_samples": identity["dropped_tail_samples"],
                    }
                    for identity in identities
                ],
                "sampling": {"applied": False, "note": "synthetic"},
            },
        )
        scheme_path.write_text(
            "\n".join(
                [
                    "synthetic scheme",
                    verifier.sha256_file(cls.csv_path),
                    verifier.sha256_file(cls.manifest_path),
                    "development_test_exposed",
                ]
            ),
            encoding="utf-8",
        )

        context = verifier.Context(
            freeze_json=cls.freeze_path,
            output_dir=output_dir,
            data_root=cls.data_root,
            sha256_mode="cached",
            do_read_through=False,
            expected_frozen_csv_sha256=verifier.sha256_file(cls.csv_path),
            expected_manifest_sha256=verifier.sha256_file(cls.manifest_path),
            expected_selection_files=SYNTHETIC_SELECTION_FILES,
            expected_role_files_per_class=dict(SYNTHETIC_ROLE_FILES),
            expected_freeze_counts={
                "total_sources": 3,
                "train": 3,
                "val": 3,
                "development_test": 0,
                "development_test_exposed": 3,
            },
        )
        return verifier.load_context_into(context)

    def _fresh_context(self, output_dir: Path) -> verifier.Context:
        return self._build(output_dir)

    def test_index_contains_every_complete_window(self) -> None:
        rows = self.context.index_rows
        expected = (
            sum(SYNTHETIC_SAMPLES.values()) // 1_000_000
        ) * len(builder.ROLE_ORDER)
        self.assertEqual(len(rows), expected)
        per_file: dict[str, int] = {}
        for row in rows:
            per_file[row["filename"]] = per_file.get(row["filename"], 0) + 1
        self.assertEqual(per_file["T0000_D00_S0000.mat"], 2)
        self.assertEqual(per_file["T0010_D00_S0000.mat"], 1)
        self.assertEqual(per_file["T0101_D00_S0000.mat"], 3)
        self.assertEqual(sum(per_file.values()), expected)

    def test_all_checks_pass_on_a_valid_tree(self) -> None:
        checks = verifier.run_checks(self.context)
        failed = [check.check_id for check in checks if not check.passed]
        self.assertEqual(failed, [], f"unexpected failures: {failed}")

    def test_check_payloads_are_json_serialisable(self) -> None:
        # the verifier report is JSON: a set or tuple key would break it late
        checks = verifier.run_checks(self.context)
        payload = json.dumps([check.as_dict() for check in checks], ensure_ascii=False)
        self.assertGreater(len(payload), 0)

    def test_every_negative_control_is_detected(self) -> None:
        controls = verifier.negative_controls(self.context, verifier.run_checks)
        undetected = [control["control"] for control in controls if not control["detected"]]
        self.assertEqual(undetected, [], f"undetected mutations: {undetected}")

    def test_builder_rejects_a_tampered_freeze_hash(self) -> None:
        tampered = self.root / "tampered_freeze.json"
        payload = json.loads(self.freeze_path.read_text(encoding="utf-8"))
        payload["frozen_split_csv_sha256"] = "0" * 64
        tampered.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            builder.load_freeze(tampered)

    def test_builder_rejects_a_wrong_file_size(self) -> None:
        source = self.sources[0]
        broken = builder.FrozenSource(**{**source.__dict__, "size_bytes": source.size_bytes + 1})
        with self.assertRaises(ValueError):
            builder.check_source_identity(broken, self.data_root, "frozen", progress=False)

    def test_builder_rejects_a_wrong_sha256(self) -> None:
        source = self.sources[0]
        broken = builder.FrozenSource(**{**source.__dict__, "expected_sha256": "1" * 64})
        with self.assertRaises(ValueError):
            builder.check_source_identity(broken, self.data_root, "compute", progress=False)

    def test_builder_rejects_a_path_outside_the_data_root(self) -> None:
        source = self.sources[0]
        other = self.root / "elsewhere.mat"
        shutil.copy2(source.file_path, other)
        broken = builder.FrozenSource(**{**source.__dict__, "file_path": other})
        with self.assertRaises(ValueError):
            builder.check_source_identity(broken, self.data_root, "frozen", progress=False)

    def test_verifier_flags_a_missing_source_row(self) -> None:
        # drop a whole source file from the frozen CSV view: the index must then
        # be rejected as incomplete
        clone = self._build(self.root / "out_missing_source")
        clone.frozen_rows = [row for row in clone.frozen_rows if row["class_id"] != "T0101"]
        checks = verifier.run_checks(clone, include_hash_group=False)
        failed = [check.check_id for check in checks if not check.passed]
        self.assertIn("A7_frozen_selection_file_count", failed)


# ---------------------------------------------------------------------------
# 3. published artifacts against the real G: sources
# ---------------------------------------------------------------------------
class RealArtifactTests(unittest.TestCase):
    """Skipped when the CASCADE3-NEWSPLIT-A artifacts have not been published."""

    @classmethod
    def setUpClass(cls) -> None:
        if not (REAL_OUTPUT_DIR / "window_index_newsplit3.csv").is_file():
            raise unittest.SkipTest(f"index not generated yet: {REAL_OUTPUT_DIR}")
        cls.rows = verifier.read_csv_rows(REAL_OUTPUT_DIR / "window_index_newsplit3.csv")
        cls.summary = json.loads(
            (REAL_OUTPUT_DIR / "newsplit3_dataset_summary.json").read_text(encoding="utf-8")
        )

    def test_published_index_hash_matches_its_summary(self) -> None:
        digest = verifier.sha256_file(REAL_OUTPUT_DIR / "window_index_newsplit3.csv")
        self.assertEqual(digest, self.summary["index"]["sha256"])
        self.assertEqual(len(self.rows), int(self.summary["index"]["rows"]))

    def test_no_legacy_path_and_no_crossing_source_file(self) -> None:
        for row in self.rows:
            self.assertTrue(str(row["file_path"]).startswith("G:\\"), row["file_path"])
            for fragment in verifier.FORBIDDEN_PATH_FRAGMENTS:
                self.assertNotIn(fragment, str(row["file_path"]))
        roles_per_file: dict[str, set[str]] = {}
        for row in self.rows:
            roles_per_file.setdefault(str(row["filename"]), set()).add(str(row["source_role"]))
        self.assertTrue(all(len(roles) == 1 for roles in roles_per_file.values()))
        self.assertEqual(len(roles_per_file), 24)

    def test_real_windows_load_finite_through_the_project_reader(self) -> None:
        first_file = self.rows[0]["filename"]
        file_rows = sorted(
            (row for row in self.rows if row["filename"] == first_file),
            key=lambda row: int(row["window_index"]),
        )
        with h5py.File(file_rows[0]["file_path"], "r") as handle:
            for position in (0, len(file_rows) // 2, len(file_rows) - 1):
                row = file_rows[position]
                start, length = int(row["start"]), int(row["length"])
                i_values = read_vector_slice(handle["RF0_I"], start, length)
                q_values = read_vector_slice(handle["RF0_Q"], start, length)
                self.assertEqual(i_values.shape, (length,))
                self.assertEqual(q_values.shape, (length,))
                self.assertTrue(np.isfinite(i_values).all())
                self.assertTrue(np.isfinite(q_values).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
