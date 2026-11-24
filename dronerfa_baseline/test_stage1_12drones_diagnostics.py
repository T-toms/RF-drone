#!/usr/bin/env python3
"""Regression tests for the STAGE1-12DRONES-DIAG validation diagnostics.

Pure-logic tests of the diagnostic's own metric and integrity functions, plus
fail-fast tests that prove the diagnostic stops and reports instead of quietly
publishing numbers when its input is incomplete or tampered with.

Everything runs on CPU, reads only CSV/JSON, never opens a MAT file and never
writes inside ``outputs/stage1_12drones_20260926/`` (temporary output goes to a
``tempfile`` directory that is removed afterwards).
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np

import diagnose_stage1_12drones_validation as diag


class _Fixture:
    """Minimal frozen-index fixture: 2 background files + 1 drone file."""

    def __init__(self) -> None:
        self.ordered: list[dict] = []
        self.expected: dict[tuple[str, str], dict] = {}
        position = 0
        for filename, label, code, count in (
            ("T0000_D00_S0100.mat", 0, "T0000", 3),
            ("T0000_D00_S0101.mat", 0, "T0000", 2),
            ("T0001_D00_S0101.mat", 1, "T0001", 4),
        ):
            for window in range(count):
                row = {
                    "index_position": position,
                    "filename": filename,
                    "window_index": window,
                    "start": window * 1_000_000,
                    "label": label,
                    "target_code": code,
                    "split": "val",
                    "source_role": "val",
                }
                self.ordered.append(row)
                self.expected[(filename, str(window))] = {
                    "label": str(label),
                    "target_code": code,
                    "start": str(window * 1_000_000),
                }
                position += 1
        self.meta = {"val_files": 3}

    def rows(self, predictions: dict[tuple[str, int], int] | None = None) -> list[dict]:
        """One full four-condition table, the shape ``check_score_integrity`` sees."""
        predictions = predictions or {}
        rows: list[dict] = []
        for condition in diag.CONDITIONS:
            for row in self.ordered:
                key = (row["filename"], row["window_index"])
                prediction = predictions.get(key, row["label"])
                margin = 1.0 if prediction == 1 else -1.0
                rows.append(
                    {
                        "condition": condition,
                        "selection_position": row["index_position"],
                        "index_position": row["index_position"],
                        "filename": row["filename"],
                        "target_code": row["target_code"],
                        "window_index": row["window_index"],
                        "start": row["start"],
                        "label": row["label"],
                        "logit_margin": margin,
                        "prediction": prediction,
                    }
                )
        return rows


class MetricTests(unittest.TestCase):
    def test_pooled_metrics_match_a_hand_computed_confusion(self) -> None:
        labels = [1, 1, 1, 1, 0, 0, 0, 0]
        predictions = [1, 1, 0, 0, 0, 0, 1, 1]
        metrics = diag.pooled_metrics(labels, predictions)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]), (2, 2, 2, 2))
        self.assertAlmostEqual(metrics["drone_recall"], 0.5)
        self.assertAlmostEqual(metrics["background_specificity"], 0.5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_pooled_metrics_reject_a_non_conserving_confusion(self) -> None:
        with self.assertRaises(ValueError):
            diag.pooled_metrics([1, 0], [1, 2])

    def test_file_equal_metric_weights_files_not_windows(self) -> None:
        rows = [
            {"filename": "a.mat", "label": 1, "prediction": 1},
            {"filename": "a.mat", "label": 1, "prediction": 1},
            {"filename": "a.mat", "label": 1, "prediction": 1},
            {"filename": "b.mat", "label": 1, "prediction": 1},
            {"filename": "b.mat", "label": 1, "prediction": 0},
            {"filename": "c.mat", "label": 0, "prediction": 0},
        ]
        metrics = diag.file_equal_metrics(rows)
        # Pooled drone recall would be 4/5 = 0.8; per file it is (1.0 + 0.5) / 2.
        self.assertAlmostEqual(metrics["file_equal_drone_recall"], 0.75)
        self.assertAlmostEqual(metrics["file_equal_background_recall"], 1.0)
        self.assertAlmostEqual(metrics["file_equal_balanced_accuracy"], 0.875)
        pooled = diag.pooled_metrics([row["label"] for row in rows], [row["prediction"] for row in rows])
        self.assertAlmostEqual(pooled["drone_recall"], 0.8)

    def test_detection_auc_endpoints_and_ties(self) -> None:
        self.assertAlmostEqual(diag.detection_auc([2.0, 3.0], [-1.0, -2.0]), 1.0)
        self.assertAlmostEqual(diag.detection_auc([-2.0, -3.0], [1.0, 2.0]), 0.0)
        self.assertAlmostEqual(diag.detection_auc([0.0, 0.0], [0.0, 0.0]), 0.5)
        # Two of three pairs ordered correctly, so 2/3 - a tie costs half a point.
        self.assertAlmostEqual(diag.detection_auc([1.0, 1.0, -1.0], [0.0]), 2.0 / 3.0)
        self.assertAlmostEqual(diag.detection_auc([0.0, 1.0], [0.0]), 0.75)

    def test_d_prime_is_zero_for_identical_distributions(self) -> None:
        self.assertAlmostEqual(diag.d_prime([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0)
        self.assertGreater(diag.d_prime([5.0, 6.0], [0.0, 1.0]), 0.0)

    def test_summary_stats_report_quantiles_and_empty_input(self) -> None:
        stats = diag.summary_stats([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["n"], 4)
        self.assertAlmostEqual(stats["median"], 2.5)
        self.assertAlmostEqual(stats["min"], 1.0)
        self.assertAlmostEqual(stats["max"], 4.0)
        empty = diag.summary_stats([])
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["median"])

    def test_oracle_threshold_is_never_worse_than_the_recorded_threshold(self) -> None:
        rng = np.random.default_rng(7)
        per_condition = {
            "clean": {
                "drone_margins": rng.normal(1.0, 1.0, size=200).tolist(),
                "background_margins": rng.normal(-0.2, 1.0, size=50).tolist(),
            },
            "-15dB": {
                "drone_margins": rng.normal(0.1, 1.0, size=200).tolist(),
                "background_margins": rng.normal(-0.1, 1.0, size=50).tolist(),
            },
        }
        report = diag.oracle_thresholds(per_condition)
        for condition, payload in report["per_condition_oracle"].items():
            self.assertGreaterEqual(payload["balanced_accuracy"] + 1e-12, payload["balanced_accuracy_at_zero"])
        self.assertGreaterEqual(
            report["shared_threshold_oracle"]["mean_balanced_accuracy"] + 1e-12,
            float(
                np.mean(
                    [
                        payload["balanced_accuracy_at_zero"]
                        for payload in report["per_condition_oracle"].values()
                    ]
                )
            ),
        )


class IntegrityTests(unittest.TestCase):
    #: The injected cross-role probe is declared as a ``train`` file, exactly as
    #: the frozen index would label it.
    ROLE_MAP = {"T9999_D00_S0000.mat": "train"}

    def setUp(self) -> None:
        self.fixture = _Fixture()

    def _problems(self, rows: list[dict]) -> list[str]:
        report = diag.check_score_integrity(
            1, rows, self.fixture.expected, self.fixture.ordered, self.fixture.meta, self.ROLE_MAP
        )
        return report["problems"]

    def test_a_consistent_table_passes(self) -> None:
        self.assertEqual(self._problems(self.fixture.rows()), [])

    def test_duplicate_key_is_rejected(self) -> None:
        rows = self.fixture.rows()
        rows.append(dict(rows[0]))
        self.assertTrue(any("duplicate keys" in problem for problem in self._problems(rows)))

    def test_missing_window_is_rejected(self) -> None:
        rows = self.fixture.rows()[1:]
        self.assertTrue(any("missing" in problem for problem in self._problems(rows)))

    def test_cross_role_row_is_rejected(self) -> None:
        rows = self.fixture.rows()
        injected = dict(rows[0])
        injected["filename"] = "T9999_D00_S0000.mat"
        self.assertTrue(any("cross-role" in problem for problem in self._problems(rows + [injected])))

    def test_label_mismatch_is_rejected(self) -> None:
        rows = self.fixture.rows()
        rows[0]["label"] = 1 - rows[0]["label"]
        self.assertTrue(any("label mismatches" in problem for problem in self._problems(rows)))

    def test_target_code_mismatch_is_rejected(self) -> None:
        rows = self.fixture.rows()
        rows[0]["target_code"] = "T9999"
        self.assertTrue(any("target-code mismatches" in problem for problem in self._problems(rows)))

    def test_start_offset_mismatch_is_rejected(self) -> None:
        rows = self.fixture.rows()
        rows[0]["start"] = rows[0]["start"] + 1
        self.assertTrue(any("start-offset mismatches" in problem for problem in self._problems(rows)))

    def test_prediction_contradicting_its_margin_is_rejected(self) -> None:
        rows = self.fixture.rows()
        rows[0]["prediction"] = 1 - rows[0]["prediction"]
        self.assertTrue(
            any("margin > 0" in problem for problem in self._problems(rows))
        )

    def test_non_finite_margin_is_rejected(self) -> None:
        rows = self.fixture.rows()
        rows[0]["logit_margin"] = float("nan")
        self.assertTrue(any("non-finite" in problem for problem in self._problems(rows)))

    def test_index_position_permutation_is_required(self) -> None:
        rows = self.fixture.rows()
        rows[0]["index_position"] = 999
        self.assertTrue(
            any("dense permutation" in problem for problem in self._problems(rows))
        )


class FailFastTests(unittest.TestCase):
    """The diagnostic must return non-zero and report, never publish quietly."""

    def setUp(self) -> None:
        # The system temp area is not writable under the session sandbox and
        # ``mkdtemp`` directories cannot be extended here, so each fail-fast test
        # gets an explicitly named scratch directory inside the project.
        self.tmp = PROJECT_DIR.parent / "tmp" / "stage1diag-tests" / self._testMethodName
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_missing_score_file_fails_and_is_reported(self) -> None:
        original = diag.SEED_ROOT
        diag.SEED_ROOT = self.tmp / "empty_seed"
        try:
            code = diag.main(["--output-dir", str(self.tmp / "out"), "--quiet"])
        finally:
            diag.SEED_ROOT = original
        self.assertEqual(code, 1)
        payload = json.loads((self.tmp / "out" / "diagnostics_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "fail")
        self.assertTrue(any("missing score file" in problem for problem in payload["problems"]))

    def test_tampered_score_rows_fail_and_are_reported(self) -> None:
        original_loader = diag.load_score_file

        def tampered(path: Path):
            rows = original_loader(path)
            if path.name.endswith("_03.csv"):
                rows.append(dict(rows[0]))
            return rows

        diag.load_score_file = tampered
        try:
            code = diag.main(["--output-dir", str(self.tmp / "out"), "--quiet"])
        finally:
            diag.load_score_file = original_loader
        self.assertEqual(code, 1)
        payload = json.loads((self.tmp / "out" / "diagnostics_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["stage"], "score file integrity")
        self.assertTrue(any("duplicate keys" in problem for problem in payload["problems"]))

    def test_missing_val_window_fails_and_is_reported(self) -> None:
        original_loader = diag.load_score_file

        def tampered(path: Path):
            rows = original_loader(path)
            if path.name.endswith("_05.csv"):
                return rows[1:]
            return rows

        diag.load_score_file = tampered
        try:
            code = diag.main(["--output-dir", str(self.tmp / "out"), "--quiet"])
        finally:
            diag.load_score_file = original_loader
        self.assertEqual(code, 1)
        payload = json.loads((self.tmp / "out" / "diagnostics_summary.json").read_text(encoding="utf-8"))
        self.assertTrue(any("missing" in problem for problem in payload["problems"]))


class PublishedArtefactTests(unittest.TestCase):
    """Cheap regression guard on the published diagnostics on disk."""

    def test_published_diagnostics_report_pass_and_completed(self) -> None:
        summary = json.loads(
            (diag.DIAG_DIR / "diagnostics_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["metric_reconciliation"]["status"], "pass")
        self.assertEqual(summary["score_integrity"]["status"], "pass")
        self.assertLessEqual(summary["metric_reconciliation"]["max_abs_float_delta"], 1e-12)
        self.assertEqual(summary["problems"], [])

    def test_published_trajectory_has_twenty_rows_and_the_selected_epoch(self) -> None:
        with (diag.DIAG_DIR / "epoch_trajectory_by_condition.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 20)
        self.assertEqual({int(row["epoch"]) for row in rows}, {1, 2, 3, 4, 5})
        self.assertEqual({row["condition"] for row in rows}, set(diag.CONDITIONS))

    def test_selected_epoch_mean_matches_the_accepted_selection_json(self) -> None:
        selection = json.loads(
            (diag.SEED_ROOT / "selection.json").read_text(encoding="utf-8")
        )
        summary = json.loads(
            (diag.DIAG_DIR / "diagnostics_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(selection["selected_epoch"], diag.SELECTED_EPOCH)
        recomputed = summary["epoch_means"]["epoch_03"]["mean_balanced_accuracy"]
        self.assertAlmostEqual(recomputed, float(selection["selected_epoch_score"]), places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
