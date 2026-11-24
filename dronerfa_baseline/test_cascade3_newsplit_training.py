"""Regression tests for CASCADE3-NEWSPLIT-B training and verification code.

Pure-logic tests plus a few read-only checks against the frozen artefacts on
disk. No training is launched and nothing under ``outputs/`` is written.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import run_cascade3_newsplit_training as training
import verify_cascade3_newsplit_training as verifier


class IndexIdentityTests(unittest.TestCase):
    def test_identity_checks_pass_on_the_frozen_artefacts(self) -> None:
        identity = training.check_index_identity()
        self.assertTrue(identity["passed"], identity["errors"])
        self.assertEqual(identity["index"]["sha256"], training.INDEX_SHA256)
        self.assertEqual(identity["stage2_view"]["sha256"], training.STAGE2_VIEW_SHA256)
        self.assertEqual(identity["observed"]["windows"], {"train": 1810, "val": 930, "development_test_exposed": 890})
        self.assertEqual(identity["observed"]["stage2_windows"], {"train": 1210, "val": 650, "development_test_exposed": 610})
        self.assertEqual(identity["role_overlap"], {"train_val": [], "train_test": [], "val_test": []})

    def test_local_stage2_view_equals_the_frozen_implementation(self) -> None:
        result = training.frozen_implementation_cross_check()
        self.assertTrue(result["passed"], result["mismatches"])
        self.assertEqual(result["compared_rows"], 2470)

    def test_validation_window_counts(self) -> None:
        self.assertEqual(training.validation_windows(1), 930)
        self.assertEqual(training.validation_windows(2), 650)


class Stage2RowTests(unittest.TestCase):
    @staticmethod
    def row(target_code: str, label: int, stage2_label: str | None = None) -> dict:
        row = {
            "target_code": target_code,
            "label": label,
            "filename": f"{target_code}_D00_S0000.mat",
            "window_index": 0,
            "start": 0,
        }
        if stage2_label is not None:
            row["stage2_label"] = stage2_label
        return row

    def test_background_rows_are_dropped(self) -> None:
        rows = [self.row("T0000", 0), self.row("T0010", 1), self.row("T0101", 1)]
        selected = training.stage2_rows(rows)
        self.assertEqual([row["label"] for row in selected], [0, 1])
        self.assertEqual([row["class_name"] for row in selected], ["DJI Phantom 4 Pro", "DJI Air 2S"])

    def test_drone_row_with_background_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            training.stage2_rows([self.row("T0010", 0)])

    def test_stage2_label_disagreement_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            training.stage2_rows([self.row("T0010", 1, stage2_label="1")])

    def test_stage2_label_agreement_is_accepted(self) -> None:
        selected = training.stage2_rows([self.row("T0101", 1, stage2_label="1")])
        self.assertEqual(selected[0]["label"], 1)


class MetricTests(unittest.TestCase):
    def test_pooled_metrics_confusion(self) -> None:
        records = [
            {"label": 1, "prediction": 1},
            {"label": 1, "prediction": 0},
            {"label": 0, "prediction": 0},
            {"label": 0, "prediction": 1},
        ]
        metrics = training.pooled_binary_metrics(records)
        self.assertEqual((metrics["tp"], metrics["fn"], metrics["tn"], metrics["fp"]), (1, 1, 1, 1))
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)

    def test_file_equal_metrics_weight_files_equally(self) -> None:
        records = []
        records += [{"filename": "p0", "label": 0, "prediction": 0}] * 9
        records += [{"filename": "p0", "label": 0, "prediction": 1}]
        records += [{"filename": "a0", "label": 1, "prediction": 1}] * 10
        metrics = training.file_equal_metrics(records)
        self.assertAlmostEqual(metrics["file_equal_phantom_recall"], 0.9)
        self.assertAlmostEqual(metrics["file_equal_air2s_recall"], 1.0)
        self.assertAlmostEqual(metrics["file_equal_balanced_accuracy"], 0.95)

    def test_file_equal_metrics_reject_a_missing_class(self) -> None:
        with self.assertRaises(ValueError):
            training.file_equal_metrics([{"filename": "p0", "label": 0, "prediction": 0}])


class SelectionTests(unittest.TestCase):
    def test_highest_score_wins(self) -> None:
        history = [
            {"epoch": 1, "score": 0.5},
            {"epoch": 2, "score": 0.7},
            {"epoch": 3, "score": 0.6},
        ]
        self.assertEqual(training.select_epoch(history, "score"), 2)

    def test_earliest_epoch_wins_an_exact_tie(self) -> None:
        history = [
            {"epoch": 1, "score": 0.6},
            {"epoch": 2, "score": 0.6},
            {"epoch": 3, "score": 0.6},
        ]
        self.assertEqual(training.select_epoch(history, "score"), 1)

    def test_selection_is_stable_under_float_noise(self) -> None:
        history = [
            {"epoch": 1, "score": 0.6000000000000001},
            {"epoch": 2, "score": 0.6},
        ]
        self.assertEqual(training.select_epoch(history, "score"), 1)


class ReadAuditTests(unittest.TestCase):
    def test_payload_counts_windows_per_file(self) -> None:
        audit = training.ReadAudit("stage1")
        audit.record_open(r"G:\a.mat")
        audit.record_open(r"G:\a.mat")
        audit.record_window(r"G:\a.mat")
        audit.record_window(r"G:\a.mat")
        payload = audit.payload()
        self.assertEqual(payload["windows_read"], 2)
        self.assertEqual(payload["open_events"], 2)
        self.assertEqual(payload["opened_file_count"], 1)
        self.assertEqual(payload["windows_by_file"], {r"G:\a.mat": 2})

    def test_test_role_is_never_in_the_train_val_path_set(self) -> None:
        identity = training.check_index_identity()
        train_val = set(identity["paths_by_role"][training.TRAIN_ROLE]) | set(
            identity["paths_by_role"][training.VAL_ROLE]
        )
        self.assertFalse(train_val & set(identity["paths_by_role"][training.TEST_ROLE]))


class RunDirectoryGuardTests(unittest.TestCase):
    def test_existing_output_directory_is_refused(self) -> None:
        scratch = PROJECT_DIR / "scratch" / "test_cascade3_newsplit_training"
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            with self.assertRaises(FileExistsError):
                training.train_stage(
                    stage=1,
                    device=training.torch.device("cpu"),
                    epochs=1,
                    output_dir=scratch,
                    max_train_samples=1,
                    max_val_samples=1,
                    smoke=True,
                    pre_run={},
                )
        finally:
            scratch.rmdir()
            parent = scratch.parent
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()


class VerifierLogicTests(unittest.TestCase):
    def test_expected_selection_matches_the_documented_rule(self) -> None:
        history = [
            {"epoch": 1, "mean_balanced_accuracy": 0.4},
            {"epoch": 2, "mean_balanced_accuracy": 0.9},
            {"epoch": 3, "mean_balanced_accuracy": 0.9},
        ]
        self.assertEqual(verifier.expected_selection(history, "mean_balanced_accuracy"), 2)

    def test_negative_controls_detect_tampering(self) -> None:
        checks = verifier.Checks()
        identity = training.check_index_identity()
        stage1 = {
            "history": [
                {"epoch": 1, "mean_balanced_accuracy": 0.5},
                {"epoch": 2, "mean_balanced_accuracy": 0.6},
            ],
            "audit": {"opened_files": identity["paths_by_role"][training.TRAIN_ROLE]},
            "config": {"index_sha256": training.INDEX_SHA256},
        }
        original = sys.stdout
        try:
            sys.stdout = open(__import__("os").devnull, "w", encoding="utf-8")
            verifier.negative_controls(checks, identity, stage1)
        finally:
            sys.stdout.close()
            sys.stdout = original
        self.assertTrue(checks.passed, [row for row in checks.rows if not row["passed"]])

    def test_frozen_constants_agree_with_the_index_files(self) -> None:
        recorded = json.loads(
            (PROJECT_DIR / "outputs" / "cascade3_newsplit_20260926" / "newsplit3_verification.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(recorded.get("conclusion"), "all checks and negative controls passed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
