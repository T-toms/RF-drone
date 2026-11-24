"""Regression tests for CASCADE3-NEWSPLIT-C evaluation and verification code.

Pure-logic tests plus a few read-only checks against the frozen artefacts on disk.
No model is trained, no GPU work is required and nothing under ``outputs/`` is
written by this module.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import run_cascade3_newsplit_evaluation as evaluator
import verify_cascade3_newsplit_evaluation as verifier


def synthetic_record(
    filename: str,
    target_code: str,
    true_class: int,
    final_prediction: int,
    stage2_executed: bool,
) -> dict:
    return {
        "filename": filename,
        "target_code": target_code,
        "target_name": evaluator.CLASS_NAMES[true_class],
        "window_index": 0,
        "true_class": true_class,
        "final_prediction": final_prediction,
        "stage2_executed": stage2_executed,
    }


def row_for(
    position: int,
    true_class: int,
    filename: str,
    *,
    window_index: int = 0,
    digest: str = "d",
) -> dict:
    return {
        "dataset_index_position": position,
        "source_role": evaluator.TEST_ROLE,
        "split": evaluator.TEST_ROLE,
        "metrics_scope": "legacy_exposed_control",
        "file_path": str(evaluator.DATA_ROOT / filename),
        "filename": filename,
        "target_code": {0: "T0000", 1: "T0010", 2: "T0101"}[true_class],
        "target_name": evaluator.CLASS_NAMES[true_class],
        "class_index": true_class,
        "class_name": evaluator.CLASS_NAMES[true_class],
        "stage1_label": 0 if true_class == 0 else 1,
        "true_class": true_class,
        "true_class_name": evaluator.CLASS_NAMES[true_class],
        "block_index": 0,
        "window_index": window_index,
        "start": window_index * evaluator.WINDOW_SAMPLES,
        "length": evaluator.WINDOW_SAMPLES,
        "source_samples": 100 * evaluator.WINDOW_SAMPLES,
        "file_sha256": "f" * 64,
        "stage1_feature_digest": digest,
    }


class FrozenTestSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = evaluator.frozen_test_rows()

    def test_composition_matches_the_frozen_index(self) -> None:
        composition = evaluator.test_split_identity(self.rows)
        self.assertEqual(composition["windows"], 890)
        self.assertEqual(composition["files"], 6)
        self.assertEqual(composition["target_counts"], evaluator.EXPECTED_TARGET_COUNTS)
        self.assertEqual(composition["class_counts"], {0: 280, 1: 300, 2: 310})
        self.assertEqual(evaluator.check_expected_composition(composition), [])

    def test_every_row_is_positioned_and_role_tagged(self) -> None:
        self.assertEqual(
            [int(row["dataset_index_position"]) for row in self.rows], list(range(890))
        )
        for row in self.rows:
            self.assertEqual(str(row["source_role"]), evaluator.TEST_ROLE)
            self.assertEqual(str(row["split"]), evaluator.TEST_ROLE)
            self.assertEqual(int(row["length"]), evaluator.WINDOW_SAMPLES)
            self.assertEqual(
                int(row["start"]), int(row["window_index"]) * evaluator.WINDOW_SAMPLES
            )
            expected = 0 if evaluator.TARGET_CODE_TO_CLASS[str(row["target_code"])] == 0 else 1
            self.assertEqual(int(row["stage1_label"]), expected)

    def test_six_frozen_files_are_the_exposed_ones(self) -> None:
        names = sorted({str(row["filename"]) for row in self.rows})
        self.assertEqual(
            names,
            [
                "T0000_D00_S0110.mat",
                "T0000_D00_S0111.mat",
                "T0010_D00_S0110.mat",
                "T0010_D00_S0111.mat",
                "T0101_D00_S0110.mat",
                "T0101_D00_S0111.mat",
            ],
        )

    def test_wrong_composition_is_rejected(self) -> None:
        composition = evaluator.test_split_identity(self.rows)
        composition["windows"] = 889
        self.assertTrue(evaluator.check_expected_composition(composition))
        composition = evaluator.test_split_identity(self.rows)
        composition["target_counts"]["T0000"] = 1
        self.assertTrue(evaluator.check_expected_composition(composition))


class DecisionRuleTests(unittest.TestCase):
    EPOCHS = {"stage1_epoch": 3, "stage2_epoch": 3}

    def test_background_row_skips_stage_two(self) -> None:
        row = row_for(0, 0, "T0000_D00_S0110.mat")
        record = evaluator.build_prediction_row(row, 2026, self.EPOCHS, [3.0, 1.0], None, None, "aa", None)
        self.assertEqual(record["stage1_margin"], -2.0)
        self.assertEqual(record["stage1_prediction"], 0)
        self.assertFalse(record["stage2_executed"])
        self.assertEqual(record["final_prediction"], 0)
        self.assertIsNone(record["stage2_margin"])
        self.assertIsNone(record["stage2_batch_id"])

    def test_positive_stage_one_margin_moves_to_stage_two(self) -> None:
        row = row_for(0, 2, "T0101_D00_S0110.mat")
        record = evaluator.build_prediction_row(row, 2026, self.EPOCHS, [-4.0, 5.0], [-6.0, 7.0], 3, "aa", "aa")
        self.assertEqual(record["stage1_prediction"], 1)
        self.assertEqual(record["stage1_decision"], "Drone")
        self.assertTrue(record["stage2_executed"])
        self.assertEqual(record["stage2_margin"], 13.0)
        self.assertEqual(record["final_prediction"], 2)
        self.assertEqual(record["final_class_name"], "DJI Air 2S")

    def test_negative_stage_two_margin_is_phantom(self) -> None:
        row = row_for(0, 1, "T0010_D00_S0110.mat")
        record = evaluator.build_prediction_row(row, 2026, self.EPOCHS, [-4.0, 5.0], [7.0, -6.0], 0, "aa", "aa")
        self.assertEqual(record["final_prediction"], 1)
        self.assertEqual(record["final_class_name"], "DJI Phantom 4 Pro")

    def test_zero_margins_fall_back_to_background_and_phantom(self) -> None:
        row = row_for(0, 0, "T0000_D00_S0110.mat")
        skipped = evaluator.build_prediction_row(row, 2026, self.EPOCHS, [2.0, 2.0], None, None, "aa", None)
        self.assertFalse(skipped["stage2_executed"])
        self.assertEqual(skipped["final_prediction"], 0)
        drone = row_for(0, 1, "T0010_D00_S0110.mat")
        tied = evaluator.build_prediction_row(drone, 2026, self.EPOCHS, [-1.0, 1.0], [3.0, 3.0], 0, "aa", "aa")
        self.assertTrue(tied["stage2_executed"])
        self.assertEqual(tied["final_prediction"], 1)

    def test_missing_stage_two_logits_for_executed_row_is_rejected(self) -> None:
        row = row_for(0, 1, "T0010_D00_S0110.mat")
        with self.assertRaises(ValueError):
            evaluator.build_prediction_row(row, 2026, self.EPOCHS, [-1.0, 1.0], None, None, "aa", None)

    def test_stage_two_logits_on_a_skipped_row_are_rejected(self) -> None:
        row = row_for(0, 0, "T0000_D00_S0110.mat")
        with self.assertRaises(ValueError):
            evaluator.build_prediction_row(row, 2026, self.EPOCHS, [1.0, -1.0], [-1.0, 1.0], 0, "aa", "aa")

    def test_stage_two_must_consume_the_stage_one_tensor(self) -> None:
        row = row_for(0, 1, "T0010_D00_S0110.mat")
        with self.assertRaises(ValueError):
            evaluator.build_prediction_row(row, 2026, self.EPOCHS, [-1.0, 1.0], [-1.0, 1.0], 0, "aa", "bb")


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = (
            [synthetic_record("A.mat", "T0000", 0, 0, False) for _ in range(3)]
            + [synthetic_record("A.mat", "T0000", 0, 1, True)]
            + [synthetic_record("B.mat", "T0010", 1, 1, True) for _ in range(3)]
            + [synthetic_record("B.mat", "T0010", 1, 2, True)]
            + [synthetic_record("C.mat", "T0101", 2, 2, True) for _ in range(2)]
            + [synthetic_record("C.mat", "T0101", 2, 1, True)]
            + [synthetic_record("C.mat", "T0101", 2, 0, False)]
        )

    def test_confusion_matrix(self) -> None:
        matrix = evaluator.confusion_matrix(self.records)
        # rows = truth (Background / Phantom / Air 2S), columns = prediction
        self.assertEqual(
            matrix.tolist(),
            [[3, 1, 0], [0, 3, 1], [1, 1, 2]],
        )

    def test_pooled_metrics(self) -> None:
        metrics = evaluator.pooled_metrics(evaluator.confusion_matrix(self.records))
        self.assertEqual(metrics["windows"], 12)
        self.assertAlmostEqual(metrics["pooled_accuracy"], 8 / 12)
        self.assertAlmostEqual(metrics["recall_background"], 0.75)
        self.assertAlmostEqual(metrics["recall_phantom"], 0.75)
        self.assertAlmostEqual(metrics["recall_air2s"], 0.5)
        self.assertAlmostEqual(metrics["precision_background"], 0.75)
        self.assertAlmostEqual(metrics["precision_phantom"], 0.6)
        self.assertAlmostEqual(metrics["precision_air2s"], 2 / 3)

    def test_file_equal_metrics(self) -> None:
        metrics = evaluator.file_equal_metrics(self.records)
        self.assertEqual(metrics["files"], 3)
        self.assertAlmostEqual(metrics["file_equal_background_recall"], 0.75)
        self.assertAlmostEqual(metrics["file_equal_phantom_recall"], 0.75)
        self.assertAlmostEqual(metrics["file_equal_air2s_recall"], 0.5)
        self.assertAlmostEqual(metrics["file_equal_accuracy"], (0.75 + 0.75 + 0.5) / 3)
        self.assertEqual(metrics["file_equal_classes_used"], 3)

    def test_stage_one_rates(self) -> None:
        per_file = evaluator.file_equal_metrics(self.records)["per_file"]
        rates = evaluator.stage1_rates(self.records, per_file)
        self.assertEqual(rates["stage1_background_false_alarms"], 1)
        self.assertAlmostEqual(rates["stage1_background_false_alarm_rate"], 0.25)
        self.assertEqual(rates["stage1_drone_misses"], 1)
        self.assertAlmostEqual(rates["stage1_drone_miss_rate"], 0.125)
        self.assertAlmostEqual(rates["stage1_drone_miss_rate_file_equal"], 0.125)
        self.assertEqual(rates["stage1_drones_passed"], 7)

    def test_stage_two_conditional_accuracy(self) -> None:
        metrics = evaluator.stage2_conditional_metrics(self.records)
        self.assertEqual(metrics["stage2_conditional_windows"], 7)
        self.assertEqual(metrics["stage2_conditional_correct"], 5)
        self.assertAlmostEqual(metrics["stage2_conditional_accuracy"], 5 / 7)

    def test_error_decomposition(self) -> None:
        errors = evaluator.error_decomposition(self.records)
        self.assertEqual(errors["error_background_to_phantom"], 1)
        self.assertEqual(errors["error_background_to_air2s"], 0)
        self.assertEqual(errors["error_phantom_to_background"], 0)
        self.assertEqual(errors["error_air2s_to_background"], 1)
        self.assertEqual(errors["error_phantom_to_air2s"], 1)
        self.assertEqual(errors["error_air2s_to_phantom"], 1)
        self.assertEqual(errors["error_windows_total"], 4)

    def test_metric_record_carries_every_declared_column(self) -> None:
        record = evaluator.metric_record(self.records)
        missing = [column for column in evaluator.METRIC_COLUMNS if column not in record]
        self.assertEqual(missing, [])
        self.assertEqual(record["windows"], 12)
        self.assertEqual(record["training_seed"], 2026)
        self.assertEqual(record["stage1_epoch"], 3)


class FeaturePipelineTests(unittest.TestCase):
    class StubDataset:
        """Minimal stand-in for the frozen window reader (no HDF5, no G: drive)."""

        def __init__(self, rows, length: int = 32768, seed: int = 7) -> None:
            self.rows = rows
            self.length = length
            self.seed = seed
            self.closed = False

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            rng = np.random.default_rng(self.seed + index)
            values = rng.standard_normal((2, self.length)).astype(np.float32)
            return torch.from_numpy(values), int(self.rows[index]["stage1_label"])

        def close(self) -> None:
            self.closed = True

    def make_rows(self, count: int) -> list[dict]:
        rows = []
        for position in range(count):
            true_class = position % 3
            row = row_for(position, true_class, "X.mat", window_index=position)
            row["length"] = 32768
            rows.append(row)
        return rows

    def test_features_have_the_frozen_shape_and_digests(self) -> None:
        rows = self.make_rows(4)
        dataset = self.StubDataset(rows)
        frontend = evaluator.SpectrogramFrontEnd(
            n_fft=evaluator.N_FFT, hop_length=evaluator.HOP_LENGTH, image_size=evaluator.IMAGE_SIZE
        )
        built = evaluator.build_features(dataset, rows, torch.device("cpu"), frontend)
        self.assertEqual(
            built["features"].shape, (4, 1, evaluator.IMAGE_SIZE, evaluator.IMAGE_SIZE)
        )
        self.assertEqual(built["features"].dtype, np.float32)
        self.assertTrue(np.isfinite(built["features"]).all())
        self.assertEqual(len(built["window_digests"]), 4 * 32)
        self.assertTrue(dataset.closed)

    def test_features_are_deterministic(self) -> None:
        rows = self.make_rows(4)
        frontend = evaluator.SpectrogramFrontEnd(
            n_fft=evaluator.N_FFT, hop_length=evaluator.HOP_LENGTH, image_size=evaluator.IMAGE_SIZE
        )
        first = evaluator.build_features(self.StubDataset(rows), rows, torch.device("cpu"), frontend)
        second = evaluator.build_features(self.StubDataset(rows), rows, torch.device("cpu"), frontend)
        self.assertEqual(first["input_probe_sha256"], second["input_probe_sha256"])
        self.assertEqual(first["window_digests"], second["window_digests"])

    def test_label_mismatch_is_rejected(self) -> None:
        rows = self.make_rows(4)
        rows[1]["true_class"] = 0
        rows[1]["stage1_label"] = 1  # stub returns 1 while the row now claims background
        frontend = evaluator.SpectrogramFrontEnd(
            n_fft=evaluator.N_FFT, hop_length=evaluator.HOP_LENGTH, image_size=evaluator.IMAGE_SIZE
        )
        with self.assertRaises(RuntimeError):
            evaluator.build_features(self.StubDataset(rows), rows, torch.device("cpu"), frontend)


class VerifierHelperTests(unittest.TestCase):
    def test_coverage_helper_detects_missing_and_duplicate_windows(self) -> None:
        rows = [
            {"dataset_index_position": position}
            for position in range(verifier.EXPECTED_WINDOWS)
        ]
        self.assertEqual(verifier.coverage_problems(rows), [])
        self.assertTrue(verifier.coverage_problems(rows[:-1]))
        duplicated = rows[:-1] + [{"dataset_index_position": 0}]
        self.assertTrue(verifier.coverage_problems(duplicated))

    def test_decision_helper_detects_a_flipped_class(self) -> None:
        row = {
            "filename": "T0010_D00_S0110.mat",
            "window_index": 1,
            "stage1_logit_background": -2.0,
            "stage1_logit_drone": 3.0,
            "stage1_margin": 5.0,
            "stage1_prediction": 1,
            "stage1_decision": "Drone",
            "stage2_executed": True,
            "stage2_batch_id": 0,
            "stage2_logit_phantom": -4.0,
            "stage2_logit_air2s": 1.0,
            "stage2_margin": 5.0,
            "final_prediction": 2,
            "shared_feature_tensor": True,
            "stage1_feature_digest": "aa",
            "stage2_feature_digest": "aa",
        }
        self.assertEqual(verifier.decision_problems([row]), [])
        flipped = dict(row, final_prediction=1)
        self.assertTrue(verifier.decision_problems([flipped]))
        unshared = dict(row, stage2_feature_digest="bb")
        self.assertTrue(verifier.decision_problems([unshared]))
        hidden = dict(row, stage2_executed=False, final_prediction=2)
        self.assertTrue(verifier.decision_problems([hidden]))

    def test_metric_helper_detects_a_shifted_value(self) -> None:
        records = [
            synthetic_record("A.mat", "T0000", 0, 0, False),
            synthetic_record("B.mat", "T0010", 1, 1, True),
            synthetic_record("C.mat", "T0101", 2, 2, True),
        ]
        own = verifier.own_summary(records)
        published = {
            "windows": 3,
            "files": 3,
            "pooled_accuracy": own["pooled_accuracy"],
            "macro_recall": own["macro_recall"],
            "balanced_accuracy": own["macro_recall"],
            "macro_precision": own["macro_precision"],
            "file_equal_accuracy": own["file_equal"]["file_equal_accuracy"],
            "file_equal_background_recall": 1.0,
            "file_equal_phantom_recall": 1.0,
            "file_equal_air2s_recall": 1.0,
            "stage1_background_false_alarms": 0,
            "stage1_background_false_alarm_rate": 0.0,
            "stage1_drone_misses": 0,
            "stage1_drone_miss_rate": 0.0,
            "stage2_conditional_windows": 2,
            "stage2_conditional_accuracy": 1.0,
            "error_windows_total": 0,
            "recall_background": 1.0,
            "recall_phantom": 1.0,
            "recall_air2s": 1.0,
            "precision_background": 1.0,
            "precision_phantom": 1.0,
            "precision_air2s": 1.0,
        }
        self.assertEqual(verifier.metric_problems(own, published, 3), [])
        forged = dict(published, pooled_accuracy=0.5)
        self.assertTrue(verifier.metric_problems(own, forged, 3))
        short = dict(published, windows=2)
        self.assertTrue(verifier.metric_problems(own, short, 3))


class PublishedArtefactTests(unittest.TestCase):
    """Read-only checks against the CASCADE3-NEWSPLIT-C run on disk (if present)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = evaluator.OUTPUT_ROOT
        cls.predictions_path = cls.root / "window_predictions.csv"
        if not cls.predictions_path.is_file():
            raise unittest.SkipTest("CASCADE3-NEWSPLIT-C has not been run yet")

    def test_predictions_match_the_frozen_decision_rules(self) -> None:
        rows = verifier.own_prediction_rows(self.predictions_path)
        self.assertEqual(len(rows), evaluator.EXPECTED_TEST_WINDOWS)
        self.assertEqual(verifier.coverage_problems(rows), [])
        self.assertEqual(verifier.decision_problems(rows), [])

    def test_published_metrics_match_an_independent_recomputation(self) -> None:
        rows = verifier.own_prediction_rows(self.predictions_path)
        own = verifier.own_summary(rows)
        published = verifier.read_json(self.root / "metrics_summary.json")["metrics"]
        self.assertEqual(verifier.metric_problems(own, published, len(rows)), [])
        self.assertEqual(
            verifier.read_json(self.root / "confusion_matrix.json")["matrix"],
            [[int(value) for value in row] for row in own["matrix"]],
        )

    def test_negative_controls_pass_on_the_published_run(self) -> None:
        rows = verifier.own_prediction_rows(self.predictions_path)
        controls = verifier.negative_controls(rows, self.root)
        self.assertEqual(controls["passed"], controls["total"])

    def test_predictions_only_cover_development_test_exposed_rows(self) -> None:
        rows = verifier.own_prediction_rows(self.predictions_path)
        self.assertEqual({str(row["source_role"]) for row in rows}, {evaluator.TEST_ROLE})
        self.assertLessEqual(
            {str(row["filename"]) for row in rows},
            {str(row["filename"]) for row in evaluator.frozen_test_rows()},
        )


if __name__ == "__main__":
    unittest.main()
