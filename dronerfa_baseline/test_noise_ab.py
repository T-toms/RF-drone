"""Checks for experimental selection and per-target evaluation semantics."""
import unittest

import torch

from run_noise_ab import (evaluate_cached, resolve_validation_seed, select_epochs,
                          summarize_predictions, validate_reference_cache)


class NoiseABTests(unittest.TestCase):
    def test_validation_noise_can_be_fixed_across_training_seeds(self):
        self.assertEqual(resolve_validation_seed(2026, None), 12026)
        self.assertEqual(resolve_validation_seed(2027, None), 12027)
        for seed in [2026, 2027, 2028]:
            self.assertEqual(resolve_validation_seed(seed, 12026), 12026)

    def test_reference_check_rejects_changed_features_or_indices(self):
        row = dict(condition="clean", samples=2, full_dataset_indices=[0, 1], feature_sha256="abc")
        validate_reference_cache([row], [dict(row)])
        for change in [dict(feature_sha256="xyz"), dict(full_dataset_indices=[1, 0]), dict(samples=3)]:
            with self.assertRaises(ValueError):
                validate_reference_cache([dict(row, **change)], [row])

    def test_selection_keeps_earliest_tie_and_uses_declared_metric(self):
        rows = [
            dict(epoch=1, clean_balanced_accuracy=0.9, mean_balanced_accuracy=0.8),
            dict(epoch=2, clean_balanced_accuracy=1.0, mean_balanced_accuracy=0.85),
            dict(epoch=3, clean_balanced_accuracy=1.0, mean_balanced_accuracy=0.95),
            dict(epoch=4, clean_balanced_accuracy=1.0, mean_balanced_accuracy=0.95),
        ]
        self.assertEqual(select_epochs(rows), {"clean_only": 2, "multi_noise": 3})

    def test_background_errors_and_target_recalls_are_separate(self):
        rows = [
            dict(label=0, prediction=0, logit_margin=-3, target_code="0000"),
            dict(label=0, prediction=1, logit_margin=1, target_code="0000"),
            dict(label=1, prediction=1, logit_margin=3, target_code="0010"),
            dict(label=1, prediction=0, logit_margin=-1, target_code="0101"),
        ]
        m = summarize_predictions(rows)
        self.assertEqual((m["tn"], m["fp"], m["fn"], m["tp"]), (1, 1, 1, 1))
        self.assertEqual(m["balanced_accuracy"], 0.5)
        self.assertEqual(m["fpr"], 0.5)
        self.assertEqual(m["phantom_recall"], 1)
        self.assertEqual(m["air2s_recall"], 0)

    def test_evaluation_preserves_source_order_and_zero_margin_is_background(self):
        logits = torch.tensor([[2., 2.], [0., 3.], [3., 1.]])
        labels = torch.tensor([0, 1, 1])
        rows = [dict(file_path="source", start=i * 10, target_code=code, label=int(labels[i]))
                for i, code in enumerate(["0000", "0010", "0101"])]
        metrics, records = evaluate_cached(
            torch.nn.Identity(), {"clean": (logits, labels, rows)},
            torch.nn.CrossEntropyLoss(), torch.device("cpu"), False, 2, 2026,
        )
        self.assertEqual([r["prediction"] for r in records], [0, 1, 0])
        self.assertEqual([r["start"] for r in records], [0, 10, 20])
        self.assertEqual(metrics[0]["balanced_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
