import unittest

from evaluate_cascade12 import summarize


class Cascade12MetricsTests(unittest.TestCase):
    def test_mutually_exclusive_error_decomposition(self):
        rows = [
            # A background false alarm.
            dict(role="development_test_exposed", condition="clean", target_code="T0000", filename="bg.mat",
                 truth_cascade=0, truth_stage2=None, stage1_prediction=1, stage2_prediction=0, cascade_prediction=1),
            # One drone missed by Stage1.
            dict(role="development_test_exposed", condition="clean", target_code="T0010", filename="drone.mat",
                 truth_cascade=2, truth_stage2=1, stage1_prediction=0, stage2_prediction=1, cascade_prediction=0),
            # One drone passed but misclassified by Stage2.
            dict(role="development_test_exposed", condition="clean", target_code="T0010", filename="drone.mat",
                 truth_cascade=2, truth_stage2=1, stage1_prediction=1, stage2_prediction=0, cascade_prediction=1),
            # One correctly classified drone.
            dict(role="development_test_exposed", condition="clean", target_code="T0010", filename="drone.mat",
                 truth_cascade=2, truth_stage2=1, stage1_prediction=1, stage2_prediction=1, cascade_prediction=2),
        ]
        summary, per_class, matrices = summarize(rows)
        self.assertEqual(summary[0]["background_false_positive"], 1)
        self.assertEqual(summary[0]["stage1_drone_missed"], 1)
        self.assertEqual(summary[0]["stage2_wrong_after_gate"], 1)
        self.assertEqual(summary[0]["total_errors"], 3)
        self.assertAlmostEqual(summary[0]["cascade_accuracy"], 0.25)
        self.assertEqual(matrices["development_test_exposed/clean"][0][1], 1)
        self.assertEqual(len(per_class), 1)


if __name__ == "__main__": unittest.main()
