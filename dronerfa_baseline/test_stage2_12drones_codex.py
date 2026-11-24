import unittest

from run_stage2_12drones import metrics, order_for_epoch
from stage2_12class_data import CODE_TO_LABEL, FULL_COUNTS, ROLE_COUNTS, load_contract_rows


class Stage2ContractTests(unittest.TestCase):
    def test_frozen_view_counts_and_positions(self):
        full, drones = load_contract_rows()
        self.assertEqual(len(full), 15790)
        for role, count in ROLE_COUNTS.items():
            role_rows = [r for r in drones if r["split"] == role]
            self.assertEqual(len(role_rows), count)
            self.assertEqual(len({r["row_index"] for r in role_rows}), count)
            self.assertTrue(all(0 <= int(r["full_split_position"]) < FULL_COUNTS[role] for r in role_rows))
            self.assertTrue(all(int(r["stage2_label"]) == CODE_TO_LABEL[r["target_code"]] for r in role_rows))
            self.assertTrue(all(int(r["cascade_label"]) == int(r["stage2_label"]) + 1 for r in role_rows))

    def test_shuffle_is_full_and_reproducible(self):
        order = order_for_epoch(1, 8990)
        self.assertEqual(len(set(order)), 8990)
        self.assertEqual(order, order_for_epoch(1, 8990))
        self.assertNotEqual(order, order_for_epoch(2, 8990))

    def test_file_class_equal_selection_does_not_double_count_extra_files(self):
        rows = []
        for code, label in CODE_TO_LABEL.items():
            # Class T0010 has two files, with one correct and one incorrect.
            files = ["a", "b"] if code == "T0010" else ["a"]
            for name in files:
                rows.append({"target_code":code,"filename":f"{code}_{name}","label":label,
                             "prediction":label if name == "a" else (label+1)%12})
        result = metrics(rows)
        self.assertAlmostEqual(result["per_class_file_equal_recall"]["T0010"], 0.5)
        self.assertAlmostEqual(result["class_file_equal_recall"], 11.5/12)


if __name__ == "__main__": unittest.main()
