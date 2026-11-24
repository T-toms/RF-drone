#!/usr/bin/env python3
"""Failure cases and a read-through check for the heldout preparation outputs.

Two failure cases were chosen for the risk that actually exists in the new code:

1. ``IndexBoundaryTests`` – the frozen window rule must never emit a window that
   crosses a 10,000,000-sample acquisition block, and must never emit overlapping
   windows.  A regression here would silently mix samples from two possibly
   discontinuous acquisitions.
2. ``HoldoutIsolationTests`` – the heldout index must keep ``split=heldout`` and
   must never collide with the already-used train/val/test files, which would
   invalidate the frozen evaluation.

Plus a read-through check that the first, middle and last *indexed* window of real
heldout files load through the project's own dataset interface and are finite.
"""

from __future__ import annotations

import csv
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dronerfa_data import load_index, read_vector_slice  # noqa: E402

INDEX = SCRIPT_DIR / "outputs" / "heldout_preparation_20260917" / "heldout_window_index.csv"
OLD_INDEX = SCRIPT_DIR / "outputs" / "window_index.csv"
BLOCK_SIZE = 10_000_000
WINDOW_SIZE = 1_000_000


class IndexBoundaryTests(unittest.TestCase):
    """Only meaningful when the generated index is present."""

    @classmethod
    def setUpClass(cls) -> None:
        if not INDEX.is_file():
            raise unittest.SkipTest(f"heldout index not generated yet: {INDEX}")
        cls.rows = load_index(INDEX)
        cls.by_file: dict[str, list[dict]] = defaultdict(list)
        for row in cls.rows:
            cls.by_file[str(row["filename"])].append(row)

    def test_no_window_crosses_a_block_boundary(self) -> None:
        offenders = [
            f"{row['filename']}:{row['start']}+{row['length']}"
            for row in self.rows
            if int(row["start"]) // BLOCK_SIZE
            != (int(row["start"]) + int(row["length"]) - 1) // BLOCK_SIZE
        ]
        self.assertEqual(offenders, [], f"跨块窗口：{offenders[:5]}")

    def test_windows_do_not_overlap_and_are_aligned(self) -> None:
        for filename, rows in self.by_file.items():
            with self.subTest(filename=filename):
                ordered = sorted(rows, key=lambda row: int(row["start"]))
                spans: list[tuple[int, int]] = []
                for row in ordered:
                    start = int(row["start"])
                    length = int(row["length"])
                    self.assertEqual(start % WINDOW_SIZE, 0, f"{filename}:{start} 未对齐")
                    self.assertEqual(length, WINDOW_SIZE)
                    spans.append((start, start + length))
                for (_, previous_end), (next_start, _) in zip(spans, spans[1:]):
                    self.assertLessEqual(previous_end, next_start, f"{filename} 存在重叠窗")

    def test_windows_stay_inside_the_stored_samples(self) -> None:
        for row in self.rows:
            self.assertLessEqual(
                int(row["start"]) + int(row["length"]),
                int(row["source_samples"]),
                f"{row['filename']}:{row['start']} 超出文件长度",
            )

    def test_window_count_matches_blocks(self) -> None:
        for filename, rows in self.by_file.items():
            with self.subTest(filename=filename):
                blocks = int(rows[0]["source_samples"]) // BLOCK_SIZE
                self.assertEqual(len(rows), blocks * (BLOCK_SIZE // WINDOW_SIZE))


class HoldoutIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not INDEX.is_file():
            raise unittest.SkipTest(f"heldout index not generated yet: {INDEX}")
        cls.rows = load_index(INDEX)

    def test_split_is_heldout_only(self) -> None:
        splits = {str(row["split"]) for row in self.rows}
        self.assertEqual(splits, {"heldout"})

    def test_no_file_or_path_overlap_with_existing_splits(self) -> None:
        if not OLD_INDEX.is_file():
            self.skipTest("existing index absent")
        with OLD_INDEX.open("r", encoding="utf-8-sig", newline="") as handle:
            old_rows = list(csv.DictReader(handle))
        old_files = {row["filename"] for row in old_rows}
        old_paths = {str(Path(row["file_path"]).resolve()) for row in old_rows}
        new_files = {str(row["filename"]) for row in self.rows}
        new_paths = {str(Path(row["file_path"]).resolve()) for row in self.rows}
        self.assertEqual(new_files & old_files, set(), "heldout 文件名与旧集合重叠")
        self.assertEqual(new_paths & old_paths, set(), "heldout 路径与旧集合重叠")

    def test_labels_follow_target_code(self) -> None:
        for row in self.rows:
            expected = 0 if str(row["target_code"]) == "0000" else 1
            self.assertEqual(int(row["label"]), expected, f"{row['filename']} 标签错误")


class WindowReadThroughTests(unittest.TestCase):
    """Real read-back of the first, middle and last indexed window of every file."""

    @classmethod
    def setUpClass(cls) -> None:
        if not INDEX.is_file():
            raise unittest.SkipTest(f"heldout index not generated yet: {INDEX}")
        cls.rows = load_index(INDEX)

    def test_first_middle_last_window_reads_through_existing_interface(self) -> None:
        import h5py

        by_file: dict[str, list[dict]] = defaultdict(list)
        for row in self.rows:
            by_file[str(row["filename"])].append(row)
        checked = 0
        for filename, rows in sorted(by_file.items()):
            ordered = sorted(rows, key=lambda row: int(row["start"]))
            for label, row in (
                ("first", ordered[0]),
                ("middle", ordered[len(ordered) // 2]),
                ("last", ordered[-1]),
            ):
                with self.subTest(filename=filename, window=label):
                    with h5py.File(str(row["file_path"]), "r") as handle:
                        i_values = read_vector_slice(
                            handle["RF0_I"], int(row["start"]), int(row["length"])
                        )
                        q_values = read_vector_slice(
                            handle["RF0_Q"], int(row["start"]), int(row["length"])
                        )
                    self.assertEqual(i_values.shape, (WINDOW_SIZE,))
                    self.assertTrue(np.isfinite(i_values).all())
                    self.assertTrue(np.isfinite(q_values).all())
                    checked += 1
        self.assertEqual(checked, 36, "应检查 12 文件 × 首/中/末 三个窗")


if __name__ == "__main__":
    unittest.main(verbosity=2)
