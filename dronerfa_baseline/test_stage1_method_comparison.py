"""Checks for BASE2 (`run_classical_baselines.py`).

Two ways to run it:

    python test_stage1_method_comparison.py                 # unit tests only
    python test_stage1_method_comparison.py --smoke         # + limited-window smoke run
    python test_stage1_method_comparison.py --verify-outputs # + independent CSV recompute

Every expected value here is recomputed from the raw prediction CSV text; the
module under test never supplies an expected metric value to these checks.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import itertools
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any, Iterator

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np

import run_classical_baselines as rcb

BASELINE_DIR = rcb.BASELINE_DIR
OLD_INDEX = rcb.DEFAULT_OLD_INDEX
HELDOUT_INDEX = rcb.DEFAULT_HELDOUT_INDEX
PYTHON = sys.executable
# Scratch space stays inside the project so the checks also run under a
# workspace-confined sandbox that denies writes to the system temp directory.
SCRATCH_ROOT = BASELINE_DIR.parent / "_test_scratch"
_SCRATCH_COUNTER = itertools.count()


@contextlib.contextmanager
def scratch_directory(prefix: str) -> Iterator[Path]:
    """Deterministic scratch directory inside the project.

    ``tempfile.mkdtemp`` is avoided on purpose: under a workspace-confined sandbox
    the directories it creates are write-protected, while directories created with
    an explicit ``mkdir`` are usable.
    """
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    path = SCRATCH_ROOT / f"{prefix}{os.getpid()}_{next(_SCRATCH_COUNTER)}"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# independent helpers (deliberately not importing the module's own metric code)
# ---------------------------------------------------------------------------
def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


_TEE_STREAMS: list[Any] = []


def attach_tee(paths: list[Path]) -> None:
    """Mirror this process's stdout/stderr into files (the sandbox blocks pipes)."""
    for stream, path in zip((sys.stdout, sys.stderr), paths):
        handle = path.open("w", encoding="utf-8")
        _TEE_STREAMS.append(handle)

        class _Tee:
            def __init__(self, original: Any, mirror: Any) -> None:
                self._original = original
                self._mirror = mirror

            def write(self, text: str) -> int:
                self._mirror.write(text)
                self._mirror.flush()
                return self._original.write(text)

            def flush(self) -> None:
                self._mirror.flush()
                self._original.flush()

        if stream is sys.stdout:
            sys.stdout = _Tee(stream, handle)
        else:
            sys.stderr = _Tee(stream, handle)


def detach_tee() -> None:
    for handle in _TEE_STREAMS:
        handle.close()
    _TEE_STREAMS.clear()


def run_cli(
    *arguments: str, check: bool = True, capture: bool = True
) -> subprocess.CompletedProcess:
    """Invoke the CLI entry point.

    The stages are driven in-process: under a workspace-confined sandbox a spawned
    child cannot be given piped or redirected stdio, so calling ``main`` directly is
    the portable way to run the same argument parser, handlers and exit code.
    ``capture=False`` instead spawns the real process so the launcher path itself is
    exercised at least once.
    """
    command = [PYTHON, str(PROJECT_DIR / "run_classical_baselines.py"), *arguments]
    if not capture:
        completed = subprocess.run(command, check=check)
        return subprocess.CompletedProcess(command, completed.returncode, "", "")
    out_buffer = io.StringIO()
    err_buffer = io.StringIO()
    with contextlib.redirect_stdout(out_buffer), contextlib.redirect_stderr(err_buffer):
        try:
            code = rcb.main(list(arguments))
        except SystemExit as exit_error:  # argparse --help / usage errors
            code = exit_error.code if isinstance(exit_error.code, int) else 1
    completed = subprocess.CompletedProcess(
        command, code, out_buffer.getvalue(), err_buffer.getvalue()
    )
    if check and code != 0:
        raise subprocess.CalledProcessError(
            code, command, output=completed.stdout, stderr=completed.stderr
        )
    return completed


def recompute_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    y = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    p = np.array([int(row["prediction"]) for row in rows], dtype=np.int64)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    total = tp + tn + fp + fn
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "windows": float(total),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "drone_recall": recall,
        "background_specificity": specificity,
        "f1": f1,
    }


def recompute_file_equal(rows: list[dict[str, str]]) -> dict[str, float]:
    groups: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        groups.setdefault((row["target_code"], row["filename"]), []).append(
            int(row["prediction"])
        )
    rates: dict[str, dict[str, float]] = {}
    for (code, name), values in groups.items():
        rates.setdefault(code, {})[name] = float(np.mean(values))
    background_fpr = float(np.mean(list(rates["0000"].values())))
    phantom = float(np.mean(list(rates["0010"].values())))
    air2s = float(np.mean(list(rates["0101"].values())))
    drone_recall = 0.5 * (phantom + air2s)
    specificity = 1.0 - background_fpr
    return {
        "file_equal_balanced_accuracy": 0.5 * (specificity + drone_recall),
        "file_equal_drone_recall": drone_recall,
        "file_equal_background_specificity": specificity,
        "file_equal_background_fpr": background_fpr,
        "file_equal_phantom_recall": phantom,
        "file_equal_air2s_recall": air2s,
    }


# ---------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------
class ConfusionAndMetricTests(unittest.TestCase):
    def test_confusion_counts_partition(self) -> None:
        labels = [0, 0, 1, 1, 1, 0]
        predictions = [0, 1, 1, 0, 1, 0]
        counts = rcb.confusion_counts(labels, predictions)
        self.assertEqual(counts, {"tn": 2, "fp": 1, "fn": 1, "tp": 2})
        self.assertEqual(sum(counts.values()), len(labels))
        values = rcb.metric_values(counts)
        self.assertAlmostEqual(values["accuracy"], 4 / 6)
        self.assertAlmostEqual(values["balanced_accuracy"], 0.5 * (2 / 3 + 2 / 3))
        self.assertAlmostEqual(values["precision"], 2 / 3)
        self.assertAlmostEqual(values["drone_recall"], 2 / 3)
        self.assertAlmostEqual(values["background_specificity"], 2 / 3)

    def test_non_binary_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rcb.confusion_counts([0, 2], [0, 0])
        with self.assertRaises(ValueError):
            rcb.confusion_counts([0, 1], [0, 3])

    def test_degenerate_confusion_is_finite(self) -> None:
        values = rcb.metric_values({"tn": 0, "fp": 0, "fn": 3, "tp": 0})
        for key, value in values.items():
            self.assertTrue(np.isfinite(value), key)


class ThresholdSelectionTests(unittest.TestCase):
    def test_threshold_separates_two_well_formed_clusters(self) -> None:
        scores = np.array([-60.0, -58.0, -40.0, -39.0])
        labels = np.array([0, 0, 1, 1])
        selection = rcb.select_energy_threshold(labels, scores)
        self.assertAlmostEqual(selection["balanced_accuracy"], 1.0)
        self.assertGreater(selection["threshold_db"], -58.0)
        self.assertLess(selection["threshold_db"], -40.0)
        predictions = (scores >= selection["threshold_db"]).astype(int)
        self.assertTrue(np.array_equal(predictions, labels))

    def test_threshold_is_deterministic_and_tie_breaks_low(self) -> None:
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        labels = np.array([0, 1, 0, 1])
        first = rcb.select_energy_threshold(labels, scores)
        second = rcb.select_energy_threshold(labels, scores)
        self.assertEqual(first["threshold_db"], second["threshold_db"])
        tied = [
            float(threshold)
            for threshold in rcb.candidate_thresholds(scores)
            if abs(
                rcb.balanced_accuracy_at_threshold(labels, scores, float(threshold))
                - first["balanced_accuracy"]
            )
            <= 1e-12
        ]
        self.assertAlmostEqual(first["threshold_db"], min(tied))

    def test_candidates_are_midpoints_between_unique_scores(self) -> None:
        scores = np.array([0.0, 1.0, 1.0, 3.0])
        candidates = rcb.candidate_thresholds(scores)
        self.assertEqual(candidates.size, 3)
        self.assertAlmostEqual(candidates[0], 0.0)
        self.assertAlmostEqual(candidates[1], 0.5)
        self.assertAlmostEqual(candidates[2], 2.0)

    def test_single_class_selection_data_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rcb.select_energy_threshold(np.array([1, 1]), np.array([0.0, 1.0]))


class FileEqualMetricTests(unittest.TestCase):
    def test_file_equal_weights_files_not_windows(self) -> None:
        records = [
            {"filename": "b0", "target_code": "0000", "label": 0, "prediction": 1},
            {"filename": "b1", "target_code": "0000", "label": 0, "prediction": 0},
            {"filename": "b1", "target_code": "0000", "label": 0, "prediction": 0},
            {"filename": "b1", "target_code": "0000", "label": 0, "prediction": 0},
            {"filename": "p0", "target_code": "0010", "label": 1, "prediction": 1},
            {"filename": "a0", "target_code": "0101", "label": 1, "prediction": 0},
        ]
        metrics = rcb.file_equal_metrics(records)
        self.assertAlmostEqual(metrics["background_file_equal_fpr"], 0.5)
        self.assertAlmostEqual(metrics["phantom_file_equal_recall"], 1.0)
        self.assertAlmostEqual(metrics["air2s_file_equal_recall"], 0.0)
        self.assertAlmostEqual(metrics["drone_file_equal_recall"], 0.5)
        self.assertAlmostEqual(metrics["file_equal_balanced_accuracy"], 0.5)
        pooled = rcb.pooled_metrics(records)
        self.assertAlmostEqual(pooled["drone_fpr"], 1 / 4)
        self.assertAlmostEqual(pooled["balanced_accuracy"], 0.5 * (0.5 + 0.75))

    def test_pooled_and_file_equal_are_not_equal_when_files_differ(self) -> None:
        records = [
            {"filename": "b0", "target_code": "0000", "label": 0, "prediction": 1},
            {"filename": "b1", "target_code": "0000", "label": 0, "prediction": 0},
            {"filename": "b1", "target_code": "0000", "label": 0, "prediction": 0},
            {"filename": "b1", "target_code": "0000", "label": 0, "prediction": 0},
            {"filename": "p0", "target_code": "0010", "label": 1, "prediction": 1},
            {"filename": "a0", "target_code": "0101", "label": 1, "prediction": 1},
        ]
        self.assertNotAlmostEqual(
            rcb.file_equal_metrics(records)["file_equal_balanced_accuracy"],
            rcb.pooled_metrics(records)["balanced_accuracy"],
        )


class ReferencePsdTests(unittest.TestCase):
    def test_numpy_reference_matches_a_manual_dft(self) -> None:
        rng = np.random.default_rng(7)
        window = (
            rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
        ).astype(np.complex64)
        taper = np.hamming(64)
        reference = rcb.numpy_reference_psd(
            window, n_fft=64, hop_length=32, window_fn=taper
        )
        centered = window - np.mean(window, dtype=np.complex128)
        power = float(np.mean(np.abs(centered) ** 2, dtype=np.float64))
        normalized = centered / np.sqrt(max(power, 1e-8))
        frames = 1 + (normalized.shape[0] - 64) // 32
        manual = []
        for index in range(frames):
            segment = normalized[index * 32 : index * 32 + 64] * taper
            spectrum = np.fft.fft(segment, n=64) / np.sqrt(64)
            manual.append(np.abs(spectrum) ** 2)
        expected = np.fft.fftshift(
            10.0 * np.log10(np.mean(np.array(manual), axis=0) + rcb.PSD_EPS)
        )
        self.assertEqual(reference.shape, (64,))
        np.testing.assert_allclose(reference, expected, atol=1e-9)

    def test_cached_window_is_the_periodic_hamming_used_by_torch(self) -> None:
        taper = rcb.hamming_window_matrix(64)
        symmetric = np.hamming(64)
        self.assertEqual(taper.shape, (64,))
        # The periodic Hamming window starts at 0.08 and its last sample is the one
        # the symmetric window would place one step outside the taper; the symmetric
        # window instead ends exactly at 0.08 with its peak in the middle.
        self.assertAlmostEqual(taper[0], 0.08, places=12)
        self.assertAlmostEqual(symmetric[-1], 0.08, places=12)
        self.assertAlmostEqual(float(taper[-1]), 0.08221502573078943, places=12)
        self.assertGreater(abs(float(taper[-1]) - float(symmetric[-1])), 1e-3)
        self.assertGreater(abs(float(taper[1]) - float(symmetric[1])), 1e-6)
        self.assertLess(abs(float(taper[1]) - float(symmetric[1])), 1e-3)

    def test_reference_is_deterministic(self) -> None:
        rng = np.random.default_rng(11)
        window = (rng.standard_normal(2048) + 1j * rng.standard_normal(2048)).astype(
            np.complex64
        )
        first = rcb.numpy_reference_psd(window, n_fft=128, hop_length=64)
        second = rcb.numpy_reference_psd(window, n_fft=128, hop_length=64)
        np.testing.assert_array_equal(first, second)


class IndexAndSourceIdentityTests(unittest.TestCase):
    def test_frozen_index_hashes(self) -> None:
        self.assertEqual(rcb.sha256_file(OLD_INDEX), rcb.OLD_INDEX_SHA256)
        self.assertEqual(rcb.sha256_file(HELDOUT_INDEX), rcb.HELDOUT_INDEX_SHA256)

    def test_source_files_are_disjoint_and_have_expected_identity(self) -> None:
        identity = rcb.source_identity(OLD_INDEX, HELDOUT_INDEX)
        self.assertTrue(identity["heldout_disjoint_from_old"])
        self.assertEqual(identity["file_overlap_count"], 0)
        # Window counts per split, and files per category, exactly as the frozen
        # indices define them: the old segment split gives train two files per
        # category, val/test one each, and the new heldout four each.
        expected = {
            "train": (920, 2),
            "val": (450, 1),
            "test": (440, 1),
            "heldout": (1820, 4),
        }
        for split, (windows, files_per_category) in expected.items():
            info = identity["per_split"][split]
            self.assertEqual(info["windows"], windows, split)
            self.assertEqual(
                sorted(info["files_per_category"]), ["0000", "0010", "0101"], split
            )
            for code, files in info["files_per_category"].items():
                self.assertEqual(len(files), files_per_category, f"{split}/{code}")
        self.assertEqual(len(identity["old_files"]), 12)
        self.assertEqual(len(identity["heldout_files"]), 12)
        self.assertEqual(
            set(identity["old_files"]) & set(identity["heldout_files"]), set()
        )

    def test_index_rows_keep_block_and_start_bookkeeping(self) -> None:
        rows = rcb.rows_by_split(OLD_INDEX, HELDOUT_INDEX)
        for split, split_rows in rows.items():
            for row in split_rows[:50]:
                self.assertLess(row["start"] + row["length"], row["source_samples"] + 1)
                self.assertEqual(row["length"], rcb.WINDOW_SAMPLES)
                self.assertIn(row["label"], (0, 1))
                self.assertEqual(split, row["split"])


class CliTests(unittest.TestCase):
    def test_module_help(self) -> None:
        result = run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for stage in ("energy", "psd-features", "psd-svm", "verify", "report"):
            self.assertIn(stage, result.stdout)

    def test_module_help_through_a_real_process(self) -> None:
        completed = subprocess.run(
            [PYTHON, str(PROJECT_DIR / "run_classical_baselines.py"), "--help"],
            check=False,
        )
        self.assertEqual(completed.returncode, 0)

    def test_each_stage_help(self) -> None:
        for stage in ("energy", "psd-features", "psd-svm", "verify", "report"):
            result = run_cli(stage, "--help")
            self.assertEqual(result.returncode, 0, f"{stage}: {result.stderr}")
            self.assertIn("--output-dir", result.stdout)

    def test_unknown_stage_fails(self) -> None:
        result = run_cli("nonsense", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr)

    def test_sklearn_fallback_import_is_recorded(self) -> None:
        info = rcb.import_sklearn(rcb.DEFAULT_SKLEARN_PATH)
        self.assertTrue(info["imported"])
        self.assertTrue(info["sklearn_version"])
        self.assertTrue(Path(info["sklearn_file"]).is_file())
        if info["fallback_used"]:
            self.assertEqual(info["fallback_path"], str(rcb.DEFAULT_SKLEARN_PATH))


# ---------------------------------------------------------------------------
# smoke run over a limited window subset
# ---------------------------------------------------------------------------
def write_smoke_indices(directory: Path, windows_per_file: int) -> tuple[Path, Path]:
    """Keep the first N windows of up to two source files per split/category.

    The old index has two train files per category but only one val/test file, so
    the number of source files is taken from the index instead of being assumed.
    """
    directory.mkdir(parents=True, exist_ok=True)
    old_rows = rcb.load_index_rows(OLD_INDEX)
    heldout_rows = rcb.load_index_rows(HELDOUT_INDEX)

    def select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
        for row in rows:
            grouped.setdefault(row["split"], {}).setdefault(row["target_code"], {}).setdefault(
                row["filename"], []
            ).append(row)
        selected: list[dict[str, Any]] = []
        for split in sorted(grouped):
            for code in sorted(grouped[split]):
                files = sorted(grouped[split][code])
                if not files:
                    raise RuntimeError(f"smoke subset has no source file for {split}/{code}")
                for name in files[:2]:
                    selected.extend(grouped[split][code][name][:windows_per_file])
        return selected

    def dump(rows: list[dict[str, Any]], path: Path) -> Path:
        fields = list(rows[0])
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    old_path = dump(select(old_rows), directory / "smoke_window_index.csv")
    heldout_path = dump(select(heldout_rows), directory / "smoke_heldout_index.csv")
    return old_path, heldout_path


def smoke_run(windows_per_file: int = 2) -> dict[str, Any]:
    with scratch_directory("smoke_") as root:
        return _smoke_run_in(root, windows_per_file)


def _smoke_run_in(root: Path, windows_per_file: int) -> dict[str, Any]:
    old_path, heldout_path = write_smoke_indices(root / "indices", windows_per_file)
    output_dir = root / "outputs"
    common = (
        "--output-dir",
        str(output_dir),
        "--old-index",
        str(old_path),
        "--heldout-index",
        str(heldout_path),
        "--sklearn-path",
        str(rcb.DEFAULT_SKLEARN_PATH),
    )
    stages = []
    for stage in (
        ("energy",),
        ("psd-features", "--device", "cpu", "--batch-size", "3", "--numpy-check", "2"),
        ("psd-svm",),
        ("verify", "--preserved", str(rcb.DEFAULT_PRESERVED), "--skip-frozen-hashes"),
        ("report",),
    ):
        result = run_cli(stage[0], *stage[1:], *common, capture=False, check=False)
        stages.append({"stage": stage[0], "returncode": result.returncode})
    required = [
        "energy_threshold/selection.json",
        "energy_threshold/test_window_predictions.csv",
        "energy_threshold/heldout_window_predictions.csv",
        "energy_threshold/metrics.csv",
        "energy_threshold/energy_distributions.png",
        "psd_svm/model.joblib",
        "psd_svm/selection.json",
        "psd_svm/test_window_predictions.csv",
        "psd_svm/heldout_window_predictions.csv",
        "psd_svm/metrics.csv",
        "psd_svm/psd_examples.png",
        "comparison_metrics.csv",
        "experiment_manifest.json",
        "report.md",
        "verification.json",
    ]
    present = {name: (output_dir / name).is_file() for name in required}
    heldout_predictions = read_rows(
        output_dir / "psd_svm" / "heldout_window_predictions.csv"
    )
    energy_heldout = read_rows(
        output_dir / "energy_threshold" / "heldout_window_predictions.csv"
    )
    indices = rcb.load_index_rows(heldout_path)
    return {
        "output_dir": str(output_dir),
        "stages": stages,
        "required_outputs": present,
        "heldout_rows": len(heldout_predictions),
        "heldout_index_rows": len(indices),
        "energy_heldout_rows": len(energy_heldout),
        "all_predictions_finite": all(
            np.isfinite(float(row["decision_function"]))
            for row in heldout_predictions
        ),
        "verification": json.loads(
            (output_dir / "verification.json").read_text(encoding="utf-8")
        ),
        "energy_selection": json.loads(
            (output_dir / "energy_threshold" / "selection.json").read_text(
                encoding="utf-8"
            )
        ),
        "psd_selection": json.loads(
            (output_dir / "psd_svm" / "selection.json").read_text(encoding="utf-8")
        ),
    }


# ---------------------------------------------------------------------------
# independent recompute against the real outputs
# ---------------------------------------------------------------------------
def verify_outputs() -> dict[str, Any]:
    report: dict[str, Any] = {"checks": [], "failures": []}

    def check(name: str, passed: bool, evidence: Any) -> None:
        report["checks"].append({"name": name, "passed": bool(passed), "evidence": evidence})
        if not passed:
            report["failures"].append(name)
        print(f"  [{'ok' if passed else 'FAIL'}] {name}")

    identity = rcb.source_identity(OLD_INDEX, HELDOUT_INDEX)
    check(
        "heldout source files are disjoint from the old split files",
        identity["heldout_disjoint_from_old"],
        {"overlap": identity["file_overlap_count"]},
    )

    for method, folder, score_field in (
        ("energy_threshold", "energy_threshold", "score_db"),
        ("psd_svm", "psd_svm", "decision_function"),
    ):
        directory = BASELINE_DIR / folder
        stored = {
            (row["method"], row["split"]): row
            for row in read_rows(directory / "metrics.csv")
        }
        for split, expected_rows in (("test", 440), ("heldout", 1820)):
            csv_path = directory / f"{split}_window_predictions.csv"
            if not csv_path.is_file():
                check(f"{method}/{split}: predictions exist", False, str(csv_path))
                continue
            rows = read_rows(csv_path)
            check(
                f"{method}/{split}: {expected_rows} prediction rows",
                len(rows) == expected_rows,
                {"rows": len(rows)},
            )
            check(
                f"{method}/{split}: every score is finite and every decision is binary",
                all(
                    np.isfinite(float(row[score_field]))
                    and int(row["prediction"]) in (0, 1)
                    for row in rows
                ),
                {"field": score_field},
            )
            independent = recompute_metrics(rows)
            file_equal = recompute_file_equal(rows)
            row = stored[(method, split)]
            check(
                f"{method}/{split}: pooled counts match metrics.csv",
                all(
                    abs(independent[key] - float(row[key])) < 1e-9
                    for key in ("tn", "fp", "fn", "tp")
                ),
                {
                    key: {"recomputed": independent[key], "stored": float(row[key])}
                    for key in ("tn", "fp", "fn", "tp")
                },
            )
            check(
                f"{method}/{split}: pooled metrics match metrics.csv",
                all(
                    abs(independent[target] - float(row[column])) < 1e-12
                    for target, column in (
                        ("accuracy", "pooled_accuracy"),
                        ("balanced_accuracy", "pooled_balanced_accuracy"),
                        ("precision", "pooled_precision"),
                        ("drone_recall", "pooled_drone_recall"),
                        ("background_specificity", "pooled_background_specificity"),
                        ("f1", "pooled_f1"),
                    )
                ),
                {
                    "accuracy": [independent["accuracy"], float(row["pooled_accuracy"])],
                    "balanced_accuracy": [
                        independent["balanced_accuracy"],
                        float(row["pooled_balanced_accuracy"]),
                    ],
                },
            )
            check(
                f"{method}/{split}: file-equal metrics match metrics.csv",
                all(
                    abs(file_equal[key] - float(row[key])) < 1e-12
                    for key in (
                        "file_equal_balanced_accuracy",
                        "file_equal_drone_recall",
                        "file_equal_background_specificity",
                        "file_equal_background_fpr",
                    )
                ),
                {
                    "balanced_accuracy": [
                        file_equal["file_equal_balanced_accuracy"],
                        float(row["file_equal_balanced_accuracy"]),
                    ]
                },
            )
            check(
                f"{method}/{split}: confusion counts conserve the row total",
                independent["tn"]
                + independent["fp"]
                + independent["fn"]
                + independent["tp"]
                == len(rows),
                {"total": len(rows)},
            )
            check(
                f"{method}/{split}: per-category source-file count matches the index",
                all(
                    len({r["filename"] for r in rows if r["target_code"] == code})
                    == len(identity["per_split"][split]["files_per_category"][code])
                    for code in ("0000", "0010", "0101")
                ),
                {
                    code: len({r["filename"] for r in rows if r["target_code"] == code})
                    for code in ("0000", "0010", "0101")
                },
            )

        selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        check(
            f"{method}: selection.json records the frozen restriction to train/val",
            (
                selection["selection_data"]["splits"] == ["train", "val"]
                if method == "energy_threshold"
                else selection["selection"]["selected_on_splits"] == ["val"]
                and selection["selection"]["fitted_on_splits"] == ["train"]
            ),
            {},
        )

    comparison = read_rows(BASELINE_DIR / "comparison_metrics.csv")
    check(
        "comparison_metrics.csv carries all four method/split rows",
        len(comparison) == 4
        and {(row["method"], row["split"]) for row in comparison}
        == {
            ("energy_threshold", "test"),
            ("energy_threshold", "heldout"),
            ("psd_svm", "test"),
            ("psd_svm", "heldout"),
        },
        {"rows": len(comparison)},
    )
    check(
        "comparison_metrics.csv aggregates the two categories of drone recall",
        all(
            abs(
                float(row["drone_recall"])
                - 0.5
                * (
                    float(row["phantom_file_equal_recall"])
                    + float(row["air2s_file_equal_recall"])
                )
            )
            < 1e-12
            for row in comparison
        ),
        {},
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run the limited smoke run")
    parser.add_argument(
        "--verify-outputs",
        action="store_true",
        help="independently recompute metrics from the real prediction CSVs",
    )
    parser.add_argument("--windows-per-file", type=int, default=2)
    parser.add_argument(
        "--log",
        type=Path,
        help="mirror stdout/stderr into this file (needed without a console pipe)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        attach_tee([args.log, args.log.with_suffix(".err.txt")])
    try:
        failures = 0
        if args.verify_outputs:
            print("independent recompute of the produced outputs")
            report = verify_outputs()
            report_path = BASELINE_DIR / "independent_recompute_test.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            failures += len(report["failures"])
        if args.smoke:
            print(f"smoke run with {args.windows_per_file} windows per source file")
            summary = smoke_run(args.windows_per_file)
            print(
                json.dumps(
                    {
                        "stages": summary["stages"],
                        "required_outputs": summary["required_outputs"],
                        "heldout_rows": summary["heldout_rows"],
                        "heldout_index_rows": summary["heldout_index_rows"],
                        "all_predictions_finite": summary["all_predictions_finite"],
                        "verification_passed": summary["verification"]["passed"],
                        "verification_failures": summary["verification"]["failures"],
                        "energy_threshold_db": summary["energy_selection"][
                            "threshold_db"
                        ],
                        "psd_selected_candidate": summary["psd_selection"]["svm"][
                            "selected_candidate_id"
                        ],
                        "numpy_reference_check": summary["psd_selection"]
                        .get("feature", {})
                        .get("source"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            for stage in summary["stages"]:
                if stage["returncode"] != 0:
                    failures += 1
            if not all(summary["required_outputs"].values()):
                failures += 1
            if summary["heldout_rows"] != summary["heldout_index_rows"]:
                failures += 1
            if not summary["verification"]["passed"]:
                failures += 1

        unittest_result = unittest.main(
            argv=[sys.argv[0]], exit=False, module=sys.modules[__name__], verbosity=2
        ).result
        failures += len(unittest_result.failures) + len(unittest_result.errors)
        print(f"failures: {failures}")
        return 1 if failures else 0
    finally:
        detach_tee()


if __name__ == "__main__":
    raise SystemExit(main())
