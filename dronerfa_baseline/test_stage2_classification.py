"""Checks for STAGE2-BASE (``run_stage2_classification.py``).

Three ways to run it::

    python test_stage2_classification.py                # offline unit tests only
    python test_stage2_classification.py --smoke        # + bounded end-to-end smoke run
    python test_stage2_classification.py --verify-outputs  # + verifier on the real root

The unit tests use tiny synthetic HDF5/index fixtures and a synthetic output tree, so
no test ever evaluates the full 1,260-window held-out split. ``--smoke`` trains two
deliberately tiny runs (a few windows, one epoch) in a scratch directory and drives
the CLI commands over them; ``--verify-outputs`` runs the independent verifier against
the real artefacts.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from typing import Any, Iterator, Sequence

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import torch

# The checks are small; a tiny thread pool keeps them from competing with a training run.
torch.set_num_threads(int(os.environ.get("STAGE2_TEST_THREADS", "2")))

import run_stage2_classification as rsc
import verify_stage2_classification as vsc
from dronerfa_data import add_complex_awgn, rms_normalize
from model import SpectrogramFrontEnd

OUTPUT_ROOT = rsc.OUTPUT_ROOT
SCRATCH_ROOT = OUTPUT_ROOT.parent / "_test_scratch"


@contextlib.contextmanager
def scratch_directory(prefix: str) -> Iterator[Path]:
    """Deterministic scratch space inside the project (sandbox-safe)."""
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    path = SCRATCH_ROOT / f"{prefix}{os.getpid()}"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# synthetic index / HDF5 fixtures
# ---------------------------------------------------------------------------
INDEX_FIELDS = (
    "file_path",
    "filename",
    "target_code",
    "target_name",
    "distance_code",
    "segment_code",
    "split",
    "label",
    "block_index",
    "window_index",
    "start",
    "length",
    "source_samples",
)


def write_synthetic_file(path: Path, windows: int, samples: int, offset: float) -> None:
    """One tiny MAT-like HDF5 file with RF0_I / RF0_Q vectors."""
    import h5py

    total = windows * samples
    base = np.arange(total, dtype=np.float64) * 1e-4 + offset
    with h5py.File(path, "w") as handle:
        handle.create_dataset("RF0_I", data=base)
        handle.create_dataset("RF0_Q", data=np.flip(base) * 0.5 + offset)


def write_synthetic_index(
    directory: Path,
    layout: Sequence[tuple[str, str, str, int]],
    split: str,
    samples: int = 4096,
) -> Path:
    """Build an index over synthetic files.

    ``layout`` entries are ``(target_code, filename, split, windows)``; the label
    column follows the stage-1 convention (1 for both drone codes).
    """
    rows: list[dict[str, Any]] = []
    for target_code, filename, row_split, windows in layout:
        path = directory / filename
        write_synthetic_file(path, windows, samples, offset=float(len(rows) + 1))
        block = 0
        for position in range(windows):
            rows.append(
                {
                    "file_path": str(path),
                    "filename": filename,
                    "target_code": target_code,
                    "target_name": "synthetic",
                    "distance_code": "00",
                    "segment_code": filename.split("_")[-1].split(".")[0],
                    "split": row_split,
                    "label": 0 if target_code == "0000" else 1,
                    "block_index": block,
                    "window_index": position,
                    "start": position * samples,
                    "length": samples,
                    "source_samples": windows * samples,
                }
            )
    index_path = directory / f"{split}_index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return index_path


def synthetic_records() -> list[dict[str, Any]]:
    """Two source files per class with unequal window counts, hand-checkable.

    ``hits`` counts *correct* windows: a Phantom window is correct when it is
    predicted Phantom (0), an Air 2S window when it is predicted Air 2S (1).
    """
    records: list[dict[str, Any]] = []
    layout = {
        0: [("phantom_a.mat", 4, 4), ("phantom_b.mat", 8, 4)],
        1: [("air2s_a.mat", 6, 3), ("air2s_b.mat", 2, 2)],
    }
    for label, files in layout.items():
        for filename, windows, hits in files:
            for window in range(windows):
                correct = window < hits
                prediction = (1 if correct else 0) if label == 1 else (0 if correct else 1)
                records.append(
                    {"filename": filename, "label": label, "prediction": prediction}
                )
    return records


# ---------------------------------------------------------------------------
# label mapping, stage-2 index and RNG rule
# ---------------------------------------------------------------------------
class Stage2IndexTests(unittest.TestCase):
    def test_label_remap_and_background_filter(self) -> None:
        rows = [
            {"target_code": "0000", "label": 0, "filename": "bg.mat"},
            {"target_code": "0010", "label": 1, "filename": "phantom.mat"},
            {"target_code": "0101", "label": 1, "filename": "air2s.mat"},
        ]
        selected = rsc.stage2_rows(rows)
        self.assertEqual([row["filename"] for row in selected], ["phantom.mat", "air2s.mat"])
        self.assertEqual([row["label"] for row in selected], [0, 1])
        self.assertEqual([row["class_name"] for row in selected], list(rsc.LABEL_TO_NAME.values()))
        self.assertEqual(rsc.CODE_TO_LABEL, {"0010": 0, "0101": 1})

    def test_drone_code_with_non_drone_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rsc.stage2_rows([{"target_code": "0010", "label": 0, "filename": "x.mat"}])

    def test_index_positions_start_at_zero_and_keep_order(self) -> None:
        index_path = rsc.OLD_INDEX
        rows = rsc.stage2_index_rows(index_path, "val")
        self.assertEqual([row["index_position"] for row in rows], list(range(len(rows))))
        self.assertEqual(len(rows), rsc.EXPECTED_SPLIT_WINDOWS["val"])
        self.assertEqual(
            [sum(1 for row in rows if row["label"] == label) for label in (0, 1)],
            list(rsc.EXPECTED_CLASS_WINDOWS["val"]),
        )
        # positions follow the frozen file order of the split
        self.assertEqual(rows[0]["filename"], "T0010_D00_S0010.mat")
        self.assertEqual(rows[-1]["filename"], "T0101_D00_S0010.mat")

    def test_all_four_splits_match_the_frozen_composition(self) -> None:
        for split, index_path in (
            ("train", rsc.OLD_INDEX),
            ("val", rsc.OLD_INDEX),
            ("test", rsc.OLD_INDEX),
            ("heldout", rsc.HELDOUT_INDEX),
        ):
            with self.subTest(split=split):
                rows = rsc.stage2_index_rows(index_path, split)
                self.assertEqual(len(rows), rsc.EXPECTED_SPLIT_WINDOWS[split])
                self.assertEqual(
                    [sum(1 for row in rows if row["label"] == label) for label in (0, 1)],
                    list(rsc.EXPECTED_CLASS_WINDOWS[split]),
                )
                files = {
                    label: {row["filename"] for row in rows if row["label"] == label}
                    for label in (0, 1)
                }
                self.assertEqual(
                    [len(files[0]), len(files[1])], list(rsc.EXPECTED_CLASS_FILES[split])
                )


class NoisePairingTests(unittest.TestCase):
    """The frozen reader's AWGN must be a property of the stage-2 position alone."""

    def setUp(self) -> None:
        self._scratch = scratch_directory("pairing")
        self.directory = self._scratch.__enter__()
        layout = [
            ("0000", "bg.mat", "val", 2),
            ("0010", "phantom.mat", "val", 3),
            ("0101", "air2s.mat", "val", 2),
        ]
        self.index_path = write_synthetic_index(self.directory, layout, "val")
        self.samples = 4096

    def tearDown(self) -> None:
        self._scratch.__exit__(None, None, None)

    def _expected_iq(
        self, filename: str, position_in_file: int, snr: float | None, rng_seed: int
    ):
        import h5py

        with h5py.File(self.directory / filename, "r") as handle:
            start = position_in_file * self.samples
            values = np.asarray(handle["RF0_I"][start : start + self.samples], dtype=np.float32)
            quadrature = np.asarray(
                handle["RF0_Q"][start : start + self.samples], dtype=np.float32
            )
        clean = rms_normalize(values + 1j * quadrature)
        if snr is None:
            return clean
        rng = np.random.default_rng(rng_seed)
        return add_complex_awgn(clean, snr, rng).astype(np.complex64)

    def test_rows_are_filtered_relabelled_and_positioned(self) -> None:
        dataset = rsc.Stage2WindowDataset(self.index_path, split="val", seed=rsc.VALIDATION_SEED)
        try:
            self.assertEqual(len(dataset), 5)
            self.assertEqual([row["label"] for row in dataset.rows], [0, 0, 0, 1, 1])
            self.assertEqual(dataset.epoch, 0)
        finally:
            dataset.close()

    def test_noise_follows_the_stage2_position(self) -> None:
        dataset = rsc.Stage2WindowDataset(
            self.index_path,
            split="val",
            fixed_snr=-10.0,
            apply_fixed_snr=True,
            seed=rsc.VALIDATION_SEED,
        )
        try:
            for position in range(len(dataset)):
                with self.subTest(position=position):
                    tensor, label = dataset[position]
                    row = dataset.rows[position]
                    expected = self._expected_iq(
                        row["filename"],
                        int(row["window_index"]),
                        -10.0,
                        rsc.VALIDATION_SEED + position,
                    )
                    observed = tensor[0].numpy() + 1j * tensor[1].numpy()
                    self.assertTrue(np.array_equal(observed, expected))
                    self.assertEqual(int(label), int(row["label"]))
        finally:
            dataset.close()

    def test_clean_windows_carry_no_noise(self) -> None:
        dataset = rsc.Stage2WindowDataset(self.index_path, split="val", seed=rsc.VALIDATION_SEED)
        try:
            tensor, _ = dataset[0]
            expected = self._expected_iq("phantom.mat", 0, None, 0)
            observed = tensor[0].numpy() + 1j * tensor[1].numpy()
            self.assertTrue(np.allclose(observed, expected, atol=1e-6))
        finally:
            dataset.close()

    def test_a_narrowed_view_keeps_the_full_split_rng(self) -> None:
        """A bounded run must see exactly the noise of the full split."""
        dataset = rsc.Stage2WindowDataset(
            self.index_path,
            split="val",
            fixed_snr=-12.0,
            apply_fixed_snr=True,
            seed=rsc.VALIDATION_SEED,
        )
        try:
            view = rsc._IndexSubsetDataset(dataset, [2, 4])
            self.assertEqual(len(view), len(dataset))
            tensor, _ = view[1]
            row = dataset.rows[4]
            expected = self._expected_iq(
                row["filename"],
                int(row["window_index"]),
                -12.0,
                rsc.VALIDATION_SEED + 4,
            )
            observed = tensor[0].numpy() + 1j * tensor[1].numpy()
            self.assertTrue(np.array_equal(observed, expected))
        finally:
            dataset.close()

    def test_noise_pairing_digest_is_frontend_independent(self) -> None:
        digest_512 = rsc.noise_pairing_digest(
            self.index_path, "val", "-10dB", rsc.VALIDATION_SEED
        )
        digest_again = rsc.noise_pairing_digest(
            self.index_path, "val", "-10dB", rsc.VALIDATION_SEED
        )
        self.assertEqual(digest_512["digest_sha256"], digest_again["digest_sha256"])
        self.assertEqual(
            [probe["rng_seed"] for probe in digest_512["probe_positions"]],
            [rsc.VALIDATION_SEED + position for position in (0, 2, 4)],
        )
        other = rsc.noise_pairing_digest(
            self.index_path, "val", "-10dB", rsc.VALIDATION_SEED + 1
        )
        self.assertNotEqual(digest_512["digest_sha256"], other["digest_sha256"])


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
class MetricTests(unittest.TestCase):
    def test_pooled_metrics_are_hand_checkable(self) -> None:
        rows = [
            {"label": 1, "prediction": 1},
            {"label": 1, "prediction": 0},
            {"label": 0, "prediction": 0},
            {"label": 0, "prediction": 1},
        ]
        metrics = rsc.stage2_pooled_metrics(rows)
        self.assertEqual((metrics["tp"], metrics["fn"], metrics["tn"], metrics["fp"]), (1, 1, 1, 1))
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision_air2s"], 0.5)
        self.assertAlmostEqual(metrics["phantom_recall"], 0.5)
        self.assertAlmostEqual(metrics["air2s_recall"], 0.5)
        self.assertAlmostEqual(metrics["f1_air2s_binary"], 0.5)
        self.assertAlmostEqual(metrics["f1_macro"], 0.5)
        self.assertEqual(
            metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"], len(rows)
        )

    def test_file_equal_metrics_weight_files_not_windows(self) -> None:
        records = synthetic_records()
        metrics = rsc.stage2_file_equal_metrics(records)
        # Phantom: 4/4 and 4/8 -> 0.75; Air2S: 3/6 and 2/2 -> 0.75
        self.assertAlmostEqual(metrics["file_equal_phantom_recall"], 0.75)
        self.assertAlmostEqual(metrics["file_equal_air2s_recall"], 0.75)
        self.assertAlmostEqual(metrics["file_equal_balanced_accuracy"], 0.75)
        self.assertEqual((metrics["files_phantom"], metrics["files_air2s"]), (2, 2))
        pooled = rsc.stage2_pooled_metrics(records)
        # pooled windows: 8/12 Phantom windows correct, 5/8 Air2S windows correct
        self.assertAlmostEqual(pooled["phantom_recall"], 8 / 12)
        self.assertAlmostEqual(pooled["air2s_recall"], 5 / 8)
        self.assertNotAlmostEqual(
            pooled["balanced_accuracy"], metrics["file_equal_balanced_accuracy"]
        )

    def test_file_equal_needs_both_classes_unless_bounded(self) -> None:
        only_phantom = [row for row in synthetic_records() if row["label"] == 0]
        with self.assertRaises(ValueError):
            rsc.stage2_file_equal_metrics(only_phantom)
        metrics = rsc.stage2_file_equal_metrics(only_phantom, require_both_classes=False)
        self.assertTrue(np.isnan(metrics["file_equal_balanced_accuracy"]))
        self.assertAlmostEqual(metrics["file_equal_phantom_recall"], 0.75)

    def test_metric_row_reports_both_aggregations(self) -> None:
        row = rsc.metric_row({"split": "heldout"}, synthetic_records())
        for field in rsc.METRIC_COLUMN_ORDER:
            self.assertIn(field, row)
        self.assertAlmostEqual(row["file_equal_balanced_accuracy"], 0.75)
        self.assertEqual(row["windows"], 20)


# ---------------------------------------------------------------------------
# selection rules
# ---------------------------------------------------------------------------
class SelectionTests(unittest.TestCase):
    def test_epoch_rule_takes_the_highest_mean_and_earliest_tie(self) -> None:
        history = [
            {"epoch": 1, "mean_file_equal_balanced_accuracy": 0.5},
            {"epoch": 2, "mean_file_equal_balanced_accuracy": 0.9},
            {"epoch": 3, "mean_file_equal_balanced_accuracy": 0.9},
            {"epoch": 4, "mean_file_equal_balanced_accuracy": 0.8},
        ]
        self.assertEqual(rsc.select_epoch(history), 2)
        history[0]["mean_file_equal_balanced_accuracy"] = 0.95
        self.assertEqual(rsc.select_epoch(history), 1)

    def _write_run(
        self,
        root: Path,
        n_fft: int,
        seed: int,
        per_condition: dict[str, float],
        selected_epoch: int = 1,
    ) -> None:
        run_dir = root / f"nfft{n_fft}" / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        rows = []
        for epoch in (1, 2):
            for condition in rsc.CONDITIONS:
                score = per_condition[condition] if epoch == selected_epoch else 0.0
                rows.append(
                    {
                        "epoch": epoch,
                        "condition": condition,
                        "windows": rsc.EXPECTED_SPLIT_WINDOWS["val"],
                        "file_equal_balanced_accuracy": score,
                        "balanced_accuracy": score,
                        "file_equal_phantom_recall": score,
                        "file_equal_air2s_recall": score,
                        "files_phantom": 1,
                        "files_air2s": 1,
                    }
                )
        rsc.write_csv(run_dir / "validation_metrics.csv", rows)
        rsc.write_json(
            run_dir / "selection.json",
            {
                "n_fft": n_fft,
                "training_seed": seed,
                "rule": rsc.EPOCH_SELECTION_RULE,
                "tie_break": rsc.TIE_BREAK,
                "selected_epoch": selected_epoch,
            },
        )
        rsc.write_json(
            run_dir / "run_config.json",
            {
                "n_fft": n_fft,
                "training_seed": seed,
                "status": "completed",
                "selected_epoch": selected_epoch,
            },
        )
        (run_dir / "best_model.pt").write_bytes(b"placeholder")

    def test_nfft_rule_uses_validation_means_only(self) -> None:
        with scratch_directory("nfft_select") as root:
            for seed in rsc.TRAINING_SEEDS:
                self._write_run(
                    root,
                    512,
                    seed,
                    {condition: 0.90 for condition in rsc.CONDITIONS},
                )
                self._write_run(
                    root,
                    1024,
                    seed,
                    {condition: 0.80 for condition in rsc.CONDITIONS},
                )
            selection = rsc.build_nfft_selection(root)
            self.assertEqual(selection["selected_n_fft"], 512)
            self.assertFalse(selection["partial_grid"])
            self.assertAlmostEqual(selection["selection_margin"], 0.10)
            text = json.dumps(selection, ensure_ascii=False).lower()
            for token in vsc.SELECTION_FORBIDDEN_TOKENS:
                self.assertNotIn(token.lower(), text)

    def test_nfft_rule_breaks_an_exact_tie_towards_512(self) -> None:
        with scratch_directory("nfft_tie") as root:
            for seed in rsc.TRAINING_SEEDS:
                self._write_run(
                    root, 512, seed, {condition: 0.75 for condition in rsc.CONDITIONS}
                )
                self._write_run(
                    root, 1024, seed, {condition: 0.75 for condition in rsc.CONDITIONS}
                )
            selection = rsc.build_nfft_selection(root)
            self.assertEqual(selection["selected_n_fft"], 512)
            self.assertTrue(selection["exact_tie"])
            self.assertEqual(selection["selected_hop_length"], 256)

    def test_select_nfft_refuses_to_rewrite_after_results_exist(self) -> None:
        with scratch_directory("nfft_locked") as root:
            for seed in rsc.TRAINING_SEEDS:
                self._write_run(
                    root, 512, seed, {condition: 0.75 for condition in rsc.CONDITIONS}
                )
                self._write_run(
                    root, 1024, seed, {condition: 0.75 for condition in rsc.CONDITIONS}
                )
            args = argparse.Namespace(output_dir=root)
            self.assertEqual(rsc.command_select_nfft(args), 0)
            lock = rsc.read_json(root / "selection_lock.json")
            self.assertEqual(
                lock["selection_sha256"], rsc.sha256_file(root / "validation_nfft_selection.json")
            )
            (root / "heldout_window_predictions.csv").write_text("x", encoding="utf-8")
            with self.assertRaises(SystemExit):
                rsc.command_select_nfft(args)


# ---------------------------------------------------------------------------
# frontend
# ---------------------------------------------------------------------------
class FrontendTests(unittest.TestCase):
    def test_hop_is_half_of_every_n_fft(self) -> None:
        self.assertEqual(rsc.NFFT_GRID, (512, 1024))
        for n_fft in rsc.NFFT_GRID:
            self.assertEqual(rsc.hop_for(n_fft), n_fft // 2)

    def test_both_frontends_emit_one_256_square(self) -> None:
        signals = torch.zeros(2, 2, 40_000)
        for n_fft in rsc.NFFT_GRID:
            frontend = SpectrogramFrontEnd(
                n_fft=n_fft, hop_length=rsc.hop_for(n_fft), image_size=256
            ).eval()
            with torch.inference_mode():
                features = frontend(signals)
            self.assertEqual(tuple(features.shape), (2, 1, 256, 256))
            self.assertTrue(torch.isfinite(features).all())


# ---------------------------------------------------------------------------
# verifier fixture: accept a consistent tree, reject tampering
# ---------------------------------------------------------------------------
class VerifierFixtureTests(unittest.TestCase):
    """Build a small synthetic output tree and drive the verifier over it."""

    nfft_values = (512, 1024)
    seeds = (2026, 2027)
    heldout_layout = [
        ("0010", "T0010_D00_S9000.mat", "heldout", 2),
        ("0010", "T0010_D00_S9001.mat", "heldout", 1),
        ("0101", "T0101_D00_S9000.mat", "heldout", 2),
        ("0101", "T0101_D00_S9001.mat", "heldout", 1),
    ]
    test_layout = [
        ("0010", "T0010_D00_S9002.mat", "test", 2),
        ("0101", "T0101_D00_S9002.mat", "test", 2),
    ]

    def setUp(self) -> None:
        self._scratch = scratch_directory("verify_fixture")
        self.root = self._scratch.__enter__()
        self.heldout_index = write_synthetic_index(
            self.root, self.heldout_layout, "heldout"
        )
        self.old_index = write_synthetic_index(self.root, self.test_layout, "test")
        self.heldout_rows = rsc.stage2_index_rows(self.heldout_index, "heldout")
        self.test_rows = rsc.stage2_index_rows(self.old_index, "test")
        self.run_configs: dict[tuple[int, int], dict[str, Any]] = {}
        self._build_runs()
        self._build_predictions("heldout", self.heldout_rows, rsc.CONDITIONS)
        self._build_predictions("test", self.test_rows, ("clean",))
        self._build_cache_manifest()
        self._build_selection()
        self._build_tables()
        self._build_figures()
        self._build_manifest_and_report()

    def tearDown(self) -> None:
        self._scratch.__exit__(None, None, None)

    # -- builders ---------------------------------------------------------
    def _build_runs(self) -> None:
        for n_fft in self.nfft_values:
            for seed in self.seeds:
                run_dir = self.root / f"nfft{n_fft}" / f"seed_{seed}"
                run_dir.mkdir(parents=True)
                selected = 1 + (seed % 2)
                history = []
                metric_rows = []
                for epoch in (1, 2):
                    score = 0.5 + 0.1 * epoch if epoch == selected else 0.4
                    history.append(
                        {
                            "epoch": epoch,
                            "train_loss": 0.5,
                            "train_balanced_accuracy": 0.5,
                            "clean_file_equal_balanced_accuracy": score,
                            "mean_file_equal_balanced_accuracy": score,
                            "seconds": 1.0,
                        }
                    )
                    for condition in rsc.CONDITIONS:
                        metric_rows.append(
                            {
                                "epoch": epoch,
                                "condition": condition,
                                "windows": len(self.heldout_rows),
                                "files_phantom": 2,
                                "files_air2s": 2,
                                "file_equal_phantom_recall": score,
                                "file_equal_air2s_recall": score,
                                "file_equal_balanced_accuracy": score,
                                "accuracy": score,
                                "balanced_accuracy": score,
                                "precision_air2s": score,
                                "precision_phantom": score,
                                "phantom_recall": score,
                                "air2s_recall": score,
                                "f1_air2s_binary": score,
                                "f1_macro": score,
                                "tn": 1,
                                "fp": 1,
                                "fn": 1,
                                "tp": 1,
                            }
                        )
                rsc.write_csv(run_dir / "training_history.csv", history)
                rsc.write_csv(run_dir / "validation_metrics.csv", metric_rows)
                config = {
                    "status": "completed",
                    "n_fft": n_fft,
                    "hop_length": rsc.hop_for(n_fft),
                    "image_size": 256,
                    "training_seed": seed,
                    "epochs": 2,
                    "batch_size": 4,
                    "train_snrs": rsc.TRAIN_SNRS,
                    "validation_seed": rsc.VALIDATION_SEED,
                    "validation_conditions": list(rsc.CONDITIONS),
                    "optimizer": {"type": "Adam", "learning_rate": 1e-3, "weight_decay": 0.0},
                    "amp": True,
                    "initial_model_sha256": f"initial-{seed}",
                    "epoch_order_sha256": [f"order-{seed}-{epoch}" for epoch in (1, 2)],
                    "selected_epoch": selected,
                    "label_mapping": {str(k): v for k, v in rsc.LABEL_TO_NAME.items()},
                }
                rsc.write_json(run_dir / "run_config.json", config)
                rsc.write_json(
                    run_dir / "selection.json",
                    {
                        "n_fft": n_fft,
                        "training_seed": seed,
                        "rule": rsc.EPOCH_SELECTION_RULE,
                        "tie_break": rsc.TIE_BREAK,
                        "selected_epoch": selected,
                    },
                )
                torch.save({"epoch": selected, "model_state": {}}, run_dir / "best_model.pt")
                self.run_configs[(n_fft, seed)] = config

    def _build_predictions(
        self, split: str, rows: Sequence[dict[str, Any]], conditions: Sequence[str]
    ) -> None:
        prediction_rows = []
        for n_fft in self.nfft_values:
            for seed in self.seeds:
                epoch = int(self.run_configs[(n_fft, seed)]["selected_epoch"])
                for condition in conditions:
                    for position, row in enumerate(rows):
                        # deliberately imperfect, deterministic and class dependent
                        margin = (1.0 if position % 3 else -1.0) * (1.0 + 0.01 * position)
                        margin += 0.05 * (seed - 2026)
                        prediction_rows.append(
                            {
                                "split": split,
                                "condition": condition,
                                "n_fft": n_fft,
                                "hop_length": rsc.hop_for(n_fft),
                                "training_seed": seed,
                                "epoch": epoch,
                                "model_id": f"nfft{n_fft}_seed{seed}",
                                "index_position": position,
                                "file_path": row["file_path"],
                                "filename": row["filename"],
                                "target_code": row["target_code"],
                                "target_name": row["target_name"],
                                "distance_code": row["distance_code"],
                                "segment_code": row["segment_code"],
                                "row_split": row["split"],
                                "label": int(row["label"]),
                                "class_name": row["class_name"],
                                "block_index": int(row["block_index"]),
                                "window_index": int(row["window_index"]),
                                "start": int(row["start"]),
                                "length": int(row["length"]),
                                "source_samples": int(row["source_samples"]),
                                "logit_phantom": 0.0,
                                "logit_air2s": margin,
                                "logit_margin": margin,
                                "air2s_score": float(1.0 / (1.0 + np.exp(-margin))),
                                "prediction": int(margin > 0.0),
                            }
                        )
        name = (
            "heldout_window_predictions.csv"
            if split == "heldout"
            else "test_clean_window_predictions.csv"
        )
        rsc.write_csv(self.root / name, prediction_rows)

    def _build_cache_manifest(self) -> None:
        entries: dict[str, list[dict[str, Any]]] = {"heldout": [], "test_clean": []}
        pairing: dict[str, list[dict[str, Any]]] = {"heldout": [], "test_clean": []}
        for key, split, conditions, windows, seed in (
            ("heldout", "heldout", rsc.CONDITIONS, len(self.heldout_rows), rsc.HELDOUT_NOISE_SEED),
            ("test_clean", "test", ("clean",), len(self.test_rows), rsc.TEST_NOISE_SEED),
        ):
            for condition in conditions:
                pairing[key].append(
                    {
                        "split": split,
                        "condition": condition,
                        "noise_seed": seed,
                        "rng_rule": (
                            "np.random.default_rng(noise_seed + epoch*len(stage2_split) + "
                            "stage2_position), epoch pinned to 0 for validation, held-out and "
                            "old-test"
                        ),
                        "digest_sha256": f"pair-{split}-{condition}",
                    }
                )
                for n_fft in self.nfft_values:
                    entries[key].append(
                        {
                            "split": split,
                            "condition": condition,
                            "n_fft": n_fft,
                            "hop_length": rsc.hop_for(n_fft),
                            "noise_seed": seed,
                            "windows": windows,
                            "shape": [windows, 1, 256, 256],
                            "dtype": "float32",
                            "feature_sha256": f"feature-{split}-{condition}-{n_fft}",
                            "window_digests_concat_sha256": f"digest-{split}-{condition}-{n_fft}",
                            "window_digest_bytes": 32 * windows,
                            "input_probe_samples": rsc.INPUT_PROBE_SAMPLES,
                            "input_probe_sha256": f"probe-{split}-{condition}",
                        }
                    )
        rsc.write_json(
            self.root / "prediction_cache_manifest.json",
            {
                "heldout": entries["heldout"],
                "test_clean": entries["test_clean"],
                "noise_pairing": pairing,
                "disk_feature_cache_written": False,
                "truncated_debug_run": True,
                "window_digest_files": None,
            },
        )

    def _build_selection(self) -> None:
        per_model = []
        values: dict[int, list[float]] = {n_fft: [] for n_fft in self.nfft_values}
        for n_fft in self.nfft_values:
            for seed in self.seeds:
                selected = int(self.run_configs[(n_fft, seed)]["selected_epoch"])
                score = 0.5 + 0.1 * selected
                values[n_fft].append(score)
                per_model.append(
                    {
                        "n_fft": n_fft,
                        "hop_length": rsc.hop_for(n_fft),
                        "training_seed": seed,
                        "selected_epoch": selected,
                        "per_condition": {
                            condition: {"file_equal_balanced_accuracy": score}
                            for condition in rsc.CONDITIONS
                        },
                        "four_condition_mean_file_equal_balacc": score,
                        "four_condition_mean_pooled_balacc": score,
                        "validation_metrics_sha256": "x",
                        "selection_json_sha256": "y",
                    }
                )
        per_nfft = [
            {
                "n_fft": n_fft,
                "hop_length": rsc.hop_for(n_fft),
                "training_seeds": list(self.seeds),
                "mean_file_equal_balacc_over_seeds": float(np.mean(values[n_fft])),
                "std_file_equal_balacc_over_seeds": 0.0,
                "mean_pooled_balacc_over_seeds": float(np.mean(values[n_fft])),
                "per_seed_file_equal_balacc": {
                    str(seed): value for seed, value in zip(self.seeds, values[n_fft])
                },
            }
            for n_fft in self.nfft_values
        ]
        selected = max(
            per_nfft, key=lambda row: (row["mean_file_equal_balacc_over_seeds"], -row["n_fft"])
        )
        selection = {
            "task": "STAGE2-BASE",
            "selection_rule": rsc.NFFT_SELECTION_RULE,
            "tie_break": rsc.NFFT_TIE_BREAK,
            "evidence_source": "old validation split only",
            "conditions": list(rsc.CONDITIONS),
            "per_model": per_model,
            "per_nfft": per_nfft,
            "selected_n_fft": selected["n_fft"],
            "selected_hop_length": rsc.hop_for(selected["n_fft"]),
            "selection_margin": 0.0,
            "exact_tie": False,
            "partial_grid": False,
        }
        rsc.write_json(self.root / "validation_nfft_selection.json", selection)
        digest = rsc.sha256_file(self.root / "validation_nfft_selection.json")
        rsc.write_json(
            self.root / "selection_lock.json",
            {"selection_sha256": digest, "selected_n_fft": selected["n_fft"]},
        )
        rsc.write_json(
            self.root / "heldout_generation_record.json",
            {
                "selection_sha256_before_evaluation": digest,
                "selection_sha256_after_evaluation": digest,
                "selection_unchanged": True,
            },
        )

    def _build_tables(self) -> None:
        merged: dict[str, list[dict[str, Any]]] = {
            "model_condition_metrics": [],
            "per_file_metrics": [],
            "nfft_summary": [],
            "paired_nfft_comparison": [],
        }
        for split, name in (
            ("heldout", "heldout_window_predictions.csv"),
            ("test", "test_clean_window_predictions.csv"),
        ):
            predictions = [rsc.cast_prediction_row(row) for row in rsc.read_csv_rows(self.root / name)]
            tables = rsc.summarize_predictions(
                predictions, split, output_root=self.root, require_both_classes=False
            )
            for key, rows in tables.items():
                merged[key].extend(rows)
            if split == "test":
                rsc.write_csv(
                    self.root / "test_clean_model_condition_metrics.csv",
                    tables["model_condition_metrics"],
                )
        rsc.write_csv(self.root / "model_condition_metrics.csv", merged["model_condition_metrics"])
        rsc.write_csv(self.root / "per_file_metrics.csv", merged["per_file_metrics"])
        rsc.write_csv(self.root / "nfft_summary.csv", merged["nfft_summary"])
        rsc.write_csv(self.root / "paired_nfft_comparison.csv", merged["paired_nfft_comparison"])
        rsc.write_json(
            self.root / "frontend_check.json",
            {
                str(n_fft): {
                    "n_fft": n_fft,
                    "hop_length": rsc.hop_for(n_fft),
                    "output_shape": [2, 1, 256, 256],
                    "finite": True,
                }
                for n_fft in self.nfft_values
            },
        )

    def _build_figures(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        windows = {}
        for code, name in (("0010", "representative_phantom.png"), ("0101", "representative_air2s.png")):
            figure, axis = plt.subplots(figsize=(5, 4))
            axis.imshow(np.arange(256 * 256).reshape(256, 256))
            figure.savefig(self.root / name, dpi=60)
            plt.close(figure)
            windows[code] = {
                "class_label": rsc.CODE_TO_LABEL[code],
                "class_name": rsc.LABEL_TO_NAME[rsc.CODE_TO_LABEL[code]],
                "filename": "fixture.mat",
                "figure": name,
                "figure_sha256": rsc.sha256_file(self.root / name),
                "n_fft_detail": {
                    str(n_fft): {
                        "model_input_shape": [256, 256],
                        "model_input_finite": True,
                    }
                    for n_fft in self.nfft_values
                },
            }
        rsc.write_json(
            self.root / "figure_evidence.json",
            {
                "rule": rsc.REPRESENTATIVE_WINDOW_RULE,
                "per_window_png_written": False,
                "windows": windows,
            },
        )

    def _build_manifest_and_report(self) -> None:
        (self.root / "report.md").write_text(
            "oracle 第二阶段结果；背景窗口不属于本任务；不是级联三分类；"
            "不能声称跨会话泛化；模拟 AWGN 不等于现场 SNR；30,240 行与 1,800 行。\n",
            encoding="utf-8",
        )
        published = {}
        for name in (
            "validation_nfft_selection.json",
            "selection_lock.json",
            "heldout_window_predictions.csv",
            "test_clean_window_predictions.csv",
            "model_condition_metrics.csv",
            "per_file_metrics.csv",
            "nfft_summary.csv",
            "paired_nfft_comparison.csv",
            "prediction_cache_manifest.json",
            "heldout_generation_record.json",
            "figure_evidence.json",
            "frontend_check.json",
            "report.md",
        ):
            path = self.root / name
            published[name] = {"bytes": path.stat().st_size, "sha256": rsc.sha256_file(path)}
        rsc.write_json(
            self.root / "experiment_manifest.json",
            {
                "scope": {"not_a_cascade": True, "three_class_performance_claimed": False},
                "design": {
                    "heldout_windows": len(self.heldout_rows),
                    "test_windows": len(self.test_rows),
                    "expected_heldout_rows": 0,
                    "expected_test_rows": 0,
                    "hop_rule": "hop_length = n_fft // 2 (exactly 50%)",
                },
                "selection": {"sha256": rsc.sha256_file(self.root / "validation_nfft_selection.json")},
                "runs_completed": 4,
                "runs_expected": 4,
                "published_files": published,
            },
        )

    # -- helpers ----------------------------------------------------------
    def run_verifier(self) -> dict[str, Any]:
        return vsc.verify_outputs(
            self.root,
            fixture=True,
            nfft_values=self.nfft_values,
            seeds=self.seeds,
            heldout_windows=len(self.heldout_rows),
            test_windows=len(self.test_rows),
            check_frozen=False,
            heldout_index=self.heldout_index,
            old_index=self.old_index,
        )

    # -- tests ------------------------------------------------------------
    def test_consistent_fixture_passes(self) -> None:
        report = self.run_verifier()
        self.assertEqual(report["failures"], [])
        self.assertTrue(report["passed"])
        self.assertEqual(report["checks"]["metrics"]["table_rows"], 16)
        self.assertTrue(report["checks"]["cache"]["heldout_probe_identical"])
    def test_tampered_prediction_is_rejected(self) -> None:
        path = self.root / "heldout_window_predictions.csv"
        rows = rsc.read_csv_rows(path)
        rows[0]["prediction"] = str(1 - int(rows[0]["prediction"]))
        rsc.write_csv(path, rows)
        report = self.run_verifier()
        self.assertFalse(report["passed"])
        self.assertTrue(
            any("identity/rule/logit" in failure for failure in report["failures"]),
            report["failures"],
        )

    def test_tampered_epoch_is_rejected(self) -> None:
        path = self.root / "heldout_window_predictions.csv"
        rows = rsc.read_csv_rows(path)
        rows[0]["epoch"] = "99"
        rsc.write_csv(path, rows)
        report = self.run_verifier()
        self.assertFalse(report["passed"])
        self.assertTrue(
            any("epoch" in failure for failure in report["failures"]), report["failures"]
        )

    def test_tampered_summary_is_rejected(self) -> None:
        path = self.root / "nfft_summary.csv"
        rows = rsc.read_csv_rows(path)
        rows[0]["file_equal_balacc_mean"] = "0.123456"
        rsc.write_csv(path, rows)
        report = self.run_verifier()
        self.assertFalse(report["passed"])
        self.assertTrue(
            any("nfft_summary.csv" in failure for failure in report["failures"]),
            report["failures"],
        )

    def test_tampered_selection_is_rejected(self) -> None:
        path = self.root / "validation_nfft_selection.json"
        payload = rsc.read_json(path)
        payload["selected_n_fft"] = 4096
        rsc.write_json(path, payload)
        report = self.run_verifier()
        self.assertFalse(report["passed"])
        self.assertTrue(
            any("configuration choice" in failure or "selection" in failure for failure in report["failures"]),
            report["failures"],
        )

    def test_selection_mentioning_another_split_is_rejected(self) -> None:
        path = self.root / "validation_nfft_selection.json"
        payload = rsc.read_json(path)
        payload["note"] = "sanity check on the heldout split"
        rsc.write_json(path, payload)
        report = self.run_verifier()
        self.assertFalse(report["passed"])
        self.assertTrue(
            any("mentions" in failure for failure in report["failures"]), report["failures"]
        )

    def test_broken_pairing_is_rejected(self) -> None:
        path = self.root / "prediction_cache_manifest.json"
        payload = rsc.read_json(path)
        payload["heldout"][1]["input_probe_sha256"] = "different"
        rsc.write_json(path, payload)
        report = self.run_verifier()
        self.assertFalse(report["passed"])
        self.assertTrue(
            any("different IQ" in failure or "probe" in failure for failure in report["failures"]),
            report["failures"],
        )

    def test_unreadable_figure_is_rejected(self) -> None:
        (self.root / "representative_phantom.png").write_bytes(b"not a png")
        report = self.run_verifier()
        self.assertFalse(report["passed"])
        self.assertTrue(
            any("PNG" in failure for failure in report["failures"]), report["failures"]
        )

    def test_png_reader_accepts_a_real_figure(self) -> None:
        width, height = vsc.read_png_dimensions(self.root / "representative_air2s.png")
        self.assertGreater(width, 100)
        self.assertGreater(height, 100)


# ---------------------------------------------------------------------------
# bounded end-to-end smoke run
# ---------------------------------------------------------------------------
def smoke_run(windows: int = 3, train_samples: int = 8, val_samples: int = 8) -> dict[str, Any]:
    """Drive every CLI command over two deliberately tiny runs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with scratch_directory("smoke") as root:
        stages: list[dict[str, Any]] = []

        def call(*argv: str) -> int:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = rsc.main([*argv, "--output-dir", str(root), "--device", device])
            stages.append({"argv": list(argv), "returncode": code, "tail": buffer.getvalue()[-200:]})
            return code

        call("frontends")
        for n_fft in rsc.NFFT_GRID:
            call(
                "train",
                "--n-fft",
                str(n_fft),
                "--seed",
                "2026",
                "--epochs",
                "1",
                "--max-train-samples",
                str(train_samples),
                "--max-val-samples",
                str(val_samples),
                "--smoke",
            )
        call("select-nfft")
        call("figures")
        call("evaluate", "--max-windows", str(windows))
        call("summarize")
        call("manifest")
        call("report")

        heldout_rows = rsc.read_csv_rows(root / "heldout_window_predictions.csv")
        cache = rsc.read_json(root / "prediction_cache_manifest.json")
        selection_text = (root / "validation_nfft_selection.json").read_text(encoding="utf-8")
        probes = {
            (row["split"], row["condition"]): row["input_probe_sha256"]
            for row in cache["heldout"] + cache["test_clean"]
        }
        per_key: dict[tuple[str, str], set[str]] = {}
        for row in cache["heldout"] + cache["test_clean"]:
            per_key.setdefault((row["split"], row["condition"]), set()).add(
                row["input_probe_sha256"]
            )
        return {
            "root": str(root),
            "stages": stages,
            "returncodes": [stage["returncode"] for stage in stages],
            "heldout_rows": len(heldout_rows),
            "truncated": cache["truncated_debug_run"],
            "pairing_identical": all(len(values) == 1 for values in per_key.values()),
            "probe_conditions": len(probes),
            "probe_groups": len(per_key),
            "selection_tokens": [
                token
                for token in vsc.SELECTION_FORBIDDEN_TOKENS
                if token.lower() in selection_text.lower()
            ],
            "selection": rsc.read_json(root / "validation_nfft_selection.json")["selected_n_fft"],
            "manifest": rsc.read_json(root / "experiment_manifest.json")["runs_completed"],
            "figures": sorted(path.name for path in root.glob("representative_*.png")),
            "evaluated_windows_not_full_split": len(heldout_rows)
            < rsc.EXPECTED_SPLIT_WINDOWS["heldout"],
        }


class SmokeRunTests(unittest.TestCase):
    def test_bounded_smoke_reaches_every_stage(self) -> None:
        summary = smoke_run()
        self.assertEqual(summary["returncodes"], [0] * len(summary["returncodes"]))
        self.assertTrue(summary["truncated"])
        self.assertTrue(summary["evaluated_windows_not_full_split"])
        self.assertTrue(summary["pairing_identical"])
        self.assertEqual(summary["selection_tokens"], [])
        self.assertEqual(summary["manifest"], 2)
        self.assertEqual(len(summary["figures"]), 2)
        # 2 n_fft x 1 seed x 4 conditions x 3 windows
        self.assertEqual(summary["heldout_rows"], 2 * 4 * 3)
        # one (split, condition) probe group per held-out condition plus test clean
        self.assertEqual(summary["probe_groups"], len(rsc.CONDITIONS) + 1)
        self.assertEqual(summary["probe_conditions"], len(rsc.CONDITIONS) + 1)


# ---------------------------------------------------------------------------
# independent recomputation over the real root
# ---------------------------------------------------------------------------
def verify_outputs() -> dict[str, Any]:
    report = vsc.verify_outputs(OUTPUT_ROOT)
    target = OUTPUT_ROOT / "verification.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run the bounded smoke run")
    parser.add_argument(
        "--verify-outputs",
        action="store_true",
        help="run the independent verifier over the real output root",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    failures = 0
    if args.verify_outputs:
        print("independent recompute of the produced artefacts")
        report = verify_outputs()
        for failure in report["failures"]:
            print(f"FAIL: {failure}")
        print(f"verifier passed={report['passed']} failures={len(report['failures'])}")
        failures += len(report["failures"])
    if args.smoke:
        print("bounded smoke run")
        summary = smoke_run()
        print(
            json.dumps(
                {
                    "returncodes": summary["returncodes"],
                    "heldout_rows": summary["heldout_rows"],
                    "pairing_identical": summary["pairing_identical"],
                    "selection": summary["selection"],
                    "selection_tokens": summary["selection_tokens"],
                    "figures": summary["figures"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        failures += sum(1 for code in summary["returncodes"] if code)
    result = unittest.main(
        argv=[sys.argv[0]], exit=False, module=sys.modules[__name__], verbosity=2
    ).result
    failures += len(result.failures) + len(result.errors)
    print(f"failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
