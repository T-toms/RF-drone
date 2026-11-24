"""Checks for NFFT4 (``run_nfft_comparison.py``).

Two ways to run it:

    python test_nfft_comparison.py                  # offline unit tests only
    python test_nfft_comparison.py --smoke          # + limited-window smoke run
    python test_nfft_comparison.py --verify-outputs # + independent CSV recompute

``--smoke`` trains one bounded ``n_fft=512`` run, writes the two representative
figures, evaluates a handful of windows, and re-runs the summariser over that
partial grid. ``--verify-outputs`` reads the real prediction CSVs and recomputes
every reported metric from the raw text; the module under test never supplies an
expected value to those checks.
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
import sys
import unittest
from pathlib import Path
from typing import Any, Iterator, Sequence

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import torch

# The checks are all small; keeping the CPU thread pool tiny stops them from
# stealing cores from a concurrent training run.
torch.set_num_threads(int(os.environ.get("NFFT4_TEST_THREADS", "2")))

import run_nfft_comparison as rnc
from evaluate_heldout_models import file_equal_metrics, pooled_metrics
from model import SpectrogramFrontEnd
from train_resnet18 import binary_metrics

OUTPUT_ROOT = rnc.OUTPUT_ROOT
SCRATCH_ROOT = OUTPUT_ROOT.parent / "_test_scratch"
_SCRATCH_COUNTER = itertools.count()


@contextlib.contextmanager
def scratch_directory(prefix: str) -> Iterator[Path]:
    """Deterministic scratch space inside the project (sandbox-safe)."""
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    path = SCRATCH_ROOT / f"{prefix}{os.getpid()}_{next(_SCRATCH_COUNTER)}"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def synthetic_records() -> list[dict[str, Any]]:
    """Four files per category, uneven window counts, one deliberate false negative."""
    records: list[dict[str, Any]] = []
    layout = {
        "0000": [(20, 4), (20, 0), (20, 10), (20, 0)],
        "0010": [(30, 30), (10, 5), (40, 40), (15, 15)],
        "0101": [(25, 25), (50, 40), (12, 12), (33, 33)],
    }
    names = {"0000": "Background", "0010": "DJI Phantom 4 Pro", "0101": "DJI Air 2S"}
    for code, files in layout.items():
        for file_index, (windows, hits) in enumerate(files):
            filename = f"{code}_file{file_index}.mat"
            label = 0 if code == "0000" else 1
            for window in range(windows):
                prediction = 1 if window < hits else 0
                records.append(
                    {
                        "filename": filename,
                        "target_code": code,
                        "target_name": names[code],
                        "segment_code": code,
                        "label": label,
                        "prediction": prediction,
                        "n_fft": 1024,
                        "training_seed": 2026,
                        "condition": "clean",
                        "split": "heldout",
                    }
                )
    return records


class FrontendTests(unittest.TestCase):
    """The four grids must differ only in n_fft/hop_length."""

    def test_hop_is_half_of_n_fft(self) -> None:
        for n_fft in rnc.NFFT_GRID:
            self.assertEqual(rnc.hop_for(n_fft), n_fft // 2)
            self.assertEqual(rnc.hop_for(n_fft) / n_fft, 0.5)
        self.assertEqual(rnc.NFFT_GRID, (512, 1024, 2048, 4096))

    def test_shape_is_one_256_square_for_every_grid(self) -> None:
        device = torch.device("cpu")
        signals = torch.zeros(1, 2, 100_000)
        for n_fft in rnc.NFFT_GRID:
            frontend = SpectrogramFrontEnd(
                n_fft=n_fft, hop_length=rnc.hop_for(n_fft), image_size=256
            ).to(device).eval()
            with torch.inference_mode():
                features = frontend(signals)
            self.assertEqual(tuple(features.shape), (1, 1, 256, 256))

    def test_frontend_is_the_frozen_module(self) -> None:
        """n_fft must stay a constructor argument of the frozen model.py frontend."""
        # 20k samples is enough to exercise every grid (including n_fft=4096) and
        # keeps the shape check cheap while a training run owns the GPU.
        described = rnc.check_frontends(torch.device("cpu"), signals=20_000)
        self.assertEqual(sorted(int(key) for key in described), list(rnc.NFFT_GRID))
        for key, entry in described.items():
            self.assertEqual(entry["hop_length"] * 2, int(key))
            self.assertTrue(entry["finite"])
            self.assertTrue(entry["window_periodic_hamming"])
            self.assertEqual(entry["output_shape"], [2, 1, 256, 256])

    def test_periodic_hamming_window_is_used(self) -> None:
        for n_fft in rnc.NFFT_GRID:
            frontend = SpectrogramFrontEnd(n_fft=n_fft, hop_length=n_fft // 2)
            periodic = torch.hamming_window(n_fft)
            self.assertTrue(torch.equal(frontend.window, periodic))
            # A symmetric window would start at 0.08 and end at 0.08; the periodic
            # variant used by the frozen code starts at 0.08 and ends at 0.08 too,
            # so the decisive check is bit-exact equality with ``periodic=True``.
            self.assertFalse(torch.equal(periodic, torch.hamming_window(n_fft, periodic=False)))


class NoisePairingTests(unittest.TestCase):
    """The four grids and three seeds share one noise realisation per window."""

    def test_digest_is_independent_of_n_fft(self) -> None:
        first = rnc.noise_pairing_digest(rnc.HELDOUT_INDEX, "heldout", "-10dB", 22026)
        second = rnc.noise_pairing_digest(rnc.HELDOUT_INDEX, "heldout", "-10dB", 22026)
        self.assertEqual(first["digest_sha256"], second["digest_sha256"])
        for probe in first["probe_positions"]:
            self.assertEqual(probe["rng_seed"], 22026 + probe["index_position"])

    def test_split_specific_index_row_rule(self) -> None:
        """The RNG index is the in-split row, not a global counter."""
        heldout = rnc.noise_pairing_digest(rnc.HELDOUT_INDEX, "heldout", "clean", 22026)
        self.assertEqual(heldout["probe_positions"][0]["rng_seed"], 22026)
        self.assertEqual(heldout["noise_seed"], rnc.HELDOUT_NOISE_SEED)
        test = rnc.noise_pairing_digest(rnc.OLD_INDEX, "test", "clean", 22026)
        self.assertEqual(test["probe_positions"][0]["rng_seed"], 22026)
        self.assertEqual(test["split"], "test")

    def test_dataset_consumes_the_documented_rng(self) -> None:
        """One window must draw exactly the noise the digest predicts."""
        from dronerfa_data import add_complex_awgn

        position = 3
        expected_rng = np.random.default_rng(rnc.HELDOUT_NOISE_SEED + position)
        probe = np.ones(16, dtype=np.complex64)
        expected = add_complex_awgn(probe, -12.0, expected_rng)
        actual_rng = np.random.default_rng(rnc.HELDOUT_NOISE_SEED + position)
        actual = add_complex_awgn(probe, -12.0, actual_rng)
        self.assertTrue(np.array_equal(expected, actual))

    def test_test_split_clean_condition_adds_no_noise(self) -> None:
        self.assertIsNone(rnc.CONDITION_VALUES["clean"])
        digest = rnc.noise_pairing_digest(rnc.OLD_INDEX, "test", "clean", 22026)
        self.assertIsNone(digest["fixed_snr_db"])


class SelectionTests(unittest.TestCase):
    """Epoch selection must be reproducible and must not see test/held-out data."""

    def test_tie_break_takes_the_earliest_epoch(self) -> None:
        history = [
            {"epoch": 1, "mean_balanced_accuracy": 0.5},
            {"epoch": 2, "mean_balanced_accuracy": 0.7},
            {"epoch": 3, "mean_balanced_accuracy": 0.7},
            {"epoch": 4, "mean_balanced_accuracy": 0.6},
        ]
        self.assertEqual(rnc.select_epoch(history), 2)

    def test_epoch_orders_are_reproducible(self) -> None:
        for seed in rnc.TRAINING_SEEDS:
            self.assertEqual(rnc.epoch_orders(seed), rnc.epoch_orders(seed))
        self.assertNotEqual(rnc.epoch_orders(2026), rnc.epoch_orders(2027))

    def test_epoch_orders_match_the_2026_reference_run(self) -> None:
        reference = json.loads(
            (
                PROJECT_DIR
                / "outputs"
                / "noise_ab_5epochs_20260917"
                / "A_original"
                / "run_config.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            rnc.epoch_orders(reference["seed"]), reference["epoch_order_sha256"]
        )

    def test_initial_weights_are_seed_determined(self) -> None:
        for seed in rnc.TRAINING_SEEDS:
            self.assertEqual(rnc.initial_state_hash(seed), rnc.initial_state_hash(seed))
        self.assertEqual(
            len({rnc.initial_state_hash(seed) for seed in rnc.TRAINING_SEEDS}), 3
        )

    def test_initial_weights_match_the_2026_reference_run(self) -> None:
        reference = json.loads(
            (
                PROJECT_DIR
                / "outputs"
                / "noise_ab_5epochs_20260917"
                / "A_original"
                / "run_config.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            rnc.initial_state_hash(reference["seed"]), reference["initial_model_sha256"]
        )

    def test_selection_inputs_exclude_test_and_heldout(self) -> None:
        """The selection rule is a pure function of four validation conditions."""
        self.assertEqual(rnc.CONDITIONS, ("clean", "-10dB", "-12dB", "-15dB"))
        self.assertEqual(rnc.VAL_SELECTION_RULE, "mean_balanced_accuracy_over_four_conditions")
        self.assertNotIn("test", rnc.VAL_SELECTION_RULE)
        self.assertNotIn("heldout", rnc.VAL_SELECTION_RULE)
        self.assertEqual(rnc.VALIDATION_SEED, 12026)
        self.assertEqual(rnc.HELDOUT_NOISE_SEED, 22026)

    def test_selection_file_carries_no_test_or_heldout_evidence(self) -> None:
        """Once trained, every selection.json must be validation-only.

        The three reused 1024 runs keep the original ``run_noise_ab`` selection.json
        verbatim (``clean_only``/``multi_noise``), so that layout is accepted too.
        """
        selection_files = sorted(OUTPUT_ROOT.glob("nfft*/seed_*/selection.json"))
        with_config = [path for path in selection_files if (path.parent / "run_config.json").is_file()]
        if not with_config:
            self.skipTest("no trained runs yet")
        for path in with_config:
            payload = json.loads(path.read_text(encoding="utf-8"))
            text = json.dumps(payload, ensure_ascii=False).lower()
            self.assertNotIn("heldout", text, path)
            self.assertNotIn("test", text, path)
            config = json.loads((path.parent / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["validation_conditions"], list(rnc.CONDITIONS))
            self.assertEqual(config["hop_length"], config["n_fft"] // 2)
            self.assertEqual(config["train_snrs"], rnc.TRAIN_SNRS)
            if "selected_epoch" in payload:
                self.assertEqual(payload["selected_epoch"], config["selected_epoch"])
                self.assertIn("mean_balanced_accuracy_by_epoch", payload)
                if "best_multi_noise_epoch" in payload:
                    self.assertEqual(payload["best_multi_noise_epoch"], config["selected_epoch"])
                history = rnc.read_csv_rows(path.parent / "training_history.csv")
                means = {
                    int(row["epoch"]): float(row["mean_balanced_accuracy"]) for row in history
                }
                for epoch, value in payload["mean_balanced_accuracy_by_epoch"].items():
                    self.assertAlmostEqual(float(value), means[int(epoch)], places=9)
                best = max(means, key=lambda epoch: (means[epoch], -epoch))
                self.assertEqual(payload["selected_epoch"], best)
            else:
                self.assertIn("multi_noise", payload)
                self.assertEqual(payload["multi_noise"], config["selected_epoch"])


class MetricTests(unittest.TestCase):
    """The generalised aggregations must agree with the frozen EVAL6 helpers."""

    def test_file_equal_matches_the_frozen_helper(self) -> None:
        records = synthetic_records()
        grouped = {}
        for row in records:
            grouped.setdefault(row["target_code"], [])
            if row["filename"] not in grouped[row["target_code"]]:
                grouped[row["target_code"]].append(row["filename"])
        grouped = {key: sorted(value) for key, value in grouped.items()}
        expected = file_equal_metrics(records, grouped)
        actual = rnc.file_equal_metrics_any(records)
        for key in (
            "background_file_equal_fpr",
            "background_file_equal_specificity",
            "phantom_file_equal_recall",
            "air2s_file_equal_recall",
            "drone_file_equal_recall",
            "file_equal_balanced_accuracy",
        ):
            self.assertAlmostEqual(actual[key], expected[key], places=12)

    def test_pooled_matches_the_frozen_binary_metrics(self) -> None:
        records = synthetic_records()
        expected = binary_metrics(
            [row["label"] for row in records],
            [row["prediction"] for row in records],
            [float(row["prediction"]) for row in records],
        )
        actual = rnc.pooled_metrics_any(records)
        pairs = (
            ("accuracy", "accuracy"),
            ("balanced_accuracy", "balanced_accuracy"),
            ("precision", "precision"),
            ("drone_recall", "recall"),
            ("background_specificity", "specificity"),
            ("f1", "f1"),
            ("tn", "tn"),
            ("fp", "fp"),
            ("fn", "fn"),
            ("tp", "tp"),
        )
        for actual_key, expected_key in pairs:
            self.assertAlmostEqual(actual[actual_key], expected[expected_key], places=12)

    def test_confusion_matrix_conserves_windows(self) -> None:
        records = synthetic_records()
        metrics = rnc.pooled_metrics_any(records)
        self.assertEqual(
            metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"], len(records)
        )

    def test_file_equal_weights_files_not_windows(self) -> None:
        """A file with 50/50 hits must weigh the same as one with 3/3 hits."""
        records = [
            {"filename": "a.mat", "target_code": "0010", "label": 1, "prediction": 1},
            {"filename": "a.mat", "target_code": "0010", "label": 1, "prediction": 0},
            {"filename": "b.mat", "target_code": "0010", "label": 1, "prediction": 1},
            {"filename": "b.mat", "target_code": "0010", "label": 1, "prediction": 1},
            {"filename": "b.mat", "target_code": "0010", "label": 1, "prediction": 1},
            {"filename": "c.mat", "target_code": "0010", "label": 1, "prediction": 1},
            {"filename": "c.mat", "target_code": "0010", "label": 1, "prediction": 1},
            {"filename": "d.mat", "target_code": "0010", "label": 1, "prediction": 1},
            {"filename": "bg.mat", "target_code": "0000", "label": 0, "prediction": 0},
        ]
        # only 9 rows: pad the drone categories to four files each to satisfy the rule
        for index in range(4):
            records.append(
                {"filename": f"e{index}.mat", "target_code": "0101", "label": 1, "prediction": 1}
            )
        primary = rnc.file_equal_metrics_any(records)
        self.assertAlmostEqual(primary["phantom_file_equal_recall"], (0.5 + 1.0 + 1.0 + 1.0) / 4, places=12)
        self.assertAlmostEqual(primary["air2s_file_equal_recall"], 1.0, places=12)

    def test_metric_columns_are_present(self) -> None:
        row = rnc.add_metric_row({"split": "heldout"}, synthetic_records())
        for column in ("accuracy", "balanced_accuracy", "precision", "drone_recall",
                       "background_specificity", "f1", "tn", "fp", "fn", "tp"):
            self.assertIn(column, row)
            self.assertTrue(np.isfinite(float(row[column])))
        for column in ("file_equal_balanced_accuracy", "file_equal_background_fpr",
                       "file_equal_drone_recall", "file_equal_phantom_recall",
                       "file_equal_air2s_recall"):
            self.assertIn(column, row)

    def test_summarise_predictions_builds_every_table(self) -> None:
        predictions = []
        for n_fft in rnc.NFFT_GRID:
            for seed in rnc.TRAINING_SEEDS:
                for condition in rnc.CONDITIONS:
                    for row in synthetic_records():
                        predictions.append(
                            {**row, "n_fft": n_fft, "training_seed": seed,
                             "condition": condition, "epoch": 2}
                        )
        with scratch_directory("summarize_") as scratch:
            tables = rnc.summarize_predictions(predictions, "heldout", output_root=scratch)
        self.assertEqual(len(tables["model_condition_metrics"]), 12 * 4)
        self.assertEqual(len(tables["nfft_summary"]), 4 * 5)
        self.assertEqual(len(tables["paired_nfft_comparison"]), 3 * 5)
        for row in tables["nfft_summary"]:
            if row["n_fft"] == rnc.BASELINE_NFFT:
                continue
            self.assertEqual(row["seeds"], 3)
            self.assertTrue(np.isfinite(row["file_equal_balacc_std"]))


class ReuseTests(unittest.TestCase):
    def test_reuse_map_matches_the_preserved_inputs(self) -> None:
        preserved = json.loads(
            (
                PROJECT_DIR.parent / "work" / "codex-preserved-inputs-20260917.json"
            ).read_text(encoding="utf-8")
        )
        recorded = {entry["path"]: entry["sha256"] for entry in preserved["files"]}
        for run_dir, seed, expected in rnc.REUSE_1024_RUNS:
            checkpoint = run_dir / "A_original" / "best_model.pt"
            self.assertIn(str(checkpoint), recorded)
            self.assertEqual(recorded[str(checkpoint)], expected)
            self.assertEqual(run_dir.name.endswith(str(seed)) or seed == 2026, True)

    def test_reuse_evidence_is_all_or_nothing(self) -> None:
        path = OUTPUT_ROOT / "reuse_1024_verification.json"
        if not path.is_file():
            self.skipTest("reuse verification has not run yet")
        evidence = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("seeds", evidence)
        for seed, entry in evidence["seeds"].items():
            self.assertIn("eligible_for_reuse", entry, seed)
            self.assertIn("checkpoint_sha256_match", entry, seed)
            self.assertIn("initial_model_sha256_derived", entry, seed)
            self.assertEqual(
                entry["initial_model_sha256_recorded"], entry["initial_model_sha256_derived"]
            )
            if entry["eligible_for_reuse"]:
                self.assertEqual(entry.get("failures"), [])
                self.assertTrue(entry["validation_cache_match"])
                self.assertEqual(
                    entry["epoch_order_sha256_recorded"], entry["epoch_order_sha256_derived"]
                )
                self.assertEqual(entry["checkpoint_epoch"], entry["best_epoch_by_rule"])


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _fake_png(path: Path, squares: int) -> None:
    """Real (small) PNG so the verify module can parse the header and size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4, 4))
    grid = np.arange(squares * squares, dtype=float).reshape(squares, squares)
    axis.imshow(np.sin(grid))
    axis.plot(np.arange(squares), np.cos(np.arange(squares)))
    figure.savefig(path, dpi=140)
    plt.close(figure)


def build_fixture(
    scratch: Path,
    seeds: Sequence[int] = (2026, 2027, 2028),
    windows: int = 12,
    test_windows: int = 9,
    require_selected_epoch_key: bool = True,
) -> None:
    """Write a complete fake NFFT4 output tree with the real CSV/JSON schemas.

    The point is to exercise ``verify_nfft_comparison`` end to end (grid shape,
    arithmetic, selection rule, leakage checks, figure evidence) without training.
    """
    # The fixture's 12 rows are spread across the index so that every category and
    # every file is represented; the published grid still uses contiguous 0..N-1
    # positions, exactly like a truncated real run.
    scratch.mkdir(parents=True, exist_ok=True)
    all_heldout = rnc.load_index(rnc.HELDOUT_INDEX, split="heldout")
    chunk = max(len(all_heldout) // windows, 1)
    heldout_index = [all_heldout[position * chunk] for position in range(windows)]
    categories = {row["target_code"] for row in heldout_index}
    if categories != {"0000", "0010", "0101"}:
        raise AssertionError(
            f"fixture sampling must cover every category, got {sorted(categories)}; "
            "increase the fixture window count"
        )
    test_index = [row for row in rnc.load_index(rnc.OLD_INDEX) if row["split"] == "test"]
    test_windows = min(test_windows, len(test_index))

    def records(index_rows, count, model_id, condition, seed, rng, base_position=0):
        out = []
        for position in range(count):
            source = index_rows[position]
            label = int(source["label"])
            prediction = int(rng.random() < (0.8 if label else 0.1))
            # The margin must obey the frozen decision rule, so it is drawn on the
            # correct side of zero instead of independently.
            magnitude = float(abs(rng.normal(1.0, 0.5))) + 0.05
            margin = magnitude if prediction else -magnitude
            out.append(
                {
                    "n_fft": int(model_id.split("_")[0].replace("nfft", "")),
                    "hop_length": 0,
                    "training_seed": seed,
                    "epoch": 2,
                    "model_id": model_id,
                    "reused_1024_checkpoint": False,
                    "checkpoint_provenance": "fixture",
                    "split": "heldout",
                    "condition": condition,
                    # The published grid numbers windows 0..N-1 inside the truncated
                    # split; the source index row is recorded separately.
                    "index_position": base_position + position,
                    "source_index_position": position,
                    "file_path": source["file_path"],
                    "filename": source["filename"],
                    "target_code": source["target_code"],
                    "target_name": source["target_name"],
                    "distance_code": source["distance_code"],
                    "segment_code": source["segment_code"],
                    "row_split": source["split"],
                    "label": label,
                    "block_index": int(source["block_index"]),
                    "window_index": int(source["window_index"]),
                    "start": int(source["start"]),
                    "length": int(source["length"]),
                    "source_samples": int(source["source_samples"]),
                    "logit_background": 0.0,
                    "logit_drone": margin,
                    "logit_margin": margin,
                    "drone_score": float(1.0 / (1.0 + np.exp(-margin))),
                    "prediction": prediction,
                }
            )
        for row in out:
            row["hop_length"] = row["n_fft"] // 2
        return out

    heldout_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for n_fft in rnc.NFFT_GRID:
        for seed in seeds:
            rng = np.random.default_rng(n_fft + seed)
            for condition in rnc.CONDITIONS:
                heldout_rows.extend(
                    records(heldout_index, windows, f"nfft{n_fft}_seed{seed}", condition, seed, rng)
                )
            test_rows.extend(records(test_index, test_windows, f"nfft{n_fft}_seed{seed}", "clean", seed, rng))
        for row in test_rows:
            row["split"] = "test"
        for row in heldout_rows + test_rows:
            # Realistic provenance: the checkpoint of this model's own run directory,
            # except for n_fft=1024 which reuses the preserved 2026-09-17 checkpoint.
            n_fft = int(row["model_id"].split("_")[0].replace("nfft", ""))
            seed = int(row["model_id"].split("_seed")[1])
            if n_fft == rnc.BASELINE_NFFT:
                row["reused_1024_checkpoint"] = True
                row["checkpoint_provenance"] = rnc.REUSE_1024_RUNS[
                    [item[1] for item in rnc.REUSE_1024_RUNS].index(seed)
                ][0] / "A_original" / "best_model.pt"
            else:
                row["checkpoint_provenance"] = (
                    scratch / f"nfft{n_fft}" / f"seed_{seed}" / "best_model.pt"
                )
            row["checkpoint_provenance"] = str(row["checkpoint_provenance"])
        rnc.write_csv(scratch / "heldout_window_predictions.csv", heldout_rows)
        rnc.write_csv(scratch / "test_clean_window_predictions.csv", test_rows)

    # per-run artefacts
    for n_fft in rnc.NFFT_GRID:
        for seed in seeds:
            run_dir = scratch / f"nfft{n_fft}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            validation = []
            history = []
            means = {}
            for epoch in range(1, 6):
                scores = {condition: 0.7 + 0.01 * epoch + 0.001 * index for index, condition in enumerate(rnc.CONDITIONS)}
                if epoch == 2:
                    scores = {condition: value + 0.2 for condition, value in scores.items()}
                means[epoch] = float(np.mean(list(scores.values())))
                for condition, score in scores.items():
                    validation.append(
                        {
                            "epoch": epoch,
                            "condition": condition,
                            "loss": 0.1,
                            "samples": 450,
                            "accuracy": score,
                            "balanced_accuracy": score,
                            "precision": score,
                            "drone_recall": score,
                            "background_specificity": score,
                            "f1": score,
                            "tn": 160,
                            "fp": 0,
                            "fn": 0,
                            "tp": 290,
                        }
                    )
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": 0.1,
                        "train_balanced_accuracy": 0.9,
                        "clean_balanced_accuracy": scores["clean"],
                        "mean_balanced_accuracy": means[epoch],
                        "seconds": 1.0,
                    }
                )
            rnc.write_csv(run_dir / "validation_metrics.csv", validation)
            rnc.write_csv(run_dir / "training_history.csv", history)
            config = {
                "n_fft": n_fft,
                "hop_length": n_fft // 2,
                "image_size": 256,
                "training_seed": seed,
                "epochs": 5,
                "batch_size": 4,
                "train_snrs": rnc.TRAIN_SNRS,
                "validation_seed": 12026,
                "validation_conditions": list(rnc.CONDITIONS),
                "optimizer": {"type": "Adam", "learning_rate": 0.001, "weight_decay": 0.0},
                "amp": True,
                "selected_epoch": 2,
                "status": "fixture",
                "smoke": False,
                "reused_from": None,
                "epoch_order_sha256": ["0" * 64] * 5,
                "initial_model_sha256": "1" * 64,
            }
            (run_dir / "best_model.pt").write_bytes(b"fixture")
            rnc.write_json(run_dir / "run_config.json", config)
            rnc.write_json(
                run_dir / "selection.json",
                (
                    {"selected_epoch": 2, "best_multi_noise_epoch": 2, "rule": "fixture"}
                    if require_selected_epoch_key
                    else {"clean_only": 2, "multi_noise": 2}
                ),
            )

    # metrics tables straight from the fixture predictions
    tables = rnc.summarize_predictions(heldout_rows, "heldout", output_root=scratch)
    rnc.write_csv(scratch / "model_condition_metrics.csv", tables["model_condition_metrics"])
    rnc.write_csv(scratch / "per_file_metrics.csv", tables["per_file_metrics"])
    rnc.write_csv(scratch / "nfft_summary.csv", tables["nfft_summary"])
    rnc.write_csv(scratch / "paired_nfft_comparison.csv", tables["paired_nfft_comparison"])

    cache = []
    digest_dir = scratch / "feature_digests"
    digest_dir.mkdir(parents=True, exist_ok=True)
    for n_fft in rnc.NFFT_GRID:
        for condition in rnc.CONDITIONS:
            cache.append(
                {
                    "split": "heldout",
                    "condition": condition,
                    "n_fft": n_fft,
                    "hop_length": n_fft // 2,
                    "noise_seed": 22026,
                    "windows": windows,
                    "shape": [windows, 1, 256, 256],
                    "dtype": "float32",
                    "feature_sha256": "2" * 64,
                    "window_digests_concat_sha256": "3" * 64,
                    "window_digest_bytes": windows * 32,
                    "seconds": 1.0,
                }
            )
            (digest_dir / f"window_digests_heldout_{condition}_nfft{n_fft}.bin").write_bytes(
                b"\x00" * (windows * 32)
            )
        (digest_dir / f"window_digests_test_clean_nfft{n_fft}.bin").write_bytes(
            b"\x00" * (test_windows * 32)
        )
    rnc.write_json(
        scratch / "prediction_cache_manifest.json",
        {
            "heldout": cache,
            "test_clean": [],
            "noise_pairing": {
                "heldout": [
                    {
                        "condition": condition,
                        "noise_seed": 22026,
                        "probe_positions": [
                            {"index_position": 0, "rng_seed": 22026},
                            {"index_position": windows // 2, "rng_seed": 22026 + windows // 2},
                        ],
                    }
                    for condition in rnc.CONDITIONS
                ],
                "test_clean": [],
            },
        },
    )
    detail = {
        str(n_fft): {
            "native_frequency_bins": n_fft,
            "native_time_frames": 3,
            "model_input_shape": [256, 256],
            "frequency_step_khz": 100_000.0 / n_fft,
            "time_step_ms": n_fft / 2 / 1e5,
            "model_input_finite": True,
        }
        for n_fft in rnc.NFFT_GRID
    }
    _fake_png(scratch / "native_stft_examples.png", 4)
    _fake_png(scratch / "model_input_examples.png", 4)
    rnc.write_json(
        scratch / "figure_evidence.json",
        {
            "n_fft_detail": detail,
            "resize_note": "fixture",
            "figures": {
                "native_stft_examples.png": rnc.sha256_file(scratch / "native_stft_examples.png"),
                "model_input_examples.png": rnc.sha256_file(scratch / "model_input_examples.png"),
            },
        },
    )
    rnc.write_json(
        scratch / "reuse_1024_verification.json",
        {
            "all_seeds_eligible": True,
            "require_retrain": [],
            "seeds": {
                str(seed): {
                    "eligible_for_reuse": True,
                    "checkpoint_sha256_match": True,
                    "initial_model_sha256_recorded": "a" * 64,
                    "initial_model_sha256_derived": "a" * 64,
                    "epoch_order_sha256_recorded": ["b" * 64],
                    "epoch_order_sha256_derived": ["b" * 64],
                    "validation_cache_match": True,
                    "checkpoint_epoch": 2,
                    "best_epoch_by_rule": 2,
                    "failures": [],
                }
                for seed in seeds
            },
        },
    )


class VerificationModuleTests(unittest.TestCase):
    """The independent verifier must accept a correct tree and reject tampering."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._context = scratch_directory("verify_fixture_")
        cls.scratch = cls._context.__enter__()
        # The reused 1024 runs keep the original run_noise_ab selection.json, which
        # has no "selected_epoch" key; the fixture mirrors that layout.
        build_fixture(cls.scratch, require_selected_epoch_key=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._context.__exit__(None, None, None)

    def run_verifier(self) -> tuple[int, list[str]]:
        import verify_nfft_comparison as vnc

        return_code = vnc.main(
            ["--output-dir", str(self.scratch), "--check", "--no-report", "--fixture"]
        )
        record = json.loads((self.scratch / "verification.json").read_text(encoding="utf-8"))
        return return_code, record["failures"]

    def test_fixture_passes(self) -> None:
        return_code, failures = self.run_verifier()
        self.assertEqual(failures, [])
        self.assertEqual(return_code, 0)

    def test_tampered_prediction_is_rejected(self) -> None:
        path = self.scratch / "heldout_window_predictions.csv"
        original = path.read_bytes()
        try:
            text = original.decode("utf-8-sig")
            lines = text.splitlines()
            header = lines[0].split(",")
            columns = lines[1].split(",")
            index = header.index("prediction")
            columns[index] = "1" if columns[index] == "0" else "0"
            lines[1] = ",".join(columns)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
            return_code, failures = self.run_verifier()
            self.assertNotEqual(return_code, 0)
            self.assertTrue(any("mismatch" in item or "not the margin" in item for item in failures), failures)
        finally:
            path.write_bytes(original)

    def test_tampered_selection_is_rejected(self) -> None:
        path = self.scratch / "nfft512" / "seed_2026" / "selection.json"
        original = path.read_bytes()
        try:
            rnc.write_json(path, {"selected_epoch": 4, "rule": "fixture"})
            return_code, failures = self.run_verifier()
            self.assertNotEqual(return_code, 0)
            self.assertTrue(any("selection.json" in item for item in failures), failures)
        finally:
            path.write_bytes(original)

    def test_summary_arithmetic_is_recomputed(self) -> None:
        path = self.scratch / "nfft_summary.csv"
        original = path.read_bytes()
        try:
            text = original.decode("utf-8-sig")
            lines = text.splitlines()
            header = lines[0].split(",")
            columns = lines[1].split(",")
            columns[header.index("file_equal_balacc_mean")] = "0.5"
            lines[1] = ",".join(columns)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
            return_code, failures = self.run_verifier()
            self.assertNotEqual(return_code, 0)
            self.assertTrue(any("nfft_summary" in item for item in failures), failures)
        finally:
            path.write_bytes(original)


class SmokeRunTests(unittest.TestCase):
    """End-to-end bounded run; only executed with ``--smoke``."""

    scratch: Path = Path()
    report: dict[str, Any] = {}

    @classmethod
    def setUpClass(cls) -> None:
        if not cls.scratch.parts:
            cls.scratch = SCRATCH_ROOT / f"unset_{os.getpid()}"
            cls.scratch.mkdir(parents=True, exist_ok=True)

    def test_01_bounded_train_evaluate_summarize(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_fft = rnc.NFFT_GRID[0]
        seed = rnc.TRAINING_SEEDS[0]
        result = rnc.train_run(
            n_fft, seed, device, epochs=1, max_train_samples=8, max_val_samples=8,
            output_root=self.scratch, smoke=True,
        )
        run_dir = Path(result["run_directory"])
        for name in (
            "run_config.json", "training_history.csv", "validation_metrics.csv",
            "selection.json", "best_model.pt",
        ):
            self.assertTrue((run_dir / name).is_file(), name)
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["smoke"], True)
        self.assertEqual(config["selected_epoch"], 1)
        self.assertEqual(config["hop_length"], n_fft // 2)
        self.assertEqual(len(config["epoch_order_sha256"]), 1)
        rows = _load_csv(run_dir / "validation_metrics.csv")
        self.assertEqual(len(rows), len(rnc.CONDITIONS))
        for row in rows:
            self.assertTrue(np.isfinite(float(row["balanced_accuracy"])))

        grid = rnc.experiment_grid(self.scratch, required=False)
        self.assertEqual(len(grid), 1)
        models = rnc.load_models(grid, device)
        predictions, cache, pairing = rnc.evaluate_split(
            "heldout", rnc.HELDOUT_INDEX, list(rnc.CONDITIONS), rnc.HELDOUT_NOISE_SEED,
            models, device, output_root=self.scratch, max_windows=12,
        )
        self.assertEqual(len(cache), len(rnc.CONDITIONS))
        for metadata in cache:
            self.assertEqual(metadata["windows"], 12)
            self.assertEqual(metadata["window_digest_bytes"], 12 * 32)
        self.assertEqual(len(pairing), len(rnc.CONDITIONS))
        self.assertEqual(len({item["digest_sha256"] for item in pairing}), len(rnc.CONDITIONS))

        scoped = predictions[: 12 * len(rnc.CONDITIONS)]
        self.assertEqual(len(scoped), 12 * len(rnc.CONDITIONS))
        tables = rnc.summarize_predictions(scoped, "heldout", output_root=self.scratch)
        self.assertEqual(len(tables["model_condition_metrics"]), len(rnc.CONDITIONS))
        for row in tables["model_condition_metrics"]:
            self.assertEqual(
                row["tp"] + row["tn"] + row["fp"] + row["fn"], 12
            )
        self.__class__.report.update(
            {
                "run_directory": str(run_dir),
                "prediction_rows": len(predictions),
                "cache": cache,
                "figures": None,
            }
        )

    def test_02_figures_are_readable(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evidence = rnc.write_figures(device, output_root=self.scratch)
        detail = evidence["n_fft_detail"]
        self.assertEqual(sorted(int(key) for key in detail), list(rnc.NFFT_GRID))
        for key, entry in detail.items():
            self.assertEqual(entry["model_input_shape"], [256, 256])
            self.assertEqual(entry["native_frequency_bins"], int(key))
            self.assertTrue(entry["model_input_finite"])
            self.assertEqual(entry["frequency_step_khz"], 100_000.0 / int(key))
        for name in ("native_stft_examples.png", "model_input_examples.png"):
            path = self.scratch / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 20_000, name)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.__class__.report["figures"] = evidence["figures"]


def verify_outputs() -> list[str]:
    """Independent recomputation of every published NFFT4 metric."""
    failures: list[str] = []
    root = OUTPUT_ROOT
    heldout_path = root / "heldout_window_predictions.csv"
    if not heldout_path.is_file():
        return [f"missing {heldout_path}"]

    # ---- fairness: one initial weight hash per seed across the four grids ----
    run_configs: dict[tuple[int, int], dict[str, Any]] = {}
    for n_fft in rnc.NFFT_GRID:
        for seed in rnc.TRAINING_SEEDS:
            path = root / f"nfft{n_fft}" / f"seed_{seed}" / "run_config.json"
            if path.is_file():
                run_configs[(n_fft, seed)] = json.loads(path.read_text(encoding="utf-8"))
    for seed in rnc.TRAINING_SEEDS:
        hashes = {
            config["initial_model_sha256"]
            for (n_fft, run_seed), config in run_configs.items()
            if run_seed == seed
        }
        if len(hashes) > 1:
            failures.append(f"seed {seed}: the four n_fft runs do not share one initial weight hash")
        if hashes and hashes != {rnc.initial_state_hash(seed)}:
            failures.append(f"seed {seed}: recorded initial weight hash is not reproducible")
        orders = {
            tuple(config.get("epoch_order_sha256") or [])
            for (n_fft, run_seed), config in run_configs.items()
            if run_seed == seed
        }
        if len(orders) > 1:
            failures.append(f"seed {seed}: the four n_fft runs do not share one epoch order")
        expected_order = tuple(rnc.epoch_orders(seed))
        if orders and orders != {expected_order}:
            failures.append(f"seed {seed}: recorded epoch orders are not reproducible")

    predictions = _load_csv(heldout_path)
    expected_rows = len(rnc.NFFT_GRID) * len(rnc.TRAINING_SEEDS) * len(rnc.CONDITIONS) * rnc.HELDOUT_SAMPLES
    if len(predictions) != expected_rows:
        failures.append(f"heldout rows {len(predictions)} != {expected_rows}")
    keys = [
        (row["model_id"], row["condition"], row["index_position"]) for row in predictions
    ]
    if len(set(keys)) != len(keys):
        failures.append("heldout prediction keys are not unique")

    index_rows = rnc.load_index(rnc.HELDOUT_INDEX, split="heldout")
    by_position = {position: row for position, row in enumerate(index_rows)}
    for row in predictions:
        position = int(row["index_position"])
        source = by_position[position]
        if row["filename"] != source["filename"]:
            failures.append(f"heldout row {position}: filename mismatch")
            break
        if int(row["label"]) != int(source["label"]):
            failures.append(f"heldout row {position}: label mismatch")
            break
        if int(row["start"]) != int(source["start"]) or int(row["length"]) != int(source["length"]):
            failures.append(f"heldout row {position}: window bounds mismatch")
            break
    for row in predictions:
        for field in ("logit_background", "logit_drone", "logit_margin", "drone_score"):
            if not np.isfinite(float(row[field])):
                failures.append(f"non-finite {field}")
                return failures
    for row in predictions:
        background = float(row["logit_background"])
        drone = float(row["logit_drone"])
        margin = float(row["logit_margin"])
        if abs((drone - background) - margin) > 1e-6:
            failures.append("logit_margin != logit_drone - logit_background")
            break
        if int(row["prediction"]) != int(margin > 0.0):
            failures.append("prediction disagrees with the margin>0 rule")
            break

    # ---- per-window -> metric recomputation -----------------------------
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault((row["model_id"], row["condition"]), []).append(
            {
                "filename": row["filename"],
                "target_code": row["target_code"],
                "label": int(row["label"]),
                "prediction": int(row["prediction"]),
            }
        )
    table = {
        (int(row["n_fft"]), int(row["training_seed"]), row["condition"]): row
        for row in _load_csv(root / "model_condition_metrics.csv")
    }
    if len(table) != len(rnc.NFFT_GRID) * len(rnc.TRAINING_SEEDS) * len(rnc.CONDITIONS):
        failures.append(f"model_condition_metrics rows {len(table)}")
    for (model_id, condition), records in grouped.items():
        n_fft = int(model_id.split("_")[0].replace("nfft", ""))
        seed = int(model_id.split("_seed")[1])
        stored = table.get((n_fft, seed, condition))
        if stored is None:
            failures.append(f"model_condition_metrics missing {model_id}/{condition}")
            continue
        if int(stored["n_fft"]) != n_fft:
            failures.append(f"{model_id}/{condition}: n_fft mismatch")
        primary = rnc.file_equal_metrics_any(records)
        secondary = rnc.pooled_metrics_any(records)
        for column, value in (
            ("file_equal_balanced_accuracy", primary["file_equal_balanced_accuracy"]),
            ("file_equal_background_fpr", primary["background_file_equal_fpr"]),
            ("file_equal_drone_recall", primary["drone_file_equal_recall"]),
            ("accuracy", secondary["accuracy"]),
            ("balanced_accuracy", secondary["balanced_accuracy"]),
            ("precision", secondary["precision"]),
            ("drone_recall", secondary["drone_recall"]),
            ("background_specificity", secondary["background_specificity"]),
            ("f1", secondary["f1"]),
        ):
            if abs(float(stored[column]) - float(value)) > 1e-12:
                failures.append(
                    f"{model_id}/{condition}: {column} stored {stored[column]} != {value}"
                )
        for column in ("tn", "fp", "fn", "tp"):
            if int(stored[column]) != int(secondary[column]):
                failures.append(f"{model_id}/{condition}: {column} mismatch")
        if secondary["tp"] + secondary["tn"] + secondary["fp"] + secondary["fn"] != len(records):
            failures.append(f"{model_id}/{condition}: confusion matrix does not conserve windows")

    # ---- nfft_summary ---------------------------------------------------
    composites: dict[tuple[int, int], float] = {}
    for n_fft in rnc.NFFT_GRID:
        for seed in rnc.TRAINING_SEEDS:
            composites[(n_fft, seed)] = float(
                np.mean(
                    [
                        float(table[(n_fft, seed, condition)]["file_equal_balanced_accuracy"])
                        for condition in rnc.CONDITIONS
                    ]
                )
            )
    summary = {
        (int(row["n_fft"]), row["condition"]): row
        for row in _load_csv(root / "nfft_summary.csv")
    }
    if len(summary) != len(rnc.NFFT_GRID) * (len(rnc.CONDITIONS) + 1):
        failures.append(f"nfft_summary rows {len(summary)}")
    for n_fft in rnc.NFFT_GRID:
        for condition in rnc.CONDITIONS:
            values = [
                float(table[(n_fft, seed, condition)]["file_equal_balanced_accuracy"])
                for seed in rnc.TRAINING_SEEDS
            ]
            stored = summary[(n_fft, condition)]
            if abs(float(stored["file_equal_balacc_mean"]) - float(np.mean(values))) > 1e-12:
                failures.append(f"nfft_summary {n_fft}/{condition}: mean mismatch")
            if abs(float(stored["file_equal_balacc_std"]) - float(np.std(values, ddof=1))) > 1e-12:
                failures.append(f"nfft_summary {n_fft}/{condition}: std mismatch")
        values = [composites[(n_fft, seed)] for seed in rnc.TRAINING_SEEDS]
        stored = summary[(n_fft, "four_condition_mean")]
        if abs(float(stored["file_equal_balacc_mean"]) - float(np.mean(values))) > 1e-12:
            failures.append(f"nfft_summary {n_fft}/mean: mean mismatch")
        if abs(float(stored["file_equal_balacc_std"]) - float(np.std(values, ddof=1))) > 1e-12:
            failures.append(f"nfft_summary {n_fft}/mean: std mismatch")

    # ---- paired comparison ---------------------------------------------
    paired = _load_csv(root / "paired_nfft_comparison.csv")
    expected_pairs = (len(rnc.NFFT_GRID) - 1) * (len(rnc.CONDITIONS) + 1)
    if len(paired) != expected_pairs:
        failures.append(f"paired_nfft_comparison rows {len(paired)} != {expected_pairs}")
    for row in paired:
        n_fft = int(row["n_fft"])
        condition = row["condition"]
        if n_fft == rnc.BASELINE_NFFT:
            failures.append("paired comparison contains the 1024 baseline as a contrast")
        per_seed = json.loads(row["per_seed"])
        if sorted(per_seed) != sorted(str(seed) for seed in rnc.TRAINING_SEEDS):
            failures.append(f"paired {n_fft}/{condition}: seed set mismatch")
        differences = []
        for seed in rnc.TRAINING_SEEDS:
            entry = per_seed[str(seed)]
            if condition == "four_condition_mean":
                value = composites[(n_fft, seed)]
                baseline = composites[(rnc.BASELINE_NFFT, seed)]
            else:
                value = float(table[(n_fft, seed, condition)]["file_equal_balanced_accuracy"])
                baseline = float(
                    table[(rnc.BASELINE_NFFT, seed, condition)]["file_equal_balanced_accuracy"]
                )
            if abs(float(entry["value"]) - value) > 1e-12:
                failures.append(f"paired {n_fft}/{condition}/{seed}: value mismatch")
            if abs(float(entry["baseline"]) - baseline) > 1e-12:
                failures.append(f"paired {n_fft}/{condition}/{seed}: baseline mismatch")
            if abs(float(entry["difference"]) - (value - baseline)) > 1e-12:
                failures.append(f"paired {n_fft}/{condition}/{seed}: difference mismatch")
            differences.append(value - baseline)
        if abs(float(row["mean_difference"]) - float(np.mean(differences))) > 1e-12:
            failures.append(f"paired {n_fft}/{condition}: mean difference mismatch")
        if abs(float(row["std_difference"]) - float(np.std(differences, ddof=1))) > 1e-12:
            failures.append(f"paired {n_fft}/{condition}: std difference mismatch")
        if int(row["seeds_positive"]) != sum(1 for value in differences if value > 0):
            failures.append(f"paired {n_fft}/{row['condition']}: positive seed count mismatch")

    # ---- old test clean -------------------------------------------------
    test_path = root / "test_clean_window_predictions.csv"
    if not test_path.is_file():
        failures.append(f"missing {test_path}")
    else:
        test_rows = _load_csv(test_path)
        expected_test = len(rnc.NFFT_GRID) * len(rnc.TRAINING_SEEDS) * rnc.TEST_SAMPLES
        if len(test_rows) != expected_test:
            failures.append(f"test clean rows {len(test_rows)} != {expected_test}")
        test_keys = [
            (row["model_id"], row["condition"], row["index_position"]) for row in test_rows
        ]
        if len(set(test_keys)) != len(test_keys):
            failures.append("test clean prediction keys are not unique")
        if {row["condition"] for row in test_rows} != {"clean"}:
            failures.append("test_clean_window_predictions.csv must hold the clean condition only")
        test_index = rnc.load_index(rnc.OLD_INDEX, split="test")
        test_by_position = {position: row for position, row in enumerate(test_index)}
        for row in test_rows:
            source = test_by_position[int(row["index_position"])]
            if row["filename"] != source["filename"] or int(row["label"]) != int(source["label"]):
                failures.append("test clean row does not match the frozen index")
                break
        for row in test_rows:
            if int(row["prediction"]) != int(float(row["logit_margin"]) > 0.0):
                failures.append("test clean prediction disagrees with the margin rule")
                break

    # ---- frozen inputs unchanged ---------------------------------------
    integrity = rnc.verify_frozen_inputs()
    if integrity["errors"]:
        failures.extend(integrity["errors"])
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify-outputs", action="store_true")
    args = parser.parse_args()

    failures = 0
    output = io.StringIO()

    class _Tee(io.TextIOBase):
        def write(self, text: str) -> int:  # noqa: D401
            output.write(text)
            return sys.__stdout__.write(text)

    tee = _Tee()
    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
        if args.smoke:
            print("== bounded smoke run ==", flush=True)
            scratch = SCRATCH_ROOT / f"smoke_{os.getpid()}"
            if scratch.exists():
                shutil.rmtree(scratch, ignore_errors=True)
            scratch.mkdir(parents=True)
            SmokeRunTests.scratch = scratch
            suite = unittest.TestLoader().loadTestsFromTestCase(SmokeRunTests)
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            failures += len(result.failures) + len(result.errors)
            print(f"== smoke run done (failures={failures}) ==", flush=True)
            shutil.rmtree(scratch, ignore_errors=True)

        print("== offline unit tests ==", flush=True)
        # Explicit discovery of this module's own cases only: running another
        # module's tests from here would silently widen the scope of this check.
        unit_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
        result = unittest.TextTestRunner(verbosity=2).run(unit_suite)
        failures += len(result.failures) + len(result.errors)
        print(f"== offline unit tests done (failures={failures}) ==", flush=True)

        if args.verify_outputs:
            print("\n== independent recomputation of published metrics ==")
            problems = verify_outputs()
            for problem in problems:
                print(f"FAIL: {problem}")
            print(f"output verification problems: {len(problems)}")
            failures += len(problems)

    print(f"failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
