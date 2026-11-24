"""Bounded unit and smoke tests for CASCADE3.

Run from the project directory::

    python dronerfa_baseline/test_cascade3_evaluation.py

Everything here is bounded: synthetic HDF5 fixtures for the reader/RNG rules, hand
computed metric fixtures, and one real but truncated smoke run (145 old-test
windows, one seed, clean condition) written to ``outputs/_test_scratch/`` — the
formal output root is never touched by the tests.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import h5py
import torch

import run_cascade3_evaluation as cascade
import verify_cascade3_evaluation as verifier
from dronerfa_data import add_complex_awgn, rms_normalize
from model import SpectrogramFrontEnd

SCRATCH_ROOT = PROJECT_DIR / "outputs" / "_test_scratch"
SMOKE_ROOT = SCRATCH_ROOT / "cascade3_smoke"
SMOKE_COPY_ROOT = SCRATCH_ROOT / "cascade3_smoke_tampered"
FORMAL_ROOT = PROJECT_DIR / "outputs" / "cascade3_evaluation_20260919"
SMOKE_WINDOWS = 145
SMOKE_SEED = 2026
SMOKE_DATASET = "old_test"
SMOKE_CONDITION = "clean"

INDEX_HEADER = [
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
]
TARGET_NAMES = {
    "0000": "Background",
    "0010": "DJI Phantom 4 Pro",
    "0101": "DJI Air 2S",
}


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def write_synthetic_index(
    root: Path, plan: list[str], length: int = 4096, split: str = "test", seed: int = 11
) -> Path:
    """Write one HDF5 file per target code plus an index with full-split positions."""
    root.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed)
    files: dict[str, Path] = {}
    for code in dict.fromkeys(plan):
        path = root / f"synthetic_{code}.mat"
        total = plan.count(code) * length
        values = generator.standard_normal(total).astype(np.float64)
        with h5py.File(path, "w") as handle:
            for name in ("RF0_I", "RF0_Q", "RF1_I", "RF1_Q"):
                # the frozen reader accepts both layouts; the real MAT files are [1, N]
                handle.create_dataset(name, data=values.reshape(1, total))
        files[code] = path
    rows = []
    cursor = {code: 0 for code in files}
    for window_index, code in enumerate(plan):
        start = cursor[code]
        cursor[code] += length
        rows.append(
            {
                "file_path": str(files[code]),
                "filename": files[code].name,
                "target_code": code,
                "target_name": TARGET_NAMES[code],
                "distance_code": "00",
                "segment_code": "0000",
                "split": split,
                "label": 0 if code == "0000" else 1,
                "block_index": 0,
                "window_index": window_index,
                "start": start,
                "length": length,
                "source_samples": plan.count(code) * length,
            }
        )
    index_path = root / "synthetic_index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return index_path


def make_record(
    filename: str,
    true_class: int,
    final_prediction: int,
    stage1_executed: bool,
    dataset: str = "heldout",
    condition: str = "clean",
    training_seed: int = 2026,
    position: int = 0,
) -> dict:
    return {
        "dataset": dataset,
        "condition": condition,
        "training_seed": training_seed,
        "dataset_index_position": position,
        "filename": filename,
        "true_class": true_class,
        "final_prediction": final_prediction,
        "stage2_executed": stage1_executed,
    }


def stage1_row(
    position: int,
    true_class: int,
    stage1: tuple[float, float],
    stage2: tuple[float, float] | None,
    stage2_batch_id: int | None = 0,
) -> dict:
    margin = float(np.float32(np.float32(stage1[1]) - np.float32(stage1[0])))
    executed = stage2 is not None
    stage2_margin = None
    final = 0
    if executed:
        stage2_margin = float(np.float32(np.float32(stage2[1]) - np.float32(stage2[0])))
        final = 2 if stage2_margin > 0 else 1
    return {
        "dataset": "heldout",
        "condition": "clean",
        "training_seed": 2026,
        "dataset_index_position": position,
        "filename": "synthetic.mat",
        "target_code": "0000" if true_class == 0 else "0010",
        "true_class": true_class,
        "stage1_logit_background": float(stage1[0]),
        "stage1_logit_drone": float(stage1[1]),
        "stage1_margin": margin,
        "stage1_prediction": 1 if margin > 0 else 0,
        "stage1_decision": "Drone" if executed else "Background",
        "stage2_executed": executed,
        "stage2_batch_id": stage2_batch_id if executed else None,
        "stage2_logit_phantom": None if stage2 is None else float(stage2[0]),
        "stage2_logit_air2s": None if stage2 is None else float(stage2[1]),
        "stage2_margin": stage2_margin,
        "final_prediction": final,
        "final_class_name": cascade.CLASS_NAMES[final],
        "stage1_feature_digest": f"digest-{position}",
        "stage2_feature_digest": f"digest-{position}" if executed else None,
    }


def read_fixture_channel(path: Path, name: str = "RF0_I") -> np.ndarray:
    """Read a fixture channel exactly like the frozen reader (float32)."""
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[name], dtype=np.float32).reshape(-1)


def frozen_normalized_window(path: Path, start: int, length: int) -> np.ndarray:
    """The frozen demean/RMS chain on a raw fixture window (float32, like the reader)."""
    i_values = read_fixture_channel(path, "RF0_I")[start : start + length]
    q_values = read_fixture_channel(path, "RF0_Q")[start : start + length]
    return rms_normalize(i_values + 1j * q_values)


def build_smoke_run() -> None:
    """Bounded real run: 145 old-test windows, seed 2026, clean, outside the formal root."""
    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)
    common = [
        "--output-dir",
        str(SMOKE_ROOT),
        "--datasets",
        SMOKE_DATASET,
        "--seeds",
        str(SMOKE_SEED),
        "--conditions",
        SMOKE_CONDITION,
        "--max-windows",
        str(SMOKE_WINDOWS),
    ]
    code = cascade.main(["evaluate", *common])
    assert code == 0, "smoke evaluate failed"
    assert cascade.main(["summarize", *common]) == 0
    assert cascade.main(["figures", *common]) == 0
    assert cascade.main(["report", *common]) == 0
    assert cascade.main(["manifest", *common]) == 0


# ---------------------------------------------------------------------------
# frozen configuration and index structure
# ---------------------------------------------------------------------------
class ConfigurationTests(unittest.TestCase):
    def test_frozen_constants_match_the_config(self) -> None:
        config = cascade.load_config()
        self.assertEqual(int(config["frontend"]["n_fft"]), cascade.N_FFT)
        self.assertEqual(int(config["frontend"]["hop_length"]), cascade.HOP_LENGTH)
        self.assertEqual(int(config["frontend"]["image_size"]), cascade.IMAGE_SIZE)
        self.assertEqual(
            tuple(int(item["seed"]) for item in config["seed_pairing"]), cascade.TRAINING_SEEDS
        )
        self.assertEqual(int(config["noise"]["seed"]), 22026)
        self.assertEqual(int(config["noise"]["epoch"]), 0)
        self.assertTrue(bool(config["frontend"]["same_tensor_for_both_stages"]))

    def test_condition_names_are_canonical(self) -> None:
        self.assertEqual(cascade.condition_name("clean"), "clean")
        self.assertEqual(cascade.condition_name(None), "clean")
        self.assertEqual(cascade.condition_name(-10), "-10")
        self.assertEqual(cascade.condition_name(-10.0), "-10")
        self.assertEqual(cascade.condition_name("-12dB"), "-12")
        self.assertEqual(cascade.condition_name("-15"), "-15")

    def test_dataset_specs_use_the_frozen_index_hashes(self) -> None:
        specs = cascade.dataset_specs(cascade.load_config())
        self.assertEqual(specs["heldout"]["expected_rows"], 21840)
        self.assertEqual(specs["heldout"]["windows"], 1820)
        self.assertEqual(specs["heldout"]["conditions"], cascade.CONDITIONS)
        self.assertEqual(specs["old_test"]["expected_rows"], 1320)
        self.assertEqual(specs["old_test"]["windows"], 440)
        self.assertEqual(specs["old_test"]["conditions"], ("clean",))
        for name, spec in specs.items():
            self.assertEqual(
                cascade.sha256_file(spec["index_path"]), spec["index_sha256"], name
            )

    def test_heldout_rows_and_class_counts(self) -> None:
        specs = cascade.dataset_specs(cascade.load_config())
        rows = cascade.dataset_rows(specs["heldout"]["index_path"], "heldout")
        self.assertEqual(len(rows), 1820)
        self.assertEqual(
            cascade.class_counts(rows),
            {0: 560, 1: 610, 2: 650},
        )
        self.assertEqual([row["dataset_index_position"] for row in rows], list(range(1820)))
        for row in rows:
            expected = 0 if row["target_code"] == "0000" else 1
            self.assertEqual(int(row["stage1_label"]), expected)

    def test_old_test_rows_and_class_counts(self) -> None:
        specs = cascade.dataset_specs(cascade.load_config())
        rows = cascade.dataset_rows(specs["old_test"]["index_path"], "test")
        self.assertEqual(len(rows), 440)
        self.assertEqual(cascade.class_counts(rows), {0: 140, 1: 140, 2: 160})

    def test_frozen_input_verification_passes(self) -> None:
        record = cascade.verify_frozen_inputs("before", SCRATCH_ROOT / "frozen_check")
        self.assertTrue(record["passed"], record["errors"])
        self.assertEqual(len(record["checkpoints"]), 6)
        self.assertTrue(all(entry["match"] for entry in record["checkpoints"].values()))
        self.assertEqual(record["preserved_inputs"]["errors"], [])
        self.assertEqual(record["original_mat_files"]["errors"], [])

    def test_published_digest_names_resolve_to_existing_files(self) -> None:
        """The published stage-1 digest files use ``test`` and the ``dB`` suffix."""
        for dataset, conditions in (
            ("heldout", cascade.CONDITIONS),
            ("old_test", ("clean",)),
        ):
            for condition in conditions:
                name = cascade.published_digest_name(dataset, condition)
                path = verifier.STAGE1_PUBLISHED_ROOT / "feature_digests" / name
                self.assertTrue(path.is_file(), f"{dataset}/{condition}: {path}")
        self.assertEqual(
            cascade.published_digest_name("old_test", "clean"),
            "window_digests_test_clean_nfft512.bin",
        )
        self.assertEqual(
            cascade.published_digest_name("heldout", "-10"),
            "window_digests_heldout_-10dB_nfft512.bin",
        )


# ---------------------------------------------------------------------------
# full-split reader, noise position and feature digests
# ---------------------------------------------------------------------------
class WindowReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_root = SCRATCH_ROOT / "cascade3_synthetic"
        if cls.fixture_root.exists():
            shutil.rmtree(cls.fixture_root)
        cls.length = 4096
        cls.plan = ["0000", "0000", "0010", "0101"]
        cls.index_path = write_synthetic_index(
            cls.fixture_root, cls.plan, length=cls.length, split="test"
        )

    def test_noise_uses_the_full_split_position(self) -> None:
        dataset = cascade.build_window_dataset(
            self.index_path, "test", "-10", 22026
        )
        try:
            for position, code in enumerate(self.plan):
                tensor, label = dataset[position]
                start = self.plan[:position].count(code) * self.length
                iq = frozen_normalized_window(
                    self.index_path.parent / f"synthetic_{code}.mat", start, self.length
                )
                rng = np.random.default_rng(22026 + position)
                expected = add_complex_awgn(iq, -10.0, rng)
                reference = np.stack((expected.real, expected.imag))
                np.testing.assert_array_equal(tensor.numpy(), reference, err_msg=f"position {position}")
                self.assertEqual(int(label), 0 if code == "0000" else 1)
        finally:
            dataset.close()

    def test_filtered_position_would_change_the_noise(self) -> None:
        """The old stage-2 rule used drone-only positions and must not be reused here."""
        dataset = cascade.build_window_dataset(self.index_path, "test", "-10", 22026)
        try:
            drone_tensor, _ = dataset[2]
        finally:
            dataset.close()
        iq = frozen_normalized_window(
            self.index_path.parent / "synthetic_0010.mat", 0, self.length
        )
        filtered = add_complex_awgn(iq, -10.0, np.random.default_rng(22026 + 0))
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(
                drone_tensor.numpy(), np.stack((filtered.real, filtered.imag))
            )

    def test_clean_condition_draws_no_noise(self) -> None:
        dataset = cascade.build_window_dataset(self.index_path, "test", "clean", 22026)
        try:
            tensor, _ = dataset[0]
        finally:
            dataset.close()
        reference = frozen_normalized_window(
            self.index_path.parent / "synthetic_0000.mat", 0, self.length
        )
        np.testing.assert_array_equal(
            tensor.numpy(), np.stack((reference.real, reference.imag))
        )

    def test_build_features_shares_one_digest_per_window(self) -> None:
        device = torch.device("cpu")
        frontend = SpectrogramFrontEnd(n_fft=512, hop_length=256, image_size=256)
        frontend.eval()
        rows = cascade.dataset_rows(self.index_path, "test")
        dataset = cascade.build_window_dataset(self.index_path, "test", "-12", 22026)
        first = cascade.build_features(dataset, rows, device, frontend)
        dataset = cascade.build_window_dataset(self.index_path, "test", "-12", 22026)
        second = cascade.build_features(dataset, rows, device, frontend)
        self.assertEqual(first["features"].shape, (4, 1, 256, 256))
        self.assertEqual(len(first["window_digests"]), 4 * 32)
        self.assertEqual(first["window_digests"], second["window_digests"])
        for position in range(4):
            digest = hashlib.sha256(
                np.ascontiguousarray(first["features"][position]).tobytes()
            ).digest()
            self.assertEqual(first["window_digests"][position * 32 : (position + 1) * 32], digest)

    def test_stage2_batches_follow_the_ascending_selection(self) -> None:
        selection = np.array([False, True, True, False, True, False, True, True, True])
        batches = cascade.stage2_batches(selection)
        self.assertEqual(batches, [[1, 2, 4, 6], [7, 8]])
        flat = [position for batch in batches for position in batch]
        self.assertEqual(flat, sorted(flat))
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(flat, list(np.nonzero(selection)[0]))
        self.assertEqual(cascade.stage2_batches(np.zeros(5, dtype=bool)), [])
        selection = np.ones(4, dtype=bool)
        self.assertEqual(cascade.stage2_batches(selection), [[0, 1, 2, 3]])


# ---------------------------------------------------------------------------
# runtime environment tier, input rebuild and the split rule counters
# ---------------------------------------------------------------------------
class EnvironmentTierTests(unittest.TestCase):
    """The published digests belong to the formal torch build *and* device."""

    FORMAL = {"torch": "2.12.1+cu130", "device": "cuda"}

    def test_comparison_tier_accepts_the_recorded_environment(self) -> None:
        environment = {"torch": "2.12.1+cu130", "device": "cuda", "numpy": "2.5.1"}
        tier, reasons = verifier.comparison_tier(environment, self.FORMAL)
        self.assertEqual(tier, verifier.TIER_FORMAL)
        self.assertEqual(reasons, [])

    def test_comparison_tier_rejects_a_cpu_device(self) -> None:
        environment = {"torch": "2.12.1+cu130", "device": "cpu"}
        tier, reasons = verifier.comparison_tier(environment, self.FORMAL)
        self.assertEqual(tier, verifier.TIER_CROSS)
        self.assertTrue(any("device cpu" in reason for reason in reasons), reasons)

    def test_comparison_tier_rejects_another_torch_build(self) -> None:
        environment = {"torch": "2.12.0+cpu", "device": "cuda"}
        tier, reasons = verifier.comparison_tier(environment, self.FORMAL)
        self.assertEqual(tier, verifier.TIER_CROSS)
        self.assertTrue(any("torch 2.12.0+cpu" in reason for reason in reasons), reasons)

    def test_comparison_tier_without_a_recorded_environment_is_cross(self) -> None:
        tier, reasons = verifier.comparison_tier({"torch": "x", "device": "cuda"}, None)
        self.assertEqual(tier, verifier.TIER_CROSS)
        self.assertTrue(reasons)

    def test_plan_reexec_only_switches_to_the_project_interpreter(self) -> None:
        fake_venv = SCRATCH_ROOT / "reexec_probe" / "python.exe"
        fake_venv.parent.mkdir(parents=True, exist_ok=True)
        fake_venv.write_bytes(b"")
        plan = verifier.plan_reexec(
            verifier.TIER_CROSS,
            enabled=True,
            marker_present=False,
            python_executable=sys.executable,
            venv_python=fake_venv,
        )
        self.assertTrue(plan["needed"], plan)
        self.assertEqual(plan["interpreter"], str(fake_venv))
        for kwargs in (
            {"enabled": False, "marker_present": False},
            {"enabled": True, "marker_present": True},
        ):
            blocked = verifier.plan_reexec(
                verifier.TIER_CROSS,
                python_executable=sys.executable,
                venv_python=fake_venv,
                **kwargs,
            )
            self.assertFalse(blocked["needed"], (kwargs, blocked))
        missing = verifier.plan_reexec(
            verifier.TIER_CROSS,
            enabled=True,
            marker_present=False,
            python_executable=sys.executable,
            venv_python=SCRATCH_ROOT / "reexec_probe" / "missing.exe",
        )
        self.assertFalse(missing["needed"], missing)
        same = verifier.plan_reexec(
            verifier.TIER_CROSS,
            enabled=True,
            marker_present=False,
            python_executable=str(fake_venv),
            venv_python=fake_venv,
        )
        self.assertFalse(same["needed"], same)
        exact = verifier.plan_reexec(
            verifier.TIER_FORMAL,
            enabled=True,
            marker_present=False,
            python_executable=sys.executable,
            venv_python=fake_venv,
        )
        self.assertFalse(exact["needed"], exact)

    def test_target_failures_depend_on_the_numeric_tier(self) -> None:
        record = {
            "input_readback_mismatches": 0,
            "digest_mismatches": 156,
            "max_abs_stage1_logit_diff": 0.0025,
            "max_abs_stage2_logit_diff": 0.0071,
            "stage1_gate_changes": 0,
            "stage1_margin_out_of_tolerance": 0,
            "stage2_class_changes": 0,
            "stage2_margin_out_of_tolerance": 0,
            "stage2_batch_id_errors": 0,
        }
        self.assertEqual(
            verifier.target_failures(record, verifier.TIER_CROSS, verifier.CROSS_ENV_LOGIT_BOUND),
            [],
        )
        formal_failures = verifier.target_failures(
            record, verifier.TIER_FORMAL, verifier.LOGIT_TOLERANCE
        )
        self.assertTrue(any("digest" in message for message in formal_failures), formal_failures)
        self.assertTrue(any("stage-1 logits" in message for message in formal_failures))
        self.assertTrue(any("stage-2 logits" in message for message in formal_failures))

        broken = dict(record, input_readback_mismatches=1)
        self.assertTrue(
            any(
                "input tensor" in message
                for message in verifier.target_failures(
                    broken, verifier.TIER_CROSS, verifier.CROSS_ENV_LOGIT_BOUND
                )
            )
        )
        unbounded = dict(record, max_abs_stage1_logit_diff=verifier.CROSS_ENV_LOGIT_BOUND * 2)
        self.assertTrue(
            any(
                "stage-1 logits" in message
                for message in verifier.target_failures(
                    unbounded, verifier.TIER_CROSS, verifier.CROSS_ENV_LOGIT_BOUND
                )
            )
        )
        flipped = dict(record, stage1_gate_changes=1, stage2_class_changes=2)
        messages = verifier.target_failures(
            flipped, verifier.TIER_CROSS, verifier.CROSS_ENV_LOGIT_BOUND
        )
        self.assertTrue(any("stage1_gate_changes" in message for message in messages), messages)
        self.assertTrue(any("stage2_class_changes" in message for message in messages), messages)

    def test_target_rule_differences_separate_drift_from_decisions(self) -> None:
        row = stage1_row(0, 1, (-1.0, 2.0), (3.0, -1.0))
        clean = verifier.target_rule_differences(
            row,
            stage1_margin=float(row["stage1_margin"]),
            expected_executed=True,
            tolerance=verifier.LOGIT_TOLERANCE,
            stage2_margin=float(row["stage2_margin"]),
            stage2_final=int(row["final_prediction"]),
            batch_index=row["stage2_batch_id"],
        )
        self.assertEqual(clean["rule_mismatches"], 0, clean)

        drifted = verifier.target_rule_differences(
            row,
            stage1_margin=float(row["stage1_margin"]) + 0.01,
            expected_executed=True,
            tolerance=verifier.LOGIT_TOLERANCE,
            stage2_margin=float(row["stage2_margin"]) + 0.01,
            stage2_final=int(row["final_prediction"]),
            batch_index=row["stage2_batch_id"],
        )
        self.assertEqual(drifted["stage1_margin_out_of_tolerance"], 1, drifted)
        self.assertEqual(drifted["stage2_margin_out_of_tolerance"], 1, drifted)
        self.assertEqual(drifted["stage1_gate_changes"], 0, drifted)
        self.assertEqual(drifted["stage2_class_changes"], 0, drifted)
        self.assertEqual(drifted["rule_mismatches"], 2, drifted)

        changed = verifier.target_rule_differences(
            row,
            stage1_margin=float(row["stage1_margin"]),
            expected_executed=False,
            tolerance=verifier.LOGIT_TOLERANCE,
        )
        self.assertEqual(changed["stage1_gate_changes"], 1, changed)
        self.assertEqual(changed["stage2_class_changes"], 0, changed)

        swapped = verifier.target_rule_differences(
            row,
            stage1_margin=float(row["stage1_margin"]),
            expected_executed=True,
            tolerance=verifier.LOGIT_TOLERANCE,
            stage2_margin=float(row["stage2_margin"]),
            stage2_final=1 if int(row["final_prediction"]) == 2 else 2,
            batch_index=row["stage2_batch_id"] + 3,
        )
        self.assertEqual(swapped["stage2_class_changes"], 1, swapped)
        self.assertEqual(swapped["stage2_batch_id_errors"], 1, swapped)
        self.assertEqual(swapped["rule_mismatches"], 2, swapped)

        skipped = verifier.target_rule_differences(
            stage1_row(0, 1, (-1.0, 2.0), None, stage2_batch_id=None),
            stage1_margin=-1.0,
            expected_executed=True,
            tolerance=verifier.LOGIT_TOLERANCE,
        )
        self.assertEqual(skipped["stage1_gate_changes"], 1, skipped)
        self.assertEqual(skipped["stage1_margin_out_of_tolerance"], 1, skipped)
        self.assertEqual(skipped["stage2_class_changes"], 0, skipped)

        inconsistent = stage1_row(0, 1, (-1.0, 2.0), (3.0, -1.0))
        inconsistent["stage2_margin"] = None
        counted = verifier.target_rule_differences(
            inconsistent,
            stage1_margin=float(inconsistent["stage1_margin"]),
            expected_executed=True,
            tolerance=verifier.LOGIT_TOLERANCE,
        )
        self.assertEqual(counted["stage2_class_changes"], 1, counted)


class InputRebuildTests(unittest.TestCase):
    """The raw+noise input path is reproducible without any GPU work."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_root = SCRATCH_ROOT / "cascade3_input_rebuild"
        if cls.fixture_root.exists():
            shutil.rmtree(cls.fixture_root)
        cls.length = 4096
        cls.plan = ["0000", "0010", "0101", "0000"]
        cls.index_path = write_synthetic_index(
            cls.fixture_root, cls.plan, length=cls.length, split="test"
        )
        cls.rows = verifier.index_rows(cls.index_path, "test")

    def test_independent_reader_matches_the_frozen_reader(self) -> None:
        for condition in ("clean", "-10", "-12"):
            dataset = cascade.build_window_dataset(self.index_path, "test", condition, 22026)
            try:
                for position in range(len(self.plan)):
                    tensor, _ = dataset[position]
                    own = verifier.build_input_tensor(
                        self.rows[position], condition, 22026, len(self.plan)
                    )
                    self.assertEqual(own.dtype, np.float32)
                    np.testing.assert_array_equal(
                        tensor.numpy(), own, err_msg=f"{condition} position {position}"
                    )
                    self.assertEqual(
                        verifier.sha256_bytes(verifier.probe_payload(own)),
                        verifier.sha256_bytes(
                            verifier.probe_payload(tensor.numpy())
                        ),
                    )
            finally:
                dataset.close()

    def test_wrong_noise_position_does_not_match(self) -> None:
        dataset = cascade.build_window_dataset(self.index_path, "test", "-12", 22026)
        try:
            tensor, _ = dataset[2]
        finally:
            dataset.close()
        wrong = dict(self.rows[2], dataset_index_position=1)
        own = verifier.build_input_tensor(wrong, "-12", 22026, len(self.plan))
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(tensor.numpy(), own)

    def test_probe_payload_uses_the_recorded_leading_samples(self) -> None:
        array = np.arange(2 * 100, dtype=np.float32).reshape(2, 100)
        self.assertEqual(
            verifier.probe_payload(array, 8),
            np.ascontiguousarray(array[:, :8]).tobytes(),
        )
        self.assertEqual(len(verifier.probe_payload(array, 8)), 64)

    def test_iter_group_inputs_walks_the_whole_split_in_order(self) -> None:
        tensors = list(verifier.iter_group_inputs(self.index_path, "test", "clean", 22026))
        self.assertEqual(len(tensors), len(self.plan))
        for position, tensor in enumerate(tensors):
            own = verifier.build_input_tensor(
                self.rows[position], "clean", 22026, len(self.plan)
            )
            np.testing.assert_array_equal(tensor, own, err_msg=f"position {position}")


# ---------------------------------------------------------------------------
# cascade decision rules: the four key paths plus boundaries
# ---------------------------------------------------------------------------
class DecisionRuleTests(unittest.TestCase):
    def _build(self, true_class: int, stage1, stage2, digest: str = "d"):
        row = {"dataset_index_position": 0, "true_class": true_class}
        spec = {"dataset": "heldout"}
        epochs = {"stage1_epoch": 5, "stage2_epoch": 3}
        for key, value in (
            ("file_path", "x.mat"),
            ("filename", "x.mat"),
            ("target_code", "0000" if true_class == 0 else "0010"),
            ("target_name", TARGET_NAMES["0000" if true_class == 0 else "0010"]),
            ("distance_code", "00"),
            ("segment_code", "0000"),
            ("split", "heldout"),
            ("stage1_label", 0 if true_class == 0 else 1),
            ("true_class_name", cascade.CLASS_NAMES[true_class]),
            ("block_index", 0),
            ("window_index", 0),
            ("start", 0),
            ("length", 1000000),
            ("source_samples", 140000000),
        ):
            row[key] = value
        stage2_values = None
        if stage2 is not None:
            stage2_values = np.asarray(stage2, dtype=np.float32)
        return cascade.build_prediction_row(
            spec,
            row,
            2026,
            epochs,
            np.asarray(stage1, dtype=np.float32),
            stage2_values,
            0 if stage2 is not None else None,
            digest,
            digest if stage2 is not None else None,
        )

    def test_background_classified_correctly(self) -> None:
        record = self._build(0, (3.0, -2.0), None)
        self.assertLess(record["stage1_margin"], 0.0)
        self.assertFalse(record["stage2_executed"])
        self.assertEqual(record["final_prediction"], 0)
        self.assertEqual(record["stage1_decision"], "Background")
        for key in ("stage2_logit_phantom", "stage2_logit_air2s", "stage2_margin"):
            self.assertIsNone(record[key])
        self.assertIsNone(record["stage2_feature_digest"])
        self.assertEqual(record["stage2_batch_id"], None)

    def test_background_false_alarm_reaches_both_machines(self) -> None:
        phantom = self._build(0, (-1.0, 2.0), (2.0, -1.0))
        self.assertTrue(phantom["stage2_executed"])
        self.assertEqual(phantom["final_prediction"], 1)
        air2s = self._build(0, (-1.0, 2.0), (-2.0, 1.0))
        self.assertTrue(air2s["stage2_executed"])
        self.assertEqual(air2s["final_prediction"], 2)

    def test_drone_missed_by_stage_one_becomes_background(self) -> None:
        for true_class in (1, 2):
            record = self._build(true_class, (4.0, -3.0), None)
            self.assertLess(record["stage1_margin"], 0.0)
            self.assertFalse(record["stage2_executed"])
            self.assertEqual(record["final_prediction"], 0)

    def test_two_machines_confused_in_stage_two(self) -> None:
        phantom_as_air2s = self._build(1, (-1.0, 2.0), (-3.0, 1.0))
        self.assertEqual(phantom_as_air2s["final_prediction"], 2)
        air2s_as_phantom = self._build(2, (-1.0, 2.0), (3.0, -1.0))
        self.assertEqual(air2s_as_phantom["final_prediction"], 1)

    def test_zero_margins_use_the_frozen_strict_inequality(self) -> None:
        record = self._build(1, (1.0, 1.0), None)
        self.assertEqual(record["stage1_margin"], 0.0)
        self.assertFalse(record["stage2_executed"])
        self.assertEqual(record["final_prediction"], 0)
        record = self._build(1, (-1.0, 2.0), (1.0, 1.0))
        self.assertEqual(record["stage2_margin"], 0.0)
        self.assertEqual(record["final_prediction"], 1)

    def test_stage2_fields_cannot_be_forged(self) -> None:
        row = {"dataset_index_position": 0, "true_class": 0}
        row.update(
            {
                "file_path": "x.mat",
                "filename": "x.mat",
                "target_code": "0000",
                "target_name": "Background",
                "distance_code": "00",
                "segment_code": "0000",
                "split": "heldout",
                "stage1_label": 0,
                "true_class_name": "Background",
                "block_index": 0,
                "window_index": 0,
                "start": 0,
                "length": 1000000,
                "source_samples": 140000000,
            }
        )
        epochs = {"stage1_epoch": 5, "stage2_epoch": 3}
        with self.assertRaises(ValueError):
            cascade.build_prediction_row(
                {"dataset": "heldout"},
                row,
                2026,
                epochs,
                np.asarray([3.0, -2.0], dtype=np.float32),
                np.asarray([1.0, -1.0], dtype=np.float32),
                0,
                "d",
                "d",
            )
        with self.assertRaises(ValueError):
            cascade.build_prediction_row(
                {"dataset": "heldout"},
                row,
                2026,
                epochs,
                np.asarray([-2.0, 3.0], dtype=np.float32),
                None,
                0,
                "d",
                None,
            )
        with self.assertRaises(ValueError):
            cascade.build_prediction_row(
                {"dataset": "heldout"},
                row,
                2026,
                epochs,
                np.asarray([-2.0, 3.0], dtype=np.float32),
                np.asarray([1.0, -1.0], dtype=np.float32),
                0,
                "d",
                None,
            )

    def test_csv_roundtrip_keeps_skipped_fields_empty(self) -> None:
        rows = [
            cascade.build_prediction_row(
                {"dataset": "heldout"},
                {
                    "dataset_index_position": position,
                    "true_class": true_class,
                    "file_path": "x.mat",
                    "filename": "x.mat",
                    "target_code": "0000" if true_class == 0 else "0010",
                    "target_name": "n",
                    "distance_code": "00",
                    "segment_code": "0000",
                    "split": "heldout",
                    "stage1_label": 0 if true_class == 0 else 1,
                    "true_class_name": cascade.CLASS_NAMES[true_class],
                    "block_index": 0,
                    "window_index": position,
                    "start": position * 1000000,
                    "length": 1000000,
                    "source_samples": 140000000,
                },
                2026,
                {"stage1_epoch": 5, "stage2_epoch": 3},
                np.asarray(stage1, dtype=np.float32),
                None if stage2 is None else np.asarray(stage2, dtype=np.float32),
                0 if stage2 is not None else None,
                f"d{position}",
                f"d{position}" if stage2 is not None else None,
            )
            for position, (true_class, stage1, stage2) in enumerate(
                [
                    (0, (3.0, -2.0), None),
                    (0, (-2.0, 3.0), (2.0, -1.0)),
                    (1, (3.0, -2.0), None),
                ]
            )
        ]
        for row in rows:
            row["condition"] = "clean"
        path = SCRATCH_ROOT / "cascade3_roundtrip.csv"
        cascade.write_csv(path, rows, cascade.PREDICTION_COLUMNS)
        raw = cascade.read_csv_rows(path)
        parsed = cascade.parse_prediction_rows(path)
        self.assertEqual(raw[0]["stage2_logit_phantom"], "")
        self.assertEqual(raw[0]["stage2_batch_id"], "")
        self.assertIsNone(parsed[0]["stage2_logit_phantom"])
        self.assertFalse(parsed[0]["stage2_executed"])
        self.assertTrue(parsed[1]["stage2_executed"])
        self.assertEqual(parsed[1]["stage2_batch_id"], 0)
        self.assertEqual(parsed[1]["stage2_feature_digest"], "d1")
        self.assertEqual(parsed[1]["final_prediction"], 1)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
class MetricTests(unittest.TestCase):
    def _records(self, matrix: np.ndarray, files: int = 1) -> list[dict]:
        records = []
        for truth in range(3):
            for prediction in range(3):
                for index in range(int(matrix[truth, prediction])):
                    records.append(
                        make_record(
                            f"file_{truth}_{index % files}.mat",
                            truth,
                            prediction,
                            prediction != 0 or truth == 0,
                            position=len(records),
                        )
                    )
        return records

    def test_pooled_metrics_match_hand_computation(self) -> None:
        matrix = np.array([[8, 1, 1], [2, 7, 1], [0, 3, 7]], dtype=np.int64)
        records = self._records(matrix)
        values = cascade.pooled_metrics(cascade.confusion_matrix(records))
        independent = verifier.pooled(records)
        self.assertEqual(values["windows"], 30)
        self.assertAlmostEqual(values["pooled_accuracy"], 22 / 30)
        self.assertAlmostEqual(values["recall_background"], 8 / 10)
        self.assertAlmostEqual(values["recall_phantom"], 7 / 10)
        self.assertAlmostEqual(values["recall_air2s"], 7 / 10)
        self.assertAlmostEqual(values["macro_recall"], (0.8 + 0.7 + 0.7) / 3)
        self.assertAlmostEqual(values["precision_background"], 8 / 10)
        self.assertAlmostEqual(values["precision_phantom"], 7 / 11)
        self.assertAlmostEqual(values["precision_air2s"], 7 / 9)
        expected_precision = (0.8 + 7 / 11 + 7 / 9) / 3
        self.assertAlmostEqual(values["macro_precision"], expected_precision)
        expected_f1 = (
            0.8
            + 2 * (7 / 11) * 0.7 / ((7 / 11) + 0.7)
            + 2 * (7 / 9) * 0.7 / ((7 / 9) + 0.7)
        ) / 3
        self.assertAlmostEqual(values["macro_f1"], expected_f1)
        for key, value in values.items():
            if key in independent and isinstance(value, float):
                self.assertTrue(verifier.close(value, independent[key]), key)

    def test_file_equal_metrics_match_hand_computation(self) -> None:
        records = [
            # class 0: file A 4/4, file B 2/4 -> 0.75
            *[make_record("A.mat", 0, 0, True, position=i) for i in range(4)],
            *[make_record("B.mat", 0, 0, True, position=4 + i) for i in range(2)],
            *[make_record("B.mat", 0, 1, True, position=6 + i) for i in range(2)],
            # class 1: file C 3/4, file D 1/4 -> 0.5
            *[make_record("C.mat", 1, 1, True, position=8 + i) for i in range(3)],
            make_record("C.mat", 1, 0, True, position=11),
            make_record("D.mat", 1, 1, True, position=12),
            *[make_record("D.mat", 1, 0, True, position=13 + i) for i in range(3)],
            # class 2: file E 4/4 -> 1.0
            *[make_record("E.mat", 2, 2, True, position=16 + i) for i in range(4)],
        ]
        values = cascade.file_equal_metrics(records)
        self.assertAlmostEqual(values["file_equal_background_recall"], 0.75)
        self.assertAlmostEqual(values["file_equal_phantom_recall"], 0.5)
        self.assertAlmostEqual(values["file_equal_air2s_recall"], 1.0)
        self.assertAlmostEqual(values["file_equal_accuracy"], (0.75 + 0.5 + 1.0) / 3)
        independent = verifier.file_equal(records)
        for key in (
            "file_equal_background_recall",
            "file_equal_phantom_recall",
            "file_equal_air2s_recall",
            "file_equal_accuracy",
        ):
            self.assertTrue(verifier.close(values[key], independent[key]), key)

    def test_stage1_rates_conditional_accuracy_and_decomposition(self) -> None:
        records = [
            make_record("bg.mat", 0, 0, False, position=0),
            make_record("bg.mat", 0, 0, False, position=1),
            make_record("bg.mat", 0, 0, False, position=2),
            make_record("bg.mat", 0, 2, True, position=3),
            make_record("ph.mat", 1, 1, True, position=4),
            make_record("ph.mat", 1, 1, True, position=5),
            make_record("ph.mat", 1, 0, False, position=6),
            make_record("ph.mat", 1, 2, True, position=7),
            make_record("air.mat", 2, 2, True, position=8),
            make_record("air.mat", 2, 2, True, position=9),
        ]
        per_file = cascade.file_equal_metrics(records)["per_file"]
        rates = cascade.stage1_rates(records, per_file)
        self.assertEqual(rates["stage1_background_false_alarms"], 1)
        self.assertAlmostEqual(rates["stage1_background_false_alarm_rate"], 0.25)
        self.assertAlmostEqual(rates["stage1_background_false_alarm_rate_file_equal"], 0.25)
        self.assertEqual(rates["stage1_drone_misses"], 1)
        self.assertAlmostEqual(rates["stage1_drone_miss_rate"], 1 / 6)
        self.assertAlmostEqual(rates["stage1_drone_miss_rate_file_equal"], (1 / 4 + 0 / 2) / 2)
        conditional = cascade.stage2_conditional_metrics(records)
        self.assertEqual(conditional["stage2_conditional_windows"], 5)
        self.assertEqual(conditional["stage2_conditional_correct"], 4)
        self.assertAlmostEqual(conditional["stage2_conditional_accuracy"], 4 / 5)
        errors = cascade.error_decomposition(records)
        self.assertEqual(errors["error_background_to_air2s"], 1)
        self.assertEqual(errors["error_phantom_to_background"], 1)
        self.assertEqual(errors["error_phantom_to_air2s"], 1)
        self.assertEqual(errors["error_air2s_to_phantom"], 0)
        self.assertEqual(errors["error_windows_total"], 3)
        independent = verifier.decomposition(records)
        for key, value in errors.items():
            self.assertTrue(verifier.close(value, independent[key]), key)

    def test_sample_standard_deviation_and_missing_classes(self) -> None:
        self.assertAlmostEqual(cascade.sample_std([0.9, 0.95, 1.0]), 0.05)
        self.assertAlmostEqual(verifier.sample_std([0.9, 0.95, 1.0]), 0.05)
        records = [make_record("bg.mat", 0, 0, False, position=index) for index in range(4)]
        values = cascade.file_equal_metrics(records)
        self.assertEqual(values["file_equal_classes_used"], 1)
        self.assertTrue(math.isnan(values["file_equal_phantom_recall"]))
        self.assertAlmostEqual(values["file_equal_accuracy"], 1.0)
        pooled = cascade.pooled_metrics(cascade.confusion_matrix(records))
        self.assertTrue(math.isnan(pooled["recall_phantom"]))

    def test_confusion_matrix_rejects_unknown_classes(self) -> None:
        records = [make_record("bg.mat", 0, 3, False, position=0)]
        with self.assertRaises(ValueError):
            cascade.confusion_matrix(records)


# ---------------------------------------------------------------------------
# bounded smoke run on real data, parity with the published stage-1 results
# ---------------------------------------------------------------------------
class SmokeRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        formal = FORMAL_ROOT / "window_predictions.csv"
        cls.formal_hash_before = verifier.sha256_file(formal) if formal.is_file() else None
        build_smoke_run()
        cls.records = cascade.parse_prediction_rows(SMOKE_ROOT / "window_predictions.csv")

    def test_smoke_row_count_and_primary_keys(self) -> None:
        self.assertEqual(len(self.records), SMOKE_WINDOWS)
        keys = {
            (
                row["dataset"],
                row["condition"],
                row["training_seed"],
                row["dataset_index_position"],
            )
            for row in self.records
        }
        self.assertEqual(len(keys), SMOKE_WINDOWS)
        self.assertEqual(
            sorted(row["dataset_index_position"] for row in self.records), list(range(SMOKE_WINDOWS))
        )

    def test_smoke_decision_rules_hold(self) -> None:
        executed = 0
        for row in self.records:
            expected_executed = float(row["stage1_margin"]) > 0.0
            self.assertEqual(bool(row["stage2_executed"]), expected_executed)
            if expected_executed:
                executed += 1
                self.assertIsNotNone(row["stage2_logit_phantom"])
                self.assertIsNotNone(row["stage2_logit_air2s"])
                self.assertEqual(row["stage2_feature_digest"], row["stage1_feature_digest"])
                expected_final = 2 if float(row["stage2_margin"]) > 0.0 else 1
            else:
                self.assertIsNone(row["stage2_logit_phantom"])
                self.assertIsNone(row["stage2_logit_air2s"])
                self.assertIsNone(row["stage2_margin"])
                self.assertIsNone(row["stage2_feature_digest"])
                expected_final = 0
            self.assertEqual(row["final_prediction"], expected_final)
        self.assertGreater(executed, 0, "the smoke run never executed stage 2")

    def test_smoke_stage1_matches_the_published_predictions(self) -> None:
        reference = {}
        path = verifier.STAGE1_PUBLISHED_ROOT / "test_clean_window_predictions.csv"
        for row in cascade.read_csv_rows(path):
            if int(row["n_fft"]) != cascade.N_FFT or int(row["training_seed"]) != SMOKE_SEED:
                continue
            if cascade.condition_name(row["condition"]) != SMOKE_CONDITION:
                continue
            reference[int(row["index_position"])] = row
        compared = 0
        for row in self.records:
            reference_row = reference.get(row["dataset_index_position"])
            self.assertIsNotNone(reference_row)
            compared += 1
            self.assertEqual(int(reference_row["prediction"]), row["stage1_prediction"])
            self.assertEqual(float(reference_row["logit_background"]), row["stage1_logit_background"])
            self.assertEqual(float(reference_row["logit_drone"]), row["stage1_logit_drone"])
            self.assertEqual(float(reference_row["logit_margin"]), row["stage1_margin"])
        self.assertEqual(compared, SMOKE_WINDOWS)

    def test_smoke_digests_match_the_published_stage1_digests(self) -> None:
        own = (SMOKE_ROOT / "feature_digests" / f"window_digests_{SMOKE_DATASET}_{SMOKE_CONDITION}.bin").read_bytes()
        reference = (
            verifier.STAGE1_PUBLISHED_ROOT
            / "feature_digests"
            / f"window_digests_test_clean_nfft{cascade.N_FFT}.bin"
        ).read_bytes()
        self.assertEqual(len(own), SMOKE_WINDOWS * 32)
        self.assertEqual(own, reference[: len(own)])

    def test_smoke_summaries_are_consistent(self) -> None:
        metrics = cascade.read_csv_rows(SMOKE_ROOT / "metrics_by_condition.csv")
        self.assertEqual(len(metrics), 1)
        row = metrics[0]
        self.assertEqual(int(row["windows"]), SMOKE_WINDOWS)
        confusion_rows = cascade.read_csv_rows(SMOKE_ROOT / "confusion_matrices.csv")
        self.assertEqual(len(confusion_rows), 9)
        self.assertEqual(sum(int(item["count"]) for item in confusion_rows), SMOKE_WINDOWS)
        per_file = cascade.read_csv_rows(SMOKE_ROOT / "per_file_metrics.csv")
        self.assertEqual(sum(int(item["windows"]) for item in per_file), SMOKE_WINDOWS)
        for name in (
            "confusion_old_test_clean.png",
            "recall_by_condition.png",
            "cascade_flow.png",
        ):
            path = SMOKE_ROOT / name
            self.assertTrue(path.is_file(), name)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_formal_output_root_untouched_by_the_smoke(self) -> None:
        formal = FORMAL_ROOT / "window_predictions.csv"
        if self.formal_hash_before is None:
            self.skipTest("formal output not produced yet")
        self.assertEqual(verifier.sha256_file(formal), self.formal_hash_before)

    def test_verifier_accepts_the_bounded_tree(self) -> None:
        result = verifier.verify_outputs(
            SMOKE_ROOT, fixture=True, check_frozen=False, check_targets_enabled=False
        )
        self.assertTrue(result["passed"], result["failures"])
        detections = result["checks"]["rejection_self_check"]["tamper_cases"]
        self.assertTrue(all(count > 0 for count in detections.values()), detections)

    def test_verifier_rejects_tampered_rows_and_digests(self) -> None:
        if SMOKE_COPY_ROOT.exists():
            shutil.rmtree(SMOKE_COPY_ROOT)
        shutil.copytree(SMOKE_ROOT, SMOKE_COPY_ROOT)
        path = SMOKE_COPY_ROOT / "window_predictions.csv"
        rows = list(cascade.read_csv_rows(path))
        skipped = next(row for row in rows if row["stage2_executed"].lower() == "false")
        skipped["stage2_logit_phantom"] = "1.5"
        skipped["stage2_margin"] = "0.5"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = verifier.verify_outputs(
            SMOKE_COPY_ROOT, fixture=True, check_frozen=False, check_targets_enabled=False
        )
        self.assertFalse(result["passed"])
        self.assertTrue(
            any("forged" in message for message in result["failures"]), result["failures"]
        )
        digest_failures: list[str] = []
        digest_file = (
            SMOKE_COPY_ROOT / "feature_digests" / f"window_digests_{SMOKE_DATASET}_{SMOKE_CONDITION}.bin"
        )
        payload = bytearray(digest_file.read_bytes())
        payload[0] ^= 0xFF
        digest_file.write_bytes(bytes(payload))
        verifier.check_digests(SMOKE_COPY_ROOT, digest_failures, {}, strict=True)
        self.assertTrue(digest_failures, "digest tampering was not detected")


# ---------------------------------------------------------------------------
# figure axes: rework 2 regression guard
# ---------------------------------------------------------------------------
FIGURE_SCRATCH_ROOT = SCRATCH_ROOT / "cascade3_figures"
AXIS_CONDITIONS = ("clean", "-10", "-12", "-15")
# third file in each class keeps the file-equal recall at 1.0 (all files perfect)
AXIS_CLASS_FILES = {0: ("bg", "0"), 1: ("phantom", "P"), 2: ("air2s", "A")}
AXIS_PREDICTIONS = {0: 0, 1: 1, 2: 2}


def axis_prediction_row(
    dataset: str, condition: str, seed: int, true_class: int, position: int
) -> dict:
    """One synthetic prediction row that satisfies the published CSV schema."""
    prefix, suffix = AXIS_CLASS_FILES[true_class]
    final = AXIS_PREDICTIONS[true_class]
    return {
        "dataset": dataset,
        "condition": condition,
        "training_seed": seed,
        "stage1_epoch": 4,
        "stage2_epoch": 2,
        "dataset_index_position": position,
        "file_path": f"{prefix}_S0000.mat",
        "filename": f"{prefix}{suffix}.mat",
        "target_code": ("0000", "0010", "0101")[true_class],
        "target_name": cascade.CLASS_NAMES[true_class],
        "distance_code": "00",
        "segment_code": "0000",
        "row_split": "test" if dataset == "old_test" else "heldout",
        "stage1_label": 0 if true_class == 0 else 1,
        "true_class": true_class,
        "true_class_name": cascade.CLASS_NAMES[true_class],
        "block_index": 0,
        "window_index": position,
        "start": 0,
        "length": 1_000_000,
        "source_samples": 10_000_000,
        "stage1_logit_background": 0.0,
        "stage1_logit_drone": 1.0 if true_class else -1.0,
        "stage1_margin": 1.0 if true_class else -1.0,
        "stage1_prediction": 1 if true_class else 0,
        "stage1_decision": "Drone" if true_class else "Background",
        "stage2_executed": bool(true_class),
        "stage2_batch_id": 0 if true_class else None,
        "stage2_logit_phantom": -1.0 if final == 2 else (1.0 if final == 1 else None),
        "stage2_logit_air2s": 1.0 if final == 2 else (-1.0 if final == 1 else None),
        "stage2_margin": 2.0 if final == 2 else (-2.0 if final == 1 else None),
        "final_prediction": final,
        "final_class_name": cascade.CLASS_NAMES[final],
        "stage1_feature_digest": f"{dataset}-{condition}-{seed}-{true_class}-{position}",
        "stage2_feature_digest": (
            f"{dataset}-{condition}-{seed}-{true_class}-{position}" if true_class else None
        ),
    }


def build_axis_predictions() -> list[dict]:
    """held-out windows under all four conditions plus an ``old_test`` clean block.

    ``heldout`` is written first and ``old_test`` last, which is exactly the order
    that used to let the single-condition dataset overwrite the four-condition
    x-axis ticks.
    """
    rows: list[dict] = []
    for dataset, conditions in (
        ("heldout", AXIS_CONDITIONS),
        ("old_test", ("clean",)),
    ):
        condition_list = list(conditions)
        for seed in cascade.TRAINING_SEEDS:
            for condition in condition_list:
                for true_class in (0, 1, 2):
                    for position in range(3):
                        rows.append(
                            axis_prediction_row(
                                dataset,
                                condition,
                                seed,
                                true_class,
                                position,
                            )
                        )
    return rows


class FigureAxisTests(unittest.TestCase):
    """The recall figure axis is a whole-figure property, not a per-dataset one."""

    def test_canonical_conditions_restore_the_frozen_order(self) -> None:
        self.assertEqual(
            cascade.canonical_conditions(["-15", "clean", "-12", "-10", "clean"]),
            list(cascade.CONDITIONS),
        )
        self.assertEqual(cascade.canonical_conditions(["clean"]), ["clean"])
        self.assertEqual(cascade.canonical_conditions([]), [])
        self.assertEqual(cascade.canonical_conditions(["clean", "-3"]), ["clean", "-3"])

    def test_figure_inventory_covers_every_required_png(self) -> None:
        expected = [
            f"confusion_{dataset}_{condition}.png"
            for dataset in verifier.DATASET_NAMES
            for condition in verifier.EXPECTED_CONDITIONS[dataset]
        ] + [cascade.RECALL_FIGURE, "cascade_flow.png"]
        self.assertEqual(list(cascade.PUBLISHED_FIGURES), expected)
        self.assertEqual(
            [cascade.RECALL_METRIC_BY_CLASS[label][1] for label in cascade.CLASS_IDS],
            [
                "file_equal_background_recall",
                "file_equal_phantom_recall",
                "file_equal_air2s_recall",
            ],
        )
    def test_single_condition_dataset_never_overwrites_the_four_ticks(self) -> None:
        """The regression: a later ``old_test``/clean block must not replace the ticks."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if FIGURE_SCRATCH_ROOT.exists():
            shutil.rmtree(FIGURE_SCRATCH_ROOT)
        rows = build_axis_predictions()
        cascade.write_csv(
            FIGURE_SCRATCH_ROOT / "window_predictions.csv", rows, cascade.PREDICTION_COLUMNS
        )
        # summarize first: the recall figure is drawn from cross_seed_summary.csv
        self.assertEqual(cascade.main(["summarize", "--output-dir", str(FIGURE_SCRATCH_ROOT)]), 0)
        self.assertEqual(cascade.main(["figures", "--output-dir", str(FIGURE_SCRATCH_ROOT)]), 0)
        evidence = json.loads(
            (FIGURE_SCRATCH_ROOT / "figure_evidence.json").read_text(encoding="utf-8")
        )
        recall = FIGURE_SCRATCH_ROOT / cascade.RECALL_FIGURE
        self.assertEqual(recall.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(evidence["figures_written"], sorted(cascade.PUBLISHED_FIGURES))
        self.assertEqual(evidence["missing_figures"], [])
        # the figure must be built from the published cross-seed table
        self.assertEqual(
            evidence["recall_axis"]["source_table_sha256"],
            verifier.sha256_file(FIGURE_SCRATCH_ROOT / "cross_seed_summary.csv"),
        )
        # 1. the rendered axis itself, the failure Codex observed in the published PNG
        image = plt.imread(recall)
        self.assertGreater(int((image[..., :3] < 0.5).all(axis=-1).sum()), 50)
        # 2. the axis record written next to the figure
        self.assertEqual(evidence["recall_axis"]["conditions"], list(AXIS_CONDITIONS))
        self.assertEqual(evidence["recall_axis"]["tick_labels"], list(AXIS_CONDITIONS))
        self.assertEqual(
            evidence["recall_axis"]["dataset_conditions"],
            {"heldout": list(AXIS_CONDITIONS), "old_test": ["clean"]},
        )
        # 3. the axes of the figure that was actually saved
        grouped = cascade.group_records(rows)
        figure, axis, _ = cascade.build_recall_figure(
            grouped, plt, summary_table=cascade.cross_seed_figure_metrics(FIGURE_SCRATCH_ROOT)
        )
        try:
            self.assertEqual(
                [text.get_text() for text in axis.get_xticklabels()], list(AXIS_CONDITIONS)
            )
            # each errorbar contributes marker line + 2 cap lines: heldout is drawn
            # first with four conditions per class, old_test last with exactly one
            self.assertEqual(
                [len(line.get_xdata()) for line in axis.lines],
                [4, 4, 4] * 3 + [1, 1, 1] * 3,
            )
            marker_lines = [axis.lines[index] for index in range(0, len(axis.lines), 3)]
            drawn = {
                float(value)
                for line in marker_lines
                for value in np.ma.filled(np.ma.asarray(line.get_xdata()), np.nan)
                if math.isfinite(float(value))
            }
            self.assertEqual(sorted(drawn), [0.0, 1.0, 2.0, 3.0])
            # every synthetic file is perfect, so every drawn point must be exactly 1.0
            for line in marker_lines:
                values = np.ma.filled(np.ma.asarray(line.get_ydata()), np.nan)
                self.assertTrue(all(math.isfinite(float(value)) for value in values), values)
                for value in values:
                    self.assertAlmostEqual(float(value), 1.0)
        finally:
            plt.close(figure)

        # the bounded tree's figures still satisfy the independent figure checker
        figure_failures: list[str] = []
        figure_record: dict = {}
        verifier.check_figures(FIGURE_SCRATCH_ROOT, figure_failures, figure_record, strict=True)
        self.assertEqual(figure_failures, [])
        self.assertEqual(sorted(figure_record["figures"]), sorted(cascade.PUBLISHED_FIGURES))
        self.assertTrue(
            all(entry["hash_ok"] for entry in figure_record["figures"].values()),
            figure_record["figures"],
        )
        # the recorded hashes match the PNG bytes on disk
        for name, entry in evidence["figures"].items():
            self.assertEqual(
                entry["sha256"], verifier.sha256_file(FIGURE_SCRATCH_ROOT / name), name
            )
            self.assertEqual(entry["bytes"], (FIGURE_SCRATCH_ROOT / name).stat().st_size, name)


# ---------------------------------------------------------------------------
# the published full run (skipped until it exists)
# ---------------------------------------------------------------------------
class PublishedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = FORMAL_ROOT / "window_predictions.csv"
        if not cls.path.is_file():
            raise unittest.SkipTest("the full cascade run has not been produced yet")
        cls.records = cascade.parse_prediction_rows(cls.path)

    def test_expected_row_counts_and_keys(self) -> None:
        by_dataset: dict[str, int] = {}
        keys = set()
        for row in self.records:
            by_dataset[row["dataset"]] = by_dataset.get(row["dataset"], 0) + 1
            key = (
                row["dataset"],
                row["condition"],
                row["training_seed"],
                row["dataset_index_position"],
            )
            self.assertNotIn(key, keys)
            keys.add(key)
        self.assertEqual(by_dataset["heldout"], 21840)
        self.assertEqual(by_dataset["old_test"], 1320)
        self.assertEqual(len(self.records), 23160)

    def test_every_group_covers_the_full_split(self) -> None:
        specs = cascade.dataset_specs(cascade.load_config())
        expected = {
            "heldout": (specs["heldout"]["windows"], cascade.CONDITIONS),
            "old_test": (specs["old_test"]["windows"], ("clean",)),
        }
        seen: dict[tuple[str, str, int], set[int]] = {}
        for row in self.records:
            key = (row["dataset"], row["condition"], row["training_seed"])
            seen.setdefault(key, set()).add(row["dataset_index_position"])
        for dataset, (windows, conditions) in expected.items():
            for condition in conditions:
                for seed in cascade.TRAINING_SEEDS:
                    self.assertEqual(
                        seen[(dataset, condition, seed)], set(range(windows)), (dataset, condition, seed)
                    )

    def test_one_feature_digest_per_window_shared_by_the_three_seeds(self) -> None:
        digests: dict[tuple[str, str, int], set[str]] = {}
        for row in self.records:
            key = (row["dataset"], row["condition"], row["dataset_index_position"])
            digests.setdefault(key, set()).add(str(row["stage1_feature_digest"]))
        self.assertTrue(all(len(values) == 1 for values in digests.values()))
        executed = [
            row for row in self.records if bool(row["stage2_executed"])
        ]
        self.assertTrue(executed)
        self.assertTrue(
            all(row["stage2_feature_digest"] == row["stage1_feature_digest"] for row in executed)
        )

    def test_stage1_parity_record_is_complete(self) -> None:
        parity = json.loads(
            (FORMAL_ROOT / "stage1_reference_parity.json").read_text(encoding="utf-8")
        )
        logits = parity["logits"]
        self.assertEqual(logits["compared_rows"], 23160)
        self.assertEqual(logits["missing_reference_rows"], 0)
        self.assertEqual(logits["prediction_mismatches"], 0)
        self.assertTrue(logits["exact_match"])
        for field in (
            "max_abs_logit_background_diff",
            "max_abs_logit_drone_diff",
            "max_abs_logit_margin_diff",
        ):
            self.assertEqual(logits[field], 0.0, field)
        self.assertTrue(parity["digests_all_identical"])
        self.assertEqual(len(parity["digests"]), 5)
        for key, entry in parity["digests"].items():
            self.assertTrue(entry.get("identical"), key)
            self.assertEqual(entry["reference_sha256"], entry["own_sha256"], key)
        self.assertFalse(parity["stage2_compared_with_oracle_csv"])

    def test_published_recall_axis_labels_all_four_conditions(self) -> None:
        """Rework 2: the published figure must label clean/-10/-12/-15, not only clean."""
        evidence = json.loads(
            (FORMAL_ROOT / "figure_evidence.json").read_text(encoding="utf-8")
        )
        axis_record = evidence.get("recall_axis") or {}
        self.assertEqual(axis_record.get("conditions"), list(cascade.CONDITIONS))
        self.assertEqual(axis_record.get("tick_labels"), list(cascade.CONDITIONS))
        self.assertEqual(
            axis_record.get("dataset_conditions"),
            {"heldout": list(cascade.CONDITIONS), "old_test": ["clean"]},
        )
        self.assertEqual(
            axis_record.get("source_table_sha256"),
            verifier.sha256_file(FORMAL_ROOT / "cross_seed_summary.csv"),
        )
        self.assertEqual(
            [entry["sha256"] for entry in evidence["figures"].values()],
            [verifier.sha256_file(FORMAL_ROOT / name) for name in evidence["figures"]],
        )

    def test_published_metrics_recompute_exactly(self) -> None:
        grouped = verifier.group_records(self.records)
        rows = {
            (row["dataset"], row["condition"], int(row["training_seed"])): row
            for row in cascade.read_csv_rows(FORMAL_ROOT / "metrics_by_condition.csv")
        }
        self.assertEqual(len(rows), len(grouped))
        checked = 0
        for key, group in grouped.items():
            values = verifier.recompute_group(group)
            row = rows[key]
            for field in (
                "file_equal_accuracy",
                "pooled_accuracy",
                "macro_recall",
                "macro_precision",
                "macro_f1",
                "stage1_background_false_alarm_rate",
                "stage1_drone_miss_rate",
                "stage2_conditional_accuracy",
                "recall_background",
                "recall_phantom",
                "recall_air2s",
            ):
                self.assertTrue(
                    verifier.close(
                        float(row[field]) if row[field] else float("nan"), values[field]
                    ),
                    f"{key} {field}",
                )
                checked += 1
            self.assertEqual(int(row["windows"]), values["windows"])
        self.assertEqual(checked, len(grouped) * 11)


if __name__ == "__main__":
    unittest.main(verbosity=2)
