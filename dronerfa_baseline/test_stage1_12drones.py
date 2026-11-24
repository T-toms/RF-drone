"""Regression tests for STAGE1-12DRONES-A index building, training and verification.

Pure-logic tests plus read-only checks against the frozen artefacts on disk.
No training is launched, no RF0 payload is read from the G: drive, and nothing
under ``outputs/`` is written.
"""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import torch

import build_stage1_12drones_index as builder
import compare_stage1_12drones_seeds as comparator
import run_stage1_12drones_training as training
import verify_stage1_12drones_training as verifier


class FrozenAuthorityTests(unittest.TestCase):
    def test_freeze_and_manifest_hashes_match_the_asserted_constants(self) -> None:
        freeze, csv_path, manifest_hash = builder.load_freeze(builder.DEFAULT_FREEZE_JSON)
        self.assertEqual(freeze["decision"], "accepted_frozen_development_split")
        self.assertEqual(builder.sha256_file(csv_path), builder.FROZEN_SPLIT_CSV_SHA256)
        self.assertEqual(manifest_hash, builder.FROZEN_MANIFEST_SHA256)
        self.assertEqual(training.FROZEN_SPLIT_CSV_SHA256, builder.FROZEN_SPLIT_CSV_SHA256)

    def test_frozen_split_shape_and_roles(self) -> None:
        sources = builder.load_frozen_sources(builder.DEFAULT_FROZEN_SPLIT_CSV)
        self.assertEqual(len(sources), 104)
        self.assertEqual(len({source.target_code for source in sources}), 13)
        role_counts: dict[str, int] = {}
        for source in sources:
            role_counts[source.role] = role_counts.get(source.role, 0) + 1
        self.assertEqual(
            role_counts,
            {"train": 62, "val": 16, "development_test": 20, "development_test_exposed": 6},
        )
        self.assertEqual(sum(1 for s in sources if s.label == 0), 8)
        self.assertEqual(sum(1 for s in sources if s.label == 1), 96)

    def test_role_expectations_split_legacy_and_new_classes(self) -> None:
        legacy = builder.expected_role_files("T0010")
        new = builder.expected_role_files("T0100")
        self.assertEqual(legacy["train"], 4)
        self.assertEqual(legacy["development_test_exposed"], 2)
        self.assertEqual(new["train"], 5)
        self.assertEqual(new["development_test"], 2)
        self.assertEqual(new["development_test_exposed"], 0)


class LabelSpaceTests(unittest.TestCase):
    def test_background_is_zero_and_every_drone_code_is_one(self) -> None:
        self.assertEqual(builder.stage1_label("T0000"), 0)
        for code in builder.DRONE_CODES:
            self.assertEqual(builder.stage1_label(code), 1)
        self.assertEqual(training.stage1_label_of("T0000"), 0)
        self.assertEqual(training.stage1_label_of("T1100"), 1)
        self.assertEqual(set(builder.TARGET_CODES), {"T0000", *builder.DRONE_CODES})
        self.assertEqual(len(builder.DRONE_CODES), 12)

    def test_mat_name_parsing_keeps_missing_distance_empty(self) -> None:
        parsed = builder.parse_mat_name("T0001_D00_S0000.mat")
        self.assertEqual(
            parsed,
            {
                "target_code": "T0001",
                "distance_code": "00",
                "segment_code": "0000",
                "has_distance_field": "True",
            },
        )
        for name in ("T1010_S0000.mat", "T1011_S0001.mat", "T1100_S0101.mat"):
            parsed = builder.parse_mat_name(name)
            self.assertEqual(parsed["distance_code"], "")
            self.assertEqual(parsed["has_distance_field"], "False")
        with self.assertRaises(ValueError):
            builder.parse_mat_name("T0010_S0000.mat")  # D00 mandatory for this code
        with self.assertRaises(ValueError):
            builder.parse_mat_name("background.mat")


class IndexSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = builder.read_csv_rows(builder.DEFAULT_OUTPUT_DIR / "window_index_stage1_12drones.csv")

    def test_index_rows_and_roles(self) -> None:
        self.assertEqual(len(self.rows), 15790)
        roles: dict[str, int] = {}
        for row in self.rows:
            roles[row["source_role"]] = roles.get(row["source_role"], 0) + 1
        self.assertEqual(
            roles,
            {"train": 9590, "val": 2360, "development_test": 2950, "development_test_exposed": 890},
        )

    def test_index_has_no_precomputed_labels_beyond_zero_and_one(self) -> None:
        labels = {row["label"] for row in self.rows}
        self.assertEqual(labels, {"0", "1"})
        for row in self.rows:
            expected = "0" if row["target_code"] == "T0000" else "1"
            self.assertEqual(row["label"], expected)
            self.assertEqual(row["stage1_label"], expected)

    def test_window_geometry_and_block_bookkeeping(self) -> None:
        for row in self.rows[:500]:
            window_index = int(row["window_index"])
            self.assertEqual(int(row["start"]), window_index * 1_000_000)
            self.assertEqual(int(row["length"]), 1_000_000)
            self.assertEqual(int(row["window_in_block"]), window_index % 10)
            self.assertEqual(int(row["block_index"]), window_index // 10)

    def test_test_roles_never_share_a_file_with_train_or_val(self) -> None:
        by_role: dict[str, set[str]] = {}
        for row in self.rows:
            by_role.setdefault(row["source_role"], set()).add(row["filename"])
        train = by_role["train"]
        val = by_role["val"]
        test = by_role["development_test"] | by_role["development_test_exposed"]
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)

    def test_summary_matches_the_index(self) -> None:
        summary = builder_json(builder.DEFAULT_OUTPUT_DIR / "dataset_summary.json")
        self.assertEqual(summary["source_files"]["total"], 104)
        self.assertEqual(summary["windows"]["total"], len(self.rows))
        self.assertEqual(summary["windows"]["by_role"]["train"], 9590)
        self.assertEqual(summary["windows"]["by_role_label"]["train"], {"0": 600, "1": 8990})
        self.assertEqual(summary["windows"]["by_role_label"]["val"], {"0": 280, "1": 2080})
        self.assertTrue(summary["index_verification"]["passed"])
        self.assertTrue(summary["index_negative_controls"]["passed"])

    def test_read_probe_never_covers_a_test_role(self) -> None:
        probe = builder_json(builder.DEFAULT_OUTPUT_DIR / "read_probe.json")
        self.assertEqual(probe["test_roles_read_count"], 0)
        probed = {
            file["filename"]
            for role in ("train", "val")
            for file in probe["bounded"][role]["files"]
        }
        test_files = {
            name for names in probe["test_roles_read"].values() for name in names
        }
        self.assertFalse(probed & test_files)
        self.assertEqual(len(probed), 78)

    def test_index_identity_gate_passes_on_disk(self) -> None:
        identity = training.check_index_identity()
        self.assertTrue(identity["passed"], identity["errors"])
        self.assertEqual(identity["index"]["sha256"], training.INDEX_SHA256)
        self.assertEqual(identity["observed"]["windows"]["train"], 9590)
        self.assertEqual(identity["observed"]["windows"]["val"], 2360)
        self.assertEqual(identity["observed"]["background_target_codes"], ["T0000"])
        self.assertEqual(len(identity["observed"]["drone_target_codes"]), 12)

    def test_per_file_window_counts_match_the_hdf5_metadata(self) -> None:
        identity = builder_json(builder.DEFAULT_OUTPUT_DIR / "source_identity.json")
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row["filename"]] = counts.get(row["filename"], 0) + 1
        self.assertEqual(len(identity["files"]), 104)
        for entry in identity["files"]:
            self.assertEqual(
                counts[entry["filename"]],
                entry["complete_windows"],
                entry["filename"],
            )


class NegativeControlDetectionTests(unittest.TestCase):
    """The index verifier must reject each tampered index variant."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = builder.load_frozen_sources(builder.DEFAULT_FROZEN_SPLIT_CSV)
        cls.rows = builder.read_csv_rows(builder.DEFAULT_OUTPUT_DIR / "window_index_stage1_12drones.csv")
        identity = builder_json(builder.DEFAULT_OUTPUT_DIR / "source_identity.json")
        cls.identities = {entry["filename"]: entry for entry in identity["files"]}
        probe = builder_json(builder.DEFAULT_OUTPUT_DIR / "read_probe.json")
        cls.probes = {role: probe["bounded"][role] for role in ("train", "val")}

    def _verify(self, rows: list[dict[str, str]]):
        return builder.verify_index(self.sources, self.identities, rows, self.probes)

    def test_baseline_index_passes(self) -> None:
        result = self._verify([dict(row) for row in self.rows])
        self.assertTrue(result["passed"], result["checks"])

    def test_cross_role_tamper_is_detected(self) -> None:
        tampered = [dict(row) for row in self.rows]
        victim = next(row for row in tampered if row["source_role"] == "train")
        victim["source_role"] = "val"
        result = self._verify(tampered)
        self.assertFalse(result["passed"])

    def test_label_tamper_is_detected(self) -> None:
        tampered = [dict(row) for row in self.rows]
        victim = next(row for row in tampered if row["label"] == "1")
        victim["label"] = "0"
        result = self._verify(tampered)
        self.assertFalse(result["passed"])

    def test_geometry_tamper_is_detected(self) -> None:
        tampered = [dict(row) for row in self.rows]
        tampered[7]["start"] = str(int(tampered[7]["start"]) + 1_000_000)
        result = self._verify(tampered)
        self.assertFalse(result["passed"])

    def test_probe_leak_is_detected(self) -> None:
        victim = next(
            source.filename for source in self.sources if source.role == "development_test"
        )
        probes = {
            role: {
                **self.probes[role],
                "files": [
                    *self.probes[role]["files"],
                    {
                        "filename": victim,
                        "role": role,
                        "target_code": "T0100",
                        "label": 1,
                        "complete_windows": 1,
                        "window_indices": [0],
                        "probes": [],
                    },
                ],
            }
            for role in ("train", "val")
        }
        result = builder.verify_index(self.sources, self.identities, self.rows, probes)
        self.assertFalse(result["passed"])


class MetricRecomputationTests(unittest.TestCase):
    def test_pooled_metrics_match_a_hand_counted_confusion_matrix(self) -> None:
        records = [
            {"label": 0, "prediction": 0},
            {"label": 0, "prediction": 1},
            {"label": 1, "prediction": 1},
            {"label": 1, "prediction": 1},
            {"label": 1, "prediction": 0},
        ]
        metrics = verifier.metrics_from_records(records)
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (1, 1, 1, 2))
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5 * (2 / 3 + 0.5))
        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        training_metrics = training.pooled_binary_metrics(records)
        self.assertEqual(metrics, training_metrics)

    def test_file_equal_metrics_weight_each_source_file_once(self) -> None:
        records = [
            {"label": 0, "prediction": 0, "filename": "bg1"},
            {"label": 1, "prediction": 1, "filename": "drone1"},
            {"label": 1, "prediction": 0, "filename": "drone1"},
            {"label": 1, "prediction": 1, "filename": "drone2"},
        ]
        metrics = verifier.file_equal_metrics(records)
        self.assertAlmostEqual(metrics["file_equal_background_recall"], 1.0)
        self.assertAlmostEqual(metrics["file_equal_drone_recall"], 0.75)
        self.assertAlmostEqual(metrics["file_equal_balanced_accuracy"], 0.875)
        self.assertEqual(metrics["files_background"], 1)
        self.assertEqual(metrics["files_drone"], 2)


class SelectionRuleTests(unittest.TestCase):
    def test_highest_score_wins(self) -> None:
        history = [
            {"epoch": 1, training.SELECTION_KEY: 0.5},
            {"epoch": 2, training.SELECTION_KEY: 0.9},
            {"epoch": 3, training.SELECTION_KEY: 0.7},
        ]
        self.assertEqual(training.select_epoch(history, training.SELECTION_KEY), 2)
        self.assertEqual(verifier.select_epoch(history), 2)

    def test_exact_tie_selects_the_earliest_epoch(self) -> None:
        history = [
            {"epoch": 1, training.SELECTION_KEY: 0.8},
            {"epoch": 2, training.SELECTION_KEY: 0.8},
            {"epoch": 3, training.SELECTION_KEY: 0.8},
        ]
        self.assertEqual(training.select_epoch(history, training.SELECTION_KEY), 1)

    def test_replay_selection_is_evenly_spread_and_keeps_both_labels(self) -> None:
        rows = [
            {"label": "1"} for _ in range(90)
        ] + [{"label": "0"} for _ in range(10)]
        selected = verifier._select_probe_rows(rows, 20)
        self.assertGreaterEqual(len(selected), 20)
        self.assertEqual({rows[position]["label"] for position in selected}, {"0", "1"})
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(verifier._select_probe_rows(rows, 0), list(range(len(rows))))


class NoiseReplicationTests(unittest.TestCase):
    """The verifier's noise/front-end copies must equal the frozen reader's."""

    def test_rms_normalisation_and_awgn_match_dronerfa_data(self) -> None:
        from dronerfa_data import add_complex_awgn, rms_normalize

        rng = np.random.default_rng(7)
        iq = (
            rng.standard_normal(4096, dtype=np.float32)
            + 1j * rng.standard_normal(4096, dtype=np.float32)
        ).astype(np.complex64)
        np.testing.assert_array_equal(verifier.rms_normalize(iq), rms_normalize(iq))
        np.testing.assert_array_equal(
            verifier.add_complex_awgn(iq, -10.0, np.random.default_rng(11)),
            add_complex_awgn(iq, -10.0, np.random.default_rng(11)),
        )

    def test_clean_condition_adds_no_noise(self) -> None:
        iq = np.ones(16, dtype=np.complex64)
        np.testing.assert_array_equal(verifier.condition_iq(iq, "clean", 5), iq)

    def test_condition_noise_reproduces_the_dataset_reader(self) -> None:
        from dronerfa_data import DroneRFaWindowDataset

        rows = builder.read_csv_rows(builder.DEFAULT_OUTPUT_DIR / "window_index_stage1_12drones.csv")
        val_rows = [row for row in rows if row["source_role"] == "val"]
        dataset = DroneRFaWindowDataset(
            builder.DEFAULT_OUTPUT_DIR / "window_index_stage1_12drones.csv",
            split="val",
            fixed_snr=-12.0,
            apply_fixed_snr=True,
            seed=verifier.VALIDATION_SEED,
        )
        self.assertEqual(len(dataset), len(val_rows))
        iq = np.asarray([0.5 + 0.25j] * 8, dtype=np.complex64)
        for position in (0, 1, 7):
            rng = np.random.default_rng(verifier.VALIDATION_SEED + position)
            from dronerfa_data import add_complex_awgn

            rng.integers(0, 1)  # the frozen reader consumes this draw
            expected = add_complex_awgn(iq, -12.0, rng)
            observed = verifier.condition_iq(iq, "-12dB", position)
            np.testing.assert_array_equal(observed, expected)
        dataset.close()

    def test_local_frontend_matches_the_model_frontend(self) -> None:
        from model import SpectrogramFrontEnd

        torch.manual_seed(3)
        signals = torch.randn(2, 2, 400_000)
        local = verifier.LocalSpectrogramFrontEnd()(signals)
        frozen = SpectrogramFrontEnd(n_fft=512, hop_length=256, image_size=256)(signals)
        self.assertEqual(tuple(local.shape), (2, 1, 256, 256))
        self.assertTrue(torch.allclose(local, frozen, atol=0.0, rtol=0.0))


class ReadAuditTests(unittest.TestCase):
    def test_audited_dataset_records_every_open_and_window(self) -> None:
        audit = training.ReadAudit("unit")
        self.assertEqual(audit.payload()["opened_file_count"], 0)
        audit.record_open(r"G:\x\a.mat")
        audit.record_open(r"G:\x\a.mat")
        audit.record_window(r"G:\x\a.mat")
        audit.record_window(r"G:\x\b.mat")
        payload = audit.payload()
        self.assertEqual(payload["open_events"], 2)
        self.assertEqual(payload["opened_file_count"], 1)
        self.assertEqual(payload["windows_read"], 2)
        self.assertEqual(payload["windows_by_file"][r"G:\x\b.mat"], 1)


class AcceptRejectSemanticsTests(unittest.TestCase):
    """The control harness must treat a violated assertion as a detection."""

    def test_detect_inverts_the_assertion(self) -> None:
        captured: list[dict] = []

        def detect(name: str, assertion, describe=None) -> None:
            try:
                passed = bool(assertion())
            except Exception as error:  # noqa: BLE001 - mirrors the verifier
                captured.append({"control": name, "detected": True, "detail": repr(error)})
                return
            captured.append({"control": name, "detected": not passed, "detail": describe})

        detect("holds", lambda: True)
        detect("violated", lambda: False)
        detect("raises", lambda: (_ for _ in ()).throw(ValueError("boom")))
        self.assertEqual(
            [(row["control"], row["detected"]) for row in captured],
            [("holds", False), ("violated", True), ("raises", True)],
        )


class CompletedRunArtefactTests(unittest.TestCase):
    """Read-only checks against the completed full run, when it is present."""

    RUN_ROOT = PROJECT_DIR / "outputs" / "stage1_12drones_20260926" / "seed_2026"

    def setUp(self) -> None:
        if not (self.RUN_ROOT / "summary.json").is_file():
            self.skipTest("full stage-1 run has not been executed yet")

    def test_run_summary_components(self) -> None:
        summary = json.loads((self.RUN_ROOT / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["phase"], "STAGE1-12DRONES-A")
        self.assertTrue(summary["identity_passed"])
        self.assertTrue(summary["read_audit_passed"])
        self.assertTrue(summary["checkpoint_reload_passed"])
        self.assertEqual(len(summary["history"]), 5)
        self.assertEqual(summary["selected_epoch"], verifier.select_epoch(summary["history"]))
        self.assertEqual(summary["read_audit_opened_files"], 78)

    def test_config_is_a_full_run(self) -> None:
        config = json.loads((self.RUN_ROOT / "run_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["mode"], "full")
        self.assertIsNone(config["max_train_samples"])
        self.assertIsNone(config["max_val_samples"])
        self.assertEqual(config["train_windows_used_per_epoch"], 9590)
        self.assertEqual(config["validation_windows_used_per_epoch"], 2360)
        self.assertEqual(config["epochs"], 5)
        self.assertTrue(config["read_audit_passed"])
        self.assertTrue(config["test_windows_never_read"])

    def test_independent_verification_passed(self) -> None:
        path = PROJECT_DIR / "outputs" / "stage1_12drones_20260926" / "verification.json"
        if not path.is_file():
            self.skipTest("verification.json has not been written yet")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(payload["passed"], payload["verification"]["failed_checks"])
        self.assertEqual(
            payload["verification"]["passed_checks"], payload["verification"]["total_checks"]
        )
        self.assertTrue(payload["negative_controls"]["passed"])
        names = {row["check"] for row in payload["verification"]["checks"]}
        for required in ("A4 window index SHA256", "C2 no development_test* file was opened"):
            self.assertIn(required, names)
        self.assertGreaterEqual(payload["read_audit"]["windows_read"], 57390)


class VerifierSourceIndependenceTests(unittest.TestCase):
    def test_verifier_does_not_import_the_training_driver(self) -> None:
        source = (PROJECT_DIR / "verify_stage1_12drones_training.py").read_text(encoding="utf-8")
        self.assertNotIn("import run_stage1_12drones_training", source)
        self.assertNotIn("import build_stage1_12drones_index", source)
        self.assertNotIn("from run_stage1_12drones_training", source)
        self.assertIn("class LocalSpectrogramFrontEnd", source)

    def test_training_driver_only_imports_the_accepted_shared_modules(self) -> None:
        source = (PROJECT_DIR / "run_stage1_12drones_training.py").read_text(encoding="utf-8")
        for forbidden in (
            "import run_cascade3_newsplit_training",
            "import run_stage2_classification",
            "import run_nfft_comparison",
        ):
            self.assertNotIn(forbidden, source)

    def test_training_never_references_a_test_role_reader(self) -> None:
        source = (PROJECT_DIR / "run_stage1_12drones_training.py").read_text(encoding="utf-8")
        self.assertIn('TEST_ROLES = ("development_test", "development_test_exposed")', source)
        self.assertIn("split=TRAIN_ROLE", source)
        self.assertIn("split=VAL_ROLE", source)
        self.assertNotIn('split="development_test"', source)
        self.assertNotIn('split="development_test_exposed"', source)


class TrainingSeedParameterisationTests(unittest.TestCase):
    """STAGE1-12DRONES-B: ``--training-seed`` must change the seed and nothing else."""

    #: The accepted STAGE1-12DRONES-A artefacts, frozen by the Codex acceptance.
    ACCEPTED_CHECKPOINT_SHA256 = (
        "37dbb815450b8ec0f798a113a7f922858fcb2be604a863eb7594eb5c52ff814b"
    )
    ACCEPTED_RUN_ROOT = PROJECT_DIR / "outputs" / "stage1_12drones_20260926" / "seed_2026"

    def setUp(self) -> None:
        self._driver_seed = training.TRAINING_SEED
        self._verifier_seed = verifier.TRAINING_SEED

    def tearDown(self) -> None:
        training.bind_training_seed(self._driver_seed)
        verifier.bind_training_seed(self._verifier_seed)

    def test_legacy_default_is_still_the_accepted_seed_2026(self) -> None:
        self.assertEqual(training.DEFAULT_TRAINING_SEED, 2026)
        self.assertEqual(training.TRAINING_SEED, 2026)
        self.assertEqual(verifier.DEFAULT_TRAINING_SEED, 2026)
        self.assertEqual(verifier.TRAINING_SEED, 2026)
        args = training.parse_args([])
        self.assertEqual(args.training_seed, 2026)
        self.assertIsNone(args.run_root)
        self.assertEqual(training.run_root_for(args.training_seed), self.ACCEPTED_RUN_ROOT)

    def test_run_directories_are_one_per_seed(self) -> None:
        seed_2026 = training.run_root_for(2026)
        seed_2027 = training.run_root_for(2027)
        self.assertNotEqual(seed_2026, seed_2027)
        self.assertEqual(seed_2026.name, "seed_2026")
        self.assertEqual(seed_2027.name, "seed_2027")
        self.assertEqual(seed_2026.parent, seed_2027.parent)
        self.assertEqual(seed_2027.parent, training.OUTPUT_ROOT)
        self.assertEqual(verifier.run_root_for(2027), seed_2027)

    def test_binding_a_seed_leaves_the_validation_seed_pinned(self) -> None:
        training.bind_training_seed(2027)
        self.assertEqual(training.TRAINING_SEED, 2027)
        self.assertEqual(training.VALIDATION_SEED, 12026)
        self.assertEqual(training.phase_for(2027), "STAGE1-12DRONES-seed2027")
        self.assertEqual(training.phase_for(2026), "STAGE1-12DRONES-A")
        training.bind_training_seed(2026)
        self.assertEqual(training.TRAINING_SEED, 2026)

    def test_every_seed_dependent_derivation_follows_the_bound_seed(self) -> None:
        """Initialisation, loader, shuffle and noise seeds must all move together."""
        source = (PROJECT_DIR / "run_stage1_12drones_training.py").read_text(encoding="utf-8")
        self.assertIn("seed_everything(TRAINING_SEED)", source)
        self.assertIn("manual_seed(TRAINING_SEED + 50_000)", source)
        self.assertIn("seed=TRAINING_SEED", source)
        # The per-epoch shuffle lives in the loader module since STAGE1-IO-OPT, and the
        # driver hands it the run seed explicitly (rework R1: over the full split, then
        # filtered to the positions actually in use).
        self.assertIn("order_for_epoch(dataset, epoch, TRAINING_SEED)", source)
        self.assertIn(
            "limited_order(\n                order_for_epoch(dataset, epoch, TRAINING_SEED), "
            "train_positions\n            )",
            source,
        )
        self.assertNotIn("manual_seed(TRAINING_SEED + epoch)", source)
        loader_source = (PROJECT_DIR / "stage1_loader_optimization.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("torch.Generator().manual_seed(int(seed) + int(epoch))", loader_source)
        # The validation side must stay pinned to its own seed for every run seed.
        self.assertIn("seed=VALIDATION_SEED", source)
        # The validation feature cache is read through an explicit position sampler, so
        # its loader generator is no longer consulted; the deterministic eval loader
        # keeps the pinned validation generator.
        self.assertIn("manual_seed(VALIDATION_SEED + 40_000)", source)
        self.assertIn("VALIDATION_SEED = 12026", source)
        for line in source.splitlines():
            if "VALIDATION_SEED" in line and "=" in line and "TRAINING_SEED" in line:
                self.fail(f"validation seed must not be derived from the run seed: {line!r}")
        verifier_source = (PROJECT_DIR / "verify_stage1_12drones_training.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("VALIDATION_SEED = 12026", verifier_source)
        self.assertNotIn("seed=VALIDATION_SEED", verifier_source)
        self.assertIn("rng = np.random.default_rng(VALIDATION_SEED + position)", verifier_source)

    def test_status_report_targets_the_requested_seed_only(self) -> None:
        training.bind_training_seed(2027)
        report = training.status_report()
        self.assertEqual(report["training_seed"], 2027)
        self.assertEqual(report["validation_seed"], 12026)
        self.assertEqual(report["run_root"], str(training.run_root_for(2027)))
        self.assertIn("seed_2027", report["runs"])
        self.assertNotIn("seed_2026", report["runs"])

    def test_verifier_binds_the_seed_and_names_its_report_per_seed(self) -> None:
        args = verifier.parse_args(["--training-seed", "2027"])
        self.assertEqual(args.training_seed, 2027)
        self.assertIsNone(args.run_root)
        self.assertIsNone(args.output)
        verifier.bind_training_seed(2027)
        self.assertEqual(verifier.TRAINING_SEED, 2027)
        self.assertEqual(verifier.run_root_for(2027).name, "seed_2027")

    def test_second_seed_cannot_overwrite_the_accepted_run(self) -> None:
        self.assertNotEqual(training.run_root_for(2027), self.ACCEPTED_RUN_ROOT)
        self.assertTrue((self.ACCEPTED_RUN_ROOT / "best_model.pt").is_file())
        self.assertEqual(
            verifier.sha256_file(self.ACCEPTED_RUN_ROOT / "best_model.pt"),
            self.ACCEPTED_CHECKPOINT_SHA256,
        )
        # The second seed may hold its own checkpoint (it does once training starts),
        # but it must be a different file in a different directory.
        candidate = training.run_root_for(2027) / "best_model.pt"
        if candidate.is_file():
            self.assertNotEqual(
                verifier.sha256_file(candidate), self.ACCEPTED_CHECKPOINT_SHA256
            )
            self.assertEqual(candidate.parent.name, "seed_2027")
            self.assertEqual(
                candidate.parent, training.OUTPUT_ROOT / "seed_2027"
            )
        self.assertNotEqual(
            self.ACCEPTED_RUN_ROOT,
            training.OUTPUT_ROOT / "seed_2027",
        )

    def test_seed_2027_artefacts_when_present(self) -> None:
        run_root = training.run_root_for(2027)
        if not (run_root / "summary.json").is_file():
            self.skipTest("seed-2027 run has not completed yet")
        summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["training_seed"], 2027)
        self.assertEqual(summary["validation_seed"], 12026)
        self.assertEqual(summary["phase"], "STAGE1-12DRONES-seed2027")
        self.assertEqual(len(summary["history"]), 5)
        self.assertEqual(summary["selected_epoch"], verifier.select_epoch(summary["history"]))
        self.assertTrue(summary["read_audit_passed"])
        self.assertTrue(summary["checkpoint_reload_passed"])
        config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["training_seed"], 2027)
        self.assertEqual(config["validation_seed"], 12026)
        self.assertEqual(config["epochs"], 5)
        self.assertEqual(config["train_windows_used_per_epoch"], 9590)
        self.assertEqual(config["validation_windows_used_per_epoch"], 2360)
        self.assertTrue(config["test_windows_never_read"])
        self.assertEqual(config["read_audit_opened_files"], 78)
        self.assertEqual(config["epochs"], len(summary["history"]))
        self.assertEqual(
            verifier.sha256_file(run_root / "best_model.pt"), config["best_model_sha256"]
        )
        payload = torch.load(run_root / "best_model.pt", map_location="cpu", weights_only=False)
        self.assertEqual(int(payload["training_seed"]), 2027)
        self.assertEqual(int(payload["validation_seed"]), 12026)
        self.assertEqual(int(payload["epoch"]), summary["selected_epoch"])
        # A different training seed must actually change the run: the shuffled
        # per-epoch orders, the initial weights and the final checkpoint differ.
        accepted = json.loads(
            (self.ACCEPTED_RUN_ROOT / "run_config.json").read_text(encoding="utf-8")
        )
        self.assertNotEqual(
            config["epoch_order_sha256"], accepted["epoch_order_sha256"]
        )
        self.assertNotEqual(
            config["initial_model_sha256"], accepted["initial_model_sha256"]
        )
        self.assertNotEqual(config["best_model_sha256"], accepted["best_model_sha256"])
        self.assertEqual(
            config["index_sha256"], accepted["index_sha256"], "frozen index must be shared"
        )
        self.assertEqual(
            config["split_source_sha256"],
            accepted["split_source_sha256"],
            "frozen split must be shared",
        )

    def test_seed_2027_verification_report_when_present(self) -> None:
        run_root = training.run_root_for(2027)
        report_path = (
            PROJECT_DIR
            / "outputs"
            / "stage1_12drones_20260926"
            / "verification_seed_2027.json"
        )
        if not report_path.is_file():
            self.skipTest("seed-2027 independent verification has not run yet")
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertTrue(payload["passed"], payload["verification"]["failed_checks"])
        self.assertEqual(
            payload["verification"]["passed_checks"], payload["verification"]["total_checks"]
        )
        self.assertTrue(payload["negative_controls"]["passed"])
        self.assertEqual(payload["training_seed"], 2027)
        self.assertEqual(payload["validation_seed"], 12026)
        self.assertEqual(Path(payload["run_root"]), run_root)
        config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["read_audit"]["epochs_completed"], config["epochs"])

    def test_seed_2026_verification_report_is_still_the_accepted_one(self) -> None:
        payload = json.loads(
            (
                PROJECT_DIR / "outputs" / "stage1_12drones_20260926" / "verification.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(payload["passed"], payload["verification"]["failed_checks"])
        self.assertEqual(payload["run_root"], str(self.ACCEPTED_RUN_ROOT))
        self.assertEqual(
            payload["negative_controls"]["detected"], payload["negative_controls"]["total"]
        )


class SeedComparisonToolTests(unittest.TestCase):
    """The seed comparison must reuse the frozen metric and decision definitions."""

    def test_self_test_controls_pass(self) -> None:
        comparator._self_test()

    def test_margin_auc_equals_the_driver_definition(self) -> None:
        from train_resnet18 import binary_metrics

        rng = np.random.default_rng(11)
        for trial in range(50):
            count = int(rng.integers(1, 25))
            background_count = int(rng.integers(1, 25))
            drone = rng.normal(0.3, 1.0, size=count)
            background = rng.normal(0.0, 1.2, size=background_count)
            if trial % 5 == 0:
                drone[:2] = 0.1
                background[:2] = 0.1
            labels = [1] * count + [0] * background_count
            probabilities = list(1 / (1 + np.exp(-drone))) + list(
                1 / (1 + np.exp(-background))
            )
            reference = binary_metrics(labels, [0] * (count + background_count), probabilities)[
                "roc_auc"
            ]
            self.assertAlmostEqual(
                comparator.auc_pairwise(drone, background), reference, places=12
            )

    def test_decode_aligns_by_index_position(self) -> None:
        rows = [
            {
                "condition": "clean",
                "index_position": 1,
                "filename": "b.mat",
                "target_code": "T0001",
                "window_index": 5,
                "start": 5_000_000,
                "label": 1,
                "margin": 0.25,
                "prediction": 1,
            },
            {
                "condition": "clean",
                "index_position": 0,
                "filename": "a.mat",
                "target_code": "T0000",
                "window_index": 0,
                "start": 0,
                "label": 0,
                "margin": -0.25,
                "prediction": 0,
            },
        ]
        decoded = comparator.decode(rows)
        self.assertEqual(list(decoded["index_position"]), [0, 1])
        self.assertEqual(list(decoded["filename"]), ["a.mat", "b.mat"])
        self.assertEqual(list(decoded["target_code"]), ["T0000", "T0001"])

    def test_comparison_pins_the_validation_seed(self) -> None:
        self.assertEqual(comparator.VALIDATION_SEED, verifier.VALIDATION_SEED)
        self.assertEqual(comparator.VALIDATION_SEED, training.VALIDATION_SEED)
        self.assertEqual(comparator.ACCEPTED_SEED, training.DEFAULT_TRAINING_SEED)
        self.assertNotEqual(comparator.CANDIDATE_SEED, comparator.ACCEPTED_SEED)
        self.assertEqual(comparator.INDEX_SHA256, verifier.INDEX_SHA256)
        self.assertEqual(comparator.run_root_for(2027), training.run_root_for(2027))

    def test_comparison_artefacts_when_present(self) -> None:
        output_dir = comparator.OUTPUT_ROOT / "seed_comparison"
        report_path = (
            PROJECT_DIR.parent / "reports" / "DroneRFa_一阶段12机型_seed复跑对照_20260927.md"
        )
        if not (output_dir / "seed_comparison.json").is_file():
            self.skipTest("seed comparison has not run yet")
        payload = json.loads((output_dir / "seed_comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["seeds"], [2026, 2027])
        self.assertEqual(payload["max_recorded_metric_delta"], 0.0)
        self.assertEqual(
            payload["identity"]["same_validation_inputs"]["validation_seed"],
            training.VALIDATION_SEED,
        )
        self.assertEqual(
            payload["identity"]["same_validation_inputs"]["accepted_validation_seed"],
            payload["identity"]["same_validation_inputs"]["candidate_validation_seed"],
        )
        for condition in comparator.CONDITIONS:
            entry = payload["per_window"][condition]
            self.assertEqual(entry["windows"], 2360)
            disagreement = entry["disagreement"]
            self.assertEqual(
                disagreement["both_correct"]
                + disagreement["both_wrong"]
                + disagreement["accepted_only_wrong"]
                + disagreement["candidate_only_wrong"],
                2360,
            )
            self.assertAlmostEqual(
                disagreement["accepted_only_wrong"]
                + disagreement["candidate_only_wrong"],
                disagreement["windows"],
                places=0,
            )
        self.assertTrue(
            any(
                payload["per_window"][condition]["disagreement"]["windows"] > 0
                for condition in comparator.CONDITIONS
            ),
            "a new training seed should move at least one validation decision",
        )
        self.assertEqual(
            payload["per_window_scope"],
            "both seeds' selected epoch"
            if payload["same_selected_epoch"]
            else f"seed 2027 selected epoch {payload['candidate_selected_epoch']}",
        )
        self.assertTrue(report_path.is_file())
        report = report_path.read_text(encoding="utf-8")
        for required in (
            "开发验证指标",
            "不是独立测试",
            "12 类机型识别率",
            "validation_seed",
            "12026",
        ):
            self.assertIn(required, report)


class SyntheticIndexRoundTripTests(unittest.TestCase):
    """Build a tiny frozen-shaped split and prove the index builder covers it."""

    def test_window_count_is_floor_of_samples(self) -> None:
        self.assertEqual(builder.window_count(1_000_000), 1)
        self.assertEqual(builder.window_count(1_999_999), 1)
        self.assertEqual(builder.window_count(150_000_000), 150)
        with self.assertRaises(ValueError):
            builder.window_count(-1)

    def test_probe_slice_offsets_stay_inside_the_window(self) -> None:
        offsets = builder._slice_offsets(5_000_000)
        self.assertEqual(len(offsets), builder.PROBE_SLICE_COUNT)
        for offset in offsets:
            self.assertGreaterEqual(offset, 5_000_000)
            self.assertLessEqual(
                offset + builder.PROBE_SLICE_SAMPLES, 5_000_000 + builder.WINDOW_SIZE
            )

    def test_write_csv_refuses_an_empty_index(self) -> None:
        scratch = PROJECT_DIR / "scratch"
        scratch.mkdir(exist_ok=True)
        target = scratch / "stage1_empty_index_probe.csv"
        try:
            with self.assertRaises(ValueError):
                builder.write_csv(target, [])
            self.assertFalse(target.exists())
        finally:
            target.unlink(missing_ok=True)

    def test_build_rows_inherits_source_role_and_label(self) -> None:
        source = builder.FrozenSource(
            filename="T0001_D00_S0000.mat",
            file_path=Path(r"G:\DroneRFa数据集\original\T0001_D00_S0000.mat"),
            target_code="T0001",
            role="train",
            size_bytes=1,
            class_id="T0001",
            class_name="DJI Mini 3",
            class_index="1",
            segment_code="0000",
            distance_code="00",
            metrics_scope="development",
            role_purpose="model fitting",
            prior_use_role="train",
            archive_segment_role="train",
            exposed_to_prior_selection="False",
            expected_sha256="a" * 64,
        )
        identities = {
            source.filename: {
                "source_samples": 2_500_000,
                "complete_windows": 2,
                "dropped_tail_samples": 500_000,
            }
        }
        rows = builder.build_rows([source], identities)
        self.assertEqual(len(rows), 2)
        for window_index, row in enumerate(rows):
            self.assertEqual(row["label"], 1)
            self.assertEqual(row["stage1_label"], 1)
            self.assertEqual(row["source_role"], "train")
            self.assertEqual(row["split"], "train")
            self.assertEqual(row["window_index"], window_index)
            self.assertEqual(row["start"], window_index * 1_000_000)
            self.assertEqual(row["window_in_block"], window_index % 10)


def builder_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
