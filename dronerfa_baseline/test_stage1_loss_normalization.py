#!/usr/bin/env python3
"""Tests for STAGE1-LOSS-NORM: criterion maths, frozen-identity gate, real-input sanity.

Three groups:

* pure-logic tests of the two normalisations (formula, tail batch, default-mode
  equality with ``nn.CrossEntropyLoss(weight=...)``, gradients, non-finite guards);
* the frozen pre-run gate: the initialisation hash and all five epoch order hashes
  must equal the accepted ``STAGE1-12DRONES-seed2027`` control, and the class
  weights must match the control's recorded float32 values;
* a bounded sanity run over **real** frozen train windows (G: drive, RF0) that
  proves, on real inputs, that the historical mode reproduces the original module
  and that the candidate mode applies the documented denominator.

Nothing here trains the formal protocol, writes under ``outputs/`` or touches a
``development_test*`` source file.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import torch
from torch import nn

import loss_normalization as ln
import run_stage1_12drones_training as base
import run_stage1_loss_normalization as entry
import verify_stage1_loss_normalization as verifier

#: Initialisation hash of the accepted seed-2027 control run.
SEED2027_INITIAL_MODEL_SHA256 = (
    "d774eeb6e3f9ec15c2b17cf5ceeab1bce68c970481f9a4d96718bc2dc358aef9"
)
#: Five epoch order hashes recorded by the accepted seed-2027 control run.
SEED2027_EPOCH_ORDER_SHA256 = (
    "79b7c0ff278f0ae521ec3219515fbb36f25d3137f9fcebb7b89cac61e3d66b80",
    "78e408fbc71af6303f15459b15f6e6d6ef2da13c5f0a133f86f80724415d1a02",
    "cb3799aa0c4246d72f671dbd10a400fc41349df7f44e4143a7d6c3efb9c462c1",
    "f480b1eb3d4b011b4c42b421dd2d597488fde6bd2fdff510fb82025af38315d7",
    "a1d2f3ff289edd125b9ae89e3c5b2e967a44c495d6e9ad85cd5d6cf231095ae4",
)
FROZEN_TRAIN_COUNTS = {0: 600, 1: 8990}
FROZEN_CLASS_WEIGHTS = (7.991666793823242, 0.5333704352378845)


class LossSpecTests(unittest.TestCase):
    def test_frozen_counts_reproduce_the_control_weights(self) -> None:
        spec = ln.build_loss_spec(ln.LOSS_MODE_GLOBAL_TRAIN_MEAN, FROZEN_TRAIN_COUNTS, 4)
        self.assertEqual(spec.class_weights, FROZEN_CLASS_WEIGHTS)
        self.assertEqual(spec.train_windows, 9590)
        self.assertEqual(spec.batch_size, 4)
        self.assertAlmostEqual(spec.mu_train, 1.0, places=6)

    def test_batch_layout_and_tail_batch(self) -> None:
        spec = ln.build_loss_spec(ln.LOSS_MODE_GLOBAL_TRAIN_MEAN, FROZEN_TRAIN_COUNTS, 4)
        sizes = spec.actual_batch_sizes
        self.assertEqual(len(sizes), spec.batches_per_epoch)
        self.assertEqual(len(sizes), 2398)
        self.assertEqual(sizes[-1], 2)
        self.assertEqual(set(sizes[:-1]), {4})
        self.assertEqual(spec.tail_batch_size, 2)
        self.assertEqual(spec.tail_batches_per_epoch, 1)
        self.assertEqual(sum(sizes), 9590)

    def test_mu_train_is_the_weighted_mean_of_the_weights(self) -> None:
        spec = ln.build_loss_spec(ln.LOSS_MODE_BATCH_WEIGHT_SUM, FROZEN_TRAIN_COUNTS, 4)
        recomputed = sum(
            count * weight for count, weight in zip(FROZEN_TRAIN_COUNTS.values(), spec.class_weights)
        ) / spec.train_windows
        self.assertAlmostEqual(spec.mu_train, recomputed, places=12)

    def test_unknown_mode_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            ln.build_loss_spec("mean_of_batch", FROZEN_TRAIN_COUNTS, 4)

    def test_payload_documents_formula_denominator_and_tail_rule(self) -> None:
        payload = ln.build_loss_spec(
            ln.LOSS_MODE_GLOBAL_TRAIN_MEAN, FROZEN_TRAIN_COUNTS, 4
        ).payload()
        self.assertEqual(payload["formula"], "L = sum_i(w_yi * CE_i) / (B_actual * mu_train)")
        self.assertIn("B_actual", payload["denominator_rule"])
        self.assertIn("2 * mu_train", payload["tail_batch_rule"])
        self.assertEqual(payload["batches_per_epoch"], 2398)


class CriterionTests(unittest.TestCase):
    """The two normalisations, on fixed logits, in float32 exactly."""

    @staticmethod
    def _weights() -> torch.Tensor:
        return torch.tensor(FROZEN_CLASS_WEIGHTS, dtype=torch.float32)

    def _logits(self, labels: list[int]) -> torch.Tensor:
        generator = torch.Generator().manual_seed(11)
        return torch.randn(len(labels), 2, generator=generator, dtype=torch.float32) * 2.0

    def test_reimplementation_agrees_with_the_original_module(self) -> None:
        """The re-implementation is the documented formula, not a byte-identical copy.

        PyTorch's own ``reduction='mean'`` path accumulates ``sum(w*CE)/sum(w)`` inside
        the C++ kernel; re-deriving it from ``reduction='none'`` can differ in the last
        bits.  That is exactly why ``batch_weight_sum`` runs use the original module
        (see :func:`build_criterion`) instead of this class, and the difference is
        asserted to stay at the floating-point-noise level.
        """
        module = nn.CrossEntropyLoss(weight=self._weights())
        max_absolute = 0.0
        max_relative = 0.0
        for labels in ([0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1], [0, 1], [0, 0]):
            logits = self._logits(labels)
            target = torch.tensor(labels, dtype=torch.long)
            reference = float(module(logits, target))
            candidate = float(
                ln.NormalizedWeightedCrossEntropy(
                    ln.LOSS_MODE_BATCH_WEIGHT_SUM, FROZEN_CLASS_WEIGHTS
                )(logits, target)
            )
            delta = abs(reference - candidate)
            max_absolute = max(max_absolute, delta)
            max_relative = max(max_relative, delta / max(abs(reference), 1e-12))
        self.assertLessEqual(max_relative, 1e-6, f"max relative delta {max_relative}")
        self.assertLessEqual(max_absolute, 1e-6, f"max absolute delta {max_absolute}")

    def test_gradients_agree_with_the_original_module(self) -> None:
        for labels in ([0, 1, 1, 1], [1, 1, 1, 1], [0, 1]):
            logits = self._logits(labels)
            target = torch.tensor(labels, dtype=torch.long)
            left = logits.clone().requires_grad_(True)
            right = logits.clone().requires_grad_(True)
            nn.CrossEntropyLoss(weight=self._weights())(left, target).backward()
            ln.NormalizedWeightedCrossEntropy(
                ln.LOSS_MODE_BATCH_WEIGHT_SUM, FROZEN_CLASS_WEIGHTS
            )(right, target).backward()
            delta = float((left.grad - right.grad).abs().max())
            self.assertLessEqual(delta, 1e-6, f"labels={labels}: max |dgrad| = {delta}")

    def test_build_criterion_default_mode_is_bitwise_the_original_module(self) -> None:
        spec = ln.build_loss_spec(ln.LOSS_MODE_BATCH_WEIGHT_SUM, FROZEN_TRAIN_COUNTS, 4)
        criterion = ln.build_criterion(spec)
        reference = nn.CrossEntropyLoss(weight=self._weights())
        for labels in ([0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1], [0, 1]):
            logits = self._logits(labels)
            target = torch.tensor(labels, dtype=torch.long)
            self.assertTrue(
                bool(torch.equal(criterion(logits, target), reference(logits, target))),
                f"labels={labels}",
            )

    def test_global_train_mean_denominator_is_the_actual_batch_size(self) -> None:
        spec = ln.build_loss_spec(ln.LOSS_MODE_GLOBAL_TRAIN_MEAN, FROZEN_TRAIN_COUNTS, 4)
        criterion = ln.NormalizedWeightedCrossEntropy(
            ln.LOSS_MODE_GLOBAL_TRAIN_MEAN,
            spec.class_weights,
            spec.mu_train,
            record_steps=True,
        )
        module = nn.CrossEntropyLoss(weight=self._weights(), reduction="none")
        for labels in ([0, 1, 1, 1], [1, 1, 1, 1], [0, 1], [0, 0]):
            logits = self._logits(labels)
            target = torch.tensor(labels, dtype=torch.long)
            loss = criterion(logits, target)
            numerator = float(module(logits, target).sum())
            expected = numerator / (len(labels) * spec.mu_train)
            self.assertAlmostEqual(float(loss), expected, places=6)
            step = criterion.steps[-1]
            self.assertEqual(step["batch_size"], len(labels))
            self.assertEqual(
                step["denominator"], float(torch.tensor(len(labels) * spec.mu_train, dtype=torch.float32))
            )

    def test_tail_batch_keeps_the_global_scale(self) -> None:
        spec = ln.build_loss_spec(ln.LOSS_MODE_GLOBAL_TRAIN_MEAN, FROZEN_TRAIN_COUNTS, 4)
        criterion = ln.NormalizedWeightedCrossEntropy(
            ln.LOSS_MODE_GLOBAL_TRAIN_MEAN,
            spec.class_weights,
            spec.mu_train,
            record_steps=True,
        )
        logits = self._logits([0, 1])
        target = torch.tensor([0, 1], dtype=torch.long)
        loss = criterion(logits, target)
        tail = criterion.steps[-1]
        self.assertEqual(tail["batch_size"], 2)
        self.assertAlmostEqual(tail["denominator"], 2 * spec.mu_train, places=6)
        # Dividing the two-window tail batch by the nominal batch size 4 would halve
        # its gradient; the recorded denominator must not do that.
        self.assertNotAlmostEqual(tail["denominator"], 4 * spec.mu_train, places=6)
        self.assertAlmostEqual(float(loss) * tail["denominator"], tail["numerator"], places=6)

    def test_batch_weight_sum_cancels_the_weights_on_a_single_class_batch(self) -> None:
        # The mechanism the task is about: on a drone-only batch the historical
        # normalisation divides the drone weight back out, so a background window in
        # a mixed batch carries a much larger share of the batch gradient than the
        # global weights would give it.
        spec = ln.build_loss_spec(ln.LOSS_MODE_GLOBAL_TRAIN_MEAN, FROZEN_TRAIN_COUNTS, 4)
        weights = torch.tensor(spec.class_weights, dtype=torch.float32)
        pure = torch.tensor([1, 1, 1, 1])
        mixed = torch.tensor([0, 1, 1, 1])
        historical_pure = float(weights[pure].sum())
        historical_mixed = float(weights[mixed].sum())
        self.assertAlmostEqual(historical_pure, 4 * FROZEN_CLASS_WEIGHTS[1], places=6)
        self.assertGreater(historical_mixed, 2 * historical_pure)
        # global_train_mean: effective per-window weight is w_c / (B * mu_train)
        effective_background = FROZEN_CLASS_WEIGHTS[0] / (4 * spec.mu_train)
        effective_drone = FROZEN_CLASS_WEIGHTS[1] / (4 * spec.mu_train)
        self.assertGreater(effective_background, 10 * effective_drone)

    def test_build_criterion_returns_the_original_module_for_the_default_mode(self) -> None:
        default_spec = ln.build_loss_spec(
            ln.LOSS_MODE_BATCH_WEIGHT_SUM, FROZEN_TRAIN_COUNTS, 4
        )
        self.assertIsInstance(ln.build_criterion(default_spec), nn.CrossEntropyLoss)
        candidate_spec = ln.build_loss_spec(
            ln.LOSS_MODE_GLOBAL_TRAIN_MEAN, FROZEN_TRAIN_COUNTS, 4
        )
        self.assertIsInstance(
            ln.build_criterion(candidate_spec), ln.NormalizedWeightedCrossEntropy
        )
        self.assertIsInstance(
            ln.build_validation_criterion(candidate_spec), nn.CrossEntropyLoss
        )

    def test_non_finite_loss_is_detectable(self) -> None:
        criterion = ln.NormalizedWeightedCrossEntropy(
            ln.LOSS_MODE_GLOBAL_TRAIN_MEAN,
            FROZEN_CLASS_WEIGHTS,
            1.0,
            record_steps=True,
        )
        logits = torch.full((4, 2), float("nan"))
        target = torch.tensor([0, 1, 1, 1], dtype=torch.long)
        loss = criterion(logits, target)
        self.assertFalse(bool(torch.isfinite(loss)))
        summary = ln.step_evidence_summary(criterion.steps)
        self.assertFalse(summary["all_finite"])

    def test_step_summary_reports_the_tail_batch(self) -> None:
        criterion = ln.NormalizedWeightedCrossEntropy(
            ln.LOSS_MODE_GLOBAL_TRAIN_MEAN, FROZEN_CLASS_WEIGHTS, 1.0, record_steps=True
        )
        for labels in ([0, 1, 1, 1], [1, 1, 1, 1], [0, 1]):
            criterion(self._logits(labels), torch.tensor(labels, dtype=torch.long))
        summary = ln.step_evidence_summary(criterion.steps)
        self.assertEqual(summary["steps"], 3)
        self.assertEqual(summary["windows"], 10)
        self.assertEqual(summary["batch_size_histogram"], {"2": 1, "4": 2})
        self.assertEqual(summary["non_nominal_batch_sizes"], [2])
        self.assertTrue(summary["all_finite"])


class ObjectiveLossProbeTests(unittest.TestCase):
    def test_primary_loss_is_unchanged_and_calls_land_in_the_right_cell(self) -> None:
        weights = torch.tensor(FROZEN_CLASS_WEIGHTS, dtype=torch.float32)
        primary = nn.CrossEntropyLoss(weight=weights)
        secondary = ln.NormalizedWeightedCrossEntropy(
            ln.LOSS_MODE_GLOBAL_TRAIN_MEAN, FROZEN_CLASS_WEIGHTS, 1.0
        )
        probe = ln.ObjectiveLossProbe(
            primary, secondary, conditions=("clean", "-10dB"), calls_per_condition=2
        )
        generator = torch.Generator().manual_seed(3)
        for call in range(6):
            logits = torch.randn(4, 2, generator=generator)
            target = torch.tensor([0, 1, 1, 1], dtype=torch.long)
            expected = primary(logits, target)
            observed = probe(logits, target)
            self.assertTrue(bool(torch.equal(expected, observed)))
        payload = probe.mean_by_cell()
        self.assertEqual(payload["calls"], 6)
        self.assertEqual(payload["cycles"], 2)
        self.assertEqual(payload["by_cycle"]["0"]["clean"]["windows"], 8)
        self.assertEqual(payload["by_cycle"]["0"]["-10dB"]["windows"], 8)
        # cycle 1 only received the first condition's two calls
        self.assertEqual(payload["by_cycle"]["1"]["clean"]["windows"], 8)
        self.assertNotIn("-10dB", payload["by_cycle"]["1"])


class FrozenIdentityPreflightTests(unittest.TestCase):
    """Read-only: no RF0 payload and no test-role file is touched."""

    def test_preflight_gate_matches_the_accepted_control(self) -> None:
        identity = base.check_index_identity()
        self.assertTrue(identity["passed"], identity["errors"])
        spec = ln.build_loss_spec(
            ln.LOSS_MODE_GLOBAL_TRAIN_MEAN,
            identity["observed"]["stage1_label_counts"][base.TRAIN_ROLE],
            base.BATCH_SIZE,
        )
        control = entry.control_reference()
        self.assertEqual(
            control["control"]["initial_model_sha256"], SEED2027_INITIAL_MODEL_SHA256
        )
        self.assertEqual(
            tuple(control["control"]["epoch_order_sha256"]), SEED2027_EPOCH_ORDER_SHA256
        )
        preflight = entry.preflight_gate(seed=2027, epochs=5, spec=spec, control=control)
        self.assertTrue(preflight["passed"], preflight["errors"])
        self.assertEqual(preflight["initial_model_sha256"], SEED2027_INITIAL_MODEL_SHA256)
        self.assertEqual(tuple(preflight["epoch_order_sha256"]), SEED2027_EPOCH_ORDER_SHA256)
        self.assertTrue(preflight["initial_model_matches_control"])
        self.assertTrue(preflight["epoch_order_matches_control"])
        self.assertEqual(preflight["split_length"], 9590)

    def test_preflight_gate_refuses_a_wrong_initial_hash(self) -> None:
        identity = base.check_index_identity()
        spec = ln.build_loss_spec(
            ln.LOSS_MODE_GLOBAL_TRAIN_MEAN,
            identity["observed"]["stage1_label_counts"][base.TRAIN_ROLE],
            base.BATCH_SIZE,
        )
        control = entry.control_reference()
        control["control"]["initial_model_sha256"] = "0" * 64
        preflight = entry.preflight_gate(seed=2027, epochs=5, spec=spec, control=control)
        self.assertFalse(preflight["passed"])
        self.assertTrue(any("initial model hash" in error for error in preflight["errors"]))

    def test_preflight_gate_refuses_a_wrong_order_hash(self) -> None:
        identity = base.check_index_identity()
        spec = ln.build_loss_spec(
            ln.LOSS_MODE_GLOBAL_TRAIN_MEAN,
            identity["observed"]["stage1_label_counts"][base.TRAIN_ROLE],
            base.BATCH_SIZE,
        )
        control = entry.control_reference()
        orders = list(control["control"]["epoch_order_sha256"])
        orders[2] = "f" * 64
        control["control"]["epoch_order_sha256"] = orders
        control["control"]["initial_model_sha256"] = SEED2027_INITIAL_MODEL_SHA256
        preflight = entry.preflight_gate(seed=2027, epochs=5, spec=spec, control=control)
        self.assertFalse(preflight["passed"])
        self.assertTrue(any("epoch 3 order hash" in error for error in preflight["errors"]))


class ResumeStateTests(unittest.TestCase):
    @staticmethod
    def _identity(**overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "index_sha256": base.INDEX_SHA256,
            "split_source_sha256": base.FROZEN_SPLIT_CSV_SHA256,
            "training_seed": 2027,
            "validation_seed": 12026,
            "epochs": 5,
            "loss_mode": ln.LOSS_MODE_GLOBAL_TRAIN_MEAN,
            "protocol_sha256": "a" * 64,
        }
        payload.update(overrides)
        return payload

    def test_identity_errors_are_reported(self) -> None:
        state = {
            "schema": base.TRAINING_STATE_SCHEMA,
            "epoch": 2,
            "next_epoch": 3,
            "identity": self._identity(),
        }
        self.assertEqual(
            base.resume_identity_errors(
                state,
                index_sha256=base.INDEX_SHA256,
                split_source_sha256=base.FROZEN_SPLIT_CSV_SHA256,
                training_seed=2027,
                validation_seed=12026,
                epochs=5,
            ),
            [],
        )
        wrong_seed = base.resume_identity_errors(
            state,
            index_sha256=base.INDEX_SHA256,
            split_source_sha256=base.FROZEN_SPLIT_CSV_SHA256,
            training_seed=2026,
            validation_seed=12026,
            epochs=5,
        )
        self.assertTrue(any("training_seed" in error for error in wrong_seed))
        finished = dict(state, epoch=5, next_epoch=6)
        errors = base.resume_identity_errors(
            finished,
            index_sha256=base.INDEX_SHA256,
            split_source_sha256=base.FROZEN_SPLIT_CSV_SHA256,
            training_seed=2027,
            validation_seed=12026,
            epochs=5,
        )
        self.assertTrue(any("already completed" in error for error in errors))

    def test_apply_training_state_restores_modules_and_refuses_a_wrong_init(self) -> None:
        base.seed_everything(2027)
        model = base.ResNet18(input_channels=1, num_classes=2)
        initial_hash = base.state_hash(model.state_dict())
        frontend = base.SpectrogramFrontEnd(n_fft=512, hop_length=256, image_size=256)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        logits = model(torch.zeros(1, 1, 256, 256))
        nn.functional.cross_entropy(logits, torch.tensor([0])).backward()
        optimizer.step()
        trained_hash = base.state_hash(model.state_dict())
        self.assertNotEqual(initial_hash, trained_hash)

        state = {
            "schema": base.TRAINING_STATE_SCHEMA,
            "epoch": 1,
            "next_epoch": 2,
            "model_state": model.state_dict(),
            "frontend_state": frontend.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "rng": base.rng_state_payload(),
            "initial_model_sha256": initial_hash,
        }
        base.seed_everything(2027)
        fresh = base.ResNet18(input_channels=1, num_classes=2)
        fresh_frontend = base.SpectrogramFrontEnd(n_fft=512, hop_length=256, image_size=256)
        fresh_optimizer = torch.optim.Adam(fresh.parameters(), lr=1e-3)
        fresh_scaler = torch.amp.GradScaler("cpu", enabled=False)
        info = base.apply_training_state(
            state,
            model=fresh,
            frontend=fresh_frontend,
            optimizer=fresh_optimizer,
            scaler=fresh_scaler,
            expected_initial_hash=initial_hash,
        )
        self.assertEqual(info["resumed_from_epoch"], 1)
        self.assertEqual(info["next_epoch"], 2)
        self.assertEqual(base.state_hash(fresh.state_dict()), trained_hash)
        self.assertIn("torch_cpu", info["rng"]["restored"])

        with self.assertRaises(RuntimeError):
            base.apply_training_state(
                state,
                model=fresh,
                frontend=fresh_frontend,
                optimizer=fresh_optimizer,
                scaler=fresh_scaler,
                expected_initial_hash="b" * 64,
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the real-input sanity run")
class RealWindowSanityTests(unittest.TestCase):
    """Bounded, real RF0 windows read once and reused by every assertion below."""

    WINDOWS = 8
    runs: dict[str, dict[str, object]] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda")
        identity = base.check_index_identity()
        if not identity["passed"]:
            raise AssertionError(f"frozen index identity failed: {identity['errors']}")
        cls.rows = [row for row in base.read_csv_rows(base.INDEX) if str(row["split"]) == base.TRAIN_ROLE]
        cls.positions = base.select_positions(cls.rows, cls.WINDOWS)
        cls.test_paths = set(identity["paths_by_role"]["development_test"]) | set(
            identity["paths_by_role"]["development_test_exposed"]
        )
        cls.train_paths = set(identity["paths_by_role"][base.TRAIN_ROLE])
        cls.default_spec = ln.build_loss_spec(
            ln.LOSS_MODE_BATCH_WEIGHT_SUM, FROZEN_TRAIN_COUNTS, 4
        )
        cls.candidate_spec = ln.build_loss_spec(
            ln.LOSS_MODE_GLOBAL_TRAIN_MEAN, FROZEN_TRAIN_COUNTS, 4
        )
        weights = torch.tensor(FROZEN_CLASS_WEIGHTS, dtype=torch.float32, device=cls.device)
        cls.runs = {
            # the exact default path of the new entry: build_criterion -> CrossEntropyLoss
            "module": cls._run(lambda: ln.build_criterion(cls.default_spec, cls.device), "module"),
            # the re-implementation of the same formula, for the tolerance comparison
            "formula": cls._run(
                lambda: ln.NormalizedWeightedCrossEntropy(
                    ln.LOSS_MODE_BATCH_WEIGHT_SUM, FROZEN_CLASS_WEIGHTS, 1.0
                ).to(cls.device),
                "formula",
            ),
            # the candidate objective
            "candidate": cls._run(
                lambda: ln.build_criterion(
                    cls.candidate_spec, cls.device, record_steps=True
                ),
                "candidate",
            ),
            # a second, independently built original module (reference for bitwise identity)
            "module_reference": cls._run(lambda: nn.CrossEntropyLoss(weight=weights), "module-reference"),
        }

    @classmethod
    def _run(cls, criterion_factory, tag: str) -> dict[str, object]:
        audit = base.ReadAudit(f"test-{tag}")
        base.seed_everything(2027)
        model = base.ResNet18(input_channels=1, num_classes=2).to(cls.device)
        frontend = base.SpectrogramFrontEnd(n_fft=512, hop_length=256, image_size=256).to(cls.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        dataset = base.AuditedWindowDataset(
            base.INDEX,
            split=base.TRAIN_ROLE,
            train_snrs=base.parse_snr_list(base.TRAIN_SNRS),
            seed=2027,
            read_audit=audit,
        )
        dataset.set_epoch(1)
        criterion = criterion_factory()
        loader = base.build_train_loader(
            dataset,
            base.LoaderConfig(num_workers=0),
            positions=cls.positions,
            key="1",
            pin_memory=True,
        )
        model.train()
        frontend.eval()
        losses: list[float] = []
        labels_seen: list[int] = []
        for signals, labels in loader:
            signals = signals.to(cls.device)
            labels = labels.to(cls.device)
            optimizer.zero_grad(set_to_none=True)
            features = base.extract_features(frontend, signals)
            with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                logits = model(features)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            labels_seen.extend(int(value) for value in labels.cpu().tolist())
        state = base.state_hash(model.state_dict())
        steps = [dict(step) for step in getattr(criterion, "steps", [])]
        dataset.close()
        del model, frontend, optimizer, scaler
        torch.cuda.empty_cache()
        return {
            "losses": losses,
            "labels": labels_seen,
            "state_hash": state,
            "windows_read": audit.windows_read,
            "opened_files": sorted(audit.opened_files),
            "steps": steps,
        }

    def test_default_path_is_bitwise_the_original_module_on_real_windows(self) -> None:
        module = self.runs["module"]
        reference = self.runs["module_reference"]
        self.assertEqual(len(module["losses"]), 2)
        self.assertEqual(module["losses"], reference["losses"])
        self.assertEqual(module["state_hash"], reference["state_hash"])

    def test_formula_reimplementation_agrees_with_the_module_on_real_windows(self) -> None:
        module = self.runs["module"]
        formula = self.runs["formula"]
        deltas = [
            abs(float(left) - float(right))
            for left, right in zip(module["losses"], formula["losses"])
        ]
        self.assertLessEqual(max(deltas), 1e-6, deltas)

    def test_candidate_mode_applies_the_global_denominator_on_real_windows(self) -> None:
        candidate = self.runs["candidate"]
        losses = candidate["losses"]
        steps = candidate["steps"]
        self.assertEqual(len(losses), 2)
        self.assertEqual(len(steps), 2)
        self.assertTrue(all(np.isfinite(float(loss)) for loss in losses))
        for loss, step in zip(losses, steps):
            self.assertEqual(step["batch_size"], 4)
            self.assertAlmostEqual(
                float(step["denominator"]), 4 * self.candidate_spec.mu_train, places=6
            )
            self.assertAlmostEqual(
                float(loss) * float(step["denominator"]), float(step["numerator"]), places=4
            )

    def test_candidate_state_differs_from_the_default_state(self) -> None:
        self.assertNotEqual(
            self.runs["module"]["state_hash"], self.runs["candidate"]["state_hash"]
        )
        self.assertNotEqual(self.runs["module"]["losses"], self.runs["candidate"]["losses"])

    def test_every_run_read_exactly_the_bounded_train_windows(self) -> None:
        for tag, run in self.runs.items():
            with self.subTest(run=tag):
                self.assertEqual(run["windows_read"], self.WINDOWS)
                opened = set(run["opened_files"])  # type: ignore[arg-type]
                self.assertFalse(opened & self.test_paths, f"{tag} touched a test file")
                self.assertTrue(opened <= self.train_paths, f"{tag} touched a foreign file")


class _NullChecks:
    """Minimal stand-in for the verifier's check collector."""

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(self, name: str, passed: bool, detail: object = None) -> bool:
        self.rows.append({"check": name, "passed": bool(passed), "detail": detail})
        return bool(passed)


class VerifierHelperTests(unittest.TestCase):
    """The independent recomputation helpers the verifier is built on."""

    def test_protocol_core_hash_ignores_the_volatile_stamp(self) -> None:
        payload = {"schema": "x", "created_at": "2026-09-28T00:00:00+00:00", "run_root": "a"}
        same = {"schema": "x", "created_at": "2026-09-29T00:00:00+00:00", "run_root": "a"}
        different = {"schema": "x", "created_at": "2026-09-28T00:00:00+00:00", "run_root": "b"}
        self.assertEqual(
            entry.protocol_core_sha256(payload), entry.protocol_core_sha256(same)
        )
        self.assertNotEqual(
            entry.protocol_core_sha256(payload), entry.protocol_core_sha256(different)
        )

    def test_recomputed_batch_layout_is_the_frozen_one(self) -> None:
        sizes = verifier.batch_shapes(9590)
        self.assertEqual(len(sizes), 2398)
        self.assertEqual(sizes[-1], 2)
        self.assertEqual(sum(sizes), 9590)

    def test_recomputed_epoch_orders_match_the_control_hashes(self) -> None:
        rows = verifier.frozen_index()
        orders = verifier.epoch_orders(rows, 5)
        hashes = [
            base.sha256_bytes(np.asarray(order, dtype=np.int64).tobytes())
            for order in orders
        ]
        self.assertEqual(tuple(hashes), SEED2027_EPOCH_ORDER_SHA256)

    def test_recomputed_batch_composition_matches_the_recorded_diagnostic(self) -> None:
        """The 77% background-free batch share must be reproducible from the index."""

        rows = verifier.frozen_index()
        spec = ln.build_loss_spec(
            ln.LOSS_MODE_GLOBAL_TRAIN_MEAN, FROZEN_TRAIN_COUNTS, 4
        )
        pack = verifier.check_batch_composition(_NullChecks(), rows, spec)
        shares = [row["share_without_background"] for row in pack["per_epoch"]]
        self.assertEqual(len(shares), 5)
        self.assertTrue(all(0.77 <= share <= 0.78 for share in shares), shares)
        self.assertTrue(
            all(row["batches_all_background"] == 0 for row in pack["per_epoch"])
        )
        tail_sizes = {len(row["tail_batch_labels"]) for row in pack["per_epoch"]}
        self.assertEqual(tail_sizes, {2})

    def test_confusion_recomputation_matches_the_accepted_implementation(self) -> None:
        records = [
            {"label": 0, "prediction": 1, "filename": "b.mat", "logit_margin": -1.0},
            {"label": 0, "prediction": 0, "filename": "b.mat", "logit_margin": 2.0},
            {"label": 1, "prediction": 1, "filename": "d.mat", "logit_margin": 0.5},
            {"label": 1, "prediction": 0, "filename": "d.mat", "logit_margin": -0.5},
        ]
        recomputed = verifier.metrics_from_records(records)
        accepted = base.pooled_binary_metrics(records)
        for key in ("windows", "tn", "fp", "fn", "tp", "accuracy", "balanced_accuracy", "drone_recall", "background_specificity", "f1"):
            self.assertAlmostEqual(float(recomputed[key]), float(accepted[key]), places=12)
        self.assertEqual(recomputed["tn"], 1)
        self.assertEqual(recomputed["fp"], 1)
        self.assertEqual(recomputed["fn"], 1)
        self.assertEqual(recomputed["tp"], 1)

    def test_file_equal_recomputation_matches_the_accepted_implementation(self) -> None:
        records = [
            {"label": 0, "prediction": 0, "filename": "b1.mat"},
            {"label": 0, "prediction": 1, "filename": "b1.mat"},
            {"label": 0, "prediction": 1, "filename": "b2.mat"},
            {"label": 1, "prediction": 1, "filename": "d1.mat"},
            {"label": 1, "prediction": 0, "filename": "d1.mat"},
            {"label": 1, "prediction": 1, "filename": "d2.mat"},
        ]
        recomputed = verifier.file_equal_from_records(records)
        accepted = base.file_equal_metrics(records)
        for key in (
            "files_background",
            "files_drone",
            "file_equal_background_recall",
            "file_equal_drone_recall",
            "file_equal_balanced_accuracy",
        ):
            self.assertAlmostEqual(float(recomputed[key]), float(accepted[key]), places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
