#!/usr/bin/env python3
"""STAGE1-IO-OPT regression tests: loader equivalence, audit identity, cleanup.

These tests are deliberately cheap: they build a tiny synthetic HDF5 fixture
(2 source files x 2 windows of 1,000,000 complex samples, ~16 MB total) so the
loader can be exercised for real -- workers spawned, HDF5 opened, AWGN drawn,
signals compared byte for byte -- without reading the G: drive or competing with a
running training job for disk bandwidth.

What is checked (rework R1):

1. ``num_workers`` in {0, 2, 4} produce byte-identical signals and labels for the
   same epoch order, over two epochs (the epoch feeds the AWGN seed) and both
   label classes.
2. The default configuration still is the reference serial loader, so existing
   invocations keep their semantics.
3. Reads and consumption share one identity, ``(phase, key, underlying split
   position)``: a non-contiguous subset, a repeated position across epochs, a
   position seen under four validation conditions and an abandoned epoch are all
   reported as the distinct events they are, and a completed traversal closes
   exactly (zero unconsumed reads, zero consumed-but-unread windows).
4. Worker events are grouped per worker process, never all called ``main``.
5. A failed consumer leaves no live worker behind and is recorded as
   delivered-but-not-completed.

Run: ``python -m unittest test_stage1_loader_optimization``.
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import h5py
import numpy as np
import torch

import stage1_loader_optimization as opt
from dronerfa_data import DroneRFaWindowDataset, parse_snr_list

FIXTURE_DIR = PROJECT_DIR / "scratch" / "loader_opt_fixture"
WINDOW_SAMPLES = 1_000_000
TRAIN_SNRS = parse_snr_list("clean,10,5,0,-5,-10,-12")
TRAIN_SEED = 2027

#: Two source files: one background (label 0), one drone (label 1).
FIXTURE_FILES = (
    ("T0000_D00_S0000.mat", "T0000", 0),
    ("T0001_D00_S0000.mat", "T0001", 1),
)
#: Four windows per file, so a 2-worker loader really has two batches to share.
WINDOWS_PER_FILE = 4
#: The real frozen index columns; the fixture must satisfy ``dronerfa_data.load_index``.
INDEX_COLUMNS = (
    "row_index",
    "file_path",
    "filename",
    "target_code",
    "target_name",
    "class_id",
    "class_index",
    "class_name",
    "distance_code",
    "segment_code",
    "source_role",
    "split",
    "metrics_scope",
    "role_purpose",
    "prior_use_role",
    "archive_segment_role",
    "exposed_to_prior_selection",
    "stage1_label",
    "label",
    "window_index",
    "window_in_block",
    "block_index",
    "start",
    "length",
    "source_samples",
    "file_sha256",
    "window_rule",
)
WINDOW_RULE = (
    "RF0 I/Q, non-overlapping 1,000,000-sample windows, stride 1,000,000, "
    "tail < 1,000,000 samples dropped"
)


def _signal_for(file_index: int, window_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, structure-bearing fixture window (not noise-only)."""

    rng = np.random.default_rng(1000 * file_index + window_index)
    t = np.arange(WINDOW_SAMPLES, dtype=np.float64)
    tone = np.exp(2j * np.pi * (0.01 + 0.003 * file_index) * t)
    if file_index == 1:  # drone file carries a second, stronger component
        tone = tone + 0.5 * np.exp(2j * np.pi * 0.2 * t)
    signal = tone * (1.0 + 0.1 * np.sin(2 * np.pi * 0.001 * t))
    signal = signal + 0.01 * (
        rng.standard_normal(WINDOW_SAMPLES) + 1j * rng.standard_normal(WINDOW_SAMPLES)
    )
    real = np.asarray(signal.real, dtype=np.float32)
    imag = np.asarray(signal.imag, dtype=np.float32)
    if window_index == 1:
        real = np.roll(real, 4096)
        imag = np.roll(imag, -4096)
    return real, imag


def build_fixture() -> tuple[Path, Path]:
    """Write the tiny HDF5 files and matching frozen-shaped index."""

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    index_path = FIXTURE_DIR / "window_index_fixture.csv"
    rows: list[dict[str, object]] = []
    row_index = 0
    for file_index, (name, code, label) in enumerate(FIXTURE_FILES):
        path = FIXTURE_DIR / name
        if not path.is_file():
            with h5py.File(path, "w") as handle:
                for channel in ("RF0_I", "RF0_Q"):
                    handle.create_dataset(
                        channel, shape=(WINDOWS_PER_FILE * WINDOW_SAMPLES,), dtype="float32"
                    )
                for window_index in range(WINDOWS_PER_FILE):
                    real, imag = _signal_for(file_index, window_index)
                    start = window_index * WINDOW_SAMPLES
                    handle["RF0_I"][start : start + WINDOW_SAMPLES] = real
                    handle["RF0_Q"][start : start + WINDOW_SAMPLES] = imag
        for window_index in range(WINDOWS_PER_FILE):
            rows.append(
                {
                    "row_index": row_index,
                    "file_path": str(path),
                    "filename": name,
                    "target_code": code,
                    "target_name": "Background" if label == 0 else "Drone fixture",
                    "class_id": code,
                    "class_index": str(label),
                    "class_name": "Background" if label == 0 else "Drone fixture",
                    "distance_code": "00",
                    "segment_code": "0000",
                    "source_role": "train",
                    "split": "train",
                    "metrics_scope": "development",
                    "role_purpose": "model fitting",
                    "prior_use_role": "train",
                    "archive_segment_role": "train",
                    "exposed_to_prior_selection": "False",
                    "stage1_label": label,
                    "label": label,
                    "window_index": window_index,
                    "window_in_block": window_index % 10,
                    "block_index": window_index // 10,
                    "start": window_index * WINDOW_SAMPLES,
                    "length": WINDOW_SAMPLES,
                    "source_samples": WINDOWS_PER_FILE * WINDOW_SAMPLES,
                    "file_sha256": "0" * 64,
                    "window_rule": WINDOW_RULE,
                }
            )
            row_index += 1
    if not index_path.is_file():
        with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
    return index_path, FIXTURE_DIR


def train_dataset(index_path: Path, seed: int = TRAIN_SEED) -> DroneRFaWindowDataset:
    return DroneRFaWindowDataset(
        index_path,
        split="train",
        train_snrs=TRAIN_SNRS,
        seed=seed,
    )


def run_epoch_digests(
    index_path: Path,
    config: opt.LoaderConfig,
    epoch: int,
    *,
    seed: int = TRAIN_SEED,
    events: "multiprocessing.Queue | None" = None,
    audit: opt.LoaderAudit | None = None,
    positions: "list[int] | None" = None,
    phase: str = opt.TRAIN_PHASE,
    key: "str | None" = None,
) -> tuple[list[dict], list[int], opt.ConsumptionLog]:
    """One full pass through the configured loader; returns per-batch digests."""

    dataset = train_dataset(index_path, seed)
    dataset.set_epoch(epoch)
    if positions is None:
        order = opt.order_for_epoch(dataset, epoch, seed)
    else:
        order = opt.limited_order(opt.order_for_epoch(dataset, epoch, seed), positions)
    consumption = opt.ConsumptionLog(
        phase=phase,
        key=str(key if key is not None else epoch),
        on_batch=(audit.drain if audit is not None else None),
    )
    if audit is not None:
        audit.register_loader(phase, consumption.key, config)
    loader = opt.build_train_loader(
        dataset, config, positions=order, phase=phase, key=consumption.key,
        audit_events=events,
    )
    digests = opt.loader_batch_digests(
        opt.consumed_batches(loader, order, consumption), order
    )
    if audit is not None:
        audit.record_consumption(consumption)
    if config.parallel:
        del loader
    if audit is not None:
        audit.close_loader(phase, consumption.key)
    dataset.close()
    return digests, order, consumption


class ReferenceDefaultsTests(unittest.TestCase):
    def test_default_config_is_the_reference_serial_loader(self) -> None:
        config = opt.LoaderConfig()
        self.assertEqual(config.num_workers, opt.REFERENCE_NUM_WORKERS)
        self.assertEqual(config.num_workers, 0)
        self.assertIsNone(config.prefetch_factor)
        self.assertIsNone(config.effective_prefetch_factor)
        self.assertFalse(config.parallel)

    def test_prefetch_factor_is_only_applied_to_parallel_loaders(self) -> None:
        serial = opt.LoaderConfig(num_workers=0, prefetch_factor=4)
        self.assertIsNone(serial.effective_prefetch_factor)
        parallel = opt.LoaderConfig(num_workers=2)
        self.assertEqual(parallel.effective_prefetch_factor, 2)
        self.assertTrue(parallel.parallel)

    def test_invalid_worker_and_prefetch_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            opt.LoaderConfig(num_workers=-1)
        with self.assertRaises(ValueError):
            opt.LoaderConfig(num_workers=2, prefetch_factor=0)

    def test_candidate_worker_counts_are_the_ones_under_test(self) -> None:
        self.assertEqual(opt.CANDIDATE_NUM_WORKERS, (0, 2, 4))


class EpochOrderTests(unittest.TestCase):
    def test_order_matches_the_frozen_driver_shuffle(self) -> None:
        index_path, _ = build_fixture()
        dataset = train_dataset(index_path)
        for epoch in (1, 2, 3):
            expected = torch.randperm(
                len(dataset),
                generator=torch.Generator().manual_seed(TRAIN_SEED + epoch),
            ).tolist()
            self.assertEqual(opt.order_for_epoch(dataset, epoch, TRAIN_SEED), expected)
            self.assertEqual(len(set(expected)), len(expected))
        dataset.close()

    def test_limited_order_keeps_the_frozen_permutation_of_a_subset(self) -> None:
        """A bounded view must not re-shuffle: it filters the full permutation."""

        full = [3, 0, 2, 1]
        self.assertEqual(opt.limited_order(full, [3, 1]), [3, 1])
        self.assertEqual(opt.limited_order(full, [0, 1, 2, 3]), full)
        self.assertEqual(opt.limited_order(full, []), [])


class LoaderEquivalenceTests(unittest.TestCase):
    """The whole point: parallel workers must not change a single input byte."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.index_path, cls.fixture_dir = build_fixture()

    def test_workers_do_not_change_signals_labels_or_order(self) -> None:
        for epoch in (1, 2):
            with self.subTest(epoch=epoch):
                reference, order, _ = run_epoch_digests(
                    self.index_path, opt.LoaderConfig(), epoch
                )
                self.assertTrue(reference, "the reference epoch produced no batches")
                self.assertEqual(
                    len(order), len(FIXTURE_FILES) * WINDOWS_PER_FILE
                )
                labels = sorted({label for batch in reference for label in batch["labels"]})
                self.assertEqual(labels, [0, 1], "the fixture must exercise both classes")
                for workers in (2, 4):
                    candidate, candidate_order, _ = run_epoch_digests(
                        self.index_path, opt.LoaderConfig(num_workers=workers), epoch
                    )
                    self.assertEqual(candidate_order, order)
                    report = opt.compare_digests(reference, candidate)
                    self.assertTrue(
                        report["passed"],
                        f"workers={workers} epoch={epoch}: {report['problems']}",
                    )
                    self.assertEqual(report["batches"], len(reference))

    def test_a_non_contiguous_subset_delivers_exactly_its_positions(self) -> None:
        """R1: the loader is indexed by underlying positions, not by view slots."""

        dataset = train_dataset(self.index_path)
        permutation = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(TRAIN_SEED + 1)
        ).tolist()
        expected = [position for position in permutation if position in (1, 3)]
        dataset.close()
        self.assertEqual(sorted(expected), [1, 3])
        for workers in (0, 2):
            with self.subTest(workers=workers):
                _, order, consumption = run_epoch_digests(
                    self.index_path,
                    opt.LoaderConfig(num_workers=workers),
                    epoch=1,
                    positions=[3, 1],
                )
                self.assertEqual(order, expected, "the frozen order must survive filtering")
                self.assertEqual(consumption.delivered, expected)
                self.assertEqual(consumption.completed, expected)

    def test_every_window_appears_exactly_once_per_epoch(self) -> None:
        _, _, consumption = run_epoch_digests(self.index_path, opt.LoaderConfig(), 1)
        self.assertEqual(len(consumption.delivered), len(set(consumption.delivered)))
        self.assertEqual(
            len(consumption.delivered), len(FIXTURE_FILES) * WINDOWS_PER_FILE
        )
        self.assertEqual(consumption.completed, consumption.delivered)
        self.assertEqual(consumption.batches_completed, consumption.batches_delivered)

    def test_epoch_changes_the_drawn_noise(self) -> None:
        first, _, _ = run_epoch_digests(self.index_path, opt.LoaderConfig(), 1)
        second, _, _ = run_epoch_digests(self.index_path, opt.LoaderConfig(), 2)
        self.assertNotEqual(
            [batch["signals_sha256"] for batch in first],
            [batch["signals_sha256"] for batch in second],
            "the AWGN seed depends on the epoch, so two epochs must differ",
        )

    def test_seed_changes_the_drawn_noise(self) -> None:
        first, _, _ = run_epoch_digests(self.index_path, opt.LoaderConfig(), 1, seed=2027)
        second, _, _ = run_epoch_digests(self.index_path, opt.LoaderConfig(), 1, seed=99)
        self.assertNotEqual(
            [batch["signals_sha256"] for batch in first],
            [batch["signals_sha256"] for batch in second],
        )

    def test_epoch_two_must_not_compare_equal_to_epoch_one(self) -> None:
        """A negative control for the comparison itself: only equal draws pass."""

        reference, _, _ = run_epoch_digests(self.index_path, opt.LoaderConfig(), 1)
        same_epoch, _, _ = run_epoch_digests(self.index_path, opt.LoaderConfig(), 1)
        other_epoch, _, _ = run_epoch_digests(self.index_path, opt.LoaderConfig(), 2)
        self.assertTrue(
            opt.compare_digests(reference, same_epoch)["passed"],
            "repeating one epoch must reproduce the same tensors",
        )
        report = opt.compare_digests(reference, other_epoch)
        self.assertFalse(report["passed"], "epoch 1 vs epoch 2 must not look identical")
        self.assertGreater(report["problem_count"], 0)


class IdentityAuditTests(unittest.TestCase):
    """R1: reads and consumption share ``(phase, key, underlying position)``."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.index_path, cls.fixture_dir = build_fixture()

    def _audit(self, workers: int = 2):
        context = multiprocessing.get_context("spawn")
        events = context.Queue()
        audit = opt.LoaderAudit("test")
        audit.set_events(events)
        return audit, events

    def _frozen_subset_order(self, positions: "list[int]", epoch: int = 1) -> list[int]:
        """The frozen shuffle filtered to ``positions``, recomputed here."""

        dataset = train_dataset(self.index_path)
        permutation = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(TRAIN_SEED + epoch)
        ).tolist()
        dataset.close()
        allowed = set(positions)
        return [position for position in permutation if position in allowed]

    def test_a_full_traversal_closes_exactly(self) -> None:
        audit, events = self._audit()
        _, order, _ = run_epoch_digests(
            self.index_path, opt.LoaderConfig(num_workers=2), 1,
            events=events, audit=audit,
        )
        report = audit.key_report(opt.TRAIN_PHASE, "1")
        self.assertEqual(report["reads"]["windows_read"], len(order))
        self.assertEqual(report["reads"]["unique_positions"], len(order))
        self.assertEqual(report["reads"]["duplicate_reads"], 0)
        self.assertEqual(report["consumed"]["windows_completed"], len(order))
        self.assertEqual(report["unconsumed_reads"]["count"], 0)
        self.assertEqual(report["unread_consumed"]["count"], 0)
        self.assertFalse(report["consumed"]["aborted"])
        self.assertTrue(report["consumed"]["exhausted"])
        # The read *order* is only defined for a single reader; the multiset is the
        # portable claim, and the parent-side consumption order is exact.
        self.assertEqual(
            report["reads"]["position_multiset_sha256"],
            opt.multiset_digest(order),
        )
        self.assertEqual(
            report["consumed"]["completed_order_sha256"],
            opt.order_digest(order),
        )

    def test_a_non_contiguous_subset_is_recorded_as_consumed(self) -> None:
        """The first implementation reported every filtered window as unconsumed."""

        expected = self._frozen_subset_order([3, 1])
        audit, events = self._audit()
        _, order, _ = run_epoch_digests(
            self.index_path, opt.LoaderConfig(num_workers=2), 1,
            events=events, audit=audit, positions=[3, 1],
        )
        self.assertEqual(order, expected)
        report = audit.key_report(opt.TRAIN_PHASE, "1")
        self.assertEqual(report["reads"]["windows_read"], 2)
        self.assertEqual(report["consumed"]["windows_completed"], 2)
        self.assertEqual(report["unconsumed_reads"]["count"], 0)
        self.assertEqual(report["unread_consumed"]["count"], 0)

    def test_the_same_position_in_two_epochs_is_two_identities(self) -> None:
        """A set keyed by position alone would have collapsed these."""

        audit, events = self._audit()
        for epoch in (1, 2):
            run_epoch_digests(
                self.index_path, opt.LoaderConfig(num_workers=2), epoch,
                events=events, audit=audit,
            )
        phase = audit.phase_report(opt.TRAIN_PHASE)
        self.assertEqual(sorted(phase["keys"]), ["1", "2"])
        for key in ("1", "2"):
            self.assertEqual(
                phase["keys"][key]["reads"]["windows_read"],
                len(FIXTURE_FILES) * WINDOWS_PER_FILE,
            )
        self.assertEqual(phase["windows_read"], 2 * len(FIXTURE_FILES) * WINDOWS_PER_FILE)
        self.assertEqual(phase["keys"]["1"]["reads"]["order_sha256"] != "", True)

    def test_four_validation_conditions_stay_separate(self) -> None:
        audit, events = self._audit()
        conditions = ("clean", "-10dB", "-12dB", "-15dB")
        for condition in conditions:
            run_epoch_digests(
                self.index_path, opt.LoaderConfig(num_workers=2), 0,
                events=events, audit=audit, phase=opt.VAL_PHASE, key=condition,
            )
        phase = audit.phase_report(opt.VAL_PHASE)
        self.assertEqual(sorted(phase["keys"]), sorted(conditions))
        self.assertEqual(
            phase["windows_read"], len(conditions) * len(FIXTURE_FILES) * WINDOWS_PER_FILE
        )
        for report in phase["keys"].values():
            self.assertEqual(report["unconsumed_reads"]["count"], 0)
            self.assertEqual(report["unread_consumed"]["count"], 0)

    def test_worker_events_are_grouped_by_process_not_called_main(self) -> None:
        audit, events = self._audit()
        _, order, _ = run_epoch_digests(
            self.index_path, opt.LoaderConfig(num_workers=2), 1,
            events=events, audit=audit,
        )
        phase = audit.phase_report(opt.TRAIN_PHASE)
        # Eight windows in batches of four: both workers must have produced events.
        self.assertEqual(len(order), 8)
        self.assertGreaterEqual(len(phase["worker_keys"]), 2)
        for worker in phase["worker_keys"]:
            self.assertRegex(worker, r"^worker\d+@pid\d+$")
        self.assertEqual(phase["worker_count"], len(phase["worker_keys"]))
        self.assertNotIn(0, phase["worker_pids"])
        # Every key is a distinct process, and none of them is the parent.
        self.assertEqual(
            len({worker.split("@pid")[1] for worker in phase["worker_keys"]}),
            len(phase["worker_keys"]),
        )

    def test_serial_loader_events_are_attributed_to_the_parent(self) -> None:
        context = multiprocessing.get_context("spawn")
        del context
        import queue as queue_module

        events = queue_module.Queue()
        audit = opt.LoaderAudit("serial")
        audit.set_events(events)
        run_epoch_digests(
            self.index_path, opt.LoaderConfig(), 1, events=events, audit=audit
        )
        phase = audit.phase_report(opt.TRAIN_PHASE)
        self.assertTrue(all(worker.startswith("main@") for worker in phase["worker_keys"]))

    def test_an_abandoned_epoch_shows_delivered_without_completion(self) -> None:
        audit, events = self._audit()
        dataset = train_dataset(self.index_path)
        dataset.set_epoch(1)
        order = opt.order_for_epoch(dataset, 1, TRAIN_SEED)
        consumption = opt.ConsumptionLog(
            phase=opt.TRAIN_PHASE, key="1", on_batch=audit.drain
        )
        loader = opt.build_train_loader(
            dataset, opt.LoaderConfig(num_workers=2), positions=order,
            phase=opt.TRAIN_PHASE, key="1", audit_events=events,
        )
        batches = opt.consumed_batches(loader, order, consumption)
        for batch in batches:
            del batch
            break  # abandon the epoch after the first delivered batch
        batches.close()  # the consumer stopped early: a real abort
        del loader
        audit.drain()
        dataset.close()
        audit.record_consumption(consumption)
        report = audit.key_report(opt.TRAIN_PHASE, "1")
        self.assertTrue(consumption.aborted)
        self.assertFalse(consumption.exhausted)
        self.assertEqual(report["consumed"]["windows_delivered"], 4)
        self.assertEqual(report["consumed"]["windows_completed"], 0)
        self.assertEqual(report["consumed"]["batches_completed"], 0)
        # Windows the abandoned run never delivered must not be claimed as read-and
        # -consumed; only real prefetch reads may appear.
        self.assertLessEqual(report["reads"]["windows_read"], len(order))
        self.assertGreater(report["unconsumed_reads"]["count"], 0)

    def test_a_failed_consumer_is_recorded_and_does_not_leak_workers(self) -> None:
        audit, events = self._audit()
        dataset = train_dataset(self.index_path)
        dataset.set_epoch(1)
        order = opt.order_for_epoch(dataset, 1, TRAIN_SEED)
        consumption = opt.ConsumptionLog(
            phase=opt.TRAIN_PHASE, key="1", on_batch=audit.drain
        )
        loader = opt.build_train_loader(
            dataset, opt.LoaderConfig(num_workers=2), positions=order,
            phase=opt.TRAIN_PHASE, key="1", audit_events=events,
        )

        class Boom(RuntimeError):
            pass

        consumed = 0
        with self.assertRaises(Boom):
            for batch in opt.consumed_batches(loader, order, consumption):
                del batch
                consumed += 1
                if consumed == 1:
                    raise Boom("consumer failed mid-epoch")
        worker_pids = [worker.pid for worker in getattr(loader, "_workers", []) or []]
        for worker in getattr(loader, "_workers", []) or []:
            self.assertFalse(worker.is_alive(), "a worker survived the failed epoch")
        del loader  # shutdown_worker called here
        audit.drain()
        dataset.close()
        audit.record_consumption(consumption)
        self.assertTrue(consumption.aborted, "the abort must be recorded")
        self.assertEqual(len(consumption.delivered), 4)
        self.assertLess(len(consumption.completed), 4)
        report = audit.key_report(opt.TRAIN_PHASE, "1")
        self.assertGreaterEqual(report["reads"]["windows_read"], 4)
        self.assertEqual(report["consumed"]["windows_delivered"], 4)
        for pid in worker_pids:
            self.assertGreater(pid, 0)

    def test_consumption_log_round_trips_through_csv(self) -> None:
        expected = self._frozen_subset_order([3, 1])
        audit, events = self._audit()
        run_epoch_digests(
            self.index_path, opt.LoaderConfig(num_workers=2), 1,
            events=events, audit=audit, positions=[3, 1],
        )
        target = PROJECT_DIR / "scratch" / "identity_audit_tests" / "consumption.csv"
        info = audit.write_consumption_csv(target)
        self.assertEqual(info["rows"], 2)
        with target.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["position"]) for row in rows], expected)
        self.assertTrue(all(int(row["completed"]) == 1 for row in rows))

    def test_phase_payload_keeps_the_legacy_aggregate_fields(self) -> None:
        audit, events = self._audit()
        run_epoch_digests(
            self.index_path, opt.LoaderConfig(num_workers=2), 1,
            events=events, audit=audit,
        )
        payload = opt.audit_from_parallel(audit, opt.TRAIN_PHASE)
        expected_files = {str(self.fixture_dir / name) for name, _, _ in FIXTURE_FILES}
        self.assertEqual(set(payload["opened_files"]), expected_files)
        self.assertEqual(payload["windows_read"], len(FIXTURE_FILES) * WINDOWS_PER_FILE)
        self.assertEqual(
            sum(payload["windows_by_file"].values()), payload["windows_read"]
        )
        self.assertEqual(payload["prefetch"]["windows_read_but_not_consumed"], 0)
        self.assertEqual(payload["drain_failures"], 0)

    def test_serial_audit_payload_keeps_the_legacy_shape(self) -> None:
        class LegacyAudit:
            def payload(self) -> dict:
                return {
                    "stage": "stage1",
                    "open_events": 2,
                    "windows_read": 4,
                    "opened_files": ["a.mat", "b.mat"],
                    "opened_file_count": 2,
                    "windows_by_file": {"a.mat": 2, "b.mat": 2},
                }

        payload = opt.serial_audit_payload(LegacyAudit(), "train")
        for field in ("open_events", "windows_read", "opened_files", "windows_by_file"):
            self.assertIn(field, payload)
        self.assertEqual(payload["split"], "train")
        self.assertEqual(payload["loader"], "reference_num_workers_0")

    def test_audit_comparison_detects_a_tampered_window_count(self) -> None:
        audit, events = self._audit()
        run_epoch_digests(
            self.index_path, opt.LoaderConfig(num_workers=2), 1,
            events=events, audit=audit,
        )
        aggregate = opt.audit_from_parallel(audit, opt.TRAIN_PHASE)
        reference = dict(aggregate)
        self.assertTrue(opt.compare_audits(reference, aggregate)["passed"])
        tampered = json.loads(json.dumps(aggregate))
        tampered["windows_read"] = int(aggregate["windows_read"]) - 1
        report = opt.compare_audits(reference, tampered)
        self.assertFalse(report["passed"])
        self.assertIn("windows_read", report["differences"])


class IdentityHelperTests(unittest.TestCase):
    def test_order_and_multiset_digests_separate_repeats_from_order(self) -> None:
        self.assertNotEqual(opt.order_digest([1, 2]), opt.order_digest([2, 1]))
        self.assertEqual(opt.multiset_digest([1, 2]), opt.multiset_digest([2, 1]))
        self.assertNotEqual(opt.multiset_digest([1, 1]), opt.multiset_digest([1, 2]))
        self.assertEqual(opt.duplicate_count([1, 2, 1]), 1)
        self.assertEqual(opt.duplicate_count([1, 2]), 0)

    def test_counter_delta_reports_missing_and_extra_positions(self) -> None:
        self.assertEqual(opt.counter_delta([1, 2, 3], [1, 2], limit=5)["count"], 1)
        self.assertEqual(opt.counter_delta([1, 1], [1], limit=5)["count"], 1)
        self.assertEqual(opt.counter_delta([1], [1, 1], limit=5)["count"], 0)

    def test_identity_digest_is_path_and_position_sensitive(self) -> None:
        first = opt.identity_digest([("a.mat", 1), ("b.mat", 2)])
        self.assertEqual(first, opt.identity_digest([("a.mat", 1), ("b.mat", 2)]))
        self.assertNotEqual(first, opt.identity_digest([("a.mat", 2), ("b.mat", 1)]))
        self.assertNotEqual(first, opt.identity_digest([("a.mat", 1), ("c.mat", 2)]))


class TensorDigestTests(unittest.TestCase):
    def test_digest_is_byte_sensitive_and_shape_sensitive(self) -> None:
        first = torch.zeros(2, 3, dtype=torch.float32)
        second = torch.zeros(2, 3, dtype=torch.float32)
        self.assertEqual(opt.tensor_digest(first), opt.tensor_digest(second))
        second[0, 0] = 1e-7
        self.assertNotEqual(opt.tensor_digest(first), opt.tensor_digest(second))
        third = torch.zeros(3, 2, dtype=torch.float32)
        self.assertNotEqual(opt.tensor_digest(first), opt.tensor_digest(third))

    def test_compare_digests_reports_the_first_difference(self) -> None:
        reference = [{"batch": 0, "size": 4, "positions": [0, 1, 2, 3], "signals_sha256": "a",
                      "labels_sha256": "b", "labels": [0, 0, 1, 1]}]
        candidate = [dict(reference[0], signals_sha256="c")]
        report = opt.compare_digests(reference, candidate)
        self.assertFalse(report["passed"])
        self.assertEqual(report["problems"][0]["issue"], "signals_sha256")


if __name__ == "__main__":
    unittest.main(verbosity=2)
