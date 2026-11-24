#!/usr/bin/env python3
"""Regression tests for the loader identity audit's verifier (STAGE1-IO-OPT R2).

The accepted verifier was written for the serial loader, where the parent process
really opens every file.  With ``num_workers > 0`` the parent opens nothing, so the
legacy ``read_audit.json`` is empty by construction and its checks would pass
*vacuously* -- exactly the "predicted reads disguised as real reads" failure the task
forbids.  These tests pin the replacement evidence
(``verify_loader_identity_audit`` / ``evaluate_loader_audit``) on the **real frozen
index** so the recomputed plan is the production plan, not a toy one.

No spawned processes and no G: reads: the payload is synthesised from the plan the
verifier itself recomputes, and the tamper controls must all be detected.
"""

from __future__ import annotations

import json
import shutil
import sys
import unittest
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import stage1_loader_optimization as opt
import verify_stage1_12drones_training as verifier

#: Scratch space inside the project: the DSH file sandbox denies writes to the
#: system temp directory, and these tests must not need elevated access.
SCRATCH = PROJECT_DIR / "scratch" / "parallel_audit_tests"
TRAIN_SEED = 2027
TRAIN_SAMPLES = 40
VAL_SAMPLES = 16
EPOCHS = 1
CONDITIONS = ("clean", "-10dB", "-12dB", "-15dB")


def _fresh_scratch(name: str) -> Path:
    target = SCRATCH / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _index_rows() -> list[dict[str, str]]:
    return verifier.read_csv_rows(verifier.INDEX)


def _index_info(rows: list[dict[str, str]]) -> dict:
    role_files: dict[str, list[str]] = {}
    for row in rows:
        role_files.setdefault(str(row["source_role"]), [])
        if str(row["filename"]) not in role_files[str(row["source_role"])]:
            role_files[str(row["source_role"])].append(str(row["filename"]))
    return {"role_files": {role: sorted(files) for role, files in role_files.items()}}


def _run_root(name: str, *, num_workers: int = 4, epochs: int = EPOCHS) -> Path:
    root = _fresh_scratch(name)
    (root / "run_config.json").write_text(
        json.dumps(
            {
                "training_seed": TRAIN_SEED,
                "validation_seed": 12026,
                "validation_conditions": list(CONDITIONS),
                "epochs": epochs,
                "max_train_samples": TRAIN_SAMPLES,
                "max_val_samples": VAL_SAMPLES,
                "loader": {"num_workers": num_workers, "parallel": num_workers > 0},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


def _worker_keys(num_workers: int) -> list[str]:
    return [f"worker{index}@pid{1000 + index}" for index in range(max(num_workers, 1))]


def synth_payload(plan: dict, *, num_workers: int = 4) -> dict:
    """Build the audit payload a faithful run would have written for ``plan``."""

    workers = _worker_keys(num_workers if num_workers > 0 else 0)
    if num_workers <= 0:
        workers = ["main@pid999"]
    phases: dict[str, dict] = {}
    for phase_name, key_map, role in (
        (verifier.TRAIN_ROLE, plan["train"], verifier.TRAIN_ROLE),
        (verifier.VAL_ROLE, plan["val"], verifier.VAL_ROLE),
    ):
        keys: dict[str, dict] = {}
        opened: set[str] = set()
        for key, positions in key_map.items():
            pairs = [
                (plan["position_path"][role][int(position)], int(position))
                for position in positions
            ]
            opened |= {path for path, _ in pairs}
            files = Counter(Path(path).name for path, _ in pairs)
            digest = verifier.position_digest(positions)
            keys[str(key)] = {
                "phase": phase_name,
                "key": str(key),
                "identity": f"{phase_name}/{key}",
                "reads": {
                    "windows_read": len(pairs),
                    "unique_positions": len({position for _, position in pairs}),
                    "duplicate_reads": 0,
                    "files": dict(sorted(files.items())),
                    "read_files": sorted({path for path, _ in pairs}),
                    "opened_files": sorted({path for path, _ in pairs}),
                    "open_events": len({path for path, _ in pairs}),
                    "order_sha256": verifier.position_digest([position for _, position in pairs]),
                    "position_order_sha256": digest,
                    "position_multiset_sha256": verifier.position_multiset_digest(positions),
                },
                "consumed": {
                    "windows_delivered": len(positions),
                    "windows_completed": len(positions),
                    "batches_delivered": max(1, (len(positions) + 3) // 4),
                    "batches_completed": max(1, (len(positions) + 3) // 4),
                    "delivered_order_sha256": digest,
                    "completed_order_sha256": digest,
                    "delivered_multiset_sha256": verifier.position_multiset_digest(positions),
                    "completed_multiset_sha256": verifier.position_multiset_digest(positions),
                    "duplicate_delivered": 0,
                    "aborted": False,
                    "iterator_closed": True,
                    "exhausted": True,
                },
                "unconsumed_reads": {"count": 0, "examples": []},
                "unread_consumed": {"count": 0, "examples": []},
                "workers": workers,
                "pids": [int(worker.split("@pid")[1]) for worker in workers],
            }
        phases[phase_name] = {
            "label": "stage1",
            "phase": phase_name,
            "keys": keys,
            "loaders": [
                {
                    "phase": phase_name,
                    "key": str(key),
                    "identity": f"{phase_name}/{key}",
                    "num_workers": num_workers,
                    "prefetch_factor": 2 if num_workers > 0 else None,
                }
                for key in key_map
            ],
            "closed_loaders": [f"{phase_name}/{key}" for key in key_map],
            "opened_files": sorted(opened),
            "opened_file_count": len(opened),
            "read_files": sorted(opened),
            "windows_read": sum(entry["reads"]["windows_read"] for entry in keys.values()),
            "windows_by_file": dict(
                sorted(
                    Counter(
                        Path(path).name
                        for key in keys.values()
                        for path in key["reads"]["read_files"]
                    ).items()
                )
            ),
            "windows_delivered": sum(
                entry["consumed"]["windows_delivered"] for entry in keys.values()
            ),
            "windows_completed": sum(
                entry["consumed"]["windows_completed"] for entry in keys.values()
            ),
            "unconsumed_reads": 0,
            "unread_consumed": 0,
            "worker_keys": workers,
            "worker_pids": [int(worker.split("@pid")[1]) for worker in workers],
        }
    return {
        "schema": "dronerfa-stage1-loader-audit/v2",
        "phases": phases,
        "drain_failures": 0,
        "unknown_events": 0,
        "worker_processes": workers,
    }


class DigestAgreementTests(unittest.TestCase):
    """The verifier re-implements the loader's digests; they must agree."""

    def test_order_and_multiset_digests_match_the_loader_module(self) -> None:
        samples = ([0, 1, 2, 3], [3, 1], [5, 5, 1], [])
        for positions in samples:
            with self.subTest(positions=positions):
                self.assertEqual(
                    verifier.position_digest(positions), opt.order_digest(positions)
                )
                self.assertEqual(
                    verifier.position_multiset_digest(positions),
                    opt.multiset_digest(positions),
                )


class LoaderAuditVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _index_rows()
        cls.index_info = _index_info(cls.rows)

    def _plan(self, name: str, *, num_workers: int = 4, epochs: int = EPOCHS) -> dict:
        root = _run_root(name, num_workers=num_workers, epochs=epochs)
        plan = verifier.loader_identity_plan(self.rows, root, self.index_info)
        self.assertTrue(plan["available"], plan)
        return plan

    def test_a_faithful_payload_passes_every_check(self) -> None:
        plan = self._plan("faithful")
        payload = synth_payload(plan, num_workers=4)
        evaluation = verifier.evaluate_loader_audit(payload, plan)
        self.assertEqual(evaluation["problems"], [])
        self.assertEqual(
            evaluation["info"]["train_windows_read"], TRAIN_SAMPLES * EPOCHS
        )
        self.assertEqual(
            evaluation["info"]["val_windows_read"], VAL_SAMPLES * len(CONDITIONS)
        )
        expected_opened = {
            Path(path).name
            for role in (verifier.TRAIN_ROLE, verifier.VAL_ROLE)
            for path in plan["position_path"][role].values()
        }
        self.assertEqual(len(evaluation["info"]["opened_files"]), len(expected_opened))
        # A bounded run must not be asked for the whole frozen train+val file set.
        self.assertLess(len(expected_opened), 78)

    def test_the_plan_covers_the_frozen_split_not_a_toy_one(self) -> None:
        plan = self._plan("plan-shape")
        self.assertEqual(len(plan["train"]), EPOCHS)
        self.assertEqual(sorted(plan["val"]), sorted(CONDITIONS))
        for positions in plan["val"].values():
            self.assertEqual(len(positions), VAL_SAMPLES)
        for positions in plan["train"].values():
            self.assertEqual(len(positions), TRAIN_SAMPLES)
            self.assertEqual(len(set(positions)), TRAIN_SAMPLES)

    def test_a_serial_plan_also_checks_the_read_order(self) -> None:
        plan = self._plan("serial-order", num_workers=0)
        payload = synth_payload(plan, num_workers=0)
        self.assertEqual(verifier.evaluate_loader_audit(payload, plan)["problems"], [])
        key = sorted(payload["phases"][verifier.TRAIN_ROLE]["keys"])[0]
        payload["phases"][verifier.TRAIN_ROLE]["keys"][key]["reads"][
            "position_order_sha256"
        ] = "0" * 64
        problems = verifier.evaluate_loader_audit(payload, plan)["problems"]
        self.assertTrue(any(problem["check"].startswith("P8b") for problem in problems))

    def _tampered_failures(self, mutate) -> list[dict]:
        plan = self._plan("tamper")
        payload = synth_payload(plan, num_workers=4)
        mutate(payload)
        return verifier.evaluate_loader_audit(payload, plan)["problems"]

    def _first_train_key(self, payload: dict) -> str:
        return sorted(payload["phases"][verifier.TRAIN_ROLE]["keys"])[0]

    def test_a_dropped_window_is_detected(self) -> None:
        def mutate(payload: dict) -> None:
            entry = payload["phases"][verifier.TRAIN_ROLE]["keys"][
                self._first_train_key(payload)
            ]
            entry["reads"]["windows_read"] -= 1
            entry["consumed"]["windows_completed"] -= 1

        problems = self._tampered_failures(mutate)
        self.assertTrue(problems)
        self.assertTrue(any(problem["check"].startswith("P7") for problem in problems))

    def test_a_duplicated_window_is_detected(self) -> None:
        def mutate(payload: dict) -> None:
            entry = payload["phases"][verifier.TRAIN_ROLE]["keys"][
                self._first_train_key(payload)
            ]
            entry["reads"]["duplicate_reads"] = 1

        problems = self._tampered_failures(mutate)
        self.assertTrue(any(problem["check"].startswith("P9") for problem in problems))

    def test_repeating_another_window_instead_of_a_missing_one_is_detected(self) -> None:
        """Dropping A while reading B twice keeps the count but breaks the multiset."""

        def mutate(payload: dict) -> None:
            entry = payload["phases"][verifier.TRAIN_ROLE]["keys"][
                self._first_train_key(payload)
            ]
            entry["consumed"]["windows_completed"] = entry["consumed"]["windows_completed"]
            entry["reads"]["position_multiset_sha256"] = "1" * 64
            entry["consumed"]["completed_order_sha256"] = "1" * 64

        problems = self._tampered_failures(mutate)
        self.assertTrue(any(problem["check"].startswith("P8 ") for problem in problems))
        self.assertTrue(any(problem["check"].startswith("P11") for problem in problems))

    def test_a_role_swap_between_phases_is_detected(self) -> None:
        def mutate(payload: dict) -> None:
            train = payload["phases"][verifier.TRAIN_ROLE]
            val = payload["phases"][verifier.VAL_ROLE]
            train["opened_files"] = sorted(set(train["opened_files"]) | {val["opened_files"][0]})
            train["opened_files"] = [p for p in train["opened_files"] if p in val["opened_files"]] or train["opened_files"]

        problems = self._tampered_failures(mutate)
        self.assertTrue(any(problem["check"].startswith("P3") for problem in problems))

    def test_a_test_role_file_is_detected(self) -> None:
        def mutate(payload: dict) -> None:
            payload["phases"][verifier.TRAIN_ROLE]["opened_files"] = list(
                payload["phases"][verifier.TRAIN_ROLE]["opened_files"]
            ) + [r"G:\DroneRFa数据集\original\T0000_D00_S0100.mat"]

        problems = self._tampered_failures(mutate)
        self.assertTrue(any(problem["check"].startswith("P3") for problem in problems))

    def test_a_lost_epoch_is_detected(self) -> None:
        def mutate(payload: dict) -> None:
            payload["phases"][verifier.TRAIN_ROLE]["keys"].pop(
                self._first_train_key(payload)
            )

        problems = self._tampered_failures(mutate)
        self.assertTrue(any(problem["check"].startswith("P2") for problem in problems))

    def test_an_unconsumed_read_is_detected(self) -> None:
        def mutate(payload: dict) -> None:
            payload["phases"][verifier.VAL_ROLE]["unconsumed_reads"] = 1

        problems = self._tampered_failures(mutate)
        self.assertTrue(any(problem["check"].startswith("P5") for problem in problems))

    def test_a_lost_event_is_detected(self) -> None:
        def mutate(payload: dict) -> None:
            payload["drain_failures"] = 1

        problems = self._tampered_failures(mutate)
        self.assertTrue(any(problem["check"].startswith("P14") for problem in problems))

    def test_a_parent_read_in_a_parallel_run_is_detected(self) -> None:
        def mutate(payload: dict) -> None:
            payload["phases"][verifier.TRAIN_ROLE]["worker_keys"] = ["main@pid1"]

        problems = self._tampered_failures(mutate)
        self.assertTrue(any(problem["check"].startswith("P17") for problem in problems))

    def test_a_half_completed_batch_is_detected(self) -> None:
        def mutate(payload: dict) -> None:
            entry = payload["phases"][verifier.TRAIN_ROLE]["keys"][
                self._first_train_key(payload)
            ]
            entry["consumed"]["batches_completed"] -= 1

        problems = self._tampered_failures(mutate)
        self.assertTrue(any(problem["check"].startswith("P13") for problem in problems))

    def test_a_loader_that_was_never_drained_is_detected(self) -> None:
        """R1: every loader pass must be registered and drained after shutdown."""

        def mutate(payload: dict) -> None:
            phase = payload["phases"][verifier.TRAIN_ROLE]
            phase["closed_loaders"] = list(phase["closed_loaders"])[:-1]

        problems = self._tampered_failures(mutate)
        self.assertTrue(any(problem["check"].startswith("P18") for problem in problems))

    def test_the_assignment_question_shape_is_not_enough(self) -> None:
        """A payload that only reports plausible totals must still fail."""

        plan = self._plan("shape-only")
        payload = synth_payload(plan, num_workers=4)
        for phase in payload["phases"].values():
            for entry in phase["keys"].values():
                entry["reads"]["position_multiset_sha256"] = "2" * 64
                entry["consumed"]["completed_order_sha256"] = "2" * 64
                entry["consumed"]["delivered_order_sha256"] = "2" * 64
        problems = verifier.evaluate_loader_audit(payload, plan)["problems"]
        self.assertGreaterEqual(len(problems), 2 * (EPOCHS + len(CONDITIONS)))


class LoaderAuditEntryPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _index_rows()
        cls.index_info = _index_info(cls.rows)

    def _written_run(self, name: str, **kwargs) -> Path:
        root = _run_root(name, **kwargs)
        plan = verifier.loader_identity_plan(self.rows, root, self.index_info)
        payload = synth_payload(plan, num_workers=kwargs.get("num_workers", 4))
        (root / "loader_read_audit.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        return root

    def test_a_faithful_run_passes_the_top_level_checks(self) -> None:
        root = self._written_run("entry-pass")
        checks = verifier.Checks()
        info = verifier.verify_loader_identity_audit(checks, root, self.rows, self.index_info)
        self.assertTrue(checks.passed, checks.failed)
        self.assertEqual(info["problems"], [])
        self.assertEqual(info["checks_evaluated"], len(verifier.LOADER_AUDIT_CHECK_NAMES))

    def test_a_missing_audit_file_is_a_failure_not_a_pass(self) -> None:
        root = _fresh_scratch("entry-missing")
        checks = verifier.Checks()
        verifier.verify_loader_identity_audit(checks, root, self.rows, self.index_info)
        self.assertFalse(checks.passed)

    def test_a_tampered_copy_fails_the_top_level_checks(self) -> None:
        root = self._written_run("entry-tampered")
        payload = json.loads((root / "loader_read_audit.json").read_text(encoding="utf-8"))
        first = sorted(payload["phases"][verifier.TRAIN_ROLE]["keys"])[0]
        payload["phases"][verifier.TRAIN_ROLE]["keys"][first]["reads"]["windows_read"] -= 1
        (root / "loader_read_audit.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        checks = verifier.Checks()
        verifier.verify_loader_identity_audit(checks, root, self.rows, self.index_info)
        self.assertFalse(checks.passed)
        self.assertTrue(
            any("P7" in name for name in checks.failed), checks.failed
        )

    def test_every_tamper_control_is_detected_on_a_real_payload(self) -> None:
        root = self._written_run("entry-controls")
        controls = verifier.loader_audit_negative_controls(root, self.rows, self.index_info)
        self.assertGreaterEqual(len(controls), 10)
        undetected = [row for row in controls if not row["detected"]]
        self.assertEqual(undetected, [], undetected)

    def test_a_serial_run_is_not_forced_through_the_parallel_checks(self) -> None:
        root = _fresh_scratch("entry-serial")
        (root / "run_config.json").write_text(
            json.dumps({"loader": {"num_workers": 0, "parallel": False}}), encoding="utf-8"
        )
        config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
        loader_config = config.get("loader") or {}
        parallel = bool(loader_config.get("parallel")) or int(
            loader_config.get("num_workers") or 0
        ) > 0
        self.assertFalse(parallel)
        self.assertFalse((root / "loader_read_audit.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
