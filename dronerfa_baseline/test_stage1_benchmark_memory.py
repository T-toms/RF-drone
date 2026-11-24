#!/usr/bin/env python3
"""Regression tests for the benchmark's process-memory sampling (STAGE1-IO-OPT R2).

R1 claimed to record process memory, but all three configurations ended up as ``{}``
in the benchmark JSON and the summary CSV had no memory column.  The cause was a
ctypes marshalling bug in ``benchmark_stage1_io.process_memory_mb``: no ``argtypes``
were declared, so ``GetCurrentProcess()``'s ``(HANDLE)-1`` pseudo handle was passed as
a 32-bit ``int``, the callee saw an invalid handle, returned ``FALSE`` and the bare
``except`` turned that into an empty mapping that the report quoted as a measurement.

These tests pin the fixed behaviour:

1. the parent reading is non-empty, finite and positive, with ``peak >= working set``;
2. the binding reports which entry point and structure size it used;
3. the current-process handle survives marshalling (the exact R1 truncation);
4. the untyped call shape still reproduces the R1 failure on this host, so the pin is
   against a reproduced bug and not against a guess (skipped, with a reason, if a
   future ctypes marshals it correctly);
5. failures are **visible**: ``process_memory_mb``/``MemoryTracker`` raise with the OS
   error text, and a non-strict tracker records ``ok=false`` instead of ``{}``;
6. the aggregate keeps every field the JSON/CSV writers read.

Scope note: the benchmark records **parent-process** memory only.  No worker or process
tree is walked, and these tests spawn no processes at all -- they measure the pytest
process itself, so there is nothing to clean up afterwards.

Run: ``python -m unittest test_stage1_benchmark_memory``.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import benchmark_stage1_io as bench  # noqa: E402

#: Fields the JSON/CSV writers of ``benchmark_stage1_io.main`` read from a payload.
REQUIRED_SUMMARY_FIELDS = (
    "ok",
    "strict",
    "scope",
    "definition",
    "sample_count",
    "iterations_aggregated",
    "samples",
    "sample_seconds_total",
    "parent_working_set_mb_max",
    "parent_peak_working_set_mb_max",
    "errors",
)


class FailingApi:
    """Stand-in binding whose read fails the way a broken host would."""

    def __init__(self) -> None:
        self.calls = 0

    def read_current(self) -> dict[str, float]:
        self.calls += 1
        raise bench.MemorySamplingError(
            "GetProcessMemoryInfo(this process) failed: GetLastError=6 (句柄无效)"
        )

    def describe(self) -> dict[str, object]:
        return {"api": "FailingApi"}


class ParentMemoryTests(unittest.TestCase):
    """The parent reading is real evidence, not an empty placeholder."""

    def test_parent_memory_is_positive_and_finite(self) -> None:
        reading = bench.process_memory_mb()
        self.assertTrue(reading, "process_memory_mb returned an empty mapping")
        self.assertEqual(
            set(reading), {"parent_working_set_mb", "parent_peak_working_set_mb"}
        )
        for name, value in reading.items():
            with self.subTest(field=name):
                self.assertIsInstance(value, float)
                self.assertTrue(math.isfinite(value), f"{name} is not finite: {value}")
                self.assertGreater(value, 0.0, f"{name} is not positive: {value}")
        self.assertGreaterEqual(
            reading["parent_peak_working_set_mb"],
            reading["parent_working_set_mb"],
            "the OS high-water mark cannot be below the current working set",
        )
        # A live CPython process with torch imported is far above 1 MiB; a truncated or
        # mis-scaled counter would not be.
        self.assertGreater(reading["parent_working_set_mb"], 1.0)

    def test_binding_reports_entry_point_and_structure_size(self) -> None:
        described = bench.win32_memory_api().describe()
        self.assertIn(
            described["api"],
            ("kernel32.K32GetProcessMemoryInfo", "psapi.GetProcessMemoryInfo"),
        )
        self.assertTrue(described["use_last_error"])
        self.assertIn("parent", described["scope"])
        if struct.calcsize("P") == 8:
            self.assertEqual(
                described["process_memory_counters_bytes"],
                72,
                "PROCESS_MEMORY_COUNTERS must be the 72-byte x64 layout passed as cb",
            )
        else:  # pragma: no cover - this project only runs on x64 Windows
            self.skipTest("32-bit layout is not the frozen benchmark layout")

    def test_current_process_handle_survives_marshalling(self) -> None:
        handle = bench.win32_memory_api().current_process_handle()
        self.assertIsInstance(handle, int)
        # GetCurrentProcess returns the documented (HANDLE)-1 pseudo handle. The R1 bug
        # was that an undeclared restype delivered it as a *32-bit* value; any handle not
        # above 2**32 means it was truncated again.
        self.assertGreater(handle, 1 << 32, f"handle looks truncated: {handle}")

    def test_untyped_call_reproduces_the_r1_failure(self) -> None:
        """The bug this module fixes is a reproduced failure, not a guess."""

        import ctypes

        api = bench.win32_memory_api()
        counters = api.counters_type()
        counters.cb = api.counters_size
        # Exactly the R1 call shape: no argtypes, no restype, psapi, no error capture.
        result = ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if result:
            self.skipTest(
                "the untyped call now succeeds on this host, so the R1 truncation cannot "
                "be reproduced here; the typed path is still covered by the other tests"
            )
        self.assertEqual(result, 0, "BOOL FALSE: the failure was silent, not an exception")
        self.assertEqual(counters.WorkingSetSize, 0)


class FailingApiTests(unittest.TestCase):
    """A failed sample must be loud, never an empty mapping reported as success."""

    def _patched(self) -> mock._patch:  # type: ignore[type-arg]
        return mock.patch.object(bench, "_API", FailingApi())

    def test_process_memory_mb_raises_with_diagnostic(self) -> None:
        with self._patched():
            with self.assertRaises(bench.MemorySamplingError) as caught:
                bench.process_memory_mb()
        message = str(caught.exception)
        self.assertIn("GetProcessMemoryInfo", message)
        self.assertIn("GetLastError=6", message)
        self.assertNotEqual(message.strip(), "")

    def test_strict_tracker_aborts_and_non_strict_records_the_error(self) -> None:
        with self._patched():
            with self.assertRaises(bench.MemorySamplingError):
                bench.MemoryTracker(strict=True).sample("epoch_mid", epoch=1)
            tracker = bench.MemoryTracker(strict=False)
            sample = tracker.sample("epoch_mid", epoch=1)

        self.assertFalse(sample["ok"])
        self.assertIsNone(sample["parent_working_set_mb"])
        self.assertIsNone(sample["parent_peak_working_set_mb"])
        self.assertIn("GetLastError=6", sample["parent_error"])
        self.assertEqual(sample["label"], "epoch_mid")
        self.assertEqual(sample["epoch"], 1)
        self.assertGreater(sample["seconds"], 0.0)
        payload = tracker.payload()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["sample_count"], 1)
        self.assertEqual(payload["parent_working_set_mb_max"], None)
        # Not an empty mapping: the failure text travels with the record.
        self.assertTrue(payload["errors"])
        self.assertTrue(all(error.strip() for error in payload["errors"]))

    def test_aggregate_marks_failed_iterations_and_empty_input(self) -> None:
        good = bench.MemoryTracker(strict=True)
        good.sample("iteration_start")
        first = good.payload()
        second = json.loads(json.dumps(first))
        # The aggregate recomputes every maximum from the samples, so the second
        # iteration's sample is what has to be larger.
        second["samples"][0]["parent_working_set_mb"] = (
            first["parent_working_set_mb_max"] + 10.0
        )

        combined = bench.aggregate_memory([first, second])
        self.assertTrue(combined["ok"])
        self.assertEqual(combined["iterations_aggregated"], 2)
        self.assertEqual(combined["sample_count"], 2)
        self.assertAlmostEqual(
            combined["parent_working_set_mb_max"], first["parent_working_set_mb_max"] + 10.0
        )
        self.assertAlmostEqual(
            combined["parent_working_set_mb_max"],
            max(
                sample["parent_working_set_mb"]
                for payload in (first, second)
                for sample in payload["samples"]
            ),
        )
        self.assertGreater(combined["sample_seconds_total"], 0.0)

        failed = json.loads(json.dumps(first))
        failed["ok"] = False
        failed["errors"] = ["epoch_mid: GetProcessMemoryInfo failed"]
        self.assertFalse(bench.aggregate_memory([first, failed])["ok"])

        empty = bench.aggregate_memory([])
        self.assertFalse(empty["ok"])
        self.assertEqual(empty["sample_count"], 0)
        self.assertTrue(empty["errors"])

    def test_summary_carries_every_field_the_writers_read(self) -> None:
        tracker = bench.MemoryTracker(strict=True)
        tracker.sample("iteration_start")
        for payload in (tracker.payload(), bench.aggregate_memory([tracker.payload()])):
            with self.subTest(payload=str(type(payload).__name__)):
                self.assertEqual(set(REQUIRED_SUMMARY_FIELDS) - set(payload), set())
                self.assertIn("Win32 GetProcessMemoryInfo", payload["definition"])
                self.assertIn("parent", payload["scope"])
                self.assertGreater(payload["sample_seconds_total"], 0.0)
                # The definition must not let a reader believe worker memory is included.
                self.assertIn("not", payload["definition"].lower())
                self.assertIn("worker", payload["definition"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
