"""Throwaway helper: run the loader tests that do not need spawned workers.

The DSH file sandbox blocks Windows named pipes, so every ``multiprocessing`` start
(and therefore every ``num_workers > 0`` DataLoader) raises ``PermissionError``
inside this session.  This runner separates "the module is broken" from "the
sandbox forbids the parallel path"; the real suite is
``test_stage1_loader_optimization.py`` run without the sandbox.

Usage: ``python scratch/run_nonworker_tests.py`` from ``dronerfa_baseline``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

BASELINE = Path(__file__).resolve().parent.parent
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))

import test_stage1_loader_optimization as tests

SKIP = ("worker", "prefetch", "failed_consumer", "audit")

loader = unittest.TestLoader()
suite = unittest.TestSuite()
for case in (
    tests.ReferenceDefaultsTests,
    tests.EpochOrderTests,
    tests.TensorDigestTests,
    tests.LoaderEquivalenceTests,
    tests.WorkerAuditTests,
    tests.WorkerCleanupTests,
):
    for name in loader.getTestCaseNames(case):
        if any(token in name for token in SKIP):
            continue
        suite.addTest(case(name))

result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
