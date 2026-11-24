"""Throwaway helper: compact unittest summary (the sandbox blocks piping to filters).

The ``if __name__ == "__main__"`` guard is mandatory here: torch's DataLoader uses
the ``spawn`` start method, which re-imports this file in every worker.  Without the
guard each worker re-ran the whole suite and crashed ("exited unexpectedly").
"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

BASELINE = Path(__file__).resolve().parent.parent
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))

DEFAULT_TARGETS = ("test_stage1_12drones", "test_stage1_loader_optimization")


def main(argv: list[str]) -> int:
    targets = argv or list(DEFAULT_TARGETS)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for target in targets:
        suite.addTests(loader.loadTestsFromName(target))

    buffer = io.StringIO()
    result = unittest.TextTestRunner(stream=buffer, verbosity=0).run(suite)
    print(
        f"ran={result.testsRun} failures={len(result.failures)} errors={len(result.errors)} "
        f"skipped={len(result.skipped)}"
    )
    for test, _ in result.failures:
        print("FAIL", test.id())
    for test, _ in result.errors:
        print("ERROR", test.id())
    for test, reason in result.skipped:
        print("SKIP", test.id(), "|", reason)
    for test, traceback_text in result.failures + result.errors:
        print("---", test.id())
        print(traceback_text.strip().splitlines()[-1][:300])
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
