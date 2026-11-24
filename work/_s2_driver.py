"""Run a project script with stdout/stderr redirected to a real file.

The DSH file sandbox blocks capturing a native program's stdio through a pipe, so
``python ... | Tee-Object`` and ``> file`` both fail or lose output. This driver
opens the log file itself (a plain file handle) and runs the target script with
``runpy``, so progress of long runs is still observable.
"""
from __future__ import annotations

import contextlib
import os
import runpy
import sys

TARGET = os.path.abspath(sys.argv[1])
LOG = os.path.abspath(sys.argv[2])
ARGS = sys.argv[3:]

directory = os.path.dirname(TARGET)
os.chdir(directory)
if directory not in sys.path:
    sys.path.insert(0, directory)

with open(LOG, "w", encoding="utf-8", errors="replace", buffering=1) as handle:
    with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
        print(f"# driver: {TARGET} {' '.join(ARGS)}", flush=True)
        sys.argv = [TARGET, *ARGS]
        try:
            runpy.run_path(TARGET, run_name="__main__")
        except SystemExit as exit_error:
            print(f"# driver: exit {exit_error.code}", flush=True)
            raise
