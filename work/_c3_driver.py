"""Run a project script in-process with stdout/stderr written to a log file.

The sandbox does not let a child process' stdio be captured through a pipe, so the
log file is opened here and the target script is executed with ``runpy``; the
process exit code is printed to the real stdout afterwards.

    python work/_c3_driver.py <script.py> <log file> [script args...]
"""
from __future__ import annotations

import os
import runpy
import sys
import traceback
from pathlib import Path


def main() -> int:
    script = Path(sys.argv[1]).resolve()
    log_path = Path(sys.argv[2]).resolve()
    script_args = sys.argv[3:]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    previous_stdout, previous_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = handle, handle
    if str(script.parent) not in sys.path:
        sys.path.insert(0, str(script.parent))
    os.chdir(script.parent)
    sys.argv = [str(script), *script_args]
    code = 0
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as error:  # noqa: PERF203
        value = error.code
        code = 0 if value is None else (value if isinstance(value, int) else 1)
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout, sys.stderr = previous_stdout, previous_stderr
        handle.close()
    print(f"driver exit {code} -> {log_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
