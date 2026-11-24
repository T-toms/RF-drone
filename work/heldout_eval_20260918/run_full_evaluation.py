"""Run the frozen held-out evaluation and mirror stdout/stderr into a log file.

The harness sandbox denies a native process whose stdout is captured through a
PowerShell pipe, so the log is produced from inside Python instead of with a
shell redirection. The evaluated script itself is unchanged.

Extra convenience flags (removed before the target sees them):
  --log-name NAME   log file name inside work/heldout_eval_20260918
  --tag NAME        append an output_dir override to the target arguments
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
TARGET = PROJECT_DIR / "dronerfa_baseline" / "evaluate_heldout_models.py"
LOG_DIR = PROJECT_DIR / "work" / "heldout_eval_20260918"
DEFAULT_OUTPUT = PROJECT_DIR / "dronerfa_baseline" / "outputs" / "heldout_evaluation_20260918"


class Mirror:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


def split_wrapper_args(argv: list[str]) -> tuple[str, str | None, list[str]]:
    log_name = "full_run.log"
    tag: str | None = None
    target: list[str] = []
    position = 0
    while position < len(argv):
        token = argv[position]
        if token == "--log-name":
            log_name = argv[position + 1]
            position += 2
            continue
        if token == "--tag":
            tag = argv[position + 1]
            position += 2
            continue
        target.append(token)
        position += 1
    return log_name, tag, target


def main() -> int:
    log_name, tag, target_args = split_wrapper_args(sys.argv[1:])
    output_dir = DEFAULT_OUTPUT if tag is None else DEFAULT_OUTPUT / tag
    log_path = LOG_DIR / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    target_argv = [str(TARGET), "--output-dir", str(output_dir), *target_args]
    sys.path.insert(0, str(TARGET.parent))
    original_stdout, original_stderr = sys.stdout, sys.stderr
    exit_code = 0
    with log_path.open("w", encoding="utf-8", buffering=1) as handle:
        sys.stdout = Mirror(original_stdout, handle)
        sys.stderr = Mirror(original_stderr, handle)
        sys.argv = target_argv
        print(f"# wrapper: target={TARGET}", flush=True)
        print(f"# wrapper: log={log_path}", flush=True)
        print(f"# wrapper: output_dir={output_dir}", flush=True)
        try:
            runpy.run_path(str(TARGET), run_name="__main__")
        except SystemExit as exit_signal:
            exit_code = int(exit_signal.code or 0)
        except BaseException:
            import traceback

            traceback.print_exc()
            exit_code = 1
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
    print(f"EXIT_CODE={exit_code} log={log_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
