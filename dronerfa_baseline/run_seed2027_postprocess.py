#!/usr/bin/env python3
"""Wait for the seed-2027 stage-1 run to finish, then verify and compare it.

Written for the STAGE1-12DRONES-B/STAGE1-IO-OPT session: the benchmark must not run
beside a full training job (shared G: drive), so this script polls for the run to
report ``status=completed`` and only then starts the serial post-processing chain:

1. independent verification of the seed-2027 artefacts
   (``verify_stage1_12drones_training.py --training-seed 2027``);
2. the seed 2026 vs seed 2027 comparison and its Markdown report
   (``compare_stage1_12drones_seeds.py``).

It never trains, never stops the running job, and never writes into ``seed_2026`` or
``seed_2027``.  Every step's stdout is appended to ``pipeline.log`` in this directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUN_ROOT = SCRIPT_DIR / "outputs" / "stage1_12drones_20260926" / "seed_2027"
PROGRESS = RUN_ROOT / "progress.json"
TRAIN_LOG = RUN_ROOT / "stage1_train.log"
#: Benchmark results live with the other STAGE1-IO-OPT artefacts.
IO_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "stage1_io_optimization_20260927"
LOG_PATH = IO_OUTPUT_ROOT / "pipeline.log"
PYTHON = sys.executable
POLL_SECONDS = 120
#: Bound the wait so a stalled run cannot leave this process hanging forever
#: (5 epochs at ~35 min plus the reload check and margin).
MAX_WAIT_SECONDS = 6 * 60 * 60


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def training_finished() -> tuple[bool, str]:
    if (RUN_ROOT / "summary.json").is_file():
        return True, "summary.json present"
    if PROGRESS.is_file():
        payload = read_json(PROGRESS)
        status = str(payload.get("status"))
        if status in {"completed", "failed"}:
            return True, f"progress.status={status}"
        return False, f"progress.status={status} epochs={payload.get('epochs_completed')}"
    return False, "no progress.json yet"


def run_step(name: str, args: list[str]) -> int:
    log(f"START {name}: {' '.join(args)}")
    started = time.perf_counter()
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            args,
            cwd=str(SCRIPT_DIR),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    log(f"END {name}: exit={result.returncode} seconds={elapsed:.1f}")
    return int(result.returncode)


def main() -> int:
    log("pipeline armed: waiting for the seed-2027 run to finish")
    deadline = time.time() + MAX_WAIT_SECONDS
    while True:
        finished, detail = training_finished()
        if finished:
            log(f"training finished: {detail}")
            break
        if time.time() > deadline:
            log(f"ABORT: still running after {MAX_WAIT_SECONDS}s ({detail})")
            return 2
        time.sleep(POLL_SECONDS)

    exits: dict[str, int] = {}
    exits["verify_seed_2027"] = run_step(
        "verify_seed_2027",
        [PYTHON, "verify_stage1_12drones_training.py", "--training-seed", "2027"],
    )
    exits["compare_seed_2026_2027"] = run_step(
        "compare_seed_2026_2027",
        [PYTHON, "compare_stage1_12drones_seeds.py"],
    )
    exits["benchmark_io"] = run_step(
        "benchmark_io",
        [
            PYTHON,
            "benchmark_stage1_io.py",
            "--windows",
            "24",
            "--iterations",
            "3",
            "--output-dir",
            str(IO_OUTPUT_ROOT),
        ],
    )
    log(f"pipeline exits: {exits}")
    print(json.dumps(exits, indent=2), flush=True)
    return 0 if all(code == 0 for code in exits.values()) else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
