#!/usr/bin/env python3
"""STAGE1-LOSS-NORM: epoch-boundary interrupt/resume determinism check.

Runs the new entry twice on a *bounded* (smoke) configuration:

* ``reference`` -- two epochs, uninterrupted;
* ``resumed`` -- the same two epochs, but the process is killed as soon as the
  epoch-1 ``last_training_state.pt`` exists (i.e. inside epoch 2) and the run is then
  restarted with ``--resume-from``.

The two runs must agree on the frozen epoch orders, the per-epoch validation
metrics, every per-window validation file and the selected-epoch checkpoint.  The
real training device is used (CUDA), and because cuDNN runs in its deterministic
mode the comparison is expected to be exact; the script reports the observed
differences instead of assuming them.

Nothing here touches the formal candidate run root or the frozen inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_stage1_loss_normalization as entry  # noqa: E402

import torch  # noqa: E402

PYTHON = Path(sys.executable)
ENTRY = SCRIPT_DIR / "run_stage1_loss_normalization.py"
DEFAULT_ROOT = entry.OUTPUT_ROOT / "resume_check"
STATE_NAME = entry.STATE_NAME

#: Bounded configuration: big enough that an epoch takes several seconds, small
#: enough that the whole check stays a smoke test.
SMOKE_TRAIN_SAMPLES = 120
SMOKE_VAL_SAMPLES = 32
SMOKE_EPOCHS = 2
LOSS_MODE = "global_train_mean"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return entry.sha256_file(Path(path))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_bytes(path: Path) -> bytes:
    return Path(path).read_bytes()


def history_without_timing(run_root: Path) -> list[dict[str, str]]:
    """``training_history.csv`` minus the wall-clock ``seconds`` column."""

    import csv

    with (run_root / "training_history.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.pop("seconds", None)
    return rows


def checkpoint_state_hashes(run_root_or_checkpoint: Path) -> dict[str, Any]:
    """Model/front-end state hashes of ``best_model.pt`` (or of one checkpoint)."""

    path = Path(run_root_or_checkpoint)
    if path.is_dir():
        path = path / "best_model.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)


def entry_command(run_root: Path, loss_mode: str, resume_from: Path | None = None) -> list[str]:
    command = [
        str(PYTHON),
        str(ENTRY),
        "--mode",
        "smoke",
        "--smoke-epochs",
        str(SMOKE_EPOCHS),
        "--smoke-train-samples",
        str(SMOKE_TRAIN_SAMPLES),
        "--smoke-val-samples",
        str(SMOKE_VAL_SAMPLES),
        "--loss-normalization",
        loss_mode,
        "--run-root",
        str(run_root),
    ]
    if resume_from is not None:
        command.extend(["--resume-from", str(resume_from)])
    else:
        command.append("--force")
    return command


def run_uninterrupted(run_root: Path, loss_mode: str) -> dict[str, Any]:
    """One normal, uninterrupted bounded run."""

    log_path = run_root.parent / "resume_reference.log"
    run_root.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("wb") as handle:
        completed = subprocess.run(
            entry_command(run_root, loss_mode),
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "role": "reference",
        "run_root": str(run_root),
        "exit_code": int(completed.returncode),
        "log": str(log_path),
        "seconds": time.perf_counter() - started,
    }


def run_interrupted_then_resumed(run_root: Path, loss_mode: str) -> dict[str, Any]:
    """Start the bounded run, kill it inside epoch 2, then resume it."""

    log_path = run_root.parent / "resume_interrupted.log"
    resume_log = run_root.parent / "resume_resumed.log"
    state_path = run_root / STATE_NAME
    if state_path.exists():
        state_path.unlink()
    run_root.mkdir(parents=True, exist_ok=True)
    command = entry_command(run_root, loss_mode)
    started = time.perf_counter()
    killed_at_state: dict[str, Any] | None = None
    with log_path.open("wb") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 900
        while time.time() < deadline:
            if process.poll() is not None:
                break
            if state_path.exists():
                try:
                    payload = read_json(run_root / entry.STATE_JSON_NAME)
                except (json.JSONDecodeError, OSError):
                    payload = None
                if payload and payload.get("latest", {}).get("epoch") == 1:
                    killed_at_state = payload["latest"]
                    process.kill()
                    break
            time.sleep(0.05)
        process.wait(timeout=120)
    interrupted_exit = process.returncode
    keras_seconds = time.perf_counter() - started
    if killed_at_state is None:
        raise RuntimeError(
            "the interrupted run finished before the epoch-1 state could be observed; "
            f"see {log_path}"
        )
    progress_after_kill = read_json(run_root / "progress.json")
    state_bytes_before_resume = sha256_file(state_path)
    resume_started = time.perf_counter()
    with resume_log.open("wb") as handle:
        resumed = subprocess.run(
            entry_command(run_root, loss_mode, resume_from=state_path),
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "role": "interrupted+resumed",
        "run_root": str(run_root),
        "killed_at_state": killed_at_state,
        "state_sha256_before_resume": state_bytes_before_resume,
        "interrupted_exit_code": int(interrupted_exit),
        "progress_at_kill": progress_after_kill,
        "interrupt_seconds": keras_seconds,
        "resume_exit_code": int(resumed.returncode),
        "resume_log": str(resume_log),
        "resume_seconds": time.perf_counter() - resume_started,
    }


def compare(reference: Path, resumed: Path) -> dict[str, Any]:
    """Artifact-by-artifact comparison of the two bounded runs."""

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    reference_config = read_json(reference / "run_config.json")
    resumed_config = read_json(resumed / "run_config.json")
    check(
        "both runs completed",
        reference_config.get("status") == "completed" and resumed_config.get("status") == "completed",
        {
            "reference": reference_config.get("status"),
            "resumed": resumed_config.get("status"),
        },
    )
    check(
        "the resumed run reports the resume and its start epoch",
        int(resumed_config.get("start_epoch", 0)) == 2
        and bool(resumed_config.get("resume", {}).get("resumed")),
        {
            "start_epoch": resumed_config.get("start_epoch"),
            "resume": resumed_config.get("resume"),
        },
    )
    check(
        "the reference run has no resume record",
        "resume" not in reference_config,
        {"resume": reference_config.get("resume")},
    )
    check(
        "initial model hash is the frozen seed-2027 value",
        reference_config["initial_model_sha256"] == resumed_config["initial_model_sha256"],
        {
            "reference": reference_config["initial_model_sha256"],
            "resumed": resumed_config["initial_model_sha256"],
        },
    )
    check(
        "epoch order hashes are identical",
        reference_config["epoch_order_sha256"] == resumed_config["epoch_order_sha256"],
        {
            "reference": reference_config["epoch_order_sha256"],
            "resumed": resumed_config["epoch_order_sha256"],
        },
    )
    check(
        "identity order hashes are identical",
        reference_config["epoch_identity_order_sha256"]
        == resumed_config["epoch_identity_order_sha256"],
        {
            "reference": reference_config["epoch_identity_order_sha256"],
            "resumed": resumed_config["epoch_identity_order_sha256"],
        },
    )
    check(
        "the same epoch was selected with the same score",
        reference_config["selected_epoch"] == resumed_config["selected_epoch"]
        and reference_config["selected_epoch_score"] == resumed_config["selected_epoch_score"],
        {
            "reference": [reference_config["selected_epoch"], reference_config["selected_epoch_score"]],
            "resumed": [resumed_config["selected_epoch"], resumed_config["selected_epoch_score"]],
        },
    )
    check(
        "training history is identical apart from wall-clock seconds",
        history_without_timing(reference) == history_without_timing(resumed),
        {
            "reference": history_without_timing(reference),
            "resumed": history_without_timing(resumed),
            "note": "the 'seconds' column is a timing measurement and is excluded",
        },
    )
    check(
        "validation metrics are byte-identical",
        read_bytes(reference / "validation_metrics.csv") == read_bytes(resumed / "validation_metrics.csv"),
        {
            "reference_sha256": sha256_file(reference / "validation_metrics.csv"),
            "resumed_sha256": sha256_file(resumed / "validation_metrics.csv"),
        },
    )
    scores: dict[str, Any] = {}
    scores_identical = True
    for epoch in range(1, SMOKE_EPOCHS + 1):
        name = f"validation_scores_epoch_{epoch:02d}.csv"
        same = read_bytes(reference / name) == read_bytes(resumed / name)
        scores_identical = scores_identical and same
        scores[name] = {
            "identical": same,
            "reference_sha256": sha256_file(reference / name),
            "resumed_sha256": sha256_file(resumed / name),
        }
    check("per-window validation files are byte-identical", scores_identical, scores)
    checkpoint_states = checkpoint_state_hashes(reference) == checkpoint_state_hashes(resumed)
    reference_epoch_checkpoint = reference / f"epoch_{int(reference_config['selected_epoch']):02d}.pt"
    resumed_epoch_checkpoint = resumed / f"epoch_{int(resumed_config['selected_epoch']):02d}.pt"
    check(
        "selected-epoch checkpoint holds bitwise identical model and front-end states",
        checkpoint_states
        and reference_epoch_checkpoint.is_file()
        and resumed_epoch_checkpoint.is_file()
        and checkpoint_state_hashes(reference_epoch_checkpoint)
        == checkpoint_state_hashes(resumed_epoch_checkpoint),
        {
            "best_model": checkpoint_state_hashes(reference),
            "best_model_resumed": checkpoint_state_hashes(resumed),
            "selected_epoch_checkpoint": [
                reference_epoch_checkpoint.name,
                resumed_epoch_checkpoint.name,
            ],
            "note": (
                "checkpoint *files* also embed run_config with timings and are "
                "therefore not byte-compared; the tensor states are"
            ),
        },
    )
    reference_epochs = {int(row["epoch"]) for row in reference_config["epoch_history"]}
    resumed_epochs = {int(row["epoch"]) for row in resumed_config["epoch_history"]}
    check(
        "both runs cover the same two epochs",
        reference_epochs == resumed_epochs == {1, 2},
        {"reference": sorted(reference_epochs), "resumed": sorted(resumed_epochs)},
    )
    reference_records = reference_config["epoch_loader_records"]
    resumed_records = resumed_config["epoch_loader_records"]
    check(
        "loader consumption records cover both epochs in both runs",
        len(reference_records) == 2
        and len(resumed_records) == 2
        and [row["batches_completed"] for row in reference_records]
        == [row["batches_completed"] for row in resumed_records],
        {
            "reference": [row["batches_completed"] for row in reference_records],
            "resumed": [row["batches_completed"] for row in resumed_records],
        },
    )
    reference_audit = read_json(reference / "loader_read_audit.json")
    resumed_audit = read_json(resumed / "loader_read_audit.json")
    check(
        "the resumed segment audits only its own epochs and says so",
        int(resumed_audit["resumed_epochs"]) == 1
        and int(resumed_audit["expected_train_windows"]) == SMOKE_TRAIN_SAMPLES
        and int(reference_audit["resumed_epochs"]) == 0
        and int(reference_audit["expected_train_windows"]) == 2 * SMOKE_TRAIN_SAMPLES,
        {
            "reference": {
                "resumed_epochs": reference_audit["resumed_epochs"],
                "expected_train_windows": reference_audit["expected_train_windows"],
                "train_windows_read": reference_audit["train"]["windows_read"],
            },
            "resumed": {
                "resumed_epochs": resumed_audit["resumed_epochs"],
                "expected_train_windows": resumed_audit["expected_train_windows"],
                "train_windows_read": resumed_audit["train"]["windows_read"],
            },
        },
    )
    check(
        "the resumed segment read exactly one epoch of training windows",
        int(resumed_audit["train"]["windows_read"]) == SMOKE_TRAIN_SAMPLES,
        {"train_windows_read": resumed_audit["train"]["windows_read"]},
    )
    check(
        "both runs kept every read inside the frozen train/val roles",
        bool(reference_audit["passed"]) and bool(resumed_audit["passed"]) and bool(resumed_config.get("read_audit_passed")),
        {
            "reference_passed": reference_audit["passed"],
            "resumed_passed": resumed_audit["passed"],
        },
    )
    state = read_json(resumed / entry.STATE_JSON_NAME)
    check(
        "the resumed run wrote its own epoch-2 state",
        int(state["latest"]["epoch"]) == 2 and len(state["writes"]) >= 1,
        {"latest": state["latest"], "writes": len(state["writes"])},
    )
    return {
        "checks": checks,
        "passed": all(row["passed"] for row in checks),
        "failed": [row for row in checks if not row["passed"]],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--loss-normalization", default=LOSS_MODE)
    parser.add_argument("--skip-runs", action="store_true")
    args = parser.parse_args(argv)
    root: Path = args.root
    reference = root / "reference"
    resumed = root / "interrupted_resumed"
    root.mkdir(parents=True, exist_ok=True)
    if not args.skip_runs:
        # This check owns its two directories and always starts from a clean slate.
        for directory in (reference, resumed):
            if directory.exists():
                shutil.rmtree(directory)

    result: dict[str, Any] = {
        "schema": "dronerfa-stage1-loss-norm-resume-check/v1",
        "generated_at": utc_now(),
        "loss_mode": args.loss_normalization,
        "configuration": {
            "mode": "smoke",
            "epochs": SMOKE_EPOCHS,
            "train_samples": SMOKE_TRAIN_SAMPLES,
            "val_samples": SMOKE_VAL_SAMPLES,
        },
    }
    if not args.skip_runs:
        result["reference_run"] = run_uninterrupted(reference, args.loss_normalization)
        result["interrupted_run"] = run_interrupted_then_resumed(
            resumed, args.loss_normalization
        )
    result["comparison"] = compare(reference, resumed)
    result["passed"] = bool(result["comparison"]["passed"])
    (root / "resume_check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "checks": len(result["comparison"]["checks"]),
                "failed": [
                    {"check": row["check"], "detail": row["detail"]}
                    for row in result["comparison"]["failed"]
                ],
                "output": str(root / "resume_check.json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
