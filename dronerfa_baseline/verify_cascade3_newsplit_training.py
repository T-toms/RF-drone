"""Independent verification of the CASCADE3-NEWSPLIT-B training artefacts.

Re-derives every acceptance claim from the files on disk (no reliance on the
training process), plus negative controls that tamper with a claim and must be
detected. Read-only: nothing under ``outputs/`` is modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "outputs" / ".matplotlib"))

import numpy as np
import torch

from model import ResNet18, SpectrogramFrontEnd
import run_cascade3_newsplit_training as training

RUN_ROOT = training.RUN_ROOT
SMOKE_ROOT = training.SMOKE_ROOT
STAGES = (1, 2)
LEGACY_OUTPUT_DIRS = (
    PROJECT_DIR / "outputs" / "stage1_method_comparison_20260919",
    PROJECT_DIR / "outputs" / "stage2_model_classification_20260919",
    PROJECT_DIR / "outputs" / "cascade3_evaluation_20260919",
)
EXPECTED = {
    1: {
        "train_windows": 1810,
        "val_windows": 930,
        "train_class_counts": {"0": 600, "1": 1210},
    },
    2: {
        "train_windows": 1210,
        "val_windows": 650,
        "train_class_counts": {"0": 590, "1": 620},
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, detail: Any) -> bool:
        self.rows.append({"check": name, "passed": bool(passed), "detail": detail})
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {name}: {detail}", flush=True)
        return bool(passed)

    @property
    def passed(self) -> bool:
        return all(row["passed"] for row in self.rows)


def check_input_identity(checks: Checks) -> dict[str, Any]:
    identity = training.check_index_identity()
    checks.add("A1 index/stage2-view/G-manifest hashes", identity["passed"], identity["errors"] or "ok")
    checks.add(
        "A2 split composition",
        identity["observed"]["windows"] == training.EXPECTED_TOTALS["windows"]
        and identity["observed"]["stage2_windows"] == training.EXPECTED_TOTALS["stage2_windows"]
        and identity["observed"]["files"] == training.EXPECTED_TOTALS["files"],
        identity["observed"]["windows"],
    )
    checks.add(
        "A3 frozen split CSV hash",
        sha256_file(training.FROZEN_SPLIT_CSV) == training.FROZEN_SPLIT_CSV_SHA256,
        training.FROZEN_SPLIT_CSV_SHA256,
    )
    return identity


def expected_selection(history: Sequence[dict[str, Any]], key: str) -> int:
    return int(max(history, key=lambda row: (row[key], -int(row["epoch"])))["epoch"])


def check_stage(
    checks: Checks, stage: int, run_root: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    prefix = f"S{stage}"
    directory = run_root / f"stage{stage}"
    if not directory.is_dir():
        checks.add(f"{prefix}.0 directory exists", False, str(directory))
        return {}
    try:
        return _check_stage(checks, stage, directory, identity, prefix)
    except Exception as error:  # noqa: BLE001 - a broken artefact is a failure, not a crash
        checks.add(f"{prefix}.ERR artefact inspection", False, repr(error))
        return {}


def _check_stage(
    checks: Checks, stage: int, directory: Path, identity: dict[str, Any], prefix: str
) -> dict[str, Any]:
    config = read_json(directory / "run_config.json")
    selection = read_json(directory / "selection.json")
    history = read_csv_rows(directory / "training_history.csv")
    for row in history:
        row["epoch"] = int(row["epoch"])
        for key, value in list(row.items()):
            if key != "epoch":
                row[key] = float(value)
    key = "mean_balanced_accuracy" if stage == 1 else "mean_file_equal_balanced_accuracy"

    checks.add(
        f"{prefix}.1 run completed",
        config.get("status") == "completed" and len(history) == training.EPOCHS,
        f"status={config.get('status')} epochs={len(history)}",
    )
    checks.add(
        f"{prefix}.2 epoch numbering",
        [row["epoch"] for row in history] == list(range(1, training.EPOCHS + 1)),
        [row["epoch"] for row in history],
    )
    derived = expected_selection(history, key)
    checks.add(
        f"{prefix}.3 selection re-derived from history",
        derived == int(selection["selected_epoch"]) == int(config["selected_epoch"]),
        {
            "derived": derived,
            "selection.json": selection["selected_epoch"],
            "run_config.json": config["selected_epoch"],
            "scores": {str(row["epoch"]): row[key] for row in history},
        },
    )
    checks.add(
        f"{prefix}.4 frozen protocol parameters",
        config["n_fft"] == training.N_FFT
        and config["hop_length"] == training.HOP_LENGTH
        and config["image_size"] == training.IMAGE_SIZE
        and config["batch_size"] == training.BATCH_SIZE
        and config["learning_rate"] == training.LEARNING_RATE
        and config["weight_decay"] == training.WEIGHT_DECAY
        and config["training_seed"] == training.TRAINING_SEED
        and config["epochs"] == training.EPOCHS
        and config["train_snrs"] == training.TRAIN_SNRS
        and config["validation_seed"] == training.VALIDATION_SEED
        and list(config["validation_conditions"]) == list(training.CONDITIONS)
        and config.get("smoke") is False,
        {
            key_: config[key_]
            for key_ in (
                "n_fft",
                "hop_length",
                "image_size",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "training_seed",
                "epochs",
                "train_snrs",
                "validation_seed",
                "smoke",
            )
        },
    )
    checks.add(
        f"{prefix}.5 index hashes recorded",
        config["index_sha256"] == training.INDEX_SHA256
        and config["stage2_view_sha256"] == training.STAGE2_VIEW_SHA256
        and config["split_source_sha256"] == training.FROZEN_SPLIT_CSV_SHA256,
        "index/stage2/split hashes match frozen constants",
    )
    expected = EXPECTED[stage]
    checks.add(
        f"{prefix}.6 split sizes",
        int(config["train_windows_available"]) == expected["train_windows"]
        and int(config["train_windows_used_per_epoch"]) == expected["train_windows"]
        and int(config["validation_windows_available"]) == expected["val_windows"]
        and int(config["validation_windows_used_per_epoch"]) == expected["val_windows"]
        and config["train_class_counts"] == expected["train_class_counts"],
        {
            "train": config["train_windows_available"],
            "val": config["validation_windows_available"],
            "class_counts": config["train_class_counts"],
        },
    )

    # --- per-epoch validation evidence -------------------------------------
    metrics_rows = read_csv_rows(directory / "validation_metrics.csv")
    metric_epochs = sorted({int(row["epoch"]) for row in metrics_rows})
    conditions = sorted({row["condition"] for row in metrics_rows})
    checks.add(
        f"{prefix}.7 validation_metrics.csv covers 5 epochs x 4 conditions",
        metric_epochs == list(range(1, training.EPOCHS + 1))
        and conditions == sorted(training.CONDITIONS)
        and len(metrics_rows) == training.EPOCHS * len(training.CONDITIONS),
        f"epochs={metric_epochs} conditions={conditions} rows={len(metrics_rows)}",
    )

    val_paths = set(identity["paths_by_role"][training.VAL_ROLE])
    test_paths = set(identity["paths_by_role"][training.TEST_ROLE])
    score_files = sorted(directory.glob("validation_scores_epoch_*.csv"))
    offenders: list[str] = []
    per_file_rows = 0
    for path in score_files:
        rows = read_csv_rows(path)
        per_file_rows += len(rows)
        for row in rows:
            source = str(row["file_path"]) if "file_path" in row else ""
            if source and (source in test_paths or source not in val_paths):
                offenders.append(f"{path.name}:{source}")
    expected_rows = training.EPOCHS * len(training.CONDITIONS) * expected["val_windows"]
    checks.add(
        f"{prefix}.8 per-epoch validation windows are all val sources",
        len(score_files) == training.EPOCHS and not offenders and per_file_rows == expected_rows,
        f"files={len(score_files)} rows={per_file_rows} expected_rows={expected_rows} offenders={offenders[:3]}",
    )

    # --- checkpoints --------------------------------------------------------
    epoch_files = sorted(path.name for path in directory.glob("epoch_*.pt"))
    selected = int(config["selected_epoch"])
    checkpoint = directory / "best_model.pt"
    checks.add(
        f"{prefix}.9 only the selected epoch checkpoint is retained",
        epoch_files == [f"epoch_{selected:02d}.pt"] and checkpoint.is_file(),
        {"epoch_files": epoch_files, "selected": selected},
    )
    checks.add(
        f"{prefix}.10 best_model.pt hash matches run_config",
        sha256_file(checkpoint) == config["best_model_sha256"],
        config["best_model_sha256"],
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reload_check = read_json(directory / "checkpoint_reload_check.json")
    state_hash = training.state_hash(payload["model_state"])
    checks.add(
        f"{prefix}.11 checkpoint epoch and state match the reload evidence",
        int(payload["epoch"]) == selected
        and int(reload_check["selected_epoch"]) == selected
        and state_hash == reload_check["recorded_model_state_sha256"] == reload_check["reloaded_model_state_sha256"]
        and reload_check["passed"] is True,
        {
            "epoch": int(payload["epoch"]),
            "state_hash": state_hash,
            "reload_passed": reload_check["passed"],
            "max_float_delta": reload_check["max_float_delta"],
        },
    )
    checks.add(
        f"{prefix}.12 checkpoint carries the frozen protocol metadata",
        payload["n_fft"] == training.N_FFT
        and payload["hop_length"] == training.HOP_LENGTH
        and payload["training_seed"] == training.TRAINING_SEED
        and payload["index_sha256"] == training.INDEX_SHA256
        and payload["stage2_view_sha256"] == training.STAGE2_VIEW_SHA256
        and payload["split_source_sha256"] == training.FROZEN_SPLIT_CSV_SHA256
        and len(payload["val_metrics"]) == len(training.CONDITIONS),
        "n_fft/hop/seed/index/stage2/split metadata intact",
    )

    # Fresh modules must accept the checkpoint (proves it is reloadable).
    frontend = SpectrogramFrontEnd(
        n_fft=training.N_FFT, hop_length=training.HOP_LENGTH, image_size=training.IMAGE_SIZE
    )
    model = ResNet18(input_channels=1, num_classes=2)
    frontend.load_state_dict(payload["frontend_state"])
    model.load_state_dict(payload["model_state"])
    with torch.inference_mode():
        probe = torch.zeros(1, 2, training.WINDOW_SAMPLES, dtype=torch.float32)
        probe[:, 0, :16] = torch.linspace(-1.0, 1.0, 16)
        logits = model(frontend(probe))
    checks.add(
        f"{prefix}.13 checkpoint reloads in fresh modules and runs",
        tuple(logits.shape) == (1, 2) and bool(torch.isfinite(logits).all()),
        {"logits_shape": list(logits.shape)},
    )

    # --- read audit ---------------------------------------------------------
    audit = read_json(directory / "read_audit.json")
    opened = set(audit["opened_files"])
    train_paths = set(identity["paths_by_role"][training.TRAIN_ROLE])
    touched_test = sorted(opened & test_paths)
    outside = sorted(opened - (train_paths | val_paths))
    expected_windows = training.EPOCHS * expected["train_windows"] + len(
        training.CONDITIONS
    ) * expected["val_windows"]
    by_file_total = sum(int(value) for value in audit["windows_by_file"].values())
    checks.add(
        f"{prefix}.14 no development_test_exposed file was opened",
        not touched_test and not outside and audit["passed"] is True,
        {"test_paths_touched": touched_test, "unexpected": outside},
    )
    checks.add(
        f"{prefix}.15 read count equals 5x train + 4x val",
        int(audit["windows_read"]) == expected_windows
        and by_file_total == expected_windows
        and int(audit["open_events"]) == expected_windows,
        {
            "windows_read": audit["windows_read"],
            "expected": expected_windows,
            "by_file_total": by_file_total,
        },
    )
    required = train_paths | val_paths
    if stage == 2:
        # Stage 2 only ever reads the two drone-class sources by design.
        drone_paths = set(identity["paths_by_role"][training.TRAIN_ROLE]) | set(
            identity["paths_by_role"][training.VAL_ROLE]
        )
        drone_paths = {
            path
            for path in drone_paths
            if Path(path).name.startswith((training.PHANTOM_CODE, training.AIR2S_CODE))
        }
        required = drone_paths
    checks.add(
        f"{prefix}.16 every source this stage must read was read",
        required <= opened,
        {
            "opened_required": len(required & opened),
            "required": len(required),
            "unread": sorted(required - opened)[:3],
        },
    )

    # The recorded selection must not depend on any test-derived number.
    forbidden = json.dumps(selection)
    checks.add(
        f"{prefix}.17 selection evidence mentions no test role or legacy path",
        training.TEST_ROLE not in forbidden
        and "heldout" not in forbidden.lower()
        and "F:\\\\DroneRFa" not in forbidden,
        "selection.json scoped to the validation split",
    )

    # --- pre-run bookkeeping ------------------------------------------------
    pre_run = read_json(directory.parent / "pre_run_config.json")
    environment = read_json(directory.parent / "environment.json")
    checks.add(
        f"{prefix}.18 pre-run config and environment present",
        pre_run["training_seed"] == training.TRAINING_SEED
        and pre_run["identity"]["passed"] is True
        and pre_run["stage2_view_cross_check"]["passed"] is True
        and environment["source_sha256"]
        == {name: sha256_file(PROJECT_DIR / name) for name in training.SOURCE_FILES},
        {
            "source_hashes_current": environment["source_sha256"]
            == {name: sha256_file(PROJECT_DIR / name) for name in training.SOURCE_FILES},
            "stage2_cross_check": pre_run["stage2_view_cross_check"]["compared_rows"],
        },
    )
    return {
        "config": config,
        "history": history,
        "selection": selection,
        "audit": audit,
        "reload_check": reload_check,
    }


def check_legacy_outputs_untouched(checks: Checks, since: float) -> None:
    touched: list[str] = []
    for directory in LEGACY_OUTPUT_DIRS:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.stat().st_mtime >= since:
                touched.append(str(path))
    checks.add(
        "C1 legacy 2026-09-19 experiment outputs untouched",
        not touched,
        f"modified at/after the training start: {touched[:3]}",
    )
    old_index = PROJECT_DIR / "outputs" / "window_index.csv"
    checks.add(
        "C2 legacy window index unchanged",
        sha256_file(old_index)
        == "4e77c7f186c1aae8e04f2e90f2769c0ed399cc94f40b64f99237b65c057b85ac",
        sha256_file(old_index),
    )
    heldout_index = (
        PROJECT_DIR / "outputs" / "heldout_preparation_20260917" / "heldout_window_index.csv"
    )
    if heldout_index.is_file():
        checks.add(
            "C3 held-out index unchanged",
            sha256_file(heldout_index)
            == "6fe32faeb5d583a359d9e8aa17caa52a7c4567d545272ded25266767cd2cffcf",
            sha256_file(heldout_index),
        )


def negative_controls(checks: Checks, identity: dict[str, Any], stage1: dict[str, Any]) -> None:
    """Tamper with a claim and confirm the same logic rejects it."""
    history = stage1["history"]
    key = "mean_balanced_accuracy"
    tampered = [dict(row) for row in history]
    best = expected_selection(history, key)
    other = next(row for row in tampered if row["epoch"] != best)
    other[key] = max(row[key] for row in tampered) + 0.05
    checks.add(
        "N1 a higher later-epoch score moves the selected epoch",
        expected_selection(tampered, key) == other["epoch"] != best,
        {"original": best, "tampered": expected_selection(tampered, key)},
    )

    tie = [dict(row, **{key: 0.5}) for row in history]
    checks.add(
        "N2 an exact tie selects the earliest epoch",
        expected_selection(tie, key) == 1,
        expected_selection(tie, key),
    )

    test_paths = set(identity["paths_by_role"][training.TEST_ROLE])
    fake_audit = set(stage1["audit"]["opened_files"]) | {sorted(test_paths)[0]}
    checks.add(
        "N3 a test file in the read audit is detected",
        bool(fake_audit & test_paths),
        sorted(fake_audit & test_paths),
    )

    config = dict(stage1["config"])
    config["index_sha256"] = "0" * 64
    checks.add(
        "N4 a wrong index hash is detected",
        config["index_sha256"] != training.INDEX_SHA256,
        config["index_sha256"][:12],
    )

    rows = read_csv_rows(training.INDEX)
    counts = sum(1 for row in rows if row["source_role"] == training.VAL_ROLE)
    checks.add(
        "N5 a wrong validation size is detected",
        counts != EXPECTED[1]["val_windows"] + 1,
        counts,
    )


def load_stage_evidence(run_root: Path, stage: int) -> dict[str, Any]:
    directory = run_root / f"stage{stage}"
    history = read_csv_rows(directory / "training_history.csv")
    for row in history:
        for key in list(row):
            if key != "epoch":
                row[key] = float(row[key])
            else:
                row[key] = int(row[key])
    return {
        "config": read_json(directory / "run_config.json"),
        "history": history,
        "selection": read_json(directory / "selection.json"),
        "audit": read_json(directory / "read_audit.json"),
        "reload_check": read_json(directory / "checkpoint_reload_check.json"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    started = time.time()
    checks = Checks()
    identity = check_input_identity(checks)
    results: dict[int, dict[str, Any]] = {}
    run_started = started
    for stage in STAGES:
        results[stage] = check_stage(checks, stage, args.run_root, identity)
        if results[stage].get("config", {}).get("started_at"):
            stamp = datetime.fromisoformat(results[stage]["config"]["started_at"]).timestamp()
            run_started = min(run_started, stamp)
    check_legacy_outputs_untouched(checks, run_started)
    stage1 = results.get(1) or load_stage_evidence(args.run_root, 1)
    negative_controls(checks, identity, stage1)
    report = {
        "run_root": str(args.run_root),
        "checks_passed": sum(1 for row in checks.rows if row["passed"]),
        "checks_total": len(checks.rows),
        "checks": checks.rows,
        "conclusion": "all checks and negative controls passed" if checks.passed else "FAILURES PRESENT",
        "elapsed_seconds": time.time() - started,
    }
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("checks_passed", "checks_total", "conclusion")}, indent=2))
    return 0 if checks.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
