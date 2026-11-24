#!/usr/bin/env python3
"""STAGE1-LOSS-NORM: independent verification of the candidate run's own claims.

The accepted ``verify_stage1_12drones_training.py`` already re-derives the frozen
protocol claims (per-window decisions replayed from raw RF0, read audit, loader
identity, tamper controls).  This verifier covers what is *new* in this task and
re-computes it from the artefacts alone:

* the pre-run protocol gate (initialisation hash and all five epoch order hashes
  against the accepted seed-2027 control) is re-run here, not trusted;
* the objective's per-step evidence: 5 x 2398 steps, the 2-window tail batch's
  denominator, ``loss == numerator / denominator`` and finiteness;
* the batch composition implied by the frozen index and the recorded epoch orders
  (how many four-window batches contain no background window at all);
* every validation per-window file: full coverage, unique keys, labels from the
  frozen index, and the recorded metrics re-derived from the decisions;
* the selection rule, the retained checkpoint and the ``best_model.pt`` payload;
* the read-audit totals compared against an independently recomputed budget;
* the epoch-boundary state file's identity and content;
* in-memory tamper controls that must be detected.

It never trains, never writes inside a run directory, and never opens the G: drive.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import loss_normalization as ln  # noqa: E402
import run_stage1_12drones_training as base  # noqa: E402
import run_stage1_loss_normalization as entry  # noqa: E402

CONDITIONS = tuple(base.CONDITIONS)
SPLIT_LENGTH = 9590
VAL_LENGTH = 2360
BATCHES_PER_EPOCH = 2398
TAIL_BATCH_SIZE = 2
EPOCHS = base.EPOCHS
DEFAULT_CANDIDATE = entry.OUTPUT_ROOT / entry.CANDIDATE_DIR_NAME
DEFAULT_CONTROL = entry.CONTROL_ROOT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, detail: Any = None) -> bool:
        self.rows.append({"check": name, "passed": bool(passed), "detail": detail})
        return bool(passed)

    @property
    def passed(self) -> bool:
        return all(row["passed"] for row in self.rows)

    def payload(self) -> dict[str, Any]:
        failed = [row for row in self.rows if not row["passed"]]
        return {
            "total_checks": len(self.rows),
            "passed_checks": len(self.rows) - len(failed),
            "failed_checks": failed,
            "checks": self.rows,
            "passed": not failed,
        }


def metrics_from_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([int(row["label"]) for row in records], dtype=np.int64)
    prediction = np.asarray([int(row["prediction"]) for row in records], dtype=np.int64)
    tp = int(np.sum((y == 1) & (prediction == 1)))
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "windows": len(y),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tp + tn) / max(len(y), 1),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "drone_recall": recall,
        "background_specificity": specificity,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def file_equal_from_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    per_file: dict[str, dict[str, float]] = {}
    for row in records:
        slot = per_file.setdefault(
            str(row["filename"]), {"label": int(row["label"]), "correct": 0.0, "windows": 0.0}
        )
        slot["correct"] += float(int(row["prediction"]) == int(row["label"]))
        slot["windows"] += 1.0
    by_class: dict[int, list[float]] = {0: [], 1: []}
    for slot in per_file.values():
        by_class[int(slot["label"])].append(slot["correct"] / max(slot["windows"], 1.0))
    background = float(np.mean(by_class[0]))
    drone = float(np.mean(by_class[1]))
    return {
        "files_background": len(by_class[0]),
        "files_drone": len(by_class[1]),
        "file_equal_background_recall": background,
        "file_equal_drone_recall": drone,
        "file_equal_balanced_accuracy": 0.5 * (background + drone),
    }


def frozen_index() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in read_csv_rows(base.INDEX):
        row: dict[str, Any] = dict(raw)
        row["window_index"] = int(row["window_index"])
        row["start"] = int(row["start"])
        row["length"] = int(row["length"])
        row["label"] = int(row["label"])
        row["stage1_label"] = int(row["stage1_label"])
        rows.append(row)
    return rows


def epoch_orders(rows: Sequence[dict[str, Any]], epochs: int, seed: int = 2027) -> list[list[int]]:
    """Recompute the frozen per-epoch shuffle over the full training split."""

    train_rows = [row for row in rows if str(row["source_role"]) == base.TRAIN_ROLE]
    length = len(train_rows)
    orders: list[list[int]] = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(
            length, generator=torch.Generator().manual_seed(int(seed) + epoch)
        ).tolist()
        orders.append([int(value) for value in order])
    return orders


def batch_shapes(order_length: int) -> list[int]:
    sizes = [base.BATCH_SIZE] * (order_length // base.BATCH_SIZE)
    remainder = order_length % base.BATCH_SIZE
    if remainder:
        sizes.append(remainder)
    return sizes


def check_protocol(checks: Checks, candidate: Path, control: Path) -> dict[str, Any]:
    protocol = read_json(candidate / entry.PROTOCOL_NAME)
    config = read_json(candidate / "run_config.json")
    recomputed = entry.protocol_core_sha256(protocol)
    checks.add(
        "V1 the protocol digest recomputes from the protocol content",
        recomputed == protocol.get("protocol_core_sha256")
        and recomputed == config.get("protocol_sha256"),
        {
            "recomputed": recomputed,
            "recorded": protocol.get("protocol_core_sha256"),
            "run_config": config.get("protocol_sha256"),
        },
    )
    candidate_parameters = protocol.get("candidate_parameters", {})
    control_payload = protocol.get("controls", {}).get("control", {})
    checks.add(
        "V2 the protocol names the candidate mode, the frozen rule and the control",
        candidate_parameters.get("mode") == ln.LOSS_MODE_GLOBAL_TRAIN_MEAN
        and candidate_parameters.get("mu_train") is not None
        and protocol["selection_rule"]["rule"] == base.SELECTION_RULE
        and protocol["selection_rule"]["key"] == base.SELECTION_KEY
        and protocol["selection_rule"]["tie_break"] == base.TIE_BREAK
        and protocol["selection_rule"]["prediction"] == base.PREDICTION_RULE
        and str(control_payload.get("run_root")) == str(control),
        {
            "mode": candidate_parameters.get("mode"),
            "control": control_payload.get("run_root"),
            "stopping_conditions": len(protocol.get("stopping_conditions", [])),
        },
    )
    identity = config.get("loss_normalization") or {}
    checks.add(
        "V3 run_config carries the same loss specification as the protocol",
        identity.get("mode") == candidate_parameters.get("mode")
        and identity.get("formula") == candidate_parameters.get("formula")
        and identity.get("denominator_rule") == candidate_parameters.get("denominator_rule")
        and identity.get("tail_batch_rule") == candidate_parameters.get("tail_batch_rule"),
        {"run_config_mode": identity.get("mode")},
    )
    return {"protocol": protocol, "config": config}


def check_preflight(checks: Checks, candidate: Path, control: Path) -> dict[str, Any]:
    recorded = read_json(candidate / "preflight.json")
    identity = base.check_index_identity()
    spec = ln.build_loss_spec(
        ln.LOSS_MODE_GLOBAL_TRAIN_MEAN,
        identity["observed"]["stage1_label_counts"][base.TRAIN_ROLE],
        base.BATCH_SIZE,
    )
    recomputed = entry.preflight_gate(
        seed=2027, epochs=EPOCHS, spec=spec, control=entry.control_reference()
    )
    control_config = read_json(control / "run_config.json")
    # A bounded run hashes the *surviving* order of its selected windows, so only a
    # full run can be compared against the control's five full-split order hashes.
    full_run = str(read_json(candidate / "run_config.json").get("mode")) == "full"
    checks.add(
        "V4 the pre-run gate matches the accepted seed-2027 initialisation and orders",
        recomputed["passed"]
        and recomputed["initial_model_sha256"] == control_config["initial_model_sha256"]
        and recomputed["epoch_order_sha256"]
        == list(control_config["epoch_order_sha256"])[:EPOCHS],
        {
            "initial_matches": recomputed["initial_model_matches_control"],
            "orders_match": recomputed["epoch_order_matches_control"],
            "orders": len(recomputed["epoch_order_sha256"]),
            "control_orders": len(control_config["epoch_order_sha256"]),
        },
    )
    checks.add(
        "V5 the recorded preflight file agrees with the recomputation",
        recorded.get("passed") is True
        and recorded.get("initial_model_sha256") == recomputed["initial_model_sha256"]
        and recorded.get("epoch_order_sha256") == recomputed["epoch_order_sha256"],
        {"recorded": recorded.get("initial_model_sha256"), "recomputed": recomputed["initial_model_sha256"]},
    )
    candidate_config = read_json(candidate / "run_config.json")
    checks.add(
        "V6 the run itself used the frozen initialisation and the frozen orders",
        candidate_config["initial_model_sha256"] == control_config["initial_model_sha256"]
        and candidate_config["class_weights"] == control_config["class_weights"]
        and (
            not full_run
            or candidate_config["epoch_order_sha256"] == control_config["epoch_order_sha256"]
        ),
        {
            "candidate_orders": candidate_config["epoch_order_sha256"],
            "control_orders": control_config["epoch_order_sha256"],
            "full_run": full_run,
        },
    )
    return {"preflight": recomputed, "spec": spec}


def check_objective_steps(
    checks: Checks, candidate: Path, spec: ln.LossSpec
) -> dict[str, Any]:
    rows = read_csv_rows(candidate / entry.OBJECTIVE_STEPS_NAME)
    payload = read_json(candidate / entry.OBJECTIVE_LOSS_NAME)
    expected_rows = EPOCHS * BATCHES_PER_EPOCH
    checks.add(
        "V7 the objective recorded exactly one step per real training batch",
        len(rows) == expected_rows
        and payload["summary"]["steps"] == expected_rows
        and int(payload["expected_steps_this_process"]) == expected_rows
        and int(payload["steps_per_epoch"]) == BATCHES_PER_EPOCH,
        {"rows": len(rows), "expected": expected_rows},
    )
    index_ok = True
    tail_ok = True
    arithmetic_ok = True
    finite_ok = True
    nominal_denominator = float(
        torch.tensor(base.BATCH_SIZE * spec.mu_train, dtype=torch.float32)
    )
    tail_denominator = float(torch.tensor(TAIL_BATCH_SIZE * spec.mu_train, dtype=torch.float32))
    tail_steps: list[dict[str, Any]] = []
    for offset, row in enumerate(rows):
        epoch = 1 + offset // BATCHES_PER_EPOCH
        batch_in_epoch = offset % BATCHES_PER_EPOCH + 1
        size = int(row["batch_size"])
        if int(row["epoch"]) != epoch or int(row["batch_in_epoch"]) != batch_in_epoch:
            index_ok = False
        expected_size = TAIL_BATCH_SIZE if batch_in_epoch == BATCHES_PER_EPOCH else base.BATCH_SIZE
        if size != expected_size:
            tail_ok = False
        denominator = float(row["denominator"])
        expected_denominator = (
            tail_denominator if batch_in_epoch == BATCHES_PER_EPOCH else nominal_denominator
        )
        if abs(denominator - expected_denominator) > 1e-6:
            tail_ok = False
        if batch_in_epoch == BATCHES_PER_EPOCH:
            tail_steps.append(
                {
                    "epoch": epoch,
                    "batch_size": size,
                    "denominator": denominator,
                    "expected": tail_denominator,
                }
            )
        loss = float(row["loss"])
        numerator = float(row["numerator"])
        if not np.isfinite(loss) or not np.isfinite(numerator) or not np.isfinite(denominator):
            finite_ok = False
        if abs(loss * denominator - numerator) > max(1e-3, 1e-3 * abs(numerator)):
            arithmetic_ok = False
    checks.add("V8 step/epoch indices follow the frozen batch layout", index_ok, {"rows": len(rows)})
    checks.add(
        "V9 every epoch ends with the 2-window tail batch and uses 2 * mu_train",
        tail_ok and len(tail_steps) == EPOCHS,
        {"tail_steps": tail_steps, "mu_train": spec.mu_train},
    )
    checks.add(
        "V10 loss == numerator / denominator for every recorded step",
        arithmetic_ok,
        {"tolerance": "1e-3 relative"},
    )
    checks.add("V11 every recorded objective value is finite", finite_ok, None)
    return {"rows": rows, "payload": payload, "tail_steps": tail_steps}


def check_batch_composition(
    checks: Checks, rows: Sequence[dict[str, Any]], spec: ln.LossSpec
) -> dict[str, Any]:
    """How often a four-window batch contains no background window at all."""

    orders = epoch_orders(rows, EPOCHS)
    train_rows = [row for row in rows if str(row["source_role"]) == base.TRAIN_ROLE]
    labels = [int(row["label"]) for row in train_rows]
    per_epoch: list[dict[str, Any]] = []
    for epoch, order in enumerate(orders, start=1):
        sizes = batch_shapes(len(order))
        cursor = 0
        pure_drone = 0
        mixed = 0
        pure_background = 0
        tail_labels: list[int] = []
        for size in sizes:
            batch_labels = [labels[position] for position in order[cursor : cursor + size]]
            if size == TAIL_BATCH_SIZE:
                tail_labels = batch_labels
            cursor += size
            background = sum(1 for value in batch_labels if value == 0)
            if background == 0:
                pure_drone += 1
            elif background == size:
                pure_background += 1
            else:
                mixed += 1
        per_epoch.append(
            {
                "epoch": epoch,
                "batches": len(sizes),
                "batches_without_background": pure_drone,
                "share_without_background": pure_drone / len(sizes),
                "batches_all_background": pure_background,
                "mixed_batches": mixed,
                "tail_batch_labels": tail_labels,
            }
        )
    share = float(np.mean([row["share_without_background"] for row in per_epoch]))
    checks.add(
        "V12 the recorded orders really produce background-free batches of the reported share",
        0.70 <= share <= 0.85 and all(row["tail_batch_labels"] for row in per_epoch),
        {"mean_share_without_background": share, "per_epoch": per_epoch},
    )
    return {"share": share, "per_epoch": per_epoch, "mu_train": spec.mu_train}


def load_validation_scores(candidate: Path, epochs: int) -> dict[int, list[dict[str, Any]]]:
    per_epoch: dict[int, list[dict[str, Any]]] = {}
    for epoch in range(1, epochs + 1):
        path = candidate / f"validation_scores_epoch_{epoch:02d}.csv"
        records: list[dict[str, Any]] = []
        for row in read_csv_rows(path):
            records.append(
                {
                    "condition": str(row["condition"]),
                    "selection_position": int(row["selection_position"]),
                    "index_position": int(row["index_position"]),
                    "filename": str(row["filename"]),
                    "target_code": str(row["target_code"]),
                    "window_index": int(row["window_index"]),
                    "start": int(row["start"]),
                    "label": int(row["label"]),
                    "logit_margin": float(row["logit_margin"]),
                    "prediction": int(row["prediction"]),
                }
            )
        per_epoch[epoch] = records
    return per_epoch


def check_validation(
    checks: Checks,
    candidate: Path,
    rows: Sequence[dict[str, Any]],
    per_epoch: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    val_rows = [row for row in rows if str(row["source_role"]) == base.VAL_ROLE]
    expected_labels = {index: int(row["label"]) for index, row in enumerate(val_rows)}
    expected_files = {index: str(row["filename"]) for index, row in enumerate(val_rows)}
    coverage_ok = True
    keys_ok = True
    labels_ok = True
    for epoch, records in per_epoch.items():
        if len(records) != len(CONDITIONS) * VAL_LENGTH:
            coverage_ok = False
        keys = {(row["condition"], row["index_position"]) for row in records}
        if len(keys) != len(records):
            keys_ok = False
        for row in records:
            index = int(row["index_position"])
            if expected_labels.get(index) != int(row["label"]) or expected_files.get(index) != str(row["filename"]):
                labels_ok = False
    checks.add(
        "V13 every epoch covers the whole validation split for all four conditions",
        coverage_ok,
        {"epochs": len(per_epoch), "rows_per_epoch": len(next(iter(per_epoch.values())))},
    )
    checks.add(
        "V14 (epoch, condition, position) keys are unique", keys_ok, None
    )
    checks.add(
        "V15 every recorded label and filename matches the frozen index",
        labels_ok,
        {"val_windows": len(val_rows)},
    )

    recorded = read_csv_rows(candidate / "validation_metrics.csv")
    deltas: dict[str, Any] = {}
    metrics_ok = True
    for row in recorded:
        epoch = int(row["epoch"])
        condition = str(row["condition"])
        records = [item for item in per_epoch[epoch] if item["condition"] == condition]
        recomputed = {**metrics_from_records(records), **file_equal_from_records(records)}
        deltas[f"{epoch}/{condition}"] = {
            "counts_match": all(
                int(row[key]) == int(recomputed[key]) for key in ("windows", "tn", "fp", "fn", "tp")
            ),
            "max_float_delta": max(
                abs(float(row[key]) - float(recomputed[key]))
                for key in (
                    "balanced_accuracy",
                    "accuracy",
                    "drone_recall",
                    "background_specificity",
                    "f1",
                    "file_equal_balanced_accuracy",
                )
            ),
        }
        metrics_ok = metrics_ok and deltas[f"{epoch}/{condition}"]["counts_match"]
        metrics_ok = metrics_ok and deltas[f"{epoch}/{condition}"]["max_float_delta"] <= 1e-12
    checks.add(
        "V16 the recorded validation metrics re-derive from the per-window decisions",
        metrics_ok,
        deltas,
    )

    selection = read_json(candidate / "selection.json")
    config = read_json(candidate / "run_config.json")
    mean_by_epoch = {
        epoch: float(
            np.mean(
                [
                    float(row["balanced_accuracy"])
                    for row in recorded
                    if int(row["epoch"]) == epoch
                ]
            )
        )
        for epoch in range(1, EPOCHS + 1)
    }
    best = max(mean_by_epoch.items(), key=lambda item: (item[1], -item[0]))
    checks.add(
        "V17 the frozen selection rule picks the recorded epoch",
        int(selection["selected_epoch"]) == best[0]
        and int(config["selected_epoch"]) == best[0]
        and abs(float(selection["selected_epoch_score"]) - best[1]) <= 1e-12
        and abs(float(config["selected_epoch_score"]) - best[1]) <= 1e-12,
        {"recomputed": {"epoch": best[0], "score": best[1]}, "recorded": selection},
    )
    checkpoint = torch.load(candidate / "best_model.pt", map_location="cpu", weights_only=False)
    checks.add(
        "V18 the retained checkpoint is the selected epoch with matching metrics",
        int(checkpoint["epoch"]) == best[0]
        and checkpoint.get("selection_rule") == base.SELECTION_RULE
        and checkpoint.get("selection_key") == base.SELECTION_KEY
        and checkpoint.get("prediction_rule") == base.PREDICTION_RULE
        and checkpoint.get("training_seed") == 2027
        and checkpoint.get("index_sha256") == base.INDEX_SHA256
        and checkpoint.get("split_source_sha256") == base.FROZEN_SPLIT_CSV_SHA256,
        {"epoch": int(checkpoint["epoch"]), "selected": best[0]},
    )
    metrics_by_condition = {
        str(row["condition"]): row
        for row in checkpoint["val_metrics"]
    }
    checkpoint_ok = True
    for condition in CONDITIONS:
        recomputed = metrics_from_records(
            [item for item in per_epoch[best[0]] if item["condition"] == condition]
        )
        stored = metrics_by_condition[condition]
        for key in ("windows", "tn", "fp", "fn", "tp"):
            checkpoint_ok = checkpoint_ok and int(stored[key]) == int(recomputed[key])
        checkpoint_ok = checkpoint_ok and abs(
            float(stored["balanced_accuracy"]) - recomputed["balanced_accuracy"]
        ) <= 1e-12
    checks.add(
        "V19 the checkpoint's validation metrics match the per-window recomputation",
        checkpoint_ok,
        {"selected_epoch": best[0]},
    )
    return {
        "mean_balanced_accuracy_by_epoch": mean_by_epoch,
        "selected_epoch": best[0],
        "selected_epoch_score": best[1],
        "metrics_deltas": deltas,
    }


def check_read_audit(checks: Checks, candidate: Path) -> dict[str, Any]:
    audit = read_json(candidate / "read_audit.json")
    loader_audit = read_json(candidate / "loader_read_audit.json")
    expected_train = EPOCHS * SPLIT_LENGTH
    # The frozen protocol builds the validation feature cache ONCE per run and reuses
    # it for every epoch, so the validation budget is one cache, not one per epoch.
    expected_val = len(CONDITIONS) * VAL_LENGTH
    expected_total = expected_train + expected_val
    checks.add(
        "V20 the read budget is exactly one pass per epoch plus one validation cache",
        int(audit["windows_read"]) == expected_total
        and int(loader_audit["expected_train_windows"]) == expected_train
        and int(loader_audit["resumed_epochs"]) == 0,
        {
            "windows_read": audit["windows_read"],
            "expected": expected_total,
            "expected_train_in_audit": loader_audit["expected_train_windows"],
        },
    )
    keys = loader_audit["train"]["keys"]
    per_key_ok = set(keys) == {str(epoch) for epoch in range(1, EPOCHS + 1)} and all(
        int(keys[key]["reads"]["windows_read"]) == SPLIT_LENGTH
        and int(keys[key]["consumed"]["windows_completed"]) == SPLIT_LENGTH
        and int(keys[key]["unconsumed_reads"]["count"]) == 0
        and int(keys[key]["unread_consumed"]["count"]) == 0
        for key in keys
    )
    val_ok = set(loader_audit["val_cache"]["keys"]) == set(CONDITIONS) and all(
        int(loader_audit["val_cache"]["keys"][condition]["reads"]["windows_read"]) == VAL_LENGTH
        for condition in CONDITIONS
    )
    checks.add(
        "V21 every epoch read and consumed exactly the training split, with no leftovers",
        per_key_ok and val_ok,
        {
            "train_keys": sorted(keys),
            "val_keys": sorted(loader_audit["val_cache"]["keys"]),
        },
    )
    checks.add(
        "V22 no development_test* source file appears in any audit",
        not audit["test_files_touched"]
        and not loader_audit["test_files_touched"]
        and audit["passed"] is True
        and loader_audit["passed"] is True
        and audit["opened_all_train_val"] is True,
        {
            "read_audit_opened_files": audit["opened_file_count"],
            "unexpected": loader_audit["unexpected_files_touched"],
        },
    )
    consumption = read_csv_rows(candidate / "loader_consumption.csv")
    train_rows = [row for row in consumption if str(row["phase"]) == base.TRAIN_ROLE]
    checks.add(
        "V23 the consumption log has one completed row per delivered window",
        len(consumption) == expected_total
        and len(train_rows) == expected_train
        and all(str(row["completed"]) == "1" for row in consumption),
        {
            "rows": len(consumption),
            "expected_total": expected_total,
            "train_rows": len(train_rows),
            "expected_train": expected_train,
        },
    )
    return {
        "audit": audit,
        "loader": loader_audit,
        "consumption_rows": len(consumption),
        "consumption_train_rows": len(train_rows),
    }


def check_state(checks: Checks, candidate: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    state_path = candidate / entry.STATE_NAME
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    identity = state.get("identity") or {}
    config = read_json(candidate / "run_config.json")
    checks.add(
        "V24 the epoch-boundary state ends at the last epoch and carries no RNG-free gaps",
        state.get("schema") == base.TRAINING_STATE_SCHEMA
        and int(state["epoch"]) == EPOCHS
        and int(state["next_epoch"]) == EPOCHS + 1
        and len(state["history"]) == EPOCHS
        and state["initial_model_sha256"] == config["initial_model_sha256"]
        and isinstance(state.get("rng"), dict)
        and state["rng"].get("torch_cpu") is not None
        and len(state["order_hashes"]) == EPOCHS,
        {
            "epoch": state.get("epoch"),
            "history": len(state.get("history", [])),
            "rng_keys": sorted((state.get("rng") or {}).keys()),
        },
    )
    checks.add(
        "V25 the state identity matches the index, the split, the seeds and the protocol",
        identity.get("index_sha256") == base.INDEX_SHA256
        and identity.get("split_source_sha256") == base.FROZEN_SPLIT_CSV_SHA256
        and identity.get("training_seed") == 2027
        and identity.get("validation_seed") == base.VALIDATION_SEED
        and identity.get("epochs") == EPOCHS
        and identity.get("loss_mode") == ln.LOSS_MODE_GLOBAL_TRAIN_MEAN
        and identity.get("protocol_sha256") == protocol.get("protocol_core_sha256"),
        identity,
    )
    restored_hash = base.state_hash(state["model_state"])
    checkpoint = torch.load(candidate / "best_model.pt", map_location="cpu", weights_only=False)
    if checkpoint["epoch"] == EPOCHS:
        checks.add(
            "V26 the final state's weights equal the retained checkpoint",
            restored_hash == base.state_hash(checkpoint["model_state"]),
            {"state": restored_hash},
        )
    else:
        checks.add(
            "V26 the final state's weights differ from an earlier selected checkpoint",
            restored_hash != base.state_hash(checkpoint["model_state"]),
            {"state": restored_hash, "selected_epoch": int(checkpoint["epoch"])},
        )
    resume_json = candidate / entry.STATE_JSON_NAME
    summary = read_json(resume_json)
    checks.add(
        "V27 the state journal records one write per epoch and no resume",
        len(summary["writes"]) == EPOCHS
        and [int(row["epoch"]) for row in summary["writes"]] == list(range(1, EPOCHS + 1))
        and int(summary["latest"]["epoch"]) == EPOCHS
        and not (candidate / entry.RESUME_EVENTS_NAME).exists(),
        {"writes": [row["epoch"] for row in summary["writes"]]},
    )
    return {"state": summary, "state_epoch": int(state["epoch"])}


def check_resume_evidence(checks: Checks) -> dict[str, Any]:
    path = entry.OUTPUT_ROOT / "resume_check" / "resume_check.json"
    if not path.is_file():
        checks.add("V28 the resume determinism check has a report", False, {"path": str(path)})
        return {}
    payload = read_json(path)
    comparisons = payload.get("comparison", {}).get("checks", [])
    checks.add(
        "V28 the epoch-boundary interrupt/resume check passed every comparison",
        bool(payload.get("passed"))
        and len(comparisons) >= 15
        and payload.get("interrupted_run", {}).get("killed_at_state", {}).get("epoch") == 1,
        {
            "passed": payload.get("passed"),
            "checks": len(comparisons),
            "failed": [row["check"] for row in payload.get("comparison", {}).get("failed", [])],
            "interrupted_exit_code": payload.get("interrupted_run", {}).get("interrupted_exit_code"),
        },
    )
    return payload


def check_paired_summary(checks: Checks) -> dict[str, Any]:
    path = entry.OUTPUT_ROOT / "comparison" / "paired_comparison.json"
    if not path.is_file():
        checks.add("V29 the paired comparison report exists", False, {"path": str(path)})
        return {}
    payload = read_json(path)
    primary = payload["primary"]
    recomputed = (
        primary["candidate"]["mean_pooled_balanced_accuracy"]
        - primary["control"]["mean_pooled_balanced_accuracy"]
    )
    checks.add(
        "V29 the paired comparison's delta equals candidate minus control",
        abs(float(primary["delta_candidate_minus_control"]) - recomputed) <= 1e-12,
        {"delta": primary["delta_candidate_minus_control"]},
    )
    return payload


def negative_controls(
    checks: Checks,
    per_epoch: dict[int, list[dict[str, Any]]],
    step_rows: Sequence[dict[str, str]],
) -> dict[str, Any]:
    """In-memory tampering that the recomputation above must reject."""

    detected: list[str] = []
    missed: list[str] = []

    def expect(name: str, condition: bool) -> None:
        (detected if condition else missed).append(name)

    records = [dict(row) for row in per_epoch[1] if row["condition"] == "clean"]
    baseline = metrics_from_records(records)
    tampered = [dict(row) for row in records]
    flipped = 0
    for row in tampered:
        if flipped < 5 and int(row["prediction"]) == int(row["label"]):
            row["prediction"] = 1 - int(row["prediction"])
            flipped += 1
    expect("a flipped per-window decision changes the recomputed confusion", metrics_from_records(tampered) != baseline)

    relabelled = [dict(row) for row in records]
    relabelled[0]["label"] = 1 - int(relabelled[0]["label"])
    expect("a changed window label changes the recomputed confusion", metrics_from_records(relabelled) != baseline)

    renamed = [dict(row) for row in records]
    renamed[0]["filename"] = "T9999_D00_S0000.mat"
    expect("a renamed source file changes the file-equal metric", file_equal_from_records(renamed) != file_equal_from_records(records))

    steps = [dict(row) for row in step_rows]
    steps[-1]["denominator"] = str(float(steps[-1]["denominator"]) * 2.0)
    tail_ok = abs(float(steps[-1]["denominator"]) - 2.0) <= 1e-6
    expect("a doubled tail-batch denominator is rejected by the tail rule", not tail_ok)

    broken = [dict(row) for row in step_rows]
    broken[0]["loss"] = str(float(broken[0]["loss"]) * 3.0)
    arithmetic_ok = abs(
        float(broken[0]["loss"]) * float(broken[0]["denominator"]) - float(broken[0]["numerator"])
    ) <= max(1e-3, 1e-3 * abs(float(broken[0]["numerator"])))
    expect("a scaled step loss is rejected by loss == numerator / denominator", not arithmetic_ok)
    checks.add(
        "V30 every negative control is detected",
        not missed,
        {"detected": detected, "missed": missed},
    )
    return {"detected": detected, "missed": missed}


def check_validation_cache_identity(
    checks: Checks, candidate: Path, control: Path
) -> dict[str, Any]:
    """The two runs must be scored on byte-identical validation inputs."""

    candidate_manifest = {
        str(row["condition"]): row
        for row in read_json(candidate / "validation_cache_manifest.json")
    }
    control_manifest = {
        str(row["condition"]): row
        for row in read_json(control / "validation_cache_manifest.json")
    }
    detail: dict[str, Any] = {}
    identical = set(candidate_manifest) == set(control_manifest)
    for condition in sorted(candidate_manifest):
        left = candidate_manifest[condition]
        right = control_manifest.get(condition, {})
        same = (
            left["feature_sha256"] == right.get("feature_sha256")
            and left["label_sha256"] == right.get("label_sha256")
            and list(left["index_positions"]) == list(right.get("index_positions", []))
        )
        identical = identical and same
        detail[condition] = {
            "feature_sha256": left["feature_sha256"],
            "control_feature_sha256": right.get("feature_sha256"),
            "identical": same,
        }
    checks.add(
        "V31 the validation cache is byte-identical to the control's",
        identical,
        detail,
    )
    return detail


def baseline_snapshot() -> dict[str, Any]:
    """Hash every file of the frozen inputs and of the accepted control runs.

    Run before and after the candidate work to show the controls were read-only:
    the loss-normalisation run writes only inside its own output root.
    """

    inputs = {
        "index": base.INDEX,
        "frozen_split_csv": base.FROZEN_SPLIT_CSV,
        "freeze_decision": base.FREEZE_JSON,
        "g_manifest": base.G_MANIFEST,
    }
    runs = {"control_seed2027": base.OUTPUT_ROOT / "seed_2027", "context_seed2026": base.OUTPUT_ROOT / "seed_2026"}
    payload: dict[str, Any] = {
        "generated_at": utc_now(),
        "inputs": {name: {"path": str(path), "sha256": base.sha256_file(path)} for name, path in inputs.items()},
        "runs": {},
    }
    for label, root in runs.items():
        files = {
            path.name: {
                "sha256": base.sha256_file(path),
                "bytes": path.stat().st_size,
                "mtime": int(path.stat().st_mtime),
            }
            for path in sorted(root.glob("*"))
            if path.is_file()
        }
        payload["runs"][label] = {"run_root": str(root), "files": files}
    return payload


def verify(
    candidate: Path,
    control: Path,
    output: Path,
    *,
    skip_preflight: bool = False,
) -> dict[str, Any]:
    global EPOCHS
    checks = Checks()
    # The run declares its own epoch count; a bounded smoke run is verified with the
    # same code path instead of being special-cased.  The full protocol is still
    # asserted to be ``base.EPOCHS`` below.
    EPOCHS = int(read_json(candidate / "run_config.json").get("epochs") or base.EPOCHS)
    rows = frozen_index()
    checks.add(
        "V0 the frozen index is intact",
        base.sha256_file(base.INDEX) == base.INDEX_SHA256 and len(rows) == base.INDEX_ROWS,
        {"rows": len(rows)},
    )
    protocol_pack = check_protocol(checks, candidate, control)
    protocol = protocol_pack["protocol"]
    config = protocol_pack["config"]
    checks.add(
        "V0b the run reported a completed full-protocol execution",
        config.get("status") == "completed"
        and config.get("mode") == "full"
        and int(config.get("epochs")) == base.EPOCHS
        and int(config.get("batch_size")) == base.BATCH_SIZE
        and float(config.get("learning_rate")) == base.LEARNING_RATE
        and float(config.get("weight_decay")) == base.WEIGHT_DECAY
        and config.get("train_snrs") == base.TRAIN_SNRS
        and config.get("validation_conditions") == list(CONDITIONS)
        and config.get("max_train_samples") is None
        and config.get("max_val_samples") is None
        and config.get("loader", {}).get("num_workers") == 0
        and "resume" not in config
        and config.get("index_sha256") == base.INDEX_SHA256
        and config.get("split_source_sha256") == base.FROZEN_SPLIT_CSV_SHA256,
        {
            "status": config.get("status"),
            "loader": config.get("loader"),
            "torch_threads": config.get("torch_threads"),
        },
    )
    if skip_preflight:
        spec = ln.build_loss_spec(
            ln.LOSS_MODE_GLOBAL_TRAIN_MEAN, {0: 600, 1: 8990}, base.BATCH_SIZE
        )
        preflight_pack = {"preflight": {}, "spec": spec}
    else:
        preflight_pack = check_preflight(checks, candidate, control)
    spec = preflight_pack["spec"]
    steps_pack = check_objective_steps(checks, candidate, spec)
    composition = check_batch_composition(checks, rows, spec)
    per_epoch = load_validation_scores(candidate, EPOCHS)
    validation = check_validation(checks, candidate, rows, per_epoch)
    audit = check_read_audit(checks, candidate)
    cache_identity = check_validation_cache_identity(checks, candidate, control)
    state = check_state(checks, candidate, protocol)
    resume = check_resume_evidence(checks)
    paired = check_paired_summary(checks)
    controls = negative_controls(checks, per_epoch, steps_pack["rows"])

    result = {
        "schema": "dronerfa-stage1-loss-norm-verification/v1",
        "generated_at": utc_now(),
        "run_root": str(candidate),
        "control_root": str(control),
        "loss_mode": ln.LOSS_MODE_GLOBAL_TRAIN_MEAN,
        "loss_spec": spec.payload(),
        "protocol_sha256": protocol.get("protocol_core_sha256"),
        "preflight": preflight_pack["preflight"],
        "objective_steps": {
            "rows": len(steps_pack["rows"]),
            "tail_steps": steps_pack["tail_steps"],
            "summary": steps_pack["payload"]["summary"],
        },
        "batch_composition": composition,
        "validation": validation,
        "read_audit": {
            "windows_read": audit["audit"]["windows_read"],
            "opened_files": audit["audit"]["opened_file_count"],
            "consumption_rows": audit["consumption_rows"],
        },
        "validation_cache_identity": cache_identity,
        "state": state,
        "resume_evidence": resume,
        "paired_comparison": paired,
        "negative_controls": controls,
        "verification": checks.payload(),
        "passed": checks.passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "passed_checks": result["verification"]["passed_checks"],
                "total_checks": result["verification"]["total_checks"],
                "failed": [row["check"] for row in result["verification"]["failed_checks"]],
                "negative_controls": f"{len(controls['detected'])}/{len(controls['detected']) + len(controls['missed'])}",
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="verification report path (default: <candidate>/loss_normalization_verification.json)",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="write a read-only hash snapshot of the frozen inputs and control runs, then exit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.snapshot is not None:
        payload = baseline_snapshot()
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"snapshot": str(args.snapshot), "runs": sorted(payload["runs"])}, indent=2))
        return 0
    output = args.output or (args.candidate / "loss_normalization_verification.json")
    result = verify(args.candidate, args.control, output, skip_preflight=args.skip_preflight)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
