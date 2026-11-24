#!/usr/bin/env python3
"""STAGE1-12DRONES-B: read-only comparison of two stage-1 training seeds.

The ``val`` split is drawn from the same ``development``/``val`` source files under
the same four frozen validation conditions and the *same* validation noise seed for
every training seed, so two runs are paired window for window.  This script never
touches the G: drive: it reads only the artefacts the driver wrote, recomputes the
metrics from the recorded per-window margins, and reports how much of the metric
difference is a disagreement about *which* windows are wrong.

Outputs are descriptive development-validation diagnostics of one frozen split.
They are not an independent test, not a 12-class identification rate, and a
per-condition oracle threshold is explicitly out of scope -- the frozen decision
rule stays ``float32(logit_drone) - float32(logit_background) > 0``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("WINDIR", r"C:\Windows")

import numpy as np

# ---------------------------------------------------------------------------
# frozen context (independent copies, not imported from the training driver)
# ---------------------------------------------------------------------------
OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "stage1_12drones_20260926"
INDEX = OUTPUT_ROOT / "window_index_stage1_12drones.csv"
INDEX_SHA256 = "5973a2fa5b8476a507fcc7a665365daf8d572f569ce5908d54f5c4bc84c48bd6"
INDEX_ROWS = 15790

ACCEPTED_SEED = 2026
CANDIDATE_SEED = 2027
VALIDATION_SEED = 12026
TRAIN_ROLE = "train"
VAL_ROLE = "val"
TEST_ROLES = ("development_test", "development_test_exposed")
BACKGROUND_CODE = "T0000"
CONDITIONS = ("clean", "-10dB", "-12dB", "-15dB")
CONDITION_ORDER = {name: position for position, name in enumerate(CONDITIONS)}
SELECTION_KEY = "mean_balanced_accuracy"
N_FFT = 512
HOP_LENGTH = 256
IMAGE_SIZE = 256
EXPECTED_VAL_FILES = 16
EXPECTED_VAL_WINDOWS = 2360
EXPECTED_EPOCHS = 5
EXPECTED_FLOAT_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run_root_for(seed: int) -> Path:
    return OUTPUT_ROOT / f"seed_{int(seed)}"


def pooled_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    prediction = np.asarray(predictions, dtype=np.int64)
    if len(y) != len(prediction):
        raise ValueError("labels and predictions must have equal length")
    tp = int(np.sum((y == 1) & (prediction == 1)))
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    if tp + tn + fp + fn != len(y):
        raise ValueError("confusion matrix does not conserve the windows")
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "windows": int(len(y)),
        "accuracy": (tp + tn) / max(len(y), 1),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "drone_recall": recall,
        "background_specificity": specificity,
        "precision": precision,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def auc_pairwise(drone_margins: np.ndarray, background_margins: np.ndarray) -> float:
    """Threshold-free rank statistic, counted directly over drone/background pairs.

    ``P(drone margin > background margin)`` with exact ties counted as one half --
    the same definition ``train_resnet18.binary_metrics`` uses.  The window counts
    here are 80 drone x 2,280 background per condition, so the 2-D comparison is
    cheap enough to avoid the rank-sum identity (whose orientation is easy to get
    wrong: the rank-sum form ``U`` counts *background above drone*).
    """
    drone = np.asarray(drone_margins, dtype=np.float64).ravel()
    background = np.asarray(background_margins, dtype=np.float64).ravel()
    if drone.size == 0 or background.size == 0:
        return float("nan")
    above = float(np.count_nonzero(drone[:, None] > background[None, :]))
    equal = float(np.count_nonzero(drone[:, None] == background[None, :]))
    return (above + 0.5 * equal) / (drone.size * background.size)


def file_equal_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    per_file: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = per_file.setdefault(
            str(row["filename"]), {"label": int(row["label"]), "correct": 0, "windows": 0}
        )
        if entry["label"] != int(row["label"]):
            raise ValueError(f"source file {row['filename']} carries two classes")
        entry["correct"] += int(int(row["prediction"]) == int(row["label"]))
        entry["windows"] += 1
    class_rates: dict[int, list[float]] = {0: [], 1: []}
    for entry in per_file.values():
        class_rates[entry["label"]].append(entry["correct"] / max(entry["windows"], 1))
    if not class_rates[0] or not class_rates[1]:
        raise ValueError("file-equal metrics need both classes")
    background = float(np.mean(class_rates[0]))
    drone = float(np.mean(class_rates[1]))
    return {
        "files_background": len(class_rates[0]),
        "files_drone": len(class_rates[1]),
        "file_equal_background_recall": background,
        "file_equal_drone_recall": drone,
        "file_equal_balanced_accuracy": 0.5 * (background + drone),
    }


def select_epoch(history: Sequence[dict[str, Any]], key: str = SELECTION_KEY) -> int:
    """Highest mean validation score; the earliest epoch wins an exact tie."""
    return int(max(history, key=lambda row: (row[key], -int(row["epoch"])))["epoch"])


# ---------------------------------------------------------------------------
# per-run loading + recomputation
# ---------------------------------------------------------------------------
def load_score_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in read_csv_rows(path):
        rows.append(
            {
                "condition": str(raw["condition"]),
                "index_position": int(raw["index_position"]),
                "filename": str(raw["filename"]),
                "target_code": str(raw["target_code"]),
                "window_index": int(raw["window_index"]),
                "start": int(raw["start"]),
                "label": int(raw["label"]),
                "margin": float(raw["logit_margin"]),
                "prediction": int(raw["prediction"]),
            }
        )
    return rows


def recompute_run(seed: int) -> dict[str, Any]:
    run_root = run_root_for(seed)
    for required in ("run_config.json", "summary.json", "selection.json", "progress.json"):
        if not (run_root / required).is_file():
            raise FileNotFoundError(f"seed {seed}: missing {run_root / required}")
    config = read_json(run_root / "run_config.json")
    summary = read_json(run_root / "summary.json")
    selection = read_json(run_root / "selection.json")
    progress = read_json(run_root / "progress.json")
    history = read_csv_rows(run_root / "training_history.csv")

    epochs = int(config["epochs"])
    if epochs != EXPECTED_EPOCHS:
        raise RuntimeError(f"seed {seed}: {epochs} epochs, expected {EXPECTED_EPOCHS}")
    if int(progress["epochs_completed"]) != epochs:
        raise RuntimeError(f"seed {seed}: progress reports {progress['epochs_completed']} epochs")
    if len(history) != epochs:
        raise RuntimeError(f"seed {seed}: training_history.csv has {len(history)} rows")
    if int(config["training_seed"]) != seed or int(summary["training_seed"]) != seed:
        raise RuntimeError(f"seed {seed}: config/summary record another training seed")
    if int(config["validation_seed"]) != VALIDATION_SEED:
        raise RuntimeError(
            f"seed {seed}: validation seed {config['validation_seed']} != {VALIDATION_SEED}"
        )

    epoch_records: dict[int, list[dict[str, Any]]] = {}
    epoch_metrics: dict[int, dict[str, dict[str, Any]]] = {}
    for epoch in range(1, epochs + 1):
        path = run_root / f"validation_scores_epoch_{epoch:02d}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"seed {seed}: missing per-window scores {path}")
        rows = load_score_file(path)
        epoch_records[epoch] = rows
        per_condition: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            selected = [row for row in rows if row["condition"] == condition]
            if len(selected) != EXPECTED_VAL_WINDOWS:
                raise RuntimeError(
                    f"seed {seed} epoch {epoch} {condition}: {len(selected)} windows, "
                    f"expected {EXPECTED_VAL_WINDOWS}"
                )
            positions = sorted(row["index_position"] for row in selected)
            if positions != list(range(EXPECTED_VAL_WINDOWS)):
                raise RuntimeError(
                    f"seed {seed} epoch {epoch} {condition}: index positions are not 0..N-1"
                )
            labels = [row["label"] for row in selected]
            if labels != [row["label"] for row in rows if row["condition"] == condition]:
                raise RuntimeError(f"seed {seed} epoch {epoch} {condition}: label order changed")
            predictions = [row["prediction"] for row in selected]
            recomputed = {
                **pooled_metrics(labels, predictions),
                **file_equal_metrics(selected),
                "margin_auc": auc_pairwise(
                    np.asarray(
                        [row["margin"] for row in selected if row["label"] == 1], dtype=np.float64
                    ),
                    np.asarray(
                        [row["margin"] for row in selected if row["label"] == 0], dtype=np.float64
                    ),
                ),
            }
            per_condition[condition] = recomputed
        epoch_metrics[epoch] = per_condition

    # Cross-check the recomputation against the driver's own recorded metrics.
    recorded_csv = read_csv_rows(run_root / "validation_metrics.csv")
    deltas: list[dict[str, Any]] = []
    for row in recorded_csv:
        epoch = int(row["epoch"])
        condition = str(row["condition"])
        recomputed = epoch_metrics[epoch][condition]
        for key in ("balanced_accuracy", "file_equal_balanced_accuracy", "accuracy"):
            delta = abs(float(row[key]) - float(recomputed[key]))
            deltas.append(
                {
                    "epoch": epoch,
                    "condition": condition,
                    "metric": key,
                    "recorded": float(row[key]),
                    "recomputed": float(recomputed[key]),
                    "abs_delta": delta,
                }
            )
        for key in ("tn", "fp", "fn", "tp", "windows"):
            if int(row[key]) != int(recomputed[key]):
                raise RuntimeError(
                    f"seed {seed} epoch {epoch} {condition}: {key} "
                    f"{row[key]} != {recomputed[key]}"
                )
    max_delta = max(row["abs_delta"] for row in deltas) if deltas else 0.0
    if max_delta > EXPECTED_FLOAT_TOLERANCE:
        raise RuntimeError(
            f"seed {seed}: recorded metrics and recomputation differ by {max_delta}"
        )

    typed_history = [
        {
            "epoch": int(row["epoch"]),
            "train_loss": float(row["train_loss"]),
            "train_balanced_accuracy": float(row["train_balanced_accuracy"]),
            SELECTION_KEY: float(row[SELECTION_KEY]),
            "mean_file_equal_balanced_accuracy": float(row["mean_file_equal_balanced_accuracy"]),
            "seconds": float(row["seconds"]),
        }
        for row in history
    ]
    expected_selected = select_epoch(typed_history)
    if expected_selected != int(selection["selected_epoch"]):
        raise RuntimeError(
            f"seed {seed}: selection.json says {selection['selected_epoch']}, rule gives "
            f"{expected_selected}"
        )
    if expected_selected != int(config["selected_epoch"]):
        raise RuntimeError(f"seed {seed}: run_config selected_epoch disagrees")
    return {
        "seed": seed,
        "run_root": run_root,
        "phase": config.get("phase"),
        "history": typed_history,
        "selected_epoch": expected_selected,
        "selected_epoch_recorded": int(selection["selected_epoch"]),
        "epoch_metrics": epoch_metrics,
        "epoch_records": epoch_records,
        "max_recorded_metric_delta": max_delta,
        "config": config,
        "summary": summary,
        "selection": selection,
        "progress": progress,
    }


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def decode(rows: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Decode one condition's per-window rows into aligned arrays."""
    ordered = sorted(rows, key=lambda row: (row["index_position"],))
    return {
        "index_position": np.asarray([row["index_position"] for row in ordered], dtype=np.int64),
        "label": np.asarray([row["label"] for row in ordered], dtype=np.int64),
        "prediction": np.asarray([row["prediction"] for row in ordered], dtype=np.int64),
        "margin": np.asarray([row["margin"] for row in ordered], dtype=np.float64),
        "filename": np.asarray([row["filename"] for row in ordered], dtype=object),
        "target_code": np.asarray([row["target_code"] for row in ordered], dtype=object),
        "window_index": np.asarray([row["window_index"] for row in ordered], dtype=np.int64),
    }


def condition_rows(run: dict[str, Any], epoch: int, condition: str) -> list[dict[str, Any]]:
    return [row for row in run["epoch_records"][epoch] if row["condition"] == condition]


def compare_condition(
    epoch: int,
    condition: str,
    accepted: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    left = decode(condition_rows(accepted, epoch, condition))
    right = decode(condition_rows(candidate, epoch, condition))
    for key in ("index_position", "label", "filename", "target_code", "window_index"):
        if not np.array_equal(left[key], right[key]):
            raise RuntimeError(
                f"{condition}: the two seeds do not share the same validation windows ({key})"
            )
    if np.array_equal(left["prediction"], right["prediction"]):
        disagreements = np.zeros(len(left["label"]), dtype=bool)
    else:
        disagreements = left["prediction"] != right["prediction"]
    both_wrong = (left["prediction"] != left["label"]) & (right["prediction"] != right["label"])
    accepted_only_wrong = (left["prediction"] != left["label"]) & (right["prediction"] == right["label"])
    candidate_only_wrong = (left["prediction"] == left["label"]) & (right["prediction"] != right["label"])
    margin_delta = right["margin"] - left["margin"]
    per_file: dict[str, dict[str, Any]] = {}
    for index, filename in enumerate(left["filename"]):
        entry = per_file.setdefault(
            str(filename),
            {
                "windows": 0,
                "accepted_correct": 0,
                "candidate_correct": 0,
                "disagreements": 0,
                "margin_delta_sum": 0.0,
                "label": int(left["label"][index]),
                "target_code": str(left["target_code"][index]),
            },
        )
        entry["windows"] += 1
        entry["accepted_correct"] += int(left["prediction"][index] == left["label"][index])
        entry["candidate_correct"] += int(right["prediction"][index] == right["label"][index])
        entry["disagreements"] += int(disagreements[index])
        entry["margin_delta_sum"] += float(margin_delta[index])
    for entry in per_file.values():
        entry["accepted_accuracy"] = entry["accepted_correct"] / max(entry["windows"], 1)
        entry["candidate_accuracy"] = entry["candidate_correct"] / max(entry["windows"], 1)
        entry["accuracy_delta"] = entry["candidate_accuracy"] - entry["accepted_accuracy"]
        entry["mean_margin_delta"] = entry["margin_delta_sum"] / max(entry["windows"], 1)
        del entry["margin_delta_sum"]
    accepted_metrics = accepted["epoch_metrics"][epoch][condition]
    candidate_metrics = candidate["epoch_metrics"][epoch][condition]
    return {
        "condition": condition,
        "windows": int(len(left["label"])),
        "accepted": accepted_metrics,
        "candidate": candidate_metrics,
        "delta": {
            "balanced_accuracy": candidate_metrics["balanced_accuracy"]
            - accepted_metrics["balanced_accuracy"],
            "file_equal_balanced_accuracy": candidate_metrics["file_equal_balanced_accuracy"]
            - accepted_metrics["file_equal_balanced_accuracy"],
            "accuracy": candidate_metrics["accuracy"] - accepted_metrics["accuracy"],
            "margin_auc": candidate_metrics["margin_auc"] - accepted_metrics["margin_auc"],
        },
        "disagreement": {
            "windows": int(disagreements.sum()),
            "fraction": float(disagreements.mean()),
            "both_wrong": int(both_wrong.sum()),
            "accepted_only_wrong": int(accepted_only_wrong.sum()),
            "candidate_only_wrong": int(candidate_only_wrong.sum()),
            "both_correct": int((~disagreements & (left["prediction"] == left["label"])).sum()),
        },
        "margin_delta": {
            "mean": float(margin_delta.mean()),
            "mean_abs": float(np.abs(margin_delta).mean()),
            "max_abs": float(np.abs(margin_delta).max()),
            "min": float(margin_delta.min()),
            "max": float(margin_delta.max()),
        },
        "per_file": {name: per_file[name] for name in sorted(per_file)},
    }


def compare_runs(accepted: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    seeds = [accepted["seed"], candidate["seed"]]
    history = []
    for epoch in range(1, EXPECTED_EPOCHS + 1):
        left = next(row for row in accepted["history"] if row["epoch"] == epoch)
        right = next(row for row in candidate["history"] if row["epoch"] == epoch)
        history.append(
            {
                "epoch": epoch,
                "accepted": left,
                "candidate": right,
                "delta": {
                    key: right[key] - left[key]
                    for key in (SELECTION_KEY, "mean_file_equal_balanced_accuracy", "train_loss")
                },
            }
        )
    epoch_table = []
    for epoch in range(1, EXPECTED_EPOCHS + 1):
        row: dict[str, Any] = {"epoch": epoch, "selected_by": []}
        if epoch == accepted["selected_epoch"]:
            row["selected_by"].append(accepted["seed"])
        if epoch == candidate["selected_epoch"]:
            row["selected_by"].append(candidate["seed"])
        for condition in CONDITIONS:
            left = accepted["epoch_metrics"][epoch][condition]
            right = candidate["epoch_metrics"][epoch][condition]
            row[f"{condition}_accepted"] = left["balanced_accuracy"]
            row[f"{condition}_candidate"] = right["balanced_accuracy"]
            row[f"{condition}_file_equal_accepted"] = left["file_equal_balanced_accuracy"]
            row[f"{condition}_file_equal_candidate"] = right["file_equal_balanced_accuracy"]
        row["mean_accepted"] = next(
            item[SELECTION_KEY] for item in accepted["history"] if item["epoch"] == epoch
        )
        row["mean_candidate"] = next(
            item[SELECTION_KEY] for item in candidate["history"] if item["epoch"] == epoch
        )
        row["mean_delta"] = row["mean_candidate"] - row["mean_accepted"]
        epoch_table.append(row)

    selected_conditions = {
        condition: compare_condition(
            accepted["selected_epoch"], condition, accepted, candidate
        )
        for condition in CONDITIONS
    }
    selected_conditions_candidate = {
        condition: compare_condition(
            candidate["selected_epoch"], condition, accepted, candidate
        )
        for condition in CONDITIONS
    }
    common_epoch = accepted["selected_epoch"] == candidate["selected_epoch"]
    identity: dict[str, Any] = {
        "accepted": {
            "seed": accepted["seed"],
            "run_root": str(accepted["run_root"]),
            "phase": accepted["phase"],
            "selected_epoch": accepted["selected_epoch"],
            "best_model_sha256": accepted["config"].get("best_model_sha256"),
            "epoch_order_sha256": accepted["config"].get("epoch_order_sha256"),
            "initial_model_sha256": accepted["config"].get("initial_model_sha256"),
            "history": accepted["history"],
        },
        "candidate": {
            "seed": candidate["seed"],
            "run_root": str(candidate["run_root"]),
            "phase": candidate["phase"],
            "selected_epoch": candidate["selected_epoch"],
            "best_model_sha256": candidate["config"].get("best_model_sha256"),
            "epoch_order_sha256": candidate["config"].get("epoch_order_sha256"),
            "initial_model_sha256": candidate["config"].get("initial_model_sha256"),
            "history": candidate["history"],
        },
        "same_validation_inputs": {
            "validation_seed": VALIDATION_SEED,
            "accepted_validation_seed": accepted["config"].get("validation_seed"),
            "candidate_validation_seed": candidate["config"].get("validation_seed"),
            "conditions": list(CONDITIONS),
            "val_windows_per_condition": EXPECTED_VAL_WINDOWS,
            "val_files": EXPECTED_VAL_FILES,
        },
        "shared_frozen_inputs": {
            "index_sha256": accepted["config"].get("index_sha256"),
            "index_sha256_candidate": candidate["config"].get("index_sha256"),
            "split_source_sha256": accepted["config"].get("split_source_sha256"),
            "split_source_sha256_candidate": candidate["config"].get("split_source_sha256"),
            "train_windows_used_per_epoch": accepted["config"].get(
                "train_windows_used_per_epoch"
            ),
            "train_windows_used_per_epoch_candidate": candidate["config"].get(
                "train_windows_used_per_epoch"
            ),
        },
        "differences": {
            "epoch_order_sha256_differ": accepted["config"].get("epoch_order_sha256")
            != candidate["config"].get("epoch_order_sha256"),
            "initial_model_sha256_differ": accepted["config"].get("initial_model_sha256")
            != candidate["config"].get("initial_model_sha256"),
            "best_model_sha256_differ": accepted["config"].get("best_model_sha256")
            != candidate["config"].get("best_model_sha256"),
        },
    }
    per_window_source = selected_conditions if common_epoch else selected_conditions_candidate
    return {
        "seeds": seeds,
        "same_selected_epoch": common_epoch,
        "accepted_selected_epoch": accepted["selected_epoch"],
        "candidate_selected_epoch": candidate["selected_epoch"],
        "history": history,
        "epoch_table": epoch_table,
        "selected_epoch_conditions": selected_conditions,
        "candidate_selected_epoch_conditions": selected_conditions_candidate,
        "per_window_scope": (
            "both seeds' selected epoch"
            if common_epoch
            else f"seed {candidate['seed']} selected epoch {candidate['selected_epoch']}"
        ),
        "per_window_source": per_window_source,
        "identity": identity,
        "max_recorded_metric_delta": max(
            accepted["max_recorded_metric_delta"], candidate["max_recorded_metric_delta"]
        ),
    }


# ---------------------------------------------------------------------------
# fixed-width markdown tables
# ---------------------------------------------------------------------------
def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [len(str(header)) for header in headers]
    for row in rows:
        widths = [max(width, len(str(cell))) for width, cell in zip(widths, row)]
    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(str(cell).ljust(width) for cell, width in zip(cells, widths)) + " |"

    output = [line(headers), "|" + "|".join("-" * (width + 2) for width in widths) + "|"]
    output.extend(line(row) for row in rows)
    return output


def fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def fmt_pp(value: float, digits: int = 4) -> str:
    return f"{float(value) * 100:+.{digits}f}"


def build_report(comparison: dict[str, Any], accepted: dict[str, Any], candidate: dict[str, Any]) -> str:
    accepted_seed = accepted["seed"]
    candidate_seed = candidate["seed"]
    left_tag = f"seed {accepted_seed}"
    right_tag = f"seed {candidate_seed}"
    lines: list[str] = []
    lines.append("# STAGE1-12DRONES-B：一阶段 12 机型训练 seed 2027 复跑与 seed 2026 对照")
    lines.append("")
    lines.append(f"生成时间（UTC）：{utc_now()}")
    lines.append("")
    lines.append("## 1. 身份与协议")
    lines.append("")
    lines.append(
        "两次训练使用同一冻结划分、同一窗口索引、同一模型/超参/噪声档位与同一验证噪声种子，"
        "唯一被参数化的是训练随机种子。"
    )
    lines.append("")
    identity_rows = [
        ["训练种子 training_seed", str(accepted_seed), str(candidate_seed)],
        ["阶段 phase", str(accepted["phase"]), str(candidate["phase"])],
        ["验证噪声种子 validation_seed", str(accepted["config"]["validation_seed"]), str(candidate["config"]["validation_seed"])],
        ["轮数 epochs", str(accepted["config"]["epochs"]), str(candidate["config"]["epochs"])],
        ["选中轮次", str(accepted["selected_epoch"]), str(candidate["selected_epoch"])],
        ["训练窗/轮", str(accepted["config"]["train_windows_used_per_epoch"]), str(candidate["config"]["train_windows_used_per_epoch"])],
        ["验证窗/轮", str(accepted["config"]["validation_windows_used_per_epoch"]), str(candidate["config"]["validation_windows_used_per_epoch"])],
        ["best_model.pt SHA256", str(accepted["config"]["best_model_sha256"])[:16] + "...", str(candidate["config"]["best_model_sha256"])[:16] + "..."],
    ]
    lines.extend(
        render_table([f"项目", f"seed {accepted_seed}（已验收）", f"seed {candidate_seed}（本轮）"], identity_rows)
    )
    lines.append("")
    lines.append(
        f"共享输入哈希：索引 `{accepted['config']['index_sha256']}`；冻结划分 "
        f"`{accepted['config']['split_source_sha256']}`。两者在两次运行中相同。"
    )
    lines.append("")
    lines.append("## 2. 每轮四条件验证 pooled BA")
    lines.append("")
    lines.append(
        "选轮规则固定为四条件 pooled BA 的均值，同分取最早轮次；下表 BA 为各条件 pooled balanced accuracy。"
    )
    lines.append("")
    epoch_rows = []
    for row in comparison["epoch_table"]:
        cells = [str(row["epoch"])]
        for condition in CONDITIONS:
            cells.append(fmt(row[f"{condition}_accepted"]))
            cells.append(fmt(row[f"{condition}_candidate"]))
        cells.append(fmt(row["mean_accepted"]))
        cells.append(fmt(row["mean_candidate"]))
        cells.append(fmt_pp(row["mean_delta"]))
        cells.append(",".join(str(seed) for seed in row["selected_by"]) or "-")
        epoch_rows.append(cells)
    headers = ["轮"]
    for condition in CONDITIONS:
        headers.extend([f"{condition} {left_tag}", f"{condition} {right_tag}"])
    headers.extend(
        [f"均值 {left_tag}", f"均值 {right_tag}", "均值差(pp)", "该轮被选中"]
    )
    lines.extend(render_table(headers, epoch_rows))
    lines.append("")
    lines.append("## 3. 选中轮次的四条件指标对照")
    lines.append("")
    lines.append(
        f"seed {accepted_seed} 选中第 {accepted['selected_epoch']} 轮，"
        f"seed {candidate_seed} 选中第 {candidate['selected_epoch']} 轮。"
    )
    lines.append("")
    selection_rows = []
    for condition in CONDITIONS:
        left = accepted["epoch_metrics"][accepted["selected_epoch"]][condition]
        right = candidate["epoch_metrics"][candidate["selected_epoch"]][condition]
        selection_rows.append(
            [
                condition,
                fmt(left["balanced_accuracy"]),
                fmt(right["balanced_accuracy"]),
                fmt_pp(right["balanced_accuracy"] - left["balanced_accuracy"]),
                fmt(left["file_equal_balanced_accuracy"]),
                fmt(right["file_equal_balanced_accuracy"]),
                fmt_pp(
                    right["file_equal_balanced_accuracy"] - left["file_equal_balanced_accuracy"]
                ),
                fmt(left["margin_auc"], 4),
                fmt(right["margin_auc"], 4),
            ]
        )
    lines.extend(
        render_table(
            [
                "条件",
                f"pooled BA {left_tag}",
                f"pooled BA {right_tag}",
                "差(pp)",
                f"file-equal BA {left_tag}",
                f"file-equal BA {right_tag}",
                "差(pp)",
                f"margin AUC {left_tag}",
                f"margin AUC {right_tag}",
            ],
            selection_rows,
        )
    )
    lines.append("")
    lines.append("混淆计数（固定零门限判决）：")
    lines.append("")
    confusion_rows = []
    for condition in CONDITIONS:
        left = accepted["epoch_metrics"][accepted["selected_epoch"]][condition]
        right = candidate["epoch_metrics"][candidate["selected_epoch"]][condition]
        confusion_rows.append(
            [
                condition,
                f"{left['tn']}/{left['fp']}/{left['fn']}/{left['tp']}",
                f"{right['tn']}/{right['fp']}/{right['fn']}/{right['tp']}",
            ]
        )
    lines.extend(
        render_table(
            ["条件", f"TN/FP/FN/TP {left_tag}", f"TN/FP/FN/TP {right_tag}"], confusion_rows
        )
    )
    lines.append("")
    lines.append("## 4. 逐窗配对差异（为什么两次结果不同）")
    lines.append("")
    lines.append(
        f"对照口径：{comparison['per_window_scope']}；每条 2,360 个验证窗，两次运行读的是同一批窗、同一噪声实现。"
    )
    lines.append("")
    agreement_rows = []
    for condition in CONDITIONS:
        item = comparison["per_window_source"][condition]
        disagreement = item["disagreement"]
        agreement_rows.append(
            [
                condition,
                str(item["windows"]),
                str(disagreement["windows"]),
                f"{disagreement['fraction'] * 100:.2f}%",
                str(disagreement["both_wrong"]),
                str(disagreement["accepted_only_wrong"]),
                str(disagreement["candidate_only_wrong"]),
                fmt(item["margin_delta"]["mean_abs"], 4),
            ]
        )
    lines.extend(
        render_table(
            [
                "条件",
                "窗数",
                "判决不同",
                "不同比例",
                "两者都错",
                f"仅 {left_tag} 错",
                f"仅 {right_tag} 错",
                "平均|margin 差|",
            ],
            agreement_rows,
        )
    )
    lines.append("")
    lines.append("## 5. 逐源文件差异（最不稳定的文件）")
    lines.append("")
    worst: list[tuple[float, str, str, dict[str, Any]]] = []
    for condition in CONDITIONS:
        for filename, entry in comparison["per_window_source"][condition]["per_file"].items():
            worst.append((abs(entry["accuracy_delta"]), condition, filename, entry))
    worst.sort(key=lambda item: item[0], reverse=True)
    file_rows = []
    for _, condition, filename, entry in worst[:12]:
        file_rows.append(
            [
                filename,
                condition,
                str(entry["windows"]),
                str(entry["target_code"]),
                fmt(entry["accepted_accuracy"], 4),
                fmt(entry["candidate_accuracy"], 4),
                fmt_pp(entry["accuracy_delta"]),
                str(entry["disagreements"]),
            ]
        )
    lines.extend(
        render_table(
            [
                "源文件",
                "条件",
                "窗数",
                "类别",
                f"acc {left_tag}",
                f"acc {right_tag}",
                "差(pp)",
                "判决不同",
            ],
            file_rows,
        )
    )
    lines.append("")
    lines.append("## 6. 结论与边界")
    lines.append("")
    lines.append(
        "- 上述全部是**开发验证指标**：验证源文件与训练源文件同批，验证集已参与选轮，"
        "不是独立测试，也不是 12 类机型识别率（阶段 1 只分背景 / 无人机）。"
    )
    lines.append(
        "- 两次运行只差训练随机种子，因此指标差异反映的是单次训练的种子敏感性；"
        "同批验证集上的差距不能外推为跨环境或跨会话泛化差异。"
    )
    lines.append(
        "- 门限固定为 `float32(logit_drone) - float32(logit_background) > 0`；"
        "本对照不使用任何按验证集调出来的门限。"
    )
    lines.append(
        "- 逐窗配对差异按同一验证窗、同一噪声实现对齐，因此“判决不同”是模型差异而非输入差异。"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _self_test() -> None:
    """Negative controls for the comparison maths; no data on disk is touched."""
    # 1. Pooled metrics on a hand-counted confusion matrix.
    labels = [0, 0, 1, 1, 1]
    predictions = [0, 1, 1, 1, 0]
    metrics = pooled_metrics(labels, predictions)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 1, 2), metrics
    assert abs(metrics["balanced_accuracy"] - 0.5 * (2 / 3 + 0.5)) < 1e-12, metrics
    assert abs(metrics["accuracy"] - 0.6) < 1e-12, metrics

    # 2. File-equal weighting: a file with more windows must not dominate.
    rows = [
        {"filename": "bg1", "label": 0, "prediction": 0},
        {"filename": "drone1", "label": 1, "prediction": 1},
        {"filename": "drone1", "label": 1, "prediction": 0},
        {"filename": "drone2", "label": 1, "prediction": 1},
    ]
    equal = file_equal_metrics(rows)
    assert abs(equal["file_equal_background_recall"] - 1.0) < 1e-12, equal
    assert abs(equal["file_equal_drone_recall"] - 0.75) < 1e-12, equal
    assert abs(equal["file_equal_balanced_accuracy"] - 0.875) < 1e-12, equal
    assert (equal["files_background"], equal["files_drone"]) == (1, 2), equal

    # 3. Rank AUC must equal the pairwise-comparison definition used by the driver.
    rng = np.random.default_rng(20260927)
    for trial in range(200):
        drone = rng.normal(0.4, 1.0, size=int(rng.integers(1, 40)))
        background = rng.normal(0.0, 1.3, size=int(rng.integers(1, 40)))
        if trial % 7 == 0:  # inject exact ties
            drone[:2] = 0.25
            background[:2] = 0.25
        expected = float(
            (
                (drone[:, None] > background[None, :]).sum()
                + 0.5 * (drone[:, None] == background[None, :]).sum()
            )
            / (drone.size * background.size)
        )
        observed = auc_pairwise(drone, background)
        assert abs(observed - expected) < 1e-12, (trial, observed, expected)

    # 4. Perfect/inverted separation controls.
    assert auc_pairwise(np.array([2.0, 3.0]), np.array([-1.0, 0.0])) == 1.0
    assert auc_pairwise(np.array([-2.0, -3.0]), np.array([1.0, 0.0])) == 0.0
    assert auc_pairwise(np.array([1.0]), np.array([1.0])) == 0.5

    # 5. The epoch rule keeps the earliest epoch on an exact tie.
    history = [
        {"epoch": 1, SELECTION_KEY: 0.8},
        {"epoch": 2, SELECTION_KEY: 0.9},
        {"epoch": 3, SELECTION_KEY: 0.9},
    ]
    assert select_epoch(history) == 2, select_epoch(history)
    tie = [{"epoch": epoch, SELECTION_KEY: 0.7} for epoch in (1, 2, 3)]
    assert select_epoch(tie) == 1, select_epoch(tie)

    # 6. ``decode`` must align rows by index position, not by file order.
    shuffled = [
        {
            "condition": "clean",
            "index_position": 1,
            "filename": "b.mat",
            "target_code": "T0001",
            "window_index": 1,
            "start": 1_000_000,
            "label": 1,
            "margin": 0.5,
            "prediction": 1,
        },
        {
            "condition": "clean",
            "index_position": 0,
            "filename": "a.mat",
            "target_code": "T0000",
            "window_index": 0,
            "start": 0,
            "label": 0,
            "margin": -0.5,
            "prediction": 0,
        },
    ]
    decoded = decode(shuffled)
    assert list(decoded["index_position"]) == [0, 1], decoded["index_position"]
    assert list(decoded["label"]) == [0, 1], decoded["label"]
    assert list(decoded["filename"]) == ["a.mat", "b.mat"], decoded["filename"]

    # 7. ``pooled_metrics`` must reject a non-conserving confusion matrix.
    try:
        pooled_metrics([0, 1], [0])
    except ValueError:
        pass
    else:
        raise AssertionError("pooled_metrics accepted mismatched lengths")


def summary_table_rows(comparison: dict[str, Any], accepted: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    del accepted, candidate
    rows: list[dict[str, Any]] = []
    for epoch_row in comparison["epoch_table"]:
        row: dict[str, Any] = {"epoch": epoch_row["epoch"]}
        for key, value in epoch_row.items():
            if key in {"epoch", "selected_by"}:
                continue
            row[key] = value
        row["selected_by"] = ";".join(str(seed) for seed in epoch_row["selected_by"])
        rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two STAGE1-12DRONES stage-1 training seeds (read-only)."
    )
    parser.add_argument("--accepted-seed", type=int, default=ACCEPTED_SEED)
    parser.add_argument("--candidate-seed", type=int, default=CANDIDATE_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / "seed_comparison",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "DroneRFa_一阶段12机型_seed复跑对照_20260927.md",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        _self_test()
        print("self-test passed: metrics, file-equal weighting and margin AUC")
        return 0

    if sha256_file(INDEX) != INDEX_SHA256:
        raise RuntimeError("the frozen window index hash changed")
    if len(read_csv_rows(INDEX)) != INDEX_ROWS:
        raise RuntimeError("the frozen window index row count changed")

    accepted = recompute_run(args.accepted_seed)
    candidate = recompute_run(args.candidate_seed)
    comparison = compare_runs(accepted, candidate)

    report = build_report(comparison, accepted, candidate)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    write_json(
        args.output_dir / "seed_comparison.json",
        {
            "schema": "dronerfa-stage1-12drones-seed-comparison/v1",
            "generated_at": utc_now(),
            "index": {"path": str(INDEX), "sha256": INDEX_SHA256, "rows": INDEX_ROWS},
            "protocol_note": (
                "development validation metrics on one frozen split; not an independent "
                "test and not a 12-class identification rate"
            ),
            **{key: value for key, value in comparison.items() if key != "per_window_source"},
            "per_window": comparison["per_window_source"],
        },
    )
    comparison_rows = summary_table_rows(comparison, accepted, candidate)
    with (args.output_dir / "seed_epoch_table.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    per_file_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for filename, entry in comparison["per_window_source"][condition]["per_file"].items():
            per_file_rows.append({"condition": condition, "filename": filename, **entry})
    with (args.output_dir / "seed_per_file_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_file_rows[0]))
        writer.writeheader()
        writer.writerows(per_file_rows)

    print(
        json.dumps(
            {
                "accepted_seed": accepted["seed"],
                "candidate_seed": candidate["seed"],
                "accepted_selected_epoch": accepted["selected_epoch"],
                "candidate_selected_epoch": candidate["selected_epoch"],
                "max_recorded_metric_delta": comparison["max_recorded_metric_delta"],
                "selected_epoch_deltas_pp": {
                    condition: round(
                        comparison["selected_epoch_conditions"][condition]["delta"][
                            "balanced_accuracy"
                        ]
                        * 100,
                        4,
                    )
                    for condition in CONDITIONS
                },
                "report": str(args.report),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
