#!/usr/bin/env python3
"""STAGE1-LOSS-NORM: paired summary of the candidate against the frozen control.

Reads only the *recorded per-window validation decisions* of three runs:

* the new candidate (``global_train_mean``, this task's one formal run);
* the accepted control (``STAGE1-12DRONES-seed2027``, ``batch_weight_sum``);
* the seed-2026 context run (a different random seed -- never a loss effect).

and writes the acceptance metrics the task asks for: per-condition TP/TN/FP/FN,
background FPR, drone FNR/recall, per-target-code detection recall with its source
file and window counts, file-equal BA, the worst condition, and a descriptive AUC.
The primary comparison is the frozen selection key (mean pooled BA over the four
validation conditions at the epoch the frozen rule picks, zero threshold).

Nothing here trains, reads RF0 payloads or changes any run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np  # noqa: E402

import run_stage1_loss_normalization as entry  # noqa: E402
import run_stage1_12drones_training as base  # noqa: E402

DEFAULT_OUTPUT = entry.OUTPUT_ROOT / "comparison"
CONDITIONS = tuple(base.CONDITIONS)
BACKGROUND_CODE = base.BACKGROUND_CODE
DRONE_CODES = (
    "T0001",
    "T0010",
    "T0011",
    "T0100",
    "T0101",
    "T0110",
    "T0111",
    "T1000",
    "T1001",
    "T1010",
    "T1011",
    "T1100",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def auc_from_margins(labels: Sequence[int], margins: Sequence[float]) -> float:
    """Mann-Whitney AUC of the detection margin (descriptive only)."""

    y = np.asarray(labels, dtype=np.int64)
    margin = np.asarray(margins, dtype=np.float64)
    positive = margin[y == 1]
    negative = margin[y == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = (positive[:, None] > negative[None, :]).sum() + 0.5 * (
        positive[:, None] == negative[None, :]
    ).sum()
    return float(comparisons / (len(positive) * len(negative)))


def confusion(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([int(row["label"]) for row in records], dtype=np.int64)
    prediction = np.asarray([int(row["prediction"]) for row in records], dtype=np.int64)
    tp = int(np.sum((y == 1) & (prediction == 1)))
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    total = len(y)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "windows": int(total),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tp + tn) / max(total, 1),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "drone_recall": recall,
        "drone_fnr": fn / max(fn + tp, 1),
        "background_specificity": specificity,
        "background_fpr": fp / max(fp + tn, 1),
        "auc_descriptive": auc_from_margins([row["label"] for row in records], [row["logit_margin"] for row in records]),
    }


def file_equal(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
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
    background = float(np.mean(by_class[0])) if by_class[0] else float("nan")
    drone = float(np.mean(by_class[1])) if by_class[1] else float("nan")
    return {
        "files_background": len(by_class[0]),
        "files_drone": len(by_class[1]),
        "file_equal_background_recall": background,
        "file_equal_drone_recall": drone,
        "file_equal_balanced_accuracy": 0.5 * (background + drone),
    }


def per_target_code(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detection recall per target code, with honest source-file/window counts."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(str(row["target_code"]), []).append(row)
    rows: list[dict[str, Any]] = []
    for code in (BACKGROUND_CODE, *DRONE_CODES):
        bucket = grouped.get(code)
        if not bucket:
            continue
        label = int(bucket[0]["label"])
        per_file: dict[str, list[int]] = {}
        for row in bucket:
            per_file.setdefault(str(row["filename"]), []).append(
                int(int(row["prediction"]) == int(row["label"]))
            )
        file_rates = [float(np.mean(values)) for values in per_file.values()]
        correct = int(sum(sum(values) for values in per_file.values()))
        row: dict[str, Any] = {
            "target_code": code,
            "stage1_label": label,
            "class": "Background" if label == 0 else "Drone",
            "source_files": len(per_file),
            "windows": len(bucket),
            "windows_correct": correct,
            "window_recall": correct / max(len(bucket), 1),
            "file_equal_recall": float(np.mean(file_rates)),
            "file_recall_min": float(np.min(file_rates)),
            "file_recall_max": float(np.max(file_rates)),
            "mean_margin": float(np.mean([float(item["logit_margin"]) for item in bucket])),
        }
        if label == 0:
            # A background window's "recall" is its specificity.
            row["background_specificity"] = row["window_recall"]
            row["background_fpr"] = 1.0 - row["window_recall"]
        rows.append(row)
    return rows


def load_run(run_root: Path, label: str) -> dict[str, Any]:
    config_path = run_root / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{label}: no run_config.json at {run_root}")
    config = read_json(config_path)
    selected = int(config["selected_epoch"])
    if int(config["training_seed"]) not in (2026, 2027):
        raise RuntimeError(f"{label}: unexpected training seed {config['training_seed']}")
    scores_path = run_root / f"validation_scores_epoch_{selected:02d}.csv"
    records = read_csv_rows(scores_path)
    per_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for row in records:
        entry_row = {
            "condition": str(row["condition"]),
            "index_position": int(row["index_position"]),
            "filename": str(row["filename"]),
            "target_code": str(row["target_code"]),
            "window_index": int(row["window_index"]),
            "label": int(row["label"]),
            "logit_margin": float(row["logit_margin"]),
            "prediction": int(row["prediction"]),
        }
        per_condition[entry_row["condition"]].append(entry_row)
    return {
        "label": label,
        "run_root": str(run_root),
        "run_config": str(config_path),
        "history": config.get("epoch_history"),
        "selected_epoch": selected,
        "selected_epoch_score": float(config["selected_epoch_score"]),
        "selection_key": config["selection_key"],
        "loss": config.get("loss"),
        "loss_normalization": config.get("loss_normalization"),
        "best_model_sha256": config.get("best_model_sha256"),
        "initial_model_sha256": config.get("initial_model_sha256"),
        "epoch_order_sha256": config.get("epoch_order_sha256"),
        "train_class_counts": config.get("train_class_counts"),
        "class_weights": config.get("class_weights"),
        "scores_path": str(scores_path),
        "records": records,
        "per_condition": per_condition,
    }


def summarise(run: dict[str, Any]) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        records = run["per_condition"][condition]
        conditions[condition] = {**confusion(records), **file_equal(records)}
    mean_ba = float(np.mean([conditions[condition]["balanced_accuracy"] for condition in CONDITIONS]))
    mean_file_equal = float(
        np.mean([conditions[condition]["file_equal_balanced_accuracy"] for condition in CONDITIONS])
    )
    worst_condition = min(
        CONDITIONS, key=lambda condition: conditions[condition]["balanced_accuracy"]
    )
    pooled_records = [row for condition in CONDITIONS for row in run["per_condition"][condition]]
    return {
        "selected_epoch": run["selected_epoch"],
        "selected_epoch_score": run["selected_epoch_score"],
        "conditions": conditions,
        "mean_pooled_balanced_accuracy": mean_ba,
        "mean_file_equal_balanced_accuracy": mean_file_equal,
        "worst_condition": worst_condition,
        "worst_condition_balanced_accuracy": conditions[worst_condition]["balanced_accuracy"],
        "pooled_over_conditions": confusion(pooled_records),
        "per_target_code": per_target_code(pooled_records),
    }


def paired_windows(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    """Per-window transitions between the two runs at the same frozen positions."""

    transitions: dict[str, int] = {}
    per_condition: dict[str, dict[str, int]] = {condition: {} for condition in CONDITIONS}
    by_class: dict[str, dict[str, int]] = {
        "Background": {"both_correct": 0, "both_wrong": 0, "control_fp_to_candidate_tn": 0, "control_tn_to_candidate_fp": 0},
        "Drone": {"both_correct": 0, "both_wrong": 0, "control_tp_to_candidate_fn": 0, "control_fn_to_candidate_tp": 0},
    }
    for condition in CONDITIONS:
        left = {row["index_position"]: row for row in candidate["per_condition"][condition]}
        right = {row["index_position"]: row for row in control["per_condition"][condition]}
        if set(left) != set(right):
            raise RuntimeError(
                f"{condition}: the two runs do not cover the same frozen positions "
                f"({len(left)} vs {len(right)})"
            )
        for position, row in left.items():
            other = right[position]
            if int(row["label"]) != int(other["label"]):
                raise RuntimeError(f"{condition}/{position}: label mismatch")
            control_prediction = int(other["prediction"])
            candidate_prediction = int(row["prediction"])
            label = int(row["label"])
            key = f"{control_prediction}->{candidate_prediction}"
            transitions[key] = transitions.get(key, 0) + 1
            per_condition[condition][key] = per_condition[condition].get(key, 0) + 1
            slot = by_class["Background" if label == 0 else "Drone"]
            if label == 0:
                if control_prediction == 0 and candidate_prediction == 0:
                    slot["both_correct"] += 1
                elif control_prediction == 1 and candidate_prediction == 1:
                    slot["both_wrong"] += 1
                elif control_prediction == 1 and candidate_prediction == 0:
                    slot["control_fp_to_candidate_tn"] += 1
                else:
                    slot["control_tn_to_candidate_fp"] += 1
            else:
                if control_prediction == 1 and candidate_prediction == 1:
                    slot["both_correct"] += 1
                elif control_prediction == 0 and candidate_prediction == 0:
                    slot["both_wrong"] += 1
                elif control_prediction == 1 and candidate_prediction == 0:
                    slot["control_tp_to_candidate_fn"] += 1
                else:
                    slot["control_fn_to_candidate_tp"] += 1
    return {
        "transitions": transitions,
        "per_condition": per_condition,
        "by_class": by_class,
        "key": (
            "'<control prediction>-><candidate prediction>', so '1->0' is a window the "
            "control detected and the candidate missed and '0->1' is a new false alarm"
        ),
    }


def trajectory(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-epoch validation score and training loss of one run, as recorded."""

    rows: list[dict[str, Any]] = []
    for row in run.get("history") or []:
        rows.append(
            {
                "epoch": int(row["epoch"]),
                "mean_pooled_balanced_accuracy": float(row["mean_balanced_accuracy"]),
                "mean_file_equal_balanced_accuracy": float(row["mean_file_equal_balanced_accuracy"]),
                "clean_balanced_accuracy": float(row["clean_balanced_accuracy"]),
                "train_objective_loss": float(row["train_loss"]),
                "seconds": float(row.get("seconds") or 0.0),
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=entry.OUTPUT_ROOT / entry.CANDIDATE_DIR_NAME)
    parser.add_argument("--control", type=Path, default=entry.CONTROL_ROOT)
    parser.add_argument("--context", type=Path, default=entry.CONTEXT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    candidate = load_run(args.candidate, "candidate_global_train_mean")
    control = load_run(args.control, "control_seed2027_batch_weight_sum")
    context = load_run(args.context, "context_seed2026_batch_weight_sum")

    summaries = {
        "candidate": summarise(candidate),
        "control": summarise(control),
        "context": summarise(context),
    }
    comparison: dict[str, Any] = {
        "schema": "dronerfa-stage1-loss-norm-comparison/v1",
        "generated_at": utc_now(),
        "primary": {
            "metric": "mean pooled balanced accuracy over the four frozen validation conditions",
            "selection_key": base.SELECTION_KEY,
            "selection_rule": base.SELECTION_RULE,
            "threshold": base.PREDICTION_RULE,
            "candidate": {
                "run_root": candidate["run_root"],
                "loss_mode": (candidate["loss_normalization"] or {}).get("mode"),
                "selected_epoch": candidate["selected_epoch"],
                "mean_pooled_balanced_accuracy": summaries["candidate"]["mean_pooled_balanced_accuracy"],
            },
            "control": {
                "run_root": control["run_root"],
                "loss_mode": "batch_weight_sum",
                "selected_epoch": control["selected_epoch"],
                "mean_pooled_balanced_accuracy": summaries["control"]["mean_pooled_balanced_accuracy"],
            },
            "delta_candidate_minus_control": summaries["candidate"]["mean_pooled_balanced_accuracy"]
            - summaries["control"]["mean_pooled_balanced_accuracy"],
            "delta_pp": 100.0
            * (
                summaries["candidate"]["mean_pooled_balanced_accuracy"]
                - summaries["control"]["mean_pooled_balanced_accuracy"]
            ),
            "context_only_seed2026": {
                "run_root": context["run_root"],
                "selected_epoch": context["selected_epoch"],
                "mean_pooled_balanced_accuracy": summaries["context"]["mean_pooled_balanced_accuracy"],
                "note": (
                    "a different random seed, provided only as context for how much a "
                    "seed alone moves the metric; it is not a loss-change effect"
                ),
            },
        },
        "per_condition": {
            condition: {
                "candidate": summaries["candidate"]["conditions"][condition],
                "control": summaries["control"]["conditions"][condition],
                "context": summaries["context"]["conditions"][condition],
                "delta_balanced_accuracy": summaries["candidate"]["conditions"][condition]["balanced_accuracy"]
                - summaries["control"]["conditions"][condition]["balanced_accuracy"],
                "delta_background_fpr": summaries["candidate"]["conditions"][condition]["background_fpr"]
                - summaries["control"]["conditions"][condition]["background_fpr"],
                "delta_drone_fnr": summaries["candidate"]["conditions"][condition]["drone_fnr"]
                - summaries["control"]["conditions"][condition]["drone_fnr"],
            }
            for condition in CONDITIONS
        },
        "worst_condition": {
            "candidate": {
                "condition": summaries["candidate"]["worst_condition"],
                "balanced_accuracy": summaries["candidate"]["worst_condition_balanced_accuracy"],
            },
            "control": {
                "condition": summaries["control"]["worst_condition"],
                "balanced_accuracy": summaries["control"]["worst_condition_balanced_accuracy"],
            },
        },
        "file_equal": {
            key: {
                "mean_file_equal_balanced_accuracy": summaries[key]["mean_file_equal_balanced_accuracy"],
                "by_condition": {
                    condition: summaries[key]["conditions"][condition]["file_equal_balanced_accuracy"]
                    for condition in CONDITIONS
                },
            }
            for key in ("candidate", "control", "context")
        },
        "paired_windows": paired_windows(candidate, control),
        "epoch_trajectory": {
            "note": (
                "per-epoch frozen validation score (mean pooled BA over the four "
                "conditions) of each run before selection; the two 2027 runs share the "
                "same orders and the same validation inputs.  'train_objective_loss' is "
                "each run's OWN normalisation and the two columns must never be compared "
                "as if they shared a definition"
            ),
            "candidate": trajectory(candidate),
            "control": trajectory(control),
            "context": trajectory(context),
        },
        "run_identity": {
            "candidate": {
                "initial_model_sha256": candidate["initial_model_sha256"],
                "best_model_sha256": candidate["best_model_sha256"],
                "epoch_order_sha256": candidate["epoch_order_sha256"],
                "train_class_counts": candidate["train_class_counts"],
                "class_weights": candidate["class_weights"],
            },
            "control": {
                "initial_model_sha256": control["initial_model_sha256"],
                "best_model_sha256": control["best_model_sha256"],
                "epoch_order_sha256": control["epoch_order_sha256"],
                "train_class_counts": control["train_class_counts"],
                "class_weights": control["class_weights"],
            },
        },
        "scope_statement": (
            "development evaluation on the frozen 16-file val split, whose source files "
            "sit in the same acquisition-time cluster as the training files; it is not "
            "an independent final test and not cross-session or cross-environment "
            "evidence, and one paired seed cannot establish cross-seed stability"
        ),
    }

    per_condition_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        row: dict[str, Any] = {"condition": condition}
        for key in ("candidate", "control", "context"):
            metrics = summaries[key]["conditions"][condition]
            for field in (
                "windows",
                "tp",
                "tn",
                "fp",
                "fn",
                "balanced_accuracy",
                "drone_recall",
                "drone_fnr",
                "background_specificity",
                "background_fpr",
                "file_equal_balanced_accuracy",
                "auc_descriptive",
            ):
                row[f"{key}_{field}"] = metrics[field]
        row["delta_ba_candidate_minus_control"] = (
            summaries["candidate"]["conditions"][condition]["balanced_accuracy"]
            - summaries["control"]["conditions"][condition]["balanced_accuracy"]
        )
        per_condition_rows.append(row)

    target_rows: list[dict[str, Any]] = []
    candidate_by_code = {row["target_code"]: row for row in summaries["candidate"]["per_target_code"]}
    control_by_code = {row["target_code"]: row for row in summaries["control"]["per_target_code"]}
    for code in sorted(candidate_by_code, key=lambda value: (value != BACKGROUND_CODE, value)):
        left = candidate_by_code[code]
        right = control_by_code[code]
        target_rows.append(
            {
                "target_code": code,
                "class": left["class"],
                "conditions_pooled": len(CONDITIONS),
                "source_files": left["source_files"],
                "windows": left["windows"],
                "windows_per_condition": left["windows"] // len(CONDITIONS),
                "candidate_windows_correct": left["windows_correct"],
                "candidate_window_recall": left["window_recall"],
                "candidate_file_equal_recall": left["file_equal_recall"],
                "control_window_recall": right["window_recall"],
                "control_file_equal_recall": right["file_equal_recall"],
                "delta_window_recall": left["window_recall"] - right["window_recall"],
                "candidate_mean_margin": left["mean_margin"],
                "control_mean_margin": right["mean_margin"],
            }
        )

    write_json(args.output / "paired_comparison.json", comparison)
    write_json(args.output / "run_summaries.json", summaries)
    write_csv(args.output / "per_condition_metrics.csv", per_condition_rows)
    write_csv(args.output / "per_target_code_recall.csv", target_rows)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_mean_ba": summaries["candidate"]["mean_pooled_balanced_accuracy"],
                "control_mean_ba": summaries["control"]["mean_pooled_balanced_accuracy"],
                "delta_pp": comparison["primary"]["delta_pp"],
                "candidate_selected_epoch": candidate["selected_epoch"],
                "control_selected_epoch": control["selected_epoch"],
                "transitions": comparison["paired_windows"]["transitions"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
