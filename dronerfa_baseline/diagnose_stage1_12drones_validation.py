#!/usr/bin/env python3
"""Read-only validation-error diagnosis for the accepted STAGE1-12DRONES-A run.

Scope of this script (task STAGE1-12DRONES-DIAG):

* It never trains, never opens a ``development_test*`` MAT, and never writes
  outside ``outputs/stage1_12drones_20260926/diagnostics/``.
* It re-derives every validation metric from the recorded per-window score files
  with a local implementation (no import of ``run_stage1_12drones_training``), so
  a bug in the training driver's own metric code cannot hide here.
* It checks that each epoch x condition score file holds exactly the 2,360 frozen
  ``val`` windows over the 16 frozen ``val`` MAT files, that the keys are unique,
  and that role/label/target-code agree with the frozen 104-file index.  Any
  missing window, duplicate key, cross-role row or label mismatch is a hard
  failure (non-zero exit) and is reported.

The task is a *detection* diagnosis: labels are Background (T0000) vs Drone
(the twelve other target codes).  A per-T-code recall in this report is therefore
"how often this drone model was detected as a drone", not a 12-way model
identification accuracy.

Outputs (all JSON/CSV, plus a console table):

    diagnostics/input_identity.json
    diagnostics/score_file_integrity.json
    diagnostics/metric_reconciliation.json
    diagnostics/validation_metrics_recomputed.csv
    diagnostics/epoch_trajectory_by_condition.csv
    diagnostics/epoch_per_file_metrics.csv
    diagnostics/epoch03_model_by_condition.csv
    diagnostics/epoch03_model_overall.csv
    diagnostics/epoch03_file_condition_breakdown.csv
    diagnostics/epoch_flip_analysis.json
    diagnostics/threshold_diagnostic.json
    diagnostics/diagnostics_summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RUN_DIR = SCRIPT_DIR / "outputs" / "stage1_12drones_20260926"
INDEX_CSV = RUN_DIR / "window_index_stage1_12drones.csv"
SEED_ROOT = RUN_DIR / "seed_2026"
DIAG_DIR = RUN_DIR / "diagnostics"
FROZEN_SPLIT_CSV = PROJECT_ROOT / "work" / "meta_split2" / "split_candidate_104_20260926.csv"

#: Frozen identities taken from the accepted reports, re-checked on every run.
FROZEN = {
    "index_sha256": "5973a2fa5b8476a507fcc7a665365daf8d572f569ce5908d54f5c4bc84c48bd6",
    "index_rows": 15790,
    "split_csv_sha256": "d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9",
    "best_model_sha256": "37dbb815450b8ec0f798a113a7f922858fcb2be604a863eb7594eb5c52ff814b",
}

VAL_ROLE = "val"
TEST_ROLES = ("development_test", "development_test_exposed")
BACKGROUND_CODE = "T0000"
CONDITIONS = ("clean", "-10dB", "-12dB", "-15dB")
EPOCHS = (1, 2, 3, 4, 5)
SELECTED_EPOCH = 3
VAL_WINDOWS = 2360
VAL_FILES = 16
BACKGROUND_FILES = 2
DRONE_FILES = 14
#: Training SNR mixture recorded in run_config.json; -15 dB is *not* in it.
TRAIN_SNRS = "clean,10,5,0,-5,-10,-12"
SELECTION_RULE = "mean_pooled_balanced_accuracy_over_four_validation_conditions"

METRIC_FLOAT_FIELDS = (
    "loss",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "drone_recall",
    "background_specificity",
    "f1",
    "file_equal_background_recall",
    "file_equal_drone_recall",
    "file_equal_balanced_accuracy",
)
METRIC_INT_FIELDS = ("samples", "windows", "tn", "fp", "fn", "tp", "files_background", "files_drone")
#: The stored CSV writes full float64 repr, so agreement must be exact modulo
#: float text round-trip.  Anything above this is a real discrepancy.
METRIC_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# independent metric implementations (deliberately re-written, not imported)
# ---------------------------------------------------------------------------
def pooled_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    """Background-vs-drone confusion counts and the accepted metric set."""
    n = len(labels)
    if n != len(predictions):
        raise ValueError("labels/predictions length mismatch")
    tp = tn = fp = fn = 0
    for y, p in zip(labels, predictions):
        if y == 1 and p == 1:
            tp += 1
        elif y == 1 and p == 0:
            fn += 1
        elif y == 0 and p == 0:
            tn += 1
        elif y == 0 and p == 1:
            fp += 1
        else:
            raise ValueError(f"unexpected label/prediction pair {(y, p)}")
    if tp + tn + fp + fn != n:
        raise ValueError("confusion matrix does not conserve the window count")
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "windows": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "drone_recall": recall,
        "background_specificity": specificity,
        "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def file_equal_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean per-file accuracy inside each class, then the two classes equally.

    Every source MAT file counts once, so a 180-window file cannot outvote a
    110-window file.
    """
    per_file: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = per_file.setdefault(
            str(row["filename"]), {"label": int(row["label"]), "correct": 0, "windows": 0}
        )
        if entry["label"] != int(row["label"]):
            raise ValueError(f"source file {row['filename']} carries two classes")
        entry["correct"] += 1 if int(row["prediction"]) == int(row["label"]) else 0
        entry["windows"] += 1
    class_rates: dict[int, list[float]] = {0: [], 1: []}
    for entry in per_file.values():
        class_rates[entry["label"]].append(entry["correct"] / entry["windows"])
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


def summary_stats(values: Sequence[float]) -> dict[str, Any]:
    """Distribution summary used for every margin / logit report."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "n": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "n": int(array.size),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def detection_auc(drone_margins: Sequence[float], background_margins: Sequence[float]) -> float:
    """Threshold-free separability: P(margin(drone) > margin(background)).

    The Mann-Whitney statistic with average ranks for ties, so the value does not
    depend on where the decision threshold sits.  This is the measure that tells
    apart "the model separates the classes worse" from "the model separates them
    equally well but its default threshold drifted".
    """
    pos = np.asarray(drone_margins, dtype=np.float64)
    neg = np.asarray(background_margins, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("detection AUC needs both classes")
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(combined.size, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)
    ordered_values = combined[order]
    cursor = 0
    while cursor < ordered_values.size:
        stop = cursor
        while stop + 1 < ordered_values.size and ordered_values[stop + 1] == ordered_values[cursor]:
            stop += 1
        if stop > cursor:
            ranks[order[cursor : stop + 1]] = (cursor + stop + 2) / 2.0
        cursor = stop + 1
    rank_sum = float(ranks[: pos.size].sum())
    return (rank_sum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)


def d_prime(drone_margins: Sequence[float], background_margins: Sequence[float]) -> float:
    """Descriptive separation in pooled standard deviations (no threshold)."""
    pos = np.asarray(drone_margins, dtype=np.float64)
    neg = np.asarray(background_margins, dtype=np.float64)
    pooled = math.sqrt(0.5 * (float(np.var(pos)) + float(np.var(neg))))
    if pooled == 0.0:
        return float("inf") if float(np.mean(pos)) != float(np.mean(neg)) else 0.0
    return (float(np.mean(pos)) - float(np.mean(neg))) / pooled


def _ba_at_threshold(drone_sorted: np.ndarray, background_sorted: np.ndarray, t: float) -> float:
    """Balanced accuracy when the decision rule is ``margin > t``."""
    n_drone = drone_sorted.size
    n_background = background_sorted.size
    if n_drone == 0 or n_background == 0:
        raise ValueError("balanced accuracy needs both classes")
    tp = n_drone - int(np.searchsorted(drone_sorted, t, side="right"))
    fp = n_background - int(np.searchsorted(background_sorted, t, side="right"))
    recall = tp / n_drone
    specificity = (n_background - fp) / n_background
    return 0.5 * (recall + specificity)


def _threshold_candidates(groups: Sequence[np.ndarray]) -> np.ndarray:
    """Every threshold that can produce a distinct decision partition."""
    pooled = np.unique(np.concatenate([np.asarray(g, dtype=np.float64) for g in groups]))
    if pooled.size == 0:
        return np.asarray([0.0])
    candidates = [pooled[0] - 1.0]
    candidates.extend(((pooled[:-1] + pooled[1:]) / 2.0).tolist())
    candidates.append(pooled[-1] + 1.0)
    return np.asarray(candidates, dtype=np.float64)


def oracle_thresholds(
    per_condition: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Best achievable threshold, per condition and shared across conditions.

    Diagnostic only: these thresholds are fitted on the same 2,360 validation
    windows that are being scored, so the resulting numbers are an optimistic
    bound on what a *calibrated* decision rule could reach.  They must never be
    reported as validation performance or used to pick between epochs.
    """
    sorted_margins = {
        condition: {
            1: np.sort(np.asarray(payload["drone_margins"], dtype=np.float64)),
            0: np.sort(np.asarray(payload["background_margins"], dtype=np.float64)),
        }
        for condition, payload in per_condition.items()
    }
    per_condition_best: dict[str, Any] = {}
    for condition, payload in sorted_margins.items():
        candidates = _threshold_candidates([payload[1], payload[0]])
        scores = np.asarray(
            [_ba_at_threshold(payload[1], payload[0], float(t)) for t in candidates]
        )
        best = int(np.argmax(scores))
        zero = _ba_at_threshold(payload[1], payload[0], 0.0)
        per_condition_best[condition] = {
            "threshold": float(candidates[best]),
            "balanced_accuracy": float(scores[best]),
            "balanced_accuracy_at_zero": float(zero),
            "gain": float(scores[best] - zero),
        }

    shared_candidates = _threshold_candidates(
        [m for payload in sorted_margins.values() for m in (payload[1], payload[0])]
    )
    shared_scores = np.asarray(
        [
            float(
                np.mean(
                    [
                        _ba_at_threshold(payload[1], payload[0], float(t))
                        for payload in sorted_margins.values()
                    ]
                )
            )
            for t in shared_candidates
        ]
    )
    best_shared = int(np.argmax(shared_scores))
    shared_threshold = float(shared_candidates[best_shared])
    return {
        "note": (
            "oracle thresholds fitted on the same 2,360 val windows; diagnostic "
            "upper bound only, not a validation score and not a selection rule"
        ),
        "per_condition_oracle": per_condition_best,
        "shared_threshold_oracle": {
            "threshold": shared_threshold,
            "mean_balanced_accuracy": float(shared_scores[best_shared]),
            "per_condition_balanced_accuracy": {
                condition: _ba_at_threshold(payload[1], payload[0], shared_threshold)
                for condition, payload in sorted_margins.items()
            },
        },
        "default_threshold": 0.0,
    }


# ---------------------------------------------------------------------------
# loading / integrity
# ---------------------------------------------------------------------------
def load_index(index_path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = read_csv_rows(index_path)
    val_rows = [row for row in rows if row["source_role"] == VAL_ROLE]
    return rows, val_rows


def build_expected_val_table(
    val_rows: Sequence[dict[str, str]]
) -> tuple[dict[tuple[str, str], dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    """Expected window table for the val split, keyed by (filename, window_index)."""
    expected: dict[tuple[str, str], dict[str, str]] = {}
    ordered: list[dict[str, str]] = []
    for position, row in enumerate(val_rows):
        key = (row["filename"], str(int(row["window_index"])))
        if key in expected:
            raise RuntimeError(f"duplicate window key in the frozen index: {key}")
        expected[key] = row
        ordered.append(
            {
                "index_position": position,
                "filename": row["filename"],
                "window_index": int(row["window_index"]),
                "start": int(row["start"]),
                "label": int(row["label"]),
                "target_code": row["target_code"],
                "split": row["split"],
                "source_role": row["source_role"],
            }
        )
    files = sorted({row["filename"] for row in val_rows})
    meta = {
        "val_rows": len(val_rows),
        "val_files": len(files),
        "background_files": len([f for f in files if f.startswith(BACKGROUND_CODE)]),
        "label_counts": {
            str(label): sum(1 for row in val_rows if int(row["label"]) == label) for label in (0, 1)
        },
        "target_code_counts": dict(
            sorted(Counter(row["target_code"] for row in val_rows).items())
        ),
        "files": files,
    }
    return expected, ordered, meta


def load_score_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in read_csv_rows(path):
        rows.append(
            {
                "condition": str(raw["condition"]),
                "selection_position": int(raw["selection_position"]),
                "index_position": int(raw["index_position"]),
                "filename": str(raw["filename"]),
                "target_code": str(raw["target_code"]),
                "window_index": int(raw["window_index"]),
                "start": int(raw["start"]),
                "label": int(raw["label"]),
                "logit_margin": float(raw["logit_margin"]),
                "prediction": int(raw["prediction"]),
            }
        )
    return rows


def check_score_integrity(
    epoch: int,
    rows: Sequence[dict[str, Any]],
    expected: dict[tuple[str, str], dict[str, str]],
    ordered: Sequence[dict[str, str]],
    meta: dict[str, Any],
    role_by_file: dict[str, str],
) -> dict[str, Any]:
    """Per-epoch-per-condition integrity of the recorded per-window scores.

    ``expected`` holds only the frozen ``val`` windows, so any row that is not in
    it is either a stray window or a window from another role.  ``role_by_file``
    resolves that case explicitly so a train/test row inside a validation score
    file is reported as cross-role leakage instead of a generic mismatch.
    """
    problems: list[str] = []
    position_to_row = {int(row["index_position"]): row for row in ordered}
    per_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        keys = [(row["filename"], str(row["window_index"])) for row in subset]
        key_counts = Counter(keys)
        duplicates = sorted(key for key, count in key_counts.items() if count > 1)
        missing = sorted(set(expected) - set(keys))
        unexpected = sorted(set(keys) - set(expected))
        cross_role_rows = sorted(
            {
                (filename, role_by_file.get(filename, "unknown"))
                for filename, _ in unexpected
                if role_by_file.get(filename) not in (None, VAL_ROLE)
            }
        )
        selections = [int(row["selection_position"]) for row in subset]
        positions = [int(row["index_position"]) for row in subset]
        label_mismatch: list[dict[str, Any]] = []
        code_mismatch: list[dict[str, Any]] = []
        start_mismatch: list[dict[str, Any]] = []
        for row in subset:
            key = (row["filename"], str(row["window_index"]))
            reference = expected.get(key)
            if reference is None:
                continue
            if int(reference["label"]) != int(row["label"]):
                label_mismatch.append(
                    {
                        "filename": row["filename"],
                        "window_index": row["window_index"],
                        "score_label": row["label"],
                        "index_label": int(reference["label"]),
                    }
                )
            if str(reference["target_code"]) != str(row["target_code"]):
                code_mismatch.append(
                    {
                        "filename": row["filename"],
                        "window_index": row["window_index"],
                        "score_code": row["target_code"],
                        "index_code": reference["target_code"],
                    }
                )
            if int(reference["start"]) != int(row["start"]):
                start_mismatch.append(
                    {
                        "filename": row["filename"],
                        "window_index": row["window_index"],
                        "score_start": row["start"],
                        "index_start": int(reference["start"]),
                    }
                )
        files_seen = sorted({row["filename"] for row in subset})
        index_positions_required = set(range(len(ordered)))
        condition_problems: list[str] = []
        if len(subset) != len(expected):
            condition_problems.append(
                f"{condition}: {len(subset)} rows, expected {len(expected)}"
            )
        if duplicates:
            condition_problems.append(f"{condition}: {len(duplicates)} duplicate keys")
        if missing:
            condition_problems.append(f"{condition}: {len(missing)} frozen val windows missing")
        if unexpected:
            condition_problems.append(
                f"{condition}: {len(unexpected)} rows outside the frozen val split"
            )
        if cross_role_rows:
            condition_problems.append(
                f"{condition}: {len(cross_role_rows)} rows belong to a non-val role "
                f"(cross-role leakage): {cross_role_rows[:5]}"
            )
        if len(files_seen) != meta["val_files"]:
            condition_problems.append(
                f"{condition}: {len(files_seen)} files, expected {meta['val_files']}"
            )
        if label_mismatch:
            condition_problems.append(f"{condition}: {len(label_mismatch)} label mismatches")
        if code_mismatch:
            condition_problems.append(f"{condition}: {len(code_mismatch)} target-code mismatches")
        if start_mismatch:
            condition_problems.append(f"{condition}: {len(start_mismatch)} start-offset mismatches")
        if sorted(positions) != sorted(index_positions_required):
            condition_problems.append(
                f"{condition}: index_position is not the dense permutation 0..{len(ordered) - 1}"
            )
        if selections != sorted(selections) or len(set(selections)) != len(selections):
            condition_problems.append(
                f"{condition}: selection_position is not a unique increasing sequence"
            )
        if any(int(row["label"]) not in (0, 1) for row in subset):
            condition_problems.append(f"{condition}: label outside {0, 1}")
        if any(int(row["prediction"]) not in (0, 1) for row in subset):
            condition_problems.append(f"{condition}: prediction outside {0, 1}")
        if any(not math.isfinite(float(row["logit_margin"])) for row in subset):
            condition_problems.append(f"{condition}: non-finite logit_margin")
        # selection_position must address the same window as index_position.
        position_conflicts = 0
        for row in subset:
            reference = position_to_row.get(int(row["index_position"]))
            if reference is None:
                position_conflicts += 1
                continue
            if (reference["filename"], reference["window_index"]) != (
                row["filename"],
                row["window_index"],
            ):
                position_conflicts += 1
        if position_conflicts:
            condition_problems.append(
                f"{condition}: {position_conflicts} index_position -> window conflicts"
            )
        # The decision rule is margin > 0; verify the recorded prediction agrees.
        decision_conflicts = [
            row
            for row in subset
            if int(row["prediction"]) != (1 if float(row["logit_margin"]) > 0.0 else 0)
        ]
        if decision_conflicts:
            condition_problems.append(
                f"{condition}: {len(decision_conflicts)} rows violate prediction = margin > 0"
            )
        zero_margins = sum(1 for row in subset if float(row["logit_margin"]) == 0.0)
        per_condition[condition] = {
            "rows": len(subset),
            "unique_keys": len(key_counts),
            "duplicate_keys": len(duplicates),
            "missing_windows": len(missing),
            "unexpected_windows": len(unexpected),
            "files": len(files_seen),
            "background_files": len([f for f in files_seen if f.startswith(BACKGROUND_CODE)]),
            "drone_files": len([f for f in files_seen if not f.startswith(BACKGROUND_CODE)]),
            "label_counts": dict(sorted(Counter(int(r["label"]) for r in subset).items())),
            "label_mismatches": len(label_mismatch),
            "target_code_mismatches": len(code_mismatch),
            "start_mismatches": len(start_mismatch),
            "decision_rule_violations": len(decision_conflicts),
            "exact_zero_margins": zero_margins,
            "cross_role_files": [f"{name} ({role})" for name, role in cross_role_rows],
            "first_unexpected": [
                {"filename": key[0], "window_index": int(key[1])} for key in unexpected[:5]
            ],
            "problems": condition_problems,
        }
        problems.extend(condition_problems)
    return {
        "epoch": epoch,
        "status": "pass" if not problems else "fail",
        "conditions": per_condition,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# per-group breakdowns
# ---------------------------------------------------------------------------
def file_condition_breakdown(
    rows: Sequence[dict[str, Any]], reference: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """One record per (source file, condition) with margins and error counts."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["filename"]), str(row["condition"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (filename, condition), subset in sorted(grouped.items()):
        label = int(subset[0]["label"])
        margins = [float(row["logit_margin"]) for row in subset]
        errors = [row for row in subset if int(row["prediction"]) != int(row["label"])]
        stats = summary_stats(margins)
        error_margins = [float(row["logit_margin"]) for row in errors]
        if label == 1:
            # A drone window is missed when the margin is <= 0; the closest miss
            # is the largest margin among the misses.
            error_metric_name = "fn"
            closest_error_margin = max(error_margins) if error_margins else None
        else:
            error_metric_name = "fp"
            # A background window is a false alarm when the margin > 0; the most
            # confident false alarm has the largest margin.
            closest_error_margin = max(error_margins) if error_margins else None
        output.append(
            {
                "filename": filename,
                "target_code": str(subset[0]["target_code"]),
                "target_name": reference.get(filename, {}).get("target_name", ""),
                "label": label,
                "class_name": "Drone" if label == 1 else "Background",
                "condition": condition,
                "windows": len(subset),
                "errors": len(errors),
                "error_kind": error_metric_name,
                "error_rate": len(errors) / len(subset),
                "recall_or_specificity": 1.0 - len(errors) / len(subset),
                "closest_error_margin": closest_error_margin,
                "margin_min": stats["min"],
                "margin_p05": stats["p05"],
                "margin_p25": stats["p25"],
                "margin_median": stats["median"],
                "margin_p75": stats["p75"],
                "margin_p95": stats["p95"],
                "margin_max": stats["max"],
                "margin_mean": stats["mean"],
                "margin_std": stats["std"],
                "share_abs_margin_lt_1": float(
                    np.mean([abs(value) < 1.0 for value in margins])
                ),
                "share_abs_margin_lt_2": float(
                    np.mean([abs(value) < 2.0 for value in margins])
                ),
            }
        )
    return output


def model_breakdown(
    rows: Sequence[dict[str, Any]], reference: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """One record per (target code, condition)."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["target_code"]), str(row["condition"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (code, condition), subset in sorted(grouped.items()):
        label = int(subset[0]["label"])
        margins = [float(row["logit_margin"]) for row in subset]
        errors = [row for row in subset if int(row["prediction"]) != int(row["label"])]
        stats = summary_stats(margins)
        output.append(
            {
                "target_code": code,
                "target_name": reference.get(code, ""),
                "class_name": "Drone" if label == 1 else "Background",
                "condition": condition,
                "files": len({row["filename"] for row in subset}),
                "windows": len(subset),
                "errors": len(errors),
                "error_kind": "fn_miss" if label == 1 else "fp_false_alarm",
                "error_rate": len(errors) / len(subset),
                "recall_or_specificity": 1.0 - len(errors) / len(subset),
                "margin_min": stats["min"],
                "margin_p25": stats["p25"],
                "margin_median": stats["median"],
                "margin_p75": stats["p75"],
                "margin_max": stats["max"],
                "margin_mean": stats["mean"],
                "margin_std": stats["std"],
            }
        )
    return output


def model_overall(
    rows: Sequence[dict[str, Any]], reference: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """One record per target code, pooled over the four conditions."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["target_code"]), []).append(row)
    output: list[dict[str, Any]] = []
    for code, subset in sorted(grouped.items()):
        label = int(subset[0]["label"])
        errors = [row for row in subset if int(row["prediction"]) != int(row["label"])]
        stats = summary_stats([float(row["logit_margin"]) for row in subset])
        per_condition_rates = {}
        for condition in CONDITIONS:
            part = [row for row in subset if row["condition"] == condition]
            if not part:
                continue
            wrong = sum(1 for row in part if int(row["prediction"]) != int(row["label"]))
            per_condition_rates[condition] = 1.0 - wrong / len(part)
        output.append(
            {
                "target_code": code,
                "target_name": reference.get(code, ""),
                "class_name": "Drone" if label == 1 else "Background",
                "files": len({row["filename"] for row in subset}),
                "windows_pooled_over_conditions": len(subset),
                "windows_per_condition": len(subset) // len(CONDITIONS),
                "errors": len(errors),
                "error_rate": len(errors) / len(subset),
                "worst_condition_rate": min(per_condition_rates.values()),
                "worst_condition": min(per_condition_rates, key=per_condition_rates.get),
                **{f"rate_{condition}": per_condition_rates[condition] for condition in CONDITIONS},
                "margin_median": stats["median"],
                "margin_mean": stats["mean"],
            }
        )
    return output


def flip_analysis(
    scores_by_epoch: dict[int, dict[tuple[str, str, str], dict[str, Any]]]
) -> dict[str, Any]:
    """How many windows change decision between epochs (paired, same noise).

    ``label{0,1}_pred{a}_to_pred{b}`` counts are kept explicit so the identity
    ``delta_FP = label0_pred0_to_pred1 - label0_pred1_to_pred0`` and
    ``delta_FN = label1_pred1_to_pred0 - label1_pred0_to_pred1`` can be checked
    against the confusion counts of the two epochs independently.
    """
    pairs = [(1, 3), (3, 4), (4, 5), (3, 5), (2, 4)]
    counter_names = (
        "label0_pred0_to_pred1",
        "label0_pred1_to_pred0",
        "label1_pred0_to_pred1",
        "label1_pred1_to_pred0",
    )
    result: dict[str, Any] = {
        "note": (
            "validation features and AWGN realisations are cached once and reused "
            "for all five epochs, so these are paired comparisons on identical inputs"
        ),
        "pairs": {},
    }
    for earlier, later in pairs:
        left = scores_by_epoch[earlier]
        right = scores_by_epoch[later]
        if set(left) != set(right):
            raise RuntimeError(f"epoch {earlier} and {later} score keys differ")
        flips: dict[str, Any] = {
            "windows": len(left),
            "decision_changes": 0,
            **{name: 0 for name in counter_names},
            "by_target_code": {},
            "by_file": {},
        }
        for key, row_left in left.items():
            row_right = right[key]
            if int(row_left["label"]) != int(row_right["label"]):
                raise RuntimeError(f"label changed for {key}")
            if str(row_left["filename"]) != str(row_right["filename"]):
                raise RuntimeError(f"filename changed for {key}")
            label = int(row_left["label"])
            before = int(row_left["prediction"])
            after = int(row_right["prediction"])
            if before == after:
                continue
            flips["decision_changes"] += 1
            name = f"label{label}_pred{before}_to_pred{after}"
            flips[name] += 1
            code = str(row_left["target_code"])
            entry = flips["by_target_code"].setdefault(code, {key: 0 for key in counter_names})
            entry[name] += 1
            filename = str(row_left["filename"])
            entry = flips["by_file"].setdefault(filename, {key: 0 for key in counter_names})
            entry[name] += 1
        flips["background_to_drone"] = (
            flips["label0_pred0_to_pred1"] + flips["label1_pred0_to_pred1"]
        )
        flips["drone_to_background"] = (
            flips["label1_pred1_to_pred0"] + flips["label0_pred1_to_pred0"]
        )
        flips["delta_fp_predicted"] = (
            flips["label0_pred0_to_pred1"] - flips["label0_pred1_to_pred0"]
        )
        flips["delta_fn_predicted"] = (
            flips["label1_pred1_to_pred0"] - flips["label1_pred0_to_pred1"]
        )
        flips["by_target_code"] = dict(sorted(flips["by_target_code"].items()))
        flips["by_file"] = dict(sorted(flips["by_file"].items()))
        result["pairs"][f"epoch{earlier}_to_epoch{later}"] = flips
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def error_run_structure(
    rows: Sequence[dict[str, Any]], reference: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Are the errors scattered, or do they form contiguous blocks of windows?

    Windows are non-overlapping and consecutive inside a source file, so a run of
    consecutive ``window_index`` values that all fail is evidence of a *local*
    stretch of the recording (a transmission gap or a weak segment) rather than a
    uniformly marginal signal.  This uses only the recorded decisions.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["filename"]), str(row["condition"])), []).append(row)
    error_sets: dict[tuple[str, str], set[int]] = {}
    for key, subset in grouped.items():
        error_sets[key] = {
            int(row["window_index"])
            for row in subset
            if int(row["prediction"]) != int(row["label"])
        }
    output: list[dict[str, Any]] = []
    for (filename, condition), subset in sorted(grouped.items()):
        label = int(subset[0]["label"])
        errors = sorted(error_sets[(filename, condition)])
        runs: list[tuple[int, int]] = []
        for index in errors:
            if runs and index == runs[-1][1] + 1:
                runs[-1] = (runs[-1][0], index)
            else:
                runs.append((index, index))
        runs_sorted = sorted(runs, key=lambda item: item[1] - item[0], reverse=True)
        clean_errors = error_sets.get((filename, "clean"), set())
        error_set = set(errors)
        overlap = (
            len(error_set & clean_errors) / len(clean_errors) if clean_errors else None
        )
        output.append(
            {
                "filename": filename,
                "target_code": str(subset[0]["target_code"]),
                "class_name": "Drone" if label == 1 else "Background",
                "condition": condition,
                "windows": len(subset),
                "errors": len(errors),
                "error_rate": len(errors) / len(subset),
                "error_runs": len(runs),
                "longest_run": (runs_sorted[0][1] - runs_sorted[0][0] + 1) if runs_sorted else 0,
                "longest_run_start_window": runs_sorted[0][0] if runs_sorted else None,
                "longest_run_end_window": runs_sorted[0][1] if runs_sorted else None,
                "top_runs": ";".join(
                    f"{end - start + 1}@{start}" for start, end in runs_sorted[:3]
                ),
                "share_errors_in_runs_ge_3": (
                    sum(end - start + 1 for start, end in runs if end - start + 1 >= 3)
                    / len(error_set)
                    if error_set
                    else None
                ),
                "share_of_clean_errors_also_wrong_here": overlap,
                "target_name": reference.get(filename, {}).get("target_name", ""),
            }
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DIAG_DIR,
        help="diagnostics output directory (default: %(default)s)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the console table")
    args = parser.parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    identity: dict[str, Any] = {
        "run": "STAGE1-12DRONES-DIAG",
        "diagnosed_at": utc_now(),
        "scope": "read-only; no training, no test-role IQ, no modification of accepted artefacts",
        "paths": {
            "run_dir": str(RUN_DIR),
            "index_csv": str(INDEX_CSV),
            "seed_root": str(SEED_ROOT),
            "frozen_split_csv": str(FROZEN_SPLIT_CSV),
            "output_dir": str(output_dir),
        },
        "files": {},
        "checks": [],
    }
    failures: list[str] = []

    # -- frozen identities --------------------------------------------------
    for label, path, expected_key, expected_extra in (
        ("window_index_stage1_12drones.csv", INDEX_CSV, "index_sha256", {"rows": FROZEN["index_rows"]}),
        ("split_candidate_104_20260926.csv", FROZEN_SPLIT_CSV, "split_csv_sha256", {}),
        ("best_model.pt", SEED_ROOT / "best_model.pt", "best_model_sha256", {}),
    ):
        if not path.is_file():
            identity["files"][label] = {"path": str(path), "present": False}
            identity["checks"].append(
                {"check": f"{label} is present", "passed": False}
            )
            failures.append(f"missing frozen input {label} at {path}")
            continue
        digest = sha256_file(path)
        record = {"path": str(path), "sha256": digest, "expected_sha256": FROZEN[expected_key]}
        record["match"] = digest == FROZEN[expected_key]
        record.update(expected_extra)
        identity["files"][label] = record
        identity["checks"].append(
            {"check": f"{label} matches the frozen hash", "passed": bool(record["match"])}
        )
        if not record["match"]:
            failures.append(f"{label} sha256 {digest} != frozen {FROZEN[expected_key]}")

    scores_paths = {
        epoch: SEED_ROOT / f"validation_scores_epoch_{epoch:02d}.csv" for epoch in EPOCHS
    }
    for epoch, path in scores_paths.items():
        if not path.is_file():
            failures.append(f"missing score file {path}")
        else:
            identity["files"][path.name] = {"path": str(path), "sha256": sha256_file(path)}
    for name in ("validation_metrics.csv", "training_history.csv", "selection.json", "run_config.json"):
        path = SEED_ROOT / name
        if not path.is_file():
            failures.append(f"missing accepted artefact {path}")
        else:
            identity["files"][name] = {"path": str(path), "sha256": sha256_file(path)}
    write_json(output_dir / "input_identity.json", identity)
    if failures:
        write_json(
            output_dir / "diagnostics_summary.json",
            {"status": "fail", "stage": "input identity", "problems": failures},
        )
        print("FAILED during input identity checks:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1

    # -- frozen index / val split ------------------------------------------
    all_rows, val_rows = load_index(INDEX_CSV)
    expected, ordered, meta = build_expected_val_table(val_rows)
    reference_by_file = {row["filename"]: row for row in all_rows}
    reference_by_code = {row["target_code"]: row["target_name"] for row in all_rows}
    role_counts = Counter(row["source_role"] for row in all_rows)
    identity["val_split"] = {
        **meta,
        "index_role_counts": dict(sorted(role_counts.items())),
        "test_roles_present_in_index": sorted({r for r in role_counts if r in TEST_ROLES}),
    }
    write_json(output_dir / "input_identity.json", identity)

    if meta["val_rows"] != VAL_WINDOWS or meta["val_files"] != VAL_FILES:
        failures.append(
            f"frozen val split has {meta['val_rows']} windows / {meta['val_files']} files, "
            f"expected {VAL_WINDOWS} / {VAL_FILES}"
        )
    if meta["background_files"] != BACKGROUND_FILES:
        failures.append(
            f"frozen val split has {meta['background_files']} background files, "
            f"expected {BACKGROUND_FILES}"
        )
    if meta["label_counts"] != {"0": 280, "1": 2080}:
        failures.append(f"unexpected val label counts {meta['label_counts']}")

    # -- load scores, verify integrity --------------------------------------
    integrity: dict[str, Any] = {
        "expected": {
            "epochs": list(EPOCHS),
            "conditions": list(CONDITIONS),
            "windows_per_epoch_per_condition": VAL_WINDOWS,
            "files_per_epoch_per_condition": VAL_FILES,
        },
        "epochs": {},
        "problems": [],
    }
    scores: dict[int, list[dict[str, Any]]] = {}
    role_by_file = {row["filename"]: row["source_role"] for row in all_rows}
    for epoch in EPOCHS:
        rows = load_score_file(scores_paths[epoch])
        scores[epoch] = rows
        report = check_score_integrity(epoch, rows, expected, ordered, meta, role_by_file)
        integrity["epochs"][f"epoch_{epoch:02d}"] = report
        integrity["problems"].extend(report["problems"])
    integrity["status"] = "pass" if not integrity["problems"] else "fail"
    write_json(output_dir / "score_file_integrity.json", integrity)
    if integrity["problems"]:
        write_json(
            output_dir / "diagnostics_summary.json",
            {"status": "fail", "stage": "score file integrity", "problems": integrity["problems"]},
        )
        print("FAILED during score-file integrity checks:", file=sys.stderr)
        for item in integrity["problems"]:
            print(f"  - {item}", file=sys.stderr)
        return 1

    # -- independent metric recomputation ----------------------------------
    stored_metrics = {
        (int(row["epoch"]), row["condition"]): row for row in read_csv_rows(SEED_ROOT / "validation_metrics.csv")
    }
    if len(stored_metrics) != len(EPOCHS) * len(CONDITIONS):
        failures.append(
            f"validation_metrics.csv has {len(stored_metrics)} epoch/condition rows, "
            f"expected {len(EPOCHS) * len(CONDITIONS)}"
        )
    recomputed_rows: list[dict[str, Any]] = []
    recon: dict[str, Any] = {"tolerance": METRIC_TOLERANCE, "epochs": {}, "problems": []}
    max_delta_overall = 0.0
    for epoch in EPOCHS:
        epoch_report: dict[str, Any] = {"conditions": {}, "problems": []}
        for condition in CONDITIONS:
            subset = [
                row
                for row in scores[epoch]
                if row["condition"] == condition
            ]
            pooled = pooled_metrics(
                [int(row["label"]) for row in subset], [int(row["prediction"]) for row in subset]
            )
            by_file = file_equal_metrics(subset)
            recomputed = {**pooled, **by_file, "samples": len(subset), "loss": None}
            stored = stored_metrics.get((epoch, condition))
            if stored is None:
                epoch_report["problems"].append(f"epoch {epoch} {condition}: no stored metric row")
                continue
            float_deltas = {
                field: abs(float(stored[field]) - float(recomputed[field]))
                for field in METRIC_FLOAT_FIELDS
                if field != "loss"
            }
            int_deltas = {
                field: int(stored[field]) - int(recomputed[field]) for field in METRIC_INT_FIELDS
            }
            worst = max(float_deltas.values())
            max_delta_overall = max(max_delta_overall, worst)
            counts_ok = all(delta == 0 for delta in int_deltas.values())
            floats_ok = worst <= METRIC_TOLERANCE
            if not floats_ok:
                epoch_report["problems"].append(
                    f"epoch {epoch} {condition}: max float delta {worst:.3e} exceeds tolerance"
                )
            if not counts_ok:
                epoch_report["problems"].append(
                    f"epoch {epoch} {condition}: count mismatch {int_deltas}"
                )
            epoch_report["conditions"][condition] = {
                "windows": recomputed["windows"],
                "max_abs_float_delta": worst,
                "count_deltas": int_deltas,
                "counts_match": counts_ok,
                "floats_match": floats_ok,
                "recomputed": {key: recomputed[key] for key in (*METRIC_FLOAT_FIELDS, *METRIC_INT_FIELDS)},
            }
            row_out: dict[str, Any] = {"epoch": epoch, "condition": condition}
            for field in (*METRIC_FLOAT_FIELDS, *METRIC_INT_FIELDS):
                row_out[f"recomputed_{field}"] = recomputed[field]
                if field in stored:
                    row_out[f"stored_{field}"] = float(stored[field]) if field in METRIC_FLOAT_FIELDS else int(stored[field])
                    if field != "loss":
                        row_out[f"delta_{field}"] = (
                            float(stored[field]) - float(recomputed[field])
                        )
            recomputed_rows.append(row_out)
        recon["epochs"][f"epoch_{epoch:02d}"] = epoch_report
        recon["problems"].extend(epoch_report["problems"])
    recon["status"] = "pass" if not recon["problems"] else "fail"
    recon["max_abs_float_delta"] = max_delta_overall
    write_json(output_dir / "metric_reconciliation.json", recon)

    write_csv(
        output_dir / "validation_metrics_recomputed.csv",
        recomputed_rows,
        list(recomputed_rows[0].keys()),
    )

    # -- epoch trajectory ---------------------------------------------------
    trajectory: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        for condition in CONDITIONS:
            subset = [row for row in scores[epoch] if row["condition"] == condition]
            pooled = pooled_metrics(
                [int(row["label"]) for row in subset], [int(row["prediction"]) for row in subset]
            )
            by_file = file_equal_metrics(subset)
            background_margins = [
                float(row["logit_margin"]) for row in subset if int(row["label"]) == 0
            ]
            drone_margins = [float(row["logit_margin"]) for row in subset if int(row["label"]) == 1]
            trajectory.append(
                {
                    "epoch": epoch,
                    "condition": condition,
                    "windows": pooled["windows"],
                    "balanced_accuracy": pooled["balanced_accuracy"],
                    "file_equal_balanced_accuracy": by_file["file_equal_balanced_accuracy"],
                    "detection_auc": detection_auc(drone_margins, background_margins),
                    "d_prime": d_prime(drone_margins, background_margins),
                    "drone_recall": pooled["drone_recall"],
                    "background_specificity": pooled["background_specificity"],
                    "file_equal_drone_recall": by_file["file_equal_drone_recall"],
                    "file_equal_background_recall": by_file["file_equal_background_recall"],
                    "precision": pooled["precision"],
                    "f1": pooled["f1"],
                    "accuracy": pooled["accuracy"],
                    "tn": pooled["tn"],
                    "fp": pooled["fp"],
                    "fn": pooled["fn"],
                    "tp": pooled["tp"],
                    "background_margin_median": float(np.median(background_margins)),
                    "background_margin_mean": float(np.mean(background_margins)),
                    "background_share_margin_gt_0": float(
                        np.mean([value > 0.0 for value in background_margins])
                    ),
                    "drone_margin_median": float(np.median(drone_margins)),
                    "drone_margin_mean": float(np.mean(drone_margins)),
                    "drone_share_margin_gt_0": float(
                        np.mean([value > 0.0 for value in drone_margins])
                    ),
                    "share_abs_margin_lt_1": float(
                        np.mean([abs(float(row["logit_margin"])) < 1.0 for row in subset])
                    ),
                }
            )
    write_csv(output_dir / "epoch_trajectory_by_condition.csv", trajectory, list(trajectory[0].keys()))

    separability_rows = [
        {
            "epoch": epoch,
            "condition": row["condition"],
            "detection_auc": row["detection_auc"],
            "d_prime": row["d_prime"],
            "balanced_accuracy_at_threshold_zero": row["balanced_accuracy"],
            "drone_recall_at_threshold_zero": row["drone_recall"],
            "background_specificity_at_threshold_zero": row["background_specificity"],
            "background_margin_mean": row["background_margin_mean"],
            "drone_margin_mean": row["drone_margin_mean"],
            "margin_mean_gap": row["drone_margin_mean"] - row["background_margin_mean"],
        }
        for epoch in EPOCHS
        for row in trajectory
        if row["epoch"] == epoch
    ]
    write_csv(
        output_dir / "epoch_threshold_free_separability.csv",
        separability_rows,
        list(separability_rows[0].keys()),
    )

    per_epoch_mean = {}
    for epoch in EPOCHS:
        part = [row for row in trajectory if row["epoch"] == epoch]
        per_epoch_mean[f"epoch_{epoch:02d}"] = {
            "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in part])),
            "mean_file_equal_balanced_accuracy": float(
                np.mean([row["file_equal_balanced_accuracy"] for row in part])
            ),
            "mean_accuracy": float(np.mean([row["accuracy"] for row in part])),
            "total_errors_over_four_conditions": int(
                sum(row["fp"] + row["fn"] for row in part)
            ),
            "total_fp_over_four_conditions": int(sum(row["fp"] for row in part)),
            "total_fn_over_four_conditions": int(sum(row["fn"] for row in part)),
            "mean_drone_recall": float(np.mean([row["drone_recall"] for row in part])),
            "mean_background_specificity": float(
                np.mean([row["background_specificity"] for row in part])
            ),
            "mean_detection_auc": float(np.mean([row["detection_auc"] for row in part])),
            "mean_d_prime": float(np.mean([row["d_prime"] for row in part])),
        }
    metric_ranking = {
        "by_mean_pooled_balanced_accuracy": [
            int(epoch)
            for epoch in sorted(
                EPOCHS,
                key=lambda e: per_epoch_mean[f"epoch_{e:02d}"]["mean_balanced_accuracy"],
                reverse=True,
            )
        ],
        "by_mean_pooled_accuracy": [
            int(epoch)
            for epoch in sorted(
                EPOCHS,
                key=lambda e: per_epoch_mean[f"epoch_{e:02d}"]["mean_accuracy"],
                reverse=True,
            )
        ],
        "by_total_errors_over_four_conditions": [
            int(epoch)
            for epoch in sorted(
                EPOCHS,
                key=lambda e: per_epoch_mean[f"epoch_{e:02d}"][
                    "total_errors_over_four_conditions"
                ],
            )
        ],
        "by_mean_detection_auc": [
            int(epoch)
            for epoch in sorted(
                EPOCHS,
                key=lambda e: per_epoch_mean[f"epoch_{e:02d}"]["mean_detection_auc"],
                reverse=True,
            )
        ],
        "note": (
            "the accepted selection rule ranks epochs by the first of these; the four "
            "rankings disagree, so the recorded 'epoch 3 is best' is a statement about "
            "that one metric and operating point, not about overall quality"
        ),
    }

    # -- per-file trajectory ------------------------------------------------
    per_file_rows: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        for condition in CONDITIONS:
            subset = [row for row in scores[epoch] if row["condition"] == condition]
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in subset:
                grouped.setdefault(str(row["filename"]), []).append(row)
            for filename, rows_for_file in sorted(grouped.items()):
                label = int(rows_for_file[0]["label"])
                errors = sum(
                    1 for row in rows_for_file if int(row["prediction"]) != int(row["label"])
                )
                margins = [float(row["logit_margin"]) for row in rows_for_file]
                per_file_rows.append(
                    {
                        "epoch": epoch,
                        "condition": condition,
                        "filename": filename,
                        "target_code": str(rows_for_file[0]["target_code"]),
                        "label": label,
                        "class_name": "Drone" if label == 1 else "Background",
                        "windows": len(rows_for_file),
                        "errors": errors,
                        "rate": 1.0 - errors / len(rows_for_file),
                        "margin_min": min(margins),
                        "margin_median": float(np.median(margins)),
                        "margin_max": max(margins),
                        "margin_mean": float(np.mean(margins)),
                    }
                )
    write_csv(output_dir / "epoch_per_file_metrics.csv", per_file_rows, list(per_file_rows[0].keys()))

    # -- epoch 3 detail -----------------------------------------------------
    selected = scores[SELECTED_EPOCH]
    file_breakdown = file_condition_breakdown(selected, reference_by_file)
    write_csv(
        output_dir / "epoch03_file_condition_breakdown.csv",
        file_breakdown,
        list(file_breakdown[0].keys()),
    )
    model_rows = model_breakdown(selected, reference_by_code)
    write_csv(output_dir / "epoch03_model_by_condition.csv", model_rows, list(model_rows[0].keys()))
    model_overall_rows = model_overall(selected, reference_by_code)
    write_csv(
        output_dir / "epoch03_model_overall.csv",
        model_overall_rows,
        list(model_overall_rows[0].keys()),
    )
    run_rows = error_run_structure(selected, reference_by_file)
    write_csv(output_dir / "epoch03_error_runs.csv", run_rows, list(run_rows[0].keys()))

    # -- paired flip analysis ----------------------------------------------
    keyed: dict[int, dict[tuple[str, str, str], dict[str, Any]]] = {}
    for epoch in EPOCHS:
        table: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in scores[epoch]:
            key = (str(row["condition"]), str(row["filename"]), str(row["window_index"]))
            if key in table:
                raise RuntimeError(f"duplicate score key {key} in epoch {epoch}")
            table[key] = row
        keyed[epoch] = table
    flips = flip_analysis(keyed)
    write_json(output_dir / "epoch_flip_analysis.json", flips)

    # -- threshold / operating point diagnostic ----------------------------
    thresholds: dict[str, Any] = {"epochs": {}}
    for epoch in EPOCHS:
        per_condition = {}
        for condition in CONDITIONS:
            subset = [row for row in scores[epoch] if row["condition"] == condition]
            per_condition[condition] = {
                "drone_margins": [
                    float(row["logit_margin"]) for row in subset if int(row["label"]) == 1
                ],
                "background_margins": [
                    float(row["logit_margin"]) for row in subset if int(row["label"]) == 0
                ],
            }
        analysis = oracle_thresholds(per_condition)
        recorded = {
            condition: float(
                next(
                    row["balanced_accuracy"]
                    for row in trajectory
                    if row["epoch"] == epoch and row["condition"] == condition
                )
            )
            for condition in CONDITIONS
        }
        analysis["recorded_mean_balanced_accuracy_at_zero"] = float(np.mean(list(recorded.values())))
        thresholds["epochs"][f"epoch_{epoch:02d}"] = analysis
    write_json(output_dir / "threshold_diagnostic.json", thresholds)

    # Only meaningful once the per-condition optima exist.
    metric_ranking["by_mean_per_condition_oracle_balanced_accuracy"] = [
        int(epoch)
        for epoch in sorted(
            EPOCHS,
            key=lambda e: float(
                np.mean(
                    [
                        item["balanced_accuracy"]
                        for item in thresholds["epochs"][f"epoch_{e:02d}"][
                            "per_condition_oracle"
                        ].values()
                    ]
                )
            ),
            reverse=True,
        )
    ]

    # -- consolidated summary ----------------------------------------------
    epoch3_by_condition = {
        row["condition"]: {
            "balanced_accuracy": row["balanced_accuracy"],
            "file_equal_balanced_accuracy": row["file_equal_balanced_accuracy"],
            "drone_recall": row["drone_recall"],
            "background_specificity": row["background_specificity"],
            "fp": row["fp"],
            "fn": row["fn"],
            "tn": row["tn"],
            "tp": row["tp"],
        }
        for row in trajectory
        if row["epoch"] == SELECTED_EPOCH
    }
    worst_model = min(
        (row for row in model_overall_rows if row["class_name"] == "Drone"),
        key=lambda row: row["error_rate"],
    )
    best_model = max(
        (row for row in model_overall_rows if row["class_name"] == "Drone"),
        key=lambda row: row["error_rate"],
    )
    worst_condition = min(
        epoch3_by_condition.items(), key=lambda item: item[1]["balanced_accuracy"]
    )
    background_overall = [row for row in model_overall_rows if row["class_name"] == "Background"]
    files_by_error: dict[str, dict[str, Any]] = {}
    for row in file_breakdown:
        entry = files_by_error.setdefault(
            row["filename"],
            {
                "filename": row["filename"],
                "target_code": row["target_code"],
                "class_name": row["class_name"],
                "windows": row["windows"],
                "errors_by_condition": {},
                "error_rate_mean_over_conditions": 0.0,
            },
        )
        entry["errors_by_condition"][row["condition"]] = row["errors"]
        entry["error_rate_mean_over_conditions"] += row["error_rate"] / len(CONDITIONS)
    ranked = sorted(
        files_by_error.values(), key=lambda entry: entry["error_rate_mean_over_conditions"], reverse=True
    )
    summary = {
        "task": "STAGE1-12DRONES-DIAG",
        "status": "completed" if not failures else "fail",
        "generated_at": utc_now(),
        "frozen_inputs_verified": True,
        "score_integrity": {
            "status": integrity["status"],
            "windows_per_epoch_per_condition": VAL_WINDOWS,
            "files_per_epoch_per_condition": VAL_FILES,
            "epochs_checked": list(EPOCHS),
        },
        "metric_reconciliation": {
            "status": recon["status"],
            "max_abs_float_delta": recon["max_abs_float_delta"],
            "tolerance": METRIC_TOLERANCE,
        },
        "selected_epoch": SELECTED_EPOCH,
        "selection_rule": SELECTION_RULE,
        "epoch3_balanced_accuracy_by_condition": epoch3_by_condition,
        "epoch_means": per_epoch_mean,
        "metric_ranking": metric_ranking,
        "worst_condition_epoch3": {
            "condition": worst_condition[0],
            **worst_condition[1],
        },
        "worst_drone_model_epoch3": worst_model,
        "best_drone_model_epoch3": best_model,
        "background_epoch3": background_overall,
        "epoch3_files_ranked_by_mean_error_rate": ranked,
        "epoch3_error_structure": {
            row["filename"] + "|" + row["condition"]: {
                "errors": row["errors"],
                "error_runs": row["error_runs"],
                "longest_run": row["longest_run"],
                "longest_run_start_window": row["longest_run_start_window"],
                "share_errors_in_runs_ge_3": row["share_errors_in_runs_ge_3"],
                "share_of_clean_errors_also_wrong_here": row[
                    "share_of_clean_errors_also_wrong_here"
                ],
            }
            for row in run_rows
            if row["errors"]
        },
        "epoch3_measured_quantities": {
            "val_background_files": BACKGROUND_FILES,
            "val_background_windows": 280,
            "val_drone_files": DRONE_FILES,
            "val_drone_windows": 2080,
            "background_windows_per_specificity_point": 1 / 280,
            "background_file_specificity_quantum": 1 / 140,
            "note": (
                "per-T-code rates are drone-vs-background detection recall, not 12-way "
                "model identification accuracy"
            ),
        },
        "training_snr_mixture": TRAIN_SNRS,
        "validation_conditions": list(CONDITIONS),
        "flip_analysis": {
            pair: {
                "decision_changes": payload["decision_changes"],
                "background_to_drone": payload["background_to_drone"],
                "drone_to_background": payload["drone_to_background"],
                "label0_pred0_to_pred1": payload["label0_pred0_to_pred1"],
                "label0_pred1_to_pred0": payload["label0_pred1_to_pred0"],
                "label1_pred0_to_pred1": payload["label1_pred0_to_pred1"],
                "label1_pred1_to_pred0": payload["label1_pred1_to_pred0"],
                "delta_fp_predicted": payload["delta_fp_predicted"],
                "delta_fn_predicted": payload["delta_fn_predicted"],
            }
            for pair, payload in flips["pairs"].items()
        },
        "threshold_diagnostic_summary": {
            epoch: {
                "recorded_mean_ba_at_zero": payload["recorded_mean_balanced_accuracy_at_zero"],
                "shared_oracle_threshold": payload["shared_threshold_oracle"]["threshold"],
                "shared_oracle_mean_ba": payload["shared_threshold_oracle"]["mean_balanced_accuracy"],
                "mean_per_condition_oracle_ba": float(
                    np.mean(
                        [
                            item["balanced_accuracy"]
                            for item in payload["per_condition_oracle"].values()
                        ]
                    )
                ),
                "max_per_condition_oracle_gain": float(
                    max(item["gain"] for item in payload["per_condition_oracle"].values())
                ),
            }
            for epoch, payload in thresholds["epochs"].items()
        },
        "threshold_free_separability": {
            f"epoch_{epoch:02d}": {
                "mean_detection_auc": per_epoch_mean[f"epoch_{epoch:02d}"]["mean_detection_auc"],
                "mean_d_prime": per_epoch_mean[f"epoch_{epoch:02d}"]["mean_d_prime"],
                "auc_by_condition": {
                    row["condition"]: row["detection_auc"]
                    for row in trajectory
                    if row["epoch"] == epoch
                },
            }
            for epoch in EPOCHS
        },
        "problems": failures,
    }
    write_json(output_dir / "diagnostics_summary.json", summary)

    if not args.quiet:
        print("epoch  cond    BA       feBA     AUC      droneR   bgSpec   tn  fp   fn   tp")
        for row in trajectory:
            print(
                f"{row['epoch']:>5}  {row['condition']:<6}  "
                f"{row['balanced_accuracy']:.6f} {row['file_equal_balanced_accuracy']:.6f} "
                f"{row['detection_auc']:.6f} "
                f"{row['drone_recall']:.6f} {row['background_specificity']:.6f} "
                f"{row['tn']:>4} {row['fp']:>4} {row['fn']:>4} {row['tp']:>5}"
            )
        print()
        print("epoch  mean BA@0   mean acc   mean AUC   mean d'   errors(4 cond)  shared-threshold oracle")
        for epoch in EPOCHS:
            label = f"epoch_{epoch:02d}"
            payload = thresholds["epochs"][label]
            means = per_epoch_mean[label]
            print(
                f"{epoch:>5}  {means['mean_balanced_accuracy']:.6f}  "
                f"{means['mean_accuracy']:.6f}  "
                f"{means['mean_detection_auc']:.6f}  "
                f"{means['mean_d_prime']:.4f}  "
                f"{means['total_errors_over_four_conditions']:>6} (FP {means['total_fp_over_four_conditions']:>4} / FN {means['total_fn_over_four_conditions']:>4})  "
                f"t={payload['shared_threshold_oracle']['threshold']:+.4f} -> "
                f"{payload['shared_threshold_oracle']['mean_balanced_accuracy']:.6f}"
            )
        print()
        print("ranking by mean pooled BA :", metric_ranking["by_mean_pooled_balanced_accuracy"])
        print("ranking by mean accuracy  :", metric_ranking["by_mean_pooled_accuracy"])
        print("ranking by total errors   :", metric_ranking["by_total_errors_over_four_conditions"])
        print("ranking by mean AUC       :", metric_ranking["by_mean_detection_auc"])
        print(
            "ranking by per-cond oracle:",
            metric_ranking["by_mean_per_condition_oracle_balanced_accuracy"],
        )
        print()
        print("epoch3 worst drone models (pooled over 4 conditions, error rate descending):")
        for row in sorted(
            (r for r in model_overall_rows if r["class_name"] == "Drone"),
            key=lambda r: r["error_rate"],
            reverse=True,
        )[:5]:
            print(
                f"  {row['target_code']} {row['target_name']:<22} "
                f"windows={row['windows_pooled_over_conditions']:>4} errors={row['errors']:>3} "
                f"rate={row['error_rate']:.4f} worst={row['worst_condition']}"
            )
        print()
        print(f"metric reconciliation: {recon['status']} (max abs float delta {max_delta_overall:.3e})")
        print(f"score integrity:       {integrity['status']}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
