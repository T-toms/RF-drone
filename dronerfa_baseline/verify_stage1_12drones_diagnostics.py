#!/usr/bin/env python3
"""Independent verification of the STAGE1-12DRONES-DIAG diagnostics.

This script deliberately does *not* import ``diagnose_stage1_12drones_validation``
(there are no imports from it, and no shared helpers); it re-derives everything
from the frozen split CSV, the frozen window index and the recorded per-window
score files, then compares its own numbers with the published diagnostics.

It also re-implements the metric/separability maths with *different* algorithms
than the diagnostic script (descending single-pass threshold sweep instead of
midpoint candidates + ``searchsorted``, pairwise comparison instead of the
Mann-Whitney rank statistic), so a shared implementation bug cannot agree with
itself.

Checks:

A  frozen input identity (split CSV, index, checkpoint, score files)
B  score files == the frozen val split, with unique keys, matching role, label,
   target code and start offset for all five epochs and four conditions
C  stored ``validation_metrics.csv`` re-derives from the per-window decisions
D  every published diagnostics artefact agrees with this script's own numbers
E  the flip analysis satisfies the exact dFP / dFN bookkeeping identity
F  no test-role MAT was ever opened (read audit of the accepted training run)
G  negative controls: tampered inputs and degenerate scores must be rejected

Read-only.  Writes only ``diagnostics/diagnostics_verification.json``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUN_DIR = SCRIPT_DIR / "outputs" / "stage1_12drones_20260926"
DIAG_DIR = RUN_DIR / "diagnostics"
SEED_ROOT = RUN_DIR / "seed_2026"
INDEX_CSV = RUN_DIR / "window_index_stage1_12drones.csv"
FROZEN_SPLIT_CSV = PROJECT_ROOT / "work" / "meta_split2" / "split_candidate_104_20260926.csv"

FROZEN = {
    "index_sha256": "5973a2fa5b8476a507fcc7a665365daf8d572f569ce5908d54f5c4bc84c48bd6",
    "split_csv_sha256": "d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9",
    "best_model_sha256": "37dbb815450b8ec0f798a113a7f922858fcb2be604a863eb7594eb5c52ff814b",
}
VAL_ROLE = "val"
TEST_ROLES = ("development_test", "development_test_exposed")
CONDITIONS = ("clean", "-10dB", "-12dB", "-15dB")
EPOCHS = (1, 2, 3, 4, 5)
SELECTED_EPOCH = 3
VAL_WINDOWS = 2360
VAL_FILES = 16
TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
class Checks:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.failed = 0

    def add(self, name: str, passed: bool, evidence: Any = None) -> bool:
        self.records.append({"check": name, "passed": bool(passed), "evidence": evidence})
        if not passed:
            self.failed += 1
        return bool(passed)

    def payload(self) -> dict[str, Any]:
        return {
            "checks": self.records,
            "passed": self.failed == 0,
            "failed": self.failed,
            "total": len(self.records),
        }


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
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


# ---------------------------------------------------------------------------
# independent metric maths
# ---------------------------------------------------------------------------
def confusion(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, int]:
    tp = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 0)
    if tp + tn + fp + fn != len(labels):
        raise ValueError("confusion matrix does not conserve the window count")
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def derived_metrics(counts: dict[str, int]) -> dict[str, float]:
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    total = tp + tn + fp + fn
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "drone_recall": recall,
        "background_specificity": specificity,
        "precision": precision,
        "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
    }


def file_equal_balanced_accuracy(rows: Sequence[dict[str, Any]]) -> float:
    per_file: dict[str, tuple[int, int, int]] = {}
    for row in rows:
        label = int(row["label"])
        correct = 1 if int(row["prediction"]) == label else 0
        seen, hits, count = per_file.get(str(row["filename"]), (label, 0, 0))
        if seen != label:
            raise ValueError(f"{row['filename']} carries two labels")
        per_file[str(row["filename"])] = (label, hits + correct, count + 1)
    by_class: dict[int, list[float]] = {0: [], 1: []}
    for label, hits, count in per_file.values():
        by_class[label].append(hits / count)
    return 0.5 * (float(np.mean(by_class[0])) + float(np.mean(by_class[1])))


def auc_pairwise(drone: Sequence[float], background: Sequence[float]) -> float:
    """AUC via explicit pairwise comparison (not the rank statistic)."""
    pos = np.asarray(drone, dtype=np.float64)[:, None]
    neg = np.asarray(background, dtype=np.float64)[None, :]
    greater = float(np.mean(pos > neg))
    ties = float(np.mean(pos == neg))
    return greater + 0.5 * ties


def sweep_curves(
    labels: np.ndarray, margins: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Descending single-pass sweep over "predict the top-k margins as drone".

    Returns the candidate boundary values (largest first), the achieved balanced
    accuracy at each boundary, and the recall/specificity pair at each boundary.
    """
    order = np.argsort(-margins, kind="mergesort")
    sorted_labels = labels[order]
    sorted_values = margins[order]
    n_positive = int(np.sum(labels == 1))
    n_negative = int(np.sum(labels == 0))
    if n_positive == 0 or n_negative == 0:
        raise ValueError("balanced accuracy needs both classes")
    tp = 0
    fp = 0
    best_values: list[float] = []
    best_scores: list[float] = []
    pairs: list[tuple[float, float]] = []
    # k = number of windows predicted drone; k = 0 has BA 0.5 by convention here.
    best_values.append(float(sorted_values[0]) + 1.0 if sorted_values.size else 1.0)
    best_scores.append(0.5 * (0.0 + 1.0))
    pairs.append((0.0, 1.0))
    for index in range(sorted_values.size):
        if int(sorted_labels[index]) == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / n_positive
        specificity = (n_negative - fp) / n_negative
        # The boundary is the value at this position: predicting drone for every
        # margin >= this value is exactly the first (index + 1) windows.
        if index + 1 < sorted_values.size:
            boundary = float(sorted_values[index + 1])
        else:
            boundary = float(sorted_values[index]) - 1.0
        best_values.append(boundary)
        best_scores.append(0.5 * (recall + specificity))
        pairs.append((recall, specificity))
    return (
        np.asarray(best_values, dtype=np.float64),
        np.asarray(best_scores, dtype=np.float64),
        np.asarray(pairs, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# score integrity (independent implementation)
# ---------------------------------------------------------------------------
def integrity_problems(
    rows: Sequence[dict[str, Any]],
    expected: dict[tuple[str, str], dict[str, str]],
    role_by_file: dict[str, str],
) -> list[str]:
    problems: list[str] = []
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        keys = [(row["filename"], str(row["window_index"])) for row in subset]
        if len(subset) != VAL_WINDOWS:
            problems.append(f"{condition}: {len(subset)} rows != {VAL_WINDOWS}")
        if len(set(keys)) != len(keys):
            problems.append(f"{condition}: duplicate window keys")
        missing = set(expected) - set(keys)
        if missing:
            problems.append(f"{condition}: {len(missing)} frozen val windows missing")
        extra = set(keys) - set(expected)
        if extra:
            roles = sorted({role_by_file.get(name, "unknown") for name, _ in extra})
            problems.append(f"{condition}: {len(extra)} rows outside the frozen val split {roles}")
        files = {row["filename"] for row in subset}
        if len(files) != VAL_FILES:
            problems.append(f"{condition}: {len(files)} files != {VAL_FILES}")
        for row in subset:
            key = (row["filename"], str(row["window_index"]))
            reference = expected.get(key)
            if reference is None:
                continue
            if int(reference["label"]) != int(row["label"]):
                problems.append(f"{condition}: label mismatch at {key}")
            if reference["target_code"] != row["target_code"]:
                problems.append(f"{condition}: target code mismatch at {key}")
            if int(reference["start"]) != int(row["start"]):
                problems.append(f"{condition}: start mismatch at {key}")
        if any(int(row["prediction"]) != (1 if float(row["logit_margin"]) > 0.0 else 0) for row in subset):
            problems.append(f"{condition}: prediction does not follow margin > 0")
    return problems


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    checks = Checks()
    result: dict[str, Any] = {
        "run": "STAGE1-12DRONES-DIAG-verify",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "scope": "read-only verification of the diagnostics artefacts",
    }

    # A. frozen identities
    identities = {
        "split_candidate_104_20260926.csv": (FROZEN_SPLIT_CSV, FROZEN["split_csv_sha256"]),
        "window_index_stage1_12drones.csv": (INDEX_CSV, FROZEN["index_sha256"]),
        "best_model.pt": (SEED_ROOT / "best_model.pt", FROZEN["best_model_sha256"]),
    }
    digest_report = {}
    for name, (path, expected_hash) in identities.items():
        digest = sha256_file(path)
        digest_report[name] = digest
        checks.add(f"A {name} matches the frozen sha256", digest == expected_hash, digest)
    result["frozen_hashes"] = digest_report

    scores_hashes = {}
    for epoch in EPOCHS:
        path = SEED_ROOT / f"validation_scores_epoch_{epoch:02d}.csv"
        scores_hashes[path.name] = sha256_file(path)
    result["score_file_hashes"] = scores_hashes
    checks.add("A all five per-epoch score files exist", len(scores_hashes) == 5, sorted(scores_hashes))

    # B. score files == frozen val split
    index_rows = read_csv_rows(INDEX_CSV)
    checks.add("A frozen index holds 15,790 rows", len(index_rows) == 15790, len(index_rows))
    role_by_file = {row["filename"]: row["source_role"] for row in index_rows}
    expected: dict[tuple[str, str], dict[str, str]] = {}
    for row in index_rows:
        if row["source_role"] != VAL_ROLE:
            continue
        expected[(row["filename"], str(int(row["window_index"])))] = row
    checks.add(
        "B frozen val split is 2,360 windows over 16 files",
        len(expected) == VAL_WINDOWS and len({name for name, _ in expected}) == VAL_FILES,
        {"windows": len(expected), "files": len({name for name, _ in expected})},
    )

    scores_by_epoch: dict[int, list[dict[str, Any]]] = {}
    integrity_evidence: dict[str, Any] = {}
    all_problems: list[str] = []
    for epoch in EPOCHS:
        rows = []
        for raw in read_csv_rows(SEED_ROOT / f"validation_scores_epoch_{epoch:02d}.csv"):
            rows.append(
                {
                    "condition": str(raw["condition"]),
                    "filename": str(raw["filename"]),
                    "target_code": str(raw["target_code"]),
                    "window_index": int(raw["window_index"]),
                    "start": int(raw["start"]),
                    "label": int(raw["label"]),
                    "logit_margin": float(raw["logit_margin"]),
                    "prediction": int(raw["prediction"]),
                    "index_position": int(raw["index_position"]),
                }
            )
        scores_by_epoch[epoch] = rows
        problems = integrity_problems(rows, expected, role_by_file)
        integrity_evidence[f"epoch_{epoch:02d}"] = {
            "rows": len(rows),
            "problems": problems[:5],
            "problem_count": len(problems),
        }
        all_problems.extend(f"epoch {epoch}: {problem}" for problem in problems)
    checks.add(
        "B every epoch/condition score file matches the frozen val split exactly",
        not all_problems,
        integrity_evidence,
    )
    result["score_integrity"] = {"problems": all_problems, "evidence": integrity_evidence}

    # C. stored validation metrics re-derive
    stored = {
        (int(row["epoch"]), row["condition"]): row
        for row in read_csv_rows(SEED_ROOT / "validation_metrics.csv")
    }
    own: dict[tuple[int, str], dict[str, Any]] = {}
    for epoch in EPOCHS:
        for condition in CONDITIONS:
            subset = [row for row in scores_by_epoch[epoch] if row["condition"] == condition]
            counts = confusion(
                [int(row["label"]) for row in subset], [int(row["prediction"]) for row in subset]
            )
            metrics = derived_metrics(counts)
            metrics.update(counts)
            metrics["windows"] = len(subset)
            metrics["file_equal_balanced_accuracy"] = file_equal_balanced_accuracy(subset)
            own[(epoch, condition)] = metrics
    worst_float = 0.0
    count_mismatches: list[str] = []
    for key, metrics in own.items():
        reference = stored.get(key)
        if reference is None:
            count_mismatches.append(f"missing stored row {key}")
            continue
        for field in ("balanced_accuracy", "accuracy", "drone_recall", "background_specificity", "precision", "f1", "file_equal_balanced_accuracy"):
            worst_float = max(worst_float, abs(float(reference[field]) - float(metrics[field])))
        for field in ("windows", "tn", "fp", "fn", "tp"):
            if int(reference[field]) != int(metrics[field]):
                count_mismatches.append(f"{key} {field}: {reference[field]} != {metrics[field]}")
    checks.add(
        "C all 20 stored validation metric rows re-derive from the per-window decisions",
        worst_float <= TOLERANCE and not count_mismatches,
        {"max_abs_float_delta": worst_float, "count_mismatches": count_mismatches[:5]},
    )

    # D1. published recomputation CSV
    published_rows = read_csv_rows(DIAG_DIR / "validation_metrics_recomputed.csv")
    recomputed_mismatch: list[str] = []
    worst_published = 0.0
    for row in published_rows:
        key = (int(row["epoch"]), row["condition"])
        metrics = own.get(key)
        if metrics is None:
            recomputed_mismatch.append(f"unexpected row {key}")
            continue
        for field in ("balanced_accuracy", "accuracy", "drone_recall", "background_specificity", "precision", "f1", "file_equal_balanced_accuracy"):
            worst_published = max(
                worst_published, abs(float(row[f"recomputed_{field}"]) - float(metrics[field]))
            )
        for field in ("windows", "tn", "fp", "fn", "tp"):
            if int(row[f"recomputed_{field}"]) != int(metrics[field]):
                recomputed_mismatch.append(f"{key} {field}")
    checks.add(
        "D1 validation_metrics_recomputed.csv agrees with this script",
        len(published_rows) == 20 and worst_published <= TOLERANCE and not recomputed_mismatch,
        {"rows": len(published_rows), "max_abs_delta": worst_published, "mismatches": recomputed_mismatch[:5]},
    )

    # D2. trajectory CSV (BA, file-equal BA, AUC, confusion)
    trajectory = read_csv_rows(DIAG_DIR / "epoch_trajectory_by_condition.csv")
    trajectory_mismatch: list[str] = []
    worst_auc = 0.0
    for row in trajectory:
        key = (int(row["epoch"]), row["condition"])
        metrics = own[key]
        subset = [item for item in scores_by_epoch[key[0]] if item["condition"] == key[1]]
        drone = [float(item["logit_margin"]) for item in subset if int(item["label"]) == 1]
        background = [float(item["logit_margin"]) for item in subset if int(item["label"]) == 0]
        auc = auc_pairwise(drone, background)
        worst_auc = max(worst_auc, abs(auc - float(row["detection_auc"])))
        for field, value in (
            ("balanced_accuracy", metrics["balanced_accuracy"]),
            ("file_equal_balanced_accuracy", metrics["file_equal_balanced_accuracy"]),
            ("accuracy", metrics["accuracy"]),
        ):
            if abs(float(row[field]) - float(value)) > TOLERANCE:
                trajectory_mismatch.append(f"{key} {field}")
        for field in ("tn", "fp", "fn", "tp"):
            if int(row[field]) != int(metrics[field]):
                trajectory_mismatch.append(f"{key} {field}")
    checks.add(
        "D2 epoch_trajectory_by_condition.csv agrees with this script (incl. AUC)",
        len(trajectory) == 20 and worst_auc <= 1e-12 and not trajectory_mismatch,
        {"rows": len(trajectory), "max_auc_delta": worst_auc, "mismatches": trajectory_mismatch[:5]},
    )

    # D3. epoch 3 per-model breakdown
    model_rows = read_csv_rows(DIAG_DIR / "epoch03_model_by_condition.csv")
    model_mismatch: list[str] = []
    for row in model_rows:
        subset = [
            item
            for item in scores_by_epoch[SELECTED_EPOCH]
            if item["condition"] == row["condition"] and item["target_code"] == row["target_code"]
        ]
        if not subset:
            model_mismatch.append(f"no windows for {row['target_code']} {row['condition']}")
            continue
        counts = confusion(
            [int(item["label"]) for item in subset], [int(item["prediction"]) for item in subset]
        )
        errors = counts["fp"] + counts["fn"]
        if int(row["windows"]) != len(subset) or int(row["errors"]) != errors:
            model_mismatch.append(
                f"{row['target_code']} {row['condition']}: {row['windows']}/{row['errors']} != {len(subset)}/{errors}"
            )
        if abs(float(row["recall_or_specificity"]) - (1 - errors / len(subset))) > TOLERANCE:
            model_mismatch.append(f"{row['target_code']} {row['condition']} rate")
    checks.add(
        "D3 epoch03_model_by_condition.csv agrees with this script",
        len(model_rows) == 52 and not model_mismatch,
        {"rows": len(model_rows), "mismatches": model_mismatch[:5]},
    )

    # D4. epoch 3 per-file breakdown
    file_rows = read_csv_rows(DIAG_DIR / "epoch03_file_condition_breakdown.csv")
    file_mismatch: list[str] = []
    for row in file_rows:
        subset = [
            item
            for item in scores_by_epoch[SELECTED_EPOCH]
            if item["condition"] == row["condition"] and item["filename"] == row["filename"]
        ]
        counts = confusion(
            [int(item["label"]) for item in subset], [int(item["prediction"]) for item in subset]
        )
        if int(row["errors"]) != counts["fp"] + counts["fn"] or int(row["windows"]) != len(subset):
            file_mismatch.append(f"{row['filename']} {row['condition']}")
    checks.add(
        "D4 epoch03_file_condition_breakdown.csv agrees with this script",
        len(file_rows) == 64 and not file_mismatch,
        {"rows": len(file_rows), "mismatches": file_mismatch[:5]},
    )

    # D5. threshold diagnostic, independent sweep
    thresholds = read_json(DIAG_DIR / "threshold_diagnostic.json")
    threshold_mismatch: list[str] = []
    for epoch in EPOCHS:
        per_condition_labels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for condition in CONDITIONS:
            subset = [row for row in scores_by_epoch[epoch] if row["condition"] == condition]
            per_condition_labels[condition] = (
                np.asarray([int(row["label"]) for row in subset], dtype=np.int64),
                np.asarray([float(row["logit_margin"]) for row in subset], dtype=np.float64),
            )
        payload = thresholds["epochs"][f"epoch_{epoch:02d}"]
        # Per-condition optimum from the descending sweep.
        for condition in CONDITIONS:
            labels, margins = per_condition_labels[condition]
            _, scores, _ = sweep_curves(labels, margins)
            best = float(np.max(scores))
            recorded = float(payload["per_condition_oracle"][condition]["balanced_accuracy"])
            if abs(best - recorded) > TOLERANCE:
                threshold_mismatch.append(f"epoch {epoch} {condition}: {best} != {recorded}")
            zero = derived_metrics(confusion(list(labels), (margins > 0.0).astype(int).tolist()))[
                "balanced_accuracy"
            ]
            recorded_zero = float(payload["per_condition_oracle"][condition]["balanced_accuracy_at_zero"])
            if abs(zero - recorded_zero) > TOLERANCE:
                threshold_mismatch.append(f"epoch {epoch} {condition}: zero-threshold BA")
        # Shared threshold: brute force over every observed margin value.
        candidates = np.unique(
            np.concatenate([margins for _, margins in per_condition_labels.values()])
        )
        best_shared = (-1.0, None)
        for boundary in candidates:
            total = 0.0
            for condition in CONDITIONS:
                labels, margins = per_condition_labels[condition]
                predictions = (margins >= boundary).astype(np.int64)
                total += derived_metrics(confusion(list(labels), predictions.tolist()))[
                    "balanced_accuracy"
                ]
            mean_score = total / len(CONDITIONS)
            if mean_score > best_shared[0]:
                best_shared = (mean_score, float(boundary))
        recorded_shared = float(payload["shared_threshold_oracle"]["mean_balanced_accuracy"])
        if abs(best_shared[0] - recorded_shared) > 1e-9:
            threshold_mismatch.append(
                f"epoch {epoch} shared oracle: {best_shared[0]} != {recorded_shared}"
            )
    checks.add(
        "D5 threshold_diagnostic.json agrees with an independent sweep",
        not threshold_mismatch,
        {"mismatches": threshold_mismatch[:5]},
    )

    # D6. summary consistency
    summary = read_json(DIAG_DIR / "diagnostics_summary.json")
    summary_mismatch: list[str] = []
    for epoch in EPOCHS:
        key = f"epoch_{epoch:02d}"
        mean_ba = float(
            np.mean(
                [
                    own[(epoch, condition)]["balanced_accuracy"]
                    for condition in CONDITIONS
                ]
            )
        )
        if abs(mean_ba - float(summary["epoch_means"][key]["mean_balanced_accuracy"])) > TOLERANCE:
            summary_mismatch.append(f"{key} mean BA")
        mean_auc = float(
            np.mean(
                [
                    auc_pairwise(
                        [
                            float(item["logit_margin"])
                            for item in scores_by_epoch[epoch]
                            if item["condition"] == condition and int(item["label"]) == 1
                        ],
                        [
                            float(item["logit_margin"])
                            for item in scores_by_epoch[epoch]
                            if item["condition"] == condition and int(item["label"]) == 0
                        ],
                    )
                    for condition in CONDITIONS
                ]
            )
        )
        if abs(mean_auc - float(summary["threshold_free_separability"][key]["mean_detection_auc"])) > 1e-12:
            summary_mismatch.append(f"{key} mean AUC")
        total_errors = sum(
            own[(epoch, condition)]["fp"] + own[(epoch, condition)]["fn"] for condition in CONDITIONS
        )
        if int(summary["epoch_means"][key]["total_errors_over_four_conditions"]) != total_errors:
            summary_mismatch.append(f"{key} total errors")
    epoch3 = float(
        np.mean([own[(SELECTED_EPOCH, condition)]["balanced_accuracy"] for condition in CONDITIONS])
    )
    if abs(epoch3 - float(summary["selection_rule"] and summary["epoch_means"]["epoch_03"]["mean_balanced_accuracy"])) > TOLERANCE:
        summary_mismatch.append("selected epoch mean BA")
    checks.add(
        "D6 diagnostics_summary.json epoch aggregates agree with this script",
        not summary_mismatch,
        {"mismatches": summary_mismatch[:5]},
    )

    # E. flip bookkeeping identity
    flips = read_json(DIAG_DIR / "epoch_flip_analysis.json")
    flip_mismatch: list[str] = []
    for pair, payload in flips["pairs"].items():
        earlier = int(pair.split("_to_epoch")[0].replace("epoch", ""))
        later = int(pair.split("_to_epoch")[1])
        totals = {
            condition: confusion(
                [int(row["label"]) for row in scores_by_epoch[earlier] if row["condition"] == condition],
                [int(row["prediction"]) for row in scores_by_epoch[earlier] if row["condition"] == condition],
            )
            for condition in CONDITIONS
        }
        totals_later = {
            condition: confusion(
                [int(row["label"]) for row in scores_by_epoch[later] if row["condition"] == condition],
                [int(row["prediction"]) for row in scores_by_epoch[later] if row["condition"] == condition],
            )
            for condition in CONDITIONS
        }
        delta_fp = sum(totals_later[c]["fp"] - totals[c]["fp"] for c in CONDITIONS)
        delta_fn = sum(totals_later[c]["fn"] - totals[c]["fn"] for c in CONDITIONS)
        if delta_fp != int(payload["delta_fp_predicted"]):
            flip_mismatch.append(f"{pair}: dFP {delta_fp} != {payload['delta_fp_predicted']}")
        if delta_fn != int(payload["delta_fn_predicted"]):
            flip_mismatch.append(f"{pair}: dFN {delta_fn} != {payload['delta_fn_predicted']}")
        if payload["decision_changes"] != (
            payload["label0_pred0_to_pred1"]
            + payload["label0_pred1_to_pred0"]
            + payload["label1_pred0_to_pred1"]
            + payload["label1_pred1_to_pred0"]
        ):
            flip_mismatch.append(f"{pair}: decision_changes does not equal the four counters")
        per_file_total = sum(
            sum(entry.values()) for entry in payload["by_file"].values()
        )
        if per_file_total != payload["decision_changes"]:
            flip_mismatch.append(f"{pair}: by_file total {per_file_total} != decision_changes")
    checks.add(
        "E flip counters satisfy the exact dFP / dFN identity against the confusion counts",
        not flip_mismatch,
        {"mismatches": flip_mismatch[:5]},
    )

    # F. leak control from the accepted run's read audit
    audit = read_json(SEED_ROOT / "read_audit.json")
    opened = [str(name) for name in audit["opened_files"]]
    opened_files = {Path(name).name for name in opened}
    test_files = {name for name, role in role_by_file.items() if role in TEST_ROLES}
    checks.add(
        "F no test-role MAT was opened by the accepted training run",
        int(audit["opened_file_count"]) == 78
        and not (opened_files & test_files)
        and audit["test_files_touched"] == []
        and bool(audit["passed"]),
        {
            "opened_file_count": audit["opened_file_count"],
            "test_files_opened": sorted(opened_files & test_files),
            "test_files_touched": audit["test_files_touched"],
        },
    )
    diagnostics_inputs = read_json(DIAG_DIR / "input_identity.json")["files"]
    mat_inputs = [name for name in diagnostics_inputs if name.lower().endswith(".mat")]
    checks.add(
        "F the diagnostics themselves read no MAT file at all",
        not mat_inputs,
        {"mat_inputs": mat_inputs},
    )

    # G. negative controls
    controls: list[dict[str, Any]] = []

    def control(name: str, detected: bool, detail: Any) -> None:
        controls.append({"control": name, "detected": bool(detected), "detail": detail})

    base_rows = scores_by_epoch[SELECTED_EPOCH]
    clean_rows = [row for row in base_rows if row["condition"] == "clean"]

    duplicated = clean_rows + [dict(clean_rows[0])]
    control(
        "G1 duplicated window key is rejected",
        bool(integrity_problems(duplicated, expected, role_by_file)),
        integrity_problems(duplicated, expected, role_by_file)[:1],
    )

    dropped = clean_rows[1:]
    problems = integrity_problems(dropped, expected, role_by_file)
    control(
        "G2 a missing val window is rejected",
        any("missing" in problem for problem in problems),
        problems[:1],
    )

    train_file = next(name for name, role in role_by_file.items() if role == "train")
    injected = dict(clean_rows[0])
    injected["filename"] = train_file
    injected["window_index"] = 0
    problems = integrity_problems(clean_rows + [injected], expected, role_by_file)
    control(
        "G3 a train-role row injected as val is rejected as cross-role",
        any("outside the frozen val split" in problem and "train" in problem for problem in problems),
        problems[:2],
    )

    relabelled = [dict(row) for row in clean_rows]
    relabelled[0]["label"] = 1 - int(relabelled[0]["label"])
    problems = integrity_problems(relabelled, expected, role_by_file)
    control(
        "G4 a flipped label is rejected",
        any("label mismatch" in problem for problem in problems),
        problems[:1],
    )

    shifted = [dict(row) for row in clean_rows]
    shifted[3]["start"] = int(shifted[3]["start"]) + 1
    problems = integrity_problems(shifted, expected, role_by_file)
    control(
        "G5 a shifted start offset is rejected",
        any("start mismatch" in problem for problem in problems),
        problems[:1],
    )

    inconsistent = [dict(row) for row in clean_rows]
    inconsistent[5]["prediction"] = 1 - int(inconsistent[5]["prediction"])
    problems = integrity_problems(inconsistent, expected, role_by_file)
    control(
        "G6 a prediction that contradicts its margin sign is rejected",
        any("margin > 0" in problem for problem in problems),
        problems[:1],
    )

    # G7: the metric maths must notice a single flipped decision.
    labels = [int(row["label"]) for row in clean_rows]
    predictions = [int(row["prediction"]) for row in clean_rows]
    baseline = derived_metrics(confusion(labels, predictions))["balanced_accuracy"]
    tampered = list(predictions)
    for position, value in enumerate(tampered):
        if value == 1 and labels[position] == 0:
            tampered[position] = 0
            break
    shifted_score = derived_metrics(confusion(labels, tampered))["balanced_accuracy"]
    control(
        "G7 one flipped background decision changes the balanced accuracy",
        abs(shifted_score - baseline) > 0,
        {"before": baseline, "after": shifted_score},
    )

    # G8: AUC must be 1.0 for a perfectly separated sample and ~0.5 for noise.
    rng = np.random.default_rng(20260927)
    perfect = auc_pairwise([2.0, 3.0, 4.0], [-3.0, -2.0, -1.0])
    noise = auc_pairwise(rng.normal(size=4000).tolist(), rng.normal(size=280).tolist())
    control(
        "G8 AUC is 1.0 for separated data and ~0.5 for pure noise",
        abs(perfect - 1.0) < 1e-12 and abs(noise - 0.5) < 0.03,
        {"perfect": perfect, "noise": noise},
    )

    # G9: the file-equal metric must react to *where* the errors sit, unlike pooled accuracy.
    synthetic = []
    for index in range(110):
        synthetic.append({"filename": "small.mat", "label": 1, "prediction": 0 if index < 11 else 1})
    for index in range(340):
        synthetic.append({"filename": "large.mat", "label": 1, "prediction": 1})
    for index in range(280):
        synthetic.append({"filename": "bg.mat", "label": 0, "prediction": 0})
    concentrated = file_equal_balanced_accuracy(synthetic)
    moved = []
    for index in range(110):
        moved.append({"filename": "small.mat", "label": 1, "prediction": 1})
    for index in range(340):
        moved.append({"filename": "large.mat", "label": 1, "prediction": 0 if index < 11 else 1})
    for index in range(280):
        moved.append({"filename": "bg.mat", "label": 0, "prediction": 0})
    dispersed = file_equal_balanced_accuracy(moved)
    pooled_same = confusion([row["label"] for row in synthetic], [row["prediction"] for row in synthetic])
    pooled_moved = confusion([row["label"] for row in moved], [row["prediction"] for row in moved])
    control(
        "G9 file-equal BA reacts to error location while pooled counts do not",
        abs(concentrated - dispersed) > 0.01 and pooled_same == pooled_moved,
        {"concentrated": concentrated, "dispersed": dispersed, "pooled_equal": pooled_same == pooled_moved},
    )

    # G10: the threshold sweep must never score below the zero-threshold rule.
    worst_gap = 0.0
    for epoch in EPOCHS:
        for condition in CONDITIONS:
            subset = [row for row in scores_by_epoch[epoch] if row["condition"] == condition]
            labels = np.asarray([int(row["label"]) for row in subset], dtype=np.int64)
            margins = np.asarray([float(row["logit_margin"]) for row in subset], dtype=np.float64)
            _, scores, _ = sweep_curves(labels, margins)
            at_zero = derived_metrics(
                confusion(labels.tolist(), (margins > 0.0).astype(int).tolist())
            )["balanced_accuracy"]
            if float(np.max(scores)) + 1e-12 < at_zero:
                worst_gap = max(worst_gap, at_zero - float(np.max(scores)))
    control(
        "G10 the swept optimum is never below the recorded zero-threshold score",
        worst_gap == 0.0,
        {"worst_gap": worst_gap},
    )

    detected = [item for item in controls if item["detected"]]
    checks.add(
        "G every negative control was detected",
        len(detected) == len(controls),
        {"detected": len(detected), "total": len(controls), "controls": controls},
    )
    result["negative_controls"] = controls

    result.update(checks.payload())
    write_json(DIAG_DIR / "diagnostics_verification.json", result)

    for record in checks.records:
        marker = "PASS" if record["passed"] else "FAIL"
        print(f"[{marker}] {record['check']}")
    print(f"\n{len(checks.records) - checks.failed}/{len(checks.records)} checks passed")
    return 0 if checks.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
