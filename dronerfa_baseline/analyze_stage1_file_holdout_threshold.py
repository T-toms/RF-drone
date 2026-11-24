#!/usr/bin/env python3
"""Assess Stage1 threshold calibration without opening development-test files.

The selected checkpoint is fixed at epoch 3. For each validation MAT, fit a
single threshold on the other 15 validation MAT files and score the held-out
file. This is a calibration diagnostic, not an independent model evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SCORES = ROOT / "outputs/stage1_12drones_20260926/seed_2026/validation_scores_epoch_03.csv"
OUT = ROOT / "outputs/stage1_12drones_20260926/diagnostics/file_holdout_threshold_20260930.json"
CONDITIONS = ("clean", "-10dB", "-12dB", "-15dB")


def metrics(rows: list[dict], threshold: float) -> dict:
    n_drone = sum(row["label"] == 1 for row in rows)
    n_bg = len(rows) - n_drone
    tp = sum(row["label"] == 1 and row["margin"] > threshold for row in rows)
    tn = sum(row["label"] == 0 and row["margin"] <= threshold for row in rows)
    return {
        "drone_recall": tp / n_drone if n_drone else None,
        "background_specificity": tn / n_bg if n_bg else None,
        "tp": tp,
        "fn": n_drone - tp,
        "tn": tn,
        "fp": n_bg - tn,
    }


def fit_threshold(rows: list[dict]) -> tuple[float, float]:
    # Exact threshold search: predictions change only at observed margins.
    margins = np.array([row["margin"] for row in rows], dtype=np.float64)
    thresholds = np.unique(np.concatenate((margins, [0.0])))
    scores = np.zeros(thresholds.size, dtype=np.float64)
    for condition in CONDITIONS:
        group = [row for row in rows if row["condition"] == condition]
        drone = np.sort([row["margin"] for row in group if row["label"] == 1])
        bg = np.sort([row["margin"] for row in group if row["label"] == 0])
        if not len(drone) or not len(bg):
            raise ValueError(f"Missing class in calibration group: {condition}")
        recall = 1 - np.searchsorted(drone, thresholds, side="right") / len(drone)
        specificity = np.searchsorted(bg, thresholds, side="right") / len(bg)
        scores += (recall + specificity) / (2 * len(CONDITIONS))
    best = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))
    best_index = min(best, key=lambda i: (abs(thresholds[i]), thresholds[i]))
    return float(thresholds[best_index]), float(scores[best_index])


def aggregate(rows: list[dict], thresholds: dict[str, float]) -> dict:
    by_condition = {}
    for condition in CONDITIONS:
        group = [row for row in rows if row["condition"] == condition]
        n_drone = sum(row["label"] == 1 for row in group)
        n_bg = len(group) - n_drone
        tp = sum(row["label"] == 1 and row["margin"] > thresholds[row["filename"]] for row in group)
        tn = sum(row["label"] == 0 and row["margin"] <= thresholds[row["filename"]] for row in group)
        by_condition[condition] = {
            "drone_recall": tp / n_drone,
            "background_specificity": tn / n_bg,
            "balanced_accuracy": (tp / n_drone + tn / n_bg) / 2,
            "tp": tp, "fn": n_drone - tp, "tn": tn, "fp": n_bg - tn,
        }
    return {
        "by_condition": by_condition,
        "mean_balanced_accuracy": sum(v["balanced_accuracy"] for v in by_condition.values()) / len(CONDITIONS),
    }


def main() -> None:
    rows = []
    with SCORES.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append({
                "condition": raw["condition"],
                "filename": raw["filename"],
                "label": int(raw["label"]),
                "margin": float(raw["logit_margin"]),
                "window_index": int(raw["window_index"]),
            })
    if len(rows) != 9440 or set(r["condition"] for r in rows) != set(CONDITIONS):
        raise ValueError("Unexpected validation score shape")
    files = sorted(set(r["filename"] for r in rows))
    if len(files) != 16:
        raise ValueError("Expected exactly 16 validation files")
    keysets = {}
    for condition in CONDITIONS:
        keys = [(r["filename"], r["window_index"]) for r in rows if r["condition"] == condition]
        if len(keys) != 2360 or len(set(keys)) != len(keys):
            raise ValueError(f"Missing or duplicate windows in {condition}")
        keysets[condition] = set(keys)
    if any(keysets[c] != keysets["clean"] for c in CONDITIONS):
        raise ValueError("Condition window keys differ")
    labels = defaultdict(set)
    for row in rows:
        labels[row["filename"]].add(row["label"])
    if any(len(v) != 1 for v in labels.values()):
        raise ValueError("Mixed labels within a source file")

    global_threshold, global_fit_ba = fit_threshold(rows)
    file_thresholds = {}
    heldout = []
    for filename in files:
        threshold, calibration_ba = fit_threshold([r for r in rows if r["filename"] != filename])
        file_thresholds[filename] = threshold
        selected = [r for r in rows if r["filename"] == filename]
        heldout.append({
            "filename": filename,
            "label": next(iter(labels[filename])),
            "threshold_fitted_without_this_file": threshold,
            "calibration_mean_ba": calibration_ba,
            "at_zero": {c: metrics([r for r in selected if r["condition"] == c], 0.0) for c in CONDITIONS},
            "at_holdout_threshold": {c: metrics([r for r in selected if r["condition"] == c], threshold) for c in CONDITIONS},
        })

    payload = {
        "purpose": "validation-only, source-file-held-out threshold calibration diagnostic",
        "not_independent_test": True,
        "selected_checkpoint": "stage1 seed 2026 epoch 3",
        "score_file": str(SCORES),
        "score_sha256": hashlib.sha256(SCORES.read_bytes()).hexdigest(),
        "selection_objective": "mean of four condition-wise pooled balanced accuracies",
        "tie_break": "threshold closest to zero, then lower threshold",
        "conditions": CONDITIONS,
        "files": len(files),
        "windows_per_condition": 2360,
        "default_zero": aggregate(rows, {f: 0.0 for f in files}),
        "full_validation_fit_optimistic": {
            "threshold": global_threshold,
            "fit_mean_balanced_accuracy": global_fit_ba,
            "metrics": aggregate(rows, {f: global_threshold for f in files}),
        },
        "leave_one_file_out": {
            "metrics": aggregate(rows, file_thresholds),
            "threshold_min": min(file_thresholds.values()),
            "threshold_max": max(file_thresholds.values()),
            "per_file": heldout,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"default mean BA: {payload['default_zero']['mean_balanced_accuracy']:.6f}")
    print(f"full-fit threshold: {global_threshold:+.6f}; mean BA: {global_fit_ba:.6f}")
    print(f"file-holdout mean BA: {payload['leave_one_file_out']['metrics']['mean_balanced_accuracy']:.6f}")
    print(f"threshold range: {min(file_thresholds.values()):+.6f} .. {max(file_thresholds.values()):+.6f}")
    print(OUT)


if __name__ == "__main__":
    main()
