#!/usr/bin/env python3
"""Paired, validation-only comparison of the selected Stage2 checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs/stage2_12drones_20260927/seed_2027"
NEW = ROOT / "outputs/stage2_minus15_20260930/seed_2027"
OUT = ROOT / "outputs/stage2_minus15_20260930/validation_comparison"
CONDITIONS = ("clean", "-10dB", "-12dB", "-15dB")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_selected(directory: Path) -> tuple[dict, dict]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed" or summary["test_files_touched"]:
        raise ValueError(f"Run incomplete or test data touched: {directory}")
    if sha256(directory / "best_model.pt") != summary["best_model_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {directory}")
    scores_path = directory / f"validation_scores_epoch_{summary['selected_epoch']:02d}.csv"
    rows = {}
    with scores_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["condition"], int(row["row_index"]))
            if key in rows:
                raise ValueError(f"Duplicate score key: {key}")
            rows[key] = {
                "filename": row["filename"], "target_code": row["target_code"],
                "label": int(row["label"]), "prediction": int(row["prediction"]),
                "full_split_position": int(row["full_split_position"]),
            }
    if len(rows) != 8320 or set(k[0] for k in rows) != set(CONDITIONS):
        raise ValueError(f"Unexpected validation score count: {len(rows)}")
    return {"summary": summary, "score_file": str(scores_path), "score_sha256": sha256(scores_path)}, rows


def accuracy(rows: list[dict], model: str) -> float:
    return sum(r[model + "_correct"] for r in rows) / len(rows)


def main() -> None:
    baseline, base_rows = load_selected(BASE)
    experiment, new_rows = load_selected(NEW)
    if set(base_rows) != set(new_rows):
        raise ValueError("Selected validation score keys differ")
    paired = []
    for key in sorted(base_rows):
        a, b = base_rows[key], new_rows[key]
        if any(a[field] != b[field] for field in ("filename", "target_code", "label", "full_split_position")):
            raise ValueError(f"Validation identity mismatch: {key}")
        paired.append({
            "condition": key[0], "row_index": key[1], "filename": a["filename"],
            "target_code": a["target_code"], "label": a["label"],
            "baseline_correct": int(a["prediction"] == a["label"]),
            "minus15_correct": int(b["prediction"] == b["label"]),
        })
    file_groups = defaultdict(list)
    for row in paired:
        file_groups[(row["condition"], row["target_code"], row["filename"])].append(row)
    file_scores = []
    for (condition, code, filename), rows in sorted(file_groups.items()):
        base_acc = accuracy(rows, "baseline")
        new_acc = accuracy(rows, "minus15")
        file_scores.append({"condition": condition, "target_code": code, "filename": filename,
                            "windows": len(rows), "baseline_accuracy": base_acc,
                            "minus15_accuracy": new_acc, "delta": new_acc - base_acc})
    class_groups = defaultdict(list)
    for row in file_scores:
        class_groups[(row["condition"], row["target_code"])].append(row)
    class_scores = []
    for (condition, code), rows in sorted(class_groups.items()):
        base_acc = sum(r["baseline_accuracy"] for r in rows) / len(rows)
        new_acc = sum(r["minus15_accuracy"] for r in rows) / len(rows)
        class_scores.append({"condition": condition, "target_code": code, "files": len(rows),
                             "baseline_file_equal_recall": base_acc,
                             "minus15_file_equal_recall": new_acc, "delta": new_acc - base_acc})
    condition_summary = []
    for condition in CONDITIONS:
        rows = [r for r in class_scores if r["condition"] == condition]
        if len(rows) != 12:
            raise ValueError(f"Expected 12 classes in {condition}")
        base_score = sum(r["baseline_file_equal_recall"] for r in rows) / 12
        new_score = sum(r["minus15_file_equal_recall"] for r in rows) / 12
        condition_summary.append({"condition": condition, "baseline": base_score,
                                  "minus15": new_score, "delta": new_score - base_score})
    mean_baseline = sum(r["baseline"] for r in condition_summary) / 4
    mean_experiment = sum(r["minus15"] for r in condition_summary) / 4
    if abs(mean_baseline - baseline["summary"]["validation_score"]) > 1e-12:
        raise ValueError("Baseline selected score does not reconcile")
    if abs(mean_experiment - experiment["summary"]["validation_score"]) > 1e-12:
        raise ValueError("Experiment selected score does not reconcile")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("per_file.csv", file_scores), ("per_class_condition.csv", class_scores),
                       ("per_condition.csv", condition_summary)):
        with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload = {
        "scope": "paired validation windows only; no development-test evaluation",
        "baseline": baseline, "minus15": experiment,
        "files_per_condition": len(file_scores) // 4,
        "windows_per_condition": len(paired) // 4,
        "mean_baseline": mean_baseline, "mean_minus15": mean_experiment,
        "delta": mean_experiment - mean_baseline,
        "per_condition": condition_summary,
        "class_condition_wins": sum(r["delta"] > 0 for r in class_scores),
        "class_condition_losses": sum(r["delta"] < 0 for r in class_scores),
        "class_condition_ties": sum(r["delta"] == 0 for r in class_scores),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("mean_baseline", "mean_minus15", "delta", "per_condition",
                                              "class_condition_wins", "class_condition_losses", "class_condition_ties")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
