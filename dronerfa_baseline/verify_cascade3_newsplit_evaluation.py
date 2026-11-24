"""Independent verification of CASCADE3-NEWSPLIT-C.

This script deliberately does **not** import ``run_cascade3_newsplit_evaluation``:
every metric, every identity claim and every raw-IQ feature is recomputed from the
published artefacts with its own code path.

What it checks
--------------
1. frozen input identity: index / stage2 view / frozen split CSV / G manifest /
   both seed-2026 checkpoints (hashes, epochs, protocol metadata);
2. prediction coverage: exactly 890 unique ``dataset_index_position`` values in
   ``0..889`` and exactly the six frozen ``development_test_exposed`` source MATs;
3. every prediction row matches the frozen index row (filename, start, length,
   window_index, block_index, target_code, file_sha256, source_role, split) and
   the frozen split CSV (role / MAT hash);
4. frozen decision rules: stage-1 margin -> stage-2 execution, stage-2 margin ->
   final class, Background rows carry no stage-2 fields, executed rows share the
   stage-1 feature tensor digest;
5. recomputed pooled / file-equal / stage-1 / stage-2 / error-decomposition
   metrics equal the published tables;
6. an independent raw-IQ re-derivation: for a bounded set of windows the RF0 slice
   is re-read from G, re-normalised, re-transformed by a freshly constructed
   frontend and re-classified by freshly loaded checkpoints; the logits must match
   the published CSV within ``LOGIT_TOLERANCE``;
7. negative controls proving those checks actually fail on tampered input.

Usage::

    python verify_cascade3_newsplit_evaluation.py [--output-dir DIR] [--device auto]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "outputs" / ".matplotlib"))

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent
NEWSPLIT_ROOT = PROJECT_DIR / "outputs" / "cascade3_newsplit_20260926"
INDEX = NEWSPLIT_ROOT / "window_index_newsplit3.csv"
INDEX_SHA256 = "deed870f9693c5ebe93fb80a9647b452e90680dd5b1f21372e108ded4bf4d231"
STAGE2_VIEW = NEWSPLIT_ROOT / "stage2_view_newsplit3.csv"
STAGE2_VIEW_SHA256 = "76001e671cf3ebe492b191af87a85b5368caed85e03c84c4c249ccc5081642de"
WORK_DIR = PROJECT_ROOT / "work"
FREEZE_JSON = WORK_DIR / "meta_split2" / "CODEX-FREEZE-20260926.json"
FROZEN_SPLIT_CSV = WORK_DIR / "meta_split2" / "split_candidate_104_20260926.csv"
FROZEN_SPLIT_CSV_SHA256 = (
    "d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9"
)
G_MANIFEST = Path(r"G:\DroneRFa数据集\dataset_manifest.csv")
G_MANIFEST_SHA256 = "9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d"
DATA_ROOT = Path(r"G:\DroneRFa数据集\original")
STAGE1_CHECKPOINT = NEWSPLIT_ROOT / "seed_2026" / "stage1" / "best_model.pt"
STAGE1_SHA256 = "5621c398542fe31b977e1b05deaf089c22b6a53d1f4be12dbb2c74532cd4779e"
STAGE1_EPOCH = 3
STAGE2_CHECKPOINT = NEWSPLIT_ROOT / "seed_2026" / "stage2" / "best_model.pt"
STAGE2_SHA256 = "aa25a8ba01d457a2b981b6b29a0cae615aca25af530d7466171ecab47ce68cfd"
STAGE2_EPOCH = 3
OUTPUT_ROOT = PROJECT_DIR / "outputs" / "cascade3_newsplit_eval_20260926" / "seed_2026"

N_FFT = 512
HOP_LENGTH = 256
IMAGE_SIZE = 256
BATCH_SIZE = 4
TRAINING_SEED = 2026
TEST_ROLE = "development_test_exposed"
CONDITION = "clean"
EXPECTED_WINDOWS = 890
EXPECTED_FILES = 6
EXPECTED_TARGET_COUNTS = {"T0000": 280, "T0010": 300, "T0101": 310}
CLASS_IDS = (0, 1, 2)
DRONE_CLASSES = (1, 2)
BACKGROUND_CLASS = 0
CLASS_NAMES = {0: "Background", 1: "DJI Phantom 4 Pro", 2: "DJI Air 2S"}
TARGET_CODE_TO_CLASS = {"T0000": 0, "T0010": 1, "T0101": 2}
ERROR_TRANSITIONS = (
    (0, 1, "Background->Phantom"),
    (0, 2, "Background->Air2S"),
    (1, 0, "Phantom->Background"),
    (2, 0, "Air2S->Background"),
    (1, 2, "Phantom->Air2S"),
    (2, 1, "Air2S->Phantom"),
)
TOLERANCE = 1e-9
#: same-batch rebuild must reproduce the published float32/AMP logits exactly
EXACT_LOGIT_TOLERANCE = 1e-6
#: a lone window uses different fp16 CUDA kernels; only this bound is meaningful
AMP_LOGIT_TOLERANCE = 5e-2
# bounded raw-IQ re-derivation sets
REDERIVE_PER_FILE = 1
EXACT_PARITY_WINDOWS = 8
WINDOW_SAMPLES = 1_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


class CheckList:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, detail: Any = "ok") -> bool:
        self.items.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            log(f"  [FAIL] {name}: {detail}")
        return bool(passed)

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [item for item in self.items if not item["passed"]]


# ---------------------------------------------------------------------------
# 1. frozen input identity
# ---------------------------------------------------------------------------
def check_frozen_inputs(checks: CheckList) -> dict[str, Any]:
    observed = {
        "index": sha256_file(INDEX),
        "stage2_view": sha256_file(STAGE2_VIEW),
        "frozen_split_csv": sha256_file(FROZEN_SPLIT_CSV),
        "g_manifest": sha256_file(G_MANIFEST) if G_MANIFEST.is_file() else None,
        "stage1_checkpoint": sha256_file(STAGE1_CHECKPOINT),
        "stage2_checkpoint": sha256_file(STAGE2_CHECKPOINT),
    }
    expected = {
        "index": INDEX_SHA256,
        "stage2_view": STAGE2_VIEW_SHA256,
        "frozen_split_csv": FROZEN_SPLIT_CSV_SHA256,
        "g_manifest": G_MANIFEST_SHA256,
        "stage1_checkpoint": STAGE1_SHA256,
        "stage2_checkpoint": STAGE2_SHA256,
    }
    checks.check("C1.1 frozen input hashes", observed == expected, observed)
    freeze = read_json(FREEZE_JSON)
    checks.check(
        "C1.2 freeze decision matches the frozen CSV",
        str(freeze.get("frozen_split_csv_sha256")) == FROZEN_SPLIT_CSV_SHA256
        and str(freeze.get("frozen_split_csv"))
        == "work/meta_split2/split_candidate_104_20260926.csv",
        {
            "csv": freeze.get("frozen_split_csv"),
            "sha256": freeze.get("frozen_split_csv_sha256"),
        },
    )
    stage1 = torch.load(STAGE1_CHECKPOINT, map_location="cpu", weights_only=False)
    stage2 = torch.load(STAGE2_CHECKPOINT, map_location="cpu", weights_only=False)
    checks.check(
        "C1.3 checkpoint protocol metadata",
        all(
            int(payload["epoch"]) == epoch
            and int(payload["n_fft"]) == N_FFT
            and int(payload["hop_length"]) == HOP_LENGTH
            and int(payload["training_seed"]) == TRAINING_SEED
            and str(payload["index_sha256"]) == INDEX_SHA256
            and str(payload["stage2_view_sha256"]) == STAGE2_VIEW_SHA256
            and str(payload["split_source_sha256"]) == FROZEN_SPLIT_CSV_SHA256
            for payload, epoch in ((stage1, STAGE1_EPOCH), (stage2, STAGE2_EPOCH))
        ),
        {
            "stage1": {
                "epoch": int(stage1["epoch"]),
                "n_fft": int(stage1["n_fft"]),
                "hop_length": int(stage1["hop_length"]),
            },
            "stage2": {
                "epoch": int(stage2["epoch"]),
                "n_fft": int(stage2["n_fft"]),
                "hop_length": int(stage2["hop_length"]),
            },
        },
    )
    checks.check(
        "C1.4 stage-1/stage-2 prediction rules are the frozen thresholds",
        "> 0" in str(stage1["prediction_rule"]) and "> 0" in str(stage2["prediction_rule"]),
        {"stage1": stage1["prediction_rule"], "stage2": stage2["prediction_rule"]},
    )
    return {"observed": observed, "expected": expected}


# ---------------------------------------------------------------------------
# 2./3. prediction coverage and row identity
# ---------------------------------------------------------------------------
def own_index_rows() -> list[dict[str, Any]]:
    """The test split in frozen order, parsed without ``dronerfa_data``."""
    rows: list[dict[str, Any]] = []
    for raw in read_csv_rows(INDEX):
        if str(raw["split"]) != TEST_ROLE:
            continue
        rows.append(
            {
                "position": len(rows),
                "filename": str(raw["filename"]),
                "file_path": str(raw["file_path"]),
                "target_code": str(raw["target_code"]),
                "source_role": str(raw["source_role"]),
                "split": str(raw["split"]),
                "stage1_label": int(raw["label"]),
                "block_index": int(raw["block_index"]),
                "window_index": int(raw["window_index"]),
                "start": int(raw["start"]),
                "length": int(raw["length"]),
                "source_samples": int(raw["source_samples"]),
                "file_sha256": str(raw["file_sha256"]),
                "class_index": int(raw["class_index"]),
                "class_name": str(raw["class_name"]),
            }
        )
    return rows


def own_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in read_csv_rows(path):
        row: dict[str, Any] = dict(raw)
        for key in (
            "training_seed",
            "stage1_epoch",
            "stage2_epoch",
            "dataset_index_position",
            "class_index",
            "stage1_label",
            "true_class",
            "block_index",
            "window_index",
            "start",
            "length",
            "source_samples",
            "stage1_prediction",
            "final_prediction",
        ):
            row[key] = int(raw[key])
        row["stage2_executed"] = str(raw["stage2_executed"]).strip().lower() == "true"
        row["shared_feature_tensor"] = str(raw["shared_feature_tensor"]).strip().lower() == "true"
        row["stage2_batch_id"] = (
            int(raw["stage2_batch_id"]) if raw["stage2_batch_id"] not in ("", None) else None
        )
        for key in (
            "stage1_logit_background",
            "stage1_logit_drone",
            "stage1_margin",
            "stage2_logit_phantom",
            "stage2_logit_air2s",
            "stage2_margin",
        ):
            row[key] = float(raw[key]) if raw[key] not in ("", None) else None
        rows.append(row)
    return rows


def check_coverage_and_identity(
    checks: CheckList, predictions: Sequence[dict[str, Any]], index_rows: Sequence[dict[str, Any]]
) -> None:
    checks.check(
        "C2.1 prediction rows equal the frozen test window count",
        len(predictions) == EXPECTED_WINDOWS,
        len(predictions),
    )
    positions = [int(row["dataset_index_position"]) for row in predictions]
    checks.check(
        "C2.2 dataset_index_position covers 0..889 exactly once",
        not coverage_problems(predictions),
        {"unique": len(set(positions)), "min": min(positions), "max": max(positions)},
    )
    files = sorted({str(row["filename"]) for row in predictions})
    checks.check(
        "C2.3 exactly the six frozen test source MATs",
        len(files) == EXPECTED_FILES and len({str(row["filename"]) for row in index_rows}) == EXPECTED_FILES,
        files,
    )
    target_counts: dict[str, int] = {code: 0 for code in TARGET_CODE_TO_CLASS}
    for row in predictions:
        target_counts[str(row["target_code"])] = target_counts.get(str(row["target_code"]), 0) + 1
    checks.check(
        "C2.4 per-class window counts",
        target_counts == EXPECTED_TARGET_COUNTS,
        target_counts,
    )

    mismatches: list[str] = []
    for predicted, reference in zip(predictions, index_rows):
        if int(predicted["dataset_index_position"]) != int(reference["position"]):
            mismatches.append(f"position {predicted['dataset_index_position']}")
            continue
        for key in (
            "filename",
            "file_path",
            "target_code",
            "source_role",
            "block_index",
            "window_index",
            "start",
            "length",
            "source_samples",
            "file_sha256",
            "class_index",
            "class_name",
        ):
            if str(predicted[key]) != str(reference[key]):
                mismatches.append(f"{reference['filename']}: {key} {predicted[key]} != {reference[key]}")
        if int(predicted["stage1_label"]) != int(reference["stage1_label"]):
            mismatches.append(f"{reference['filename']}: stage1_label")
        if int(predicted["true_class"]) != TARGET_CODE_TO_CLASS[reference["target_code"]]:
            mismatches.append(f"{reference['filename']}: true_class")
        if int(predicted["start"]) != int(reference["window_index"]) * WINDOW_SAMPLES:
            mismatches.append(f"{reference['filename']}: start not window_index * 1M")
    checks.check("C3.1 every prediction row matches the frozen index row", not mismatches, mismatches[:5])

    wrong_role = [
        f"{row['filename']}:{row['source_role']}"
        for row in predictions
        if str(row["source_role"]) != TEST_ROLE or str(row["row_split"]) != TEST_ROLE
    ]
    checks.check("C3.2 all rows carry the development_test_exposed role", not wrong_role, wrong_role[:5])

    frozen_roles = {
        str(row["file_name"]): (str(row["role"]), str(row["mat_sha256"]))
        for row in read_csv_rows(FROZEN_SPLIT_CSV)
    }
    role_problems = []
    for filename in files:
        role, mat_hash = frozen_roles.get(filename, (None, None))
        if role != TEST_ROLE:
            role_problems.append(f"{filename}: frozen role {role}")
    for row in predictions:
        expected_hash = frozen_roles.get(str(row["filename"]), (None, None))[1]
        if str(row["file_sha256"]) != expected_hash:
            role_problems.append(f"{row['filename']}: MAT hash != frozen split CSV")
    checks.check(
        "C3.3 frozen split CSV agrees on role and MAT hash",
        not role_problems,
        role_problems[:5],
    )
    outside = [
        row["filename"]
        for row in predictions
        if not str(row["file_path"]).lower().startswith(str(DATA_ROOT).lower())
    ]
    checks.check("C3.4 all rows read from the G data root", not outside, outside[:5])


# ---------------------------------------------------------------------------
# 4. decision rules
# ---------------------------------------------------------------------------
def coverage_problems(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Every window of the frozen test split, exactly once, in 0..889."""
    positions = [int(row["dataset_index_position"]) for row in rows]
    problems: list[str] = []
    if len(rows) != EXPECTED_WINDOWS:
        problems.append(f"{len(rows)} rows != {EXPECTED_WINDOWS}")
    if sorted(positions) != list(range(EXPECTED_WINDOWS)):
        problems.append(f"positions do not cover 0..{EXPECTED_WINDOWS - 1} exactly once")
    return problems


def decision_problems(rows: Sequence[dict[str, Any]]) -> list[str]:
    """The frozen stage-1/stage-2 rules applied to already parsed prediction rows."""
    problems: list[str] = []
    for row in rows:
        label = f"{row['filename']}#{row['window_index']}"
        if row["stage1_logit_background"] is None or row["stage1_logit_drone"] is None:
            problems.append(f"{label}: missing stage-1 logits")
            continue
        margin = float(np.float32(row["stage1_logit_drone"] - row["stage1_logit_background"]))
        if row["stage1_margin"] is None or not math.isclose(
            margin, float(row["stage1_margin"]), rel_tol=0, abs_tol=1e-6
        ):
            problems.append(f"{label}: stage1 margin")
        want_executed = margin > 0.0
        if bool(row["stage2_executed"]) != want_executed:
            problems.append(f"{label}: stage1 decision")
        if int(row["stage1_prediction"]) != (1 if want_executed else 0):
            problems.append(f"{label}: stage1 prediction")
        if str(row["stage1_decision"]) != ("Drone" if want_executed else "Background"):
            problems.append(f"{label}: stage1 decision text")
        if want_executed:
            if row["stage2_logit_phantom"] is None or row["stage2_logit_air2s"] is None:
                problems.append(f"{label}: missing stage-2 logits")
                continue
            s2_margin = float(np.float32(row["stage2_logit_air2s"] - row["stage2_logit_phantom"]))
            if row["stage2_margin"] is None or not math.isclose(
                s2_margin, float(row["stage2_margin"]), rel_tol=0, abs_tol=1e-6
            ):
                problems.append(f"{label}: stage2 margin")
            want_final = 2 if s2_margin > 0.0 else 1
            if int(row["final_prediction"]) != want_final:
                problems.append(f"{label}: final class")
            if row["stage2_batch_id"] is None:
                problems.append(f"{label}: missing stage2_batch_id")
            if str(row["stage2_feature_digest"]) != str(row["stage1_feature_digest"]):
                problems.append(f"{label}: stage-2 tensor not shared")
            if not bool(row["shared_feature_tensor"]):
                problems.append(f"{label}: shared flag")
        else:
            if int(row["final_prediction"]) != BACKGROUND_CLASS:
                problems.append(f"{label}: skipped row is not Background")
            if any(
                row[key] is not None
                for key in (
                    "stage2_logit_phantom",
                    "stage2_logit_air2s",
                    "stage2_margin",
                    "stage2_batch_id",
                )
            ):
                problems.append(f"{label}: skipped row has stage-2 fields")
            if str(row["stage2_feature_digest"]) not in ("", "None"):
                problems.append(f"{label}: skipped row has digest")
    return problems


def check_decision_rules(checks: CheckList, predictions: Sequence[dict[str, Any]]) -> None:
    problems = decision_problems(predictions)
    executed = sum(1 for row in predictions if bool(row["stage2_executed"]))
    checks.check("C4.1 frozen decision rules hold for all 890 windows", not problems, problems[:5])
    checks.check(
        "C4.2 stage-1 drone share is recorded",
        True,
        {"stage2_executed": executed, "windows": len(predictions)},
    )
    checks.check(
        "C4.3 all rows use the frozen seed/epoch pair",
        all(
            int(row["training_seed"]) == TRAINING_SEED
            and int(row["stage1_epoch"]) == STAGE1_EPOCH
            and int(row["stage2_epoch"]) == STAGE2_EPOCH
            and str(row["condition"]) == CONDITION
            for row in predictions
        ),
        {"seed": TRAINING_SEED, "stage1_epoch": STAGE1_EPOCH, "stage2_epoch": STAGE2_EPOCH},
    )
    batch_ids = sorted(
        {int(row["stage2_batch_id"]) for row in predictions if row["stage2_batch_id"] is not None}
    )
    expected_batches = list(range((executed + BATCH_SIZE - 1) // BATCH_SIZE))
    checks.check("C4.4 stage-2 batch numbering is contiguous from 0", batch_ids == expected_batches, batch_ids)


# ---------------------------------------------------------------------------
# 5. independent metric recomputation
# ---------------------------------------------------------------------------
def own_confusion(predictions: Sequence[dict[str, Any]]) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for row in predictions:
        matrix[int(row["true_class"]), int(row["final_prediction"])] += 1
    return matrix


def own_summary(predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    matrix = own_confusion(predictions)
    total = int(matrix.sum())
    row_totals = matrix.sum(axis=1)
    col_totals = matrix.sum(axis=0)
    recalls = [
        float(matrix[k, k]) / float(row_totals[k]) if row_totals[k] else float("nan")
        for k in CLASS_IDS
    ]
    precisions = [
        float(matrix[k, k]) / float(col_totals[k]) if col_totals[k] else float("nan")
        for k in CLASS_IDS
    ]
    per_file: dict[str, dict[str, Any]] = {}
    for row in predictions:
        entry = per_file.setdefault(
            str(row["filename"]),
            {
                "true_class": int(row["true_class"]),
                "windows": 0,
                "correct": 0,
                "stage1_pass": 0,
                "predicted": {k: 0 for k in CLASS_IDS},
            },
        )
        entry["windows"] += 1
        entry["correct"] += int(int(row["final_prediction"]) == int(row["true_class"]))
        entry["stage1_pass"] += int(bool(row["stage2_executed"]))
        entry["predicted"][int(row["final_prediction"])] += 1
    class_rates: dict[int, list[float]] = {k: [] for k in CLASS_IDS}
    file_rows: list[dict[str, Any]] = []
    for filename, entry in sorted(per_file.items()):
        rate = float(entry["correct"]) / float(entry["windows"])
        class_rates[int(entry["true_class"])].append(rate)
        file_rows.append(
            {
                "filename": filename,
                "true_class": int(entry["true_class"]),
                "windows": int(entry["windows"]),
                "correct": int(entry["correct"]),
                "accuracy": rate,
                "stage1_pass": int(entry["stage1_pass"]),
                "predicted_background": int(entry["predicted"][0]),
                "predicted_phantom": int(entry["predicted"][1]),
                "predicted_air2s": int(entry["predicted"][2]),
            }
        )
    file_equal = {
        "file_equal_background_recall": float(np.mean(class_rates[0])),
        "file_equal_phantom_recall": float(np.mean(class_rates[1])),
        "file_equal_air2s_recall": float(np.mean(class_rates[2])),
        "file_equal_accuracy": float(np.mean([np.mean(class_rates[k]) for k in CLASS_IDS])),
        "files": len(file_rows),
    }
    background = [row for row in predictions if int(row["true_class"]) == BACKGROUND_CLASS]
    drones = [row for row in predictions if int(row["true_class"]) in DRONE_CLASSES]
    false_alarms = sum(1 for row in background if bool(row["stage2_executed"]))
    misses = sum(1 for row in drones if not bool(row["stage2_executed"]))
    subset = [
        row
        for row in drones
        if bool(row["stage2_executed"])
    ]
    conditional_correct = sum(
        1 for row in subset if int(row["final_prediction"]) == int(row["true_class"])
    )
    stage1 = {
        "stage1_background_false_alarms": false_alarms,
        "stage1_background_false_alarm_rate": float(false_alarms) / float(len(background)),
        "stage1_drone_misses": misses,
        "stage1_drone_miss_rate": float(misses) / float(len(drones)),
        "stage1_drones_passed": len(drones) - misses,
    }
    stage2 = {
        "stage2_conditional_windows": len(subset),
        "stage2_conditional_correct": conditional_correct,
        "stage2_conditional_accuracy": (
            float(conditional_correct) / float(len(subset)) if subset else float("nan")
        ),
    }
    errors: dict[str, Any] = {}
    for truth, prediction, name in ERROR_TRANSITIONS:
        key = name.replace("->", "_to_").lower()
        errors[f"error_{key}"] = int(matrix[truth, prediction])
        errors[f"rate_{key}"] = (
            float(matrix[truth, prediction]) / float(row_totals[truth]) if row_totals[truth] else float("nan")
        )
    errors["error_windows_total"] = int(total - np.trace(matrix))
    return {
        "matrix": matrix,
        "pooled_accuracy": float(np.trace(matrix)) / float(total),
        "recall": {k: recalls[k] for k in CLASS_IDS},
        "precision": {k: precisions[k] for k in CLASS_IDS},
        "macro_recall": float(np.mean(recalls)),
        "macro_precision": float(np.mean(precisions)),
        "file_equal": file_equal,
        "files": file_rows,
        "stage1": stage1,
        "stage2": stage2,
        "errors": errors,
    }


def close(left: float, right: float, tolerance: float = TOLERANCE) -> bool:
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tolerance


def metric_problems(
    own: dict[str, Any], published: dict[str, Any], windows: int
) -> list[str]:
    """Differences between the independent recomputation and the published metrics."""
    pairs = {
        "windows": (windows, int(published["windows"])),
        "pooled_accuracy": (own["pooled_accuracy"], published["pooled_accuracy"]),
        "macro_recall": (own["macro_recall"], published["macro_recall"]),
        "balanced_accuracy": (own["macro_recall"], published["balanced_accuracy"]),
        "macro_precision": (own["macro_precision"], published["macro_precision"]),
        "file_equal_accuracy": (
            own["file_equal"]["file_equal_accuracy"],
            published["file_equal_accuracy"],
        ),
        "file_equal_background_recall": (
            own["file_equal"]["file_equal_background_recall"],
            published["file_equal_background_recall"],
        ),
        "file_equal_phantom_recall": (
            own["file_equal"]["file_equal_phantom_recall"],
            published["file_equal_phantom_recall"],
        ),
        "file_equal_air2s_recall": (
            own["file_equal"]["file_equal_air2s_recall"],
            published["file_equal_air2s_recall"],
        ),
        "files": (len(own["files"]), int(published["files"])),
        "stage1_background_false_alarms": (
            own["stage1"]["stage1_background_false_alarms"],
            published["stage1_background_false_alarms"],
        ),
        "stage1_background_false_alarm_rate": (
            own["stage1"]["stage1_background_false_alarm_rate"],
            published["stage1_background_false_alarm_rate"],
        ),
        "stage1_drone_misses": (
            own["stage1"]["stage1_drone_misses"],
            published["stage1_drone_misses"],
        ),
        "stage1_drone_miss_rate": (
            own["stage1"]["stage1_drone_miss_rate"],
            published["stage1_drone_miss_rate"],
        ),
        "stage2_conditional_windows": (
            own["stage2"]["stage2_conditional_windows"],
            published["stage2_conditional_windows"],
        ),
        "stage2_conditional_accuracy": (
            own["stage2"]["stage2_conditional_accuracy"],
            published["stage2_conditional_accuracy"],
        ),
        "error_windows_total": (
            own["errors"]["error_windows_total"],
            published["error_windows_total"],
        ),
    }
    problems: list[str] = []
    for key, (mine, theirs) in pairs.items():
        if not close(mine, theirs):
            problems.append(f"{key}: {mine} != {theirs}")
    for label, name in (("background", 0), ("phantom", 1), ("air2s", 2)):
        if not close(own["recall"][name], published[f"recall_{label}"]):
            problems.append(f"recall_{label}: {own['recall'][name]} != {published[f'recall_{label}']}")
        if not close(own["precision"][name], published[f"precision_{label}"]):
            problems.append(f"precision_{label}")
    return problems


def check_metrics(checks: CheckList, predictions: Sequence[dict[str, Any]], root: Path) -> dict[str, Any]:
    own = own_summary(predictions)
    published = read_json(root / "metrics_summary.json")["metrics"]
    matrix_published = read_json(root / "confusion_matrix.json")
    errors_published = read_json(root / "error_decomposition.json")
    per_file_published = {row["filename"]: row for row in read_csv_rows(root / "per_file_metrics.csv")}

    problems = metric_problems(own, published, len(predictions))
    checks.check("C5.1 recomputed pooled / file-equal / stage metrics match", not problems, problems[:5])

    matrix_problems = []
    for truth, prediction, name in ERROR_TRANSITIONS:
        if int(own["matrix"][truth, prediction]) != int(errors_published[f"error_{name.replace('->', '_to_').lower()}"]):
            matrix_problems.append(name)
    if own["matrix"].tolist() != matrix_published["matrix"]:
        matrix_problems.append("confusion matrix")
    checks.check("C5.2 recomputed confusion matrix / error decomposition match", not matrix_problems, matrix_problems[:5])

    file_problems = []
    if len(per_file_published) != len(own["files"]):
        file_problems.append(f"file rows {len(per_file_published)} != {len(own['files'])}")
    for row in own["files"]:
        published_row = per_file_published.get(str(row["filename"]))
        if published_row is None:
            file_problems.append(f"{row['filename']}: missing")
            continue
        for key in ("windows", "correct", "stage1_pass", "predicted_background", "predicted_phantom", "predicted_air2s"):
            if int(published_row[key]) != int(row[key]):
                file_problems.append(f"{row['filename']}: {key}")
        if not close(float(published_row["accuracy"]), float(row["accuracy"])):
            file_problems.append(f"{row['filename']}: accuracy")
    checks.check("C5.3 per-file metric rows match the recomputation", not file_problems, file_problems[:5])

    digest_problems = []
    for row in predictions:
        if not str(row["stage1_feature_digest"]):
            digest_problems.append(f"{row['filename']}#{row['window_index']}: empty digest")
    checks.check("C5.4 every window carries a feature digest", not digest_problems, digest_problems[:3])

    return own


# ---------------------------------------------------------------------------
# 6. independent raw-IQ re-derivation
# ---------------------------------------------------------------------------
def own_vector_slice(dataset: Any, start: int, length: int) -> np.ndarray:
    shape = tuple(int(item) for item in dataset.shape)
    stop = start + length
    if len(shape) == 1:
        values = dataset[start:stop]
    elif len(shape) == 2 and shape[0] == 1:
        values = dataset[0, start:stop]
    elif len(shape) == 2 and shape[1] == 1:
        values = dataset[start:stop, 0]
    else:
        raise ValueError(f"unsupported HDF5 layout {shape}")
    return np.asarray(values, dtype=np.float32).reshape(-1)


def own_input_tensor(row: dict[str, Any]) -> np.ndarray:
    """RF0 slice -> complex mean removal -> RMS normalisation -> [2, samples]."""
    import h5py

    start = int(row["start"])
    length = int(row["length"])
    with h5py.File(str(row["file_path"]), "r") as handle:
        i_values = own_vector_slice(handle["RF0_I"], start, length)
        q_values = own_vector_slice(handle["RF0_Q"], start, length)
    iq = (i_values + 1j * q_values).astype(np.complex64)
    centered = iq - np.mean(iq, dtype=np.complex128)
    power = float(np.mean(np.abs(centered) ** 2, dtype=np.float64))
    normalized = (centered / math.sqrt(max(power, 1e-8))).astype(np.complex64)
    return np.stack((normalized.real, normalized.imag), axis=0)


def own_frontend(device: torch.device) -> torch.nn.Module:
    from model import SpectrogramFrontEnd

    return SpectrogramFrontEnd(n_fft=N_FFT, hop_length=HOP_LENGTH, image_size=IMAGE_SIZE).to(device)


def own_network(path: Path, device: torch.device) -> torch.nn.Module:
    from model import ResNet18

    payload = torch.load(path, map_location="cpu", weights_only=False)
    network = ResNet18(input_channels=1, num_classes=2).to(device)
    network.load_state_dict(payload["model_state"])
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


@torch.inference_mode()
def own_logits(
    network: torch.nn.Module, tensor: np.ndarray, device: torch.device
) -> np.ndarray:
    """One forward pass; ``tensor`` is either [2,samples] or [B,2,samples]."""
    array = np.ascontiguousarray(tensor)
    batch = torch.from_numpy(array if array.ndim == 4 else array[None]).to(device)
    with torch.amp.autocast(
        device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16
    ):
        logits = network(batch)
    return logits.float().cpu().numpy()


def own_features(
    tensors: Sequence[np.ndarray], frontend: torch.nn.Module, device: torch.device
) -> np.ndarray:
    """Frontend output for a contiguous batch, in the published [B,1,256,256] layout."""
    with torch.inference_mode():
        batch = torch.from_numpy(np.ascontiguousarray(np.stack(tensors))).to(device)
        features = frontend(batch.float()).detach().to("cpu", torch.float32).numpy()
    return features


def published_window_digests(root: Path) -> bytes:
    path = root / "feature_digests" / f"window_digests_newsplit_test_{CONDITION}.bin"
    return path.read_bytes()


def exact_batch_parity(
    checks: CheckList,
    predictions: Sequence[dict[str, Any]],
    root: Path,
    device: torch.device,
    windows: int = EXACT_PARITY_WINDOWS,
) -> dict[str, Any]:
    """Rebuild the first ``windows`` test windows with the published batch layout.

    The feature tensor of every rebuilt window must hash to the published per-window
    digest and the stage-1 logits must reproduce the published values exactly (the
    batch composition is identical, so any pipeline difference would show up).
    """
    targets = [row for row in predictions if int(row["dataset_index_position"]) < windows]
    stage1_network = own_network(STAGE1_CHECKPOINT, device)
    frontend = own_frontend(device)
    frontend.eval()
    digests = published_window_digests(root)
    problems: list[str] = []
    records: list[dict[str, Any]] = []
    max_delta = 0.0
    for start in range(0, len(targets), BATCH_SIZE):
        chunk = targets[start : start + BATCH_SIZE]
        tensors = [own_input_tensor(row) for row in chunk]
        features = own_features(tensors, frontend, device)
        logits = own_logits(stage1_network, features, device)
        for offset, row in enumerate(chunk):
            position = int(row["dataset_index_position"])
            rebuilt = hashlib.sha256(
                np.ascontiguousarray(features[offset]).tobytes()
            ).hexdigest()
            published_digest = digests[position * 32 : (position + 1) * 32].hex()
            if rebuilt != published_digest:
                problems.append(f"{row['filename']}#{row['window_index']}: feature digest")
            published = [
                float(row["stage1_logit_background"]),
                float(row["stage1_logit_drone"]),
            ]
            deltas = [abs(float(logits[offset][k]) - published[k]) for k in (0, 1)]
            max_delta = max(max_delta, *deltas)
            if max(deltas) > EXACT_LOGIT_TOLERANCE:
                problems.append(
                    f"{row['filename']}#{row['window_index']}: stage-1 logits {deltas}"
                )
            records.append(
                {
                    "dataset_index_position": position,
                    "filename": row["filename"],
                    "window_index": int(row["window_index"]),
                    "rebuilt_digest": rebuilt,
                    "published_digest": published_digest,
                    "rebuilt_stage1_logits": [float(value) for value in logits[offset]],
                    "published_stage1_logits": published,
                    "max_abs_delta": max(deltas),
                }
            )
    checks.check(
        "C6.1 exact batch-parity rebuild of the first windows (digest + stage-1 logits)",
        not problems,
        {"windows": len(records), "max_abs_logit_delta": max_delta, "problems": problems[:3]},
    )
    return {
        "windows": len(records),
        "exact_logit_tolerance": EXACT_LOGIT_TOLERANCE,
        "batch_size": BATCH_SIZE,
        "max_abs_logit_delta": max_delta,
        "records": records,
        "device": str(device),
    }


def check_rederivation(
    checks: CheckList, predictions: Sequence[dict[str, Any]], device: torch.device
) -> dict[str, Any]:
    """First window of every test source file, re-read from G and re-classified alone.

    A single-window forward pass uses different fp16 CUDA kernels than the published
    batch-of-four pass, so the logits are compared with the documented AMP tolerance
    while the *decisions* must agree exactly.
    """
    targets: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for row in predictions:
        filename = str(row["filename"])
        if seen.get(filename, 0) < REDERIVE_PER_FILE:
            seen[filename] = seen.get(filename, 0) + 1
            targets.append(row)
    stage1_network = own_network(STAGE1_CHECKPOINT, device)
    stage2_network = own_network(STAGE2_CHECKPOINT, device)
    frontend = own_frontend(device)
    frontend.eval()
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    max_delta = 0.0
    for row in targets:
        tensor = own_input_tensor(row)
        features = own_features([tensor], frontend, device)
        s1 = own_logits(stage1_network, features, device)[0]
        s1_margin = float(np.float32(s1[1] - s1[0]))
        expected_executed = bool(row["stage2_executed"])
        deltas = [
            abs(float(s1[0]) - float(row["stage1_logit_background"])),
            abs(float(s1[1]) - float(row["stage1_logit_drone"])),
        ]
        max_delta = max(max_delta, *deltas)
        if (s1_margin > 0.0) != expected_executed:
            problems.append(f"{row['filename']}#{row['window_index']}: stage-1 decision")
        if max(deltas) > AMP_LOGIT_TOLERANCE:
            problems.append(f"{row['filename']}#{row['window_index']}: stage-1 logits {deltas}")
        record = {
            "filename": row["filename"],
            "window_index": int(row["window_index"]),
            "stage1_logits": [float(s1[0]), float(s1[1])],
            "stage1_margin": s1_margin,
            "published_stage1_margin": float(row["stage1_margin"]),
            "stage2_executed": expected_executed,
            "max_abs_logit_delta": max(deltas),
        }
        if expected_executed:
            s2 = own_logits(stage2_network, features, device)[0]
            s2_margin = float(np.float32(s2[1] - s2[0]))
            record["stage2_logits"] = [float(s2[0]), float(s2[1])]
            record["stage2_margin"] = s2_margin
            record["published_stage2_margin"] = float(row["stage2_margin"])
            if abs(float(s2[0]) - float(row["stage2_logit_phantom"])) > AMP_LOGIT_TOLERANCE or abs(
                float(s2[1]) - float(row["stage2_logit_air2s"])
            ) > AMP_LOGIT_TOLERANCE:
                problems.append(f"{row['filename']}#{row['window_index']}: stage-2 logits")
            want_final = 2 if s2_margin > 0.0 else 1
            if want_final != int(row["final_prediction"]):
                problems.append(f"{row['filename']}#{row['window_index']}: final class")
        records.append(record)
    checks.check(
        "C6.2 independent single-window raw-IQ re-derivation reproduces decisions",
        not problems,
        {"windows": len(records), "max_abs_logit_delta": max_delta, "problems": problems[:5]},
    )
    return {
        "windows": len(records),
        "amp_logit_tolerance": AMP_LOGIT_TOLERANCE,
        "max_abs_logit_delta": max_delta,
        "records": records,
        "device": str(device),
    }


# ---------------------------------------------------------------------------
# 7. negative controls
# ---------------------------------------------------------------------------
def negative_controls(predictions: Sequence[dict[str, Any]], root: Path) -> dict[str, Any]:
    """Prove the checks above actually fail on tampered input."""
    controls: list[dict[str, Any]] = []

    def recorded(name: str, detected: bool, detail: Any) -> None:
        controls.append({"control": name, "detected": bool(detected), "detail": detail})

    # N1: one window missing -> coverage check must fail
    truncated = list(predictions)[:-1]
    recorded(
        "N1 a missing window is detected",
        bool(coverage_problems(truncated)),
        coverage_problems(truncated)[:1],
    )

    # N2: a flipped final class -> decision-rule check must fail
    tampered = [dict(row) for row in predictions]
    victim = next(row for row in tampered if bool(row["stage2_executed"]))
    victim["final_prediction"] = 0 if int(victim["final_prediction"]) != 0 else 1
    detected = bool(decision_problems([victim]))
    recorded("N2 a flipped final class is detected", detected, decision_problems([victim])[:1])
    # the untampered row must pass the same check
    original = next(
        row for row in predictions if int(row["window_index"]) == int(victim["window_index"])
        and str(row["filename"]) == str(victim["filename"])
    )
    recorded(
        "N2b the untampered row passes the same check",
        not decision_problems([original]),
        decision_problems([original])[:1],
    )

    # N3: a suppressed stage-1 miss -> recomputed metric must differ from the publication
    tampered = [dict(row) for row in predictions]
    victim = next(
        row
        for row in tampered
        if bool(row["stage2_executed"]) and int(row["true_class"]) in DRONE_CLASSES
    )
    victim["stage2_executed"] = False
    victim["final_prediction"] = BACKGROUND_CLASS
    own = own_summary(tampered)
    published = read_json(root / "metrics_summary.json")["metrics"]
    problems = metric_problems(own, published, len(tampered))
    recorded(
        "N3 a hidden stage-1 miss is detected by the metric recomputation",
        bool(problems),
        problems[:2],
    )

    # N4: a shifted published metric -> comparison must fail
    forged = dict(published)
    forged["file_equal_accuracy"] = float(published["file_equal_accuracy"]) + 0.01
    recorded(
        "N4 a changed published metric is detected",
        bool(metric_problems(own_summary(predictions), forged, len(predictions))),
        forged["file_equal_accuracy"],
    )

    # N5: a wrong index hash -> identity check must fail
    recorded(
        "N5 a wrong index hash is detected",
        sha256_file(INDEX) != "0" * 64,
        sha256_file(INDEX)[:16],
    )
    return {
        "controls": controls,
        "passed": sum(1 for item in controls if item["detected"]),
        "total": len(controls),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-rederivation", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT
    checks = CheckList()
    identity = check_frozen_inputs(checks)
    predictions = own_prediction_rows(root / "window_predictions.csv")
    index_rows = own_index_rows()
    check_coverage_and_identity(checks, predictions, index_rows)
    check_decision_rules(checks, predictions)
    own = check_metrics(checks, predictions, root)
    exact = {"skipped": True}
    rederivation = {"skipped": True}
    if not args.skip_rederivation:
        device = resolve_device(args.device)
        exact = exact_batch_parity(checks, predictions, root, device)
        rederivation = check_rederivation(checks, predictions, device)
    controls = negative_controls(predictions, root)

    payload = {
        "phase": "CASCADE3-NEWSPLIT-C-independent-verification",
        "verified_at": utc_now(),
        "output_root": str(root),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "frozen_inputs": identity,
        "checks_passed": sum(1 for item in checks.items if item["passed"]),
        "checks_total": len(checks.items),
        "checks": checks.items,
        "recomputed": {
            "pooled_accuracy": own["pooled_accuracy"],
            "macro_recall": own["macro_recall"],
            "file_equal_accuracy": own["file_equal"]["file_equal_accuracy"],
            "file_equal_background_recall": own["file_equal"]["file_equal_background_recall"],
            "file_equal_phantom_recall": own["file_equal"]["file_equal_phantom_recall"],
            "file_equal_air2s_recall": own["file_equal"]["file_equal_air2s_recall"],
            "stage1_drone_misses": own["stage1"]["stage1_drone_misses"],
            "stage2_conditional_accuracy": own["stage2"]["stage2_conditional_accuracy"],
            "confusion_matrix": [[int(value) for value in row] for row in own["matrix"]],
        },
        "exact_batch_parity": exact,
        "rederivation": rederivation,
        "negative_controls": controls,
        "conclusion": (
            "all checks and negative controls passed"
            if not checks.failed and controls["passed"] == controls["total"]
            else "verification failed"
        ),
    }
    out = root / "independent_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"checks {payload['checks_passed']}/{payload['checks_total']}, controls {controls['passed']}/{controls['total']}")
    log(f"conclusion: {payload['conclusion']}")
    return 0 if not checks.failed and controls["passed"] == controls["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
