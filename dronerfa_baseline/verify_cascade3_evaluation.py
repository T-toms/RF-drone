"""Independent verification of the CASCADE3 artefacts.

Reads only the produced CSV/JSON/PNG files, the frozen configuration and the raw
IQ data, and recomputes every reported number with its own code: the run module
never supplies an expected value to these checks.  In addition a bounded target set
is re-derived from raw IQ with the documented batch composition (contiguous chunk of
4 over the full split for the shared feature tensor, contiguous chunk of 4 over the
executed rows for stage 2) and compared with the published logits and digests.

    python verify_cascade3_evaluation.py [--output-dir ROOT] [--fixture]
                                        [--targets-per-class 3] [--skip-targets]
                                        [--device auto|cuda|cpu] [--no-reexec]
                                        [--input-probe-groups heldout:clean]

Runtime environment
-------------------
The formal run pinned its arithmetic in ``environment.json`` (torch build + device).
The published per-window feature digests and logits are byte-reproducible only under
that arithmetic: the CUDA frontend/AMP path the run used.  Recomputing on CPU (or
with another torch build) is *not* the formal pipeline — the STFT and the fp32/fp16
kernels differ in the last bits, so nearly every digest differs and the logits move
by ~1e-3.  The verifier therefore records both environments, classifies the run as

* ``formal_environment`` - torch build and device match the formal run; the strict
  criteria apply (byte-identical window digests, |dlogit| <= 1e-4, identical
  discrete decisions), and
* ``cross_environment`` - they do not match, so the digest criterion is *not*
  applicable; the verifier then requires exact inputs (own raw-IQ/noise rebuild and
  the recorded input probe), bounded numeric drift (<= CROSS_ENV_LOGIT_BOUND) and
  identical discrete decisions instead.

When the current interpreter cannot reproduce the formal arithmetic but the project
``.venv`` interpreter is present, the verifier re-executes itself with that
interpreter (``--no-reexec`` disables this, and the report always records it).

``--fixture`` skips the frozen-input re-hash and the full-size expectations, which
lets ``test_cascade3_evaluation.py`` point the verifier at a small bounded tree and
prove that tampered predictions, digests or metric tables are rejected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "outputs" / ".matplotlib"))

import numpy as np
import torch

from model import ResNet18, SpectrogramFrontEnd
from train_resnet18 import extract_features

CONFIG_PATH = PROJECT_ROOT / "work" / "cascade3-config-20260919.json"
OUTPUT_ROOT = PROJECT_DIR / "outputs" / "cascade3_evaluation_20260919"
PRESERVED_INPUTS = PROJECT_ROOT / "work" / "codex-preserved-inputs-20260917.json"
STAGE1_PUBLISHED_ROOT = PROJECT_DIR / "outputs" / "stage1_method_comparison_20260919" / "nfft"

N_FFT = 512
HOP_LENGTH = 256
IMAGE_SIZE = 256
BATCH_SIZE = 4
TRAINING_SEEDS = (2026, 2027, 2028)
CONDITIONS = ("clean", "-10", "-12", "-15")
DATASET_NAMES = ("heldout", "old_test")
CLASS_NAMES = {0: "Background", 1: "DJI Phantom 4 Pro", 2: "DJI Air 2S"}
CLASS_IDS = (0, 1, 2)
TARGET_CODE_TO_CLASS = {"0000": 0, "0010": 1, "0101": 2}
DRONE_CLASSES = (1, 2)
BACKGROUND_CLASS = 0
NOISE_SEED = 22026
NOISE_EPOCH = 0
FEATURE_DTYPE = np.float32

EXPECTED_WINDOWS = {"heldout": 1820, "old_test": 440}
EXPECTED_CONDITIONS = {"heldout": CONDITIONS, "old_test": ("clean",)}
EXPECTED_ROWS = {"heldout": 21840, "old_test": 1320}
EXPECTED_CLASS_WINDOWS = {
    "heldout": {0: 560, 1: 610, 2: 650},
    "old_test": {0: 140, 1: 140, 2: 160},
}
CROSS_SEED_METRICS = (
    "file_equal_accuracy",
    "file_equal_background_recall",
    "file_equal_phantom_recall",
    "file_equal_air2s_recall",
    "pooled_accuracy",
    "macro_recall",
    "macro_precision",
    "macro_f1",
    "recall_background",
    "recall_phantom",
    "recall_air2s",
    "stage1_background_false_alarm_rate",
    "stage1_drone_miss_rate",
    "stage2_conditional_accuracy",
)
ERROR_TRANSITIONS = (
    (0, 1, "Background->Phantom"),
    (0, 2, "Background->Air2S"),
    (1, 0, "Phantom->Background"),
    (2, 0, "Air2S->Background"),
    (1, 2, "Phantom->Air2S"),
    (2, 1, "Air2S->Phantom"),
)
METRIC_NUMERIC_FIELDS = (
    "file_equal_accuracy",
    "file_equal_background_recall",
    "file_equal_phantom_recall",
    "file_equal_air2s_recall",
    "pooled_accuracy",
    "macro_recall",
    "balanced_accuracy",
    "macro_precision",
    "macro_f1",
    "recall_background",
    "recall_phantom",
    "recall_air2s",
    "precision_background",
    "precision_phantom",
    "precision_air2s",
    "f1_background",
    "f1_phantom",
    "f1_air2s",
    "stage1_background_false_alarm_rate",
    "stage1_background_false_alarm_rate_file_equal",
    "stage1_drone_miss_rate",
    "stage1_drone_miss_rate_file_equal",
    "stage2_conditional_accuracy",
)
TOLERANCE = 1e-9
LOGIT_TOLERANCE = 1e-4
# Cross-environment bound: the largest stage-1/stage-2 logit drift actually measured
# between the formal CUDA/fp16-AMP path and a pure CPU recomputation over the frozen
# target set is 0.0025 (stage 1) and 0.0071 (stage 2).  The bound below is ~7x that
# measured maximum and is deliberately not fitted to it; it only ever applies when
# the runtime environment does not match the formal one, and even then the discrete
# decisions of every target window must still agree exactly.
CROSS_ENV_LOGIT_BOUND = 5e-2
TIER_FORMAL = "formal_environment"
TIER_CROSS = "cross_environment"
FORMAL_ENVIRONMENT_FILE = "environment.json"
INPUT_PROBE_SAMPLES = 4096
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
REEXEC_MARKER = "CASCADE3_VERIFIER_REEXEC"
REEXEC_ORIGIN = "CASCADE3_VERIFIER_REEXEC_FROM"
TARGET_RULE_COUNTERS = (
    "stage1_gate_changes",
    "stage1_margin_out_of_tolerance",
    "stage2_class_changes",
    "stage2_margin_out_of_tolerance",
    "stage2_batch_id_errors",
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
REQUIRED_REPORT_TOKENS = (
    "CASCADE3",
    "n_fft=512",
    "hop_length=256",
    "stage2_executed",
    "文件等权",
    "描述性",
    "跨会话",
    "不重训",
    "oracle",
    "1,000,000",
    "22026",
)


# ---------------------------------------------------------------------------
# small independent helpers
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# runtime environment and numeric-comparison tier
# ---------------------------------------------------------------------------
def resolve_device(requested: str = "auto") -> torch.device:
    """``auto`` prefers CUDA, ``cuda`` insists on it, ``cpu`` forces the CPU."""
    if requested == "cpu":
        return torch.device("cpu")
    available = bool(torch.cuda.is_available())
    if requested == "cuda" and not available:
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    if requested == "cuda" or (requested == "auto" and available):
        return torch.device("cuda")
    return torch.device("cpu")


def runtime_environment(device: torch.device) -> dict[str, Any]:
    """Everything that decides whether the formal arithmetic can be reproduced."""
    record: dict[str, Any] = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torch_cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "numpy": str(np.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device.type,
    }
    if device.type == "cuda":
        record["device_name"] = torch.cuda.get_device_name(0)
    return record


def comparison_tier(
    environment: dict[str, Any], formal: dict[str, Any] | None
) -> tuple[str, list[str]]:
    """``formal_environment`` only when the runtime can reproduce the recorded run."""
    reasons: list[str] = []
    if not formal:
        reasons.append(f"the formal run did not record {FORMAL_ENVIRONMENT_FILE}")
        return TIER_CROSS, reasons
    if str(formal.get("torch")) != str(environment.get("torch")):
        reasons.append(
            f"torch {environment.get('torch')} does not match the formal {formal.get('torch')}"
        )
    if str(formal.get("device")) != str(environment.get("device")):
        reasons.append(
            f"device {environment.get('device')} does not match the formal {formal.get('device')}"
        )
    return (TIER_FORMAL if not reasons else TIER_CROSS), reasons


def plan_reexec(
    tier: str,
    *,
    enabled: bool,
    marker_present: bool,
    python_executable: str,
    venv_python: Path = VENV_PYTHON,
) -> dict[str, Any]:
    """Decide whether to re-run this verifier under the project interpreter."""
    plan: dict[str, Any] = {"needed": False, "reason": None, "interpreter": None}
    if tier == TIER_FORMAL:
        plan["reason"] = "the current interpreter already reproduces the formal run"
        return plan
    if not enabled:
        plan["reason"] = "disabled by --no-reexec"
        return plan
    if marker_present:
        plan["reason"] = "already re-executed once; using the cross-environment tier"
        return plan
    if not venv_python.is_file():
        plan["reason"] = f"{venv_python} does not exist"
        return plan
    if Path(python_executable).resolve() == Path(venv_python).resolve():
        plan["reason"] = "the current interpreter already is the project interpreter"
        return plan
    plan.update(
        {
            "needed": True,
            "reason": "the current interpreter cannot reproduce the formal arithmetic",
            "interpreter": str(venv_python),
        }
    )
    return plan


# ---------------------------------------------------------------------------
# GPU-independent input path (own reader, own noise, own probe payload)
# ---------------------------------------------------------------------------
def probe_payload(array: np.ndarray, samples: int = INPUT_PROBE_SAMPLES) -> bytes:
    """The recorded input probe payload: the leading samples of one window tensor."""
    return np.ascontiguousarray(array[:, :samples]).tobytes()


def own_vector_slice(dataset: Any, start: int, length: int) -> np.ndarray:
    """Own RF0 channel slicer for the three documented HDF5 layouts."""
    stop = start + length
    shape = tuple(int(item) for item in dataset.shape)
    if len(shape) == 1:
        values = dataset[start:stop]
    elif len(shape) == 2 and shape[0] == 1:
        values = dataset[0, start:stop]
    elif len(shape) == 2 and shape[1] == 1:
        values = dataset[start:stop, 0]
    else:
        raise ValueError(f"unsupported HDF5 layout {shape}")
    return np.asarray(values, dtype=np.float32).reshape(-1)


def build_input_tensor(
    row: dict[str, Any], condition: str, noise_seed: int, split_length: int
) -> np.ndarray:
    """Rebuild one raw+noise input window without ``dronerfa_data`` and without any GPU.

    The documented chain is reproduced independently: RF0 I/Q slice -> remove the
    complex mean -> RMS normalize -> optional complex AWGN drawn from
    ``default_rng(noise_seed + epoch*len(full_split) + full_split_position)`` with the
    epoch pinned to 0 -> ``[2, samples]`` float32.
    """
    import h5py

    start = int(row["start"])
    length = int(row["length"])
    with h5py.File(str(row["file_path"]), "r") as handle:
        i_values = own_vector_slice(handle["RF0_I"], start, length)
        q_values = own_vector_slice(handle["RF0_Q"], start, length)
    iq = i_values + 1j * q_values
    centered = iq - np.mean(iq, dtype=np.complex128)
    power = float(np.mean(np.abs(centered) ** 2, dtype=np.float64))
    normalized = (centered / math.sqrt(max(power, 1e-8))).astype(np.complex64)
    if condition != "clean":
        snr_db = float(condition)
        rng = np.random.default_rng(
            noise_seed + 0 * int(split_length) + int(row["dataset_index_position"])
        )
        signal_power = float(np.mean(np.abs(normalized) ** 2, dtype=np.float64))
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        scale = math.sqrt(noise_power / 2.0)
        noise = scale * (
            rng.standard_normal(normalized.shape, dtype=np.float32)
            + 1j * rng.standard_normal(normalized.shape, dtype=np.float32)
        )
        normalized = (normalized + noise).astype(np.complex64)
    return np.stack((normalized.real, normalized.imag), axis=0)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: Any, right: Any, tolerance: float = TOLERANCE) -> bool:
    left_nan = left is None or (isinstance(left, float) and not np.isfinite(left))
    right_nan = right is None or (isinstance(right, float) and not np.isfinite(right))
    if left_nan or right_nan:
        return bool(left_nan and right_nan)
    return abs(float(left) - float(right)) <= tolerance


def text_to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def parse_prediction_rows(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in read_csv_rows(path):
        record: dict[str, Any] = dict(raw)
        for key in (
            "training_seed",
            "stage1_epoch",
            "stage2_epoch",
            "dataset_index_position",
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
            record[key] = int(str(raw[key]).strip())
        for key in ("stage1_logit_background", "stage1_logit_drone", "stage1_margin"):
            record[key] = float(str(raw[key]).strip())
        for key in ("stage2_logit_phantom", "stage2_logit_air2s", "stage2_margin"):
            record[key] = text_to_float(raw.get(key))
        record["stage2_executed"] = str(raw.get("stage2_executed", "")).strip().lower() == "true"
        batch_id = str(raw.get("stage2_batch_id", "")).strip()
        record["stage2_batch_id"] = int(batch_id) if batch_id else None
        record["stage2_feature_digest"] = (
            str(raw.get("stage2_feature_digest", "")).strip() or None
        )
        records.append(record)
    return records


def index_rows(index_path: Path, split: str) -> list[dict[str, Any]]:
    """Own reader of a frozen index (no code shared with the run module).

    ``dataset_index_position`` is the position **inside the requested split**, exactly
    like the frozen reader: the old index holds train/val/test in one file, so the row
    number of the file is not the row position of the split.
    """
    rows: list[dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["split"] != split:
                continue
            code = raw["target_code"]
            if code not in TARGET_CODE_TO_CLASS:
                raise ValueError(f"unexpected target_code {code}")
            rows.append(
                {
                    "dataset_index_position": len(rows),
                    "file_path": raw["file_path"],
                    "filename": raw["filename"],
                    "target_code": code,
                    "target_name": raw["target_name"],
                    "distance_code": raw["distance_code"],
                    "segment_code": raw["segment_code"],
                    "row_split": raw["split"],
                    "stage1_label": int(raw["label"]),
                    "true_class": TARGET_CODE_TO_CLASS[code],
                    "block_index": int(raw["block_index"]),
                    "window_index": int(raw["window_index"]),
                    "start": int(raw["start"]),
                    "length": int(raw["length"]),
                    "source_samples": int(raw["source_samples"]),
                }
            )
    return rows


def config_dataset_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    datasets = config["datasets"]
    return {
        "heldout": {
            "split": "heldout",
            "index_path": PROJECT_ROOT / str(datasets["heldout_index"]),
            "index_sha256": str(datasets["heldout_index_sha256"]),
        },
        "old_test": {
            "split": "test",
            "index_path": PROJECT_ROOT / str(datasets["old_test_index"]),
            "index_sha256": str(datasets["old_index_sha256"]),
        },
    }


def checkpoint_expectations(config: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for pairing in config["seed_pairing"]:
        for stage in (1, 2):
            entries.append(
                {
                    "training_seed": int(pairing["seed"]),
                    "stage": stage,
                    "path": PROJECT_ROOT / str(pairing[f"stage{stage}_checkpoint"]),
                    "sha256": str(pairing[f"stage{stage}_sha256"]),
                    "bytes": int(pairing[f"stage{stage}_bytes"]),
                    "epoch": int(pairing[f"stage{stage}_epoch"]),
                }
            )
    return entries


# ---------------------------------------------------------------------------
# independent metric recomputation
# ---------------------------------------------------------------------------
def confusion(records: Sequence[dict[str, Any]]) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for record in records:
        matrix[int(record["true_class"]), int(record["final_prediction"])] += 1
    return matrix


def pooled(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    matrix = confusion(records)
    total = int(matrix.sum())
    rows = matrix.sum(axis=1)
    columns = matrix.sum(axis=0)
    recall = {
        label: (float(matrix[label, label]) / float(rows[label]) if rows[label] else float("nan"))
        for label in CLASS_IDS
    }
    precision = {
        label: (
            float(matrix[label, label]) / float(columns[label])
            if columns[label]
            else float("nan")
        )
        for label in CLASS_IDS
    }
    f1 = {
        label: (
            2 * precision[label] * recall[label] / (precision[label] + recall[label])
            if recall[label] + precision[label] > 0
            else 0.0
        )
        for label in CLASS_IDS
    }
    finite_recall = [value for value in recall.values() if np.isfinite(value)]
    finite_precision = [value for value in precision.values() if np.isfinite(value)]
    return {
        "windows": total,
        "background_windows": int(rows[0]),
        "phantom_windows": int(rows[1]),
        "air2s_windows": int(rows[2]),
        "pooled_accuracy": float(np.trace(matrix)) / float(total) if total else float("nan"),
        "macro_recall": float(np.mean(finite_recall)) if finite_recall else float("nan"),
        "balanced_accuracy": float(np.mean(finite_recall)) if finite_recall else float("nan"),
        "macro_precision": float(np.mean(finite_precision)) if finite_precision else float("nan"),
        "macro_f1": float(np.mean(list(f1.values()))) if total else float("nan"),
        "recall_background": recall[0],
        "recall_phantom": recall[1],
        "recall_air2s": recall[2],
        "precision_background": precision[0],
        "precision_phantom": precision[1],
        "precision_air2s": precision[2],
        "f1_background": f1[0],
        "f1_phantom": f1[1],
        "f1_air2s": f1[2],
    }


def file_equal(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for record in records:
        filename = str(record["filename"])
        truth = int(record["true_class"])
        entry = files.setdefault(
            filename,
            {
                "true_class": truth,
                "windows": 0,
                "correct": 0,
                "stage1_pass": 0,
                "predicted": {label: 0 for label in CLASS_IDS},
            },
        )
        entry["windows"] += 1
        entry["correct"] += int(int(record["final_prediction"]) == truth)
        entry["stage1_pass"] += int(bool(record["stage2_executed"]))
        entry["predicted"][int(record["final_prediction"])] += 1
    class_rates: dict[int, list[float]] = {label: [] for label in CLASS_IDS}
    for entry in files.values():
        class_rates[int(entry["true_class"])].append(
            float(entry["correct"]) / float(max(int(entry["windows"]), 1))
        )
    class_rate = {
        label: (float(np.mean(class_rates[label])) if class_rates[label] else float("nan"))
        for label in CLASS_IDS
    }
    available = [class_rate[label] for label in CLASS_IDS if np.isfinite(class_rate[label])]
    return {
        "files_background": len(class_rates[0]),
        "files_phantom": len(class_rates[1]),
        "files_air2s": len(class_rates[2]),
        "file_equal_background_recall": class_rate[0],
        "file_equal_phantom_recall": class_rate[1],
        "file_equal_air2s_recall": class_rate[2],
        "file_equal_accuracy": float(np.mean(available)) if available else float("nan"),
        "file_equal_classes_used": len(available),
        "files": files,
    }


def stage1_statistics(
    records: Sequence[dict[str, Any]], files: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    background = [record for record in records if int(record["true_class"]) == BACKGROUND_CLASS]
    drones = [record for record in records if int(record["true_class"]) in DRONE_CLASSES]
    false_alarms = sum(1 for record in background if bool(record["stage2_executed"]))
    misses = sum(1 for record in drones if not bool(record["stage2_executed"]))
    background_files = [
        entry for entry in files.values() if int(entry["true_class"]) == BACKGROUND_CLASS
    ]
    drone_files = [entry for entry in files.values() if int(entry["true_class"]) in DRONE_CLASSES]
    false_alarm_file_rates = [
        float(entry["stage1_pass"]) / float(max(int(entry["windows"]), 1))
        for entry in background_files
    ]
    miss_file_rates = [
        float(int(entry["windows"]) - int(entry["stage1_pass"]))
        / float(max(int(entry["windows"]), 1))
        for entry in drone_files
    ]
    return {
        "stage1_background_false_alarms": false_alarms,
        "stage1_background_false_alarm_rate": (
            float(false_alarms) / float(len(background)) if background else float("nan")
        ),
        "stage1_background_false_alarm_rate_file_equal": (
            float(np.mean(false_alarm_file_rates)) if false_alarm_file_rates else float("nan")
        ),
        "stage1_drone_misses": misses,
        "stage1_drone_miss_rate": float(misses) / float(len(drones)) if drones else float("nan"),
        "stage1_drone_miss_rate_file_equal": (
            float(np.mean(miss_file_rates)) if miss_file_rates else float("nan")
        ),
        "stage1_drones_passed": len(drones) - misses,
    }


def conditional(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    subset = [
        record
        for record in records
        if int(record["true_class"]) in DRONE_CLASSES and bool(record["stage2_executed"])
    ]
    correct = sum(
        1 for record in subset if int(record["final_prediction"]) == int(record["true_class"])
    )
    return {
        "stage2_conditional_windows": len(subset),
        "stage2_conditional_correct": correct,
        "stage2_conditional_accuracy": (
            float(correct) / float(len(subset)) if subset else float("nan")
        ),
    }


def decomposition(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    matrix = confusion(records)
    rows = matrix.sum(axis=1)
    result: dict[str, Any] = {}
    for truth, prediction, name in ERROR_TRANSITIONS:
        key = name.replace("->", "_to_").lower()
        count = int(matrix[truth, prediction])
        result[f"error_{key}"] = count
        result[f"rate_{key}"] = (
            float(count) / float(rows[truth]) if rows[truth] else float("nan")
        )
    result["error_windows_total"] = int(matrix.sum() - np.trace(matrix))
    return result


def recompute_group(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pooled_values = pooled(records)
    file_values = file_equal(records)
    files = file_values.pop("files")
    return {
        **pooled_values,
        **file_values,
        **stage1_statistics(records, files),
        **conditional(records),
        **decomposition(records),
        "per_file": {
            name: {
                "true_class": int(entry["true_class"]),
                "windows": int(entry["windows"]),
                "correct": int(entry["correct"]),
                "accuracy": float(entry["correct"]) / float(max(int(entry["windows"]), 1)),
                "stage1_pass": int(entry["stage1_pass"]),
                "predicted_background": int(entry["predicted"][0]),
                "predicted_phantom": int(entry["predicted"][1]),
                "predicted_air2s": int(entry["predicted"][2]),
            }
            for name, entry in files.items()
        },
    }


def sample_std(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def group_records(
    records: Sequence[dict[str, Any]]
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(
            (str(record["dataset"]), str(record["condition"]), int(record["training_seed"])), []
        ).append(record)
    return grouped


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_frozen_inputs(failures: list[str], record: dict[str, Any], root: Path) -> None:
    config = read_json(CONFIG_PATH)
    config_sha = sha256_file(CONFIG_PATH)
    record["config_path"] = str(CONFIG_PATH)
    record["config_sha256"] = config_sha
    if int(config["frontend"]["n_fft"]) != N_FFT or int(config["frontend"]["hop_length"]) != HOP_LENGTH:
        failures.append("config frontend is no longer n_fft=512 / hop_length=256")
    if int(config["frontend"]["image_size"]) != IMAGE_SIZE:
        failures.append("config image_size is no longer 256")
    spec_by_name = config_dataset_specs(config)
    record["indices"] = {}
    for name, spec in spec_by_name.items():
        actual = sha256_file(spec["index_path"])
        record["indices"][name] = {
            "path": str(spec["index_path"]),
            "expected": spec["index_sha256"],
            "actual": actual,
            "match": actual == spec["index_sha256"],
        }
        if actual != spec["index_sha256"]:
            failures.append(f"{name} index SHA256 mismatch")
    record["checkpoints"] = {}
    for entry in checkpoint_expectations(config):
        key = f"stage{entry['stage']}_seed{entry['training_seed']}"
        if not entry["path"].is_file():
            failures.append(f"missing checkpoint {entry['path']}")
            record["checkpoints"][key] = {"exists": False}
            continue
        actual = sha256_file(entry["path"])
        actual_bytes = entry["path"].stat().st_size
        record["checkpoints"][key] = {
            "path": str(entry["path"]),
            "expected": entry["sha256"],
            "actual": actual,
            "bytes": actual_bytes,
            "match": actual == entry["sha256"] and actual_bytes == entry["bytes"],
        }
        if actual != entry["sha256"]:
            failures.append(f"{key} checkpoint SHA256 mismatch: {actual}")
        if actual_bytes != entry["bytes"]:
            failures.append(f"{key} checkpoint size mismatch: {actual_bytes}")
    preserved = read_json(PRESERVED_INPUTS)
    preserved_errors: list[str] = []
    for item in preserved["files"]:
        path = Path(str(item["path"]))
        if not path.is_file():
            preserved_errors.append(f"missing {path}")
            continue
        actual = sha256_file(path)
        if actual != str(item["sha256"]):
            preserved_errors.append(f"{path} SHA256 mismatch")
    record["preserved_inputs"] = {
        "files": len(preserved["files"]),
        "errors": preserved_errors,
    }
    failures.extend(preserved_errors)
    for stage, name in (("before", "frozen_inputs_verification_before.json"), ("after", "frozen_inputs_verification.json")):
        path = root / name
        if not path.is_file():
            failures.append(f"missing {name}")
            continue
        stored = read_json(path)
        record[f"{stage}_record"] = {
            "file": name,
            "sha256": sha256_file(path),
            "passed": stored.get("passed"),
            "config_sha256": stored.get("config_sha256"),
            "errors": stored.get("errors"),
        }
        if stored.get("passed") is not True:
            failures.append(f"{name} reports passed={stored.get('passed')}")
        if stored.get("config_sha256") != config_sha:
            failures.append(f"{name} was produced for another config hash")
    before = root / "frozen_inputs_verification_before.json"
    after = root / "frozen_inputs_verification.json"
    if before.is_file() and after.is_file():
        unchanged = read_json(before).get("config_sha256") == read_json(after).get("config_sha256")
        record["config_unchanged_during_inference"] = unchanged
        if not unchanged:
            failures.append("the frozen config hash changed between the two integrity checks")


def validate_records(
    records: Sequence[dict[str, Any]],
    dataset: str,
    split: str,
    index_path: Path,
    epochs: dict[int, dict[str, int]],
    failures: list[str],
    record: dict[str, Any],
    strict: bool,
) -> dict[int, dict[str, Any]]:
    """Primary key, coverage, identity, label mapping and epoch checks."""
    if not records:
        if strict:
            failures.append(f"{dataset}: no prediction rows")
        return {}
    expected_windows = EXPECTED_WINDOWS[dataset]
    expected_conditions = EXPECTED_CONDITIONS[dataset]
    expected_seeds = list(TRAINING_SEEDS)
    keys: set[tuple[str, str, int, int]] = set()
    seen: dict[tuple[str, int], set[int]] = {}
    duplicated = 0
    identity_errors = 0
    label_errors = 0
    epoch_errors = 0
    for one in records:
        key = (
            str(one["dataset"]),
            str(one["condition"]),
            int(one["training_seed"]),
            int(one["dataset_index_position"]),
        )
        if key in keys:
            duplicated += 1
        keys.add(key)
        seen.setdefault((key[1], key[2]), set()).add(key[3])
        if int(one["true_class"]) not in CLASS_IDS:
            label_errors += 1
        expected_class = TARGET_CODE_TO_CLASS[str(one["target_code"])]
        if expected_class != int(one["true_class"]):
            label_errors += 1
        expected_label = 0 if expected_class == BACKGROUND_CLASS else 1
        if int(one["stage1_label"]) != expected_label:
            label_errors += 1
        if str(one["true_class_name"]) != CLASS_NAMES[expected_class]:
            label_errors += 1
        if str(one["final_class_name"]) != CLASS_NAMES[int(one["final_prediction"])]:
            label_errors += 1
        if int(one["training_seed"]) in epochs:
            if int(one["stage1_epoch"]) != epochs[int(one["training_seed"])]["stage1_epoch"]:
                epoch_errors += 1
            if int(one["stage2_epoch"]) != epochs[int(one["training_seed"])]["stage2_epoch"]:
                epoch_errors += 1
    rows = index_rows(index_path, split)
    by_position = {row["dataset_index_position"]: row for row in rows}
    for one in records:
        position = int(one["dataset_index_position"])
        row = by_position.get(position)
        if row is None:
            identity_errors += 1
            continue
        for column in (
            "file_path",
            "filename",
            "target_code",
            "target_name",
            "distance_code",
            "segment_code",
            "block_index",
            "window_index",
            "start",
            "length",
            "source_samples",
        ):
            reference = row[column]
            actual: Any = one[column]
            if column in {"block_index", "window_index", "start", "length", "source_samples"}:
                actual = int(actual)
                reference = int(reference)
            if actual != reference:
                identity_errors += 1
    coverage_errors: list[str] = []
    for condition in expected_conditions:
        for seed in expected_seeds:
            positions = seen.get((condition, seed), set())
            expected = set(range(expected_windows))
            if positions != expected:
                coverage_errors.append(
                    f"{dataset}/{condition}/seed{seed}: {len(positions)} positions, "
                    f"missing {len(expected - positions)}, extra {len(positions - expected)}"
                )
    record.update(
        {
            "rows": len(records),
            "expected_rows": EXPECTED_ROWS[dataset],
            "duplicate_primary_keys": duplicated,
            "identity_errors": identity_errors,
            "label_errors": label_errors,
            "epoch_errors": epoch_errors,
            "coverage_errors": coverage_errors,
        }
    )
    if strict:
        if len(records) != EXPECTED_ROWS[dataset]:
            failures.append(f"{dataset}: {len(records)} rows, expected {EXPECTED_ROWS[dataset]}")
        if duplicated:
            failures.append(f"{dataset}: {duplicated} duplicated primary keys")
        if identity_errors:
            failures.append(f"{dataset}: {identity_errors} identity mismatches against the index")
        if label_errors:
            failures.append(f"{dataset}: {label_errors} class/label mapping violations")
        if epoch_errors:
            failures.append(f"{dataset}: {epoch_errors} rows carry an epoch outside the config")
        failures.extend(coverage_errors)
    return by_position


def check_prediction_rules(
    records: Sequence[dict[str, Any]],
    failures: list[str],
    record: dict[str, Any],
    strict: bool = True,
) -> None:
    """stage2_executed iff stage-1 margin > 0, mapping, finiteness and digests."""
    margin_errors = 0
    executed_errors = 0
    mapping_errors = 0
    forged_logits = 0
    missing_logits = 0
    non_finite = 0
    digest_errors = 0
    batch_errors = 0
    digest_lookup = {
        (str(one["dataset"]), str(one["condition"]), int(one["training_seed"]), int(one["dataset_index_position"])): one
        for one in records
    }
    for one in records:
        expected_margin = float(np.float32(np.float32(one["stage1_logit_drone"]) - np.float32(one["stage1_logit_background"])))
        if abs(float(one["stage1_margin"]) - expected_margin) > 1e-6:
            margin_errors += 1
        expected_stage1 = 1 if float(one["stage1_margin"]) > 0.0 else 0
        if int(one["stage1_prediction"]) != expected_stage1:
            mapping_errors += 1
        executed = bool(one["stage2_executed"])
        if executed != (expected_stage1 == 1):
            executed_errors += 1
        if str(one["stage1_decision"]) != ("Drone" if executed else "Background"):
            mapping_errors += 1
        if executed:
            if one["stage2_logit_phantom"] is None or one["stage2_logit_air2s"] is None:
                missing_logits += 1
                continue
            if one["stage2_margin"] is None:
                missing_logits += 1
                continue
            if not all(
                np.isfinite(
                    [
                        float(one["stage2_logit_phantom"]),
                        float(one["stage2_logit_air2s"]),
                        float(one["stage2_margin"]),
                    ]
                )
            ):
                non_finite += 1
            expected_stage2_margin = float(
                np.float32(np.float32(one["stage2_logit_air2s"]) - np.float32(one["stage2_logit_phantom"]))
            )
            if abs(float(one["stage2_margin"]) - expected_stage2_margin) > 1e-6:
                margin_errors += 1
            expected_final = 2 if float(one["stage2_margin"]) > 0.0 else 1
            if int(one["final_prediction"]) != expected_final:
                mapping_errors += 1
            if not one["stage2_feature_digest"]:
                digest_errors += 1
            elif str(one["stage2_feature_digest"]) != str(one["stage1_feature_digest"]):
                digest_errors += 1
            if one["stage2_batch_id"] is None or int(one["stage2_batch_id"]) < 0:
                batch_errors += 1
        else:
            if any(
                one[key] is not None
                for key in ("stage2_logit_phantom", "stage2_logit_air2s", "stage2_margin")
            ):
                forged_logits += 1
            if one["stage2_feature_digest"] is not None:
                forged_logits += 1
            if int(one["final_prediction"]) != BACKGROUND_CLASS:
                mapping_errors += 1
        if not np.isfinite(
            [
                float(one["stage1_logit_background"]),
                float(one["stage1_logit_drone"]),
                float(one["stage1_margin"]),
            ]
        ).all():
            non_finite += 1
    # the batch id must equal the ascending selection chunk index of its own group
    selection: dict[tuple[str, str, int], list[int]] = {}
    for one in records:
        key = (str(one["dataset"]), str(one["condition"]), int(one["training_seed"]))
        if bool(one["stage2_executed"]):
            selection.setdefault(key, []).append(int(one["dataset_index_position"]))
    expected_batch: dict[tuple[str, str, int, int], int] = {}
    for key, positions in selection.items():
        for index, start in enumerate(range(0, len(positions), BATCH_SIZE)):
            for position in positions[start : start + BATCH_SIZE]:
                expected_batch[(key[0], key[1], key[2], position)] = index
    for one in records:
        if not bool(one["stage2_executed"]):
            continue
        key = (
            str(one["dataset"]),
            str(one["condition"]),
            int(one["training_seed"]),
            int(one["dataset_index_position"]),
        )
        if expected_batch.get(key) != one["stage2_batch_id"]:
            batch_errors += 1
    record.update(
        {
            "rows": len(records),
            "stage1_margin_errors": margin_errors,
            "stage2_executed_errors": executed_errors,
            "label_mapping_errors": mapping_errors,
            "forged_stage2_fields": forged_logits,
            "missing_stage2_logits": missing_logits,
            "non_finite_logits": non_finite,
            "digest_mismatch": digest_errors,
            "batch_id_errors": batch_errors,
            "executed_rows": sum(1 for one in records if bool(one["stage2_executed"])),
            "unique_digest_keys": len(digest_lookup),
        }
    )
    if not strict:
        return
    for label, count in (
        ("stage-1 margin recomputation", margin_errors),
        ("stage2_executed iff margin>0", executed_errors),
        ("final label mapping", mapping_errors),
        ("forged stage-2 fields on skipped rows", forged_logits),
        ("missing stage-2 logits on executed rows", missing_logits),
        ("non-finite logits", non_finite),
        ("stage-1/stage-2 digest equality", digest_errors),
        ("stage2 batch id rule", batch_errors),
    ):
        if count:
            failures.append(f"prediction rules: {count} violation(s) of {label}")


def check_metric_tables(
    root: Path,
    records: Sequence[dict[str, Any]],
    failures: list[str],
    record: dict[str, Any],
    strict: bool,
) -> None:
    grouped = group_records(records)
    recomputed = {key: recompute_group(value) for key, value in grouped.items()}

    confusion_path = root / "confusion_matrices.csv"
    if confusion_path.is_file():
        mismatches = 0
        checked = 0
        rows = read_csv_rows(confusion_path)
        for key, values in recomputed.items():
            matrix = confusion(grouped[key])
            for truth in CLASS_IDS:
                for prediction in CLASS_IDS:
                    match = [
                        row
                        for row in rows
                        if row["dataset"] == key[0]
                        and row["condition"] == key[1]
                        and int(row["training_seed"]) == key[2]
                        and int(row["true_class"]) == truth
                        and int(row["predicted_class"]) == prediction
                    ]
                    if len(match) != 1:
                        mismatches += 1
                        continue
                    checked += 1
                    if int(match[0]["count"]) != int(matrix[truth, prediction]):
                        mismatches += 1
        record["confusion_matrix"] = {
            "rows": len(rows),
            "checked_cells": checked,
            "mismatches": mismatches,
        }
        if strict and mismatches:
            failures.append(f"confusion_matrices.csv: {mismatches} mismatching cells")

    metrics_path = root / "metrics_by_condition.csv"
    if metrics_path.is_file():
        rows = read_csv_rows(metrics_path)
        lookup = {
            (row["dataset"], row["condition"], int(row["training_seed"])): row for row in rows
        }
        field_mismatches: list[str] = []
        count_mismatches = 0
        for key, values in recomputed.items():
            row = lookup.get(key)
            if row is None:
                field_mismatches.append(f"missing metric row {key}")
                continue
            for field in METRIC_NUMERIC_FIELDS:
                if field not in row:
                    continue
                if not close(text_to_float(row.get(field)), values.get(field)):
                    field_mismatches.append(f"{key} {field}: {row.get(field)} != {values.get(field)}")
            for field in (
                "windows",
                "background_windows",
                "phantom_windows",
                "air2s_windows",
                "files_background",
                "files_phantom",
                "files_air2s",
                "stage1_background_false_alarms",
                "stage1_drone_misses",
                "stage1_drones_passed",
                "stage2_conditional_windows",
                "stage2_conditional_correct",
                "error_windows_total",
            ):
                if field in row and int(row[field]) != int(values[field]):
                    count_mismatches += 1
        record["metrics_by_condition"] = {
            "rows": len(rows),
            "groups": len(recomputed),
            "mismatches": len(field_mismatches) + count_mismatches,
            "examples": field_mismatches[:5],
        }
        if strict and (field_mismatches or count_mismatches):
            failures.append(
                f"metrics_by_condition.csv: {len(field_mismatches)} field and "
                f"{count_mismatches} count mismatches"
            )

    per_file_path = root / "per_file_metrics.csv"
    if per_file_path.is_file():
        rows = read_csv_rows(per_file_path)
        lookup = {
            (row["dataset"], row["condition"], int(row["training_seed"]), row["filename"]): row
            for row in rows
        }
        mismatches = 0
        for key, values in recomputed.items():
            for filename, entry in values["per_file"].items():
                row = lookup.get((key[0], key[1], key[2], filename))
                if row is None:
                    mismatches += 1
                    continue
                if (
                    int(row["windows"]) != entry["windows"]
                    or int(row["correct"]) != entry["correct"]
                    or int(row["stage1_pass"]) != entry["stage1_pass"]
                    or int(row["predicted_background"]) != entry["predicted_background"]
                    or int(row["predicted_phantom"]) != entry["predicted_phantom"]
                    or int(row["predicted_air2s"]) != entry["predicted_air2s"]
                    or not close(text_to_float(row["accuracy"]), entry["accuracy"])
                    or int(row["true_class"]) != entry["true_class"]
                ):
                    mismatches += 1
        record["per_file_metrics"] = {
            "rows": len(rows),
            "expected_rows": sum(len(values["per_file"]) for values in recomputed.values()),
            "mismatches": mismatches,
        }
        if strict and mismatches:
            failures.append(f"per_file_metrics.csv: {mismatches} mismatching file rows")

    error_path = root / "error_decomposition.csv"
    if error_path.is_file():
        rows = read_csv_rows(error_path)
        lookup = {
            (row["dataset"], row["condition"], int(row["training_seed"]), row["transition"]): row
            for row in rows
        }
        mismatches = 0
        for key, values in recomputed.items():
            for truth, prediction, name in ERROR_TRANSITIONS:
                row = lookup.get((key[0], key[1], key[2], name))
                if row is None:
                    mismatches += 1
                    continue
                expected_count = values[f"error_{name.replace('->', '_to_').lower()}"]
                expected_rate = values[f"rate_{name.replace('->', '_to_').lower()}"]
                if int(row["count"]) != expected_count or not close(
                    text_to_float(row["rate_of_true_class"]), expected_rate
                ):
                    mismatches += 1
        record["error_decomposition"] = {"rows": len(rows), "mismatches": mismatches}
        if strict and mismatches:
            failures.append(f"error_decomposition.csv: {mismatches} mismatching rows")

    summary_path = root / "cross_seed_summary.csv"
    if summary_path.is_file():
        rows = read_csv_rows(summary_path)
        mismatches = 0
        checked = 0
        for row in rows:
            dataset = row["dataset"]
            condition = row["condition"]
            metric = row["metric"]
            if condition == "four_condition_mean":
                values = []
                for seed in TRAINING_SEEDS:
                    seed_values = [
                        recomputed[key][metric]
                        for key in recomputed
                        if key[0] == dataset and key[2] == seed
                    ]
                    seed_values = [float(value) for value in seed_values if np.isfinite(float(value))]
                    if seed_values:
                        values.append(float(np.mean(seed_values)))
            else:
                values = [
                    float(recomputed[(dataset, condition, seed)][metric])
                    for seed in TRAINING_SEEDS
                    if (dataset, condition, seed) in recomputed
                ]
            checked += 1
            if int(row["seeds"]) != len(values):
                mismatches += 1
                continue
            own_values = [float(item) for item in str(row["seed_values"]).split(",") if item]
            if len(own_values) != len(values) or any(
                not close(left, right) for left, right in zip(own_values, values)
            ):
                mismatches += 1
            if not close(text_to_float(row["mean"]), float(np.mean(values)) if values else float("nan")):
                mismatches += 1
            if not close(text_to_float(row["sample_std"]), sample_std(values)):
                mismatches += 1
        record["cross_seed_summary"] = {"rows": len(rows), "checked": checked, "mismatches": mismatches}
        if strict and mismatches:
            failures.append(f"cross_seed_summary.csv: {mismatches} mismatching rows")


def check_digests(
    root: Path, failures: list[str], record: dict[str, Any], strict: bool
) -> None:
    manifest_path = root / "prediction_cache_manifest.json"
    if not manifest_path.is_file():
        failures.append("missing prediction_cache_manifest.json")
        return
    manifest = read_json(manifest_path)
    entries: list[dict[str, Any]] = []
    for cache in manifest.get("cache", []):
        dataset = str(cache["dataset"])
        condition = str(cache["condition"])
        path = root / str(cache["window_digest_file"])
        entry: dict[str, Any] = {
            "dataset": dataset,
            "condition": condition,
            "windows": int(cache["windows"]),
            "file": str(path),
        }
        if not path.is_file():
            failures.append(f"missing digest file {path}")
            entry["exists"] = False
            entries.append(entry)
            continue
        payload = path.read_bytes()
        entry["exists"] = True
        entry["bytes"] = len(payload)
        entry["bytes_ok"] = len(payload) == int(cache["windows"]) * 32
        entry["file_sha256"] = sha256_file(path)
        entry["file_sha256_ok"] = entry["file_sha256"] == cache["window_digest_file_sha256"]
        entry["concat_sha256_ok"] = sha256_bytes(payload) == cache["window_digests_concat_sha256"]
        split_token = "test" if dataset == "old_test" else dataset
        condition_token = "clean" if condition == "clean" else f"{condition}dB"
        reference_path = (
            STAGE1_PUBLISHED_ROOT
            / "feature_digests"
            / f"window_digests_{split_token}_{condition_token}_nfft{N_FFT}.bin"
        )
        if not reference_path.is_file() and strict:
            failures.append(f"missing published stage-1 digest binary {reference_path}")
        if reference_path.is_file():
            reference = reference_path.read_bytes()
            entry["stage1_reference"] = str(reference_path)
            entry["stage1_reference_bytes"] = len(reference)
            entry["stage1_reference_identical"] = reference[: len(payload)] == payload
            if strict and not entry["stage1_reference_identical"]:
                failures.append(
                    f"{dataset}/{condition}: per-window digests differ from the published "
                    "stage-1 nfft512 digests"
                )
        else:
            entry["stage1_reference"] = None
        if strict:
            if not entry["bytes_ok"]:
                failures.append(f"{dataset}/{condition}: digest file has {entry['bytes']} bytes")
            if not entry["file_sha256_ok"]:
                failures.append(f"{dataset}/{condition}: digest file hash mismatch")
            if not entry["concat_sha256_ok"]:
                failures.append(f"{dataset}/{condition}: digest concatenation hash mismatch")
        entries.append(entry)
    record["cache_entries"] = entries
    record["truncated_debug_run"] = bool(manifest.get("truncated_debug_run"))
    record["prediction_rows_in_manifest"] = manifest.get("prediction_rows")
    parity = manifest.get("parity_summary", {})
    record["parity_summary"] = parity
    if strict and record["truncated_debug_run"]:
        failures.append("the published cache manifest says truncated_debug_run=true")


def check_stage1_parity(
    root: Path, records: Sequence[dict[str, Any]], failures: list[str], record: dict[str, Any], strict: bool
) -> None:
    parity_path = root / "stage1_reference_parity.json"
    if not parity_path.is_file():
        failures.append("missing stage1_reference_parity.json")
        return
    parity = read_json(parity_path)
    record["stored"] = {
        "compared_rows": parity["logits"].get("compared_rows"),
        "max_abs_logit_background_diff": parity["logits"].get("max_abs_logit_background_diff"),
        "max_abs_logit_drone_diff": parity["logits"].get("max_abs_logit_drone_diff"),
        "max_abs_logit_margin_diff": parity["logits"].get("max_abs_logit_margin_diff"),
        "prediction_mismatches": parity["logits"].get("prediction_mismatches"),
        "exact_match": parity["logits"].get("exact_match"),
        "digests_all_identical": parity.get("digests_all_identical"),
        "stage2_compared_with_oracle_csv": parity.get("stage2_compared_with_oracle_csv"),
    }
    reference: dict[tuple[str, str, int, int], dict[str, str]] = {}
    mapping = {
        "heldout": STAGE1_PUBLISHED_ROOT / "heldout_window_predictions.csv",
        "old_test": STAGE1_PUBLISHED_ROOT / "test_clean_window_predictions.csv",
    }
    for dataset, path in mapping.items():
        if not path.is_file():
            if strict:
                failures.append(f"missing published stage-1 predictions {path}")
            continue
        for row in read_csv_rows(path):
            if int(row["n_fft"]) != N_FFT:
                continue
            condition = row["condition"]
            if condition.endswith("dB"):
                condition = condition[:-2]
            reference[(dataset, condition, int(row["training_seed"]), int(row["index_position"]))] = row
    compared = 0
    missing = 0
    max_diffs = {"background": 0.0, "drone": 0.0, "margin": 0.0}
    prediction_mismatches = 0
    for one in records:
        key = (
            str(one["dataset"]),
            str(one["condition"]),
            int(one["training_seed"]),
            int(one["dataset_index_position"]),
        )
        row = reference.get(key)
        if row is None:
            missing += 1
            continue
        compared += 1
        for column, own, name in (
            ("logit_background", "stage1_logit_background", "background"),
            ("logit_drone", "stage1_logit_drone", "drone"),
            ("logit_margin", "stage1_margin", "margin"),
        ):
            max_diffs[name] = max(max_diffs[name], abs(float(row[column]) - float(one[own])))
        if int(row["prediction"]) != int(one["stage1_prediction"]):
            prediction_mismatches += 1
    record["recomputed"] = {
        "compared_rows": compared,
        "missing_reference_rows": missing,
        "max_abs_diffs": max_diffs,
        "prediction_mismatches": prediction_mismatches,
        "exact_match": (
            compared > 0
            and missing == 0
            and prediction_mismatches == 0
            and all(value == 0.0 for value in max_diffs.values())
        ),
    }
    if strict:
        if missing:
            failures.append(f"stage-1 parity: {missing} prediction rows have no reference row")
        if prediction_mismatches:
            failures.append(
                f"stage-1 parity: {prediction_mismatches} stage-1 decisions differ from the "
                "published stage-1 nfft512 predictions"
            )
        if any(value > 1e-6 for value in max_diffs.values()):
            failures.append(f"stage-1 parity: logit differences exceed 1e-6 ({max_diffs})")
        if parity.get("stage2_compared_with_oracle_csv") is not False:
            failures.append("stage 2 must not claim parity with the published oracle CSV")


def check_cache_manifest(
    root: Path, failures: list[str], record: dict[str, Any], strict: bool
) -> None:
    manifest = read_json(root / "prediction_cache_manifest.json")
    record["prediction_rows"] = manifest.get("prediction_rows")
    record["datasets"] = manifest.get("datasets")
    record["seeds"] = manifest.get("seeds")
    record["conditions"] = manifest.get("conditions")
    record["n_fft"] = manifest.get("n_fft")
    record["hop_length"] = manifest.get("hop_length")
    record["rng_rule_contains_noise_seed"] = "noise_seed" in str(manifest.get("rng_rule", ""))
    if strict:
        if manifest.get("n_fft") != N_FFT or manifest.get("hop_length") != HOP_LENGTH:
            failures.append("cache manifest does not describe n_fft=512 / hop_length=256")
        if list(manifest.get("seeds") or []) != list(TRAINING_SEEDS):
            failures.append("cache manifest does not use the three frozen training seeds")
        if manifest.get("prediction_rows") != sum(
            EXPECTED_ROWS[name] for name in manifest.get("datasets") or []
        ):
            failures.append("cache manifest prediction_rows disagrees with the frozen row counts")


class FeatureChunkCache:
    """Rebuilds the shared float32 feature tensor exactly as the run module does.

    The window reader and the frontend are the frozen ones; the batch composition is
    the documented one (contiguous chunk of four rows over the full split), because
    the STFT output of a row is not bit-identical across batch compositions.  The CPU
    input tensors of every built chunk are kept so the raw+noise input path can be
    digested independently of any GPU work.
    """

    def __init__(self, index_path: Path, split: str, condition: str, device: torch.device):
        from dronerfa_data import DroneRFaWindowDataset

        self.condition = condition
        self.snr = None if condition == "clean" else float(condition)
        self.dataset = DroneRFaWindowDataset(
            index_path,
            split=split,
            fixed_snr=self.snr,
            apply_fixed_snr=self.snr is not None,
            seed=NOISE_SEED,
        )
        if self.dataset.epoch != NOISE_EPOCH:
            raise RuntimeError("unexpected epoch in the fresh reader")
        self.windows = len(self.dataset)
        self.device = device
        self.frontend = SpectrogramFrontEnd(
            n_fft=N_FFT, hop_length=HOP_LENGTH, image_size=IMAGE_SIZE
        ).to(device)
        self.frontend.eval()
        self.chunks: dict[int, np.ndarray] = {}
        self.input_digests: dict[int, list[tuple[str, str]]] = {}

    def chunk(self, start: int) -> np.ndarray:
        cached = self.chunks.get(start)
        if cached is not None:
            return cached
        stop = min(start + BATCH_SIZE, self.windows)
        signals = [self.dataset[position][0] for position in range(start, stop)]
        # digest the raw+noise inputs here: they are pure CPU/numpy and independent
        # of any GPU work, so they stay comparable across environments.  Both the
        # recorded probe payload (leading samples, the rule of the formal run) and
        # the whole tensor are kept.
        self.input_digests[start] = [
            (
                sha256_bytes(probe_payload(np.ascontiguousarray(signal.numpy()))),
                sha256_bytes(np.ascontiguousarray(signal.numpy()).tobytes()),
            )
            for signal in signals
        ]
        batch = torch.stack(signals).to(self.device)
        with torch.inference_mode():
            features = extract_features(self.frontend, batch)
        array = features.detach().to("cpu", torch.float32).numpy()
        if array.shape != (stop - start, 1, IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"unexpected feature shape {array.shape}")
        self.chunks[start] = array
        return array

    def row(self, position: int) -> np.ndarray:
        start = position - position % BATCH_SIZE
        return self.chunk(start)[position - start]

    def input_digest(self, position: int) -> tuple[str, str]:
        """(probe payload digest, whole tensor digest) of one raw+noise input window."""
        start = position - position % BATCH_SIZE
        self.chunk(start)
        return self.input_digests[start][position - start]

    def close(self) -> None:
        self.dataset.close()


def iter_group_inputs(
    index_path: Path, split: str, condition: str, noise_seed: int
):
    """Yield the raw+noise ``[2, samples]`` tensors of a whole split, in split order."""
    from dronerfa_data import DroneRFaWindowDataset

    snr = None if condition == "clean" else float(condition)
    dataset = DroneRFaWindowDataset(
        index_path,
        split=split,
        fixed_snr=snr,
        apply_fixed_snr=snr is not None,
        seed=noise_seed,
    )
    try:
        if dataset.epoch != NOISE_EPOCH:
            raise RuntimeError("unexpected epoch in the fresh reader")
        for position in range(len(dataset)):
            tensor, _ = dataset[position]
            yield np.ascontiguousarray(tensor.numpy())
    finally:
        dataset.close()

    def close(self) -> None:
        self.dataset.close()


def load_networks(device: torch.device) -> dict[int, dict[str, Any]]:
    config = read_json(CONFIG_PATH)
    networks: dict[int, dict[str, Any]] = {}
    for pairing in config["seed_pairing"]:
        seed = int(pairing["seed"])
        stage1 = torch.load(
            PROJECT_ROOT / str(pairing["stage1_checkpoint"]), map_location="cpu", weights_only=False
        )
        stage2 = torch.load(
            PROJECT_ROOT / str(pairing["stage2_checkpoint"]), map_location="cpu", weights_only=False
        )
        if int(stage1["epoch"]) != int(pairing["stage1_epoch"]):
            raise RuntimeError(f"stage-1 epoch drift for seed {seed}")
        if int(stage2["epoch"]) != int(pairing["stage2_epoch"]):
            raise RuntimeError(f"stage-2 epoch drift for seed {seed}")
        first = ResNet18(input_channels=1, num_classes=2).to(device)
        first.load_state_dict(stage1["model_state"])
        first.eval()
        second = ResNet18(input_channels=1, num_classes=2).to(device)
        second.load_state_dict(stage2["model_state"])
        second.eval()
        for network in (first, second):
            for parameter in network.parameters():
                parameter.requires_grad_(False)
        networks[seed] = {
            "stage1": first,
            "stage2": second,
            "frontend_state": stage1["frontend_state"],
        }
        del stage1, stage2
    return networks


def forward(network: ResNet18, features: np.ndarray, device: torch.device) -> np.ndarray:
    batch = torch.from_numpy(np.ascontiguousarray(features)).to(device)
    with torch.inference_mode():
        with torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16
        ):
            logits = network(batch)
    return logits.float().cpu().numpy()


def choose_targets(records: Sequence[dict[str, Any]], per_class: int) -> list[int]:
    by_class: dict[int, list[int]] = {label: [] for label in CLASS_IDS}
    for one in records:
        by_class[int(one["true_class"])].append(int(one["dataset_index_position"]))
    targets: list[int] = []
    for label in CLASS_IDS:
        values = sorted(by_class[label])
        if not values:
            continue
        picks = [values[0], values[len(values) // 2], values[-1]][: max(per_class, 1)]
        targets.extend(picks)
    executed_background = [
        int(one["dataset_index_position"])
        for one in records
        if int(one["true_class"]) == BACKGROUND_CLASS and bool(one["stage2_executed"])
    ]
    skipped_drone = [
        int(one["dataset_index_position"])
        for one in records
        if int(one["true_class"]) in DRONE_CLASSES and not bool(one["stage2_executed"])
    ]
    for extra in (executed_background[:1], skipped_drone[:1]):
        targets.extend(extra)
    unique: list[int] = []
    for position in targets:
        if position not in unique:
            unique.append(position)
    return unique


def target_rule_differences(
    one: dict[str, Any],
    *,
    stage1_margin: float,
    expected_executed: bool,
    tolerance: float,
    stage2_margin: float | None = None,
    stage2_final: int | None = None,
    batch_index: int | None = None,
) -> dict[str, int]:
    """Split one recomputed target window into auditable mismatch counters.

    A numeric drift beyond the tolerance is *not* a decision change: the decisions
    (stage-1 gate, stage-2 machine) are counted separately from the margins.
    """
    counters = {key: 0 for key in TARGET_RULE_COUNTERS}
    if expected_executed != bool(one["stage2_executed"]):
        counters["stage1_gate_changes"] += 1
    if abs(float(stage1_margin) - float(one["stage1_margin"])) > tolerance:
        counters["stage1_margin_out_of_tolerance"] += 1
    if expected_executed and bool(one["stage2_executed"]):
        if one["stage2_margin"] is None:
            # the CSV carries no stage-2 decision for a row its own rules executed
            counters["stage2_class_changes"] += 1
        else:
            if stage2_final is not None and int(stage2_final) != int(one["final_prediction"]):
                counters["stage2_class_changes"] += 1
            if stage2_margin is not None and abs(
                float(stage2_margin) - float(one["stage2_margin"])
            ) > tolerance:
                counters["stage2_margin_out_of_tolerance"] += 1
            if batch_index is not None and one["stage2_batch_id"] != int(batch_index):
                counters["stage2_batch_id_errors"] += 1
    counters["rule_mismatches"] = sum(counters[key] for key in TARGET_RULE_COUNTERS)
    return counters


def target_failures(record: dict[str, Any], tier: str, tolerance: float) -> list[str]:
    """The failure messages of the target recomputation for one numeric tier.

    In the cross-environment tier the per-window byte digests cannot match by
    construction, so they are recorded but never turned into a failure; the exact
    input read-back, the bounded numeric drift and the discrete decisions are.
    """
    failures: list[str] = []
    if int(record["input_readback_mismatches"]):
        failures.append(
            f"target recomputation: {record['input_readback_mismatches']} raw+noise input "
            "tensor(s) do not match the independent reader"
        )
    if tier == TIER_FORMAL and int(record["digest_mismatches"]):
        failures.append(
            f"target recomputation: {record['digest_mismatches']} window digest(s) "
            "do not match the CSV"
        )
    if float(record["max_abs_stage1_logit_diff"]) > tolerance:
        failures.append(
            f"target recomputation: stage-1 logits differ by "
            f"{record['max_abs_stage1_logit_diff']} (> {tolerance})"
        )
    if float(record["max_abs_stage2_logit_diff"]) > tolerance:
        failures.append(
            f"target recomputation: stage-2 logits differ by "
            f"{record['max_abs_stage2_logit_diff']} (> {tolerance})"
        )
    for key in TARGET_RULE_COUNTERS:
        if int(record[key]):
            failures.append(
                f"target recomputation: {record[key]} violation(s) of {key}"
            )
    return failures


def check_targets(
    root: Path,
    records: Sequence[dict[str, Any]],
    failures: list[str],
    record: dict[str, Any],
    per_class: int,
    strict: bool,
    device_request: str = "auto",
    tier: str = TIER_FORMAL,
) -> None:
    """Re-derive inputs, stage-1/stage-2 logits and digests for a bounded target set."""
    if per_class <= 0:
        record["skipped"] = True
        return
    device = resolve_device(device_request)
    config = read_json(CONFIG_PATH)
    index_by_dataset = {
        name: PROJECT_ROOT / str(config["datasets"][key])
        for name, key in (
            ("heldout", "heldout_index"),
            ("old_test", "old_test_index"),
        )
    }
    split_by_dataset = {"heldout": "heldout", "old_test": "test"}
    tolerance = LOGIT_TOLERANCE if tier == TIER_FORMAL else CROSS_ENV_LOGIT_BOUND
    grouped = group_records(records)
    networks = load_networks(device)
    seeds = sorted({key[2] for key in grouped})
    index_by_position: dict[str, dict[int, dict[str, Any]]] = {}
    for dataset in DATASET_NAMES:
        index_by_position[dataset] = {
            int(row["dataset_index_position"]): row
            for row in index_rows(index_by_dataset[dataset], split_by_dataset[dataset])
        }
    targets_record: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    counters = {key: 0 for key in TARGET_RULE_COUNTERS}
    max_stage1_diff = 0.0
    max_stage2_diff = 0.0
    digest_errors = 0
    stage2_digest_errors = 0
    input_checks = 0
    input_mismatches = 0
    checked = 0
    try:
        for dataset in DATASET_NAMES:
            conditions = sorted(
                {key[1] for key in grouped if key[0] == dataset},
                key=lambda item: CONDITIONS.index(item) if item in CONDITIONS else 99,
            )
            for condition in conditions:
                base_seed = seeds[0]
                base_group = grouped.get((dataset, condition, base_seed))
                if base_group is None:
                    continue
                targets = choose_targets(base_group, per_class)
                cache = FeatureChunkCache(
                    index_by_dataset[dataset], split_by_dataset[dataset], condition, device
                )
                selection_by_seed: dict[int, list[int]] = {}
                for seed in seeds:
                    group = grouped.get((dataset, condition, seed))
                    if group is None:
                        continue
                    selection_by_seed[seed] = [
                        int(one["dataset_index_position"])
                        for one in sorted(group, key=lambda item: int(item["dataset_index_position"]))
                        if bool(one["stage2_executed"])
                    ]
                for seed in seeds:
                    group = grouped.get((dataset, condition, seed))
                    if group is None:
                        continue
                    by_position = {int(one["dataset_index_position"]): one for one in group}
                    stage1_logits: dict[int, np.ndarray] = {}
                    for position in targets:
                        one = by_position.get(position)
                        if one is None:
                            continue
                        checked += 1
                        chunk_start = position - position % BATCH_SIZE
                        chunk = cache.chunk(chunk_start)
                        own_digest = sha256_bytes(
                            np.ascontiguousarray(chunk[position - chunk_start]).tobytes()
                        )
                        digest_match = own_digest == str(one["stage1_feature_digest"])
                        if not digest_match:
                            digest_errors += 1
                        # --- input exactness, independent of any GPU work ---------
                        index_entry = index_by_position[dataset].get(position)
                        if index_entry is None:
                            input_mismatches += 1
                            input_digest = None
                            input_full_digest = None
                            reference_digest = None
                            reference_full_digest = None
                            input_match = None
                        else:
                            input_digest, input_full_digest = cache.input_digest(position)
                            own_input = build_input_tensor(
                                index_entry, condition, NOISE_SEED, cache.windows
                            )
                            input_checks += 1
                            reference_digest = sha256_bytes(probe_payload(own_input))
                            reference_full_digest = sha256_bytes(
                                np.ascontiguousarray(own_input).tobytes()
                            )
                            input_match = (
                                input_digest == reference_digest
                                and input_full_digest == reference_full_digest
                            )
                            if not input_match:
                                input_mismatches += 1
                        if chunk_start not in stage1_logits:
                            stage1_logits[chunk_start] = forward(
                                networks[seed]["stage1"], chunk, device
                            )
                        logits = stage1_logits[chunk_start][position - chunk_start]
                        diff = max(
                            abs(float(logits[0]) - float(one["stage1_logit_background"])),
                            abs(float(logits[1]) - float(one["stage1_logit_drone"])),
                        )
                        max_stage1_diff = max(max_stage1_diff, diff)
                        margin = float(np.float32(logits[1] - logits[0]))
                        expected_executed = margin > 0.0
                        entry: dict[str, Any] = {
                            "dataset": dataset,
                            "condition": condition,
                            "training_seed": seed,
                            "position": position,
                            "true_class": int(one["true_class"]),
                            "input_probe_digest": input_digest,
                            "input_full_digest": input_full_digest,
                            "independent_input_probe_digest": reference_digest,
                            "independent_input_full_digest": reference_full_digest,
                            "input_match": input_match,
                            "digest_match": digest_match,
                            "recomputed_digest": own_digest,
                            "stage1_max_abs_logit_diff": diff,
                            "stage1_margin_diff": abs(margin - float(one["stage1_margin"])),
                            "gate_flip": expected_executed != bool(one["stage2_executed"]),
                        }
                        if not expected_executed or not bool(one["stage2_executed"]):
                            # either stage 1 itself rejects the row, or the CSV skipped a
                            # row that stage 1 passes here: in both cases there is no
                            # published stage-2 batch to rebuild for this window
                            counters_local = target_rule_differences(
                                one,
                                stage1_margin=margin,
                                expected_executed=expected_executed,
                                tolerance=tolerance,
                            )
                            if not bool(one["stage2_executed"]):
                                entry["stage2_skipped_in_csv"] = True
                            for key in TARGET_RULE_COUNTERS:
                                counters[key] += counters_local[key]
                            target_rows.append(entry)
                            continue
                        positions = selection_by_seed[seed]
                        batch_index = positions.index(position) // BATCH_SIZE
                        batch = positions[batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE]
                        rows = [cache.row(item) for item in batch]
                        batch_features = np.stack(rows)
                        stage2_logits = forward(networks[seed]["stage2"], batch_features, device)
                        offset = batch.index(position)
                        stage2_margin = float(
                            np.float32(stage2_logits[offset][1] - stage2_logits[offset][0])
                        )
                        expected_final = 2 if stage2_margin > 0.0 else 1
                        if one["stage2_margin"] is None:
                            # the CSV says the row was executed but carries no stage-2
                            # numbers (the static checks already reject that); it is
                            # counted as a decision mismatch instead of comparing to None
                            counters_local = target_rule_differences(
                                one,
                                stage1_margin=margin,
                                expected_executed=True,
                                tolerance=tolerance,
                            )
                        else:
                            stage2_diff = max(
                                abs(
                                    float(stage2_logits[offset][0])
                                    - float(one["stage2_logit_phantom"])
                                ),
                                abs(
                                    float(stage2_logits[offset][1])
                                    - float(one["stage2_logit_air2s"])
                                ),
                            )
                            max_stage2_diff = max(max_stage2_diff, stage2_diff)
                            own_stage2_digest = sha256_bytes(
                                np.ascontiguousarray(rows[offset]).tobytes()
                            )
                            stage2_digest_match = own_stage2_digest == str(
                                one["stage2_feature_digest"]
                            )
                            if not stage2_digest_match:
                                stage2_digest_errors += 1
                            counters_local = target_rule_differences(
                                one,
                                stage1_margin=margin,
                                expected_executed=True,
                                tolerance=tolerance,
                                stage2_margin=stage2_margin,
                                stage2_final=expected_final,
                                batch_index=batch_index,
                            )
                            entry.update(
                                {
                                    "stage2_digest_match": stage2_digest_match,
                                    "recomputed_stage2_digest": own_stage2_digest,
                                    "stage2_max_abs_logit_diff": stage2_diff,
                                    "stage2_margin_diff": abs(
                                        stage2_margin - float(one["stage2_margin"])
                                    ),
                                    "class_flip": expected_final
                                    != int(one["final_prediction"]),
                                    "batch_id_match": one["stage2_batch_id"] == batch_index,
                                }
                            )
                        for key in TARGET_RULE_COUNTERS:
                            counters[key] += counters_local[key]
                        target_rows.append(entry)
                    targets_record.append(
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "training_seed": seed,
                            "targets": len(targets),
                        }
                    )
                cache.close()
    finally:
        for network in networks.values():
            del network["stage1"], network["stage2"]
        if device.type == "cuda":
            torch.cuda.empty_cache()
    record.update(
        {
            "checked_windows": checked,
            "max_abs_stage1_logit_diff": max_stage1_diff,
            "max_abs_stage2_logit_diff": max_stage2_diff,
            "digest_mismatches": digest_errors,
            "stage2_digest_mismatches": stage2_digest_errors,
            "digest_comparison": (
                "byte_exact"
                if tier == TIER_FORMAL
                else "not_applicable_cross_environment"
            ),
            "input_checks": input_checks,
            "input_readback_mismatches": input_mismatches,
            "input_probe_payload_samples": INPUT_PROBE_SAMPLES,
            "numeric_tier": tier,
            "tolerance": tolerance,
            "groups": targets_record,
            "targets": target_rows,
        }
    )
    record.update(counters)
    record["rule_mismatches"] = sum(int(record[key]) for key in TARGET_RULE_COUNTERS)
    if strict:
        failures.extend(target_failures(record, tier, tolerance))


def check_input_probe(
    root: Path,
    failures: list[str],
    record: dict[str, Any],
    groups: Sequence[str],
    strict: bool,
) -> None:
    """Recompute the recorded whole-split input probe (raw IQ + noise, no GPU).

    The formal run digested the leading samples of every window tensor of a group in
    split order into ``input_probe_sha256``.  Recomputing exactly that rolling digest
    with the frozen reader ties the published predictions to byte-identical *inputs*
    without involving the GPU STFT at all.
    """
    manifest_path = root / "prediction_cache_manifest.json"
    if not manifest_path.is_file():
        record["skipped"] = "missing prediction_cache_manifest.json"
        return
    manifest = read_json(manifest_path)
    samples = int(manifest.get("input_probe_samples") or INPUT_PROBE_SAMPLES)
    entries: dict[tuple[str, str], dict[str, Any]] = {
        (str(cache["dataset"]), str(cache["condition"])): cache
        for cache in manifest.get("cache", [])
    }
    config = read_json(CONFIG_PATH)
    specs = {
        "heldout": (
            PROJECT_ROOT / str(config["datasets"]["heldout_index"]),
            "heldout",
            int(config["noise"]["seed"]),
        ),
        "old_test": (
            PROJECT_ROOT / str(config["datasets"]["old_test_index"]),
            "test",
            int(config["noise"]["seed"]),
        ),
    }
    results: list[dict[str, Any]] = []
    for token in groups:
        dataset, _, condition = token.partition(":")
        if dataset not in specs or not condition:
            failures.append(f"input probe: unknown group token {token!r}")
            continue
        index_path, split, noise_seed = specs[dataset]
        entry = entries.get((dataset, condition))
        if entry is None:
            failures.append(f"input probe: no manifest entry for {dataset}/{condition}")
            continue
        digest = hashlib.sha256()
        started = time.perf_counter()
        windows = 0
        for tensor in iter_group_inputs(index_path, split, condition, noise_seed):
            digest.update(probe_payload(tensor, samples))
            windows += 1
        actual = digest.hexdigest()
        item = {
            "dataset": dataset,
            "condition": condition,
            "windows": windows,
            "samples_per_window": samples,
            "recorded_probe_sha256": entry.get("input_probe_sha256"),
            "recomputed_probe_sha256": actual,
            "match": actual == str(entry.get("input_probe_sha256")),
            "seconds": time.perf_counter() - started,
        }
        results.append(item)
        if strict and not item["match"]:
            failures.append(
                f"input probe {dataset}/{condition}: recomputed {actual} does not match the "
                f"recorded {entry.get('input_probe_sha256')}"
            )
    record["groups"] = results
    record["checked"] = len(results)


def read_png_dimensions(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()[:33]
    if payload[:8] != PNG_SIGNATURE:
        raise ValueError(f"{path} is not a PNG file")
    width = int.from_bytes(payload[16:20], "big")
    height = int.from_bytes(payload[20:24], "big")
    return width, height


def check_figures(root: Path, failures: list[str], record: dict[str, Any], strict: bool) -> None:
    evidence_path = root / "figure_evidence.json"
    if not evidence_path.is_file():
        failures.append("missing figure_evidence.json")
        return
    evidence = read_json(evidence_path)
    record["figures"] = {}
    expected_confusions = [
        f"confusion_{dataset}_{condition}.png"
        for dataset in DATASET_NAMES
        for condition in EXPECTED_CONDITIONS[dataset]
    ]
    required = expected_confusions + ["recall_by_condition.png", "cascade_flow.png"]
    for name in required:
        path = root / name
        entry: dict[str, Any] = {"exists": path.is_file()}
        if path.is_file():
            entry["sha256"] = sha256_file(path)
            try:
                entry["dimensions"] = list(read_png_dimensions(path))
            except ValueError as error:
                entry["error"] = str(error)
                if strict:
                    failures.append(str(error))
            listed = (evidence.get("figures") or {}).get(name)
            entry["listed_sha256"] = listed.get("sha256") if listed else None
            entry["hash_ok"] = bool(listed) and listed.get("sha256") == entry["sha256"]
            if strict and not entry["hash_ok"]:
                failures.append(f"{name} is not listed with a matching hash in figure_evidence.json")
        elif strict:
            failures.append(f"missing figure {name}")
        record["figures"][name] = entry


def check_manifest(root: Path, failures: list[str], record: dict[str, Any], strict: bool) -> None:
    path = root / "experiment_manifest.json"
    if not path.is_file():
        if strict:
            failures.append("missing experiment_manifest.json")
        return
    manifest = read_json(path)
    mismatches: list[str] = []
    for name, entry in (manifest.get("published_files") or {}).items():
        if name.endswith("/"):
            for inner, inner_entry in entry.items():
                inner_path = root / name.rstrip("/") / inner
                if not inner_path.is_file():
                    mismatches.append(f"missing {inner_path}")
                    continue
                if sha256_file(inner_path) != inner_entry["sha256"]:
                    mismatches.append(f"{inner_path} hash mismatch")
            continue
        published = root / name
        if not published.is_file():
            mismatches.append(f"missing {name}")
            continue
        if sha256_file(published) != entry["sha256"]:
            mismatches.append(f"{name} hash mismatch")
    record["published_files"] = len(manifest.get("published_files") or {})
    record["mismatches"] = mismatches
    record["config_sha256"] = manifest.get("config", {}).get("sha256")
    if strict:
        failures.extend(mismatches)
        if manifest.get("config", {}).get("sha256") != sha256_file(CONFIG_PATH):
            failures.append("experiment_manifest.json was produced for another config hash")


def check_report(root: Path, failures: list[str], record: dict[str, Any], strict: bool) -> None:
    path = root / "report.md"
    if not path.is_file():
        if strict:
            failures.append("missing report.md")
        return
    text = path.read_text(encoding="utf-8")
    missing = [token for token in REQUIRED_REPORT_TOKENS if token not in text]
    record.update({"bytes": len(text.encode("utf-8")), "missing_tokens": missing})
    if strict and missing:
        failures.append(f"report.md is missing required statements: {missing}")


def self_check_rejections(record: dict[str, Any], failures: list[str]) -> None:
    """Prove the rule checker rejects the four key paths when the CSV is tampered."""
    def make_row(
        position: int,
        true_class: int,
        stage1: tuple[float, float],
        stage2: tuple[float, float] | None,
        batch_id: int | None = 0,
        seed: int = 2026,
        condition: str = "clean",
    ) -> dict[str, Any]:
        margin = float(np.float32(np.float32(stage1[1]) - np.float32(stage1[0])))
        executed = stage2 is not None
        final = 0
        if executed:
            stage2_margin = float(np.float32(np.float32(stage2[1]) - np.float32(stage2[0])))
            final = 2 if stage2_margin > 0 else 1
        else:
            stage2_margin = None
        return {
            "dataset": "heldout",
            "condition": condition,
            "training_seed": seed,
            "dataset_index_position": position,
            "true_class": true_class,
            "stage1_logit_background": float(stage1[0]),
            "stage1_logit_drone": float(stage1[1]),
            "stage1_margin": margin,
            "stage1_prediction": 1 if margin > 0 else 0,
            "stage1_decision": "Drone" if executed else "Background",
            "stage2_executed": executed,
            "stage2_batch_id": batch_id if executed else None,
            "stage2_logit_phantom": None if stage2 is None else float(stage2[0]),
            "stage2_logit_air2s": None if stage2 is None else float(stage2[1]),
            "stage2_margin": stage2_margin,
            "final_prediction": final,
            "final_class_name": CLASS_NAMES[final],
            "stage1_feature_digest": f"digest-{position}",
            "stage2_feature_digest": f"digest-{position}" if executed else None,
        }

    base = [
        make_row(0, 0, (3.0, -2.0), None),
        make_row(1, 0, (-1.0, 2.0), (2.0, -1.0)),
        make_row(2, 0, (-1.0, 2.0), (-2.0, 1.0)),
        make_row(3, 1, (4.0, -3.0), None),
        make_row(4, 2, (4.0, -3.0), None),
        make_row(5, 1, (-1.0, 2.0), (3.0, -1.0)),
        make_row(6, 2, (-1.0, 2.0), (-3.0, 1.0)),
    ]
    clean_failures: list[str] = []
    check_prediction_rules(base, clean_failures, {}, strict=True)
    if clean_failures:
        failures.append(f"self-check: the untampered sample was rejected: {clean_failures}")

    def tamper(mutator) -> int:
        rows = [dict(row) for row in base]
        mutator(rows)
        local: list[str] = []
        check_prediction_rules(rows, local, {}, strict=True)
        return len(local)

    cases = {
        "background classified correctly (no stage 2)": lambda rows: rows[0].update(
            {"final_prediction": 1}
        ),
        "background false alarm routed to a machine": lambda rows: rows[1].update(
            {"stage2_executed": False, "stage2_logit_phantom": 2.0, "stage2_logit_air2s": -1.0}
        ),
        "drone missed by stage 1": lambda rows: rows[3].update(
            {"stage2_executed": True, "stage2_logit_phantom": 1.0, "stage2_logit_air2s": -1.0}
        ),
        "the two machines swapped in stage 2": lambda rows: rows[5].update(
            {"final_prediction": 2}
        ),
        "forged stage-2 logits on a skipped row": lambda rows: rows[0].update(
            {"stage2_logit_phantom": 1.0, "stage2_margin": 0.5}
        ),
        "stage-2 input digest differs from stage 1": lambda rows: rows[1].update(
            {"stage2_feature_digest": "other"}
        ),
        "batch id does not follow the ascending selection": lambda rows: rows[1].update(
            {"stage2_batch_id": 7}
        ),
        "non-finite stage-1 logit": lambda rows: rows[0].update(
            {"stage1_logit_drone": float("nan")}
        ),
    }
    detections = {}
    for name, mutator in cases.items():
        detections[name] = tamper(mutator)
    record["tamper_cases"] = detections
    undetected = [name for name, count in detections.items() if count == 0]
    if undetected:
        failures.append(f"self-check: tampering not detected for {undetected}")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def verify_outputs(
    root: Path | None = None,
    fixture: bool = False,
    check_frozen: bool = True,
    targets_per_class: int = 3,
    check_targets_enabled: bool = True,
    device: str = "auto",
    input_probe_groups: Sequence[str] = (),
    reexec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(root) if root is not None else OUTPUT_ROOT
    strict = not fixture
    failures: list[str] = []
    checks: dict[str, Any] = {
        "root": str(root),
        "fixture_mode": fixture,
        "strict": strict,
        "tolerance": TOLERANCE,
        "logit_tolerance": LOGIT_TOLERANCE,
    }
    started = time.perf_counter()
    config = read_json(CONFIG_PATH)
    epochs = {
        int(pairing["seed"]): {
            "stage1_epoch": int(pairing["stage1_epoch"]),
            "stage2_epoch": int(pairing["stage2_epoch"]),
        }
        for pairing in config["seed_pairing"]
    }
    # ------------------------------------------------------------------
    # runtime environment vs the arithmetic the formal run pinned
    # ------------------------------------------------------------------
    environment_record: dict[str, Any] = {"requested_device": device}
    formal_path = root / FORMAL_ENVIRONMENT_FILE
    formal_environment = read_json(formal_path) if formal_path.is_file() else None
    environment_record["formal"] = formal_environment
    environment_record["reexec"] = reexec
    if fixture:
        # a bounded synthetic tree has no recorded formal environment; the strict
        # criteria are kept so the tamper-rejection self-checks stay as strict as before
        device_object = resolve_device("cpu")
        tier = TIER_FORMAL
        environment_record["recomputation"] = runtime_environment(device_object)
        environment_record["numeric_comparison_tier"] = tier
        environment_record["criterion"] = (
            "fixture mode: bounded synthetic tree, strict criteria on synthetic values"
        )
    else:
        try:
            device_object = resolve_device(device)
            environment = runtime_environment(device_object)
            tier, reasons = comparison_tier(environment, formal_environment)
            environment_record["recomputation"] = environment
            environment_record["environment_mismatch_reasons"] = reasons
            environment_record["numeric_comparison_tier"] = tier
            environment_record["cross_environment_logit_bound"] = CROSS_ENV_LOGIT_BOUND
            environment_record["criterion"] = (
                "byte-exact window digests, |dlogit| <= 1e-4, identical discrete decisions"
                if tier == TIER_FORMAL
                else "digest criterion not applicable; exact raw+noise inputs, "
                f"|dlogit| <= {CROSS_ENV_LOGIT_BOUND}, identical discrete decisions"
            )
        except RuntimeError as error:
            device_object = torch.device("cpu")
            tier = TIER_CROSS
            environment_record["error"] = str(error)
            environment_record["numeric_comparison_tier"] = tier
            failures.append(str(error))
    checks["environment"] = environment_record
    if check_frozen:
        check_frozen_inputs(failures, checks.setdefault("frozen_inputs", {}), root)
    else:
        checks["frozen_inputs"] = {"skipped": True}

    prediction_path = root / "window_predictions.csv"
    if not prediction_path.is_file():
        failures.append(f"missing {prediction_path}")
        return {
            "task": "CASCADE3 independent verification",
            "generated_at": utc_now(),
            "root": str(root),
            "passed": False,
            "failures": failures,
            "checks": checks,
        }
    records = parse_prediction_rows(prediction_path)
    checks["prediction_file"] = {
        "path": str(prediction_path),
        "sha256": sha256_file(prediction_path),
        "rows": len(records),
    }
    spec_by_name = config_dataset_specs(config)
    for dataset in DATASET_NAMES:
        subset = [one for one in records if str(one["dataset"]) == dataset]
        if not subset:
            if strict:
                failures.append(f"{dataset}: no prediction rows")
            continue
        validate_records(
            subset,
            dataset,
            spec_by_name[dataset]["split"],
            spec_by_name[dataset]["index_path"],
            epochs,
            failures,
            checks.setdefault("table", {}).setdefault(dataset, {}),
            strict,
        )
    check_prediction_rules(records, failures, checks.setdefault("decision_rules", {}), True)
    check_digests(root, failures, checks.setdefault("digests", {}), strict)
    check_cache_manifest(root, failures, checks.setdefault("cache_manifest", {}), strict)
    check_stage1_parity(root, records, failures, checks.setdefault("stage1_parity", {}), strict)
    check_metric_tables(root, records, failures, checks.setdefault("metrics", {}), strict)
    check_figures(root, failures, checks.setdefault("figures", {}), strict)
    check_manifest(root, failures, checks.setdefault("manifest", {}), strict)
    check_report(root, failures, checks.setdefault("report", {}), strict)
    self_check_rejections(checks.setdefault("rejection_self_check", {}), failures)
    if check_targets_enabled:
        if check_frozen:
            check_targets(
                root,
                records,
                failures,
                checks.setdefault("target_recomputation", {}),
                targets_per_class,
                strict,
                device_request=device_object.type,
                tier=tier,
            )
        else:
            checks["target_recomputation"] = {"skipped": "frozen inputs not checked"}
    else:
        checks["target_recomputation"] = {"skipped": "disabled"}
    if input_probe_groups:
        check_input_probe(
            root,
            failures,
            checks.setdefault("input_probe", {}),
            list(input_probe_groups),
            strict,
        )
    else:
        checks["input_probe"] = {
            "skipped": "no --input-probe-groups requested",
            "hint": "e.g. --input-probe-groups heldout:clean (about 4 minutes per full split)",
        }
    checks["seconds"] = time.perf_counter() - started
    return {
        "task": "CASCADE3 independent verification",
        "generated_at": utc_now(),
        "root": str(root),
        "fixture_mode": fixture,
        "numeric_comparison_tier": tier,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "checks": checks,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None, help="write the JSON here as well")
    parser.add_argument("--fixture", action="store_true", help="bounded synthetic tree, no frozen checks")
    parser.add_argument("--skip-frozen", action="store_true")
    parser.add_argument("--targets-per-class", type=int, default=3)
    parser.add_argument("--skip-targets", action="store_true")
    parser.add_argument("--no-write", action="store_true", help="do not write verification.json")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="device for the target recomputation (default: cuda when available)",
    )
    parser.add_argument(
        "--no-reexec",
        action="store_true",
        help="never re-execute under the project .venv interpreter",
    )
    parser.add_argument(
        "--input-probe-groups",
        default="",
        help="comma separated dataset:condition tokens whose recorded whole-split input "
        "probe is recomputed (e.g. heldout:clean,old_test:clean)",
    )
    return parser.parse_args(argv)


def maybe_reexec(args: argparse.Namespace, argv: Sequence[str], root: Path) -> int | None:
    """Re-run this verifier under the project interpreter when that is required.

    The published digests belong to the formal CUDA/torch build.  If the current
    interpreter cannot reproduce that arithmetic (typically a CPU-only torch), the
    strict byte comparison would report a failure that says nothing about the run.
    Re-executing the *same* command with the project ``.venv`` interpreter restores
    the formal arithmetic; the child records this in its report.  ``--no-reexec`` and
    a missing ``.venv`` leave the run in the recorded cross-environment tier.
    """
    if args.fixture:
        return None
    try:
        device = resolve_device(args.device)
        environment = runtime_environment(device)
    except RuntimeError:
        environment = {"torch": str(torch.__version__), "device": "unknown"}
    formal_path = root / FORMAL_ENVIRONMENT_FILE
    formal = read_json(formal_path) if formal_path.is_file() else None
    tier, reasons = comparison_tier(environment, formal)
    plan = plan_reexec(
        tier,
        enabled=not args.no_reexec,
        marker_present=os.environ.get(REEXEC_MARKER) == "1",
        python_executable=sys.executable,
    )
    if not plan["needed"]:
        if tier == TIER_CROSS:
            print(
                f"environment: cross-environment tier ({'; '.join(reasons)}); "
                f"not re-executing because {plan['reason']}",
                flush=True,
            )
        return None
    command = [str(plan["interpreter"]), str(Path(__file__).resolve()), *[str(item) for item in argv]]
    child_environment = dict(os.environ)
    child_environment[REEXEC_MARKER] = "1"
    child_environment[REEXEC_ORIGIN] = json.dumps(
        {
            "python": sys.executable,
            "torch": environment.get("torch"),
            "device": environment.get("device"),
            "reasons": reasons,
        }
    )
    print(
        f"environment: {plan['reason']} ({'; '.join(reasons)}); "
        f"re-executing with {plan['interpreter']}",
        flush=True,
    )
    try:
        completed = subprocess.run(command, cwd=os.getcwd(), env=child_environment, check=False)
    except OSError as error:
        print(f"environment: re-execution failed ({error}); continuing in this interpreter", flush=True)
        return None
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    root = args.output_dir or OUTPUT_ROOT
    reexec = None
    origin = os.environ.get(REEXEC_ORIGIN)
    if origin:
        try:
            reexec = {"parent": json.loads(origin), "marker": REEXEC_MARKER}
        except json.JSONDecodeError:
            reexec = {"parent": origin, "marker": REEXEC_MARKER}
    code = maybe_reexec(args, argv, root)
    if code is not None:
        return code
    result = verify_outputs(
        root,
        fixture=args.fixture,
        check_frozen=not args.skip_frozen and not args.fixture,
        targets_per_class=args.targets_per_class,
        check_targets_enabled=not args.skip_targets,
        device=args.device,
        input_probe_groups=[
            token.strip() for token in args.input_probe_groups.split(",") if token.strip()
        ],
        reexec=reexec,
    )
    if not args.no_write:
        (root).mkdir(parents=True, exist_ok=True)
        (root / "verification.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        f"passed={result['passed']} failures={len(result['failures'])} "
        f"tier={result.get('numeric_comparison_tier')} "
        f"({result['checks'].get('seconds', 0):.1f}s)",
        flush=True,
    )
    for message in result["failures"]:
        print(f"  FAIL {message}", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
