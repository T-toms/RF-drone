"""STAGE2-BASE: oracle machine-type classification (Phantom 4 Pro vs Air 2S).

Only windows whose true label is a drone enter this stage; the two classes are

    0 = DJI Phantom 4 Pro (stage-1 ``target_code`` ``0010``)
    1 = DJI Air 2S        (stage-1 ``target_code`` ``0101``)

The background class is not part of this task and no first-stage model is called:
this is not a cascade three-class system.

Stepwise commands (``--output-dir`` selects the root, default below)::

    python run_stage2_classification.py verify-inputs   [--stage before|after]
    python run_stage2_classification.py frontends
    python run_stage2_classification.py train          [--n-fft 512] [--seed 2026]
    python run_stage2_classification.py select-nfft
    python run_stage2_classification.py figures
    python run_stage2_classification.py evaluate       [--max-windows N]
    python run_stage2_classification.py summarize
    python run_stage2_classification.py manifest
    python run_stage2_classification.py all

``test_stage2_classification.py`` runs the bounded unit/smoke checks and
``verify_stage2_classification.py`` independently recomputes every reported
metric from the raw CSVs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "outputs" / ".matplotlib"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from dronerfa_data import (
    DroneRFaWindowDataset,
    add_complex_awgn,
    load_index,
    parse_snr_list,
    read_vector_slice,
    rms_normalize,
    stratified_limit,
)
from model import ResNet18, SpectrogramFrontEnd
from train_resnet18 import extract_features, seed_everything, train_epoch

# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------
OUTPUT_ROOT = PROJECT_DIR / "outputs" / "stage2_model_classification_20260919"
OLD_INDEX = PROJECT_DIR / "outputs" / "window_index.csv"
HELDOUT_INDEX = (
    PROJECT_DIR / "outputs" / "heldout_preparation_20260917" / "heldout_window_index.csv"
)
PRESERVED_INPUTS = PROJECT_DIR.parent / "work" / "codex-preserved-inputs-20260917.json"
ORIGINAL_MAT_DIR = Path(r"C:\Users\Administrator\Documents\DroneRFa数据集")

OLD_INDEX_SHA256 = "4e77c7f186c1aae8e04f2e90f2769c0ed399cc94f40b64f99237b65c057b85ac"
HELDOUT_INDEX_SHA256 = (
    "6fe32faeb5d583a359d9e8aa17caa52a7c4567d545272ded25266767cd2cffcf"
)

NFFT_GRID: tuple[int, ...] = (512, 1024)
HOP_DIVISOR = 2
IMAGE_SIZE = 256
TRAINING_SEEDS: tuple[int, ...] = (2026, 2027, 2028)
EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
TRAIN_SNRS = "clean,10,5,0,-5,-10,-12"
VALIDATION_SEED = 12026
HELDOUT_NOISE_SEED = 22026
TEST_NOISE_SEED_OFFSET = 20_000
# The old-test split is evaluated clean only, so no noise is ever drawn for it and
# this seed is recorded for completeness only (it is inert).
TEST_NOISE_SEED = TRAINING_SEEDS[0] + TEST_NOISE_SEED_OFFSET

CONDITION_VALUES: dict[str, float | None] = {
    "clean": None,
    "-10dB": -10.0,
    "-12dB": -12.0,
    "-15dB": -15.0,
}
CONDITIONS: tuple[str, ...] = ("clean", "-10dB", "-12dB", "-15dB")
TEST_CONDITIONS: tuple[str, ...] = ("clean",)

# stage-2 label space: 0 = Phantom 4 Pro, 1 = Air 2S
PHANTOM_CODE = "0010"
AIR2S_CODE = "0101"
CODE_TO_LABEL: dict[str, int] = {PHANTOM_CODE: 0, AIR2S_CODE: 1}
LABEL_TO_NAME: dict[int, str] = {0: "DJI Phantom 4 Pro", 1: "DJI Air 2S"}
STAGE1_DRONE_LABEL = 1

# old index split sizes after filtering, and the per-class window counts
EXPECTED_SPLIT_WINDOWS: dict[str, int] = {
    "train": 620,
    "val": 290,
    "test": 300,
    "heldout": 1260,
}
EXPECTED_CLASS_WINDOWS: dict[str, tuple[int, int]] = {
    "train": (310, 310),
    "val": (140, 150),
    "test": (140, 160),
    "heldout": (610, 650),
}
EXPECTED_CLASS_FILES: dict[str, tuple[int, int]] = {
    "train": (2, 2),
    "val": (1, 1),
    "test": (1, 1),
    "heldout": (4, 4),
}
HELDOUT_CONDITIONS = len(CONDITIONS)
EXPECTED_HELDOUT_ROWS = (
    len(NFFT_GRID) * len(TRAINING_SEEDS) * len(CONDITIONS) * EXPECTED_SPLIT_WINDOWS["heldout"]
)
EXPECTED_TEST_ROWS = len(NFFT_GRID) * len(TRAINING_SEEDS) * EXPECTED_SPLIT_WINDOWS["test"]

EPOCH_SELECTION_RULE = "mean_file_equal_balanced_accuracy_over_four_conditions"
TIE_BREAK = "earliest_epoch"
NFFT_SELECTION_RULE = "mean_over_training_seeds_of_the_four_condition_validation_score"
NFFT_TIE_BREAK = "smaller_n_fft_wins_an_exact_tie"
BASELINE_NFFT = 1024
SELECTION_LOCK_RULE = (
    "the configuration choice is written and hashed before any held-out or old-test "
    "prediction is produced; later evaluation re-checks the hash and never rewrites it"
)
PREDICTION_RULE = (
    "float32(logit_air2s) - float32(logit_phantom) > 0 -> Air 2S (label 1), else Phantom (label 0)"
)
FEATURE_DTYPE = np.float32
INPUT_PROBE_SAMPLES = 4096
REPRESENTATIVE_WINDOW_RULE = (
    "first row of the frozen held-out index whose target_code is '0010' (Phantom) and the "
    "first row whose target_code is '0101' (Air 2S); the rule is fixed before any training "
    "or evaluation and depends only on the frozen index file order"
)
SELECTION_FORBIDDEN_TOKENS = ("heldout", "test", "F:\\DroneRFa")

METRIC_COLUMN_ORDER = (
    "windows",
    "files_phantom",
    "files_air2s",
    "file_equal_phantom_recall",
    "file_equal_air2s_recall",
    "file_equal_balanced_accuracy",
    "accuracy",
    "balanced_accuracy",
    "precision_air2s",
    "precision_phantom",
    "phantom_recall",
    "air2s_recall",
    "f1_air2s_binary",
    "f1_macro",
    "tn",
    "fp",
    "fn",
    "tp",
)

PREDICTION_COLUMNS = (
    "split",
    "condition",
    "n_fft",
    "hop_length",
    "training_seed",
    "epoch",
    "model_id",
    "index_position",
    "file_path",
    "filename",
    "target_code",
    "target_name",
    "distance_code",
    "segment_code",
    "row_split",
    "label",
    "class_name",
    "block_index",
    "window_index",
    "start",
    "length",
    "source_samples",
    "logit_phantom",
    "logit_air2s",
    "logit_margin",
    "air2s_score",
    "prediction",
)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path, label: str | None = None) -> str:
    digest = hashlib.sha256()
    if label:
        log(f"  hash start [{label}] {path.name}")
    started = time.perf_counter()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    result = digest.hexdigest()
    if label:
        log(
            f"  hash done  [{label}] {path.name} {path.stat().st_size} bytes "
            f"sha256={result} ({time.perf_counter() - started:.1f}s)"
        )
    return result


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def hop_for(n_fft: int) -> int:
    return n_fft // HOP_DIVISOR


def nfft_label(n_fft: int) -> str:
    return f"nfft{n_fft}"


def run_directory(n_fft: int, seed: int, output_root: Path | None = None) -> Path:
    return (output_root or OUTPUT_ROOT) / nfft_label(n_fft) / f"seed_{seed}"


def state_hash(state: dict[str, torch.Tensor]) -> str:
    """Byte-exact hash of a state_dict, identical to ``run_noise_ab.state_hash``."""
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def mean_or_zero(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def spread(values: Sequence[float]) -> float:
    """Sample standard deviation, or 0 for a single value."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


# ---------------------------------------------------------------------------
# stage-2 index view
# ---------------------------------------------------------------------------
def stage2_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the two drone target codes and remap their labels to 0/1.

    The original index order is preserved, so the resulting list defines the
    stable, 0-based stage-2 positions used by every split.
    """
    selected: list[dict[str, Any]] = []
    for row in rows:
        code = str(row["target_code"])
        if code not in CODE_TO_LABEL:
            continue
        stage1_label = int(row["label"])
        if stage1_label != STAGE1_DRONE_LABEL:
            raise ValueError(
                f"index row {row.get('filename')} target {code} carries stage-1 label "
                f"{stage1_label}; expected a drone window ({STAGE1_DRONE_LABEL})"
            )
        selected.append(
            {
                **row,
                "stage1_label": stage1_label,
                "label": CODE_TO_LABEL[code],
                "class_name": LABEL_TO_NAME[CODE_TO_LABEL[code]],
            }
        )
    return selected


def stage2_index_rows(index_path: Path, split: str) -> list[dict[str, Any]]:
    """Filtered rows of one split with their stable 0-based stage-2 position."""
    rows = stage2_rows(load_index(index_path, split=split))
    return [{"index_position": position, **row} for position, row in enumerate(rows)]


class Stage2WindowDataset(DroneRFaWindowDataset):
    """The frozen lazy reader restricted to the two drone classes.

    ``DroneRFaWindowDataset`` is used unchanged through inheritance; only the row
    list is narrowed and relabelled. The frozen RNG rule therefore keeps working,
    but is now evaluated on the stage-2 index::

        default_rng(seed + epoch * len(self) + index)

    with ``len(self)`` the number of drone windows of this split and ``index`` the
    stage-2 position. Validation, held-out and test read their split with a fixed
    split-level noise seed and ``epoch`` pinned to 0, so the added noise is a
    property of (split, condition, stage-2 position) alone and is paired
    window-by-window across the two ``n_fft`` values and the three training seeds.
    """

    def __init__(
        self,
        index_path: Path,
        split: str,
        train_snrs: list[float | None] | None = None,
        fixed_snr: float | None = None,
        apply_fixed_snr: bool = False,
        max_samples: int | None = None,
        seed: int = 2026,
    ) -> None:
        super().__init__(index_path, split=split, seed=seed)
        self.rows = stage2_rows(self.rows)
        if max_samples is not None and max_samples > 0:
            self.rows = stratified_limit(self.rows, max_samples)
        self.train_snrs = train_snrs
        self.fixed_snr = fixed_snr
        self.apply_fixed_snr = apply_fixed_snr


class _IndexSubsetDataset:
    """View a ``Stage2WindowDataset`` through a list of stage-2 positions.

    ``__len__`` reports the full split length so the frozen AWGN rule keeps using
    the real stage-2 position, while ``__getitem__`` only reads the requested rows.
    A bounded (smoke) run therefore sees exactly the noise of the full run.
    """

    def __init__(self, dataset: Stage2WindowDataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def sampler(self) -> list[int]:
        return list(range(len(self.indices)))

    def __getitem__(self, position: int):
        return self.dataset[self.indices[position]]

    def close(self) -> None:
        self.dataset.close()


def split_composition(split: str, index_path: Path) -> dict[str, Any]:
    rows = stage2_index_rows(index_path, split)
    codes = [row["target_code"] for row in rows]
    counts = {int(label): sum(1 for row in rows if int(row["label"]) == label) for label in (0, 1)}
    files: dict[int, set[str]] = {0: set(), 1: set()}
    for row in rows:
        files[int(row["label"])].add(row["filename"])
    return {
        "split": split,
        "index": str(index_path),
        "windows": len(rows),
        "class_windows": counts,
        "class_files": {label: sorted(names) for label, names in files.items()},
        "target_codes": sorted(set(codes)),
        "first_position_identity": {
            key: rows[0][key] for key in ("filename", "start", "length", "target_code", "label")
        }
        if rows
        else None,
        "last_position_identity": {
            key: rows[-1][key] for key in ("filename", "start", "length", "target_code", "label")
        }
        if rows
        else None,
    }


# ---------------------------------------------------------------------------
# metrics (positive class = Air 2S = label 1)
# ---------------------------------------------------------------------------
def stage2_pooled_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pooled window counts plus the full metric set used by every table."""
    y = np.asarray([int(row["label"]) for row in records], dtype=np.int64)
    prediction = np.asarray([int(row["prediction"]) for row in records], dtype=np.int64)
    tp = int(np.sum((y == 1) & (prediction == 1)))
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    if tp + tn + fp + fn != len(y):
        raise ValueError("confusion matrix does not conserve the window count")
    phantom_recall = tn / max(tn + fp, 1)
    air2s_recall = tp / max(tp + fn, 1)
    precision_air2s = tp / max(tp + fp, 1)
    precision_phantom = tn / max(tn + fn, 1)
    f1_air2s = 2 * precision_air2s * air2s_recall / max(precision_air2s + air2s_recall, 1e-12)
    f1_phantom = 2 * precision_phantom * phantom_recall / max(precision_phantom + phantom_recall, 1e-12)
    return {
        "windows": int(len(y)),
        "accuracy": (tp + tn) / max(len(y), 1),
        "balanced_accuracy": 0.5 * (phantom_recall + air2s_recall),
        "precision_air2s": precision_air2s,
        "precision_phantom": precision_phantom,
        "phantom_recall": phantom_recall,
        "air2s_recall": air2s_recall,
        "f1_air2s_binary": f1_air2s,
        "f1_macro": 0.5 * (f1_phantom + f1_air2s),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def stage2_file_equal_metrics(
    records: Sequence[dict[str, Any]], require_both_classes: bool = True
) -> dict[str, Any]:
    """Equal weight per source file inside each class, then the two classes equally.

    This is the primary aggregation of STAGE2-BASE. ``f1`` is deliberately absent:
    a file-equal F1 is not defined by the task, so only pooled F1 is reported.

    A bounded debug run can truncate the split before the first window of the second
    class; ``require_both_classes=False`` then reports ``nan`` instead of failing, and
    the truncated flag stays visible in ``prediction_cache_manifest.json``.
    """
    per_file: dict[str, dict[str, Any]] = {}
    for row in records:
        entry = per_file.setdefault(
            str(row["filename"]),
            {"label": int(row["label"]), "correct": 0, "windows": 0},
        )
        if entry["label"] != int(row["label"]):
            raise ValueError(f"source file {row['filename']} carries two different classes")
        entry["correct"] += int(int(row["prediction"]) == int(row["label"]))
        entry["windows"] += 1
    class_rates: dict[int, list[float]] = {0: [], 1: []}
    per_file_rate: dict[str, float] = {}
    for filename, entry in per_file.items():
        rate = entry["correct"] / max(entry["windows"], 1)
        class_rates[entry["label"]].append(rate)
        per_file_rate[filename] = rate
    missing = [label for label, values in class_rates.items() if not values]
    if missing and require_both_classes:
        raise ValueError(f"file-equal metrics need both classes; missing {missing}")

    def class_rate(label: int) -> float:
        values = class_rates[label]
        return float(np.mean(values)) if values else float("nan")

    phantom = class_rate(0)
    air2s = class_rate(1)
    return {
        "files_phantom": len(class_rates[0]),
        "files_air2s": len(class_rates[1]),
        "file_equal_phantom_recall": phantom,
        "file_equal_air2s_recall": air2s,
        "file_equal_balanced_accuracy": 0.5 * (phantom + air2s),
        "per_file_recall": per_file_rate,
        "per_file_windows": {name: entry["windows"] for name, entry in per_file.items()},
    }


def metric_row(
    base: dict[str, Any],
    records: Sequence[dict[str, Any]],
    require_both_classes: bool = True,
) -> dict[str, Any]:
    primary = stage2_file_equal_metrics(records, require_both_classes=require_both_classes)
    pooled = stage2_pooled_metrics(records)
    return {
        **base,
        **{key: primary[key] for key in ("files_phantom", "files_air2s")},
        "file_equal_phantom_recall": primary["file_equal_phantom_recall"],
        "file_equal_air2s_recall": primary["file_equal_air2s_recall"],
        "file_equal_balanced_accuracy": primary["file_equal_balanced_accuracy"],
        "windows": pooled["windows"],
        "accuracy": pooled["accuracy"],
        "balanced_accuracy": pooled["balanced_accuracy"],
        "precision_air2s": pooled["precision_air2s"],
        "precision_phantom": pooled["precision_phantom"],
        "phantom_recall": pooled["phantom_recall"],
        "air2s_recall": pooled["air2s_recall"],
        "f1_air2s_binary": pooled["f1_air2s_binary"],
        "f1_macro": pooled["f1_macro"],
        "tn": pooled["tn"],
        "fp": pooled["fp"],
        "fn": pooled["fn"],
        "tp": pooled["tp"],
    }


# ---------------------------------------------------------------------------
# frozen input integrity
# ---------------------------------------------------------------------------
def verify_frozen_inputs() -> dict[str, Any]:
    """Re-hash the frozen inputs and re-check the stage-2 split composition."""
    errors: list[str] = []
    old_hash = sha256_file(OLD_INDEX, label="old index")
    if old_hash != OLD_INDEX_SHA256:
        errors.append(f"old index SHA256 mismatch: {old_hash}")
    heldout_hash = sha256_file(HELDOUT_INDEX, label="heldout index")
    if heldout_hash != HELDOUT_INDEX_SHA256:
        errors.append(f"heldout index SHA256 mismatch: {heldout_hash}")

    preserved = read_json(PRESERVED_INPUTS)
    preserved_records = []
    total = len(preserved["files"])
    for position, entry in enumerate(preserved["files"], start=1):
        path = Path(entry["path"])
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_hash = (
            sha256_file(path, label=f"preserved {position}/{total} {path.name}")
            if exists
            else None
        )
        match = bool(
            exists and actual_size == entry["bytes"] and actual_hash == entry["sha256"]
        )
        preserved_records.append(
            {
                "path": entry["path"],
                "expected_bytes": entry["bytes"],
                "actual_bytes": actual_size,
                "expected_sha256": entry["sha256"],
                "actual_sha256": actual_hash,
                "match": match,
            }
        )
        if not match:
            errors.append(f"preserved input changed: {entry['path']}")

    # The 12 original MAT files are 11.5 GiB in total; only their recorded sizes
    # are re-checked here, which is cheap and still detects any replacement.
    mat_records = []
    for entry in preserved.get("original_mat_sizes", []):
        path = ORIGINAL_MAT_DIR / entry["name"]
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        match = bool(exists and actual_size == entry["bytes"])
        mat_records.append(
            {
                "name": entry["name"],
                "path": str(path),
                "expected_bytes": entry["bytes"],
                "actual_bytes": actual_size,
                "match": match,
            }
        )
        if not match:
            errors.append(f"original MAT changed: {path}")

    composition = {}
    for split, index_path in (
        ("train", OLD_INDEX),
        ("val", OLD_INDEX),
        ("test", OLD_INDEX),
        ("heldout", HELDOUT_INDEX),
    ):
        entry = split_composition(split, index_path)
        entry["expected_windows"] = EXPECTED_SPLIT_WINDOWS[split]
        entry["expected_class_windows"] = list(EXPECTED_CLASS_WINDOWS[split])
        entry["expected_class_files"] = list(EXPECTED_CLASS_FILES[split])
        observed_windows = entry["windows"]
        observed_classes = [entry["class_windows"].get(0, 0), entry["class_windows"].get(1, 0)]
        observed_files = [len(entry["class_files"][0]), len(entry["class_files"][1])]
        entry["windows_match"] = observed_windows == EXPECTED_SPLIT_WINDOWS[split]
        entry["class_windows_match"] = observed_classes == list(EXPECTED_CLASS_WINDOWS[split])
        entry["class_files_match"] = observed_files == list(EXPECTED_CLASS_FILES[split])
        entry["label_mapping"] = {"0": LABEL_TO_NAME[0], "1": LABEL_TO_NAME[1]}
        if not entry["windows_match"]:
            errors.append(
                f"{split}: {observed_windows} windows, expected {EXPECTED_SPLIT_WINDOWS[split]}"
            )
        if not entry["class_windows_match"]:
            errors.append(f"{split}: class windows {observed_classes} != {list(EXPECTED_CLASS_WINDOWS[split])}")
        if not entry["class_files_match"]:
            errors.append(f"{split}: class files {observed_files} != {list(EXPECTED_CLASS_FILES[split])}")
        composition[split] = entry

    return {
        "created_at": utc_now(),
        "old_index": {
            "path": str(OLD_INDEX),
            "sha256": old_hash,
            "expected": OLD_INDEX_SHA256,
            "match": old_hash == OLD_INDEX_SHA256,
        },
        "heldout_index": {
            "path": str(HELDOUT_INDEX),
            "sha256": heldout_hash,
            "expected": HELDOUT_INDEX_SHA256,
            "match": heldout_hash == HELDOUT_INDEX_SHA256,
        },
        "preserved_inputs": {
            "reference": str(PRESERVED_INPUTS),
            "reference_sha256": sha256_file(PRESERVED_INPUTS),
            "checked": len(preserved_records),
            "all_match": all(item["match"] for item in preserved_records),
            "files": preserved_records,
        },
        "original_mat_sizes_checked": {
            "directory": str(ORIGINAL_MAT_DIR),
            "checked": len(mat_records),
            "all_match": all(item["match"] for item in mat_records),
            "full_sha256_recomputed": False,
            "note": "11.5 GiB of source MAT data is size-checked, not re-hashed",
            "files": mat_records,
        },
        "stage2_composition": composition,
        "process_files": {
            name: sha256_file(PROJECT_DIR / name)
            for name in (
                "run_stage2_classification.py",
                "dronerfa_data.py",
                "model.py",
                "train_resnet18.py",
            )
        },
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# frontends
# ---------------------------------------------------------------------------
def frontend_for(n_fft: int, device: torch.device) -> SpectrogramFrontEnd:
    frontend = SpectrogramFrontEnd(
        n_fft=n_fft, hop_length=hop_for(n_fft), image_size=IMAGE_SIZE
    ).to(device)
    frontend.eval()
    return frontend


def check_frontends(device: torch.device, signals: int = 200_000) -> dict[str, Any]:
    """Cheap shape/finiteness check of the two frozen stage-2 frontends."""
    generator = torch.Generator().manual_seed(20260919)
    batch = torch.randn(2, 2, signals, generator=generator)
    report: dict[str, Any] = {}
    for n_fft in NFFT_GRID:
        frontend = frontend_for(n_fft, device)
        hop = hop_for(n_fft)
        frames = (signals - n_fft) // hop + 1
        with torch.inference_mode():
            features = extract_features(frontend, batch.to(device))
        window = frontend.window.detach().cpu()
        report[str(n_fft)] = {
            "n_fft": n_fft,
            "hop_length": hop,
            "hop_ratio": hop / n_fft,
            "native_frames_at_1e6_samples": (1_000_000 - n_fft) // hop + 1,
            "native_frames_probe": frames,
            "output_shape": list(features.shape),
            "output_is_reference_shape": list(features.shape) == [2, 1, IMAGE_SIZE, IMAGE_SIZE],
            "finite": bool(torch.isfinite(features).all()),
            "dtype": str(features.dtype),
            "window_periodic_hamming": bool(
                torch.allclose(window, torch.hamming_window(n_fft), atol=0.0, rtol=0.0)
            ),
            "window_first": float(window[0]),
            "window_last": float(window[-1]),
        }
        del frontend, features
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report


# ---------------------------------------------------------------------------
# validation cache and per-window probes
# ---------------------------------------------------------------------------
def input_probe_payload(tensor: torch.Tensor, samples: int = INPUT_PROBE_SAMPLES) -> bytes:
    """Byte payload of the first ``samples`` IQ samples of one [2, length] window.

    Hashing a fixed prefix of the actual noisy input proves that the same signal
    reached both frontends (and every training seed) without hashing 8 MiB per
    window.
    """
    array = np.ascontiguousarray(tensor[:, :samples].detach().cpu().numpy())
    return array.tobytes()


def derive_initial_state_hash(seed: int) -> str:
    """Re-derive the ResNet18 initialisation hash for a training seed."""
    seed_everything(seed)
    model = ResNet18(input_channels=1, num_classes=2)
    result = state_hash(model.state_dict())
    del model
    return result


def build_validation_cache(
    n_fft: int, device: torch.device, max_val_samples: int | None = None
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]], list[dict[str, Any]]]:
    """Four fixed old-validation conditions as float32 features, shared by all seeds.

    The dataset keeps the full split so the AWGN seed stays
    ``VALIDATION_SEED + stage2_position``; only the selection is narrowed for smoke runs.
    """
    frontend = frontend_for(n_fft, device)
    cached: dict[str, tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]] = {}
    metadata: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        dataset = Stage2WindowDataset(
            OLD_INDEX,
            split="val",
            fixed_snr=CONDITION_VALUES[condition],
            apply_fixed_snr=CONDITION_VALUES[condition] is not None,
            seed=VALIDATION_SEED,
        )
        if dataset.epoch != 0:
            raise RuntimeError("validation dataset epoch must stay 0")
        selected = stratified_limit(dataset.rows, max_val_samples)
        positions = {id(row): index for index, row in enumerate(dataset.rows)}
        indices = [positions[id(row)] for row in selected]
        # Carry the stage-2 position into the cache so every per-window record of a
        # bounded run still names the window of the full split.
        selected = [
            {**row, "index_position": position} for row, position in zip(selected, indices)
        ]
        view = _IndexSubsetDataset(dataset, indices)
        loader = DataLoader(
            view,
            batch_size=BATCH_SIZE,
            sampler=view.sampler(),
            num_workers=0,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(VALIDATION_SEED + 30_000),
        )
        tensors: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        probe_digest = hashlib.sha256()
        window_digests = bytearray()
        started = time.perf_counter()
        with torch.inference_mode():
            for batch, (signals, targets) in enumerate(loader, 1):
                for row in signals:
                    probe_digest.update(input_probe_payload(row))
                features = extract_features(frontend, signals.to(device))
                array = features.detach().to("cpu", torch.float32).numpy()
                if not np.isfinite(array).all():
                    raise RuntimeError(f"non-finite validation features in {condition}")
                for row in array:
                    window_digests.extend(
                        hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest()
                    )
                tensors.append(features.cpu())
                labels.append(targets)
                if batch % 25 == 0 or batch == len(loader):
                    log(f"  val cache nfft{n_fft} {condition} {batch}/{len(loader)}")
        xx, yy = torch.cat(tensors), torch.cat(labels)
        cached[condition] = (xx, yy, selected)
        metadata.append(
            {
                "split": "val",
                "condition": condition,
                "n_fft": n_fft,
                "hop_length": hop_for(n_fft),
                "noise_seed": VALIDATION_SEED,
                "samples": len(selected),
                "stage2_positions": indices,
                "shape": [len(selected), 1, IMAGE_SIZE, IMAGE_SIZE],
                "dtype": "float32",
                "feature_sha256": sha256_bytes(xx.numpy().tobytes()),
                "window_digests_concat_sha256": sha256_bytes(bytes(window_digests)),
                "window_digest_bytes": len(window_digests),
                "input_probe_samples": INPUT_PROBE_SAMPLES,
                "input_probe_sha256": probe_digest.hexdigest(),
                "seconds": time.perf_counter() - started,
            }
        )
        dataset.close()
    del frontend
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cached, metadata


@torch.inference_mode()
def evaluate_cached(
    model: ResNet18,
    cached: dict[str, tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]],
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-epoch validation metrics and per-window records, batch size fixed at 4."""
    model.eval()
    metrics_all: list[dict[str, Any]] = []
    records_all: list[dict[str, Any]] = []
    for condition, (xx, yy, rows) in cached.items():
        loader = DataLoader(
            TensorDataset(xx, yy),
            batch_size=BATCH_SIZE,
            shuffle=False,
            generator=torch.Generator().manual_seed(VALIDATION_SEED + 40_000),
        )
        window_records: list[dict[str, Any]] = []
        total_loss = 0.0
        cursor = 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            with torch.amp.autocast(
                device_type=device.type, enabled=amp_enabled, dtype=torch.float16
            ):
                logits = model(features)
                loss = criterion(logits, labels)
            total_loss += float(loss) * len(labels)
            logits = logits.float().cpu().numpy()
            labels_np = labels.cpu().numpy()
            margins = logits[:, 1] - logits[:, 0]
            predictions = (margins > 0.0).astype(np.int64)
            for offset in range(len(labels_np)):
                source = rows[cursor + offset]
                window_records.append(
                    {
                        "condition": condition,
                        "selection_position": cursor + offset,
                        "index_position": int(source.get("index_position", cursor + offset)),
                        "filename": source["filename"],
                        "target_code": source["target_code"],
                        "label": int(labels_np[offset]),
                        "logit_phantom": float(logits[offset, 0]),
                        "logit_air2s": float(logits[offset, 1]),
                        "logit_margin": float(margins[offset]),
                        "prediction": int(predictions[offset]),
                    }
                )
            cursor += len(labels_np)
        metrics_all.append(
            {
                "condition": condition,
                "loss": total_loss / max(len(rows), 1),
                **metric_row({"condition": condition}, window_records),
            }
        )
        records_all.extend(window_records)
    # ``rows`` entries are index rows, so the label column is already 0/1.
    for metric in metrics_all:
        for key in METRIC_COLUMN_ORDER:
            if isinstance(metric.get(key), (int, float)):
                continue
            raise RuntimeError(f"validation metric {key} is not numeric")
    return metrics_all, records_all


def select_epoch(history: Sequence[dict[str, Any]]) -> int:
    """Mean of the four validation file-equal balanced accuracies; earliest epoch wins ties."""
    return int(
        max(
            history,
            key=lambda row: (row["mean_file_equal_balanced_accuracy"], -int(row["epoch"])),
        )["epoch"]
    )


# ---------------------------------------------------------------------------
# training one (n_fft, seed) run
# ---------------------------------------------------------------------------
def train_run(
    n_fft: int,
    seed: int,
    device: torch.device,
    epochs: int = EPOCHS,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    output_root: Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train one (n_fft, seed) run from scratch and write every per-run artefact."""
    root = output_root or OUTPUT_ROOT
    run_dir = run_directory(n_fft, seed, root)
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    hop = hop_for(n_fft)
    started_at = utc_now()
    amp_enabled = device.type == "cuda"
    log(f"Training nfft{n_fft} hop{hop} seed {seed} -> {run_dir}")

    cached, cache_metadata = build_validation_cache(n_fft, device, max_val_samples)
    write_json(run_dir / "validation_cache_manifest.json", cache_metadata)
    for row in cache_metadata:
        log(
            f"  validation cache {row['condition']}: samples={row['samples']} "
            f"feature_sha256={row['feature_sha256']} ({row['seconds']:.1f}s)"
        )

    seed_everything(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # The model is built immediately after seeding, so its initial weights are a
    # function of the training seed alone and are identical for both n_fft values.
    model = ResNet18(input_channels=1, num_classes=2).to(device)
    initial_hash = state_hash(model.state_dict())
    derived_hash = derive_initial_state_hash(seed)
    if initial_hash != derived_hash:
        raise RuntimeError(
            f"initial ResNet18 weights for seed {seed} do not reproduce the derived hash"
        )
    frontend = frontend_for(n_fft, device)

    dataset = Stage2WindowDataset(
        OLD_INDEX,
        split="train",
        train_snrs=parse_snr_list(TRAIN_SNRS),
        max_samples=max_train_samples,
        seed=seed,
    )
    counts = {label: sum(1 for row in dataset.rows if int(row["label"]) == label) for label in (0, 1)}
    if not all(counts.values()):
        raise ValueError(f"both machine-type classes are required, got {counts}")
    weights = torch.tensor(
        [len(dataset) / (2 * counts[label]) for label in (0, 1)], dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    config = {
        "phase": "STAGE2-BASE",
        "task": "oracle machine-type classification, Phantom 4 Pro (0) vs Air 2S (1)",
        "n_fft": n_fft,
        "hop_length": hop,
        "hop_ratio": hop / n_fft,
        "image_size": IMAGE_SIZE,
        "training_seed": seed,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "train_snrs": TRAIN_SNRS,
        "train_samples": len(dataset),
        "class_counts": counts,
        "class_weights": weights.detach().cpu().tolist(),
        "initial_model_sha256": initial_hash,
        "initial_model_sha256_derived": derived_hash,
        "validation_seed": VALIDATION_SEED,
        "validation_conditions": list(CONDITIONS),
        "selection": EPOCH_SELECTION_RULE,
        "tie_break": TIE_BREAK,
        "prediction": PREDICTION_RULE,
        "label_mapping": {str(label): LABEL_TO_NAME[label] for label in (0, 1)},
        "index_sha256": sha256_file(OLD_INDEX),
        "frontend": {
            "n_fft": n_fft,
            "hop_length": hop,
            "image_size": IMAGE_SIZE,
            "window": "torch.hamming_window(n_fft) periodic",
            "implementation": "model.SpectrogramFrontEnd (frozen)",
        },
        "optimizer": {
            "type": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "amp": amp_enabled,
        "amp_dtype": "float16",
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "smoke": smoke,
        "started_at": started_at,
        "source_sha256": {
            name: sha256_file(PROJECT_DIR / name)
            for name in (
                "run_stage2_classification.py",
                "dronerfa_data.py",
                "model.py",
                "train_resnet18.py",
            )
        },
        "heldout_index_sha256": sha256_file(HELDOUT_INDEX),
        "status": "running",
    }
    write_json(run_dir / "run_config.json", config)

    history: list[dict[str, Any]] = []
    order_hashes: list[str] = []
    checkpoint_path = run_dir / "best_model.pt"
    try:
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            dataset.set_epoch(epoch)
            order = torch.randperm(
                len(dataset), generator=torch.Generator().manual_seed(seed + epoch)
            ).tolist()
            order_hashes.append(sha256_bytes(np.asarray(order, dtype=np.int64).tobytes()))
            loader = DataLoader(
                Subset(dataset, order),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
                generator=torch.Generator().manual_seed(seed + epoch + 50_000),
            )
            log(f"  epoch {epoch}/{epochs} nfft{n_fft} seed {seed}")
            train_loss, train_metrics = train_epoch(
                frontend, model, loader, criterion, optimizer, scaler, device, amp_enabled
            )
            metrics, records = evaluate_cached(model, cached, criterion, device, amp_enabled)
            mean_score = float(
                np.mean([row["file_equal_balanced_accuracy"] for row in metrics])
            )
            clean = next(row for row in metrics if row["condition"] == "clean")
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                    "train_air2s_recall": train_metrics["recall"],
                    "clean_file_equal_balanced_accuracy": clean["file_equal_balanced_accuracy"],
                    "mean_file_equal_balanced_accuracy": mean_score,
                    "seconds": time.perf_counter() - started,
                }
            )
            write_csv(
                run_dir / "validation_metrics.csv",
                _accumulate_validation_metrics(run_dir, epoch, metrics),
            )
            write_csv(run_dir / "training_history.csv", history)
            write_csv(run_dir / f"validation_scores_epoch_{epoch:02d}.csv", records)
            torch.save(
                {
                    "epoch": epoch,
                    "frontend_state": frontend.state_dict(),
                    "model_state": model.state_dict(),
                    "n_fft": n_fft,
                    "hop_length": hop,
                    "validation_seed": VALIDATION_SEED,
                    "val_metrics": metrics,
                    "selection_rule": EPOCH_SELECTION_RULE,
                    "tie_break": TIE_BREAK,
                    "prediction_rule": PREDICTION_RULE,
                    "training_seed": seed,
                    "initial_model_sha256": initial_hash,
                    "run_config": config,
                },
                run_dir / f"epoch_{epoch:02d}.pt",
            )
            selection = {
                "n_fft": n_fft,
                "training_seed": seed,
                "rule": EPOCH_SELECTION_RULE,
                "tie_break": TIE_BREAK,
                "mean_file_equal_balanced_accuracy_by_epoch": {
                    str(row["epoch"]): row["mean_file_equal_balanced_accuracy"] for row in history
                },
                "selected_epoch": select_epoch(history),
            }
            write_json(run_dir / "selection.json", selection)
            if selection["selected_epoch"] == epoch:
                shutil.copyfile(run_dir / f"epoch_{epoch:02d}.pt", checkpoint_path)
            log(
                f"  epoch {epoch}: mean file-equal BA={mean_score:.6f} "
                f"({history[-1]['seconds']:.1f}s) selected={selection['selected_epoch']}"
            )
            for metric in metrics:
                log(
                    f"    {metric['condition']}: file-equal BA="
                    f"{metric['file_equal_balanced_accuracy']:.6f} pooled BA="
                    f"{metric['balanced_accuracy']:.6f} tp={metric['tp']} tn={metric['tn']} "
                    f"fp={metric['fp']} fn={metric['fn']}"
                )
    finally:
        dataset.close()

    selected = select_epoch(history)
    config["epoch_order_sha256"] = order_hashes
    config["selected_epoch"] = selected
    config["retained_epoch_checkpoints"] = []
    config["removed_epoch_checkpoints"] = []
    for path in sorted(run_dir.glob("epoch_*.pt")):
        if path.name == f"epoch_{selected:02d}.pt":
            config["retained_epoch_checkpoints"].append(path.name)
            continue
        path.unlink()
        config["removed_epoch_checkpoints"].append(path.name)
    config["completed_at"] = utc_now()
    config["status"] = "completed"
    write_json(run_dir / "run_config.json", config)
    log(f"  kept {config['retained_epoch_checkpoints']}, removed {len(config['removed_epoch_checkpoints'])} epoch checkpoints")

    release_run_resources(model, optimizer, scaler, cached)
    return {"run_directory": str(run_dir), "config": config, "history": history}


def release_run_resources(
    model: Any = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    cached: dict[str, Any] | None = None,
) -> None:
    """Best-effort cleanup that must never abort the surrounding pipeline."""
    try:
        del model, optimizer, scaler
        if cached is not None:
            cached.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as error:  # pragma: no cover - defensive
        log(f"WARNING: resource cleanup failed ({error!r}); continuing")


def _accumulate_validation_metrics(
    run_dir: Path, epoch: int, metrics: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    path = run_dir / "validation_metrics.csv"
    rows = read_csv_rows(path) if path.is_file() else []
    rows.extend(
        {"epoch": epoch, **{key: str(value) for key, value in metric.items()}}
        for metric in metrics
    )
    return rows


# ---------------------------------------------------------------------------
# n_fft selection (old validation only)
# ---------------------------------------------------------------------------
def experiment_grid(
    output_root: Path | None = None, required: bool = True
) -> list[dict[str, Any]]:
    """Every (n_fft, seed) run; ``required=False`` allows a partial grid."""
    root = output_root or OUTPUT_ROOT
    grid = []
    for n_fft in NFFT_GRID:
        for seed in TRAINING_SEEDS:
            run_dir = run_directory(n_fft, seed, root)
            config_path = run_dir / "run_config.json"
            if not config_path.is_file():
                if required:
                    raise FileNotFoundError(f"missing run: {run_dir}")
                continue
            config = read_json(config_path)
            checkpoint = run_dir / "best_model.pt"
            if not checkpoint.is_file():
                if required:
                    raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
                continue
            grid.append(
                {
                    "n_fft": n_fft,
                    "hop_length": hop_for(n_fft),
                    "training_seed": seed,
                    "run_directory": run_dir,
                    "config": config,
                    "checkpoint": checkpoint,
                    "epoch": int(config["selected_epoch"]),
                }
            )
    return grid


def run_validation_evidence(n_fft: int, seed: int, output_root: Path | None = None) -> dict[str, Any]:
    """Old-validation evidence of one selected epoch, read back from its own artefacts."""
    run_dir = run_directory(n_fft, seed, output_root)
    selection = read_json(run_dir / "selection.json")
    selected_epoch = int(selection["selected_epoch"])
    metrics_path = run_dir / "validation_metrics.csv"
    rows = [
        row for row in read_csv_rows(metrics_path) if int(row["epoch"]) == selected_epoch
    ]
    if len(rows) != len(CONDITIONS):
        raise RuntimeError(
            f"{run_dir}: {len(rows)} validation rows for epoch {selected_epoch}, "
            f"expected {len(CONDITIONS)}"
        )
    by_condition = {
        row["condition"]: {
            "windows": int(float(row["windows"])),
            "file_equal_balanced_accuracy": float(row["file_equal_balanced_accuracy"]),
            "pooled_balanced_accuracy": float(row["balanced_accuracy"]),
            "file_equal_phantom_recall": float(row["file_equal_phantom_recall"]),
            "file_equal_air2s_recall": float(row["file_equal_air2s_recall"]),
        }
        for row in rows
    }
    ordered = [by_condition[condition] for condition in CONDITIONS]
    return {
        "n_fft": n_fft,
        "hop_length": hop_for(n_fft),
        "training_seed": seed,
        "selected_epoch": selected_epoch,
        "selection_rule": selection["rule"],
        "tie_break": selection["tie_break"],
        "per_condition": by_condition,
        "four_condition_mean_file_equal_balacc": mean_or_zero(
            [row["file_equal_balanced_accuracy"] for row in ordered]
        ),
        "four_condition_mean_pooled_balacc": mean_or_zero(
            [row["pooled_balanced_accuracy"] for row in ordered]
        ),
        "validation_metrics_sha256": sha256_file(metrics_path),
        "selection_json_sha256": sha256_file(run_dir / "selection.json"),
    }


def build_nfft_selection(output_root: Path | None = None) -> dict[str, Any]:
    """The frozen n_fft rule, evaluated on old-validation evidence only."""
    root = output_root or OUTPUT_ROOT
    per_model = []
    seeds_used: set[int] = set()
    for n_fft in NFFT_GRID:
        for seed in TRAINING_SEEDS:
            config_path = run_directory(n_fft, seed, root) / "run_config.json"
            if not config_path.is_file():
                continue
            per_model.append(run_validation_evidence(n_fft, seed, root))
            seeds_used.add(seed)
    if not per_model:
        raise RuntimeError("no completed run is available for the configuration choice")
    for n_fft in NFFT_GRID:
        if not any(row["n_fft"] == n_fft for row in per_model):
            raise RuntimeError(f"no completed run for n_fft={n_fft}; both grids are required")

    per_nfft = []
    for n_fft in NFFT_GRID:
        rows = [row for row in per_model if row["n_fft"] == n_fft]
        values = [row["four_condition_mean_file_equal_balacc"] for row in rows]
        pooled = [row["four_condition_mean_pooled_balacc"] for row in rows]
        per_nfft.append(
            {
                "n_fft": n_fft,
                "hop_length": hop_for(n_fft),
                "training_seeds": [row["training_seed"] for row in rows],
                "mean_file_equal_balacc_over_seeds": mean_or_zero(values),
                "std_file_equal_balacc_over_seeds": spread(values),
                "mean_pooled_balacc_over_seeds": mean_or_zero(pooled),
                "per_seed_file_equal_balacc": {
                    str(row["training_seed"]): row["four_condition_mean_file_equal_balacc"]
                    for row in rows
                },
            }
        )
    # Higher mean wins; an exact tie goes to the cheaper (smaller) n_fft.
    best = max(
        per_nfft,
        key=lambda row: (row["mean_file_equal_balacc_over_seeds"], -row["n_fft"]),
    )
    runner_up = min(
        (row for row in per_nfft if row["n_fft"] != best["n_fft"]),
        key=lambda row: row["n_fft"],
        default=None,
    )
    margin = (
        best["mean_file_equal_balacc_over_seeds"] - runner_up["mean_file_equal_balacc_over_seeds"]
        if runner_up
        else float("nan")
    )
    return {
        "task": "STAGE2-BASE",
        "purpose": "choose between the two STFT sizes using old-validation evidence only",
        "selection_rule": NFFT_SELECTION_RULE,
        "tie_break": NFFT_TIE_BREAK,
        "evidence_source": "old validation split only (window_index.csv, split='val')",
        "conditions": list(CONDITIONS),
        "validation_noise_seed": VALIDATION_SEED,
        "epoch_rule": EPOCH_SELECTION_RULE,
        "seeds_requested": list(TRAINING_SEEDS),
        "seeds_available": sorted(seeds_used),
        "partial_grid": sorted(seeds_used) != list(TRAINING_SEEDS),
        "per_model": per_model,
        "per_nfft": per_nfft,
        "selected_n_fft": best["n_fft"],
        "selected_hop_length": hop_for(best["n_fft"]),
        "selection_margin": margin,
        "exact_tie": bool(runner_up and margin == 0.0),
        "leakage_guard": (
            "only the old validation split was read; no other split of the project is "
            "referenced by this file"
        ),
        "written_at": utc_now(),
    }


def command_select_nfft(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    selection_path = root / "validation_nfft_selection.json"
    lock_path = root / "selection_lock.json"
    if (root / "heldout_window_predictions.csv").is_file():
        raise SystemExit(
            "refusing to (re)write the configuration choice after evaluation results exist: "
            f"{root / 'heldout_window_predictions.csv'} is already present"
        )
    selection = build_nfft_selection(root)
    write_json(selection_path, selection)
    lock = {
        "task": "STAGE2-BASE",
        "selection_file": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "selected_n_fft": selection["selected_n_fft"],
        "per_model_evidence_sha256": {
            f"nfft{row['n_fft']}_seed{row['training_seed']}": {
                "validation_metrics_sha256": row["validation_metrics_sha256"],
                "selection_json_sha256": row["selection_json_sha256"],
            }
            for row in selection["per_model"]
        },
        "locked_at": utc_now(),
        "rule": SELECTION_LOCK_RULE,
    }
    write_json(lock_path, lock)
    log(
        f"Selected n_fft={selection['selected_n_fft']} "
        f"(mean file-equal validation BA {selection['per_nfft'][0]['mean_file_equal_balacc_over_seeds']:.6f} "
        f"vs {selection['per_nfft'][1]['mean_file_equal_balacc_over_seeds']:.6f}); "
        f"locked at {lock['selection_sha256']}"
    )
    return 0


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def representative_windows() -> dict[str, dict[str, Any]]:
    """One fixed Phantom window and one fixed Air 2S window from the frozen index."""
    rows = stage2_index_rows(HELDOUT_INDEX, "heldout")
    result: dict[str, dict[str, Any]] = {}
    for code in (PHANTOM_CODE, AIR2S_CODE):
        for row in rows:
            if row["target_code"] == code:
                result[code] = row
                break
        else:
            raise RuntimeError(f"held-out index has no row with target_code {code}")
    return result


def read_representative_iq(row: dict[str, Any]) -> np.ndarray:
    import h5py

    with h5py.File(row["file_path"], "r") as handle:
        start = int(row["start"])
        length = int(row["length"])
        i_values = read_vector_slice(handle["RF0_I"], start, length)
        q_values = read_vector_slice(handle["RF0_Q"], start, length)
    return rms_normalize(i_values + 1j * q_values)


def native_spectrogram(iq: np.ndarray, n_fft: int, device: torch.device) -> np.ndarray:
    """Native log-power STFT of one window, float32, same convention as the model."""
    window = torch.hamming_window(n_fft).to(device)
    signal = torch.from_numpy(np.ascontiguousarray(iq)).to(device)
    with torch.inference_mode():
        spectra = torch.stft(
            signal,
            n_fft=n_fft,
            hop_length=hop_for(n_fft),
            win_length=n_fft,
            window=window,
            center=False,
            normalized=True,
            onesided=False,
            return_complex=True,
        )
        log_power = 10.0 * torch.log10(spectra.abs().square() + 1e-10)
        log_power = torch.fft.fftshift(log_power, dim=0)
    return log_power.detach().to("cpu", torch.float32).numpy()


def write_figures(device: torch.device, output_root: Path | None = None) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = output_root or OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    sample_rate = 100e6
    evidence: dict[str, Any] = {
        "rule": REPRESENTATIVE_WINDOW_RULE,
        "stft_convention": (
            "torch.stft center=False normalized=True onesided=False, periodic Hamming, fftshift"
        ),
        "n_fft_grid": list(NFFT_GRID),
        "per_window_png_written": False,
        "windows": {},
        "figures": {},
    }
    for code, row in representative_windows().items():
        iq = read_representative_iq(row)
        probe = torch.from_numpy(np.stack((iq.real, iq.imag), axis=0)).unsqueeze(0).to(device)
        native: dict[int, np.ndarray] = {}
        model_inputs: dict[int, np.ndarray] = {}
        detail: dict[str, Any] = {}
        for n_fft in NFFT_GRID:
            spectra = native_spectrogram(iq, n_fft, device)
            native[n_fft] = spectra
            frontend = frontend_for(n_fft, device)
            with torch.inference_mode():
                features = extract_features(frontend, probe)[0, 0].detach().cpu().numpy()
            model_inputs[n_fft] = features
            detail[str(n_fft)] = {
                "native_shape_frequency_time": [int(spectra.shape[0]), int(spectra.shape[1])],
                "native_frequency_bins": int(spectra.shape[0]),
                "native_time_frames": int(spectra.shape[1]),
                "model_input_shape": [int(features.shape[0]), int(features.shape[1])],
                "model_input_shape_ok": list(features.shape) == [IMAGE_SIZE, IMAGE_SIZE],
                "model_input_finite": bool(np.isfinite(features).all()),
                "time_step_ms": hop_for(n_fft) / sample_rate * 1e3,
                "frequency_step_khz": sample_rate / n_fft / 1e3,
                "native_spectrogram_sha256": sha256_bytes(
                    np.ascontiguousarray(spectra, dtype=np.float32).tobytes()
                ),
                "model_input_sha256": sha256_bytes(
                    np.ascontiguousarray(features, dtype=np.float32).tobytes()
                ),
            }
            del frontend
            if device.type == "cuda":
                torch.cuda.empty_cache()

        half_mhz = sample_rate / 1e6 / 2.0
        figure, axes = plt.subplots(4, 1, figsize=(11.0, 15.0), constrained_layout=True)
        for axis, n_fft in zip(axes[:2], NFFT_GRID):
            spectra = native[n_fft]
            extent = (
                0.0,
                spectra.shape[1] * hop_for(n_fft) / sample_rate * 1e3,
                -half_mhz,
                half_mhz,
            )
            image = axis.imshow(spectra, aspect="auto", origin="lower", extent=extent, cmap="magma")
            axis.set_title(
                f"native STFT n_fft={n_fft}, hop={hop_for(n_fft)}: "
                f"{spectra.shape[0]} freq x {spectra.shape[1]} time "
                f"({sample_rate / n_fft / 1e3:.1f} kHz x {hop_for(n_fft) / sample_rate * 1e3:.4f} ms)"
            )
            axis.set_xlabel("time within the window (ms)")
            axis.set_ylabel("offset from RF0 channel centre (MHz)")
            figure.colorbar(image, ax=axis, label="10log10|X|^2 (dB)")
        for axis, n_fft in zip(axes[2:], NFFT_GRID):
            features = model_inputs[n_fft]
            image = axis.imshow(features, aspect="auto", origin="lower", cmap="viridis")
            axis.set_title(
                f"actual model input n_fft={n_fft}: {features.shape[0]}x{features.shape[1]} "
                f"z-scored log power (bilinear resize of the "
                f"{native[n_fft].shape[0]}x{native[n_fft].shape[1]} native grid)"
            )
            axis.set_xlabel("resized time axis (column, 256 bins)")
            axis.set_ylabel("resized frequency axis (row, 256 bins)")
            figure.colorbar(image, ax=axis, label="z-score")
        figure.suptitle(
            f"{LABEL_TO_NAME[CODE_TO_LABEL[code]]} representative window: {row['filename']} "
            f"start={row['start']} len={row['length']} label={row['label']} "
            f"(fixed rule, no per-window PNGs)",
            fontsize=11,
        )
        figure_path = root / f"representative_{'phantom' if code == PHANTOM_CODE else 'air2s'}.png"
        figure.savefig(figure_path, dpi=150)
        plt.close(figure)
        evidence["windows"][code] = {
            "class_label": CODE_TO_LABEL[code],
            "class_name": LABEL_TO_NAME[CODE_TO_LABEL[code]],
            "filename": row["filename"],
            "file_path": row["file_path"],
            "index_position": int(row["index_position"]),
            "start": int(row["start"]),
            "length": int(row["length"]),
            "split": row["split"],
            "n_fft_detail": detail,
            "figure": figure_path.name,
            "figure_sha256": sha256_file(figure_path),
        }
        evidence["figures"][figure_path.name] = sha256_file(figure_path)
        log(f"  wrote {figure_path.name}")
    write_json(root / "figure_evidence.json", evidence)
    return evidence


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def load_models(grid: Sequence[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    loaded = []
    for entry in grid:
        checkpoint = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
        if int(checkpoint["epoch"]) != entry["epoch"]:
            raise RuntimeError(f"checkpoint epoch mismatch in {entry['checkpoint']}")
        frontend = frontend_for(entry["n_fft"], device)
        frontend.load_state_dict(checkpoint["frontend_state"])
        frontend.eval()
        network = ResNet18(input_channels=1, num_classes=2).to(device)
        network.load_state_dict(checkpoint["model_state"])
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        loaded.append(
            {
                **entry,
                "network": network,
                "frontend": frontend,
                "checkpoint_sha256": sha256_file(entry["checkpoint"]),
            }
        )
        del checkpoint
    return loaded


def noise_pairing_digest(
    index_path: Path, split: str, condition: str, noise_seed: int
) -> dict[str, Any]:
    """Re-derive the added AWGN per window straight from the frozen index.

    Three representative stage-2 positions (first/middle/last) are probed with the
    frozen ``add_complex_awgn`` on a fixed 8-sample delta probe using
    ``default_rng(noise_seed + stage2_position)``. The digest depends only on
    (index, split, condition, noise seed), so it must be identical for the two
    ``n_fft`` values and for every training seed.
    """
    rows = stage2_index_rows(index_path, split)
    if not rows:
        raise ValueError(f"index {index_path} has no rows for split {split}")
    snr = CONDITION_VALUES[condition]
    digest = hashlib.sha256()
    probes = []
    for position in sorted({0, len(rows) // 2, len(rows) - 1}):
        row = rows[position]
        rng_seed = noise_seed + position
        rng = np.random.default_rng(rng_seed)
        if snr is None:
            draw = rng.standard_normal(8, dtype=np.float32)
            payload = np.asarray(draw, dtype=np.float32).tobytes()
        else:
            probe = np.zeros(8, dtype=np.complex64)
            probe[0] = 1.0
            noisy = add_complex_awgn(probe, snr, rng)
            payload = np.asarray(noisy, dtype=np.complex64).tobytes()
        digest.update(payload)
        probes.append(
            {
                "index_position": position,
                "filename": row["filename"],
                "start": int(row["start"]),
                "rng_seed": rng_seed,
                "probe_draw_sha256": sha256_bytes(payload),
            }
        )
    return {
        "split": split,
        "condition": condition,
        "fixed_snr_db": snr,
        "noise_seed": noise_seed,
        "rng_rule": (
            "np.random.default_rng(noise_seed + epoch*len(stage2_split) + stage2_position), "
            "epoch pinned to 0 for validation, held-out and old-test"
        ),
        "probe_algorithm": (
            "clean: 8 standard_normal(float32) draws per stage-2 position; noisy: frozen "
            "add_complex_awgn on an 8-sample delta probe"
        ),
        "probe_positions": probes,
        "digest_sha256": digest.hexdigest(),
    }


def evaluate_split(
    split: str,
    index_path: Path,
    conditions: Sequence[str],
    noise_seed: int,
    models: Sequence[dict[str, Any]],
    device: torch.device,
    cache_dir: Path | None = None,
    write_cache_digests: bool = False,
    output_root: Path | None = None,
    max_windows: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Predict every (model, condition, window) triple of one split.

    The float32 feature cache is built once per (n_fft, condition) and shared by the
    three training seeds of that n_fft. ``max_windows`` truncates the prefix of the
    split before any feature is built, so a bounded run only reads those windows.
    """
    root = output_root or OUTPUT_ROOT
    amp_enabled = device.type == "cuda"
    rows = stage2_index_rows(index_path, split)
    if max_windows is not None:
        rows = rows[:max_windows]
    if not rows:
        raise ValueError(f"index {index_path} has no rows for split {split}")
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    windows = len(rows)
    prediction_rows: list[dict[str, Any]] = []
    cache_metadata: list[dict[str, Any]] = []
    pairing: list[dict[str, Any]] = []

    for condition in conditions:
        snr = CONDITION_VALUES[condition]
        pairing.append(noise_pairing_digest(index_path, split, condition, noise_seed))
        for n_fft in NFFT_GRID:
            entries = [entry for entry in models if entry["n_fft"] == n_fft]
            if not entries:
                continue
            log(f"  features {split} {condition} nfft{n_fft} ({windows} windows)")
            dataset = Stage2WindowDataset(
                index_path,
                split=split,
                fixed_snr=snr,
                apply_fixed_snr=snr is not None,
                seed=noise_seed,
            )
            if dataset.epoch != 0:
                raise RuntimeError("dataset epoch must stay 0 so the RNG index is the position")
            feature_array = np.empty((windows, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=FEATURE_DTYPE)
            window_digests = bytearray()
            probe_digest = hashlib.sha256()
            started = time.perf_counter()
            frontend = entries[0]["frontend"]
            try:
                for start in range(0, windows, BATCH_SIZE):
                    stop = min(start + BATCH_SIZE, windows)
                    signals = []
                    for position in range(start, stop):
                        tensor, label = dataset[position]
                        if int(label) != int(labels[position]):
                            raise RuntimeError("dataset labels disagree with the index")
                        probe_digest.update(input_probe_payload(tensor))
                        signals.append(tensor)
                    batch = torch.stack(signals).to(device, non_blocking=True)
                    with torch.inference_mode():
                        batch_features = extract_features(frontend, batch)
                    array = batch_features.detach().to("cpu", torch.float32).numpy()
                    if array.shape != (stop - start, 1, IMAGE_SIZE, IMAGE_SIZE):
                        raise RuntimeError(f"unexpected feature shape {array.shape}")
                    if not np.isfinite(array).all():
                        raise RuntimeError(f"non-finite features in {split}/{condition}")
                    feature_array[start:stop] = array
                    for row in array:
                        window_digests.extend(
                            hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest()
                        )
                    if (start // BATCH_SIZE + 1) % 100 == 0 or stop == windows:
                        elapsed = time.perf_counter() - started
                        log(
                            f"    {stop}/{windows} ({elapsed:.1f}s, "
                            f"{stop / max(elapsed, 1e-9):.1f} win/s)"
                        )
            finally:
                dataset.close()
            metadata = {
                "split": split,
                "condition": condition,
                "n_fft": n_fft,
                "hop_length": hop_for(n_fft),
                "noise_seed": noise_seed,
                "windows": windows,
                "shape": [windows, 1, IMAGE_SIZE, IMAGE_SIZE],
                "dtype": "float32",
                "feature_sha256": sha256_bytes(feature_array.tobytes()),
                "window_digests_concat_sha256": sha256_bytes(bytes(window_digests)),
                "window_digest_bytes": len(window_digests),
                "input_probe_samples": INPUT_PROBE_SAMPLES,
                "input_probe_sha256": probe_digest.hexdigest(),
                "seconds": time.perf_counter() - started,
            }
            cache_metadata.append(metadata)
            if cache_dir is not None and write_cache_digests:
                cache_dir.mkdir(parents=True, exist_ok=True)
                (
                    cache_dir / f"window_digests_{split}_{condition}_{nfft_label(n_fft)}.bin"
                ).write_bytes(bytes(window_digests))
            del window_digests

            for entry in entries:
                logits_all = np.empty((windows, 2), dtype=np.float32)
                margins_all = np.empty(windows, dtype=np.float32)
                with torch.inference_mode():
                    for start in range(0, windows, BATCH_SIZE):
                        stop = min(start + BATCH_SIZE, windows)
                        batch = torch.from_numpy(
                            np.ascontiguousarray(feature_array[start:stop])
                        ).to(device)
                        with torch.amp.autocast(
                            device_type=device.type, enabled=amp_enabled, dtype=torch.float16
                        ):
                            logits = entry["network"](batch)
                        logits = logits.float().cpu().numpy()
                        logits_all[start:stop] = logits
                        margins_all[start:stop] = logits[:, 1] - logits[:, 0]
                if not np.isfinite(logits_all).all() or not np.isfinite(margins_all).all():
                    raise RuntimeError(
                        f"non-finite logits for nfft{n_fft} seed {entry['training_seed']}"
                    )
                predictions = (margins_all > 0.0).astype(np.int64)
                for position, row in enumerate(rows):
                    margin = float(margins_all[position])
                    prediction_rows.append(
                        {
                            "split": split,
                            "condition": condition,
                            "n_fft": n_fft,
                            "hop_length": hop_for(n_fft),
                            "training_seed": entry["training_seed"],
                            "epoch": entry["epoch"],
                            "model_id": f"nfft{n_fft}_seed{entry['training_seed']}",
                            "index_position": row["index_position"],
                            "file_path": row["file_path"],
                            "filename": row["filename"],
                            "target_code": row["target_code"],
                            "target_name": row["target_name"],
                            "distance_code": row["distance_code"],
                            "segment_code": row["segment_code"],
                            "row_split": row["split"],
                            "label": int(row["label"]),
                            "class_name": row["class_name"],
                            "block_index": int(row["block_index"]),
                            "window_index": int(row["window_index"]),
                            "start": int(row["start"]),
                            "length": int(row["length"]),
                            "source_samples": int(row["source_samples"]),
                            "logit_phantom": float(logits_all[position, 0]),
                            "logit_air2s": float(logits_all[position, 1]),
                            "logit_margin": margin,
                            "air2s_score": float(1.0 / (1.0 + np.exp(-np.float64(margin)))),
                            "prediction": int(predictions[position]),
                        }
                    )
                log(
                    f"    nfft{n_fft} seed {entry['training_seed']} {split}/{condition}: "
                    f"air2s {int(predictions.sum())}/{windows}"
                )
            del feature_array
            if device.type == "cuda":
                torch.cuda.empty_cache()
    del models
    return prediction_rows, cache_metadata, pairing


def command_evaluate(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    selection_path = root / "validation_nfft_selection.json"
    lock_path = root / "selection_lock.json"
    if not selection_path.is_file() or not lock_path.is_file():
        raise SystemExit(
            "the configuration choice must be written first: run `select-nfft` before `evaluate`"
        )
    lock = read_json(lock_path)
    selection_hash_before = sha256_file(selection_path)
    if selection_hash_before != lock["selection_sha256"]:
        raise SystemExit(
            f"{selection_path} was modified after it was locked "
            f"({selection_hash_before} != {lock['selection_sha256']})"
        )
    device = resolve_device(args.device)
    conditions = conditions_from(args)
    grid = experiment_grid(root, required=False)
    if not grid:
        raise SystemExit(f"no completed run under {root}; train the grid first")
    if args.max_windows is None:
        covered = {(entry["n_fft"], entry["training_seed"]) for entry in grid}
        expected = {(n_fft, seed) for n_fft in NFFT_GRID for seed in TRAINING_SEEDS}
        if covered != expected:
            raise SystemExit(
                f"the full evaluation needs all {len(expected)} runs; missing "
                f"{sorted(expected - covered)}"
            )
    log(f"Configuration choice (old validation only): n_fft={lock['selected_n_fft']}")

    models = load_models(grid, device)
    started = time.perf_counter()
    heldout_rows, heldout_cache, heldout_pairing = evaluate_split(
        "heldout",
        HELDOUT_INDEX,
        conditions,
        HELDOUT_NOISE_SEED,
        models,
        device,
        cache_dir=root / "feature_digests",
        write_cache_digests=args.max_windows is None,
        output_root=root,
        max_windows=args.max_windows,
    )
    log(f"Held-out predictions: {len(heldout_rows)} rows ({time.perf_counter() - started:.1f}s)")
    write_csv(root / "heldout_window_predictions.csv", heldout_rows)

    models = load_models(grid, device)
    test_rows, test_cache, test_pairing = evaluate_split(
        "test",
        OLD_INDEX,
        TEST_CONDITIONS,
        TEST_NOISE_SEED,
        models,
        device,
        cache_dir=root / "feature_digests",
        write_cache_digests=args.max_windows is None,
        output_root=root,
        max_windows=args.max_windows,
    )
    log(f"Old-test clean predictions: {len(test_rows)} rows")
    write_csv(root / "test_clean_window_predictions.csv", test_rows)

    truncated = args.max_windows is not None
    if not truncated:
        if len(heldout_rows) != EXPECTED_HELDOUT_ROWS:
            raise RuntimeError(
                f"held-out grid produced {len(heldout_rows)} rows, expected {EXPECTED_HELDOUT_ROWS}"
            )
        if len(test_rows) != EXPECTED_TEST_ROWS:
            raise RuntimeError(
                f"old-test grid produced {len(test_rows)} rows, expected {EXPECTED_TEST_ROWS}"
            )
    write_json(
        root / "prediction_cache_manifest.json",
        {
            "heldout": heldout_cache,
            "test_clean": test_cache,
            "noise_pairing": {"heldout": heldout_pairing, "test_clean": test_pairing},
            "cache_location": (
                "in-memory float32 per (split, condition, n_fft); freed before the next grid"
            ),
            "disk_feature_cache_written": False,
            "window_digest_files": None if truncated else "feature_digests/",
            "max_windows": args.max_windows,
            "truncated_debug_run": truncated,
            "expected_heldout_rows": EXPECTED_HELDOUT_ROWS,
            "expected_test_rows": EXPECTED_TEST_ROWS,
        },
    )
    selection_hash_after = sha256_file(selection_path)
    record = {
        "task": "STAGE2-BASE",
        "selection_file": str(selection_path),
        "selection_sha256_before_evaluation": selection_hash_before,
        "selection_sha256_after_evaluation": selection_hash_after,
        "selection_unchanged": selection_hash_before == selection_hash_after,
        "selected_n_fft": lock["selected_n_fft"],
        "heldout_rows": len(heldout_rows),
        "test_rows": len(test_rows),
        "truncated_debug_run": truncated,
        "generated_at": utc_now(),
    }
    write_json(root / "heldout_generation_record.json", record)
    if not record["selection_unchanged"]:
        raise RuntimeError("the configuration choice changed while predictions were produced")
    log(f"Configuration choice unchanged after evaluation ({selection_hash_after})")
    return 0


def conditions_from(args: argparse.Namespace) -> list[str]:
    conditions = [token.strip() for token in args.conditions.split(",") if token.strip()]
    for condition in conditions:
        if condition not in CONDITION_VALUES:
            raise SystemExit(f"unknown condition {condition}")
    return conditions


# ---------------------------------------------------------------------------
# summarisation
# ---------------------------------------------------------------------------
def cast_prediction_row(row: dict[str, str]) -> dict[str, Any]:
    cast: dict[str, Any] = dict(row)
    for key in ("n_fft", "hop_length", "training_seed", "epoch", "index_position", "label", "prediction"):
        cast[key] = int(cast[key])
    for key in ("logit_phantom", "logit_air2s", "logit_margin", "air2s_score"):
        cast[key] = float(cast[key])
    for key in ("block_index", "window_index", "start", "length", "source_samples"):
        cast[key] = int(cast[key])
    return cast


def summarize_predictions(
    predictions: Sequence[dict[str, Any]],
    split: str,
    output_root: Path | None = None,
    require_both_classes: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Build every metric table from the per-window prediction rows."""
    scoped = [row for row in predictions if row["split"] == split]
    if not scoped:
        raise ValueError(f"no predictions for split {split}")
    # The old-test split is evaluated clean only, so the condition set is read back
    # from the rows instead of assuming the four held-out conditions.
    present_conditions = [
        condition
        for condition in CONDITIONS
        if any(row["condition"] == condition for row in scoped)
    ]
    keys = sorted(
        {(row["n_fft"], row["training_seed"], row["condition"]) for row in scoped},
        key=lambda item: (item[0], item[1], present_conditions.index(item[2])),
    )
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in scoped:
        grouped.setdefault((row["n_fft"], row["training_seed"], row["condition"]), []).append(row)

    per_file_rows: list[dict[str, Any]] = []
    model_condition_rows: list[dict[str, Any]] = []
    composites: dict[tuple[int, int], dict[str, float]] = {}
    for n_fft, seed, condition in keys:
        rows = grouped[(n_fft, seed, condition)]
        base = {
            "split": split,
            "n_fft": n_fft,
            "hop_length": hop_for(n_fft),
            "training_seed": seed,
            "epoch": rows[0]["epoch"],
            "condition": condition,
            "model_id": f"nfft{n_fft}_seed{seed}",
        }
        model_condition_rows.append(
            metric_row(base, rows, require_both_classes=require_both_classes)
        )
        for filename in sorted({row["filename"] for row in rows}):
            file_rows = [row for row in rows if row["filename"] == filename]
            label = int(file_rows[0]["label"])
            correct = sum(int(row["prediction"]) == label for row in file_rows)
            per_file_rows.append(
                {
                    **base,
                    "filename": filename,
                    "target_code": file_rows[0]["target_code"],
                    "target_name": file_rows[0]["target_name"],
                    "segment_code": file_rows[0]["segment_code"],
                    "class_label": label,
                    "class_name": LABEL_TO_NAME[label],
                    "windows": len(file_rows),
                    "correct": correct,
                    "errors": len(file_rows) - correct,
                    "correct_rate": correct / len(file_rows),
                    "error_rate": (len(file_rows) - correct) / len(file_rows),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    metric_by_key = {
        (row["n_fft"], row["training_seed"], row["condition"]): row for row in model_condition_rows
    }
    present_nfft = sorted({n_fft for n_fft, _, _ in metric_by_key})
    present_seeds = sorted({seed for _, seed, _ in metric_by_key})
    for n_fft in present_nfft:
        for seed in present_seeds:
            if (n_fft, seed, present_conditions[0]) not in metric_by_key:
                continue
            rows = [metric_by_key[(n_fft, seed, condition)] for condition in present_conditions]
            composites[(n_fft, seed)] = {
                "file_equal_balanced_accuracy": mean_or_zero(
                    [row["file_equal_balanced_accuracy"] for row in rows]
                ),
                "balanced_accuracy": mean_or_zero([row["balanced_accuracy"] for row in rows]),
                "file_equal_phantom_recall": mean_or_zero(
                    [row["file_equal_phantom_recall"] for row in rows]
                ),
                "file_equal_air2s_recall": mean_or_zero(
                    [row["file_equal_air2s_recall"] for row in rows]
                ),
            }
    for n_fft in present_nfft:
        for condition in present_conditions:
            seeds = [seed for seed in present_seeds if (n_fft, seed, condition) in metric_by_key]
            if not seeds:
                continue
            values = [
                metric_by_key[(n_fft, seed, condition)]["file_equal_balanced_accuracy"]
                for seed in seeds
            ]
            pooled = [
                metric_by_key[(n_fft, seed, condition)]["balanced_accuracy"] for seed in seeds
            ]
            summary_rows.append(
                {
                    "split": split,
                    "n_fft": n_fft,
                    "hop_length": hop_for(n_fft),
                    "condition": condition,
                    "seeds": len(seeds),
                    "file_equal_balacc_mean": mean_or_zero(values),
                    "file_equal_balacc_std": spread(values),
                    "file_equal_balacc_min": float(np.min(values)),
                    "file_equal_balacc_max": float(np.max(values)),
                    "pooled_balacc_mean": mean_or_zero(pooled),
                    "pooled_balacc_std": spread(pooled),
                    "per_seed_file_equal_balacc": json.dumps(
                        {str(seed): value for seed, value in zip(seeds, values)}
                    ),
                }
            )
        composite_seeds = sorted(seed for seed in present_seeds if (n_fft, seed) in composites)
        values = [composites[(n_fft, seed)]["file_equal_balanced_accuracy"] for seed in composite_seeds]
        pooled = [composites[(n_fft, seed)]["balanced_accuracy"] for seed in composite_seeds]
        composite_label = (
            "four_condition_mean" if len(present_conditions) == len(CONDITIONS) else "condition_mean"
        )
        summary_rows.append(
            {
                "split": split,
                "n_fft": n_fft,
                "hop_length": hop_for(n_fft),
                "condition": composite_label,
                "seeds": len(composite_seeds),
                "file_equal_balacc_mean": mean_or_zero(values),
                "file_equal_balacc_std": spread(values),
                "file_equal_balacc_min": float(np.min(values)),
                "file_equal_balacc_max": float(np.max(values)),
                "pooled_balacc_mean": mean_or_zero(pooled),
                "pooled_balacc_std": spread(pooled),
                "per_seed_file_equal_balacc": json.dumps(
                    {str(seed): value for seed, value in zip(composite_seeds, values)}
                ),
            }
        )
    for n_fft in present_nfft:
        if n_fft == BASELINE_NFFT or BASELINE_NFFT not in present_nfft:
            continue
        for condition in [*present_conditions, composite_label]:
            differences = []
            pooled_differences = []
            for seed in present_seeds:
                if condition == composite_label:
                    if (n_fft, seed) not in composites or (BASELINE_NFFT, seed) not in composites:
                        continue
                    value = composites[(n_fft, seed)]["file_equal_balanced_accuracy"]
                    baseline = composites[(BASELINE_NFFT, seed)]["file_equal_balanced_accuracy"]
                    value_pooled = composites[(n_fft, seed)]["balanced_accuracy"]
                    baseline_pooled = composites[(BASELINE_NFFT, seed)]["balanced_accuracy"]
                else:
                    if (
                        (n_fft, seed, condition) not in metric_by_key
                        or (BASELINE_NFFT, seed, condition) not in metric_by_key
                    ):
                        continue
                    value = metric_by_key[(n_fft, seed, condition)]["file_equal_balanced_accuracy"]
                    baseline = metric_by_key[(BASELINE_NFFT, seed, condition)][
                        "file_equal_balanced_accuracy"
                    ]
                    value_pooled = metric_by_key[(n_fft, seed, condition)]["balanced_accuracy"]
                    baseline_pooled = metric_by_key[(BASELINE_NFFT, seed, condition)][
                        "balanced_accuracy"
                    ]
                differences.append(
                    {
                        "training_seed": seed,
                        "value": value,
                        "baseline": baseline,
                        "difference": value - baseline,
                        "pooled_value": value_pooled,
                        "pooled_baseline": baseline_pooled,
                        "pooled_difference": value_pooled - baseline_pooled,
                    }
                )
                pooled_differences.append(value_pooled - baseline_pooled)
            if not differences:
                continue
            deltas = [row["difference"] for row in differences]
            paired_rows.append(
                {
                    "split": split,
                    "n_fft": n_fft,
                    "baseline_n_fft": BASELINE_NFFT,
                    "condition": condition,
                    "seeds": len(differences),
                    "mean_difference": mean_or_zero(deltas),
                    "min_difference": float(np.min(deltas)),
                    "max_difference": float(np.max(deltas)),
                    "range_difference": float(np.max(deltas) - np.min(deltas)),
                    "std_difference": spread(deltas),
                    "seeds_positive": int(sum(1 for value in deltas if value > 0)),
                    "seeds_negative": int(sum(1 for value in deltas if value < 0)),
                    "seeds_zero": int(sum(1 for value in deltas if value == 0)),
                    "all_seeds_same_direction": all(value > 0 for value in deltas)
                    or all(value < 0 for value in deltas),
                    "per_seed": json.dumps(
                        {
                            str(row["training_seed"]): {
                                "value": row["value"],
                                "baseline": row["baseline"],
                                "difference": row["difference"],
                                "pooled_difference": row["pooled_difference"],
                            }
                            for row in differences
                        }
                    ),
                    "mean_pooled_difference": mean_or_zero(pooled_differences),
                }
            )
    return {
        "model_condition_metrics": model_condition_rows,
        "per_file_metrics": per_file_rows,
        "nfft_summary": summary_rows,
        "paired_nfft_comparison": paired_rows,
    }


def command_summarize(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    cache_manifest_path = root / "prediction_cache_manifest.json"
    truncated = bool(
        read_json(cache_manifest_path).get("truncated_debug_run")
        if cache_manifest_path.is_file()
        else False
    )
    heldout = [cast_prediction_row(row) for row in read_csv_rows(root / "heldout_window_predictions.csv")]
    tables = summarize_predictions(
        heldout, "heldout", output_root=root, require_both_classes=not truncated
    )
    write_csv(root / "model_condition_metrics.csv", tables["model_condition_metrics"])
    write_csv(root / "per_file_metrics.csv", tables["per_file_metrics"])
    write_csv(root / "nfft_summary.csv", tables["nfft_summary"])
    write_csv(root / "paired_nfft_comparison.csv", tables["paired_nfft_comparison"])
    test = [cast_prediction_row(row) for row in read_csv_rows(root / "test_clean_window_predictions.csv")]
    test_tables = summarize_predictions(
        test, "test", output_root=root, require_both_classes=not truncated
    )
    write_csv(root / "test_clean_model_condition_metrics.csv", test_tables["model_condition_metrics"])
    log(
        f"Wrote model_condition_metrics.csv ({len(tables['model_condition_metrics'])}), "
        f"per_file_metrics.csv ({len(tables['per_file_metrics'])}), "
        f"nfft_summary.csv ({len(tables['nfft_summary'])}), "
        f"paired_nfft_comparison.csv ({len(tables['paired_nfft_comparison'])}), "
        f"test_clean_model_condition_metrics.csv ({len(test_tables['model_condition_metrics'])})"
    )
    return 0


# ---------------------------------------------------------------------------
# manifest and report
# ---------------------------------------------------------------------------
def write_experiment_manifest(
    root: Path, extra_commands: dict[str, Any] | None = None
) -> dict[str, Any]:
    grid = experiment_grid(root, required=False)
    runs = []
    for entry in grid:
        config = entry["config"]
        run_dir = entry["run_directory"]
        runs.append(
            {
                "n_fft": entry["n_fft"],
                "hop_length": entry["hop_length"],
                "hop_ratio": entry["hop_length"] / entry["n_fft"],
                "training_seed": entry["training_seed"],
                "selected_epoch": entry["epoch"],
                "status": config.get("status"),
                "checkpoint": str(run_dir / "best_model.pt"),
                "checkpoint_sha256": sha256_file(run_dir / "best_model.pt"),
                "initial_model_sha256": config.get("initial_model_sha256"),
                "epoch_order_sha256": config.get("epoch_order_sha256"),
                "train_snrs": config.get("train_snrs"),
                "validation_seed": config.get("validation_seed"),
                "validation_conditions": config.get("validation_conditions"),
                "optimizer": config.get("optimizer"),
                "amp": config.get("amp"),
                "batch_size": config.get("batch_size"),
                "image_size": config.get("image_size"),
                "run_config": str(run_dir / "run_config.json"),
                "training_history": str(run_dir / "training_history.csv"),
                "validation_metrics": str(run_dir / "validation_metrics.csv"),
                "selection": str(run_dir / "selection.json"),
            }
        )
    published = {}
    for name in (
        "validation_nfft_selection.json",
        "selection_lock.json",
        "heldout_window_predictions.csv",
        "test_clean_window_predictions.csv",
        "model_condition_metrics.csv",
        "test_clean_model_condition_metrics.csv",
        "per_file_metrics.csv",
        "nfft_summary.csv",
        "paired_nfft_comparison.csv",
        "representative_phantom.png",
        "representative_air2s.png",
        "figure_evidence.json",
        "prediction_cache_manifest.json",
        "heldout_generation_record.json",
        "frontend_check.json",
        "frozen_inputs_verification.json",
        "frozen_inputs_verification_after.json",
        "report.md",
        "verification.json",
    ):
        path = root / name
        if path.is_file():
            published[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    selection_path = root / "validation_nfft_selection.json"
    selection = read_json(selection_path) if selection_path.is_file() else None
    manifest = {
        "task": "STAGE2-BASE: oracle machine-type classification (Phantom 4 Pro vs Air 2S)",
        "created_at": utc_now(),
        "output_root": str(root),
        "scope": {
            "input": "drone windows only (stage-1 target codes 0010 and 0101)",
            "label_mapping": {str(label): LABEL_TO_NAME[label] for label in (0, 1)},
            "excluded": "background windows; no first-stage prediction is called",
            "not_a_cascade": True,
            "three_class_performance_claimed": False,
        },
        "design": {
            "n_fft_grid": list(NFFT_GRID),
            "hop_rule": "hop_length = n_fft // 2 (exactly 50%)",
            "image_size": IMAGE_SIZE,
            "training_seeds": list(TRAINING_SEEDS),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": {
                "type": "Adam",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
            "amp": True,
            "amp_dtype": "float16",
            "train_snrs": TRAIN_SNRS,
            "validation_conditions": list(CONDITIONS),
            "validation_seed": VALIDATION_SEED,
            "epoch_selection_rule": EPOCH_SELECTION_RULE,
            "epoch_tie_break": TIE_BREAK,
            "nfft_selection_rule": NFFT_SELECTION_RULE,
            "nfft_tie_break": NFFT_TIE_BREAK,
            "nfft_selection_evidence": "old validation split only",
            "prediction_rule": PREDICTION_RULE,
            "primary_aggregation": (
                "per-source-file correct rate averaged inside each machine class, then the two "
                "classes weighted equally (file-equal balanced accuracy)"
            ),
            "secondary_aggregation": "pooled over all windows, confusion counts retained",
            "heldout_noise_seed": HELDOUT_NOISE_SEED,
            "test_noise_seed": TEST_NOISE_SEED,
            "heldout_rng_rule": (
                "np.random.default_rng(noise_seed + epoch*len(stage2_split) + stage2_position), "
                "epoch pinned to 0"
            ),
            "heldout_windows": EXPECTED_SPLIT_WINDOWS["heldout"],
            "test_windows": EXPECTED_SPLIT_WINDOWS["test"],
            "expected_heldout_rows": EXPECTED_HELDOUT_ROWS,
            "expected_test_rows": EXPECTED_TEST_ROWS,
        },
        "fairness": {
            "same_initial_weights_within_seed": (
                "initial_model_sha256 is recorded per run and must be identical across the two "
                "n_fft values of one training seed; the paired unit test asserts it"
            ),
            "same_epoch_orders_within_seed": (
                "epoch_order_sha256 is derived from torch.randperm(620, generator=seed+epoch) and "
                "must be identical across the two n_fft values of one seed"
            ),
            "same_validation_inputs": (
                "one float32 validation cache per n_fft, reused by that grid's three seeds"
            ),
            "paired_noise_across_grids": (
                "the same split/condition noise is a property of the stage-2 position; the input "
                "probe digest of a (split, condition) must be identical for both n_fft values"
            ),
            "test_and_heldout_not_used_for_selection": True,
        },
        "feature_cache": {
            "location": "in-memory float32 CPU array per (split, condition, n_fft)",
            "disk_feature_cache_written": False,
            "window_digest_files": "feature_digests/window_digests_<split>_<condition>_<nfft>.bin",
            "manifest": str(root / "prediction_cache_manifest.json"),
            "per_window_png_written": False,
        },
        "selection": {
            "file": str(selection_path),
            "sha256": sha256_file(selection_path) if selection_path.is_file() else None,
            "selected_n_fft": selection["selected_n_fft"] if selection else None,
            "locked_by": str(root / "selection_lock.json"),
            "heldout_generation_record": str(root / "heldout_generation_record.json"),
        },
        "runs": runs,
        "runs_completed": sum(1 for entry in runs if entry["status"] == "completed"),
        "runs_expected": len(NFFT_GRID) * len(TRAINING_SEEDS),
        "published_files": published,
        "source_sha256": {
            name: sha256_file(PROJECT_DIR / name)
            for name in (
                "run_stage2_classification.py",
                "test_stage2_classification.py",
                "verify_stage2_classification.py",
                "dronerfa_data.py",
                "model.py",
                "train_resnet18.py",
            )
        },
        "frozen_inputs": {
            "old_index": str(OLD_INDEX),
            "old_index_sha256": OLD_INDEX_SHA256,
            "heldout_index": str(HELDOUT_INDEX),
            "heldout_index_sha256": HELDOUT_INDEX_SHA256,
            "reference": str(PRESERVED_INPUTS),
        },
        "extra_commands": extra_commands or {},
        "writes_or_modifies_original_data": False,
    }
    write_json(root / "experiment_manifest.json", manifest)
    return manifest


def command_manifest(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    manifest = write_experiment_manifest(root)
    log(
        f"Wrote experiment_manifest.json: {manifest['runs_completed']}/{manifest['runs_expected']} runs complete"
    )
    return 0


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def command_report(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    manifest = read_json(root / "experiment_manifest.json")
    selection = read_json(root / "validation_nfft_selection.json")
    summary = read_csv_rows(root / "nfft_summary.csv")
    metrics = read_csv_rows(root / "model_condition_metrics.csv")
    per_file = read_csv_rows(root / "per_file_metrics.csv")
    paired = read_csv_rows(root / "paired_nfft_comparison.csv")
    test_metrics = read_csv_rows(root / "test_clean_model_condition_metrics.csv")

    def four_condition_row(n_fft: int) -> dict[str, str]:
        return next(
            row
            for row in summary
            if int(row["n_fft"]) == n_fft and row["condition"] == "four_condition_mean"
        )

    lines: list[str] = []
    lines.append("# STAGE2-BASE：已知输入为无人机时的机型分类（Phantom 4 Pro vs Air 2S）")
    lines.append("")
    lines.append(
        "本报告是 **oracle 第二阶段结果**：所有输入窗口的真实标签都是无人机，"
        "模型只回答“这是 Phantom 4 Pro 还是 Air 2S”。"
    )
    lines.append("")
    lines.append("## 1. 结论摘要（oracle 第二阶段）")
    lines.append("")
    lines.append(
        "主口径为文件等权平衡准确率：同一机型内先按源文件求正确率并等权平均，"
        "再对两个机型等权平均（file-equal balanced accuracy）。括号内为 pooled 窗口口径。"
    )
    lines.append("")
    lines.append("| n_fft | hop | 三 seed 四条件均值 (file-equal BA) | 标准差 | pooled 均值 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for n_fft in NFFT_GRID:
        row = four_condition_row(n_fft)
        lines.append(
            f"| {n_fft} | {row['hop_length']} | {pct(float(row['file_equal_balacc_mean']))} | "
            f"{100 * float(row['file_equal_balacc_std']):.2f}pp | {pct(float(row['pooled_balacc_mean']))} |"
        )
    lines.append("")
    lines.append(
        f"选中的配置为 **n_fft={selection['selected_n_fft']}**，"
        f"该选择只使用旧验证集（见 `validation_nfft_selection.json`），"
        "不读取旧 test 或新留出结果；差值 "
        f"{100 * float(selection['selection_margin']):.2f}pp。"
    )
    if selection.get("exact_tie"):
        lines.append("两档完全并列，按冻结规则取计算量较小的 512。")
    lines.append("")
    lines.append("## 2. 任务与口径边界")
    lines.append("")
    lines.append("- 背景窗口不属于本任务：本阶段输入只有真实标签为无人机的窗口。")
    lines.append(
        "- 本结果**不是**级联系统的三分类性能：没有接入第一阶段预测，"
        "也没有评估“背景 / 无人机 / 机型”的联合指标。"
    )
    lines.append(
        "- 旧训练集每类只有 2 个源文件，旧验证/旧 test 每类各 1 个源文件，"
        "新留出集每类 4 个源文件。"
    )
    lines.append(
        "- 训练/验证中的噪声是人工复 AWGN，不等于现场绝对 SNR；"
        "因此**不能**声称跨会话、跨场景或跨设备泛化。"
    )
    lines.append("")
    lines.append("## 3. 数据与标签映射")
    lines.append("")
    lines.append("| label | 机型 | 原 target_code | train | val | test | heldout |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for label, code in ((0, PHANTOM_CODE), (1, AIR2S_CODE)):
        counts = [EXPECTED_CLASS_WINDOWS[split][label] for split in ("train", "val", "test", "heldout")]
        lines.append(
            f"| {label} | {LABEL_TO_NAME[label]} | {code} | "
            + " | ".join(str(value) for value in counts)
            + " |"
        )
    lines.append("")
    lines.append(
        "stage2 index position 在每个 split 内按冻结索引原顺序重新从 0 编号；"
        "相同 split/condition 的噪声只由 (split, condition, stage2 position) 决定，"
        "因此在两个 n_fft 和三个训练种子之间逐窗配对。"
    )
    lines.append("")
    lines.append("## 4. 训练与选择")
    lines.append("")
    lines.append(
        f"- 从头训练 ResNet18（`input_channels=1, num_classes=2`），"
        f"{EPOCHS} 轮、batch {BATCH_SIZE}、Adam lr {LEARNING_RATE}、weight_decay {WEIGHT_DECAY}、"
        "CUDA AMP float16（前端 float32）。"
    )
    lines.append(
        f"- 训练噪声 A_original：{TRAIN_SNRS} 等概率；验证条件 {', '.join(CONDITIONS)}，"
        f"validation noise seed {VALIDATION_SEED}。"
    )
    lines.append(
        "- 同一 seed 的两档共享同一初始 ResNet18 权重哈希与同一 epoch 窗口顺序（见各 run_config.json）。"
    )
    lines.append(
        f"- 选 epoch 规则：四档验证文件等权 BA 的算术平均最高，完全并列取最早 epoch"
        f"（{selection['epoch_rule']}）。"
    )
    lines.append("")
    lines.append("| n_fft | seed | 选中 epoch | 四条件 file-equal BA |")
    lines.append("|---:|---:|---:|---:|")
    for row in selection["per_model"]:
        lines.append(
            f"| {row['n_fft']} | {row['training_seed']} | {row['selected_epoch']} | "
            f"{pct(row['four_condition_mean_file_equal_balacc'])} |"
        )
    lines.append("")
    lines.append("## 5. 新留出集四条件结果（主口径）")
    lines.append("")
    lines.append("| n_fft | seed | condition | file-equal BA | pooled BA | Phantom recall | Air2S recall | F1 (Air2S) | TN/FP/FN/TP |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---|")
    for row in sorted(
        metrics,
        key=lambda item: (int(item["n_fft"]), int(item["training_seed"]), CONDITIONS.index(item["condition"])),
    ):
        lines.append(
            f"| {row['n_fft']} | {row['training_seed']} | {row['condition']} | "
            f"{pct(float(row['file_equal_balanced_accuracy']))} | {pct(float(row['balanced_accuracy']))} | "
            f"{pct(float(row['file_equal_phantom_recall']))} | {pct(float(row['file_equal_air2s_recall']))} | "
            f"{pct(float(row['f1_air2s_binary']))} | "
            f"{row['tn']}/{row['fp']}/{row['fn']}/{row['tp']} |"
        )
    lines.append("")
    lines.append("### 5.1 跨 seed 汇总")
    lines.append("")
    lines.append("| n_fft | condition | seeds | file-equal BA 均值 | 标准差 | pooled 均值 |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for row in sorted(
        summary,
        key=lambda item: (
            int(item["n_fft"]),
            CONDITIONS.index(item["condition"]) if item["condition"] in CONDITIONS else 99,
        ),
    ):
        lines.append(
            f"| {row['n_fft']} | {row['condition']} | {row['seeds']} | "
            f"{pct(float(row['file_equal_balacc_mean']))} | "
            f"{100 * float(row['file_equal_balacc_std']):.2f}pp | "
            f"{pct(float(row['pooled_balacc_mean']))} |"
        )
    lines.append("")
    lines.append("### 5.2 同 seed 的 512 相对 1024 配对差值")
    lines.append("")
    lines.append("| condition | seeds | 平均差值 | 最小 | 最大 | 同向 | 逐 seed 差值 |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for row in paired:
        lines.append(
            f"| {row['condition']} | {row['seeds']} | {100 * float(row['mean_difference']):+.2f}pp | "
            f"{100 * float(row['min_difference']):+.2f}pp | {100 * float(row['max_difference']):+.2f}pp | "
            f"{'是' if row['all_seeds_same_direction'] == 'True' else '否'} | "
            f"{json.dumps(json.loads(row['per_seed']), ensure_ascii=False)} |"
        )
    lines.append("")
    lines.append("n=3 个 seed，窗口与噪声条件相关，差值只作描述性结果，不做显著性结论。")
    lines.append("")
    lines.append("## 6. 逐源文件结果")
    lines.append("")
    lines.append("| n_fft | seed | condition | 文件 | 机型 | 窗口 | 正确率 |")
    lines.append("|---:|---:|---|---|---|---:|---:|")
    for row in per_file:
        lines.append(
            f"| {row['n_fft']} | {row['training_seed']} | {row['condition']} | {row['filename']} | "
            f"{row['class_name']} | {row['windows']} | {pct(float(row['correct_rate']))} |"
        )
    lines.append("")
    lines.append("## 7. 旧 test clean（附加）")
    lines.append("")
    lines.append("| n_fft | seed | file-equal BA | pooled BA | TN/FP/FN/TP |")
    lines.append("|---:|---:|---:|---:|---|")
    for row in sorted(test_metrics, key=lambda item: (int(item["n_fft"]), int(item["training_seed"]))):
        lines.append(
            f"| {row['n_fft']} | {row['training_seed']} | "
            f"{pct(float(row['file_equal_balanced_accuracy']))} | {pct(float(row['balanced_accuracy']))} | "
            f"{row['tn']}/{row['fp']}/{row['fn']}/{row['tp']} |"
        )
    lines.append("")
    lines.append("## 8. 产物与复核")
    lines.append("")
    lines.append("- 配置选择：`validation_nfft_selection.json`（只含旧验证证据）与锁定记录 `selection_lock.json`。")
    lines.append("- 生成留出结果前后，选择文件哈希被重新核对，见 `heldout_generation_record.json`。")
    lines.append(
        "- 原始窗口预测：`heldout_window_predictions.csv`（30,240 行）、"
        "`test_clean_window_predictions.csv`（1,800 行）。"
    )
    lines.append("- 独立复算：`verification.json`；单元/有界 smoke 测试见 `test_stage2_classification.py`。")
    lines.append(
        "- 代表窗图：`representative_phantom.png`、`representative_air2s.png`"
        "（同一固定窗的 512/1024 原生 STFT 与实际 1x256x256 模型输入，不保存逐窗 PNG）。"
    )
    lines.append("")
    lines.append("## 9. 不能声称的内容")
    lines.append("")
    lines.append("- 不是级联三分类性能，也没有使用第一阶段输出。")
    lines.append("- 旧训练集每类仅 2 个源文件；新留出每类 4 个文件，仍不足以证明跨会话泛化。")
    lines.append("- 模拟 AWGN 档位不是现场绝对 SNR。")
    lines.append("- n=3 个训练种子的差值不做统计显著性结论。")
    lines.append("")
    report_path = root / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Wrote {report_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=(
            "verify-inputs",
            "frontends",
            "train",
            "select-nfft",
            "figures",
            "evaluate",
            "summarize",
            "report",
            "manifest",
            "all",
        ),
    )
    parser.add_argument("--n-fft", type=int, choices=NFFT_GRID)
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default="cuda", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--stage",
        default="before",
        choices=("before", "after"),
        help="which frozen-input record `verify-inputs` writes",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        help="Smoke only: stratify the training split down to this many windows.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        help="Smoke only: stratify the validation split down to this many windows.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        help="Smoke only: truncate the evaluated split and skip full-size assertions.",
    )
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--smoke", action="store_true", help="Mark outputs as a smoke run.")
    return parser.parse_args(argv)


def command_verify_inputs(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    integrity = verify_frozen_inputs()
    name = (
        "frozen_inputs_verification.json"
        if args.stage == "before"
        else "frozen_inputs_verification_after.json"
    )
    write_json(root / name, integrity)
    if integrity["errors"]:
        for error in integrity["errors"]:
            log(f"REJECTED: {error}")
        return 2
    log(f"Frozen inputs verified ({args.stage}); wrote {name}")
    return 0


def command_frontends(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    report = check_frontends(device)
    write_json(root / "frontend_check.json", report)
    for key, value in report.items():
        log(f"  nfft{key}: shape={value['output_shape']} finite={value['finite']}")
    return 0


def command_train(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    targets = [
        (n_fft, seed)
        for n_fft in ([args.n_fft] if args.n_fft else NFFT_GRID)
        for seed in ([args.seed] if args.seed else TRAINING_SEEDS)
    ]
    for n_fft, seed in targets:
        run_dir = run_directory(n_fft, seed, root)
        config_path = run_dir / "run_config.json"
        if config_path.is_file():
            status = read_json(config_path).get("status")
            if status == "completed":
                log(f"skip completed run {run_dir} (status={status})")
                continue
            raise RuntimeError(
                f"{run_dir} holds an unfinished run (status={status!r}); remove the directory "
                f"to retrain it"
            )
        if run_dir.exists():
            raise RuntimeError(
                f"{run_dir} exists without run_config.json; remove it before retraining"
            )
        train_run(
            n_fft,
            seed,
            device,
            epochs=args.epochs,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            output_root=root,
            smoke=args.smoke,
        )
    return 0


def command_figures(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    device = resolve_device(args.device)
    evidence = write_figures(device, output_root=root)
    for code, entry in evidence["windows"].items():
        log(f"  class {code}: {entry['filename']} -> {entry['figure']}")
    return 0


def command_all(args: argparse.Namespace) -> int:
    for function in (
        command_verify_inputs,
        command_frontends,
        command_train,
        command_select_nfft,
        command_figures,
        command_evaluate,
        command_summarize,
        command_manifest,
        command_report,
    ):
        code = function(args)
        if code:
            return code
    after = argparse.Namespace(**{**vars(args), "stage": "after"})
    return command_verify_inputs(after)


COMMANDS = {
    "verify-inputs": command_verify_inputs,
    "frontends": command_frontends,
    "train": command_train,
    "select-nfft": command_select_nfft,
    "figures": command_figures,
    "evaluate": command_evaluate,
    "summarize": command_summarize,
    "report": command_report,
    "manifest": command_manifest,
    "all": command_all,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
