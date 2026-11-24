"""CASCADE3-NEWSPLIT-B: retrain the frozen two-stage baseline on the new split.

Frozen protocol (identical to the accepted NFFT4 / STAGE2-BASE / CASCADE3 input
and noise handling; only the source-file split is new):

* RF0 I/Q, non-overlapping 1,000,000-sample windows read at the row ``start``;
* per-window mean removal + RMS normalisation (``dronerfa_data.rms_normalize``);
* training noise ``A_original``: complex AWGN at
  ``clean,10,5,0,-5,-10,-12`` dB drawn per window with the frozen rule
  ``default_rng(seed + epoch * len(split) + position)``;
* Hamming STFT ``n_fft=512``, ``hop=256``, 256x256 log-power image
  (``model.SpectrogramFrontEnd``, unchanged);
* ``model.ResNet18(input_channels=1, num_classes=2)``, Adam lr=1e-3, wd=0,
  batch size 4, class-weighted cross entropy, AMP float16, seed 2026, 5 epochs;
* Stage 1 = Background (T0000) vs Drone (T0010/T0101);
  Stage 2 = DJI Phantom 4 Pro (T0010) vs DJI Air 2S (T0101);

Epoch selection uses only the new validation split under the four frozen
validation conditions (clean/-10/-12/-15 dB, validation noise seed 12026), with
the earliest epoch winning ties. The ``development_test_exposed`` split is never
read by this script and its file paths are asserted absent from the read audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from collections import Counter
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
    parse_snr_list,
    stratified_limit,
)
from model import ResNet18, SpectrogramFrontEnd
from train_resnet18 import extract_features, seed_everything, train_epoch

# ---------------------------------------------------------------------------
# frozen inputs of CASCADE3-NEWSPLIT-A / META-SPLIT2
# ---------------------------------------------------------------------------
NEWSPLIT_ROOT = PROJECT_DIR / "outputs" / "cascade3_newsplit_20260926"
INDEX = NEWSPLIT_ROOT / "window_index_newsplit3.csv"
INDEX_SHA256 = "deed870f9693c5ebe93fb80a9647b452e90680dd5b1f21372e108ded4bf4d231"
STAGE2_VIEW = NEWSPLIT_ROOT / "stage2_view_newsplit3.csv"
STAGE2_VIEW_SHA256 = "76001e671cf3ebe492b191af87a85b5368caed85e03c84c4c249ccc5081642de"

WORK_DIR = PROJECT_DIR.parent / "work"
FREEZE_JSON = WORK_DIR / "meta_split2" / "CODEX-FREEZE-20260926.json"
FROZEN_SPLIT_CSV = WORK_DIR / "meta_split2" / "split_candidate_104_20260926.csv"
FROZEN_SPLIT_CSV_SHA256 = (
    "d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9"
)
G_MANIFEST = Path(r"G:\DroneRFa数据集\dataset_manifest.csv")
G_MANIFEST_SHA256 = "9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d"
DATA_ROOT = Path(r"G:\DroneRFa数据集\original")

RUN_ROOT = NEWSPLIT_ROOT / "seed_2026"
SMOKE_ROOT = RUN_ROOT / "smoke"

# ---------------------------------------------------------------------------
# frozen training protocol
# ---------------------------------------------------------------------------
N_FFT = 512
HOP_LENGTH = 256
IMAGE_SIZE = 256
EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
TRAINING_SEED = 2026
TRAIN_SNRS = "clean,10,5,0,-5,-10,-12"
VALIDATION_SEED = 12026
CONDITION_VALUES: dict[str, float | None] = {
    "clean": None,
    "-10dB": -10.0,
    "-12dB": -12.0,
    "-15dB": -15.0,
}
CONDITIONS: tuple[str, ...] = ("clean", "-10dB", "-12dB", "-15dB")
TIE_BREAK = "earliest_epoch"
STAGE1_PREDICTION_RULE = "float32(logit_drone) - float32(logit_background) > 0"
STAGE2_PREDICTION_RULE = "float32(logit_air2s) - float32(logit_phantom) > 0"
STAGE1_SELECTION_RULE = "mean_pooled_balanced_accuracy_over_four_validation_conditions"
STAGE2_SELECTION_RULE = "mean_file_equal_balanced_accuracy_over_four_validation_conditions"

TRAIN_ROLE = "train"
VAL_ROLE = "val"
TEST_ROLE = "development_test_exposed"
EXPECTED_ROLES = (TRAIN_ROLE, VAL_ROLE, TEST_ROLE)

BACKGROUND_CODE = "T0000"
PHANTOM_CODE = "T0010"
AIR2S_CODE = "T0101"
STAGE1_DRONE_LABEL = 1
STAGE2_CODE_TO_LABEL: dict[str, int] = {PHANTOM_CODE: 0, AIR2S_CODE: 1}
STAGE2_LABEL_TO_NAME = {0: "DJI Phantom 4 Pro", 1: "DJI Air 2S"}
WINDOW_SAMPLES = 1_000_000
RELOAD_TOLERANCE = 1e-9
SOURCE_FILES = ("run_cascade3_newsplit_training.py", "dronerfa_data.py", "model.py", "train_resnet18.py")

# Expected composition of the frozen index (guards against a swapped input).
EXPECTED_TOTALS: dict[str, Any] = {
    "windows": {TRAIN_ROLE: 1810, VAL_ROLE: 930, TEST_ROLE: 890},
    "files": {TRAIN_ROLE: 12, VAL_ROLE: 6, TEST_ROLE: 6},
    "stage2_windows": {TRAIN_ROLE: 1210, VAL_ROLE: 650, TEST_ROLE: 610},
    "stage1_label_counts": {
        TRAIN_ROLE: {0: 600, 1: 1210},
        VAL_ROLE: {0: 280, 1: 650},
        TEST_ROLE: {0: 280, 1: 610},
    },
    "target_counts": {
        TRAIN_ROLE: {BACKGROUND_CODE: 600, PHANTOM_CODE: 590, AIR2S_CODE: 620},
        VAL_ROLE: {BACKGROUND_CODE: 280, PHANTOM_CODE: 310, AIR2S_CODE: 340},
        TEST_ROLE: {BACKGROUND_CODE: 280, PHANTOM_CODE: 300, AIR2S_CODE: 310},
    },
}


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
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


def state_hash(state: dict[str, torch.Tensor]) -> str:
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


# ---------------------------------------------------------------------------
# frozen input identity
# ---------------------------------------------------------------------------
def frozen_role_map() -> dict[str, str]:
    """Map every frozen source MAT name to its role in the meta-split freeze."""
    if not FREEZE_JSON.is_file():
        raise FileNotFoundError(f"missing freeze decision: {FREEZE_JSON}")
    freeze = read_json(FREEZE_JSON)
    if freeze.get("frozen_split_csv") != "work/meta_split2/split_candidate_104_20260926.csv":
        raise RuntimeError(f"unexpected frozen split path in {FREEZE_JSON}")
    if freeze.get("frozen_split_csv_sha256") != FROZEN_SPLIT_CSV_SHA256:
        raise RuntimeError("freeze decision records an unexpected frozen split hash")
    observed = sha256_file(FROZEN_SPLIT_CSV)
    if observed != FROZEN_SPLIT_CSV_SHA256:
        raise RuntimeError(
            f"frozen split CSV hash mismatch: {observed} != {FROZEN_SPLIT_CSV_SHA256}"
        )
    roles: dict[str, str] = {}
    for row in read_csv_rows(FROZEN_SPLIT_CSV):
        filename = str(row["file_name"])
        role = str(row["role"])
        if filename in roles and roles[filename] != role:
            raise RuntimeError(f"{filename} carries two roles in the frozen split")
        roles[filename] = role
    return roles


def check_index_identity() -> dict[str, Any]:
    """Re-hash the frozen index artefacts and re-check every compositional claim."""
    errors: list[str] = []
    index_hash = sha256_file(INDEX)
    if index_hash != INDEX_SHA256:
        errors.append(f"index SHA256 mismatch: {index_hash}")
    view_hash = sha256_file(STAGE2_VIEW)
    if view_hash != STAGE2_VIEW_SHA256:
        errors.append(f"stage2 view SHA256 mismatch: {view_hash}")
    if G_MANIFEST.is_file():
        manifest_hash = sha256_file(G_MANIFEST)
        if manifest_hash != G_MANIFEST_SHA256:
            errors.append(f"G dataset manifest SHA256 mismatch: {manifest_hash}")
    else:
        manifest_hash = None
        errors.append(f"G dataset manifest not reachable: {G_MANIFEST}")

    roles = frozen_role_map()
    rows = read_csv_rows(INDEX)
    for row in rows:
        row["_window_index"] = int(row["window_index"])
        row["_start"] = int(row["start"])
        row["_length"] = int(row["length"])
        row["_label"] = int(row["label"])

    role_of_file: dict[str, str] = {}
    for row in rows:
        filename = str(row["filename"])
        role = str(row["source_role"])
        if str(row["split"]) != role:
            errors.append(f"{filename}: split != source_role")
            break
        previous = role_of_file.setdefault(filename, role)
        if previous != role:
            errors.append(f"{filename} appears under two roles in the index")
            break
        if roles.get(filename) != role:
            errors.append(
                f"{filename}: index role {role} != frozen role {roles.get(filename)}"
            )
            break
        if not str(row["file_path"]).lower().startswith(str(DATA_ROOT).lower()):
            errors.append(f"{filename}: file_path outside the G data root")
            break
        if row["_length"] != WINDOW_SAMPLES:
            errors.append(f"{filename}: window length {row['_length']}")
            break
        if row["_start"] != row["_window_index"] * WINDOW_SAMPLES:
            errors.append(
                f"{filename}: start {row['_start']} != window_index * {WINDOW_SAMPLES}"
            )
            break

    counts: dict[str, Counter] = {role: Counter() for role in EXPECTED_ROLES}
    files: dict[str, set[str]] = {role: set() for role in EXPECTED_ROLES}
    paths: dict[str, set[str]] = {role: set() for role in EXPECTED_ROLES}
    label_counts: dict[str, Counter] = {role: Counter() for role in EXPECTED_ROLES}
    target_counts: dict[str, Counter] = {role: Counter() for role in EXPECTED_ROLES}
    stage2_counts: Counter = Counter()
    for row in rows:
        role = str(row["source_role"])
        counts[role][1] += 1
        files[role].add(str(row["filename"]))
        paths[role].add(str(row["file_path"]))
        label_counts[role][int(row["label"])] += 1
        target_counts[role][str(row["target_code"])] += 1
        if str(row["target_code"]) in STAGE2_CODE_TO_LABEL:
            stage2_counts[role] += 1

    observed = {
        "windows": {role: counts[role][1] for role in EXPECTED_ROLES},
        "files": {role: len(files[role]) for role in EXPECTED_ROLES},
        "stage2_windows": {role: stage2_counts[role] for role in EXPECTED_ROLES},
        "stage1_label_counts": {
            role: dict(sorted(label_counts[role].items())) for role in EXPECTED_ROLES
        },
        "target_counts": {
            role: dict(sorted(target_counts[role].items())) for role in EXPECTED_ROLES
        },
    }
    for key, expected in EXPECTED_TOTALS.items():
        if observed[key] != expected:
            errors.append(f"{key} mismatch: {observed[key]} != {expected}")

    overlap = {
        "train_val": sorted(files[TRAIN_ROLE] & files[VAL_ROLE]),
        "train_test": sorted(files[TRAIN_ROLE] & files[TEST_ROLE]),
        "val_test": sorted(files[VAL_ROLE] & files[TEST_ROLE]),
    }
    if any(overlap.values()):
        errors.append(f"source files cross roles: {overlap}")

    # The stage-2 view must be exactly the drone projection of the frozen index.
    projection = stage2_projection(rows)
    view_rows = read_csv_rows(STAGE2_VIEW)
    if len(view_rows) != len(projection):
        errors.append(f"stage2 view has {len(view_rows)} rows, projection {len(projection)}")
    else:
        for expected_row, actual in zip(projection, view_rows):
            if (
                int(actual["stage2_position_in_split"]) != expected_row["position_in_split"]
                or int(actual["row_index"]) != expected_row["row_index"]
                or str(actual["filename"]) != expected_row["filename"]
                or int(actual["start"]) != expected_row["start"]
                or int(actual["stage2_label"]) != expected_row["stage2_label"]
            ):
                errors.append("stage2 view deviates from the index projection")
                break

    return {
        "index": {"path": str(INDEX), "sha256": index_hash, "rows": len(rows)},
        "stage2_view": {"path": str(STAGE2_VIEW), "sha256": view_hash, "rows": len(view_rows)},
        "g_manifest": {"path": str(G_MANIFEST), "sha256": manifest_hash},
        "frozen_split_csv": {"path": str(FROZEN_SPLIT_CSV), "sha256": FROZEN_SPLIT_CSV_SHA256},
        "files_by_role": {role: sorted(files[role]) for role in EXPECTED_ROLES},
        "paths_by_role": {role: sorted(paths[role]) for role in EXPECTED_ROLES},
        "observed": observed,
        "role_overlap": overlap,
        "errors": errors,
        "passed": not errors,
    }


def stage2_projection(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pure projection of the frozen index onto the two drone classes."""
    cursor: Counter = Counter()
    projected: list[dict[str, Any]] = []
    for row in rows:
        code = str(row["target_code"])
        if code not in STAGE2_CODE_TO_LABEL:
            continue
        role = str(row["source_role"])
        projected.append(
            {
                "role": role,
                "position_in_split": cursor[role],
                "row_index": int(row["row_index"]),
                "filename": str(row["filename"]),
                "start": int(row["start"]),
                "stage2_label": STAGE2_CODE_TO_LABEL[code],
            }
        )
        cursor[role] += 1
    return projected


def frozen_implementation_cross_check() -> dict[str, Any]:
    """Compare the local stage-2 view against the frozen STAGE2-BASE implementation.

    ``run_stage2_classification`` keys its two drone classes by the legacy
    ``0010``/``0101`` target codes, while the new index writes ``T0010``/``T0101``.
    The comparison therefore strips the leading ``T`` before handing the rows to
    the frozen function; everything else is passed through unchanged.
    """
    from run_stage2_classification import stage2_rows as frozen_stage2_rows

    rows = read_csv_rows(INDEX)
    for row in rows:
        row["label"] = int(row["label"])
    local = stage2_projection(rows)
    legacy_rows = [{**row, "target_code": str(row["target_code"])[1:]} for row in rows]
    reference: list[dict[str, Any]] = []
    for role in (TRAIN_ROLE, VAL_ROLE, TEST_ROLE):
        split_rows = [row for row in legacy_rows if str(row["source_role"]) == role]
        for position, row in enumerate(frozen_stage2_rows(split_rows)):
            reference.append(
                {
                    "role": role,
                    "position_in_split": position,
                    "row_index": int(row["row_index"]),
                    "filename": str(row["filename"]),
                    "start": int(row["start"]),
                    "stage2_label": int(row["label"]),
                }
            )
    local_map = {(row["role"], row["position_in_split"]): row for row in local}
    reference_map = {(row["role"], row["position_in_split"]): row for row in reference}
    mismatches = [
        {"key": key, "local": local_map.get(key), "frozen": reference_map.get(key)}
        for key in sorted(set(local_map) | set(reference_map))
        if local_map.get(key) != reference_map.get(key)
    ]
    return {
        "compared_rows": len(local),
        "frozen_rows": len(reference),
        "mismatches": mismatches[:5],
        "mismatch_count": len(mismatches),
        "passed": not mismatches,
    }


# ---------------------------------------------------------------------------
# read audit + datasets
# ---------------------------------------------------------------------------
class ReadAudit:
    """Records every HDF5 file this process opens and every window it reads."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.opened_files: set[str] = set()
        self.windows_by_file: Counter = Counter()
        self.windows_read = 0
        self.open_events = 0

    def record_open(self, path: str) -> None:
        self.open_events += 1
        self.opened_files.add(str(path))

    def record_window(self, path: str) -> None:
        self.windows_read += 1
        self.windows_by_file[str(path)] += 1

    def payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "open_events": self.open_events,
            "windows_read": self.windows_read,
            "opened_files": sorted(self.opened_files),
            "opened_file_count": len(self.opened_files),
            "windows_by_file": dict(sorted(self.windows_by_file.items())),
        }


class _AuditMixin:
    _read_audit: ReadAudit

    def _file(self, path: str):
        self._read_audit.record_open(str(path))
        return super()._file(path)

    def __getitem__(self, index: int):
        value = super().__getitem__(index)
        self._read_audit.record_window(str(self.rows[index]["file_path"]))
        return value


class AuditedWindowDataset(_AuditMixin, DroneRFaWindowDataset):
    """Stage-1 reader: one label per window, audited file access."""

    def __init__(self, *args: Any, read_audit: ReadAudit, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._read_audit = read_audit


class Stage2WindowDataset(DroneRFaWindowDataset):
    """The frozen stage-2 view: drone rows only, labels remapped to 0/1.

    Identical in behaviour to ``run_stage2_classification.Stage2WindowDataset``;
    re-declared here so this driver only depends on ``dronerfa_data``/``model``.
    """

    def __init__(
        self,
        index_path: Path,
        split: str,
        train_snrs: list[float | None] | None = None,
        fixed_snr: float | None = None,
        apply_fixed_snr: bool = False,
        max_samples: int | None = None,
        seed: int = TRAINING_SEED,
    ) -> None:
        super().__init__(index_path, split=split, seed=seed)
        self.rows = stage2_rows(self.rows)
        if max_samples is not None and max_samples > 0:
            self.rows = stratified_limit(self.rows, max_samples)
        self.train_snrs = train_snrs
        self.fixed_snr = fixed_snr
        self.apply_fixed_snr = apply_fixed_snr


class AuditedStage2Dataset(_AuditMixin, Stage2WindowDataset):
    def __init__(self, *args: Any, read_audit: ReadAudit, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._read_audit = read_audit


def validation_windows(stage: int) -> int:
    """Number of validation windows of one stage, straight from the frozen index."""
    rows = read_csv_rows(INDEX)
    if stage == 1:
        return sum(1 for row in rows if str(row["split"]) == VAL_ROLE)
    return sum(
        1
        for row in rows
        if str(row["split"]) == VAL_ROLE and str(row["target_code"]) in STAGE2_CODE_TO_LABEL
    )


def stage2_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the two drone target codes and remap their labels to 0/1."""
    selected: list[dict[str, Any]] = []
    for row in rows:
        code = str(row["target_code"])
        if code not in STAGE2_CODE_TO_LABEL:
            continue
        if int(row["label"]) != STAGE1_DRONE_LABEL:
            raise ValueError(
                f"{row.get('filename')} target {code} carries stage-1 label "
                f"{row['label']}; expected a drone window"
            )
        if row.get("stage2_label") not in (None, ""):
            declared = int(row["stage2_label"])
            if declared != STAGE2_CODE_TO_LABEL[code]:
                raise ValueError(
                    f"{row.get('filename')} stage2_label {declared} disagrees with "
                    f"target {code}"
                )
        label = STAGE2_CODE_TO_LABEL[code]
        selected.append({**row, "label": label, "class_name": STAGE2_LABEL_TO_NAME[label]})
    return selected


class _IndexSubsetDataset:
    """View a dataset through a list of positions without changing its length.

    ``__len__`` reports the full split length so the frozen AWGN rule
    ``default_rng(seed + epoch * len(dataset) + position)`` keeps producing the
    same noise as the unbounded run; ``__getitem__`` only reads the requested rows.
    """

    def __init__(self, dataset: Any, indices: Sequence[int]) -> None:
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


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def pooled_binary_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([int(row["label"]) for row in records], dtype=np.int64)
    prediction = np.asarray([int(row["prediction"]) for row in records], dtype=np.int64)
    tp = int(np.sum((y == 1) & (prediction == 1)))
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    if tp + tn + fp + fn != len(y):
        raise ValueError("confusion matrix does not conserve the window count")
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "windows": int(len(y)),
        "accuracy": (tp + tn) / max(len(y), 1),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "drone_recall": recall,
        "background_specificity": specificity,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def file_equal_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Equal weight per source file inside each class, then the two classes equally."""
    per_file: dict[str, dict[str, Any]] = {}
    for row in records:
        entry = per_file.setdefault(
            str(row["filename"]),
            {"label": int(row["label"]), "correct": 0, "windows": 0},
        )
        if entry["label"] != int(row["label"]):
            raise ValueError(f"source file {row['filename']} carries two classes")
        entry["correct"] += int(int(row["prediction"]) == int(row["label"]))
        entry["windows"] += 1
    class_rates: dict[int, list[float]] = {0: [], 1: []}
    per_file_rate: dict[str, float] = {}
    for filename, entry in per_file.items():
        rate = entry["correct"] / max(entry["windows"], 1)
        class_rates[entry["label"]].append(rate)
        per_file_rate[filename] = rate
    if not class_rates[0] or not class_rates[1]:
        raise ValueError("file-equal metrics need both classes in the split")
    phantom = float(np.mean(class_rates[0]))
    air2s = float(np.mean(class_rates[1]))
    return {
        "files_phantom": len(class_rates[0]),
        "files_air2s": len(class_rates[1]),
        "file_equal_phantom_recall": phantom,
        "file_equal_air2s_recall": air2s,
        "file_equal_balanced_accuracy": 0.5 * (phantom + air2s),
        "per_file_recall": per_file_rate,
        "per_file_windows": {name: entry["windows"] for name, entry in per_file.items()},
    }


# ---------------------------------------------------------------------------
# validation caches
# ---------------------------------------------------------------------------
def _cache_dataset(
    stage: int,
    condition: str,
    read_audit: ReadAudit,
    max_val_samples: int | None,
) -> tuple[Any, list[dict[str, Any]]]:
    if stage == 1:
        dataset = AuditedWindowDataset(
            INDEX,
            split=VAL_ROLE,
            fixed_snr=CONDITION_VALUES[condition],
            apply_fixed_snr=CONDITION_VALUES[condition] is not None,
            seed=VALIDATION_SEED,
            read_audit=read_audit,
        )
    else:
        dataset = AuditedStage2Dataset(
            INDEX,
            split=VAL_ROLE,
            fixed_snr=CONDITION_VALUES[condition],
            apply_fixed_snr=CONDITION_VALUES[condition] is not None,
            seed=VALIDATION_SEED,
            read_audit=read_audit,
        )
    if dataset.epoch != 0:
        raise RuntimeError("validation dataset epoch must stay 0")
    selected = stratified_limit(dataset.rows, max_val_samples)
    positions = {id(row): index for index, row in enumerate(dataset.rows)}
    indices = [positions[id(row)] for row in selected]
    selected = [
        {**row, "index_position": position} for row, position in zip(selected, indices)
    ]
    return _IndexSubsetDataset(dataset, indices), selected


def build_validation_cache(
    stage: int,
    device: torch.device,
    read_audit: ReadAudit,
    max_val_samples: int | None = None,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]], list[dict[str, Any]]]:
    frontend = SpectrogramFrontEnd(
        n_fft=N_FFT, hop_length=HOP_LENGTH, image_size=IMAGE_SIZE
    ).to(device)
    frontend.eval()
    cached: dict[str, tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]] = {}
    metadata: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        view, selected = _cache_dataset(stage, condition, read_audit, max_val_samples)
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
        started = time.perf_counter()
        with torch.inference_mode():
            for batch, (signals, targets) in enumerate(loader, 1):
                features = extract_features(frontend, signals.to(device))
                if not bool(torch.isfinite(features).all()):
                    raise RuntimeError(f"non-finite validation features in {condition}")
                tensors.append(features.cpu())
                labels.append(targets)
                if batch % 50 == 0 or batch == len(loader):
                    log(f"  val cache {condition} {batch}/{len(loader)}")
        xx, yy = torch.cat(tensors), torch.cat(labels)
        cached[condition] = (xx, yy, selected)
        metadata.append(
            {
                "split": VAL_ROLE,
                "stage": stage,
                "condition": condition,
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "noise_seed": VALIDATION_SEED,
                "samples": len(selected),
                "index_positions": view.indices,
                "shape": [len(selected), 1, IMAGE_SIZE, IMAGE_SIZE],
                "dtype": "float32",
                "feature_sha256": sha256_bytes(xx.numpy().tobytes()),
                "label_sha256": sha256_bytes(yy.numpy().astype(np.int64).tobytes()),
                "seconds": time.perf_counter() - started,
            }
        )
        view.close()
    del frontend
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cached, metadata


@torch.inference_mode()
def evaluate_cached(
    stage: int,
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
    for condition in CONDITIONS:
        xx, yy, rows = cached[condition]
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
                        "window_index": int(source["window_index"]),
                        "start": int(source["start"]),
                        "label": int(labels_np[offset]),
                        "logit_margin": float(margins[offset]),
                        "prediction": int(predictions[offset]),
                    }
                )
            cursor += len(labels_np)
        metrics_all.append(
            {
                "stage": stage,
                "condition": condition,
                "loss": total_loss / max(len(rows), 1),
                "samples": len(rows),
                **pooled_binary_metrics(window_records),
                **(
                    {
                        key: value
                        for key, value in file_equal_metrics(window_records).items()
                        if key not in ("per_file_recall", "per_file_windows")
                    }
                    if stage == 2
                    else {}
                ),
            }
        )
        records_all.extend(window_records)
    return metrics_all, records_all


def select_epoch(history: Sequence[dict[str, Any]], key: str) -> int:
    """Highest mean validation score; the earliest epoch wins an exact tie."""
    return int(max(history, key=lambda row: (row[key], -int(row["epoch"])))["epoch"])


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "source_sha256": {name: sha256_file(PROJECT_DIR / name) for name in SOURCE_FILES},
        "recorded_at": utc_now(),
    }


def make_loader(
    dataset: Any, order: Sequence[int], seed: int, device: torch.device
) -> DataLoader:
    return DataLoader(
        Subset(dataset, list(order)) if order else dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed + 50_000),
    )


def train_stage(
    stage: int,
    device: torch.device,
    epochs: int,
    output_dir: Path,
    max_train_samples: int | None,
    max_val_samples: int | None,
    smoke: bool,
    pre_run: dict[str, Any],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {output_dir}")
    output_dir.mkdir(parents=True)
    stage_name = f"stage{stage}"
    started_at = utc_now()
    amp_enabled = device.type == "cuda"
    audit = ReadAudit(stage_name)
    log(f"[{stage_name}] training seed {TRAINING_SEED} -> {output_dir}")

    cached, cache_metadata = build_validation_cache(
        stage, device, audit, max_val_samples
    )
    write_json(output_dir / "validation_cache_manifest.json", cache_metadata)
    for row in cache_metadata:
        log(
            f"  validation cache {row['condition']}: samples={row['samples']} "
            f"feature_sha256={row['feature_sha256'][:16]}... ({row['seconds']:.1f}s)"
        )

    seed_everything(TRAINING_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = ResNet18(input_channels=1, num_classes=2).to(device)
    initial_hash = state_hash(model.state_dict())
    frontend = SpectrogramFrontEnd(
        n_fft=N_FFT, hop_length=HOP_LENGTH, image_size=IMAGE_SIZE
    ).to(device)

    dataset_class = AuditedWindowDataset if stage == 1 else AuditedStage2Dataset
    dataset = dataset_class(
        INDEX,
        split=TRAIN_ROLE,
        train_snrs=parse_snr_list(TRAIN_SNRS),
        seed=TRAINING_SEED,
        read_audit=audit,
    )
    if max_train_samples is not None and max_train_samples > 0:
        selected = stratified_limit(dataset.rows, max_train_samples)
        positions = {id(row): index for index, row in enumerate(dataset.rows)}
        bounded = _IndexSubsetDataset(dataset, [positions[id(row)] for row in selected])
        train_view: Any = bounded
    else:
        train_view = dataset

    counts = {
        label: sum(1 for row in dataset.rows if int(row["label"]) == label) for label in (0, 1)
    }
    if not all(counts.values()):
        raise ValueError(f"both training classes are required, got {counts}")
    weights = torch.tensor(
        [len(dataset) / (2 * counts[label]) for label in (0, 1)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    stage1 = stage == 1
    config: dict[str, Any] = {
        "phase": "CASCADE3-NEWSPLIT-B",
        "stage": stage_name,
        "task": (
            "Background (T0000) vs Drone (T0010/T0101)"
            if stage1
            else "oracle machine type: Phantom 4 Pro (T0010)=0 vs Air 2S (T0101)=1"
        ),
        "split_source": "work/meta_split2/split_candidate_104_20260926.csv",
        "split_source_sha256": FROZEN_SPLIT_CSV_SHA256,
        "index": str(INDEX),
        "index_sha256": INDEX_SHA256,
        "stage2_view": str(STAGE2_VIEW),
        "stage2_view_sha256": STAGE2_VIEW_SHA256,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "hop_ratio": HOP_LENGTH / N_FFT,
        "image_size": IMAGE_SIZE,
        "window_samples": WINDOW_SAMPLES,
        "training_seed": TRAINING_SEED,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": {"type": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
        "train_snrs": TRAIN_SNRS,
        "train_windows_available": len(dataset),
        "train_windows_used_per_epoch": len(train_view),
        "train_class_counts": counts,
        "class_weights": weights.detach().cpu().tolist(),
        "initial_model_sha256": initial_hash,
        "validation_split": VAL_ROLE,
        "validation_seed": VALIDATION_SEED,
        "validation_conditions": list(CONDITIONS),
        "validation_windows_available": validation_windows(stage),
        "validation_windows_used_per_epoch": len(cache_metadata[0]["index_positions"]),
        "selection": STAGE1_SELECTION_RULE if stage1 else STAGE2_SELECTION_RULE,
        "tie_break": TIE_BREAK,
        "prediction": STAGE1_PREDICTION_RULE if stage1 else STAGE2_PREDICTION_RULE,
        "label_mapping": (
            {"0": "Background", "1": "Drone"}
            if stage1
            else {str(label): STAGE2_LABEL_TO_NAME[label] for label in (0, 1)}
        ),
        "frontend": {
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "image_size": IMAGE_SIZE,
            "window": "torch.hamming_window(n_fft) periodic",
            "implementation": "model.SpectrogramFrontEnd (frozen)",
        },
        "amp": amp_enabled,
        "amp_dtype": "float16",
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "smoke": smoke,
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "started_at": started_at,
        "source_sha256": pre_run["source_sha256"],
        "test_role": TEST_ROLE,
        "test_windows_never_read": None,
        "status": "running",
    }
    write_json(output_dir / "run_config.json", config)
    write_json(output_dir / "read_audit.json", audit.payload())

    history: list[dict[str, Any]] = []
    order_hashes: list[str] = []
    checkpoint_path = output_dir / "best_model.pt"
    selection_key = (
        "mean_balanced_accuracy" if stage1 else "mean_file_equal_balanced_accuracy"
    )
    try:
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            dataset.set_epoch(epoch)
            if max_train_samples is not None and max_train_samples > 0:
                order = torch.randperm(
                    len(dataset), generator=torch.Generator().manual_seed(TRAINING_SEED + epoch)
                ).tolist()
                position_of = {index: slot for slot, index in enumerate(train_view.indices)}
                order = [position_of[row] for row in order if row in position_of]
            else:
                order = torch.randperm(
                    len(dataset), generator=torch.Generator().manual_seed(TRAINING_SEED + epoch)
                ).tolist()
            order_hashes.append(sha256_bytes(np.asarray(order, dtype=np.int64).tobytes()))
            loader = make_loader(train_view, order, TRAINING_SEED + epoch, device)
            log(f"  [{stage_name}] epoch {epoch}/{epochs}")
            train_loss, train_metrics = train_epoch(
                frontend, model, loader, criterion, optimizer, scaler, device, amp_enabled
            )
            metrics, records = evaluate_cached(
                stage, model, cached, criterion, device, amp_enabled
            )
            if stage1:
                mean_score = float(np.mean([row["balanced_accuracy"] for row in metrics]))
                clean_key = "balanced_accuracy"
                clean_value = next(
                    row for row in metrics if row["condition"] == "clean"
                )["balanced_accuracy"]
                extra = {
                    "train_f1": train_metrics["f1"],
                    "train_drone_recall": train_metrics["recall"],
                }
            else:
                mean_score = float(
                    np.mean([row["file_equal_balanced_accuracy"] for row in metrics])
                )
                clean_key = "file_equal_balanced_accuracy"
                clean_value = next(
                    row for row in metrics if row["condition"] == "clean"
                )["file_equal_balanced_accuracy"]
                extra = {
                    "train_f1": train_metrics["f1"],
                    "train_air2s_recall": train_metrics["recall"],
                }
            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                **extra,
                f"clean_{clean_key}": clean_value,
                selection_key: mean_score,
                "seconds": time.perf_counter() - epoch_started,
            }
            history.append(history_row)
            write_csv(
                output_dir / "validation_metrics.csv",
                accumulate_validation_metrics(output_dir, epoch, metrics),
            )
            write_csv(output_dir / "training_history.csv", history)
            write_csv(output_dir / f"validation_scores_epoch_{epoch:02d}.csv", records)
            torch.save(
                {
                    "epoch": epoch,
                    "frontend_state": frontend.state_dict(),
                    "model_state": model.state_dict(),
                    "n_fft": N_FFT,
                    "hop_length": HOP_LENGTH,
                    "validation_seed": VALIDATION_SEED,
                    "val_metrics": metrics,
                    "selection_rule": config["selection"],
                    "tie_break": TIE_BREAK,
                    "prediction_rule": config["prediction"],
                    "training_seed": TRAINING_SEED,
                    "initial_model_sha256": initial_hash,
                    "index_sha256": INDEX_SHA256,
                    "stage2_view_sha256": STAGE2_VIEW_SHA256,
                    "split_source_sha256": FROZEN_SPLIT_CSV_SHA256,
                    "run_config": config,
                },
                output_dir / f"epoch_{epoch:02d}.pt",
            )
            selected_epoch = select_epoch(history, selection_key)
            selection = {
                "stage": stage_name,
                "n_fft": N_FFT,
                "training_seed": TRAINING_SEED,
                "rule": config["selection"],
                "tie_break": TIE_BREAK,
                "validation_split": VAL_ROLE,
                "conditions": list(CONDITIONS),
                f"{selection_key}_by_epoch": {
                    str(row["epoch"]): row[selection_key] for row in history
                },
                "selected_epoch": selected_epoch,
                "selected_epoch_score": next(
                    row[selection_key] for row in history if row["epoch"] == selected_epoch
                ),
            }
            write_json(output_dir / "selection.json", selection)
            if selected_epoch == epoch:
                shutil.copyfile(output_dir / f"epoch_{epoch:02d}.pt", checkpoint_path)
            write_json(
                output_dir / "progress.json",
                {
                    "stage": stage_name,
                    "status": "running",
                    "epochs_requested": epochs,
                    "epochs_completed": epoch,
                    "last_completed_epoch": epoch,
                    "selected_epoch_so_far": selected_epoch,
                    "last_epoch_seconds": history_row["seconds"],
                    "updated_at": utc_now(),
                },
            )
            write_json(output_dir / "read_audit.json", audit.payload())
            log(
                f"  epoch {epoch}: {selection_key}={mean_score:.6f} "
                f"({history_row['seconds']:.1f}s) selected={selected_epoch}"
            )
            for metric in metrics:
                suffix = (
                    f"BA={metric['balanced_accuracy']:.6f}"
                    if stage1
                    else f"file-equal BA={metric['file_equal_balanced_accuracy']:.6f} "
                    f"pooled BA={metric['balanced_accuracy']:.6f}"
                )
                log(
                    f"    {metric['condition']}: {suffix} "
                    f"tp={metric['tp']} tn={metric['tn']} "
                    f"fp={metric['fp']} fn={metric['fn']}"
                )
    finally:
        train_view.close()

    selected = select_epoch(history, selection_key)
    config["epoch_order_sha256"] = order_hashes
    config["selected_epoch"] = selected
    config["selected_epoch_score"] = next(
        row[selection_key] for row in history if row["epoch"] == selected
    )
    config["epoch_history"] = history
    config["retained_epoch_checkpoints"] = []
    config["removed_epoch_checkpoints"] = []
    for path in sorted(output_dir.glob("epoch_*.pt")):
        if path.name == f"epoch_{selected:02d}.pt":
            config["retained_epoch_checkpoints"].append(path.name)
            continue
        path.unlink()
        config["removed_epoch_checkpoints"].append(path.name)
    config["best_model_sha256"] = sha256_file(checkpoint_path)
    config["completed_at"] = utc_now()
    config["status"] = "completed"
    write_json(output_dir / "run_config.json", config)
    write_json(
        output_dir / "progress.json",
        {
            "stage": stage_name,
            "status": "completed",
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "selected_epoch": selected,
            "updated_at": utc_now(),
        },
    )
    log(
        f"  [{stage_name}] kept {config['retained_epoch_checkpoints']}, "
        f"removed {len(config['removed_epoch_checkpoints'])} non-selected checkpoints"
    )

    # The read audit must never contain a development_test_exposed source file.
    test_paths = set(pre_run["identity"]["paths_by_role"][TEST_ROLE])
    train_val_paths = set(pre_run["identity"]["paths_by_role"][TRAIN_ROLE]) | set(
        pre_run["identity"]["paths_by_role"][VAL_ROLE]
    )
    audit_payload = audit.payload()
    audit_payload["test_files_touched"] = sorted(set(audit.opened_files) & test_paths)
    audit_payload["unexpected_files_touched"] = sorted(
        set(audit.opened_files) - train_val_paths
    )
    audit_payload["opened_all_train"] = train_val_paths <= set(audit.opened_files)
    audit_payload["passed"] = not (
        audit_payload["test_files_touched"] or audit_payload["unexpected_files_touched"]
    )
    write_json(output_dir / "read_audit.json", audit_payload)
    if not audit_payload["passed"]:
        raise RuntimeError(f"read audit failed for {stage_name}: {audit_payload}")
    config["test_windows_never_read"] = True
    config["read_audit_windows_read"] = audit_payload["windows_read"]
    config["read_audit_opened_files"] = audit_payload["opened_file_count"]
    config["read_audit_passed"] = True
    write_json(output_dir / "run_config.json", config)

    reload_check = verify_checkpoint_reload(
        stage, output_dir, checkpoint_path, cached, criterion, device, amp_enabled
    )
    write_json(output_dir / "checkpoint_reload_check.json", reload_check)
    if not reload_check["passed"]:
        raise RuntimeError(f"checkpoint reload check failed for {stage_name}")

    del model, optimizer, scaler
    cached.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "stage": stage_name,
        "run_directory": str(output_dir),
        "config": config,
        "history": history,
        "reload_check": reload_check,
        "read_audit": audit_payload,
    }


def accumulate_validation_metrics(
    output_dir: Path, epoch: int, metrics: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    path = output_dir / "validation_metrics.csv"
    rows = read_csv_rows(path) if path.is_file() else []
    rows.extend({"epoch": epoch, **{key: str(value) for key, value in row.items()}} for row in metrics)
    return rows


def verify_checkpoint_reload(
    stage: int,
    output_dir: Path,
    checkpoint_path: Path,
    cached: dict[str, tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]],
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    """Reload the best checkpoint in fresh modules and reproduce its val metrics."""
    selection = read_json(output_dir / "selection.json")
    selected = int(selection["selected_epoch"])
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    frontend = SpectrogramFrontEnd(
        n_fft=N_FFT, hop_length=HOP_LENGTH, image_size=IMAGE_SIZE
    ).to(device)
    model = ResNet18(input_channels=1, num_classes=2).to(device)
    frontend.load_state_dict(payload["frontend_state"])
    model.load_state_dict(payload["model_state"])
    metrics, _ = evaluate_cached(stage, model, cached, criterion, device, amp_enabled)
    recorded = payload["val_metrics"]
    count_keys = ("tn", "fp", "fn", "tp")
    float_keys = (
        ("balanced_accuracy",)
        if stage == 1
        else ("file_equal_balanced_accuracy", "balanced_accuracy")
    )
    comparisons = []
    max_float_delta = 0.0
    for expected, actual in zip(recorded, metrics):
        if expected["condition"] != actual["condition"]:
            raise RuntimeError("validation condition order changed on reload")
        for key in count_keys:
            comparisons.append(
                {
                    "condition": expected["condition"],
                    "metric": key,
                    "recorded": expected[key],
                    "reloaded": actual[key],
                    "match": bool(expected[key] == actual[key]),
                    "exact": True,
                }
            )
        for key in float_keys:
            delta = abs(float(expected[key]) - float(actual[key]))
            max_float_delta = max(max_float_delta, delta)
            comparisons.append(
                {
                    "condition": expected["condition"],
                    "metric": key,
                    "recorded": expected[key],
                    "reloaded": actual[key],
                    "match": bool(delta <= RELOAD_TOLERANCE),
                    "exact": bool(expected[key] == actual[key]),
                    "abs_delta": delta,
                }
            )
    mismatches = [row for row in comparisons if not row["match"]]
    result = {
        "stage": f"stage{stage}",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "selected_epoch": selected,
        "epoch_matches_selection": int(payload["epoch"]) == selected,
        "recorded_model_state_sha256": state_hash(payload["model_state"]),
        "reloaded_model_state_sha256": state_hash(model.state_dict()),
        "comparisons": len(comparisons),
        "float_tolerance": RELOAD_TOLERANCE,
        "max_float_delta": max_float_delta,
        "mismatches": mismatches[:10],
    }
    result["state_dict_identical"] = (
        result["recorded_model_state_sha256"] == result["reloaded_model_state_sha256"]
    )
    result["passed"] = bool(
        result["epoch_matches_selection"]
        and result["state_dict_identical"]
        and not mismatches
    )
    del model, frontend
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def status_report() -> dict[str, Any]:
    report: dict[str, Any] = {"run_root": str(RUN_ROOT), "stages": {}}
    for root in (RUN_ROOT, SMOKE_ROOT):
        for stage in (1, 2):
            directory = root / f"stage{stage}"
            entry: dict[str, Any] = {"directory": str(directory), "exists": directory.is_dir()}
            if entry["exists"]:
                progress = directory / "progress.json"
                config = directory / "run_config.json"
                checkpoint = directory / "best_model.pt"
                entry["progress"] = read_json(progress) if progress.is_file() else None
                if config.is_file():
                    payload = read_json(config)
                    entry["status"] = payload.get("status")
                    entry["selected_epoch"] = payload.get("selected_epoch")
                entry["best_model"] = (
                    {
                        "path": str(checkpoint),
                        "bytes": checkpoint.stat().st_size,
                        "sha256": sha256_file(checkpoint),
                    }
                    if checkpoint.is_file()
                    else None
                )
                entry["epoch_checkpoints"] = sorted(
                    path.name for path in directory.glob("epoch_*.pt")
                )
            report["stages"][f"{root.name}/{directory.name}"] = entry
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CASCADE3-NEWSPLIT-B: two-stage retraining on the frozen new split."
    )
    parser.add_argument("--stage", choices=("stage1", "stage2", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--smoke-train-samples", type=int, default=24)
    parser.add_argument("--smoke-val-samples", type=int, default=8)
    parser.add_argument("--status", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.status:
        print(json.dumps(status_report(), ensure_ascii=False, indent=2))
        return 0

    root = SMOKE_ROOT if args.smoke else RUN_ROOT
    epochs = args.smoke_epochs if args.smoke else args.epochs
    max_train = args.smoke_train_samples if args.smoke else None
    max_val = args.smoke_val_samples if args.smoke else None
    if args.smoke:
        epochs = min(epochs, 1)

    device = resolve_device(args.device)
    identity = check_index_identity()
    if not identity["passed"]:
        raise RuntimeError(f"frozen index identity check failed: {identity['errors']}")
    cross_check = frozen_implementation_cross_check()
    if not cross_check["passed"]:
        raise RuntimeError(f"stage-2 view cross-check failed: {cross_check}")
    environment = environment_record(device)

    pre_run = {
        "phase": "CASCADE3-NEWSPLIT-B",
        "run_root": str(root),
        "smoke": args.smoke,
        "stage": args.stage,
        "epochs": epochs,
        "max_train_samples": max_train,
        "max_val_samples": max_val,
        "training_seed": TRAINING_SEED,
        "protocol": {
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "train_snrs": TRAIN_SNRS,
            "validation_seed": VALIDATION_SEED,
            "conditions": list(CONDITIONS),
            "tie_break": TIE_BREAK,
            "stage1_selection": STAGE1_SELECTION_RULE,
            "stage2_selection": STAGE2_SELECTION_RULE,
            "stage1_prediction": STAGE1_PREDICTION_RULE,
            "stage2_prediction": STAGE2_PREDICTION_RULE,
        },
        "identity": identity,
        "stage2_view_cross_check": cross_check,
        "environment": environment,
        "source_sha256": environment["source_sha256"],
        "written_at": utc_now(),
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "pre_run_config.json", pre_run)
    write_json(root / "environment.json", environment)
    write_json(root / "input_identity.json", identity)
    log(json.dumps({"identity_passed": identity["passed"], "run_root": str(root)}, ensure_ascii=False))

    results: list[dict[str, Any]] = []
    stages = (1, 2) if args.stage == "all" else (int(args.stage[-1]),)
    for stage in stages:
        result = train_stage(
            stage,
            device,
            epochs,
            root / f"stage{stage}",
            max_train,
            max_val,
            args.smoke,
            pre_run,
        )
        results.append(result)

    summary = {
        "phase": "CASCADE3-NEWSPLIT-B",
        "smoke": args.smoke,
        "epochs": epochs,
        "training_seed": TRAINING_SEED,
        "identity_passed": identity["passed"],
        "stages": [
            {
                "stage": row["stage"],
                "run_directory": row["run_directory"],
                "selected_epoch": row["config"]["selected_epoch"],
                "selected_epoch_score": row["config"]["selected_epoch_score"],
                "selection_rule": row["config"]["selection"],
                "best_model_sha256": row["config"]["best_model_sha256"],
                "history": row["history"],
                "read_audit_passed": row["read_audit"]["passed"],
                "read_audit_windows": row["read_audit"]["windows_read"],
                "checkpoint_reload_passed": row["reload_check"]["passed"],
            }
            for row in results
        ],
        "completed_at": utc_now(),
    }
    write_json(root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
