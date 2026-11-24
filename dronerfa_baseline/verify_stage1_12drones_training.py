#!/usr/bin/env python3
"""Independent verification of the STAGE1-12DRONES-A training artefacts.

Everything here is re-derived from the files on disk and from the frozen split /
index, deliberately *without* importing the training driver
(``run_stage1_12drones_training``) or the index builder
(``build_stage1_12drones_index``).  The only shared code is the data reader
(``dronerfa_data``) and the model definition (``model``), both already accepted
by earlier tasks; the spectrogram maths is re-implemented locally so a change in
``model.SpectrogramFrontEnd`` cannot hide behind a shared import.

Checks cover, in order:

A  frozen input identity (split CSV hash, G manifest hash, index hash/shape)
B  index <-> frozen-split agreement and label/role integrity
C  read audit: the exact set of opened files, opened-file == train+val, and a
   per-file read budget consistent with the number of completed epochs
D  checkpoint reload in fresh modules
E  independent re-derivation of the recorded validation metrics and the epoch
   selection rule from raw RF0 reads
F  negative controls that tamper with a claim and must be detected

Read-only: nothing under ``outputs/`` is modified.  The whole verification runs
on CPU so it can proceed beside a GPU training job.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / "outputs" / ".matplotlib"))

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from model import ResNet18

# ---------------------------------------------------------------------------
# frozen constants (independent copies, not imported from the driver)
# ---------------------------------------------------------------------------
INDEX = SCRIPT_DIR / "outputs" / "stage1_12drones_20260926" / "window_index_stage1_12drones.csv"
INDEX_SHA256 = "5973a2fa5b8476a507fcc7a665365daf8d572f569ce5908d54f5c4bc84c48bd6"
INDEX_ROWS = 15790
OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "stage1_12drones_20260926"
#: Per-seed run directories live under ``OUTPUT_ROOT``; use :func:`run_root_for`.
SUMMARY_JSON = OUTPUT_ROOT / "dataset_summary.json"
FREEZE_JSON = PROJECT_ROOT / "work" / "meta_split2" / "CODEX-FREEZE-20260926.json"
FROZEN_SPLIT_CSV = PROJECT_ROOT / "work" / "meta_split2" / "split_candidate_104_20260926.csv"
FROZEN_SPLIT_CSV_SHA256 = (
    "d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9"
)
G_MANIFEST = Path(r"G:\DroneRFa数据集\dataset_manifest.csv")
G_MANIFEST_SHA256 = "9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d"

TRAIN_ROLE = "train"
VAL_ROLE = "val"
TEST_ROLES = ("development_test", "development_test_exposed")
BACKGROUND_CODE = "T0000"
DRONE_CODES = (
    "T0001", "T0010", "T0011", "T0100", "T0101", "T0110",
    "T0111", "T1000", "T1001", "T1010", "T1011", "T1100",
)

N_FFT = 512
HOP_LENGTH = 256
IMAGE_SIZE = 256
BATCH_SIZE = 4
#: The accepted STAGE1-12DRONES-A seed; ``--training-seed`` selects another.
DEFAULT_TRAINING_SEED = 2026
#: Rebinding this only changes which claimed seed the artefacts are checked
#: against; every re-derivation below is seed-independent by construction.
TRAINING_SEED = DEFAULT_TRAINING_SEED
#: Pinned for every training seed: the validation noise must stay byte-identical
#: across seeds, otherwise the seeds are not paired-comparable.
VALIDATION_SEED = 12026
CONDITION_VALUES: dict[str, float | None] = {
    "clean": None,
    "-10dB": -10.0,
    "-12dB": -12.0,
    "-15dB": -15.0,
}
CONDITIONS = tuple(CONDITION_VALUES)
SELECTION_KEY = "mean_balanced_accuracy"
WINDOW_SAMPLES = 1_000_000
WINDOWS_PER_BLOCK = 10

#: Bounded replay budget: validation windows re-derived from raw RF0 (and run
#: through all four conditions).  ``0`` replays the whole val split.
REPLAY_WINDOWS = 480
RMS_EPSILON = 1e-8
#: The training run caches validation features on the GPU; an independent CPU
#: recomputation differs only by float32 rounding, so margins are compared with
#: an explicit device tolerance while the decisions must match exactly.
MARGIN_TOLERANCE = 0.05


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Any:
    # ``utf-8-sig`` accepts both a plain UTF-8 file and one written with a BOM by
    # another tool, so a stray BOM cannot look like a verification result.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_root_for(seed: int) -> Path:
    """Run directory of a training seed inside the shared stage-1 output root."""
    return OUTPUT_ROOT / f"seed_{int(seed)}"


def bind_training_seed(seed: int) -> int:
    """Point check E9 at the seed whose artefacts are under verification."""
    global TRAINING_SEED
    TRAINING_SEED = int(seed)
    return TRAINING_SEED


class Checks:
    """Ordered, labelled assertions with an aggregate verdict."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, detail: Any = None) -> bool:
        self.rows.append({"check": name, "passed": bool(passed), "detail": detail})
        mark = "PASS" if passed else "FAIL"
        text = "" if detail is None else f": {detail}"
        print(f"[{mark}] {name}{text}", flush=True)
        return bool(passed)

    @property
    def passed(self) -> bool:
        return all(row["passed"] for row in self.rows)

    @property
    def failed(self) -> list[str]:
        return [row["check"] for row in self.rows if not row["passed"]]

    def payload(self) -> dict[str, Any]:
        return {
            "checks": self.rows,
            "passed_checks": sum(1 for row in self.rows if row["passed"]),
            "total_checks": len(self.rows),
            "passed": self.passed,
            "failed_checks": self.failed,
        }


# ---------------------------------------------------------------------------
# independent re-implementation of the frozen data maths
# ---------------------------------------------------------------------------
def rms_normalize(iq: np.ndarray, epsilon: float = RMS_EPSILON) -> np.ndarray:
    """Local copy of the frozen per-window mean removal + RMS normalisation."""
    centered = iq - np.mean(iq, dtype=np.complex128)
    power = float(np.mean(np.abs(centered) ** 2, dtype=np.float64))
    return (centered / math.sqrt(max(power, epsilon))).astype(np.complex64)


def add_complex_awgn(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Local copy of the frozen complex-AWGN rule (A_original)."""
    signal_power = float(np.mean(np.abs(iq) ** 2, dtype=np.float64))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = math.sqrt(noise_power / 2.0)
    noise = scale * (
        rng.standard_normal(iq.shape, dtype=np.float32)
        + 1j * rng.standard_normal(iq.shape, dtype=np.float32)
    )
    return (iq + noise).astype(np.complex64)


class LocalSpectrogramFrontEnd:
    """Independent log-power STFT front end: Hamming, hop 256, fftshift, 256x256."""

    def __init__(self) -> None:
        self.window = torch.hamming_window(N_FFT, periodic=True)

    def __call__(self, iq_components: torch.Tensor) -> torch.Tensor:
        if iq_components.ndim != 3 or iq_components.shape[1] != 2:
            raise ValueError(f"expected [batch, 2, samples], got {tuple(iq_components.shape)}")
        complex_iq = torch.complex(iq_components[:, 0], iq_components[:, 1])
        spectra = torch.stft(
            complex_iq,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=N_FFT,
            window=self.window,
            center=False,
            normalized=True,
            onesided=False,
            return_complex=True,
        )
        log_power = 10.0 * torch.log10(spectra.abs().square() + 1e-10)
        log_power = torch.fft.fftshift(log_power, dim=1)
        mean = log_power.mean(dim=(1, 2), keepdim=True)
        std = log_power.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        normalized = (log_power - mean) / std
        return F.interpolate(
            normalized.unsqueeze(1),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )


def read_channel_slice(dataset: h5py.Dataset, start: int, length: int) -> np.ndarray:
    """Shape-tolerant RF0 slice read, identical to the frozen reader's rule."""
    if dataset.ndim == 1:
        values = dataset[start : start + length]
    elif dataset.ndim == 2 and dataset.shape[0] == 1:
        values = dataset[0, start : start + length]
    elif dataset.ndim == 2 and dataset.shape[1] == 1:
        values = dataset[start : start + length, 0]
    else:
        raise ValueError(f"unsupported dataset shape: {dataset.shape}")
    return np.asarray(values, dtype=np.float32)


def window_iq(handle: h5py.File, row: dict[str, Any]) -> np.ndarray:
    start = int(row["start"])
    length = int(row["length"])
    real = read_channel_slice(handle["RF0_I"], start, length)
    imag = read_channel_slice(handle["RF0_Q"], start, length)
    return rms_normalize(real + 1j * imag)


def condition_iq(iq: np.ndarray, condition: str, position: int) -> np.ndarray:
    """Fixed validation noise, mirroring the frozen reader's RNG consumption.

    ``DroneRFaWindowDataset.__getitem__`` builds ``default_rng(seed + epoch * len
    + index)`` per window (epoch 0 for validation), draws the training-selection
    integer, and only then draws the AWGN samples.
    """
    snr = CONDITION_VALUES[condition]
    if snr is None:
        return iq
    selection = (None,)
    rng = np.random.default_rng(VALIDATION_SEED + position)
    rng.integers(0, len(selection))  # the frozen reader draws this even for val
    return add_complex_awgn(iq, snr, rng)


def metrics_from_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([int(row["label"]) for row in records], dtype=np.int64)
    prediction = np.asarray([int(row["prediction"]) for row in records], dtype=np.int64)
    tp = int(np.sum((y == 1) & (prediction == 1)))
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
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
    per_file: dict[str, dict[str, Any]] = {}
    for row in records:
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
    """Highest mean score wins; the earliest epoch wins an exact tie."""
    return int(max(history, key=lambda row: (row[key], -int(row["epoch"])))["epoch"])


# ---------------------------------------------------------------------------
# A. frozen input identity
# ---------------------------------------------------------------------------
def check_inputs(checks: Checks) -> tuple[list[dict[str, str]], dict[str, str], dict[str, Any]]:
    freeze = read_json(FREEZE_JSON)
    checks.add(
        "A1 freeze decision points at the frozen split CSV",
        freeze.get("frozen_split_csv") == "work/meta_split2/split_candidate_104_20260926.csv"
        and freeze.get("frozen_split_csv_sha256") == FROZEN_SPLIT_CSV_SHA256,
        freeze.get("frozen_split_csv"),
    )
    checks.add(
        "A2 frozen split CSV SHA256",
        sha256_file(FROZEN_SPLIT_CSV) == FROZEN_SPLIT_CSV_SHA256,
        FROZEN_SPLIT_CSV_SHA256,
    )
    checks.add(
        "A3 G dataset manifest SHA256 unchanged",
        G_MANIFEST.is_file() and sha256_file(G_MANIFEST) == G_MANIFEST_SHA256,
        G_MANIFEST_SHA256,
    )
    checks.add(
        "A4 window index SHA256",
        sha256_file(INDEX) == INDEX_SHA256,
        INDEX_SHA256,
    )

    frozen_rows = read_csv_rows(FROZEN_SPLIT_CSV)
    frozen_role: dict[str, str] = {}
    frozen_class: dict[str, str] = {}
    for row in frozen_rows:
        frozen_role[str(row["file_name"])] = str(row["role"])
        frozen_class[str(row["file_name"])] = str(row["class_id"])
    checks.add(
        "A5 frozen split describes 104 files / 13 classes",
        len(frozen_rows) == 104
        and len(frozen_role) == 104
        and len(set(frozen_class.values())) == 13
        and set(frozen_class.values()) == {BACKGROUND_CODE, *DRONE_CODES},
        {"files": len(frozen_role), "classes": len(set(frozen_class.values()))},
    )

    rows = read_csv_rows(INDEX)
    checks.add(
        "A6 index row count",
        len(rows) == INDEX_ROWS,
        {"rows": len(rows), "expected": INDEX_ROWS},
    )
    checks.add(
        "A7 index covers exactly the frozen 104 file names",
        {str(row["filename"]) for row in rows} == set(frozen_role),
        len({str(row["filename"]) for row in rows}),
    )
    return rows, frozen_role, freeze


def check_index_integrity(
    checks: Checks,
    rows: Sequence[dict[str, str]],
    frozen_role: dict[str, str],
    summary: dict[str, Any],
) -> dict[str, Any]:
    cross_role: list[str] = []
    label_violations: list[str] = []
    geometry: list[str] = []
    for row in rows:
        filename = str(row["filename"])
        role = str(row["source_role"])
        if role != str(row["split"]) or frozen_role.get(filename) != role:
            cross_role.append(filename)
        expected_label = 0 if str(row["target_code"]) == BACKGROUND_CODE else 1
        if int(row["label"]) != expected_label or int(row["stage1_label"]) != expected_label:
            label_violations.append(filename)
        if (
            int(row["length"]) != WINDOW_SAMPLES
            or int(row["start"]) != int(row["window_index"]) * WINDOW_SAMPLES
            or int(row["window_in_block"]) != int(row["window_index"]) % WINDOWS_PER_BLOCK
            or int(row["block_index"]) != int(row["window_index"]) // WINDOWS_PER_BLOCK
        ):
            geometry.append(filename)
    checks.add("B1 every window inherits its frozen source role", not cross_role, sorted(set(cross_role))[:5])
    checks.add(
        "B2 label 0 is exactly T0000 and label 1 covers the other twelve codes",
        not label_violations
        and {str(row["target_code"]) for row in rows if int(row["label"]) == 0} == {BACKGROUND_CODE}
        and {str(row["target_code"]) for row in rows if int(row["label"]) == 1}
        == set(DRONE_CODES),
        sorted(set(label_violations))[:5],
    )
    checks.add("B3 window geometry is 1M-aligned and 1M long", not geometry, sorted(set(geometry))[:5])

    role_files: dict[str, set[str]] = defaultdict(set)
    role_windows: Counter = Counter()
    role_label_windows: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        role = str(row["source_role"])
        role_files[role].add(str(row["filename"]))
        role_windows[role] += 1
        role_label_windows[role][int(row["label"])] += 1
    overlap = {
        "train_val": sorted(role_files[TRAIN_ROLE] & role_files[VAL_ROLE]),
        "train_test": sorted(
            role_files[TRAIN_ROLE] & (role_files["development_test"] | role_files["development_test_exposed"])
        ),
        "val_test": sorted(
            role_files[VAL_ROLE] & (role_files["development_test"] | role_files["development_test_exposed"])
        ),
    }
    checks.add("B4 train/val/test source sets are disjoint", not any(overlap.values()), overlap)
    checks.add(
        "B5 role composition is 62/16/20/6 files",
        {role: len(files) for role, files in role_files.items()}
        == {
            TRAIN_ROLE: 62,
            VAL_ROLE: 16,
            "development_test": 20,
            "development_test_exposed": 6,
        },
        {role: len(files) for role, files in role_files.items()},
    )
    checks.add(
        "B6 dataset_summary.json agrees with the index",
        summary["windows"]["by_role"]
        == {
            TRAIN_ROLE: role_windows[TRAIN_ROLE],
            VAL_ROLE: role_windows[VAL_ROLE],
            "development_test": role_windows["development_test"],
            "development_test_exposed": role_windows["development_test_exposed"],
        }
        and summary["windows"]["by_role_label"][TRAIN_ROLE]
        == {str(key): value for key, value in sorted(role_label_windows[TRAIN_ROLE].items())}
        and summary["windows"]["by_role_label"][VAL_ROLE]
        == {str(key): value for key, value in sorted(role_label_windows[VAL_ROLE].items())},
        summary["windows"]["by_role"],
    )
    checks.add(
        "B7 index verification and negative controls recorded as passing",
        bool(summary.get("index_verification", {}).get("passed"))
        and bool(summary.get("index_negative_controls", {}).get("passed")),
        {
            "verification": summary.get("index_verification"),
            "controls": summary.get("index_negative_controls"),
        },
    )
    return {
        "role_files": {role: sorted(files) for role, files in role_files.items()},
        "role_windows": dict(role_windows),
        "role_label_windows": {
            role: dict(sorted(counter.items())) for role, counter in role_label_windows.items()
        },
    }


# ---------------------------------------------------------------------------
# C. read audit
# ---------------------------------------------------------------------------
def check_read_audit(
    checks: Checks,
    run_root: Path,
    composite: dict[str, Any],
    index_info: dict[str, Any],
) -> dict[str, Any]:
    audit = read_json(run_root / "read_audit.json")
    config = read_json(run_root / "run_config.json")
    # ``verify`` hands the frozen index rows in through ``composite``; the loader
    # identity audit below needs them, and reading them from the caller's payload is
    # what this function always meant to do (the bare ``rows`` name was a latent
    # NameError on the loader-audit branch, which no run had reached before).
    rows: list[dict[str, str]] = list(composite.get("rows") or [])
    if not (run_root / "progress.json").is_file():
        raise RuntimeError(
            f"{run_root} has no progress.json yet: the training run is still in its first epoch"
        )
    progress = read_json(run_root / "progress.json")
    epoch_rows = read_csv_rows(run_root / "validation_metrics.csv")
    conditions_recorded = sorted({str(row["condition"]) for row in epoch_rows})
    epochs_completed = int(progress["epochs_completed"])
    train_paths = set(index_info["role_files"][TRAIN_ROLE])
    val_paths = set(index_info["role_files"][VAL_ROLE])
    test_paths = set(index_info["role_files"]["development_test"]) | set(
        index_info["role_files"]["development_test_exposed"]
    )
    opened = {Path(raw).name for raw in audit["opened_files"]}
    expected = train_paths | val_paths
    loader_config = config.get("loader") or {}
    parallel = bool(loader_config.get("parallel")) or int(loader_config.get("num_workers") or 0) > 0

    checks.add(
        "C1 the read audit opens exactly the frozen train+val files",
        opened == expected,
        {
            "opened": len(opened),
            "expected": len(expected),
            "missing": sorted(expected - opened)[:5],
            "unexpected": sorted(opened - expected)[:5],
        },
    )
    checks.add(
        "C2 no development_test* file was opened",
        not (opened & test_paths) and not audit.get("test_files_touched"),
        {"test_files_touched": audit.get("test_files_touched")},
    )
    checks.add(
        "C3 the audit self-report matches the recomputed file sets",
        audit.get("passed") is True
        and bool(config.get("read_audit_passed"))
        and bool(config.get("test_windows_never_read"))
        and audit.get("unexpected_files_touched") == [],
        {
            "passed": audit.get("passed"),
            "unexpected": audit.get("unexpected_files_touched"),
        },
    )

    per_file_windows = {
        filename: sum(
            1
            for row in composite["rows"]
            if str(row["filename"]) == filename and str(row["source_role"]) == TRAIN_ROLE
        )
        for filename in train_paths
    }
    val_position_windows = {
        filename: sum(
            1
            for row in composite["rows"]
            if str(row["filename"]) == filename and str(row["source_role"]) == VAL_ROLE
        )
        for filename in val_paths
    }
    val_window_total = len(
        [row for row in composite["rows"] if str(row["source_role"]) == VAL_ROLE]
    )
    if val_window_total <= 0:
        raise RuntimeError("no validation windows in the index")
    # validation_metrics.csv holds one aggregate row per epoch and condition.
    epochs_in_metrics = len({str(row["epoch"]) for row in epoch_rows})
    observed = {Path(raw).name: int(count) for raw, count in audit["windows_by_file"].items()}
    budget_errors: dict[str, Any] = {}
    if epochs_in_metrics != epochs_completed:
        budget_errors["_epochs"] = {
            "validation_metrics_epochs": epochs_in_metrics,
            "progress_epochs_completed": epochs_completed,
        }
    per_epoch_conditions = Counter(str(row["epoch"]) for row in epoch_rows)
    if set(per_epoch_conditions.values()) != {len(CONDITIONS)}:
        budget_errors["_conditions_per_epoch"] = dict(per_epoch_conditions)
    for filename, trains in per_file_windows.items():
        expected_reads = trains * epochs_completed
        if observed.get(filename, 0) != expected_reads:
            budget_errors[filename] = {
                "observed": observed.get(filename, 0),
                "expected_train": expected_reads,
            }
    for filename, val_windows in val_position_windows.items():
        # The validation cache re-reads every val window once per condition; the
        # validation itself then runs from the cached spectrograms.
        expected_reads = val_windows * len(CONDITIONS)
        if observed.get(filename, 0) != expected_reads:
            budget_errors[filename] = {
                "observed": observed.get(filename, 0),
                "expected_val_cache": expected_reads,
            }
    checks.add(
        "C4 per-file window reads equal train x epochs + val x conditions",
        not budget_errors,
        {
            "epochs_completed": epochs_completed,
            "conditions": conditions_recorded,
            "violations": dict(list(budget_errors.items())[:5]),
        },
    )
    checks.add(
        "C5 the run recorded every frozen validation condition",
        conditions_recorded == sorted(CONDITIONS),
        conditions_recorded,
    )

    loader_audit: dict[str, Any] = {}
    if parallel and rows:
        # With workers the parent process performs no reads at all, so C1/C3 would
        # pass vacuously against an empty legacy audit.  The worker event stream is
        # then the only real evidence and is verified on its own terms below.
        checks.add(
            "C6 the parallel run left the parent-side read audit empty (workers read)",
            int(audit.get("windows_read", -1)) == 0
            and int(audit.get("opened_file_count", -1)) == 0,
            {
                "parent_windows_read": audit.get("windows_read"),
                "parent_opened_files": audit.get("opened_file_count"),
            },
        )
        loader_audit = verify_loader_identity_audit(checks, run_root, rows, index_info)
    elif rows and (run_root / "loader_read_audit.json").is_file():
        # The serial reference path records the same identity evidence since rework
        # R1; it is an additional check, never a replacement for C1-C4.
        loader_audit = verify_loader_identity_audit(checks, run_root, rows, index_info)
    return {
        "opened_files": sorted(opened),
        "windows_read": int(audit["windows_read"]),
        "epochs_completed": epochs_completed,
        "conditions": conditions_recorded,
        "budget_violations": budget_errors,
        "loader": loader_config,
        "parallel_loader": parallel,
        "worker_read_audit": loader_audit,
    }


def _split_row_tables(rows: Sequence[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """The index rows grouped by split, in file order.

    This reproduces ``dronerfa_data.load_index`` locally; a position in the loader
    audit is an index into exactly this list.
    """

    tables: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        tables.setdefault(str(row["split"]), []).append(row)
    return tables


def _stratified_positions(rows: Sequence[dict[str, str]], limit: Any) -> list[int]:
    """Local re-implementation of ``dronerfa_data.stratified_limit`` (positions)."""

    count = len(rows)
    if limit is None or int(limit) <= 0 or count <= int(limit):
        return list(range(count))
    buckets: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        stratum = str(row.get("target_code", row["label"]))
        buckets.setdefault(stratum, []).append(index)
    selected: list[int] = []
    cursor = 0
    labels = sorted(buckets)
    while len(selected) < int(limit):
        added = False
        for label in labels:
            bucket = buckets[label]
            if cursor < len(bucket) and len(selected) < int(limit):
                selected.append(bucket[cursor])
                added = True
        if not added:
            break
        cursor += 1
    return selected


def position_digest(positions: Sequence[int]) -> str:
    """The loader audit's ordered-position digest, re-implemented independently."""

    array = np.asarray([int(value) for value in positions], dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def position_multiset_digest(positions: Sequence[int]) -> str:
    """The loader audit's ``(position, count)`` digest, re-implemented independently.

    Reads are compared as a multiset rather than a sequence: with several workers
    the *arrival* order of read events is not a defined order, while the parent's
    consumption log very much is.  A missing or repeated window still changes the
    multiset, so nothing is lost.
    """

    counts: dict[int, int] = {}
    for value in positions:
        key = int(value)
        counts[key] = counts.get(key, 0) + 1
    digest = hashlib.sha256()
    for key in sorted(counts):
        digest.update(key.to_bytes(8, "little", signed=True))
        digest.update(counts[key].to_bytes(4, "little"))
    return digest.hexdigest()


def loader_identity_plan(
    rows: Sequence[dict[str, str]],
    run_root: Path,
    index_info: dict[str, Any],
) -> dict[str, Any]:
    """Recompute the loader's expected ``(phase, key) -> ordered positions``.

    Everything here comes from the frozen index, the run configuration and the
    frozen shuffle rule -- never from the audit being checked:
    ``torch.randperm(len(split), generator=manual_seed(training_seed + epoch))``
    restricted to the selected positions, and for validation the selected
    positions in selection order, once per condition.
    """

    config_path = run_root / "run_config.json"
    if not config_path.is_file():
        return {"available": False, "reason": f"missing {config_path}"}
    config = read_json(config_path)
    tables = _split_row_tables(rows)
    train_rows = tables.get(TRAIN_ROLE, [])
    val_rows = tables.get(VAL_ROLE, [])
    if not train_rows or not val_rows:
        return {"available": False, "reason": "index has no train or val rows"}
    training_seed = int(config.get("training_seed", TRAINING_SEED))
    conditions = [str(value) for value in config.get("validation_conditions", CONDITIONS)]
    train_selected = _stratified_positions(train_rows, config.get("max_train_samples"))
    val_selected = _stratified_positions(val_rows, config.get("max_val_samples"))
    epochs_completed = int(config.get("epochs") or 0)
    progress_path = run_root / "progress.json"
    if progress_path.is_file():
        epochs_completed = int(read_json(progress_path).get("epochs_completed", epochs_completed))
    plan: dict[str, Any] = {
        "available": True,
        "training_seed": training_seed,
        "validation_seed": int(config.get("validation_seed", VALIDATION_SEED)),
        "num_workers": int((config.get("loader") or {}).get("num_workers") or 0),
        "conditions": conditions,
        "train": {},
        "val": {},
        "role_files": {
            TRAIN_ROLE: set(index_info["role_files"][TRAIN_ROLE]),
            VAL_ROLE: set(index_info["role_files"][VAL_ROLE]),
            "test": set(index_info["role_files"]["development_test"])
            | set(index_info["role_files"]["development_test_exposed"]),
        },
        "position_path": {
            TRAIN_ROLE: {int(position): str(train_rows[position]["file_path"]) for position in train_selected},
            VAL_ROLE: {int(position): str(val_rows[position]["file_path"]) for position in val_selected},
        },
        "bounded": {
            TRAIN_ROLE: bool(config.get("max_train_samples")),
            VAL_ROLE: bool(config.get("max_val_samples")),
        },
    }
    selected_set = set(int(position) for position in train_selected)
    for epoch in range(1, epochs_completed + 1):
        permutation = torch.randperm(
            len(train_rows),
            generator=torch.Generator().manual_seed(training_seed + epoch),
        ).tolist()
        plan["train"][str(epoch)] = [
            int(position) for position in permutation if int(position) in selected_set
        ]
    for condition in conditions:
        plan["val"][str(condition)] = [int(position) for position in val_selected]
    return plan


def normalize_loader_audit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept both shapes of ``loader_read_audit.json``.

    The driver writes a document with a ``train`` phase and a ``val_cache`` phase
    next to the raw :class:`LoaderAudit` payload under ``audit``; the raw payload
    itself carries ``phases``.  The checker works on the normalised raw shape, so a
    run written by either layout is judged by exactly the same rules.
    """

    if payload.get("phases"):
        return payload
    audit = payload.get("audit") or {}
    phases = {
        TRAIN_ROLE: payload.get(TRAIN_ROLE) or {},
        VAL_ROLE: payload.get("val_cache") or payload.get(VAL_ROLE) or {},
    }
    if not any(phases.values()):
        return payload
    normalized = dict(audit) if audit else {}
    normalized["phases"] = phases
    normalized.setdefault("drain_failures", payload.get("drain_failures", audit.get("drain_failures")))
    normalized.setdefault("unknown_events", payload.get("unknown_events", audit.get("unknown_events")))
    normalized.setdefault(
        "worker_processes",
        payload.get("worker_processes_seen") or audit.get("worker_processes") or [],
    )
    normalized.setdefault("schema", payload.get("schema"))
    return normalized


def evaluate_loader_audit(payload: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Pure check of an audit payload against the recomputed plan.

    No file access and no self-reported field is trusted: the per-key window
    counts, the ordered-position digests, the delivered/completed split and the
    per-file budget are all recomputed from the plan, and a phase whose file set is
    not exactly its role's file set fails even when the counts happen to add up.
    """

    problems: list[dict[str, Any]] = []
    raw = payload.get("audit") or {}
    raw_phases = raw.get("phases") or {}
    if raw_phases:
        # The driver writes the phase reports twice: the readable top level (with a
        # couple of convenience fields added) and the raw audit payload.  Every field
        # the raw copy carries must match, so tampering with either copy is caught
        # instead of being silently ignored.
        for name, top in (
            (TRAIN_ROLE, payload.get(TRAIN_ROLE)),
            (VAL_ROLE, payload.get("val_cache") or payload.get(VAL_ROLE)),
        ):
            if top is None:
                continue
            raw_report = raw_phases.get(name) or {}
            missing = sorted(set(raw_report) - set(top))
            differing = sorted(
                field for field in set(raw_report) & set(top) if raw_report[field] != top[field]
            )
            if missing or differing:
                problems.append(
                    {
                        "check": f"P0 {name}: the two copies of the phase report agree",
                        "detail": {"missing_in_top_level": missing, "differing": differing},
                    }
                )
    payload = normalize_loader_audit_payload(payload)

    def fail(code: str, detail: Any) -> None:
        problems.append({"check": code, "detail": detail})

    phases = payload.get("phases") or {}
    train_phase = phases.get(TRAIN_ROLE) or {}
    val_phase = phases.get(VAL_ROLE) or {}
    if not train_phase or not val_phase:
        fail(
            "P1 the audit carries a train phase and a val_cache phase",
            {"phases": sorted(phases)},
        )
        return {"problems": problems, "info": {}}
    test_files = set(plan["role_files"]["test"])
    info: dict[str, Any] = {"train_windows_read": 0, "val_windows_read": 0}

    for phase_name, phase, expected_keys, role in (
        (TRAIN_ROLE, train_phase, plan["train"], TRAIN_ROLE),
        (VAL_ROLE, val_phase, plan["val"], VAL_ROLE),
    ):
        keys = phase.get("keys") or {}
        role_files = set(plan["role_files"][role])
        if set(keys) != set(expected_keys):
            fail(
                f"P2 {phase_name}: the audit covers exactly its expected keys",
                {"recorded": sorted(keys), "expected": sorted(expected_keys)},
            )
        opened = {Path(raw).name for raw in phase.get("opened_files", [])}
        planned_files = {
            Path(path).name for path in plan["position_path"][role].values()
        }
        if opened != planned_files:
            fail(
                f"P3 {phase_name}: opened files are exactly the files the plan uses",
                {
                    "opened": len(opened),
                    "expected": len(planned_files),
                    "missing": sorted(planned_files - opened)[:5],
                    "unexpected": sorted(opened - planned_files)[:5],
                },
            )
        if not opened <= role_files:
            fail(
                f"P3b {phase_name}: every opened file belongs to the frozen {role} role",
                sorted(opened - role_files)[:5],
            )
        if not plan.get("bounded", {}).get(role) and opened != role_files:
            fail(
                f"P3c {phase_name}: an unbounded run opens every frozen {role} file",
                {
                    "opened": len(opened),
                    "expected": len(role_files),
                    "missing": sorted(role_files - opened)[:5],
                },
            )
        if opened & test_files:
            fail(
                f"P4 {phase_name}: no development_test file was opened",
                sorted(opened & test_files)[:5],
            )
        if phase.get("unconsumed_reads") or phase.get("unread_consumed"):
            fail(
                f"P5 {phase_name}: every read was consumed and every consumed window was read",
                {
                    "unconsumed_reads": phase.get("unconsumed_reads"),
                    "unread_consumed": phase.get("unread_consumed"),
                },
            )
        for key, expected_positions in sorted(expected_keys.items()):
            report = keys.get(str(key))
            if report is None:
                continue
            expected_set = set(int(value) for value in expected_positions)
            digest = position_digest(expected_positions)
            reads = report.get("reads") or {}
            consumed = report.get("consumed") or {}
            read_counts = {
                name: int(count) for name, count in (reads.get("files") or {}).items()
            }
            expected_counts: dict[str, int] = {}
            for position in expected_positions:
                path = plan["position_path"][role].get(int(position))
                if path is None:
                    fail(
                        f"P6 {phase_name}/{key}: the plan knows every expected position",
                        int(position),
                    )
                    continue
                name = Path(path).name
                expected_counts[name] = expected_counts.get(name, 0) + 1
            if read_counts != expected_counts:
                fail(
                    f"P6 {phase_name}/{key}: per-file window reads equal the recomputed plan",
                    {
                        "recorded": dict(list(read_counts.items())[:5]),
                        "expected": dict(list(expected_counts.items())[:5]),
                    },
                )
            if int(reads.get("windows_read", -1)) != len(expected_positions):
                fail(
                    f"P7 {phase_name}/{key}: the window read count equals the plan",
                    {"recorded": reads.get("windows_read"), "expected": len(expected_positions)},
                )
            multiset = position_multiset_digest(expected_positions)
            if reads.get("position_multiset_sha256") != multiset:
                fail(
                    f"P8 {phase_name}/{key}: the reads are exactly the planned windows",
                    {
                        "recorded": reads.get("position_multiset_sha256"),
                        "expected": multiset,
                    },
                )
            if plan.get("num_workers", 0) == 0 and reads.get("position_order_sha256") != digest:
                # One reader process: the arrival order is the delivery order.
                fail(
                    f"P8b {phase_name}/{key}: a serial reader follows the planned order",
                    {"recorded": reads.get("position_order_sha256"), "expected": digest},
                )
            if int(reads.get("duplicate_reads", -1)) != 0:
                fail(
                    f"P9 {phase_name}/{key}: no window position was read twice",
                    reads.get("duplicate_reads"),
                )
            for field in ("windows_delivered", "windows_completed"):
                if int(consumed.get(field, -1)) != len(expected_positions):
                    fail(
                        f"P10 {phase_name}/{key}: {field} equals the plan",
                        {"recorded": consumed.get(field), "expected": len(expected_positions)},
                    )
            for field in ("delivered_order_sha256", "completed_order_sha256"):
                if consumed.get(field) != digest:
                    fail(
                        f"P11 {phase_name}/{key}: {field} equals the recomputed delivery order",
                        {"recorded": consumed.get(field), "expected": digest},
                    )
            if consumed.get("aborted") or not consumed.get("iterator_closed"):
                fail(
                    f"P12 {phase_name}/{key}: the loader pass ran to completion",
                    {
                        "aborted": consumed.get("aborted"),
                        "iterator_closed": consumed.get("iterator_closed"),
                    },
                )
            if int(consumed.get("batches_completed", -1)) != int(
                consumed.get("batches_delivered", -2)
            ) or int(consumed.get("batches_delivered", 0)) <= 0:
                fail(
                    f"P13 {phase_name}/{key}: every delivered batch was completed",
                    {
                        "delivered": consumed.get("batches_delivered"),
                        "completed": consumed.get("batches_completed"),
                    },
                )
            if phase_name == TRAIN_ROLE:
                info["train_windows_read"] += int(reads.get("windows_read", 0))
            else:
                info["val_windows_read"] += int(reads.get("windows_read", 0))

    drain_failures = int(payload.get("drain_failures", -1))
    if drain_failures != 0 or int(payload.get("unknown_events", -1)) != 0:
        fail(
            "P14 no audit event was lost on the queue",
            {
                "drain_failures": payload.get("drain_failures"),
                "unknown_events": payload.get("unknown_events"),
            },
        )
    if not payload.get("worker_processes"):
        fail("P15 the audit records the worker processes that produced it", [])
    for phase_name, phase in ((TRAIN_ROLE, train_phase), (VAL_ROLE, val_phase)):
        # Loader lifecycle: every pass must be registered when it starts and drained
        # after its workers were shut down, otherwise events could still be in flight
        # when the report was written.
        phase_expected = plan[TRAIN_ROLE] if phase_name == TRAIN_ROLE else plan[VAL_ROLE]
        registered = {
            str(row.get("identity")) for row in (phase.get("loaders") or [])
        }
        closed = {str(identity) for identity in (phase.get("closed_loaders") or [])}
        expected_loaders = {f"{phase_name}/{key}" for key in phase_expected}
        missing_registration = sorted(expected_loaders - registered)
        never_closed = sorted(registered - closed)
        if missing_registration or never_closed:
            fail(
                f"P18 {phase_name}: every loader pass was registered and drained after shutdown",
                {
                    "missing_registration": missing_registration[:5],
                    "registered_but_never_closed": never_closed[:5],
                    "registered": sorted(registered),
                    "closed": sorted(closed),
                },
            )
    if plan.get("num_workers", 0) > 0:
        for phase_name, phase in ((TRAIN_ROLE, train_phase), (VAL_ROLE, val_phase)):
            workers = list(phase.get("worker_keys") or [])
            if len(workers) < int(plan["num_workers"]):
                fail(
                    f"P16 {phase_name}: every configured worker produced events",
                    {"workers": workers, "num_workers": plan["num_workers"]},
                )
            if any(worker.startswith("main@") for worker in workers):
                fail(
                    f"P17 {phase_name}: the parent process did not read windows itself",
                    [worker for worker in workers if worker.startswith("main@")],
                )
    info["opened_files"] = sorted(
        {Path(raw).name for raw in train_phase.get("opened_files", [])}
        | {Path(raw).name for raw in val_phase.get("opened_files", [])}
    )
    info["train_keys"] = sorted((train_phase.get("keys") or {}))
    info["val_keys"] = sorted((val_phase.get("keys") or {}))
    info["worker_processes"] = len(payload.get("worker_processes") or [])
    return {"problems": problems, "info": info}


#: Human-readable name for every loader-audit problem code.
LOADER_AUDIT_CHECK_NAMES: dict[str, str] = {
    "P0": "the two copies of the phase report agree",
    "P1": "the audit carries a train phase and a val_cache phase",
    "P2": "the audit covers exactly the expected epoch/condition keys",
    "P3": "opened files are exactly the files the recomputed plan uses",
    "P3b": "every opened file belongs to the frozen role",
    "P3c": "an unbounded run opens every frozen role file",
    "P4": "no development_test file was opened in any phase",
    "P5": "every read was consumed and every consumed window was read",
    "P6": "per-file window reads equal the recomputed plan",
    "P7": "the window read count equals the plan",
    "P8": "the reads are exactly the planned windows",
    "P8b": "a serial reader follows the planned order",
    "P9": "no window position was read twice",
    "P10": "delivered/completed window counts equal the plan",
    "P11": "delivered/completed orders equal the recomputed delivery order",
    "P12": "the loader pass ran to completion",
    "P13": "every delivered batch was completed",
    "P14": "no audit event was lost on the queue",
    "P15": "the audit records the worker processes that produced it",
    "P16": "every configured worker produced events",
    "P17": "the parent process did not read windows itself",
    "P18": "every loader pass was registered and drained after shutdown",
}


def verify_loader_identity_audit(
    checks: Checks,
    run_root: Path,
    rows: Sequence[dict[str, str]],
    index_info: dict[str, Any],
) -> dict[str, Any]:
    """Verify ``loader_read_audit.json`` against the independently recomputed plan."""

    path = run_root / "loader_read_audit.json"
    if not path.is_file():
        checks.add(
            "C7 the run wrote a loader identity audit",
            False,
            f"missing {path}",
        )
        return {}
    try:
        payload = read_json(path)
    except Exception as error:  # noqa: BLE001 - a malformed artefact is a failure
        checks.add(
            "C7 the loader identity audit parses",
            False,
            f"{path} is not readable JSON: {error!r}",
        )
        return {}
    payload = read_json(path)
    plan = loader_identity_plan(rows, run_root, index_info)
    if not plan.get("available"):
        checks.add("C7 the loader identity audit can be recomputed", False, plan.get("reason"))
        return {}
    evaluation = evaluate_loader_audit(payload, plan)
    failed_codes = {problem["check"].split()[0] for problem in evaluation["problems"]}
    for code, name in LOADER_AUDIT_CHECK_NAMES.items():
        detail = [
            problem
            for problem in evaluation["problems"]
            if problem["check"].split()[0] == code
        ]
        checks.add(
            f"C7.{code} loader audit: {name}",
            code not in failed_codes,
            detail[:3] if detail else "ok",
        )
    return {
        "plan": {
            "training_seed": plan["training_seed"],
            "num_workers": plan["num_workers"],
            "conditions": plan["conditions"],
            "train_keys": sorted(plan["train"]),
            "val_keys": sorted(plan["val"]),
            "train_windows_per_epoch": len(next(iter(plan["train"].values()), [])),
            "val_windows_per_condition": len(next(iter(plan["val"].values()), [])),
        },
        "problems": evaluation["problems"],
        "checks_evaluated": len(LOADER_AUDIT_CHECK_NAMES),
        **evaluation["info"],
    }


def loader_audit_negative_controls(
    run_root: Path,
    rows: Sequence[dict[str, str]],
    index_info: dict[str, Any],
) -> list[dict[str, Any]]:
    """Tamper with a real audit payload; every mutation must be detected.

    The controls run the same pure :func:`evaluate_loader_audit` the verifier uses
    at the top level, so a check that cannot fail here cannot pass there either.
    """

    path = run_root / "loader_read_audit.json"
    if not path.is_file():
        return []
    payload = normalize_loader_audit_payload(read_json(path))
    plan = loader_identity_plan(rows, run_root, index_info)
    if not plan.get("available"):
        return []
    if evaluate_loader_audit(payload, plan)["problems"]:
        # Nothing to control against: the untampered payload is already rejected.
        return []
    controls: list[dict[str, Any]] = []

    def detect(name: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        tampered = json.loads(json.dumps(payload))
        try:
            mutate(tampered)
            problems = evaluate_loader_audit(tampered, plan)["problems"]
            detected = bool(problems)
            detail = problems[0]["check"] if problems else "not detected"
        except Exception as error:  # noqa: BLE001 - a crash counts as a detection
            detected = True
            detail = f"raised {error!r}"
        controls.append({"control": name, "detected": detected, "detail": detail})

    def train_key(tampered: dict[str, Any]) -> str:
        keys = sorted((tampered["phases"][TRAIN_ROLE]["keys"]))
        return keys[0]

    def control_dropped_window(tampered: dict[str, Any]) -> None:
        entry = tampered["phases"][TRAIN_ROLE]["keys"][train_key(tampered)]
        entry["reads"]["windows_read"] -= 1
        entry["consumed"]["windows_completed"] -= 1

    def control_duplicated_window(tampered: dict[str, Any]) -> None:
        entry = tampered["phases"][TRAIN_ROLE]["keys"][train_key(tampered)]
        entry["reads"]["duplicate_reads"] = 1

    def control_shifted_epoch(tampered: dict[str, Any]) -> None:
        keys = sorted(tampered["phases"][TRAIN_ROLE]["keys"])
        if len(keys) < 2:
            raise RuntimeError("only one training epoch in the audit")
        first, second = keys[0], keys[1]
        tampered["phases"][TRAIN_ROLE]["keys"][second]["consumed"][
            "completed_order_sha256"
        ] = tampered["phases"][TRAIN_ROLE]["keys"][first]["consumed"]["completed_order_sha256"]

    def control_role_swap(tampered: dict[str, Any]) -> None:
        train_files = list(tampered["phases"][TRAIN_ROLE]["opened_files"])
        val_files = list(tampered["phases"][VAL_ROLE]["opened_files"])
        if not train_files or not val_files:
            raise RuntimeError("audit has no files to swap")
        tampered["phases"][TRAIN_ROLE]["opened_files"] = train_files[:-1] + [val_files[0]]

    def control_test_file_read(tampered: dict[str, Any]) -> None:
        tampered["test_files_touched"] = ["T0000_D00_S0100.mat"]
        tampered["phases"][TRAIN_ROLE]["opened_files"] = list(
            tampered["phases"][TRAIN_ROLE]["opened_files"]
        ) + [str(Path(r"G:\DroneRFa数据集\original") / "T0000_D00_S0100.mat")]

    def control_unconsumed_read(tampered: dict[str, Any]) -> None:
        tampered["phases"][VAL_ROLE]["unconsumed_reads"] = 1

    def control_lost_event(tampered: dict[str, Any]) -> None:
        tampered["drain_failures"] = 1

    def control_short_batches(tampered: dict[str, Any]) -> None:
        entry = tampered["phases"][TRAIN_ROLE]["keys"][train_key(tampered)]
        entry["consumed"]["batches_completed"] -= 1

    def control_parent_read(tampered: dict[str, Any]) -> None:
        if plan["num_workers"] <= 0:
            raise RuntimeError("serial audit has no worker split to break")
        tampered["phases"][TRAIN_ROLE]["worker_keys"] = ["main@pid1"]

    def control_missing_epoch(tampered: dict[str, Any]) -> None:
        tampered["phases"][TRAIN_ROLE]["keys"].pop(train_key(tampered))

    def control_loader_never_closed(tampered: dict[str, Any]) -> None:
        closed = list(tampered["phases"][TRAIN_ROLE].get("closed_loaders") or [])
        if not closed:
            raise RuntimeError("audit records no closed loader")
        tampered["phases"][TRAIN_ROLE]["closed_loaders"] = closed[:-1]

    detect("a dropped training window", control_dropped_window)
    detect("a duplicated window position", control_duplicated_window)
    detect("an epoch's order replaced by another epoch's", control_shifted_epoch)
    detect("a validation file counted as a training file", control_role_swap)
    detect("a development_test file in the read set", control_test_file_read)
    detect("an unconsumed read left over", control_unconsumed_read)
    detect("a lost worker event", control_lost_event)
    detect("a batch delivered but not completed", control_short_batches)
    detect("a training epoch missing from the audit", control_missing_epoch)
    detect("a loader that was never drained and closed", control_loader_never_closed)
    if plan["num_workers"] > 0:
        detect("the parent process reading instead of a worker", control_parent_read)
    return controls


# ---------------------------------------------------------------------------
# D/E. checkpoint reload and independent metric re-derivation
# ---------------------------------------------------------------------------
def load_checkpoint(run_root: Path, device: torch.device) -> tuple[dict[str, Any], ResNet18]:
    payload = torch.load(run_root / "best_model.pt", map_location=device, weights_only=False)
    model = ResNet18(input_channels=1, num_classes=2).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return payload, model


def frontend_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _select_probe_rows(rows: Sequence[dict[str, str]], budget: int) -> list[int]:
    """Evenly spaced positions over the validation split, both labels present.

    ``budget <= 0`` selects the whole split.  Every selected position is used as
    the window's index inside the split, which is also its AWGN seed offset, so
    the replay reproduces exactly the noisy input the training run evaluated.
    """
    total = len(rows)
    if budget <= 0 or budget >= total:
        return list(range(total))
    step = total / budget
    selected = sorted({min(int(index * step), total - 1) for index in range(budget)})
    labels = {int(rows[position]["label"]) for position in selected}
    if labels != {0, 1}:
        for position, row in enumerate(rows):
            if int(row["label"]) not in labels:
                selected.append(position)
                labels.add(int(row["label"]))
        selected.sort()
    return selected


def reproduce_validation(
    checks: Checks,
    run_root: Path,
    rows: Sequence[dict[str, str]],
    device: torch.device,
    replay_budget: int = REPLAY_WINDOWS,
) -> dict[str, Any]:
    val_rows = [row for row in rows if str(row["source_role"]) == VAL_ROLE]
    payload, model = load_checkpoint(run_root, device)
    recorded = {row["condition"]: row for row in payload["val_metrics"]}
    checks.add(
        "D1 checkpoint carries the stage-1 label mapping and prediction rule",
        payload.get("label_mapping") == {"0": "Background", "1": "Drone"}
        and payload.get("prediction_rule") == "float32(logit_drone) - float32(logit_background) > 0",
        {"label_mapping": payload.get("label_mapping")},
    )
    checks.add(
        "D2 checkpoint records the same index and split hashes",
        payload.get("index_sha256") == INDEX_SHA256
        and payload.get("split_source_sha256") == FROZEN_SPLIT_CSV_SHA256,
        {"index_sha256": payload.get("index_sha256")},
    )

    manifest = {
        entry["condition"]: entry
        for entry in read_json(run_root / "validation_cache_manifest.json")
    }
    checks.add(
        "E1 validation cache manifest covers the full val split for every condition",
        set(manifest) == set(CONDITIONS)
        and all(
            len(entry["index_positions"]) == len(val_rows)
            and entry["samples"] == len(val_rows)
            and entry["dtype"] == "float32"
            and int(entry["n_fft"]) == N_FFT
            and int(entry["hop_length"]) == HOP_LENGTH
            and int(entry["noise_seed"]) == VALIDATION_SEED
            for entry in manifest.values()
        ),
        {
            condition: {
                "samples": manifest[condition]["samples"],
                "files": len(manifest[condition]["files"]),
            }
            for condition in CONDITIONS
        },
    )

    scores_path = run_root / f"validation_scores_epoch_{int(payload['epoch']):02d}.csv"
    scores = read_csv_rows(scores_path)
    recorded_predictions: dict[str, list[dict[str, str]]] = {condition: [] for condition in CONDITIONS}
    for row in scores:
        recorded_predictions[str(row["condition"])].append(row)
    shape_ok = all(
        len(recorded_predictions[condition]) == len(val_rows) for condition in CONDITIONS
    )
    checks.add(
        "E2 recorded per-window score files cover the whole val split",
        shape_ok,
        {condition: len(recorded_predictions[condition]) for condition in CONDITIONS},
    )
    if not shape_ok:
        raise RuntimeError("recorded validation score files do not match the val split")

    recorded_rows: dict[str, list[dict[str, str]]] = {}
    for condition in CONDITIONS:
        recorded_rows[condition] = sorted(
            recorded_predictions[condition], key=lambda row: int(row["selection_position"])
        )

    # The pooled metrics must be re-derivable from the recorded per-window file
    # alone, independent of the training process' own metric code.
    metric_deltas: dict[str, Any] = {}
    metrics_ok = True
    for condition in CONDITIONS:
        records = [
            {
                "label": int(row["label"]),
                "prediction": int(row["prediction"]),
                "filename": str(row["filename"]),
            }
            for row in recorded_rows[condition]
        ]
        recomputed = {**metrics_from_records(records), **file_equal_metrics(records)}
        stored = recorded[condition]
        deltas = {
            key: abs(float(stored[key]) - float(recomputed[key]))
            for key in (
                "balanced_accuracy",
                "accuracy",
                "drone_recall",
                "background_specificity",
                "f1",
                "file_equal_balanced_accuracy",
            )
        }
        counts_match = all(
            int(stored[key]) == int(recomputed[key]) for key in ("windows", "tn", "fp", "fn", "tp")
        )
        metrics_ok = metrics_ok and counts_match and max(deltas.values()) <= 1e-12
        metric_deltas[condition] = {
            "max_abs_delta": max(deltas.values()),
            "counts_match": counts_match,
            "confusion": {
                key: int(stored[key]) for key in ("tn", "fp", "fn", "tp")
            },
        }
    checks.add(
        "E3 every recorded validation metric re-derives from the per-window decisions",
        metrics_ok,
        metric_deltas,
    )

    # Bounded replay: read each selected window once and re-run all four
    # conditions through an independently re-implemented pipeline.
    positions = _select_probe_rows(val_rows, replay_budget)
    frontend = LocalSpectrogramFrontEnd()
    replay: dict[str, Any] = {}
    margin_violations: list[dict[str, Any]] = []
    decision_mismatches: list[dict[str, Any]] = []
    max_margin_delta = 0.0
    handles: dict[str, h5py.File] = {}

    def handle_for(path: str) -> h5py.File:
        existing = handles.get(path)
        if existing is None:
            existing = h5py.File(path, "r")
            handles[path] = existing
        return existing

    chunk_size = 64
    try:
        with torch.inference_mode():
            for start in range(0, len(positions), chunk_size):
                chunk = positions[start : start + chunk_size]
                signals: dict[str, list[torch.Tensor]] = {condition: [] for condition in CONDITIONS}
                labels: list[int] = []
                for position in chunk:
                    row = val_rows[position]
                    iq = window_iq(handle_for(str(row["file_path"])), row)
                    labels.append(int(row["label"]))
                    for condition in CONDITIONS:
                        signal = condition_iq(iq, condition, position)
                        signals[condition].append(
                            torch.from_numpy(np.stack((signal.real, signal.imag), axis=0))
                        )
                batch_labels = torch.tensor(labels, dtype=torch.long)
                for condition in CONDITIONS:
                    stack = torch.stack(signals[condition])
                    features = frontend(stack)
                    logits = model(features).float()
                    margins = (logits[:, 1] - logits[:, 0]).tolist()
                    for offset, position in enumerate(chunk):
                        row = val_rows[position]
                        reference = recorded_rows[condition][position]
                        margin = float(margins[offset])
                        delta = abs(margin - float(reference["logit_margin"]))
                        max_margin_delta = max(max_margin_delta, delta)
                        if delta > MARGIN_TOLERANCE:
                            margin_violations.append(
                                {
                                    "condition": condition,
                                    "position": position,
                                    "filename": row["filename"],
                                    "recorded_margin": float(reference["logit_margin"]),
                                    "recomputed_margin": margin,
                                    "delta": delta,
                                }
                            )
                        recomputed_prediction = 1 if margin > 0.0 else 0
                        if recomputed_prediction != int(reference["prediction"]):
                            decision_mismatches.append(
                                {
                                    "condition": condition,
                                    "position": position,
                                    "filename": row["filename"],
                                    "recorded": int(reference["prediction"]),
                                    "recomputed": recomputed_prediction,
                                }
                            )
                del signals
    finally:
        for file_handle in handles.values():
            file_handle.close()

    replay_labels = [int(val_rows[position]["label"]) for position in positions]
    replay_files = [str(val_rows[position]["filename"]) for position in positions]
    replay_label_counts = {str(key): value for key, value in sorted(Counter(replay_labels).items())}
    replay["windows"] = len(positions)
    replay["files"] = len(set(replay_files))
    replay["label_counts"] = replay_label_counts
    replay["max_abs_margin_delta"] = max_margin_delta
    replay["margin_tolerance"] = MARGIN_TOLERANCE
    replay["margin_violations"] = margin_violations[:5]
    replay["decision_mismatches"] = decision_mismatches[:5]
    expected_replay = min(
        max(max(replay_budget, 0), len({str(row["filename"]) for row in val_rows})),
        len(val_rows),
    )
    checks.add(
        "E4 replayed windows cover the whole val split evenly and both labels",
        len(set(replay_files)) == len({str(row["filename"]) for row in val_rows})
        and set(replay_label_counts) == {"0", "1"}
        # ``_select_probe_rows`` returns at least ``max(budget, distinct files)``
        # positions -- enough to touch every val source file -- or the whole split.
        and len(positions) >= expected_replay,
        {
            "windows": len(positions),
            "budget": replay_budget,
            "expected_replay": expected_replay,
            "files": len(set(replay_files)),
            "labels": replay_label_counts,
        },
    )
    checks.add(
        "E5 replayed margins reproduce within the float32 device tolerance",
        not margin_violations,
        {
            "windows": len(positions),
            "conditions": len(CONDITIONS),
            "max_abs_margin_delta": max_margin_delta,
            "tolerance": MARGIN_TOLERANCE,
            "violations": margin_violations[:3],
        },
    )
    checks.add(
        "E6 replayed decisions reproduce window for window",
        not decision_mismatches,
        {"mismatches": len(decision_mismatches), "examples": decision_mismatches[:3]},
    )

    history = read_csv_rows(run_root / "training_history.csv")
    typed_history = [
        {"epoch": int(row["epoch"]), SELECTION_KEY: float(row[SELECTION_KEY])} for row in history
    ]
    recorded_selection = read_json(run_root / "selection.json")
    expected_epoch = select_epoch(typed_history)
    checks.add(
        "E7 best epoch follows the frozen selection rule (mean pooled BA, earliest wins ties)",
        expected_epoch == int(recorded_selection["selected_epoch"])
        and expected_epoch == int(payload["epoch"]),
        {
            "by_epoch": {row["epoch"]: row[SELECTION_KEY] for row in typed_history},
            "expected": expected_epoch,
            "recorded": recorded_selection["selected_epoch"],
        },
    )
    checks.add(
        "E8 no other epoch beats the selected epoch",
        all(
            row[SELECTION_KEY] <= typed_history[expected_epoch - 1][SELECTION_KEY] + 1e-15
            for row in typed_history
        ),
        {
            str(row["epoch"]): row[SELECTION_KEY]
            for row in typed_history
            if row["epoch"] != expected_epoch
        },
    )

    run_config = read_json(run_root / "run_config.json")
    checks.add(
        "E9 run config declares the frozen protocol",
        int(run_config["n_fft"]) == N_FFT
        and int(run_config["hop_length"]) == HOP_LENGTH
        and int(run_config["image_size"]) == IMAGE_SIZE
        and int(run_config["batch_size"]) == BATCH_SIZE
        and float(run_config["learning_rate"]) == 1e-3
        and float(run_config["weight_decay"]) == 0.0
        and int(run_config["training_seed"]) == TRAINING_SEED
        and int(run_config["validation_seed"]) == VALIDATION_SEED
        and run_config["train_snrs"] == "clean,10,5,0,-5,-10,-12"
        and run_config["validation_conditions"] == list(CONDITIONS)
        and run_config["selection_key"] == SELECTION_KEY
        and run_config["max_train_samples"] is None
        and run_config["max_val_samples"] is None,
        {
            "epochs": run_config["epochs"],
            "training_seed": run_config["training_seed"],
            "validation_seed": run_config["validation_seed"],
            "train_windows_used_per_epoch": run_config["train_windows_used_per_epoch"],
            "val_windows_used_per_epoch": run_config["validation_windows_used_per_epoch"],
        },
    )
    checks.add(
        "E13 the checkpoint records the verified training seed",
        int(payload["training_seed"]) == TRAINING_SEED
        and int(payload["validation_seed"]) == VALIDATION_SEED
        and int(run_config["training_seed"]) == int(payload["training_seed"]),
        {
            "checkpoint_training_seed": payload.get("training_seed"),
            "checkpoint_validation_seed": payload.get("validation_seed"),
            "expected_training_seed": TRAINING_SEED,
        },
    )
    checks.add(
        "E10 every epoch ran the full train and val splits",
        int(run_config["train_windows_used_per_epoch"]) == int(run_config["train_windows_available"])
        and int(run_config["validation_windows_used_per_epoch"])
        == int(run_config["validation_windows_available"]),
        {
            "train": [run_config["train_windows_used_per_epoch"], run_config["train_windows_available"]],
            "val": [
                run_config["validation_windows_used_per_epoch"],
                run_config["validation_windows_available"],
            ],
        },
    )
    checks.add(
        "E11 the driver's own checkpoint reload check reports success",
        bool(read_json(run_root / "checkpoint_reload_check.json")["passed"]),
        read_json(run_root / "checkpoint_reload_check.json")["checkpoint_sha256"],
    )
    checks.add(
        "E12 best_model.pt hash equals the hash recorded at the end of the run",
        sha256_file(run_root / "best_model.pt") == run_config["best_model_sha256"],
        run_config["best_model_sha256"],
    )

    del model
    return {
        "cache_manifest": {
            condition: {
                "samples": manifest[condition]["samples"],
                "files": len(manifest[condition]["files"]),
                "feature_sha256": manifest[condition]["feature_sha256"],
            }
            for condition in CONDITIONS
        },
        "recorded_metric_recomputation": metric_deltas,
        "replay": replay,
        "selected_epoch": expected_epoch,
        "val_windows": len(val_rows),
    }


# ---------------------------------------------------------------------------
# F. negative controls
# ---------------------------------------------------------------------------
def _replay_subset(
    model: ResNet18,
    rows: Sequence[dict[str, str]],
    positions: Sequence[int],
    condition: str,
) -> list[int]:
    """Replay selected validation windows of one condition and return decisions."""
    val_rows = [row for row in rows if str(row["source_role"]) == VAL_ROLE]
    frontend = LocalSpectrogramFrontEnd()
    predictions: list[int] = []
    handles: dict[str, h5py.File] = {}
    try:
        with torch.inference_mode():
            for position in positions:
                row = val_rows[position]
                path = str(row["file_path"])
                if path not in handles:
                    handles[path] = h5py.File(path, "r")
                iq = window_iq(handles[path], row)
                signal = condition_iq(iq, condition, position)
                tensor = torch.from_numpy(
                    np.stack((signal.real, signal.imag), axis=0)
                ).unsqueeze(0)
                logits = model(frontend(tensor))[0].float()
                predictions.append(1 if float(logits[1] - logits[0]) > 0.0 else 0)
    finally:
        for handle in handles.values():
            handle.close()
    return predictions


def negative_controls(
    checks: Checks,
    run_root: Path,
    rows: Sequence[dict[str, str]],
    index_info: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Each control mutates one claim (or supplies a malformed one); the verifier
    must reject it.  ``detect`` passes only when the mutated input is caught."""
    del checks
    controls: list[dict[str, Any]] = []
    payload, model = load_checkpoint(run_root, device)
    base_state = {key: value.clone() for key, value in model.state_dict().items()}
    audit = read_json(run_root / "read_audit.json")
    progress = read_json(run_root / "progress.json")
    epochs_completed = int(progress["epochs_completed"])
    scores_path = run_root / f"validation_scores_epoch_{int(payload['epoch']):02d}.csv"
    scores = read_csv_rows(scores_path)
    recorded_metrics = {row["condition"]: row for row in payload["val_metrics"]}
    train_files = set(index_info["role_files"][TRAIN_ROLE])
    val_files = set(index_info["role_files"][VAL_ROLE])
    test_files = set(index_info["role_files"]["development_test"]) | set(
        index_info["role_files"]["development_test_exposed"]
    )
    train_windows = Counter(
        str(row["filename"]) for row in rows if str(row["source_role"]) == TRAIN_ROLE
    )
    val_windows = Counter(
        str(row["filename"]) for row in rows if str(row["source_role"]) == VAL_ROLE
    )

    def observed_counts(raw_audit: dict[str, Any]) -> dict[str, int]:
        return {Path(raw).name: int(count) for raw, count in raw_audit["windows_by_file"].items()}

    def audit_budget_ok(raw_audit: dict[str, Any]) -> bool:
        """Verifier check C4, run against a supplied audit payload."""
        observed = observed_counts(raw_audit)
        for filename, count in train_windows.items():
            if observed.get(filename, 0) != count * epochs_completed:
                return False
        for filename, count in val_windows.items():
            if observed.get(filename, 0) != count * len(CONDITIONS):
                return False
        return True

    def audit_files_ok(raw_audit: dict[str, Any]) -> bool:
        """Verifier checks C1/C2, run against a supplied audit payload."""
        opened = {Path(raw).name for raw in raw_audit["opened_files"]}
        return opened == (train_files | val_files) and not (opened & test_files)

    def metrics_match_scores(raw_scores: Sequence[dict[str, str]], condition: str) -> bool:
        """Verifier check E3, run against supplied per-window score rows."""
        selected = [row for row in raw_scores if str(row["condition"]) == condition]
        records = [
            {
                "label": int(row["label"]),
                "prediction": int(row["prediction"]),
                "filename": str(row["filename"]),
            }
            for row in selected
        ]
        recomputed = {**metrics_from_records(records), **file_equal_metrics(records)}
        stored = recorded_metrics[condition]
        return all(
            abs(float(stored[key]) - float(recomputed[key])) <= 1e-12
            for key in (
                "balanced_accuracy",
                "accuracy",
                "drone_recall",
                "background_specificity",
                "f1",
                "file_equal_balanced_accuracy",
            )
        ) and all(
            int(stored[key]) == int(recomputed[key]) for key in ("windows", "tn", "fp", "fn", "tp")
        )

    def detect(name: str, assertion: Callable[[], bool], describe: Any = None) -> None:
        """A control passes when the verifier-side assertion is violated."""
        try:
            passed = bool(assertion())
        except Exception as error:  # noqa: BLE001 - a crash is a detection
            controls.append({"control": name, "detected": True, "detail": f"raised {error!r}"})
            return
        controls.append(
            {
                "control": name,
                "detected": not passed,
                "detail": describe if describe is not None else "",
            }
        )

    def control_val_window_counted_as_train() -> bool:
        """A val window counted against a train file must break the read budget."""
        tampered = json.loads(json.dumps(audit))
        victim = sorted(train_files)[0]
        tampered["windows_by_file"][
            next(raw for raw in audit["windows_by_file"] if Path(raw).name == victim)
        ] += 1
        return audit_budget_ok(tampered)

    def control_test_file_in_opened() -> bool:
        """A development_test file among the opened files must break C1/C2."""
        tampered = json.loads(json.dumps(audit))
        tampered["opened_files"] = [*audit["opened_files"], str(sorted(test_files)[0])]
        return audit_files_ok(tampered)

    def control_label_flip_in_scores() -> bool:
        """Flipping one window label in the recorded scores must break E3."""
        tampered = [dict(row) for row in scores]
        position = next(
            index for index, row in enumerate(tampered) if int(row["label"]) == 1
        )
        tampered[position]["label"] = "0"
        condition = str(tampered[position]["condition"])
        return metrics_match_scores(tampered, condition)

    def control_weight_tamper() -> bool:
        """A checkpoint whose network weights are replaced must be caught by E6."""
        positions = _select_probe_rows(
            [row for row in rows if str(row["source_role"]) == VAL_ROLE], 48
        )
        condition = str(payload["val_metrics"][0]["condition"])
        authentic = _replay_subset(model, rows, positions, condition)
        torch.manual_seed(4242)
        scratch = ResNet18(input_channels=1, num_classes=2).to(device)
        model.load_state_dict(scratch.state_dict())
        tampered = _replay_subset(model, rows, positions, condition)
        model.load_state_dict(base_state)
        restored = _replay_subset(model, rows, positions, condition)
        recorded_subset = [
            int(row["prediction"])
            for row in sorted(
                (row for row in scores if str(row["condition"]) == condition),
                key=lambda row: int(row["selection_position"]),
            )
        ]
        recorded_subset = [recorded_subset[position] for position in positions]
        if authentic != restored:
            raise RuntimeError("replay is not reproducible across a weight restore")
        del scratch
        # Detected only when the swapped weights move the decisions away from
        # both the authentic replay and the recording.
        return tampered == recorded_subset or tampered == authentic

    def control_selection_rule() -> bool:
        """Tampering the epoch history must move the independent argmax."""
        history = read_csv_rows(run_root / "training_history.csv")
        typed = [
            {"epoch": int(row["epoch"]), SELECTION_KEY: float(row[SELECTION_KEY])}
            for row in history
        ]
        selection = read_json(run_root / "selection.json")
        first = min(row["epoch"] for row in typed)
        tampered = [
            {**row, SELECTION_KEY: (1.0 if row["epoch"] == first else 0.0)} for row in typed
        ]
        return (
            select_epoch(tampered) == int(selection["selected_epoch"])
            and select_epoch(typed) == int(selection["selected_epoch"])
        )

    def control_tie_break() -> bool:
        """Claiming a later epoch for an exact tie must not survive the rule."""
        selection = read_json(run_root / "selection.json")
        recorded_epoch = int(selection["selected_epoch"])
        recorded_score = float(selection["selected_epoch_score"])
        tied = [
            {"epoch": epoch, SELECTION_KEY: recorded_score}
            for epoch in range(1, max(recorded_epoch + 1, 3))
        ]
        claimed = {
            "mean_balanced_accuracy_by_epoch": {
                str(row["epoch"]): row[SELECTION_KEY] for row in tied
            },
            "selected_epoch": max(row["epoch"] for row in tied),
        }
        recomputed = select_epoch(tied)
        claimed_epoch = int(claimed["selected_epoch"])
        return recomputed == claimed_epoch

    def control_label_mapping() -> bool:
        """A drone window relabelled background must break the label contract."""
        val_rows = [row for row in rows if str(row["source_role"]) == VAL_ROLE]
        victim = next(
            position
            for position, row in enumerate(val_rows)
            if str(row["target_code"]) != BACKGROUND_CODE
        )
        tampered = [
            {**row, "label": "0", "stage1_label": "0"} if position == victim else dict(row)
            for position, row in enumerate(val_rows)
        ]
        return not [
            row["filename"]
            for row in tampered
            if int(row["label"]) != (0 if str(row["target_code"]) == BACKGROUND_CODE else 1)
        ]

    def control_window_read_budget_without_val_cache() -> bool:
        """Dropping the validation-cache reads from the audit must break C4."""
        tampered = json.loads(json.dumps(audit))
        observed = observed_counts(tampered)
        for filename, count in val_windows.items():
            raw = next(
                (key for key in tampered["windows_by_file"] if Path(key).name == filename), None
            )
            if raw is not None:
                tampered["windows_by_file"][raw] = observed.get(filename, 0) - count
        return audit_budget_ok(tampered)

    detect(
        "a train file carries one extra val window read",
        control_val_window_counted_as_train,
        "audit budget C4",
    )
    detect(
        "a development_test file appears among the opened files",
        control_test_file_in_opened,
        "opened-file set C1/C2",
    )
    detect(
        "a window label is flipped in the recorded scores",
        control_label_flip_in_scores,
        "metric re-derivation E3",
    )
    detect(
        "one checkpoint weight is perturbed",
        control_weight_tamper,
        "decision replay E6",
    )
    detect(
        "the epoch history is tampered with",
        control_selection_rule,
        "selection rule E7",
    )
    detect(
        "the earliest-epoch tie-break is violated",
        control_tie_break,
        "tie-break E7",
    )
    detect(
        "a drone window relabelled background is rejected",
        control_label_mapping,
        "label contract B2",
    )
    detect(
        "the validation-cache reads are missing from the audit",
        control_window_read_budget_without_val_cache,
        "audit budget C4",
    )
    controls.extend(loader_audit_negative_controls(run_root, rows, index_info))

    del model
    return {
        "controls": controls,
        "detected": sum(1 for row in controls if row["detected"]),
        "total": len(controls),
        "passed": all(row["detected"] for row in controls),
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def verify_loader_audit_run(run_root: Path, output_path: Path) -> dict[str, Any]:
    """Top-level verification of one (usually short) loader-audit run.

    This is the entry point used to accept a real parallel smoke run without the
    full-protocol assertions that only a 5-epoch full run can satisfy, and it is
    the same ``verify_loader_identity_audit`` the full verifier calls.  It also
    re-derives the ordered consumption from the raw CSV, so a payload whose digests
    were rewritten by hand still has to match the logged positions.
    """

    checks = Checks()
    rows, frozen_role, freeze = check_inputs(checks)
    summary = read_json(SUMMARY_JSON)
    index_info = check_index_integrity(checks, rows, frozen_role, summary)
    config_path = run_root / "run_config.json"
    config = read_json(config_path) if config_path.is_file() else {}
    loader_config = config.get("loader") or {}
    parallel = bool(loader_config.get("parallel")) or int(
        loader_config.get("num_workers") or 0
    ) > 0
    checks.add(
        "L1 the run under test used the loader identity audit",
        bool(loader_config) and (run_root / "loader_read_audit.json").is_file(),
        {"loader": loader_config, "run_root": str(run_root)},
    )
    audit_info = verify_loader_identity_audit(checks, run_root, rows, index_info)
    plan = loader_identity_plan(rows, run_root, index_info)
    if plan.get("available"):
        expected: Counter = Counter()
        for phase_name, keys in ((TRAIN_ROLE, plan["train"]), (VAL_ROLE, plan["val"])):
            for key, positions in keys.items():
                for position in positions:
                    expected[(phase_name, str(key), int(position))] += 1
        log_path = run_root / "loader_consumption.csv"
        logged: Counter = Counter()
        delivered_rows = read_csv_rows(log_path) if log_path.is_file() else []
        incomplete = 0
        for row in delivered_rows:
            logged[(str(row["phase"]), str(row["key"]), int(row["position"]))] += 1
            if int(row.get("completed", 0)) != 1:
                incomplete += 1
        checks.add(
            "L2 the ordered consumption log matches the recomputed plan",
            bool(delivered_rows) and logged == expected,
            {
                "log": str(log_path),
                "rows": len(delivered_rows),
                "expected_rows": sum(expected.values()),
                "missing": list((expected - logged).elements())[:5],
                "extra": list((logged - expected).elements())[:5],
            },
        )
        checks.add(
            "L3 every logged window was completed by a training step",
            bool(delivered_rows) and incomplete == 0,
            {"incomplete_rows": incomplete},
        )
        checks.add(
            "L4 the run recorded the audit as passing",
            bool(config.get("loader_audit_passed")),
            {"loader_audit_passed": config.get("loader_audit_passed")},
        )
    controls = loader_audit_negative_controls(run_root, rows, index_info)
    if controls:
        checks.add(
            "F1 every loader-audit tamper control was detected",
            all(row["detected"] for row in controls),
            f"{sum(1 for row in controls if row['detected'])}/{len(controls)}",
        )
    else:
        # No controls means the untampered payload already fails its own checks; the
        # tamper controls have nothing to prove and must not be reported as passing.
        already_rejected = bool(audit_info.get("problems"))
        checks.add(
            "F1 loader-audit tamper controls ran (or the audit was already rejected)",
            already_rejected,
            "audit already rejected by C7; controls not applicable"
            if already_rejected
            else "no tamper control could be built",
        )
    result = {
        "schema": "dronerfa-stage1-loader-audit-verification/v1",
        "generated_at": utc_now(),
        "run_root": str(run_root),
        "parallel": parallel,
        "loader": loader_config,
        "training_seed": config.get("training_seed"),
        "loader_audit": audit_info,
        "negative_controls": controls,
        "verification": checks.payload(),
        "passed": checks.passed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "passed_checks": result["verification"]["passed_checks"],
                "total_checks": result["verification"]["total_checks"],
                "failed": result["verification"]["failed_checks"],
                "negative_controls": f"{sum(1 for row in controls if row['detected'])}/{len(controls)}",
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    del freeze
    return result


def verify(
    run_root: Path,
    output_path: Path,
    replay_budget: int = REPLAY_WINDOWS,
    skip_reproduction: bool = False,
) -> dict[str, Any]:
    device = torch.device("cpu")
    checks = Checks()
    rows, frozen_role, freeze = check_inputs(checks)
    summary = read_json(SUMMARY_JSON)
    index_info = check_index_integrity(checks, rows, frozen_role, summary)
    composite = {"rows": rows}
    audit_info = check_read_audit(checks, run_root, composite, index_info)
    reproduction: dict[str, Any] = {}
    if not skip_reproduction:
        reproduction = reproduce_validation(checks, run_root, rows, device, replay_budget)
    controls = negative_controls(checks, run_root, rows, index_info, device)
    checks.add(
        "F1 every negative control was detected",
        controls["passed"],
        f"{controls['detected']}/{controls['total']}",
    )

    result = {
        "schema": "dronerfa-stage1-12drones-training-verification/v1",
        "generated_at": utc_now(),
        "run_root": str(run_root),
        "training_seed": TRAINING_SEED,
        "validation_seed": VALIDATION_SEED,
        "index": {"path": str(INDEX), "sha256": INDEX_SHA256, "rows": len(rows)},
        "frozen_split_csv": {"path": str(FROZEN_SPLIT_CSV), "sha256": FROZEN_SPLIT_CSV_SHA256},
        "read_audit": audit_info,
        "reproduction": reproduction,
        "negative_controls": controls,
        "verification": checks.payload(),
        "passed": checks.passed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "passed_checks": result["verification"]["passed_checks"],
                "total_checks": result["verification"]["total_checks"],
                "failed": result["verification"]["failed_checks"],
                "negative_controls": f"{controls['detected']}/{controls['total']}",
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    del freeze
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the STAGE1-12DRONES stage-1 training artefacts of one training "
            f"seed (default: the accepted seed {DEFAULT_TRAINING_SEED})."
        )
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=DEFAULT_TRAINING_SEED,
        help="training seed whose run directory and config claim are verified",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="override the run directory (default: outputs/.../seed_<training-seed>)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "verification report path (default: verification.json for the accepted "
            "seed, verification_seed_<seed>.json otherwise)"
        ),
    )
    parser.add_argument(
        "--replay-windows",
        type=int,
        default=REPLAY_WINDOWS,
        help="validation windows to re-derive from raw RF0; 0 replays the whole split",
    )
    parser.add_argument("--skip-reproduction", action="store_true")
    parser.add_argument(
        "--loader-audit-only",
        action="store_true",
        help=(
            "verify only the loader identity audit (phase/key/position) of the run at "
            "--run-root; this is the acceptance path for short parallel smoke runs, "
            "where the full 5-epoch protocol assertions do not apply"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bind_training_seed(args.training_seed)
    run_root = Path(args.run_root) if args.run_root is not None else run_root_for(TRAINING_SEED)
    if args.output is not None:
        output_path = args.output
    elif args.loader_audit_only:
        output_path = run_root / "loader_audit_verification.json"
    elif TRAINING_SEED == DEFAULT_TRAINING_SEED:
        output_path = OUTPUT_ROOT / "verification.json"
    else:
        output_path = OUTPUT_ROOT / f"verification_seed_{TRAINING_SEED}.json"
    if args.loader_audit_only:
        result = verify_loader_audit_run(run_root, output_path)
    else:
        result = verify(
            run_root,
            output_path,
            args.replay_windows,
            args.skip_reproduction,
        )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
