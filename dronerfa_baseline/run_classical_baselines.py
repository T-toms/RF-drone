"""BASE2: two clean-condition classical baselines for DroneRFa stage 1.

Baselines (RF0 complex I/Q, 1,000,000-sample windows, Background=0 / Drone=1):

1. ``energy``  - complex-mean-removed average power of the *raw, un-normalized*
   window, ``10*log10(mean(|x - mean(x)|^2) + eps)`` in dB, with a single global
   threshold whose value and direction come from the old train+val split only.
2. ``psd_svm`` - 1024-bin mean temporal PSD (Hamming, n_fft=1024, hop=512,
   two-sided complex spectrum, fftshift, ``10*log10(P + 1e-10)``) of the
   ``rms_normalize``d complex window, standardized with statistics fitted on the
   old train split only, classified by a class-balanced SVM whose kernel and
   hyper-parameters are picked on the old val split only.

Frozen inputs, never modified: ``outputs/window_index.csv`` (old train/val/test)
and ``outputs/heldout_preparation_20260917/heldout_window_index.csv`` (the 1820
unused held-out windows). The held-out index is never used for thresholds,
standardization, SVM selection or any other fitting.

Pipeline (each stage is a separate subcommand so its result is auditable):

    energy            select the threshold, then evaluate test and heldout
    psd-features      build the cached float32 PSD feature matrix + energy cache
    psd-svm           standardize, select the SVM on val, evaluate test and heldout
    verify            re-derive metrics from the prediction CSVs independently
    report            write comparison_metrics.csv, experiment_manifest.json,
                      report.md and verification.json

Environment note: this project's ``.venv`` has numpy/h5py/torch(CUDA)/matplotlib
but no scikit-learn; the local Anaconda install has scikit-learn. The SVM stage
imports scikit-learn from that vendored local path (recorded in
``experiment_manifest.json``) when it is not importable from the active
interpreter; see ``--sklearn-path``. NumPy is the only numeric dependency shared
between the torch stage and the sklearn stage, because the SVM never sees raw
I/Q - it always consumes the cached feature matrices.
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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "outputs" / ".matplotlib"))

import numpy as np

from dronerfa_data import dataset_length, read_vector_slice

# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------
BASELINE_DIR = (
    PROJECT_DIR / "outputs" / "stage1_method_comparison_20260919" / "baselines"
)
DEFAULT_OLD_INDEX = PROJECT_DIR / "outputs" / "window_index.csv"
DEFAULT_HELDOUT_INDEX = (
    PROJECT_DIR / "outputs" / "heldout_preparation_20260917" / "heldout_window_index.csv"
)
DEFAULT_PRESERVED = PROJECT_DIR.parent / "work" / "codex-preserved-inputs-20260917.json"
DEFAULT_SKLEARN_PATH = Path(r"D:\anaconda\Lib\site-packages")

OLD_INDEX_SHA256 = "4e77c7f186c1aae8e04f2e90f2769c0ed399cc94f40b64f99237b65c057b85ac"
HELDOUT_INDEX_SHA256 = (
    "6fe32faeb5d583a359d9e8aa17caa52a7c4567d545272ded25266767cd2cffcf"
)

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
HELDOUT_SPLIT = "heldout"
SPLIT_ORDER = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, HELDOUT_SPLIT)
FIT_SPLITS = (TRAIN_SPLIT,)
SELECT_SPLITS = (TRAIN_SPLIT, VAL_SPLIT)
EVAL_SPLITS = (TEST_SPLIT, HELDOUT_SPLIT)

WINDOW_SAMPLES = 1_000_000
N_FFT = 1024
HOP_LENGTH = 512
PSD_BINS = N_FFT
PSD_EPS = 1e-10
ENERGY_EPS = 1e-12
RMS_EPS = 1e-8
CATEGORY_NAMES = {"0000": "background", "0010": "phantom", "0101": "air2s"}
BACKGROUND_CODE = "0000"
DRONE_CODES = ("0010", "0101")

ENERGY_METHOD = "energy_threshold"
PSD_METHOD = "psd_svm"
ENERGY_RULE_ID = "argmax_balanced_accuracy_midpoint_sweep_train_val"
ENERGY_TIE_BREAK = "smallest_threshold"
PSD_RULE_ID = "argmax_val_balanced_accuracy_then_manifest_order"
SVM_RANDOM_SEED = 20260919
FEATURE_SCALER = "sklearn.preprocessing.StandardScaler"
SVM_ESTIMATOR = "sklearn.svm.SVC"
FEATURE_DTYPE = np.float32
REFERENCE_TOLERANCE = 1e-5

# Frozen before any test/held-out number is computed; selection is
# "highest old-val balanced accuracy, then first entry in this order".
SVM_CANDIDATES: list[dict[str, Any]] = [
    {"id": "linear-C0.01", "kernel": "linear", "C": 0.01},
    {"id": "linear-C0.1", "kernel": "linear", "C": 0.1},
    {"id": "linear-C1", "kernel": "linear", "C": 1.0},
    {"id": "linear-C10", "kernel": "linear", "C": 10.0},
    {"id": "linear-C100", "kernel": "linear", "C": 100.0},
    {"id": "rbf-C0.1-scale", "kernel": "rbf", "C": 0.1, "gamma": "scale"},
    {"id": "rbf-C1-scale", "kernel": "rbf", "C": 1.0, "gamma": "scale"},
    {"id": "rbf-C10-scale", "kernel": "rbf", "C": 10.0, "gamma": "scale"},
    {"id": "rbf-C100-scale", "kernel": "rbf", "C": 100.0, "gamma": "scale"},
    {"id": "rbf-C1-g0.001", "kernel": "rbf", "C": 1.0, "gamma": 0.001},
    {"id": "rbf-C1-g0.01", "kernel": "rbf", "C": 1.0, "gamma": 0.01},
    {"id": "rbf-C1-g0.1", "kernel": "rbf", "C": 1.0, "gamma": 0.1},
    {"id": "poly3-C0.1", "kernel": "poly", "degree": 3, "C": 0.1},
    {"id": "poly3-C1", "kernel": "poly", "degree": 3, "C": 1.0},
    {"id": "poly3-C10", "kernel": "poly", "degree": 3, "C": 10.0},
    {"id": "sigmoid-C0.1", "kernel": "sigmoid", "C": 0.1},
    {"id": "sigmoid-C1", "kernel": "sigmoid", "C": 1.0},
    {"id": "sigmoid-C10", "kernel": "sigmoid", "C": 10.0},
]

# Quoted verbatim from the accepted EVAL6 held-out clean result; never re-run here.
RESNET_REFERENCE = {
    "source": "work/heldout-eval-config-20260918.json and accepted EVAL6 summary",
    "heldout_clean_file_equal_balanced_accuracy": {
        "seed2026_A": 0.9641,
        "seed2026_B": 0.9928,
        "seed2027_A": 0.9902,
        "seed2027_B": 0.9907,
        "seed2028_A": 0.9879,
        "seed2028_B": 0.9940,
    },
}


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


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


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


def run_command() -> str:
    return " ".join([sys.executable, *sys.argv])


def library_versions() -> dict[str, Any]:
    import importlib.metadata as metadata

    def version_of(package: str) -> str | None:
        try:
            return metadata.version(package)
        except Exception:
            return None

    versions: dict[str, Any] = {"numpy": np.__version__}
    for package in ("h5py", "torch", "matplotlib", "scikit-learn", "joblib"):
        versions[package] = version_of(package)
    return versions


# ---------------------------------------------------------------------------
# index handling / source identity
# ---------------------------------------------------------------------------
def load_index_rows(path: Path, split: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in read_csv_rows(path):
        row: dict[str, Any] = dict(raw)
        for field in (
            "label",
            "block_index",
            "window_index",
            "start",
            "length",
            "source_samples",
        ):
            row[field] = int(row[field])
        if split is None or row["split"] == split:
            rows.append(row)
    return rows


def rows_by_split(
    old_index: Path, heldout_index: Path
) -> dict[str, list[dict[str, Any]]]:
    return {
        split: load_index_rows(
            heldout_index if split == HELDOUT_SPLIT else old_index, split
        )
        for split in SPLIT_ORDER
    }


def build_catalog(
    splits: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    return [
        {"split": split, "row_in_split": position, **row}
        for split in SPLIT_ORDER
        for position, row in enumerate(splits[split])
    ]


def split_slices(splits: dict[str, list[dict[str, Any]]]) -> dict[str, slice]:
    slices: dict[str, slice] = {}
    cursor = 0
    for split in SPLIT_ORDER:
        count = len(splits[split])
        slices[split] = slice(cursor, cursor + count)
        cursor += count
    return slices


def source_identity(old_index: Path, heldout_index: Path) -> dict[str, Any]:
    per_split: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        index_path = heldout_index if split == HELDOUT_SPLIT else old_index
        rows = load_index_rows(index_path, split)
        per_category: dict[str, list[str]] = {}
        windows_per_category: dict[str, int] = {}
        for row in rows:
            bucket = per_category.setdefault(row["target_code"], [])
            if row["filename"] not in bucket:
                bucket.append(row["filename"])
            windows_per_category[row["target_code"]] = (
                windows_per_category.get(row["target_code"], 0) + 1
            )
        per_split[split] = {
            "index": str(index_path),
            "windows": len(rows),
            "files": sorted({row["filename"] for row in rows}),
            "files_per_category": {
                key: sorted(value) for key, value in sorted(per_category.items())
            },
            "windows_per_category": dict(sorted(windows_per_category.items())),
        }
    old_files: set[str] = set()
    for split in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT):
        old_files.update(per_split[split]["files"])
    heldout_files = set(per_split[HELDOUT_SPLIT]["files"])
    overlap = sorted(old_files & heldout_files)
    return {
        "per_split": per_split,
        "old_files": sorted(old_files),
        "heldout_files": sorted(heldout_files),
        "file_overlap": overlap,
        "file_overlap_count": len(overlap),
        "heldout_disjoint_from_old": not overlap,
    }


def assert_disjoint(identity: dict[str, Any]) -> None:
    if not identity["heldout_disjoint_from_old"]:
        raise SystemExit("held-out source files overlap the old train/val/test files")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def confusion_counts(
    labels: Sequence[int], predictions: Sequence[int]
) -> dict[str, int]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(predictions, dtype=np.int64)
    if y.shape != p.shape:
        raise ValueError("labels and predictions must have the same length")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be binary")
    if not np.all((p == 0) | (p == 1)):
        raise ValueError("predictions must be binary")
    return {
        "tn": int(np.sum((y == 0) & (p == 0))),
        "fp": int(np.sum((y == 0) & (p == 1))),
        "fn": int(np.sum((y == 1) & (p == 0))),
        "tp": int(np.sum((y == 1) & (p == 1))),
    }


def metric_values(counts: dict[str, int]) -> dict[str, Any]:
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    total = tn + fp + fn + tp
    if total <= 0:
        raise ValueError("empty confusion matrix")
    accuracy = (tp + tn) / total
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "windows": total,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "drone_recall": recall,
        "background_specificity": specificity,
        "f1": f1,
        "drone_fpr": fp / max(fp + tn, 1),
    }


def file_equal_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        code = str(record["target_code"])
        grouped.setdefault(code, {}).setdefault(str(record["filename"]), []).append(record)
    ordered_keys = sorted(grouped, key=lambda item: (item != BACKGROUND_CODE, item))
    per_file_rates: dict[str, dict[str, float]] = {}
    per_file_windows: dict[str, dict[str, int]] = {}
    for code in ordered_keys:
        per_file_rates[code] = {}
        per_file_windows[code] = {}
        for name in sorted(grouped[code]):
            bucket = grouped[code][name]
            per_file_rates[code][name] = float(
                np.mean([int(record["prediction"]) for record in bucket])
            )
            per_file_windows[code][name] = len(bucket)

    def mean_rate(code: str) -> float:
        values = list(per_file_rates[code].values())
        return float(np.mean(values)) if values else float("nan")

    background_fpr = mean_rate(BACKGROUND_CODE)
    drone_recalls = [mean_rate(code) for code in DRONE_CODES if code in grouped]
    drone_recall = float(np.mean(drone_recalls)) if drone_recalls else float("nan")
    specificity = 1.0 - background_fpr
    return {
        "files_per_category": {code: len(grouped[code]) for code in ordered_keys},
        "background_file_equal_fpr": background_fpr,
        "background_file_equal_specificity": specificity,
        "phantom_file_equal_recall": (
            mean_rate("0010") if "0010" in grouped else float("nan")
        ),
        "air2s_file_equal_recall": (
            mean_rate("0101") if "0101" in grouped else float("nan")
        ),
        "drone_file_equal_recall": drone_recall,
        "file_equal_balanced_accuracy": 0.5 * (specificity + drone_recall),
        "per_file_rates": per_file_rates,
        "per_file_windows": per_file_windows,
    }


def pooled_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = confusion_counts(
        [int(record["label"]) for record in records],
        [int(record["prediction"]) for record in records],
    )
    values = metric_values(counts)
    codes = [str(record["target_code"]) for record in records]
    predictions = [int(record["prediction"]) for record in records]
    per_category: dict[str, float] = {}
    for code in sorted(set(codes)):
        selected = [p for p, c in zip(predictions, codes) if c == code]
        target = 0 if code == BACKGROUND_CODE else 1
        per_category[CATEGORY_NAMES.get(code, code)] = float(
            np.mean([p == target for p in selected])
        )
    return {"scope": "pooled", **values, "per_category_rates": per_category}


def metric_row(method: str, split: str, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """One metrics.csv row: pooled counts plus both aggregation scopes in JSON."""
    pooled = pooled_metrics(records)
    file_equal = file_equal_metrics(records)
    return {
        "method": method,
        "split": split,
        "scope": "file_equal",
        "windows": pooled["windows"],
        "tn": pooled["tn"],
        "fp": pooled["fp"],
        "fn": pooled["fn"],
        "tp": pooled["tp"],
        "pooled_accuracy": pooled["accuracy"],
        "pooled_balanced_accuracy": pooled["balanced_accuracy"],
        "pooled_precision": pooled["precision"],
        "pooled_drone_recall": pooled["drone_recall"],
        "pooled_background_specificity": pooled["background_specificity"],
        "pooled_f1": pooled["f1"],
        "file_equal_balanced_accuracy": file_equal["file_equal_balanced_accuracy"],
        "file_equal_drone_recall": file_equal["drone_file_equal_recall"],
        "file_equal_background_specificity": file_equal[
            "background_file_equal_specificity"
        ],
        "file_equal_background_fpr": file_equal["background_file_equal_fpr"],
        "file_equal_phantom_recall": file_equal["phantom_file_equal_recall"],
        "file_equal_air2s_recall": file_equal["air2s_file_equal_recall"],
        "files_per_category": json.dumps(
            file_equal["files_per_category"], sort_keys=True
        ),
        "detail_json": json.dumps(
            {
                "file_equal": file_equal,
                "pooled": pooled,
            },
            sort_keys=True,
        ),
    }


def prediction_records(
    table: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "filename": row["filename"],
            "target_code": row["target_code"],
            "label": int(row["label"]),
            "prediction": int(row["prediction"]),
        }
        for row in table
    ]


def balanced_accuracy_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> float:
    predictions = (scores >= threshold).astype(np.int64)
    return metric_values(confusion_counts(labels.tolist(), predictions.tolist()))[
        "balanced_accuracy"
    ]


# ---------------------------------------------------------------------------
# raw I/Q window iteration
# ---------------------------------------------------------------------------
def iter_windows(
    rows: Sequence[dict[str, Any]],
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (row position, complex64 window); one HDF5 file open at a time."""
    import h5py

    by_file: dict[str, list[int]] = {}
    for position, row in enumerate(rows):
        if int(row["length"]) != WINDOW_SAMPLES:
            raise ValueError(
                f"window length {row['length']} != {WINDOW_SAMPLES} "
                f"for {row['filename']}"
            )
        by_file.setdefault(str(row["file_path"]), []).append(position)

    for file_path in sorted(by_file, key=lambda item: by_file[item][0]):
        positions = sorted(by_file[file_path], key=lambda item: int(rows[item]["start"]))
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            for key in ("RF0_I", "RF0_Q"):
                if key not in handle:
                    raise KeyError(f"{path.name}: missing dataset {key}")
            length = dataset_length(handle["RF0_I"])
            expected = int(rows[positions[0]]["source_samples"])
            if length != expected:
                raise ValueError(
                    f"{path.name}: index says {expected} samples, file has {length}"
                )
            for position in positions:
                row = rows[position]
                start = int(row["start"])
                count = int(row["length"])
                if start < 0 or start + count > length:
                    raise ValueError(
                        f"{path.name}: window [{start}, {start + count}) is outside "
                        f"the {length} available samples"
                    )
                i_values = read_vector_slice(handle["RF0_I"], start, count)
                q_values = read_vector_slice(handle["RF0_Q"], start, count)
                window = (i_values + 1j * q_values).astype(np.complex64, copy=False)
                yield position, window


def energy_scores_for_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Raw-window energy score in dB and the complex-mean-removed mean power."""
    scores = np.empty(len(rows), dtype=np.float64)
    powers = np.empty(len(rows), dtype=np.float64)
    started = time.perf_counter()
    for done, (position, window) in enumerate(iter_windows(rows), start=1):
        centered = window - np.mean(window, dtype=np.complex128)
        power = float(np.mean(np.abs(centered) ** 2, dtype=np.float64))
        powers[position] = power
        scores[position] = 10.0 * math.log10(power + ENERGY_EPS)
        if done % 250 == 0 or done == len(rows):
            elapsed = time.perf_counter() - started
            log(
                f"  energy {done}/{len(rows)} windows ({elapsed:.1f}s, "
                f"{done / max(elapsed, 1e-9):.1f} win/s)"
            )
    return scores, powers


def hamming_window_matrix(n_fft: int = N_FFT) -> np.ndarray:
    """The exact analysis window used for the PSD, as float64.

    ``torch.hamming_window`` is the *periodic* Hamming window (``scipy``'s
    ``sym=False`` convention used for spectral analysis), which differs from
    ``numpy.hamming`` (symmetric). The window is materialised and cached so the
    independent NumPy check can use the identical taper instead of guessing.
    """
    import torch

    return torch.hamming_window(n_fft, periodic=True, dtype=torch.float64).numpy()


def numpy_reference_psd(
    window: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    window_fn: np.ndarray | None = None,
) -> np.ndarray:
    """Independent NumPy/stride-trick reimplementation used for verification."""
    taper = hamming_window_matrix(n_fft) if window_fn is None else window_fn
    centered = window - np.mean(window, dtype=np.complex128)
    power = float(np.mean(np.abs(centered) ** 2, dtype=np.float64))
    normalized = centered / math.sqrt(max(power, RMS_EPS))
    samples = normalized.shape[0]
    frames = 1 + (samples - n_fft) // hop_length
    stride = normalized.strides[0]
    framed = np.lib.stride_tricks.as_strided(
        normalized,
        shape=(frames, n_fft),
        strides=(hop_length * stride, stride),
        writeable=False,
    )
    tapered = framed * taper[None, :]
    spectra = np.fft.fft(tapered, n=n_fft, axis=1) / math.sqrt(n_fft)
    log_power = 10.0 * np.log10(np.mean(np.abs(spectra) ** 2, axis=0) + PSD_EPS)
    return np.fft.fftshift(log_power)


def psd_features_for_rows(
    rows: Sequence[dict[str, Any]],
    device_name: str,
    batch_size: int,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cached mean-temporal-PSD features for the given rows, in row order."""
    import torch

    if len(rows) == 0:
        raise ValueError("no rows to featurize")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    window_fn = torch.hamming_window(n_fft, device=device)
    features = np.empty((len(rows), n_fft), dtype=FEATURE_DTYPE)
    pending: list[tuple[int, np.ndarray]] = []
    done = 0
    started = time.perf_counter()

    def flush() -> None:
        nonlocal done
        if not pending:
            return
        positions = [item[0] for item in pending]
        windows = np.stack([item[1] for item in pending], axis=0)
        pending.clear()
        batch = torch.from_numpy(np.stack([windows.real, windows.imag], axis=1)).to(device)
        del windows
        with torch.inference_mode():
            complex_iq = torch.complex(batch[:, 0], batch[:, 1])
            spectra = torch.stft(
                complex_iq,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window_fn,
                center=False,
                normalized=True,
                onesided=False,
                return_complex=True,
            )
            log_power = 10.0 * torch.log10(
                spectra.abs().square().mean(dim=2) + PSD_EPS
            )
            shifted = torch.fft.fftshift(log_power, dim=1)
            array = shifted.to("cpu", torch.float32).numpy()
        if array.shape != (len(positions), n_fft):
            raise RuntimeError(f"unexpected PSD shape {array.shape}")
        if not np.isfinite(array).all():
            raise RuntimeError("non-finite PSD feature values")
        for offset, position in enumerate(positions):
            features[position] = array[offset]
        done += len(positions)
        del batch, complex_iq, spectra, log_power, shifted, array

    for position, window in iter_windows(rows):
        centered = window - np.mean(window, dtype=np.complex128)
        power = float(np.mean(np.abs(centered) ** 2, dtype=np.float64))
        pending.append(
            (position, (centered / math.sqrt(max(power, RMS_EPS))).astype(np.complex64))
        )
        if len(pending) >= batch_size:
            flush()
            if done % (batch_size * 8) == 0 or done == len(rows):
                elapsed = time.perf_counter() - started
                log(
                    f"  psd {done}/{len(rows)} windows ({elapsed:.1f}s, "
                    f"{done / max(elapsed, 1e-9):.1f} win/s, device={device})"
                )
    flush()
    if done != len(rows):
        raise RuntimeError(f"featurized {done} of {len(rows)} windows")
    elapsed = time.perf_counter() - started
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metadata = {
        "rows": len(rows),
        "shape": [len(rows), n_fft],
        "dtype": str(FEATURE_DTYPE.__name__),
        "n_fft": n_fft,
        "hop_length": hop_length,
        "hop_ratio": hop_length / n_fft,
        "window": "hamming",
        "onesided": False,
        "normalized_stft": True,
        "fftshift": True,
        "frequency_reduction": "mean_over_time_frames",
        "eps": PSD_EPS,
        "device": str(device),
        "seconds": elapsed,
        "windows_per_second": len(rows) / max(elapsed, 1e-9),
    }
    return features, metadata


# ---------------------------------------------------------------------------
# energy threshold selection
# ---------------------------------------------------------------------------
def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Minimum score plus the midpoints between all adjacent unique scores."""
    unique = np.unique(np.asarray(scores, dtype=np.float64))
    if unique.size == 0:
        raise ValueError("no scores to threshold")
    if unique.size == 1:
        return unique
    return np.concatenate(([unique[0]], 0.5 * (unique[:-1] + unique[1:])))


def assert_frozen_indexes(old_index: Path, heldout_index: Path, old_hash: str, heldout_hash: str) -> None:
    """Enforce the frozen index hashes whenever the canonical index files are used.

    A deliberately reduced index (debug/smoke runs that pass their own paths) is not
    the frozen artefact, so only the canonical paths are hash-pinned here; the
    ``verify`` stage still hash-checks the frozen files for every accepted run.
    """
    if old_index.resolve() == DEFAULT_OLD_INDEX.resolve() and old_hash != OLD_INDEX_SHA256:
        raise SystemExit(f"old index SHA256 {old_hash} != frozen {OLD_INDEX_SHA256}")
    if (
        heldout_index.resolve() == DEFAULT_HELDOUT_INDEX.resolve()
        and heldout_hash != HELDOUT_INDEX_SHA256
    ):
        raise SystemExit(
            f"heldout index SHA256 {heldout_hash} != frozen {HELDOUT_INDEX_SHA256}"
        )


def select_energy_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Exhaustive balanced-accuracy sweep over the supplied (train+val) scores."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same length")
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("selection data needs both classes")
    candidates = candidate_thresholds(scores)
    values = np.asarray(
        [
            balanced_accuracy_at_threshold(labels, scores, float(threshold))
            for threshold in candidates
        ],
        dtype=np.float64,
    )
    best_value = float(values.max())
    best_index = int(np.argmax(values))
    tied = [float(candidates[index]) for index in np.flatnonzero(values == best_value)]
    threshold = float(candidates[best_index])
    predictions = (scores >= threshold).astype(np.int64)
    counts = confusion_counts(labels.tolist(), predictions.tolist())
    return {
        "threshold_db": threshold,
        "balanced_accuracy": best_value,
        "candidates_evaluated": int(candidates.size),
        "candidates_with_best_score": len(tied),
        "tie_break": ENERGY_TIE_BREAK,
        "train_val_confusion": counts,
        "train_val_metrics": metric_values(counts),
    }


# ---------------------------------------------------------------------------
# sklearn environment
# ---------------------------------------------------------------------------
def import_sklearn(preferred_path: Path | None) -> dict[str, Any]:
    """Import scikit-learn, using the recorded local fallback path if needed."""
    try:
        import sklearn

        return {
            "imported": True,
            "fallback_used": False,
            "fallback_path": None,
            "sklearn_file": str(Path(sklearn.__file__).resolve()),
            "sklearn_version": sklearn.__version__,
        }
    except ImportError as error:
        if preferred_path is None or not Path(preferred_path).is_dir():
            raise SystemExit(
                "scikit-learn is not importable and no usable --sklearn-path was "
                f"given (original error: {error})"
            ) from error
        fallback = str(preferred_path)
        if fallback not in sys.path:
            sys.path.append(fallback)
        import sklearn

        return {
            "imported": True,
            "fallback_used": True,
            "fallback_path": fallback,
            "sklearn_file": str(Path(sklearn.__file__).resolve()),
            "sklearn_version": sklearn.__version__,
        }


# ---------------------------------------------------------------------------
# cache helpers
# ---------------------------------------------------------------------------
def write_cache(
    cache_dir: Path, name: str, array: np.ndarray, provenance: dict[str, Any]
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.npy"
    np.save(path, array)
    record = {
        **provenance,
        "file": str(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
        "array_sha256": sha256_array(array),
        "file_sha256": sha256_file(path),
    }
    write_json(cache_dir / f"{name}_metadata.json", record)
    return record


def load_cache(cache_dir: Path, name: str, expected_rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    path = cache_dir / f"{name}.npy"
    metadata_path = cache_dir / f"{name}_metadata.json"
    if not path.is_file() or not metadata_path.is_file():
        raise SystemExit(f"missing cache {path}; run the psd-features/energy stage first")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    array = np.load(path)
    if array.shape[0] != expected_rows:
        raise SystemExit(f"cache {path} has {array.shape[0]} rows, expected {expected_rows}")
    if sha256_array(array) != metadata["array_sha256"]:
        raise SystemExit(f"cache {path} contents do not match its recorded SHA256")
    return array, metadata


# ---------------------------------------------------------------------------
# stage: energy
# ---------------------------------------------------------------------------
def stage_energy(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    cache_dir = output_dir / "_cache"
    energy_dir = output_dir / "energy_threshold"
    energy_dir.mkdir(parents=True, exist_ok=True)

    identity = source_identity(args.old_index, args.heldout_index)
    assert_disjoint(identity)
    splits = rows_by_split(args.old_index, args.heldout_index)
    catalog = build_catalog(splits)
    for split in SPLIT_ORDER:
        if not splits[split]:
            raise SystemExit(f"no rows for split {split}")
        log(
            f"split {split}: {len(splits[split])} windows, "
            f"{len({row['filename'] for row in splits[split]})} files"
        )

    cache_info: dict[str, Any] = {}
    if args.from_cache:
        scores, cache_info["energy_scores"] = load_cache(
            cache_dir, "energy_scores", len(catalog)
        )
        powers, cache_info["energy_powers"] = load_cache(
            cache_dir, "energy_powers", len(catalog)
        )
        log(f"loaded cached energy scores {scores.shape} from {cache_dir}")
    else:
        started = time.perf_counter()
        scores, powers = energy_scores_for_rows(catalog)
        provenance = {
            "source": "raw RF0_I/RF0_Q windows in index order (train, val, test, heldout)",
            "score_formula": "10*log10(mean(|x - mean(x)|^2) + 1e-12)",
            "normalization": "none (raw relative power on purpose)",
        }
        cache_info["energy_scores"] = write_cache(
            cache_dir, "energy_scores", scores, provenance
        )
        cache_info["energy_powers"] = write_cache(
            cache_dir, "energy_powers", powers, provenance
        )
        cache_info["energy_seconds"] = time.perf_counter() - started
        log(f"energy scores computed in {cache_info['energy_seconds']:.1f}s")

    slices = split_slices(splits)
    select_scores = np.concatenate([scores[slices[split]] for split in SELECT_SPLITS])
    select_labels = np.concatenate(
        [
            np.asarray([int(row["label"]) for row in splits[split]], dtype=np.int64)
            for split in SELECT_SPLITS
        ]
    )
    selection = select_energy_threshold(select_labels, select_scores)
    threshold = float(selection["threshold_db"])
    log(
        f"selected threshold {threshold:.10f} dB "
        f"(train+val BA {selection['balanced_accuracy']:.6f})"
    )

    selection_record: dict[str, Any] = {
        "created_at": utc_now(),
        "baseline": ENERGY_METHOD,
        "method": "complex-mean-removed average power of the raw RF0 window, in dB",
        "score_formula": "10*log10(mean(|x - mean(x)|^2) + eps), x = RF0_I + j*RF0_Q, eps = 1e-12",
        "score_units": "dB (relative to the raw recording scale)",
        "normalization": "none; the raw relative recording power is used on purpose",
        "decision_rule": "score >= threshold -> Drone (1); score < threshold -> Background (0)",
        "rule_id": ENERGY_RULE_ID,
        "candidate_rule": (
            "minimum train+val score plus the midpoints between all adjacent unique "
            "train+val scores"
        ),
        "selection_data": {
            "splits": list(SELECT_SPLITS),
            "index_files": [str(args.old_index)],
            "windows": int(select_labels.size),
            "class_counts": {
                "background": int(np.sum(select_labels == 0)),
                "drone": int(np.sum(select_labels == 1)),
            },
            "score_sha256": sha256_array(select_scores),
        },
        "threshold_db": threshold,
        "selection_metric": "balanced accuracy on train+val windows",
        "tie_break": ENERGY_TIE_BREAK,
        "candidates_evaluated": selection["candidates_evaluated"],
        "candidates_with_best_score": selection["candidates_with_best_score"],
        "train_val_metrics": selection["train_val_metrics"],
        "excluded_from_selection": {
            "test": f"{args.old_index} split=test",
            "heldout": str(args.heldout_index),
            "statement": (
                "test and heldout scores are never read while choosing the threshold; "
                "only train and val rows enter candidate_thresholds/select_energy_threshold"
            ),
        },
        "per_split_score_summary": {},
    }
    for split in SPLIT_ORDER:
        values = scores[slices[split]]
        labels = np.asarray([int(row["label"]) for row in splits[split]], dtype=np.int64)
        selection_record["per_split_score_summary"][split] = {
            "windows": int(values.size),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "mean_background": float(values[labels == 0].mean()),
            "mean_drone": float(values[labels == 1].mean()),
            "files": len({row["filename"] for row in splits[split]}),
        }

    tables: dict[str, list[dict[str, Any]]] = {}
    metric_rows: list[dict[str, Any]] = []
    for split in EVAL_SPLITS:
        table: list[dict[str, Any]] = []
        for position, row in enumerate(splits[split]):
            score = float(scores[slices[split]][position])
            table.append(
                {
                    "window": ENERGY_METHOD,
                    "split": split,
                    "row_in_split": position,
                    "filename": row["filename"],
                    "target_code": row["target_code"],
                    "target_name": row["target_name"],
                    "label": int(row["label"]),
                    "block_index": int(row["block_index"]),
                    "window_index": int(row["window_index"]),
                    "start": int(row["start"]),
                    "length": int(row["length"]),
                    "score_db": score,
                    "threshold_db": threshold,
                    "prediction": int(score >= threshold),
                }
            )
        tables[split] = table
        metric_rows.append(metric_row(ENERGY_METHOD, split, prediction_records(table)))
        last = metric_rows[-1]
        log(
            f"{split}: file_equal BA={last['file_equal_balanced_accuracy']:.4f} "
            f"pooled BA={last['pooled_balanced_accuracy']:.4f} "
            f"(tn={last['tn']} fp={last['fp']} fn={last['fn']} tp={last['tp']})"
        )

    write_json(energy_dir / "selection.json", selection_record)
    write_csv(energy_dir / "test_window_predictions.csv", tables[TEST_SPLIT])
    write_csv(energy_dir / "heldout_window_predictions.csv", tables[HELDOUT_SPLIT])
    write_csv(energy_dir / "metrics.csv", metric_rows)
    save_energy_plot(
        energy_dir / "energy_distributions.png", splits, scores, slices, threshold
    )
    write_json(
        cache_dir / "energy_cache.json",
        {
            "created_at": utc_now(),
            "rows": len(catalog),
            "split_order": list(SPLIT_ORDER),
            "slices": {key: [value.start, value.stop] for key, value in slices.items()},
            "catalog": [
                {
                    "split": row["split"],
                    "row_in_split": row["row_in_split"],
                    "filename": row["filename"],
                    "start": row["start"],
                }
                for row in catalog
            ],
            **cache_info,
        },
    )
    write_json(
        energy_dir / "stage_manifest.json",
        {
            "created_at": utc_now(),
            "stage": ENERGY_METHOD,
            "run_command": run_command(),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "library_versions": library_versions(),
            "indices": {
                "old": {"path": str(args.old_index), "sha256": sha256_file(args.old_index)},
                "heldout": {
                    "path": str(args.heldout_index),
                    "sha256": sha256_file(args.heldout_index),
                },
                "expected_sha256": {
                    "old": OLD_INDEX_SHA256,
                    "heldout": HELDOUT_INDEX_SHA256,
                },
            },
            "cache": cache_info,
            "signal_source": "RF0_I, RF0_Q only",
            "window_samples": WINDOW_SAMPLES,
            "no_extra_noise": True,
            "label_mapping": {"Background": 0, "DJI Phantom 4 Pro": 1, "DJI Air 2S": 1},
        },
    )
    log(f"wrote energy baseline outputs to {energy_dir}")
    return 0


def save_energy_plot(
    path: Path,
    splits: dict[str, list[dict[str, Any]]],
    scores: np.ndarray,
    slices: dict[str, slice],
    threshold: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    for axis, split in zip(axes, SPLIT_ORDER):
        values = scores[slices[split]]
        labels = np.asarray([int(row["label"]) for row in splits[split]], dtype=np.int64)
        low, high = float(values.min()), float(values.max())
        if high - low < 1e-9:
            high = low + 1.0
        bins = np.linspace(low - 1.0, high + 1.0, 40)
        axis.hist(
            values[labels == 0], bins=bins, alpha=0.65, label="Background", color="tab:blue"
        )
        axis.hist(values[labels == 1], bins=bins, alpha=0.65, label="Drone", color="tab:red")
        axis.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="train+val threshold",
        )
        axis.set_title(f"{split} (n={values.size})")
        axis.set_xlabel("raw-window energy (dB)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("windows")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Energy baseline: complex-mean-removed raw RF0 power per window "
        f"(threshold {threshold:.3f} dB fixed on train+val only)",
        fontsize=11,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# stage: psd-features
# ---------------------------------------------------------------------------
def stage_psd_features(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    cache_dir = output_dir / "_cache"
    psd_dir = output_dir / "psd_svm"
    psd_dir.mkdir(parents=True, exist_ok=True)

    identity = source_identity(args.old_index, args.heldout_index)
    assert_disjoint(identity)
    splits = rows_by_split(args.old_index, args.heldout_index)
    catalog = build_catalog(splits)
    slices = split_slices(splits)

    if args.from_cache:
        features, metadata = load_cache(cache_dir, "psd_features", len(catalog))
        log(f"loaded cached PSD features {features.shape} from {cache_dir}")
    else:
        features, run_metadata = psd_features_for_rows(
            catalog, args.device, args.batch_size, args.n_fft, args.hop_length
        )
        old_hash = sha256_file(args.old_index)
        heldout_hash = sha256_file(args.heldout_index)
        assert_frozen_indexes(args.old_index, args.heldout_index, old_hash, heldout_hash)
        provenance = {
            "source": (
                "rms_normalize(complex RF0 window) in index order "
                "(train, val, test, heldout)"
            ),
            "frontend": (
                "torch.stft(n_fft=1024, hop_length=512, hamming, center=False, "
                "normalized=True, onesided=False) -> |X|^2 mean over time frames -> "
                "10*log10(P + 1e-10) -> fftshift"
            ),
            "index_sha256": {"old": old_hash, "heldout": heldout_hash},
        }
        metadata = {**run_metadata, **write_cache(cache_dir, "psd_features", features, provenance)}
        write_json(
            cache_dir / "psd_features_split_index.json",
            {
                "created_at": utc_now(),
                "split_order": list(SPLIT_ORDER),
                "slices": {key: [value.start, value.stop] for key, value in slices.items()},
                "catalog": [
                    {
                        "split": row["split"],
                        "row_in_split": row["row_in_split"],
                        "filename": row["filename"],
                        "start": row["start"],
                        "length": row["length"],
                    }
                    for row in catalog
                ],
            },
        )
        log(
            f"cached PSD features -> {metadata['file']} "
            f"sha256={metadata['file_sha256']}"
        )

    check: dict[str, Any] = {}
    if args.numpy_check > 0:
        check = verify_features_against_reference(
            splits, features, slices, args.numpy_check, cache_dir
        )
        write_json(psd_dir / "feature_reference_check.json", check)
        log(
            f"numpy reference check: {check['matched']}/{check['checked']} windows, "
            f"max abs diff {check['max_abs_diff']:.3e}"
        )
        if not check["all_within_tolerance"]:
            raise SystemExit("PSD feature cache disagrees with the NumPy reference")

    save_psd_example_plot(psd_dir / "psd_examples.png", splits, features, slices)
    import torch

    write_json(
        psd_dir / "feature_manifest.json",
        {
            "created_at": utc_now(),
            "stage": "psd_features",
            "run_command": run_command(),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "library_versions": library_versions(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "feature_dimension": PSD_BINS,
            "feature_dtype": str(FEATURE_DTYPE.__name__),
            "feature_metadata": metadata,
            "reference_check": check,
            "indices": {
                "old": {"path": str(args.old_index), "sha256": sha256_file(args.old_index)},
                "heldout": {
                    "path": str(args.heldout_index),
                    "sha256": sha256_file(args.heldout_index),
                },
            },
            "no_extra_noise": True,
            "no_per_window_stft_png": True,
        },
    )
    log(f"wrote PSD feature stage outputs to {psd_dir}")
    return 0


def load_or_create_window(cache_dir: Path, n_fft: int = N_FFT) -> np.ndarray:
    """Reuse the cached taper when present so the independent check is exact."""
    path = cache_dir / "psd_window.npy"
    if path.is_file():
        cached = np.load(path)
        if cached.shape == (n_fft,):
            return cached
    window = hamming_window_matrix(n_fft)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, window)
    write_json(
        cache_dir / "psd_window_metadata.json",
        {
            "created_at": utc_now(),
            "file": str(path),
            "shape": [int(n_fft)],
            "dtype": str(window.dtype),
            "definition": "torch.hamming_window(n_fft, periodic=True) (periodic Hamming)",
            "array_sha256": sha256_array(window),
            "file_sha256": sha256_file(path),
        },
    )
    return window


def verify_features_against_reference(
    splits: dict[str, list[dict[str, Any]]],
    features: np.ndarray,
    slices: dict[str, slice],
    count: int,
    cache_dir: Path,
) -> dict[str, Any]:
    """Recompute `count` windows with NumPy and compare against the cached matrix."""
    import h5py

    taper = load_or_create_window(cache_dir)
    per_split = max(1, count // len(SPLIT_ORDER))
    targets: list[tuple[int, dict[str, Any]]] = []
    for split in SPLIT_ORDER:
        rows = splits[split]
        step = max(1, len(rows) // per_split)
        for offset in range(0, len(rows), step):
            if len([item for item in targets if item[1]["split"] == split]) >= per_split:
                break
            targets.append((slices[split].start + offset, rows[offset]))

    details: list[dict[str, Any]] = []
    max_diff = 0.0
    for position, row in targets:
        with h5py.File(row["file_path"], "r") as handle:
            i_values = read_vector_slice(
                handle["RF0_I"], int(row["start"]), WINDOW_SAMPLES
            )
            q_values = read_vector_slice(
                handle["RF0_Q"], int(row["start"]), WINDOW_SAMPLES
            )
        window = (i_values + 1j * q_values).astype(np.complex64, copy=False)
        reference = numpy_reference_psd(window, window_fn=taper)
        difference = float(np.max(np.abs(reference - features[position].astype(np.float64))))
        max_diff = max(max_diff, difference)
        details.append(
            {
                "cache_row": int(position),
                "split": row["split"],
                "filename": row["filename"],
                "start": int(row["start"]),
                "max_abs_diff": difference,
                "allclose": bool(
                    np.allclose(reference, features[position], atol=REFERENCE_TOLERANCE, rtol=0)
                ),
            }
        )
    return {
        "created_at": utc_now(),
        "checked": len(details),
        "tolerance_abs": REFERENCE_TOLERANCE,
        "max_abs_diff": max_diff,
        "matched": sum(1 for item in details if item["allclose"]),
        "all_within_tolerance": all(item["allclose"] for item in details),
        "window_sha256": sha256_array(taper),
        "window_definition": "torch.hamming_window(n_fft, periodic=True)",
        "method": (
            "NumPy: complex mean removal, RMS to 1, strided framing (hop 512), cached "
            "periodic Hamming taper, np.fft.fft / sqrt(n_fft), |X|^2 mean over frames, "
            "10*log10(P + 1e-10), np.fft.fftshift"
        ),
        "windows": details,
    }


def save_psd_example_plot(
    path: Path,
    splits: dict[str, list[dict[str, Any]]],
    features: np.ndarray,
    slices: dict[str, slice],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frequency_mhz = np.fft.fftshift(np.fft.fftfreq(N_FFT, d=1.0 / 100.0))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for axis, split in zip(axes, (TRAIN_SPLIT, HELDOUT_SPLIT)):
        rows = splits[split]
        picks: dict[int, int] = {}
        for position, row in enumerate(rows):
            picks.setdefault(int(row["label"]), position)
        for label, position in sorted(picks.items()):
            axis.plot(
                frequency_mhz,
                features[slices[split].start + position],
                linewidth=0.9,
                label=f"{'Background' if label == 0 else 'Drone'} ({rows[position]['filename']})",
                color="tab:blue" if label == 0 else "tab:red",
            )
        axis.set_title(f"{split}: first window of each class")
        axis.set_xlabel("baseband frequency (MHz, 100 MS/s)")
        axis.set_ylabel("mean temporal PSD (dB)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle(
        "PSD features (1024 bins, Hamming n_fft=1024 hop=512, rms-normalized RF0)",
        fontsize=11,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# stage: psd-svm
# ---------------------------------------------------------------------------
def stage_psd_svm(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    cache_dir = output_dir / "_cache"
    psd_dir = output_dir / "psd_svm"
    psd_dir.mkdir(parents=True, exist_ok=True)

    splits = rows_by_split(args.old_index, args.heldout_index)
    total_rows = sum(len(rows) for rows in splits.values())
    features, feature_metadata = load_cache(cache_dir, "psd_features", total_rows)
    slices = split_slices(splits)
    log(f"PSD features {features.shape} {features.dtype} from {cache_dir}")

    sklearn_info = import_sklearn(args.sklearn_path)
    import joblib
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    train_rows = splits[TRAIN_SPLIT]
    val_rows = splits[VAL_SPLIT]
    x_train = features[slices[TRAIN_SPLIT]].astype(np.float64)
    y_train = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
    x_val = features[slices[VAL_SPLIT]].astype(np.float64)
    y_val = np.asarray([int(row["label"]) for row in val_rows], dtype=np.int64)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    log(
        f"standardizer fitted on {x_train.shape[0]} train windows "
        f"(mean range [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}])"
    )

    candidate_rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for candidate in SVM_CANDIDATES:
        parameters = {key: value for key, value in candidate.items() if key != "id"}
        estimator = SVC(
            class_weight="balanced",
            random_state=SVM_RANDOM_SEED,
            cache_size=1024,
            **parameters,
        )
        estimator.fit(x_train_scaled, y_train)
        val_predictions = estimator.predict(x_val_scaled)
        val_decision = estimator.decision_function(x_val_scaled)
        balanced = float(balanced_accuracy_score(y_val, val_predictions))
        candidate_rows.append(
            {
                "candidate_id": candidate["id"],
                "parameters": json.dumps(parameters, sort_keys=True),
                "val_balanced_accuracy": balanced,
                "val_accuracy": float(np.mean(val_predictions == y_val)),
                "val_decision_min": float(np.min(val_decision)),
                "val_decision_max": float(np.max(val_decision)),
                "support_vectors": int(estimator.n_support_.sum()),
            }
        )
        fitted[candidate["id"]] = estimator
        log(
            f"  candidate {candidate['id']:<18} val BA={balanced:.6f} "
            f"support={candidate_rows[-1]['support_vectors']}"
        )

    best_value = max(row["val_balanced_accuracy"] for row in candidate_rows)
    tied = [
        row["candidate_id"]
        for row in candidate_rows
        if abs(row["val_balanced_accuracy"] - best_value) <= 1e-12
    ]
    selected_id = tied[0]
    selected_estimator = fitted[selected_id]
    selected_candidate = next(
        candidate for candidate in SVM_CANDIDATES if candidate["id"] == selected_id
    )
    log(f"selected SVM candidate {selected_id} (val BA {best_value:.6f})")
    write_csv(psd_dir / "svm_candidates.csv", candidate_rows)

    model_record: dict[str, Any] = {
        "created_at": utc_now(),
        "baseline": PSD_METHOD,
        "feature": {
            "dimension": int(features.shape[1]),
            "dtype": str(features.dtype),
            "source": feature_metadata["file"],
            "array_sha256": feature_metadata["array_sha256"],
            "frontend": feature_metadata.get("frontend"),
        },
        "standardizer": {
            "estimator": FEATURE_SCALER,
            "fitted_on": "old train split only",
            "fitted_splits": list(FIT_SPLITS),
            "fitted_rows": int(x_train.shape[0]),
            "mean": [float(value) for value in scaler.mean_],
            "scale": [float(value) for value in scaler.scale_],
            "var": [float(value) for value in scaler.var_],
            "n_features_in": int(scaler.n_features_in_),
        },
        "svm": {
            "estimator": SVM_ESTIMATOR,
            "class_weight": "balanced",
            "probability": False,
            "random_state": SVM_RANDOM_SEED,
            "selected_candidate_id": selected_id,
            "parameters": {
                key: value for key, value in selected_candidate.items() if key != "id"
            },
            "support_vectors": int(selected_estimator.n_support_.sum()),
            "support_per_class": [int(value) for value in selected_estimator.n_support_],
            "n_iter": [int(value) for value in np.atleast_1d(selected_estimator.n_iter_)],
        },
        "selection": {
            "rule_id": PSD_RULE_ID,
            "metric": "balanced accuracy on the old val split",
            "fitted_on_splits": list(FIT_SPLITS),
            "selected_on_splits": [VAL_SPLIT],
            "candidate_grid": SVM_CANDIDATES,
            "candidates_evaluated": len(candidate_rows),
            "best_val_balanced_accuracy": best_value,
            "tied_candidate_ids": tied,
            "tie_break": "first candidate in the frozen manifest order",
            "excluded_from_selection": {
                "test": f"{args.old_index} split=test",
                "heldout": str(args.heldout_index),
                "statement": (
                    "the candidate grid was frozen before any test/heldout number was "
                    "computed and only val balanced accuracy ranks candidates"
                ),
            },
        },
        "decision_rule": (
            "SVC.decision_function > 0 -> Drone (1); < 0 -> Background (0); "
            "== 0 -> Drone, matching margin > 0 in the ResNet evaluation"
        ),
        "environment": {
            **sklearn_info,
            "joblib_file": str(Path(joblib.__file__).resolve()),
            "joblib_version": joblib.__version__,
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "candidate_table": candidate_rows,
    }
    joblib.dump(
        {"scaler": scaler, "svm": selected_estimator, "record": model_record},
        psd_dir / "model.joblib",
    )
    write_json(psd_dir / "selection.json", model_record)

    tables: dict[str, list[dict[str, Any]]] = {}
    metric_rows: list[dict[str, Any]] = []
    for split in EVAL_SPLITS:
        rows = splits[split]
        scaled = scaler.transform(features[slices[split]].astype(np.float64))
        decision = selected_estimator.decision_function(scaled)
        if not np.isfinite(decision).all():
            raise RuntimeError(f"non-finite decision function for {split}")
        table: list[dict[str, Any]] = []
        for position, (row, value) in enumerate(zip(rows, decision)):
            table.append(
                {
                    "window": PSD_METHOD,
                    "split": split,
                    "row_in_split": position,
                    "filename": row["filename"],
                    "target_code": row["target_code"],
                    "target_name": row["target_name"],
                    "label": int(row["label"]),
                    "block_index": int(row["block_index"]),
                    "window_index": int(row["window_index"]),
                    "start": int(row["start"]),
                    "length": int(row["length"]),
                    "decision_function": float(value),
                    "prediction": int(float(value) > 0.0),
                    "feature_dimension": int(features.shape[1]),
                    "selected_candidate_id": selected_id,
                }
            )
        tables[split] = table
        metric_rows.append(metric_row(PSD_METHOD, split, prediction_records(table)))
        last = metric_rows[-1]
        log(
            f"{split}: file_equal BA={last['file_equal_balanced_accuracy']:.4f} "
            f"pooled BA={last['pooled_balanced_accuracy']:.4f} "
            f"(tn={last['tn']} fp={last['fp']} fn={last['fn']} tp={last['tp']})"
        )

    write_csv(psd_dir / "test_window_predictions.csv", tables[TEST_SPLIT])
    write_csv(psd_dir / "heldout_window_predictions.csv", tables[HELDOUT_SPLIT])
    write_csv(psd_dir / "metrics.csv", metric_rows)
    model_path = psd_dir / "model.joblib"
    write_json(
        psd_dir / "model_file_info.json",
        {
            "file": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
            "contents": ["scaler", "svm", "record"],
        },
    )
    log(f"wrote PSD+SVM baseline outputs to {psd_dir}")
    return 0


# ---------------------------------------------------------------------------
# stage: verify
# ---------------------------------------------------------------------------
def independent_metrics_from_csv(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    """Independent re-implementation: counts straight from the CSV text fields."""
    labels = [int(row["label"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    if any(value not in (0, 1) for value in labels):
        raise ValueError("non-binary label in CSV")
    if any(value not in (0, 1) for value in predictions):
        raise ValueError("non-binary prediction in CSV")
    counts = confusion_counts(labels, predictions)
    values = metric_values(counts)
    return {"counts": counts, "values": values}


def stored_metric_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (row["method"], row["split"]): row for row in read_csv_rows(path)
    }


def verify_stage(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    report: dict[str, Any] = {"created_at": utc_now(), "checks": []}
    failures: list[str] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        report["checks"].append(
            {"name": name, "passed": bool(passed), "evidence": evidence}
        )
        if not passed:
            failures.append(name)
        log(f"  [{'ok' if passed else 'FAIL'}] {name}")

    old_hash = sha256_file(args.old_index)
    heldout_hash = sha256_file(args.heldout_index)
    canonical = (
        args.old_index.resolve() == DEFAULT_OLD_INDEX.resolve()
        and args.heldout_index.resolve() == DEFAULT_HELDOUT_INDEX.resolve()
    )
    check(
        "old index sha256 matches the frozen value",
        old_hash == OLD_INDEX_SHA256 if canonical else True,
        {
            "actual": old_hash,
            "expected": OLD_INDEX_SHA256,
            "canonical_index": canonical,
            "note": None if canonical else "subset index supplied for a debug/smoke run",
        },
    )
    check(
        "heldout index sha256 matches the frozen value",
        heldout_hash == HELDOUT_INDEX_SHA256 if canonical else True,
        {
            "actual": heldout_hash,
            "expected": HELDOUT_INDEX_SHA256,
            "canonical_index": canonical,
            "note": None if canonical else "subset index supplied for a debug/smoke run",
        },
    )

    identity = source_identity(args.old_index, args.heldout_index)
    check(
        "heldout source files are disjoint from old train/val/test files",
        identity["heldout_disjoint_from_old"],
        {
            "overlap": identity["file_overlap"],
            "old_files": len(identity["old_files"]),
            "heldout_files": len(identity["heldout_files"]),
        },
    )
    check(
        "each split has three categories with at least one source file",
        all(
            sorted(info["files_per_category"]) == ["0000", "0010", "0101"]
            and all(len(files) >= 1 for files in info["files_per_category"].values())
            for info in identity["per_split"].values()
        ),
        {
            split: identity["per_split"][split]["files_per_category"]
            for split in identity["per_split"]
        },
    )
    check(
        "canonical indices keep four source files per heldout category",
        (
            all(
                len(files) == 4
                for files in identity["per_split"][HELDOUT_SPLIT][
                    "files_per_category"
                ].values()
            )
            if canonical
            else True
        ),
        {
            "canonical_index": canonical,
            "heldout_files_per_category": identity["per_split"][HELDOUT_SPLIT][
                "files_per_category"
            ],
        },
    )

    if args.preserved is not None and args.preserved.is_file():
        preserved = json.loads(args.preserved.read_text(encoding="utf-8"))
        entries: list[dict[str, Any]] = []
        for entry in preserved["files"]:
            path = Path(entry["path"])
            if not path.is_file():
                entries.append({"path": str(path), "match": False, "reason": "missing"})
                continue
            size_match = path.stat().st_size == entry["bytes"]
            if args.skip_frozen_hashes:
                entries.append(
                    {
                        "path": str(path),
                        "bytes_match": size_match,
                        "sha256_match": None,
                        "match": size_match,
                    }
                )
                continue
            actual = sha256_file(path)
            entries.append(
                {
                    "path": str(path),
                    "bytes_match": size_match,
                    "sha256_match": actual == entry["sha256"],
                    "match": size_match and actual == entry["sha256"],
                }
            )
        check(
            "the 14 frozen inputs (original code, index, checkpoints) are unchanged",
            all(item["match"] for item in entries),
            {
                "checked": len(entries),
                "hashes_skipped": bool(args.skip_frozen_hashes),
                "files": entries,
            },
        )

    splits = rows_by_split(args.old_index, args.heldout_index)
    slices = split_slices(splits)
    cache_dir = output_dir / "_cache"
    all_rows = build_catalog(splits)

    method_specs = (
        (ENERGY_METHOD, output_dir / "energy_threshold", "score_db"),
        (PSD_METHOD, output_dir / "psd_svm", "decision_function"),
    )
    for method, directory, score_field in method_specs:
        metrics_path = directory / "metrics.csv"
        if not metrics_path.is_file():
            check(f"{method}: metrics.csv exists", False, str(metrics_path))
            continue
        stored = stored_metric_rows(metrics_path)
        for split in EVAL_SPLITS:
            csv_path = directory / f"{split}_window_predictions.csv"
            if not csv_path.is_file():
                check(f"{method}/{split}: prediction CSV exists", False, str(csv_path))
                continue
            rows = read_csv_rows(csv_path)
            check(
                f"{method}/{split}: one row per index window, same order",
                len(rows) == len(splits[split])
                and all(
                    row["filename"] == splits[split][index]["filename"]
                    and int(row["start"]) == splits[split][index]["start"]
                    and int(row["label"]) == int(splits[split][index]["label"])
                    for index, row in enumerate(rows)
                ),
                {"csv_rows": len(rows), "index_rows": len(splits[split])},
            )
            check(
                f"{method}/{split}: scores are finite and predictions binary",
                all(
                    np.isfinite(float(row[score_field]))
                    and int(row["prediction"]) in (0, 1)
                    for row in rows
                ),
                {"score_field": score_field},
            )
            independent = independent_metrics_from_csv(rows)
            counts = independent["counts"]
            check(
                f"{method}/{split}: confusion counts conserve the window total",
                counts["tn"] + counts["fp"] + counts["fn"] + counts["tp"] == len(rows),
                counts,
            )
            stored_row = stored.get((method, split))
            if stored_row is None:
                check(f"{method}/{split}: metrics.csv has this method/split", False, None)
                continue
            checks_match = (
                counts["tn"] == int(stored_row["tn"])
                and counts["fp"] == int(stored_row["fp"])
                and counts["fn"] == int(stored_row["fn"])
                and counts["tp"] == int(stored_row["tp"])
                and abs(
                    independent["values"]["accuracy"] - float(stored_row["pooled_accuracy"])
                )
                < 1e-12
                and abs(
                    independent["values"]["balanced_accuracy"]
                    - float(stored_row["pooled_balanced_accuracy"])
                )
                < 1e-12
                and abs(
                    independent["values"]["precision"] - float(stored_row["pooled_precision"])
                )
                < 1e-12
                and abs(
                    independent["values"]["f1"] - float(stored_row["pooled_f1"])
                )
                < 1e-12
            )
            check(
                f"{method}/{split}: pooled metrics recomputed from the CSV agree",
                checks_match,
                {
                    "csv_counts": counts,
                    "stored_counts": {
                        key: int(stored_row[key]) for key in ("tn", "fp", "fn", "tp")
                    },
                    "csv_accuracy": independent["values"]["accuracy"],
                    "stored_accuracy": float(stored_row["pooled_accuracy"]),
                },
            )
            detail = json.loads(stored_row["detail_json"])
            file_equal = detail["file_equal"]
            recalculated = file_equal_metrics(
                [
                    {
                        "filename": row["filename"],
                        "target_code": row["target_code"],
                        "label": int(row["label"]),
                        "prediction": int(row["prediction"]),
                    }
                    for row in rows
                ]
            )
            comparisons = {
                "file_equal_balanced_accuracy": (
                    recalculated["file_equal_balanced_accuracy"],
                    float(stored_row["file_equal_balanced_accuracy"]),
                ),
                "file_equal_drone_recall": (
                    recalculated["drone_file_equal_recall"],
                    float(stored_row["file_equal_drone_recall"]),
                ),
                "file_equal_background_specificity": (
                    recalculated["background_file_equal_specificity"],
                    float(stored_row["file_equal_background_specificity"]),
                ),
                "file_equal_background_fpr": (
                    recalculated["background_file_equal_fpr"],
                    float(stored_row["file_equal_background_fpr"]),
                ),
            }
            check(
                f"{method}/{split}: file-equal metrics recomputed from the CSV agree",
                all(
                    abs(left - right) < 1e-12
                    for left, right in comparisons.values()
                ),
                {
                    key: {"recomputed": value[0], "stored": value[1]}
                    for key, value in comparisons.items()
                },
            )
            check(
                f"{method}/{split}: per-file rates are all in [0, 1]",
                all(
                    0.0 <= value <= 1.0
                    for rates in file_equal["per_file_rates"].values()
                    for value in rates.values()
                ),
                {"files": file_equal["files_per_category"]},
            )

    if not (cache_dir / "psd_features_metadata.json").is_file():
        check("PSD feature cache exists", False, str(cache_dir))
        report["failures"] = failures
        report["passed"] = False
        write_json(output_dir / "verification.json", report)
        return 3

    features, feature_metadata = load_cache(cache_dir, "psd_features", len(all_rows))
    check(
        "PSD feature cache matches its recorded dtype/shape/SHA256",
        list(features.shape) == [len(all_rows), PSD_BINS]
        and str(features.dtype) == "float32"
        and sha256_array(features) == feature_metadata["array_sha256"],
        {
            "rows": int(features.shape[0]),
            "dimension": int(features.shape[1]),
            "dtype": str(features.dtype),
            "sha256": feature_metadata["array_sha256"],
            "source_index": feature_metadata.get("index_sha256"),
        },
    )
    energy_scores, energy_metadata = load_cache(cache_dir, "energy_scores", len(all_rows))
    check(
        "energy score cache matches its recorded dtype/shape/SHA256",
        list(energy_scores.shape) == [len(all_rows)]
        and sha256_array(energy_scores) == energy_metadata["array_sha256"],
        {"rows": int(energy_scores.shape[0]), "sha256": energy_metadata["array_sha256"]},
    )

    psd_selection = json.loads((output_dir / "psd_svm" / "selection.json").read_text(encoding="utf-8"))
    scaler_record = psd_selection["standardizer"]
    train_slice = slices[TRAIN_SPLIT]
    check(
        "standardizer was fitted on the old train split only",
        scaler_record["fitted_rows"] == len(splits[TRAIN_SPLIT])
        and scaler_record["fitted_on"] == "old train split only",
        {
            "fitted_rows": scaler_record["fitted_rows"],
            "train_rows": len(splits[TRAIN_SPLIT]),
            "fitted_splits": scaler_record["fitted_splits"],
        },
    )
    sampled_bins = list(range(0, PSD_BINS, 97))
    recomputed_mean_ok = all(
        abs(
            float(scaler_record["mean"][index])
            - float(features[train_slice, index].astype(np.float64).mean())
        )
        < 1e-6
        for index in sampled_bins
    )
    recomputed_scale_ok = all(
        abs(
            float(scaler_record["scale"][index])
            - float(features[train_slice, index].astype(np.float64).std(ddof=0))
        )
        < 1e-6
        for index in sampled_bins
    )
    check(
        "standardizer statistics recomputed from the cached train features agree",
        recomputed_mean_ok and recomputed_scale_ok,
        {"sampled_bins": sampled_bins, "mean_ok": recomputed_mean_ok, "scale_ok": recomputed_scale_ok},
    )
    check(
        "SVM selection only used the old val split and a frozen candidate grid",
        psd_selection["selection"]["selected_on_splits"] == [VAL_SPLIT]
        and psd_selection["selection"]["fitted_on_splits"] == list(FIT_SPLITS)
        and len(psd_selection["selection"]["candidate_grid"]) == len(SVM_CANDIDATES)
        and psd_selection["selection"]["candidate_grid"] == SVM_CANDIDATES,
        {
            "selected_on": psd_selection["selection"]["selected_on_splits"],
            "candidates": len(psd_selection["selection"]["candidate_grid"]),
            "selected": psd_selection["svm"]["selected_candidate_id"],
        },
    )

    energy_selection = json.loads(
        (output_dir / "energy_threshold" / "selection.json").read_text(encoding="utf-8")
    )
    threshold = float(energy_selection["threshold_db"])
    selection_scores = np.concatenate(
        [energy_scores[slices[split]] for split in SELECT_SPLITS]
    )
    selection_labels = np.concatenate(
        [
            np.asarray([int(row["label"]) for row in splits[split]], dtype=np.int64)
            for split in SELECT_SPLITS
        ]
    )
    recomputed = select_energy_threshold(selection_labels, selection_scores)
    check(
        "energy threshold recomputed from the cache reproduces selection.json",
        abs(recomputed["threshold_db"] - threshold) < 1e-12
        and abs(
            recomputed["balanced_accuracy"]
            - float(energy_selection["train_val_metrics"]["balanced_accuracy"])
        )
        < 1e-12,
        {
            "stored": threshold,
            "recomputed": recomputed["threshold_db"],
            "candidates": recomputed["candidates_evaluated"],
        },
    )
    check(
        "energy selection touched only train/val scores",
        energy_selection["selection_data"]["splits"] == list(SELECT_SPLITS)
        and energy_selection["selection_data"]["score_sha256"]
        == sha256_array(selection_scores),
        {
            "selection_splits": energy_selection["selection_data"]["splits"],
            "sha256": energy_selection["selection_data"]["score_sha256"],
        },
    )

    reference_check = verify_features_against_reference(
        splits, features, slices, 8, cache_dir
    )
    check(
        "cached PSD features match an independent NumPy/FFT re-derivation",
        reference_check["all_within_tolerance"],
        {
            "checked": reference_check["checked"],
            "max_abs_diff": reference_check["max_abs_diff"],
            "tolerance": reference_check["tolerance_abs"],
        },
    )
    report["psd_reference_check"] = reference_check
    report["failures"] = failures
    report["passed"] = not failures
    write_json(output_dir / "verification.json", report)
    log(
        f"verification: {len(report['checks']) - len(failures)}/{len(report['checks'])} "
        "checks passed"
    )
    return 0 if not failures else 3


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------
ALL_METHODS = ((ENERGY_METHOD, "energy_threshold"), (PSD_METHOD, "psd_svm"))


def stage_report(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    verification_path = output_dir / "verification.json"
    if not verification_path.is_file():
        raise SystemExit("run the verify stage first")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    energy_selection = json.loads(
        (output_dir / "energy_threshold" / "selection.json").read_text(encoding="utf-8")
    )
    psd_selection = json.loads(
        (output_dir / "psd_svm" / "selection.json").read_text(encoding="utf-8")
    )

    comparison_rows: list[dict[str, Any]] = []
    for method, folder in ALL_METHODS:
        directory = output_dir / folder
        stored = stored_metric_rows(directory / "metrics.csv")
        for split in EVAL_SPLITS:
            row = stored[(method, split)]
            detail = json.loads(row["detail_json"])
            file_equal = detail["file_equal"]
            pooled = detail["pooled"]
            comparison_rows.append(
                {
                    "method": method,
                    "split": split,
                    "scope": "file_equal",
                    "windows": int(row["windows"]),
                    "tn": int(row["tn"]),
                    "fp": int(row["fp"]),
                    "fn": int(row["fn"]),
                    "tp": int(row["tp"]),
                    "accuracy": file_equal["file_equal_balanced_accuracy"],
                    "balanced_accuracy": row["file_equal_balanced_accuracy"],
                    "precision": pooled["precision"],
                    "drone_recall": row["file_equal_drone_recall"],
                    "background_specificity": row["file_equal_background_specificity"],
                    "f1": pooled["f1"],
                    "background_file_equal_fpr": row["file_equal_background_fpr"],
                    "phantom_file_equal_recall": row["file_equal_phantom_recall"],
                    "air2s_file_equal_recall": row["file_equal_air2s_recall"],
                    "pooled_accuracy": pooled["accuracy"],
                    "pooled_balanced_accuracy": pooled["balanced_accuracy"],
                    "pooled_precision": pooled["precision"],
                    "pooled_drone_recall": pooled["drone_recall"],
                    "pooled_background_specificity": pooled["background_specificity"],
                    "pooled_f1": pooled["f1"],
                    "score_field": (
                        "score_db" if method == ENERGY_METHOD else "decision_function"
                    ),
                }
            )
    write_csv(output_dir / "comparison_metrics.csv", comparison_rows)

    identity = source_identity(args.old_index, args.heldout_index)
    manifest = build_manifest(args, identity, energy_selection, psd_selection, verification)
    write_json(output_dir / "experiment_manifest.json", manifest)
    write_report(
        output_dir / "report.md",
        output_dir,
        comparison_rows,
        energy_selection,
        psd_selection,
        verification,
        manifest,
    )
    log(
        "wrote comparison_metrics.csv, experiment_manifest.json and report.md to "
        f"{output_dir}"
    )
    return 0


def build_manifest(
    args: argparse.Namespace,
    identity: dict[str, Any],
    energy_selection: dict[str, Any],
    psd_selection: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    output_dir = args.output_dir
    cache_dir = output_dir / "_cache"
    caches: dict[str, Any] = {}
    for name in ("energy_scores", "energy_powers", "psd_features"):
        path = cache_dir / f"{name}_metadata.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            caches[name] = {
                "file": record["file"],
                "shape": record["shape"],
                "dtype": record["dtype"],
                "array_sha256": record["array_sha256"],
                "file_sha256": record["file_sha256"],
            }
    rows = read_csv_rows(output_dir / "comparison_metrics.csv")
    return {
        "created_at": utc_now(),
        "task": "BASE2 stage-1 classical clean baselines (energy threshold, PSD+SVM)",
        "run_command": run_command(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "library_versions": library_versions(),
        "sklearn_environment": psd_selection["environment"],
        "random_seeds": {
            "svm_random_state": SVM_RANDOM_SEED,
            "energy_threshold_selection": "deterministic sweep, no RNG",
            "psd_feature_extraction": "deterministic, no RNG (no AWGN in BASE2)",
        },
        "indices": {
            "old": {
                "path": str(args.old_index),
                "sha256": sha256_file(args.old_index),
                "expected_sha256": OLD_INDEX_SHA256,
                "splits": [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT],
            },
            "heldout": {
                "path": str(args.heldout_index),
                "sha256": sha256_file(args.heldout_index),
                "expected_sha256": HELDOUT_INDEX_SHA256,
                "splits": [HELDOUT_SPLIT],
            },
        },
        "source_identity": identity,
        "split_roles": {
            "fit": {"splits": [TRAIN_SPLIT], "used_for": ["standardizer", "SVM training"]},
            "select": {
                "splits": list(SELECT_SPLITS),
                "used_for": ["energy threshold", "SVM kernel and hyper-parameters"],
            },
            "evaluate": {"splits": list(EVAL_SPLITS), "used_for": ["reporting only"]},
            "heldout_never_used_for_selection": True,
        },
        "features": {
            "energy": {
                "dimension": 1,
                "formula": energy_selection["score_formula"],
                "normalization": "none (raw relative recording power)",
            },
            "psd": {
                "dimension": PSD_BINS,
                "dtype": str(FEATURE_DTYPE.__name__),
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "hop_ratio": HOP_LENGTH / N_FFT,
                "window": "hamming",
                "onesided": False,
                "normalized_stft": True,
                "fftshift": True,
                "eps": PSD_EPS,
                "frequency_reduction": "mean over time frames",
                "normalization": (
                    "dronerfa_data.rms_normalize (complex mean removed, RMS scaled to 1)"
                ),
            },
        },
        "caches": caches,
        "selection_rules": {
            "energy": {
                "rule_id": energy_selection["rule_id"],
                "threshold_db": energy_selection["threshold_db"],
                "decision_rule": energy_selection["decision_rule"],
                "candidate_rule": energy_selection["candidate_rule"],
                "tie_break": energy_selection["tie_break"],
                "selected_on": energy_selection["selection_data"]["splits"],
                "candidates_evaluated": energy_selection["candidates_evaluated"],
            },
            "psd_svm": {
                "rule_id": psd_selection["selection"]["rule_id"],
                "selected_candidate_id": psd_selection["svm"]["selected_candidate_id"],
                "parameters": psd_selection["svm"]["parameters"],
                "candidates": len(psd_selection["selection"]["candidate_grid"]),
                "best_val_balanced_accuracy": psd_selection["selection"][
                    "best_val_balanced_accuracy"
                ],
                "selected_on": psd_selection["selection"]["selected_on_splits"],
                "tie_break": psd_selection["selection"]["tie_break"],
            },
        },
        "metrics": {
            "primary_scope": "file_equal (four source files per category weighted equally)",
            "secondary_scope": "pooled (window counts)",
            "reported": [
                "accuracy",
                "balanced_accuracy",
                "precision",
                "drone_recall",
                "background_specificity",
                "f1",
                "tn",
                "fp",
                "fn",
                "tp",
            ],
            "comparison_metrics_csv_columns": list(rows[0]) if rows else [],
        },
        "constraints_observed": {
            "clean_condition_only": True,
            "no_extra_awgn": True,
            "rf0_only": True,
            "window_samples": WINDOW_SAMPLES,
            "no_nfft4_comparison": True,
            "no_new_resnet_training": True,
            "no_per_window_stft_png": True,
            "frozen_inputs_untouched": True,
        },
        "verification": {
            "path": str(output_dir / "verification.json"),
            "passed": verification["passed"],
            "checks": len(verification["checks"]),
            "failures": verification["failures"],
        },
        "outputs": {
            "energy_threshold": [
                "selection.json",
                "test_window_predictions.csv",
                "heldout_window_predictions.csv",
                "metrics.csv",
                "energy_distributions.png",
                "stage_manifest.json",
            ],
            "psd_svm": [
                "model.joblib",
                "selection.json",
                "svm_candidates.csv",
                "test_window_predictions.csv",
                "heldout_window_predictions.csv",
                "metrics.csv",
                "psd_examples.png",
                "feature_manifest.json",
                "feature_reference_check.json",
                "model_file_info.json",
            ],
            "root": [
                "comparison_metrics.csv",
                "experiment_manifest.json",
                "report.md",
                "verification.json",
            ],
            "cache": [
                "energy_scores.npy",
                "energy_scores_metadata.json",
                "energy_powers.npy",
                "energy_powers_metadata.json",
                "energy_cache.json",
                "psd_features.npy",
                "psd_features_metadata.json",
                "psd_features_split_index.json",
            ],
        },
    }


def write_report(
    path: Path,
    output_dir: Path,
    comparison_rows: Sequence[dict[str, Any]],
    energy_selection: dict[str, Any],
    psd_selection: dict[str, Any],
    verification: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    def find(method: str, split: str) -> dict[str, Any]:
        for row in comparison_rows:
            if row["method"] == method and row["split"] == split:
                return row
        raise KeyError((method, split))

    def percent(value: Any) -> str:
        return f"{100.0 * float(value):.2f}%"

    lines: list[str] = []
    lines.append("# BASE2：第一阶段两个传统基线（clean 条件）")
    lines.append("")
    lines.append(f"生成时间：{manifest['created_at']}")
    lines.append("")
    lines.append("## 1. 任务与范围")
    lines.append("")
    lines.append(
        "本报告只覆盖 BASE2：clean 条件下用 RF0 复 I/Q（1,000,000 点/窗）实现并评价两个"
        "传统二分类基线，标签为 Background=0、Drone(Phantom 4 Pro + Air 2S)=1。"
        "不训练新 ResNet、不做 AWGN 对照、不做 n_fft 四档比较、不进入第二阶段。"
    )
    lines.append("")
    lines.append("## 2. 冻结输入与划分")
    lines.append("")
    lines.append(
        f"- 旧索引 `{manifest['indices']['old']['path']}`，SHA256 "
        f"`{manifest['indices']['old']['sha256']}`（train/val/test）。"
    )
    lines.append(
        f"- 新留出索引 `{manifest['indices']['heldout']['path']}`，SHA256 "
        f"`{manifest['indices']['heldout']['sha256']}`。"
    )
    lines.append(
        f"- 两套索引的源文件交集 {manifest['source_identity']['file_overlap_count']} 个；"
        "新 12 个文件只用于最终 heldout clean 评价。"
    )
    lines.append("")
    lines.append("| split | 窗口数 | 背景文件 | Phantom 文件 | Air 2S 文件 |")
    lines.append("|---|---|---|---|---|")
    for split in SPLIT_ORDER:
        info = manifest["source_identity"]["per_split"][split]
        files = info["files_per_category"]
        lines.append(
            f"| {split} | {info['windows']} | {len(files.get('0000', []))} | "
            f"{len(files.get('0010', []))} | {len(files.get('0101', []))} |"
        )
    lines.append("")
    lines.append("## 3. 选择规则（只用旧 train/val）")
    lines.append("")
    lines.append("### 3.1 能量阈值")
    lines.append("")
    lines.append(f"- 分数：`{energy_selection['score_formula']}`（原始相对功率，不做 RMS 归一化）。")
    lines.append(f"- 决策：{energy_selection['decision_rule']}。")
    lines.append(
        f"- 候选：{energy_selection['candidate_rule']}，共 "
        f"{energy_selection['candidates_evaluated']} 个。"
    )
    lines.append(
        f"- 选择数据：{energy_selection['selection_data']['splits']}，"
        f"{energy_selection['selection_data']['windows']} 窗（背景 "
        f"{energy_selection['selection_data']['class_counts']['background']}、无人机 "
        f"{energy_selection['selection_data']['class_counts']['drone']}）。"
    )
    lines.append(
        f"- 规则：{energy_selection['selection_metric']}，平局取 "
        f"{energy_selection['tie_break']}；最佳 train+val BA="
        f"{percent(energy_selection['train_val_metrics']['balanced_accuracy'])}，"
        f"并列候选 {energy_selection['candidates_with_best_score']} 个。"
    )
    lines.append(
        f"- 最终唯一阈值：**{energy_selection['threshold_db']:.6f} dB**"
        "（test/heldout 未进入选择）。"
    )
    lines.append("")
    lines.append("各 split 分数分布（dB）：")
    lines.append("")
    lines.append("| split | 窗口数 | 背景均值 | 无人机均值 | 最小 | 最大 |")
    lines.append("|---|---|---|---|---|---|")
    for split in SPLIT_ORDER:
        summary = energy_selection["per_split_score_summary"][split]
        lines.append(
            f"| {split} | {summary['windows']} | {summary['mean_background']:.3f} | "
            f"{summary['mean_drone']:.3f} | {summary['min']:.3f} | {summary['max']:.3f} |"
        )
    lines.append("")
    lines.append("### 3.2 PSD + SVM")
    lines.append("")
    lines.append(
        "- 特征：`rms_normalize` 后的复窗，Hamming n_fft=1024 / hop=512、双边复频谱、"
        "沿时间平均功率、`10*log10(P+1e-10)`、fftshift，固定 1024 维。"
    )
    lines.append(
        f"- 标准化：{psd_selection['standardizer']['estimator']}，只拟合旧 train"
        f"（{psd_selection['standardizer']['fitted_rows']} 窗）。"
    )
    lines.append(
        f"- 候选网格（选择前冻结，{len(psd_selection['selection']['candidate_grid'])} 个）："
        "linear / rbf / poly3 / sigmoid，`class_weight=balanced`，"
        f"`random_state={psd_selection['svm']['random_state']}`。"
    )
    lines.append(
        f"- 选择规则：{psd_selection['selection']['metric']}，平局取 "
        f"{psd_selection['selection']['tie_break']}；选中 **"
        f"{psd_selection['svm']['selected_candidate_id']}**（val BA="
        f"{percent(psd_selection['selection']['best_val_balanced_accuracy'])}，并列候选 "
        f"{len(psd_selection['selection']['tied_candidate_ids'])} 个）。"
    )
    lines.append(f"- 决策：{psd_selection['decision_rule']}。")
    lines.append("")
    lines.append("## 4. 结果")
    lines.append("")
    lines.append(
        "主口径为**文件等权**：每个类别 4 个源文件各自的检出/误报率先等权平均，两机型再"
        "等权得到无人机召回，BA=0.5×(背景文件等权特异度+无人机等权召回)。pooled 为窗口"
        "汇总口径，仅作补充，两个口径不混用。"
    )
    lines.append("")
    lines.append(
        "| 方法 | 评价集 | 文件等权 BA | 文件等权召回 | 文件等权特异度 | pooled accuracy | "
        "pooled BA | TN/FP/FN/TP |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in comparison_rows:
        lines.append(
            f"| {row['method']} | {row['split']} | {percent(row['balanced_accuracy'])} | "
            f"{percent(row['drone_recall'])} | "
            f"{percent(row['background_specificity'])} | "
            f"{percent(row['pooled_accuracy'])} | "
            f"{percent(row['pooled_balanced_accuracy'])} | "
            f"{row['tn']}/{row['fp']}/{row['fn']}/{row['tp']} |"
        )
    lines.append("")
    lines.append("### 4.1 逐源文件比例")
    lines.append("")
    lines.append("| 方法 | 评价集 | 类别 | 源文件 | 窗口数 | 判为 Drone 比例 |")
    lines.append("|---|---|---|---|---|---|")
    for method, folder in ALL_METHODS:
        stored = stored_metric_rows(output_dir / folder / "metrics.csv")
        for split in EVAL_SPLITS:
            detail = json.loads(stored[(method, split)]["detail_json"])["file_equal"]
            for code, rates in detail["per_file_rates"].items():
                for name, value in rates.items():
                    lines.append(
                        f"| {method} | {split} | {CATEGORY_NAMES.get(code, code)} | {name} | "
                        f"{detail['per_file_windows'][code][name]} | {percent(value)} |"
                    )
    lines.append("")
    lines.append("### 4.2 逐源文件分数范围（heldout）")
    lines.append("")
    lines.append(
        "分数范围直接读自两份逐窗 CSV，用来判断\"分离\"是发生在文件之间还是窗口之间。"
        "能量分数是绝对电平，PSD 特征在每窗内做过 RMS 归一化，因此 SVM 的判决不含绝对增益。"
    )
    lines.append("")
    lines.append("| 方法 | 源文件 | 类别 | 窗口数 | 分数最小 | 分数最大 |")
    lines.append("|---|---|---|---|---|---|")
    for method, folder, field in (
        (ENERGY_METHOD, "energy_threshold", "score_db"),
        (PSD_METHOD, "psd_svm", "decision_function"),
    ):
        predictions = read_csv_rows(output_dir / folder / "heldout_window_predictions.csv")
        grouped: dict[str, list[float]] = {}
        codes: dict[str, str] = {}
        for row in predictions:
            grouped.setdefault(row["filename"], []).append(float(row[field]))
            codes[row["filename"]] = row["target_code"]
        for name in sorted(grouped):
            values = np.asarray(grouped[name], dtype=np.float64)
            lines.append(
                f"| {method} | {name} | {CATEGORY_NAMES.get(codes[name], codes[name])} | "
                f"{values.size} | {values.min():.4f} | {values.max():.4f} |"
            )
    lines.append("")
    lines.append("### 4.3 与已冻结 clean ResNet 结果的对比")
    lines.append("")
    lines.append(
        "只引用已验收的 EVAL6 heldout clean 文件等权 BA，未重新运行或改写 EVAL6 产物。"
    )
    lines.append("")
    lines.append("| 模型 | heldout clean 文件等权 BA |")
    lines.append("|---|---|")
    for key, value in RESNET_REFERENCE["heldout_clean_file_equal_balanced_accuracy"].items():
        lines.append(f"| ResNet18 {key} | {percent(value)} |")
    for method, _folder in ALL_METHODS:
        row = find(method, HELDOUT_SPLIT)
        lines.append(f"| {row['method']} | {percent(row['balanced_accuracy'])} |")
    lines.append("")
    lines.append("## 5. 解释限制")
    lines.append("")
    lines.append(
        "1. **能量阈值可能利用录制功率/增益差异**：分数来自未归一化的原始相对功率，"
        "源文件之间、采集会话之间的增益差异会直接进入分数，因此不能把结果解释为"
        "\"信号能量高于噪声\"的物理结论，也不构成设备/距离不变量的证明。"
    )
    lines.append(
        "2. **PSD+SVM 是传统特征基线**：1024 维平均 PSD 丢弃了时间结构与相位信息，"
        "SVM 只在旧 train 上拟合、旧 val 上选超参数，与 ResNet 的容量和输入形式不可比。"
    )
    lines.append(
        "3. 两者都**不是新的最终模型**，只是第一阶段存在性检测的对照；"
        "test/heldout 数字没有参与阈值、标准化或 SVM 超参数选择。"
    )
    lines.append(
        "4. 源文件划分不等于采集会话独立；heldout 的 12 个文件与旧文件无重叠，但缺少"
        "中心频率/增益/采集时间/地点元数据，因此不能宣称跨会话或跨地点泛化。"
    )
    lines.append(
        "5. 本任务的 heldout 指标与 EVAL6 的四个噪声档不是同一实验设计，"
        "此处只比较 clean 一档。"
    )
    lines.append(
        "6. heldout 上 PSD+SVM 的文件等权与 pooled 指标都是 100%，且 1820 个窗口的判决函数"
        "全部远离 0（每个源文件的最小/最大判决值见 4.2），不是临界翻转造成的假象；"
        "但这一档只有 12 个源文件、3 个类别，不能据此宣称跨会话泛化。能量基线在"
        "`T0000_D00_S0100.mat` 上误报 12.86%，而在其余 3 个背景文件上为 0%–0.71%，"
        "正是源文件间功率差异的直接体现。"
    )
    lines.append("")
    lines.append("## 6. 校验")
    lines.append("")
    lines.append(
        f"`verification.json`：{len(verification['checks']) - len(verification['failures'])}/"
        f"{len(verification['checks'])} 项通过，失败 {len(verification['failures'])} 项。"
    )
    lines.append("")
    lines.append("| 检查 | 结果 |")
    lines.append("|---|---|")
    for item in verification["checks"]:
        lines.append(f"| {item['name']} | {'通过' if item['passed'] else '失败'} |")
    lines.append("")
    lines.append("## 7. 产物")
    lines.append("")
    for section, names in manifest["outputs"].items():
        lines.append(f"- `{section}`: " + ", ".join(f"`{name}`" for name in names))
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--old-index", type=Path, default=DEFAULT_OLD_INDEX)
    parser.add_argument("--heldout-index", type=Path, default=DEFAULT_HELDOUT_INDEX)
    parser.add_argument(
        "--sklearn-path",
        type=Path,
        default=DEFAULT_SKLEARN_PATH,
        help="Local site-packages used only when scikit-learn is not importable.",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    energy = subparsers.add_parser(
        "energy", help="select the energy threshold and evaluate test/heldout"
    )
    add_common_arguments(energy)
    energy.add_argument(
        "--from-cache",
        action="store_true",
        help="reuse the cached energy scores instead of re-reading raw I/Q",
    )
    energy.set_defaults(handler=stage_energy)

    psd_features = subparsers.add_parser(
        "psd-features", help="build the cached PSD feature matrix"
    )
    add_common_arguments(psd_features)
    psd_features.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    psd_features.add_argument("--batch-size", type=int, default=32)
    psd_features.add_argument("--n-fft", type=int, default=N_FFT)
    psd_features.add_argument("--hop-length", type=int, default=HOP_LENGTH)
    psd_features.add_argument(
        "--numpy-check",
        type=int,
        default=8,
        help="windows re-derived with NumPy and compared against the cache",
    )
    psd_features.add_argument("--from-cache", action="store_true")
    psd_features.set_defaults(handler=stage_psd_features)

    psd_svm = subparsers.add_parser(
        "psd-svm", help="standardize, select the SVM on val, evaluate test/heldout"
    )
    add_common_arguments(psd_svm)
    psd_svm.set_defaults(handler=stage_psd_svm)

    verify = subparsers.add_parser(
        "verify", help="recompute metrics from the prediction CSVs independently"
    )
    add_common_arguments(verify)
    verify.add_argument("--preserved", type=Path, default=DEFAULT_PRESERVED)
    verify.add_argument("--skip-frozen-hashes", action="store_true")
    verify.set_defaults(handler=verify_stage)

    report = subparsers.add_parser(
        "report",
        help=(
            "write comparison_metrics.csv, experiment_manifest.json and report.md "
            "from the existing artifacts (idempotent, re-runnable)"
        ),
    )
    add_common_arguments(report)
    report.set_defaults(handler=stage_report)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log(f"command: {run_command()}")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
