"""NFFT4: first-stage STFT-size ablation for the DroneRFa RF0 detector.

Everything except the STFT grid is frozen:

* inputs      RF0_I/RF0_Q, 1,000,000-sample windows, Background=0 / Drone=1
* splits      ``outputs/window_index.csv`` (train/val/test) and
              ``outputs/heldout_preparation_20260917/heldout_window_index.csv``
* training    seeds 2026/2027/2028, 5 epochs, batch 4, Adam lr 1e-3, wd 0,
              A_original training SNR plan ``clean,10,5,0,-5,-10,-12``, AMP on
* validation  conditions ``clean,-10,-12,-15``, fixed AWGN seed 12026
* selection   mean of the four validation balanced accuracies over the 5 epochs,
              earliest epoch wins ties; test and held-out never participate
* frontend    the frozen ``model.SpectrogramFrontEnd`` (periodic Hamming via
              ``torch.hamming_window``, ``center=False``, ``normalized=True``,
              ``onesided=False``, log power, fftshift, per-sample z-score,
              bilinear resize to 256x256)
* inference   CUDA AMP float16, batch 4, margin = float32(logit1) - float32(logit0),
              decision margin > 0

Only ``n_fft``/``hop_length`` change, always with ``hop_length = n_fft // 2``:

    512/256, 1024/512, 2048/1024, 4096/2048

The three ``n_fft=1024`` A_original checkpoints from 2026-09-17 may only be
reused after every reuse prerequisite has been re-derived and matched
(``--reuse-1024``); the check is all-or-nothing and never looks at any test or
held-out score.

Subcommands::

    verify-inputs   frozen hashes + 1024 reuse equivalence (writes JSON evidence)
    frontends       smoke check of the four frontends (finite values, shapes)
    train           train one (n_fft, seed) run, or every missing run
    figures         native STFT + actual 256x256 model-input figures
    evaluate        held-out (4 conditions) and old-test-clean predictions
    summarize       model_condition_metrics / paired_nfft_comparison / nfft_summary
    manifest        experiment_manifest.json over the whole grid
    verify          (see test_nfft_comparison.py --verify-outputs)
    all             verify-inputs -> frontends -> train -> figures -> evaluate
                    -> summarize -> manifest

Outputs live only under
``outputs/stage1_method_comparison_20260919/nfft/``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    snr_name,
    stratified_limit,
)
from evaluate_heldout_models import file_equal_metrics, pooled_metrics
from model import ResNet18, SpectrogramFrontEnd
from train_resnet18 import extract_features, seed_everything, train_epoch

# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------
OUTPUT_ROOT = (
    PROJECT_DIR / "outputs" / "stage1_method_comparison_20260919" / "nfft"
)
OLD_INDEX = PROJECT_DIR / "outputs" / "window_index.csv"
HELDOUT_INDEX = (
    PROJECT_DIR / "outputs" / "heldout_preparation_20260917" / "heldout_window_index.csv"
)
PRESERVED_INPUTS = PROJECT_DIR.parent / "work" / "codex-preserved-inputs-20260917.json"
HELDOUT_CONFIG = PROJECT_DIR.parent / "work" / "heldout-eval-config-20260918.json"

OLD_INDEX_SHA256 = "4e77c7f186c1aae8e04f2e90f2769c0ed399cc94f40b64f99237b65c057b85ac"
HELDOUT_INDEX_SHA256 = (
    "6fe32faeb5d583a359d9e8aa17caa52a7c4567d545272ded25266767cd2cffcf"
)

NFFT_GRID: tuple[int, ...] = (512, 1024, 2048, 4096)
HOP_DIVISOR = 2
IMAGE_SIZE = 256
TRAINING_SEEDS: tuple[int, ...] = (2026, 2027, 2028)
EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
TRAIN_SNRS = "clean,10,5,0,-5,-10,-12"
VALIDATION_SEED = 12026
TRAIN_SAMPLES = 920
VAL_SAMPLES = 450
TEST_SAMPLES = 440
HELDOUT_SAMPLES = 1820
CONDITION_VALUES: dict[str, float | None] = {
    "clean": None,
    "-10dB": -10.0,
    "-12dB": -12.0,
    "-15dB": -15.0,
}
CONDITIONS: tuple[str, ...] = ("clean", "-10dB", "-12dB", "-15dB")
HELDOUT_NOISE_SEED = 22026
TEST_NOISE_SEED_OFFSET = 20_000
VAL_SELECTION_RULE = "mean_balanced_accuracy_over_four_conditions"
TIE_BREAK = "earliest_epoch"
PREDICTION_RULE = "float32(logit_drone) - float32(logit_background) > 0"
BACKGROUND_CODE = "0000"
DRONE_CODES = ("0010", "0101")
CATEGORY_NAMES = {"0000": "background", "0010": "phantom", "0101": "air2s"}
BASELINE_NFFT = 1024
FEATURE_DTYPE = np.float32
METRIC_COLUMNS = (
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
)
REPRESENTATIVE_WINDOW_RULE = (
    "heldout index row 0 of the frozen held-out index "
    "(first row of the file, chosen before any run)"
)

# The three A_original n_fft=1024 runs of 2026-09-17, in the order
# (run directory, seed, checkpoint sha256 from work/codex-preserved-inputs-20260917.json).
REUSE_1024_RUNS: tuple[tuple[Path, int, str], ...] = (
    (
        PROJECT_DIR / "outputs" / "noise_ab_5epochs_20260917",
        2026,
        "298f6fd959c43f6fd69b9399175060d81ae29c49e6aa68d976dbfbe4023fe645",
    ),
    (
        PROJECT_DIR / "outputs" / "noise_ab_replicates_20260917" / "seed_2027",
        2027,
        "b06898b0018576b6e060fe781bd3871fe703500bb3710f65921c658a45d894c7",
    ),
    (
        PROJECT_DIR / "outputs" / "noise_ab_replicates_20260917" / "seed_2028",
        2028,
        "e1c2dce6b0679011b95e819c18d60a252d17a304ad8907bc0c960c5dde75c2ef",
    ),
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


def run_directory(n_fft: int, seed: int) -> Path:
    return OUTPUT_ROOT / nfft_label(n_fft) / f"seed_{seed}"


def run_command() -> str:
    return subprocess.list2cmdline([sys.executable, *sys.argv])


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


def file_equal_metrics_any(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Equal weight per source file inside each category, then drone categories equal.

    Generalises :func:`evaluate_heldout_models.file_equal_metrics` to any number of
    files per category so the old test split (three files per category) can be
    summarised with the same primary rule as the held-out split. A category that
    the sample does not contain at all is reported as ``nan``; the full evaluation
    always contains all three, and ``verification.json`` enforces that.
    """
    grouped: dict[str, list[str]] = {}
    for row in records:
        bucket = grouped.setdefault(row["target_code"], [])
        if row["filename"] not in bucket:
            bucket.append(row["filename"])
    grouped = {key: sorted(value) for key, value in grouped.items()}
    unknown = set(grouped) - {BACKGROUND_CODE, *DRONE_CODES}
    if unknown:
        raise ValueError(f"unexpected categories: {sorted(unknown)}")
    rates: dict[str, dict[str, float]] = {}
    for category, names in grouped.items():
        rates[category] = {}
        for name in names:
            rows = [r for r in records if r["filename"] == name]
            if not rows:
                raise ValueError(f"no rows for file {name}")
            rates[category][name] = sum(int(r["prediction"]) for r in rows) / len(rows)

    def category_rate(code: str) -> float:
        return float(np.mean(list(rates[code].values()))) if code in rates else float("nan")

    background_fpr = category_rate(BACKGROUND_CODE)
    phantom_recall = category_rate("0010")
    air2s_recall = category_rate("0101")
    drone_recall = 0.5 * (phantom_recall + air2s_recall)
    specificity = 1.0 - background_fpr
    balanced = 0.5 * (specificity + drone_recall)
    return {
        "files_per_category": {key: len(value) for key, value in grouped.items()},
        "background_file_equal_fpr": background_fpr,
        "background_file_equal_specificity": specificity,
        "phantom_file_equal_recall": phantom_recall,
        "air2s_file_equal_recall": air2s_recall,
        "drone_file_equal_recall": drone_recall,
        "file_equal_balanced_accuracy": balanced,
        "per_file_rates": rates,
    }


def pooled_metrics_any(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled window counts plus the full metric set used by every report table."""
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    prediction = np.asarray([r["prediction"] for r in records], dtype=np.int64)
    tp = int(np.sum((y == 1) & (prediction == 1)))
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    if tp + tn + fp + fn != len(y):
        raise ValueError("confusion matrix does not conserve the window count")
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    accuracy = (tp + tn) / max(len(y), 1)
    return {
        "windows": int(len(y)),
        "accuracy": accuracy,
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


# ---------------------------------------------------------------------------
# frozen input integrity
# ---------------------------------------------------------------------------
def verify_frozen_inputs() -> dict[str, Any]:
    """Re-hash the frozen inputs before and after the run."""
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
    return {
        "old_index": {"path": str(OLD_INDEX), "sha256": old_hash,
                      "expected": OLD_INDEX_SHA256, "match": old_hash == OLD_INDEX_SHA256},
        "heldout_index": {"path": str(HELDOUT_INDEX), "sha256": heldout_hash,
                          "expected": HELDOUT_INDEX_SHA256,
                          "match": heldout_hash == HELDOUT_INDEX_SHA256},
        "heldout_config": {
            "path": str(HELDOUT_CONFIG),
            "sha256": sha256_file(HELDOUT_CONFIG),
        },
        "preserved_inputs": {
            "reference": str(PRESERVED_INPUTS),
            "reference_sha256": sha256_file(PRESERVED_INPUTS),
            "checked": len(preserved_records),
            "all_match": all(item["match"] for item in preserved_records),
            "files": preserved_records,
        },
        "process_files": {
            name: sha256_file(PROJECT_DIR / name)
            for name in (
                "dronerfa_data.py",
                "model.py",
                "train_resnet18.py",
                "run_noise_ab.py",
                "evaluate_heldout_models.py",
            )
        },
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# n_fft=1024 checkpoint reuse equivalence
# ---------------------------------------------------------------------------
def epoch_orders(seed: int, epochs: int = EPOCHS, samples: int = TRAIN_SAMPLES) -> list[str]:
    """Recompute the paired training orders exactly as ``run_noise_ab.run_arm`` did."""
    orders = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(
            samples, generator=torch.Generator().manual_seed(seed + epoch)
        ).tolist()
        orders.append(sha256_bytes(np.asarray(order, dtype=np.int64).tobytes()))
    return orders


def initial_state_hash(seed: int) -> str:
    """Re-derive the ResNet18 initialisation hash for a training seed."""
    seed_everything(seed)
    model = ResNet18(input_channels=1, num_classes=2)
    result = state_hash(model.state_dict())
    del model
    return result


class _IndexSubsetDataset:
    """View a ``DroneRFaWindowDataset`` through a list of original row indices.

    ``__len__`` reports the original dataset length so the frozen AWGN rule
    ``default_rng(seed + epoch*len(dataset) + index)`` keeps producing exactly the
    same noise for a given window, while ``__getitem__`` only reads the rows the
    caller actually wants. The full validation run keeps every row, so this only
    makes bounded (smoke) runs cheap.
    """

    def __init__(self, dataset: DroneRFaWindowDataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def sampler(self) -> list[int]:
        """Positions to iterate, so a narrow selection never walks the full set."""
        return list(range(len(self.indices)))

    def __getitem__(self, position: int):
        return self.dataset[self.indices[position]]

    def close(self) -> None:
        self.dataset.close()


def build_validation_cache(
    n_fft: int, device: torch.device, max_val_samples: int | None = None
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]], list[dict[str, Any]]]:
    """Four fixed validation conditions, float32 features shared by every seed.

    Mirrors ``run_noise_ab.cache_validation``: the dataset keeps the full validation
    index so the AWGN seed stays ``VALIDATION_SEED + index``, and only the selection
    (identity of the original rows) is narrowed.
    """
    frontend = SpectrogramFrontEnd(
        n_fft=n_fft, hop_length=hop_for(n_fft), image_size=IMAGE_SIZE
    ).to(device)
    frontend.eval()
    cached: dict[str, tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]] = {}
    metadata: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        dataset = DroneRFaWindowDataset(
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
        view = _IndexSubsetDataset(dataset, indices)
        loader = DataLoader(
            view,
            batch_size=BATCH_SIZE,
            sampler=view.sampler(),
            num_workers=0,
            pin_memory=True,
            generator=torch.Generator().manual_seed(VALIDATION_SEED + 30_000),
        )
        tensors: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        started = time.perf_counter()
        with torch.inference_mode():
            for batch, (signals, targets) in enumerate(loader, 1):
                tensors.append(extract_features(frontend, signals.to(device)).cpu())
                labels.append(targets)
                if batch % 25 == 0 or batch == len(loader):
                    log(f"  val cache nfft{n_fft} {condition} {batch}/{len(loader)}")
        xx, yy = torch.cat(tensors), torch.cat(labels)
        cached[condition] = (xx, yy, selected)
        metadata.append(
            {
                "condition": condition,
                "n_fft": n_fft,
                "hop_length": hop_for(n_fft),
                "samples": len(selected),
                "full_dataset_indices": indices,
                "feature_sha256": sha256_bytes(xx.numpy().tobytes()),
                "seconds": time.perf_counter() - started,
            }
        )
        dataset.close()
    del frontend
    torch.cuda.empty_cache()
    return cached, metadata


def verify_reuse_1024(device: torch.device, write_evidence: bool = True) -> dict[str, Any]:
    """Prove the three 2026-09-17 n_fft=1024 A_original runs match this configuration.

    Every prerequisite is re-derived from the artefacts themselves and compared
    against what this script would produce. Nothing about test or held-out scores
    is consulted, and a single mismatch rejects reuse for that seed (the caller then
    retrains it).
    """
    checks: dict[str, Any] = {"seeds": {}, "require_retrain": []}
    expected_snrs = ",".join(snr_name(value) for value in parse_snr_list(TRAIN_SNRS))
    cache_manifest_by_run: dict[Path, list[dict[str, Any]]] = {}
    started = time.perf_counter()

    for run_dir, seed, expected_checkpoint_hash in REUSE_1024_RUNS:
        entry: dict[str, Any] = {
            "run_directory": str(run_dir),
            "training_seed": seed,
            "checkpoint_expected_sha256": expected_checkpoint_hash,
        }
        failures: list[str] = []
        config_path = run_dir / "A_original" / "run_config.json"
        checkpoint_path = run_dir / "A_original" / "best_model.pt"
        selection_path = run_dir / "A_original" / "selection.json"
        history_path = run_dir / "A_original" / "training_history.csv"
        metrics_path = run_dir / "A_original" / "validation_metrics.csv"
        for path in (config_path, checkpoint_path, selection_path, history_path, metrics_path):
            if not path.is_file():
                failures.append(f"missing artefact: {path}")
        if failures:
            entry["failures"] = failures
            entry["eligible_for_reuse"] = False
            checks["seeds"][str(seed)] = entry
            checks["require_retrain"].append(seed)
            continue

        config = read_json(config_path)
        entry["declared_config"] = config

        # 1. hyper-parameters, optimizer, AMP and noise plan
        config_fields = {
            "seed": (config.get("seed"), seed),
            "epochs": (config.get("epochs"), EPOCHS),
            "batch_size": (config.get("batch_size"), BATCH_SIZE),
            "train_snrs": (config.get("train_snrs"), TRAIN_SNRS),
            "train_samples": (config.get("train_samples"), TRAIN_SAMPLES),
            "selection": (config.get("selection"), "mean_balanced_accuracy"),
            "tie_break": (config.get("tie_break"), "earliest epoch"),
            "prediction": (config.get("prediction"), PREDICTION_RULE),
            "index_sha256": (config.get("index_sha256"), OLD_INDEX_SHA256),
            "amp": (bool(config.get("amp")), True),
            "validation_conditions": (list(config.get("validation_conditions", [])), list(CONDITIONS)),
        }
        for key, (actual, expected) in config_fields.items():
            if actual != expected:
                failures.append(f"config.{key}: {actual!r} != {expected!r}")
        frontend = config.get("frontend", {})
        for key, expected in (
            ("n_fft", BASELINE_NFFT),
            ("hop_length", hop_for(BASELINE_NFFT)),
            ("image_size", IMAGE_SIZE),
        ):
            if frontend.get(key) != expected:
                failures.append(f"config.frontend.{key}: {frontend.get(key)!r} != {expected!r}")
        optimizer = config.get("optimizer", {})
        for key, expected in (
            ("type", "Adam"),
            ("learning_rate", LEARNING_RATE),
            ("weight_decay", WEIGHT_DECAY),
        ):
            if optimizer.get(key) != expected:
                failures.append(f"config.optimizer.{key}: {optimizer.get(key)!r} != {expected!r}")
        if not history_path.is_file():
            failures.append("training_history.csv is required to re-derive the orders")
        else:
            history = read_csv_rows(history_path)
            if len(history) != EPOCHS:
                failures.append(f"history has {len(history)} epochs, expected {EPOCHS}")
            recorded = config.get("epoch_order_sha256")
            derived = epoch_orders(seed)
            entry["epoch_order_sha256_recorded"] = recorded
            entry["epoch_order_sha256_derived"] = derived
            if recorded != derived:
                failures.append("epoch order hashes differ from the re-derived paired orders")
            reference_orders = [row["epoch"] for row in history]
            if reference_orders != [str(i) for i in range(1, EPOCHS + 1)]:
                failures.append(f"unexpected epoch numbering in history: {reference_orders}")

        # 2. initial weights
        derived_initial = initial_state_hash(seed)
        entry["initial_model_sha256_recorded"] = config.get("initial_model_sha256")
        entry["initial_model_sha256_derived"] = derived_initial
        if config.get("initial_model_sha256") != derived_initial:
            failures.append(
                "initial ResNet18 weights for this seed do not reproduce the recorded hash"
            )

        # 3. training source snapshot
        snapshot = run_dir / "source_snapshot"
        source_checks = []
        for name in ("dronerfa_data.py", "model.py", "train_resnet18.py", "window_index.csv"):
            snapshot_path = snapshot / name
            if not snapshot_path.is_file():
                failures.append(f"missing source snapshot file: {name}")
                continue
            snapshot_hash = sha256_file(snapshot_path)
            current_path = OLD_INDEX if name == "window_index.csv" else PROJECT_DIR / name
            current_hash = sha256_file(current_path)
            source_checks.append(
                {"name": name, "snapshot_sha256": snapshot_hash,
                 "current_sha256": current_hash, "match": snapshot_hash == current_hash}
            )
            if snapshot_hash != current_hash:
                failures.append(f"source snapshot differs from the frozen file: {name}")
        entry["source_snapshot"] = source_checks

        # 4. validation cache: identical windows and identical noise/features
        cache_path = run_dir / "validation_cache_manifest.json"
        if not cache_path.is_file():
            failures.append("missing validation_cache_manifest.json")
        else:
            reference_cache = read_json(cache_path)
            key = run_dir.resolve()
            if key not in cache_manifest_by_run:
                cache_manifest_by_run[key] = reference_cache
            recorded_by_condition = {row["condition"]: row for row in reference_cache}
            entry["validation_cache_reference"] = {
                condition: {
                    "samples": recorded_by_condition[condition]["samples"],
                    "feature_sha256": recorded_by_condition[condition]["feature_sha256"],
                    "full_dataset_indices_sha256": sha256_bytes(
                        np.asarray(
                            recorded_by_condition[condition]["full_dataset_indices"],
                            dtype=np.int64,
                        ).tobytes()
                    ),
                }
                for condition in CONDITIONS
            }
            if config.get("validation_seed") not in (None, VALIDATION_SEED):
                failures.append(
                    f"validation seed {config.get('validation_seed')} != {VALIDATION_SEED}"
                )

        # 5. selection evidence points at the checkpoint epoch
        selection = read_json(selection_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        epoch = int(checkpoint["epoch"])
        entry["checkpoint_epoch"] = epoch
        entry["selection_json"] = selection
        best_epoch, best_score = None, None
        for row in read_csv_rows(metrics_path):
            row_epoch = int(row["epoch"])
            if row_epoch != epoch:
                continue
            score = float(row["balanced_accuracy"])
            best_score = score if best_score is None else max(best_score, score)
        if best_score is None:
            failures.append(f"validation_metrics.csv has no rows for epoch {epoch}")
        recorded_means = {}
        by_epoch: dict[int, list[float]] = {}
        for row in read_csv_rows(metrics_path):
            by_epoch.setdefault(int(row["epoch"]), []).append(float(row["balanced_accuracy"]))
        for row_epoch, values in by_epoch.items():
            recorded_means[row_epoch] = float(np.mean(values))
        history_means = {
            int(row["epoch"]): float(row["mean_balanced_accuracy"])
            for row in read_csv_rows(history_path)
        }
        if set(recorded_means) != set(history_means):
            failures.append("validation metrics and history cover different epochs")
        else:
            for row_epoch, value in recorded_means.items():
                if abs(value - history_means[row_epoch]) > 1e-9:
                    failures.append(
                        f"epoch {row_epoch}: validation mean {value} != history {history_means[row_epoch]}"
                    )
            best_derived = max(recorded_means, key=lambda item: (recorded_means[item], -item))
            entry["best_epoch_by_rule"] = best_derived
            if selection.get("multi_noise") != best_derived:
                failures.append(
                    f"selection.json multi_noise={selection.get('multi_noise')} but the rule gives {best_derived}"
                )
            if epoch != best_derived:
                failures.append(f"checkpoint epoch {epoch} is not the rule-selected epoch {best_derived}")
        checkpoint_run_config = checkpoint.get("run_config", {})
        entry["checkpoint_run_config_match"] = (
            checkpoint_run_config.get("index_sha256") == OLD_INDEX_SHA256
            and int(checkpoint_run_config.get("seed", -1)) == seed
            and checkpoint_run_config.get("selection") == "mean_balanced_accuracy"
        )
        if not entry["checkpoint_run_config_match"]:
            failures.append("checkpoint run_config does not match seed/index/selection")
        if list(checkpoint.get("frontend_state", {}).keys()) != []:
            failures.append(
                f"unexpected frontend_state keys: {list(checkpoint.get('frontend_state', {}).keys())}"
            )
        entry["checkpoint_metric_epoch_match"] = best_score is not None and epoch == int(
            selection.get("multi_noise", -1)
        )
        del checkpoint

        actual_checkpoint_hash = sha256_file(checkpoint_path)
        entry["checkpoint_sha256"] = actual_checkpoint_hash
        entry["checkpoint_sha256_match"] = actual_checkpoint_hash == expected_checkpoint_hash
        if not entry["checkpoint_sha256_match"]:
            failures.append("checkpoint SHA256 differs from the preserved-input record")

        entry["failures"] = failures
        entry["eligible_for_reuse"] = not failures
        if failures:
            checks["require_retrain"].append(seed)
        checks["seeds"][str(seed)] = entry

    # Re-derive the four-condition validation cache once and compare with the runs.
    log("Re-deriving the n_fft=1024 validation cache for the reuse check")
    _, metadata = build_validation_cache(BASELINE_NFFT, device)
    derived_cache = {row["condition"]: row for row in metadata}
    index_digests = {}
    for condition, row in derived_cache.items():
        index_digests[condition] = sha256_bytes(
            np.asarray(row["full_dataset_indices"], dtype=np.int64).tobytes()
        )
    checks["derived_validation_cache"] = {
        condition: {
            "samples": derived_cache[condition]["samples"],
            "feature_sha256": derived_cache[condition]["feature_sha256"],
            "full_dataset_indices_sha256": index_digests[condition],
            "seconds": derived_cache[condition]["seconds"],
        }
        for condition in CONDITIONS
    }
    for run_dir, seed, _ in REUSE_1024_RUNS:
        entry = checks["seeds"][str(seed)]
        reference = entry.pop("validation_cache_reference", None)
        if reference is None:
            continue
        mismatches = []
        for condition in CONDITIONS:
            if reference[condition]["samples"] != derived_cache[condition]["samples"]:
                mismatches.append(f"{condition}: samples differ")
            if reference[condition]["feature_sha256"] != derived_cache[condition]["feature_sha256"]:
                mismatches.append(f"{condition}: feature sha256 differs")
            if reference[condition]["full_dataset_indices_sha256"] != index_digests[condition]:
                mismatches.append(f"{condition}: validation window identities differ")
        entry["validation_cache_match"] = not mismatches
        entry["validation_cache_mismatches"] = mismatches
        if mismatches:
            entry["failures"] = list(entry.get("failures", [])) + mismatches
            entry["eligible_for_reuse"] = False
            if seed not in checks["require_retrain"]:
                checks["require_retrain"].append(seed)

    checks["all_seeds_eligible"] = all(
        entry.get("eligible_for_reuse") for entry in checks["seeds"].values()
    )
    checks["reuse_scope"] = (
        "Prerequisites only: this check never reads any test or held-out prediction "
        "or metric, and it does not add noise-condition choices to the comparison."
    )
    checks["seconds"] = time.perf_counter() - started
    if write_evidence:
        write_json(OUTPUT_ROOT / "reuse_1024_verification.json", checks)
    for seed, entry in checks["seeds"].items():
        log(
            f"  reuse seed {seed}: eligible={entry.get('eligible_for_reuse')} "
            f"failures={entry.get('failures')}"
        )
    return checks


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
    """Cheap shape/finiteness check of the four frozen frontends."""
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
            "finite": bool(torch.isfinite(features).all()),
            "window_periodic_hamming": bool(
                torch.allclose(window, torch.hamming_window(n_fft), atol=0.0, rtol=0.0)
            ),
            "window_first": float(window[0]),
            "window_last": float(window[-1]),
            "window_symmetric_equals_periodic": bool(
                torch.allclose(
                    window, torch.hamming_window(n_fft, periodic=False), atol=1e-12
                )
            ),
        }
        del frontend, features
        torch.cuda.empty_cache()
    return report


# ---------------------------------------------------------------------------
# training one (n_fft, seed) run
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_cached(
    model: ResNet18,
    cached: dict[str, tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-epoch validation metrics and per-window records, batch size fixed at 4."""
    model.eval()
    metrics_all: list[dict[str, Any]] = []
    records_all: list[dict[str, Any]] = []
    for condition, (xx, yy, rows) in cached.items():
        loader = DataLoader(
            TensorDataset(xx, yy), batch_size=BATCH_SIZE, shuffle=False,
            generator=torch.Generator().manual_seed(VALIDATION_SEED + 40_000),
        )
        labels_all: list[int] = []
        predictions_all: list[int] = []
        total_loss = 0.0
        cursor = 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=True, dtype=torch.float16):
                logits = model(features)
                loss = criterion(logits, labels)
            total_loss += float(loss) * len(labels)
            logits = logits.float()
            margins = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            predictions = (margins > 0.0).astype(np.int64)
            labels_np = labels.cpu().numpy()
            for offset in range(len(labels_np)):
                source = rows[cursor + offset]
                records_all.append(
                    {
                        "condition": condition,
                        "validation_index_in_selection": cursor + offset,
                        "file_path": source["file_path"],
                        "filename": source["filename"],
                        "target_code": source["target_code"],
                        "label": int(labels_np[offset]),
                        "logit_margin": float(margins[offset]),
                        "prediction": int(predictions[offset]),
                    }
                )
            labels_all.extend(labels_np.tolist())
            predictions_all.extend(predictions.tolist())
            cursor += len(labels_np)
        counts = pooled_metrics_any(
            [
                {"label": label, "prediction": prediction, "target_code": source["target_code"],
                 "filename": source["filename"]}
                for label, prediction, source in zip(labels_all, predictions_all, rows)
            ]
        )
        metrics_all.append(
            {
                "condition": condition,
                "loss": total_loss / len(rows),
                "samples": len(rows),
                "accuracy": counts["accuracy"],
                "balanced_accuracy": counts["balanced_accuracy"],
                "precision": counts["precision"],
                "drone_recall": counts["drone_recall"],
                "background_specificity": counts["background_specificity"],
                "f1": counts["f1"],
                "tn": counts["tn"],
                "fp": counts["fp"],
                "fn": counts["fn"],
                "tp": counts["tp"],
            }
        )
    return metrics_all, records_all


def select_epoch(history: list[dict[str, Any]]) -> int:
    """Mean of the four validation balanced accuracies; earliest epoch wins ties."""
    return int(max(history, key=lambda row: row["mean_balanced_accuracy"])["epoch"])


def finalize_interrupted_run(
    n_fft: int,
    seed: int,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Finish the bookkeeping of a run whose script died after its last epoch.

    Only valid when every epoch artefact is present and the rule-selected epoch
    already has both ``epoch_XX.pt`` and ``best_model.pt``. Training itself is
    never redone or altered: the function re-derives the selected epoch from
    ``training_history.csv``, re-checks the checkpoint epoch, prunes the other
    epoch files and writes the completion marker.
    """
    root = output_root or OUTPUT_ROOT
    run_dir = root / nfft_label(n_fft) / f"seed_{seed}"
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"no run_config.json in {run_dir}")
    config = read_json(config_path)
    if config.get("status") in ("completed", "reused_after_equivalence_verification"):
        return {"status": "already_finalized", "config": config}
    history = read_csv_rows(run_dir / "training_history.csv")
    if len(history) < EPOCHS:
        raise RuntimeError(
            f"{run_dir} holds {len(history)}/{EPOCHS} epochs; a partial run must be retrained"
        )
    selected = select_epoch(
        [
            {"epoch": int(row["epoch"]), "mean_balanced_accuracy": float(row["mean_balanced_accuracy"])}
            for row in history
        ]
    )
    selected_checkpoint = run_dir / f"epoch_{selected:02d}.pt"
    best = run_dir / "best_model.pt"
    if not selected_checkpoint.is_file() or not best.is_file():
        raise RuntimeError(f"{run_dir} is missing the selected epoch checkpoint")
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    checkpoint_epoch = int(checkpoint["epoch"])
    del checkpoint
    if checkpoint_epoch != selected:
        raise RuntimeError(
            f"{run_dir}: best_model.pt holds epoch {checkpoint_epoch}, the rule selects {selected}"
        )
    retained, removed = [], []
    for path in sorted(run_dir.glob("epoch_*.pt")):
        if path.name == selected_checkpoint.name:
            retained.append(path.name)
            continue
        path.unlink()
        removed.append(path.name)
    config["selected_epoch"] = selected
    config["retained_epoch_checkpoints"] = retained
    config["removed_epoch_checkpoints"] = removed
    config["completed_at"] = utc_now()
    config["status"] = "completed"
    config["finalized_after_interruption"] = True
    config["finalization_note"] = (
        "the training script was killed after the last epoch; only the bookkeeping "
        "was completed, no training was repeated or changed"
    )
    write_json(config_path, config)
    log(f"Finalized interrupted run {run_dir} at epoch {selected}; removed {len(removed)} epoch files")
    return {"status": "finalized", "config": config}


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
    """Train one (n_fft, seed) run and write every per-run artefact."""
    root = output_root or OUTPUT_ROOT
    run_dir = root / nfft_label(n_fft) / f"seed_{seed}"
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    hop = hop_for(n_fft)
    started_at = utc_now()
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
    frontend = frontend_for(n_fft, device)
    model = ResNet18(input_channels=1, num_classes=2).to(device)
    initial_hash = state_hash(model.state_dict())

    dataset = DroneRFaWindowDataset(
        OLD_INDEX,
        split="train",
        train_snrs=parse_snr_list(TRAIN_SNRS),
        max_samples=max_train_samples,
        seed=seed,
    )
    counts = {label: sum(row["label"] == label for row in dataset.rows) for label in (0, 1)}
    if not all(counts.values()):
        raise ValueError("both training classes are required")
    weights = torch.tensor(
        [len(dataset) / (2 * counts[k]) for k in (0, 1)], device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler(device.type, enabled=True)

    config = {
        "phase": "NFFT4",
        "n_fft": n_fft,
        "hop_length": hop,
        "hop_ratio": hop / n_fft,
        "image_size": IMAGE_SIZE,
        "training_seed": seed,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "train_snrs": TRAIN_SNRS,
        "train_samples": len(dataset),
        "class_counts": counts,
        "initial_model_sha256": initial_hash,
        "validation_seed": VALIDATION_SEED,
        "validation_conditions": list(CONDITIONS),
        "selection": VAL_SELECTION_RULE,
        "tie_break": TIE_BREAK,
        "prediction": PREDICTION_RULE,
        "index_sha256": sha256_file(OLD_INDEX),
        "frontend": {
            "n_fft": n_fft,
            "hop_length": hop,
            "image_size": IMAGE_SIZE,
            "window": "torch.hamming_window(n_fft) periodic",
            "implementation": "model.SpectrogramFrontEnd (frozen)",
        },
        "optimizer": {"type": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
        "amp": True,
        "amp_dtype": "float16",
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "smoke": smoke,
        "started_at": started_at,
        "source_sha256": {
            name: sha256_file(PROJECT_DIR / name)
            for name in ("run_nfft_comparison.py", "dronerfa_data.py", "model.py", "train_resnet18.py")
        },
        "heldout_index_sha256": sha256_file(HELDOUT_INDEX),
        "heldout_samples": HELDOUT_SAMPLES,
        "test_samples": TEST_SAMPLES,
        "status": "running",
    }

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
                Subset(dataset, order), batch_size=BATCH_SIZE, shuffle=False,
                num_workers=0, pin_memory=True,
                generator=torch.Generator().manual_seed(seed + epoch + 50_000),
            )
            log(f"  epoch {epoch}/{epochs} nfft{n_fft} seed {seed}")
            train_loss, train_metrics = train_epoch(
                frontend, model, loader, criterion, optimizer, scaler, device, True
            )
            metrics, records = evaluate_cached(model, cached, criterion, device)
            mean_score = float(np.mean([row["balanced_accuracy"] for row in metrics]))
            clean = next(row for row in metrics if row["condition"] == "clean")
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                    "clean_balanced_accuracy": clean["balanced_accuracy"],
                    "mean_balanced_accuracy": mean_score,
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
                    "selection_rule": VAL_SELECTION_RULE,
                    "tie_break": TIE_BREAK,
                    "prediction_rule": PREDICTION_RULE,
                    "training_seed": seed,
                    "run_config": config,
                },
                run_dir / f"epoch_{epoch:02d}.pt",
            )
            selection = {
                "n_fft": n_fft,
                "training_seed": seed,
                "rule": VAL_SELECTION_RULE,
                "tie_break": TIE_BREAK,
                "mean_balanced_accuracy_by_epoch": {
                    str(row["epoch"]): row["mean_balanced_accuracy"] for row in history
                },
                "selected_epoch": select_epoch(history),
                "best_multi_noise_epoch": select_epoch(history),
            }
            write_json(run_dir / "selection.json", selection)
            if selection["selected_epoch"] == epoch:
                shutil.copyfile(run_dir / f"epoch_{epoch:02d}.pt", checkpoint_path)
            log(
                f"  epoch {epoch}: mean BA={mean_score:.6f} "
                f"({history[-1]['seconds']:.1f}s) selected={selection['selected_epoch']}"
            )
            for metric in metrics:
                log(
                    f"    {metric['condition']}: BA={metric['balanced_accuracy']:.6f} "
                    f"tp={metric['tp']} tn={metric['tn']} fp={metric['fp']} fn={metric['fn']}"
                )
    finally:
        dataset.close()

    config["epoch_order_sha256"] = order_hashes
    config["selected_epoch"] = select_epoch(history)
    config["status"] = "training_complete"
    config["reused_from"] = None
    write_json(run_dir / "run_config.json", config)
    # Only the selected epoch keeps its full checkpoint (identical to best_model.pt);
    # the other epochs stay auditable through validation_metrics.csv and
    # validation_scores_epoch_XX.csv.
    selected = int(config["selected_epoch"])
    retained, removed = [], []
    for path in sorted(run_dir.glob("epoch_*.pt")):
        if path.name == f"epoch_{selected:02d}.pt":
            retained.append(path.name)
            continue
        path.unlink()
        removed.append(path.name)
    config["retained_epoch_checkpoints"] = retained
    config["removed_epoch_checkpoints"] = removed
    config["completed_at"] = utc_now()
    config["status"] = "completed"
    write_json(run_dir / "run_config.json", config)
    log(f"  kept {retained}, removed {len(removed)} non-selected epoch checkpoints")
    release_run_resources(model, optimizer, scaler, cached)
    return {"run_directory": str(run_dir), "config": config, "history": history}


def release_run_resources(
    model: Any = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    cached: dict[str, Any] | None = None,
) -> None:
    """Best-effort cleanup that must never abort the surrounding pipeline.

    ``run_config.json`` is finalised before this runs, so a failure here cannot
    leave a half-written run behind; it is reported and execution continues with
    the next (n_fft, seed) configuration.
    """
    try:
        del model, optimizer, scaler
        if cached is not None:
            cached.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as error:  # pragma: no cover - defensive
        log(f"WARNING: resource cleanup failed ({error!r}); continuing")


def _accumulate_validation_metrics(
    run_dir: Path, epoch: int, metrics: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Append this epoch's per-condition validation metrics to the running CSV."""
    path = run_dir / "validation_metrics.csv"
    rows = read_csv_rows(path) if path.is_file() else []
    rows.extend({"epoch": epoch, **{key: str(value) for key, value in metric.items()}} for metric in metrics)
    return rows


def link_reused_checkpoint(source: Path, destination: Path) -> str:
    """Expose a frozen checkpoint inside the reuse directory without touching it.

    A hard link keeps the byte content identical (so the recorded SHA256 still
    matches) and never modifies the original file. Copying is the portable
    fallback when the two paths live on different volumes.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"existing {destination} does not match {source}")
        return "already_present"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def register_reused_run(
    n_fft: int,
    seed: int,
    device: torch.device,
    reuse_check: dict[str, Any],
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Write the per-run artefacts of a verified reused n_fft=1024 checkpoint."""
    root = output_root or OUTPUT_ROOT
    run_dir = root / nfft_label(n_fft) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    entry = reuse_check["seeds"][str(seed)]
    source_run = Path(entry["run_directory"]) / "A_original"
    checkpoint_path = source_run / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = int(checkpoint["epoch"])
    del checkpoint

    reference_config = read_json(source_run / "run_config.json")
    shutil.copyfile(source_run / "selection.json", run_dir / "selection.json")
    shutil.copyfile(source_run / "training_history.csv", run_dir / "training_history.csv")
    shutil.copyfile(source_run / "validation_metrics.csv", run_dir / "validation_metrics.csv")
    # run_noise_ab writes the shared cache manifest either beside the arm directory
    # (2026 run) or at the run root (replicate runs); look in both places.
    cache_candidates = (
        source_run / "validation_cache_manifest.json",
        source_run.parent / "validation_cache_manifest.json",
    )
    for candidate in cache_candidates:
        if candidate.is_file():
            write_json(run_dir / "validation_cache_manifest.json", read_json(candidate))
            break
    else:
        raise FileNotFoundError(f"no validation_cache_manifest.json near {source_run}")
    for epoch_index in range(1, EPOCHS + 1):
        source_scores = source_run / f"validation_scores_epoch_{epoch_index:02d}.csv"
        if source_scores.is_file():
            shutil.copyfile(source_scores, run_dir / f"validation_scores_epoch_{epoch_index:02d}.csv")

    selection = read_json(source_run / "selection.json")
    link_mode = link_reused_checkpoint(checkpoint_path, run_dir / "best_model.pt")
    config = {
        "phase": "NFFT4",
        "n_fft": n_fft,
        "hop_length": hop_for(n_fft),
        "hop_ratio": hop_for(n_fft) / n_fft,
        "image_size": IMAGE_SIZE,
        "training_seed": seed,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "train_snrs": TRAIN_SNRS,
        "train_samples": TRAIN_SAMPLES,
        "class_counts": reference_config.get("class_counts"),
        "initial_model_sha256": reference_config.get("initial_model_sha256"),
        "validation_seed": VALIDATION_SEED,
        "validation_conditions": list(CONDITIONS),
        "selection": reference_config.get("selection"),
        "tie_break": reference_config.get("tie_break"),
        "prediction": reference_config.get("prediction"),
        "index_sha256": reference_config.get("index_sha256"),
        "frontend": {
            "n_fft": n_fft,
            "hop_length": hop_for(n_fft),
            "image_size": IMAGE_SIZE,
            "window": "torch.hamming_window(n_fft) periodic",
            "implementation": "model.SpectrogramFrontEnd (frozen)",
        },
        "optimizer": reference_config.get("optimizer"),
        "amp": True,
        "amp_dtype": "float16",
        "torch_version": reference_config.get("torch_version"),
        "gpu": reference_config.get("gpu"),
        "epoch_order_sha256": reference_config.get("epoch_order_sha256"),
        "selected_epoch": selection["multi_noise"],
        "status": "reused_after_equivalence_verification",
        "reused_from": str(checkpoint_path),
        "reused_checkpoint_sha256": entry["checkpoint_sha256"],
        "reused_checkpoint_link": link_mode,
        "reuse_verification": {
            "verdict": "eligible",
            "evidence_file": str((output_root or OUTPUT_ROOT) / "reuse_1024_verification.json"),
            "checks": {
                key: entry[key]
                for key in (
                    "initial_model_sha256_recorded",
                    "initial_model_sha256_derived",
                    "epoch_order_sha256_recorded",
                    "epoch_order_sha256_derived",
                    "validation_cache_match",
                    "checkpoint_sha256",
                    "checkpoint_sha256_match",
                    "checkpoint_epoch",
                    "best_epoch_by_rule",
                    "selection_json",
                    "source_snapshot",
                    "checkpoint_run_config_match",
                )
                if key in entry
            },
        },
        "source_sha256": {
            name: sha256_file(PROJECT_DIR / name)
            for name in ("run_nfft_comparison.py", "dronerfa_data.py", "model.py", "train_resnet18.py")
        },
        "heldout_index_sha256": sha256_file(HELDOUT_INDEX),
        "heldout_samples": HELDOUT_SAMPLES,
        "test_samples": TEST_SAMPLES,
        "completed_at": utc_now(),
    }
    write_json(run_dir / "run_config.json", config)
    log(f"Registered reused run nfft{n_fft} seed {seed} -> {run_dir}")
    return {"run_directory": str(run_dir), "config": config, "reused": True}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def representative_window() -> dict[str, Any]:
    rows = load_index(HELDOUT_INDEX, split="heldout")
    return rows[0]


def read_representative_iq(row: dict[str, Any]) -> np.ndarray:
    import h5py

    with h5py.File(row["file_path"], "r") as handle:
        start = int(row["start"])
        length = int(row["length"])
        i_values = read_vector_slice(handle["RF0_I"], start, length)
        q_values = read_vector_slice(handle["RF0_Q"], start, length)
    return rms_normalize(i_values + 1j * q_values)


def native_spectrogram(
    iq: np.ndarray, n_fft: int, device: torch.device
) -> np.ndarray:
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
    row = representative_window()
    iq = read_representative_iq(row)
    probe = torch.from_numpy(np.stack((iq.real, iq.imag), axis=0)).unsqueeze(0).to(device)
    sample_rate = 100e6
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
        frequency_offsets = (
            (np.arange(n_fft) - (n_fft // 2)) / n_fft * sample_rate / 1e6
        )
        detail[str(n_fft)] = {
            "native_shape_frequency_time": [int(spectra.shape[0]), int(spectra.shape[1])],
            "native_frequency_bins": int(spectra.shape[0]),
            "native_time_frames": int(spectra.shape[1]),
            "model_input_shape": [int(features.shape[0]), int(features.shape[1])],
            "time_step_ms": hop_for(n_fft) / sample_rate * 1e3,
            "frequency_step_khz": sample_rate / n_fft / 1e3,
            "frequency_offset_min_mhz": float(frequency_offsets[0]),
            "frequency_offset_max_mhz": float(frequency_offsets[-1]),
            "model_input_finite": bool(np.isfinite(features).all()),
        }
        del frontend
        torch.cuda.empty_cache()

    figure, axes = plt.subplots(4, 1, figsize=(11.5, 12.4), constrained_layout=True)
    for axis, n_fft in zip(axes, NFFT_GRID):
        spectra = native[n_fft]
        # fftshift puts 鈭抐s/2 first; the grid is n_fft evenly spaced bins of fs/n_fft.
        half_mhz = sample_rate / 1e6 / 2.0
        extent = (
            0.0,
            spectra.shape[1] * hop_for(n_fft) / sample_rate * 1e3,
            -half_mhz,
            half_mhz,
        )
        image = axis.imshow(
            spectra, aspect="auto", origin="lower", extent=extent, cmap="magma"
        )
        axis.set_title(
            f"n_fft={n_fft}, hop={hop_for(n_fft)} - native grid "
            f"{spectra.shape[0]} freq x {spectra.shape[1]} time "
            f"({sample_rate / n_fft / 1e3:.1f} kHz x {hop_for(n_fft) / sample_rate * 1e3:.4f} ms)"
        )
        axis.set_xlabel("time within the window (ms)")
        axis.set_ylabel("offset from RF0 channel centre (MHz)")
        figure.colorbar(image, ax=axis, label="10log10|X|^2 (dB)")
    figure.suptitle(
        f"Frozen frontend, one fixed representative window: {row['filename']} "
        f"start={row['start']} len={row['length']} label={row['label']}",
        fontsize=11,
    )
    figure.savefig(root / "native_stft_examples.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(4, 1, figsize=(10.5, 12.4), constrained_layout=True)
    for axis, n_fft in zip(axes, NFFT_GRID):
        features = model_inputs[n_fft]
        image = axis.imshow(features, aspect="auto", origin="lower", cmap="viridis")
        axis.set_title(
            f"n_fft={n_fft}: actual model input {features.shape[0]}x{features.shape[1]} "
            f"(z-scored log power, bilinear resize of the "
            f"{native[n_fft].shape[0]}x{native[n_fft].shape[1]} native grid)"
        )
        axis.set_xlabel("resized time axis (column, 256 bins)")
        axis.set_ylabel("resized frequency axis (row, 256 bins)")
        figure.colorbar(image, ax=axis, label="z-score")
    figure.suptitle(
        "Actual 1x256x256 tensors fed to ResNet18; the bilinear resize compresses "
        "the native time-frequency grid, so the four panels share no common "
        "physical bin size",
        fontsize=11,
    )
    figure.savefig(root / "model_input_examples.png", dpi=150)
    plt.close(figure)

    # Native-time-frequency content check for the "same window" claim.
    reference_len = None
    same_length = True
    for n_fft in NFFT_GRID:
        covered = (native[n_fft].shape[1] - 1) * hop_for(n_fft) + n_fft
        if reference_len is None:
            reference_len = covered
        same_length &= covered == reference_len
    evidence = {
        "representative_window": {
            "rule": REPRESENTATIVE_WINDOW_RULE,
            "filename": row["filename"],
            "file_path": row["file_path"],
            "start": int(row["start"]),
            "length": int(row["length"]),
            "label": int(row["label"]),
            "target_code": row["target_code"],
            "split": row["split"],
        },
        "stft_convention": "torch.stft center=False normalized=True onesided=False, periodic Hamming, fftshift",
        "n_fft_detail": detail,
        "sampled_samples_per_grid": {
            str(n_fft): int((native[n_fft].shape[1] - 1) * hop_for(n_fft) + n_fft)
            for n_fft in NFFT_GRID
        },
        "all_grids_cover_the_same_window_prefix": bool(same_length),
        "resize_note": (
            "model_input_examples.png shows the real 256x256 tensor; bilinear resizing "
            "compresses the native time-frequency grid, so panel rows/columns are not "
            "the same physical spacing across n_fft"
        ),
        "figures": {
            "native_stft_examples.png": sha256_file(root / "native_stft_examples.png"),
            "model_input_examples.png": sha256_file(root / "model_input_examples.png"),
        },
        "per_window_png_written": False,
    }
    write_json(root / "figure_evidence.json", evidence)
    return evidence


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def experiment_grid(
    output_root: Path | None = None, required: bool = True
) -> list[dict[str, Any]]:
    """Every (n_fft, seed) run of the grid; ``required=False`` allows a partial grid."""
    root = output_root or OUTPUT_ROOT
    grid = []
    for n_fft in NFFT_GRID:
        for seed in TRAINING_SEEDS:
            run_dir = root / nfft_label(n_fft) / f"seed_{seed}"
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
            reused_from = config.get("reused_from")
            grid.append(
                {
                    "n_fft": n_fft,
                    "hop_length": hop_for(n_fft),
                    "training_seed": seed,
                    "run_directory": run_dir,
                    "config": config,
                    "checkpoint": checkpoint,
                    "checkpoint_provenance": str(reused_from) if reused_from else str(checkpoint),
                    "epoch": int(config["selected_epoch"]),
                    "reused": reused_from is not None,
                }
            )
    return grid


def load_models(
    grid: list[dict[str, Any]], device: torch.device
) -> list[dict[str, Any]]:
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


def index_rows_with_position(index_path: Path, split: str) -> list[dict[str, Any]]:
    rows = load_index(index_path, split=split)
    return [{"index_position": position, **row} for position, row in enumerate(rows)]


def noise_pairing_digest(
    index_path: Path, split: str, condition: str, noise_seed: int
) -> dict[str, Any]:
    """Re-derive the added AWGN per window straight from the frozen index.

    Three representative windows (first/middle/last) are probed. For each one the
    frozen ``add_complex_awgn`` is called on a fixed 8-sample delta probe with
    ``default_rng(noise_seed + row_position)``, so the digest pins the exact RNG
    draw of that window without reading any MAT data. The digest is a property of
    (index, condition, seed) alone, therefore identical across the four ``n_fft``
    grids and the three training seeds by construction; the recorded hash lets a
    reviewer confirm that the same code path produced every grid.
    """
    rows = load_index(index_path, split=split)
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
        "rng_rule": "np.random.default_rng(noise_seed + epoch*len(dataset) + row_index), epoch pinned to 0",
        "probe_algorithm": (
            "clean: 8 standard_normal(float32) draws per row_position; noisy: frozen "
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
    models: list[dict[str, Any]],
    device: torch.device,
    cache_dir: Path | None = None,
    write_cache_digests: bool = False,
    output_root: Path | None = None,
    max_windows: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Predict every (model, condition, window) triple of one split.

    The float32 feature cache is built once per (n_fft, condition) and shared by the
    three training seeds of that n_fft, which is exactly how the frozen EVAL6 code
    shared one cache across six models. ``max_windows`` truncates the prefix of the
    split before any feature is built, so a bounded run only reads those windows.
    """
    root = output_root or OUTPUT_ROOT
    rows = index_rows_with_position(index_path, split)
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
            dataset = DroneRFaWindowDataset(
                index_path,
                split=split,
                fixed_snr=snr,
                apply_fixed_snr=snr is not None,
                seed=noise_seed,
            )
            if dataset.epoch != 0:
                raise RuntimeError("dataset epoch must stay 0 so the RNG index is the row")
            features = np.empty((windows, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=FEATURE_DTYPE)
            window_digests = bytearray()
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
                        signals.append(tensor)
                    batch = torch.stack(signals).to(device, non_blocking=True)
                    with torch.inference_mode():
                        batch_features = extract_features(frontend, batch)
                    array = batch_features.detach().to("cpu", torch.float32).numpy()
                    if array.shape != (stop - start, 1, IMAGE_SIZE, IMAGE_SIZE):
                        raise RuntimeError(f"unexpected feature shape {array.shape}")
                    if not np.isfinite(array).all():
                        raise RuntimeError(f"non-finite features in {split}/{condition}")
                    features[start:stop] = array
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
                "feature_sha256": sha256_bytes(features.tobytes()),
                "window_digests_concat_sha256": sha256_bytes(bytes(window_digests)),
                "window_digest_bytes": len(window_digests),
                "seconds": time.perf_counter() - started,
            }
            cache_metadata.append(metadata)
            if cache_dir is not None and write_cache_digests:
                cache_dir.mkdir(parents=True, exist_ok=True)
                (cache_dir / f"window_digests_{split}_{condition}_{nfft_label(n_fft)}.bin").write_bytes(
                    bytes(window_digests)
                )
            del window_digests

            for entry in entries:
                logits_all = np.empty((windows, 2), dtype=np.float32)
                margins_all = np.empty(windows, dtype=np.float32)
                with torch.inference_mode():
                    for start in range(0, windows, BATCH_SIZE):
                        stop = min(start + BATCH_SIZE, windows)
                        batch = torch.from_numpy(
                            np.ascontiguousarray(features[start:stop])
                        ).to(device)
                        with torch.amp.autocast(
                            device_type=device.type, enabled=True, dtype=torch.float16
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
                            "n_fft": n_fft,
                            "hop_length": hop_for(n_fft),
                            "training_seed": entry["training_seed"],
                            "epoch": entry["epoch"],
                            "model_id": f"nfft{n_fft}_seed{entry['training_seed']}",
                            "reused_1024_checkpoint": bool(entry["reused"]),
                            "checkpoint_provenance": entry["checkpoint_provenance"],
                            "split": split,
                            "condition": condition,
                            "index_position": row["index_position"],
                            "file_path": row["file_path"],
                            "filename": row["filename"],
                            "target_code": row["target_code"],
                            "target_name": row["target_name"],
                            "distance_code": row["distance_code"],
                            "segment_code": row["segment_code"],
                            "row_split": row["split"],
                            "label": int(row["label"]),
                            "block_index": int(row["block_index"]),
                            "window_index": int(row["window_index"]),
                            "start": int(row["start"]),
                            "length": int(row["length"]),
                            "source_samples": int(row["source_samples"]),
                            "logit_background": float(logits_all[position, 0]),
                            "logit_drone": float(logits_all[position, 1]),
                            "logit_margin": margin,
                            "drone_score": float(1.0 / (1.0 + np.exp(-np.float64(margin)))),
                            "prediction": int(predictions[position]),
                        }
                    )
                log(
                    f"    nfft{n_fft} seed {entry['training_seed']} {split}/{condition}: "
                    f"drone {int(predictions.sum())}/{windows}"
                )
            del features
            torch.cuda.empty_cache()
    del models
    return prediction_rows, cache_metadata, pairing


# ---------------------------------------------------------------------------
# summarisation
# ---------------------------------------------------------------------------
def spread(values: Sequence[float]) -> float:
    """Sample standard deviation, or 0 for a single value (avoids a numpy warning)."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def add_metric_row(
    base: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    primary = file_equal_metrics_any(records)
    secondary = pooled_metrics_any(records)
    return {
        **base,
        "windows": len(records),
        "files_per_category": ";".join(
            f"{key}:{value}" for key, value in sorted(primary["files_per_category"].items())
        ),
        "file_equal_balanced_accuracy": primary["file_equal_balanced_accuracy"],
        "file_equal_drone_recall": primary["drone_file_equal_recall"],
        "file_equal_background_specificity": primary["background_file_equal_specificity"],
        "file_equal_background_fpr": primary["background_file_equal_fpr"],
        "file_equal_phantom_recall": primary["phantom_file_equal_recall"],
        "file_equal_air2s_recall": primary["air2s_file_equal_recall"],
        "accuracy": secondary["accuracy"],
        "balanced_accuracy": secondary["balanced_accuracy"],
        "precision": secondary["precision"],
        "drone_recall": secondary["drone_recall"],
        "background_specificity": secondary["background_specificity"],
        "f1": secondary["f1"],
        "tn": secondary["tn"],
        "fp": secondary["fp"],
        "fn": secondary["fn"],
        "tp": secondary["tp"],
    }


def summarize_predictions(
    predictions: list[dict[str, Any]],
    split: str,
    output_root: Path | None = None,
) -> dict[str, Path]:
    """Build every metric table from the per-window prediction rows."""
    root = output_root or OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    scoped = [row for row in predictions if row["split"] == split]
    if not scoped:
        raise ValueError(f"no predictions for split {split}")
    keys = sorted(
        {(row["n_fft"], row["training_seed"], row["condition"]) for row in scoped},
        key=lambda item: (item[0], item[1], CONDITIONS.index(item[2])),
    )
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in scoped:
        grouped.setdefault(
            (row["n_fft"], row["training_seed"], row["condition"]), []
        ).append(row)

    per_file_rows: list[dict[str, Any]] = []
    model_condition_rows: list[dict[str, Any]] = []
    composites: dict[tuple[int, int], dict[str, Any]] = {}
    for n_fft, seed, condition in keys:
        rows = grouped[(n_fft, seed, condition)]
        epoch = rows[0]["epoch"]
        base = {
            "split": split,
            "n_fft": n_fft,
            "hop_length": hop_for(n_fft),
            "training_seed": seed,
            "epoch": epoch,
            "condition": condition,
            "model_id": f"nfft{n_fft}_seed{seed}",
        }
        model_condition_rows.append(add_metric_row(base, rows))
        files = sorted({row["filename"] for row in rows})
        for filename in files:
            file_rows = [row for row in rows if row["filename"] == filename]
            label = int(file_rows[0]["label"])
            hits = sum(int(row["prediction"]) for row in file_rows)
            if label == 1:
                counts = {"tp": hits, "fn": len(file_rows) - hits, "tn": 0, "fp": 0}
                detection = hits / len(file_rows)
                false_positive = ""
            else:
                counts = {"tp": 0, "fn": 0, "tn": len(file_rows) - hits, "fp": hits}
                detection = ""
                false_positive = hits / len(file_rows)
            per_file_rows.append(
                {
                    **base,
                    "filename": filename,
                    "target_code": file_rows[0]["target_code"],
                    "target_name": file_rows[0].get("target_name", ""),
                    "segment_code": file_rows[0].get("segment_code", ""),
                    "role": "background" if label == 0 else "drone",
                    "windows": len(file_rows),
                    "file_equal_drone_detection_rate": detection,
                    "file_equal_background_fpr": false_positive,
                    **counts,
                }
            )

    # Cross-seed summary per (n_fft, condition) and paired differences against 1024.
    # The grid is read back from the predictions so a bounded run summarises exactly
    # what it produced instead of assuming the full three-seed grid.
    summary_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    metric_by_key = {
        (row["n_fft"], row["training_seed"], row["condition"]): row
        for row in model_condition_rows
    }
    present_nfft = sorted({n_fft for n_fft, _, _ in metric_by_key})
    present_seeds = sorted({seed for _, seed, _ in metric_by_key})
    for n_fft in present_nfft:
        for seed in present_seeds:
            if (n_fft, seed, CONDITIONS[0]) not in metric_by_key:
                continue
            rows = [metric_by_key[(n_fft, seed, condition)] for condition in CONDITIONS]
            composites[(n_fft, seed)] = {
                "file_equal_balanced_accuracy": float(
                    np.mean([row["file_equal_balanced_accuracy"] for row in rows])
                ),
                "balanced_accuracy": float(
                    np.mean([row["balanced_accuracy"] for row in rows])
                ),
                "file_equal_drone_recall": float(
                    np.mean([row["file_equal_drone_recall"] for row in rows])
                ),
                "file_equal_background_fpr": float(
                    np.mean([row["file_equal_background_fpr"] for row in rows])
                ),
            }
    for n_fft in present_nfft:
        for condition in CONDITIONS:
            seeds = [
                seed
                for seed in present_seeds
                if (n_fft, seed, condition) in metric_by_key
            ]
            values = [
                metric_by_key[(n_fft, seed, condition)]["file_equal_balanced_accuracy"]
                for seed in seeds
            ]
            pooled = [
                metric_by_key[(n_fft, seed, condition)]["balanced_accuracy"]
                for seed in seeds
            ]
            summary_rows.append(
                {
                    "n_fft": n_fft,
                    "hop_length": hop_for(n_fft),
                    "condition": condition,
                    "seeds": len(seeds),
                    "file_equal_balacc_mean": float(np.mean(values)),
                    "file_equal_balacc_std": spread(values),
                    "file_equal_balacc_min": float(np.min(values)),
                    "file_equal_balacc_max": float(np.max(values)),
                    "pooled_balacc_mean": float(np.mean(pooled)),
                    "pooled_balacc_std": spread(pooled),
                    "per_seed_file_equal_balacc": json.dumps(
                        {str(seed): value for seed, value in zip(seeds, values)}
                    ),
                }
            )
        composite_seeds = sorted(
            seed for seed in present_seeds if (n_fft, seed) in composites
        )
        values = [
            composites[(n_fft, seed)]["file_equal_balanced_accuracy"] for seed in composite_seeds
        ]
        pooled = [composites[(n_fft, seed)]["balanced_accuracy"] for seed in composite_seeds]
        summary_rows.append(
            {
                "n_fft": n_fft,
                "hop_length": hop_for(n_fft),
                "condition": "four_condition_mean",
                "seeds": len(composite_seeds),
                "file_equal_balacc_mean": float(np.mean(values)),
                "file_equal_balacc_std": spread(values),
                "file_equal_balacc_min": float(np.min(values)),
                "file_equal_balacc_max": float(np.max(values)),
                "pooled_balacc_mean": float(np.mean(pooled)),
                "pooled_balacc_std": spread(pooled),
                "per_seed_file_equal_balacc": json.dumps(
                    {str(seed): value for seed, value in zip(composite_seeds, values)}
                ),
            }
        )
    for n_fft in present_nfft:
        if n_fft == BASELINE_NFFT or (BASELINE_NFFT not in present_nfft):
            continue
        for condition in CONDITIONS + ("four_condition_mean",):
            differences = []
            pooled_differences = []
            for seed in present_seeds:
                if condition == "four_condition_mean":
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
                    "mean_difference": float(np.mean(deltas)),
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
                    "mean_pooled_difference": float(np.mean(pooled_differences)),
                }
            )
    return {
        "model_condition_metrics": model_condition_rows,
        "per_file_metrics": per_file_rows,
        "nfft_summary": summary_rows,
        "paired_nfft_comparison": paired_rows,
    }


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
            "figures",
            "evaluate",
            "summarize",
            "manifest",
            "verify",
            "all",
        ),
    )
    parser.add_argument("--n-fft", type=int, choices=NFFT_GRID)
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--reuse-1024",
        action="store_true",
        help="Reuse a 2026-09-17 n_fft=1024 checkpoint, but only after every "
        "reuse prerequisite is re-derived and matched.",
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
        help="Smoke only: truncate the evaluated split and skip full-set assertions.",
    )
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--smoke", action="store_true", help="Mark outputs as a smoke run.")
    parser.add_argument(
        "--complete-interrupted-runs",
        action="store_true",
        help="Finish only the bookkeeping of a run whose script died after its last epoch; "
        "training is never repeated or altered.",
    )
    return parser.parse_args(argv)


def conditions_from(args: argparse.Namespace) -> list[str]:
    conditions = [token.strip() for token in args.conditions.split(",") if token.strip()]
    for condition in conditions:
        if condition not in CONDITION_VALUES:
            raise SystemExit(f"unknown condition {condition}")
    return conditions


def command_verify_inputs(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    integrity = verify_frozen_inputs()
    write_json(root / "frozen_inputs_verification.json", integrity)
    if integrity["errors"]:
        for error in integrity["errors"]:
            log(f"REJECTED: {error}")
        return 2
    log("Frozen inputs verified")
    device = resolve_device(args.device)
    reuse = verify_reuse_1024(device)
    log(f"Reuse eligibility: {json.dumps({k: v.get('eligible_for_reuse') for k, v in reuse['seeds'].items()})}")
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
    reuse = None
    for n_fft, seed in targets:
        run_dir = root / nfft_label(n_fft) / f"seed_{seed}"
        config_path = run_dir / "run_config.json"
        if config_path.is_file():
            status = read_json(config_path).get("status")
            if status in ("completed", "reused_after_equivalence_verification"):
                log(f"skip completed run {run_dir} (status={status})")
                continue
            if args.complete_interrupted_runs:
                finalize_interrupted_run(n_fft, seed, output_root=root)
                continue
            raise RuntimeError(
                f"{run_dir} holds an unfinished run (status={status!r}); "
                f"re-run with --complete-interrupted-runs to finish its bookkeeping "
                f"or remove the directory to retrain it"
            )
        if run_dir.exists():
            raise RuntimeError(
                f"{run_dir} exists without run_config.json; remove it before retraining"
            )
        if n_fft == BASELINE_NFFT and args.reuse_1024:
            if reuse is None:
                reuse = verify_reuse_1024(device)
            if reuse["seeds"].get(str(seed), {}).get("eligible_for_reuse"):
                register_reused_run(n_fft, seed, device, reuse, output_root=root)
                continue
            log(f"Reuse rejected for seed {seed}; retraining instead")
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
    log(json.dumps(evidence["n_fft_detail"], ensure_ascii=False, indent=2))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    conditions = conditions_from(args)
    grid = experiment_grid(root)
    if args.max_windows is not None:
        log(f"Smoke evaluation on {args.max_windows} windows per split")
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
        ["clean"],
        TRAINING_SEEDS[0] + TEST_NOISE_SEED_OFFSET,
        models,
        device,
        cache_dir=root / "feature_digests",
        write_cache_digests=args.max_windows is None,
        output_root=root,
        max_windows=args.max_windows,
    )
    log(f"Old-test clean predictions: {len(test_rows)} rows")
    write_csv(root / "test_clean_window_predictions.csv", test_rows)

    write_json(
        root / "prediction_cache_manifest.json",
        {
            "heldout": heldout_cache,
            "test_clean": test_cache,
            "noise_pairing": {"heldout": heldout_pairing, "test_clean": test_pairing},
            "cache_location": "in-memory float32 per (split, condition, n_fft); freed before the next condition",
            "disk_feature_cache_written": False,
            "window_digest_files": None if args.max_windows is not None else "feature_digests/",
            "max_windows": args.max_windows,
            "truncated_debug_run": args.max_windows is not None,
        },
    )
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    heldout = read_csv_rows(root / "heldout_window_predictions.csv")
    for row in heldout:
        row["n_fft"] = int(row["n_fft"])
        row["training_seed"] = int(row["training_seed"])
        row["epoch"] = int(row["epoch"])
        row["index_position"] = int(row["index_position"])
        row["label"] = int(row["label"])
        row["prediction"] = int(row["prediction"])
        row["split"] = "heldout"
    tables = summarize_predictions(heldout, "heldout", output_root=root)
    write_csv(root / "model_condition_metrics.csv", tables["model_condition_metrics"])
    write_csv(root / "per_file_metrics.csv", tables["per_file_metrics"])
    write_csv(root / "nfft_summary.csv", tables["nfft_summary"])
    write_csv(root / "paired_nfft_comparison.csv", tables["paired_nfft_comparison"])
    log(
        f"Wrote model_condition_metrics.csv ({len(tables['model_condition_metrics'])}), "
        f"per_file_metrics.csv ({len(tables['per_file_metrics'])}), "
        f"nfft_summary.csv ({len(tables['nfft_summary'])}), "
        f"paired_nfft_comparison.csv ({len(tables['paired_nfft_comparison'])})"
    )
    return 0


def write_experiment_manifest(
    root: Path, extra_commands: dict[str, Any] | None = None
) -> dict[str, Any]:
    """One auditable entry point for the whole NFFT4 run."""
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
                "reused": entry["reused"],
                "reused_from": config.get("reused_from"),
                "reused_checkpoint_sha256": config.get("reused_checkpoint_sha256"),
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
        "heldout_window_predictions.csv",
        "test_clean_window_predictions.csv",
        "model_condition_metrics.csv",
        "per_file_metrics.csv",
        "nfft_summary.csv",
        "paired_nfft_comparison.csv",
        "native_stft_examples.png",
        "model_input_examples.png",
        "figure_evidence.json",
        "prediction_cache_manifest.json",
        "reuse_1024_verification.json",
        "frozen_inputs_verification.json",
        "report.md",
        "verification.json",
    ):
        path = root / name
        if path.is_file():
            published[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "task": "NFFT4: first-stage STFT-size ablation",
        "created_at": utc_now(),
        "output_root": str(root),
        "design": {
            "n_fft_grid": list(NFFT_GRID),
            "hop_rule": "hop_length = n_fft // 2 (exactly 50%)",
            "image_size": IMAGE_SIZE,
            "training_seeds": list(TRAINING_SEEDS),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": {"type": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
            "amp": True,
            "amp_dtype": "float16",
            "train_snrs": TRAIN_SNRS,
            "validation_conditions": list(CONDITIONS),
            "validation_seed": VALIDATION_SEED,
            "selection_rule": VAL_SELECTION_RULE,
            "tie_break": TIE_BREAK,
            "prediction_rule": PREDICTION_RULE,
            "primary_aggregation": "per-source-file rates, four files per category equally weighted",
            "secondary_aggregation": "pooled over all held-out windows, counts retained",
            "heldout_noise_seed": HELDOUT_NOISE_SEED,
            "heldout_rng_rule": "np.random.default_rng(noise_seed + epoch*len(dataset) + row_index), epoch pinned to 0",
        },
        "fairness": {
            "same_initial_weights_within_seed": (
                "initial_model_sha256 is recorded per run and must be identical across the "
                "four n_fft values of one training seed; the paired unit test asserts it"
            ),
            "same_epoch_orders_within_seed": (
                "epoch_order_sha256 is derived from torch.randperm(920, generator=seed+epoch) "
                "and checked against the 2026-09-17 reference runs"
            ),
            "same_validation_inputs": (
                "one float32 validation cache per n_fft, reused by that grid's three seeds; "
                "the n_fft=1024 cache was re-derived and matched byte for byte"
            ),
            "test_and_heldout_not_used_for_selection": True,
        },
        "reuse_1024": {
            "evidence": str(root / "reuse_1024_verification.json"),
            "rule": (
                "the three 2026-09-17 A_original checkpoints are reused only where every "
                "prerequisite was re-derived and matched; a mismatch forces a retrain"
            ),
        },
        "feature_cache": {
            "location": "in-memory float32 CPU array per (split, condition, n_fft)",
            "disk_feature_cache_written": False,
            "window_digest_files": "feature_digests/window_digests_<split>_<condition>_<nfft>.bin",
            "digest_rule": "one SHA256 per window over the float32 1x256x256 tensor, concatenated",
            "manifest": str(root / "prediction_cache_manifest.json"),
            "per_window_png_written": False,
        },
        "runs": runs,
        "runs_completed": sum(1 for entry in runs if entry["status"] in ("completed", "reused_after_equivalence_verification")),
        "runs_expected": len(NFFT_GRID) * len(TRAINING_SEEDS),
        "published_files": published,
        "source_sha256": {
            name: sha256_file(PROJECT_DIR / name)
            for name in (
                "run_nfft_comparison.py",
                "test_nfft_comparison.py",
                "verify_nfft_comparison.py",
                "dronerfa_data.py",
                "model.py",
                "train_resnet18.py",
                "run_noise_ab.py",
                "evaluate_heldout_models.py",
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


def command_all(args: argparse.Namespace) -> int:
    for function in (
        command_verify_inputs,
        command_frontends,
        command_train,
        command_figures,
        command_evaluate,
        command_summarize,
        command_manifest,
    ):
        code = function(args)
        if code:
            return code
    return 0


COMMANDS = {
    "verify-inputs": command_verify_inputs,
    "frontends": command_frontends,
    "train": command_train,
    "figures": command_figures,
    "evaluate": command_evaluate,
    "summarize": command_summarize,
    "manifest": command_manifest,
    "all": command_all,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "verify":
        raise SystemExit(
            "run `python test_nfft_comparison.py --verify-outputs` for the independent "
            "recomputation of every reported metric"
        )
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
