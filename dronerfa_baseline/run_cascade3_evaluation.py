"""CASCADE3: direct stage-1 detection to stage-2 machine classification cascade.

The already accepted stage-1 detector (Background vs Drone) and the accepted
stage-2 classifier (Phantom 4 Pro vs Air 2S) are chained **without retraining and
without any threshold tuning**.  Three seed-matched cascades are evaluated:

    seed 2026 : stage-1 nfft512 seed2026  -> stage-2 nfft512 seed2026
    seed 2027 : stage-1 nfft512 seed2027  -> stage-2 nfft512 seed2027
    seed 2028 : stage-1 nfft512 seed2028  -> stage-2 nfft512 seed2028

Frozen configuration: ``work/cascade3-config-20260919.json`` (only evidence
source).  Per dataset/condition/window exactly one noisy signal and exactly one
``1x256x256`` float32 feature tensor are produced; the noise RNG position is the
**full split row position** (background rows included) and the very same tensor is
handed to stage 1 and, if stage 1 passes, to stage 2.

Stepwise commands (``--output-dir`` selects the root, default below)::

    python run_cascade3_evaluation.py verify-inputs [--stage before|after]
    python run_cascade3_evaluation.py evaluate [--max-windows N] [--datasets heldout,old_test]
    python run_cascade3_evaluation.py summarize
    python run_cascade3_evaluation.py figures
    python run_cascade3_evaluation.py report
    python run_cascade3_evaluation.py manifest
    python run_cascade3_evaluation.py all

``test_cascade3_evaluation.py`` runs the bounded unit/smoke checks and
``verify_cascade3_evaluation.py`` independently recomputes the published tables
from the raw prediction CSV and re-derives stage-1/stage-2 logits from raw IQ for
a bounded target set.

No model is trained here, the held-out split is never used for any choice, and no
network parameter or decision threshold is modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "outputs" / ".matplotlib"))

import numpy as np
import torch

from dronerfa_data import DroneRFaWindowDataset, load_index
from model import ResNet18, SpectrogramFrontEnd
from train_resnet18 import extract_features

# ---------------------------------------------------------------------------
# frozen protocol constants (all cross-checked against the frozen config)
# ---------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / "work" / "cascade3-config-20260919.json"
OUTPUT_ROOT = PROJECT_DIR / "outputs" / "cascade3_evaluation_20260919"
PRESERVED_INPUTS = PROJECT_ROOT / "work" / "codex-preserved-inputs-20260917.json"
STAGE1_PUBLISHED_ROOT = PROJECT_DIR / "outputs" / "stage1_method_comparison_20260919" / "nfft"

N_FFT = 512
HOP_LENGTH = 256
IMAGE_SIZE = 256
BATCH_SIZE = 4
TRAINING_SEEDS: tuple[int, ...] = (2026, 2027, 2028)
FEATURE_DTYPE = np.float32
INPUT_PROBE_SAMPLES = 4096
NOISE_EPOCH = 0
DATASET_NAMES: tuple[str, ...] = ("heldout", "old_test")

# config keys ``conditions_db`` -> canonical condition names used everywhere below
CONDITION_VALUES: dict[str, float | None] = {
    "clean": None,
    "-10": -10.0,
    "-12": -12.0,
    "-15": -15.0,
}
CONDITIONS: tuple[str, ...] = ("clean", "-10", "-12", "-15")
TEST_CONDITIONS: tuple[str, ...] = ("clean",)

CLASS_NAMES: dict[int, str] = {
    0: "Background",
    1: "DJI Phantom 4 Pro",
    2: "DJI Air 2S",
}
CLASS_IDS: tuple[int, ...] = (0, 1, 2)
TARGET_CODE_TO_CLASS: dict[str, int] = {"0000": 0, "0010": 1, "0101": 2}
DRONE_CLASSES: tuple[int, ...] = (1, 2)
BACKGROUND_CLASS = 0

RNG_RULE = (
    "np.random.default_rng(noise_seed + epoch*len(full_split) + full_split_row_position), "
    "epoch pinned to 0; background rows keep their position and therefore their draw"
)
FEATURE_BATCH_RULE = (
    "full split rows in frozen index order, contiguous chunks of batch_size=4 "
    "(byte-identical to the frozen stage-1 nfft512 feature protocol)"
)
STAGE1_RULE = "run stage 2 iff float32(logit_drone) - float32(logit_background) > 0"
STAGE2_RULE = (
    "Air 2S (class 2) iff float32(logit_air2s) - float32(logit_phantom) > 0, "
    "otherwise Phantom 4 Pro (class 1)"
)
STAGE2_BATCH_RULE = (
    "executed rows only, ascending dataset_index_position, contiguous chunks of "
    "batch_size=4; the chunk start is recorded per row as stage2_batch_id"
)
STAGE2_SKIPPED_RULE = (
    "a row whose stage-1 margin is <= 0 is never passed to stage 2: final class "
    "Background (0), stage2_executed=false and all stage-2 fields left empty"
)
PARITY_RULE = (
    "the cascade stage-1 logits must reproduce the published stage-1 nfft512 "
    "predictions of the same split/condition/seed exactly, and the per-window "
    "feature digests must reproduce the published stage-1 digest binaries byte "
    "for byte; stage 2 is NOT compared with the published oracle CSV because its "
    "RNG row positions differ"
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "dataset",
    "condition",
    "training_seed",
    "stage1_epoch",
    "stage2_epoch",
    "dataset_index_position",
    "file_path",
    "filename",
    "target_code",
    "target_name",
    "distance_code",
    "segment_code",
    "row_split",
    "stage1_label",
    "true_class",
    "true_class_name",
    "block_index",
    "window_index",
    "start",
    "length",
    "source_samples",
    "stage1_logit_background",
    "stage1_logit_drone",
    "stage1_margin",
    "stage1_prediction",
    "stage1_decision",
    "stage2_executed",
    "stage2_batch_id",
    "stage2_logit_phantom",
    "stage2_logit_air2s",
    "stage2_margin",
    "final_prediction",
    "final_class_name",
    "stage1_feature_digest",
    "stage2_feature_digest",
)

METRIC_COLUMNS: tuple[str, ...] = (
    "dataset",
    "condition",
    "training_seed",
    "stage1_epoch",
    "stage2_epoch",
    "windows",
    "background_windows",
    "phantom_windows",
    "air2s_windows",
    "file_equal_accuracy",
    "file_equal_classes_used",
    "file_equal_background_recall",
    "file_equal_phantom_recall",
    "file_equal_air2s_recall",
    "files_background",
    "files_phantom",
    "files_air2s",
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
    "stage1_background_false_alarms",
    "stage1_background_false_alarm_rate",
    "stage1_background_false_alarm_rate_file_equal",
    "stage1_drone_misses",
    "stage1_drone_miss_rate",
    "stage1_drone_miss_rate_file_equal",
    "stage1_drones_passed",
    "stage2_conditional_windows",
    "stage2_conditional_correct",
    "stage2_conditional_accuracy",
    "error_background_to_phantom",
    "error_background_to_air2s",
    "error_phantom_to_background",
    "error_air2s_to_background",
    "error_phantom_to_air2s",
    "error_air2s_to_phantom",
    "error_windows_total",
)

CROSS_SEED_METRICS: tuple[str, ...] = (
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

ERROR_TRANSITIONS: tuple[tuple[int, int, str], ...] = (
    (0, 1, "Background->Phantom"),
    (0, 2, "Background->Air2S"),
    (1, 0, "Phantom->Background"),
    (2, 0, "Air2S->Background"),
    (1, 2, "Phantom->Air2S"),
    (2, 1, "Air2S->Phantom"),
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


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = list(columns) if columns is not None else list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    """``None`` becomes an empty field (used by the skipped stage-2 columns)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def mean_or_nan(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def sample_std(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def condition_name(value: Any) -> str:
    """Canonical condition name: ``clean``, ``-10``, ``-12``, ``-15``.

    Older artefacts spell the noisy conditions ``-10dB``; the suffix is accepted
    here so the published stage-1 CSVs can be compared with the cascade.
    """
    if value is None:
        return "clean"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "clean"
        lowered = text.lower()
        if lowered in {"clean", "none"}:
            return "clean"
        if lowered.endswith("db"):
            text = text[:-2].strip()
        return f"{float(text):g}"
    return f"{float(value):g}"


def feature_digest(array: np.ndarray) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).digest()


def input_probe_payload(tensor: torch.Tensor, samples: int = INPUT_PROBE_SAMPLES) -> bytes:
    array = np.ascontiguousarray(tensor[:, :samples].detach().cpu().numpy())
    return array.tobytes()


# ---------------------------------------------------------------------------
# frozen configuration
# ---------------------------------------------------------------------------
def load_config(path: Path | None = None) -> dict[str, Any]:
    config = read_json(path or CONFIG_PATH)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("unexpected cascade3 config schema_version")
    if str(config.get("task")) != "CASCADE3":
        raise ValueError("config task is not CASCADE3")
    frontend = config["frontend"]
    if int(frontend["n_fft"]) != N_FFT or int(frontend["hop_length"]) != HOP_LENGTH:
        raise ValueError("frozen frontend must stay n_fft=512 / hop_length=256")
    if int(frontend["image_size"]) != IMAGE_SIZE:
        raise ValueError("frozen frontend image_size must stay 256")
    if str(frontend["rf_channel"]) != "RF0":
        raise ValueError("frozen frontend must stay on RF0")
    conditions = tuple(condition_name(item) for item in config["noise"]["conditions_db"])
    if conditions != CONDITIONS:
        raise ValueError(f"unexpected condition list {conditions}")
    seeds = tuple(int(item["seed"]) for item in config["seed_pairing"])
    if seeds != TRAINING_SEEDS:
        raise ValueError(f"unexpected seed pairing {seeds}")
    if int(config["noise"]["seed"]) != 22026 or int(config["noise"]["epoch"]) != NOISE_EPOCH:
        raise ValueError("frozen noise seed/epoch must stay 22026/0")
    if bool(frontend.get("same_tensor_for_both_stages")) is not True:
        raise ValueError("the config must keep one shared tensor for both stages")
    labels = {int(key): str(value) for key, value in config["decision"]["labels"].items()}
    if labels != CLASS_NAMES:
        raise ValueError(f"unexpected label space {labels}")
    return config


def dataset_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    datasets = config["datasets"]
    expected = config["expected_prediction_rows"]
    noise_seed = int(config["noise"]["seed"])
    return {
        "heldout": {
            "dataset": "heldout",
            "split": "heldout",
            "index_path": PROJECT_ROOT / str(datasets["heldout_index"]),
            "index_sha256": str(datasets["heldout_index_sha256"]),
            "conditions": CONDITIONS,
            "windows": int(datasets["heldout_rows_per_condition_seed"]),
            "noise_seed": noise_seed,
            "expected_rows": int(expected["heldout"]),
            "expected_class_windows": {
                TARGET_CODE_TO_CLASS["0000"]: int(datasets["heldout_class_rows"]["Background"]),
                TARGET_CODE_TO_CLASS["0010"]: int(
                    datasets["heldout_class_rows"]["DJI Phantom 4 Pro"]
                ),
                TARGET_CODE_TO_CLASS["0101"]: int(datasets["heldout_class_rows"]["DJI Air 2S"]),
            },
        },
        "old_test": {
            "dataset": "old_test",
            "split": "test",
            "index_path": PROJECT_ROOT / str(datasets["old_test_index"]),
            "index_sha256": str(datasets["old_index_sha256"]),
            "conditions": TEST_CONDITIONS,
            "windows": int(datasets["old_test_rows_per_seed"]),
            "noise_seed": noise_seed,
            "expected_rows": int(expected["old_test_clean"]),
            "expected_class_windows": None,
        },
    }


def checkpoint_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for pairing in config["seed_pairing"]:
        seed = int(pairing["seed"])
        entries.append(
            {
                "training_seed": seed,
                "stage": 1,
                "checkpoint": PROJECT_ROOT / str(pairing["stage1_checkpoint"]),
                "sha256": str(pairing["stage1_sha256"]),
                "bytes": int(pairing["stage1_bytes"]),
                "epoch": int(pairing["stage1_epoch"]),
            }
        )
        entries.append(
            {
                "training_seed": seed,
                "stage": 2,
                "checkpoint": PROJECT_ROOT / str(pairing["stage2_checkpoint"]),
                "sha256": str(pairing["stage2_sha256"]),
                "bytes": int(pairing["stage2_bytes"]),
                "epoch": int(pairing["stage2_epoch"]),
            }
        )
    return entries


def seed_epochs(config: dict[str, Any]) -> dict[int, dict[str, int]]:
    return {
        int(pairing["seed"]): {
            "stage1_epoch": int(pairing["stage1_epoch"]),
            "stage2_epoch": int(pairing["stage2_epoch"]),
        }
        for pairing in config["seed_pairing"]
    }


# ---------------------------------------------------------------------------
# dataset rows and window reading
# ---------------------------------------------------------------------------
def dataset_rows(index_path: Path, split: str) -> list[dict[str, Any]]:
    """Full split rows in frozen index order, with the three-class truth attached.

    ``dataset_index_position`` is the position inside the **full** split (background
    rows included).  It is the RNG position of the frozen AWGN rule and the primary
    key component of every prediction row.
    """
    rows: list[dict[str, Any]] = []
    for position, row in enumerate(load_index(index_path, split=split)):
        code = str(row["target_code"])
        if code not in TARGET_CODE_TO_CLASS:
            raise ValueError(f"unexpected target_code {code} in {index_path}")
        true_class = TARGET_CODE_TO_CLASS[code]
        stage1_label = int(row["label"])
        expected_label = 0 if true_class == BACKGROUND_CLASS else 1
        if stage1_label != expected_label:
            raise ValueError(
                f"index row {row['filename']} target {code} carries stage-1 label "
                f"{stage1_label}; expected {expected_label}"
            )
        rows.append(
            {
                **row,
                "dataset_index_position": position,
                "stage1_label": stage1_label,
                "true_class": true_class,
                "true_class_name": CLASS_NAMES[true_class],
            }
        )
    return rows


def class_counts(rows: Iterable[dict[str, Any]]) -> dict[int, int]:
    counts = {label: 0 for label in CLASS_IDS}
    for row in rows:
        counts[int(row["true_class"])] += 1
    return counts


def build_window_dataset(
    index_path: Path, split: str, condition: str, noise_seed: int
) -> DroneRFaWindowDataset:
    """Frozen lazy reader over the whole split (background rows included).

    ``DroneRFaWindowDataset`` is used unchanged, so the AWGN rule stays exactly
    ``default_rng(noise_seed + epoch * len(split) + row_position)`` with the row
    position of the full split and ``epoch`` pinned to 0.
    """
    snr = CONDITION_VALUES[condition]
    dataset = DroneRFaWindowDataset(
        index_path,
        split=split,
        fixed_snr=snr,
        apply_fixed_snr=snr is not None,
        seed=noise_seed,
    )
    if dataset.epoch != NOISE_EPOCH:
        raise RuntimeError("dataset epoch must stay 0 so the RNG index is the full-split row")
    return dataset


def read_window(dataset: DroneRFaWindowDataset, position: int) -> tuple[torch.Tensor, int]:
    """Read one window through the frozen reader (labels travel with the tensor)."""
    tensor, label = dataset[position]
    return tensor, int(label)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def load_cascade_models(
    config: dict[str, Any], device: torch.device, seeds: Sequence[int] = TRAINING_SEEDS
) -> dict[int, dict[str, Any]]:
    """Load the seed-matched stage-1/stage-2 checkpoint pairs (no training)."""
    epochs = seed_epochs(config)
    loaded: dict[int, dict[str, Any]] = {}
    frontend_states: dict[int, dict[str, Any]] = {}
    for entry in checkpoint_entries(config):
        seed = int(entry["training_seed"])
        if seed not in seeds:
            continue
        checkpoint_path: Path = entry["checkpoint"]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing checkpoint {checkpoint_path}")
        actual = sha256_file(checkpoint_path, label=f"{'stage1' if entry['stage'] == 1 else 'stage2'} seed {seed}")
        if actual != entry["sha256"]:
            raise RuntimeError(f"checkpoint SHA256 mismatch for {checkpoint_path}: {actual}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint["epoch"]) != int(entry["epoch"]):
            raise RuntimeError(
                f"checkpoint epoch {checkpoint['epoch']} != config epoch {entry['epoch']} "
                f"for {checkpoint_path}"
            )
        frontend = SpectrogramFrontEnd(
            n_fft=N_FFT, hop_length=HOP_LENGTH, image_size=IMAGE_SIZE
        ).to(device)
        frontend.load_state_dict(checkpoint["frontend_state"])
        frontend.eval()
        network = ResNet18(input_channels=1, num_classes=2).to(device)
        network.load_state_dict(checkpoint["model_state"])
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        record = loaded.setdefault(
            seed,
            {
                "training_seed": seed,
                "stage1_epoch": epochs[seed]["stage1_epoch"],
                "stage2_epoch": epochs[seed]["stage2_epoch"],
                "frontend": frontend,
            },
        )
        key = "stage1_network" if entry["stage"] == 1 else "stage2_network"
        record[key] = network
        record[f"{key}_checkpoint"] = str(checkpoint_path)
        record[f"{key}_sha256"] = actual
        record[f"{key}_epoch"] = int(entry["epoch"])
        frontend_states.setdefault(seed, {})[entry["stage"]] = checkpoint["frontend_state"]
        del checkpoint
    missing = [seed for seed in seeds if seed not in loaded or "stage2_network" not in loaded[seed]]
    if missing:
        raise RuntimeError(f"incomplete seed pairing for {missing}")
    states = [loaded[seed]["frontend"].state_dict() for seed in seeds]
    reference = states[0]
    if any(set(state) != set(reference) for state in states):
        raise RuntimeError("stage-1/stage-2 frontends disagree on their parameter keys")
    for seed in seeds:
        record = loaded[seed]
        record["frontend_state_keys"] = sorted(record["frontend"].state_dict().keys())
    return loaded


def network_logits(network: ResNet18, features: np.ndarray, device: torch.device) -> np.ndarray:
    """Batched float16-AMP forward pass over a float32 feature array (batch 4)."""
    windows = int(features.shape[0])
    logits_all = np.empty((windows, 2), dtype=np.float32)
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for start in range(0, windows, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, windows)
            batch = torch.from_numpy(np.ascontiguousarray(features[start:stop])).to(device)
            with torch.amp.autocast(
                device_type=device.type, enabled=amp_enabled, dtype=torch.float16
            ):
                logits = network(batch)
            logits_all[start:stop] = logits.float().cpu().numpy()
    if not np.isfinite(logits_all).all():
        raise RuntimeError("non-finite logits")
    return logits_all


def stage2_batches(selection: np.ndarray, batch_size: int = BATCH_SIZE) -> list[list[int]]:
    """Ascending positions that passed stage 1, chunked by ``batch_size``."""
    positions = [int(index) for index in np.nonzero(np.asarray(selection, dtype=bool))[0]]
    return [positions[start : start + batch_size] for start in range(0, len(positions), batch_size)]


# ---------------------------------------------------------------------------
# feature building
# ---------------------------------------------------------------------------
def build_features(
    dataset: DroneRFaWindowDataset,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
    frontend: SpectrogramFrontEnd,
) -> dict[str, Any]:
    """One noisy signal and one float32 feature tensor per row of the full split.

    The loop mirrors the frozen stage-1 nfft512 protocol (contiguous batch of 4 over
    the full split), so the tensors and their per-window digests are byte-identical
    to the published stage-1 feature cache for the same split/condition.
    """
    windows = len(rows)
    features = np.empty((windows, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=FEATURE_DTYPE)
    window_digests = bytearray()
    probe_digest = hashlib.sha256()
    started = time.perf_counter()
    try:
        for start in range(0, windows, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, windows)
            signals = []
            for position in range(start, stop):
                tensor, label = read_window(dataset, position)
                expected = int(rows[position]["stage1_label"])
                if rows[position]["dataset_index_position"] != position:
                    raise RuntimeError("dataset row position mismatch")
                if int(label) != expected:
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
                raise RuntimeError("non-finite features")
            features[start:stop] = array
            for row in array:
                window_digests.extend(feature_digest(row))
            if (start // BATCH_SIZE + 1) % 200 == 0 or stop == windows:
                elapsed = time.perf_counter() - started
                log(
                    f"    {stop}/{windows} ({elapsed:.1f}s, "
                    f"{stop / max(elapsed, 1e-9):.1f} win/s)"
                )
    finally:
        dataset.close()
    return {
        "features": features,
        "window_digests": bytes(window_digests),
        "input_probe_sha256": probe_digest.hexdigest(),
        "seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# cascade evaluation
# ---------------------------------------------------------------------------
def build_prediction_row(
    spec: dict[str, Any],
    row: dict[str, Any],
    seed: int,
    epochs: dict[str, int],
    stage1_logits: Sequence[float],
    stage2_logits: Sequence[float] | None,
    stage2_batch_id: int | None,
    stage1_digest: str,
    stage2_digest: str | None,
) -> dict[str, Any]:
    """Assemble one cascade prediction row and apply the frozen decision rules.

    Margins are computed in float32 exactly like the frozen stage-1/stage-2
    evaluation code (``logits[:, 1] - logits[:, 0]`` on a float32 array), so the
    published stage-1 predictions can be reproduced bit for bit.
    """
    stage1 = np.asarray(stage1_logits, dtype=np.float32)
    logit_background = float(stage1[0])
    logit_drone = float(stage1[1])
    margin = float(np.float32(stage1[1] - stage1[0]))
    stage1_prediction = 1 if margin > 0.0 else 0
    executed = stage1_prediction == 1
    if executed:
        if stage2_logits is None:
            raise ValueError("stage 2 must be executed when the stage-1 margin is positive")
        stage2 = np.asarray(stage2_logits, dtype=np.float32)
        logit_phantom = float(stage2[0])
        logit_air2s = float(stage2[1])
        stage2_margin = float(np.float32(stage2[1] - stage2[0]))
        final = 2 if stage2_margin > 0.0 else 1
        if stage2_digest is None:
            raise ValueError("an executed stage-2 row must carry the digest of its input")
    else:
        if stage2_logits is not None:
            raise ValueError("stage 2 must not produce logits for a skipped row")
        logit_phantom = logit_air2s = stage2_margin = None
        final = BACKGROUND_CLASS
        stage2_digest = None
    return {
        "dataset": spec["dataset"],
        "condition": None,
        "training_seed": int(seed),
        "stage1_epoch": int(epochs["stage1_epoch"]),
        "stage2_epoch": int(epochs["stage2_epoch"]),
        "dataset_index_position": int(row["dataset_index_position"]),
        "file_path": row["file_path"],
        "filename": row["filename"],
        "target_code": row["target_code"],
        "target_name": row["target_name"],
        "distance_code": row["distance_code"],
        "segment_code": row["segment_code"],
        "row_split": row["split"],
        "stage1_label": int(row["stage1_label"]),
        "true_class": int(row["true_class"]),
        "true_class_name": row["true_class_name"],
        "block_index": int(row["block_index"]),
        "window_index": int(row["window_index"]),
        "start": int(row["start"]),
        "length": int(row["length"]),
        "source_samples": int(row["source_samples"]),
        "stage1_logit_background": logit_background,
        "stage1_logit_drone": logit_drone,
        "stage1_margin": margin,
        "stage1_prediction": stage1_prediction,
        "stage1_decision": "Drone" if executed else "Background",
        "stage2_executed": executed,
        "stage2_batch_id": stage2_batch_id,
        "stage2_logit_phantom": logit_phantom,
        "stage2_logit_air2s": logit_air2s,
        "stage2_margin": stage2_margin,
        "final_prediction": final,
        "final_class_name": CLASS_NAMES[final],
        "stage1_feature_digest": stage1_digest,
        "stage2_feature_digest": stage2_digest,
    }


def evaluate_dataset(
    spec: dict[str, Any],
    models: dict[int, dict[str, Any]],
    device: torch.device,
    root: Path,
    conditions: Sequence[str],
    seeds: Sequence[int],
    max_windows: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cascade every condition/window of one dataset for the requested seeds."""
    epochs = {
        seed: {
            "stage1_epoch": int(models[seed]["stage1_epoch"]),
            "stage2_epoch": int(models[seed]["stage2_epoch"]),
        }
        for seed in seeds
    }
    all_rows = dataset_rows(spec["index_path"], spec["split"])
    rows = all_rows if max_windows is None else all_rows[:max_windows]
    windows = len(rows)
    if not rows:
        raise ValueError(f"{spec['dataset']} has no rows")
    prediction_rows: list[dict[str, Any]] = []
    cache_metadata: list[dict[str, Any]] = []
    digest_dir = root / "feature_digests"
    frontend = models[seeds[0]]["frontend"]
    for condition in conditions:
        log(f"  features {spec['dataset']} {condition} ({windows} windows, nfft{N_FFT}/hop{HOP_LENGTH})")
        dataset = build_window_dataset(spec["index_path"], spec["split"], condition, spec["noise_seed"])
        built = build_features(dataset, rows, device, frontend)
        features = built["features"]
        digest_dir.mkdir(parents=True, exist_ok=True)
        digest_path = digest_dir / f"window_digests_{spec['dataset']}_{condition}.bin"
        digest_path.write_bytes(built["window_digests"])
        metadata = {
            "dataset": spec["dataset"],
            "split": spec["split"],
            "condition": condition,
            "fixed_snr_db": CONDITION_VALUES[condition],
            "noise_seed": spec["noise_seed"],
            "noise_epoch": NOISE_EPOCH,
            "rng_rule": RNG_RULE,
            "feature_batch_rule": FEATURE_BATCH_RULE,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "image_size": IMAGE_SIZE,
            "windows": windows,
            "shape": [windows, 1, IMAGE_SIZE, IMAGE_SIZE],
            "dtype": "float32",
            "feature_sha256": sha256_bytes(features.tobytes()),
            "window_digest_bytes": len(built["window_digests"]),
            "window_digests_concat_sha256": sha256_bytes(built["window_digests"]),
            "window_digest_file": str(digest_path.relative_to(root)),
            "window_digest_file_sha256": sha256_file(digest_path),
            "input_probe_samples": INPUT_PROBE_SAMPLES,
            "input_probe_sha256": built["input_probe_sha256"],
            "seconds": built["seconds"],
        }
        cache_metadata.append(metadata)
        log(
            f"    feature_sha256={metadata['feature_sha256']} "
            f"digests_sha256={metadata['window_digests_concat_sha256']} "
            f"({metadata['seconds']:.1f}s)"
        )
        for seed in seeds:
            entry = models[seed]
            stage1 = network_logits(entry["stage1_network"], features, device)
            margins = (stage1[:, 1] - stage1[:, 0]).astype(np.float32)
            selection = margins > 0.0
            stage2_logits = np.full((windows, 2), np.nan, dtype=np.float32)
            stage2_digests: dict[int, str] = {}
            batch_id_by_position: dict[int, int] = {}
            batches = stage2_batches(selection)
            with torch.inference_mode():
                for batch_id, positions in enumerate(batches):
                    batch_array = np.ascontiguousarray(features[positions])
                    batch = torch.from_numpy(batch_array).to(device)
                    with torch.amp.autocast(
                        device_type=device.type,
                        enabled=device.type == "cuda",
                        dtype=torch.float16,
                    ):
                        logits = entry["stage2_network"](batch)
                    logits = logits.float().cpu().numpy()
                    if not np.isfinite(logits).all():
                        raise RuntimeError(
                            f"non-finite stage-2 logits for seed {seed} {condition}"
                        )
                    stage2_logits[positions] = logits
                    for offset, position in enumerate(positions):
                        stage2_digests[int(position)] = sha256_bytes(
                            np.ascontiguousarray(batch_array[offset]).tobytes()
                        )
                        batch_id_by_position[int(position)] = batch_id
            executed = int(selection.sum())
            log(
                f"    seed {seed} {spec['dataset']}/{condition}: stage1 drone "
                f"{executed}/{windows}, stage2 batches {len(batches)}"
            )
            for position, row in enumerate(rows):
                digest = feature_digest(features[position])
                record = build_prediction_row(
                    spec,
                    row,
                    seed,
                    epochs[seed],
                    stage1[position],
                    stage2_logits[position] if bool(selection[position]) else None,
                    batch_id_by_position.get(position),
                    digest.hex(),
                    stage2_digests.get(position),
                )
                record["condition"] = condition
                prediction_rows.append(record)
        del features
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return prediction_rows, cache_metadata


# ---------------------------------------------------------------------------
# published stage-1 reference parity
# ---------------------------------------------------------------------------
def published_digest_name(dataset: str, condition: str, n_fft: int = N_FFT) -> str:
    """File name of the published stage-1 digest binary for one group.

    The published artefacts spell the split ``test`` (not ``old_test``) and append
    ``dB`` to the noisy conditions, e.g. ``window_digests_heldout_-10dB_nfft512.bin``
    and ``window_digests_test_clean_nfft512.bin``.
    """
    split_token = "test" if dataset == "old_test" else dataset
    condition_token = "clean" if condition == "clean" else f"{condition}dB"
    return f"window_digests_{split_token}_{condition_token}_nfft{n_fft}.bin"


def stage1_reference_parity(
    predictions: Sequence[dict[str, Any]],
    cache_metadata: Sequence[dict[str, Any]],
    root: Path,
) -> dict[str, Any]:
    """Compare the cascade stage-1 columns with the published stage-1 nfft512 results."""
    result: dict[str, Any] = {
        "rule": PARITY_RULE,
        "stage1_root": str(STAGE1_PUBLISHED_ROOT),
        "logits": {},
        "digests": {},
        "stage2_compared_with_oracle_csv": False,
        "stage2_note": (
            "stage 2 must NOT match the published oracle CSV: its RNG row positions were "
            "computed after filtering to drone rows only"
        ),
    }
    index_by_group: dict[tuple[str, str, int, int], dict[str, str]] = {}
    reference_files = {
        "heldout": STAGE1_PUBLISHED_ROOT / "heldout_window_predictions.csv",
        "old_test": STAGE1_PUBLISHED_ROOT / "test_clean_window_predictions.csv",
    }
    for dataset, path in reference_files.items():
        if not path.is_file():
            result["logits"][dataset] = {"skipped": f"missing {path}"}
            continue
        rows = read_csv_rows(path)
        for row in rows:
            if int(row["n_fft"]) != N_FFT:
                continue
            key = (
                dataset,
                condition_name(row["condition"]),
                int(row["training_seed"]),
                int(row["index_position"]),
            )
            index_by_group[key] = row
    groups: dict[tuple[str, str, int], int] = {}
    max_diffs = {"logit_background": 0.0, "logit_drone": 0.0, "logit_margin": 0.0}
    mismatches = 0
    missing_reference = 0
    compared = 0
    for record in predictions:
        key = (
            str(record["dataset"]),
            str(record["condition"]),
            int(record["training_seed"]),
            int(record["dataset_index_position"]),
        )
        reference = index_by_group.get(key)
        if reference is None:
            missing_reference += 1
            continue
        compared += 1
        group = (str(record["dataset"]), str(record["condition"]), int(record["training_seed"]))
        groups[group] = groups.get(group, 0) + 1
        for column, reference_column in (
            ("logit_background", "stage1_logit_background"),
            ("logit_drone", "stage1_logit_drone"),
            ("logit_margin", "stage1_margin"),
        ):
            difference = abs(float(record[reference_column]) - float(reference[column]))
            max_diffs[column] = max(max_diffs[column], difference)
        if condition_name(reference["condition"]) != str(record["condition"]):
            mismatches += 1
        if int(reference["prediction"]) != int(record["stage1_prediction"]):
            mismatches += 1
    result["logits"] = {
        "compared_rows": compared,
        "missing_reference_rows": missing_reference,
        "groups": [
            {"dataset": key[0], "condition": key[1], "training_seed": key[2], "rows": value}
            for key, value in sorted(groups.items())
        ],
        "max_abs_logit_background_diff": max_diffs["logit_background"],
        "max_abs_logit_drone_diff": max_diffs["logit_drone"],
        "max_abs_logit_margin_diff": max_diffs["logit_margin"],
        "prediction_mismatches": mismatches,
        "exact_match": (
            max_diffs["logit_background"] == 0.0
            and max_diffs["logit_drone"] == 0.0
            and max_diffs["logit_margin"] == 0.0
            and mismatches == 0
            and missing_reference == 0
            and compared > 0
        ),
    }
    for metadata in cache_metadata:
        dataset = str(metadata["dataset"])
        condition = str(metadata["condition"])
        reference_path = (
            STAGE1_PUBLISHED_ROOT
            / "feature_digests"
            / published_digest_name(dataset, condition, int(metadata.get("n_fft", N_FFT)))
        )
        entry: dict[str, Any] = {"reference": str(reference_path)}
        if not reference_path.is_file():
            entry["skipped"] = "missing published stage-1 digest binary"
        else:
            reference_bytes = reference_path.read_bytes()
            own_bytes = (root / str(metadata["window_digest_file"])).read_bytes()
            entry.update(
                {
                    "reference_bytes": len(reference_bytes),
                    "own_bytes": len(own_bytes),
                    "reference_sha256": sha256_bytes(reference_bytes),
                    "own_sha256": sha256_bytes(own_bytes),
                    "identical": reference_bytes == own_bytes,
                }
            )
        result["digests"][f"{dataset}:{condition}"] = entry
    result["digests_all_identical"] = all(
        bool(entry.get("identical")) for entry in result["digests"].values()
    ) and bool(result["digests"])
    return result


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def confusion_matrix(records: Sequence[dict[str, Any]]) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for record in records:
        truth = int(record["true_class"])
        prediction = int(record["final_prediction"])
        if truth not in CLASS_IDS or prediction not in CLASS_IDS:
            raise ValueError(f"class outside 0..2: truth={truth} prediction={prediction}")
        matrix[truth, prediction] += 1
    return matrix


def pooled_metrics(matrix: np.ndarray) -> dict[str, Any]:
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("empty confusion matrix")
    row_totals = matrix.sum(axis=1)
    column_totals = matrix.sum(axis=0)
    recalls = [
        float(matrix[label, label]) / float(row_totals[label]) if row_totals[label] else float("nan")
        for label in CLASS_IDS
    ]
    precisions = [
        float(matrix[label, label]) / float(column_totals[label])
        if column_totals[label]
        else float("nan")
        for label in CLASS_IDS
    ]
    f1_scores = [
        2 * precisions[label] * recalls[label] / (precisions[label] + recalls[label])
        if recalls[label] + precisions[label] > 0
        else 0.0
        for label in CLASS_IDS
    ]
    finite_recalls = [value for value in recalls if np.isfinite(value)]
    finite_precisions = [value for value in precisions if np.isfinite(value)]
    result = {
        "windows": total,
        "correct": int(np.trace(matrix)),
        "pooled_accuracy": float(np.trace(matrix)) / float(total),
        "macro_recall": mean_or_nan(finite_recalls),
        "balanced_accuracy": mean_or_nan(finite_recalls),
        "macro_precision": mean_or_nan(finite_precisions),
        "macro_f1": mean_or_nan(f1_scores),
    }
    for label, name in (("background", 0), ("phantom", 1), ("air2s", 2)):
        result[f"recall_{label}"] = recalls[name]
        result[f"precision_{label}"] = precisions[name]
        result[f"f1_{label}"] = f1_scores[name]
    for label, name in (("background", 0), ("phantom", 1), ("air2s", 2)):
        result[f"{label}_windows"] = int(row_totals[name])
    return result


def file_equal_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per source file inside each class, classes equally weighted (primary metric)."""
    per_file: dict[str, dict[str, Any]] = {}
    for record in records:
        filename = str(record["filename"])
        truth = int(record["true_class"])
        entry = per_file.setdefault(
            filename,
            {
                "true_class": truth,
                "windows": 0,
                "correct": 0,
                "predicted": {label: 0 for label in CLASS_IDS},
                "stage1_pass": 0,
            },
        )
        if entry["true_class"] != truth:
            raise ValueError(f"source file {filename} carries two different classes")
        entry["windows"] += 1
        entry["correct"] += int(int(record["final_prediction"]) == truth)
        entry["predicted"][int(record["final_prediction"])] += 1
        entry["stage1_pass"] += int(bool(record["stage2_executed"]))
    class_rates: dict[int, list[float]] = {label: [] for label in CLASS_IDS}
    per_file_rows: list[dict[str, Any]] = []
    for filename, entry in sorted(per_file.items()):
        rate = float(entry["correct"]) / float(max(entry["windows"], 1))
        class_rates[int(entry["true_class"])].append(rate)
        per_file_rows.append(
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
    class_rate = {
        label: float(np.mean(class_rates[label])) if class_rates[label] else float("nan")
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
        "file_equal_accuracy": mean_or_nan(available),
        "file_equal_classes_used": len(available),
        "per_file": per_file_rows,
        "per_file_rate": {row["filename"]: row["accuracy"] for row in per_file_rows},
    }


def stage1_rates(records: Sequence[dict[str, Any]], per_file: Sequence[dict[str, Any]]) -> dict[str, Any]:
    background = [record for record in records if int(record["true_class"]) == BACKGROUND_CLASS]
    drones = [record for record in records if int(record["true_class"]) in DRONE_CLASSES]
    false_alarms = sum(1 for record in background if bool(record["stage2_executed"]))
    misses = sum(1 for record in drones if not bool(record["stage2_executed"]))
    file_lookup = {str(row["filename"]): row for row in per_file}
    background_files = [
        row for row in per_file if int(row["true_class"]) == BACKGROUND_CLASS
    ]
    drone_files = [row for row in per_file if int(row["true_class"]) in DRONE_CLASSES]
    false_alarm_rates = [
        float(row["stage1_pass"]) / float(max(int(row["windows"]), 1)) for row in background_files
    ]
    miss_rates = [
        float(int(row["windows"]) - int(row["stage1_pass"])) / float(max(int(row["windows"]), 1))
        for row in drone_files
    ]
    return {
        "stage1_background_false_alarms": false_alarms,
        "stage1_background_false_alarm_rate": (
            float(false_alarms) / float(len(background)) if background else float("nan")
        ),
        "stage1_background_false_alarm_rate_file_equal": mean_or_nan(false_alarm_rates),
        "stage1_drone_misses": misses,
        "stage1_drone_miss_rate": (
            float(misses) / float(len(drones)) if drones else float("nan")
        ),
        "stage1_drone_miss_rate_file_equal": mean_or_nan(miss_rates),
        "stage1_drones_passed": len(drones) - misses,
        "per_file_lookup_size": len(file_lookup),
    }


def stage2_conditional_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy of stage 2 on true drones that actually passed stage 1."""
    subset = [
        record
        for record in records
        if int(record["true_class"]) in DRONE_CLASSES and bool(record["stage2_executed"])
    ]
    correct = sum(1 for record in subset if int(record["final_prediction"]) == int(record["true_class"]))
    return {
        "stage2_conditional_windows": len(subset),
        "stage2_conditional_correct": correct,
        "stage2_conditional_accuracy": (
            float(correct) / float(len(subset)) if subset else float("nan")
        ),
    }


def error_decomposition(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    matrix = confusion_matrix(records)
    row_totals = matrix.sum(axis=1)
    result: dict[str, Any] = {}
    for truth, prediction, name in ERROR_TRANSITIONS:
        count = int(matrix[truth, prediction])
        key = name.replace("->", "_to_").lower()
        result[f"error_{key}"] = count
        result[f"rate_{key}"] = (
            float(count) / float(row_totals[truth]) if row_totals[truth] else float("nan")
        )
    result["error_windows_total"] = int(matrix.sum() - np.trace(matrix))
    return result


def metric_record(records: Sequence[dict[str, Any]], columns: Sequence[str]) -> dict[str, Any]:
    matrix = confusion_matrix(records)
    pooled = pooled_metrics(matrix)
    file_equal = file_equal_metrics(records)
    per_file = file_equal.pop("per_file")
    rates = stage1_rates(records, per_file)
    rates.pop("per_file_lookup_size")
    conditional = stage2_conditional_metrics(records)
    errors = error_decomposition(records)
    row = {**pooled, **file_equal, **rates, **conditional, **errors}
    if columns is not None:
        return {key: row.get(key) for key in columns}
    return row


def group_records(
    records: Sequence[dict[str, Any]]
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["dataset"]), str(record["condition"]), int(record["training_seed"]))
        grouped.setdefault(key, []).append(record)
    return grouped


# ---------------------------------------------------------------------------
# integrity checks of the frozen inputs
# ---------------------------------------------------------------------------
def mat_reference_sizes() -> dict[str, int]:
    """Frozen MAT byte sizes: download manifest (old) + preparation summary (held-out)."""
    sizes: dict[str, int] = {}
    manifest_path = PROJECT_ROOT / "dronerfa_manifest.json"
    if manifest_path.is_file():
        for entry in read_json(manifest_path).get("entries", []):
            name = str(entry.get("name", ""))
            if name.lower().endswith(".mat"):
                sizes[Path(name).name] = int(entry["unpacked_size"])
    summary_path = (
        PROJECT_DIR
        / "outputs"
        / "heldout_preparation_20260917"
        / "heldout_dataset_summary.json"
    )
    if summary_path.is_file():
        for entry in read_json(summary_path).get("files", []):
            sizes[str(entry["filename"])] = int(entry["file_size_bytes"])
    return sizes


def verify_frozen_inputs(stage: str, root: Path | None = None) -> dict[str, Any]:
    root = root or OUTPUT_ROOT
    config = load_config()
    errors: list[str] = []
    config_sha = sha256_file(CONFIG_PATH, label="frozen config")
    spec_by_name = dataset_specs(config)
    record: dict[str, Any] = {
        "stage": stage,
        "checked_at": utc_now(),
        "config_path": str(CONFIG_PATH),
        "config_sha256": config_sha,
        "indices": {},
        "checkpoints": {},
        "preserved_inputs": {},
        "original_mat_files": {},
        "errors": errors,
    }
    for name, spec in spec_by_name.items():
        actual = sha256_file(spec["index_path"], label=f"{name} index")
        record["indices"][name] = {
            "path": str(spec["index_path"]),
            "split": spec["split"],
            "expected_sha256": spec["index_sha256"],
            "actual_sha256": actual,
            "match": actual == spec["index_sha256"],
            "bytes": spec["index_path"].stat().st_size,
        }
        if actual != spec["index_sha256"]:
            errors.append(f"{name} index SHA256 mismatch: {actual}")
        rows = dataset_rows(spec["index_path"], spec["split"])
        counts = class_counts(rows)
        record["indices"][name].update(
            {
                "rows": len(rows),
                "expected_rows": spec["windows"],
                "class_windows": {CLASS_NAMES[key]: value for key, value in counts.items()},
                "files_per_class": {
                    CLASS_NAMES[label]: sorted(
                        {row["filename"] for row in rows if int(row["true_class"]) == label}
                    )
                    for label in CLASS_IDS
                },
            }
        )
        if len(rows) != spec["windows"]:
            errors.append(f"{name} index has {len(rows)} rows, expected {spec['windows']}")
        if spec["expected_class_windows"] is not None:
            for label, expected in spec["expected_class_windows"].items():
                if counts[int(label)] != int(expected):
                    errors.append(
                        f"{name} class {CLASS_NAMES[int(label)]} has {counts[int(label)]} "
                        f"windows, expected {expected}"
                    )
    for entry in checkpoint_entries(config):
        path: Path = entry["checkpoint"]
        label = f"{'stage1' if entry['stage'] == 1 else 'stage2'} seed {entry['training_seed']}"
        if not path.is_file():
            errors.append(f"missing checkpoint {path}")
            record["checkpoints"][label] = {"path": str(path), "exists": False}
            continue
        actual = sha256_file(path, label=label)
        actual_bytes = path.stat().st_size
        match = actual == entry["sha256"] and actual_bytes == entry["bytes"]
        record["checkpoints"][label] = {
            "path": str(path),
            "exists": True,
            "expected_sha256": entry["sha256"],
            "actual_sha256": actual,
            "expected_bytes": entry["bytes"],
            "actual_bytes": actual_bytes,
            "epoch_in_config": entry["epoch"],
            "match": match,
        }
        if actual != entry["sha256"]:
            errors.append(f"{label} checkpoint SHA256 mismatch: {actual}")
        if actual_bytes != entry["bytes"]:
            errors.append(f"{label} checkpoint size mismatch: {actual_bytes}")

    preserved = read_json(PRESERVED_INPUTS)
    preserved_errors: list[str] = []
    for index, item in enumerate(preserved["files"], start=1):
        path = Path(str(item["path"]))
        if not path.is_file():
            preserved_errors.append(f"missing preserved input {path}")
            continue
        actual = sha256_file(path)
        if actual != str(item.get("sha256")):
            preserved_errors.append(f"preserved input {path} SHA256 mismatch: {actual}")
    record["preserved_inputs"] = {
        "path": str(PRESERVED_INPUTS),
        "files": len(preserved["files"]),
        "checked": len(preserved["files"]) - len(preserved_errors),
        "errors": preserved_errors,
    }
    errors.extend(preserved_errors)

    mat_errors: list[str] = []
    mat_sizes: dict[str, Any] = {}
    expected_sizes = mat_reference_sizes()
    for name, spec in spec_by_name.items():
        for row in dataset_rows(spec["index_path"], spec["split"]):
            path = Path(str(row["file_path"]))
            key = str(path)
            if key in mat_sizes:
                continue
            if not path.is_file():
                mat_errors.append(f"missing MAT {path}")
                continue
            size = path.stat().st_size
            expected = expected_sizes.get(path.name)
            mat_sizes[key] = size
            if expected is not None and int(expected) != int(size):
                mat_errors.append(
                    f"MAT {path} has {size} bytes, the preparation record says {expected}"
                )
    record["original_mat_files"] = {
        "files": len(mat_sizes),
        "bytes": int(sum(mat_sizes.values())),
        "sizes": mat_sizes,
        "reference_sizes": {name: expected_sizes[name] for name in sorted(expected_sizes)},
        "errors": mat_errors,
        "note": (
            "sizes are compared against the frozen download manifest (old MAT) and the "
            "held-out preparation summary (new MAT); the 11.5 GiB of data is not re-hashed here"
        ),
    }
    errors.extend(mat_errors)

    previous = root / (
        "frozen_inputs_verification.json" if stage == "after" else "frozen_inputs_verification_before.json"
    )
    record["passed"] = not errors
    record["verification_file"] = str(previous)
    write_json(previous, record)
    log(f"frozen inputs checked ({stage}): errors={len(errors)} -> {previous.name}")
    for message in errors:
        log(f"  ERROR {message}")
    return record


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def command_verify_inputs(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    record = verify_frozen_inputs(args.stage, root)
    return 0 if record["passed"] else 1


def evaluate_command_conditions(args: argparse.Namespace, spec: dict[str, Any]) -> list[str]:
    if not args.conditions:
        return list(spec["conditions"])
    requested = [condition_name(item) for item in args.conditions.split(",") if item.strip()]
    unknown = [item for item in requested if item not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}")
    if args.max_windows is None and requested != list(spec["conditions"]):
        raise SystemExit(
            f"a full run must use the frozen conditions {list(spec['conditions'])} for {spec['dataset']}"
        )
    return requested


def command_evaluate(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    config = load_config()
    config_sha_before = sha256_file(CONFIG_PATH)
    spec_by_name = dataset_specs(config)
    names = (
        [item.strip() for item in args.datasets.split(",") if item.strip()]
        if args.datasets
        else list(DATASET_NAMES)
    )
    unknown = [name for name in names if name not in spec_by_name]
    if unknown:
        raise SystemExit(f"unknown datasets {unknown}")
    seeds = (
        [int(item) for item in args.seeds.split(",") if item.strip()]
        if args.seeds
        else list(TRAINING_SEEDS)
    )
    if args.max_windows is None and seeds != list(TRAINING_SEEDS):
        raise SystemExit("a full run must use the three frozen training seeds")
    device = resolve_device(args.device)
    log(f"device {device}")
    log(f"config {CONFIG_PATH} sha256={config_sha_before}")
    started = time.perf_counter()
    models = load_cascade_models(config, device, seeds)
    predictions: list[dict[str, Any]] = []
    cache_metadata: list[dict[str, Any]] = []
    for name in names:
        spec = spec_by_name[name]
        conditions = evaluate_command_conditions(args, spec)
        log(f"dataset {name} ({spec['split']}) conditions={conditions} seeds={seeds}")
        rows, cache = evaluate_dataset(
            spec,
            models,
            device,
            root,
            conditions,
            seeds,
            max_windows=args.max_windows,
        )
        predictions.extend(rows)
        cache_metadata.extend(cache)
    predictions.sort(
        key=lambda row: (
            DATASET_NAMES.index(str(row["dataset"])),
            int(row["training_seed"]),
            CONDITIONS.index(str(row["condition"])) if str(row["condition"]) in CONDITIONS else 99,
            int(row["dataset_index_position"]),
        )
    )
    if args.max_windows is None:
        for name in names:
            spec = spec_by_name[name]
            expected = spec["expected_rows"] * len(seeds) // len(TRAINING_SEEDS)
            present = sum(1 for row in predictions if str(row["dataset"]) == name)
            if present != expected:
                raise SystemExit(f"{name} produced {present} rows, expected {expected}")
    write_csv(root / "window_predictions.csv", predictions, PREDICTION_COLUMNS)
    parity = stage1_reference_parity(predictions, cache_metadata, root)
    write_json(root / "stage1_reference_parity.json", parity)
    cache_manifest = {
        "rule": "one noisy signal and one float32 feature tensor per dataset/condition/window",
        "rng_rule": RNG_RULE,
        "feature_batch_rule": FEATURE_BATCH_RULE,
        "stage1_rule": STAGE1_RULE,
        "stage2_rule": STAGE2_RULE,
        "stage2_batch_rule": STAGE2_BATCH_RULE,
        "stage2_skipped_rule": STAGE2_SKIPPED_RULE,
        "truncated_debug_run": args.max_windows is not None,
        "max_windows": args.max_windows,
        "seeds": seeds,
        "datasets": names,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "conditions": list(CONDITIONS),
        "config_path": str(CONFIG_PATH),
        "config_sha256": config_sha_before,
        "prediction_rows": len(predictions),
        "cache": cache_metadata,
        "parity_summary": {
            "logits_exact_match": parity["logits"].get("exact_match"),
            "digests_all_identical": parity.get("digests_all_identical"),
        },
        "seconds": time.perf_counter() - started,
    }
    write_json(root / "prediction_cache_manifest.json", cache_manifest)
    write_json(
        root / "evaluation_run.json",
        {
            "started_at": utc_now(),
            "dataset": names,
            "seeds": seeds,
            "conditions": list(CONDITIONS),
            "max_windows": args.max_windows,
            "prediction_rows": len(predictions),
            "device": str(device),
            "config_sha256": config_sha_before,
            "config_sha256_after": sha256_file(CONFIG_PATH),
            "seconds": time.perf_counter() - started,
        },
    )
    log(f"wrote {root / 'window_predictions.csv'} ({len(predictions)} rows)")
    return 0


def parse_prediction_rows(path: Path) -> list[dict[str, Any]]:
    """Read the prediction CSV and cast the numeric/boolean columns."""
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
            record[key] = int(raw[key])
        for key in (
            "stage1_logit_background",
            "stage1_logit_drone",
            "stage1_margin",
        ):
            record[key] = float(raw[key])
        for key in ("stage2_logit_phantom", "stage2_logit_air2s", "stage2_margin"):
            text = (raw.get(key) or "").strip()
            record[key] = float(text) if text else None
        record["stage2_executed"] = (raw["stage2_executed"] or "").strip().lower() == "true"
        text = (raw.get("stage2_batch_id") or "").strip()
        record["stage2_batch_id"] = int(text) if text else None
        record["stage2_feature_digest"] = (raw.get("stage2_feature_digest") or "").strip() or None
        records.append(record)
    return records


def command_parity(args: argparse.Namespace) -> int:
    """Recompute the published stage-1 parity evidence from the existing artefacts.

    Nothing is re-inferred: the prediction CSV and the digest binaries are read back
    and compared with the published stage-1 nfft512 results, which keeps the parity
    evidence reproducible after a naming fix in the reporting code.
    """
    root = args.output_dir or OUTPUT_ROOT
    predictions = parse_prediction_rows(root / "window_predictions.csv")
    cache_path = root / "prediction_cache_manifest.json"
    cache_manifest = read_json(cache_path)
    parity = stage1_reference_parity(predictions, cache_manifest["cache"], root)
    write_json(root / "stage1_reference_parity.json", parity)
    cache_manifest["parity_summary"] = {
        "logits_exact_match": parity["logits"].get("exact_match"),
        "digests_all_identical": parity.get("digests_all_identical"),
    }
    write_json(cache_path, cache_manifest)
    log(
        f"parity: rows={parity['logits'].get('compared_rows')} "
        f"exact={parity['logits'].get('exact_match')} "
        f"digests_identical={parity['digests_all_identical']}"
    )
    for key, entry in parity["digests"].items():
        log(f"  {key}: identical={entry.get('identical')} {entry.get('own_bytes')} bytes")
    return 0


def build_confusion_rows(
    predictions: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (dataset, condition, seed), group in sorted(group_records(predictions).items()):
        matrix = confusion_matrix(group)
        for truth in CLASS_IDS:
            for prediction in CLASS_IDS:
                rows.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "training_seed": seed,
                        "true_class": truth,
                        "true_class_name": CLASS_NAMES[truth],
                        "predicted_class": prediction,
                        "predicted_class_name": CLASS_NAMES[prediction],
                        "count": int(matrix[truth, prediction]),
                    }
                )
    return rows


def command_summarize(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    predictions = parse_prediction_rows(root / "window_predictions.csv")
    grouped = group_records(predictions)
    metric_rows: list[dict[str, Any]] = []
    per_file_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for (dataset, condition, seed), group in sorted(grouped.items()):
        row = metric_record(group, METRIC_COLUMNS)
        row.update({"dataset": dataset, "condition": condition, "training_seed": seed})
        stages = group[0]
        row["stage1_epoch"] = int(stages["stage1_epoch"])
        row["stage2_epoch"] = int(stages["stage2_epoch"])
        metric_rows.append({key: row.get(key) for key in METRIC_COLUMNS})
        file_equal = file_equal_metrics(group)
        for entry in file_equal["per_file"]:
            per_file_rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "training_seed": seed,
                    "filename": entry["filename"],
                    "true_class": entry["true_class"],
                    "true_class_name": CLASS_NAMES[int(entry["true_class"])],
                    "windows": entry["windows"],
                    "correct": entry["correct"],
                    "accuracy": entry["accuracy"],
                    "stage1_pass": entry["stage1_pass"],
                    "predicted_background": entry["predicted_background"],
                    "predicted_phantom": entry["predicted_phantom"],
                    "predicted_air2s": entry["predicted_air2s"],
                }
            )
        errors = error_decomposition(group)
        matrix = confusion_matrix(group)
        for truth, prediction, name in ERROR_TRANSITIONS:
            error_rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "training_seed": seed,
                    "transition": name,
                    "true_class": truth,
                    "predicted_class": prediction,
                    "count": int(matrix[truth, prediction]),
                    "rate_of_true_class": errors[f"rate_{name.replace('->', '_to_').lower()}"],
                }
            )
    summary_rows: list[dict[str, Any]] = []
    datasets = sorted({key[0] for key in grouped})
    for dataset in datasets:
        conditions = sorted(
            {key[1] for key in grouped if key[0] == dataset},
            key=lambda item: CONDITIONS.index(item) if item in CONDITIONS else 99,
        )
        seeds = sorted({key[2] for key in grouped if key[0] == dataset})
        for condition in conditions:
            group_rows = [
                row
                for row in metric_rows
                if row["dataset"] == dataset and row["condition"] == condition
            ]
            for metric in CROSS_SEED_METRICS:
                values = [float(row[metric]) for row in group_rows if row.get(metric) is not None]
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "metric": metric,
                        "seeds": len(values),
                        "seed_values": ",".join(f"{value:.10g}" for value in values),
                        "mean": mean_or_nan(values),
                        "sample_std": sample_std(values),
                    }
                )
        if len(conditions) > 1:
            for metric in CROSS_SEED_METRICS:
                per_seed_means = []
                for seed in seeds:
                    values = [
                        float(row[metric])
                        for row in metric_rows
                        if row["dataset"] == dataset
                        and row["training_seed"] == seed
                        and row.get(metric) is not None
                        and np.isfinite(float(row[metric]))
                    ]
                    if values:
                        per_seed_means.append(float(np.mean(values)))
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "condition": "four_condition_mean",
                        "metric": metric,
                        "seeds": len(per_seed_means),
                        "seed_values": ",".join(f"{value:.10g}" for value in per_seed_means),
                        "mean": mean_or_nan(per_seed_means),
                        "sample_std": sample_std(per_seed_means),
                    }
                )
    write_csv(root / "confusion_matrices.csv", build_confusion_rows(predictions))
    write_json(
        root / "confusion_matrices.json",
        {
            "classes": {str(key): value for key, value in CLASS_NAMES.items()},
            "matrices": [
                {
                    "dataset": dataset,
                    "condition": condition,
                    "training_seed": seed,
                    "matrix": confusion_matrix(group).tolist(),
                    "row_totals": confusion_matrix(group).sum(axis=1).tolist(),
                }
                for (dataset, condition, seed), group in sorted(grouped.items())
            ],
        },
    )
    write_csv(root / "metrics_by_condition.csv", metric_rows, METRIC_COLUMNS)
    write_json(root / "metrics_by_condition.json", metric_rows)
    write_csv(root / "per_file_metrics.csv", per_file_rows)
    write_csv(root / "error_decomposition.csv", error_rows)
    write_csv(root / "cross_seed_summary.csv", summary_rows)
    log(
        f"summarize: {len(metric_rows)} model-condition rows, {len(per_file_rows)} per-file rows, "
        f"{len(error_rows)} error rows, {len(summary_rows)} cross-seed rows"
    )
    return 0


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
RECALL_FIGURE = "recall_by_condition.png"
RECALL_METRIC_BY_CLASS: dict[int, tuple[str, str]] = {
    0: ("background", "file_equal_background_recall"),
    1: ("phantom", "file_equal_phantom_recall"),
    2: ("air2s", "file_equal_air2s_recall"),
}
CROSS_SEED_FIGURE_METRICS: tuple[str, ...] = ("file_equal_accuracy",) + tuple(
    RECALL_METRIC_BY_CLASS[label][1] for label in (0, 1, 2)
)
PUBLISHED_FIGURES: tuple[str, ...] = (
    "confusion_heldout_clean.png",
    "confusion_heldout_-10.png",
    "confusion_heldout_-12.png",
    "confusion_heldout_-15.png",
    "confusion_old_test_clean.png",
    RECALL_FIGURE,
    "cascade_flow.png",
)


def canonical_conditions(values: Iterable[str]) -> list[str]:
    """Every observed condition once, in the frozen ``CONDITIONS`` order.

    The recall figure draws one x position per entry of this global list, so a
    later dataset that only observed ``clean`` (``old_test``) can never shrink the
    axis back to a single labelled tick.
    """
    unique = {str(value) for value in values}
    known = [condition for condition in CONDITIONS if condition in unique]
    unknown = sorted(unique.difference(CONDITIONS))
    return known + unknown


def dataset_conditions(
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]], dataset: str
) -> list[str]:
    """Conditions observed for one dataset, frozen order, no duplicates."""
    return canonical_conditions(key[1] for key in grouped if key[0] == dataset)


def cross_seed_figure_metrics(root: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    """``(dataset, condition, metric) -> {mean, sample_std}`` from ``cross_seed_summary.csv``.

    The recall figure is drawn from these published numbers instead of recomputing
    them, so ``figure_evidence.json`` can pin the figure to ``cross_seed_summary.csv``.
    """
    table: dict[tuple[str, str, str], dict[str, float]] = {}
    path = root / "cross_seed_summary.csv"
    if not path.is_file():
        return table
    for row in read_csv_rows(path):
        metric = str(row.get("metric", ""))
        if metric not in CROSS_SEED_FIGURE_METRICS:
            continue
        try:
            mean = float(row["mean"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            sample_std = float(row["sample_std"])
        except (KeyError, TypeError, ValueError):
            sample_std = float("nan")
        table[(str(row["dataset"]), str(row["condition"]), metric)] = {
            "mean": mean,
            "sample_std": sample_std,
        }
    return table


def build_recall_figure(
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]],
    plt,
    summary_table: dict[tuple[str, str, str], dict[str, float]] | None = None,
):
    """One x position per frozen condition for *every* dataset (rework 2).

    Returns ``(figure, axis, conditions)``.  Every dataset/condition pair is drawn at
    ``conditions.index(condition)``, the ticks are set exactly once for the whole
    figure, and a dataset that only observed ``clean`` (``old_test``) contributes
    that single position without touching the axis ticks.

    ``summary_table`` optionally supplies the already published cross-seed means
    (``cross_seed_summary.csv``); without it the numbers are recomputed from
    ``grouped``.
    """
    datasets = sorted({key[0] for key in grouped})
    conditions = canonical_conditions(key[1] for key in grouped)
    positions = {condition: float(index) for index, condition in enumerate(conditions)}

    figure, axis = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    colors = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c"}
    for dataset in datasets:
        seeds = sorted({key[2] for key in grouped if key[0] == dataset})
        observed = dataset_conditions(grouped, dataset)
        for label in CLASS_IDS:
            metric = RECALL_METRIC_BY_CLASS[label][1]
            means, errors = [], []
            for condition in observed:
                if summary_table is not None:
                    entry = summary_table.get((dataset, condition, metric))
                    mean = float(entry["mean"]) if entry else float("nan")
                    error = float(entry["sample_std"]) if entry else float("nan")
                    mean = mean if np.isfinite(mean) else float("nan")
                    error = error if np.isfinite(error) else float("nan")
                else:
                    values = [
                        float(metric_record(grouped[(dataset, condition, seed)], None)[metric])
                        for seed in seeds
                        if (dataset, condition, seed) in grouped
                    ]
                    values = [value for value in values if np.isfinite(value)]
                    mean = float(np.mean(values)) if values else float("nan")
                    error = sample_std(values)
                means.append(mean)
                errors.append(error)
            axis.errorbar(
                [positions[condition] for condition in observed],
                means,
                yerr=errors,
                marker="o",
                capsize=3,
                color=colors[label],
                label=f"{dataset}: {CLASS_NAMES[label]}",
                alpha=0.9,
            )
    axis.set_xticks([positions[condition] for condition in conditions], conditions)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("noise condition (AWGN, epoch 0, seed 22026)")
    axis.set_ylabel("per-class file-equal recall (mean +/- sample std over 3 seeds)")
    axis.set_title("CASCADE3 per-class recall across conditions")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="lower left", ncol=2)
    return figure, axis, conditions


def command_figures(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.output_dir or OUTPUT_ROOT
    predictions = parse_prediction_rows(root / "window_predictions.csv")
    grouped = group_records(predictions)
    datasets = sorted({key[0] for key in grouped})
    seeds = sorted({key[2] for key in grouped})
    evidence: dict[str, Any] = {
        "rule": "aggregate figures only, no per-window PNG",
        "source": str(root / "window_predictions.csv"),
        "expected_figures": list(PUBLISHED_FIGURES),
        "figures": {},
    }

    def save(figure, name: str) -> None:
        path = root / name
        figure.savefig(path, dpi=150)
        plt.close(figure)
        evidence["figures"][name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        log(f"  wrote {name}")

    for dataset in datasets:
        conditions = dataset_conditions(grouped, dataset)
        for condition in conditions:
            available = [seed for seed in seeds if (dataset, condition, seed) in grouped]
            figure, axes = plt.subplots(
                1, len(available), figsize=(4.6 * len(available), 4.8), squeeze=False, constrained_layout=True
            )
            for axis, seed in zip(axes[0], available):
                group = grouped[(dataset, condition, seed)]
                matrix = confusion_matrix(group)
                row_totals = matrix.sum(axis=1)
                image = axis.imshow(matrix, cmap="Blues")
                for truth in CLASS_IDS:
                    for prediction in CLASS_IDS:
                        count = int(matrix[truth, prediction])
                        share = count / row_totals[truth] if row_totals[truth] else 0.0
                        axis.text(
                            prediction,
                            truth,
                            f"{count}\n{100 * share:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if count > 0.6 * max(1, matrix.max()) else "black",
                        )
                axis.set_xticks(CLASS_IDS, ["Background", "Phantom", "Air2S"], fontsize=8)
                axis.set_yticks(CLASS_IDS, ["Background", "Phantom", "Air2S"], fontsize=8)
                axis.set_xlabel("predicted (cascade)")
                axis.set_ylabel("true class")
                accuracy = float(np.trace(matrix)) / float(max(int(matrix.sum()), 1))
                axis.set_title(
                    f"seed {seed} | {dataset} {condition}\nwindows {int(matrix.sum())}, "
                    f"accuracy {100 * accuracy:.2f}%",
                    fontsize=9,
                )
                figure.colorbar(image, ax=axis, shrink=0.85)
            figure.suptitle(
                f"CASCADE3 confusion matrices (counts / share of the true class), {dataset}, "
                f"condition={condition}",
                fontsize=11,
            )
            save(figure, f"confusion_{dataset}_{condition}.png")

    summary_table = cross_seed_figure_metrics(root)
    recall_figure, recall_axis, recall_conditions = build_recall_figure(
        grouped, plt, summary_table=summary_table
    )
    save(recall_figure, RECALL_FIGURE)
    evidence["recall_axis"] = {
        "conditions": recall_conditions,
        "tick_labels": [text.get_text() for text in recall_axis.get_xticklabels()],
        "datasets": datasets,
        "dataset_conditions": {
            dataset: dataset_conditions(grouped, dataset) for dataset in datasets
        },
        "source_table": str(root / "cross_seed_summary.csv"),
        "source_table_sha256": (
            sha256_file(root / "cross_seed_summary.csv")
            if (root / "cross_seed_summary.csv").is_file()
            else None
        ),
        "rule": (
            "one x position per frozen condition shared by every dataset; a dataset "
            "that observed fewer conditions only contributes its own positions and "
            "never rewrites the axis ticks"
        ),
    }

    figure, axis = plt.subplots(figsize=(9.6, 5.6), constrained_layout=True)
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 8)
    axis.axis("off")

    def box(x: float, y: float, width: float, height: float, text: str, color: str) -> None:
        axis.add_patch(
            plt.Rectangle((x, y), width, height, facecolor=color, edgecolor="black", alpha=0.85)
        )
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(x1: float, y1: float, x2: float, y2: float, text: str = "") -> None:
        axis.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
        )
        if text:
            axis.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2, text, fontsize=8, ha="left", va="center")

    box(0.2, 5.6, 2.6, 1.3, "RF0 I/Q window\n1,000,000 samples", "#dfe7f5")
    box(3.4, 5.6, 3.0, 1.3, "demean + RMS + complex AWGN\n(seed 22026 + full-split row)", "#dfe7f5")
    box(7.0, 5.6, 4.6, 1.3, "Hamming STFT n_fft=512, hop=256\nlog power + fftshift + z-score -> 1x256x256", "#dfe7f5")
    box(3.4, 3.3, 5.0, 1.3, "Stage-1 ResNet18 (seed-matched, nfft512 epoch locked)\nmargin = logit_drone - logit_background", "#e3f2df")
    box(8.8, 3.3, 2.9, 1.3, "one float32 tensor\nshared by both stages", "#f6e6d0")
    box(0.2, 0.9, 3.0, 1.2, "final: Background (class 0)\nstage2_executed = false", "#f7d9d9")
    box(4.6, 0.9, 3.6, 1.2, "Stage-2 ResNet18 (seed-matched)\nmargin = logit_air2s - logit_phantom", "#e3f2df")
    box(8.9, 1.6, 2.8, 0.9, "Air 2S (class 2)", "#f7d9d9")
    box(8.9, 0.4, 2.8, 0.9, "Phantom 4 Pro (class 1)", "#f7d9d9")
    arrow(2.8, 6.25, 3.4, 6.25)
    arrow(6.4, 6.25, 7.0, 6.25)
    arrow(9.3, 5.6, 9.3, 4.6)
    arrow(6.0, 4.6, 6.0, 5.6)
    arrow(3.6, 3.3, 2.2, 2.1, "margin <= 0")
    arrow(6.4, 3.3, 6.4, 2.1, "margin > 0")
    arrow(8.2, 1.8, 8.9, 2.05, "margin > 0")
    arrow(8.2, 1.2, 8.9, 0.85, "margin <= 0")
    axis.set_title(
        "CASCADE3 flow: one noisy signal and one feature tensor per window, "
        "seed-matched stage-1 -> stage-2, no threshold tuning",
        fontsize=10,
    )
    save(figure, "cascade_flow.png")

    evidence["figures_written"] = sorted(evidence["figures"])
    evidence["missing_figures"] = [
        name for name in PUBLISHED_FIGURES if name not in evidence["figures"]
    ]
    write_json(root / "figure_evidence.json", evidence)
    return 0


# ---------------------------------------------------------------------------
# manifest and report
# ---------------------------------------------------------------------------
def command_manifest(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    config = load_config()
    cache = read_json(root / "prediction_cache_manifest.json")
    manifest: dict[str, Any] = {
        "task": "CASCADE3",
        "generated_at": utc_now(),
        "output_root": str(root),
        "config": {
            "path": str(CONFIG_PATH),
            "sha256": sha256_file(CONFIG_PATH),
            "task_scope_exclusions": config.get("scope_exclusions"),
            "selection_policy": config.get("selection_policy"),
            "decision": config.get("decision"),
        },
        "protocol": {
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "frontend": config["frontend"]["preprocessing"],
            "rng_rule": RNG_RULE,
            "feature_batch_rule": FEATURE_BATCH_RULE,
            "stage1_rule": STAGE1_RULE,
            "stage2_rule": STAGE2_RULE,
            "stage2_batch_rule": STAGE2_BATCH_RULE,
            "stage2_skipped_rule": STAGE2_SKIPPED_RULE,
            "truncated_debug_run": cache.get("truncated_debug_run"),
        },
        "inputs": {},
        "counts": {},
        "published_files": {},
        "checks": {},
    }
    verification_before = root / "frozen_inputs_verification_before.json"
    verification_after = root / "frozen_inputs_verification.json"
    for name, path in (("before", verification_before), ("after", verification_after)):
        if path.is_file():
            record = read_json(path)
            manifest["inputs"][name] = {
                "file": path.name,
                "sha256": sha256_file(path),
                "passed": record.get("passed"),
                "config_sha256": record.get("config_sha256"),
                "errors": record.get("errors"),
            }
    if verification_before.is_file() and verification_after.is_file():
        manifest["config_unchanged_during_inference"] = (
            read_json(verification_before).get("config_sha256")
            == read_json(verification_after).get("config_sha256")
        )
    prediction_path = root / "window_predictions.csv"
    if prediction_path.is_file():
        rows = parse_prediction_rows(prediction_path)
        manifest["counts"]["prediction_rows"] = len(rows)
        manifest["counts"]["by_dataset"] = {
            dataset: sum(1 for row in rows if str(row["dataset"]) == dataset)
            for dataset in sorted({str(row["dataset"]) for row in rows})
        }
        manifest["counts"]["by_dataset_condition_seed"] = [
            {
                "dataset": key[0],
                "condition": key[1],
                "training_seed": key[2],
                "rows": len(group),
            }
            for key, group in sorted(group_records(rows).items())
        ]
        manifest["counts"]["class_windows"] = {
            CLASS_NAMES[label]: sum(1 for row in rows if int(row["true_class"]) == label)
            for label in CLASS_IDS
        }
    for name in (
        "window_predictions.csv",
        "prediction_cache_manifest.json",
        "stage1_reference_parity.json",
        "confusion_matrices.csv",
        "confusion_matrices.json",
        "metrics_by_condition.csv",
        "metrics_by_condition.json",
        "per_file_metrics.csv",
        "error_decomposition.csv",
        "cross_seed_summary.csv",
        "figure_evidence.json",
        "report.md",
        "verification.json",
        "evaluation_run.json",
    ):
        path = root / name
        if path.is_file():
            manifest["published_files"][name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    digest_dir = root / "feature_digests"
    if digest_dir.is_dir():
        manifest["published_files"]["feature_digests/"] = {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(digest_dir.glob("*.bin"))
        }
    parity_path = root / "stage1_reference_parity.json"
    if parity_path.is_file():
        parity = read_json(parity_path)
        manifest["checks"]["stage1_reference_parity"] = {
            "logits": {key: value for key, value in parity["logits"].items() if key != "groups"},
            "digests_all_identical": parity.get("digests_all_identical"),
        }
    verification_path = root / "verification.json"
    if verification_path.is_file():
        verification = read_json(verification_path)
        manifest["checks"]["independent_verification"] = {
            "passed": verification.get("passed"),
            "failures": verification.get("failures"),
            "generated_at": verification.get("generated_at"),
        }
    environment = root / "environment.json"
    if environment.is_file():
        manifest["environment"] = read_json(environment)
    write_json(root / "experiment_manifest.json", manifest)
    log(f"manifest written ({len(manifest['published_files'])} published entries)")
    return 0


def command_environment(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    import matplotlib
    import h5py

    device = resolve_device(args.device)
    record = {
        "generated_at": utc_now(),
        "python": sys_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "matplotlib": matplotlib.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "amp": "torch.amp.autocast float16 for both stages; frontend stays float32",
        "git_commit": None,
    }
    write_json(root / "environment.json", record)
    return 0


def sys_version() -> str:
    import sys

    return sys.version


def command_report(args: argparse.Namespace) -> int:
    root = args.output_dir or OUTPUT_ROOT
    metrics = read_csv_rows(root / "metrics_by_condition.csv")
    summary = read_csv_rows(root / "cross_seed_summary.csv")
    confusion = read_csv_rows(root / "confusion_matrices.csv")
    per_file = read_csv_rows(root / "per_file_metrics.csv")
    errors = read_csv_rows(root / "error_decomposition.csv")
    cache = read_json(root / "prediction_cache_manifest.json")
    parity = read_json(root / "stage1_reference_parity.json")
    verification_path = root / "verification.json"

    def summary_value(dataset: str, condition: str, metric: str, column: str) -> str:
        for row in summary:
            if (
                row["dataset"] == dataset
                and row["condition"] == condition
                and row["metric"] == metric
            ):
                return row[column]
        return ""

    def number(text: str) -> float:
        return float(text) if text else float("nan")

    datasets = sorted({row["dataset"] for row in metrics})
    lines: list[str] = []
    lines.append("# CASCADE3：第一阶段检测 + 第二阶段机型分类直接串联（三分类）")
    lines.append("")
    lines.append(
        "本报告把已验收的第一阶段 Background/Drone 检测器与已验收的第二阶段 "
        "Phantom 4 Pro / Air 2S 分类器**直接串联**，得到 Background / DJI Phantom 4 Pro / "
        "DJI Air 2S 三分类结果。三个训练种子各自串联（2026-2026、2027-2027、2028-2028），"
        "不做跨种子投票或集成。"
    )
    lines.append("")
    lines.append(
        "冻结协议：RF0 I/Q、1,000,000 点窗、`n_fft=512`、`hop_length=256`、`image_size=256`；"
        "每个数据集/条件/窗口只生成一次噪声与一次 `1x256x256` float32 特征张量，"
        "同一张张量先给第一阶段，通过后原样交给第二阶段；"
        "噪声 RNG 位置取**完整 split 的行号**（背景行同样占位），seed 22026、epoch 0。"
        "第一阶段规则 `logit_drone - logit_background > 0` 才执行第二阶段，"
        "第二阶段规则 `logit_air2s - logit_phantom > 0` 判 Air 2S，否则 Phantom 4 Pro。"
        "不重训、不调门槛、不连接既有预测 CSV，全部从原始 IQ 重算。"
    )
    lines.append("")
    lines.append("## 1. 结论摘要（文件等权主口径）")
    lines.append("")
    lines.append(
        "主口径：先算每个源文件在其真实类上的正确率，类内源文件等权，再三类等权"
        "（file-equal accuracy）；表中同时列出 pooled 窗口口径，两者不可混用。"
        "`four_condition_mean` 行先在每个 seed 内对四档求均值，再对三个 seed 求均值与样本标准差。"
    )
    lines.append("")
    lines.append("| 数据集 | 条件 | 三 seed 文件等权准确率均值 | 样本标准差 | pooled 准确率均值 | 宏召回均值 | 宏 F1 均值 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        conditions = []
        for row in summary:
            if row["dataset"] == dataset and row["condition"] not in conditions:
                conditions.append(row["condition"])
        for condition in conditions:
            lines.append(
                f"| {dataset} | {condition} | "
                f"{pct(number(summary_value(dataset, condition, 'file_equal_accuracy', 'mean')))} | "
                f"{100 * number(summary_value(dataset, condition, 'file_equal_accuracy', 'sample_std')):.2f}pp | "
                f"{pct(number(summary_value(dataset, condition, 'pooled_accuracy', 'mean')))} | "
                f"{pct(number(summary_value(dataset, condition, 'macro_recall', 'mean')))} | "
                f"{pct(number(summary_value(dataset, condition, 'macro_f1', 'mean')))} |"
            )
    lines.append("")
    lines.append("## 2. 任务与口径边界")
    lines.append("")
    lines.append("- 本任务是**评价**：不训练模型、不用新 heldout 选择配置或门槛、不做噪声方式实验。")
    lines.append(
        "- 第三类“Background”不是新模型：它等于第一阶段判为背景（margin <= 0）的结果，"
        "此时 `stage2_executed=false`，第二阶段 logits 与 margin 留空，不做任何伪造。"
    )
    lines.append(
        "- 新 heldout 已参与此前的 STFT 点数消融与第二阶段选型背景，因此这里是**描述性留出评价**，"
        "不能称“全新未触碰的最终测试集”，也不能称跨会话或跨场景泛化。"
    )
    lines.append(
        "- 新 heldout 每类只有 4 个源文件、旧 test 每类只有 1 个源文件；"
        "四档条件是人工复 AWGN，不等于现场绝对 SNR。"
    )
    lines.append(
        "- 旧 test 只做 clean：四档噪声只在 heldout 上评价，两个数据集的行数与条件数因此不同。"
    )
    lines.append("")
    lines.append("## 3. 数据与规模")
    lines.append("")
    lines.append("| 数据集 | split | 条件 | 每个 seed 级联的预测行数 | 类别窗口 |")
    lines.append("|---|---|---|---:|---|")
    dataset_seen: list[str] = []
    for entry in cache["cache"]:
        if entry["dataset"] in dataset_seen:
            continue
        dataset_seen.append(entry["dataset"])
        group = [
            row
            for row in metrics
            if row["dataset"] == entry["dataset"] and int(row["training_seed"]) == min(
                int(item["training_seed"]) for item in metrics if item["dataset"] == entry["dataset"]
            )
        ]
        windows = group[0]["windows"] if group else ""
        background = group[0]["background_windows"] if group else ""
        phantom = group[0]["phantom_windows"] if group else ""
        air2s = group[0]["air2s_windows"] if group else ""
        conditions = sorted(
            {item["condition"] for item in cache["cache"] if item["dataset"] == entry["dataset"]},
            key=lambda item: CONDITIONS.index(item),
        )
        lines.append(
            f"| {entry['dataset']} | {entry['split']} | {', '.join(conditions)} | {windows} | "
            f"Background {background} / Phantom {phantom} / Air2S {air2s} |"
        )
    lines.append("")
    lines.append(
        f"合计预测行数 {cache['prediction_rows']}（heldout 21,840 + 旧 test clean 1,320），"
        "主键为 (dataset, condition, training_seed, dataset_index_position)。"
    )
    lines.append("")
    lines.append("## 4. 每个 seed/条件的完整指标")
    lines.append("")
    lines.append(
        "| 数据集 | 条件 | seed | 文件等权准确率 | pooled 准确率 | 宏召回 | 宏精确率 | 宏 F1 | "
        "Background recall | Phantom recall | Air2S recall |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in metrics:
        lines.append(
            f"| {row['dataset']} | {row['condition']} | {row['training_seed']} | "
            f"{pct(number(row['file_equal_accuracy']))} | {pct(number(row['pooled_accuracy']))} | "
            f"{pct(number(row['macro_recall']))} | {pct(number(row['macro_precision']))} | "
            f"{pct(number(row['macro_f1']))} | {pct(number(row['recall_background']))} | "
            f"{pct(number(row['recall_phantom']))} | {pct(number(row['recall_air2s']))} |"
        )
    lines.append("")
    lines.append("## 5. 跨 seed 均值与样本标准差（n=3，仅描述性）")
    lines.append("")
    lines.append("| 数据集 | 条件 | 指标 | seed 值 | 均值 | 样本标准差 |")
    lines.append("|---|---|---|---|---:|---:|")
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['condition']} | {row['metric']} | {row['seed_values']} | "
            f"{100 * number(row['mean']):.2f}% | {100 * number(row['sample_std']):.2f}pp |"
        )
    lines.append("")
    lines.append("## 6. 第一阶段在级联中的行为")
    lines.append("")
    lines.append("| 数据集 | 条件 | seed | 背景误报率 | 无人机漏检率 | 通过第一阶段的真实无人机 | 第二阶段条件准确率 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in metrics:
        lines.append(
            f"| {row['dataset']} | {row['condition']} | {row['training_seed']} | "
            f"{pct(number(row['stage1_background_false_alarm_rate']))} "
            f"({row['stage1_background_false_alarms']} 窗) | "
            f"{pct(number(row['stage1_drone_miss_rate']))} ({row['stage1_drone_misses']} 窗) | "
            f"{row['stage1_drones_passed']} | {pct(number(row['stage2_conditional_accuracy']))} |"
        )
    lines.append("")
    lines.append(
        "“第二阶段条件准确率”只在真实无人机且第一阶段 margin > 0 的窗口上计算，"
        "因此它描述的是第二阶段在级联真实输入上的表现，而不是 oracle 二分类结果。"
    )
    lines.append("")
    lines.append("## 7. 最终错误分解")
    lines.append("")
    lines.append("| 数据集 | 条件 | seed | " + " | ".join(name for _, _, name in ERROR_TRANSITIONS) + " | 总错误 |")
    lines.append("|---|---|---:|" + "---:|" * (len(ERROR_TRANSITIONS) + 1))
    error_lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in errors:
        error_lookup.setdefault(
            (row["dataset"], row["condition"], row["training_seed"]), {}
        )[row["transition"]] = row["count"]
    for row in metrics:
        key = (row["dataset"], row["condition"], row["training_seed"])
        counts = error_lookup.get(key, {})
        lines.append(
            f"| {row['dataset']} | {row['condition']} | {row['training_seed']} | "
            + " | ".join(str(counts.get(name, 0)) for _, _, name in ERROR_TRANSITIONS)
            + f" | {row['error_windows_total']} |"
        )
    lines.append("")
    lines.append("## 8. 逐源文件（支持文件等权主口径）")
    lines.append("")
    lines.append("| 数据集 | 条件 | seed | 源文件 | 真实类 | 窗数 | 正确 | 正确率 | 通过第一阶段 |")
    lines.append("|---|---|---:|---|---|---:|---:|---:|---:|")
    for row in per_file:
        lines.append(
            f"| {row['dataset']} | {row['condition']} | {row['training_seed']} | {row['filename']} | "
            f"{row['true_class_name']} | {row['windows']} | {row['correct']} | "
            f"{pct(number(row['accuracy']))} | {row['stage1_pass']} |"
        )
    lines.append("")
    lines.append("## 9. 一致性核对")
    lines.append("")
    logits = parity["logits"]
    lines.append(
        f"- 第一阶段 nfft512 与既有同协议预测的一致性：比较 {logits.get('compared_rows')} 行，"
        f"最大 |Δlogit_background| = {logits.get('max_abs_logit_background_diff')}，"
        f"最大 |Δlogit_drone| = {logits.get('max_abs_logit_drone_diff')}，"
        f"最大 |Δlogit_margin| = {logits.get('max_abs_logit_margin_diff')}，"
        f"判定不一致数 = {logits.get('prediction_mismatches')}。"
    )
    lines.append(
        f"- 逐窗特征摘要与既有第一阶段 nfft512 摘要文件逐字节一致："
        f"{parity.get('digests_all_identical')}。"
    )
    lines.append(
        "- 第二阶段**不**与既有 oracle 二分类 CSV 比较：oracle 结果的 RNG 行位置是"
        "“过滤后的无人机行号”，级联按完整 split 行号生成噪声，两者本来就不该相同。"
    )
    lines.append(
        f"- 特征摘要文件位于 `feature_digests/`，每个窗口 32 字节；特征缓存清单见 "
        "`prediction_cache_manifest.json`。"
    )
    lines.append(
        "- 冻结输入（配置、两个索引、六个检查点、14 个既有冻结输入）在推理前后各核对一次，"
        "见 `frozen_inputs_verification_before.json` 与 `frozen_inputs_verification.json`。"
    )
    if verification_path.is_file():
        verification = read_json(verification_path)
        lines.append(
            f"- 独立验证器：passed={verification.get('passed')}，"
            f"failures={len(verification.get('failures') or [])}（`verification.json`）。"
        )
    lines.append("")
    lines.append("## 10. 图")
    lines.append("")
    lines.append("- `confusion_<dataset>_<condition>.png`：每个 seed 的三分类混淆矩阵（计数与按真实类占比）。")
    lines.append("- `recall_by_condition.png`：三类文件等权召回率随噪声条件变化（三 seed 均值 ± 样本标准差）。")
    lines.append("- `cascade_flow.png`：串联流程示意（同一张量、seed 配对、两条 margin 规则）。")
    lines.append("- 不生成逐窗 PNG。")
    lines.append("")
    lines.append("## 11. 未做与不能声称的内容")
    lines.append("")
    lines.append("- 未训练、未微调、未改阈值、未改前端、未使用 heldout 做任何选择。")
    lines.append("- 未做噪声方式对照、未加未知类、未做任意信号上传界面、未恢复任何定时器。")
    lines.append(
        "- 不能据此声称跨会话/跨场景/跨设备泛化，也不能把四档人工 AWGN 当作现场绝对 SNR 性能。"
    )
    lines.append(
        "- 三个种子并非独立数据源：同一批窗口、同一份噪声被三个种子共享，"
        "跨 seed 标准差只反映模型初始化与训练顺序的差异。"
    )
    lines.append("")
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"report written ({len(lines)} lines)")
    return 0


def command_all(args: argparse.Namespace) -> int:
    before = command_verify_inputs(argparse.Namespace(output_dir=args.output_dir, stage="before"))
    if before != 0:
        log("frozen inputs failed the pre-inference check; aborting")
        return 1
    command_environment(argparse.Namespace(output_dir=args.output_dir, device=args.device))
    if command_evaluate(args) != 0:
        return 1
    command_summarize(args)
    command_figures(args)
    command_report(args)
    command_manifest(args)
    after = command_verify_inputs(argparse.Namespace(output_dir=args.output_dir, stage="after"))
    command_manifest(args)
    return 0 if after == 0 else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "verify-inputs",
            "environment",
            "evaluate",
            "parity",
            "summarize",
            "figures",
            "report",
            "manifest",
            "all",
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stage", choices=["before", "after"], default="before")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--datasets", default=None, help="comma separated subset of heldout,old_test")
    parser.add_argument("--seeds", default=None, help="comma separated subset of 2026,2027,2028")
    parser.add_argument("--conditions", default=None, help="comma separated subset (bounded runs only)")
    parser.add_argument("--max-windows", type=int, default=None, help="bounded prefix of each split")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir)
    handlers = {
        "verify-inputs": command_verify_inputs,
        "environment": command_environment,
        "evaluate": command_evaluate,
        "parity": command_parity,
        "summarize": command_summarize,
        "figures": command_figures,
        "report": command_report,
        "manifest": command_manifest,
        "all": command_all,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
