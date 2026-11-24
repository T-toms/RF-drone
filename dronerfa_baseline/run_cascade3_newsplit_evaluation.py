"""CASCADE3-NEWSPLIT-C: clean cascade evaluation on the new frozen three-class split.

The two checkpoints accepted in CASCADE3-NEWSPLIT-B (seed 2026) are chained
**without retraining and without any threshold tuning** and evaluated on the six
``development_test_exposed`` source MATs of the frozen 104-file development split:

    stage 1 : Background (T0000) vs Drone (T0010/T0101)
    stage 2 : DJI Phantom 4 Pro (T0010) vs DJI Air 2S (T0101)

Frozen decision rules (identical to CASCADE3 / CASCADE3-NEWSPLIT-B)::

    stage 2 runs iff float32(logit_drone) - float32(logit_background) > 0
    final class = Air 2S  iff float32(logit_air2s) - float32(logit_phantom) > 0
                  Phantom 4 Pro otherwise
    a window whose stage-1 margin is <= 0 is final class Background

Frozen input pipeline (identical to the accepted stage-1/stage-2 protocol):
RF0 I/Q, non-overlapping 1,000,000-sample windows read at the row ``start``,
per-window mean removal + RMS normalisation, Hamming STFT ``n_fft=512`` /
``hop=256``, 256x256 log-power image.  Exactly one feature tensor per window is
built and the **same tensor** is handed to stage 1 and, if stage 1 passes, to
stage 2.  The only condition evaluated here is ``clean`` (no AWGN); the old
multi-level AWGN sweep is deliberately not resumed.

Stepwise commands (``--output-dir`` selects the root, default below)::

    python run_cascade3_newsplit_evaluation.py verify-inputs
    python run_cascade3_newsplit_evaluation.py evaluate [--device auto|cuda|cpu]
    python run_cascade3_newsplit_evaluation.py summarize
    python run_cascade3_newsplit_evaluation.py manifest
    python run_cascade3_newsplit_evaluation.py all

No model is trained here, nothing is written to the G: data drive, and the
accepted CASCADE3-NEWSPLIT-A/B artefacts and the legacy 2026-09-19 experiment
outputs are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
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
# frozen inputs (CASCADE3-NEWSPLIT-A / B, META-SPLIT2)
# ---------------------------------------------------------------------------
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

TRAIN_RUN_ROOT = NEWSPLIT_ROOT / "seed_2026"
STAGE1_CHECKPOINT = TRAIN_RUN_ROOT / "stage1" / "best_model.pt"
STAGE1_SHA256 = "5621c398542fe31b977e1b05deaf089c22b6a53d1f4be12dbb2c74532cd4779e"
STAGE1_EPOCH = 3
STAGE2_CHECKPOINT = TRAIN_RUN_ROOT / "stage2" / "best_model.pt"
STAGE2_SHA256 = "aa25a8ba01d457a2b981b6b29a0cae615aca25af530d7466171ecab47ce68cfd"
STAGE2_EPOCH = 3

OUTPUT_ROOT = PROJECT_DIR / "outputs" / "cascade3_newsplit_eval_20260926" / "seed_2026"

# ---------------------------------------------------------------------------
# frozen evaluation protocol
# ---------------------------------------------------------------------------
N_FFT = 512
HOP_LENGTH = 256
IMAGE_SIZE = 256
BATCH_SIZE = 4
FEATURE_DTYPE = np.float32
INPUT_PROBE_SAMPLES = 4096
TRAINING_SEED = 2026
DATASET_NAME = "newsplit_test"
TEST_ROLE = "development_test_exposed"
CONDITION = "clean"
# only this condition is evaluated in this task; see the module docstring
CONDITIONS: tuple[str, ...] = ("clean",)

WINDOW_SAMPLES = 1_000_000
BACKGROUND_CLASS = 0
CLASS_IDS: tuple[int, ...] = (0, 1, 2)
DRONE_CLASSES: tuple[int, ...] = (1, 2)
CLASS_NAMES: dict[int, str] = {
    0: "Background",
    1: "DJI Phantom 4 Pro",
    2: "DJI Air 2S",
}
TARGET_CODE_TO_CLASS: dict[str, int] = {"T0000": 0, "T0010": 1, "T0101": 2}

EXPECTED_TEST_WINDOWS = 890
EXPECTED_TEST_FILES = 6
EXPECTED_TARGET_COUNTS: dict[str, int] = {"T0000": 280, "T0010": 300, "T0101": 310}

STAGE1_RULE = "run stage 2 iff float32(logit_drone) - float32(logit_background) > 0"
STAGE2_RULE = (
    "Air 2S (class 2) iff float32(logit_air2s) - float32(logit_phantom) > 0, "
    "otherwise Phantom 4 Pro (class 1)"
)
THRESHOLD_RULE = (
    "both margins use the fixed threshold 0.0; no threshold, epoch, hyper-parameter "
    "or condition was selected on the test split"
)
FEATURE_RULE = (
    "one float32 1x256x256 tensor per window; the identical tensor object is handed "
    "to stage 1 and, when stage 1 passes, to stage 2"
)
FEATURE_BATCH_RULE = (
    "test split rows in frozen index order, contiguous chunks of batch_size=4 "
    "(same layout as the accepted nfft512 feature protocol)"
)
STAGE2_BATCH_RULE = (
    "executed rows only, ascending dataset_index_position, contiguous chunks of "
    "batch_size=4; the chunk index is recorded per row as stage2_batch_id"
)

ERROR_TRANSITIONS: tuple[tuple[int, int, str], ...] = (
    (0, 1, "Background->Phantom"),
    (0, 2, "Background->Air2S"),
    (1, 0, "Phantom->Background"),
    (2, 0, "Air2S->Background"),
    (1, 2, "Phantom->Air2S"),
    (2, 1, "Air2S->Phantom"),
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "dataset",
    "condition",
    "training_seed",
    "stage1_epoch",
    "stage2_epoch",
    "dataset_index_position",
    "source_role",
    "row_split",
    "metrics_scope",
    "file_path",
    "filename",
    "target_code",
    "target_name",
    "class_index",
    "class_name",
    "stage1_label",
    "true_class",
    "true_class_name",
    "block_index",
    "window_index",
    "start",
    "length",
    "source_samples",
    "file_sha256",
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
    "shared_feature_tensor",
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
    "files",
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

PER_FILE_COLUMNS: tuple[str, ...] = (
    "filename",
    "target_code",
    "target_name",
    "true_class",
    "windows",
    "correct",
    "accuracy",
    "stage1_pass",
    "stage1_miss",
    "stage1_false_alarm",
    "predicted_background",
    "predicted_phantom",
    "predicted_air2s",
)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(columns) if columns is not None else list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_or_nan(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def feature_digest(array: np.ndarray) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).digest()


def input_probe_payload(tensor: torch.Tensor, samples: int = INPUT_PROBE_SAMPLES) -> bytes:
    array = np.ascontiguousarray(tensor[:, :samples].detach().cpu().numpy())
    return array.tobytes()


# ---------------------------------------------------------------------------
# frozen test rows
# ---------------------------------------------------------------------------
def frozen_test_rows() -> list[dict[str, Any]]:
    """The 890 ``development_test_exposed`` rows in frozen index order.

    ``dataset_index_position`` is the position inside the frozen test split and is
    the primary key of every prediction row.  Every identity claim about the row is
    re-checked here instead of being trusted from the CSV.
    """
    rows: list[dict[str, Any]] = []
    for position, row in enumerate(load_index(INDEX, split=TEST_ROLE)):
        code = str(row["target_code"])
        if code not in TARGET_CODE_TO_CLASS:
            raise ValueError(f"unexpected target_code {code} in {INDEX}")
        true_class = TARGET_CODE_TO_CLASS[code]
        stage1_label = int(row["label"])
        expected_label = 0 if true_class == BACKGROUND_CLASS else 1
        if stage1_label != expected_label:
            raise ValueError(
                f"{row['filename']}: index stage-1 label {stage1_label} != {expected_label}"
            )
        if str(row["source_role"]) != TEST_ROLE or str(row["split"]) != TEST_ROLE:
            raise ValueError(f"{row['filename']}: row is not a {TEST_ROLE} row")
        if str(row["file_path"]).lower().find(str(DATA_ROOT).lower()) != 0:
            raise ValueError(f"{row['filename']}: file_path outside the G data root")
        if int(row["length"]) != WINDOW_SAMPLES:
            raise ValueError(f"{row['filename']}: window length {row['length']}")
        if int(row["start"]) != int(row["window_index"]) * WINDOW_SAMPLES:
            raise ValueError(f"{row['filename']}: start is not window_index * 1M")
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


def test_split_identity(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Composition of the frozen test split (guards against a swapped input)."""
    target_counts = {code: 0 for code in TARGET_CODE_TO_CLASS}
    class_counts = {label: 0 for label in CLASS_IDS}
    files: dict[str, set[str]] = {}
    for row in rows:
        target_counts[str(row["target_code"])] += 1
        class_counts[int(row["true_class"])] += 1
        files.setdefault(str(row["filename"]), set()).add(str(row["true_class"]))
    return {
        "windows": len(rows),
        "files": len(files),
        "target_counts": target_counts,
        "class_counts": class_counts,
        "file_names": sorted(files),
    }


def check_expected_composition(composition: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if int(composition["windows"]) != EXPECTED_TEST_WINDOWS:
        errors.append(f"test windows {composition['windows']} != {EXPECTED_TEST_WINDOWS}")
    if int(composition["files"]) != EXPECTED_TEST_FILES:
        errors.append(f"test files {composition['files']} != {EXPECTED_TEST_FILES}")
    if dict(composition["target_counts"]) != EXPECTED_TARGET_COUNTS:
        errors.append(f"target counts {composition['target_counts']} != {EXPECTED_TARGET_COUNTS}")
    return errors


# ---------------------------------------------------------------------------
# frozen input identity
# ---------------------------------------------------------------------------
def verify_input_identity(stage: str, root: Path | None = None) -> dict[str, Any]:
    """Re-hash every frozen input and re-check the test-split composition."""
    root = root or OUTPUT_ROOT
    record: dict[str, Any] = {
        "stage": stage,
        "checked_at": utc_now(),
        "index": str(INDEX),
        "index_sha256": None,
        "stage2_view_sha256": None,
        "g_manifest_sha256": None,
        "frozen_split_csv_sha256": None,
        "checkpoints": {},
        "composition": None,
        "checks": [],
        "errors": [],
    }
    errors: list[str] = record["errors"]
    checks: list[dict[str, Any]] = record["checks"]

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    index_hash = sha256_file(INDEX)
    record["index_sha256"] = index_hash
    check("index SHA256", index_hash == INDEX_SHA256, index_hash)

    view_hash = sha256_file(STAGE2_VIEW)
    record["stage2_view_sha256"] = view_hash
    check("stage2 view SHA256", view_hash == STAGE2_VIEW_SHA256, view_hash)

    if G_MANIFEST.is_file():
        manifest_hash = sha256_file(G_MANIFEST)
        record["g_manifest_sha256"] = manifest_hash
        check("G manifest SHA256", manifest_hash == G_MANIFEST_SHA256, manifest_hash)
    else:
        check("G manifest reachable", False, str(G_MANIFEST))

    split_hash = sha256_file(FROZEN_SPLIT_CSV)
    record["frozen_split_csv_sha256"] = split_hash
    check("frozen split CSV SHA256", split_hash == FROZEN_SPLIT_CSV_SHA256, split_hash)

    freeze = read_json(FREEZE_JSON)
    check(
        "freeze decision points at the frozen CSV",
        str(freeze.get("frozen_split_csv"))
        == "work/meta_split2/split_candidate_104_20260926.csv"
        and str(freeze.get("frozen_split_csv_sha256")) == FROZEN_SPLIT_CSV_SHA256,
        {
            "frozen_split_csv": freeze.get("frozen_split_csv"),
            "frozen_split_csv_sha256": freeze.get("frozen_split_csv_sha256"),
        },
    )

    for label, path, expected, epoch in (
        ("stage1", STAGE1_CHECKPOINT, STAGE1_SHA256, STAGE1_EPOCH),
        ("stage2", STAGE2_CHECKPOINT, STAGE2_SHA256, STAGE2_EPOCH),
    ):
        if not path.is_file():
            check(f"{label} checkpoint present", False, str(path))
            continue
        digest = sha256_file(path)
        record["checkpoints"][label] = {
            "path": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
            "expected_sha256": expected,
            "expected_epoch": epoch,
        }
        check(f"{label} checkpoint SHA256", digest == expected, digest)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        check(f"{label} checkpoint epoch", int(payload["epoch"]) == epoch, int(payload["epoch"]))
        check(
            f"{label} checkpoint frozen index identity",
            str(payload.get("index_sha256")) == INDEX_SHA256
            and str(payload.get("stage2_view_sha256")) == STAGE2_VIEW_SHA256
            and str(payload.get("split_source_sha256")) == FROZEN_SPLIT_CSV_SHA256,
            {
                "index_sha256": payload.get("index_sha256"),
                "stage2_view_sha256": payload.get("stage2_view_sha256"),
                "split_source_sha256": payload.get("split_source_sha256"),
            },
        )
        check(
            f"{label} checkpoint protocol",
            int(payload.get("n_fft", 0)) == N_FFT
            and int(payload.get("hop_length", 0)) == HOP_LENGTH
            and int(payload.get("training_seed", 0)) == TRAINING_SEED,
            {
                "n_fft": payload.get("n_fft"),
                "hop_length": payload.get("hop_length"),
                "training_seed": payload.get("training_seed"),
            },
        )
        del payload

    rows = frozen_test_rows()
    composition = test_split_identity(rows)
    record["composition"] = composition
    for problem in check_expected_composition(composition):
        check("test split composition", False, problem)
    if not any(item["check"] == "test split composition" for item in checks):
        check("test split composition", True, composition["windows"])
    missing = [row["filename"] for row in rows if not Path(str(row["file_path"])).is_file()]
    check("every test MAT exists on G", not missing, missing[:5])
    bad_root = [
        row["filename"]
        for row in rows
        if not str(row["file_path"]).lower().startswith(str(DATA_ROOT).lower())
    ]
    check("no test path outside the G data root", not bad_root, bad_root[:5])

    record["passed"] = not errors
    if root is not None:
        write_json(root / f"frozen_inputs_verification_{stage}.json", record)
    return record


# ---------------------------------------------------------------------------
# features and logits
# ---------------------------------------------------------------------------
def build_window_dataset(rows: Sequence[dict[str, Any]]) -> DroneRFaWindowDataset:
    """Frozen lazy reader over the whole test split, clean condition (no AWGN).

    ``DroneRFaWindowDataset`` is used unchanged; with ``train_snrs=None`` and
    ``apply_fixed_snr=False`` no noise is drawn, so the window content is a pure
    function of RF0 and the frozen preprocessing.
    """
    dataset = DroneRFaWindowDataset(
        INDEX,
        split=TEST_ROLE,
        train_snrs=None,
        fixed_snr=None,
        apply_fixed_snr=False,
        seed=TRAINING_SEED,
    )
    if len(dataset) != len(rows):
        raise RuntimeError(f"reader exposes {len(dataset)} rows, index has {len(rows)}")
    if dataset.epoch != 0:
        raise RuntimeError("window dataset epoch must stay 0")
    return dataset


def build_features(
    dataset: DroneRFaWindowDataset,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
    frontend: SpectrogramFrontEnd,
) -> dict[str, Any]:
    """One float32 1x256x256 feature tensor per test window, in frozen order."""
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
                tensor, label = dataset[position]
                if int(rows[position]["dataset_index_position"]) != position:
                    raise RuntimeError("dataset row position mismatch")
                expected = 0 if int(rows[position]["true_class"]) == BACKGROUND_CLASS else 1
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
            if (start // BATCH_SIZE + 1) % 50 == 0 or stop == windows:
                elapsed = time.perf_counter() - started
                log(
                    f"    {stop}/{windows} features ({elapsed:.1f}s, "
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


def load_cascade_models(device: torch.device) -> dict[str, Any]:
    """Load the accepted seed-2026 stage-1/stage-2 checkpoints (hash-checked)."""
    models: dict[str, Any] = {}
    for label, path, expected in (
        ("stage1", STAGE1_CHECKPOINT, STAGE1_SHA256),
        ("stage2", STAGE2_CHECKPOINT, STAGE2_SHA256),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{label} checkpoint SHA256 mismatch: {actual}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        frontend = SpectrogramFrontEnd(
            n_fft=N_FFT, hop_length=HOP_LENGTH, image_size=IMAGE_SIZE
        ).to(device)
        frontend.load_state_dict(payload["frontend_state"])
        frontend.eval()
        network = ResNet18(input_channels=1, num_classes=2).to(device)
        network.load_state_dict(payload["model_state"])
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        models[f"{label}_network"] = network
        models[f"{label}_checkpoint"] = str(path)
        models[f"{label}_sha256"] = actual
        models[f"{label}_epoch"] = int(payload["epoch"])
        if label == "stage1":
            models["frontend"] = frontend
        del payload
    return models


# ---------------------------------------------------------------------------
# cascade decisions
# ---------------------------------------------------------------------------
def build_prediction_row(
    row: dict[str, Any],
    seed: int,
    epochs: dict[str, int],
    stage1_logits: Sequence[float],
    stage2_logits: Sequence[float] | None,
    stage2_batch_id: int | None,
    stage1_digest: str,
    stage2_digest: str | None,
) -> dict[str, Any]:
    """Assemble one cascade prediction row and apply the frozen decision rules."""
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
        if stage2_digest != stage1_digest:
            raise ValueError("stage 2 must consume the stage-1 feature tensor")
    else:
        if stage2_logits is not None:
            raise ValueError("stage 2 must not produce logits for a skipped row")
        logit_phantom = logit_air2s = stage2_margin = None
        final = BACKGROUND_CLASS
        stage2_digest = None
    return {
        "dataset": DATASET_NAME,
        "condition": CONDITION,
        "training_seed": int(seed),
        "stage1_epoch": int(epochs["stage1_epoch"]),
        "stage2_epoch": int(epochs["stage2_epoch"]),
        "dataset_index_position": int(row["dataset_index_position"]),
        "source_role": row["source_role"],
        "row_split": row["split"],
        "metrics_scope": row.get("metrics_scope", "development"),
        "file_path": row["file_path"],
        "filename": row["filename"],
        "target_code": row["target_code"],
        "target_name": row["target_name"],
        "class_index": int(row["class_index"]),
        "class_name": row["class_name"],
        "stage1_label": int(row["stage1_label"]),
        "true_class": int(row["true_class"]),
        "true_class_name": row["true_class_name"],
        "block_index": int(row["block_index"]),
        "window_index": int(row["window_index"]),
        "start": int(row["start"]),
        "length": int(row["length"]),
        "source_samples": int(row["source_samples"]),
        "file_sha256": row["file_sha256"],
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
        "shared_feature_tensor": True,
        "stage1_feature_digest": stage1_digest,
        "stage2_feature_digest": stage2_digest,
    }


def evaluate_split(
    models: dict[str, Any], device: torch.device, root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cascade the frozen clean test split (890 windows) with the seed-2026 pair."""
    rows = frozen_test_rows()
    composition = test_split_identity(rows)
    problems = check_expected_composition(composition)
    if problems:
        raise RuntimeError(f"unexpected test split composition: {problems}")
    epochs = {
        "stage1_epoch": int(models["stage1_epoch"]),
        "stage2_epoch": int(models["stage2_epoch"]),
    }
    windows = len(rows)
    prediction_rows: list[dict[str, Any]] = []
    cache_metadata: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        if condition != CONDITION:
            raise RuntimeError("this task only evaluates the clean condition")
        log(f"  features {DATASET_NAME}/{condition}: {windows} windows, nfft{N_FFT}/hop{HOP_LENGTH}")
        dataset = build_window_dataset(rows)
        built = build_features(dataset, rows, device, models["frontend"])
        features = built["features"]
        metadata = {
            "dataset": DATASET_NAME,
            "split": TEST_ROLE,
            "condition": condition,
            "fixed_snr_db": None,
            "awgn_applied": False,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "image_size": IMAGE_SIZE,
            "feature_rule": FEATURE_RULE,
            "feature_batch_rule": FEATURE_BATCH_RULE,
            "windows": windows,
            "shape": [windows, 1, IMAGE_SIZE, IMAGE_SIZE],
            "dtype": "float32",
            "feature_sha256": sha256_bytes(features.tobytes()),
            "window_digest_bytes": len(built["window_digests"]),
            "window_digests_concat_sha256": sha256_bytes(built["window_digests"]),
            "input_probe_samples": INPUT_PROBE_SAMPLES,
            "input_probe_sha256": built["input_probe_sha256"],
            "seconds": built["seconds"],
        }
        cache_metadata.append(metadata)
        digest_path = root / "feature_digests" / f"window_digests_{DATASET_NAME}_{condition}.bin"
        digest_path.parent.mkdir(parents=True, exist_ok=True)
        digest_path.write_bytes(built["window_digests"])
        metadata["window_digest_file"] = str(digest_path.relative_to(root))
        metadata["window_digest_file_sha256"] = sha256_file(digest_path)
        log(
            f"    feature_sha256={metadata['feature_sha256']} "
            f"digests={metadata['window_digests_concat_sha256']} ({metadata['seconds']:.1f}s)"
        )

        stage1 = network_logits(models["stage1_network"], features, device)
        margins = (stage1[:, 1] - stage1[:, 0]).astype(np.float32)
        selection = margins > 0.0
        stage2_logits = np.full((windows, 2), np.nan, dtype=np.float32)
        stage2_digests: dict[int, str] = {}
        batch_id_by_position: dict[int, int] = {}
        positions_all = [int(index) for index in np.nonzero(selection)[0]]
        batches = [
            positions_all[start : start + BATCH_SIZE]
            for start in range(0, len(positions_all), BATCH_SIZE)
        ]
        with torch.inference_mode():
            for batch_id, positions in enumerate(batches):
                batch_array = np.ascontiguousarray(features[positions])
                batch = torch.from_numpy(batch_array).to(device)
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda",
                    dtype=torch.float16,
                ):
                    logits = models["stage2_network"](batch)
                logits = logits.float().cpu().numpy()
                if not np.isfinite(logits).all():
                    raise RuntimeError("non-finite stage-2 logits")
                stage2_logits[positions] = logits
                for offset, position in enumerate(positions):
                    stage2_digests[int(position)] = sha256_bytes(
                        np.ascontiguousarray(batch_array[offset]).tobytes()
                    )
                    batch_id_by_position[int(position)] = batch_id
        executed = int(selection.sum())
        log(
            f"    seed {TRAINING_SEED} {DATASET_NAME}/{condition}: stage-1 drone "
            f"{executed}/{windows}, stage-2 batches {len(batches)}"
        )
        for position, row in enumerate(rows):
            digest = feature_digest(features[position]).hex()
            record = build_prediction_row(
                row,
                TRAINING_SEED,
                epochs,
                stage1[position],
                stage2_logits[position] if bool(selection[position]) else None,
                batch_id_by_position.get(position),
                digest,
                stage2_digests.get(position),
            )
            prediction_rows.append(record)
        del features
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(prediction_rows) != windows:
        raise RuntimeError(f"{len(prediction_rows)} prediction rows for {windows} windows")
    return prediction_rows, {"windows": windows, "conditions": cache_metadata}


# ---------------------------------------------------------------------------
# metrics (recomputed from the prediction rows; the verifier re-implements them)
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
    result: dict[str, Any] = {
        "windows": total,
        "correct": int(np.trace(matrix)),
        "pooled_accuracy": float(np.trace(matrix)) / float(total),
        "macro_recall": mean_or_nan(recalls),
        "balanced_accuracy": mean_or_nan(recalls),
        "macro_precision": mean_or_nan(precisions),
        "macro_f1": mean_or_nan(f1_scores),
    }
    for label, name in (("background", 0), ("phantom", 1), ("air2s", 2)):
        result[f"recall_{label}"] = recalls[name]
        result[f"precision_{label}"] = precisions[name]
        result[f"f1_{label}"] = f1_scores[name]
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
                "target_code": str(record["target_code"]),
                "target_name": str(record["target_name"]),
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
        stage1_pass = int(entry["stage1_pass"])
        is_drone = int(entry["true_class"]) in DRONE_CLASSES
        per_file_rows.append(
            {
                "filename": filename,
                "target_code": entry["target_code"],
                "target_name": entry["target_name"],
                "true_class": int(entry["true_class"]),
                "windows": int(entry["windows"]),
                "correct": int(entry["correct"]),
                "accuracy": rate,
                "stage1_pass": stage1_pass,
                "stage1_miss": int(entry["windows"]) - stage1_pass if is_drone else None,
                "stage1_false_alarm": stage1_pass if not is_drone else None,
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
        "files": len(per_file_rows),
        "files_background": len(class_rates[0]),
        "files_phantom": len(class_rates[1]),
        "files_air2s": len(class_rates[2]),
        "file_equal_background_recall": class_rate[0],
        "file_equal_phantom_recall": class_rate[1],
        "file_equal_air2s_recall": class_rate[2],
        "file_equal_accuracy": mean_or_nan(available),
        "file_equal_classes_used": len(available),
        "per_file": per_file_rows,
    }


def stage1_rates(records: Sequence[dict[str, Any]], per_file: Sequence[dict[str, Any]]) -> dict[str, Any]:
    background = [record for record in records if int(record["true_class"]) == BACKGROUND_CLASS]
    drones = [record for record in records if int(record["true_class"]) in DRONE_CLASSES]
    false_alarms = sum(1 for record in background if bool(record["stage2_executed"]))
    misses = sum(1 for record in drones if not bool(record["stage2_executed"]))
    background_files = [row for row in per_file if int(row["true_class"]) == BACKGROUND_CLASS]
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


def metric_record(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    matrix = confusion_matrix(records)
    file_equal = file_equal_metrics(records)
    per_file = file_equal.pop("per_file")
    row: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "condition": CONDITION,
        "training_seed": TRAINING_SEED,
        "stage1_epoch": STAGE1_EPOCH,
        "stage2_epoch": STAGE2_EPOCH,
    }
    row.update({key: value for key, value in pooled_metrics(matrix).items() if key != "correct"})
    row.update(file_equal)
    row.update(stage1_rates(records, per_file))
    row.update(stage2_conditional_metrics(records))
    row.update(error_decomposition(records))
    return row


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def command_verify_inputs(args: argparse.Namespace) -> int:
    root = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT
    record = verify_input_identity(args.stage, root)
    for item in record["checks"]:
        log(f"  [{'ok' if item['passed'] else 'FAIL'}] {item['check']}: {item['detail']}")
    log(f"input identity: {'PASS' if record['passed'] else 'FAIL'} ({len(record['checks'])} checks)")
    return 0 if record["passed"] else 1


def command_evaluate(args: argparse.Namespace) -> int:
    root = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    before = verify_input_identity("before", root)
    if not before["passed"]:
        log("frozen inputs failed verification; refusing to evaluate")
        return 1
    device = resolve_device(args.device)
    log(f"device: {device}")
    models = load_cascade_models(device)
    predictions, cache = evaluate_split(models, device, root)
    write_csv(root / "window_predictions.csv", predictions, PREDICTION_COLUMNS)
    write_json(root / "feature_cache_metadata.json", cache)
    write_json(
        root / "input_identity.json",
        {
            "phase": "CASCADE3-NEWSPLIT-C",
            "written_at": utc_now(),
            "index": str(INDEX),
            "index_sha256": INDEX_SHA256,
            "stage2_view": str(STAGE2_VIEW),
            "stage2_view_sha256": STAGE2_VIEW_SHA256,
            "frozen_split_csv": str(FROZEN_SPLIT_CSV),
            "frozen_split_csv_sha256": FROZEN_SPLIT_CSV_SHA256,
            "g_manifest": str(G_MANIFEST),
            "g_manifest_sha256": G_MANIFEST_SHA256,
            "stage1_checkpoint": str(STAGE1_CHECKPOINT),
            "stage1_sha256": STAGE1_SHA256,
            "stage1_epoch": STAGE1_EPOCH,
            "stage2_checkpoint": str(STAGE2_CHECKPOINT),
            "stage2_sha256": STAGE2_SHA256,
            "stage2_epoch": STAGE2_EPOCH,
            "training_seed": TRAINING_SEED,
            "test_role": TEST_ROLE,
            "conditions": list(CONDITIONS),
            "feature_rule": FEATURE_RULE,
            "stage1_rule": STAGE1_RULE,
            "stage2_rule": STAGE2_RULE,
            "threshold_rule": THRESHOLD_RULE,
            "evaluation_script": str(Path(__file__).resolve()),
            "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
        },
    )
    write_json(
        root / "environment.json",
        {
            "written_at": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
            ),
        },
    )
    after = verify_input_identity("after", root)
    summary = {
        "phase": "CASCADE3-NEWSPLIT-C",
        "condition": CONDITION,
        "windows": len(predictions),
        "stage1_drone_windows": sum(1 for row in predictions if row["stage2_executed"]),
        "inputs_verified_before": before["passed"],
        "inputs_verified_after": after["passed"],
        "seconds": time.perf_counter() - started,
        "completed_at": utc_now(),
    }
    write_json(root / "evaluation_run.json", summary)
    log(
        f"evaluate: {summary['windows']} windows, stage-1 drone "
        f"{summary['stage1_drone_windows']}, {summary['seconds']:.1f}s"
    )
    return 0


def parse_prediction_rows(path: Path) -> list[dict[str, Any]]:
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
        for key in (
            "stage1_logit_background",
            "stage1_logit_drone",
            "stage1_margin",
            "stage2_logit_phantom",
            "stage2_logit_air2s",
            "stage2_margin",
        ):
            row[key] = float(raw[key]) if raw[key] not in (None, "") else None
        row["stage2_executed"] = str(raw["stage2_executed"]).strip().lower() == "true"
        row["shared_feature_tensor"] = str(raw["shared_feature_tensor"]).strip().lower() == "true"
        row["stage2_batch_id"] = (
            int(raw["stage2_batch_id"]) if raw["stage2_batch_id"] not in (None, "") else None
        )
        rows.append(row)
    if not rows:
        raise ValueError(f"no prediction rows in {path}")
    return rows


def command_summarize(args: argparse.Namespace) -> int:
    root = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT
    predictions = parse_prediction_rows(root / "window_predictions.csv")
    records = [row for row in predictions if row["condition"] == CONDITION]
    metrics = metric_record(records)
    matrix = confusion_matrix(records)
    file_equal = file_equal_metrics(records)
    per_file_rows = file_equal.pop("per_file")
    write_csv(root / "per_file_metrics.csv", per_file_rows, PER_FILE_COLUMNS)
    write_csv(root / "metrics_by_condition.csv", [metrics], METRIC_COLUMNS)
    write_json(root / "metrics_by_condition.json", {CONDITION: metrics, "per_file": per_file_rows})
    with (root / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *[CLASS_NAMES[label] for label in CLASS_IDS]])
        for label in CLASS_IDS:
            writer.writerow([CLASS_NAMES[label], *[int(value) for value in matrix[label]]])
    write_json(
        root / "confusion_matrix.json",
        {
            "labels": [CLASS_NAMES[label] for label in CLASS_IDS],
            "matrix": [[int(value) for value in row] for row in matrix],
        },
    )
    decomposition = error_decomposition(records)
    write_csv(
        root / "error_decomposition.csv",
        [
            {
                "transition": name,
                "count": int(matrix[truth, prediction]),
                "rate_in_true_class": decomposition[f"rate_{name.replace('->', '_to_').lower()}"],
            }
            for truth, prediction, name in ERROR_TRANSITIONS
        ],
        ["transition", "count", "rate_in_true_class"],
    )
    write_json(root / "error_decomposition.json", decomposition)
    write_json(
        root / "stage_decomposition.json",
        {
            "stage1": {
                "rule": STAGE1_RULE,
                "false_alarms": metrics["stage1_background_false_alarms"],
                "false_alarm_rate": metrics["stage1_background_false_alarm_rate"],
                "false_alarm_rate_file_equal": metrics[
                    "stage1_background_false_alarm_rate_file_equal"
                ],
                "drone_misses": metrics["stage1_drone_misses"],
                "drone_miss_rate": metrics["stage1_drone_miss_rate"],
                "drone_miss_rate_file_equal": metrics["stage1_drone_miss_rate_file_equal"],
                "drones_passed": metrics["stage1_drones_passed"],
            },
            "stage2": {
                "rule": STAGE2_RULE,
                "conditional_windows": metrics["stage2_conditional_windows"],
                "conditional_correct": metrics["stage2_conditional_correct"],
                "conditional_accuracy": metrics["stage2_conditional_accuracy"],
                "errors": {
                    "phantom_to_air2s": metrics["error_phantom_to_air2s"],
                    "air2s_to_phantom": metrics["error_air2s_to_phantom"],
                },
            },
        },
    )
    write_json(
        root / "metrics_summary.json",
        {
            "phase": "CASCADE3-NEWSPLIT-C",
            "dataset": DATASET_NAME,
            "condition": CONDITION,
            "primary_metric": "file_equal_accuracy",
            "primary_metric_note": (
                "per source file accuracy averaged with equal weight per class; the six "
                "development_test_exposed files were previously used by the legacy "
                "experiments and are a descriptive control, not a fresh final test"
            ),
            "thresholds": {"stage1": 0.0, "stage2": 0.0, "rule": THRESHOLD_RULE},
            "metrics": metrics,
            "confusion_matrix": [[int(value) for value in row] for row in matrix],
            "summarized_at": utc_now(),
        },
    )
    log(
        f"summarize: file-equal BA {metrics['file_equal_accuracy']:.6f}, pooled "
        f"{metrics['pooled_accuracy']:.6f}, stage-1 miss {metrics['stage1_drone_misses']}, "
        f"stage-2 errors {metrics['error_phantom_to_air2s'] + metrics['error_air2s_to_phantom']}"
    )
    return 0


def command_manifest(args: argparse.Namespace) -> int:
    root = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT
    artifacts: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "experiment_manifest.json":
            artifacts[str(path.relative_to(root))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    write_json(
        root / "experiment_manifest.json",
        {
            "phase": "CASCADE3-NEWSPLIT-C",
            "written_at": utc_now(),
            "output_root": str(root),
            "frozen_inputs": {
                "index": {"path": str(INDEX), "sha256": INDEX_SHA256},
                "stage2_view": {"path": str(STAGE2_VIEW), "sha256": STAGE2_VIEW_SHA256},
                "frozen_split_csv": {
                    "path": str(FROZEN_SPLIT_CSV),
                    "sha256": FROZEN_SPLIT_CSV_SHA256,
                },
                "g_manifest": {"path": str(G_MANIFEST), "sha256": G_MANIFEST_SHA256},
                "stage1_checkpoint": {
                    "path": str(STAGE1_CHECKPOINT),
                    "sha256": STAGE1_SHA256,
                    "epoch": STAGE1_EPOCH,
                },
                "stage2_checkpoint": {
                    "path": str(STAGE2_CHECKPOINT),
                    "sha256": STAGE2_SHA256,
                    "epoch": STAGE2_EPOCH,
                },
            },
            "protocol": {
                "training_seed": TRAINING_SEED,
                "test_role": TEST_ROLE,
                "conditions": list(CONDITIONS),
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "feature_rule": FEATURE_RULE,
                "stage1_rule": STAGE1_RULE,
                "stage2_rule": STAGE2_RULE,
                "threshold_rule": THRESHOLD_RULE,
            },
            "artifacts": artifacts,
        },
    )
    log(f"manifest: {len(artifacts)} artifacts recorded")
    return 0


def command_all(args: argparse.Namespace) -> int:
    code = command_evaluate(args)
    if code != 0:
        return code
    code = command_summarize(args)
    if code != 0:
        return code
    return command_manifest(args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("verify-inputs", "evaluate", "summarize", "manifest", "all"),
        help="step to run",
    )
    parser.add_argument("--output-dir", default=None, help="override the output root")
    parser.add_argument(
        "--stage",
        default="before",
        choices=("before", "after"),
        help="label of the verify-inputs record",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    handlers = {
        "verify-inputs": command_verify_inputs,
        "evaluate": command_evaluate,
        "summarize": command_summarize,
        "manifest": command_manifest,
        "all": command_all,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
