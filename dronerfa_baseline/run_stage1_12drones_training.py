#!/usr/bin/env python3
"""STAGE1-12DRONES: stage-1 Background vs Drone training on the frozen 104-MAT split.

``STAGE1-12DRONES-A`` ran this protocol at training seed 2026 and was accepted;
``STAGE1-12DRONES-B`` re-runs the *identical* protocol at training seed 2027 to
measure how much a single training seed moves the frozen validation metrics.
``--training-seed`` switches the run; it defaults to the accepted 2026 seed so
old invocations keep their exact semantics.

Frozen protocol (identical to the accepted CASCADE3-NEWSPLIT-B stage 1; only the
label space and the source-file split are new):

* RF0 I/Q, non-overlapping 1,000,000-sample windows read at the row ``start``;
* per-window mean removal + RMS normalisation (``dronerfa_data.rms_normalize``);
* training noise ``A_original``: complex AWGN at
  ``clean,10,5,0,-5,-10,-12`` dB drawn per window with the frozen rule
  ``default_rng(seed + epoch * len(split) + position)``, where ``seed`` is the
  requested training seed;
* Hamming STFT ``n_fft=512``, ``hop=256``, 256x256 log-power image
  (``model.SpectrogramFrontEnd``, unchanged);
* ``model.ResNet18(input_channels=1, num_classes=2)``, Adam lr=1e-3, wd=0,
  batch size 4, class-weighted cross entropy, AMP float16, 5 epochs;
* the requested training seed drives weight initialisation (``seed_everything``),
  the per-epoch shuffle (``training_seed + epoch``), the DataLoader generator
  (``training_seed + 50_000``) and the per-window training noise
  (``training_seed + epoch * len(split) + position``);
* stage 1 = Background (T0000) = 0 vs Drone (the twelve other target codes) = 1;
* epoch selection uses only the ``val`` files under the four frozen validation
  conditions (clean/-10/-12/-15 dB), maximising the mean pooled balanced
  accuracy; the earliest epoch wins an exact tie.  The validation noise seed is
  pinned at 12026 for *every* training seed so two seeds see byte-identical
  validation inputs and stay paired-comparable.

Every window inherits the role of its source MAT, and the 26
``development_test`` / ``development_test_exposed`` files are never opened: the
read audit fails the run if any of their paths appears among the opened files.

Nothing here evaluates a test role, and nothing here is stage 2.
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
from typing import Any, Callable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / "outputs" / ".matplotlib"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dronerfa_data import DroneRFaWindowDataset, parse_snr_list
from model import ResNet18, SpectrogramFrontEnd
from stage1_loader_optimization import (
    CANDIDATE_NUM_WORKERS,
    REFERENCE_NUM_WORKERS,
    REFERENCE_PREFETCH_FACTOR,
    VAL_PHASE,
    ConsumptionLog,
    LoaderAudit,
    LoaderConfig,
    audit_from_parallel,
    build_cache_loader,
    build_train_loader,
    consumed_batches,
    identity_digest,
    limited_order,
    loader_from_args,
    order_for_epoch,
    tensor_digest,
)
from train_resnet18 import binary_metrics, extract_features, seed_everything

# ---------------------------------------------------------------------------
# frozen inputs of STAGE1-12DRONES-A
# ---------------------------------------------------------------------------
OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "stage1_12drones_20260926"
INDEX = OUTPUT_ROOT / "window_index_stage1_12drones.csv"
INDEX_SHA256 = "5973a2fa5b8476a507fcc7a665365daf8d572f569ce5908d54f5c4bc84c48bd6"
INDEX_ROWS = 15790
SUMMARY_JSON = OUTPUT_ROOT / "dataset_summary.json"

WORK_DIR = PROJECT_ROOT / "work"
FREEZE_JSON = WORK_DIR / "meta_split2" / "CODEX-FREEZE-20260926.json"
FROZEN_SPLIT_CSV = WORK_DIR / "meta_split2" / "split_candidate_104_20260926.csv"
FROZEN_SPLIT_CSV_SHA256 = (
    "d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9"
)
G_MANIFEST = Path(r"G:\DroneRFa数据集\dataset_manifest.csv")
G_MANIFEST_SHA256 = "9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d"
DATA_ROOT = Path(r"G:\DroneRFa数据集\original")

RUN_ROOT_SUFFIX = "seed_{seed}"
SMOKE_ROOT = OUTPUT_ROOT / "smoke"
LOG_NAME = "stage1_train.log"

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
#: The accepted STAGE1-12DRONES-A seed.  ``--training-seed`` overrides the run
#: seed; keeping this the default preserves the accepted invocation semantics.
DEFAULT_TRAINING_SEED = 2026
#: Seeds already trained on this frozen split, with the phase that produced them.
ACCEPTED_TRAINING_SEEDS: dict[int, str] = {2026: "STAGE1-12DRONES-A"}
#: Rebound to the requested seed by :func:`bind_training_seed` *before* any helper
#: that reads it runs.  Module global (rather than a threaded parameter) so the
#: accepted seed-2026 numerics stay untouched.
TRAINING_SEED = DEFAULT_TRAINING_SEED
TRAIN_SNRS = "clean,10,5,0,-5,-10,-12"
#: Never parameterised: the validation noise seed is shared by every training
#: seed, which is what makes two seeds comparable window for window.
VALIDATION_SEED = 12026
CONDITION_VALUES: dict[str, float | None] = {
    "clean": None,
    "-10dB": -10.0,
    "-12dB": -12.0,
    "-15dB": -15.0,
}
CONDITIONS: tuple[str, ...] = ("clean", "-10dB", "-12dB", "-15dB")
TIE_BREAK = "earliest_epoch"
SELECTION_KEY = "mean_balanced_accuracy"
SELECTION_RULE = (
    "mean_pooled_balanced_accuracy_over_four_validation_conditions"
)
PREDICTION_RULE = "float32(logit_drone) - float32(logit_background) > 0"
WINDOW_SAMPLES = 1_000_000
#: Bookkeeping columns kept for schema parity with the legacy three-class index.
WINDOWS_PER_BLOCK = 10
RELOAD_TOLERANCE = 1e-9

BACKGROUND_CODE = "T0000"
LABEL_BACKGROUND = 0
LABEL_DRONE = 1
LABEL_MAPPING = {"0": "Background", "1": "Drone"}

TRAIN_ROLE = "train"
VAL_ROLE = "val"
TEST_ROLES = ("development_test", "development_test_exposed")
EXPECTED_ROLES = (TRAIN_ROLE, VAL_ROLE, *TEST_ROLES)

SOURCE_FILES = (
    "build_stage1_12drones_index.py",
    "run_stage1_12drones_training.py",
    "stage1_loader_optimization.py",
    "dronerfa_data.py",
    "model.py",
    "train_resnet18.py",
)

# Expected composition of the frozen index (guards against a swapped input).
EXPECTED_TOTALS: dict[str, Any] = {
    "windows": {
        TRAIN_ROLE: 9590,
        VAL_ROLE: 2360,
        "development_test": 2950,
        "development_test_exposed": 890,
    },
    "files": {
        TRAIN_ROLE: 62,
        VAL_ROLE: 16,
        "development_test": 20,
        "development_test_exposed": 6,
    },
    "stage1_label_counts": {
        TRAIN_ROLE: {0: 600, 1: 8990},
        VAL_ROLE: {0: 280, 1: 2080},
    },
}

#: Compact per-batch progress interval (2380+ training batches per epoch).
PROGRESS_EVERY = 100


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_root_for(seed: int) -> Path:
    """``outputs/stage1_12drones_20260926/seed_<seed>`` (one directory per seed)."""
    return OUTPUT_ROOT / RUN_ROOT_SUFFIX.format(seed=int(seed))


def phase_for(seed: int) -> str:
    """Phase label of a seed: an accepted seed keeps its original phase name."""
    seed = int(seed)
    return ACCEPTED_TRAINING_SEEDS.get(seed, f"STAGE1-12DRONES-seed{seed}")


def bind_training_seed(seed: int) -> int:
    """Point the protocol at one training seed and return it.

    Must run before any helper that derives a path or a seed from
    ``TRAINING_SEED``.  Validation keeps ``VALIDATION_SEED`` regardless.
    """
    global TRAINING_SEED
    TRAINING_SEED = int(seed)
    return TRAINING_SEED


class Tee:
    """Append everything written through ``log`` to a file as well as stdout."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def write(self, message: str) -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


_TEE: Tee | None = None


def log(message: str) -> None:
    if _TEE is None:
        print(message, flush=True)
    else:
        _TEE.write(message)


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
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


# ---------------------------------------------------------------------------
# epoch-boundary state (STAGE1-LOSS-NORM): save / restore hooks
# ---------------------------------------------------------------------------
#: The state schema is a *new* artefact of the loss-normalisation run; the accepted
#: seed-2026/seed-2027 checkpoints are never rewritten or resumed.
TRAINING_STATE_SCHEMA = "dronerfa-stage1-training-state/v1"
#: Identity keys a restored state must agree on before an epoch is skipped.
RESUME_IDENTITY_KEYS = (
    "index_sha256",
    "split_source_sha256",
    "training_seed",
    "validation_seed",
    "epochs",
)


def rng_state_payload() -> dict[str, Any]:
    """Python/NumPy/torch-CPU/torch-CUDA RNG states, in one picklable dict."""

    import random

    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng_state(payload: dict[str, Any]) -> dict[str, Any]:
    """Restore what :func:`rng_state_payload` saved; returns what was restored.

    Both generators take a **CPU** ``uint8`` state tensor, so a state payload loaded
    with ``map_location="cpu"`` is used as-is; moving the CUDA state to the GPU would
    make ``set_rng_state_all`` reject it.
    """

    import random

    restored: list[str] = []
    if "python" in payload:
        random.setstate(payload["python"])
        restored.append("python")
    if "numpy" in payload:
        np.random.set_state(payload["numpy"])
        restored.append("numpy")
    if "torch_cpu" in payload:
        torch.set_rng_state(
            torch.as_tensor(payload["torch_cpu"], dtype=torch.uint8).cpu()
        )
        restored.append("torch_cpu")
    if "torch_cuda" in payload and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [
                torch.as_tensor(state, dtype=torch.uint8).cpu()
                for state in payload["torch_cuda"]
            ]
        )
        restored.append("torch_cuda")
    return {
        "restored": restored,
        "note": (
            "the frozen protocol's per-epoch order and per-window noise come from "
            "explicit generators (torch.Generator().manual_seed(seed+epoch) and "
            "numpy.random.default_rng(seed + epoch*len(split) + position)); the "
            "global RNG states are saved and restored for completeness"
        ),
    }


def resume_identity_errors(
    state: dict[str, Any],
    *,
    index_sha256: str,
    split_source_sha256: str,
    training_seed: int,
    validation_seed: int,
    epochs: int,
) -> list[str]:
    """Every reason the saved state must not be resumed into this invocation."""

    errors: list[str] = []
    if state.get("schema") != TRAINING_STATE_SCHEMA:
        errors.append(f"state schema {state.get('schema')!r} != {TRAINING_STATE_SCHEMA!r}")
    identity = state.get("identity") or {}
    expected = {
        "index_sha256": index_sha256,
        "split_source_sha256": split_source_sha256,
        "training_seed": int(training_seed),
        "validation_seed": int(validation_seed),
        "epochs": int(epochs),
    }
    for key in RESUME_IDENTITY_KEYS:
        if identity.get(key) != expected[key]:
            errors.append(f"state identity {key}={identity.get(key)!r} != {expected[key]!r}")
    completed = int(state.get("epoch") or 0)
    if completed < 1:
        errors.append(f"state has no completed epoch (epoch={state.get('epoch')!r})")
    if completed >= int(epochs):
        errors.append(f"state already completed epoch {completed} of {epochs}")
    if int(state.get("next_epoch") or 0) != completed + 1:
        errors.append(
            f"state next_epoch={state.get('next_epoch')!r} != {completed + 1}"
        )
    return errors


def apply_training_state(
    state: dict[str, Any],
    *,
    model: nn.Module,
    frontend: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    expected_initial_hash: str,
) -> dict[str, Any]:
    """Load a saved epoch-boundary state into freshly built modules.

    The freshly built model must hash to the same initial value the saved run
    started from, which is what proves the resume used the same initialisation.
    """

    saved_initial = str(state.get("initial_model_sha256", ""))
    if saved_initial and saved_initial != expected_initial_hash:
        raise RuntimeError(
            "resume initialisation mismatch: saved "
            f"{saved_initial} != rebuilt {expected_initial_hash}"
        )
    model.load_state_dict(state["model_state"])
    frontend.load_state_dict(state["frontend_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if state.get("scaler_state") is not None:
        scaler.load_state_dict(state["scaler_state"])
    rng = restore_rng_state(state.get("rng") or {})
    return {
        "resumed_from_epoch": int(state["epoch"]),
        "next_epoch": int(state["next_epoch"]),
        "initial_model_sha256": saved_initial,
        "restored_model_sha256": state_hash(model.state_dict()),
        "rng": rng,
    }


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


def stage1_label_of(target_code: str) -> int:
    return LABEL_BACKGROUND if target_code == BACKGROUND_CODE else LABEL_DRONE


def check_index_identity() -> dict[str, Any]:
    """Re-hash the frozen index and re-check every compositional claim."""
    errors: list[str] = []
    index_hash = sha256_file(INDEX)
    if index_hash != INDEX_SHA256:
        errors.append(f"index SHA256 mismatch: {index_hash}")
    if G_MANIFEST.is_file():
        manifest_hash = sha256_file(G_MANIFEST)
        if manifest_hash != G_MANIFEST_SHA256:
            errors.append(f"G dataset manifest SHA256 mismatch: {manifest_hash}")
    else:
        manifest_hash = None
        errors.append(f"G dataset manifest not reachable: {G_MANIFEST}")

    roles = frozen_role_map()
    rows = read_csv_rows(INDEX)
    if len(rows) != INDEX_ROWS:
        errors.append(f"index row count {len(rows)} != {INDEX_ROWS}")
    for row in rows:
        row["_window_index"] = int(row["window_index"])
        row["_start"] = int(row["start"])
        row["_length"] = int(row["length"])
        row["_label"] = int(row["label"])
        row["_stage1_label"] = int(row["stage1_label"])

    role_of_file: dict[str, str] = {}
    label_errors: list[str] = []
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
        if (
            int(row["window_in_block"]) != row["_window_index"] % WINDOWS_PER_BLOCK
            or int(row["block_index"]) != row["_window_index"] // WINDOWS_PER_BLOCK
        ):
            errors.append(f"{filename}: window_in_block/block_index disagree with window_index")
            break
        expected_label = stage1_label_of(str(row["target_code"]))
        if row["_label"] != expected_label or row["_stage1_label"] != expected_label:
            label_errors.append(filename)
    if label_errors:
        errors.append(f"stage-1 label violations: {sorted(set(label_errors))[:5]}")

    counts: dict[str, Counter] = {role: Counter() for role in EXPECTED_ROLES}
    files: dict[str, set[str]] = {role: set() for role in EXPECTED_ROLES}
    paths: dict[str, set[str]] = {role: set() for role in EXPECTED_ROLES}
    label_counts: dict[str, Counter] = {role: Counter() for role in EXPECTED_ROLES}
    target_counts: dict[str, Counter] = {role: Counter() for role in EXPECTED_ROLES}
    for row in rows:
        role = str(row["source_role"])
        counts[role][1] += 1
        files[role].add(str(row["filename"]))
        paths[role].add(str(row["file_path"]))
        label_counts[role][int(row["label"])] += 1
        target_counts[role][str(row["target_code"])] += 1

    observed = {
        "windows": {role: counts[role][1] for role in EXPECTED_ROLES},
        "files": {role: len(files[role]) for role in EXPECTED_ROLES},
        "stage1_label_counts": {
            role: dict(sorted(label_counts[role].items())) for role in (TRAIN_ROLE, VAL_ROLE)
        },
        "background_target_codes": sorted(
            {
                str(row["target_code"])
                for row in rows
                if int(row["label"]) == LABEL_BACKGROUND
            }
        ),
        "drone_target_codes": sorted(
            {str(row["target_code"]) for row in rows if int(row["label"]) == LABEL_DRONE}
        ),
    }
    for key, expected in EXPECTED_TOTALS.items():
        if observed[key] != expected:
            errors.append(f"{key} mismatch: {observed[key]} != {expected}")
    if observed["background_target_codes"] != [BACKGROUND_CODE]:
        errors.append(
            f"background label is not exclusive to {BACKGROUND_CODE}: "
            f"{observed['background_target_codes']}"
        )
    if len(observed["drone_target_codes"]) != 12 or BACKGROUND_CODE in observed["drone_target_codes"]:
        errors.append(f"drone label does not cover exactly 12 target codes: {observed['drone_target_codes']}")

    overlap = {
        "train_val": sorted(files[TRAIN_ROLE] & files[VAL_ROLE]),
        "train_test": sorted(
            files[TRAIN_ROLE] & (files["development_test"] | files["development_test_exposed"])
        ),
        "val_test": sorted(
            files[VAL_ROLE] & (files["development_test"] | files["development_test_exposed"])
        ),
    }
    if any(overlap.values()):
        errors.append(f"source files cross roles: {overlap}")

    return {
        "index": {"path": str(INDEX), "sha256": index_hash, "rows": len(rows)},
        "g_manifest": {"path": str(G_MANIFEST), "sha256": manifest_hash},
        "frozen_split_csv": {"path": str(FROZEN_SPLIT_CSV), "sha256": FROZEN_SPLIT_CSV_SHA256},
        "files_by_role": {role: sorted(files[role]) for role in EXPECTED_ROLES},
        "paths_by_role": {role: sorted(paths[role]) for role in EXPECTED_ROLES},
        "observed": observed,
        "role_overlap": overlap,
        "errors": errors,
        "passed": not errors,
    }


def group_file_balance(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-source-file class balance of a split (diagnostic, not a selection key)."""
    per_file: dict[str, Counter] = {}
    for row in rows:
        per_file.setdefault(str(row["filename"]), Counter())[int(row["label"])] += 1
    single_class = sum(1 for counter in per_file.values() if len(counter) == 1)
    return {
        "files": len(per_file),
        "files_with_both_labels": len(per_file) - single_class,
        "files_with_one_label": single_class,
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


class AuditedWindowDataset(DroneRFaWindowDataset):
    """Stage-1 reader: one label per window, audited file access."""

    def __init__(self, *args: Any, read_audit: ReadAudit, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._read_audit = read_audit

    def _file(self, path: str):
        self._read_audit.record_open(str(path))
        return super()._file(path)

    def __getitem__(self, index: int):
        value = super().__getitem__(index)
        self._read_audit.record_window(str(self.rows[index]["file_path"]))
        return value


def select_positions(
    rows: Sequence[dict[str, Any]], max_samples: int | None
) -> list[int]:
    """Underlying split positions of the bounded-view selection, in selection order.

    ``stratified_limit`` returns references into ``rows``, so the bounded smoke
    selects the very same windows as the accepted protocol and the loader is then
    handed *positions into the split*, never slots of a view.  A row table without
    duplicate objects keeps the mapping exact.
    """
    from dronerfa_data import stratified_limit

    if max_samples is not None and max_samples > 0:
        selected = stratified_limit(list(rows), max_samples)
        positions = {id(row): index for index, row in enumerate(rows)}
        return [positions[id(row)] for row in selected]
    return list(range(len(rows)))


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
    """Equal weight per source file inside each class, then the two classes equally.

    This is the metric that exposes how concentrated a stage-1 split is: with
    twelve drone classes against one background class the pooled metric is
    dominated by drone windows, so both numbers are reported.
    """
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
    for entry in per_file.values():
        class_rates[entry["label"]].append(entry["correct"] / max(entry["windows"], 1))
    if not class_rates[0] or not class_rates[1]:
        raise ValueError("file-equal metrics need both classes in the split")
    background = float(np.mean(class_rates[0]))
    drone = float(np.mean(class_rates[1]))
    return {
        "files_background": len(class_rates[0]),
        "files_drone": len(class_rates[1]),
        "file_equal_background_recall": background,
        "file_equal_drone_recall": drone,
        "file_equal_balanced_accuracy": 0.5 * (background + drone),
    }


def select_epoch(history: Sequence[dict[str, Any]], key: str) -> int:
    """Highest mean validation score; the earliest epoch wins an exact tie."""
    return int(max(history, key=lambda row: (row[key], -int(row["epoch"])))["epoch"])


# ---------------------------------------------------------------------------
# validation caches
# ---------------------------------------------------------------------------
def _cache_dataset(
    condition: str,
    read_audit: ReadAudit,
    max_val_samples: int | None,
    loader_config: LoaderConfig,
    loader_audit: LoaderAudit | None = None,
    pin_memory: bool = False,
) -> tuple[DataLoader, list[dict[str, Any]], list[int], AuditedWindowDataset]:
    """One validation-condition loader over the selected windows, in selection order.

    The loader is always built by the optimisation module (also for
    ``num_workers=0``), so the validation cache carries the same
    ``(phase, key, position)`` identity evidence as the training epochs instead of
    only a parent-side file tally.
    """
    dataset = AuditedWindowDataset(
        INDEX,
        split=VAL_ROLE,
        fixed_snr=CONDITION_VALUES[condition],
        apply_fixed_snr=CONDITION_VALUES[condition] is not None,
        seed=VALIDATION_SEED,
        read_audit=read_audit,
    )
    if dataset.epoch != 0:
        raise RuntimeError("validation dataset epoch must stay 0")
    indices = select_positions(dataset.rows, max_val_samples)
    selected = [
        {**dataset.rows[position], "index_position": int(position)} for position in indices
    ]
    if loader_audit is not None:
        loader_audit.register_loader(VAL_ROLE, condition, loader_config)
    loader = build_cache_loader(
        dataset,
        loader_config,
        positions=indices,
        batch_size=BATCH_SIZE,
        phase=VAL_ROLE,
        key=condition,
        audit_events=(loader_audit.events if loader_audit is not None else None),
        pin_memory=pin_memory,
    )
    return loader, selected, indices, dataset


def build_validation_cache(
    device: torch.device,
    read_audit: ReadAudit,
    max_val_samples: int | None = None,
    loader_config: LoaderConfig | None = None,
    loader_audit: LoaderAudit | None = None,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]], list[dict[str, Any]]]:
    frontend = SpectrogramFrontEnd(
        n_fft=N_FFT, hop_length=HOP_LENGTH, image_size=IMAGE_SIZE
    ).to(device)
    frontend.eval()
    config = loader_config if loader_config is not None else LoaderConfig()
    cached: dict[str, tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]] = {}
    metadata: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        cache_source, selected, cache_indices, cache_dataset = _cache_dataset(
            condition,
            read_audit,
            max_val_samples,
            config,
            loader_audit,
            pin_memory=device.type == "cuda",
        )
        consumption = ConsumptionLog(
            phase=VAL_ROLE,
            key=condition,
            on_batch=(loader_audit.drain if loader_audit is not None else None),
        )
        loader = cache_source
        total_batches = len(loader)
        tensors: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        started = time.perf_counter()
        with torch.inference_mode():
            for batch, (signals, targets) in enumerate(
                consumed_batches(loader, cache_indices, consumption), 1
            ):
                features = extract_features(frontend, signals.to(device))
                if not bool(torch.isfinite(features).all()):
                    raise RuntimeError(f"non-finite validation features in {condition}")
                tensors.append(features.cpu())
                labels.append(targets)
                if batch % 100 == 0 or batch == total_batches:
                    log(f"  val cache {condition} {batch}/{total_batches}")
        consumption_record: dict[str, Any] | None = None
        if loader_audit is not None:
            consumption_record = loader_audit.record_consumption(consumption)
            if consumption.aborted or consumption.batches_completed != total_batches:
                raise RuntimeError(
                    f"validation cache {condition} did not complete: "
                    f"{consumption.batches_completed}/{total_batches} batches"
                )
        xx, yy = torch.cat(tensors), torch.cat(labels)
        cached[condition] = (xx, yy, selected)
        metadata.append(
            {
                "split": VAL_ROLE,
                "stage": 1,
                "condition": condition,
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "noise_seed": VALIDATION_SEED,
                "samples": len(selected),
                "index_positions": list(cache_indices),
                "shape": [len(selected), 1, IMAGE_SIZE, IMAGE_SIZE],
                "dtype": "float32",
                "feature_sha256": sha256_bytes(xx.numpy().tobytes()),
                "label_sha256": sha256_bytes(yy.numpy().astype(np.int64).tobytes()),
                "files": sorted({str(row["filename"]) for row in selected}),
                "seconds": time.perf_counter() - started,
                "loader": config.payload(),
                "consumption": consumption_record,
            }
        )
        if config.parallel:
            del loader  # shut the workers down before the next condition
        cache_dataset.close()
        if loader_audit is not None:
            # Worker shutdown flushes its events; collect them per condition, before
            # the next condition's workers start, so no event is orphaned in the queue.
            loader_audit.close_loader(VAL_ROLE, condition)
    del frontend
    if loader_audit is not None:
        loader_audit.drain()
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
                "stage": 1,
                "condition": condition,
                "loss": total_loss / max(len(rows), 1),
                "samples": len(rows),
                **pooled_binary_metrics(window_records),
                **file_equal_metrics(window_records),
            }
        )
        records_all.extend(window_records)
    return metrics_all, records_all


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
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "logical_cores": os.cpu_count(),
        "pythonpath_override": os.environ.get("PYTHONPATH"),
        "source_sha256": {name: sha256_file(SCRIPT_DIR / name) for name in SOURCE_FILES},
        "recorded_at": utc_now(),
    }


def train_one_epoch(
    frontend: SpectrogramFrontEnd,
    model: ResNet18,
    dataset: Any,
    positions: Sequence[int],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    loader_config: LoaderConfig | None = None,
    consumption: ConsumptionLog | None = None,
    audit_events: Any = None,
    epoch: int = 0,
    observer: Callable[[dict[str, Any]], None] | None = None,
    legacy_subset_loader: bool = False,
) -> tuple[float, dict[str, Any]]:
    """One full pass over an explicit *underlying position* order.

    ``positions`` are indices into the split's own row table, so the sampler, the
    read event and the consumption entry share one coordinate system.  The loader
    is then a plain ``sampler=positions`` DataLoader and the serial
    (``num_workers=0``) and parallel paths differ only in the ``DataLoader``
    arguments, never in the maths, the batch order or the noise.

    ``legacy_subset_loader`` rebuilds the historical
    ``Subset(dataset, positions)`` + ``num_workers=0`` loader; it exists so the
    rework can prove that this refactor did not change the serial training maths,
    and is never used by a real run.

    ``observer`` receives one record after every completed optimisation step
    (batch identity, input digests, loss, parameter/BN/scheduler state).  It is a
    pure measurement hook: nothing it does feeds back into training.
    """

    import hashlib

    model.train()
    frontend.eval()
    config = loader_config if loader_config is not None else LoaderConfig()
    parallel = config.parallel
    if legacy_subset_loader:
        loader = DataLoader(
            torch.utils.data.Subset(dataset, [int(position) for position in positions]),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=REFERENCE_NUM_WORKERS,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(TRAINING_SEED + 50_000),
        )
    else:
        loader = build_train_loader(
            dataset,
            config,
            positions=positions,
            key=str(epoch),
            audit_events=audit_events,
            pin_memory=device.type == "cuda",
        )
    delivered = [int(position) for position in positions]
    total_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    probabilities_all: list[float] = []
    started = time.perf_counter()
    batches: Any = (
        consumed_batches(loader, delivered, consumption)
        if consumption is not None
        else loader
    )
    cursor = 0
    for batch_index, (signals, labels) in enumerate(batches, start=1):
        batch_positions = delivered[cursor : cursor + int(labels.shape[0])]
        cursor += int(labels.shape[0])
        # Pure measurement hook: digests are taken on the CPU batch the loader
        # produced, before it is moved or consumed, so they cannot be perturbed by
        # the training step and cannot perturb it either.
        batch_evidence: dict[str, Any] | None = None
        if observer is not None:
            batch_evidence = {
                "positions": batch_positions,
                "signals_sha256": tensor_digest(signals),
                "labels_sha256": tensor_digest(labels.reshape(-1).to(torch.int64)),
                "signals_dtype": str(signals.dtype),
                "labels": [int(value) for value in labels.tolist()],
            }
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        features = extract_features(frontend, signals)
        with torch.amp.autocast(
            device_type=device.type, enabled=amp_enabled, dtype=torch.float16
        ):
            logits = model(features)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
        predictions = logits.detach().argmax(dim=1)
        batch_size = labels.shape[0]
        total_loss += float(loss.detach()) * batch_size
        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())
        probabilities_all.extend(probabilities.cpu().tolist())

        if observer is not None and batch_evidence is not None:
            digest = hashlib.sha256()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    digest.update(parameter.grad.detach().float().cpu().numpy().tobytes())
            observer(
                {
                    "phase": (consumption.phase if consumption is not None else ""),
                    "key": (consumption.key if consumption is not None else ""),
                    "batch": batch_index,
                    "loss": float(loss.detach()),
                    "scaler_scale": float(scaler.get_scale()),
                    "metrics": binary_metrics(
                        labels.cpu().tolist(),
                        predictions.cpu().tolist(),
                        probabilities.cpu().tolist(),
                    ),
                    "parameter_sha256": state_hash(model.state_dict()),
                    "gradient_sha256": digest.hexdigest(),
                    **batch_evidence,
                }
            )

        if batch_index % PROGRESS_EVERY == 0 or batch_index == len(loader):
            elapsed = time.perf_counter() - started
            rate = len(labels_all) / max(elapsed, 1e-9)
            log(
                f"    train batch {batch_index:5d}/{len(loader):5d} "
                f"loss={total_loss / max(len(labels_all), 1):.4f} rate={rate:.1f} samples/s"
            )
    if parallel:
        del loader  # shut the workers down before the parent reads the audit
    return (
        total_loss / max(len(labels_all), 1),
        binary_metrics(labels_all, predictions_all, probabilities_all),
    )


def train_stage1(
    device: torch.device,
    epochs: int,
    output_dir: Path,
    max_train_samples: int | None,
    max_val_samples: int | None,
    mode: str,
    pre_run: dict[str, Any],
    loader_config: LoaderConfig | None = None,
    *,
    train_criterion: nn.Module | None = None,
    eval_criterion: nn.Module | None = None,
    extra_config: dict[str, Any] | None = None,
    epoch_callback: Callable[..., None] | None = None,
    start_epoch: int = 1,
    initial_state: dict[str, Any] | None = None,
    resume_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the frozen stage-1 protocol.

    The five keyword-only arguments were added for ``STAGE1-LOSS-NORM`` and are all
    inert by default, so an invocation that does not pass them executes exactly the
    code path the accepted seed-2026/seed-2027 runs executed:

    ``train_criterion`` / ``eval_criterion``
        objective used for the optimisation step / for the recorded validation loss.
        Both default to the historical class-weighted cross entropy; the validation
        loss definition must stay the original one whichever objective trains.
    ``extra_config``
        additive ``run_config.json`` keys (collision with an existing key is refused).
    ``epoch_callback``
        called once per completed epoch with the live modules, so a driver can
        persist an epoch-boundary state without the protocol owning that format.
    ``start_epoch`` / ``initial_state`` / ``resume_record``
        epoch-boundary resume: history, order hashes and loader records continue from
        the saved state, and the saved modules/RNG are restored before the loop.
    """
    if initial_state is None and (output_dir / "best_model.pt").is_file():
        raise FileExistsError(
            f"refusing to overwrite the existing run at {output_dir / 'best_model.pt'}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    amp_enabled = device.type == "cuda"
    audit = ReadAudit("stage1")
    active_loader = loader_config if loader_config is not None else LoaderConfig()
    parallel_loader = active_loader.parallel
    # The identity audit runs for both paths: with workers the parent performs no
    # reads, and with the serial reference loader the parent-side ReadAudit cannot
    # say *which* epoch or condition a read belonged to.  A spawn queue is only
    # needed when the readers really are separate processes.
    if parallel_loader:
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        events: Any = context.Queue()
    else:
        import queue as queue_module

        events = queue_module.Queue()
    loader_audit = LoaderAudit("stage1_parallel" if parallel_loader else "stage1_serial")
    loader_audit.set_events(events)
    log(
        f"[stage1] 12-drone Background-vs-Drone training, seed {TRAINING_SEED} -> {output_dir} "
        f"(loader num_workers={active_loader.num_workers}, "
        f"prefetch_factor={active_loader.effective_prefetch_factor}, "
        f"torch_threads={torch.get_num_threads()})"
    )

    cached, cache_metadata = build_validation_cache(
        device, audit, max_val_samples, active_loader, loader_audit
    )
    cache_phase_audit: dict[str, Any] = loader_audit.phase_report(VAL_PHASE)
    log(
        f"  loader cache phase: files={cache_phase_audit['opened_file_count']} "
        f"windows={cache_phase_audit['windows_read']} "
        f"consumed={cache_phase_audit['windows_completed']} "
        f"unconsumed_reads={cache_phase_audit['unconsumed_reads']} "
        f"workers={cache_phase_audit['worker_keys']}"
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

    dataset = AuditedWindowDataset(
        INDEX,
        split=TRAIN_ROLE,
        train_snrs=parse_snr_list(TRAIN_SNRS),
        seed=TRAINING_SEED,
        read_audit=audit,
    )
    train_positions = select_positions(dataset.rows, max_train_samples)

    counts = {
        label: sum(1 for row in dataset.rows if int(row["label"]) == label)
        for label in (LABEL_BACKGROUND, LABEL_DRONE)
    }
    if not all(counts.values()):
        raise ValueError(f"both training classes are required, got {counts}")
    class_weights = torch.tensor(
        [len(dataset) / (2 * counts[label]) for label in (LABEL_BACKGROUND, LABEL_DRONE)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # The objective that trains and the objective whose loss is *recorded* on the
    # validation cache are separate handles: STAGE1-LOSS-NORM trains a different
    # normalisation while the recorded validation loss keeps the original definition.
    training_criterion = train_criterion if train_criterion is not None else criterion
    validation_criterion = eval_criterion if eval_criterion is not None else criterion
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    resume_info: dict[str, Any] | None = None
    if initial_state is not None:
        resume_info = apply_training_state(
            initial_state,
            model=model,
            frontend=frontend,
            optimizer=optimizer,
            scaler=scaler,
            expected_initial_hash=initial_hash,
        )

    val_rows = [row for row in read_csv_rows(INDEX) if str(row["split"]) == VAL_ROLE]
    config: dict[str, Any] = {
        "phase": phase_for(TRAINING_SEED),
        "stage": "stage1",
        "task": "Background (T0000) vs Drone (the twelve other target codes)",
        "task_variant": (
            "STAGE1-12DRONES-A accepted protocol re-run at an independent training seed"
            if TRAINING_SEED != DEFAULT_TRAINING_SEED
            else "STAGE1-12DRONES-A accepted protocol"
        ),
        "mode": mode,
        "run_root_suffix": RUN_ROOT_SUFFIX.format(seed=TRAINING_SEED),
        "protocol_identical_across_seeds": (
            "only the training seed differs; validation noise seed stays "
            f"{VALIDATION_SEED}"
        ),
        "split_source": "work/meta_split2/split_candidate_104_20260926.csv",
        "split_source_sha256": FROZEN_SPLIT_CSV_SHA256,
        "g_manifest": str(G_MANIFEST),
        "g_manifest_sha256": G_MANIFEST_SHA256,
        "index": str(INDEX),
        "index_sha256": INDEX_SHA256,
        "index_rows": INDEX_ROWS,
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
        "optimizer": {
            "type": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "loss": "class-weighted cross entropy (weight = n_windows / (2 * n_class_windows))",
        "train_snrs": TRAIN_SNRS,
        "train_files": len({str(row["filename"]) for row in dataset.rows}),
        "train_windows_available": len(dataset),
        "train_windows_used_per_epoch": len(train_positions),
        "train_class_counts": counts,
        "class_weights": class_weights.detach().cpu().tolist(),
        "train_balance": group_file_balance(dataset.rows),
        "val_balance": group_file_balance(val_rows),
        "initial_model_sha256": initial_hash,
        "validation_split": VAL_ROLE,
        "validation_seed": VALIDATION_SEED,
        "validation_conditions": list(CONDITIONS),
        "validation_files": len({str(row["filename"]) for row in val_rows}),
        "validation_windows_available": len(val_rows),
        "validation_windows_used_per_epoch": len(cache_metadata[0]["index_positions"]),
        "selection": SELECTION_RULE,
        "selection_key": SELECTION_KEY,
        "tie_break": TIE_BREAK,
        "prediction": PREDICTION_RULE,
        "label_mapping": LABEL_MAPPING,
        "frontend": {
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "image_size": IMAGE_SIZE,
            "window": "torch.hamming_window(n_fft) periodic",
            "implementation": "model.SpectrogramFrontEnd (frozen)",
        },
        "amp": amp_enabled,
        "amp_dtype": "float16",
        "loader": {
            **active_loader.payload(),
            "identity": "(phase, key, underlying_split_position)",
            "parent_torch_threads": torch.get_num_threads(),
            "parent_torch_interop_threads": torch.get_num_interop_threads(),
            "reference_serial_path": not parallel_loader,
        },
        "data_loader_optimization": (
            "stage1_loader_optimization.LoaderConfig"
            if parallel_loader
            else "reference loader (num_workers=0) through the same audited code path"
        ),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_threads_requested": pre_run.get("torch_threads_requested"),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "smoke": mode == "smoke",
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "started_at": started_at,
        "source_sha256": pre_run["source_sha256"],
        "test_roles": list(TEST_ROLES),
        "test_windows_never_read": None,
        "status": "running",
    }
    if extra_config:
        collisions = sorted(set(extra_config) & set(config))
        if collisions:
            raise ValueError(
                f"extra_config would overwrite frozen run_config keys: {collisions}"
            )
        config.update(extra_config)
    if resume_record is not None:
        config["resume"] = resume_record
        config["resume_restore"] = resume_info
        config["start_epoch"] = int(start_epoch)
    write_json(output_dir / "run_config.json", config)
    write_json(output_dir / "read_audit.json", audit.payload())

    history: list[dict[str, Any]] = []
    order_hashes: list[str] = []
    identity_order_hashes: list[str] = []
    epoch_loader_records: list[dict[str, Any]] = []
    resumed_epochs = 0
    if initial_state is not None:
        history = [dict(row) for row in initial_state.get("history", [])]
        order_hashes = [str(value) for value in initial_state.get("order_hashes", [])]
        identity_order_hashes = [
            str(value) for value in initial_state.get("identity_order_hashes", [])
        ]
        epoch_loader_records = [
            dict(row) for row in initial_state.get("epoch_loader_records", [])
        ]
        resumed_epochs = len(history)
    checkpoint_path = output_dir / "best_model.pt"
    try:
        for epoch in range(int(start_epoch), epochs + 1):
            epoch_started = time.perf_counter()
            dataset.set_epoch(epoch)
            # The frozen shuffle is always over the *full* split length; a bounded
            # run then keeps the selected positions, so the noise rule and the
            # surviving order match the accepted protocol exactly.
            order = limited_order(
                order_for_epoch(dataset, epoch, TRAINING_SEED), train_positions
            )
            order_hashes.append(sha256_bytes(np.asarray(order, dtype=np.int64).tobytes()))
            identity_order_hashes.append(
                identity_digest(
                    [
                        (str(dataset.rows[position]["file_path"]), int(position))
                        for position in order
                    ]
                )
            )
            loader_audit.register_loader(TRAIN_ROLE, str(epoch), active_loader)
            # Non-persistent workers: rebuild after set_epoch so every worker sees
            # the new epoch, which the frozen noise rule depends on.
            consumption = ConsumptionLog(
                phase=TRAIN_ROLE,
                key=str(epoch),
                # Drain while the epoch runs: a prefetching loader produces worker
                # events far faster than one queue drain at the end could hold.
                on_batch=loader_audit.drain,
            )
            log(f"  [stage1] epoch {epoch}/{epochs}")
            train_loss, train_metrics = train_one_epoch(
                frontend,
                model,
                dataset,
                order,
                training_criterion,
                optimizer,
                scaler,
                device,
                amp_enabled,
                active_loader,
                consumption,
                events,
                epoch=epoch,
            )
            epoch_loader_records.append(loader_audit.record_consumption(consumption))
            if consumption.aborted or consumption.batches_completed != consumption.batches_delivered:
                raise RuntimeError(
                    f"epoch {epoch} did not complete every delivered batch: "
                    f"{consumption.batches_completed}/{consumption.batches_delivered}"
                )
            loader_audit.close_loader(TRAIN_ROLE, str(epoch))
            record = loader_audit.key_report(TRAIN_ROLE, str(epoch))
            log(
                f"    loader: read {record['reads']['windows_read']} windows, "
                f"completed {record['consumed']['windows_completed']} "
                f"({record['consumed']['batches_completed']}/"
                f"{record['consumed']['batches_delivered']} batches), "
                f"unconsumed_reads={record['unconsumed_reads']['count']}, "
                f"unread_consumed={record['unread_consumed']['count']}"
            )
            metrics, records = evaluate_cached(
                model, cached, validation_criterion, device, amp_enabled
            )
            mean_score = float(np.mean([row["balanced_accuracy"] for row in metrics]))
            mean_file_equal = float(
                np.mean([row["file_equal_balanced_accuracy"] for row in metrics])
            )
            clean_row = next(row for row in metrics if row["condition"] == "clean")
            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "train_f1": train_metrics["f1"],
                "train_drone_recall": train_metrics["recall"],
                "clean_balanced_accuracy": clean_row["balanced_accuracy"],
                "mean_balanced_accuracy": mean_score,
                "mean_file_equal_balanced_accuracy": mean_file_equal,
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
                    "image_size": IMAGE_SIZE,
                    "validation_seed": VALIDATION_SEED,
                    "val_metrics": metrics,
                    "selection_rule": SELECTION_RULE,
                    "selection_key": SELECTION_KEY,
                    "tie_break": TIE_BREAK,
                    "prediction_rule": PREDICTION_RULE,
                    "label_mapping": LABEL_MAPPING,
                    "training_seed": TRAINING_SEED,
                    "initial_model_sha256": initial_hash,
                    "index_sha256": INDEX_SHA256,
                    "split_source_sha256": FROZEN_SPLIT_CSV_SHA256,
                    "run_config": config,
                },
                output_dir / f"epoch_{epoch:02d}.pt",
            )
            selected_epoch = select_epoch(history, SELECTION_KEY)
            selection = {
                "stage": "stage1",
                "n_fft": N_FFT,
                "training_seed": TRAINING_SEED,
                "rule": SELECTION_RULE,
                "key": SELECTION_KEY,
                "tie_break": TIE_BREAK,
                "validation_split": VAL_ROLE,
                "conditions": list(CONDITIONS),
                "mean_balanced_accuracy_by_epoch": {
                    str(row["epoch"]): row[SELECTION_KEY] for row in history
                },
                "selected_epoch": selected_epoch,
                "selected_epoch_score": next(
                    row[SELECTION_KEY] for row in history if row["epoch"] == selected_epoch
                ),
            }
            write_json(output_dir / "selection.json", selection)
            if selected_epoch == epoch:
                shutil.copyfile(output_dir / f"epoch_{epoch:02d}.pt", checkpoint_path)
            write_json(
                output_dir / "progress.json",
                {
                    "stage": "stage1",
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
            if epoch_callback is not None:
                # Pure hook for a driver-owned artefact (epoch-boundary state).  It
                # runs after every frozen artefact of this epoch is on disk and
                # before the next epoch starts; it cannot change the maths.
                epoch_callback(
                    epoch=epoch,
                    model=model,
                    frontend=frontend,
                    optimizer=optimizer,
                    scaler=scaler,
                    history=history,
                    order_hashes=order_hashes,
                    identity_order_hashes=identity_order_hashes,
                    epoch_loader_records=epoch_loader_records,
                    metrics=metrics,
                    selection=selection,
                    config=config,
                    initial_model_sha256=initial_hash,
                    resumed_epochs=resumed_epochs,
                )
            log(
                f"  epoch {epoch}: mean pooled BA={mean_score:.6f} "
                f"mean file-equal BA={mean_file_equal:.6f} "
                f"({history_row['seconds']:.1f}s) selected={selected_epoch}"
            )
            for metric in metrics:
                log(
                    f"    {metric['condition']}: BA={metric['balanced_accuracy']:.6f} "
                    f"file-equal BA={metric['file_equal_balanced_accuracy']:.6f} "
                    f"tp={metric['tp']} tn={metric['tn']} fp={metric['fp']} fn={metric['fn']}"
                )
    finally:
        dataset.close()

    selected = select_epoch(history, SELECTION_KEY)
    config["epoch_order_sha256"] = order_hashes
    config["epoch_identity_order_sha256"] = identity_order_hashes
    config["epoch_order_coordinate"] = "underlying split position"
    config["epoch_loader_records"] = epoch_loader_records
    config["selected_epoch"] = selected
    config["selected_epoch_score"] = next(
        row[SELECTION_KEY] for row in history if row["epoch"] == selected
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
            "stage": "stage1",
            "status": "completed",
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "selected_epoch": selected,
            "updated_at": utc_now(),
        },
    )
    log(
        f"  [stage1] kept {config['retained_epoch_checkpoints']}, "
        f"removed {len(config['removed_epoch_checkpoints'])} non-selected checkpoints"
    )

    # The read audit must never contain a development_test* source file.
    test_paths = set(pre_run["identity"]["paths_by_role"]["development_test"]) | set(
        pre_run["identity"]["paths_by_role"]["development_test_exposed"]
    )
    train_val_paths = set(pre_run["identity"]["paths_by_role"][TRAIN_ROLE]) | set(
        pre_run["identity"]["paths_by_role"][VAL_ROLE]
    )
    audit_payload = audit.payload()
    audit_payload["test_files_touched"] = sorted(set(audit.opened_files) & test_paths)
    audit_payload["unexpected_files_touched"] = sorted(
        set(audit.opened_files) - train_val_paths
    )
    audit_payload["opened_all_train_val"] = train_val_paths <= set(audit.opened_files)
    audit_payload["train_val_files_expected"] = len(train_val_paths)
    audit_payload["passed"] = not (
        audit_payload["test_files_touched"] or audit_payload["unexpected_files_touched"]
    )
    write_json(output_dir / "read_audit.json", audit_payload)
    if not audit_payload["passed"]:
        raise RuntimeError(f"read audit failed for stage1: {audit_payload}")
    config["test_windows_never_read"] = True
    config["read_audit_windows_read"] = audit_payload["windows_read"]
    config["read_audit_opened_files"] = audit_payload["opened_file_count"]
    config["read_audit_passed"] = True
    config["read_audit_opened_all_train_val"] = audit_payload["opened_all_train_val"]
    write_json(output_dir / "run_config.json", config)

    if loader_audit is not None:
        loader_audit.drain()
        loader_payload = {
            "schema": "dronerfa-stage1-loader-audit/v2",
            "stage": "stage1",
            "loader": active_loader.payload(),
            "training_seed": TRAINING_SEED,
            "validation_seed": VALIDATION_SEED,
            "conditions": list(CONDITIONS),
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
            "train": audit_from_parallel(loader_audit, TRAIN_ROLE),
            "val_cache": cache_phase_audit,
            "audit": loader_audit.payload(),
            "serial_reference_audit": {
                "windows_read": audit_payload["windows_read"],
                "opened_file_count": audit_payload["opened_file_count"],
                "note": (
                    "with workers the parent process performs no HDF5 reads, so the "
                    "parent-side ReadAudit is 0; the worker events above are then the "
                    "only real evidence and the verifier checks them directly"
                ),
            },
            "worker_processes_seen": sorted(
                {tally.pid for tally in loader_audit.tallies.values() if tally.pid > 0}
            ),
            "epoch_loader_records": epoch_loader_records,
            # Only the epochs that ran in *this* process are audited here; a resumed
            # segment reports its predecessor separately instead of pretending the
            # reads happened here.
            "expected_train_windows": len(train_positions)
            * (len(history) - resumed_epochs),
            "resumed_epochs": resumed_epochs,
            "resume": resume_record,
        }
        touched = (
            set(loader_payload["train"]["opened_files"])
            | set(loader_payload["val_cache"]["opened_files"])
        )
        loader_payload["test_files_touched"] = sorted(touched & test_paths)
        loader_payload["unexpected_files_touched"] = sorted(touched - train_val_paths)
        loader_payload["passed"] = not (
            loader_payload["test_files_touched"] or loader_payload["unexpected_files_touched"]
        )
        write_json(output_dir / "loader_read_audit.json", loader_payload)
        consumption_log = loader_audit.write_consumption_csv(
            output_dir / "loader_consumption.csv"
        )
        config["loader_consumption_log"] = consumption_log
        if not loader_payload["passed"]:
            raise RuntimeError(f"loader identity audit failed: {loader_payload}")
        config["parallel_loader_audit_passed"] = bool(parallel_loader)
        config["loader_audit_passed"] = True
        config["loader_audit_schema"] = "dronerfa-stage1-loader-audit/v2"
        config["parallel_loader_worker_pids"] = loader_payload["worker_processes_seen"]
        write_json(output_dir / "run_config.json", config)

    reload_check = verify_checkpoint_reload(
        output_dir, checkpoint_path, cached, validation_criterion, device, amp_enabled
    )
    write_json(output_dir / "checkpoint_reload_check.json", reload_check)
    if not reload_check["passed"]:
        raise RuntimeError("checkpoint reload check failed for stage1")

    del model, optimizer, scaler
    cached.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "stage": "stage1",
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
    rows.extend(
        {"epoch": epoch, **{key: str(value) for key, value in row.items()}} for row in metrics
    )
    return rows


def verify_checkpoint_reload(
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
    metrics, _ = evaluate_cached(model, cached, criterion, device, amp_enabled)
    recorded = payload["val_metrics"]
    comparisons: list[dict[str, Any]] = []
    max_float_delta = 0.0
    for expected, actual in zip(recorded, metrics):
        if expected["condition"] != actual["condition"]:
            raise RuntimeError("validation condition order changed on reload")
        for key in ("tn", "fp", "fn", "tp"):
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
        for key in ("balanced_accuracy", "file_equal_balanced_accuracy", "accuracy"):
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
        "stage": "stage1",
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
def status_report(run_root: Path | None = None) -> dict[str, Any]:
    """Progress of the requested seed's run (plus the shared smoke directory)."""
    primary = Path(run_root) if run_root is not None else run_root_for(TRAINING_SEED)
    report: dict[str, Any] = {
        "training_seed": TRAINING_SEED,
        "validation_seed": VALIDATION_SEED,
        "run_root": str(primary),
        "smoke_root": str(SMOKE_ROOT),
        "runs": {},
    }
    roots = (primary,) if primary == SMOKE_ROOT else (primary, SMOKE_ROOT)
    for root in roots:
        entry: dict[str, Any] = {"directory": str(root), "exists": root.is_dir()}
        if entry["exists"]:
            progress = root / "progress.json"
            config = root / "run_config.json"
            checkpoint = root / "best_model.pt"
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
            entry["epoch_checkpoints"] = sorted(path.name for path in root.glob("epoch_*.pt"))
        report["runs"][root.name] = entry
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STAGE1-12DRONES: stage-1 Background vs Drone training on the frozen 104-MAT split."
    )
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--training-seed",
        type=int,
        default=DEFAULT_TRAINING_SEED,
        help=(
            "training random seed (default: the accepted STAGE1-12DRONES-A seed "
            f"{DEFAULT_TRAINING_SEED}); selects outputs/.../seed_<seed>/ and leaves "
            f"the validation noise seed pinned at {VALIDATION_SEED}"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="override the output directory (default: seed-<training-seed> directory)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=REFERENCE_NUM_WORKERS,
        help=(
            "DataLoader worker processes (default: 0 = the reference serial loader; "
            f"candidate values {list(CANDIDATE_NUM_WORKERS)}). Windows spawn, non-persistent, "
            "one lazy HDF5 handle set per worker"
        ),
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help=(
            "batches prefetched per worker (default: 2 when --num-workers > 0, ignored "
            "for the serial loader)"
        ),
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help=(
            "torch CPU threads for the parent process (default: None = torch's own "
            "default, i.e. the historical behaviour). With --num-workers > 0 the "
            "default oversubscribes this 16-core host: the parent takes 10 threads "
            "while every worker also decompresses gzip chunks. The value actually in "
            "force is recorded in environment.json / run_config.json"
        ),
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--smoke-train-samples", type=int, default=40)
    parser.add_argument("--smoke-val-samples", type=int, default=16)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing run directory (refused by default)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    global _TEE
    args = parse_args(argv)
    bind_training_seed(args.training_seed)
    if args.status:
        print(
            json.dumps(
                status_report(args.run_root), ensure_ascii=False, indent=2
            )
        )
        return 0

    device = resolve_device(args.device)
    loader_config = loader_from_args(args.num_workers, args.prefetch_factor)
    if args.torch_threads is not None:
        if args.torch_threads < 1:
            raise ValueError(f"--torch-threads must be >= 1, got {args.torch_threads}")
        torch.set_num_threads(int(args.torch_threads))
        os.environ["OMP_NUM_THREADS"] = str(int(args.torch_threads))
    identity = check_index_identity()
    if not identity["passed"]:
        raise RuntimeError(f"frozen index identity check failed: {identity['errors']}")
    environment = environment_record(device)
    environment["torch_threads_requested"] = args.torch_threads
    environment["torch_threads_actual"] = torch.get_num_threads()
    environment["torch_interop_threads"] = torch.get_num_interop_threads()
    pre_run = {
        "phase": phase_for(TRAINING_SEED),
        "mode": args.mode,
        "epochs": args.epochs if args.mode == "full" else min(args.smoke_epochs, 1),
        "training_seed": TRAINING_SEED,
        "accepted_seed_reference": DEFAULT_TRAINING_SEED,
        "loader": loader_config.payload(),
        "torch_threads_requested": args.torch_threads,
        "torch_threads_actual": torch.get_num_threads(),
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
            "selection": SELECTION_RULE,
            "prediction": PREDICTION_RULE,
            "label_mapping": LABEL_MAPPING,
        },
        "identity": identity,
        "environment": environment,
        "source_sha256": environment["source_sha256"],
        "written_at": utc_now(),
    }
    smoke = args.mode == "smoke"
    root = (
        Path(args.run_root)
        if args.run_root is not None
        else (SMOKE_ROOT if smoke else run_root_for(TRAINING_SEED))
    )
    if (root / "best_model.pt").is_file() and not args.force:
        raise FileExistsError(
            f"a trained checkpoint already exists at {root / 'best_model.pt'}; "
            "refusing to overwrite it without --force"
        )
    if args.force and root.is_dir():
        shutil.rmtree(root)
    write_json(root / "pre_run_config.json", pre_run)
    write_json(root / "environment.json", environment)
    write_json(root / "input_identity.json", identity)
    log(
        json.dumps(
            {
                "identity_passed": identity["passed"],
                "training_seed": TRAINING_SEED,
                "validation_seed": VALIDATION_SEED,
                "run_root": str(root),
                "train_windows": identity["observed"]["windows"][TRAIN_ROLE],
                "val_windows": identity["observed"]["windows"][VAL_ROLE],
            },
            ensure_ascii=False,
        )
    )
    if args.identity_only:
        return 0

    if smoke:
        epochs = min(args.smoke_epochs, 1)
        max_train: int | None = args.smoke_train_samples
        max_val: int | None = args.smoke_val_samples
    else:
        epochs = args.epochs
        max_train = None
        max_val = None

    _TEE = Tee(root / ("smoke_train.log" if smoke else LOG_NAME))
    try:
        result = train_stage1(
            device,
            epochs,
            root,
            max_train,
            max_val,
            args.mode,
            pre_run,
            loader_config,
        )
    finally:
        _TEE.close()
        _TEE = None

    summary = {
        "phase": phase_for(TRAINING_SEED),
        "mode": args.mode,
        "epochs": epochs,
        "training_seed": TRAINING_SEED,
        "validation_seed": VALIDATION_SEED,
        "loader": loader_config.payload(),
        "torch_threads_requested": args.torch_threads,
        "torch_threads_actual": torch.get_num_threads(),
        "identity_passed": identity["passed"],
        "run_directory": result["run_directory"],
        "selected_epoch": result["config"]["selected_epoch"],
        "selected_epoch_score": result["config"]["selected_epoch_score"],
        "selection_rule": result["config"]["selection"],
        "best_model_sha256": result["config"]["best_model_sha256"],
        "history": result["history"],
        "read_audit_passed": result["read_audit"]["passed"],
        "read_audit_windows_read": result["read_audit"]["windows_read"],
        "read_audit_opened_files": result["read_audit"]["opened_file_count"],
        "loader_audit_passed": result["config"].get("loader_audit_passed"),
        "checkpoint_reload_passed": result["reload_check"]["passed"],
        "completed_at": utc_now(),
    }
    write_json(root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
