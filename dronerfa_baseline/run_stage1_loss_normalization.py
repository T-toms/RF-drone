#!/usr/bin/env python3
"""STAGE1-LOSS-NORM: seed-2027 loss-normalisation comparison entry point.

One new formal candidate run of the frozen stage-1 ``Background vs Drone`` protocol
(``work/deepseek-tasks/20260927-stage1-loss-normalization.md``):

* identical inputs and protocol to the accepted ``STAGE1-12DRONES-seed2027`` run
  (same frozen index, same source-file split, same seed, same order/noise rules,
  same ``num_workers=0`` reference loader, same validation cache and selection rule);
* the *only* intended difference is the loss normalisation, selected with
  ``--loss-normalization`` (default ``batch_weight_sum`` = the historical behaviour);
* the recorded validation loss stays the original class-weighted cross entropy, so
  the candidate objective loss is stored under its own name only;
* every epoch boundary writes ``last_training_state.pt`` (modules, optimiser,
  GradScaler, RNG states, history, selection, protocol/code/index identity), and
  ``--resume-from`` accepts exactly such a file.

Full runs in the default mode are refused unless ``--allow-full-control-arm`` is
passed: the task authorises **one** new candidate run and forbids privately adding a
second full control arm.  Bounded smoke runs are allowed in both modes and are the
evidence that the default mode is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / "outputs" / ".matplotlib"))

import numpy as np
import torch

import loss_normalization as ln
import run_stage1_12drones_training as base

# ---------------------------------------------------------------------------
# this task's own paths
# ---------------------------------------------------------------------------
OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "stage1_loss_normalization_20260927"
CANDIDATE_DIR_NAME = "seed_2027_global_train_mean"
DEFAULT_MODE_DIR_NAME = "seed_2027_batch_weight_sum"
SMOKE_DIR_NAME = "smoke"
PROTOCOL_NAME = "protocol.json"
STATE_NAME = "last_training_state.pt"
STATE_JSON_NAME = "last_training_state.json"
RESUME_EVENTS_NAME = "resume_events.json"
OBJECTIVE_STEPS_NAME = "objective_loss_steps.csv"
OBJECTIVE_LOSS_NAME = "objective_loss.json"
VALIDATION_OBJECTIVE_NAME = "validation_objective_loss.json"
LOSS_SPEC_NAME = "loss_spec.json"

#: The accepted control whose selected-epoch score is the comparison denominator.
CONTROL_ROOT = base.OUTPUT_ROOT / "seed_2027"
CONTROL_PHASE = "STAGE1-12DRONES-seed2027"
#: Context only: a different random seed is *not* a loss-change effect.
CONTEXT_ROOT = base.OUTPUT_ROOT / "seed_2026"
CONTEXT_PHASE = "STAGE1-12DRONES-A"

#: Code whose hashes the run records (the frozen protocol plus this task's modules).
SOURCE_FILES = (
    *base.SOURCE_FILES,
    "loss_normalization.py",
    "run_stage1_loss_normalization.py",
)

DEFAULT_TRAINING_SEED = 2027
PROGRESS_EVERY = base.PROGRESS_EVERY


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return base.sha256_file(Path(path))


def write_json(path: Path, payload: Any) -> None:
    base.write_json(Path(path), payload)


def read_json(path: Path) -> Any:
    return base.read_json(Path(path))


# ---------------------------------------------------------------------------
# epoch-boundary state
# ---------------------------------------------------------------------------
def state_identity(
    *,
    index_sha256: str,
    split_source_sha256: str,
    training_seed: int,
    validation_seed: int,
    epochs: int,
    loss_mode: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    return {
        "index_sha256": index_sha256,
        "split_source_sha256": split_source_sha256,
        "training_seed": int(training_seed),
        "validation_seed": int(validation_seed),
        "epochs": int(epochs),
        "loss_mode": str(loss_mode),
        "protocol_sha256": str(protocol_sha256),
    }


def save_training_state(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    frontend: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    history: Sequence[dict[str, Any]],
    order_hashes: Sequence[str],
    identity_order_hashes: Sequence[str],
    epoch_loader_records: Sequence[dict[str, Any]],
    selection: dict[str, Any],
    initial_model_sha256: str,
    identity: dict[str, Any],
    loss_payload: dict[str, Any],
) -> dict[str, Any]:
    """Write ``last_training_state.pt`` atomically and describe it in JSON."""

    payload = {
        "schema": base.TRAINING_STATE_SCHEMA,
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "model_state": model.state_dict(),
        "frontend_state": frontend.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng": base.rng_state_payload(),
        "history": [dict(row) for row in history],
        "order_hashes": [str(value) for value in order_hashes],
        "identity_order_hashes": [str(value) for value in identity_order_hashes],
        "epoch_loader_records": [dict(row) for row in epoch_loader_records],
        "selection": dict(selection),
        "initial_model_sha256": str(initial_model_sha256),
        "identity": dict(identity),
        "loss": dict(loss_payload),
        "saved_at": utc_now(),
        "resume_scope": (
            "explicit epoch boundaries only: the state is written after a completed "
            "epoch, never mid-epoch, so an interrupted epoch has to be re-run"
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "selected_epoch_so_far": int(selection.get("selected_epoch", 0)),
        "saved_at": payload["saved_at"],
    }


def load_training_state(path: Path, device: torch.device) -> dict[str, Any]:
    """Load an epoch-boundary state.

    ``map_location="cpu"`` is deliberate: the RNG states must stay CPU byte tensors
    (``torch.set_rng_state`` / ``torch.cuda.set_rng_state_all`` reject GPU ones), and
    ``load_state_dict`` moves the module/optimiser tensors onto the live device by
    itself.
    """

    del device
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def path_for(args: argparse.Namespace, mode: str) -> Path:
    if args.run_root is not None:
        return Path(args.run_root)
    if args.mode == "smoke":
        return OUTPUT_ROOT / SMOKE_DIR_NAME
    if mode == ln.LOSS_MODE_BATCH_WEIGHT_SUM:
        return OUTPUT_ROOT / DEFAULT_MODE_DIR_NAME
    return OUTPUT_ROOT / CANDIDATE_DIR_NAME


def control_reference() -> dict[str, Any]:
    config_path = CONTROL_ROOT / "run_config.json"
    payload = read_json(config_path) if config_path.is_file() else {}
    context_path = CONTEXT_ROOT / "run_config.json"
    context = read_json(context_path) if context_path.is_file() else {}
    return {
        "control": {
            "phase": CONTROL_PHASE,
            "run_root": str(CONTROL_ROOT),
            "run_config": str(config_path),
            "run_config_sha256": sha256_file(config_path) if config_path.is_file() else None,
            "loss": payload.get("loss"),
            "selected_epoch": payload.get("selected_epoch"),
            "selected_epoch_score": payload.get("selected_epoch_score"),
            "best_model_sha256": payload.get("best_model_sha256"),
            "initial_model_sha256": payload.get("initial_model_sha256"),
            "epoch_order_sha256": payload.get("epoch_order_sha256"),
            "class_weights": payload.get("class_weights"),
            "loader": payload.get("loader"),
            "note": (
                "accepted STAGE1-12DRONES-seed2027 control; it predates the loader "
                "audit, so it has no loader_read_audit.json"
            ),
        },
        "context_only": {
            "phase": CONTEXT_PHASE,
            "run_root": str(CONTEXT_ROOT),
            "selected_epoch": context.get("selected_epoch"),
            "selected_epoch_score": context.get("selected_epoch_score"),
            "note": "independent random seed; context only, never a loss-change effect",
        },
    }


def preflight_gate(
    *,
    seed: int,
    epochs: int,
    spec: ln.LossSpec,
    control: dict[str, Any],
) -> dict[str, Any]:
    """Hard pre-run gate: initialisation, epoch order and weights vs the control.

    Runs before any window is read.  The initial model hash and every frozen epoch
    order hash must equal the accepted ``STAGE1-12DRONES-seed2027`` values, which is
    what makes the loss normalisation the *only* intended difference between the two
    runs.  A mismatch raises here instead of after three hours of training.
    """

    dataset = base.DroneRFaWindowDataset(
        base.INDEX, split=base.TRAIN_ROLE, seed=int(seed)
    )
    split_length = len(dataset)
    order_hashes = [
        base.sha256_bytes(
            np.asarray(base.order_for_epoch(dataset, epoch, int(seed)), dtype=np.int64).tobytes()
        )
        for epoch in range(1, int(epochs) + 1)
    ]
    base.seed_everything(int(seed))
    model = base.ResNet18(input_channels=1, num_classes=2)
    initial_hash = base.state_hash(model.state_dict())
    del model
    dataset.close()

    control_payload = control["control"]
    expected_initial = control_payload.get("initial_model_sha256")
    expected_orders = list(control_payload.get("epoch_order_sha256") or [])
    expected_weights = [float(value) for value in (control_payload.get("class_weights") or [])]
    errors: list[str] = []
    if expected_initial is None:
        errors.append("the control run_config does not record initial_model_sha256")
    elif initial_hash != expected_initial:
        errors.append(
            f"initial model hash {initial_hash} != control {expected_initial}"
        )
    if len(expected_orders) < int(epochs):
        errors.append(
            f"the control records {len(expected_orders)} epoch order hashes, need {epochs}"
        )
    else:
        for epoch, (observed, expected) in enumerate(
            zip(order_hashes, expected_orders), start=1
        ):
            if observed != expected:
                errors.append(
                    f"epoch {epoch} order hash {observed} != control {expected}"
                )
    if expected_weights and max(
        abs(left - right) for left, right in zip(spec.class_weights, expected_weights)
    ) > 1e-9:
        errors.append(
            f"class weights {list(spec.class_weights)} != control {expected_weights}"
        )
    return {
        "checked_at": utc_now(),
        "split_length": split_length,
        "training_seed": int(seed),
        "initial_model_sha256": initial_hash,
        "control_initial_model_sha256": expected_initial,
        "initial_model_matches_control": initial_hash == expected_initial,
        "epoch_order_sha256": order_hashes,
        "control_epoch_order_sha256": expected_orders[: int(epochs)],
        "epoch_order_matches_control": order_hashes == expected_orders[: int(epochs)],
        "class_weights": list(spec.class_weights),
        "control_class_weights": expected_weights,
        "loss_mode": spec.mode,
        "mu_train": spec.mu_train,
        "errors": errors,
        "passed": not errors,
    }


def build_protocol(
    *,
    args: argparse.Namespace,
    root: Path,
    identity: dict[str, Any],
    spec: ln.LossSpec,
    seed: int,
    loader_payload: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "dronerfa-stage1-loss-norm-protocol/v1",
        "task": "STAGE1-LOSS-NORM",
        "created_at": utc_now(),
        "run_root": str(root),
        "mode": args.mode,
        "hypothesis": (
            "with batch_size=4 on a 600/8990 background/drone split, the historical "
            "weighted CE normalises by the batch's own weight sum, so a drone-only "
            "batch cancels the class weights entirely; dividing by the actual batch "
            "size times the global training mean weight restores the global scale"
        ),
        "primary_comparison": {
            "metric": (
                "mean pooled balanced accuracy over the four frozen validation "
                "conditions at the epoch selected by the frozen rule, zero threshold"
            ),
            "candidate": str(root),
            "candidate_loss_mode": spec.mode,
            "control": str(CONTROL_ROOT),
            "control_loss_mode": ln.LOSS_MODE_BATCH_WEIGHT_SUM,
            "context_only": str(CONTEXT_ROOT),
            "statement": (
                "one paired training-seed-2027 comparison; it can support a single "
                "paired effect estimate and must not be reported as cross-seed "
                "stability"
            ),
        },
        "selection_rule": {
            "split": base.VAL_ROLE,
            "conditions": list(base.CONDITIONS),
            "rule": base.SELECTION_RULE,
            "key": base.SELECTION_KEY,
            "tie_break": base.TIE_BREAK,
            "prediction": base.PREDICTION_RULE,
        },
        "stopping_conditions": [
            "training loss, validation features or the objective become non-finite",
            "the initial model hash differs from the frozen seed-2027 initialisation",
            "the per-epoch order hash differs from the accepted seed-2027 order",
            "a read audit opens a development_test* source file or an unexpected file",
            "the loader consumption audit does not close exactly",
            "a required permission, device or protocol precondition is not met",
        ],
        "candidate_parameters": spec.payload(),
        "frozen_protocol": {
            "n_fft": base.N_FFT,
            "hop_length": base.HOP_LENGTH,
            "image_size": base.IMAGE_SIZE,
            "batch_size": base.BATCH_SIZE,
            "epochs": effective_epochs(args),
            "learning_rate": base.LEARNING_RATE,
            "weight_decay": base.WEIGHT_DECAY,
            "optimizer": "Adam",
            "amp": "float16 autocast + GradScaler",
            "train_snrs": base.TRAIN_SNRS,
            "validation_seed": base.VALIDATION_SEED,
            "window_samples": base.WINDOW_SAMPLES,
            "label_mapping": base.LABEL_MAPPING,
            "loader": loader_payload,
            "torch_threads_actual": environment.get("torch_threads"),
            "torch_threads_requested": args.torch_threads,
        },
        "inputs": {
            "index": str(base.INDEX),
            "index_sha256": identity["index"]["sha256"],
            "index_rows": identity["index"]["rows"],
            "frozen_split_csv": str(base.FROZEN_SPLIT_CSV),
            "frozen_split_csv_sha256": base.FROZEN_SPLIT_CSV_SHA256,
            "g_manifest": str(base.G_MANIFEST),
            "g_manifest_sha256": identity["g_manifest"]["sha256"],
            "observed": identity["observed"],
        },
        "code_sha256": environment["source_sha256"],
        "device": {
            "device": str(environment.get("device")),
            "gpu": environment.get("gpu"),
            "torch": environment.get("torch"),
            "cuda_version": environment.get("cuda_version"),
        },
        "controls": control_reference(),
    }


def protocol_core_sha256(payload: dict[str, Any]) -> str:
    """Hash of the protocol *content*: everything except the volatile stamps.

    ``protocol.json`` carries a ``created_at`` timestamp and its own digest, neither
    of which may enter the resume gate -- a resumed segment builds a fresh file and
    must still match the protocol the interrupted segment started from.
    """

    core = {
        key: value
        for key, value in payload.items()
        if key not in ("created_at", "protocol_core_sha256")
    }
    blob = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def effective_epochs(args: argparse.Namespace) -> int:
    """How many epochs this invocation will really run.

    The protocol demands :data:`base.EPOCHS` for a full run; a bounded smoke run uses
    ``--smoke-epochs``.  The protocol file, the state identity and run_config all
    quote this one number so the resume gate compares like with like.
    """

    if args.mode == "full":
        epochs = int(args.epochs)
        if epochs != base.EPOCHS:
            raise RuntimeError(
                f"the frozen protocol is {base.EPOCHS} epochs; refusing --epochs {epochs}"
            )
        return epochs
    return int(args.smoke_epochs)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seed = int(args.training_seed)
    base.bind_training_seed(seed)
    if args.status:
        print(json.dumps(status_report(Path(args.run_root) if args.run_root else None), ensure_ascii=False, indent=2))
        return 0

    device = base.resolve_device(args.device)
    loader_config = base.loader_from_args(args.num_workers, args.prefetch_factor)
    identity = base.check_index_identity()
    if not identity["passed"]:
        raise RuntimeError(f"frozen index identity check failed: {identity['errors']}")
    spec = ln.build_loss_spec(
        args.loss_normalization,
        identity["observed"]["stage1_label_counts"][base.TRAIN_ROLE],
        base.BATCH_SIZE,
    )
    if spec.train_windows != identity["observed"]["windows"][base.TRAIN_ROLE]:
        raise RuntimeError(
            f"loss spec train windows {spec.train_windows} != "
            f"index train windows {identity['observed']['windows'][base.TRAIN_ROLE]}"
        )

    if args.mode == "full":
        if device.type != "cuda":
            raise RuntimeError(
                f"a full run requires CUDA, got device={device}; report the device "
                "instead of silently training on the CPU"
            )
        if int(args.num_workers) != 0:
            raise RuntimeError(
                "the frozen loader configuration for full runs is num_workers=0 "
                f"(Codex decision 2026-09-28); got {args.num_workers}"
            )
        if spec.mode == ln.LOSS_MODE_BATCH_WEIGHT_SUM and not args.allow_full_control_arm:
            raise RuntimeError(
                "STAGE1-LOSS-NORM authorises exactly one new full run (the candidate). "
                "A full batch_weight_sum arm would be a second control arm and is "
                "refused without --allow-full-control-arm."
            )

    if args.torch_threads is not None:
        if args.torch_threads < 1:
            raise ValueError(f"--torch-threads must be >= 1, got {args.torch_threads}")
        torch.set_num_threads(int(args.torch_threads))
        os.environ["OMP_NUM_THREADS"] = str(int(args.torch_threads))

    environment = base.environment_record(device)
    environment["source_sha256"] = {
        name: sha256_file(SCRIPT_DIR / name) for name in SOURCE_FILES
    }
    environment["torch_threads_requested"] = args.torch_threads
    environment["torch_threads_actual"] = torch.get_num_threads()
    environment["absolute_path"] = str(SCRIPT_DIR)
    environment["platform_detail"] = platform.platform()
    environment["recorded_at"] = utc_now()

    root = path_for(args, spec.mode)
    resuming = args.resume_from is not None
    if not resuming and (root / "best_model.pt").is_file() and not args.force:
        raise FileExistsError(
            f"a trained checkpoint already exists at {root / 'best_model.pt'}; "
            "refusing to overwrite a completed run"
        )
    root.mkdir(parents=True, exist_ok=True)

    protocol = build_protocol(
        args=args,
        root=root,
        identity=identity,
        spec=spec,
        seed=seed,
        loader_payload=loader_config.payload(),
        environment=environment,
    )
    protocol_sha256 = protocol_core_sha256(protocol)
    protocol["protocol_core_sha256"] = protocol_sha256
    write_json(root / PROTOCOL_NAME, protocol)
    protocol_file_sha256 = sha256_file(root / PROTOCOL_NAME)
    write_json(root / LOSS_SPEC_NAME, spec.payload())
    write_json(root / "input_identity.json", identity)
    write_json(root / "environment.json", environment)

    # The effective epoch count is decided before anything is written, so the state
    # identity, the resume gate and run_config all quote the same number.
    epochs = effective_epochs(args)
    control = control_reference()
    preflight = preflight_gate(seed=seed, epochs=epochs, spec=spec, control=control)
    preflight["mode"] = args.mode
    preflight["run_root"] = str(root)
    write_json(root / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError(
            "preflight gate failed; refusing to train: " + "; ".join(preflight["errors"])
        )

    state_identity_payload = state_identity(
        index_sha256=base.INDEX_SHA256,
        split_source_sha256=base.FROZEN_SPLIT_CSV_SHA256,
        training_seed=seed,
        validation_seed=base.VALIDATION_SEED,
        epochs=epochs,
        loss_mode=spec.mode,
        protocol_sha256=protocol_sha256,
    )
    pre_run = {
        "phase": f"STAGE1-LOSS-NORM-{spec.mode}",
        "mode": args.mode,
        "epochs": epochs,
        "training_seed": seed,
        "loader": loader_config.payload(),
        "torch_threads_requested": args.torch_threads,
        "torch_threads_actual": torch.get_num_threads(),
        "loss_spec": spec.payload(),
        "protocol_sha256": protocol_sha256,
        "identity": identity,
        "environment": environment,
        "source_sha256": environment["source_sha256"],
        "written_at": utc_now(),
    }

    initial_state: dict[str, Any] | None = None
    resume_record: dict[str, Any] | None = None
    start_epoch = 1
    if resuming:
        resume_path = Path(args.resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"no training state at {resume_path}")
        state = load_training_state(resume_path, device)
        errors = base.resume_identity_errors(
            state,
            index_sha256=base.INDEX_SHA256,
            split_source_sha256=base.FROZEN_SPLIT_CSV_SHA256,
            training_seed=seed,
            validation_seed=base.VALIDATION_SEED,
            epochs=epochs,
        )
        saved_loss = (state.get("loss") or {}).get("mode")
        if saved_loss != spec.mode:
            errors.append(f"state loss mode {saved_loss!r} != requested {spec.mode!r}")
        saved_protocol = (state.get("identity") or {}).get("protocol_sha256")
        if saved_protocol != protocol_sha256:
            errors.append(
                f"state protocol {saved_protocol!r} != this run's {protocol_sha256!r}"
            )
        if errors:
            raise RuntimeError(f"refusing to resume: {errors}")
        initial_state = state
        start_epoch = int(state["next_epoch"])
        resume_record = {
            "resumed": True,
            "state_path": str(resume_path),
            "state_sha256": sha256_file(resume_path),
            "state_epoch": int(state["epoch"]),
            "start_epoch": start_epoch,
            "resumed_at": utc_now(),
            "state_history_epochs": len(state.get("history", [])),
            "note": (
                "the loader/read audit of this process covers only the epochs listed "
                "in its own records; the pre-resume epoch reads are recorded in the "
                "saved segment evidence, not here"
            ),
        }
        write_json(root / RESUME_EVENTS_NAME, resume_record)

    write_json(root / "pre_run_config.json", pre_run)

    training_criterion = ln.build_criterion(spec, device, record_steps=True)
    validation_criterion = ln.build_validation_criterion(spec, device)
    candidate_probe_criterion = ln.build_criterion(spec, device, record_steps=False)
    train_rows = [row for row in base.read_csv_rows(base.INDEX) if str(row["split"]) == base.TRAIN_ROLE]
    val_rows = [row for row in base.read_csv_rows(base.INDEX) if str(row["split"]) == base.VAL_ROLE]
    smoke = args.mode != "full"
    max_train = None if not smoke else int(args.smoke_train_samples)
    max_val = None if not smoke else int(args.smoke_val_samples)
    selected_train = base.select_positions(train_rows, max_train)
    selected_val = base.select_positions(val_rows, max_val)
    calls_per_condition = -(-len(selected_val) // base.BATCH_SIZE)
    probe = ln.ObjectiveLossProbe(
        validation_criterion,
        candidate_probe_criterion,
        conditions=base.CONDITIONS,
        calls_per_condition=calls_per_condition,
    )
    state_records: list[dict[str, Any]] = []

    def epoch_callback(**kwargs: Any) -> None:
        record = save_training_state(
            root / STATE_NAME,
            epoch=int(kwargs["epoch"]),
            model=kwargs["model"],
            frontend=kwargs["frontend"],
            optimizer=kwargs["optimizer"],
            scaler=kwargs["scaler"],
            history=kwargs["history"],
            order_hashes=kwargs["order_hashes"],
            identity_order_hashes=kwargs["identity_order_hashes"],
            epoch_loader_records=kwargs["epoch_loader_records"],
            selection=kwargs["selection"],
            initial_model_sha256=kwargs["initial_model_sha256"],
            identity=state_identity_payload,
            loss_payload=spec.payload(),
        )
        state_records.append(record)
        write_json(
            root / STATE_JSON_NAME,
            {
                "schema": base.TRAINING_STATE_SCHEMA,
                "state_file": STATE_NAME,
                "writes": state_records,
                "latest": record,
                "resume_scope": (
                    "epoch boundaries only; an epoch interrupted mid-way is re-run "
                    "from the previous boundary"
                ),
            },
        )
        base.log(f"    state: saved {STATE_NAME} at epoch {record['epoch']} ({record['bytes']} bytes)")

    base.log(
        json.dumps(
            {
                "identity_passed": identity["passed"],
                "training_seed": seed,
                "validation_seed": base.VALIDATION_SEED,
                "run_root": str(root),
                "loss_mode": spec.mode,
                "mu_train": spec.mu_train,
                "loader": loader_config.payload(),
                "start_epoch": start_epoch,
            },
            ensure_ascii=False,
        )
    )
    if args.identity_only:
        return 0

    extra_config = {
        "loss_normalization": spec.payload(),
        "loss_normalization_task": "STAGE1-LOSS-NORM",
        "protocol_sha256": protocol_sha256,
        "protocol_json_sha256": protocol_file_sha256,
        "protocol_path": str(root / PROTOCOL_NAME),
        "control_run_root": str(CONTROL_ROOT),
        "control_phase": CONTROL_PHASE,
        "context_run_root": str(CONTEXT_ROOT),
        "validation_loss_definition": (
            "original class-weighted cross entropy (unchanged); the candidate "
            "objective loss is stored in validation_objective_loss.json"
        ),
        "resume_scope": "epoch boundaries only",
    }

    base._TEE = base.Tee(root / ("smoke_train.log" if args.mode == "smoke" else base.LOG_NAME))
    started = time.perf_counter()
    try:
        result = base.train_stage1(
            device,
            epochs,
            root,
            max_train,
            max_val,
            args.mode,
            pre_run,
            loader_config,
            train_criterion=training_criterion,
            eval_criterion=probe,
            extra_config=extra_config,
            epoch_callback=epoch_callback,
            start_epoch=start_epoch,
            initial_state=initial_state,
            resume_record=resume_record,
        )
    finally:
        base._TEE.close()
        base._TEE = None
    elapsed = time.perf_counter() - started

    steps = list(getattr(training_criterion, "steps", []))
    steps_per_epoch = -(-len(selected_train) // base.BATCH_SIZE)
    steps_summary = ln.step_evidence_summary(steps)
    pre_resume_epochs = len(initial_state.get("history", [])) if initial_state else 0
    epochs_this_process = len(result["history"]) - pre_resume_epochs
    write_json(
        root / OBJECTIVE_LOSS_NAME,
        {
            "mode": spec.mode,
            "spec": spec.payload(),
            "steps_recorded": bool(steps),
            "steps_per_epoch": steps_per_epoch,
            "expected_steps_this_process": steps_per_epoch * epochs_this_process,
            "epochs_this_process": epochs_this_process,
            "pre_resume_epochs": pre_resume_epochs,
            "start_epoch": start_epoch,
            "summary": steps_summary,
            "steps_note": (
                "per-step numerator/denominator evidence is recorded for the candidate "
                "normalisation; in the historical batch_weight_sum mode the criterion "
                "is the original nn.CrossEntropyLoss module, which is intentionally "
                "not instrumented"
            ),
            "denominator_check": {
                "rule": ln.DENOMINATOR_RULES[spec.mode],
                "tail_batch_rule": ln.TAIL_BATCH_RULES[spec.mode],
                "tail_batch_size": spec.tail_batch_size,
                "tail_batches_per_epoch": spec.tail_batches_per_epoch,
                "expected_tail_denominator": (
                    float(torch.tensor(spec.tail_batch_size * spec.mu_train, dtype=torch.float32))
                    if spec.mode == ln.LOSS_MODE_GLOBAL_TRAIN_MEAN
                    else None
                ),
            },
        },
    )
    if steps:
        rows = []
        for index, step in enumerate(steps):
            rows.append(
                {
                    "step": index + 1,
                    "epoch": start_epoch + index // steps_per_epoch,
                    "batch_in_epoch": index % steps_per_epoch + 1,
                    "batch_size": int(step["batch_size"]),
                    "denominator": float(step["denominator"]),
                    "numerator": float(step["numerator"]),
                    "loss": float(step["loss"]),
                }
            )
        base.write_csv(root / OBJECTIVE_STEPS_NAME, rows)
    validation_objective = probe.mean_by_cell()
    validation_objective["cycles_note"] = (
        "cycle k is the validation evaluation of training epoch k+1; the final cycle "
        "is the checkpoint-reload re-evaluation of the selected epoch"
    )
    validation_objective["validation_cache_samples"] = len(selected_val)
    write_json(root / VALIDATION_OBJECTIVE_NAME, validation_objective)
    write_json(
        root / STATE_JSON_NAME,
        {
            "schema": base.TRAINING_STATE_SCHEMA,
            "state_file": STATE_NAME,
            "writes": state_records,
            "latest": state_records[-1] if state_records else None,
            "resume_scope": (
                "epoch boundaries only; an epoch interrupted mid-way is re-run from "
                "the previous boundary"
            ),
            "state_exists": (root / STATE_NAME).is_file(),
        },
    )

    summary = {
        "phase": f"STAGE1-LOSS-NORM-{spec.mode}",
        "mode": args.mode,
        "epochs": epochs,
        "training_seed": seed,
        "validation_seed": base.VALIDATION_SEED,
        "loss_mode": spec.mode,
        "loss_spec": spec.payload(),
        "loader": loader_config.payload(),
        "torch_threads_actual": torch.get_num_threads(),
        "run_directory": str(root),
        "selected_epoch": result["config"]["selected_epoch"],
        "selected_epoch_score": result["config"]["selected_epoch_score"],
        "selection_rule": result["config"]["selection"],
        "initial_model_sha256": result["config"]["initial_model_sha256"],
        "best_model_sha256": result["config"]["best_model_sha256"],
        "epoch_order_sha256": result["config"]["epoch_order_sha256"],
        "history": result["history"],
        "read_audit_passed": result["read_audit"]["passed"],
        "read_audit_windows_read": result["read_audit"]["windows_read"],
        "read_audit_opened_files": result["read_audit"]["opened_file_count"],
        "loader_audit_passed": result["config"].get("loader_audit_passed"),
        "checkpoint_reload_passed": result["reload_check"]["passed"],
        "objective_steps": steps_summary,
        "start_epoch": start_epoch,
        "resumed": bool(resuming),
        "seconds": elapsed,
        "completed_at": utc_now(),
    }
    write_json(root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def status_report(run_root: Path | None = None) -> dict[str, Any]:
    root = Path(run_root) if run_root is not None else OUTPUT_ROOT / CANDIDATE_DIR_NAME
    report: dict[str, Any] = {"run_root": str(root), "exists": root.is_dir()}
    if report["exists"]:
        progress = root / "progress.json"
        config = root / "run_config.json"
        summary = root / "summary.json"
        report["progress"] = read_json(progress) if progress.is_file() else None
        report["status"] = read_json(config).get("status") if config.is_file() else None
        report["summary"] = read_json(summary) if summary.is_file() else None
        state = root / STATE_JSON_NAME
        report["state"] = read_json(state) if state.is_file() else None
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "STAGE1-LOSS-NORM: one seed-2027 stage-1 run at a chosen loss "
            "normalisation, on the frozen 104-MAT / 12-drone split."
        )
    )
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--epochs", type=int, default=base.EPOCHS)
    parser.add_argument(
        "--loss-normalization",
        choices=ln.LOSS_MODES,
        default=ln.DEFAULT_LOSS_MODE,
        help=(
            "batch_weight_sum (default, the historical PyTorch weighted CE) or "
            "global_train_mean (the candidate: divide by the actual batch size times "
            "mu_train)"
        ),
    )
    parser.add_argument("--training-seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=base.REFERENCE_NUM_WORKERS)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--smoke-train-samples", type=int, default=40)
    parser.add_argument("--smoke-val-samples", type=int, default=16)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--allow-full-control-arm",
        action="store_true",
        help=(
            "permit a full (5-epoch) run in the default batch_weight_sum mode; the "
            "task authorises only one new full candidate run, so this needs explicit "
            "Codex approval"
        ),
    )
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
