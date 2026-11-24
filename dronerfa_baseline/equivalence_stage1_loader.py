#!/usr/bin/env python3
"""STAGE1-IO-OPT-R1 (R3): a real, same-version 0/2/4 training-step comparison.

Why this exists
---------------
The first round compared two smoke runs that were produced by *different source
revisions* (``run_stage1_12drones_training.py`` SHA256 ``516aaeb8...`` at 08:00 vs
``a857ba27...`` at 10:22, ``stage1_loader_optimization.py`` ``f1fc9318...`` vs
``7159d5eb...``) and then claimed training equivalence from a validation-feature-cache
hash.  The two runs really did train differently (``train_loss`` 1.0145 vs 0.9355),
because the code that produced them differed.  Nothing about that pair proves or
disproves anything about the loader, and it is kept only as an invalidated record.

This script redoes the comparison inside one process, on one code revision, with

* the *real* frozen train windows from G: (no synthetic fixture),
* the *real* ``train_one_epoch`` the training driver uses -- not a re-implementation,
* a full optimisation step: class-weighted CE + AMP float16 + ``scaler.scale`` +
  ``backward`` + ``GradScaler.step`` + ``update`` + Adam,
* the same initialisation, the same per-epoch shuffle, the same PER-WINDOW noise
  (epoch and position both enter the AWGN seed), the same batch size and order,
* a fourth column, ``legacy_serial``, that rebuilds the historical
  ``Subset(dataset, positions)`` + ``num_workers=0`` loader through the same function
  body, so "the refactor changed the serial maths" can be separated from
  "the workers changed the maths".

Per batch it records the input digests, the labels, the loss, the scaler scale, the
gradient digest, the parameter digest (which includes every BatchNorm running
statistic and ``num_batches_tracked``) and the batch metrics; per epoch it records
the optimiser and scaler state digests.  Everything is compared against the
reference column, and the run is written to a new directory.

Usage::

    python equivalence_stage1_loader.py --torch-threads 2
    python equivalence_stage1_loader.py --deterministic   # separates CUDA nondeterminism
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import platform
import statistics
import sys
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
from torch import nn

from model import ResNet18, SpectrogramFrontEnd
from stage1_loader_optimization import (
    ConsumptionLog,
    LoaderAudit,
    LoaderConfig,
    limited_order,
    order_for_epoch,
)
from train_resnet18 import seed_everything

import run_stage1_12drones_training as drv

DEFAULT_OUT = (
    SCRIPT_DIR / "outputs" / "stage1_io_optimization_20260927" / "rework1" / "equivalence"
)
#: Column order: the historical loader first, then the unified path at 0/2/4 workers.
MODES = ("legacy_serial", "workers0", "workers2", "workers4")
MODE_WORKERS = {"legacy_serial": 0, "workers0": 0, "workers2": 2, "workers4": 4}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def state_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            digest.update(value.detach().float().cpu().numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def optimizer_digest(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    state = optimizer.state_dict()
    for key in sorted(state["param_groups"][0] if state["param_groups"] else []):
        digest.update(str(key).encode())
    for parameter_id, entry in sorted(state["state"].items()):
        digest.update(str(parameter_id).encode())
        for name in sorted(entry):
            value = entry[name]
            digest.update(name.encode())
            if isinstance(value, torch.Tensor):
                digest.update(value.detach().float().cpu().numpy().tobytes())
            else:
                digest.update(str(value).encode())
    return digest.hexdigest()


def build_column(
    positions: Sequence[int],
    mode: str,
    device: torch.device,
    epochs: int,
    training_seed: int,
) -> dict[str, Any]:
    """Run ``epochs`` real training epochs for one loader configuration."""

    workers = MODE_WORKERS[mode]
    context = multiprocessing.get_context("spawn")
    events = context.Queue()
    audit = LoaderAudit(f"equivalence_{mode}")
    audit.set_events(events)

    read_audit = drv.ReadAudit(f"equivalence_{mode}")
    dataset = drv.AuditedWindowDataset(
        drv.INDEX,
        split=drv.TRAIN_ROLE,
        train_snrs=drv.parse_snr_list(drv.TRAIN_SNRS),
        seed=training_seed,
        read_audit=read_audit,
    )
    observed_positions = drv.select_positions(dataset.rows, len(positions))
    if observed_positions != list(positions):
        raise RuntimeError(
            f"the per-column window selection drifted for {mode}: "
            f"{observed_positions[:5]} != {list(positions)[:5]}"
        )

    seed_everything(training_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = ResNet18(input_channels=1, num_classes=2).to(device)
    frontend = SpectrogramFrontEnd(
        n_fft=drv.N_FFT, hop_length=drv.HOP_LENGTH, image_size=drv.IMAGE_SIZE
    ).to(device)
    counts = {
        label: sum(1 for row in dataset.rows if int(row["label"]) == label)
        for label in (drv.LABEL_BACKGROUND, drv.LABEL_DRONE)
    }
    class_weights = torch.tensor(
        [len(dataset) / (2 * counts[label]) for label in (drv.LABEL_BACKGROUND, drv.LABEL_DRONE)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=drv.LEARNING_RATE, weight_decay=drv.WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    config = LoaderConfig(num_workers=workers)

    column: dict[str, Any] = {
        "mode": mode,
        "num_workers": workers,
        "legacy_subset_loader": mode == "legacy_serial",
        "initial_model_sha256": drv.state_hash(model.state_dict()),
        "class_weights": class_weights.detach().cpu().tolist(),
        "batches": [],
        "epochs": [],
        "loader": config.payload(),
        "started_at": utc_now(),
    }
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch)
        order = limited_order(order_for_epoch(dataset, epoch, training_seed), positions)
        consumption = ConsumptionLog(
            phase=drv.TRAIN_ROLE, key=str(epoch), on_batch=audit.drain
        )
        records: list[dict[str, Any]] = []
        started = time.perf_counter()
        loss, metrics = drv.train_one_epoch(
            frontend,
            model,
            dataset,
            order,
            criterion,
            optimizer,
            scaler,
            device,
            device.type == "cuda",
            config,
            consumption,
            events,
            epoch=epoch,
            observer=records.append,
            legacy_subset_loader=(mode == "legacy_serial"),
        )
        seconds = time.perf_counter() - started
        audit.drain()
        audit.record_consumption(consumption)
        audit.close_loader(drv.TRAIN_ROLE, str(epoch))
        report = audit.key_report(drv.TRAIN_ROLE, str(epoch))
        column["batches"].append(records)
        column["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "metrics": metrics,
                "seconds": seconds,
                "positions": order,
                "position_digest": drv.identity_digest(
                    [(str(dataset.rows[position]["file_path"]), int(position)) for position in order]
                ),
                "parameter_sha256": drv.state_hash(model.state_dict()),
                "optimizer_sha256": optimizer_digest(optimizer),
                "scaler_sha256": state_digest(scaler.state_dict()),
                "scaler_scale": float(scaler.get_scale()),
                "parent_windows_read": read_audit.windows_read,
                "loader_audit": {
                    "reads": report["reads"]["windows_read"],
                    "completed": report["consumed"]["windows_completed"],
                    "delivered": report["consumed"]["windows_delivered"],
                    "unconsumed_reads": report["unconsumed_reads"]["count"],
                    "unread_consumed": report["unread_consumed"]["count"],
                    "batches_delivered": report["consumed"]["batches_delivered"],
                    "batches_completed": report["consumed"]["batches_completed"],
                    "workers": report["workers"],
                    "identity_events": bool(report["workers"]) and not mode == "legacy_serial",
                },
            }
        )
    column["final_model_sha256"] = drv.state_hash(model.state_dict())
    column["final_params"] = {
        key: value.detach().float().cpu().clone() for key, value in model.state_dict().items()
    }
    column["finished_at"] = utc_now()
    dataset.close()
    del model, optimizer, scaler, frontend
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return column


def compare_columns(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Batch-by-batch and epoch-by-epoch comparison of two columns."""

    problems: list[dict[str, Any]] = []
    if reference["initial_model_sha256"] != candidate["initial_model_sha256"]:
        problems.append(
            {
                "issue": "initial_model_sha256",
                "reference": reference["initial_model_sha256"],
                "candidate": candidate["initial_model_sha256"],
            }
        )
    digest_keys = ("signals_sha256", "labels_sha256", "labels", "positions")
    loss_deltas: list[dict[str, Any]] = []
    digest_mismatches: list[dict[str, Any]] = []
    gradient_mismatches: list[dict[str, Any]] = []
    parameter_mismatches: list[dict[str, Any]] = []
    for epoch_index, (left_epoch, right_epoch) in enumerate(
        zip(reference["batches"], candidate["batches"]), start=1
    ):
        if len(left_epoch) != len(right_epoch):
            problems.append(
                {
                    "issue": "batch_count",
                    "epoch": epoch_index,
                    "reference": len(left_epoch),
                    "candidate": len(right_epoch),
                }
            )
        for left, right in zip(left_epoch, right_epoch):
            for key in digest_keys:
                if left.get(key) != right.get(key):
                    digest_mismatches.append(
                        {"epoch": epoch_index, "batch": left["batch"], "field": key}
                    )
            if left.get("gradient_sha256") != right.get("gradient_sha256"):
                gradient_mismatches.append(
                    {"epoch": epoch_index, "batch": left["batch"]}
                )
            if left.get("parameter_sha256") != right.get("parameter_sha256"):
                parameter_mismatches.append(
                    {"epoch": epoch_index, "batch": left["batch"]}
                )
            loss_deltas.append(
                {
                    "epoch": epoch_index,
                    "batch": left["batch"],
                    "reference": left["loss"],
                    "candidate": right["loss"],
                    "abs_delta": abs(float(left["loss"]) - float(right["loss"])),
                }
            )
    max_tensor_delta = 0.0
    different_tensors = 0
    for key, value in reference["final_params"].items():
        other = candidate["final_params"].get(key)
        if other is None:
            different_tensors += 1
            continue
        delta = float((value.double() - other.double()).abs().max())
        max_tensor_delta = max(max_tensor_delta, delta)
        if delta > 0:
            different_tensors += 1
    epoch_problems: list[dict[str, Any]] = []
    for left, right in zip(reference["epochs"], candidate["epochs"]):
        for key in ("optimizer_sha256", "scaler_sha256", "position_digest"):
            if left.get(key) != right.get(key):
                epoch_problems.append(
                    {"epoch": left["epoch"], "field": key}
                )
    if digest_mismatches:
        problems.append({"issue": "input_digests", "count": len(digest_mismatches),
                         "examples": digest_mismatches[:5]})
    if gradient_mismatches:
        problems.append({"issue": "gradient_digests", "count": len(gradient_mismatches),
                         "examples": gradient_mismatches[:5]})
    if parameter_mismatches:
        problems.append({"issue": "per_batch_parameters", "count": len(parameter_mismatches),
                         "examples": parameter_mismatches[:5]})
    if epoch_problems:
        problems.append({"issue": "epoch_state", "count": len(epoch_problems),
                         "examples": epoch_problems[:5]})
    losses = [row["abs_delta"] for row in loss_deltas]
    return {
        "reference": reference["mode"],
        "candidate": candidate["mode"],
        "batches_compared": len(loss_deltas),
        "initial_model_identical": reference["initial_model_sha256"]
        == candidate["initial_model_sha256"],
        "input_digest_mismatches": len(digest_mismatches),
        "gradient_digest_mismatches": len(gradient_mismatches),
        "per_batch_parameter_mismatches": len(parameter_mismatches),
        "loss_exact_matches": sum(1 for value in losses if value == 0.0),
        "loss_max_abs_delta": max(losses) if losses else 0.0,
        "loss_mean_abs_delta": statistics.fmean(losses) if losses else 0.0,
        "final_model_sha256_reference": reference["final_model_sha256"],
        "final_model_sha256_candidate": candidate["final_model_sha256"],
        "final_model_identical": reference["final_model_sha256"]
        == candidate["final_model_sha256"],
        "final_state_tensors_differing": different_tensors,
        "final_state_max_abs_delta": max_tensor_delta,
        "epoch_state_problems": epoch_problems,
        "problems": problems,
        # Equivalence claim: identical inputs *and* identical optimiser trajectory.
        "inputs_identical": not digest_mismatches and reference["initial_model_sha256"]
        == candidate["initial_model_sha256"],
        "trajectory_identical": not (
            gradient_mismatches or parameter_mismatches or epoch_problems
        )
        and reference["final_model_sha256"] == candidate["final_model_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Same-version 0/2/4 real training-step equivalence (STAGE1-IO-OPT R3)."
    )
    parser.add_argument("--windows", type=int, default=48, help="frozen train windows per epoch")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--training-seed", type=int, default=2027)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="enable torch.use_deterministic_algorithms to separate CUDA nondeterminism",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if sha256_file(drv.INDEX) != drv.INDEX_SHA256:
        raise RuntimeError("the frozen window index hash changed; refusing to compare")
    if args.torch_threads is not None:
        torch.set_num_threads(int(args.torch_threads))
        os.environ["OMP_NUM_THREADS"] = str(int(args.torch_threads))
    drv.bind_training_seed(args.training_seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if args.device == "cpu":
        device = torch.device("cpu")

    # The position set is fixed once, from the frozen split, and shared by every
    # column: 48 windows in the exact order ``select_positions`` produces.
    probe = drv.AuditedWindowDataset(
        drv.INDEX,
        split=drv.TRAIN_ROLE,
        train_snrs=drv.parse_snr_list(drv.TRAIN_SNRS),
        seed=args.training_seed,
        read_audit=drv.ReadAudit("probe"),
    )
    positions = drv.select_positions(probe.rows, args.windows)
    probe.close()

    columns: dict[str, dict[str, Any]] = {}
    for mode in args.modes:
        started = time.perf_counter()
        print(f"[equivalence] {mode} ...", flush=True)
        column = build_column(positions, mode, device, args.epochs, args.training_seed)
        columns[mode] = column
        print(
            f"[equivalence] {mode}: loss={[round(row['train_loss'], 6) for row in column['epochs']]}"
            f" final={column['final_model_sha256'][:16]} ({time.perf_counter() - started:.1f}s)",
            flush=True,
        )

    reference_mode = args.modes[0]
    reference = columns[reference_mode]
    comparisons = {
        mode: compare_columns(reference, columns[mode])
        for mode in args.modes
        if mode != reference_mode
    }
    for mode, comparison in comparisons.items():
        print(
            f"[equivalence] {reference_mode} vs {mode}: "
            f"inputs_identical={comparison['inputs_identical']} "
            f"trajectory_identical={comparison['trajectory_identical']} "
            f"loss_max_abs_delta={comparison['loss_max_abs_delta']:.3e} "
            f"final_delta={comparison['final_state_max_abs_delta']:.3e}",
            flush=True,
        )

    run_id = f"equiv-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    payload = {
        "schema": "dronerfa-stage1-loader-equivalence/v2",
        "run_id": run_id,
        "generated_at": utc_now(),
        "reference_mode": reference_mode,
        "modes": args.modes,
        "windows_requested": args.windows,
        "windows_used": len(positions),
        "positions": list(positions),
        "epochs": args.epochs,
        "training_seed": args.training_seed,
        "batch_size": drv.BATCH_SIZE,
        "deterministic_algorithms": bool(args.deterministic),
        "index": {"path": str(drv.INDEX), "sha256": sha256_file(drv.INDEX)},
        "device": str(device),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch_threads": torch.get_num_threads(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "use_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "logical_cores": os.cpu_count(),
        },
        "source_sha256": {
            name: sha256_file(SCRIPT_DIR / name)
            for name in (
                "run_stage1_12drones_training.py",
                "stage1_loader_optimization.py",
                "equivalence_stage1_loader.py",
                "dronerfa_data.py",
                "model.py",
                "train_resnet18.py",
            )
        },
        "columns": {
            mode: {
                "mode": column["mode"],
                "num_workers": column["num_workers"],
                "legacy_subset_loader": column["legacy_subset_loader"],
                "initial_model_sha256": column["initial_model_sha256"],
                "final_model_sha256": column["final_model_sha256"],
                "class_weights": column["class_weights"],
                "loader": column["loader"],
                "started_at": column["started_at"],
                "finished_at": column["finished_at"],
                "epochs": column["epochs"],
                "batches": column["batches"],
            }
            for mode, column in columns.items()
        },
        "comparisons": comparisons,
        # The loader's claim is about the *inputs*: every column must consume the very
        # same windows with the very same noise, byte for byte.  Whether the optimiser
        # trajectory is then bit-identical too is a separate question about CUDA
        # nondeterminism, reported next to it rather than hidden inside it.
        "passed": all(
            comparison["inputs_identical"] for comparison in comparisons.values()
        )
        if comparisons
        else False,
        "trajectory_identical_all": all(
            comparison["trajectory_identical"] for comparison in comparisons.values()
        )
        if comparisons
        else False,
    }
    write_json(args.out_dir / f"equivalence_{run_id}.json", payload)
    write_json(args.out_dir / "equivalence_latest.json", payload)

    rows: list[dict[str, Any]] = []
    for mode, comparison in comparisons.items():
        rows.append(
            {
                "reference": comparison["reference"],
                "candidate": comparison["candidate"],
                "batches_compared": comparison["batches_compared"],
                "inputs_identical": comparison["inputs_identical"],
                "trajectory_identical": comparison["trajectory_identical"],
                "input_digest_mismatches": comparison["input_digest_mismatches"],
                "gradient_digest_mismatches": comparison["gradient_digest_mismatches"],
                "per_batch_parameter_mismatches": comparison["per_batch_parameter_mismatches"],
                "loss_exact_matches": comparison["loss_exact_matches"],
                "loss_max_abs_delta": comparison["loss_max_abs_delta"],
                "loss_mean_abs_delta": comparison["loss_mean_abs_delta"],
                "final_model_identical": comparison["final_model_identical"],
                "final_state_tensors_differing": comparison["final_state_tensors_differing"],
                "final_state_max_abs_delta": comparison["final_state_max_abs_delta"],
            }
        )
    with (args.out_dir / "equivalence_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"run_id": run_id, "passed": payload["passed"],
                      "trajectory_identical_all": payload["trajectory_identical_all"],
                      "summary": rows},
                     ensure_ascii=False, indent=2)[:4000], flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
