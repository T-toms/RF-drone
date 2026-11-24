#!/usr/bin/env python3
"""STAGE1-IO-OPT-R2: bounded, real *training-step* throughput benchmark.

What changed in R2
------------------
The R1 run claimed to record process memory, but every configuration ended up as
``{}`` and the summary CSV had no memory column at all.  R2 fixes the measurement
(see "Process memory" below), records parent **and** worker memory, adds the memory
columns to the CSV, and refuses to report a run whose memory sampling failed.

What changed from the first benchmark
-------------------------------------
The first version timed ``torch.inference_mode()`` + ``model.eval()`` forward passes
with no backward, no Adam and no ``GradScaler`` step, so it could not support a claim
about end-to-end training throughput.  It also wrote a JSON without a run id,
environment, thread settings or per-iteration raw records, and its report quoted
numbers that did not match its own CSV.

This version runs the real optimisation step the frozen protocol uses:

* class-weighted cross entropy, AMP float16 autocast, ``scaler.scale(loss).backward()``,
  ``scaler.step(optimizer)``, ``scaler.update()``, Adam lr 1e-3;
* the same bounded window order for every configuration, the same initialisation
  (recorded as a digest and asserted equal across columns), the same batch size 4;
* warm-up iterations that are excluded from the timed range, and a rotated
  configuration order per iteration so a drifting machine cannot favour one column.

Timing honesty
--------------
* ``wall_seconds`` is measured with ``perf_counter`` around the whole iteration and
  includes one final ``torch.cuda.synchronize()``; it is the number used for
  windows/s.
* The per-region split (loader wait / H2D / front-end / fwd+bwd / optimiser step) is
  measured with a ``torch.cuda.synchronize()`` at every region boundary, so it is
  *synchronous region timing*: it removes the CPU/GPU overlap and therefore does not
  add up to ``wall_seconds``.  It answers "where does the work go", not "how fast is
  the pipeline".
* ``gpu_busy_seconds`` comes from CUDA events around the GPU regions on the default
  stream, so it is a CUDA-event busy time, **not** ``nvidia-smi`` utilisation.  The
  JSON says so, and ``--nvidia-smi`` can add an independent optional sample if the
  host exposes ``nvidia-smi``.
* ``seconds_loader_wait`` is the only region that is *not* GPU work; it is the time
  the training loop spent blocked on the next batch.

Process memory (R2)
-------------------
``parent_working_set_mb`` is the working set of *this* Python process (the training
loop) and ``parent_peak_working_set_mb`` is that process's OS high-water mark *since
process start*, both read with Win32 ``GetProcessMemoryInfo``.  These are
**parent-process numbers only**: not total process memory, not machine-wide memory and
not the memory of DataLoader worker processes (no worker/process tree is walked, so
nothing here may be quoted as "total memory").  Samples are taken at iteration start,
epoch start, once mid-epoch (inside the timed region, so its cost is recorded) and at
epoch end (after the wall timer stops).

A failed sample raises ``MemorySamplingError`` and aborts the run instead of writing
an empty mapping that a report could quote as a successful measurement;
``--allow-memory-failure`` records ``ok=false`` plus the OS error text instead.

Usage::

    python benchmark_stage1_io.py --windows 48 --epochs 2 --iterations 3 --torch-threads 2
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
import subprocess
import sys
import threading
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

import run_stage1_12drones_training as drv
from model import ResNet18, SpectrogramFrontEnd
from stage1_loader_optimization import (
    ConsumptionLog,
    LoaderAudit,
    LoaderConfig,
    limited_order,
    order_digest,
    order_for_epoch,
    tensor_digest,
)
from train_resnet18 import seed_everything

OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "stage1_12drones_20260926"
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "outputs" / "stage1_io_optimization_20260927" / "rework1" / "benchmark"
)
DEFAULT_WINDOWS = 48
DEFAULT_EPOCHS = 2
DEFAULT_ITERATIONS = 3
WORKER_CHOICES = (0, 2, 4)


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


MIB = float(1 << 20)
MEMORY_DEFINITION = (
    "Win32 GetProcessMemoryInfo, read with explicit ctypes signatures. These are "
    "PARENT-PROCESS numbers only: parent_working_set_mb is the working set of this "
    "Python process (the training loop) and parent_peak_working_set_mb is its OS "
    "high-water mark since process start, so it can include allocations made before the "
    "benchmark. They are not total process memory, not machine-wide memory, not the RSS "
    "of DataLoader worker processes and not a sum over any process tree."
)


class MemorySamplingError(RuntimeError):
    """The OS process-memory counters could not be read.

    Raised instead of returning an empty mapping, so a failed measurement can never be
    quoted as a successful one.
    """


class Win32MemoryApi:
    """Typed binding of ``GetProcessMemoryInfo`` for the current process.

    Why the explicit signatures matter (R1 bug, reproduced by
    ``test_stage1_benchmark_memory``): the old code called
    ``ctypes.windll.psapi.GetProcessMemoryInfo`` with no ``argtypes``.  ctypes then
    marshalled ``GetCurrentProcess()``'s ``(HANDLE)-1`` pseudo handle as a 32-bit
    ``int``, the callee saw an invalid handle, returned ``FALSE``
    (``ERROR_INVALID_HANDLE``) and the surrounding ``except``/fall-through turned that
    into ``{}`` -- which the report then quoted as "process memory recorded".
    Declaring ``argtypes``/``restype`` makes the pseudo handle survive the call.  The
    documented ``K32GetProcessMemoryInfo`` entry point in ``kernel32`` is preferred,
    with ``psapi`` only as a fallback; the binding is verified by a real read before it
    is accepted.  ``PROCESS_MEMORY_COUNTERS`` is 72 bytes on x64 and is passed as ``cb``
    exactly as the API requires.
    """

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise MemorySamplingError(
                f"process memory counters need Windows, this interpreter reports {sys.platform!r}"
            )
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        self.counters_type = PROCESS_MEMORY_COUNTERS
        self.counters_size = int(ctypes.sizeof(PROCESS_MEMORY_COUNTERS))

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        self._kernel32 = kernel32

        self.get_process_memory_info, self.source = self._bind_memory_info(ctypes, wintypes)

    def _bind_memory_info(self, ctypes: Any, wintypes: Any) -> tuple[Any, str]:
        candidates: list[tuple[str, Any]] = [
            (
                "kernel32.K32GetProcessMemoryInfo",
                getattr(self._kernel32, "K32GetProcessMemoryInfo", None),
            )
        ]
        try:
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
        except OSError as error:  # pragma: no cover - psapi is the fallback only
            candidates.append(("psapi.GetProcessMemoryInfo", None))
            self._psapi_error = repr(error)
        else:
            self._psapi_error = None
            candidates.append(
                ("psapi.GetProcessMemoryInfo", getattr(psapi, "GetProcessMemoryInfo", None))
            )

        problems: list[str] = []
        for source, function in candidates:
            if function is None:
                problems.append(f"{source}: entry point unavailable")
                continue
            function.restype = wintypes.BOOL
            function.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(self.counters_type),
                wintypes.DWORD,
            ]
            try:
                self._read_counters(function, self.current_process_handle(), "binding probe")
            except MemorySamplingError as error:
                problems.append(f"{source}: {error}")
                continue
            return function, source
        detail = "; ".join(problems) if problems else "no candidate entry point"
        raise MemorySamplingError(f"GetProcessMemoryInfo is unusable on this host: {detail}")

    def error_text(self) -> str:
        """Current ``GetLastError`` as ``code (message)`` (the DLLs use_last_error)."""

        code = int(self._ctypes.get_last_error())
        try:
            message = self._ctypes.FormatError(code).strip()
        except Exception:  # pragma: no cover - defensive, formatting must not mask the code
            message = "unknown error text"
        return f"GetLastError={code} ({message})"

    def current_process_handle(self) -> int:
        """Pseudo handle of this process.

        ``GetCurrentProcess`` returns the documented ``(HANDLE)-1`` pseudo handle, which
        has the same numeric value as ``INVALID_HANDLE_VALUE``; it is a valid argument
        for ``GetProcessMemoryInfo`` and must not be rejected here.  The reason it is
        read through a declared ``restype`` is precisely that an undeclared call
        truncates it to 32 bits and makes the API call fail.
        """

        handle = self._kernel32.GetCurrentProcess()
        if handle is None or int(handle) == 0:
            raise MemorySamplingError(f"GetCurrentProcess returned {handle!r}")
        return int(handle)

    def _read_counters(self, function: Any, handle: Any, owner: str) -> dict[str, float]:
        counters = self.counters_type()
        counters.cb = self.counters_size
        if not function(handle, self._ctypes.byref(counters), counters.cb):
            raise MemorySamplingError(f"GetProcessMemoryInfo({owner}) failed: {self.error_text()}")
        working_set = float(counters.WorkingSetSize)
        peak = float(counters.PeakWorkingSetSize)
        if not working_set > 0.0 or not peak > 0.0:
            raise MemorySamplingError(
                f"GetProcessMemoryInfo({owner}) returned non-positive counters "
                f"(working_set={working_set}, peak={peak}); the reading is unusable"
            )
        return {"working_set_mb": working_set / MIB, "peak_working_set_mb": peak / MIB}

    def read_current(self) -> dict[str, float]:
        return self._read_counters(
            self.get_process_memory_info, self.current_process_handle(), "this process"
        )

    def describe(self) -> dict[str, Any]:
        return {
            "api": self.source,
            "use_last_error": True,
            "scope": "parent process only (this interpreter)",
            "process_memory_counters_bytes": self.counters_size,
            "psapi_load_error": self._psapi_error,
        }


_API_LOCK = threading.Lock()
_API: Win32MemoryApi | None = None


def win32_memory_api() -> Win32MemoryApi:
    """Cached, signature-verified Win32 memory binding; raises on any failure."""

    global _API
    with _API_LOCK:
        if _API is None:
            try:
                _API = Win32MemoryApi()
            except MemorySamplingError:
                raise
            except Exception as error:  # pragma: no cover - binding failures are the error
                raise MemorySamplingError(
                    f"cannot bind the Win32 process-memory API: {error!r}"
                ) from error
        return _API


def process_memory_mb() -> dict[str, float]:
    """Working set / peak working set of *this* process, in MiB.

    Returns ``{"parent_working_set_mb": float, "parent_peak_working_set_mb": float}``.
    Raises :class:`MemorySamplingError` with the OS error text when the counters
    cannot be read -- it never returns an empty mapping that a report could mistake
    for a successful measurement.  ``parent_peak_working_set_mb`` is the OS high-water
    mark since process start, so it can include allocations made before the benchmark.
    """

    reading = win32_memory_api().read_current()
    return {
        "parent_working_set_mb": reading["working_set_mb"],
        "parent_peak_working_set_mb": reading["peak_working_set_mb"],
    }


def _max_or_none(samples: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
    return max(values) if values else None


def _memory_summary(
    samples: Sequence[dict[str, Any]],
    errors: Sequence[str],
    *,
    iterations: int,
    strict: bool,
) -> dict[str, Any]:
    """Aggregate parent-process memory samples into the record written to JSON/CSV."""

    return {
        "ok": not errors and all(bool(sample["ok"]) for sample in samples),
        "strict": bool(strict),
        "scope": "parent process only (the training-loop interpreter)",
        "definition": MEMORY_DEFINITION,
        "sample_count": len(samples),
        "iterations_aggregated": int(iterations),
        "samples": list(samples),
        "sample_seconds_total": sum(float(sample["seconds"]) for sample in samples),
        "parent_working_set_mb_max": _max_or_none(samples, "parent_working_set_mb"),
        "parent_peak_working_set_mb_max": _max_or_none(samples, "parent_peak_working_set_mb"),
        "errors": list(errors),
    }


def aggregate_memory(payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fold the per-iteration memory payloads of one configuration into one record."""

    payloads = [payload for payload in payloads if payload]
    if not payloads:
        return {
            "ok": False,
            "strict": None,
            "scope": "parent process only (the training-loop interpreter)",
            "definition": MEMORY_DEFINITION,
            "sample_count": 0,
            "iterations_aggregated": 0,
            "samples": [],
            "sample_seconds_total": 0.0,
            "parent_working_set_mb_max": None,
            "parent_peak_working_set_mb_max": None,
            "errors": ["no memory samples were collected"],
        }
    summary = _memory_summary(
        [sample for payload in payloads for sample in payload.get("samples", [])],
        [error for payload in payloads for error in payload.get("errors", [])],
        iterations=sum(int(payload.get("iterations_aggregated", 1)) for payload in payloads),
        strict=bool(payloads[0].get("strict")),
    )
    summary["ok"] = all(bool(payload.get("ok")) for payload in payloads)
    return summary


class MemoryTracker:
    """Parent-process memory samples with an explicit ok/error status."""

    def __init__(self, *, strict: bool = True) -> None:
        self.strict = bool(strict)
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []

    def sample(self, label: str, *, epoch: int | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        record: dict[str, Any] = {"label": label, "epoch": epoch, "sampled_at": utc_now()}
        ok = True
        try:
            record.update(process_memory_mb())
        except MemorySamplingError as error:
            ok = False
            record["parent_working_set_mb"] = None
            record["parent_peak_working_set_mb"] = None
            record["parent_error"] = str(error)
            self.errors.append(f"{label}: {error}")
        record["seconds"] = time.perf_counter() - started
        record["ok"] = ok
        self.samples.append(record)
        if not ok and self.strict:
            raise MemorySamplingError(
                f"process-memory sampling failed at {label}: {'; '.join(self.errors)}. "
                "Pass --allow-memory-failure to record the failure instead of aborting."
            )
        return record

    def payload(self) -> dict[str, Any]:
        return _memory_summary(self.samples, self.errors, iterations=1, strict=self.strict)


def nvidia_smi_snapshot() -> dict[str, Any]:
    """Optional independent utilisation sample; never required for the result."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if completed.returncode != 0:
            return {"available": False, "error": completed.stderr.strip()[:200]}
        values = [part.strip() for part in completed.stdout.strip().splitlines()[0].split(",")]
        return {
            "available": True,
            "utilization_gpu_percent": float(values[0]),
            "memory_used_mb": float(values[1]),
            "note": "single instantaneous sample, taken outside the timed loop",
        }
    except Exception as error:  # pragma: no cover - optional probe
        return {"available": False, "error": repr(error)[:200]}


def run_iteration(
    index_path: Path,
    positions: Sequence[int],
    workers: int,
    prefetch_factor: int,
    device: torch.device,
    *,
    epochs: int,
    training_seed: int,
    audit_events_enabled: bool = True,
    instrument: bool = False,
    memory_strict: bool = True,
) -> dict[str, Any]:
    """One timed pass over ``positions`` × ``epochs`` with the real training step.

    ``instrument=False`` (the headline measurement) keeps the pipeline asynchronous:
    the only synchronisation is one ``torch.cuda.synchronize()`` after the epoch, so
    prefetching workers can overlap with GPU compute exactly as they would in a real
    run.  ``instrument=True`` adds a ``synchronize()`` at every region boundary,
    which removes that overlap on purpose: it answers "where does the time go" and
    is explicitly *not* the throughput number.

    ``memory_strict=True`` aborts the iteration (raising ``MemorySamplingError``) if the
    parent-process memory sample fails, so no run can be reported with empty memory.
    """

    memory = MemoryTracker(strict=memory_strict)
    config = LoaderConfig(num_workers=workers, prefetch_factor=prefetch_factor)
    context = multiprocessing.get_context("spawn")
    events = context.Queue() if audit_events_enabled else None
    audit = LoaderAudit(f"bench_workers_{workers}")
    audit.set_events(events if audit_events_enabled else None)
    read_audit = drv.ReadAudit(f"bench_workers_{workers}")
    dataset = drv.AuditedWindowDataset(
        index_path,
        split=drv.TRAIN_ROLE,
        train_snrs=drv.parse_snr_list(drv.TRAIN_SNRS),
        seed=training_seed,
        read_audit=read_audit,
    )
    if drv.select_positions(dataset.rows, len(positions)) != list(positions):
        raise RuntimeError("the benchmark window selection drifted")

    seed_everything(training_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = ResNet18(input_channels=1, num_classes=2).to(device)
    initial_hash = drv.state_hash(model.state_dict())
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
    model.train()
    frontend.eval()

    memory.sample("iteration_start")
    epoch_records: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch)
        order = limited_order(order_for_epoch(dataset, epoch, training_seed), positions)
        consumption = ConsumptionLog(phase=drv.TRAIN_ROLE, key=str(epoch), on_batch=audit.drain)
        audit.register_loader(drv.TRAIN_ROLE, str(epoch), config)
        loader = drv.build_train_loader(
            dataset,
            config,
            positions=order,
            key=str(epoch),
            audit_events=events,
            pin_memory=device.type == "cuda",
        )
        memory.sample("epoch_start", epoch=epoch)
        loader_seconds = 0.0
        h2d_seconds = 0.0
        frontend_seconds = 0.0
        model_seconds = 0.0
        step_seconds = 0.0
        windows = 0
        digests: list[dict[str, Any]] = []
        gpu_intervals: list[float] = []
        started = time.perf_counter()
        previous = started
        batch_iterator = drv.consumed_batches(loader, order, consumption)
        for signals, labels in batch_iterator:
            arrived = time.perf_counter()
            loader_seconds += arrived - previous
            digests.append(
                {
                    "batch": len(digests) + 1,
                    "size": int(labels.shape[0]),
                    "signals_sha256": tensor_digest(signals),
                    "labels_sha256": tensor_digest(labels.reshape(-1).to(torch.int64)),
                    "labels": [int(value) for value in labels.tolist()],
                }
            )
            if instrument and device.type == "cuda":
                torch.cuda.synchronize()
            mark = time.perf_counter()
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if instrument and device.type == "cuda":
                torch.cuda.synchronize()
            after_h2d = time.perf_counter()
            h2d_seconds += after_h2d - mark
            start_event = (
                torch.cuda.Event(enable_timing=True)
                if instrument and device.type == "cuda"
                else None
            )
            stop_event = (
                torch.cuda.Event(enable_timing=True)
                if instrument and device.type == "cuda"
                else None
            )
            if start_event is not None:
                start_event.record()
            optimizer.zero_grad(set_to_none=True)
            features = drv.extract_features(frontend, signals)
            if instrument and device.type == "cuda":
                torch.cuda.synchronize()
            after_frontend = time.perf_counter()
            frontend_seconds += after_frontend - after_h2d
            with torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16
            ):
                logits = model(features)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if instrument and device.type == "cuda":
                torch.cuda.synchronize()
            after_model = time.perf_counter()
            model_seconds += after_model - after_frontend
            scaler.step(optimizer)
            scaler.update()
            float(loss.detach())
            if stop_event is not None:
                stop_event.record()
            if instrument and device.type == "cuda":
                # The elapsed time of a CUDA event pair is only defined once both
                # events completed, so the synchronise has to come first.
                torch.cuda.synchronize()
            after_step = time.perf_counter()
            step_seconds += after_step - after_model
            if start_event is not None and stop_event is not None:
                gpu_intervals.append(start_event.elapsed_time(stop_event) / 1000.0)
            windows += int(labels.shape[0])
            previous = time.perf_counter()
            if len(digests) == 1:
                # Sampled once per epoch while the workers are certainly alive and
                # prefetching.  This one is inside the timed region; its cost is the
                # sample's ``seconds`` field and is reported with the results.
                memory.sample("epoch_mid", epoch=epoch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_seconds = time.perf_counter() - started
        # Taken after the wall timer stopped, so it cannot inflate any throughput
        # number; the loader (and therefore its workers) is still alive here.
        memory.sample("epoch_end", epoch=epoch)
        audit.drain()
        audit.record_consumption(consumption)
        audit.close_loader(drv.TRAIN_ROLE, str(epoch))
        report = audit.key_report(drv.TRAIN_ROLE, str(epoch))
        epoch_records.append(
            {
                "epoch": epoch,
                "instrumented": bool(instrument),
                "windows": windows,
                "batches": consumption.batches_delivered,
                "batches_completed": consumption.batches_completed,
                "positions_sha256": order_digest(order),
                "wall_seconds": total_seconds,
                "windows_per_second": windows / max(total_seconds, 1e-9),
                "seconds_loader_wait": loader_seconds,
                "seconds_h2d": h2d_seconds,
                "seconds_frontend": frontend_seconds,
                "seconds_model_fwd_bwd": model_seconds,
                "seconds_optimizer_step": step_seconds,
                "synchronous_region_seconds": (
                    loader_seconds
                    + h2d_seconds
                    + frontend_seconds
                    + model_seconds
                    + step_seconds
                ),
                "gpu_busy_seconds": float(sum(gpu_intervals)),
                "gpu_busy_fraction_of_wall": float(sum(gpu_intervals))
                / max(total_seconds, 1e-9),
                "loss_last_batch": float(loss.detach()),
                "scaler_scale": float(scaler.get_scale()),
                "digests": digests,
                "audit": {
                    "reads": report["reads"]["windows_read"],
                    "completed": report["consumed"]["windows_completed"],
                    "delivered": report["consumed"]["windows_delivered"],
                    "unconsumed_reads": report["unconsumed_reads"]["count"],
                    "unread_consumed": report["unread_consumed"]["count"],
                    "workers": report["workers"],
                    "duplicate_reads": report["reads"]["duplicate_reads"],
                },
            }
        )
        del loader
        audit.close_loader(drv.TRAIN_ROLE, str(epoch))
    final_hash = drv.state_hash(model.state_dict())
    dataset.close()
    del model, optimizer, scaler, frontend
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "num_workers": workers,
        "prefetch_factor": config.effective_prefetch_factor,
        "loader": config.payload(),
        "initial_model_sha256": initial_hash,
        "final_model_sha256": final_hash,
        "epochs": epoch_records,
        "parent_windows_read": read_audit.windows_read,
        "memory": memory.payload(),
    }


def summarize(rows: Sequence[dict[str, Any]], workers: int) -> dict[str, Any]:
    if not rows:
        return {"num_workers": workers, "iterations": 0}
    rates = [row["windows_per_second"] for row in rows]
    regions = {
        "seconds_loader_wait": statistics.fmean(row["seconds_loader_wait"] for row in rows),
        "seconds_h2d": statistics.fmean(row["seconds_h2d"] for row in rows),
        "seconds_frontend": statistics.fmean(row["seconds_frontend"] for row in rows),
        "seconds_model_fwd_bwd": statistics.fmean(row["seconds_model_fwd_bwd"] for row in rows),
        "seconds_optimizer_step": statistics.fmean(row["seconds_optimizer_step"] for row in rows),
    }
    wall = statistics.fmean(row["wall_seconds"] for row in rows)
    busy = statistics.fmean(row["gpu_busy_seconds"] for row in rows)
    return {
        "num_workers": workers,
        "iterations": len(rows),
        "instrumented": bool(rows[0].get("instrumented")),
        "windows_per_second_mean": statistics.fmean(rates),
        "windows_per_second_min": min(rates),
        "windows_per_second_max": max(rates),
        "windows_per_second_stdev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        "wall_seconds_mean": wall,
        "wall_seconds_definition": (
            "asynchronous training loop, synchronised once at the end of the epoch "
            "(the headline throughput number)"
            if not rows[0].get("instrumented")
            else "region-synchronised loop: overlap removed, not a throughput number"
        ),
        "region_seconds": regions,
        "region_seconds_synchronous_note": (
            "each region is bounded by torch.cuda.synchronize(), so these do not sum "
            "to the pipelined wall time"
        ),
        "gpu_busy_seconds_mean": busy,
        "gpu_busy_fraction_of_wall": busy / max(wall, 1e-9),
        "gpu_busy_definition": (
            "CUDA event elapsed time around front-end + fwd/bwd + optimiser step per "
            "batch on the default stream, summed and divided by wall time; this is not "
            "nvidia-smi utilisation"
        ),
        "windows_total": sum(row["windows"] for row in rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded real-training-step loader throughput benchmark."
    )
    parser.add_argument("--windows", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--workers", type=int, nargs="+", default=list(WORKER_CHOICES))
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--training-seed", type=int, default=2027)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="parent torch CPU threads (default: torch's own default = 10 on this host)",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--nvidia-smi", action="store_true", help="add an optional GPU sample")
    parser.add_argument(
        "--audit-events",
        choices=("on", "off"),
        default="on",
        help=(
            "diagnostic only: 'off' removes the per-window worker event stream so the "
            "cost of the identity audit can be separated from the cost of running "
            "several readers. A run with events off has no read evidence and must "
            "never be reported as an accepted throughput configuration"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-memory-failure",
        action="store_true",
        help=(
            "record a failed process-memory sample (memory_sampling.ok=false plus the OS "
            "error text) instead of aborting. The default is to fail loudly, so no run can "
            "report an empty memory record as a successful measurement"
        ),
    )
    args = parser.parse_args(argv)

    if sha256_file(drv.INDEX) != drv.INDEX_SHA256:
        raise RuntimeError("the frozen window index hash changed; refusing to benchmark")
    active = [
        path
        for path in OUTPUT_ROOT.glob("seed_*/progress.json")
        if json.loads(path.read_text(encoding="utf-8")).get("status") == "running"
    ]
    if active:
        raise RuntimeError(
            f"a stage-1 training run is still active ({[str(p) for p in active]}); "
            "benchmark only after it stops so the numbers are not contention artefacts"
        )
    if args.torch_threads is not None:
        if args.torch_threads < 1:
            raise ValueError(f"--torch-threads must be >= 1, got {args.torch_threads}")
        torch.set_num_threads(int(args.torch_threads))
        os.environ["OMP_NUM_THREADS"] = str(int(args.torch_threads))
    drv.bind_training_seed(args.training_seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if args.device == "cpu":
        device = torch.device("cpu")

    probe = drv.AuditedWindowDataset(
        drv.INDEX,
        split=drv.TRAIN_ROLE,
        train_snrs=drv.parse_snr_list(drv.TRAIN_SNRS),
        seed=args.training_seed,
        read_audit=drv.ReadAudit("probe"),
    )
    positions = drv.select_positions(probe.rows, args.windows)
    probe.close()
    if len(positions) < drv.BATCH_SIZE:
        raise RuntimeError(f"need at least {drv.BATCH_SIZE} windows, got {len(positions)}")

    run_id = f"bench-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    started_at = utc_now()
    gpu_before = nvidia_smi_snapshot() if args.nvidia_smi else {"available": False}

    # Fail fast on the memory probe: the R1 run discovered the broken API only after
    # writing a JSON that claimed memory had been recorded.
    memory_strict = not args.allow_memory_failure
    try:
        memory_api = win32_memory_api().describe()
        process_memory_mb()
    except MemorySamplingError as error:
        if memory_strict:
            raise
        memory_api = {"api": None, "error": str(error)}
        print(f"[warning] process-memory sampling is unavailable: {error}", flush=True)

    raw: dict[int, list[dict[str, Any]]] = {workers: [] for workers in args.workers}
    warmups: dict[int, list[dict[str, Any]]] = {workers: [] for workers in args.workers}
    for warmup_index in range(max(args.warmup, 0)):
        order_this_round = list(args.workers[warmup_index % len(args.workers):]) + list(
            args.workers[: warmup_index % len(args.workers)]
        )
        for workers in order_this_round:
            print(f"[warmup {warmup_index + 1}] workers={workers}", flush=True)
            warmups[workers].append(
                run_iteration(
                    drv.INDEX,
                    positions,
                    workers,
                    args.prefetch_factor,
                    device,
                    epochs=1,
                    training_seed=args.training_seed,
                    audit_events_enabled=args.audit_events == "on",
                    instrument=False,
                    memory_strict=memory_strict,
                )
            )
    for iteration in range(1, args.iterations + 1):
        rotated = list(args.workers[(iteration - 1) % len(args.workers):]) + list(
            args.workers[: (iteration - 1) % len(args.workers)]
        )
        for workers in rotated:
            print(f"[iteration {iteration}] workers={workers}", flush=True)
            record = run_iteration(
                drv.INDEX,
                positions,
                workers,
                args.prefetch_factor,
                device,
                epochs=args.epochs,
                training_seed=args.training_seed,
                audit_events_enabled=args.audit_events == "on",
                instrument=False,
                memory_strict=memory_strict,
            )
            record["iteration"] = iteration
            raw[workers].append(record)

    # Region diagnostic: one synchronised, non-overlapping pass per configuration.
    # It is reported separately and never mixed into the throughput statistics.
    region_diagnostics: dict[str, Any] = {}
    for workers in args.workers:
        print(f"[region diagnostic] workers={workers}", flush=True)
        diagnostic = run_iteration(
            drv.INDEX,
            positions,
            workers,
            args.prefetch_factor,
            device,
            epochs=1,
            training_seed=args.training_seed,
            audit_events_enabled=args.audit_events == "on",
            instrument=True,
            memory_strict=memory_strict,
        )
        region_diagnostics[str(workers)] = summarize(diagnostic["epochs"], workers)

    configurations: dict[str, Any] = {}
    for workers in args.workers:
        flattened = [epoch for record in raw[workers] for epoch in record["epochs"]]
        entry = summarize(flattened, workers)
        entry["initial_model_sha256"] = sorted(
            {record["initial_model_sha256"] for record in raw[workers]}
        )
        entry["final_model_sha256"] = sorted(
            {record["final_model_sha256"] for record in raw[workers]}
        )
        entry["prefetch_factor"] = raw[workers][-1]["prefetch_factor"]
        entry["batches"] = sum(epoch["batches"] for epoch in flattened)
        entry["batches_completed"] = sum(epoch["batches_completed"] for epoch in flattened)
        entry["audit"] = {
            "reads": sum(epoch["audit"]["reads"] for epoch in flattened),
            "completed": sum(epoch["audit"]["completed"] for epoch in flattened),
            "delivered": sum(epoch["audit"]["delivered"] for epoch in flattened),
            "unconsumed_reads": sum(epoch["audit"]["unconsumed_reads"] for epoch in flattened),
            "unread_consumed": sum(epoch["audit"]["unread_consumed"] for epoch in flattened),
            "duplicate_reads": sum(epoch["audit"]["duplicate_reads"] for epoch in flattened),
            "workers": sorted({w for epoch in flattened for w in epoch["audit"]["workers"]}),
            "parent_windows_read": sum(record["parent_windows_read"] for record in raw[workers]),
        }
        entry["memory"] = aggregate_memory([record["memory"] for record in raw[workers]])
        entry["iterations_detail"] = [
            {
                "iteration": record["iteration"],
                "epochs": [
                    {
                        "epoch": epoch["epoch"],
                        "windows": epoch["windows"],
                        "windows_per_second": epoch["windows_per_second"],
                        "wall_seconds": epoch["wall_seconds"],
                        "positions_sha256": epoch["positions_sha256"],
                        "batch_digest_sha256": hashlib.sha256(
                            json.dumps(epoch["digests"], sort_keys=True).encode()
                        ).hexdigest(),
                    }
                    for epoch in record["epochs"]
                ],
            }
            for record in raw[workers]
        ]
        configurations[str(workers)] = entry

    baseline = configurations[str(args.workers[0])]["windows_per_second_mean"]
    for workers in args.workers:
        entry = configurations[str(workers)]
        entry["speedup_vs_first_configuration"] = entry["windows_per_second_mean"] / max(
            baseline, 1e-9
        )

    # Equivalence guard: identical inputs across columns, per epoch, per iteration.
    def epoch_digests(record: dict[str, Any]) -> list[str]:
        return [
            hashlib.sha256(json.dumps(epoch["digests"], sort_keys=True).encode()).hexdigest()
            for epoch in record["epochs"]
        ]

    reference_digests = [epoch_digests(record) for record in raw[args.workers[0]]]
    equivalence: dict[str, Any] = {}
    for workers in args.workers:
        candidate_digests = [epoch_digests(record) for record in raw[workers]]
        equivalence[str(workers)] = {
            "digest_lists_identical": candidate_digests == reference_digests,
            "epochs_compared": sum(len(entry) for entry in candidate_digests),
        }
    payload = {
        "schema": "dronerfa-stage1-io-benchmark/v2",
        "run_id": run_id,
        "generated_at": utc_now(),
        "started_at": started_at,
        "finished_at": utc_now(),
        "mode": "real_train_step",
        "audit_events": args.audit_events,
        "audit_events_note": (
            "diagnostic configuration without the worker event stream: it has no read "
            "evidence and is not an acceptable throughput configuration"
            if args.audit_events == "off"
            else "worker read events are recorded as in a real run"
        ),
        "loss": "class-weighted cross entropy (same weights as the frozen run)",
        "optimizer": {"type": "Adam", "lr": drv.LEARNING_RATE, "weight_decay": drv.WEIGHT_DECAY},
        "amp": {"enabled": device.type == "cuda", "dtype": "float16"},
        "grad_scaler": "torch.amp.GradScaler, scale/step/update every batch",
        "index": {"path": str(drv.INDEX), "sha256": sha256_file(drv.INDEX)},
        "windows_requested": args.windows,
        "windows_used": len(positions),
        "positions": list(positions),
        "batch_size": drv.BATCH_SIZE,
        "epochs_per_iteration": args.epochs,
        "iterations": args.iterations,
        "warmup_iterations": args.warmup,
        "warmup_seconds": {
            str(workers): sum(
                epoch["wall_seconds"]
                for record in warmups[workers]
                for epoch in record["epochs"]
            )
            for workers in args.workers
        },
        "configuration_order_per_iteration": [
            list(args.workers[(iteration - 1) % len(args.workers):])
            + list(args.workers[: (iteration - 1) % len(args.workers)])
            for iteration in range(1, args.iterations + 1)
        ],
        "timed_range": "from the first batch request to the final torch.cuda.synchronize()",
        "training_seed": args.training_seed,
        "device": str(device),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch_threads_requested": args.torch_threads,
            "torch_threads_actual": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "logical_cores": os.cpu_count(),
        },
        "source_sha256": {
            name: sha256_file(SCRIPT_DIR / name)
            for name in (
                "benchmark_stage1_io.py",
                "stage1_loader_optimization.py",
                "run_stage1_12drones_training.py",
                "dronerfa_data.py",
                "model.py",
                "train_resnet18.py",
            )
        },
        "configurations": configurations,
        "memory_sampling": {
            "ok": all(
                bool(configurations[str(workers)]["memory"]["ok"]) for workers in args.workers
            ),
            "strict": memory_strict,
            "scope": "parent process only (the training-loop interpreter)",
            "fields": {
                "parent_working_set_mb": "working set of this process, MiB",
                "parent_peak_working_set_mb": (
                    "this process's OS high-water mark since process start, MiB"
                ),
            },
            "not_measured": (
                "DataLoader worker memory: no worker/process tree is walked, so these "
                "numbers must not be quoted as total process or machine memory"
            ),
            "definition": MEMORY_DEFINITION,
            "api": memory_api,
            "sampling_points": [
                "iteration_start",
                "epoch_start",
                "epoch_mid (inside the timed region; cost recorded per sample)",
                "epoch_end (after the wall timer stops)",
            ],
            "failures": sorted(
                {
                    error
                    for workers in args.workers
                    for error in configurations[str(workers)]["memory"]["errors"]
                }
            ),
        },
        "region_diagnostics": region_diagnostics,
        "equivalence": equivalence,
        "gpu_sample_before": gpu_before,
        "gpu_sample_after": nvidia_smi_snapshot() if args.nvidia_smi else {"available": False},
        "notes": [
            "windows/s uses the asynchronous pipelined wall time (one synchronise per "
            "epoch); the per-region breakdown comes from a separate synchronised pass "
            "and therefore does not sum to it",
            "gpu_busy_fraction is a CUDA-event measure with its own definition, not "
            "nvidia-smi utilisation",
            "the page cache cannot be flushed from Python, so the first iteration is the "
            "only 'cold' one; warm-up iterations are excluded from the summary",
            "inputs_identical_to_first compares the per-batch signal/label digests of "
            "every configuration against the first one, so a fast-but-wrong loader "
            "cannot be reported as a win",
            "memory columns come from Win32 GetProcessMemoryInfo with explicit ctypes "
            "signatures and are PARENT-PROCESS numbers only (this interpreter): they are "
            "not total process memory, not machine-wide memory and not worker memory; "
            "memory_sampling.ok=false would mean a sample failed and no memory number "
            "may be quoted from this run",
        ],
    }
    if not payload["memory_sampling"]["ok"]:
        payload["notes"].append(
            "MEMORY SAMPLING FAILED for at least one configuration "
            f"({payload['memory_sampling']['failures']}); the memory fields of this run are "
            "not usable evidence"
        )
        print(
            "[warning] memory sampling failed; see memory_sampling.failures in the JSON: "
            f"{payload['memory_sampling']['failures']}",
            flush=True,
        )
    write_json(args.out_dir / f"benchmark_{run_id}.json", payload)
    write_json(args.out_dir / "benchmark_latest.json", payload)

    rows: list[dict[str, Any]] = []
    for workers in args.workers:
        entry = configurations[str(workers)]
        diagnostic = region_diagnostics.get(str(workers), {})
        diagnostic_regions = diagnostic.get("region_seconds", {})
        memory = entry["memory"]
        rows.append(
            {
                "num_workers": workers,
                "prefetch_factor": entry["prefetch_factor"],
                "windows_per_second_mean": entry["windows_per_second_mean"],
                "windows_per_second_stdev": entry["windows_per_second_stdev"],
                "windows_per_second_min": entry["windows_per_second_min"],
                "windows_per_second_max": entry["windows_per_second_max"],
                "wall_seconds_mean": entry["wall_seconds_mean"],
                "speedup_vs_first_configuration": entry["speedup_vs_first_configuration"],
                "diag_seconds_loader_wait": diagnostic_regions.get("seconds_loader_wait", ""),
                "diag_seconds_h2d": diagnostic_regions.get("seconds_h2d", ""),
                "diag_seconds_frontend": diagnostic_regions.get("seconds_frontend", ""),
                "diag_seconds_model_fwd_bwd": diagnostic_regions.get("seconds_model_fwd_bwd", ""),
                "diag_seconds_optimizer_step": diagnostic_regions.get("seconds_optimizer_step", ""),
                "diag_gpu_busy_fraction_of_wall": diagnostic.get("gpu_busy_fraction_of_wall", ""),
                "windows_read": entry["audit"]["reads"],
                "windows_completed": entry["audit"]["completed"],
                "unconsumed_reads": entry["audit"]["unconsumed_reads"],
                "parent_windows_read": entry["audit"]["parent_windows_read"],
                "worker_processes": len(entry["audit"]["workers"]),
                "parent_working_set_mb": memory["parent_working_set_mb_max"],
                "parent_peak_working_set_mb": memory["parent_peak_working_set_mb_max"],
                "memory_scope": memory["scope"],
                "memory_samples": memory["sample_count"],
                "memory_sample_seconds": memory["sample_seconds_total"],
                "memory_sampling_ok": memory["ok"],
                "inputs_identical_to_first": equivalence[str(workers)]["digest_lists_identical"],
                "run_id": run_id,
                "audit_events": args.audit_events,
                "torch_threads": torch.get_num_threads(),
            }
        )
    with (args.out_dir / "benchmark_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "output": str(args.out_dir),
                "torch_threads": torch.get_num_threads(),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
