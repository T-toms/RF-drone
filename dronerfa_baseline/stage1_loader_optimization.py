#!/usr/bin/env python3
"""STAGE1-IO-OPT: an opt-in parallel window loader for the frozen stage-1 protocol.

The accepted STAGE1-12DRONES runs read every 1,000,000-sample window through a
single ``num_workers=0`` DataLoader: one CPU thread does the HDF5 read, the RMS
normalisation and the AWGN draw while the GPU waits, which is why the observed GPU
utilisation sits in the low tens of percent.  This module keeps that serial path as
the untouched reference (``num_workers=0``, ``prefetch_factor=None`` -- the default
of every existing entry point) and adds an equivalent parallel path.

What is preserved bit for bit
-----------------------------
* window geometry and order: the same ``window_index``/``start`` rows, and the same
  per-epoch shuffle ``torch.randperm(len(split), generator=manual_seed(seed + epoch))``
  applied to the very same split length;
* the noise rule ``numpy.random.default_rng(seed + epoch * len(split) + position)``
  with the training-selection draw consumed before the AWGN samples -- so position
  *and* epoch both matter, and the epoch has to reach every worker;
* the preprocessed tensors the model consumes: ``torch.from_numpy`` of the
  complex64 ``RF0_I + 1j*RF0_Q`` window after mean removal, RMS normalisation and
  the same AWGN draw (CPU float32, no cast);
* batch size, and the fact that a shuffle exposes every selected window once.

Identity model (rework R1)
--------------------------
Reads and consumption are both keyed by the same triple

    ``(phase, key, underlying split position)``

where ``phase`` is ``train`` / ``val`` and ``key`` is the epoch number (``"1"`` ..)
for training or the validation condition (``clean`` / ``-10dB`` / ...) for the
validation cache.  A "position" is always an index into the *split's own row table*
(``DroneRFaWindowDataset.rows``), never a slot of some view: the loader is handed a
``PositionView`` whose ``__getitem__`` takes the underlying position directly, so a
sampler index and a read index are the same number by construction and the slot /
position mismatch of the first implementation cannot come back.

* worker events record real opens and real reads (emitted only after the read
  succeeded) with their phase/key/position, are grouped by ``worker<id>@pid<pid>``
  instead of a single ``main`` bucket, and are aggregated per key -- so two epochs
  or two conditions touching the same position no longer collapse into one set;
* the parent records what it *delivered* and what it *completed* separately, in
  order, per batch: ``delivered`` is the batch the loader produced, ``completed``
  is a batch whose training step ran to the end.  A batch that was delivered but
  whose step raised stays in ``delivered`` only, which is the honest failure
  evidence;
* a full traversal closes exactly: every read is a consumed window and every
  consumed window was read, so ``unconsumed_reads == undelivered_reads == 0``.
  Reads left over after an abandoned epoch are reported as such instead of being
  hidden by a ``range()`` estimate (the previous implementation's bug);
* per-key order is preserved as a SHA256 of the ordered position list plus, for
  short runs, a full CSV of the delivered/completed positions.

Worker epoch propagation
------------------------
Workers are deliberately **not** persistent.  The dataset holds a mutable ``epoch``
attribute and a persistent worker would keep its own stale copy.  The loader is
therefore rebuilt once per epoch *after* ``set_epoch``, and once per validation
condition, which under the Windows ``spawn`` start method also gives every worker a
freshly unpickled dataset that opens its own lazy HDF5 handles.  No open handle is
ever inherited across processes.

Real I/O and GPU timings are measured separately, in ``benchmark_stage1_io.py``:
the point of this module is equivalence and auditability first, speed second.
"""

from __future__ import annotations

import csv
import hashlib
import queue as queue_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

#: Reference loader configuration; every existing entry point keeps this.
REFERENCE_NUM_WORKERS = 0
REFERENCE_PREFETCH_FACTOR: int | None = None
#: Worker counts the task asks to compare.
CANDIDATE_NUM_WORKERS = (0, 2, 4)
#: The frozen batch size; the parallel path must not change the optimisation step.
BATCH_SIZE = 4

TRAIN_PHASE = "train"
VAL_PHASE = "val"


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoaderConfig:
    """Validated loader settings.

    ``prefetch_factor`` is forced to ``None`` for ``num_workers=0`` because PyTorch
    rejects the argument in that case; ``None`` therefore means "the reference
    semantics" and is reported as such rather than silently defaulted.
    """

    num_workers: int = REFERENCE_NUM_WORKERS
    prefetch_factor: int | None = REFERENCE_PREFETCH_FACTOR

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.num_workers == 0:
            if self.prefetch_factor is not None:
                object.__setattr__(self, "prefetch_factor", None)
        elif self.prefetch_factor is None:
            object.__setattr__(self, "prefetch_factor", 2)
        elif self.prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1, got {self.prefetch_factor}")

    @property
    def parallel(self) -> bool:
        return self.num_workers > 0

    @property
    def effective_prefetch_factor(self) -> int | None:
        """What is actually handed to ``DataLoader`` (``None`` when serial)."""
        return self.prefetch_factor if self.parallel else None

    def payload(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "prefetch_factor": self.effective_prefetch_factor,
            "requested_prefetch_factor": self.prefetch_factor,
            "parallel": self.parallel,
            "persistent_workers": False,
            "start_method": "spawn",
            "epoch_propagation": (
                "loader rebuilt after set_epoch (non-persistent workers); the "
                "validation cache rebuilds one loader per condition"
            ),
            "identity_model": "(phase, key, underlying_split_position)",
        }


def loader_from_args(num_workers: int, prefetch_factor: int | None) -> LoaderConfig:
    """Build a :class:`LoaderConfig` from CLI values."""
    return LoaderConfig(num_workers=int(num_workers), prefetch_factor=prefetch_factor)


# ---------------------------------------------------------------------------
# identity helpers
# ---------------------------------------------------------------------------
def identity_digest(pairs: Sequence[tuple[str, int]]) -> str:
    """SHA256 over an ordered ``(path, position)`` sequence."""

    digest = hashlib.sha256()
    for path, position in pairs:
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(int(position).to_bytes(8, "little", signed=True))
        digest.update(b"\x1f")
    return digest.hexdigest()


def order_digest(positions: Sequence[int]) -> str:
    """SHA256 over an ordered position list (delivery order preserved)."""

    array = np.asarray([int(value) for value in positions], dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def multiset_digest(positions: Sequence[int]) -> str:
    """SHA256 over ``(position, count)`` pairs, so duplicates cannot be hidden."""

    counts: dict[int, int] = {}
    for value in positions:
        key = int(value)
        counts[key] = counts.get(key, 0) + 1
    digest = hashlib.sha256()
    for key in sorted(counts):
        digest.update(key.to_bytes(8, "little", signed=True))
        digest.update(counts[key].to_bytes(4, "little"))
    return digest.hexdigest()


def duplicate_count(positions: Sequence[int]) -> int:
    """How many entries are repeats of an earlier entry (0 for a strict pass)."""

    seen: set[int] = set()
    repeats = 0
    for value in positions:
        key = int(value)
        if key in seen:
            repeats += 1
        seen.add(key)
    return repeats


def counter_delta(left: Sequence[int], right: Sequence[int], limit: int = 20) -> dict[str, Any]:
    """``left - right`` as a multiset, with a bounded example list."""

    counts: dict[int, int] = {}
    for value in right:
        key = int(value)
        counts[key] = counts.get(key, 0) + 1
    missing: list[int] = []
    for value in left:
        key = int(value)
        if counts.get(key, 0) > 0:
            counts[key] -= 1
        else:
            missing.append(key)
    return {"count": len(missing), "examples": missing[:limit]}


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
#: Event kinds a worker dataset can emit.
OPEN_EVENT = "open"
WINDOW_EVENT = "window"


def worker_identity() -> str:
    """A stable per-process tag: ``worker<id>`` inside a worker, ``main`` outside."""

    info = torch.utils.data.get_worker_info()
    if info is None:
        return "main"
    return f"worker{info.id}"


def _current_pid() -> int:
    import os

    return os.getpid()


class WorkerAuditDataset(Dataset):
    """Wrap a split dataset and report real file opens / window reads.

    The wrapper measures, it does not predict: events are emitted from inside
    ``__getitem__`` after the underlying read succeeded, so a failed read never
    shows up as a read.  ``index`` is an underlying split position, because a
    :class:`PositionView` above this wrapper passes positions through unchanged.

    ``phase`` and ``key`` are set *before* the ``DataLoader`` is built, so the
    pickled worker copy carries the identity of the epoch or validation condition
    it is about to read for.
    """

    def __init__(
        self,
        dataset: Dataset,
        events: "queue_module.Queue | None" = None,
        phase: str = "",
        key: str = "",
        path_rows: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.dataset = dataset
        self._events = events
        self.phase = str(phase)
        self.key = str(key)
        self._opened: set[str] = set()
        if path_rows is not None:
            self._path_rows: Sequence[dict[str, Any]] = path_rows
        else:
            own = getattr(dataset, "path_rows", None)
            if own is None:
                own = getattr(dataset, "rows", None)
            if own is None:
                raise TypeError(
                    "WorkerAuditDataset needs the underlying row table: pass path_rows "
                    "when wrapping a subset view"
                )
            self._path_rows = own

    # -- identity passthrough -------------------------------------------------
    @property
    def rows(self) -> list[dict[str, Any]]:
        return getattr(self.dataset, "rows")

    @property
    def path_rows(self) -> Sequence[dict[str, Any]]:
        return self._path_rows

    @property
    def epoch(self) -> int:
        return getattr(self.dataset, "epoch", 0)

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.dataset, "set_epoch", None)
        if setter is not None:
            setter(int(epoch))

    def set_identity(self, phase: str, key: str) -> None:
        self.phase = str(phase)
        self.key = str(key)

    def close(self) -> None:
        closer = getattr(self.dataset, "close", None)
        if closer is not None:
            closer()

    def __len__(self) -> int:
        return len(self.dataset)

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self._events is None:
            return
        try:
            self._events.put(
                {
                    "kind": kind,
                    "worker": worker_identity(),
                    "pid": _current_pid(),
                    "phase": self.phase,
                    "key": self.key,
                    **payload,
                }
            )
        except Exception:  # pragma: no cover - a broken queue must not fail the epoch
            pass

    def __getitem__(self, index: int):
        row = self._path_rows[int(index)]
        path = str(row["file_path"])
        if path not in self._opened:
            self._opened.add(path)
            self._emit(OPEN_EVENT, {"file": path})
        value = self.dataset[int(index)]
        self._emit(
            WINDOW_EVENT,
            {"file": path, "position": int(index), "filename": str(row.get("filename", ""))},
        )
        return value


class PositionView(Dataset):
    """Index a split dataset by *underlying split position*.

    ``__len__`` still reports the full split length because the frozen AWGN rule
    ``seed + epoch * len(split) + position`` is defined on it, while
    ``__getitem__`` forwards the position unchanged.  That is what keeps a
    sampler index, a read index and an audit position the same number.
    """

    def __init__(self, dataset: Dataset, path_rows: Sequence[dict[str, Any]] | None = None) -> None:
        self.dataset = dataset
        if path_rows is not None:
            self.path_rows: Sequence[dict[str, Any]] = path_rows
        else:
            own = getattr(dataset, "path_rows", None)
            if own is None:
                own = getattr(dataset, "rows", None)
            if own is None:
                raise TypeError("PositionView needs the underlying row table")
            self.path_rows = own

    @property
    def rows(self) -> list[dict[str, Any]]:
        return getattr(self.dataset, "rows")

    @property
    def epoch(self) -> int:
        return getattr(self.dataset, "epoch", 0)

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.dataset, "set_epoch", None)
        if setter is not None:
            setter(int(epoch))

    def set_identity(self, phase: str, key: str) -> None:
        setter = getattr(self.dataset, "set_identity", None)
        if setter is not None:
            setter(phase, key)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, position: int):
        return self.dataset[int(position)]

    def close(self) -> None:
        closer = getattr(self.dataset, "close", None)
        if closer is not None:
            closer()


class MemmapWindowView(Dataset):
    """Validation feature cache backed by a memory-mapped ``.npy`` pair.

    Pickling a ``TensorDataset`` holding 2,360 x 1 x 256 x 256 float32 tensors per
    worker would cost more than the epoch saves, so the cache is written to disk
    once and memory-mapped per worker instead.  The returned tensors are float32
    and element-wise identical to the in-memory ones.
    """

    def __init__(self, features_path: Path, labels_path: Path) -> None:
        self.features_path = Path(features_path)
        self.labels_path = Path(labels_path)
        self._features: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def _load(self) -> None:
        if self._features is None:
            self._features = np.load(self.features_path, mmap_mode="r")
            self._labels = np.load(self.labels_path, mmap_mode="r")

    def __len__(self) -> int:
        self._load()
        return int(self._features.shape[0])

    def __getitem__(self, index: int):
        self._load()
        feature = np.array(self._features[index], dtype=np.float32, copy=True)
        label = int(self._labels[index])
        return torch.from_numpy(feature), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# consumption log (what the parent really consumed)
# ---------------------------------------------------------------------------
@dataclass
class ConsumptionLog:
    """Ordered record of what the training loop delivered and completed.

    ``delivered`` grows when the loader hands a batch over; ``completed`` grows
    only after the consumer's body for that batch finished, i.e. after the
    optimiser step ran.  A batch killed by an exception therefore shows up as
    delivered-but-not-completed, and an abandoned epoch shows up as a short
    ``completed`` list rather than a silently full one.
    """

    phase: str = ""
    key: str = ""
    delivered: list[int] = field(default_factory=list)
    completed: list[int] = field(default_factory=list)
    batches_delivered: int = 0
    batches_completed: int = 0
    aborted: bool = False
    iterator_closed: bool = False
    #: True only when the sampler order was consumed to its end.
    exhausted: bool = False
    #: Optional hook run after every delivered batch; the driver uses it to pull
    #: worker audit events out of the queue while the workers are still busy, so a
    #: long epoch cannot overflow an unread queue.
    on_batch: "Callable[[], None] | None" = None
    _open_batch: list[int] = field(default_factory=list)

    def note_delivered(self, positions: Sequence[int]) -> None:
        batch = [int(position) for position in positions]
        self._open_batch = batch
        self.delivered.extend(batch)

    def batch_delivered(self) -> None:
        self.batches_delivered += 1
        if self.on_batch is not None:
            self.on_batch()

    def batch_completed(self) -> None:
        self.completed.extend(self._open_batch)
        self._open_batch = []
        self.batches_completed += 1

    def payload(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "key": self.key,
            "identity": f"{self.phase}/{self.key}",
            "batches_delivered": self.batches_delivered,
            "batches_completed": self.batches_completed,
            "windows_delivered": len(self.delivered),
            "windows_completed": len(self.completed),
            "delivered_order_sha256": order_digest(self.delivered),
            "completed_order_sha256": order_digest(self.completed),
            "delivered_multiset_sha256": multiset_digest(self.delivered),
            "completed_multiset_sha256": multiset_digest(self.completed),
            "aborted": bool(self.aborted),
            "iterator_closed": bool(self.iterator_closed),
            "exhausted": bool(self.exhausted),
        }


def consumed_batches(
    loader: DataLoader,
    sampler_positions: Sequence[int],
    consumption: ConsumptionLog,
) -> Iterator[Any]:
    """Iterate ``loader`` and record the positions it delivered, in order.

    ``DataLoader`` does not expose the indices of the batch it just produced, so
    the batch is mapped back through the sampler order that built it:
    ``sampler_positions[cursor : cursor + size]``.  With a :class:`PositionView`
    those are underlying split positions, not view slots.

    Completion is recorded when the consumer asks for the *next* batch, i.e. once
    the body it ran for this batch returned; a consumer that stops early (or whose
    body raises) leaves that batch as delivered-but-not-completed, which is the
    honest evidence of an interrupted epoch.
    """

    cursor = 0
    try:
        for batch in loader:
            size = int(batch[1].shape[0])
            batch_positions = sampler_positions[cursor : cursor + size]
            if len(batch_positions) != size:
                raise RuntimeError(
                    "the loader produced more windows than the sampler order contains"
                )
            cursor += size
            consumption.note_delivered(batch_positions)
            consumption.batch_delivered()
            yield batch
            # Runs when the consumer asks for the next batch, i.e. after its body
            # (and therefore the optimiser step) finished.
            consumption.batch_completed()
        consumption.exhausted = cursor == len(sampler_positions)
    except GeneratorExit:
        # The consumer stopped early (break / garbage collection): a real abort.
        consumption.aborted = True
        raise
    except BaseException:
        consumption.aborted = True
        raise
    finally:
        consumption.iterator_closed = True


# ---------------------------------------------------------------------------
# samplers and loaders
# ---------------------------------------------------------------------------
def order_for_epoch(dataset: Dataset, epoch: int, seed: int) -> list[int]:
    """The frozen per-epoch shuffle, materialised over the *full* split length."""

    return torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(int(seed) + int(epoch))
    ).tolist()


def limited_order(
    full_order: Sequence[int], selected_positions: Sequence[int]
) -> list[int]:
    """Keep the frozen shuffle order, restricted to the positions in use.

    ``IndexSubset``-style bounded runs permute the full split and then keep the
    selected rows, so the surviving order is a filtered permutation -- never a
    fresh shuffle of the subset.
    """

    allowed = {int(position) for position in selected_positions}
    return [int(position) for position in full_order if int(position) in allowed]


def _build_dataset(
    dataset: Dataset,
    *,
    phase: str,
    key: str,
    audit_events: "queue_module.Queue | None",
) -> Dataset:
    """Audit wrapper below a position view, so events carry real positions."""

    path_rows = getattr(dataset, "path_rows", None) or getattr(dataset, "rows", None)
    if path_rows is None:
        raise TypeError("cannot audit a dataset without a row table")
    if audit_events is None:
        return PositionView(dataset, path_rows=path_rows)
    wrapped = WorkerAuditDataset(
        dataset, events=audit_events, phase=phase, key=key, path_rows=path_rows
    )
    return PositionView(wrapped, path_rows=path_rows)


def _loader_kwargs(config: LoaderConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"batch_size": int(BATCH_SIZE), "shuffle": False}
    worker_init: Callable[..., None] | None = None
    if config.parallel:
        kwargs["num_workers"] = config.num_workers
        kwargs["prefetch_factor"] = config.effective_prefetch_factor
        kwargs["persistent_workers"] = False
        worker_init = _worker_init_fn
    return {"kwargs": kwargs, "worker_init": worker_init}


def build_train_loader(
    dataset: Dataset,
    config: LoaderConfig,
    *,
    positions: Sequence[int],
    phase: str = TRAIN_PHASE,
    key: str = "",
    audit_events: "queue_module.Queue | None" = None,
    pin_memory: bool = False,
) -> DataLoader:
    """Training loader over an explicit *underlying position* order.

    ``positions`` already encodes the epoch's shuffle, so the loader itself must
    not shuffle again; that keeps the serial and parallel paths on one code path.
    """

    view = _build_dataset(dataset, phase=phase, key=key, audit_events=audit_events)
    built = _loader_kwargs(config)
    return DataLoader(
        view,
        sampler=[int(position) for position in positions],
        pin_memory=bool(pin_memory),
        worker_init_fn=built["worker_init"],
        **built["kwargs"],
    )


def build_cache_loader(
    dataset: Dataset,
    config: LoaderConfig,
    *,
    positions: Sequence[int],
    batch_size: int,
    phase: str = VAL_PHASE,
    key: str = "",
    audit_events: "queue_module.Queue | None" = None,
    pin_memory: bool = False,
) -> DataLoader:
    """Validation cache loader: the selected positions, once, in selection order."""

    view = _build_dataset(dataset, phase=phase, key=key, audit_events=audit_events)
    built = _loader_kwargs(config)
    kwargs = built["kwargs"]
    kwargs["batch_size"] = int(batch_size)
    return DataLoader(
        view,
        sampler=[int(position) for position in positions],
        pin_memory=bool(pin_memory),
        worker_init_fn=built["worker_init"],
        **kwargs,
    )


def _worker_init_fn(worker_id: int) -> None:  # pragma: no cover - runs in a child
    """One compute thread per worker; the parent keeps torch's default."""

    del worker_id
    torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# audit aggregation
# ---------------------------------------------------------------------------
@dataclass
class WorkerEventTally:
    """Per worker-process event tally, keyed by ``worker<id>@pid<pid>``."""

    tag: str = ""
    pid: int = -1
    opens: dict[tuple[str, str, str], int] = field(default_factory=dict)
    reads: dict[tuple[str, str, str, int], int] = field(default_factory=dict)
    read_order: list[tuple[str, str, str, int]] = field(default_factory=list)
    phases: set[str] = field(default_factory=set)
    keys: set[tuple[str, str]] = field(default_factory=set)

    def payload(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "pid": self.pid,
            "opens": int(sum(self.opens.values())),
            "reads": int(sum(self.reads.values())),
            "phases": sorted(self.phases),
            "keys": sorted(f"{phase}/{key}" for phase, key in self.keys),
        }


class LoaderAudit:
    """Parent-side aggregation of worker events plus parent consumption.

    Everything is keyed by ``(phase, key)`` so a training epoch, a second training
    epoch over the same positions, and a validation condition over the same
    positions never merge into one set.

    ``opened_files`` / ``windows_by_file`` / ``windows_read`` / ``open_events`` keep
    the legacy aggregate field names, but they are now computed *from the recorded
    events* rather than from a predicted range.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.events: "queue_module.Queue | None" = None
        self.tallies: dict[str, WorkerEventTally] = {}
        #: (phase, key) -> aggregated read/consumption state.
        self.keys: dict[tuple[str, str], dict[str, Any]] = {}
        self.opens: dict[tuple[str, str], dict[str, int]] = {}
        self.loaders: list[dict[str, Any]] = []
        self.closed_loaders: list[str] = []
        self.drain_failures = 0
        self.unknown_events = 0

    # -- wiring ---------------------------------------------------------------
    def set_events(self, events: "queue_module.Queue | None") -> None:
        self.events = events

    def register_loader(self, phase: str, key: str, config: LoaderConfig) -> None:
        self.loaders.append(
            {
                "phase": str(phase),
                "key": str(key),
                "identity": f"{phase}/{key}",
                "num_workers": config.num_workers,
                "prefetch_factor": config.effective_prefetch_factor,
            }
        )

    def close_loader(self, phase: str, key: str) -> int:
        """Drain after a loader was destroyed; returns the number of drained events."""

        drained = self.drain()
        identity = f"{phase}/{key}"
        if identity not in self.closed_loaders:
            self.closed_loaders.append(identity)
        return drained

    def _key_state(self, phase: str, key: str) -> dict[str, Any]:
        state = self.keys.get((str(phase), str(key)))
        if state is None:
            state = {
                "phase": str(phase),
                "key": str(key),
                "reads": [],  # ordered (path, position)
                "consumed_delivered": [],
                "consumed_completed": [],
                "batches_delivered": 0,
                "batches_completed": 0,
                "aborted": False,
                "iterator_closed": False,
                "exhausted": False,
                "workers": set(),
                "pids": set(),
            }
            self.keys[(str(phase), str(key))] = state
        return state

    # -- worker-side path ------------------------------------------------------
    def drain(self) -> int:
        """Move every queued worker event into the parent tally; returns the count."""

        if self.events is None:
            return 0
        drained = 0
        while True:
            try:
                event = self.events.get_nowait()
            except queue_module.Empty:
                break
            except Exception:  # pragma: no cover - closed queue
                self.drain_failures += 1
                break
            kind = str(event.get("kind", ""))
            phase = str(event.get("phase", ""))
            key = str(event.get("key", ""))
            if phase == "" or key == "":
                self.unknown_events += 1
                drained += 1
                continue
            state = self._key_state(phase, key)
            tag = str(event.get("worker", "unknown"))
            pid = int(event.get("pid", -1))
            worker_key = f"{tag}@pid{pid}"
            tally = self.tallies.get(worker_key)
            if tally is None:
                tally = WorkerEventTally(tag=tag, pid=pid)
                self.tallies[worker_key] = tally
            tally.phases.add(phase)
            tally.keys.add((phase, key))
            state["workers"].add(worker_key)
            if pid > 0:
                state["pids"].add(pid)
            if kind == OPEN_EVENT:
                path = str(event["file"])
                slot = (phase, key, path)
                tally.opens[slot] = tally.opens.get(slot, 0) + 1
                per_phase = self.opens.setdefault((phase, key), {})
                per_phase[path] = per_phase.get(path, 0) + 1
            elif kind == WINDOW_EVENT:
                path = str(event["file"])
                position = int(event["position"])
                slot = (phase, key, path, position)
                tally.reads[slot] = tally.reads.get(slot, 0) + 1
                tally.read_order.append(slot)
                state["reads"].append((path, position))
            else:
                self.unknown_events += 1
            drained += 1
        return drained

    def record_consumption(self, consumption: ConsumptionLog) -> dict[str, Any]:
        """Attach one loader pass's delivered/completed identity to its key."""

        self.drain()
        state = self._key_state(consumption.phase, consumption.key)
        state["consumed_delivered"].extend(consumption.delivered)
        state["consumed_completed"].extend(consumption.completed)
        state["batches_delivered"] += consumption.batches_delivered
        state["batches_completed"] += consumption.batches_completed
        state["aborted"] = bool(state["aborted"] or consumption.aborted)
        state["iterator_closed"] = bool(
            state["iterator_closed"] or consumption.iterator_closed
        )
        state["exhausted"] = bool(state["exhausted"] or consumption.exhausted)
        return consumption.payload()

    # -- reporting -------------------------------------------------------------
    def _per_file_counts(self, entries: Sequence[tuple[str, int]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path, _ in entries:
            name = Path(path).name
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))

    def key_report(self, phase: str, key: str, *, example_limit: int = 20) -> dict[str, Any]:
        state = self._key_state(phase, key)
        reads: list[tuple[str, int]] = list(state["reads"])
        read_positions = [position for _, position in reads]
        delivered: list[int] = list(state["consumed_delivered"])
        completed: list[int] = list(state["consumed_completed"])
        unconsumed = counter_delta(read_positions, completed, example_limit)
        unread = counter_delta(completed, read_positions, example_limit)
        opens = self.opens.get((str(phase), str(key)), {})
        return {
            "phase": str(phase),
            "key": str(key),
            "identity": f"{phase}/{key}",
            "reads": {
                "windows_read": len(reads),
                "unique_positions": len(set(read_positions)),
                "duplicate_reads": duplicate_count(read_positions),
                "files": self._per_file_counts(reads),
                "read_files": sorted({path for path, _ in reads}),
                "opened_files": sorted(opens),
                "open_events": int(sum(opens.values())),
                "order_sha256": identity_digest(reads),
                "position_order_sha256": order_digest(read_positions),
                "position_multiset_sha256": multiset_digest(read_positions),
            },
            "consumed": {
                "windows_delivered": len(delivered),
                "windows_completed": len(completed),
                "batches_delivered": int(state["batches_delivered"]),
                "batches_completed": int(state["batches_completed"]),
                "delivered_order_sha256": order_digest(delivered),
                "completed_order_sha256": order_digest(completed),
                "delivered_multiset_sha256": multiset_digest(delivered),
                "completed_multiset_sha256": multiset_digest(completed),
                "duplicate_delivered": duplicate_count(delivered),
                "aborted": bool(state["aborted"]),
                "iterator_closed": bool(state["iterator_closed"]),
                "exhausted": bool(state["exhausted"]),
            },
            # Reads no completed batch needed: read-ahead at the end of a full
            # traversal.  Exactly 0 for a fully consumed epoch.
            "unconsumed_reads": unconsumed,
            # Completed windows with no matching read event: proof that an event was
            # lost (or that a read was never recorded).  Must be 0.
            "unread_consumed": unread,
            "workers": sorted(state["workers"]),
            "pids": sorted(state["pids"]),
        }

    def phase_report(self, phase: str, *, example_limit: int = 20) -> dict[str, Any]:
        keys = sorted(key for (entry_phase, key) in self.keys if entry_phase == str(phase))
        per_key = {key: self.key_report(phase, key, example_limit=example_limit) for key in keys}
        all_reads: list[tuple[str, int]] = []
        delivered = 0
        completed = 0
        unconsumed = 0
        unread = 0
        workers: set[str] = set()
        pids: set[int] = set()
        for report in per_key.values():
            state = self.keys[(str(phase), report["key"])]
            all_reads.extend(state["reads"])
            delivered += report["consumed"]["windows_delivered"]
            completed += report["consumed"]["windows_completed"]
            unconsumed += report["unconsumed_reads"]["count"]
            unread += report["unread_consumed"]["count"]
            workers |= set(report["workers"])
            pids |= set(report["pids"])
        legacy = self._per_file_counts(all_reads)
        opened: set[str] = set()
        open_events = 0
        for (entry_phase, _), counts in self.opens.items():
            if entry_phase == str(phase):
                opened |= set(counts)
                open_events += int(sum(counts.values()))
        return {
            "label": self.label,
            "phase": str(phase),
            "keys": per_key,
            "opened_files": sorted(opened),
            "opened_file_count": len(opened),
            "open_events": open_events,
            "read_files": sorted({path for path, _ in all_reads}),
            "windows_read": len(all_reads),
            "windows_by_file": legacy,
            "windows_delivered": delivered,
            "windows_completed": completed,
            "unconsumed_reads": unconsumed,
            "unread_consumed": unread,
            "duplicate_reads": duplicate_count([position for _, position in all_reads]),
            "position_order_sha256": order_digest([position for _, position in all_reads]),
            "position_multiset_sha256": multiset_digest([position for _, position in all_reads]),
            "identity_order_sha256": identity_digest(all_reads),
            "worker_count": len(workers),
            "worker_keys": sorted(workers),
            "worker_pids": sorted(pid for pid in pids if pid > 0),
            "keys_seen": keys,
            "loaders": [row for row in self.loaders if row["phase"] == str(phase)],
            "closed_loaders": [
                row for row in self.closed_loaders if row.split("/")[0] == str(phase)
            ],
            "drain_failures": self.drain_failures,
            "unknown_events": self.unknown_events,
        }

    def payload(self) -> dict[str, Any]:
        phases = sorted({phase for phase, _ in self.keys})
        return {
            "schema": "dronerfa-stage1-loader-audit/v2",
            "label": self.label,
            "identity_model": "(phase, key, underlying split position)",
            "phases": {phase: self.phase_report(phase) for phase in phases},
            "loaders": list(self.loaders),
            "closed_loaders": list(self.closed_loaders),
            "worker_processes": sorted(self.tallies),
            "worker_keys": {key: tally.payload() for key, tally in sorted(self.tallies.items())},
            "drain_failures": self.drain_failures,
            "unknown_events": self.unknown_events,
            "consumption_note": (
                "delivered = the loader handed the batch over; completed = the batch's "
                "training step ran to the end.  Both are recorded in delivery order."
            ),
        }

    def write_consumption_csv(self, path: Path) -> dict[str, Any]:
        """Persist the ordered consumption log (one row per delivered window)."""

        path.parent.mkdir(parents=True, exist_ok=True)
        rows = 0
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["phase", "key", "position", "delivered_index", "completed"])
            for (phase, key), state in sorted(self.keys.items()):
                completed_counts: dict[int, int] = {}
                for position in state["consumed_completed"]:
                    completed_counts[position] = completed_counts.get(position, 0) + 1
                for index, position in enumerate(state["consumed_delivered"]):
                    is_completed = completed_counts.get(position, 0) > 0
                    if is_completed:
                        completed_counts[position] -= 1
                    writer.writerow([phase, key, position, index, int(is_completed)])
                    rows += 1
        return {"path": str(path), "rows": rows}


def serial_audit_payload(audit: Any, split: str) -> dict[str, Any]:
    """Legacy payload from the driver's own serial :class:`ReadAudit`."""

    payload = audit.payload()
    payload["split"] = split
    payload["loader"] = "reference_num_workers_0"
    payload["windows_expected"] = int(payload.get("windows_read", 0))
    return payload


def audit_from_parallel(
    audit: LoaderAudit,
    split: str,
    *,
    windows_expected: int | None = None,
) -> dict[str, Any]:
    """Phase payload from any loader path, with the legacy aggregate field names.

    ``windows_expected`` is a *self-reported* driver expectation kept for
    continuity with the first payload shape; it is never used as evidence, because
    the verifier recomputes the expected identity from the frozen split.
    """

    payload = audit.phase_report(split)
    if windows_expected is not None:
        payload["self_reported_expected_windows"] = int(windows_expected)
    payload["loader"] = (
        "parallel_num_workers>0" if payload["worker_count"] else "reference_num_workers_0"
    )
    payload["prefetch"] = {
        "windows_read_by_workers": int(payload["windows_read"]),
        "windows_read_but_not_consumed": int(payload["unconsumed_reads"]),
        "unconsumed_examples": {
            key: report["unconsumed_reads"]["examples"]
            for key, report in payload["keys"].items()
            if report["unconsumed_reads"]["count"]
        },
    }
    return payload


def compare_audits(
    reference: dict[str, Any], parallel: dict[str, Any]
) -> dict[str, Any]:
    """Field-by-field comparison of the two audit payloads."""

    fields = ("opened_files", "windows_by_file", "windows_read", "open_events", "phase")
    differences: dict[str, Any] = {}
    for field_name in fields:
        left = reference.get(field_name)
        right = parallel.get(field_name)
        if left != right:
            differences[field_name] = {"reference": left, "parallel": right}
    return {
        "fields": list(fields),
        "differences": differences,
        "opened_files_identical": reference.get("opened_files") == parallel.get("opened_files"),
        "windows_by_file_identical": reference.get("windows_by_file")
        == parallel.get("windows_by_file"),
        "windows_read_identical": reference.get("windows_read") == parallel.get("windows_read"),
        "passed": not differences,
    }


# ---------------------------------------------------------------------------
# equivalence helpers
# ---------------------------------------------------------------------------
def tensor_digest(value: torch.Tensor) -> str:
    """SHA256 of the raw float32 buffer, so equality is byte equality."""

    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def loader_batch_digests(
    loader: DataLoader, sampler_positions: Sequence[int]
) -> list[dict[str, Any]]:
    """Ordered per-batch digest of signals/labels for one pass over ``loader``."""

    cursor = 0
    digests: list[dict[str, Any]] = []
    for batch_index, (signals, labels) in enumerate(loader):
        size = int(labels.shape[0])
        positions = [int(value) for value in sampler_positions[cursor : cursor + size]]
        cursor += size
        digests.append(
            {
                "batch": batch_index,
                "size": size,
                "positions": positions,
                "signals_sha256": tensor_digest(signals),
                "labels_sha256": tensor_digest(labels.reshape(-1).to(torch.int64)),
                "signals_dtype": str(signals.dtype),
                "labels": [int(value) for value in labels.tolist()],
            }
        )
    return digests


def compare_digests(
    reference: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Compare two digest lists batch by batch."""

    problems: list[dict[str, Any]] = []
    if len(reference) != len(candidate):
        problems.append(
            {"issue": "batch_count", "reference": len(reference), "candidate": len(candidate)}
        )
    for index, (left, right) in enumerate(zip(reference, candidate)):
        for key in ("size", "positions", "signals_sha256", "labels_sha256", "labels"):
            if left.get(key) != right.get(key):
                problems.append(
                    {"issue": key, "batch": index, "reference": left.get(key), "candidate": right.get(key)}
                )
    return {
        "batches": min(len(reference), len(candidate)),
        "problems": problems[:10],
        "problem_count": len(problems),
        "passed": not problems,
        "identical": not problems,
    }
