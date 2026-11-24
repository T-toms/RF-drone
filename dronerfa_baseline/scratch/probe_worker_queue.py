"""Probe: which queue type can be pickled into spawned DataLoader workers?

`ctx.Queue()` is only shareable through process inheritance, so putting one inside a
Dataset fails under spawn ("An attempt has been made to start a new process before
the current process has finished its bootstrapping phase"). A `Manager().Queue()` is
a proxy and pickles fine. This probe measures the event throughput too, because the
audit emits one open + one window event per (file, window) access.
"""

from __future__ import annotations

import multiprocessing
import sys
import time
from pathlib import Path

BASELINE = Path(__file__).resolve().parent.parent
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))

import torch
from torch.utils.data import DataLoader, Dataset


class EventDataset(Dataset):
    def __init__(self, events, count: int = 64) -> None:
        self.events = events
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        self.events.put({"kind": "window", "index": index, "pid": multiprocessing.current_process().pid})
        return torch.zeros(4, dtype=torch.float32), torch.tensor(index % 2)


def try_queue(kind: str, events, workers: int = 2) -> tuple[bool, str, float, int]:
    dataset = EventDataset(events)
    loader = DataLoader(dataset, batch_size=4, num_workers=workers)
    started = time.perf_counter()
    try:
        batches = 0
        for _ in loader:
            batches += 1
        del loader
    except Exception as error:  # noqa: BLE001 - the point is to report the failure
        return False, f"{type(error).__name__}: {error}", time.perf_counter() - started, 0
    elapsed = time.perf_counter() - started
    drained = 0
    while True:
        try:
            events.get_nowait()
            drained += 1
        except Exception:
            break
    return True, f"{batches} batches", elapsed, drained


def main() -> int:
    ctx = multiprocessing.get_context("spawn")

    print("== ctx.Queue() placed inside the Dataset ==")
    try:
        events = ctx.Queue()
        ok, detail, seconds, drained = try_queue("ctx", events)
        print(f"  ok={ok} detail={detail[:120]} seconds={seconds:.2f} drained={drained}")
    except Exception as error:
        print(f"  construction failed: {type(error).__name__}: {error}")

    print("== Manager().Queue() placed inside the Dataset ==")
    manager = ctx.Manager()
    try:
        events = manager.Queue()
        ok, detail, seconds, drained = try_queue("manager", events)
        print(f"  ok={ok} detail={detail[:120]} seconds={seconds:.2f} drained={drained}")
        # Throughput probe: 2000 events across 2 workers.
        big = manager.Queue()
        started = time.perf_counter()
        ok2, detail2, seconds2, drained2 = try_queue("manager-big", big, workers=2)
        print(f"  big: ok={ok2} drained={drained2} seconds={seconds2:.2f}")
        # Drain cost alone
        bulk = manager.Queue()
        for index in range(5000):
            bulk.put({"kind": "window", "index": index, "pid": 1, "file": "x" * 40})
        started = time.perf_counter()
        count = 0
        while True:
            try:
                bulk.get_nowait()
                count += 1
            except Exception:
                break
        print(f"  drain 5000 manager events in {time.perf_counter() - started:.2f}s (got {count})")
    finally:
        manager.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
