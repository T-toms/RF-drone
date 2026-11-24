"""Probe: can a spawned DataLoader worker run under the current sandbox?

Prints the full traceback, because the sandbox reports a bare WinError 5 that
this helper has to attribute before any escalation is requested.
"""

from __future__ import annotations

import multiprocessing
import sys
import traceback
from pathlib import Path

BASELINE = Path(__file__).resolve().parent.parent
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))

import torch
from torch.utils.data import DataLoader, Dataset


class Tiny(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int):
        return torch.full((2, 3), float(index)), index % 2


def main() -> int:
    print("torch", torch.__version__, flush=True)
    context = multiprocessing.get_context("spawn")
    events = context.Queue()
    try:
        loader = DataLoader(
            Tiny(), batch_size=2, num_workers=2, sampler=list(range(8)),
            worker_init_fn=lambda worker_id: torch.set_num_threads(1),
        )
        batches = 0
        for signals, labels in loader:
            events.put({"see": int(labels[0])})
            batches += 1
        print("batches", batches, flush=True)
        del loader
        drained = 0
        while True:
            try:
                events.get_nowait()
            except Exception:
                break
            drained += 1
        print("events", drained, flush=True)
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
