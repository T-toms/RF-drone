"""Probe: run the loaders path with real workers and surface child stderr."""

from __future__ import annotations

import multiprocessing
import sys
import traceback
from pathlib import Path

BASELINE = Path(__file__).resolve().parent.parent
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))


def main() -> int:
    from test_stage1_loader_optimization import build_fixture, train_dataset
    import stage1_loader_optimization as opt

    index_path, fixture_dir = build_fixture()
    print("fixture:", index_path, "files:", [p.name for p in fixture_dir.glob("*.mat")])

    ctx = multiprocessing.get_context("spawn")
    events = ctx.Queue()
    try:
        dataset = train_dataset(index_path)
        dataset.set_epoch(1)
        order = opt.order_for_epoch(dataset, 1, 2027)
        print("order:", order)
        config = opt.LoaderConfig(num_workers=2)
        consumption = opt.ConsumptionLog(split="train", epoch=1)
        loader = opt.build_train_loader(dataset, config, order=order, audit_events=events)
        print("loader dataset type:", type(loader.dataset).__name__)
        total = 0
        for signals, labels in opt.consumed_batches(loader, order, consumption):
            total += int(labels.shape[0])
            print("  batch", tuple(signals.shape), labels.tolist())
        print("total windows:", total)
        del loader
        dataset.close()
    except Exception:
        traceback.print_exc()
        return 1
    drained = 0
    while True:
        try:
            event = events.get_nowait()
            drained += 1
            if drained <= 4:
                print("  event:", event)
        except Exception:
            break
    print("drained events:", drained)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
