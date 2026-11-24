"""Measure the single-line cost of the per-batch digest that the benchmark records.

Rework R4 requires the batch-data hash comparison to be either outside the timed
range or its cost listed on its own line.  The benchmark keeps the digests inside
the timed loop (so every configuration is timed by identical code), so this probe
prices exactly that block on real G: windows and writes the number next to the
benchmark it belongs to.

Usage::

    python scratch/measure_digest_cost.py --windows 12 --repeats 5 \
      --out outputs/stage1_io_optimization_20260927/rework1/benchmark/digest_cost.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASELINE = Path(__file__).resolve().parent.parent
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))

import torch

import run_stage1_12drones_training as drv
from stage1_loader_optimization import LoaderConfig, limited_order, order_for_epoch, tensor_digest


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--benchmark-batches",
        type=int,
        default=12,
        help="batches per epoch in the benchmark this cost belongs to (48 windows / 4)",
    )
    parser.add_argument("--benchmark-wall-seconds", type=float, default=5.751627566665168)
    parser.add_argument("--training-seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    drv.bind_training_seed(args.training_seed)
    dataset = drv.AuditedWindowDataset(
        drv.INDEX,
        split=drv.TRAIN_ROLE,
        train_snrs=drv.parse_snr_list(drv.TRAIN_SNRS),
        seed=args.training_seed,
        read_audit=drv.ReadAudit("digest-cost"),
    )
    positions = drv.select_positions(dataset.rows, args.windows)
    dataset.set_epoch(1)
    order = limited_order(order_for_epoch(dataset, 1, args.training_seed), positions)
    loader = drv.build_train_loader(
        dataset, LoaderConfig(), positions=order, key="1", audit_events=None
    )

    samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    for signals, labels in loader:
        samples.append((signals, labels))
    del loader
    dataset.close()

    per_batch_ms: list[float] = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        for signals, labels in samples:
            tensor_digest(signals)
            tensor_digest(labels.reshape(-1).to(torch.int64))
        per_batch_ms.append((time.perf_counter() - started) * 1000.0 / max(len(samples), 1))

    batches_per_epoch = len(samples)
    mean_ms = sum(per_batch_ms) / len(per_batch_ms)
    benchmark_epoch_ms = mean_ms * args.benchmark_batches
    payload = {
        "schema": "dronerfa-stage1-io-digest-cost/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "windows": len(positions),
        "batches": batches_per_epoch,
        "batch_size": drv.BATCH_SIZE,
        "repeats": args.repeats,
        "digest_ms_per_batch_mean": mean_ms,
        "digest_ms_per_batch_min": min(per_batch_ms),
        "digest_ms_per_batch_max": max(per_batch_ms),
        "digest_ms_per_probe_epoch": mean_ms * batches_per_epoch,
        "benchmark_equivalent": {
            "batches_per_epoch": args.benchmark_batches,
            "epoch_wall_seconds": args.benchmark_wall_seconds,
            "digest_seconds_per_epoch": benchmark_epoch_ms / 1000.0,
            "share_of_epoch_wall": (benchmark_epoch_ms / 1000.0)
            / max(args.benchmark_wall_seconds, 1e-9),
            "reference_run": "bench-20260927T114431Z configuration num_workers=0",
        },
        "note": (
            "cost of tensor_digest(signals) + tensor_digest(labels) as called inside the "
            "benchmark's timed loop; the load itself is excluded.  The digest is recorded "
            "so every configuration is proven to consume identical inputs; it applies "
            "identically to all three configurations, so the comparison is unaffected, "
            "while the absolute windows/s is depressed by this share."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
