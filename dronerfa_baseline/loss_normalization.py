#!/usr/bin/env python3
"""STAGE1-LOSS-NORM: the two loss normalisations compared by this task.

The frozen stage-1 protocol trains ``Background vs Drone`` with a *class-weighted*
cross entropy and ``batch_size=4``.  ``work/stage1-batch-balance-diagnostic-20260927.json``
showed that roughly 77% of the four-window batches contain no background window at
all, so the batch that is actually optimised rarely contains the class the weights
are supposed to protect.  Two different things can be meant by "class-weighted
cross entropy" on such a batch:

``batch_weight_sum`` (the historical / PyTorch default)
    ``L = sum_i(w_yi * CE_i) / sum_i(w_yi)``

    The denominator is the *batch's own* weight sum, so a drone-only batch cancels
    the drone weight completely (``w1 / (4*w1) = 1/4`` each) and the class weight
    only ever acts *between* windows of the same batch.  This is exactly what
    ``torch.nn.CrossEntropyLoss(weight=...)`` computes and is therefore the
    historical behaviour that must stay bit-for-bit reproducible.

``global_train_mean`` (this task's candidate)
    ``L = sum_i(w_yi * CE_i) / (B_actual * mu_train)``

    ``mu_train = sum_c(n_c * w_c) / N`` is computed from the **training** counts
    only (``N=9590``, ``n0=600``, ``n1=8990``, ``w_c = N/(2*n_c)``) and is ``1`` by
    construction.  The denominator is the *actual* number of windows in the batch
    times ``mu_train`` -- never the nominal ``batch_size``, because the frozen split
    has 9590 windows and 9590 = 2397*4 + 2, so every epoch ends on a two-window tail
    batch.  Dividing that tail batch by 4 would silently shrink its gradient by 2x.

The module is deliberately standalone: it neither imports nor modifies the frozen
data/model modules, and it is the only place the formulas live.  It also records,
per optimiser step, the numerator and the denominator that were really used, which
is what lets a verifier re-derive the objective instead of trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

#: Historical normalisation: ``sum(w*CE) / sum(w)`` over the batch.
LOSS_MODE_BATCH_WEIGHT_SUM = "batch_weight_sum"
#: Candidate normalisation: ``sum(w*CE) / (B_actual * mu_train)``.
LOSS_MODE_GLOBAL_TRAIN_MEAN = "global_train_mean"
LOSS_MODES: tuple[str, ...] = (LOSS_MODE_BATCH_WEIGHT_SUM, LOSS_MODE_GLOBAL_TRAIN_MEAN)
#: The frozen protocol's default; every existing run keeps this.
DEFAULT_LOSS_MODE = LOSS_MODE_BATCH_WEIGHT_SUM

FORMULAS: dict[str, str] = {
    LOSS_MODE_BATCH_WEIGHT_SUM: "L = sum_i(w_yi * CE_i) / sum_i(w_yi)",
    LOSS_MODE_GLOBAL_TRAIN_MEAN: "L = sum_i(w_yi * CE_i) / (B_actual * mu_train)",
}
DENOMINATOR_RULES: dict[str, str] = {
    LOSS_MODE_BATCH_WEIGHT_SUM: (
        "sum of the per-window class weights present in the batch (PyTorch "
        "CrossEntropyLoss(weight=..., reduction='mean') semantics)"
    ),
    LOSS_MODE_GLOBAL_TRAIN_MEAN: (
        "actual batch size B_actual times mu_train; the 2-window tail batch uses "
        "2 * mu_train, never batch_size=4"
    ),
}
TAIL_BATCH_RULES: dict[str, str] = {
    LOSS_MODE_BATCH_WEIGHT_SUM: (
        "the tail batch is re-normalised by its own two weights, exactly as the "
        "frozen protocol always did"
    ),
    LOSS_MODE_GLOBAL_TRAIN_MEAN: (
        "the tail batch keeps the global per-window scale: denominator = 2 * mu_train"
    ),
}


def class_weights_from_counts(counts: Mapping[int, int]) -> list[float]:
    """Frozen weights ``w_c = N / (2 * n_c)`` for classes ``0..C-1`` in order.

    The values are rounded through ``float32`` because that is the dtype the frozen
    runs put in the criterion and record in ``run_config.json``; keeping the float64
    value here would make a hash/equality comparison against the accepted control
    fail on the last bits for no reason.
    """

    if not counts:
        raise ValueError("counts must not be empty")
    labels = sorted(int(label) for label in counts)
    total = sum(int(counts[label]) for label in labels)
    if total <= 0:
        raise ValueError("counts must be positive")
    for label in labels:
        if int(counts[label]) <= 0:
            raise ValueError(f"class {label} has no windows: {counts}")
    return [
        float(torch.tensor(total / (len(labels) * int(counts[label])), dtype=torch.float32))
        for label in labels
    ]


def mu_train_from_counts(
    counts: Mapping[int, int], weights: Sequence[float]
) -> float:
    """``mu_train = sum_c(n_c * w_c) / N`` -- the mean per-window weight."""

    labels = sorted(int(label) for label in counts)
    if len(weights) != len(labels):
        raise ValueError(
            f"weights ({len(weights)}) do not match the class count ({len(labels)})"
        )
    total = sum(int(counts[label]) for label in labels)
    weighted = sum(
        int(counts[label]) * float(weight) for label, weight in zip(labels, weights)
    )
    return weighted / total


@dataclass(frozen=True)
class LossSpec:
    """Everything a run has to record about the objective it trained with."""

    mode: str
    class_weights: tuple[float, ...]
    train_counts: dict[int, int]
    train_windows: int
    mu_train: float
    batch_size: int
    tail_batch_size: int
    tail_batches_per_epoch: int

    def payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "formula": FORMULAS[self.mode],
            "denominator_rule": DENOMINATOR_RULES[self.mode],
            "tail_batch_rule": TAIL_BATCH_RULES[self.mode],
            "class_weights": list(self.class_weights),
            "weight_rule": "w_c = N / (2 * n_c) from the training split only",
            "train_counts": {str(key): int(value) for key, value in sorted(self.train_counts.items())},
            "train_windows": int(self.train_windows),
            "mu_train": float(self.mu_train),
            "mu_train_definition": "sum_c(n_c * w_c) / N over the training split only",
            "batch_size": int(self.batch_size),
            "tail_batch_size": int(self.tail_batch_size),
            "tail_batches_per_epoch": int(self.tail_batches_per_epoch),
            "batches_per_epoch": self.batches_per_epoch,
            "validation_loss": (
                "unchanged original class-weighted cross entropy "
                "(nn.CrossEntropyLoss(weight=class_weights)); the candidate objective "
                "is stored under a different name and must never be compared to it"
            ),
        }

    @property
    def batches_per_epoch(self) -> int:
        return -(-int(self.train_windows) // int(self.batch_size))

    @property
    def actual_batch_sizes(self) -> list[int]:
        sizes = [int(self.batch_size)] * (int(self.train_windows) // int(self.batch_size))
        remainder = int(self.train_windows) % int(self.batch_size)
        if remainder:
            sizes.append(remainder)
        return sizes


def build_loss_spec(
    mode: str,
    train_counts: Mapping[int, int],
    batch_size: int,
) -> LossSpec:
    """Validate a mode against the frozen training counts and describe it."""

    if mode not in LOSS_MODES:
        raise ValueError(f"unknown loss normalisation {mode!r}; expected one of {LOSS_MODES}")
    counts = {int(label): int(value) for label, value in train_counts.items()}
    weights = class_weights_from_counts(counts)
    mu_train = mu_train_from_counts(counts, weights)
    total = sum(counts.values())
    tail = total % int(batch_size)
    return LossSpec(
        mode=str(mode),
        class_weights=tuple(float(value) for value in weights),
        train_counts=counts,
        train_windows=total,
        mu_train=float(mu_train),
        batch_size=int(batch_size),
        tail_batch_size=int(tail),
        tail_batches_per_epoch=(1 if tail else 0),
    )


class NormalizedWeightedCrossEntropy(nn.Module):
    """Class-weighted cross entropy under one of the two normalisations.

    ``record_steps`` keeps a per-call record of the numerator, the denominator and
    the actual batch size, which the run persists as objective-step evidence.  It is
    off by default so unit tests and quick probes carry no bookkeeping.
    """

    def __init__(
        self,
        mode: str,
        weight: Sequence[float] | torch.Tensor,
        mu_train: float = 1.0,
        *,
        record_steps: bool = False,
    ) -> None:
        super().__init__()
        if mode not in LOSS_MODES:
            raise ValueError(f"unknown loss normalisation {mode!r}")
        tensor = torch.as_tensor(weight, dtype=torch.float32)
        if tensor.ndim != 1 or tensor.numel() < 2:
            raise ValueError(f"class weights must be a 1-D vector with >= 2 entries, got {tuple(tensor.shape)}")
        self.mode = str(mode)
        self.mu_train = float(mu_train)
        self.register_buffer("weight", tensor)
        self.record_steps = bool(record_steps)
        self.steps: list[dict[str, float | int]] = []

    # -- the two normalisations ------------------------------------------------
    def numerator(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``sum_i(w_yi * CE_i)`` -- identical for both modes."""

        per_window = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        return per_window.sum()

    def denominator(self, target: torch.Tensor) -> torch.Tensor:
        batch = int(target.shape[0])
        if self.mode == LOSS_MODE_GLOBAL_TRAIN_MEAN:
            return torch.as_tensor(
                float(batch) * self.mu_train, dtype=torch.float32, device=target.device
            )
        return self.weight[target].sum()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        numerator = self.numerator(logits, target)
        denominator = self.denominator(target)
        loss = numerator / denominator
        if self.record_steps:
            self.steps.append(
                {
                    "batch_size": int(target.shape[0]),
                    "denominator": float(denominator.detach()),
                    "numerator": float(numerator.detach()),
                    "loss": float(loss.detach()),
                }
            )
        return loss


def build_criterion(
    spec: LossSpec,
    device: torch.device | str | None = None,
    *,
    record_steps: bool = False,
) -> nn.Module:
    """Build the training criterion for ``spec``.

    ``batch_weight_sum`` returns the *original* ``nn.CrossEntropyLoss(weight=...)``
    module rather than the re-implementation above: the historical mode must not
    depend on this task's arithmetic at all.  ``global_train_mean`` necessarily uses
    :class:`NormalizedWeightedCrossEntropy`, whose ``batch_weight_sum`` branch is
    kept and unit-tested against the module so the two implementations are known to
    agree.
    """

    if spec.mode == LOSS_MODE_BATCH_WEIGHT_SUM:
        criterion: nn.Module = nn.CrossEntropyLoss(
            weight=torch.tensor(spec.class_weights, dtype=torch.float32)
        )
    else:
        criterion = NormalizedWeightedCrossEntropy(
            spec.mode,
            spec.class_weights,
            spec.mu_train,
            record_steps=record_steps,
        )
    if device is not None:
        criterion = criterion.to(torch.device(device))
    return criterion


def build_validation_criterion(
    spec: LossSpec, device: torch.device | str | None = None
) -> nn.Module:
    """The frozen validation/selection loss, always the original weighted CE."""

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(spec.class_weights, dtype=torch.float32)
    )
    if device is not None:
        criterion = criterion.to(torch.device(device))
    return criterion


class ObjectiveLossProbe(nn.Module):
    """Wrap the validation criterion and accumulate a second objective loss.

    ``forward`` returns the *frozen validation loss* unchanged, so the recorded
    ``loss`` column keeps its original definition.  The candidate objective loss of
    the very same logits is accumulated separately and is reported under its own
    name only.

    The caller declares the call layout (how many optimiser-free calls one
    validation condition takes, and in which order the conditions are visited), so
    every recorded call can be attributed to a ``(cycle, condition)`` cell without
    guessing.  A "cycle" is one full sweep over the four validation conditions.
    """

    def __init__(
        self,
        primary: nn.Module,
        secondary: nn.Module,
        *,
        conditions: Sequence[str],
        calls_per_condition: int,
    ) -> None:
        super().__init__()
        if not conditions:
            raise ValueError("conditions must not be empty")
        if int(calls_per_condition) < 1:
            raise ValueError(f"calls_per_condition must be >= 1, got {calls_per_condition}")
        self.primary = primary
        self.secondary = secondary
        self.conditions = tuple(str(value) for value in conditions)
        self.calls_per_condition = int(calls_per_condition)
        self.calls: list[dict[str, Any]] = []

    @property
    def calls_per_cycle(self) -> int:
        return self.calls_per_condition * len(self.conditions)

    def cell_of(self, call_index: int) -> tuple[int, str]:
        cycle = int(call_index) // self.calls_per_cycle
        offset = int(call_index) % self.calls_per_cycle
        return cycle, self.conditions[offset // self.calls_per_condition]

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.primary(logits, target)
        with torch.no_grad():
            secondary = float(self.secondary(logits, target).detach())
        self.calls.append(
            {
                "call": len(self.calls),
                "windows": int(target.shape[0]),
                "loss_sum": secondary * int(target.shape[0]),
            }
        )
        return loss

    def reset(self) -> None:
        self.calls.clear()

    def mean_by_cell(self) -> dict[str, Any]:
        cells: dict[tuple[int, str], dict[str, float]] = {}
        for call in self.calls:
            key = self.cell_of(int(call["call"]))
            entry = cells.setdefault(key, {"loss_sum": 0.0, "windows": 0.0})
            entry["loss_sum"] += float(call["loss_sum"])
            entry["windows"] += float(call["windows"])
        cycles = sorted({key[0] for key in cells})
        return {
            "conditions": list(self.conditions),
            "calls_per_condition": self.calls_per_condition,
            "calls_per_cycle": self.calls_per_cycle,
            "calls": len(self.calls),
            "cycles": len(cycles),
            "by_cycle": {
                str(cycle): {
                    condition: {
                        "windows": int(entry["windows"]),
                        "objective_loss": entry["loss_sum"] / max(entry["windows"], 1.0),
                    }
                    for (entry_cycle, condition), entry in sorted(cells.items())
                    if entry_cycle == cycle
                }
                for cycle in sorted({key[0] for key in cells})
            },
            "note": (
                "objective loss of the *candidate* normalisation evaluated on the same "
                "validation logits; it is not the recorded validation loss and the two "
                "numbers must not be compared as if they shared a definition"
            ),
        }


def step_evidence_summary(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the recorded objective steps into a checkable summary."""

    sizes = [int(step["batch_size"]) for step in steps]
    denominators = [float(step["denominator"]) for step in steps]
    numerators = [float(step["numerator"]) for step in steps]
    losses = [float(step["loss"]) for step in steps]
    tail = [index for index, size in enumerate(sizes) if size != max(sizes, default=0)]
    finite = all(value == value and abs(value) != float("inf") for value in losses)
    return {
        "steps": len(steps),
        "windows": int(sum(sizes)),
        "batch_size_histogram": {str(size): sizes.count(size) for size in sorted(set(sizes))},
        "non_nominal_batch_steps": tail,
        "non_nominal_batch_sizes": [sizes[index] for index in tail],
        "denominator_min": min(denominators) if denominators else None,
        "denominator_max": max(denominators) if denominators else None,
        "numerator_min": min(numerators) if numerators else None,
        "numerator_max": max(numerators) if numerators else None,
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "all_finite": bool(finite),
    }


def reset_steps(criterion: nn.Module) -> None:
    steps = getattr(criterion, "steps", None)
    if steps is not None:
        steps.clear()


__all__ = [
    "DEFAULT_LOSS_MODE",
    "DENOMINATOR_RULES",
    "FORMULAS",
    "LOSS_MODES",
    "LOSS_MODE_BATCH_WEIGHT_SUM",
    "LOSS_MODE_GLOBAL_TRAIN_MEAN",
    "LossSpec",
    "NormalizedWeightedCrossEntropy",
    "ObjectiveLossProbe",
    "TAIL_BATCH_RULES",
    "build_criterion",
    "build_loss_spec",
    "build_validation_criterion",
    "class_weights_from_counts",
    "mu_train_from_counts",
    "reset_steps",
    "step_evidence_summary",
]
