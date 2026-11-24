from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


TARGET_NAMES = {
    "0000": "Background",
    "0010": "DJI Phantom 4 Pro",
    "0101": "DJI Air 2S",
}

SEGMENT_SPLITS = {
    "0000": "train",
    "0001": "train",
    "0010": "val",
    "0011": "test",
}


def dataset_length(dataset: h5py.Dataset) -> int:
    if dataset.ndim == 1:
        return int(dataset.shape[0])
    if dataset.ndim == 2 and 1 in dataset.shape:
        return int(max(dataset.shape))
    raise ValueError(f"Unsupported dataset shape: {dataset.shape}")


def read_vector_slice(
    dataset: h5py.Dataset, start: int, length: int
) -> np.ndarray:
    stop = start + length
    if dataset.ndim == 1:
        values = dataset[start:stop]
    elif dataset.ndim == 2 and dataset.shape[0] == 1:
        values = dataset[0, start:stop]
    elif dataset.ndim == 2 and dataset.shape[1] == 1:
        values = dataset[start:stop, 0]
    else:
        raise ValueError(f"Unsupported dataset shape: {dataset.shape}")
    return np.asarray(values, dtype=np.float32)


def load_index(index_path: Path, split: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            row["label"] = int(row["label"])
            row["block_index"] = int(row["block_index"])
            row["window_index"] = int(row["window_index"])
            row["start"] = int(row["start"])
            row["length"] = int(row["length"])
            row["source_samples"] = int(row["source_samples"])
            if split is None or row["split"] == split:
                rows.append(row)
    return rows


def stratified_limit(
    rows: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or len(rows) <= limit:
        return rows

    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        stratum = str(row.get("target_code", row["label"]))
        by_label.setdefault(stratum, []).append(row)

    selected: list[dict[str, Any]] = []
    labels = sorted(by_label)
    cursor = 0
    while len(selected) < limit:
        added = False
        for label in labels:
            bucket = by_label[label]
            if cursor < len(bucket) and len(selected) < limit:
                selected.append(bucket[cursor])
                added = True
        if not added:
            break
        cursor += 1
    return selected


def rms_normalize(iq: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    centered = iq - np.mean(iq, dtype=np.complex128)
    power = float(np.mean(np.abs(centered) ** 2, dtype=np.float64))
    return (centered / math.sqrt(max(power, epsilon))).astype(np.complex64)


def add_complex_awgn(
    iq: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    signal_power = float(np.mean(np.abs(iq) ** 2, dtype=np.float64))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = math.sqrt(noise_power / 2.0)
    noise = scale * (
        rng.standard_normal(iq.shape, dtype=np.float32)
        + 1j * rng.standard_normal(iq.shape, dtype=np.float32)
    )
    return (iq + noise).astype(np.complex64)


def parse_snr_list(value: str) -> list[float | None]:
    snrs: list[float | None] = []
    for token in value.split(","):
        item = token.strip().lower()
        if not item:
            continue
        if item in {"clean", "none"}:
            snrs.append(None)
        else:
            snrs.append(float(item))
    if not snrs:
        raise ValueError("SNR list must contain at least one value")
    return snrs


def snr_name(snr_db: float | None) -> str:
    if snr_db is None:
        return "clean"
    return f"{snr_db:g}dB"


def class_counts(rows: Iterable[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in rows:
        label = int(row["label"])
        counts[label] = counts.get(label, 0) + 1
    return counts


class DroneRFaWindowDataset:
    """Lazy HDF5 window reader compatible with PyTorch DataLoader."""

    def __init__(
        self,
        index_path: Path,
        split: str,
        train_snrs: list[float | None] | None = None,
        fixed_snr: float | None = None,
        apply_fixed_snr: bool = False,
        max_samples: int | None = None,
        seed: int = 2026,
    ) -> None:
        self.rows = stratified_limit(load_index(index_path, split), max_samples)
        self.train_snrs = train_snrs
        self.fixed_snr = fixed_snr
        self.apply_fixed_snr = apply_fixed_snr
        self.seed = seed
        self.epoch = 0
        self._files: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _file(self, path: str) -> h5py.File:
        handle = self._files.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._files[path] = handle
        return handle

    def __getitem__(self, index: int):
        import torch

        row = self.rows[index]
        handle = self._file(row["file_path"])
        start = int(row["start"])
        length = int(row["length"])
        i_values = read_vector_slice(handle["RF0_I"], start, length)
        q_values = read_vector_slice(handle["RF0_Q"], start, length)
        iq = rms_normalize(i_values + 1j * q_values)

        rng = np.random.default_rng(self.seed + self.epoch * len(self) + index)
        if self.train_snrs is not None:
            snr_db = self.train_snrs[int(rng.integers(0, len(self.train_snrs)))]
            if snr_db is not None:
                iq = add_complex_awgn(iq, snr_db, rng)
        elif self.apply_fixed_snr and self.fixed_snr is not None:
            iq = add_complex_awgn(iq, self.fixed_snr, rng)

        tensor = torch.from_numpy(np.stack((iq.real, iq.imag), axis=0))
        return tensor, int(row["label"])

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
