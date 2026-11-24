"""Frozen twelve-drone view of the accepted Stage1 window index.

Only CSV metadata is read while building this contract.  Test IQ remains closed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from dronerfa_data import add_complex_awgn, read_vector_slice, rms_normalize

HERE = Path(__file__).resolve().parent
INDEX = HERE / "outputs/stage1_12drones_20260926/window_index_stage1_12drones.csv"
INDEX_SHA256 = "5973a2fa5b8476a507fcc7a665365daf8d572f569ce5908d54f5c4bc84c48bd6"
SPLIT = HERE.parent / "work/meta_split2/split_candidate_104_20260926.csv"
SPLIT_SHA256 = "d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9"
OUT = HERE / "outputs/stage2_12drones_20260927/contract"
TARGET_CODES = ("T0001", "T0010", "T0011", "T0100", "T0101", "T0110", "T0111", "T1000", "T1001", "T1010", "T1011", "T1100")
CODE_TO_LABEL = {code: i for i, code in enumerate(TARGET_CODES)}
ROLE_COUNTS = {"train": 8990, "val": 2080, "development_test": 2950, "development_test_exposed": 610}
ROLE_FILES = {"train": 58, "val": 14, "development_test": 20, "development_test_exposed": 4}
FULL_COUNTS = {"train": 9590, "val": 2360, "development_test": 2950, "development_test_exposed": 890}
NOISE_SEEDS = {"val": 12026, "development_test": 22026, "development_test_exposed": 32026}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_contract_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if sha256(INDEX) != INDEX_SHA256 or sha256(SPLIT) != SPLIT_SHA256:
        raise RuntimeError("Frozen index or split changed")
    with INDEX.open(encoding="utf-8-sig", newline="") as f:
        full = list(csv.DictReader(f))
    if len(full) != 15790 or len({int(r["row_index"]) for r in full}) != 15790:
        raise RuntimeError("Index length or row_index uniqueness changed")
    full.sort(key=lambda r: int(r["row_index"]))
    positions = defaultdict(int)
    drones = []
    file_roles = defaultdict(set)
    unique_keys = set()
    for row in full:
        role = row["split"]
        code = row["target_code"]
        if role not in FULL_COUNTS or code not in ("T0000", *TARGET_CODES):
            raise RuntimeError(f"Unknown role or T code: {role}/{code}")
        if int(row["label"]) != (code != "T0000"):
            raise RuntimeError("Stage1 label disagrees with T code")
        key = (role, row["file_path"], int(row["start"]), int(row["length"]))
        if key in unique_keys:
            raise RuntimeError(f"Duplicate source window: {key}")
        unique_keys.add(key)
        file_roles[row["file_path"]].add(role)
        row["full_split_position"] = str(positions[role])
        positions[role] += 1
        if code != "T0000":
            label = CODE_TO_LABEL[code]
            drones.append({**row, "stage2_label": str(label), "cascade_label": str(label + 1)})
    if dict(positions) != FULL_COUNTS:
        raise RuntimeError(f"Full role counts changed: {dict(positions)}")
    if any(len(roles) != 1 for roles in file_roles.values()):
        raise RuntimeError("Source file crosses roles")
    counts = Counter(r["split"] for r in drones)
    files = {role: len({r["file_path"] for r in drones if r["split"] == role}) for role in ROLE_COUNTS}
    if dict(counts) != ROLE_COUNTS or files != ROLE_FILES:
        raise RuntimeError(f"Drone role counts changed: {dict(counts)}, {files}")
    for role in ROLE_COUNTS:
        if set(r["stage2_label"] for r in drones if r["split"] == role) != {str(i) for i in range(12) if any(x["split"] == role and x["target_code"] == TARGET_CODES[i] for x in drones)}:
            raise RuntimeError("Stage2 labels incomplete")
    if len({r["file_path"] for r in full}) != 104:
        raise RuntimeError("Source file count changed")
    return full, drones


def read_iq(row: dict[str, str], *, seed: int, epoch: int, full_role_length: int, train_snrs: tuple[float | None, ...] | None = None, fixed_snr: float | None = None) -> torch.Tensor:
    """Replicate the accepted Stage1 preprocessing at the full-split coordinate."""
    with h5py.File(row["file_path"], "r") as f:
        start, length = int(row["start"]), int(row["length"])
        iq = rms_normalize(read_vector_slice(f["RF0_I"], start, length) + 1j * read_vector_slice(f["RF0_Q"], start, length))
    rng = np.random.default_rng(seed + epoch * full_role_length + int(row["full_split_position"]))
    if train_snrs is not None:
        snr = train_snrs[int(rng.integers(0, len(train_snrs)))]
    else:
        snr = fixed_snr
    if snr is not None:
        iq = add_complex_awgn(iq, snr, rng)
    return torch.from_numpy(np.stack((iq.real, iq.imag), axis=0))


class Stage2WindowDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, str]], role: str, *, seed: int, train_snrs: tuple[float | None, ...] | None = None, fixed_snr: float | None = None):
        self.rows = [r for r in rows if r["split"] == role]
        self.role, self.seed, self.train_snrs, self.fixed_snr = role, seed, train_snrs, fixed_snr
        self.epoch = 0
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index: int):
        row = self.rows[index]
        return read_iq(row, seed=self.seed, epoch=self.epoch, full_role_length=FULL_COUNTS[self.role], train_snrs=self.train_snrs, fixed_snr=self.fixed_snr), int(row["stage2_label"])


def write_contract(out: Path = OUT) -> dict:
    full, drones = load_contract_rows()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "stage2_index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(drones[0]))
        writer.writeheader()
        writer.writerows(drones)
    mapping = {"background_cascade_label": 0, "target_to_stage2": CODE_TO_LABEL, "target_to_cascade": {code: i + 1 for i, code in enumerate(TARGET_CODES)}}
    (out / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    by_code_role = {code: dict(Counter(r["split"] for r in drones if r["target_code"] == code)) for code in TARGET_CODES}
    protocol = {"schema": "dronerfa-stage2-12class-contract/v1", "index_sha256": INDEX_SHA256, "split_sha256": SPLIT_SHA256, "view_sha256": sha256(out / "stage2_index.csv"), "mapping_sha256": sha256(out / "mapping.json"), "role_counts": ROLE_COUNTS, "role_files": ROLE_FILES, "full_role_counts": FULL_COUNTS, "by_code_role": by_code_role, "full_split_position": "zero-based within original full role before background filtering, ordered by row_index", "train_noise_rng": "default_rng(2027 + epoch*9590 + full_split_position)", "validation_noise_rng": "default_rng(12026 + full_split_position)", "test_noise_seeds": {"development_test": 22026, "development_test_exposed": 32026}, "test_iq_opened": False}
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(write_contract(args.output), ensure_ascii=False, indent=2))
