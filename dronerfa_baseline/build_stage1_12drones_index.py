#!/usr/bin/env python3
"""STAGE1-12DRONES-A step 1: window index for the frozen 104-MAT 13-class split.

The index is derived *only* from the Codex-frozen development split
(``work/meta_split2/CODEX-FREEZE-20260926.json`` ->
``work/meta_split2/split_candidate_104_20260926.csv``).  All 104 source MATs of
the 13 classes (background ``T0000`` + twelve drone classes) are indexed; every
source keeps the single role the frozen CSV gives it, so no source file can
cross train/val/test.

Stage-1 label space (this is *not* the class id):

* ``T0000`` (Background) -> ``label``/``stage1_label`` 0
* every other target code  -> ``label``/``stage1_label`` 1 (Drone)

Window rule (unchanged from the project reader):

* RF0 channel only, 100 MS/s complex baseband, read as ``RF0_I + 1j * RF0_Q``,
* every complete 1,000,000-sample (10.0 ms) window of the stored record,
* non-overlapping, stride == window size, 1M-aligned; the trailing remainder
  (``source_samples % 1_000_000``) is dropped.

Outputs (default ``dronerfa_baseline/outputs/stage1_12drones_20260926/``)::

    window_index_stage1_12drones.csv   window-level index (all 104 MATs)
    dataset_summary.json               machine-readable counts per role/class
    source_identity.json               104 source MATs: size, SHA256, HDF5 shape
    read_probe.json                    bounded real-RF0 read probes (train/val)
    index_verification.json            self-check of every index invariant
    data_scheme_stage1_12drones.md     short data-scheme note

This script trains nothing.  It never reads an RF0 window of a
``development_test`` / ``development_test_exposed`` source; those files only
appear as metadata rows.  It does not touch the legacy three-class index
(``outputs/cascade3_newsplit_20260926/``), old checkpoints, old experiment
outputs, or anything on the G: data drive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_FREEZE_JSON = PROJECT_ROOT / "work" / "meta_split2" / "CODEX-FREEZE-20260926.json"
#: Resolved from the freeze decision, which is the only role authority.
DEFAULT_FROZEN_SPLIT_CSV = PROJECT_ROOT / "work" / "meta_split2" / "split_candidate_104_20260926.csv"
DEFAULT_DATA_ROOT = Path(r"G:\DroneRFa数据集\original")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "stage1_12drones_20260926"

#: Background target code; every other code is a drone class in stage 1.
BACKGROUND_CODE = "T0000"
#: The twelve drone target codes present in the frozen 104-MAT split.
DRONE_CODES = (
    "T0001",
    "T0010",
    "T0011",
    "T0100",
    "T0101",
    "T0110",
    "T0111",
    "T1000",
    "T1001",
    "T1010",
    "T1011",
    "T1100",
)
TARGET_CODES = (BACKGROUND_CODE, *DRONE_CODES)
LABEL_BACKGROUND = 0
LABEL_DRONE = 1
LABEL_NAME = {LABEL_BACKGROUND: "Background", LABEL_DRONE: "Drone"}

ROLE_ORDER = ("train", "val", "development_test", "development_test_exposed")
#: Roles whose RF0 payload may be read by any stage-1 training artefact.
READABLE_ROLES = ("train", "val")
#: Roles that are metadata-only for this phase.
TEST_ROLES = ("development_test", "development_test_exposed")

EXPECTED_FILES_PER_CLASS = 8
EXPECTED_TOTAL_FILES = 104
EXPECTED_ROLE_FILES = {
    "train": 62,
    "val": 16,
    "development_test": 20,
    "development_test_exposed": 6,
}
#: Frozen per-class file counts.  The three classes that were already part of the
#: project before META-SPLIT2 (T0000 background, T0010 Phantom 4 Pro, T0101
#: Air 2S) were split 4/2/2/2; the ten newly downloaded drone classes were split
#: 5/1/2/0 by META-SPLIT2.
LEGACY_CODES = ("T0000", "T0010", "T0101")
EXPECTED_ROLE_FILES_LEGACY = {
    "train": 4,
    "val": 2,
    "development_test": 0,
    "development_test_exposed": 2,
}
EXPECTED_ROLE_FILES_NEW = {
    "train": 5,
    "val": 1,
    "development_test": 2,
    "development_test_exposed": 0,
}


def expected_role_files(target_code: str) -> dict[str, int]:
    """Frozen per-role file counts of one target code."""
    return (
        EXPECTED_ROLE_FILES_LEGACY
        if target_code in LEGACY_CODES
        else EXPECTED_ROLE_FILES_NEW
    )

WINDOW_SIZE = 1_000_000
#: The published DroneRFa records are stored in 10,000,000-sample blocks; the
#: two bookkeeping columns are kept only for schema parity with the legacy index.
BLOCK_SIZE = 10_000_000
WINDOWS_PER_BLOCK = BLOCK_SIZE // WINDOW_SIZE
SAMPLE_RATE_HZ = 100_000_000
REQUIRED_DATASETS = ("RF0_I", "RF0_Q")
OPTIONAL_DATASETS = ("RF1_I", "RF1_Q")

#: Frozen values this script asserts before doing any work.
FROZEN_SPLIT_CSV_SHA256 = "d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9"
FROZEN_MANIFEST_SHA256 = "9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d"

#: Bounded read probe geometry: three slices per channel per probed window.
PROBE_SLICE_SAMPLES = 4096
PROBE_SLICE_COUNT = 3
#: One full-window payload per role, compared sample for sample by the verifier.
FULL_PROBE_FILES_PER_ROLE = 1

WINDOW_RULE = (
    "RF0 I/Q, non-overlapping 1,000,000-sample windows, stride 1,000,000, "
    "tail < 1,000,000 samples dropped"
)

#: Paths that must never appear in a generated index (legacy F: data drive or
#: the old C:\...\Documents junction used by the first baseline).
FORBIDDEN_PATH_FRAGMENTS = (
    "F:\\",
    "F:/",
    "junction",
    "Documents\\DroneRFa",
    "Documents/DroneRFa",
)

#: The upstream archive segment table must never be read as a role authority.
FORBIDDEN_AUTHORITY_FILES = ("source_table_104.csv",)

FILE_PATTERN = re.compile(
    r"^T(?P<target>\d{4})_D(?P<distance>\d{2})_S(?P<segment>\d{4})\.mat$",
    re.IGNORECASE,
)
#: T1010 / T1011 / T1100 are stored without a distance field in the archive.
DISTANCE_OPTIONAL_CODES = ("T1010", "T1011", "T1100")
FILE_PATTERN_NO_DISTANCE = re.compile(
    r"^T(?P<target>\d{4})_S(?P<segment>\d{4})\.mat$", re.IGNORECASE
)

CSV_FIELDS = (
    "row_index",
    "file_path",
    "filename",
    "target_code",
    "target_name",
    "class_id",
    "class_index",
    "class_name",
    "distance_code",
    "segment_code",
    "source_role",
    "split",
    "metrics_scope",
    "role_purpose",
    "prior_use_role",
    "archive_segment_role",
    "exposed_to_prior_selection",
    "stage1_label",
    "label",
    "window_index",
    "window_in_block",
    "block_index",
    "start",
    "length",
    "source_samples",
    "file_sha256",
    "window_rule",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def dataset_length(dataset: h5py.Dataset) -> int:
    """Sample count of a stored channel, tolerating (1, N) and (N, 1) layouts."""
    if dataset.ndim == 1:
        return int(dataset.shape[0])
    if dataset.ndim == 2 and 1 in dataset.shape:
        return int(max(dataset.shape))
    raise ValueError(f"unsupported dataset shape: {dataset.shape}")


def window_count(source_samples: int, window_size: int = WINDOW_SIZE) -> int:
    if window_size <= 0:
        raise ValueError("window size must be positive")
    if source_samples < 0:
        raise ValueError("source sample count must be non-negative")
    return source_samples // window_size


def stage1_label(target_code: str) -> int:
    """Background -> 0, every drone target code -> 1."""
    return LABEL_BACKGROUND if target_code == BACKGROUND_CODE else LABEL_DRONE


def parse_mat_name(filename: str) -> dict[str, str]:
    """Split a DroneRFa MAT name; T1010/T1011/T1100 carry no distance field."""
    match = FILE_PATTERN.match(filename)
    if match is not None:
        return {
            "target_code": f"T{match.group('target')}",
            "distance_code": match.group("distance"),
            "segment_code": match.group("segment"),
            "has_distance_field": "True",
        }
    match = FILE_PATTERN_NO_DISTANCE.match(filename)
    if match is None:
        raise ValueError(f"{filename}: unexpected DroneRFa file name")
    target = f"T{match.group('target')}"
    if target not in DISTANCE_OPTIONAL_CODES:
        raise ValueError(
            f"{filename}: target {target} must carry a distance field in its name"
        )
    return {
        "target_code": target,
        "distance_code": "",
        "segment_code": match.group("segment"),
        "has_distance_field": "False",
    }


def resolve_authority_path(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


_FILE_SIZE_CACHE: dict[str, int] = {}


def _file_size(raw: str) -> int:
    """Memoised ``stat().st_size``; 0 marks a missing path."""
    cached = _FILE_SIZE_CACHE.get(raw)
    if cached is None:
        try:
            cached = Path(raw).stat().st_size
        except OSError:
            cached = 0
        _FILE_SIZE_CACHE[raw] = cached
    return cached


# ---------------------------------------------------------------------------
# frozen authority
# ---------------------------------------------------------------------------
@dataclass
class FrozenSource:
    """One frozen source MAT and the role the frozen CSV assigns to it."""

    filename: str
    file_path: Path
    target_code: str
    role: str
    size_bytes: int
    class_id: str
    class_name: str
    class_index: str
    segment_code: str
    distance_code: str
    metrics_scope: str
    role_purpose: str
    prior_use_role: str
    archive_segment_role: str
    exposed_to_prior_selection: str
    expected_sha256: str
    row: dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> int:
        return stage1_label(self.target_code)


def load_freeze(freeze_path: Path) -> tuple[dict[str, Any], Path, str]:
    """Load the freeze decision and prove it still describes the frozen CSV."""
    if not freeze_path.is_file():
        raise FileNotFoundError(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != "dronerfa-meta-split2-codex-freeze/v1":
        raise ValueError(f"unexpected freeze schema: {freeze.get('schema')!r}")
    if freeze.get("decision") != "accepted_frozen_development_split":
        raise ValueError(f"unexpected freeze decision: {freeze.get('decision')!r}")

    csv_path = resolve_authority_path(str(freeze["frozen_split_csv"]))
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    actual = sha256_file(csv_path)
    expected = str(freeze["frozen_split_csv_sha256"])
    if actual != expected:
        raise ValueError(
            f"frozen split CSV sha256 mismatch: expected {expected}, got {actual}"
        )
    if expected != FROZEN_SPLIT_CSV_SHA256:
        raise ValueError(
            f"freeze decision hash {expected} != asserted {FROZEN_SPLIT_CSV_SHA256}"
        )

    manifest_path = Path(str(freeze["source_manifest"]))
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_actual = sha256_file(manifest_path)
    manifest_expected = str(freeze["source_manifest_sha256"])
    if manifest_actual != manifest_expected:
        raise ValueError(
            "dataset manifest sha256 mismatch: "
            f"expected {manifest_expected}, got {manifest_actual}"
        )
    if manifest_expected != FROZEN_MANIFEST_SHA256:
        raise ValueError(
            f"manifest hash {manifest_expected} != asserted {FROZEN_MANIFEST_SHA256}"
        )

    counts = freeze.get("counts", {})
    if (
        int(counts.get("total_sources", -1)) != EXPECTED_TOTAL_FILES
        or int(counts.get("train", -1)) != EXPECTED_ROLE_FILES["train"]
        or int(counts.get("val", -1)) != EXPECTED_ROLE_FILES["val"]
        or int(counts.get("development_test", -1)) != EXPECTED_ROLE_FILES["development_test"]
        or int(counts.get("development_test_exposed", -1))
        != EXPECTED_ROLE_FILES["development_test_exposed"]
    ):
        raise ValueError(f"freeze decision counts are unexpected: {counts}")
    return freeze, csv_path, manifest_actual


def load_frozen_sources(csv_path: Path) -> list[FrozenSource]:
    """Read all 104 frozen rows and re-check the frozen shape of the split."""
    rows = read_csv_rows(csv_path)
    if len(rows) != EXPECTED_TOTAL_FILES:
        raise ValueError(
            f"expected {EXPECTED_TOTAL_FILES} frozen rows, got {len(rows)}"
        )

    sources: list[FrozenSource] = []
    for row in rows:
        filename = str(row["file_name"])
        parsed = parse_mat_name(filename)
        target = parsed["target_code"]
        class_id = str(row["class_id"])
        if class_id != target:
            raise ValueError(
                f"{filename}: file name target {target} disagrees with class_id {class_id}"
            )
        if str(row["archive_s_code"]) != parsed["segment_code"]:
            raise ValueError(
                f"{filename}: file name segment {parsed['segment_code']} disagrees with "
                f"archive_s_code {row['archive_s_code']}"
            )
        role = str(row["role"])
        if role not in ROLE_ORDER:
            raise ValueError(f"{filename}: unexpected role {role!r}")
        sources.append(
            FrozenSource(
                filename=filename,
                file_path=Path(str(row["g_path"])),
                target_code=target,
                role=role,
                size_bytes=int(row["size_bytes"]),
                class_id=class_id,
                class_name=str(row["class_name"]),
                class_index=str(row["class_index"]),
                segment_code=parsed["segment_code"],
                distance_code=parsed["distance_code"],
                metrics_scope=str(row["metrics_scope"]),
                role_purpose=str(row["role_purpose"]),
                prior_use_role=str(row["prior_use_role"]),
                archive_segment_role=str(row["archive_segment_role"]),
                exposed_to_prior_selection=str(row["exposed_to_prior_selection"]),
                expected_sha256=str(row["mat_sha256"]).lower(),
                row=row,
            )
        )

    names = [source.filename for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("duplicate source file names in the frozen split")
    if {source.target_code for source in sources} != set(TARGET_CODES):
        raise ValueError("frozen split does not cover the expected 13 classes")

    for target in TARGET_CODES:
        count = sum(1 for source in sources if source.target_code == target)
        if count != EXPECTED_FILES_PER_CLASS:
            raise ValueError(
                f"class {target}: expected {EXPECTED_FILES_PER_CLASS} files, got {count}"
            )
        expected_roles = expected_role_files(target)
        observed_roles = Counter(
            source.role for source in sources if source.target_code == target
        )
        for role in ROLE_ORDER:
            if observed_roles.get(role, 0) != expected_roles[role]:
                raise ValueError(
                    f"class {target} role {role}: expected {expected_roles[role]} files, "
                    f"got {observed_roles.get(role, 0)}"
                )

    observed_role_files = Counter(source.role for source in sources)
    for role, expected in EXPECTED_ROLE_FILES.items():
        if observed_role_files.get(role, 0) != expected:
            raise ValueError(
                f"role {role}: expected {expected} files, got {observed_role_files.get(role, 0)}"
            )
    return sources


# ---------------------------------------------------------------------------
# per-source metadata + bounded probes
# ---------------------------------------------------------------------------
def check_source_identity(source: FrozenSource, data_root: Path) -> dict[str, Any]:
    """Existence, size and HDF5 structure of one frozen source MAT.

    The frozen CSV already carries the accepted SHA256 of every MAT, so this
    step deliberately does not re-hash ~100 GiB; ``sha256`` is copied from the
    frozen authority and marked as such.
    """
    path = source.file_path
    if not path.is_file():
        raise FileNotFoundError(path)
    if not is_within(path, data_root):
        raise ValueError(f"{path} is outside the data root {data_root}")
    lowered = str(path).lower()
    for fragment in FORBIDDEN_PATH_FRAGMENTS:
        if fragment.lower() in lowered:
            raise ValueError(f"{path} matches a forbidden legacy path fragment {fragment!r}")

    size = path.stat().st_size
    if size != source.size_bytes:
        raise ValueError(
            f"{path.name}: size {size} does not match frozen size {source.size_bytes}"
        )

    with h5py.File(path, "r") as handle:
        keys = sorted(handle.keys())
        missing = [key for key in REQUIRED_DATASETS if key not in handle]
        if missing:
            raise KeyError(f"{path.name}: missing RF0 datasets {missing}")
        shapes = {key: list(handle[key].shape) for key in keys}
        dtypes = {key: str(handle[key].dtype) for key in keys}
        chunks = {
            key: (list(handle[key].chunks) if handle[key].chunks else None) for key in keys
        }
        compression = {
            key: [handle[key].compression, handle[key].compression_opts] for key in keys
        }
        lengths = {key: dataset_length(handle[key]) for key in REQUIRED_DATASETS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{path.name}: RF0_I/RF0_Q lengths differ: {lengths}")
        source_samples = next(iter(lengths.values()))
        if source_samples < WINDOW_SIZE:
            raise ValueError(
                f"{path.name}: {source_samples} samples cannot hold one {WINDOW_SIZE}-sample window"
            )

    return {
        "filename": source.filename,
        "g_path": str(path),
        "role": source.role,
        "target_code": source.target_code,
        "label": source.label,
        "size_bytes": size,
        "sha256": source.expected_sha256,
        "sha256_source": "frozen_csv_manifest_audited",
        "source_samples": source_samples,
        "dropped_tail_samples": source_samples % WINDOW_SIZE,
        "complete_windows": window_count(source_samples),
        "hdf5_datasets": keys,
        "hdf5_shapes": shapes,
        "hdf5_dtypes": dtypes,
        "hdf5_chunks": chunks,
        "hdf5_compression": compression,
        "has_rf1": all(key in keys for key in OPTIONAL_DATASETS),
        "mtime_utc": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def _slice_offsets(window_start: int) -> list[int]:
    """Start offsets of the three bounded probe slices inside one window."""
    if PROBE_SLICE_COUNT == 3:
        return [
            window_start,
            window_start + WINDOW_SIZE // 2 - PROBE_SLICE_SAMPLES // 2,
            window_start + WINDOW_SIZE - PROBE_SLICE_SAMPLES,
        ]
    step = max((WINDOW_SIZE - PROBE_SLICE_SAMPLES) // max(PROBE_SLICE_COUNT - 1, 1), 1)
    return [window_start + index * step for index in range(PROBE_SLICE_COUNT)]


def read_channel_slice(handle: h5py.File, name: str, start: int, length: int) -> np.ndarray:
    """Shape-tolerant read of ``name[start:start+length]`` as float32."""
    dataset = handle[name]
    if dataset.ndim == 1:
        values = dataset[start : start + length]
    elif dataset.ndim == 2 and dataset.shape[0] == 1:
        values = dataset[0, start : start + length]
    elif dataset.ndim == 2 and dataset.shape[1] == 1:
        values = dataset[start : start + length, 0]
    else:
        raise ValueError(f"unsupported dataset shape: {dataset.shape}")
    return np.asarray(values, dtype=np.float32)


def probe_windows(
    sources: Sequence[FrozenSource], role: str, slices_per_window: int | None = None
) -> dict[str, Any]:
    """Bounded real-RF0 reads for one role: first/middle/last window of each file.

    Reads at most ``PROBE_SLICE_COUNT * PROBE_SLICE_SAMPLES`` samples per channel
    per window instead of a whole 1M-sample payload, so the probe proves the
    stored I/Q is readable and finite without copying ~100 GiB.
    """
    del slices_per_window  # geometry is frozen by PROBE_SLICE_COUNT
    selected = [source for source in sources if source.role == role]
    if not selected:
        raise ValueError(f"no frozen sources carry role {role!r}")
    files: list[dict[str, Any]] = []
    started = time.perf_counter()
    for source in selected:
        identity = check_source_identity(source, DEFAULT_DATA_ROOT)
        windows = int(identity["complete_windows"])
        window_indices = sorted({0, windows // 2, windows - 1})
        probes: list[dict[str, Any]] = []
        with h5py.File(source.file_path, "r") as handle:
            for window_index in window_indices:
                window_start = window_index * WINDOW_SIZE
                per_channel: dict[str, Any] = {}
                for channel in REQUIRED_DATASETS:
                    parts = [
                        read_channel_slice(handle, channel, offset, PROBE_SLICE_SAMPLES)
                        for offset in _slice_offsets(window_start)
                    ]
                    joined = np.concatenate(parts)
                    if not bool(np.isfinite(joined).all()):
                        raise ValueError(
                            f"{source.filename} window {window_index} {channel}: non-finite samples"
                        )
                    per_channel[channel] = {
                        "slices": list(_slice_offsets(window_start)),
                        "slice_samples": PROBE_SLICE_SAMPLES,
                        "samples": int(joined.size),
                        "dtype": str(joined.dtype),
                        "crc32": f"{zlib.crc32(joined.tobytes()) & 0xFFFFFFFF:08x}",
                        "sha256": sha256_bytes(joined.tobytes()),
                        "min": float(joined.min()),
                        "max": float(joined.max()),
                        "mean": float(joined.mean(dtype=np.float64)),
                    }
                probes.append(
                    {"window_index": window_index, "start": window_start, "channels": per_channel}
                )
        files.append(
            {
                "filename": source.filename,
                "role": role,
                "target_code": source.target_code,
                "label": source.label,
                "complete_windows": windows,
                "window_indices": window_indices,
                "probes": probes,
            }
        )
    samples_read = sum(
        PROBE_SLICE_COUNT * PROBE_SLICE_SAMPLES * len(REQUIRED_DATASETS) * len(row["probes"])
        for row in files
    )
    return {
        "role": role,
        "files": files,
        "file_count": len(files),
        "probed_windows": sum(len(row["probes"]) for row in files),
        "samples_read": samples_read,
        "seconds": time.perf_counter() - started,
    }


def probe_full_window(source: FrozenSource) -> dict[str, Any]:
    """Read one whole 1M-sample window per channel; the verifier re-derives it."""
    with h5py.File(source.file_path, "r") as handle:
        per_channel: dict[str, Any] = {}
        for channel in REQUIRED_DATASETS:
            values = read_channel_slice(handle, channel, 0, WINDOW_SIZE)
            if not bool(np.isfinite(values).all()):
                raise ValueError(f"{source.filename} {channel}: non-finite samples")
            per_channel[channel] = {
                "samples": int(values.size),
                "dtype": str(values.dtype),
                "sha256": sha256_bytes(values.tobytes()),
                "crc32": f"{zlib.crc32(values.tobytes()) & 0xFFFFFFFF:08x}",
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean(dtype=np.float64)),
                "sum": float(values.sum(dtype=np.float64)),
            }
    return {
        "filename": source.filename,
        "role": source.role,
        "target_code": source.target_code,
        "window_index": 0,
        "start": 0,
        "length": WINDOW_SIZE,
        "channels": per_channel,
    }


# ---------------------------------------------------------------------------
# index construction
# ---------------------------------------------------------------------------
def build_rows(sources: Sequence[FrozenSource], identities: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        identity = identities[source.filename]
        label = stage1_label(source.target_code)
        for window_index in range(int(identity["complete_windows"])):
            rows.append(
                {
                    "row_index": len(rows),
                    "file_path": str(source.file_path),
                    "filename": source.filename,
                    "target_code": source.target_code,
                    "target_name": source.class_name,
                    "class_id": source.class_id,
                    "class_index": source.class_index,
                    "class_name": source.class_name,
                    "distance_code": source.distance_code,
                    "segment_code": source.segment_code,
                    "source_role": source.role,
                    "split": source.role,
                    "metrics_scope": source.metrics_scope,
                    "role_purpose": source.role_purpose,
                    "prior_use_role": source.prior_use_role,
                    "archive_segment_role": source.archive_segment_role,
                    "exposed_to_prior_selection": source.exposed_to_prior_selection,
                    "stage1_label": label,
                    "label": label,
                    "window_index": window_index,
                    "window_in_block": window_index % WINDOWS_PER_BLOCK,
                    "block_index": window_index // WINDOWS_PER_BLOCK,
                    "start": window_index * WINDOW_SIZE,
                    "length": WINDOW_SIZE,
                    "source_samples": int(identity["source_samples"]),
                    "file_sha256": source.expected_sha256,
                    "window_rule": WINDOW_RULE,
                }
            )
    return rows


def summarize(
    sources: Sequence[FrozenSource],
    identities: dict[str, dict[str, Any]],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    files_by_role: dict[str, int] = {role: 0 for role in ROLE_ORDER}
    windows_by_role: dict[str, int] = {role: 0 for role in ROLE_ORDER}
    windows_by_role_label: dict[str, dict[str, int]] = {role: {} for role in ROLE_ORDER}
    windows_by_role_target: dict[str, dict[str, int]] = {role: {} for role in ROLE_ORDER}
    files_by_target: dict[str, int] = {code: 0 for code in TARGET_CODES}
    windows_by_target: dict[str, int] = {code: 0 for code in TARGET_CODES}
    samples_by_role: dict[str, int] = {role: 0 for role in ROLE_ORDER}

    for source in sources:
        files_by_role[source.role] += 1
        files_by_target[source.target_code] += 1
        samples_by_role[source.role] += int(identities[source.filename]["source_samples"])
    for row in rows:
        role = str(row["source_role"])
        label = str(row["label"])
        target = str(row["target_code"])
        windows_by_role[role] += 1
        windows_by_target[target] += 1
        windows_by_role_label[role][label] = windows_by_role_label[role].get(label, 0) + 1
        windows_by_role_target[role][target] = windows_by_role_target[role].get(target, 0) + 1

    return {
        "schema": "dronerfa-stage1-12drones-dataset-summary/v1",
        "generated_at": utc_now(),
        "data_root": str(DEFAULT_DATA_ROOT),
        "window_size": WINDOW_SIZE,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_rule": WINDOW_RULE,
        "label_space": {
            "0": LABEL_NAME[0],
            "1": LABEL_NAME[1],
            "background_target_code": BACKGROUND_CODE,
            "drone_target_codes": list(DRONE_CODES),
        },
        "roles": {
            "readable": list(READABLE_ROLES),
            "metadata_only": list(TEST_ROLES),
            "source_roles": list(ROLE_ORDER),
        },
        "source_files": {
            "total": len(sources),
            "by_role": files_by_role,
            "by_target_code": files_by_target,
            "class_count": len(TARGET_CODES),
            "files_per_class": EXPECTED_FILES_PER_CLASS,
        },
        "windows": {
            "total": len(rows),
            "by_role": windows_by_role,
            "by_role_label": {
                role: dict(sorted(windows_by_role_label[role].items())) for role in ROLE_ORDER
            },
            "by_role_target_code": {
                role: dict(sorted(windows_by_role_target[role].items())) for role in ROLE_ORDER
            },
            "by_target_code": windows_by_target,
        },
        "samples": {
            "total": sum(samples_by_role.values()),
            "by_role": samples_by_role,
            "dropped_tail_samples_total": sum(
                int(identity["dropped_tail_samples"]) for identity in identities.values()
            ),
        },
        "training_inputs": {
            "train_windows": windows_by_role["train"],
            "val_windows": windows_by_role["val"],
            "train_class_counts": windows_by_role_label["train"],
            "val_class_counts": windows_by_role_label["val"],
            "sampling": "all train/val windows, no subsampling",
        },
    }


# ---------------------------------------------------------------------------
# verification of the generated index
# ---------------------------------------------------------------------------
def verify_index(
    sources: Sequence[FrozenSource],
    identities: dict[str, dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    probes: dict[str, Any],
) -> dict[str, Any]:
    """Every structural invariant of the generated index, as named checks."""
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    frozen_by_name = {source.filename: source for source in sources}
    add(
        "C01 frozen source count is 104",
        len(sources) == EXPECTED_TOTAL_FILES,
        len(sources),
    )
    role_files = Counter(source.role for source in sources)
    add(
        "C02 frozen role file counts",
        all(role_files.get(role, 0) == count for role, count in EXPECTED_ROLE_FILES.items()),
        dict(role_files),
    )
    per_class = Counter(source.target_code for source in sources)
    add(
        "C03 13 classes x 8 files",
        len(per_class) == len(TARGET_CODES)
        and set(per_class) == set(TARGET_CODES)
        and all(count == EXPECTED_FILES_PER_CLASS for count in per_class.values()),
        dict(sorted(per_class.items())),
    )

    index_files = {str(row["filename"]) for row in rows}
    add(
        "C04 index file set equals the frozen 104",
        index_files == set(frozen_by_name),
        sorted(index_files ^ set(frozen_by_name))[:5],
    )

    role_of_file: dict[str, str] = {}
    label_of_file: dict[str, int] = {}
    target_of_file: dict[str, str] = {}
    cross_role: list[str] = []
    label_errors: list[str] = []
    alignment_errors: list[str] = []
    geometry_errors: list[str] = []
    block_errors: list[str] = []
    for row in rows:
        filename = str(row["filename"])
        source = frozen_by_name.get(filename)
        if source is None:
            continue
        if role_of_file.setdefault(filename, str(row["source_role"])) != str(row["source_role"]):
            cross_role.append(filename)
        if str(row["split"]) != str(row["source_role"]):
            cross_role.append(filename)
        if str(row["source_role"]) != source.role:
            cross_role.append(filename)
        expected_label = stage1_label(source.target_code)
        label_of_file[filename] = int(row["label"])
        target_of_file[filename] = str(row["target_code"])
        if int(row["label"]) != expected_label or int(row["stage1_label"]) != expected_label:
            label_errors.append(filename)
        if int(row["start"]) != int(row["window_index"]) * WINDOW_SIZE:
            alignment_errors.append(filename)
        if (
            int(row["length"]) != WINDOW_SIZE
            or int(row["source_samples"]) != int(identities[filename]["source_samples"])
        ):
            geometry_errors.append(filename)
        if int(row["window_in_block"]) != int(row["window_index"]) % WINDOWS_PER_BLOCK or int(
            row["block_index"]
        ) != int(row["window_index"]) // WINDOWS_PER_BLOCK:
            block_errors.append(filename)
    add("C05 no source file crosses roles", not cross_role, sorted(set(cross_role))[:5])
    add(
        "C06 stage1 labels follow the frozen target code",
        not label_errors,
        sorted(set(label_errors))[:5],
    )
    add("C07 window start = window_index * 1e6", not alignment_errors, sorted(set(alignment_errors))[:5])
    add(
        "C08 window length and source_samples match the HDF5 metadata",
        not geometry_errors,
        sorted(set(geometry_errors))[:5],
    )
    add(
        "C08b window_in_block / block_index follow the 10M-sample block layout",
        not block_errors,
        sorted(set(block_errors))[:5],
    )

    window_rows_by_file = Counter(str(row["filename"]) for row in rows)
    count_mismatch = {
        filename: {"index": window_rows_by_file.get(filename, 0), "hdf5": int(identities[filename]["complete_windows"])}
        for filename in identities
        if window_rows_by_file.get(filename, 0) != int(identities[filename]["complete_windows"])
    }
    add("C09 window rows per file equal floor(samples/1e6)", not count_mismatch, count_mismatch)

    duplicate_keys = [
        key
        for key, count in Counter(
            (str(row["filename"]), int(row["window_index"])) for row in rows
        ).items()
        if count != 1
    ]
    add("C10 (file, window_index) is unique", not duplicate_keys, duplicate_keys[:5])

    existing = [str(row["file_path"]) for row in rows if not _file_size(str(row["file_path"]))]
    add("C11 every indexed path exists", not existing, sorted(set(existing))[:5])

    outside = [
        str(row["file_path"])
        for row in rows
        if not is_within(Path(str(row["file_path"])), DEFAULT_DATA_ROOT)
    ]
    add("C12 every indexed path is inside the G data root", not outside, sorted(set(outside))[:5])

    forbidden = [
        str(row["file_path"])
        for row in rows
        if any(
            fragment.lower() in str(row["file_path"]).lower()
            for fragment in FORBIDDEN_PATH_FRAGMENTS
        )
    ]
    add("C13 no legacy F:/junction path in the index", not forbidden, sorted(set(forbidden))[:5])

    sizes_bad = {
        filename: {"observed": _file_size(identity["g_path"]), "frozen": identity["size_bytes"]}
        for filename, identity in identities.items()
        if _file_size(identity["g_path"]) != identity["size_bytes"]
    }
    add("C14 on-disk sizes match the frozen manifest", not sizes_bad, sizes_bad)

    hashes_bad = {
        filename: identity["sha256"]
        for filename, identity in identities.items()
        if identity["sha256"] != frozen_by_name[filename].expected_sha256
    }
    add("C15 index file_sha256 equals the frozen manifest hash", not hashes_bad, hashes_bad)

    hash_mismatch = [
        str(row["filename"])
        for row in rows
        if str(row["file_sha256"]) != frozen_by_name[str(row["filename"])].expected_sha256
    ]
    add("C16 every row carries its source MAT SHA256", not hash_mismatch, sorted(set(hash_mismatch))[:5])

    role_file_sets = {role: {s.filename for s in sources if s.role == role} for role in ROLE_ORDER}
    overlaps = {
        "train_val": sorted(role_file_sets["train"] & role_file_sets["val"]),
        "train_dev_test": sorted(role_file_sets["train"] & role_file_sets["development_test"]),
        "val_dev_test": sorted(role_file_sets["val"] & role_file_sets["development_test"]),
        "train_exposed": sorted(role_file_sets["train"] & role_file_sets["development_test_exposed"]),
        "val_exposed": sorted(role_file_sets["val"] & role_file_sets["development_test_exposed"]),
        "dev_test_exposed": sorted(
            role_file_sets["development_test"] & role_file_sets["development_test_exposed"]
        ),
    }
    add("C17 train/val/test file sets are mutually exclusive", not any(overlaps.values()), overlaps)

    exposed_rows = {str(row["filename"]) for row in rows if str(row["source_role"]) in TEST_ROLES}
    readable_rows = {str(row["filename"]) for row in rows if str(row["source_role"]) in READABLE_ROLES}
    add(
        "C18 test roles are metadata-only (no overlapping rows with train/val)",
        not (exposed_rows & readable_rows),
        sorted(exposed_rows & readable_rows)[:5],
    )

    probe_files = {
        str(file["filename"]) for role in READABLE_ROLES for file in probes[role]["files"]
    }
    probe_test = sorted(probe_files & set(role_file_sets["development_test"]))
    probe_exposed = sorted(probe_files & set(role_file_sets["development_test_exposed"]))
    add("C19 raw RF0 probes never touch development_test", not probe_test, probe_test)
    add("C20 raw RF0 probes never touch development_test_exposed", not probe_exposed, probe_exposed)
    add(
        "C21 raw RF0 probes cover every train and val file",
        probe_files == set(role_file_sets["train"]) | set(role_file_sets["val"]),
        {"probed": len(probe_files)},
    )

    label_by_role: dict[str, Counter] = {role: Counter() for role in ROLE_ORDER}
    for row in rows:
        label_by_role[str(row["source_role"])][int(row["label"])] += 1
    add(
        "C22 no drone file carries the background label",
        all(
            not (
                row["target_code"] != BACKGROUND_CODE
                and int(row["label"]) != LABEL_DRONE
            )
            and not (row["target_code"] == BACKGROUND_CODE and int(row["label"]) != LABEL_BACKGROUND)
            for row in rows
        ),
        {role: dict(sorted(label_by_role[role].items())) for role in ROLE_ORDER},
    )

    distance_flags = {
        code: sorted(
            {
                str(row["distance_code"])
                for row in rows
                if str(row["target_code"]) == code
            }
        )
        for code in DISTANCE_OPTIONAL_CODES
    }
    add(
        "C23 T1010/T1011/T1100 carry no invented distance code",
        all(values == [""] for values in distance_flags.values()),
        distance_flags,
    )

    other_distances = sorted(
        {
            str(row["distance_code"])
            for row in rows
            if str(row["target_code"]) not in DISTANCE_OPTIONAL_CODES
            and str(row["target_code"]) != BACKGROUND_CODE
        }
    )
    add(
        "C24 the nine D-field drone classes all carry D00",
        other_distances == ["00"],
        other_distances,
    )
    background_distances = sorted(
        {str(row["distance_code"]) for row in rows if str(row["target_code"]) == BACKGROUND_CODE}
    )
    add(
        "C25 background rows keep the archive D00 scene code",
        background_distances == ["00"],
        background_distances,
    )

    label_check = {
        "background_windows": sum(1 for row in rows if int(row["label"]) == LABEL_BACKGROUND),
        "drone_windows": sum(1 for row in rows if int(row["label"]) == LABEL_DRONE),
        "background_targets": sorted(
            {str(row["target_code"]) for row in rows if int(row["label"]) == LABEL_BACKGROUND}
        ),
        "drone_target_count": len(
            {str(row["target_code"]) for row in rows if int(row["label"]) == LABEL_DRONE}
        ),
    }
    add(
        "C26 background label maps to exactly T0000 and 12 drone codes map to 1",
        label_check["background_targets"] == [BACKGROUND_CODE]
        and label_check["drone_target_count"] == len(DRONE_CODES)
        and label_check["background_windows"] > 0
        and label_check["drone_windows"] > 0,
        label_check,
    )

    passed = all(row["passed"] for row in checks)
    return {
        "schema": "dronerfa-stage1-12drones-index-verification/v1",
        "generated_at": utc_now(),
        "checks": checks,
        "passed_checks": sum(1 for row in checks if row["passed"]),
        "total_checks": len(checks),
        "passed": passed,
    }


def negative_controls(
    sources: Sequence[FrozenSource],
    identities: dict[str, dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    probes: dict[str, Any],
) -> dict[str, Any]:
    """Tamper with a copy of the index and require the verifier to reject it."""
    controls: list[dict[str, Any]] = []

    def run(name: str, mutate) -> None:
        tampered = [dict(row) for row in rows]
        mutate(tampered)
        result = verify_index(sources, identities, tampered, probes)
        failed = [row["check"] for row in result["checks"] if not row["passed"]]
        controls.append(
            {
                "control": name,
                "detected": not result["passed"],
                "failed_checks": failed,
            }
        )

    def swap_role(tampered: list[dict[str, Any]]) -> None:
        """Give one train source's first window the ``val`` role."""
        victim = next(
            str(row["filename"])
            for row in tampered
            if str(row["source_role"]) == "train" and int(row["window_index"]) == 0
        )
        for row in tampered:
            if str(row["filename"]) == victim and int(row["window_index"]) == 0:
                row["source_role"] = "val"
                row["split"] = "val"

    def flip_label(tampered: list[dict[str, Any]]) -> None:
        """Relabel one drone window as background."""
        for row in tampered:
            if int(row["label"]) == LABEL_DRONE and int(row["window_index"]) == 0:
                row["label"] = LABEL_BACKGROUND
                row["stage1_label"] = LABEL_BACKGROUND
                return
        raise AssertionError("no drone window to tamper with")

    def misalign(tampered: list[dict[str, Any]]) -> None:
        tampered[3]["start"] = int(tampered[3]["start"]) + 1

    def bad_hash(tampered: list[dict[str, Any]]) -> None:
        tampered[5]["file_sha256"] = "0" * 64

    def drop_file(tampered: list[dict[str, Any]]) -> None:
        victim = str(tampered[-1]["filename"])
        tampered[:] = [row for row in tampered if str(row["filename"]) != victim]

    def leak_probe() -> None:
        """Smuggle a development_test file into the raw read probe."""
        victim = next(
            source.filename for source in sources if source.role == "development_test"
        )
        leaked = {
            role: {
                **probes[role],
                "files": [
                    *probes[role]["files"],
                    {
                        "filename": victim,
                        "role": role,
                        "target_code": "unknown",
                        "label": LABEL_DRONE,
                        "complete_windows": 0,
                        "window_indices": [],
                        "probes": [],
                    },
                ],
            }
            for role in READABLE_ROLES
        }
        result = verify_index(sources, identities, rows, leaked)
        failed = [row["check"] for row in result["checks"] if not row["passed"]]
        controls.append(
            {
                "control": "development_test file smuggled into the raw read probe",
                "detected": not result["passed"],
                "failed_checks": failed,
            }
        )

    run("train window relabelled as val", swap_role)
    run("drone window labelled background", flip_label)
    run("window start off by one sample", misalign)
    run("source MAT hash replaced", bad_hash)
    run("one source file dropped from the index", drop_file)
    leak_probe()

    return {
        "schema": "dronerfa-stage1-12drones-index-negative-controls/v1",
        "generated_at": utc_now(),
        "controls": controls,
        "detected": sum(1 for row in controls if row["detected"]),
        "total": len(controls),
        "passed": all(row["detected"] for row in controls),
    }


# ---------------------------------------------------------------------------
# scheme note
# ---------------------------------------------------------------------------
def scheme_markdown(summary: dict[str, Any], identity_hashes: dict[str, Any]) -> str:
    lines = [
        "# STAGE1-12DRONES-A 数据入口方案（104 MAT / 13 类）",
        "",
        f"生成时间（UTC）：{summary['generated_at']}",
        "",
        "## 冻结输入",
        "",
        "- 划分权威：`work/meta_split2/CODEX-FREEZE-20260926.json` → "
        "`work/meta_split2/split_candidate_104_20260926.csv` "
        f"(SHA256 `{FROZEN_SPLIT_CSV_SHA256}`)",
        f"- G 正式清单：`G:\\DroneRFa数据集\\dataset_manifest.csv` (SHA256 `{FROZEN_MANIFEST_SHA256}`)",
        f"- 数据根：`{summary['data_root']}`；104 个 MAT 全部按冻结 `role` 索引，源文件不跨角色。",
        "",
        "## Stage1 标签空间",
        "",
        "- `T0000` = Background = 0；其余 12 个 T 码 = Drone = 1。",
        "- `class_id` 等冻结字段原样保留，仅用于溯源，不参与 Stage1 损失。",
        "",
        "## 窗口规则",
        "",
        f"- {summary['window_rule']}；每窗 {summary['window_size']} 采样 @ {summary['sample_rate_hz']} Hz（10.0 ms）。",
        "",
        "## 角色与窗口统计",
        "",
        "| role | 文件数 | 窗口数 | label 0 | label 1 | 本阶段用途 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    usage = {
        "train": "模型拟合（全部窗口）",
        "val": "选型（clean/-10/-12/-15 dB 四条件）",
        "development_test": "不读取，仅元数据登记",
        "development_test_exposed": "不读取，仅元数据登记",
    }
    for role in ROLE_ORDER:
        by_label = summary["windows"]["by_role_label"][role]
        lines.append(
            f"| {role} | {summary['source_files']['by_role'][role]} | "
            f"{summary['windows']['by_role'][role]} | {by_label.get('0', 0)} | "
            f"{by_label.get('1', 0)} | {usage[role]} |"
        )
    lines += [
        "",
        "## 源身份与核验",
        "",
        f"- 索引 SHA256：`{identity_hashes['index_sha256']}`",
        f"- 汇总 JSON SHA256：`{identity_hashes['summary_sha256']}`",
        f"- 源身份 JSON SHA256：`{identity_hashes['identity_sha256']}`",
        f"- 读取探针 JSON SHA256：`{identity_hashes['probe_sha256']}`",
        "",
        "## 限制",
        "",
        "- 冻结划分是**开发用途**划分：新 10 类的 20 个测试文件与此前未参与选型但同属一个文件时间簇，"
        "旧 3 类的 6 个测试文件此前已暴露；不得称 13 类独立最终测试或跨采集会话。",
        "- 9 个机型文件名含 `D00`（论文表 3：20～40 m）；`T1010/T1011/T1100` 文件名无距离字段，距离不可确认，"
        "本索引不推定其距离。",
        "- Stage1 只输出 Background vs Drone，本阶段不对任何测试角色做评价。",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def build_index(output_dir: Path, freeze_json: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.glob("window_index_stage1_12drones.csv")):
        raise FileExistsError(
            f"refusing to overwrite an existing index: {output_dir / 'window_index_stage1_12drones.csv'}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    freeze, csv_path, manifest_hash = load_freeze(freeze_json)
    sources = load_frozen_sources(csv_path)
    print(f"frozen sources: {len(sources)} files, {len(TARGET_CODES)} classes", flush=True)

    identities: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for position, source in enumerate(sources, 1):
        identities[source.filename] = check_source_identity(source, DEFAULT_DATA_ROOT)
        if position % 20 == 0 or position == len(sources):
            print(f"  metadata {position}/{len(sources)}", flush=True)
    metadata_seconds = time.perf_counter() - started

    rows = build_rows(sources, identities)
    print(f"window rows: {len(rows)}", flush=True)

    probes: dict[str, Any] = {}
    for role in READABLE_ROLES:
        probes[role] = probe_windows(sources, role)
        print(
            f"  probe {role}: files={probes[role]['file_count']} "
            f"windows={probes[role]['probed_windows']} samples={probes[role]['samples_read']} "
            f"({probes[role]['seconds']:.1f}s)",
            flush=True,
        )
    full_probes = []
    for role in READABLE_ROLES:
        candidates = sorted(
            (source for source in sources if source.role == role), key=lambda item: item.filename
        )
        for source in candidates[:FULL_PROBE_FILES_PER_ROLE]:
            full_probes.append(probe_full_window(source))
    for role in TEST_ROLES:
        candidates = sorted(
            (source for source in sources if source.role == role), key=lambda item: item.filename
        )
        if candidates:
            # Metadata-only: record which file would be probed, without reading it.
            full_probes.append(
                {
                    "filename": candidates[0].filename,
                    "role": role,
                    "target_code": candidates[0].target_code,
                    "read": False,
                    "reason": "test role is metadata-only for STAGE1-12DRONES-A",
                }
            )

    summary = summarize(sources, identities, rows)
    summary["freeze_inputs"] = {
        "freeze_json": str(freeze_json),
        "freeze_decision": freeze.get("decision"),
        "frozen_split_csv": str(csv_path),
        "frozen_split_csv_sha256": FROZEN_SPLIT_CSV_SHA256,
        "g_manifest": str(freeze.get("source_manifest")),
        "g_manifest_sha256": manifest_hash,
    }
    summary["timings"] = {"metadata_seconds": metadata_seconds}

    verification = verify_index(sources, identities, rows, probes)
    controls = negative_controls(sources, identities, rows, probes)

    index_path = output_dir / "window_index_stage1_12drones.csv"
    write_csv(index_path, rows)
    summary_path = output_dir / "dataset_summary.json"
    identity_path = output_dir / "source_identity.json"
    probe_path = output_dir / "read_probe.json"
    verification_path = output_dir / "index_verification.json"
    controls_path = output_dir / "index_negative_controls.json"
    scheme_path = output_dir / "data_scheme_stage1_12drones.md"

    identity_hashes = {
        "index_sha256": sha256_file(index_path),
        "index_rows": len(rows),
        "summary_sha256": "",
        "identity_sha256": "",
        "probe_sha256": "",
    }
    write_json(
        identity_path,
        {
            "schema": "dronerfa-stage1-12drones-source-identity/v1",
            "generated_at": utc_now(),
            "data_root": str(DEFAULT_DATA_ROOT),
            "sha256_policy": (
                "MAT SHA256 copied from the Codex-accepted G manifest and frozen split CSV; "
                "not recomputed, to avoid re-hashing ~100 GiB for documentation"
            ),
            "files": [identities[source.filename] for source in sources],
        },
    )
    write_json(
        probe_path,
        {
            "schema": "dronerfa-stage1-12drones-read-probe/v1",
            "generated_at": utc_now(),
            "probe_geometry": {
                "slices_per_channel": PROBE_SLICE_COUNT,
                "slice_samples": PROBE_SLICE_SAMPLES,
                "window_indices_per_file": "first, middle, last",
                "channels": list(REQUIRED_DATASETS),
                "full_window_files_per_role": FULL_PROBE_FILES_PER_ROLE,
            },
            "roles": {
                role: {
                    "file_count": probes[role]["file_count"],
                    "probed_windows": probes[role]["probed_windows"],
                    "samples_read": probes[role]["samples_read"],
                    "seconds": probes[role]["seconds"],
                }
                for role in READABLE_ROLES
            },
            "bounded": probes,
            "full_windows": full_probes,
            "test_roles_read": {
                role: sorted(source.filename for source in sources if source.role == role)
                for role in TEST_ROLES
            },
            "test_roles_read_count": 0,
        },
    )
    write_json(output_dir / "index_negative_controls.json", controls)
    write_json(verification_path, verification)
    summary["artifacts"] = identity_hashes
    summary["index_verification"] = {
        "passed": verification["passed"],
        "passed_checks": verification["passed_checks"],
        "total_checks": verification["total_checks"],
    }
    summary["index_negative_controls"] = {
        "passed": controls["passed"],
        "detected": controls["detected"],
        "total": controls["total"],
    }
    write_json(summary_path, summary)
    identity_hashes["summary_sha256"] = sha256_file(summary_path)
    identity_hashes["identity_sha256"] = sha256_file(identity_path)
    identity_hashes["probe_sha256"] = sha256_file(probe_path)
    scheme_path.write_text(scheme_markdown(summary, identity_hashes), encoding="utf-8")

    result = {
        "output_dir": str(output_dir),
        "index": {"path": str(index_path), "sha256": identity_hashes["index_sha256"], "rows": len(rows)},
        "summary": {"path": str(summary_path), "sha256": identity_hashes["summary_sha256"]},
        "source_identity": {"path": str(identity_path), "sha256": identity_hashes["identity_sha256"]},
        "read_probe": {"path": str(probe_path), "sha256": identity_hashes["probe_sha256"]},
        "verification_path": str(verification_path),
        "negative_controls_path": str(controls_path),
        "scheme_path": str(scheme_path),
        "verification": verification,
        "negative_controls": controls,
        "windows_by_role": summary["windows"]["by_role"],
        "windows_by_role_label": summary["windows"]["by_role_label"],
    }
    write_json(output_dir / "index_build_result.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STAGE1-12DRONES-A: window index for the frozen 104-MAT 13-class split."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--freeze-json", type=Path, default=DEFAULT_FREEZE_JSON)
    parser.add_argument("--verify-only", action="store_true", help="re-check an existing index")
    return parser.parse_args(argv)


def verify_existing(output_dir: Path, freeze_json: Path) -> int:
    """Re-derive the invariants of an index that already exists on disk."""
    _, csv_path, _ = load_freeze(freeze_json)
    sources = load_frozen_sources(csv_path)
    identities: dict[str, dict[str, Any]] = {}
    for source in sources:
        with h5py.File(source.file_path, "r") as handle:
            samples = dataset_length(handle[REQUIRED_DATASETS[0]])
        identities[source.filename] = {
            "g_path": str(source.file_path),
            "size_bytes": source.size_bytes,
            "source_samples": samples,
            "dropped_tail_samples": samples % WINDOW_SIZE,
            "complete_windows": window_count(samples),
            "sha256": source.expected_sha256,
        }
    rows = read_csv_rows(output_dir / "window_index_stage1_12drones.csv")
    payload = json.loads((output_dir / "read_probe.json").read_text(encoding="utf-8"))
    probes = {role: payload["bounded"][role] for role in READABLE_ROLES}
    result = verify_index(sources, identities, rows, probes)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_only:
        return verify_existing(args.output_dir, args.freeze_json)
    result = build_index(args.output_dir, args.freeze_json)
    print(
        json.dumps(
            {
                "index": result["index"],
                "windows_by_role": result["windows_by_role"],
                "windows_by_role_label": result["windows_by_role_label"],
                "index_verification": {
                    "passed": result["verification"]["passed"],
                    "passed_checks": result["verification"]["passed_checks"],
                    "total_checks": result["verification"]["total_checks"],
                },
                "negative_controls": {
                    "passed": result["negative_controls"]["passed"],
                    "detected": result["negative_controls"]["detected"],
                    "total": result["negative_controls"]["total"],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if result["verification"]["passed"] and result["negative_controls"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
