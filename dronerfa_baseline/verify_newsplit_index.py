#!/usr/bin/env python3
"""Independent verification of the CASCADE3-NEWSPLIT-A window index.

This checker deliberately re-derives everything from the frozen authority files
and from the HDF5 headers on G: instead of reusing the builder's row logic:

* the frozen split CSV is read again and its SHA256 is compared with the hash
  recorded in ``CODEX-FREEZE-20260926.json``;
* the G: manifest SHA256 is compared with the hash the freeze decision recorded;
* every one of the 24 MATs is re-checked for existence, size, SHA256 and RF0
  readability (``--sha256-mode recompute`` re-reads all bytes);
* every window of the produced index is re-derived from the stored sample count
  (``floor(source_samples / 1_000_000)`` complete, non-overlapping 1M windows)
  and compared field by field with the CSV;
* the legacy experiment artifacts are hashed to prove this step did not touch
  them;
* a set of mutated copies of the index must be rejected (negative controls), so
  the checks are shown to be able to fail.

Usage::

    python dronerfa_baseline/verify_newsplit_index.py --sha256-mode recompute
    python dronerfa_baseline/verify_newsplit_index.py --sha256-mode cached --no-read-through

Exit code 0 only when every check and every negative control passes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_FREEZE_JSON = PROJECT_ROOT / "work" / "meta_split2" / "CODEX-FREEZE-20260926.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "cascade3_newsplit_20260926"
DEFAULT_DATA_ROOT = Path(r"G:\DroneRFa数据集\original")

TARGET_CODES = ("T0000", "T0010", "T0101")
CLASS_INDEX = {"T0000": 0, "T0010": 1, "T0101": 2}
CLASS_NAME = {"T0000": "Background", "T0010": "DJI Phantom 4 Pro", "T0101": "DJI Air 2S"}
STAGE2_LABEL = {"T0010": 0, "T0101": 1}
ROLE_ORDER = ("train", "val", "development_test_exposed")
ROLE_FILES_PER_CLASS = {"train": 4, "val": 2, "development_test_exposed": 2}
WINDOW_SIZE = 1_000_000
BLOCK_SIZE = 10_000_000
REQUIRED_DATASETS = ("RF0_I", "RF0_Q")
FORBIDDEN_PATH_FRAGMENTS = (
    "F:\\",
    "F:/",
    "junction",
    "Documents\\DroneRFa",
    "Documents/DroneRFa",
)
#: Provenance-frozen legacy artifacts that this step must not have modified.
PROTECTED_ARTIFACTS = {
    "outputs/window_index.csv": (
        "4e77c7f186c1aae8e04f2e90f2769c0ed399cc94f40b64f99237b65c057b85ac",
        284507,
    ),
    "outputs/heldout_preparation_20260917/heldout_window_index.csv": (
        "6fe32faeb5d583a359d9e8aa17caa52a7c4567d545272ded25266767cd2cffcf",
        267537,
    ),
}
FROZEN_MANIFEST_SHA256 = "9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d"
FROZEN_SPLIT_SHA256 = "d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9"
FROZEN_SELECTION_FILES = 24
FROZEN_COUNTS = {
    "total_sources": 104,
    "train": 62,
    "val": 16,
    "development_test": 20,
    "development_test_exposed": 6,
}


# ---------------------------------------------------------------------------
# small helpers (independent of the builder)
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stored_length(dataset: h5py.Dataset) -> int:
    if dataset.ndim == 1:
        return int(dataset.shape[0])
    if dataset.ndim == 2 and 1 in dataset.shape:
        return int(max(dataset.shape))
    raise ValueError(f"unsupported dataset shape: {dataset.shape}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


@dataclass
class Check:
    check_id: str
    group: str
    passed: bool
    expected: Any
    actual: Any
    evidence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "group": self.group,
            "passed": bool(self.passed),
            "expected": self.expected,
            "actual": self.actual,
            "evidence": self.evidence,
        }


def json_safe(value: Any) -> Any:
    """Make a check payload JSON-serialisable (sets, tuple dict keys)."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((json_safe(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def compare(check_id: str, group: str, expected: Any, actual: Any, evidence: str = "") -> Check:
    expected = json_safe(expected)
    actual = json_safe(actual)
    return Check(check_id, group, expected == actual, expected, actual, evidence)


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------
@dataclass
class Context:
    """Everything a check may look at; mutated copies drive negative controls."""

    freeze_json: Path
    output_dir: Path
    data_root: Path
    sha256_mode: str = "cached"
    do_read_through: bool = False
    #: Expectations. ``load_context`` pins them to the frozen constants; only an
    #: in-process test may relax them for a synthetic authority tree.
    expected_frozen_csv_sha256: str = FROZEN_SPLIT_SHA256
    expected_manifest_sha256: str = FROZEN_MANIFEST_SHA256
    expected_selection_files: int = FROZEN_SELECTION_FILES
    expected_role_files_per_class: dict[str, int] = field(
        default_factory=lambda: dict(ROLE_FILES_PER_CLASS)
    )
    expected_freeze_counts: dict[str, Any] = field(default_factory=lambda: dict(FROZEN_COUNTS))
    freeze: dict[str, Any] = field(default_factory=dict)
    frozen_rows: list[dict[str, str]] = field(default_factory=list)
    frozen_csv_path: Path = Path()
    frozen_csv_sha256: str = ""
    manifest_path: Path = Path()
    manifest_sha256: str = ""
    index_rows: list[dict[str, str]] = field(default_factory=list)
    stage2_rows: list[dict[str, str]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)
    index_path: Path = Path()
    stage2_path: Path = Path()
    summary_path: Path = Path()
    identity_path: Path = Path()
    scheme_path: Path = Path()
    index_sha256: str = ""
    stage2_sha256: str = ""
    source_samples: dict[str, int] = field(default_factory=dict)
    source_sha256: dict[str, str] = field(default_factory=dict)
    hdf5_probe: dict[str, Any] = field(default_factory=dict)
    protected_hashes: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_context_into(ctx: Context) -> Context:
    """Fill a context (whose paths are already set) from the authority and artifacts."""
    ctx.freeze = json.loads(ctx.freeze_json.read_text(encoding="utf-8"))

    raw_csv = str(ctx.freeze["frozen_split_csv"])
    candidate = Path(raw_csv)
    ctx.frozen_csv_path = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
    ctx.frozen_csv_sha256 = sha256_file(ctx.frozen_csv_path)
    ctx.manifest_path = Path(str(ctx.freeze["source_manifest"]))
    ctx.manifest_sha256 = sha256_file(ctx.manifest_path)

    ctx.frozen_rows = [
        row
        for row in read_csv_rows(ctx.frozen_csv_path)
        if str(row.get("class_id", "")) in TARGET_CODES
    ]

    ctx.index_path = ctx.output_dir / "window_index_newsplit3.csv"
    ctx.stage2_path = ctx.output_dir / "stage2_view_newsplit3.csv"
    ctx.summary_path = ctx.output_dir / "newsplit3_dataset_summary.json"
    ctx.identity_path = ctx.output_dir / "newsplit3_source_identity.json"
    ctx.scheme_path = ctx.output_dir / "newsplit3_scheme.md"

    ctx.index_rows = read_csv_rows(ctx.index_path)
    ctx.stage2_rows = read_csv_rows(ctx.stage2_path)
    ctx.summary = json.loads(ctx.summary_path.read_text(encoding="utf-8"))
    ctx.identity = json.loads(ctx.identity_path.read_text(encoding="utf-8"))
    ctx.index_sha256 = sha256_file(ctx.index_path)
    ctx.stage2_sha256 = sha256_file(ctx.stage2_path)
    for item in ctx.identity.get("files", []):
        ctx.source_sha256[str(item.get("filename"))] = str(item.get("sha256", "")).lower()

    for name in PROTECTED_ARTIFACTS:
        path = SCRIPT_DIR / name
        ctx.protected_hashes[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "size": path.stat().st_size if path.is_file() else None,
        }
    return ctx


def load_context(args: argparse.Namespace) -> Context:
    freeze_json = args.freeze_json
    if not freeze_json.is_absolute():
        freeze_json = (PROJECT_ROOT / freeze_json).resolve()
    ctx = Context(
        freeze_json=freeze_json,
        output_dir=args.output_dir,
        data_root=args.data_root,
        sha256_mode=args.sha256_mode,
        do_read_through=not args.no_read_through,
    )
    return load_context_into(ctx)


# ---------------------------------------------------------------------------
# group A: authority and source identity
# ---------------------------------------------------------------------------
def group_a_authority(ctx: Context) -> list[Check]:
    checks: list[Check] = []
    checks.append(
        compare(
            "A1_freeze_schema",
            "authority",
            ("dronerfa-meta-split2-codex-freeze/v1", "accepted_frozen_development_split"),
            (ctx.freeze.get("schema"), ctx.freeze.get("decision")),
            str(ctx.freeze_json),
        )
    )
    checks.append(
        compare(
            "A2_frozen_split_csv_sha256",
            "authority",
            ctx.expected_frozen_csv_sha256,
            ctx.frozen_csv_sha256,
            str(ctx.frozen_csv_path),
        )
    )
    checks.append(
        compare(
            "A3_freeze_declares_same_csv_sha256",
            "authority",
            str(ctx.freeze.get("frozen_split_csv_sha256")),
            ctx.frozen_csv_sha256,
        )
    )
    checks.append(
        compare(
            "A4_manifest_sha256",
            "authority",
            ctx.expected_manifest_sha256,
            ctx.manifest_sha256,
            str(ctx.manifest_path),
        )
    )
    checks.append(
        compare(
            "A5_freeze_declares_same_manifest_sha256",
            "authority",
            str(ctx.freeze.get("source_manifest_sha256")),
            ctx.manifest_sha256,
        )
    )

    role_counts = {
        f"{code}:{role}": sum(
            1
            for row in ctx.frozen_rows
            if row["class_id"] == code and row["role"] == role
        )
        for code in TARGET_CODES
        for role in ROLE_ORDER
    }
    checks.append(
        compare(
            "A6_frozen_selection_4_2_2",
            "authority",
            {
                f"{code}:{role}": ctx.expected_role_files_per_class[role]
                for code in TARGET_CODES
                for role in ROLE_ORDER
            },
            role_counts,
            "files per class and role in the frozen CSV, keyed class:role",
        )
    )
    checks.append(
        compare(
            "A7_frozen_selection_file_count",
            "authority",
            ctx.expected_selection_files,
            len(ctx.frozen_rows),
            "T0000 + T0010 + T0101 rows of the frozen split CSV",
        )
    )
    checks.append(
        compare(
            "A8_freeze_counts",
            "authority",
            ctx.expected_freeze_counts,
            ctx.freeze.get("counts"),
        )
    )

    path_problems: list[str] = []
    missing: list[str] = []
    size_problems: list[str] = []
    for row in ctx.frozen_rows:
        path = Path(str(row["g_path"]))
        if not path.is_file():
            missing.append(str(path))
            continue
        if path.stat().st_size != int(row["size_bytes"]):
            size_problems.append(
                f"{path.name}: {path.stat().st_size} != {row['size_bytes']}"
            )
        if not _is_within(path, ctx.data_root):
            path_problems.append(f"{path} outside {ctx.data_root}")
        for fragment in FORBIDDEN_PATH_FRAGMENTS:
            if fragment in str(path):
                path_problems.append(f"{path} contains forbidden fragment {fragment!r}")
    checks.append(compare("A9_all_paths_exist", "authority", [], missing))
    checks.append(compare("A10_sizes_match_frozen_csv", "authority", [], size_problems))
    checks.append(
        compare(
            "A11_paths_inside_data_root_and_clean",
            "authority",
            [],
            path_problems,
            f"data root {ctx.data_root}",
        )
    )
    return checks


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def group_a_hashes(ctx: Context) -> list[Check]:
    """SHA256 of the 24 MATs; the heavy mode re-reads every byte."""
    checks: list[Check] = []
    problems: list[str] = []
    recomputed = 0
    for row in ctx.frozen_rows:
        path = Path(str(row["g_path"]))
        expected = str(row["mat_sha256"]).lower()
        if ctx.sha256_mode == "recompute":
            actual = sha256_file(path)
            recomputed += 1
        else:
            actual = ctx.source_sha256.get(path.name, "")
        if actual != expected:
            problems.append(f"{path.name}: {actual or 'missing'} != {expected}")
    checks.append(
        compare(
            "A12_mat_sha256_matches_frozen_csv",
            "authority",
            [],
            problems,
            f"{len(ctx.frozen_rows)} files, {recomputed} recomputed from disk "
            f"(mode={ctx.sha256_mode})",
        )
    )
    duplicate_hashes = [
        digest
        for digest in {str(row["mat_sha256"]).lower() for row in ctx.frozen_rows}
        if sum(1 for row in ctx.frozen_rows if str(row["mat_sha256"]).lower() == digest) > 1
    ]
    checks.append(
        compare(
            "A13_mat_sha256_pairwise_distinct",
            "authority",
            [],
            sorted(duplicate_hashes),
            "no two of the 24 source MATs share a hash",
        )
    )
    return checks


def group_a_rf0(ctx: Context) -> list[Check]:
    """RF0 readability and the stored sample counts used by the window rule."""
    checks: list[Check] = []
    problems: list[str] = []
    probing = ctx.sha256_mode == "recompute" or not ctx.source_samples
    for row in ctx.frozen_rows:
        filename = str(row["file_name"])
        path = Path(str(row["g_path"]))
        if filename in ctx.source_samples and not probing:
            continue
        try:
            with h5py.File(path, "r") as handle:
                missing = [key for key in REQUIRED_DATASETS if key not in handle]
                if missing:
                    problems.append(f"{filename}: missing {missing}")
                    continue
                lengths = {key: stored_length(handle[key]) for key in REQUIRED_DATASETS}
                if len(set(lengths.values())) != 1:
                    problems.append(f"{filename}: RF0_I/RF0_Q lengths differ {lengths}")
                    continue
                samples = next(iter(lengths.values()))
                ctx.source_samples[filename] = samples
                ctx.hdf5_probe[filename] = {
                    "datasets": sorted(handle.keys()),
                    "rf0_shape": list(handle["RF0_I"].shape),
                    "rf0_dtype": str(handle["RF0_I"].dtype),
                    "samples": samples,
                }
                if samples < WINDOW_SIZE:
                    problems.append(f"{filename}: only {samples} samples < one window")
        except OSError as error:  # pragma: no cover - unreadable file
            problems.append(f"{filename}: {error}")
    checks.append(
        compare(
            "A14_rf0_readable_equal_length",
            "authority",
            [],
            problems,
            f"{len(ctx.source_samples)} of {len(ctx.frozen_rows)} MATs probed through "
            "h5py (RF0_I/RF0_Q)",
        )
    )
    derived = {
        filename: samples // WINDOW_SIZE for filename, samples in ctx.source_samples.items()
    }
    checks.append(
        compare(
            "A15_frozen_rows_have_stored_samples",
            "authority",
            len(ctx.frozen_rows),
            len(ctx.source_samples),
            "every frozen MAT yielded a stored sample count",
        )
    )
    checks.append(
        compare(
            "A16_window_count_derived_from_disk",
            "authority",
            sum(derived.values()),
            len(ctx.index_rows),
            f"sum(floor(source_samples/1e6)) = {sum(derived.values())}",
        )
    )
    return checks


# ---------------------------------------------------------------------------
# group B: index structure
# ---------------------------------------------------------------------------
def _to_int(value: Any, default: int = -1) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def group_b_index(ctx: Context) -> list[Check]:
    checks: list[Check] = []
    rows = ctx.index_rows
    frozen_by_file = {str(row["file_name"]): row for row in ctx.frozen_rows}
    frozen_role = {name: str(row["role"]) for name, row in frozen_by_file.items()}
    frozen_code = {name: str(row["class_id"]) for name, row in frozen_by_file.items()}
    frozen_samples = {name: ctx.source_samples.get(name) for name in frozen_by_file}

    checks.append(
        compare(
            "B1_index_sha256_matches_summary",
            "index",
            str(ctx.summary.get("index", {}).get("sha256")),
            ctx.index_sha256,
            str(ctx.index_path),
        )
    )
    checks.append(
        compare(
            "B2_row_count_matches_summary",
            "index",
            int(ctx.summary.get("index", {}).get("rows", -1)),
            len(rows),
        )
    )
    checks.append(
        compare(
            "B3_fields_match_summary",
            "index",
            list(ctx.summary.get("index", {}).get("fields", [])),
            list(rows[0].keys()) if rows else [],
        )
    )

    # per-file coverage: exact index sequence, alignment, in-file span
    coverage_problems: list[str] = []
    overlap_problems: list[str] = []
    label_problems: list[str] = []
    role_problems: list[str] = []
    sample_problems: list[str] = []
    window_rule_values: set[str] = set()
    seen_files: set[str] = set()
    row_index_problems: int = 0
    by_file: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_file.setdefault(str(row["filename"]), []).append(row)
    for position, row in enumerate(rows):
        if _to_int(row.get("row_index")) != position:
            row_index_problems += 1
        window_rule_values.add(str(row.get("window_rule", "")))

    for filename, file_rows in by_file.items():
        seen_files.add(filename)
        if filename not in frozen_by_file:
            coverage_problems.append(f"{filename}: not part of the frozen selection")
            continue
        expected_samples = frozen_samples.get(filename)
        expected_windows = (
            None if expected_samples is None else expected_samples // WINDOW_SIZE
        )
        ordered = sorted(file_rows, key=lambda item: _to_int(item["start"]))
        indices = [_to_int(item["window_index"]) for item in ordered]
        if indices != list(range(len(ordered))):
            coverage_problems.append(
                f"{filename}: window_index {indices[:5]}... is not 0..{len(ordered) - 1}"
            )
        if expected_windows is None:
            coverage_problems.append(f"{filename}: no stored sample count to compare with")
        elif len(ordered) != expected_windows:
            coverage_problems.append(
                f"{filename}: {len(ordered)} windows indexed, disk implies {expected_windows}"
            )
        spans: list[tuple[int, int]] = []
        for item in ordered:
            start = _to_int(item["start"])
            length = _to_int(item["length"])
            samples = _to_int(item["source_samples"])
            index = _to_int(item["window_index"])
            if length != WINDOW_SIZE:
                coverage_problems.append(f"{filename}: length {length} != {WINDOW_SIZE}")
            if start % WINDOW_SIZE != 0:
                coverage_problems.append(f"{filename}: start {start} not 1M aligned")
            if start != index * WINDOW_SIZE:
                coverage_problems.append(
                    f"{filename}: start {start} != window_index {index} * {WINDOW_SIZE}"
                )
            if _to_int(item["block_index"]) != start // BLOCK_SIZE:
                coverage_problems.append(f"{filename}: block_index mismatch at {start}")
            if _to_int(item["window_in_block"]) != (start % BLOCK_SIZE) // WINDOW_SIZE:
                coverage_problems.append(f"{filename}: window_in_block mismatch at {start}")
            if start + length > samples:
                coverage_problems.append(f"{filename}: window {start} runs past {samples}")
            if frozen_samples.get(filename) is not None and samples != frozen_samples[filename]:
                sample_problems.append(
                    f"{filename}: source_samples {samples} != disk {frozen_samples[filename]}"
                )
            spans.append((start, start + length))
        for (_, previous_end), (next_start, _) in zip(spans, spans[1:]):
            if previous_end != next_start:
                overlap_problems.append(
                    f"{filename}: window boundary {previous_end} -> {next_start}"
                )

        # labels and roles are constant per source file and come from the frozen CSV
        code = frozen_code[filename]
        role = frozen_role[filename]
        if len({str(item["target_code"]) for item in ordered}) != 1:
            label_problems.append(f"{filename}: several target codes in one file")
        if len({str(item["source_role"]) for item in ordered}) != 1:
            role_problems.append(f"{filename}: several source roles in one file")
        for item in ordered:
            if str(item["target_code"]) != code:
                label_problems.append(f"{filename}: target_code {item['target_code']} != frozen {code}")
                break
        for item in ordered:
            if str(item["source_role"]) != role or str(item["split"]) != role:
                role_problems.append(
                    f"{filename}: role {item['source_role']}/{item['split']} != frozen {role}"
                )
                break

    missing_files = sorted(set(frozen_by_file) - seen_files)
    if missing_files:
        coverage_problems.append(f"files with no indexed window: {missing_files}")

    # label semantics over every row
    for row in rows:
        code = str(row["target_code"])
        class_index = _to_int(row["class_index"])
        stage1 = _to_int(row["stage1_label"])
        eligible = _to_int(row["stage2_eligible"])
        expected_class = CLASS_INDEX.get(code, -99)
        if class_index != expected_class:
            label_problems.append(f"row {row['row_index']}: class_index {class_index} != {expected_class}")
        if str(row["class_name"]) != CLASS_NAME.get(code, "?"):
            label_problems.append(f"row {row['row_index']}: class_name {row['class_name']}")
        if stage1 != (0 if expected_class == 0 else 1):
            label_problems.append(f"row {row['row_index']}: stage1_label {stage1}")
        if _to_int(row["label"]) != stage1:
            label_problems.append(f"row {row['row_index']}: label != stage1_label")
        if eligible != (0 if expected_class == 0 else 1):
            label_problems.append(f"row {row['row_index']}: stage2_eligible {eligible}")
        if eligible == 0:
            if str(row["stage2_label"]) != "":
                label_problems.append(f"row {row['row_index']}: background has stage2_label")
        elif _to_int(row["stage2_label"]) != STAGE2_LABEL.get(code, -99):
            label_problems.append(f"row {row['row_index']}: stage2_label {row['stage2_label']}")
    if len(label_problems) > 12:
        label_problems = label_problems[:12] + ["..."]

    checks.append(compare("B4_row_index_is_contiguous", "index", 0, row_index_problems))
    checks.append(
        compare(
            "B5_per_file_window_coverage_exact",
            "index",
            [],
            coverage_problems,
            "window_index 0..n-1, start = index*1M, length 1M, inside stored samples",
        )
    )
    checks.append(
        compare(
            "B6_no_gap_no_overlap",
            "index",
            [],
            overlap_problems,
            "windows tile each source file contiguously with stride == window size",
        )
    )
    checks.append(compare("B7_label_semantics", "index", [], label_problems))
    checks.append(
        compare(
            "B8_source_role_matches_frozen_csv",
            "index",
            [],
            role_problems,
            "role/split copied from frozen CSV, one role per source file",
        )
    )
    checks.append(
        compare(
            "B9_source_samples_match_disk",
            "index",
            [],
            sample_problems,
        )
    )
    checks.append(
        compare(
            "B10_single_window_rule_string",
            "index",
            1,
            len(window_rule_values),
            f"values: {sorted(window_rule_values)[:2]}",
        )
    )

    # frozen bookkeeping columns carried into the index
    bookkeeping_problems: list[str] = []
    for filename, file_rows in by_file.items():
        if filename not in frozen_by_file:
            continue
        frozen = frozen_by_file[filename]
        expected = {
            "metrics_scope": str(frozen["metrics_scope"]),
            "prior_use_role": str(frozen["prior_use_role"]),
            "archive_segment_role": str(frozen["archive_segment_role"]),
            "exposed_to_prior_selection": str(frozen["exposed_to_prior_selection"]),
            "role_purpose": str(frozen["role_purpose"]),
            "file_sha256": str(frozen["mat_sha256"]).lower(),
        }
        for key, value in expected.items():
            if str(file_rows[0][key]) != value:
                bookkeeping_problems.append(
                    f"{filename}: {key} {file_rows[0][key]!r} != frozen {value!r}"
                )
    checks.append(
        compare(
            "B11_frozen_bookkeeping_columns_copied",
            "index",
            [],
            bookkeeping_problems,
            "metrics_scope / prior_use_role / archive_segment_role / exposure / sha256",
        )
    )

    # role totals recomputed from the frozen CSV
    role_windows: dict[str, int] = {role: 0 for role in ROLE_ORDER}
    class_windows: dict[str, int] = {code: 0 for code in TARGET_CODES}
    for row in rows:
        role_windows[str(row["source_role"])] = role_windows.get(str(row["source_role"]), 0) + 1
        class_windows[str(row["target_code"])] = class_windows.get(str(row["target_code"]), 0) + 1
    checks.append(
        compare(
            "B12_windows_by_role_matches_summary",
            "index",
            ctx.summary.get("counts", {}).get("windows_by_role"),
            role_windows,
        )
    )
    checks.append(
        compare(
            "B13_windows_by_class_matches_summary",
            "index",
            ctx.summary.get("counts", {}).get("windows_by_class"),
            class_windows,
        )
    )
    index_file_role: dict[str, set[str]] = {}
    for row in rows:
        index_file_role.setdefault(str(row["filename"]), set()).add(str(row["source_role"]))
    checks.append(
        compare(
            "B14_file_to_role_map_is_the_frozen_map",
            "index",
            {name: [role] for name, role in sorted(frozen_role.items())},
            {name: sorted(roles) for name, roles in sorted(index_file_role.items())},
            "no source MAT appears under two roles",
        )
    )
    return checks


# ---------------------------------------------------------------------------
# group C: stage-2 view
# ---------------------------------------------------------------------------
def group_c_stage2(ctx: Context) -> list[Check]:
    checks: list[Check] = []
    expected_rows = [row for row in ctx.index_rows if _to_int(row["stage2_eligible"]) == 1]
    checks.append(
        compare(
            "C1_view_is_exact_filter_of_index",
            "stage2",
            [str(row["row_index"]) for row in expected_rows],
            [str(row["row_index"]) for row in ctx.stage2_rows],
            "view keeps index order and only stage2_eligible rows",
        )
    )
    problems: list[str] = []
    per_split = {role: 0 for role in ROLE_ORDER}
    for position, row in enumerate(ctx.stage2_rows):
        if _to_int(row["stage2_position"]) != position:
            problems.append(f"row {position}: stage2_position {row['stage2_position']}")
        role = str(row["source_role"])
        if _to_int(row["stage2_position_in_split"]) != per_split.get(role, 0):
            problems.append(
                f"row {position}: stage2_position_in_split "
                f"{row['stage2_position_in_split']} != {per_split.get(role, 0)} for {role}"
            )
        per_split[role] = per_split.get(role, 0) + 1
        if str(row["target_code"]) == "T0000":
            problems.append(f"row {position}: background row inside the stage-2 view")
        if _to_int(row["stage2_label"]) not in (0, 1):
            problems.append(f"row {position}: stage2_label {row['stage2_label']}")
        if str(row["split"]) != role:
            problems.append(f"row {position}: split {row['split']} != {role}")
    checks.append(
        compare(
            "C2_positions_stable_and_background_free",
            "stage2",
            [],
            problems,
            "global 0-based stage2_position plus per-split 0-based position",
        )
    )
    checks.append(
        compare(
            "C3_view_sha256_matches_summary",
            "stage2",
            str(ctx.summary.get("stage2_view", {}).get("sha256")),
            ctx.stage2_sha256,
        )
    )
    counts_by_role_class = {
        role: {
            code: sum(
                1
                for row in ctx.stage2_rows
                if str(row["source_role"]) == role and str(row["target_code"]) == code
            )
            for code in ("T0010", "T0101")
        }
        for role in ROLE_ORDER
    }
    summary_stage2 = ctx.summary.get("stage2_view", {}).get("counts", {}).get(
        "windows_by_role_class", {}
    )
    checks.append(
        compare(
            "C4_view_counts_match_summary",
            "stage2",
            counts_by_role_class,
            {
                role: {
                    code: summary_stage2.get(role, {}).get(code)
                    for code in ("T0010", "T0101")
                }
                for role in ROLE_ORDER
            },
        )
    )
    expected_per_split = {
        role: sum(
            1
            for row in ctx.index_rows
            if str(row["source_role"]) == role and _to_int(row["stage2_eligible"]) == 1
        )
        for role in ROLE_ORDER
    }
    checks.append(
        compare(
            "C5_view_per_split_counts_equal_drone_rows_of_index",
            "stage2",
            expected_per_split,
            per_split,
            "stage-2 view keeps every drone window of the three-class index exactly once",
        )
    )
    return checks


# ---------------------------------------------------------------------------
# group D: isolation from the frozen legacy experiments
# ---------------------------------------------------------------------------
def group_d_isolation(ctx: Context) -> list[Check]:
    checks: list[Check] = []
    problems: list[str] = []
    for name, (digest, size) in PROTECTED_ARTIFACTS.items():
        record = ctx.protected_hashes.get(name, {})
        if not record.get("exists"):
            problems.append(f"{name}: missing")
            continue
        if record.get("sha256") != digest:
            problems.append(f"{name}: sha256 {record.get('sha256')} != {digest}")
        if record.get("size") != size:
            problems.append(f"{name}: size {record.get('size')} != {size}")
    checks.append(
        compare(
            "D1_legacy_indexes_unchanged",
            "isolation",
            [],
            problems,
            "outputs/window_index.csv and the heldout index keep their frozen hashes",
        )
    )

    new_files = sorted(path.name for path in ctx.output_dir.iterdir() if path.is_file())
    known = {
        "window_index_newsplit3.csv",
        "stage2_view_newsplit3.csv",
        "newsplit3_dataset_summary.json",
        "newsplit3_source_identity.json",
        "newsplit3_scheme.md",
        "newsplit3_verification.json",
    }
    checks.append(
        compare(
            "D2_outputs_confined_to_new_directory",
            "isolation",
            [],
            sorted(set(new_files) - known),
            f"new artifact directory holds {new_files}",
        )
    )
    forbidden = [
        str(row["file_path"])
        for row in ctx.index_rows
        if any(fragment in str(row["file_path"]) for fragment in FORBIDDEN_PATH_FRAGMENTS)
    ]
    checks.append(
        compare(
            "D3_no_legacy_or_foreign_paths_in_index",
            "isolation",
            [],
            forbidden,
            "no F: drive, junction or Documents path in the index",
        )
    )
    outside = [
        str(row["file_path"])
        for row in ctx.index_rows
        if not _is_within(Path(str(row["file_path"])), ctx.data_root)
    ]
    checks.append(
        compare(
            "D4_every_index_path_inside_data_root",
            "isolation",
            [],
            outside,
            f"data root {ctx.data_root}",
        )
    )
    checks.append(
        compare(
            "D5_no_sampling_or_rule_change",
            "isolation",
            (False, "all_complete_1m_windows"),
            (
                bool(ctx.summary.get("sampling", {}).get("applied")),
                "all_complete_1m_windows",
            ),
            str(ctx.summary.get("sampling", {}).get("note", ""))[:80],
        )
    )
    return checks


# ---------------------------------------------------------------------------
# group E: summary document vs recomputation
# ---------------------------------------------------------------------------
def group_e_summary(ctx: Context) -> list[Check]:
    checks: list[Check] = []
    rows = ctx.index_rows
    frozen_by_file = {str(row["file_name"]): row for row in ctx.frozen_rows}

    recomputed_files: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row["filename"])
        entry = recomputed_files.setdefault(
            name,
            {
                "filename": name,
                "target_code": str(row["target_code"]),
                "source_role": str(row["source_role"]),
                "windows": 0,
                "source_samples": _to_int(row["source_samples"]),
                "sha256": str(row["file_sha256"]),
            },
        )
        entry["windows"] += 1

    summary_files = {str(item["filename"]): item for item in ctx.summary.get("files", [])}
    problems: list[str] = []
    if set(summary_files) != set(recomputed_files):
        problems.append(
            f"summary files {sorted(set(summary_files) ^ set(recomputed_files))[:4]} differ"
        )
    for name, entry in recomputed_files.items():
        item = summary_files.get(name)
        if item is None:
            continue
        for key in ("target_code", "source_role", "windows", "source_samples", "sha256"):
            if str(item.get(key)) != str(entry[key]):
                problems.append(f"{name}: summary {key} {item.get(key)} != {entry[key]}")
        frozen = frozen_by_file.get(name, {})
        if str(item.get("g_path")) != str(frozen.get("g_path")):
            problems.append(f"{name}: summary g_path differs from frozen CSV")
        if str(item.get("sha256_source")) not in ("recomputed", "frozen_csv"):
            problems.append(f"{name}: unknown sha256_source {item.get('sha256_source')!r}")
    checks.append(
        compare(
            "E1_summary_per_file_recomputed",
            "summary",
            [],
            problems,
            "24 entries: role, class, windows, samples, sha256, g_path",
        )
    )

    tail_problems = [
        f"{name}: {item.get('dropped_tail_samples')}"
        for name, item in summary_files.items()
        if _to_int(item.get("dropped_tail_samples")) != (
            _to_int(item.get("source_samples")) % WINDOW_SIZE
        )
    ]
    total_tail = sum(_to_int(item.get("dropped_tail_samples")) for item in summary_files.values())
    checks.append(
        compare(
            "E2_dropped_tail_is_the_sub_window_remainder",
            "summary",
            [],
            tail_problems,
            f"total dropped tail samples = {total_tail}",
        )
    )
    checks.append(
        compare(
            "E3_summary_totals",
            "summary",
            (
                len(rows),
                ctx.expected_selection_files,
                ctx.stage2_sha256,
            ),
            (
                int(ctx.summary.get("counts", {}).get("total_windows", -1)),
                int(ctx.summary.get("source_file_count", -1)),
                str(ctx.summary.get("stage2_view", {}).get("sha256")),
            ),
        )
    )
    checks.append(
        compare(
            "E4_identity_document_lists_every_source",
            "summary",
            ctx.expected_selection_files,
            len(ctx.identity.get("files", [])),
            str(ctx.identity_path),
        )
    )
    identity_sha_problems = [
        item["filename"]
        for item in ctx.identity.get("files", [])
        if str(item.get("frozen_sha256", "")).lower()
        != str(item.get("sha256", "")).lower()
    ]
    checks.append(
        compare(
            "E5_identity_sha256_agrees_with_frozen_hash",
            "summary",
            [],
            identity_sha_problems,
        )
    )
    scheme_text = ctx.scheme_path.read_text(encoding="utf-8") if ctx.scheme_path.is_file() else ""
    checks.append(
        compare(
            "E6_scheme_document_present_and_names_authority",
            "summary",
            True,
            all(
                token in scheme_text
                for token in (
                    ctx.expected_frozen_csv_sha256,
                    ctx.expected_manifest_sha256,
                    "development_test_exposed",
                )
            ),
            f"{len(scheme_text)} characters",
        )
    )
    return checks


# ---------------------------------------------------------------------------
# read-through probe: every window reachable through the project reader
# ---------------------------------------------------------------------------
def read_through_probe(ctx: Context, sample_positions: int = 3) -> dict[str, Any]:
    from dronerfa_data import read_vector_slice  # imported late; the real reader

    records: list[dict[str, Any]] = []
    problems: list[str] = []
    by_file: dict[str, list[dict[str, str]]] = {}
    for row in ctx.index_rows:
        by_file.setdefault(str(row["filename"]), []).append(row)

    for filename, file_rows in sorted(by_file.items()):
        ordered = sorted(file_rows, key=lambda item: _to_int(item["start"]))
        picks = sorted(
            {
                0,
                len(ordered) // 2,
                len(ordered) - 1,
            }
        )[:sample_positions]
        starts = [ordered[position] for position in picks]
        path = Path(str(ordered[0]["file_path"]))
        try:
            with h5py.File(path, "r") as handle:
                for row in starts:
                    start = _to_int(row["start"])
                    length = _to_int(row["length"])
                    began = time.time()
                    i_values = read_vector_slice(handle["RF0_I"], start, length)
                    q_values = read_vector_slice(handle["RF0_Q"], start, length)
                    elapsed = time.time() - began
                    finite = bool(
                        i_values.shape == (length,)
                        and q_values.shape == (length,)
                        and (i_values == i_values).all()
                        and (q_values == q_values).all()
                    )
                    records.append(
                        {
                            "filename": filename,
                            "start": start,
                            "length": length,
                            "finite": finite,
                            "seconds": round(elapsed, 3),
                            "mb_per_s": round(length * 16 / 1e6 / max(elapsed, 1e-6), 1),
                        }
                    )
                    if not finite:
                        problems.append(f"{filename}:{start} not finite or wrong length")
        except OSError as error:  # pragma: no cover
            problems.append(f"{filename}: {error}")

    seconds = [record["seconds"] for record in records]
    return {
        "files_probed": len(by_file),
        "windows_probed": len(records),
        "positions": "first, middle, last indexed window of every source MAT",
        "problems": problems,
        "median_seconds_per_window": (
            round(sorted(seconds)[len(seconds) // 2], 3) if seconds else None
        ),
        "max_seconds_per_window": round(max(seconds), 3) if seconds else None,
        "samples": records,
    }


# ---------------------------------------------------------------------------
# negative controls
# ---------------------------------------------------------------------------
def _fresh_rows(ctx: Context) -> list[dict[str, str]]:
    return copy.deepcopy(ctx.index_rows)


def _mutated_context(ctx: Context, index_rows: list[dict[str, str]], **kwargs: Any) -> Context:
    clone = copy.deepcopy(ctx)
    clone.index_rows = index_rows
    # negative controls must stay cheap: no re-read of the 24 MATs
    clone.sha256_mode = "cached"
    for key, value in kwargs.items():
        setattr(clone, key, value)
    return clone


def negative_controls(ctx: Context, check_runner: Callable[[Context], list[Check]]) -> list[dict[str, Any]]:
    """Every mutation must make at least one check fail."""

    def _first_row_of_first_file(rows: list[dict[str, str]]) -> dict[str, str]:
        first_file = sorted({row["filename"] for row in rows})[0]
        return next(row for row in rows if row["filename"] == first_file)

    def _last_row_of_largest_file(rows: list[dict[str, str]]) -> dict[str, str]:
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["filename"]] = counts.get(row["filename"], 0) + 1
        filename = max(sorted(counts), key=lambda name: counts[name])
        file_rows = [row for row in rows if row["filename"] == filename]
        return max(file_rows, key=lambda row: _to_int(row["window_index"]))

    def drop_a_window(clone: Context) -> Context:
        rows = _fresh_rows(clone)
        rows.remove(_first_row_of_first_file(rows))
        return _mutated_context(clone, rows)

    def duplicate_a_window(clone: Context) -> Context:
        rows = _fresh_rows(clone)
        rows.append(copy.deepcopy(_first_row_of_first_file(rows)))
        return _mutated_context(clone, rows)

    def shift_a_window(clone: Context) -> Context:
        rows = _fresh_rows(clone)
        victim = _last_row_of_largest_file(rows)
        victim["start"] = str(_to_int(victim["start"]) - WINDOW_SIZE)
        return _mutated_context(clone, rows)

    def flip_a_label(clone: Context) -> Context:
        rows = _fresh_rows(clone)
        victim = next(row for row in rows if row["target_code"] == "T0010")
        victim["class_index"] = "2"
        return _mutated_context(clone, rows)

    def cross_a_role(clone: Context) -> Context:
        rows = _fresh_rows(clone)
        roles = {row["source_role"] for row in rows}
        donor = "val" if "val" in roles else sorted(roles)[0]
        victim = next(row for row in rows if row["source_role"] == donor)
        target = "train" if donor != "train" else sorted(roles)[-1]
        for row in rows:
            if row["filename"] == victim["filename"]:
                row["source_role"] = target
                row["split"] = target
        return _mutated_context(clone, rows)

    def foreign_path(clone: Context) -> Context:
        rows = _fresh_rows(clone)
        rows[0]["file_path"] = f"F:\\DroneRFa数据集\\original\\{rows[0]['filename']}"
        return _mutated_context(clone, rows)

    def drop_a_file(clone: Context) -> Context:
        rows = _fresh_rows(clone)
        victim = sorted({row["filename"] for row in rows})[0]
        return _mutated_context(
            clone, [row for row in rows if row["filename"] != victim]
        )

    def background_into_stage2(clone: Context) -> Context:
        background = next(
            row for row in clone.index_rows if str(row["target_code"]) == "T0000"
        )
        injected = dict(background)
        injected["stage2_position"] = str(len(clone.stage2_rows))
        injected["stage2_position_in_split"] = "0"
        injected["stage2_label"] = "0"
        return _mutated_context(clone, clone.index_rows, stage2_rows=clone.stage2_rows + [injected])

    def tamper_summary(clone: Context) -> Context:
        summary = copy.deepcopy(clone.summary)
        summary["counts"]["windows_by_role"]["train"] += 1
        return _mutated_context(clone, clone.index_rows, summary=summary)

    def truncate_index_file(clone: Context) -> Context:
        rows = _fresh_rows(clone)
        return _mutated_context(clone, rows, index_sha256="0" * 64)

    mutations = [
        ("N1_drop_one_window", drop_a_window, "B5_per_file_window_coverage_exact"),
        ("N2_duplicate_one_window", duplicate_a_window, "B5_per_file_window_coverage_exact"),
        ("N3_overlapping_window_start", shift_a_window, "B6_no_gap_no_overlap"),
        ("N4_wrong_class_label", flip_a_label, "B7_label_semantics"),
        ("N5_source_file_crosses_roles", cross_a_role, "B8_source_role_matches_frozen_csv"),
        ("N6_foreign_F_drive_path", foreign_path, "D3_no_legacy_or_foreign_paths_in_index"),
        ("N7_missing_source_file_rows", drop_a_file, "B5_per_file_window_coverage_exact"),
        ("N8_background_row_in_stage2_view", background_into_stage2, "C2_positions_stable_and_background_free"),
        ("N9_tampered_summary_count", tamper_summary, "B12_windows_by_role_matches_summary"),
        ("N10_index_sha256_mismatch", truncate_index_file, "B1_index_sha256_matches_summary"),
    ]

    results: list[dict[str, Any]] = []
    for name, mutate, expected_check in mutations:
        # the hash group is skipped on purpose: it would re-read 24 MATs ten times
        checks = check_runner(mutate(ctx), include_hash_group=False)
        failed = [check.check_id for check in checks if not check.passed]
        results.append(
            {
                "control": name,
                "expected_to_fail": expected_check,
                "failed_checks": failed,
                "detected": expected_check in failed,
            }
        )
    return results


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
ALL_GROUPS: tuple[Callable[[Context], list[Check]], ...] = (
    group_a_authority,
    group_a_rf0,
    group_b_index,
    group_c_stage2,
    group_d_isolation,
    group_e_summary,
)


def run_checks(ctx: Context, include_hash_group: bool = True) -> list[Check]:
    checks: list[Check] = []
    for group in ALL_GROUPS:
        checks.extend(group(ctx))
    if include_hash_group:
        checks.extend(group_a_hashes(ctx))
    return checks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently verify the CASCADE3-NEWSPLIT-A window index."
    )
    parser.add_argument("--freeze-json", type=Path, default=DEFAULT_FREEZE_JSON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sha256-mode",
        choices=("recompute", "cached"),
        default="recompute",
        help="recompute reads all 24 MATs again; cached trusts the identity record",
    )
    parser.add_argument("--no-read-through", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    ctx = load_context(args)
    print(f"index      : {ctx.index_path} sha256={ctx.index_sha256} rows={len(ctx.index_rows)}")
    print(f"stage-2    : {ctx.stage2_path} sha256={ctx.stage2_sha256} rows={len(ctx.stage2_rows)}")
    print(f"sha256 mode: {args.sha256_mode}")

    checks = run_checks(ctx)
    read_through = (
        read_through_probe(ctx) if not args.no_read_through else {"skipped": True}
    )
    if not args.no_read_through:
        checks.append(
            compare(
                "F1_all_indexed_windows_readable",
                "read_through",
                [],
                read_through["problems"],
                f"{read_through['windows_probed']} windows through "
                f"dronerfa_data.read_vector_slice; median "
                f"{read_through['median_seconds_per_window']}s/window",
            )
        )

    controls = negative_controls(ctx, run_checks)
    failed = [check for check in checks if not check.passed]
    control_failures = [control for control in controls if not control["detected"]]

    report = {
        "schema": "dronerfa-cascade3-newsplit3-verification/v1",
        "task": "CASCADE3-NEWSPLIT-A",
        "verified_at_utc": utc_now(),
        "authority": {
            "freeze_json": str(ctx.freeze_json),
            "freeze_decision": ctx.freeze.get("decision"),
            "frozen_split_csv": str(ctx.frozen_csv_path),
            "frozen_split_csv_sha256": ctx.frozen_csv_sha256,
            "source_manifest": str(ctx.manifest_path),
            "source_manifest_sha256": ctx.manifest_sha256,
            "data_root": str(ctx.data_root),
            "sha256_mode": args.sha256_mode,
        },
        "artifacts": {
            "index": {
                "path": str(ctx.index_path),
                "sha256": ctx.index_sha256,
                "rows": len(ctx.index_rows),
            },
            "stage2_view": {
                "path": str(ctx.stage2_path),
                "sha256": ctx.stage2_sha256,
                "rows": len(ctx.stage2_rows),
            },
            "summary": {
                "path": str(ctx.summary_path),
                "sha256": sha256_file(ctx.summary_path),
            },
            "identity": {
                "path": str(ctx.identity_path),
                "sha256": sha256_file(ctx.identity_path),
            },
            "scheme": {
                "path": str(ctx.scheme_path),
                "sha256": sha256_file(ctx.scheme_path),
            },
        },
        "protected_artifacts": ctx.protected_hashes,
        "hdf5_probe": ctx.hdf5_probe,
        "checks": [check.as_dict() for check in checks],
        "check_summary": {
            "total": len(checks),
            "passed": len(checks) - len(failed),
            "failed": len(failed),
            "failed_ids": [check.check_id for check in failed],
        },
        "negative_controls": controls,
        "negative_control_summary": {
            "total": len(controls),
            "detected": len(controls) - len(control_failures),
            "undetected": [control["control"] for control in control_failures],
        },
        "read_through": read_through,
        "runtime_seconds": round(time.time() - started, 2),
        "conclusion": (
            "all checks and negative controls passed"
            if not failed and not control_failures
            else "FAILED: "
            + ", ".join([check.check_id for check in failed] + [c["control"] for c in control_failures])
        ),
    }
    report_path = args.report or (ctx.output_dir / "newsplit3_verification.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for check in checks:
        mark = "PASS" if check.passed else "FAIL"
        print(f"  [{mark}] {check.check_id}: {check.evidence}")
    for control in controls:
        mark = "PASS" if control["detected"] else "FAIL"
        print(
            f"  [{mark}] control {control['control']} -> "
            f"{control['failed_checks'][:2]}"
        )
    print(
        f"checks {len(checks) - len(failed)}/{len(checks)} passed; "
        f"negative controls {len(controls) - len(control_failures)}/{len(controls)} detected"
    )
    print(f"report: {report_path}")
    return 0 if not failed and not control_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
