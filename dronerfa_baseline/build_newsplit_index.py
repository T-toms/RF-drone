#!/usr/bin/env python3
"""CASCADE3-NEWSPLIT-A: window index for the frozen 2026-09-26 three-class split.

The index is derived *only* from the Codex-frozen development split
(``work/meta_split2/CODEX-FREEZE-20260926.json`` -> ``split_candidate_104_20260926.csv``).
Three target codes are used: T0000 (Background), T0010 (DJI Phantom 4 Pro) and
T0101 (DJI Air 2S); every source MAT keeps the single role the frozen CSV gives
it, so no source file can cross train/val/test.

Window rule (unchanged from the project reader):

* RF0 channel only, 100 MS/s complex baseband, read as RF0_I + 1j * RF0_Q,
* every complete 1,000,000-sample (10.0 ms) window of the stored record,
* non-overlapping, stride == window size, 1M-aligned; the trailing remainder
  (``source_samples % 1_000_000``) is dropped.

This script does not train anything and does not touch the legacy
``outputs/window_index.csv``, the heldout index, old checkpoints, old experiment
outputs, or anything on the G: data drive.

Outputs (default ``dronerfa_baseline/outputs/cascade3_newsplit_20260926/``)::

    window_index_newsplit3.csv     window-level index (all three classes)
    stage2_view_newsplit3.csv      drone-only projection with stage-2 positions
    newsplit3_dataset_summary.json machine-readable counts per class and role
    newsplit3_source_identity.json 24 source MATs: size, SHA256, HDF5 shape
    newsplit3_scheme.md            short data-scheme note
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_FREEZE_JSON = PROJECT_ROOT / "work" / "meta_split2" / "CODEX-FREEZE-20260926.json"
DEFAULT_DATA_ROOT = Path(r"G:\DroneRFa数据集\original")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "cascade3_newsplit_20260926"
DEFAULT_OUTPUT_NAME = "cascade3_newsplit_20260926"

#: Target codes of the re-trained three-class two-stage baseline.
TARGET_CODES = ("T0000", "T0010", "T0101")
CLASS_INDEX = {"T0000": 0, "T0010": 1, "T0101": 2}
CLASS_NAME = {
    "T0000": "Background",
    "T0010": "DJI Phantom 4 Pro",
    "T0101": "DJI Air 2S",
}
#: Stage-2 label space: Phantom 4 Pro -> 0, Air 2S -> 1 (background excluded).
STAGE2_LABEL = {"T0010": 0, "T0101": 1}

ROLE_ORDER = ("train", "val", "development_test_exposed")
#: Frozen file counts per class and role (4 / 2 / 2).
ROLE_FILES_PER_CLASS = {"train": 4, "val": 2, "development_test_exposed": 2}
SELECTED_FILE_COUNT = 24

WINDOW_SIZE = 1_000_000
BLOCK_SIZE = 10_000_000
SAMPLE_RATE_HZ = 100_000_000
#: RF0 is the only acquisition channel consumed by the frozen reader.
REQUIRED_DATASETS = ("RF0_I", "RF0_Q")
OPTIONAL_DATASETS = ("RF1_I", "RF1_Q")

#: Paths that must never appear in a generated index (legacy F: data drive or
#: the old C:\...\Documents junction used by the first baseline).
FORBIDDEN_PATH_FRAGMENTS = (
    "F:\\",
    "F:/",
    "junction",
    "Documents\\DroneRFa",
    "Documents/DroneRFa",
)

WINDOW_RULE = (
    "RF0 I/Q, non-overlapping 1,000,000-sample windows, stride 1,000,000, "
    "tail < 1,000,000 samples dropped"
)

FILE_PATTERN = re.compile(
    r"^T(?P<target>\d{4})_D(?P<distance>\d{2})_S(?P<segment>\d{4})\.mat$",
    re.IGNORECASE,
)

CSV_FIELDS = (
    "row_index",
    "file_path",
    "filename",
    "target_code",
    "target_name",
    "distance_code",
    "segment_code",
    "source_role",
    "split",
    "metrics_scope",
    "role_purpose",
    "prior_use_role",
    "archive_segment_role",
    "exposed_to_prior_selection",
    "class_index",
    "class_name",
    "stage1_label",
    "stage2_eligible",
    "stage2_label",
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

STAGE2_FIELDS = (
    "stage2_position",
    "stage2_position_in_split",
    "row_index",
    "file_path",
    "filename",
    "target_code",
    "target_name",
    "source_role",
    "split",
    "metrics_scope",
    "class_index",
    "class_name",
    "stage2_label",
    "window_index",
    "start",
    "length",
    "source_samples",
    "file_sha256",
)


# ---------------------------------------------------------------------------
# helpers
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


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


def resolve_authority_path(raw: str) -> Path:
    """Resolve a freeze-file path: absolute wins, otherwise project-relative."""
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


# ---------------------------------------------------------------------------
# authority
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
    segment_code: str
    distance_code: str
    metrics_scope: str
    role_purpose: str
    prior_use_role: str
    archive_segment_role: str
    exposed_to_prior_selection: str
    expected_sha256: str
    row: dict[str, str] = field(default_factory=dict)


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
    return freeze, csv_path, manifest_actual


def load_frozen_sources(
    csv_path: Path,
    targets: Sequence[str],
    role_files_per_class: dict[str, int] | None = None,
    expected_total: int = SELECTED_FILE_COUNT,
) -> list[FrozenSource]:
    """Select the target classes from the frozen split CSV and check its shape."""
    wanted = tuple(targets)
    expected_roles = dict(role_files_per_class or ROLE_FILES_PER_CLASS)
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header")
        for row in reader:
            if str(row.get("class_id", "")) in wanted:
                rows.append(dict(row))

    if len(rows) != expected_total:
        raise ValueError(
            f"expected {expected_total} frozen rows for {wanted}, got {len(rows)}"
        )

    sources: list[FrozenSource] = []
    for row in rows:
        target = str(row["class_id"])
        role = str(row["role"])
        if role not in ROLE_ORDER:
            raise ValueError(f"{row['file_name']}: unexpected role {role!r}")
        match = FILE_PATTERN.match(str(row["file_name"]))
        if match is None:
            raise ValueError(f"{row['file_name']}: unexpected DroneRFa file name")
        if f"T{match.group('target')}" != target:
            raise ValueError(
                f"{row['file_name']}: file name target {match.group('target')} "
                f"disagrees with class_id {target}"
            )
        if match.group("segment") != str(row["archive_s_code"]):
            raise ValueError(
                f"{row['file_name']}: file name segment {match.group('segment')} "
                f"disagrees with archive_s_code {row['archive_s_code']}"
            )
        sources.append(
            FrozenSource(
                filename=str(row["file_name"]),
                file_path=Path(str(row["g_path"])),
                target_code=target,
                role=role,
                size_bytes=int(row["size_bytes"]),
                class_id=target,
                class_name=str(row["class_name"]),
                segment_code=str(row["archive_s_code"]),
                distance_code=match.group("distance"),
                metrics_scope=str(row["metrics_scope"]),
                role_purpose=str(row["role_purpose"]),
                prior_use_role=str(row["prior_use_role"]),
                archive_segment_role=str(row["archive_segment_role"]),
                exposed_to_prior_selection=str(row["exposed_to_prior_selection"]),
                expected_sha256=str(row["mat_sha256"]).lower(),
                row=row,
            )
        )

    if len({source.filename for source in sources}) != len(sources):
        raise ValueError("duplicate source file names in the frozen selection")
    for target in wanted:
        for role in ROLE_ORDER:
            count = sum(
                1 for source in sources if source.target_code == target and source.role == role
            )
            expected = expected_roles[role]
            if count != expected:
                raise ValueError(
                    f"class {target} role {role}: expected {expected} files, got {count}"
                )
    if {source.target_code for source in sources} != set(wanted):
        raise ValueError("frozen selection does not cover every requested target code")
    return sources


def check_source_identity(
    source: FrozenSource,
    data_root: Path,
    sha256_mode: str,
    progress: bool = True,
) -> dict[str, Any]:
    """Existence, size, SHA256, HDF5 shape and dtype of one frozen source MAT."""
    path = source.file_path
    if not path.is_file():
        raise FileNotFoundError(path)
    if not is_within(path, data_root):
        raise ValueError(f"{path} is outside the data root {data_root}")

    size = path.stat().st_size
    if size != source.size_bytes:
        raise ValueError(
            f"{path.name}: size {size} does not match frozen size {source.size_bytes}"
        )

    if sha256_mode == "compute":
        started = time.time()
        actual_sha256 = sha256_file(path)
        if progress:
            print(
                f"  sha256 {path.name} {size} bytes -> {actual_sha256} "
                f"({time.time() - started:.1f}s)",
                flush=True,
            )
        recomputed = True
    elif sha256_mode == "frozen":
        actual_sha256 = source.expected_sha256
        recomputed = False
    else:
        raise ValueError(f"unsupported sha256 mode: {sha256_mode}")

    if actual_sha256 != source.expected_sha256:
        raise ValueError(
            f"{path.name}: sha256 mismatch, frozen {source.expected_sha256}, "
            f"observed {actual_sha256}"
        )

    with h5py.File(path, "r") as handle:
        keys = sorted(handle.keys())
        missing = [key for key in REQUIRED_DATASETS if key not in handle]
        if missing:
            raise KeyError(f"{path.name}: missing RF0 datasets {missing}")
        shapes = {key: list(handle[key].shape) for key in keys}
        dtypes = {key: str(handle[key].dtype) for key in keys}
        lengths = {key: dataset_length(handle[key]) for key in REQUIRED_DATASETS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{path.name}: RF0_I/RF0_Q lengths differ: {lengths}")
        source_samples = next(iter(lengths.values()))

    return {
        "filename": source.filename,
        "g_path": str(path),
        "size_bytes": size,
        "sha256": actual_sha256,
        "sha256_source": "recomputed" if recomputed else "frozen_csv",
        "frozen_sha256": source.expected_sha256,
        "source_samples": source_samples,
        "dropped_tail_samples": source_samples % WINDOW_SIZE,
        "complete_windows": window_count(source_samples),
        "hdf5_datasets": keys,
        "hdf5_shapes": shapes,
        "hdf5_dtypes": dtypes,
        "has_rf1": all(key in keys for key in OPTIONAL_DATASETS),
        "mtime_utc": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
    }


# ---------------------------------------------------------------------------
# window planning
# ---------------------------------------------------------------------------
def plan_file_windows(
    source: FrozenSource, source_samples: int, file_sha256: str | None = None
) -> list[dict[str, Any]]:
    """Every complete 1M window of one source file, in acquisition order."""
    class_index = CLASS_INDEX[source.target_code]
    stage1_label = 0 if class_index == 0 else 1
    stage2_eligible = 0 if class_index == 0 else 1
    stage2_label = "" if stage2_eligible == 0 else str(STAGE2_LABEL[source.target_code])
    sha = file_sha256 if file_sha256 is not None else source.expected_sha256

    rows: list[dict[str, Any]] = []
    for index in range(window_count(source_samples)):
        start = index * WINDOW_SIZE
        rows.append(
            {
                "file_path": str(source.file_path),
                "filename": source.filename,
                "target_code": source.target_code,
                "target_name": source.class_name,
                "distance_code": source.distance_code,
                "segment_code": source.segment_code,
                "source_role": source.role,
                "split": source.role,
                "metrics_scope": source.metrics_scope,
                "role_purpose": source.role_purpose,
                "prior_use_role": source.prior_use_role,
                "archive_segment_role": source.archive_segment_role,
                "exposed_to_prior_selection": source.exposed_to_prior_selection,
                "class_index": class_index,
                "class_name": source.class_name,
                "stage1_label": stage1_label,
                "stage2_eligible": stage2_eligible,
                "stage2_label": stage2_label,
                "label": stage1_label,
                "window_index": index,
                "window_in_block": index % (BLOCK_SIZE // WINDOW_SIZE),
                "block_index": start // BLOCK_SIZE,
                "start": start,
                "length": WINDOW_SIZE,
                "source_samples": source_samples,
                "file_sha256": sha,
                "window_rule": WINDOW_RULE,
            }
        )
    return rows


def stage2_view(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drone-only projection with stable per-split stage-2 positions.

    Stage 2 must never see a Background row, so the view is a pure filter of the
    three-class index: the row order of the index is preserved, file order first
    and window order second.
    """
    counters: dict[str, int] = {}
    view: list[dict[str, Any]] = []
    for row in rows:
        if int(row["stage2_eligible"]) != 1:
            continue
        role = str(row["source_role"])
        position_in_split = counters.get(role, 0)
        counters[role] = position_in_split + 1
        view.append(
            {
                "stage2_position": len(view),
                "stage2_position_in_split": position_in_split,
                "row_index": int(row["row_index"]),
                "file_path": row["file_path"],
                "filename": row["filename"],
                "target_code": row["target_code"],
                "target_name": row["target_name"],
                "source_role": role,
                "split": role,
                "metrics_scope": row["metrics_scope"],
                "class_index": int(row["class_index"]),
                "class_name": row["class_name"],
                "stage2_label": int(row["stage2_label"]),
                "window_index": int(row["window_index"]),
                "start": int(row["start"]),
                "length": int(row["length"]),
                "source_samples": int(row["source_samples"]),
                "file_sha256": row["file_sha256"],
            }
        )
    return view


def count_table(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Counts per role, per class, per role x class and per stage-1 label.

    The stage-2 view does not carry ``stage1_label`` (every row there is a
    drone), so the stage-1 split is derived from ``class_index`` when the column
    is absent.
    """
    rows = list(rows)
    by_role: dict[str, int] = {role: 0 for role in ROLE_ORDER}
    by_class: dict[str, int] = {code: 0 for code in TARGET_CODES}
    by_role_class: dict[str, dict[str, int]] = {
        role: {code: 0 for code in TARGET_CODES} for role in ROLE_ORDER
    }
    by_role_stage1 = {role: {"background": 0, "drone": 0} for role in ROLE_ORDER}
    files_by_role_class: dict[str, dict[str, set[str]]] = {
        role: {code: set() for code in TARGET_CODES} for role in ROLE_ORDER
    }
    for row in rows:
        role = str(row["source_role"])
        code = str(row["target_code"])
        by_role[role] = by_role.get(role, 0) + 1
        by_class[code] = by_class.get(code, 0) + 1
        by_role_class[role][code] += 1
        if "stage1_label" in row:
            stage1_label = int(row["stage1_label"])
        else:
            stage1_label = 0 if int(row["class_index"]) == 0 else 1
        key = "background" if stage1_label == 0 else "drone"
        by_role_stage1[role][key] += 1
        files_by_role_class[role][code].add(str(row["filename"]))
    return {
        "windows_by_role": by_role,
        "windows_by_class": by_class,
        "windows_by_role_class": by_role_class,
        "windows_by_role_stage1_label": by_role_stage1,
        "files_by_role_class": {
            role: {code: len(names) for code, names in classes.items()}
            for role, classes in files_by_role_class.items()
        },
        "files_by_role": {
            role: sum(len(names) for names in classes.values())
            for role, classes in files_by_role_class.items()
        },
        "total_windows": len(rows),
    }


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def render_scheme(
    freeze: dict[str, Any],
    frozen_csv_sha: str,
    manifest_sha: str,
    identities: Sequence[dict[str, Any]],
    counts: dict[str, Any],
    stage2_counts: dict[str, Any],
    index_sha: str,
    stage2_sha: str,
    generated_at: str,
    data_root: Path,
) -> str:
    legacy_train, legacy_val, legacy_test = 920, 450, 440
    lines: list[str] = [
        "# CASCADE3-NEWSPLIT-A 数据方案（三类新冻结划分窗口索引）",
        "",
        f"生成时间（UTC）：{generated_at}",
        "",
        "## 权威输入身份",
        "",
        f"- 冻结决定：`{freeze['frozen_split_csv']}`，SHA256 `{frozen_csv_sha}`"
        f"（由 `CODEX-FREEZE-20260926.json` 冻结）。",
        f"- G 盘正式清单：`{freeze['source_manifest']}`，SHA256 `{manifest_sha}`。",
        f"- 数据根：`{data_root}`；24 个 MAT 的名称、大小与 SHA256 均与冻结表逐项一致。",
        "- 角色只取自冻结 CSV；`work/meta_split1/source_table_104.csv` 的同名列在本流程中未被读取。",
        "",
        "## 窗口规则",
        "",
        "- 通道：RF0，I/Q 各一列实数，100 MS/s 复数基带（与既有 reader 一致）。",
        f"- 窗口：{WINDOW_SIZE:,} 采样（10.0 ms），非重叠，步长 = 窗口长度，1M 对齐。",
        f"- 每个源文件取全部完整 1M 窗：`windows = floor(source_samples / {WINDOW_SIZE:,})`，"
        "尾段不足 1M 丢弃（本批 24 个文件尾段均为 0）。",
        "- 10M 采集块是 1M 的整数倍，因此窗口不会跨越 10M 块边界。",
        f"- `window_index` 是文件内 0 起的全局窗序号；`block_index`/`window_in_block` 是同一窗口在 "
        f"{BLOCK_SIZE:,} 采样块内的等价定位。",
        "",
        "## 标签空间",
        "",
        "- `class_index`（最终三类）：0 = Background（T0000），1 = DJI Phantom 4 Pro（T0010），"
        "2 = DJI Air 2S（T0101）。",
        "- `stage1_label`（阶段 1 二元）：0 = Background，1 = Drone（与旧索引 `label` 语义一致，"
        "`label` 列等于 `stage1_label`）。",
        "- `stage2_label`（阶段 2）：仅两个无人机类有值，0 = Phantom 4 Pro，1 = Air 2S；"
        "Background 行 `stage2_eligible=0` 且 `stage2_label` 为空。",
        "",
        "## 规模（窗口数 / 源文件数）",
        "",
        "| 角色 | 文件 | 窗口 | Background | Phantom 4 Pro | Air 2S | 阶段1 背景 | 阶段1 无人机 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for role in ROLE_ORDER:
        rc = counts["windows_by_role_class"][role]
        s1 = counts["windows_by_role_stage1_label"][role]
        lines.append(
            f"| {role} | {counts['files_by_role'][role]} | {counts['windows_by_role'][role]} | "
            f"{rc['T0000']} | {rc['T0010']} | {rc['T0101']} | {s1['background']} | {s1['drone']} |"
        )
    lines += [
        f"| 合计 | 24 | {counts['total_windows']} | {counts['windows_by_class']['T0000']} | "
        f"{counts['windows_by_class']['T0010']} | {counts['windows_by_class']['T0101']} | "
        f"{sum(counts['windows_by_role_stage1_label'][r]['background'] for r in ROLE_ORDER)} | "
        f"{sum(counts['windows_by_role_stage1_label'][r]['drone'] for r in ROLE_ORDER)} |",
        "",
        f"阶段 2 视图（仅无人机类）：训练 {stage2_counts['windows_by_role']['train']}，"
        f"验证 {stage2_counts['windows_by_role']['val']}，"
        f"测试 {stage2_counts['windows_by_role']['development_test_exposed']}，"
        f"合计 {stage2_counts['total_windows']} 窗。",
        "",
        "## 与旧索引的规模对比（仅供 Codex 决定是否抽样）",
        "",
        "| 项 | 旧 12 文件索引 | 本索引 | 比值 |",
        "| --- | --- | --- | --- |",
        f"| 训练窗 | {legacy_train} | {counts['windows_by_role']['train']} | "
        f"{counts['windows_by_role']['train'] / legacy_train:.2f}x |",
        f"| 验证窗 | {legacy_val} | {counts['windows_by_role']['val']} | "
        f"{counts['windows_by_role']['val'] / legacy_val:.2f}x |",
        f"| 测试窗 | {legacy_test} | {counts['windows_by_role']['development_test_exposed']} | "
        f"{counts['windows_by_role']['development_test_exposed'] / legacy_test:.2f}x |",
        "",
        "本流程没有抽样、没有改窗口规则；是否抽样或缩短训练由 Codex 决定。",
        "",
        "## 使用约束",
        "",
        "- 一个源 MAT 只出现在一个角色中；窗口继承源文件角色，未来任何派生划分都必须继续继承。",
        "- `development_test_exposed` 的 6 个 MAT 此前已用于旧实验（`exposed_to_prior_selection=True`），"
        "只能作已暴露开发对照，不得称为全新 final test 或跨会话泛化。",
        "- 测试文件不得用于训练、选 epoch、选阈值或选超参数；训练视图只用 `source_role == \"train\"`。",
        "- 本流程未训练模型、未改动 G 盘数据、未改动任何历史实验产物。",
        "",
        "## 产物",
        "",
        f"- `window_index_newsplit3.csv`（SHA256 `{index_sha}`）：全部 "
        f"{counts['total_windows']} 窗。",
        f"- `stage2_view_newsplit3.csv`（SHA256 `{stage2_sha}`）：无人机窗投影，"
        "含稳定 `stage2_position`。",
        "- `newsplit3_dataset_summary.json`：机器可读计数与逐文件明细。",
        "- `newsplit3_source_identity.json`：24 个 MAT 的大小、SHA256、HDF5 形状。",
        "- `newsplit3_verification.json`：独立核验与反向对照结果（由 "
        "`verify_newsplit_index.py` 生成）。",
        "",
        "## 复现命令",
        "",
        "```",
        "python dronerfa_baseline/build_newsplit_index.py \\",
        "  --freeze-json work/meta_split2/CODEX-FREEZE-20260926.json \\",
        r"  --data-root 'G:\DroneRFa数据集\original'",
        "python dronerfa_baseline/verify_newsplit_index.py --sha256-mode recompute",
        "python -m unittest dronerfa_baseline.test_newsplit_index -v",
        "```",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the CASCADE3-NEWSPLIT-A window index from the Codex-frozen "
            "three-class development split."
        )
    )
    parser.add_argument("--freeze-json", type=Path, default=DEFAULT_FREEZE_JSON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-custom-output-dir",
        action="store_true",
        help="allow an output directory that is not the frozen CASCADE3-NEWSPLIT-A path",
    )
    parser.add_argument(
        "--sha256-mode",
        choices=("compute", "frozen"),
        default="compute",
        help="compute re-reads every MAT; frozen trusts the frozen CSV hash value",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()

    output_dir: Path = args.output_dir.resolve()
    if (
        output_dir.name != DEFAULT_OUTPUT_NAME
        and not args.allow_custom_output_dir
    ):
        raise SystemExit(
            "refusing to write outside the frozen output directory "
            f"{DEFAULT_OUTPUT_DIR} (pass --allow-custom-output-dir to override): {output_dir}"
        )
    data_root: Path = args.data_root.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"data root not found: {data_root}")

    freeze_path = args.freeze_json if args.freeze_json.is_absolute() else (
        PROJECT_ROOT / args.freeze_json
    ).resolve()
    freeze, csv_path, manifest_sha = load_freeze(freeze_path)
    frozen_csv_sha = str(freeze["frozen_split_csv_sha256"])
    print(f"freeze decision : {freeze_path}")
    print(f"frozen split CSV: {csv_path} sha256={frozen_csv_sha}")
    print(f"G manifest      : {freeze['source_manifest']} sha256={manifest_sha}")

    sources = load_frozen_sources(csv_path, TARGET_CODES)
    print(
        f"selected sources: {len(sources)} files, targets {TARGET_CODES}, "
        f"roles {[f'{r}={ROLE_FILES_PER_CLASS[r]}/class' for r in ROLE_ORDER]}"
    )

    print("verifying source identity ...", flush=True)
    identities: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for source in sorted(sources, key=lambda item: (item.target_code, item.filename)):
        identity = check_source_identity(source, data_root, args.sha256_mode)
        identities.append(identity)
        rows.extend(
            plan_file_windows(
                source, int(identity["source_samples"]), str(identity["sha256"])
            )
        )

    for position, row in enumerate(rows):
        row["row_index"] = position

    counts = count_table(rows)
    view = stage2_view(rows)
    stage2_counts = count_table(view)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "window_index_newsplit3.csv"
    stage2_path = output_dir / "stage2_view_newsplit3.csv"
    summary_path = output_dir / "newsplit3_dataset_summary.json"
    identity_path = output_dir / "newsplit3_source_identity.json"
    scheme_path = output_dir / "newsplit3_scheme.md"
    existing = [
        path
        for path in (index_path, stage2_path, summary_path, identity_path, scheme_path)
        if path.exists()
    ]
    if existing and not args.force:
        raise SystemExit(
            "refusing to overwrite existing outputs (pass --force): "
            + ", ".join(path.name for path in existing)
        )

    write_csv(index_path, CSV_FIELDS, rows)
    write_csv(stage2_path, STAGE2_FIELDS, view)
    index_sha = sha256_file(index_path)
    stage2_sha = sha256_file(stage2_path)
    generated_at = utc_now()

    per_file = []
    for identity in identities:
        source = next(item for item in sources if item.filename == identity["filename"])
        per_file.append(
            {
                "filename": source.filename,
                "target_code": source.target_code,
                "target_name": source.class_name,
                "source_role": source.role,
                "segment_code": source.segment_code,
                "metrics_scope": source.metrics_scope,
                "prior_use_role": source.prior_use_role,
                "archive_segment_role": source.archive_segment_role,
                "exposed_to_prior_selection": source.exposed_to_prior_selection,
                "g_path": identity["g_path"],
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
                "sha256_source": identity["sha256_source"],
                "source_samples": identity["source_samples"],
                "dropped_tail_samples": identity["dropped_tail_samples"],
                "windows": identity["complete_windows"],
            }
        )

    identity_payload = {
        "schema": "dronerfa-cascade3-newsplit3-source-identity/v1",
        "task": "CASCADE3-NEWSPLIT-A",
        "generated_at_utc": generated_at,
        "freeze_json": str(freeze_path),
        "frozen_split_csv": str(csv_path),
        "frozen_split_csv_sha256": frozen_csv_sha,
        "source_manifest": str(freeze["source_manifest"]),
        "source_manifest_sha256": manifest_sha,
        "data_root": str(data_root),
        "target_codes": list(TARGET_CODES),
        "sha256_mode": args.sha256_mode,
        "files": identities,
    }
    write_json(identity_path, identity_payload)

    summary = {
        "schema": "dronerfa-cascade3-newsplit3-dataset-summary/v1",
        "task": "CASCADE3-NEWSPLIT-A",
        "generated_at_utc": generated_at,
        "authority": {
            "freeze_json": str(freeze_path),
            "freeze_decision": freeze.get("decision"),
            "frozen_split_csv": str(csv_path),
            "frozen_split_csv_sha256": frozen_csv_sha,
            "source_manifest": str(freeze["source_manifest"]),
            "source_manifest_sha256": manifest_sha,
            "canonical_prior_use_role_source": "frozen split CSV (prior_use_role column)",
        },
        "data_root": str(data_root),
        "window_rule": WINDOW_RULE,
        "window_size_samples": WINDOW_SIZE,
        "window_duration_ms_at_100MSps": WINDOW_SIZE / (SAMPLE_RATE_HZ / 1000.0),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "acquisition_block_size_samples": BLOCK_SIZE,
        "channel": "RF0",
        "target_codes": list(TARGET_CODES),
        "class_index": CLASS_INDEX,
        "class_name": CLASS_NAME,
        "stage1_label_space": {"0": "Background", "1": "Drone"},
        "stage2_label_space": {"0": "DJI Phantom 4 Pro", "1": "DJI Air 2S"},
        "roles": list(ROLE_ORDER),
        "role_files_per_class": ROLE_FILES_PER_CLASS,
        "source_file_count": len(sources),
        "counts": counts,
        "stage2_view": {
            "file": stage2_path.name,
            "sha256": stage2_sha,
            "counts": stage2_counts,
            "rule": "pure filter of the three-class index on stage2_eligible == 1",
        },
        "index": {
            "file": index_path.name,
            "sha256": index_sha,
            "rows": len(rows),
            "fields": list(CSV_FIELDS),
        },
        "files": per_file,
        "leakage_rule": (
            "one source MAT -> exactly one role; windows inherit the role of their "
            "source MAT; test files must not be used for fitting, epoch selection, "
            "threshold selection or hyper-parameter selection"
        ),
        "claim_limits": [
            "the 6 development_test_exposed MATs were already used by earlier "
            "experiments and are an exposed development control, not a fresh final test",
            "the three classes are one file-time cluster each; this split is not "
            "cross-session, cross-receiver or open-set evidence",
            "no model was trained by this step",
        ],
        "legacy_scale_reference": {
            "index": "outputs/window_index.csv",
            "sha256": "4e77c7f186c1aae8e04f2e90f2769c0ed399cc94f40b64f99237b65c057b85ac",
            "train_windows": 920,
            "val_windows": 450,
            "test_windows": 440,
            "note": "12-file legacy index kept untouched; used only as a scale reference",
        },
        "sampling": {
            "applied": False,
            "note": (
                "every complete 1M window is indexed; no subsampling or rule change "
                "was applied, the scale decision belongs to Codex"
            ),
        },
    }
    write_json(summary_path, summary)

    scheme_path.write_text(
        render_scheme(
            freeze=freeze,
            frozen_csv_sha=frozen_csv_sha,
            manifest_sha=manifest_sha,
            identities=identities,
            counts=counts,
            stage2_counts=stage2_counts,
            index_sha=index_sha,
            stage2_sha=stage2_sha,
            generated_at=generated_at,
            data_root=data_root,
        ),
        encoding="utf-8",
    )

    print(f"index           : {index_path} ({len(rows)} rows) sha256={index_sha}")
    print(f"stage-2 view    : {stage2_path} ({len(view)} rows) sha256={stage2_sha}")
    print(f"summary         : {summary_path}")
    print(f"identity        : {identity_path}")
    print(f"scheme          : {scheme_path}")
    print(f"windows by role : {counts['windows_by_role']}")
    print(f"windows by class: {counts['windows_by_class']}")
    print(f"stage-2 by role : {stage2_counts['windows_by_role']}")
    print(f"total           : {len(rows)} windows in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
