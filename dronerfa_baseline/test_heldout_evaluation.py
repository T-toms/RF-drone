"""Necessary rejection and aggregation test cases for the frozen held-out evaluation.

Runs without touching the 12 GB of source data:

  * index hash rejection              (synthetic index + temporary config)
  * preserved-input hash rejection     (small synthetic file)
  * checkpoint sha256 rejection        (temporary config copy)
  * checkpoint epoch rejection         (temporary config copy)
  * checkpoint training-index rejection(temporary config copy)
  * selection.json evidence rejection  (temporary selection.json copy)
  * file-equal vs pooled aggregation on unequal-length files, with values that
    were computed by hand before the run
  * threshold rule consistency

Every rejection case asserts the exact error text so a silent pass is impossible.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from evaluate_heldout_models import (  # noqa: E402
    file_equal_metrics,
    pooled_metrics,
    verify_checkpoints,
    verify_frozen_inputs,
)

CONFIG_PATH = PROJECT_DIR.parent / "work" / "heldout-eval-config-20260918.json"
# The harness sandbox denies writes to the system temp directory, so synthetic
# fixtures live in a scratch directory inside the workspace.
SCRATCH_ROOT = PROJECT_DIR / "outputs" / "heldout_evaluation_20260918" / "test_scratch"


@contextmanager
def scratch(name: str):
    path = SCRATCH_ROOT / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
EXPECTED_ERRORS: dict[str, str] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name: str, condition: bool, detail: str) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}: {detail}", flush=True)
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def minimal_config(tmp: Path, **overrides) -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    index_path = tmp / "index.csv"
    index_path.write_text("file_path,filename,label\nfake,F.mat,0\n", encoding="utf-8")
    data_dir = tmp / "data"
    data_dir.mkdir(exist_ok=True)
    config.update(
        {
            "index": str(index_path),
            "index_sha256": sha256_file(index_path),
            "data_root": str(data_dir),
            "file_sha256": {},
            "preserved_inputs": str(tmp / "preserved.json"),
        }
    )
    config.update(overrides)
    return config


def write_preserved(tmp: Path, files: list[dict]) -> Path:
    path = tmp / "preserved.json"
    path.write_text(json.dumps({"files": files}), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# case 1: index hash rejection
# ---------------------------------------------------------------------------
def case_index_hash_rejection() -> None:
    with scratch("case1_index_hash") as tmp:
        config = minimal_config(tmp)
        write_preserved(tmp, [])
        errors: list[str] = []
        verify_frozen_inputs(config, errors)
        check("index hash accepted when correct", errors == [], f"errors={errors}")

        config["index_sha256"] = "0" * 64
        errors = []
        verify_frozen_inputs(config, errors)
        check(
            "index hash rejection",
            len(errors) == 1 and errors[0].startswith("index SHA256 mismatch"),
            f"errors={errors}",
        )


# ---------------------------------------------------------------------------
# case 2: preserved-input rejection
# ---------------------------------------------------------------------------
def case_preserved_input_rejection() -> None:
    with scratch("case2_preserved_input") as tmp:
        small = tmp / "small_code.py"
        small.write_text("print('ok')\n", encoding="utf-8")
        good = {
            "path": str(small),
            "bytes": small.stat().st_size,
            "sha256": sha256_file(small),
        }
        config = minimal_config(tmp)
        write_preserved(tmp, [good])
        errors: list[str] = []
        verify_frozen_inputs(config, errors)
        check("preserved input accepted when unchanged", errors == [], f"errors={errors}")

        small.write_text("print('tampered')\n", encoding="utf-8")
        errors = []
        verify_frozen_inputs(config, errors)
        check(
            "preserved input rejection",
            len(errors) == 1 and "preserved input changed" in errors[0],
            f"errors={errors}",
        )


# ---------------------------------------------------------------------------
# case 3/4/5/6: checkpoint verification (temporary config copies only)
# ---------------------------------------------------------------------------
def case_checkpoint_rejections() -> None:
    with scratch("case3_checkpoint") as tmp:
        config = minimal_config(tmp)
        write_preserved(tmp, [])

        # The genuine training index, taken from the frozen preserved-inputs record.
        preserved = json.loads(
            (PROJECT_DIR.parent / "work" / "codex-preserved-inputs-20260917.json").read_text(
                encoding="utf-8"
            )
        )
        training_entry = next(
            entry
            for entry in preserved["files"]
            if Path(entry["path"]) == PROJECT_DIR / "outputs" / "window_index.csv"
        )
        training_index_sha256 = training_entry["sha256"]
        errors: list[str] = []
        records = verify_checkpoints(config, training_index_sha256, errors)
        check(
            "six real checkpoints accepted",
            errors == [] and len(records) == 6,
            f"errors={errors}",
        )

        # 3: wrong checkpoint sha256.
        config_bad_hash = json.loads(json.dumps(config))
        config_bad_hash["models"][0]["sha256"] = "f" * 64
        errors = []
        verify_checkpoints(config_bad_hash, training_index_sha256, errors)
        check(
            "checkpoint sha256 rejection",
            any("checkpoint SHA256 mismatch" in item for item in errors),
            f"errors={errors}",
        )

        # 4: wrong declared epoch (checkpoint itself untouched, read only).
        config_bad_epoch = json.loads(json.dumps(config))
        config_bad_epoch["models"][0]["epoch"] = 99
        errors = []
        verify_checkpoints(config_bad_epoch, training_index_sha256, errors)
        check(
            "checkpoint epoch rejection",
            any("epoch mismatch" in item or "does not select epoch" in item for item in errors),
            f"errors={errors}",
        )

        # 5: checkpoint trained against a different index.
        errors = []
        verify_checkpoints(config, "1" * 64, errors)
        check(
            "training index rejection",
            len(errors) == 6 and all("different index" in item for item in errors),
            f"{len(errors)} errors",
        )

        # 6: selection.json evidence rejection, on a temporary copy of the arm dir.
        arm_src = Path(config["models"][0]["checkpoint"]).parent
        arm_dir = tmp / "arm_copy"
        shutil.copytree(arm_src, arm_dir, ignore=shutil.ignore_patterns("epoch_*.pt"))
        selection_path = arm_dir / "selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection["multi_noise"] = int(config["models"][0]["epoch"]) + 1
        selection_path.write_text(json.dumps(selection), encoding="utf-8")
        for entry in preserved["files"]:
            if Path(entry["path"]).parent == arm_src:
                break
        else:
            raise AssertionError("real checkpoint is not a preserved input")
        check(
            "temporary arm copy exists for selection test",
            (arm_dir / "selection.json").is_file()
            and (arm_dir / "best_model.pt").is_file(),
            str(arm_dir),
        )
        config_bad_selection = json.loads(json.dumps(config))
        config_bad_selection["models"][0]["checkpoint"] = str(arm_dir / "best_model.pt")
        errors = []
        verify_checkpoints(config_bad_selection, training_index_sha256, errors)
        check(
            "selection.json evidence rejection",
            len(errors) == 1 and "does not select epoch" in errors[0],
            f"errors={errors}",
        )


# ---------------------------------------------------------------------------
# case 7: unequal-length files, file-equal versus pooled
# ---------------------------------------------------------------------------
def synthetic_records() -> tuple[list[dict], dict[str, list[str]]]:
    """Hand-computed fixture: 1000 windows spread over twelve files."""
    plan = {
        "T0000_D00_S0100.mat": ("0000", 140, 0.20),
        "T0000_D00_S0101.mat": ("0000", 140, 0.10),
        "T0000_D00_S0110.mat": ("0000", 140, 0.25),
        "T0000_D00_S0111.mat": ("0000", 140, 0.05),
        "T0010_D00_S0100.mat": ("0010", 160, 0.75),
        "T0010_D00_S0101.mat": ("0010", 150, 0.90),
        "T0010_D00_S0110.mat": ("0010", 150, 0.80),
        "T0010_D00_S0111.mat": ("0010", 150, 0.70),
        "T0101_D00_S0100.mat": ("0101", 160, 0.85),
        "T0101_D00_S0101.mat": ("0101", 180, 0.60),
        "T0101_D00_S0110.mat": ("0101", 160, 0.95),
        "T0101_D00_S0111.mat": ("0101", 150, 0.80),
    }
    records: list[dict] = []
    grouped: dict[str, list[str]] = {}
    for name, (code, count, rate) in plan.items():
        grouped.setdefault(code, []).append(name)
        hits = round(rate * count)
        label = 0 if code == "0000" else 1
        for position in range(count):
            prediction = 1 if position < hits else 0
            records.append(
                {
                    "filename": name,
                    "target_code": code,
                    "label": label,
                    "prediction": prediction,
                    "heldout_index": len(records),
                }
            )
    return records, grouped


def case_unequal_length_aggregation() -> None:
    records, grouped = synthetic_records()
    lengths = {name: sum(1 for r in records if r["filename"] == name) for name in grouped["0000"]}
    all_lengths = {r["filename"]: 0 for r in records}
    for row in records:
        all_lengths[row["filename"]] += 1
    check(
        "fixture has unequal file lengths",
        len(set(all_lengths.values())) > 1,
        f"{len(records)} windows over {len(all_lengths)} files, "
        f"lengths={sorted(set(all_lengths.values()))}",
    )
    check(
        "fixture keeps the four-per-category structure",
        all(len(names) == 4 for names in grouped.values()),
        f"{ {key: len(value) for key, value in grouped.items()} } (background example {lengths})",
    )
    primary = file_equal_metrics(records, grouped)
    secondary = pooled_metrics(records)

    check(
        "background file-equal FPR",
        abs(primary["background_file_equal_fpr"] - 0.15) < 1e-12,
        f"{primary['background_file_equal_fpr']:.6f}",
    )
    check(
        "phantom file-equal recall",
        abs(primary["phantom_file_equal_recall"] - 0.7875) < 1e-12,
        f"{primary['phantom_file_equal_recall']:.6f}",
    )
    check(
        "air2s file-equal recall",
        abs(primary["air2s_file_equal_recall"] - 0.8) < 1e-12,
        f"{primary['air2s_file_equal_recall']:.6f}",
    )
    check(
        "file-equal balanced accuracy",
        abs(primary["file_equal_balanced_accuracy"] - 0.821875) < 1e-12,
        f"{primary['file_equal_balanced_accuracy']:.6f}",
    )
    check(
        "pooled window recall",
        secondary["tp"] == 996 and secondary["fn"] == 264,
        f"tp={secondary['tp']} fn={secondary['fn']}",
    )
    check(
        "pooled window FPR",
        secondary["fp"] == 84 and secondary["tn"] == 476,
        f"fp={secondary['fp']} tn={secondary['tn']}",
    )
    check(
        "pooled balanced accuracy",
        abs(secondary["pooled_balanced_accuracy"] - 0.5 * (996 / 1260 + 476 / 560))
        < 1e-12,
        f"{secondary['pooled_balanced_accuracy']:.12f}",
    )
    difference = abs(
        primary["file_equal_balanced_accuracy"] - secondary["pooled_balanced_accuracy"]
    )
    check(
        "file-equal and pooled are distinct on unequal files",
        difference > 1e-9,
        f"file_equal={primary['file_equal_balanced_accuracy']:.12f} "
        f"pooled={secondary['pooled_balanced_accuracy']:.12f} difference={difference:.12f}",
    )

    # A deliberately skewed fixture: one tiny file that detects everything and one
    # large file that detects nothing separate the two aggregations strongly.
    skewed_records: list[dict] = []
    skewed_grouped = {"0000": [], "0010": [], "0101": []}
    plan = {
        "0000": [(4, 0.0), (100, 0.0), (100, 0.0), (100, 0.0)],
        "0010": [(4, 1.0), (100, 0.0), (100, 0.0), (100, 0.0)],
        "0101": [(4, 1.0), (100, 0.0), (100, 0.0), (100, 0.0)],
    }
    for code, entries in plan.items():
        for position, (count, rate) in enumerate(entries):
            name = f"T{code}_D00_S01{position:02d}.mat"
            skewed_grouped[code].append(name)
            hits = round(rate * count)
            for window in range(count):
                skewed_records.append(
                    {
                        "filename": name,
                        "target_code": code,
                        "label": 0 if code == "0000" else 1,
                        "prediction": 1 if window < hits else 0,
                        "heldout_index": len(skewed_records),
                    }
                )
    skewed_primary = file_equal_metrics(skewed_records, skewed_grouped)
    skewed_secondary = pooled_metrics(skewed_records)
    check(
        "skewed fixture file-equal FPR",
        abs(skewed_primary["background_file_equal_fpr"] - 0.0) < 1e-12,
        f"{skewed_primary['background_file_equal_fpr']:.6f}",
    )
    check(
        "skewed fixture file-equal drone recall",
        abs(skewed_primary["drone_file_equal_recall"] - 0.25) < 1e-12,
        f"{skewed_primary['drone_file_equal_recall']:.6f}",
    )
    check(
        "skewed fixture pooled recall",
        skewed_secondary["tp"] == 8
        and skewed_secondary["fn"] == 600
        and abs(skewed_secondary["pooled_recall"] - 8 / 608) < 1e-12,
        f"tp={skewed_secondary['tp']} fn={skewed_secondary['fn']} "
        f"recall={skewed_secondary['pooled_recall']:.6f}",
    )
    skew_difference = abs(
        skewed_primary["file_equal_balanced_accuracy"]
        - skewed_secondary["pooled_balanced_accuracy"]
    )
    check(
        "skewed fixture separates file-equal from pooled",
        skew_difference > 0.11,
        f"file_equal={skewed_primary['file_equal_balanced_accuracy']:.6f} "
        f"pooled={skewed_secondary['pooled_balanced_accuracy']:.6f} "
        f"difference={skew_difference:.6f}",
    )

    # Aggregation must refuse an incomplete category instead of silently averaging.
    if True:
        broken = {key: list(value) for key, value in grouped.items()}
        broken["0101"] = broken["0101"][:3]
        try:
            file_equal_metrics(records, broken)
        except ValueError as error:
            check("incomplete category rejected", "expected four source files" in str(error), str(error))
        else:
            raise AssertionError("incomplete category was not rejected")


# ---------------------------------------------------------------------------
# case 8: threshold rule
# ---------------------------------------------------------------------------
def case_threshold_rule() -> None:
    import numpy as np

    margins = np.asarray([-1.0e-7, 0.0, 1.0e-7, -3.5, 2.5], dtype=np.float32)
    predictions = (margins > 0.0).astype(np.int64)
    check(
        "threshold rule margin>0",
        predictions.tolist() == [0, 0, 1, 0, 1],
        f"{predictions.tolist()}",
    )


def main() -> int:
    case_index_hash_rejection()
    case_preserved_input_rejection()
    case_checkpoint_rejections()
    case_unequal_length_aggregation()
    case_threshold_rule()
    print("ALL TEST CASES PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
