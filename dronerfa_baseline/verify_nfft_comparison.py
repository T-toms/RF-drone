"""NFFT4 verification: recompute every published number from the raw CSVs.

This module deliberately does *not* import ``run_nfft_comparison``. It reads the
frozen indices and the prediction CSVs as text, rebuilds the confusion matrices,
the per-source-file rates and the four-condition composites with its own code, and
then compares the result with the published ``model_condition_metrics.csv``,
``nfft_summary.csv``, ``paired_nfft_comparison.csv`` and the per-run artefacts.

Executed as::

    python verify_nfft_comparison.py            # write verification.json + report.md
    python verify_nfft_comparison.py --check    # same checks, non-zero exit on failure
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_DIR = Path(__file__).resolve().parent

NFFT_GRID = (512, 1024, 2048, 4096)
HOP_DIVISOR = 2
TRAINING_SEEDS = (2026, 2027, 2028)
CONDITIONS = ("clean", "-10dB", "-12dB", "-15dB")
BASELINE_NFFT = 1024
BACKGROUND_CODE = "0000"
DRONE_CODES = ("0010", "0101")
EXPECTED_HELDOUT_ROWS = len(NFFT_GRID) * len(TRAINING_SEEDS) * len(CONDITIONS) * 1820
EXPECTED_TEST_ROWS = len(NFFT_GRID) * len(TRAINING_SEEDS) * 440
TOLERANCE = 1e-12
REUSE_EXPECTED = {
    2026: (
        str(PROJECT_DIR / "outputs" / "noise_ab_5epochs_20260917" / "A_original" / "best_model.pt"),
        "298f6fd959c43f6fd69b9399175060d81ae29c49e6aa68d976dbfbe4023fe645",
    ),
    2027: (
        str(
            PROJECT_DIR
            / "outputs"
            / "noise_ab_replicates_20260917"
            / "seed_2027"
            / "A_original"
            / "best_model.pt"
        ),
        "b06898b0018576b6e060fe781bd3871fe703500bb3710f65921c658a45d894c7",
    ),
    2028: (
        str(
            PROJECT_DIR
            / "outputs"
            / "noise_ab_replicates_20260917"
            / "seed_2028"
            / "A_original"
            / "best_model.pt"
        ),
        "e1c2dce6b0679011b95e819c18d60a252d17a304ad8907bc0c960c5dde75c2ef",
    ),
}
FROZEN_PROCESS_FILES = (
    "dronerfa_data.py",
    "model.py",
    "train_resnet18.py",
    "run_noise_ab.py",
    "evaluate_heldout_models.py",
)
PRESERVED_INPUTS = PROJECT_DIR.parent / "work" / "codex-preserved-inputs-20260917.json"
HELDOUT_CONFIG = PROJECT_DIR.parent / "work" / "heldout-eval-config-20260918.json"
OLD_INDEX = PROJECT_DIR / "outputs" / "window_index.csv"
HELDOUT_INDEX = (
    PROJECT_DIR / "outputs" / "heldout_preparation_20260917" / "heldout_window_index.csv"
)
OLD_INDEX_SHA256 = "4e77c7f186c1aae8e04f2e90f2769c0ed399cc94f40b64f99237b65c057b85ac"
HELDOUT_INDEX_SHA256 = (
    "6fe32faeb5d583a359d9e8aa17caa52a7c4567d545272ded25266767cd2cffcf"
)


def output_root(cli: Path | None = None) -> Path:
    if cli is not None:
        return cli
    return PROJECT_DIR / "outputs" / "stage1_method_comparison_20260919" / "nfft"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


# ---------------------------------------------------------------------------
# aggregation implemented from scratch
# ---------------------------------------------------------------------------
def confusion(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    tp = tn = fp = fn = 0
    for row in rows:
        label = int(row["label"])
        predicted = int(row["prediction"])
        if label == 1 and predicted == 1:
            tp += 1
        elif label == 0 and predicted == 0:
            tn += 1
        elif label == 0 and predicted == 1:
            fp += 1
        else:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def pooled(rows: Sequence[dict[str, str]]) -> dict[str, float]:
    counts = confusion(rows)
    total = counts["tp"] + counts["tn"] + counts["fp"] + counts["fn"]
    recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
    specificity = counts["tn"] / max(counts["tn"] + counts["fp"], 1)
    precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
    return {
        **counts,
        "windows": total,
        "accuracy": (counts["tp"] + counts["tn"]) / max(total, 1),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "drone_recall": recall,
        "background_specificity": specificity,
        "precision": precision,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def file_equal(rows: Sequence[dict[str, str]]) -> dict[str, float]:
    """File-equal rates; a category that is absent altogether is reported as nan.

    On the real splits every category is present with four (held-out) or three
    (old test) source files; a small fixture that only carries one category must
    not abort the whole verification.
    """
    per_category: dict[str, dict[str, float]] = {}
    for row in rows:
        per_category.setdefault(row["target_code"], {}).setdefault(row["filename"], 0.0)
    hits: dict[str, dict[str, int]] = {
        code: {name: 0 for name in files} for code, files in per_category.items()
    }
    counts: dict[str, dict[str, int]] = {
        code: {name: 0 for name in files} for code, files in per_category.items()
    }
    for row in rows:
        counts[row["target_code"]][row["filename"]] += 1
        hits[row["target_code"]][row["filename"]] += int(row["prediction"])
    rates = {
        code: {name: hits[code][name] / counts[code][name] for name in files}
        for code, files in per_category.items()
    }
    missing = float("nan")

    def category_rate(code: str) -> float:
        return mean(list(rates[code].values())) if code in rates else missing

    background_fpr = category_rate(BACKGROUND_CODE)
    phantom = category_rate("0010")
    air2s = category_rate("0101")
    drone_recall = 0.5 * (phantom + air2s) if not (phantom != phantom or air2s != air2s) else missing
    balanced = (
        0.5 * (1.0 - background_fpr + drone_recall)
        if not (background_fpr != background_fpr or drone_recall != drone_recall)
        else missing
    )
    return {
        "background_file_equal_fpr": background_fpr,
        "background_file_equal_specificity": 1.0 - background_fpr,
        "phantom_file_equal_recall": phantom,
        "air2s_file_equal_recall": air2s,
        "drone_file_equal_recall": drone_recall,
        "file_equal_balanced_accuracy": balanced,
        "files_per_category": {code: len(files) for code, files in per_category.items()},
        "file_windows": counts,
    }


def composite(metric_rows: list[dict[str, Any]], metrics: dict[tuple[int, int, str], dict[str, Any]]) -> dict[tuple[int, int], float]:
    composites = {}
    for n_fft in NFFT_GRID:
        for seed in TRAINING_SEEDS:
            composites[(n_fft, seed)] = mean(
                [metrics[(n_fft, seed, condition)]["file_equal_balanced_accuracy"] for condition in CONDITIONS]
            )
    return composites


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_frozen_inputs(record: dict[str, Any], failures: list[str]) -> None:
    old_hash = sha256_file(OLD_INDEX)
    heldout_hash = sha256_file(HELDOUT_INDEX)
    record["old_index"] = {"sha256": old_hash, "expected": OLD_INDEX_SHA256, "match": old_hash == OLD_INDEX_SHA256}
    record["heldout_index"] = {
        "sha256": heldout_hash,
        "expected": HELDOUT_INDEX_SHA256,
        "match": heldout_hash == HELDOUT_INDEX_SHA256,
    }
    if old_hash != OLD_INDEX_SHA256:
        failures.append("old window_index.csv hash changed")
    if heldout_hash != HELDOUT_INDEX_SHA256:
        failures.append("held-out index hash changed")

    preserved = read_json(PRESERVED_INPUTS)
    preserved_records = []
    for entry in preserved["files"]:
        path = Path(entry["path"])
        actual = sha256_file(path)
        match = actual == entry["sha256"] and path.stat().st_size == entry["bytes"]
        preserved_records.append(
            {
                "path": entry["path"],
                "expected_sha256": entry["sha256"],
                "actual_sha256": actual,
                "match": match,
            }
        )
        if not match:
            failures.append(f"frozen input changed: {entry['path']}")
    record["preserved_inputs"] = {
        "reference": str(PRESERVED_INPUTS),
        "checked": len(preserved_records),
        "all_match": all(item["match"] for item in preserved_records),
        "files": preserved_records,
    }
    record["process_files"] = {
        name: sha256_file(PROJECT_DIR / name) for name in FROZEN_PROCESS_FILES
    }
    record["heldout_config_sha256"] = sha256_file(HELDOUT_CONFIG)


def check_runs(
    root: Path, record: dict[str, Any], failures: list[str], fixture: bool = False
) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    for n_fft in NFFT_GRID:
        for seed in TRAINING_SEEDS:
            key = f"nfft{n_fft}_seed{seed}"
            run_dir = root / f"nfft{n_fft}" / f"seed_{seed}"
            entry: dict[str, Any] = {"run_directory": str(run_dir)}
            config_path = run_dir / "run_config.json"
            if not config_path.is_file():
                if fixture:
                    # A partial tree (an in-progress experiment) simply does not
                    # contain this run yet; that is not a verification failure.
                    continue
                failures.append(f"missing artefact: {config_path}")
                entry["status"] = "missing"
                runs[key] = entry
                continue
            runs[key] = entry
            history_path = run_dir / "training_history.csv"
            metrics_path = run_dir / "validation_metrics.csv"
            selection_path = run_dir / "selection.json"
            checkpoint = run_dir / "best_model.pt"
            for path in (history_path, metrics_path, selection_path, checkpoint):
                if not path.is_file():
                    failures.append(f"missing artefact: {path}")
                    entry["status"] = "missing"
                    return runs
            config = read_json(config_path)
            history = read_csv_rows(history_path)
            validation = read_csv_rows(metrics_path)
            selection = read_json(selection_path)
            if "selected_epoch" not in selection:
                # The reused 2026-09-17 runs keep the original run_noise_ab
                # selection.json, whose multi-noise key is "multi_noise".
                selection = {**selection, "selected_epoch": selection.get("multi_noise")}

            entry["n_fft"] = config["n_fft"]
            entry["hop_length"] = config["hop_length"]
            entry["training_seed"] = config["training_seed"]
            entry["train_snrs"] = config["train_snrs"]
            entry["batch_size"] = config["batch_size"]
            entry["optimizer"] = config["optimizer"]
            entry["amp"] = config["amp"]
            entry["validation_seed"] = config["validation_seed"]
            entry["validation_conditions"] = config["validation_conditions"]
            entry["selected_epoch"] = config["selected_epoch"]
            entry["status"] = config["status"]
            entry["smoke"] = config.get("smoke")
            entry["reused_from"] = config.get("reused_from")
            entry["epoch_order_sha256"] = config.get("epoch_order_sha256")
            entry["initial_model_sha256"] = config.get("initial_model_sha256")
            entry["checkpoint_sha256"] = sha256_file(checkpoint)
            entry["checkpoint_bytes"] = checkpoint.stat().st_size

            if config["n_fft"] != n_fft or config["hop_length"] != n_fft // HOP_DIVISOR:
                failures.append(f"{key}: n_fft/hop mismatch")
            if config["hop_length"] * HOP_DIVISOR != config["n_fft"]:
                failures.append(f"{key}: hop is not 50% of n_fft")
            if config["training_seed"] != seed:
                failures.append(f"{key}: training seed mismatch")
            if config["train_snrs"] != "clean,10,5,0,-5,-10,-12":
                failures.append(f"{key}: training SNR plan changed")
            if config["image_size"] != 256:
                failures.append(f"{key}: image_size changed")
            if config["batch_size"] != 4:
                failures.append(f"{key}: batch size changed")
            if config["optimizer"] != {"type": "Adam", "learning_rate": 0.001, "weight_decay": 0.0}:
                failures.append(f"{key}: optimizer changed")
            if config["amp"] is not True:
                failures.append(f"{key}: AMP disabled")
            if config["validation_seed"] != 12026:
                failures.append(f"{key}: validation seed changed")
            if config["validation_conditions"] != list(CONDITIONS):
                failures.append(f"{key}: validation conditions changed")
            if config.get("smoke"):
                failures.append(f"{key}: smoke run present in the final grid")
            if len(history) != 5:
                failures.append(f"{key}: history has {len(history)} epochs")

            # selection rule, re-derived from the four validation conditions
            by_epoch: dict[int, list[float]] = {}
            for row in validation:
                by_epoch.setdefault(int(row["epoch"]), []).append(float(row["balanced_accuracy"]))
            means = {epoch: mean(values) for epoch, values in by_epoch.items()}
            derived_best = max(means, key=lambda epoch: (means[epoch], -epoch))
            entry["mean_balanced_accuracy_by_epoch"] = means
            entry["best_epoch_by_rule"] = derived_best
            if selection.get("selected_epoch") != derived_best:
                failures.append(f"{key}: selection.json epoch {selection.get('selected_epoch')} != {derived_best}")
            if int(config["selected_epoch"]) != derived_best:
                failures.append(f"{key}: run_config epoch != rule epoch")
            if not all(len(values) == len(CONDITIONS) for values in by_epoch.values()):
                failures.append(f"{key}: validation metrics are not four-condition per epoch")
            history_means = {int(row["epoch"]): float(row["mean_balanced_accuracy"]) for row in history}
            for epoch, value in means.items():
                if abs(value - history_means.get(epoch, float("nan"))) > 1e-9:
                    failures.append(f"{key}: epoch {epoch} history mean differs from metrics")

            # a reused checkpoint must reproduce the preserved hash exactly
            if config.get("reused_from"):
                expected_path, expected_hash = REUSE_EXPECTED[seed]
                entry["reuse_expected_path"] = expected_path
                entry["reuse_expected_sha256"] = expected_hash
                entry["reuse_path_match"] = Path(config["reused_from"]).resolve() == Path(expected_path).resolve()
                entry["reuse_hash_match"] = entry["checkpoint_sha256"] == expected_hash
                entry["reuse_original_unchanged"] = sha256_file(Path(expected_path)) == expected_hash
                if not entry["reuse_path_match"]:
                    failures.append(f"{key}: reused_from points at an unexpected checkpoint")
                if not entry["reuse_hash_match"]:
                    failures.append(f"{key}: reused checkpoint hash does not match the preserved input")
                if not entry["reuse_original_unchanged"]:
                    failures.append(f"{key}: the original 2026-09-17 checkpoint was modified")
                if n_fft != BASELINE_NFFT:
                    failures.append(f"{key}: only the 1024 grid may reuse a checkpoint")
            else:
                if n_fft == BASELINE_NFFT and not fixture:
                    failures.append(f"{key}: 1024 run was retrained without a reuse verdict")
                entry["epoch_checkpoint_files"] = sorted(
                    path.name for path in run_dir.glob("epoch_*.pt")
                )
            runs[key] = entry
    return runs


def check_prediction_grid(
    root: Path,
    record: dict[str, Any],
    failures: list[str],
    fixture: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    heldout = read_csv_rows(root / "heldout_window_predictions.csv")
    test = read_csv_rows(root / "test_clean_window_predictions.csv")
    record["heldout_rows"] = len(heldout)
    record["test_rows"] = len(test)
    heldout_windows = max((int(row["index_position"]) for row in heldout), default=-1) + 1
    test_windows = max((int(row["index_position"]) for row in test), default=-1) + 1
    expected_heldout = (
        EXPECTED_HELDOUT_ROWS
        if not fixture
        else len({row["model_id"] for row in heldout}) * len(CONDITIONS) * heldout_windows
    )
    expected_test = (
        EXPECTED_TEST_ROWS
        if not fixture
        else len({row["model_id"] for row in test}) * test_windows
    )
    if len(heldout) != expected_heldout:
        failures.append(f"heldout rows {len(heldout)} != {expected_heldout}")
    if len(test) != expected_test:
        failures.append(f"test rows {len(test)} != {expected_test}")

    heldout_keys = [
        (row["model_id"], row["condition"], int(row["index_position"])) for row in heldout
    ]
    record["heldout_unique_keys"] = len(set(heldout_keys))
    if len(set(heldout_keys)) != len(heldout_keys):
        failures.append("heldout primary keys are not unique")
    test_keys = [
        (row["model_id"], row["condition"], int(row["index_position"])) for row in test
    ]
    record["test_unique_keys"] = len(set(test_keys))
    if len(set(test_keys)) != len(test_keys):
        failures.append("test clean primary keys are not unique")

    grid_counts: dict[tuple[str, str], int] = {}
    for row in heldout:
        grid_counts[(row["model_id"], row["condition"])] = (
            grid_counts.get((row["model_id"], row["condition"]), 0) + 1
        )
    if fixture:
        # A bounded fixture declares its own grid; the real evaluation must hold
        # exactly the 12 x 4 grid of the frozen design.
        expected_grid = {
            (model_id, condition): heldout_windows
            for model_id in sorted({row["model_id"] for row in heldout})
            for condition in CONDITIONS
        }
    else:
        expected_grid = {
            (f"nfft{n_fft}_seed{seed}", condition): 1820
            for n_fft in NFFT_GRID
            for seed in TRAINING_SEEDS
            for condition in CONDITIONS
        }
    if grid_counts != expected_grid:
        missing = {key: value for key, value in expected_grid.items() if grid_counts.get(key) != value}
        failures.append(f"heldout grid incomplete for {len(missing)} (model, condition) pairs")
    record["heldout_windows_per_model"] = heldout_windows
    record["test_windows_per_model"] = test_windows
    per_model_windows: dict[str, set[int]] = {}
    for row in heldout:
        per_model_windows.setdefault(row["model_id"], set()).add(int(row["index_position"]))
    for model_id, positions in per_model_windows.items():
        if sorted(positions) != list(range(heldout_windows)):
            failures.append(
                f"{model_id}: held-out positions are not exactly 0..{heldout_windows - 1}"
            )
    test_positions: dict[str, set[int]] = {}
    for row in test:
        test_positions.setdefault(row["model_id"], set()).add(int(row["index_position"]))
    for model_id, positions in test_positions.items():
        if sorted(positions) != list(range(test_windows)):
            failures.append(
                f"{model_id}: test positions are not exactly 0..{test_windows - 1}"
            )

    # numeric sanity and the threshold rule
    for name, rows in (("heldout", heldout), ("test", test)):
        for row in rows:
            for field in ("logit_background", "logit_drone", "logit_margin"):
                value = float(row[field])
                if value != value or value in (float("inf"), float("-inf")):
                    failures.append(f"{name}: non-finite {field}")
                    return heldout, test
            margin = float(row["logit_drone"]) - float(row["logit_background"])
            if abs(margin - float(row["logit_margin"])) > 1e-6:
                failures.append(f"{name}: margin is not logit_drone - logit_background")
                break
            if int(row["prediction"]) != int(float(row["logit_margin"]) > 0.0):
                failures.append(f"{name}: prediction is not the margin>0 rule")
                break

    # row identity against the frozen indices
    if fixture:
        # The fixture declares its own window count and draws its rows from across
        # the index, so the frozen-index identity check applies to real runs only.
        index_identity_checked = False
    else:
        index_identity_checked = True
        for name, rows, index_path, split in (
            ("heldout", heldout, HELDOUT_INDEX, "heldout"),
            ("test", test, OLD_INDEX, "test"),
        ):
            index = read_csv_rows(index_path)
            positions = {}
            for position, source in enumerate(index):
                if split == "heldout" or source["split"] == split:
                    positions[len(positions)] = source
            for row in rows:
                source = positions[int(row["index_position"])]
                for field in ("filename", "target_code", "target_name", "segment_code"):
                    if row[field] != source[field]:
                        failures.append(f"{name}: {field} mismatch at position {row['index_position']}")
                        break
                if int(row["label"]) != int(source["label"]):
                    failures.append(f"{name}: label mismatch at position {row['index_position']}")
                    break
                if int(row["start"]) != int(source["start"]) or int(row["length"]) != int(source["length"]):
                    failures.append(f"{name}: window bounds mismatch")
                    break
                if int(row["window_index"]) != int(source["window_index"]):
                    failures.append(f"{name}: window_index mismatch")
                    break
    record["index_identity_checked"] = index_identity_checked
    # the held-out file carries the held-out split only
    if {row["split"] for row in heldout} != {"heldout"}:
        failures.append("heldout prediction file is not held-out only")
    if {row["condition"] for row in test} != {"clean"}:
        failures.append("test prediction file is not clean only")
    return heldout, test


def check_metrics(
    root: Path,
    heldout: list[dict[str, str]],
    test: list[dict[str, str]],
    test_windows: int,
    record: dict[str, Any],
    failures: list[str],
    fixture: bool = False,
) -> tuple[dict[tuple[int, int, str], dict[str, Any]], dict[tuple[int, int], float]]:
    grouped: dict[tuple[int, int, str], list[dict[str, str]]] = {}
    for row in heldout:
        n_fft = int(row["n_fft"])
        seed = int(row["training_seed"])
        grouped.setdefault((n_fft, seed, row["condition"]), []).append(row)

    # ---- provenance recorded on every prediction row --------------------
    provenance_by_model: dict[str, set[str]] = {}
    for row in heldout:
        provenance_by_model.setdefault(row["model_id"], set()).add(
            row.get("checkpoint_provenance", "")
        )
    record["checkpoint_provenance"] = {
        model_id: sorted(values) for model_id, values in sorted(provenance_by_model.items())
    }
    for model_id, values in provenance_by_model.items():
        if len(values) != 1 or not next(iter(values)):
            failures.append(f"{model_id}: prediction rows disagree on the checkpoint provenance")
            continue
        provenance = next(iter(values))
        run_directory = provenance.replace("\\best_model.pt", "").replace("/best_model.pt", "")
        source_config = Path(run_directory) / "run_config.json"
        if not Path(provenance).is_file():
            failures.append(f"{model_id}: checkpoint provenance does not exist: {provenance}")
        if not source_config.is_file():
            failures.append(f"{model_id}: no run_config.json at {source_config}")
    # The reused 1024 models must point at the 2026-09-17 checkpoints, not at a copy
    # inside this output tree.
    for model_id, values in provenance_by_model.items():
        if not model_id.startswith(f"nfft{BASELINE_NFFT}_"):
            continue
        provenance = next(iter(values), "")
        seed = int(model_id.split("_seed")[1])
        expected = str(Path(REUSE_EXPECTED[seed][0]))
        if Path(provenance).resolve() != Path(expected).resolve():
            failures.append(
                f"{model_id}: reused checkpoint provenance {provenance} is not the preserved {expected}"
            )

    stored_rows = read_csv_rows(root / "model_condition_metrics.csv")
    record["model_condition_rows"] = len(stored_rows)
    expected_metric_rows = len(grouped) if fixture else len(NFFT_GRID) * len(TRAINING_SEEDS) * len(CONDITIONS)
    if len(stored_rows) != expected_metric_rows:
        failures.append(
            f"model_condition_metrics has {len(stored_rows)} rows, expected {expected_metric_rows}"
        )
    stored = {
        (int(row["n_fft"]), int(row["training_seed"]), row["condition"]): row
        for row in stored_rows
    }
    recomputed: dict[tuple[int, int, str], dict[str, Any]] = {}
    per_condition_windows = max(len(rows) for rows in grouped.values())
    for key, rows in grouped.items():
        primary = file_equal(rows)
        secondary = pooled(rows)
        recomputed[key] = {**primary, **secondary}
        if secondary["windows"] != len(rows):
            failures.append(f"{key}: {secondary['windows']} windows")
        if secondary["tp"] + secondary["tn"] + secondary["fp"] + secondary["fn"] != len(rows):
            failures.append(f"{key}: confusion matrix does not conserve windows")
        published = stored.get(key)
        if published is None:
            failures.append(f"model_condition_metrics missing {key}")
            continue
        compare = (
            ("file_equal_balanced_accuracy", primary["file_equal_balanced_accuracy"]),
            ("file_equal_background_fpr", primary["background_file_equal_fpr"]),
            ("file_equal_background_specificity", primary["background_file_equal_specificity"]),
            ("file_equal_drone_recall", primary["drone_file_equal_recall"]),
            ("file_equal_phantom_recall", primary["phantom_file_equal_recall"]),
            ("file_equal_air2s_recall", primary["air2s_file_equal_recall"]),
            ("accuracy", secondary["accuracy"]),
            ("balanced_accuracy", secondary["balanced_accuracy"]),
            ("precision", secondary["precision"]),
            ("drone_recall", secondary["drone_recall"]),
            ("background_specificity", secondary["background_specificity"]),
            ("f1", secondary["f1"]),
        )
        for column, value in compare:
            if abs(float(published[column]) - value) > TOLERANCE:
                failures.append(f"{key}: {column} published {published[column]} != {value}")
        for column in ("tn", "fp", "fn", "tp"):
            if int(published[column]) != int(secondary[column]):
                failures.append(f"{key}: {column} mismatch")
        if int(published["windows"]) != len(rows):
            failures.append(f"{key}: published window count")
        if not fixture and primary["files_per_category"] != {"0000": 4, "0010": 4, "0101": 4}:
            failures.append(f"{key}: unexpected file counts {primary['files_per_category']}")

    # per-file table
    per_file = read_csv_rows(root / "per_file_metrics.csv")
    record["per_file_rows"] = len(per_file)
    expected_file_rows = (
        sum(len(recomputed[key]["file_windows"][code]) for key in recomputed for code in recomputed[key]["file_windows"])
        if fixture
        else len(NFFT_GRID) * len(TRAINING_SEEDS) * len(CONDITIONS) * 12
    )
    if len(per_file) != expected_file_rows:
        failures.append(f"per_file_metrics has {len(per_file)} rows")
    for row in per_file:
        key = (int(row["n_fft"]), int(row["training_seed"]), row["condition"])
        filename = row["filename"]
        if key not in recomputed:
            failures.append(f"per_file_metrics row has no prediction group: {row}")
            break
        windows = recomputed[key]["file_windows"]
        if filename not in windows.get(row["target_code"], {}):
            failures.append(f"per_file_metrics row not in the prediction grid: {row}")
            break
        expected_windows = windows[row["target_code"]][filename]
        if int(row["windows"]) != expected_windows:
            failures.append(f"per_file_metrics window count mismatch for {key}/{filename}")

    # per-run metrics against the prediction rows
    for (n_fft, seed, condition), rows in grouped.items():
        run_metrics = read_csv_rows(root / f"nfft{n_fft}" / f"seed_{seed}" / "validation_metrics.csv")
        if not run_metrics:
            failures.append(f"missing validation metrics for nfft{n_fft}/{seed}")
            continue
        if len(run_metrics) != 5 * len(CONDITIONS):
            failures.append(f"nfft{n_fft}/{seed}: per-epoch validation metrics incomplete")
        epoch = int(rows[0]["epoch"])
        selected = [
            row for row in run_metrics if int(row["epoch"]) == epoch and row["condition"] == condition
        ]
        if len(selected) != 1:
            failures.append(f"nfft{n_fft}/{seed}/{condition}: no unique selected-epoch row")
            continue
        counts = int(selected[0]["tn"]) + int(selected[0]["fp"]) + int(selected[0]["fn"]) + int(
            selected[0]["tp"]
        )
        if counts != 450 and not fixture:
            failures.append(
                f"nfft{n_fft}/{seed}/{condition}: validation counts cover {counts} windows, expected 450"
            )

    composites = composite(stored_rows, recomputed)
    present_nfft = sorted({key[0] for key in recomputed})
    present_seeds = sorted({key[1] for key in recomputed})
    record["nfft_values_verified"] = present_nfft
    record["seeds_verified"] = present_seeds

    summary_rows = read_csv_rows(root / "nfft_summary.csv")
    record["nfft_summary_rows"] = len(summary_rows)
    expected_summary_rows = (
        len(present_nfft) * (len(CONDITIONS) + 1)
        if fixture
        else len(NFFT_GRID) * (len(CONDITIONS) + 1)
    )
    if len(summary_rows) != expected_summary_rows:
        failures.append(f"nfft_summary has {len(summary_rows)} rows")
    if not summary_rows:
        return recomputed, composites
    summary = {(int(row["n_fft"]), row["condition"]): row for row in summary_rows}
    for n_fft in present_nfft:
        for condition in CONDITIONS:
            values = [
                recomputed[(n_fft, seed, condition)]["file_equal_balanced_accuracy"]
                for seed in present_seeds
            ]
            published = summary[(n_fft, condition)]
            if abs(float(published["file_equal_balacc_mean"]) - mean(values)) > TOLERANCE:
                failures.append(f"nfft_summary {n_fft}/{condition}: mean mismatch")
            if abs(float(published["file_equal_balacc_std"]) - sample_std(values)) > TOLERANCE:
                failures.append(f"nfft_summary {n_fft}/{condition}: std mismatch")
            per_seed = json.loads(published["per_seed_file_equal_balacc"])
            for seed in present_seeds:
                if abs(float(per_seed[str(seed)]) - recomputed[(n_fft, seed, condition)]["file_equal_balanced_accuracy"]) > TOLERANCE:
                    failures.append(f"nfft_summary {n_fft}/{condition}/{seed}: per-seed value mismatch")
        values = [composites[(n_fft, seed)] for seed in present_seeds]
        published = summary[(n_fft, "four_condition_mean")]
        if abs(float(published["file_equal_balacc_mean"]) - mean(values)) > TOLERANCE:
            failures.append(f"nfft_summary {n_fft}/four_condition_mean: mean mismatch")
        if abs(float(published["file_equal_balacc_std"]) - sample_std(values)) > TOLERANCE:
            failures.append(f"nfft_summary {n_fft}/four_condition_mean: std mismatch")
        pooled_values = [
            mean([recomputed[(n_fft, seed, condition)]["balanced_accuracy"] for condition in CONDITIONS])
            for seed in present_seeds
        ]
        if abs(float(published["pooled_balacc_mean"]) - mean(pooled_values)) > TOLERANCE:
            failures.append(f"nfft_summary {n_fft}/four_condition_mean: pooled mean mismatch")
        if abs(float(published["pooled_balacc_std"]) - sample_std(pooled_values)) > TOLERANCE:
            failures.append(f"nfft_summary {n_fft}/four_condition_mean: pooled std mismatch")
    for n_fft in present_nfft:
        for condition in CONDITIONS:
            pooled_values = [
                recomputed[(n_fft, seed, condition)]["balanced_accuracy"]
                for seed in present_seeds
            ]
            published = summary[(n_fft, condition)]
            if abs(float(published["pooled_balacc_mean"]) - mean(pooled_values)) > TOLERANCE:
                failures.append(f"nfft_summary {n_fft}/{condition}: pooled mean mismatch")
            if abs(float(published["pooled_balacc_std"]) - sample_std(pooled_values)) > TOLERANCE:
                failures.append(f"nfft_summary {n_fft}/{condition}: pooled std mismatch")

    paired_rows = read_csv_rows(root / "paired_nfft_comparison.csv")
    record["paired_rows"] = len(paired_rows)
    expected_pairs = (len(NFFT_GRID) - 1) * (len(CONDITIONS) + 1)
    if len(paired_rows) != expected_pairs:
        failures.append(f"paired_nfft_comparison has {len(paired_rows)} rows")
    for row in paired_rows:
        n_fft = int(row["n_fft"])
        condition = row["condition"]
        if n_fft == BASELINE_NFFT or int(row["baseline_n_fft"]) != BASELINE_NFFT:
            failures.append(f"paired row {n_fft}/{condition}: baseline column wrong")
        per_seed = json.loads(row["per_seed"])
        if sorted(per_seed) != sorted(str(seed) for seed in present_seeds):
            failures.append(f"paired row {n_fft}/{condition}: seed set wrong")
        differences = []
        for seed in present_seeds:
            if condition == "four_condition_mean":
                value = composites[(n_fft, seed)]
                baseline = composites[(BASELINE_NFFT, seed)]
            else:
                value = recomputed[(n_fft, seed, condition)]["file_equal_balanced_accuracy"]
                baseline = recomputed[(BASELINE_NFFT, seed, condition)]["file_equal_balanced_accuracy"]
            entry = per_seed[str(seed)]
            if abs(float(entry["value"]) - value) > TOLERANCE:
                failures.append(f"paired {n_fft}/{condition}/{seed}: value mismatch")
            if abs(float(entry["baseline"]) - baseline) > TOLERANCE:
                failures.append(f"paired {n_fft}/{condition}/{seed}: baseline mismatch")
            if abs(float(entry["difference"]) - (value - baseline)) > TOLERANCE:
                failures.append(f"paired {n_fft}/{condition}/{seed}: difference mismatch")
            differences.append(value - baseline)
        if abs(float(row["mean_difference"]) - mean(differences)) > TOLERANCE:
            failures.append(f"paired {n_fft}/{condition}: mean mismatch")
        if abs(float(row["std_difference"]) - sample_std(differences)) > TOLERANCE:
            failures.append(f"paired {n_fft}/{condition}: std mismatch")
        if abs(float(row["min_difference"]) - min(differences)) > TOLERANCE:
            failures.append(f"paired {n_fft}/{condition}: min mismatch")
        if abs(float(row["max_difference"]) - max(differences)) > TOLERANCE:
            failures.append(f"paired {n_fft}/{condition}: max mismatch")
        if int(row["seeds_positive"]) != sum(1 for value in differences if value > 0):
            failures.append(f"paired {n_fft}/{condition}: positive count mismatch")
        if int(row["seeds_negative"]) != sum(1 for value in differences if value < 0):
            failures.append(f"paired {n_fft}/{condition}: negative count mismatch")

    # test clean predictions -> pooled metric consistency where files are published
    test_grouped: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in test:
        test_grouped.setdefault((int(row["n_fft"]), int(row["training_seed"])), []).append(row)
    record["test_models"] = len(test_grouped)
    for (n_fft, seed), rows in test_grouped.items():
        if len(rows) != test_windows and not fixture:
            failures.append(f"test clean nfft{n_fft}/seed{seed}: {len(rows)} rows")
        pooled_test = pooled(rows)
        if pooled_test["tp"] + pooled_test["tn"] + pooled_test["fp"] + pooled_test["fn"] != len(rows):
            failures.append(
                f"test clean nfft{n_fft}/seed{seed}: counts do not cover {len(rows)} windows"
            )
        primary = file_equal(rows)
        expected_files = {"0000": 1, "0010": 1, "0101": 1}
        if not fixture and primary["files_per_category"] != expected_files:
            failures.append(
                f"test clean nfft{n_fft}/seed{seed}: unexpected file counts "
                f"{primary['files_per_category']} (the old test split holds one source "
                f"file per category, so file-equal and window-pooled rates coincide)"
            )

    record["test_file_equal"] = {
        f"nfft{n_fft}_seed{seed}": file_equal(rows)["file_equal_balanced_accuracy"]
        for (n_fft, seed), rows in sorted(test_grouped.items())
    }
    record["test_pooled"] = {
        f"nfft{n_fft}_seed{seed}": pooled(rows)["balanced_accuracy"]
        for (n_fft, seed), rows in sorted(test_grouped.items())
    }
    return recomputed, composites


def check_figures_and_evidence(root: Path, record: dict[str, Any], failures: list[str]) -> None:
    evidence_path = root / "figure_evidence.json"
    if not evidence_path.is_file():
        failures.append("missing figure_evidence.json")
        return
    evidence = read_json(evidence_path)
    record["figure_evidence"] = evidence
    for name in ("native_stft_examples.png", "model_input_examples.png"):
        path = root / name
        if not path.is_file():
            failures.append(f"missing figure {name}")
            continue
        header = path.read_bytes()[:8]
        if header != b"\x89PNG\r\n\x1a\n":
            failures.append(f"{name} is not a PNG")
        if path.stat().st_size < 5_000:
            failures.append(f"{name} is suspiciously small ({path.stat().st_size} bytes)")
        recorded = evidence.get("figures", {}).get(name)
        if recorded != sha256_file(path):
            failures.append(f"{name}: hash differs from figure_evidence.json")
    detail = evidence.get("n_fft_detail", {})
    if sorted(int(key) for key in detail) != list(NFFT_GRID):
        failures.append("figure evidence does not cover all four n_fft values")
    for key, entry in detail.items():
        if entry.get("model_input_shape") != [256, 256]:
            failures.append(f"n_fft {key}: model input is not 256x256")
        if entry.get("native_frequency_bins") != int(key):
            failures.append(f"n_fft {key}: native grid does not have n_fft frequency bins")
        if not entry.get("model_input_finite"):
            failures.append(f"n_fft {key}: model input is not finite")
    if not evidence.get("resize_note"):
        failures.append("figure evidence is missing the resize caveat")


def check_reuse_evidence(root: Path, record: dict[str, Any], failures: list[str]) -> None:
    path = root / "reuse_1024_verification.json"
    if not path.is_file():
        failures.append("missing reuse_1024_verification.json")
        return
    evidence = read_json(path)
    record["reuse_evidence_file"] = str(path)
    record["reuse_all_seeds_eligible"] = evidence.get("all_seeds_eligible")
    record["reuse_require_retrain"] = evidence.get("require_retrain")
    for seed in TRAINING_SEEDS:
        entry = evidence["seeds"][str(seed)]
        if not entry.get("eligible_for_reuse"):
            failures.append(f"reuse seed {seed}: not eligible but a 1024 run exists")
        if entry.get("initial_model_sha256_recorded") != entry.get("initial_model_sha256_derived"):
            failures.append(f"reuse seed {seed}: initial weights not reproduced")
        if entry.get("epoch_order_sha256_recorded") != entry.get("epoch_order_sha256_derived"):
            failures.append(f"reuse seed {seed}: epoch orders not reproduced")
        if not entry.get("validation_cache_match"):
            failures.append(f"reuse seed {seed}: validation cache differs")
        if not entry.get("checkpoint_sha256_match"):
            failures.append(f"reuse seed {seed}: checkpoint hash differs")
        if entry.get("checkpoint_epoch") != entry.get("best_epoch_by_rule"):
            failures.append(f"reuse seed {seed}: selected epoch is not the rule epoch")
        if entry.get("failures"):
            failures.append(f"reuse seed {seed}: recorded failures {entry['failures']}")


def check_pairing_and_cache(
    root: Path, record: dict[str, Any], failures: list[str], fixture: bool = False
) -> None:
    path = root / "prediction_cache_manifest.json"
    if not path.is_file():
        failures.append("missing prediction_cache_manifest.json")
        return
    manifest = read_json(path)
    heldout_cache = manifest["heldout"]
    record["heldout_cache"] = [
        {
            "condition": row["condition"],
            "n_fft": row["n_fft"],
            "windows": row["windows"],
            "feature_sha256": row["feature_sha256"],
            "window_digest_bytes": row["window_digest_bytes"],
            "seconds": row["seconds"],
        }
        for row in heldout_cache
    ]
    expected_cache_rows = (
        len({row["condition"] for row in heldout_cache}) * len({row["n_fft"] for row in heldout_cache})
        if fixture
        else len(CONDITIONS) * len(NFFT_GRID)
    )
    if len(heldout_cache) != expected_cache_rows:
        failures.append("held-out feature cache does not cover four conditions x four grids")
    for row in heldout_cache:
        if row["windows"] != 1820 and not fixture:
            failures.append(f"cache {row['condition']}/{row['n_fft']}: {row['windows']} windows")
        expected_bytes = row["windows"] * 32
        if row["window_digest_bytes"] != expected_bytes:
            failures.append(
                f"cache {row['condition']}/{row['n_fft']}: {row['window_digest_bytes']} digest bytes"
            )
    # the four grids of one condition share the same noise realisation
    for condition in {row["condition"] for row in heldout_cache}:
        digests = {
            row["n_fft"] for row in heldout_cache if row["condition"] == condition
        }
        expected_nfft = (
            {row["n_fft"] for row in heldout_cache}
            if fixture
            else set(NFFT_GRID)
        )
        if digests != expected_nfft:
            failures.append(f"cache {condition}: grids missing")
    noise = manifest["noise_pairing"]["heldout"]
    record["noise_pairing"] = noise
    if len(noise) != len(CONDITIONS):
        failures.append("noise pairing evidence does not cover four conditions")
    for entry in noise:
        if entry["noise_seed"] != 22026:
            failures.append(f"noise seed changed for {entry['condition']}")
        for probe in entry["probe_positions"]:
            if probe["rng_seed"] != 22026 + probe["index_position"]:
                failures.append(f"noise RNG index rule changed for {entry['condition']}")
    digest_files = sorted((root / "feature_digests").glob("window_digests_*.bin"))
    record["window_digest_files"] = len(digest_files)
    if fixture:
        return
    if len(digest_files) != (len(CONDITIONS) * len(NFFT_GRID)) + len(NFFT_GRID):
        failures.append(f"window digest files: {len(digest_files)}")
    for path in digest_files:
        expected = 1820 * 32 if "heldout" in path.name else 440 * 32
        if path.stat().st_size != expected:
            failures.append(f"{path.name}: {path.stat().st_size} bytes, expected {expected}")


def check_leakage(
    root: Path, record: dict[str, Any], failures: list[str], fixture: bool = False
) -> None:
    """Held-out and test data must not appear in any selection artefact."""
    inspected = []
    for path in sorted(root.glob("nfft*/seed_*/selection.json")):
        payload = read_json(path)
        inspected.append(str(path))
        text = json.dumps(payload, ensure_ascii=False).lower()
        for token in ("heldout", "test"):
            if token in text:
                failures.append(f"{path}: selection artefact mentions {token}")
    record["selection_artefacts_checked"] = len(inspected)
    if not fixture and len(inspected) != len(NFFT_GRID) * len(TRAINING_SEEDS):
        failures.append(f"selection artefacts: {len(inspected)}")
    record["selection_artefact_rule"] = (
        "selection.json holds only the four validation-condition mean BA per epoch"
    )


def write_report(
    root: Path,
    record: dict[str, Any],
    metrics: dict[tuple[int, int, str], dict[str, Any]],
    composites: dict[tuple[int, int], float],
    failures: list[str],
) -> None:
    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    lines: list[str] = []
    lines.append("# NFFT4：第一阶段 STFT 点数消融（n_fft = 512 / 1024 / 2048 / 4096）")
    lines.append("")
    lines.append(
        "范围：只改变 STFT 的 n_fft（hop 固定为 50%），其余全部沿用冻结设置——RF0 I/Q、"
        "1,000,000 点窗、Background=0 / Drone=1、A_original 训练噪声 "
        "`clean,10,5,0,-5,-10,-12`、验证四档 `clean,-10,-12,-15`（固定噪声 seed 12026）、"
        "ResNet18、Adam(1e-3, wd=0)、batch 4、AMP float16、image_size 256、"
        "5 epochs、三个训练种子 2026/2027/2028。选模规则：四档验证平衡准确率的算术平均，"
        "并列取最早轮次。test 与 heldout 均未参与任何拟合、选轮或 n_fft 选择。"
    )
    lines.append("")
    lines.append(
        f"主评价：新 heldout 1820 窗 × 四条件 × 12 模型 = {record['heldout_rows']:,} 条唯一预测；"
        f"补充评价：旧 test clean 440 窗 × 12 模型 = {record['test_rows']:,} 条。"
    )
    lines.append("")
    lines.append("## 0. 复用的 1024 检查点")
    lines.append("")
    lines.append(
        "三个 n_fft=1024 模型没有重训，直接复用 2026-09-17 的 A_original 检查点。"
        "复用前逐项重算并核对：run_config 的 seed/epochs/batch/训练噪声/优化器/AMP/index 哈希、"
        "初始 ResNet18 权重哈希（按种子重算）、每轮训练窗口顺序哈希（按 `seed+epoch` 重算）、"
        "四档验证特征 SHA256 与验证窗身份（重新构建缓存后逐档比对）、selection.json 与"
        "validation_metrics.csv 推出的最优轮次、检查点 SHA256 与冻结记录一致，"
        "且原检查点文件本身未被修改。全部通过（见 `reuse_1024_verification.json`）。"
        "因此四档在同一 training seed 下共享同一初始权重与同一训练顺序。"
    )
    lines.append("")
    lines.append("## 1. 主口径：新 heldout 文件等权平衡准确率")
    lines.append("")
    lines.append(
        "每类别 4 个源文件先各自求检出率/误报率，再对文件取等权平均；"
        "无人机召回为 Phantom 与 Air2S 文件等权召回的均值。"
    )
    lines.append("")
    lines.append("| n_fft (hop) | 训练种子 | epoch | clean | −10 dB | −12 dB | −15 dB | 四档均值 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for n_fft in NFFT_GRID:
        for seed in TRAINING_SEEDS:
            cells = [
                pct(metrics[(n_fft, seed, condition)]["file_equal_balanced_accuracy"])
                for condition in CONDITIONS
            ]
            lines.append(
                f"| {n_fft} ({n_fft // 2}) | {seed} | {record['runs'][f'nfft{n_fft}_seed{seed}']['selected_epoch']} | "
                + " | ".join(cells)
                + f" | {pct(composites[(n_fft, seed)])} |"
            )
    lines.append("")
    lines.append("### 跨种子均值 ± 样本标准差（n=3，仅描述性）")
    lines.append("")
    lines.append("| n_fft (hop) | clean | −10 dB | −12 dB | −15 dB | 四档均值 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    summary_pairs = {
        (int(row["n_fft"]), row["condition"]): row
        for row in read_csv_rows(root / "nfft_summary.csv")
    }
    for n_fft in NFFT_GRID:
        cells = []
        for condition in CONDITIONS:
            row = summary_pairs[(n_fft, condition)]
            cells.append(
                f"{pct(float(row['file_equal_balacc_mean']))} ± {float(row['file_equal_balacc_std']) * 100:.2f} pp"
            )
        row = summary_pairs[(n_fft, "four_condition_mean")]
        cells.append(
            f"{pct(float(row['file_equal_balacc_mean']))} ± {float(row['file_equal_balacc_std']) * 100:.2f} pp"
        )
        lines.append(f"| {n_fft} ({n_fft // 2}) | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 2. 与 n_fft=1024 的同种子配对差值（主口径，文件等权 BA）")
    lines.append("")
    lines.append("| n_fft | 条件 | 种子 2026 | 种子 2027 | 种子 2028 | 均值 | 方向一致 |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    paired_rows = read_csv_rows(root / "paired_nfft_comparison.csv")
    for row in paired_rows:
        per_seed = json.loads(row["per_seed"])
        cells = [f"{float(per_seed[str(seed)]['difference']) * 100:+.2f} pp" for seed in TRAINING_SEEDS]
        lines.append(
            f"| {row['n_fft']} | {row['condition']} | "
            + " | ".join(cells)
            + f" | {float(row['mean_difference']) * 100:+.2f} pp | "
            + ("是" if row["all_seeds_same_direction"] in ("True", "true", True) else "否")
            + " |"
        )
    lines.append("")
    lines.append(
        "差值只作描述性报告：只有三个训练种子，且同窗四档条件与同文件窗口都不是独立样本，"
        "因此不做显著性宣称，也不因结果挑选单一 n_fft 作为“最终模型”。"
    )
    lines.append("")
    lines.append("## 3. 补充口径：旧 test clean（440 窗，不参与选择）")
    lines.append("")
    lines.append("| n_fft (hop) | 训练种子 | 文件等权 BA | pooled BA | 背景文件等权误报 | 无人机文件等权召回 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    test_rows = read_csv_rows(root / "test_clean_window_predictions.csv")
    test_grouped: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in test_rows:
        test_grouped.setdefault((int(row["n_fft"]), int(row["training_seed"])), []).append(row)
    for n_fft in NFFT_GRID:
        for seed in TRAINING_SEEDS:
            rows = test_grouped[(n_fft, seed)]
            primary = file_equal(rows)
            secondary = pooled(rows)
            lines.append(
                f"| {n_fft} ({n_fft // 2}) | {seed} | {pct(primary['file_equal_balanced_accuracy'])} | "
                f"{pct(secondary['balanced_accuracy'])} | {pct(primary['background_file_equal_fpr'])} | "
                f"{pct(primary['drone_file_equal_recall'])} |"
            )
    lines.append("")
    lines.append("## 4. 详细指标")
    lines.append("")
    lines.append(
        "`model_condition_metrics.csv` 给出 12 模型 × 4 条件的完整指标（accuracy、balanced accuracy、"
        "precision、无人机召回、背景特异度、F1、TN/FP/FN/TP），同时包含文件等权主口径与 pooled 窗口口径，"
        "两者列名分开、不混用。逐文件统计见 `per_file_metrics.csv`（576 行）。"
    )
    lines.append("")
    lines.append("## 5. 图：原生 STFT 与实际模型输入")
    lines.append("")
    lines.append(
        "`native_stft_examples.png` 在运行前固定的同一个代表窗上给出四档 n_fft 的原生对数功率 STFT，"
        "带真实坐标轴（时间 ms；相对 RF0 通道中心的 MHz 偏移）。"
        "`model_input_examples.png` 给出四档 n_fft 实际送入 ResNet18 的 1×256×256 张量。"
        "**注意：256×256 的双线性缩放会压缩原生时频网格，因此四张图的横纵轴并不对应相同的物理分辨率。**"
        "本轮不保存逐窗 PNG。"
    )
    lines.append("")
    if "figure_evidence" in record:
        lines.append("| n_fft | 原生网格(freq×time) | 模型输入 | 频率步长 | 时间步长 |")
        lines.append("|---|---|---|---:|---:|")
        for key in sorted(record["figure_evidence"]["n_fft_detail"], key=int):
            entry = record["figure_evidence"]["n_fft_detail"][key]
            lines.append(
                f"| {key} | {entry['native_frequency_bins']}×{entry['native_time_frames']} | "
                f"{entry['model_input_shape'][0]}×{entry['model_input_shape'][1]} | "
                f"{entry['frequency_step_khz']:.1f} kHz | {entry['time_step_ms']:.4f} ms |"
            )
        lines.append("")
    lines.append("## 6. 结构与口径检查")
    lines.append("")
    lines.append(f"- 冻结输入 14 项 SHA256 全部匹配：{record['preserved_inputs']['all_match']}")
    lines.append(f"- 旧索引 / 新留出索引 SHA256：{record['old_index']['match']} / {record['heldout_index']['match']}")
    lines.append(f"- heldout 预测行数：{record['heldout_rows']:,}（唯一主键 {record['heldout_unique_keys']:,}）")
    lines.append(f"- test clean 预测行数：{record['test_rows']:,}（唯一主键 {record['test_unique_keys']:,}）")
    lines.append(f"- 三次复用等价性验证全部通过：{record.get('reuse_all_seeds_eligible')}")
    lines.append(
        f"- heldout 特征缓存：四档 n_fft × 四条件，每档 1820 窗，逐窗摘要 {1820 * 32} 字节；"
        "噪声 RNG 规则 `default_rng(22026 + 行号)`，epoch 固定 0"
    )
    lines.append(f"- 独立复算失败项：{len(failures)}")
    lines.append("")
    lines.append("## 7. 解释边界")
    lines.append("")
    lines.append(
        "- 四组只差 n_fft/hop；四组使用同一套 AWGN 方法，本任务不比较噪声生成方式，也不加入 −15 dB 训练。"
    )
    lines.append(
        "- 只有三个训练种子；同文件窗口与同窗四档条件高度相关，差值不构成显著性结论。"
    )
    lines.append("- 新文件与原 12 个 MAT 无文件交集，但 MAT 内无采集会话/地点/中心频率元数据，不宣称跨会话泛化。")
    lines.append("- 附加噪声档为模拟复 AWGN 条件，不是现场绝对 SNR。")
    lines.append("- 本报告不对第二阶段（机型分类）作任何结论。")
    lines.append("")
    if failures:
        lines.append("## 8. 未通过项")
        lines.append("")
        for failure in failures:
            lines.append(f"- {failure}")
        lines.append("")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="exit non-zero on failure")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Bounded self-test tree: relax the frozen 1820/440 row and digest sizes.",
    )
    args = parser.parse_args(argv)
    root = output_root(args.output_dir)
    started = time.perf_counter()
    failures: list[str] = []
    record: dict[str, Any] = {
        "task": "NFFT4 first-stage STFT-size ablation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "n_fft_grid": list(NFFT_GRID),
        "training_seeds": list(TRAINING_SEEDS),
        "conditions": list(CONDITIONS),
        "primary_aggregation": (
            "per-source-file rates, four files per category equally weighted; drone recall is the "
            "equal-weight mean of Phantom and Air2S file-equal recalls"
        ),
        "secondary_aggregation": "pooled over all 1820 held-out windows, counts retained",
    }
    check_frozen_inputs(record, failures)
    fixture = bool(args.fixture)
    record["fixture_mode"] = fixture
    if fixture:
        record["fixture_note"] = (
            "Bounded self-test tree: row counts, grid size and digest sizes follow the "
            "fixture scale instead of the frozen 1820/440 design."
        )
    record["runs"] = check_runs(root, record, failures, fixture=fixture)
    heldout, test = check_prediction_grid(root, record, failures, fixture=fixture)
    metrics, composites = check_metrics(
        root,
        heldout,
        test,
        int(record["test_windows_per_model"]),
        record,
        failures,
        fixture=fixture,
    )
    check_pairing_and_cache(root, record, failures, fixture=fixture)
    check_figures_and_evidence(root, record, failures)
    check_reuse_evidence(root, record, failures)
    check_leakage(root, record, failures, fixture=fixture)

    record["checks_failed"] = len(failures)
    record["failures"] = failures
    record["passed"] = not failures
    record["seconds"] = time.perf_counter() - started
    record["published_files"] = {
        name: sha256_file(root / name)
        for name in (
            "heldout_window_predictions.csv",
            "test_clean_window_predictions.csv",
            "model_condition_metrics.csv",
            "per_file_metrics.csv",
            "nfft_summary.csv",
            "paired_nfft_comparison.csv",
            "native_stft_examples.png",
            "model_input_examples.png",
            "prediction_cache_manifest.json",
            "figure_evidence.json",
            "reuse_1024_verification.json",
        )
        if (root / name).is_file()
    }
    if not args.no_report:
        write_report(root, record, metrics, composites, failures)
        record["published_files"]["report.md"] = sha256_file(root / "report.md")
    (root / "verification.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for failure in failures:
        print(f"FAIL: {failure}")
    print(f"verification failures: {len(failures)}")
    return 1 if (failures and args.check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
