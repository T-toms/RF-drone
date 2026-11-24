"""Independent verification of the frozen held-out evaluation products.

Checks, in order:

  1. manifest consistency: index hash, six checkpoint hashes/epochs, prediction count,
     preserved-input hashes, per-window digest files
  2. real-IQ feature recheck: for a fixed 36-window set (first/middle/last window of
     every source file), rebuild the float32 features through the existing dataset +
     frontend from the raw MAT files and compare the per-window SHA256 with
     window_digests_<condition>.bin; also confirm the AWGN is reproducible by reading
     the same window twice and comparing the tensors
  3. CSV spot check: replay the ORIGINAL global batch-4 boundaries. For every target
     window i the batch start is (i // 4) * 4 with stop = min(start + 4, 1820), so the
     frontend float32 pass and the CUDA AMP float16 network pass see exactly the batch
     composition of the production run. One rebuilt feature set per condition is shared
     by all six models. Logits and predictions are compared against
     window_predictions.csv with an absolute tolerance of 1e-4 (not widened).
  4. two-run parity: the four complete feature SHA256 values and the 43,680 prediction
     rows of this run are compared with the preserved first-run evidence
  5. aggregation recompute: per-file / per-condition metrics, the paired differences and
     the uniqueness of the paired key set are recomputed from the CSV
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dronerfa_data import DroneRFaWindowDataset  # noqa: E402
from evaluate_heldout_models import (  # noqa: E402
    CONDITION_VALUES,
    file_equal_metrics,
    pooled_metrics,
    predict_window_range,
)
from model import ResNet18, SpectrogramFrontEnd  # noqa: E402
from train_resnet18 import extract_features  # noqa: E402

OUTPUT_DIR = PROJECT_DIR / "outputs" / "heldout_evaluation_20260918"
RUN1_DIR = OUTPUT_DIR / "run1_evidence"
CONFIG_PATH = PROJECT_DIR.parent / "work" / "heldout-eval-config-20260918.json"
INDEX_PATH = PROJECT_DIR / "outputs" / "heldout_preparation_20260917" / "heldout_window_index.csv"
CONDITIONS = ["clean", "-10dB", "-12dB", "-15dB"]
MODEL_ORDER = [
    (2026, "A_original"),
    (2026, "B_plus_minus15"),
    (2027, "A_original"),
    (2027, "B_plus_minus15"),
    (2028, "A_original"),
    (2028, "B_plus_minus15"),
]
TRAINING_SEEDS = sorted({seed for seed, _ in MODEL_ORDER})
PRODUCTION_BATCH_SIZE = 4
LOGIT_ABS_TOLERANCE = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Results:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.failures: list[str] = []

    def record(self, name: str, passed: bool, detail: Any) -> None:
        self.checks.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)
        if not passed:
            self.failures.append(name)


def load_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    **raw,
                    "training_seed": int(raw["training_seed"]),
                    "heldout_index": int(raw["heldout_index"]),
                    "label": int(raw["label"]),
                    "logit_background": float(raw["logit_background"]),
                    "logit_drone": float(raw["logit_drone"]),
                    "logit_margin": float(raw["logit_margin"]),
                    "drone_score": float(raw["drone_score"]),
                    "prediction": int(raw["prediction"]),
                }
            )
    return rows


def load_grouping() -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    grouped: dict[str, list[str]] = {}
    order: list[dict[str, Any]] = []
    with INDEX_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for position, row in enumerate(csv.DictReader(handle)):
            grouped.setdefault(row["target_code"], [])
            if row["filename"] not in grouped[row["target_code"]]:
                grouped[row["target_code"]].append(row["filename"])
            order.append({"heldout_index": position, **row})
    return {key: sorted(value) for key, value in grouped.items()}, order


def choose_windows(index_order: list[dict[str, Any]]) -> list[int]:
    """Fixed 36 targets: first, middle and last window of each of the twelve files."""
    by_file: dict[str, list[int]] = {}
    for position, row in enumerate(index_order):
        by_file.setdefault(row["filename"], []).append(position)
    chosen: list[int] = []
    for filename in sorted(by_file, key=lambda name: by_file[name][0]):
        positions = by_file[filename]
        chosen.extend([positions[0], positions[len(positions) // 2], positions[-1]])
    return sorted(set(chosen))


def batch_start(position: int, total: int) -> int:
    """Start index of the production batch that contains `position`."""
    return (position // PRODUCTION_BATCH_SIZE) * PRODUCTION_BATCH_SIZE


def rebuild_features_for_batches(
    index_path: Path,
    condition: str,
    noise_seed: int,
    batch_starts: list[int],
    total_windows: int,
    device: torch.device,
    frontend: SpectrogramFrontEnd,
) -> dict[int, np.ndarray]:
    """Rebuild the frontend features of complete production batches, keyed by window."""
    snr = CONDITION_VALUES[condition]
    dataset = DroneRFaWindowDataset(
        index_path,
        split="heldout",
        fixed_snr=snr,
        apply_fixed_snr=snr is not None,
        seed=noise_seed,
    )
    features_by_window: dict[int, np.ndarray] = {}
    try:
        for start in batch_starts:
            stop = min(start + PRODUCTION_BATCH_SIZE, total_windows)
            signals = torch.stack([dataset[position][0] for position in range(start, stop)])
            signals = signals.to(device)
            with torch.inference_mode():
                batch_features = extract_features(frontend, signals)
            array = batch_features.detach().to("cpu", torch.float32).numpy()
            for offset, position in enumerate(range(start, stop)):
                features_by_window[position] = array[offset]
    finally:
        dataset.close()
    return features_by_window


def main() -> int:
    results = Results()
    manifest = json.loads((OUTPUT_DIR / "evaluation_manifest.json").read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    predictions = load_predictions(OUTPUT_DIR / "window_predictions.csv")
    grouped, index_order = load_grouping()

    # ---- 1. manifest consistency ----------------------------------------
    results.record(
        "manifest prediction rows",
        manifest["prediction_rows"] == len(predictions) == 43680,
        f"{manifest['prediction_rows']} manifest / {len(predictions)} csv",
    )
    results.record(
        "index hash",
        manifest["index"]["sha256"] == config["index_sha256"] == sha256_file(INDEX_PATH),
        manifest["index"]["sha256"],
    )
    checkpoint_ok = all(
        item["sha256_match"] and item["epoch_match"] for item in manifest["checkpoints"]
    )
    results.record(
        "checkpoint hashes and epochs",
        checkpoint_ok and len(manifest["checkpoints"]) == 6,
        [
            f"{item['training_seed']}/{item['arm']}: epoch={item['epoch']} "
            f"sha_match={item['sha256_match']} selection={item['run_config_selection']}"
            for item in manifest["checkpoints"]
        ],
    )
    results.record(
        "preserved inputs unchanged",
        manifest["preserved_inputs"]["all_match"],
        f"{manifest['preserved_inputs']['checked']} files",
    )
    digest_sizes = {
        condition: (OUTPUT_DIR / f"window_digests_{condition}.bin").stat().st_size
        if (OUTPUT_DIR / f"window_digests_{condition}.bin").is_file()
        else None
        for condition in CONDITIONS
    }
    results.record(
        "per-window digest files hold 32 bytes per window",
        all(size == 32 * len(index_order) for size in digest_sizes.values()),
        f"{digest_sizes} (expected {32 * len(index_order)})",
    )

    # ---- 2. real IQ feature recheck -------------------------------------
    windows = choose_windows(index_order)
    batch_starts = sorted({batch_start(position, len(index_order)) for position in windows})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        results.record("CUDA available for AMP parity", False, str(device))
    amp_enabled = device.type == "cuda"
    noise_seed = int(config["noise_seed"])
    frontend = SpectrogramFrontEnd(n_fft=1024, hop_length=512, image_size=256).to(device)
    frontend.eval()

    digest_ok = True
    digest_detail: dict[str, Any] = {}
    features_by_condition: dict[str, dict[int, np.ndarray]] = {}
    for condition in CONDITIONS:
        digest_bytes = (OUTPUT_DIR / f"window_digests_{condition}.bin").read_bytes()
        if len(digest_bytes) != 32 * len(index_order):
            digest_ok = False
            continue
        rebuilt = rebuild_features_for_batches(
            INDEX_PATH, condition, noise_seed, batch_starts, len(index_order), device, frontend
        )
        features_by_condition[condition] = rebuilt
        mismatches = [
            position
            for position in windows
            if hashlib.sha256(np.ascontiguousarray(rebuilt[position]).tobytes()).digest()
            != digest_bytes[position * 32 : (position + 1) * 32]
        ]
        # Same window read twice must reproduce the same AWGN tensor.
        dataset = DroneRFaWindowDataset(
            INDEX_PATH,
            split="heldout",
            fixed_snr=CONDITION_VALUES[condition],
            apply_fixed_snr=CONDITION_VALUES[condition] is not None,
            seed=noise_seed,
        )
        try:
            repeat_mismatches = [
                position
                for position in windows
                if not torch.equal(dataset[position][0], dataset[position][0])
            ]
        finally:
            dataset.close()
        digest_detail[condition] = {
            "windows_checked": len(windows),
            "distinct_production_batches_rebuilt": len(batch_starts),
            "feature_digest_mismatches": mismatches,
            "repeat_read_mismatches": repeat_mismatches,
        }
        digest_ok = digest_ok and not mismatches and not repeat_mismatches
    results.record(
        "real IQ feature recheck against cached digests (production batch 4)",
        digest_ok,
        json.dumps(digest_detail, ensure_ascii=False),
    )

    # ---- 3. CSV spot check on the original batch boundaries -------------
    csv_lookup = {
        (row["training_seed"], row["arm"], row["condition"], row["heldout_index"]): row
        for row in predictions
    }
    tolerance_detail: dict[str, Any] = {}
    spot_ok = True
    for seed, arm in MODEL_ORDER:
        model_entry = next(
            item
            for item in config["models"]
            if item["training_seed"] == seed and item["arm"] == arm
        )
        checkpoint = torch.load(model_entry["checkpoint"], map_location="cpu", weights_only=False)
        model_frontend = SpectrogramFrontEnd(n_fft=1024, hop_length=512, image_size=256).to(device)
        model_frontend.load_state_dict(checkpoint["frontend_state"])
        model_frontend.eval()
        network = ResNet18(input_channels=1, num_classes=2).to(device)
        network.load_state_dict(checkpoint["model_state"])
        network.eval()
        per_condition: dict[str, Any] = {}
        for condition in CONDITIONS:
            rebuilt = features_by_condition[condition]
            max_logit_gap = 0.0
            max_margin_gap = 0.0
            prediction_mismatches: list[int] = []
            target_set = set(windows)
            for start in batch_starts:
                stop = min(start + PRODUCTION_BATCH_SIZE, len(index_order))
                targets = [
                    position for position in range(start, stop) if position in target_set
                ]
                if not targets:
                    continue
                batch = np.stack([rebuilt[position] for position in range(start, stop)])
                logits, margins, flags = predict_window_range(
                    network, batch, 0, stop - start, PRODUCTION_BATCH_SIZE, device, amp_enabled
                )
                for position in targets:
                    offset = position - start
                    stored = csv_lookup[(seed, arm, condition, position)]
                    max_logit_gap = max(
                        max_logit_gap,
                        abs(logits[offset, 0] - np.float32(stored["logit_background"])),
                        abs(logits[offset, 1] - np.float32(stored["logit_drone"])),
                    )
                    max_margin_gap = max(
                        max_margin_gap,
                        abs(margins[offset] - np.float32(stored["logit_margin"])),
                    )
                    if int(flags[offset]) != stored["prediction"]:
                        prediction_mismatches.append(position)
            per_condition[condition] = {
                "targets": len(windows),
                "max_abs_logit_gap_vs_csv": float(max_logit_gap),
                "max_abs_margin_gap_vs_csv": float(max_margin_gap),
                "prediction_mismatches_vs_csv": prediction_mismatches,
            }
            if (
                max_logit_gap > LOGIT_ABS_TOLERANCE
                or max_margin_gap > LOGIT_ABS_TOLERANCE
                or prediction_mismatches
            ):
                spot_ok = False
        tolerance_detail[f"seed{seed}_{arm}"] = {
            "epoch": model_entry["epoch"],
            "conditions": per_condition,
        }
        del checkpoint, network, model_frontend
        torch.cuda.empty_cache()
    results.record(
        f"CSV spot check, original batch-4 alignment (tolerance {LOGIT_ABS_TOLERANCE})",
        spot_ok,
        json.dumps(tolerance_detail, ensure_ascii=False),
    )

    # Threshold consistency across every stored row.
    threshold_violations = [
        row
        for row in predictions
        if int(float(row["logit_margin"]) > 0.0) != row["prediction"]
        or abs(row["logit_margin"] - (row["logit_drone"] - row["logit_background"])) > 1e-5
    ]
    results.record(
        "threshold and margin consistency over all rows",
        not threshold_violations,
        f"violations={len(threshold_violations)}",
    )
    non_finite = [
        row
        for row in predictions
        if not all(
            np.isfinite(row[key])
            for key in ("logit_background", "logit_drone", "logit_margin", "drone_score")
        )
    ]
    results.record("finite logits everywhere", not non_finite, f"non_finite={len(non_finite)}")

    # ---- 4. two-run parity ---------------------------------------------
    if RUN1_DIR.is_dir():
        run1_manifest = json.loads(
            (RUN1_DIR / "evaluation_manifest_run1.json").read_text(encoding="utf-8")
        )
        run1_features = {
            item["condition"]: item["feature_sha256"]
            for item in run1_manifest["feature_cache"]["conditions"]
        }
        run2_features = {
            item["condition"]: item["feature_sha256"]
            for item in manifest["feature_cache"]["conditions"]
        }
        feature_parity = {
            condition: {
                "run1": run1_features.get(condition),
                "run2": run2_features.get(condition),
                "match": run1_features.get(condition) == run2_features.get(condition),
            }
            for condition in CONDITIONS
        }
        results.record(
            "four complete feature SHA256 match the first run",
            all(item["match"] for item in feature_parity.values()),
            json.dumps(feature_parity, ensure_ascii=False),
        )
        run1_predictions = load_predictions(RUN1_DIR / "window_predictions_run1.csv")
        run1_lookup = {
            (row["training_seed"], row["arm"], row["condition"], row["heldout_index"]): row
            for row in run1_predictions
        }
        differences: list[str] = []
        max_logit_delta = 0.0
        prediction_changes = 0
        if len(run1_predictions) != len(predictions):
            differences.append(
                f"row count {len(run1_predictions)} vs {len(predictions)}"
            )
        for row in predictions:
            key = (row["training_seed"], row["arm"], row["condition"], row["heldout_index"])
            first = run1_lookup.get(key)
            if first is None:
                differences.append(f"missing run1 row {key}")
                continue
            max_logit_delta = max(
                max_logit_delta,
                abs(row["logit_background"] - first["logit_background"]),
                abs(row["logit_drone"] - first["logit_drone"]),
            )
            if row["prediction"] != first["prediction"]:
                prediction_changes += 1
        results.record(
            "43,680 prediction rows identical between runs",
            not differences and prediction_changes == 0,
            f"row_differences={len(differences)} prediction_changes={prediction_changes} "
            f"max_abs_logit_delta={max_logit_delta:.3e}",
        )
    else:
        results.record("first run evidence present", False, str(RUN1_DIR))

    # ---- 5. aggregation recompute ---------------------------------------
    recompute_detail: dict[str, Any] = {}
    aggregation_ok = True
    model_metrics_rows = list(
        csv.DictReader(
            (OUTPUT_DIR / "model_condition_metrics.csv").open("r", encoding="utf-8-sig", newline="")
        )
    )
    for model_row in model_metrics_rows:
        seed = int(model_row["training_seed"])
        arm = model_row["arm"]
        condition = model_row["condition"]
        bucket = [
            row
            for row in predictions
            if row["training_seed"] == seed
            and row["arm"] == arm
            and row["condition"] == condition
        ]
        primary = file_equal_metrics(bucket, grouped)
        secondary = pooled_metrics(bucket)
        gaps = {
            "file_equal_balanced_accuracy": abs(
                primary["file_equal_balanced_accuracy"]
                - float(model_row["file_equal_balanced_accuracy"])
            ),
            "background_file_equal_fpr": abs(
                primary["background_file_equal_fpr"]
                - float(model_row["background_file_equal_fpr"])
            ),
            "phantom_file_equal_recall": abs(
                primary["phantom_file_equal_recall"]
                - float(model_row["phantom_file_equal_recall"])
            ),
            "air2s_file_equal_recall": abs(
                primary["air2s_file_equal_recall"]
                - float(model_row["air2s_file_equal_recall"])
            ),
            "pooled_tp": abs(secondary["tp"] - int(model_row["pooled_tp"])),
            "pooled_balanced_accuracy": abs(
                secondary["pooled_balanced_accuracy"]
                - float(model_row["pooled_balanced_accuracy"])
            ),
        }
        worst = max(gaps.values())
        recompute_detail[f"{seed}_{arm}_{condition}"] = worst
        if worst > 1e-9:
            aggregation_ok = False
    results.record(
        "model_condition_metrics recompute from CSV",
        aggregation_ok and len(model_metrics_rows) == 24,
        f"{len(model_metrics_rows)} rows, max_gap={max(recompute_detail.values()):.3e}",
    )

    per_file_ok = True
    per_file_rows = list(
        csv.DictReader(
            (OUTPUT_DIR / "per_file_metrics.csv").open("r", encoding="utf-8-sig", newline="")
        )
    )
    for row in per_file_rows:
        key = (
            int(row["training_seed"]),
            row["arm"],
            row["condition"],
            row["filename"],
        )
        bucket = [
            item
            for item in predictions
            if (item["training_seed"], item["arm"], item["condition"], item["filename"]) == key
        ]
        expected_windows = len(bucket)
        expected_hits = sum(item["prediction"] for item in bucket)
        if int(row["windows"]) != expected_windows:
            per_file_ok = False
        if row["role"] == "drone":
            if int(row["tp"]) != expected_hits or int(row["fn"]) != expected_windows - expected_hits:
                per_file_ok = False
        else:
            if int(row["fp"]) != expected_hits or int(row["tn"]) != expected_windows - expected_hits:
                per_file_ok = False
    results.record(
        "per_file_metrics recompute from CSV",
        per_file_ok and len(per_file_rows) == 288,
        f"{len(per_file_rows)} rows checked",
    )

    paired_rows = list(
        csv.DictReader(
            (OUTPUT_DIR / "paired_comparison.csv").open("r", encoding="utf-8-sig", newline="")
        )
    )
    paired_keys = [(int(row["training_seed"]), row["condition"]) for row in paired_rows]
    expected_keys = {
        (seed, condition)
        for seed in TRAINING_SEEDS
        for condition in CONDITIONS + ["four_condition_mean"]
    }
    results.record(
        "paired_comparison holds 15 unique complete keys",
        len(paired_rows) == 15
        and len(set(paired_keys)) == 15
        and set(paired_keys) == expected_keys,
        f"rows={len(paired_rows)} unique={len(set(paired_keys))} expected={len(expected_keys)}",
    )
    model_metrics = {
        (int(row["training_seed"]), row["arm"], row["condition"]): row
        for row in model_metrics_rows
    }
    paired_ok = True
    paired_gap = 0.0
    for row in paired_rows:
        seed = int(row["training_seed"])
        condition = row["condition"]
        if condition == "four_condition_mean":
            a_value = float(
                np.mean(
                    [
                        float(
                            model_metrics[(seed, "A_original", item)][
                                "file_equal_balanced_accuracy"
                            ]
                        )
                        for item in CONDITIONS
                    ]
                )
            )
            b_value = float(
                np.mean(
                    [
                        float(
                            model_metrics[(seed, "B_plus_minus15", item)][
                                "file_equal_balanced_accuracy"
                            ]
                        )
                        for item in CONDITIONS
                    ]
                )
            )
        else:
            a_value = float(
                model_metrics[(seed, "A_original", condition)]["file_equal_balanced_accuracy"]
            )
            b_value = float(
                model_metrics[(seed, "B_plus_minus15", condition)]["file_equal_balanced_accuracy"]
            )
        gap = max(
            abs(a_value - float(row["A_file_equal_balacc"])),
            abs(b_value - float(row["B_file_equal_balacc"])),
            abs((b_value - a_value) - float(row["B_minus_A_file_equal_balacc"])),
        )
        paired_gap = max(paired_gap, gap)
        if gap > 1e-12:
            paired_ok = False
    results.record(
        "paired_comparison recompute",
        paired_ok,
        f"{len(paired_rows)} rows, max_gap={paired_gap:.3e}",
    )

    summary = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_target_windows": windows,
        "distinct_production_batches_per_condition": len(batch_starts),
        "production_batch_size": PRODUCTION_BATCH_SIZE,
        "logit_abs_tolerance": LOGIT_ABS_TOLERANCE,
        "checks": results.checks,
        "failures": results.failures,
        "passed": not results.failures,
    }
    (OUTPUT_DIR / "independent_verification.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"verification finished: {len(results.checks)} checks, "
        f"{len(results.failures)} failures",
        flush=True,
    )
    return 0 if not results.failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
