"""Frozen held-out evaluation of the six selected DroneRFa checkpoints.

Protocol (frozen by Codex, see work/heldout-eval-config-20260918.json):
  * inputs   : 12 unused source files under F:, index heldout_window_index.csv (1820 rows)
  * frontend : existing RF0 I/Q -> rms_normalize -> complex AWGN -> Hamming STFT
               (n_fft=1024, hop=512) -> log power -> fftshift -> per-sample normalize
               -> 1x256x256 float32, built once per condition and shared by all six models
  * inference: existing ResNet18, CUDA AMP float16, batch 4,
               margin = float32(logit_drone) - float32(logit_background)
  * decision : margin > 0 -> Drone, margin == 0 -> Background (threshold 0, unchanged)
  * noise    : DroneRFaWindowDataset(index, split='heldout', fixed_snr=..., seed=22026)
               with epoch pinned to 0, so window i uses rng seed 22026 + global row index
  * primary  : per-source-file rates, four files per category equally weighted
  * secondary: pooled window counts

No training, threshold tuning, epoch/model re-selection or index reordering happens here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent

import numpy as np
import torch

from dronerfa_data import DroneRFaWindowDataset, load_index
from model import ResNet18, SpectrogramFrontEnd
from train_resnet18 import extract_features

# ---------------------------------------------------------------------------
# constants fixed before the run
# ---------------------------------------------------------------------------
CHANNELS = 1
IMAGE_SIZE = 256
FEATURE_DTYPE = np.float32
CONDITION_VALUES = {"clean": None, "-10dB": -10.0, "-12dB": -12.0, "-15dB": -15.0}
CATEGORY_NAMES = {"0000": "background", "0010": "phantom", "0101": "air2s"}
DRONE_CODES = ("0010", "0101")
BACKGROUND_CODE = "0000"
PROGRESS_EVERY = 25


def sha256_file(path: Path, label: str | None = None) -> str:
    """Hash a file; optional label prints start/end progress so long inputs stay visible."""
    digest = hashlib.sha256()
    if label:
        print(f"  hash start [{label}] {path.name}", flush=True)
    started = time.perf_counter()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    result = digest.hexdigest()
    if label:
        print(
            f"  hash done  [{label}] {path.name} "
            f"{path.stat().st_size} bytes sha256={result} "
            f"({time.perf_counter() - started:.1f}s)",
            flush=True,
        )
    return result


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command() -> str:
    return subprocess.list2cmdline([sys.executable, *sys.argv])


# ---------------------------------------------------------------------------
# integrity of frozen inputs
# ---------------------------------------------------------------------------
def verify_frozen_inputs(config: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    index_path = Path(config["index"])
    actual_index_hash = sha256_file(index_path, label="heldout index")
    if actual_index_hash != config["index_sha256"]:
        errors.append(
            f"index SHA256 mismatch: {actual_index_hash} != {config['index_sha256']}"
        )

    data_root = Path(config["data_root"])
    files: dict[str, Any] = {}
    total = len(config["file_sha256"])
    for position, (name, expected) in enumerate(config["file_sha256"].items(), start=1):
        path = data_root / name
        if not path.is_file():
            errors.append(f"missing data file: {path}")
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path, label=f"data {position}/{total} {name}")
        files[name] = {
            "path": str(path),
            "bytes": actual_size,
            "sha256": actual_hash,
            "sha256_expected": expected,
            "sha256_match": actual_hash == expected,
        }
        if actual_hash != expected:
            errors.append(f"data SHA256 mismatch for {name}")

    preserved_path = Path(config["preserved_inputs"])
    preserved = json.loads(preserved_path.read_text(encoding="utf-8"))
    preserved_check = []
    preserved_total = len(preserved["files"])
    for position, entry in enumerate(preserved["files"], start=1):
        path = Path(entry["path"])
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_hash = (
            sha256_file(path, label=f"preserved {position}/{preserved_total} {path.name}")
            if exists
            else None
        )
        match = bool(
            exists and actual_size == entry["bytes"] and actual_hash == entry["sha256"]
        )
        preserved_check.append(
            {
                "path": entry["path"],
                "expected_bytes": entry["bytes"],
                "expected_sha256": entry["sha256"],
                "actual_bytes": actual_size,
                "actual_sha256": actual_hash,
                "match": match,
            }
        )
        if not match:
            errors.append(f"preserved input changed: {entry['path']}")

    return {
        "index_path": str(index_path),
        "index_sha256": actual_index_hash,
        "index_sha256_matches_config": actual_index_hash == config["index_sha256"],
        "data_files": files,
        "preserved_inputs": {
            "reference": str(preserved_path),
            "reference_sha256": sha256_file(preserved_path),
            "checked": len(preserved_check),
            "all_match": all(item["match"] for item in preserved_check),
            "files": preserved_check,
        },
    }


def verify_checkpoints(
    config: dict[str, Any], training_index_sha256: str, errors: list[str]
) -> list[dict[str, Any]]:
    """Load every checkpoint and check its hash, declared epoch and selection policy."""
    records: list[dict[str, Any]] = []
    for model in config["models"]:
        path = Path(model["checkpoint"])
        record: dict[str, Any] = {
            "training_seed": model["training_seed"],
            "arm": model["arm"],
            "declared_epoch": model["epoch"],
            "declared_selection_policy": model["selection_policy"],
            "checkpoint": str(path),
            "sha256_expected": model["sha256"],
        }
        if not path.is_file():
            errors.append(f"missing checkpoint: {path}")
            record["status"] = "missing"
            records.append(record)
            continue
        actual_hash = sha256_file(path)
        record["sha256"] = actual_hash
        record["sha256_match"] = actual_hash == model["sha256"]
        if actual_hash != model["sha256"]:
            errors.append(f"checkpoint SHA256 mismatch: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        epoch = int(checkpoint["epoch"])
        run_config = checkpoint.get("run_config", {})
        record.update(
            {
                "epoch": epoch,
                "epoch_match": epoch == int(model["epoch"]),
                "run_config_selection": run_config.get("selection"),
                "run_config_tie_break": run_config.get("tie_break"),
                "run_config_prediction": run_config.get("prediction"),
                "run_config_index_sha256": run_config.get("index_sha256"),
                "run_config_training_seed": run_config.get("seed"),
                "run_config_train_snrs": run_config.get("train_snrs"),
                "has_frontend_state": "frontend_state" in checkpoint,
                "has_model_state": "model_state" in checkpoint,
            }
        )
        if epoch != int(model["epoch"]):
            errors.append(f"checkpoint epoch mismatch: {path} epoch={epoch}")
        if run_config.get("selection") != "mean_balanced_accuracy":
            errors.append(f"unexpected selection rule in {path}")
        if run_config.get("index_sha256") != training_index_sha256:
            errors.append(
                f"checkpoint {path} was trained against a different index "
                f"({run_config.get('index_sha256')})"
            )
        if run_config.get("seed") is not None and int(run_config["seed"]) != int(
            model["training_seed"]
        ):
            errors.append(f"checkpoint {path} training seed mismatch")

        selection_path = path.parent / "selection.json"
        history_path = path.parent / "training_history.csv"
        evidence: dict[str, Any] = {}
        if selection_path.is_file():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            evidence["selection_json"] = selection
            evidence["selection_json_best_multi_noise_epoch"] = selection.get(
                model["selection_policy"]
            )
            if selection.get(model["selection_policy"]) != epoch:
                errors.append(
                    f"{selection_path} does not select epoch {epoch} for "
                    f"{model['selection_policy']}"
                )
        elif history_path.is_file():
            with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
                history = list(csv.DictReader(handle))
            best = max(history, key=lambda row: float(row["mean_balanced_accuracy"]))
            evidence["history_path"] = str(history_path)
            evidence["history_best_multi_noise_epoch"] = int(best["epoch"])
            if int(best["epoch"]) != epoch:
                errors.append(
                    f"{history_path} best multi_noise epoch is {best['epoch']}, not {epoch}"
                )
        else:
            evidence["note"] = "no selection.json or training_history.csv beside checkpoint"
        record["selection_evidence"] = evidence
        record["status"] = "ok"
        records.append(record)
        del checkpoint
    return records


# ---------------------------------------------------------------------------
# aggregation helpers (also exercised by test_heldout_evaluation.py)
# ---------------------------------------------------------------------------
def file_equal_metrics(
    records: list[dict[str, Any]], grouped_files: dict[str, list[str]]
) -> dict[str, Any]:
    rates: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for category, names in grouped_files.items():
        rates[category] = {}
        counts[category] = {}
        for name in names:
            rows = [r for r in records if r["filename"] == name]
            if not rows:
                raise ValueError(f"no rows for file {name}")
            hits = sum(int(r["prediction"]) for r in rows)
            rates[category][name] = hits / len(rows)
            counts[category][name] = len(rows)
        if len(rates[category]) != 4:
            raise ValueError(f"expected four source files for category {category}")

    background_fpr = float(np.mean(list(rates[BACKGROUND_CODE].values())))
    phantom_recall = float(np.mean(list(rates["0010"].values())))
    air2s_recall = float(np.mean(list(rates["0101"].values())))
    drone_recall = 0.5 * (phantom_recall + air2s_recall)
    specificity = 1.0 - background_fpr
    return {
        "background_file_equal_fpr": background_fpr,
        "background_file_equal_specificity": specificity,
        "phantom_file_equal_recall": phantom_recall,
        "air2s_file_equal_recall": air2s_recall,
        "drone_file_equal_recall": drone_recall,
        "file_equal_balanced_accuracy": 0.5 * (specificity + drone_recall),
        "per_file_rates": rates,
        "per_file_windows": counts,
    }


def pooled_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    p = np.asarray([r["prediction"] for r in records], dtype=np.int64)
    codes = np.asarray([r["target_code"] for r in records])
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    per_category: dict[str, float] = {}
    for code in (*DRONE_CODES, BACKGROUND_CODE):
        mask = codes == code
        if code == BACKGROUND_CODE:
            per_category[CATEGORY_NAMES[code]] = float(np.mean(p[mask] == 0))
        else:
            per_category[CATEGORY_NAMES[code]] = float(np.mean(p[mask] == 1))
    return {
        "windows": int(len(y)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pooled_recall": recall,
        "pooled_specificity": specificity,
        "pooled_balanced_accuracy": 0.5 * (recall + specificity),
        "pooled_window_fpr": fp / max(fp + tn, 1),
        "pooled_phantom_window_recall": per_category["phantom"],
        "pooled_air2s_window_recall": per_category["air2s"],
        "pooled_background_window_specificity": per_category["background"],
    }


# ---------------------------------------------------------------------------
# feature production
# ---------------------------------------------------------------------------
def compute_condition_features(
    index_path: Path,
    condition: str,
    noise_seed: int,
    batch_size: int,
    windows: int,
    device: torch.device,
    frontend: SpectrogramFrontEnd,
) -> tuple[np.ndarray, dict[str, Any]]:
    snr = CONDITION_VALUES[condition]
    dataset = DroneRFaWindowDataset(
        index_path,
        split="heldout",
        fixed_snr=snr,
        apply_fixed_snr=snr is not None,
        seed=noise_seed,
    )
    if dataset.epoch != 0:
        raise RuntimeError("dataset epoch must stay 0 so the RNG index is the global row")
    if len(dataset) < windows:
        raise RuntimeError(f"dataset has {len(dataset)} rows, need {windows}")
    features = np.empty((windows, CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtype=FEATURE_DTYPE)
    digest = hashlib.sha256()
    window_digests = bytearray()
    labels = np.empty(windows, dtype=np.int64)
    started = time.perf_counter()
    try:
        for start in range(0, windows, batch_size):
            stop = min(start + batch_size, windows)
            signals = []
            for position in range(start, stop):
                tensor, label = dataset[position]
                signals.append(tensor)
                labels[position] = label
            batch = torch.stack(signals).to(device, non_blocking=True)
            with torch.inference_mode():
                batch_features = extract_features(frontend, batch)
            array = batch_features.detach().to("cpu", torch.float32).numpy()
            if array.shape != (stop - start, CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(f"unexpected feature shape {array.shape}")
            if not np.isfinite(array).all():
                raise RuntimeError(f"non-finite feature values in condition {condition}")
            features[start:stop] = array
            for row in array:
                window_digest = hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest()
                window_digests.extend(window_digest)
                digest.update(window_digest)
            del batch, batch_features
            if (start // batch_size + 1) % PROGRESS_EVERY == 0 or stop == windows:
                elapsed = time.perf_counter() - started
                print(
                    f"  features {condition} {stop}/{windows} "
                    f"({elapsed:.1f}s, {stop / max(elapsed, 1e-9):.1f} win/s)",
                    flush=True,
                )
    finally:
        dataset.close()

    elapsed = time.perf_counter() - started
    digest_bytes = bytes(window_digests)
    metadata = {
        "condition": condition,
        "fixed_snr_db": snr,
        "apply_fixed_snr": snr is not None,
        "noise_seed": noise_seed,
        "dataset_epoch": 0,
        "rng_rule": "np.random.default_rng(noise_seed + epoch*len(dataset) + global_row_index)",
        "windows": windows,
        "shape": [windows, CHANNELS, IMAGE_SIZE, IMAGE_SIZE],
        "dtype": str(FEATURE_DTYPE.__name__),
        "feature_sha256": sha256_bytes(features.tobytes()),
        "window_digest_bytes": len(digest_bytes),
        "window_digests_concat_sha256": digest.hexdigest(),
        "seconds": elapsed,
    }
    return features, {"metadata": metadata, "window_digests": digest_bytes, "labels": labels}


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
def load_checkpoint_models(
    config: dict[str, Any], device: torch.device
) -> list[dict[str, Any]]:
    loaded = []
    for model in config["models"]:
        path = Path(model["checkpoint"])
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        frontend = SpectrogramFrontEnd(n_fft=1024, hop_length=512, image_size=256).to(device)
        frontend.load_state_dict(checkpoint["frontend_state"])
        frontend.eval()
        network = ResNet18(input_channels=1, num_classes=2).to(device)
        network.load_state_dict(checkpoint["model_state"])
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        loaded.append(
            {
                "training_seed": int(model["training_seed"]),
                "arm": model["arm"],
                "epoch": int(model["epoch"]),
                "checkpoint": str(path),
                "network": network,
                "frontend": frontend,
            }
        )
        del checkpoint
    return loaded


@torch.inference_mode()
def predict_window_range(
    network: ResNet18,
    features: np.ndarray,
    start: int,
    stop: int,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Float32 logits, drone-minus-background margin and the threshold-0 decision."""
    count = stop - start
    logits_out = np.empty((count, 2), dtype=np.float32)
    margins = np.empty(count, dtype=np.float32)
    cursor = 0
    for offset in range(start, stop, batch_size):
        end = min(offset + batch_size, stop)
        batch = torch.from_numpy(np.ascontiguousarray(features[offset:end])).to(device)
        with torch.amp.autocast(
            device_type=device.type, enabled=amp_enabled, dtype=torch.float16
        ):
            logits = network(batch)
        logits = logits.float().cpu().numpy()
        rows = logits.shape[0]
        logits_out[cursor : cursor + rows] = logits
        margins[cursor : cursor + rows] = logits[:, 1] - logits[:, 0]
        cursor += rows
    predictions = (margins > 0.0).astype(np.int64)
    return logits_out, margins, predictions


# ---------------------------------------------------------------------------
# main flow
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR.parent / "work" / "heldout-eval-config-20260918.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--conditions",
        default="clean,-10dB,-12dB,-15dB",
        help="Comma separated subset of the frozen conditions, order preserved.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        help="Debug only: truncate the index and skip aggregation across files.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mixed precision for the ResNet, matching the original runs.",
    )
    parser.add_argument(
        "--write-index-rows",
        action="store_true",
        help="Also write index_rows.csv (window_predictions.csv already carries every field).",
    )
    return parser.parse_args()


def build_index_rows(
    index_path: Path, max_windows: int | None
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = load_index(index_path, split="heldout")
    if max_windows is not None:
        rows = rows[:max_windows]
    enriched = [{"heldout_index": position, **row} for position, row in enumerate(rows)]
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    return enriched, labels


def files_by_category(index_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in index_rows:
        bucket = grouped.setdefault(row["target_code"], [])
        if row["filename"] not in bucket:
            bucket.append(row["filename"])
    return {key: sorted(value) for key, value in grouped.items()}


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config["output_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = [token.strip() for token in args.conditions.split(",") if token.strip()]
    for condition in conditions:
        if condition not in CONDITION_VALUES:
            raise SystemExit(f"unknown condition {condition}")
    if args.batch_size < 1:
        raise SystemExit("batch size must be positive")

    started_at = utc_now()
    errors: list[str] = []
    print(f"Config: {args.config}", flush=True)
    print(f"Output: {output_dir}", flush=True)

    integrity = verify_frozen_inputs(config, errors)
    # Checkpoint provenance refers to the original training index; the heldout
    # index is independently verified above against its own frozen hash.
    training_index_path = PROJECT_DIR / "outputs" / "window_index.csv"
    training_index_sha256 = next(
        (
            entry["expected_sha256"]
            for entry in integrity["preserved_inputs"]["files"]
            if Path(entry["path"]) == training_index_path
        ),
        None,
    )
    if training_index_sha256 is None:
        errors.append("original training index is absent from preserved inputs")
        checkpoints = []
    else:
        checkpoints = verify_checkpoints(config, training_index_sha256, errors)
    if errors:
        write_json(
            output_dir / "aborted_run.json",
            {
                "started_at": started_at,
                "finished_at": utc_now(),
                "stage": "frozen input verification",
                "errors": errors,
            },
        )
        for item in errors:
            print(f"REJECTED: {item}", flush=True)
        return 2

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    index_rows, labels = build_index_rows(Path(config["index"]), args.max_windows)
    expected_windows = int(config["expected_counts"]["windows"])
    truncated = args.max_windows is not None and args.max_windows < expected_windows
    if not truncated and len(index_rows) != expected_windows:
        raise SystemExit(f"index has {len(index_rows)} rows, expected {expected_windows}")
    grouped = files_by_category(index_rows)
    print(
        "windows per category: "
        + ", ".join(
            f"{key}={int(np.sum([r['target_code'] == key for r in index_rows]))}"
            for key in sorted(grouped)
        ),
        flush=True,
    )
    print(f"files per category: {grouped}", flush=True)

    model_info = load_checkpoint_models(config, device)
    for info in model_info:
        print(
            f"model seed={info['training_seed']} arm={info['arm']} epoch={info['epoch']}",
            flush=True,
        )

    windows = len(index_rows)
    prediction_rows: list[dict[str, Any]] = []
    condition_metadata: list[dict[str, Any]] = []

    for condition in conditions:
        print(f"Condition {condition}: building shared float32 feature cache", flush=True)
        features, cache = compute_condition_features(
            Path(config["index"]),
            condition,
            int(config["noise_seed"]),
            args.batch_size,
            windows,
            device,
            model_info[0]["frontend"],
        )
        metadata = cache["metadata"]
        if not np.array_equal(cache["labels"], labels):
            raise RuntimeError(f"dataset labels disagree with the index for {condition}")
        # Write the concatenated per-window digests, not their aggregate hash,
        # so the verifier can compare any raw-IQ window with the evaluated cache.
        if metadata["windows"] * 32 != len(cache["window_digests"]):
            raise RuntimeError(
                f"expected {metadata['windows'] * 32} digest bytes, "
                f"got {len(cache['window_digests'])}"
            )
        condition_metadata.append(metadata)
        (output_dir / f"window_digests_{condition}.bin").write_bytes(cache["window_digests"])
        print(
            f"  {condition}: shape={metadata['shape']} dtype={metadata['dtype']} "
            f"feature_sha256={metadata['feature_sha256']} ({metadata['seconds']:.1f}s)",
            flush=True,
        )
        for info in model_info:
            logits, margins, predictions = predict_window_range(
                info["network"],
                features,
                0,
                windows,
                args.batch_size,
                device,
                amp_enabled,
            )
            if not np.isfinite(logits).all() or not np.isfinite(margins).all():
                raise RuntimeError(
                    f"non-finite logits for seed {info['training_seed']} {info['arm']} {condition}"
                )
            for position, row in enumerate(index_rows):
                margin = float(margins[position])
                prediction_rows.append(
                    {
                        "training_seed": info["training_seed"],
                        "arm": info["arm"],
                        "epoch": info["epoch"],
                        "condition": condition,
                        "heldout_index": row["heldout_index"],
                        "file_path": row["file_path"],
                        "filename": row["filename"],
                        "target_code": row["target_code"],
                        "target_name": row["target_name"],
                        "distance_code": row["distance_code"],
                        "segment_code": row["segment_code"],
                        "split": row["split"],
                        "label": row["label"],
                        "block_index": row["block_index"],
                        "window_index": row["window_index"],
                        "start": row["start"],
                        "length": row["length"],
                        "source_samples": row["source_samples"],
                        "logit_background": float(logits[position, 0]),
                        "logit_drone": float(logits[position, 1]),
                        "logit_margin": margin,
                        "drone_score": float(
                            1.0 / (1.0 + np.exp(-np.float64(margin)))
                        ),
                        "prediction": int(predictions[position]),
                    }
                )
            print(
                f"  seed={info['training_seed']} {info['arm']} {condition}: "
                f"predicted drone {int(predictions.sum())}/{windows}",
                flush=True,
            )
        del features, cache
        torch.cuda.empty_cache()

    write_csv(output_dir / "window_predictions.csv", prediction_rows)
    if args.write_index_rows:
        write_csv(
            output_dir / "index_rows.csv",
            [
                {
                    "heldout_index": row["heldout_index"],
                    "file_path": row["file_path"],
                    "filename": row["filename"],
                    "target_code": row["target_code"],
                    "target_name": row["target_name"],
                    "distance_code": row["distance_code"],
                    "segment_code": row["segment_code"],
                    "split": row["split"],
                    "label": row["label"],
                    "block_index": row["block_index"],
                    "window_index": row["window_index"],
                    "start": row["start"],
                    "length": row["length"],
                    "source_samples": row["source_samples"],
                }
                for row in index_rows
            ],
        )
    finished_at = utc_now()

    gpu = None
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_bytes": int(total_bytes),
            "free_bytes_after_run": int(free_bytes),
        }

    manifest = {
        "created_at": started_at,
        "finished_at": finished_at,
        "status": "completed_predictions",
        "task": "frozen held-out evaluation of six selected checkpoints",
        "run_command": run_command(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": gpu,
        "amp": amp_enabled,
        "amp_dtype": "float16" if amp_enabled else None,
        "batch_size": args.batch_size,
        "batch_size_note": (
            "config default 4; per-window logits are batch independent because the "
            "frontend normalizes each sample and eval-mode BatchNorm is per sample"
        ),
        "conditions": conditions,
        "condition_values_db": {name: CONDITION_VALUES[name] for name in conditions},
        "noise_seed": config["noise_seed"],
        "noise_index_rule": config["noise_index_rule"],
        "epoch_pinned": 0,
        "decision_rule": config["inference"]["decision"],
        "threshold": config["inference"]["threshold"],
        "truncated_debug_run": bool(truncated),
        "index": {
            "path": config["index"],
            "sha256": integrity["index_sha256"],
            "sha256_matches_config": integrity["index_sha256_matches_config"],
            "rows": windows,
            "split": "heldout",
            "order_preserved": True,
            "windows_per_category": {
                key: int(np.sum([r["target_code"] == key for r in index_rows]))
                for key in sorted(grouped)
            },
            "files_per_category": grouped,
        },
        "data_files": integrity["data_files"],
        "preserved_inputs": integrity["preserved_inputs"],
        "checkpoints": checkpoints,
        "frozen_config": {
            "path": str(args.config),
            "sha256": sha256_file(args.config),
            "frontend": config["frontend"],
            "aggregation": config["aggregation"],
            "interpretation_limits": config["interpretation_limits"],
            "constraints": config["constraints"],
        },
        "source_sha256": {
            name: sha256_file(PROJECT_DIR / name)
            for name in (
                "evaluate_heldout_models.py",
                "dronerfa_data.py",
                "model.py",
                "train_resnet18.py",
            )
        },
        "feature_cache": {
            "location": (
                "in-memory float32 CPU array per condition, freed before the next condition"
            ),
            "expected_bytes_per_condition": int(
                windows * CHANNELS * IMAGE_SIZE * IMAGE_SIZE * 4
            ),
            "shared_by_all_models": True,
            "disk_cache_written": False,
            "window_digest_files": [
                f"window_digests_{condition}.bin" for condition in conditions
            ],
            "conditions": condition_metadata,
        },
        "resume_strategy": (
            "window_predictions.csv is written once, after every condition/model pair "
            "completed; per-condition feature sha256 and per-window digests are written "
            "immediately after each cache is built so a failed run is auditable and can "
            "be restarted condition by condition without re-deriving the protocol"
        ),
        "expected_counts": config["expected_counts"],
        "prediction_rows": len(prediction_rows),
        "index_rows_written": bool(args.write_index_rows),
        "max_windows_argument": args.max_windows,
    }
    write_json(output_dir / "evaluation_manifest.json", manifest)
    print(f"Wrote {len(prediction_rows)} prediction rows to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
