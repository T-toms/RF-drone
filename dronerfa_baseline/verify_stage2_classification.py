"""Independent verification of the STAGE2-BASE artefacts.

Reads only the produced CSV/JSON/PNG files and the frozen indices, and recomputes
every reported number from the raw text with its own code: the run module never
supplies an expected value to these checks.

    python verify_stage2_classification.py [--output-dir ROOT] [--fixture]

``--fixture`` skips the frozen-input re-hash and the full-size expectations, which
lets ``test_stage2_classification.py`` point the verifier at a small synthetic tree
and prove that tampered predictions, summaries or selection files are rejected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_DIR / "outputs" / "stage2_model_classification_20260919"
OLD_INDEX = PROJECT_DIR / "outputs" / "window_index.csv"
HELDOUT_INDEX = (
    PROJECT_DIR / "outputs" / "heldout_preparation_20260917" / "heldout_window_index.csv"
)
PRESERVED_INPUTS = PROJECT_DIR.parent / "work" / "codex-preserved-inputs-20260917.json"
ORIGINAL_MAT_DIR = Path(r"C:\Users\Administrator\Documents\DroneRFa数据集")

OLD_INDEX_SHA256 = "4e77c7f186c1aae8e04f2e90f2769c0ed399cc94f40b64f99237b65c057b85ac"
HELDOUT_INDEX_SHA256 = "6fe32faeb5d583a359d9e8aa17caa52a7c4567d545272ded25266767cd2cffcf"

NFFT_GRID = (512, 1024)
TRAINING_SEEDS = (2026, 2027, 2028)
CONDITIONS = ("clean", "-10dB", "-12dB", "-15dB")
TEST_CONDITIONS = ("clean",)
HELDOUT_CONDITIONS_USED = 4
HELDOUT_WINDOWS = 1260
TEST_WINDOWS = 300
EXPECTED_HELDOUT_ROWS = len(NFFT_GRID) * len(TRAINING_SEEDS) * HELDOUT_CONDITIONS_USED * HELDOUT_WINDOWS
EXPECTED_TEST_ROWS = len(NFFT_GRID) * len(TRAINING_SEEDS) * TEST_WINDOWS
CODE_TO_LABEL = {"0010": 0, "0101": 1}
LABEL_TO_NAME = {0: "DJI Phantom 4 Pro", 1: "DJI Air 2S"}
PREDICTION_RULE = "logit_margin > 0"
TOLERANCE = 1e-12
SELECTION_FORBIDDEN_TOKENS = ("heldout", "test", "F:\\DroneRFa")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

METRIC_FIELDS = (
    "file_equal_phantom_recall",
    "file_equal_air2s_recall",
    "file_equal_balanced_accuracy",
    "accuracy",
    "balanced_accuracy",
    "precision_air2s",
    "precision_phantom",
    "phantom_recall",
    "air2s_recall",
    "f1_air2s_binary",
    "f1_macro",
)


# ---------------------------------------------------------------------------
# small independent helpers
# ---------------------------------------------------------------------------
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


def close(left: float, right: float, tolerance: float = TOLERANCE) -> bool:
    if left is None or right is None:
        return False
    if np.isnan(left) and np.isnan(right):
        return True
    return abs(float(left) - float(right)) <= tolerance


CONDITION_VALUES: dict[str, float | None] = {
    "clean": None,
    "-10dB": -10.0,
    "-12dB": -12.0,
    "-15dB": -15.0,
}
TRAIN_SNRS = "clean,10,5,0,-5,-10,-12"
TRAIN_WINDOWS = 620
EPOCHS = 5
VALIDATION_SEED = 12026
HELDOUT_NOISE_SEED = 22026
# The old-test split is evaluated clean, so this seed only documents the inert value
# the run recorded (2026 + 20_000).
TEST_NOISE_SEED = 22026


def derive_initial_state_hash(seed: int) -> str:
    """Re-derive the ResNet18 initialisation of a training seed from the frozen model."""
    import random

    import torch

    from model import ResNet18

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = ResNet18(input_channels=1, num_classes=2)
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    del model
    return digest.hexdigest()


def derive_epoch_order_hashes(seed: int) -> list[str]:
    """Re-derive the paired epoch window orders of one training seed."""
    import torch

    orders = []
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(
            TRAIN_WINDOWS, generator=torch.Generator().manual_seed(seed + epoch)
        ).tolist()
        orders.append(
            hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest()
        )
    return orders


def recompute_pairing_digest(
    index_path: Path, split: str, condition: str, noise_seed: int
) -> dict[str, Any]:
    """Re-derive the added-AWGN probe digest of one (split, condition)."""
    from dronerfa_data import add_complex_awgn

    rows = index_rows(index_path, split)
    snr = CONDITION_VALUES[condition]
    digest = hashlib.sha256()
    seeds = []
    for position in sorted({0, len(rows) // 2, len(rows) - 1}):
        rng_seed = noise_seed + position
        seeds.append(rng_seed)
        rng = np.random.default_rng(rng_seed)
        if snr is None:
            payload = np.asarray(
                rng.standard_normal(8, dtype=np.float32), dtype=np.float32
            ).tobytes()
        else:
            probe = np.zeros(8, dtype=np.complex64)
            probe[0] = 1.0
            payload = np.asarray(add_complex_awgn(probe, snr, rng), dtype=np.complex64).tobytes()
        digest.update(payload)
    return {"digest_sha256": digest.hexdigest(), "rng_seeds": seeds}


def index_rows(index_path: Path, split: str) -> list[dict[str, str]]:
    """Stage-2 view of a split, recomputed here from the frozen index text."""
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    selected = []
    for row in rows:
        code = row["target_code"]
        if code not in CODE_TO_LABEL:
            continue
        if int(row["label"]) != 1:
            raise ValueError(f"{row['filename']} carries a non-drone label for a drone code")
        selected.append({**row, "label": CODE_TO_LABEL[code]})
    return selected


def pooled_from_rows(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Pooled window metrics; positive class is Air 2S (label 1)."""
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    prediction = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
    tp = int(np.sum((y == 1) & (prediction == 1)))
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    phantom_recall = tn / max(tn + fp, 1)
    air2s_recall = tp / max(tp + fn, 1)
    precision_air2s = tp / max(tp + fp, 1)
    precision_phantom = tn / max(tn + fn, 1)
    f1_air2s = 2 * precision_air2s * air2s_recall / max(precision_air2s + air2s_recall, 1e-12)
    f1_phantom = 2 * precision_phantom * phantom_recall / max(precision_phantom + phantom_recall, 1e-12)
    return {
        "windows": float(len(y)),
        "accuracy": (tp + tn) / max(len(y), 1),
        "balanced_accuracy": 0.5 * (phantom_recall + air2s_recall),
        "precision_air2s": precision_air2s,
        "precision_phantom": precision_phantom,
        "phantom_recall": phantom_recall,
        "air2s_recall": air2s_recall,
        "f1_air2s_binary": f1_air2s,
        "f1_macro": 0.5 * (f1_phantom + f1_air2s),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def file_equal_from_rows(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Per-file rates averaged inside a class, then the two classes averaged."""
    per_file: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = per_file.setdefault(
            str(row["filename"]), {"label": int(row["label"]), "correct": 0, "windows": 0}
        )
        if entry["label"] != int(row["label"]):
            raise ValueError(f"file {row['filename']} mixes two classes")
        entry["correct"] += int(int(row["prediction"]) == int(row["label"]))
        entry["windows"] += 1
    rates = {0: [], 1: []}
    for entry in per_file.values():
        rates[entry["label"]].append(entry["correct"] / max(entry["windows"], 1))
    phantom = float(np.mean(rates[0])) if rates[0] else float("nan")
    air2s = float(np.mean(rates[1])) if rates[1] else float("nan")
    return {
        "files_phantom": float(len(rates[0])),
        "files_air2s": float(len(rates[1])),
        "file_equal_phantom_recall": phantom,
        "file_equal_air2s_recall": air2s,
        "file_equal_balanced_accuracy": 0.5 * (phantom + air2s),
    }


def sample_std(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def read_png_dimensions(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    if not payload.startswith(PNG_SIGNATURE):
        raise ValueError(f"{path.name} is not a PNG file")
    if payload[12:16] != b"IHDR":
        raise ValueError(f"{path.name} has no IHDR chunk")
    width = int.from_bytes(payload[16:20], "big")
    height = int.from_bytes(payload[20:24], "big")
    return width, height


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_frozen_inputs(failures: list[str], record: dict[str, Any]) -> None:
    old_hash = sha256_file(OLD_INDEX)
    heldout_hash = sha256_file(HELDOUT_INDEX)
    record["old_index"] = {
        "sha256": old_hash,
        "expected": OLD_INDEX_SHA256,
        "match": old_hash == OLD_INDEX_SHA256,
    }
    record["heldout_index"] = {
        "sha256": heldout_hash,
        "expected": HELDOUT_INDEX_SHA256,
        "match": heldout_hash == HELDOUT_INDEX_SHA256,
    }
    if old_hash != OLD_INDEX_SHA256:
        failures.append(f"old index changed: {old_hash}")
    if heldout_hash != HELDOUT_INDEX_SHA256:
        failures.append(f"heldout index changed: {heldout_hash}")

    preserved = read_json(PRESERVED_INPUTS)
    entries = []
    for entry in preserved["files"]:
        path = Path(entry["path"])
        match = bool(
            path.is_file()
            and path.stat().st_size == entry["bytes"]
            and sha256_file(path) == entry["sha256"]
        )
        entries.append({"path": entry["path"], "match": match})
        if not match:
            failures.append(f"preserved input changed: {entry['path']}")
    record["preserved_inputs"] = {
        "checked": len(entries),
        "all_match": all(item["match"] for item in entries),
        "files": entries,
    }

    mat_entries = []
    for entry in preserved.get("original_mat_sizes", []):
        path = ORIGINAL_MAT_DIR / entry["name"]
        match = bool(path.is_file() and path.stat().st_size == entry["bytes"])
        mat_entries.append({"name": entry["name"], "match": match})
        if not match:
            failures.append(f"original MAT size changed: {path}")
    record["original_mat_sizes"] = {
        "checked": len(mat_entries),
        "all_match": all(item["match"] for item in mat_entries),
        "full_sha256_recomputed": False,
    }


def check_runs(
    root: Path,
    failures: list[str],
    record: dict[str, Any],
    nfft_values: Sequence[int],
    seeds: Sequence[int],
    strict: bool,
) -> dict[tuple[int, int], dict[str, Any]]:
    runs: dict[tuple[int, int], dict[str, Any]] = {}
    for n_fft in nfft_values:
        for seed in seeds:
            run_dir = root / f"nfft{n_fft}" / f"seed_{seed}"
            entry: dict[str, Any] = {"run_directory": str(run_dir)}
            config_path = run_dir / "run_config.json"
            selection_path = run_dir / "selection.json"
            history_path = run_dir / "training_history.csv"
            metrics_path = run_dir / "validation_metrics.csv"
            checkpoint_path = run_dir / "best_model.pt"
            missing = [
                path.name
                for path in (config_path, selection_path, history_path, metrics_path, checkpoint_path)
                if not path.is_file()
            ]
            if missing:
                failures.append(f"{run_dir}: missing {missing}")
                record[f"nfft{n_fft}_seed{seed}"] = entry
                continue
            config = read_json(config_path)
            selection = read_json(selection_path)
            history = read_csv_rows(history_path)
            metrics = read_csv_rows(metrics_path)
            try:
                import torch

                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                checkpoint_epoch: int | None = int(checkpoint["epoch"])
                del checkpoint
            except Exception as error:  # pragma: no cover - corrupt checkpoint
                failures.append(f"{run_dir}: best_model.pt could not be read ({error!r})")
                checkpoint_epoch = None
            entry["checkpoint_epoch"] = checkpoint_epoch
            entry["status"] = config.get("status")
            entry["config"] = {
                key: config.get(key)
                for key in (
                    "n_fft",
                    "hop_length",
                    "image_size",
                    "training_seed",
                    "epochs",
                    "batch_size",
                    "train_snrs",
                    "validation_seed",
                    "validation_conditions",
                    "optimizer",
                    "amp",
                    "initial_model_sha256",
                    "epoch_order_sha256",
                    "selected_epoch",
                    "label_mapping",
                )
            }
            if config.get("status") != "completed":
                failures.append(f"{run_dir}: status {config.get('status')!r} is not completed")
            for key, expected in (("n_fft", n_fft), ("training_seed", seed)):
                if config.get(key) != expected:
                    failures.append(f"{run_dir}: config.{key}={config.get(key)!r} != {expected!r}")
            if config.get("hop_length") != n_fft // 2:
                failures.append(f"{run_dir}: hop_length is not n_fft // 2")
            if config.get("image_size") != 256:
                failures.append(f"{run_dir}: image_size is not 256")
            optimizer = config.get("optimizer", {})
            if optimizer.get("type") != "Adam" or optimizer.get("learning_rate") != 1e-3:
                failures.append(f"{run_dir}: optimizer is not Adam lr 1e-3")
            if optimizer.get("weight_decay") not in (0.0, 0):
                failures.append(f"{run_dir}: weight_decay is not 0")
            if config.get("train_snrs") != "clean,10,5,0,-5,-10,-12":
                failures.append(f"{run_dir}: train_snrs is not the frozen A_original list")
            if list(config.get("validation_conditions", [])) != list(CONDITIONS):
                failures.append(f"{run_dir}: validation conditions differ from the frozen four")
            if config.get("validation_seed") != 12026:
                failures.append(f"{run_dir}: validation seed is not 12026")
            if config.get("label_mapping") != {"0": LABEL_TO_NAME[0], "1": LABEL_TO_NAME[1]}:
                failures.append(f"{run_dir}: label mapping is not Phantom=0 / Air2S=1")

            # epoch selection rule, re-derived from the history alone
            by_epoch: dict[int, list[float]] = {}
            for row in metrics:
                by_epoch.setdefault(int(row["epoch"]), []).append(
                    float(row["file_equal_balanced_accuracy"])
                )
            means = {epoch: float(np.mean(values)) for epoch, values in by_epoch.items()}
            if not means:
                failures.append(f"{run_dir}: validation_metrics.csv has no rows")
            else:
                best = max(means, key=lambda epoch: (means[epoch], -epoch))
                entry["selected_epoch_by_rule"] = best
                entry["selection_mean_by_epoch"] = means
                if int(selection.get("selected_epoch", -1)) != best:
                    failures.append(
                        f"{run_dir}: selection.json selected_epoch "
                        f"{selection.get('selected_epoch')} != rule {best}"
                    )
                if config.get("selected_epoch") != best:
                    failures.append(f"{run_dir}: config.selected_epoch != rule {best}")
                if checkpoint_epoch is not None and checkpoint_epoch != best:
                    failures.append(
                        f"{run_dir}: best_model.pt holds epoch {checkpoint_epoch}, rule selects {best}"
                    )
                for row in history:
                    epoch = int(row["epoch"])
                    if epoch not in means:
                        failures.append(f"{run_dir}: history epoch {epoch} has no validation rows")
                        continue
                    if not close(float(row["mean_file_equal_balanced_accuracy"]), means[epoch]):
                        failures.append(
                            f"{run_dir}: history epoch {epoch} mean "
                            f"{row['mean_file_equal_balanced_accuracy']} != {means[epoch]}"
                        )
                if len(history) != int(config.get("epochs", 0)):
                    failures.append(f"{run_dir}: history has {len(history)} epochs")
                if len(by_epoch) != len(history):
                    failures.append(f"{run_dir}: validation metrics and history cover different epochs")
                entry["validation_rows"] = len(metrics)
                entry["windows_per_condition"] = sorted(
                    {int(float(row["windows"])) for row in metrics}
                )
                if strict and entry["windows_per_condition"] != [290]:
                    failures.append(
                        f"{run_dir}: validation windows {entry['windows_per_condition']} != [290]"
                    )
                if len(metrics) != len(CONDITIONS) * len(history):
                    failures.append(f"{run_dir}: validation_metrics.csv row count is inconsistent")
            record[f"nfft{n_fft}_seed{seed}"] = entry
            runs[(n_fft, seed)] = entry

    # fairness: same initial weights and same epoch orders across the two n_fft
    for seed in seeds:
        entries = [runs.get((n_fft, seed)) for n_fft in nfft_values]
        if any(entry is None for entry in entries):
            continue
        initial = {entry["config"].get("initial_model_sha256") for entry in entries}
        orders = {json.dumps(entry["config"].get("epoch_order_sha256")) for entry in entries}
        record.setdefault("fairness", {})[f"seed_{seed}"] = {
            "initial_model_sha256": sorted(value for value in initial if value),
            "initial_identical": len(initial) == 1 and None not in initial,
            "epoch_order_identical": len(orders) == 1,
        }
        if len(initial) != 1 or None in initial:
            failures.append(f"seed {seed}: the two n_fft runs have different initial weights")
        if len(orders) != 1:
            failures.append(f"seed {seed}: the two n_fft runs have different epoch orders")
        expected_orders = len(entries[0]["config"].get("epoch_order_sha256") or [])
        if expected_orders != int(entries[0]["config"].get("epochs", 0)):
            failures.append(f"seed {seed}: epoch order hashes do not cover every epoch")
        if strict:
            # Re-derive both from the frozen model definition and the frozen train size.
            rederived_initial = derive_initial_state_hash(seed)
            record["fairness"][f"seed_{seed}"]["initial_model_sha256_rederived"] = rederived_initial
            if rederived_initial not in initial:
                failures.append(
                    f"seed {seed}: the recorded initial weights do not reproduce from the seed"
                )
            rederived_orders = derive_epoch_order_hashes(seed)
            record["fairness"][f"seed_{seed}"]["epoch_order_sha256_rederived"] = rederived_orders
            if json.dumps(rederived_orders) not in orders:
                failures.append(
                    f"seed {seed}: the recorded epoch orders do not reproduce from the seed"
                )
    return runs


def check_selection(
    root: Path,
    failures: list[str],
    record: dict[str, Any],
    runs: dict[tuple[int, int], dict[str, Any]],
    strict: bool,
) -> dict[str, Any]:
    selection_path = root / "validation_nfft_selection.json"
    lock_path = root / "selection_lock.json"
    if not selection_path.is_file() or not lock_path.is_file():
        failures.append("validation_nfft_selection.json / selection_lock.json missing")
        return {}
    text = selection_path.read_text(encoding="utf-8")
    lowered = text.lower()
    tokens = [token for token in SELECTION_FORBIDDEN_TOKENS if token.lower() in lowered]
    record["forbidden_tokens"] = tokens
    if tokens:
        failures.append(f"the configuration choice mentions {tokens}")
    selection = json.loads(text)
    lock = read_json(lock_path)
    digest = sha256_file(selection_path)
    record["sha256"] = digest
    record["lock_sha256"] = lock.get("selection_sha256")
    record["selected_n_fft"] = selection.get("selected_n_fft")
    if lock.get("selection_sha256") != digest:
        failures.append("selection_lock.json does not hold the current selection hash")

    # re-derive the rule from the per-run validation evidence
    per_nfft: dict[int, list[float]] = {}
    for row in selection.get("per_model", []):
        key = (int(row["n_fft"]), int(row["training_seed"]))
        entry = runs.get(key)
        if entry is None:
            failures.append(f"the configuration choice cites a missing run {key}")
            continue
        means = entry.get("selection_mean_by_epoch", {})
        selected = int(row["selected_epoch"])
        if selected not in means:
            failures.append(f"the configuration choice cites epoch {selected} of {key}")
            continue
        if not close(float(row["four_condition_mean_file_equal_balacc"]), means[selected]):
            failures.append(f"the configuration choice misreports the validation score of {key}")
        per_nfft.setdefault(key[0], []).append(means[selected])
    expected_best = None
    if per_nfft and all(per_nfft.get(n_fft) for n_fft in NFFT_GRID):
        averages = {n_fft: float(np.mean(values)) for n_fft, values in per_nfft.items()}
        expected_best = max(averages, key=lambda n_fft: (averages[n_fft], -n_fft))
        record["recomputed_means"] = averages
        if selection.get("selected_n_fft") != expected_best:
            failures.append(
                f"selected n_fft {selection.get('selected_n_fft')} != recomputed {expected_best}"
            )
        if strict and len(per_nfft[min(per_nfft)]) != len(TRAINING_SEEDS):
            failures.append("the configuration choice does not use all three training seeds")
    record["seeds_used"] = sorted({int(row["training_seed"]) for row in selection.get("per_model", [])})
    return selection


def check_generation_record(
    root: Path, failures: list[str], record: dict[str, Any], selection_digest: str | None
) -> None:
    path = root / "heldout_generation_record.json"
    if not path.is_file():
        failures.append("heldout_generation_record.json missing")
        return
    payload = read_json(path)
    current = sha256_file(root / "validation_nfft_selection.json")
    record.update(
        {
            "before": payload.get("selection_sha256_before_evaluation"),
            "after": payload.get("selection_sha256_after_evaluation"),
            "unchanged": payload.get("selection_unchanged"),
            "current": current,
        }
    )
    if not payload.get("selection_unchanged"):
        failures.append("the generation record reports a changed configuration choice")
    for key in ("selection_sha256_before_evaluation", "selection_sha256_after_evaluation"):
        if payload.get(key) != current:
            failures.append(f"the generation record {key} does not match the current file")
    if selection_digest is not None and current != selection_digest:
        failures.append("the configuration choice changed after it was locked")


def check_prediction_table(
    path: Path,
    split: str,
    index_path: Path,
    failures: list[str],
    record: dict[str, Any],
    expected_windows: int,
    expected_conditions: Sequence[str],
    expected_rows: int,
    nfft_values: Sequence[int],
    seeds: Sequence[int],
    selection_by_model: dict[tuple[int, int], dict[str, Any]],
    strict: bool,
) -> list[dict[str, Any]]:
    if not path.is_file():
        failures.append(f"{path.name} missing")
        return []
    rows = read_csv_rows(path)
    record["rows"] = len(rows)
    record["expected_rows"] = expected_rows
    if strict and len(rows) != expected_rows:
        failures.append(f"{path.name}: {len(rows)} rows != {expected_rows}")

    reference = index_rows(index_path, split)
    reference_windows = len(reference)
    record["index_windows"] = reference_windows
    if strict and reference_windows != expected_windows:
        failures.append(
            f"{path.name}: the frozen index has {reference_windows} windows != {expected_windows}"
        )

    keys: dict[tuple[int, int, str], dict[int, dict[str, str]]] = {}
    identity_errors = 0
    for row in rows:
        key = (int(row["n_fft"]), int(row["training_seed"]), row["condition"])
        position = int(row["index_position"])
        group = keys.setdefault(key, {})
        if position in group:
            failures.append(f"{path.name}: duplicate key {key} position {position}")
            continue
        group[position] = row
        if position < reference_windows:
            source = reference[position]
            if row["filename"] != source["filename"] or int(row["start"]) != int(source["start"]):
                identity_errors += 1
            if int(row["length"]) != int(source["length"]):
                identity_errors += 1
            if strict and int(row["length"]) != 1_000_000:
                identity_errors += 1
            if row["target_code"] != source["target_code"]:
                identity_errors += 1
            if int(row["label"]) != CODE_TO_LABEL[source["target_code"]]:
                identity_errors += 1
            if row["class_name"] != LABEL_TO_NAME[CODE_TO_LABEL[source["target_code"]]]:
                identity_errors += 1
        if int(row["hop_length"]) != int(row["n_fft"]) // 2:
            identity_errors += 1
        margin = float(row["logit_margin"])
        phantom = float(row["logit_phantom"])
        air2s = float(row["logit_air2s"])
        if not np.isfinite([phantom, air2s, margin]).all():
            identity_errors += 1
        if abs((air2s - phantom) - margin) > 1e-6:
            identity_errors += 1
        if int(row["prediction"]) != int(margin > 0.0):
            identity_errors += 1
        score = float(row["air2s_score"])
        if score < 0.0 or score > 1.0:
            identity_errors += 1
    record["identity_or_rule_errors"] = identity_errors
    if identity_errors:
        failures.append(f"{path.name}: {identity_errors} identity/rule/logit violations")

    expected_keys = {
        (n_fft, seed, condition)
        for n_fft in nfft_values
        for seed in seeds
        for condition in expected_conditions
    }
    record["groups"] = len(keys)
    record["expected_groups"] = len(expected_keys)
    if strict and set(keys) != expected_keys:
        failures.append(
            f"{path.name}: groups differ from the frozen grid "
            f"(missing {sorted(expected_keys - set(keys))[:4]}, "
            f"extra {sorted(set(keys) - expected_keys)[:4]})"
        )
    coverage_errors = 0
    for key, group in keys.items():
        positions = sorted(group)
        if positions != list(range(reference_windows)):
            coverage_errors += 1
            continue
        epoch = int(group[0]["epoch"])
        expected_epoch = selection_by_model.get((key[0], key[1]), {}).get("selected_epoch")
        if expected_epoch is not None and epoch != expected_epoch:
            failures.append(f"{path.name}: {key} uses epoch {epoch} != selected {expected_epoch}")
        if group[0]["split"] != split:
            failures.append(f"{path.name}: {key} has split {group[0]['split']!r}")
        if group[0]["model_id"] != f"nfft{key[0]}_seed{key[1]}":
            failures.append(f"{path.name}: {key} has model_id {group[0]['model_id']!r}")
    record["coverage_errors"] = coverage_errors
    if coverage_errors:
        failures.append(f"{path.name}: {coverage_errors} groups do not cover 0..{reference_windows - 1}")

    records = []
    for key, group in keys.items():
        ordered = [group[position] for position in sorted(group)]
        for row in ordered:
            records.append(
                {
                    "n_fft": key[0],
                    "training_seed": key[1],
                    "condition": key[2],
                    "filename": row["filename"],
                    "label": int(row["label"]),
                    "prediction": int(row["prediction"]),
                }
            )
    return records


def check_metric_tables(
    root: Path,
    failures: list[str],
    record: dict[str, Any],
    heldout_records: Sequence[dict[str, Any]],
    split: str,
    expected_conditions: Sequence[str],
    strict: bool,
    table_name: str = "model_condition_metrics.csv",
    per_file_required: bool = True,
) -> dict[tuple[int, int, str], dict[str, float]]:
    path = root / table_name
    if not path.is_file():
        failures.append(f"{table_name} missing")
        return {}
    rows = read_csv_rows(path)
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in heldout_records:
        if split == "test" and row["condition"] not in expected_conditions:
            continue
        grouped.setdefault(
            (row["n_fft"], row["training_seed"], row["condition"]), []
        ).append(row)
    scoped = [
        row
        for row in rows
        if row["split"] == split
        and row["condition"] in set(expected_conditions)
    ]
    record["table_rows"] = len(scoped)
    record["recomputed_groups"] = len(grouped)
    if strict and len(scoped) != len(grouped):
        failures.append(
            f"{table_name} has {len(scoped)} {split} rows != {len(grouped)} groups"
        )
    metrics: dict[tuple[int, int, str], dict[str, float]] = {}
    for row in scoped:
        key = (int(row["n_fft"]), int(row["training_seed"]), row["condition"])
        group = grouped.get(key)
        if group is None:
            failures.append(f"{table_name} reports an unknown group {key}")
            continue
        pooled = pooled_from_rows(group)
        primary = file_equal_from_rows(group)
        expected = {**pooled, **primary}
        metrics[key] = expected
        for field in METRIC_FIELDS + ("tn", "fp", "fn", "tp", "windows"):
            if field not in row:
                failures.append(f"{table_name} lacks column {field}")
                continue
            if not close(float(row[field]), expected[field]):
                failures.append(
                    f"{key} {field}: reported {row[field]} != recomputed {expected[field]}"
                )
        if int(pooled["tn"] + pooled["fp"] + pooled["fn"] + pooled["tp"]) != len(group):
            failures.append(f"{key}: confusion matrix does not conserve the window count")

    # per-file table
    per_file_path = root / "per_file_metrics.csv"
    if not per_file_path.is_file():
        if per_file_required:
            failures.append("per_file_metrics.csv missing")
    else:
        per_file_rows = [
            row
            for row in read_csv_rows(per_file_path)
            if row["split"] == split and row["condition"] in set(expected_conditions)
        ]
        if per_file_required and not per_file_rows:
            failures.append(
                f"per_file_metrics.csv publishes no {split} rows although they are required"
            )
        expected_files: dict[tuple[int, int, str, str], dict[str, Any]] = {}
        for key, group in grouped.items():
            by_file: dict[str, list[dict[str, Any]]] = {}
            for item in group:
                by_file.setdefault(item["filename"], []).append(item)
            for filename, items in by_file.items():
                correct = sum(
                    int(int(item["prediction"]) == int(item["label"])) for item in items
                )
                expected_files[(key[0], key[1], key[2], filename)] = {
                    "windows": len(items),
                    "correct": correct,
                    "correct_rate": correct / len(items),
                    "class_label": int(items[0]["label"]),
                }
        record["per_file_rows"] = len(per_file_rows)
        record["per_file_expected"] = len(expected_files)
        if strict and per_file_required and len(per_file_rows) != len(expected_files):
            failures.append(
                f"per_file_metrics.csv has {len(per_file_rows)} {split} rows "
                f"!= {len(expected_files)}"
            )
        for row in per_file_rows:
            key = (
                int(row["n_fft"]),
                int(row["training_seed"]),
                row["condition"],
                row["filename"],
            )
            expected = expected_files.get(key)
            if expected is None:
                failures.append(f"per_file_metrics.csv reports an unknown row {key}")
                continue
            if int(row["windows"]) != expected["windows"]:
                failures.append(f"per_file_metrics.csv {key}: window count differs")
            if int(row["correct"]) != expected["correct"]:
                failures.append(f"per_file_metrics.csv {key}: correct count differs")
            if not close(float(row["correct_rate"]), expected["correct_rate"]):
                failures.append(f"per_file_metrics.csv {key}: correct rate differs")
            if int(row["class_label"]) != expected["class_label"]:
                failures.append(f"per_file_metrics.csv {key}: class label differs")

    # cross-seed summary and paired differences
    summary_path = root / "nfft_summary.csv"
    paired_path = root / "paired_nfft_comparison.csv"
    if not summary_path.is_file() or not paired_path.is_file():
        failures.append("nfft_summary.csv / paired_nfft_comparison.csv missing")
        return metrics
    summary_rows = [
        row
        for row in read_csv_rows(summary_path)
        if row["split"] == split and row["condition"] in set(expected_conditions)
    ]
    composites: dict[tuple[int, int], float] = {}
    composites_pooled: dict[tuple[int, int], float] = {}
    for key, value in metrics.items():
        composites.setdefault((key[0], key[1]), []).append(
            value["file_equal_balanced_accuracy"]
        )  # type: ignore[arg-type]
        composites_pooled.setdefault((key[0], key[1]), []).append(value["balanced_accuracy"])  # type: ignore[arg-type]
    composites = {key: float(np.mean(value)) for key, value in composites.items()}  # type: ignore[arg-type]
    composites_pooled = {
        key: float(np.mean(value)) for key, value in composites_pooled.items()  # type: ignore[arg-type]
    }
    record["summary_rows"] = len(summary_rows)
    checked_groups = 0
    for row in summary_rows:
        n_fft = int(row["n_fft"])
        condition = row["condition"]
        if condition in expected_conditions:
            values = [
                metrics[(n_fft, seed, condition)]["file_equal_balanced_accuracy"]
                for seed in TRAINING_SEEDS
                if (n_fft, seed, condition) in metrics
            ]
        else:
            values = [
                composites[(n_fft, seed)]
                for seed in TRAINING_SEEDS
                if (n_fft, seed) in composites
            ]
        if not values:
            failures.append(f"nfft_summary.csv row {n_fft}/{condition} has no source group")
            continue
        checked_groups += 1
        if not close(float(row["file_equal_balacc_mean"]), float(np.mean(values))):
            failures.append(f"nfft_summary.csv {n_fft}/{condition}: mean differs")
        if not close(float(row["file_equal_balacc_std"]), sample_std(values)):
            failures.append(f"nfft_summary.csv {n_fft}/{condition}: std differs")
        if int(row["seeds"]) != len(values):
            failures.append(f"nfft_summary.csv {n_fft}/{condition}: seed count differs")
    record["summary_groups_checked"] = checked_groups

    paired_rows = [
        row
        for row in read_csv_rows(paired_path)
        if row["split"] == split and row["condition"] in set(expected_conditions)
    ]
    record["paired_rows"] = len(paired_rows)
    for row in paired_rows:
        n_fft = int(row["n_fft"])
        baseline = int(row["baseline_n_fft"])
        condition = row["condition"]
        differences = []
        for seed in TRAINING_SEEDS:
            if condition in expected_conditions:
                if (n_fft, seed, condition) not in metrics or (
                    baseline,
                    seed,
                    condition,
                ) not in metrics:
                    continue
                value = metrics[(n_fft, seed, condition)]["file_equal_balanced_accuracy"]
                base = metrics[(baseline, seed, condition)]["file_equal_balanced_accuracy"]
            else:
                if (n_fft, seed) not in composites or (baseline, seed) not in composites:
                    continue
                value = composites[(n_fft, seed)]
                base = composites[(baseline, seed)]
            differences.append(value - base)
        if not differences:
            failures.append(f"paired_nfft_comparison.csv row {n_fft}/{condition} has no source")
            continue
        if not close(float(row["mean_difference"]), float(np.mean(differences))):
            failures.append(f"paired_nfft_comparison.csv {n_fft}/{condition}: mean differs")
        if not close(float(row["std_difference"]), sample_std(differences)):
            failures.append(f"paired_nfft_comparison.csv {n_fft}/{condition}: std differs")
        if int(row["seeds"]) != len(differences):
            failures.append(f"paired_nfft_comparison.csv {n_fft}/{condition}: seed count differs")
    return metrics


def check_cache_manifest(
    root: Path,
    failures: list[str],
    record: dict[str, Any],
    nfft_values: Sequence[int],
    split_windows: dict[str, int],
    strict: bool,
) -> None:
    path = root / "prediction_cache_manifest.json"
    if not path.is_file():
        failures.append("prediction_cache_manifest.json missing")
        return
    payload = read_json(path)
    if payload.get("disk_feature_cache_written"):
        failures.append("a disk feature cache was reported, expected in-memory only")
    for key, split in (("heldout", "heldout"), ("test_clean", "test")):
        entries = {int(row["n_fft"]): row for row in payload.get(key, [])}
        record[f"{key}_nfft_entries"] = sorted(entries)
        if strict and sorted(entries) != sorted(nfft_values):
            failures.append(f"prediction_cache_manifest.json {key} does not cover both grids")
        probes = {row["input_probe_sha256"] for row in payload.get(key, [])}
        per_condition: dict[str, set[str]] = {}
        for row in payload.get(key, []):
            per_condition.setdefault(row["condition"], set()).add(row["input_probe_sha256"])
        record[f"{key}_probe_identical"] = all(
            len(digests) == 1 for digests in per_condition.values()
        )
        if not record[f"{key}_probe_identical"]:
            for condition, digests in per_condition.items():
                if len(digests) != 1:
                    failures.append(
                        f"{key}/{condition}: the same noise produced different IQ for the two "
                        f"n_fft grids"
                    )
        record[f"{key}_probe_digests"] = len(probes)
        for condition, digests in per_condition.items():
            if len(digests) != 1:
                failures.append(f"{key}/{condition}: input probe digest differs across n_fft")
            windows = {int(row["windows"]) for row in payload.get(key, []) if row["condition"] == condition}
            if strict and windows != {split_windows[split]}:
                failures.append(
                    f"{key}/{condition}: windows {sorted(windows)} != {split_windows[split]}"
                )
            for row in payload.get(key, []):
                if row["condition"] != condition:
                    continue
                if int(row["window_digest_bytes"]) != 32 * int(row["windows"]):
                    failures.append(
                        f"{key}/{condition}/nfft{row['n_fft']}: per-window digests are not "
                        f"32 bytes per window"
                    )
                if int(row["shape"][0]) != int(row["windows"]):
                    failures.append(f"{key}/{condition}: cached feature shape disagrees")
        if not payload.get("truncated_debug_run"):
            for row in payload.get(key, []):
                digest_path = root / "feature_digests" / (
                    f"window_digests_{'test' if split == 'test' else 'heldout'}_"
                    f"{row['condition']}_nfft{row['n_fft']}.bin"
                )
                if not digest_path.is_file():
                    failures.append(f"missing window digest file {digest_path.name}")
                    continue
                if sha256_file(digest_path) != row["window_digests_concat_sha256"]:
                    failures.append(f"window digest file {digest_path.name} does not match")
    pairing = payload.get("noise_pairing", {})
    for key, expected_split, expected_conditions, index_path, noise_seed in (
        ("heldout", "heldout", CONDITIONS, HELDOUT_INDEX, HELDOUT_NOISE_SEED),
        ("test_clean", "test", TEST_CONDITIONS, OLD_INDEX, TEST_NOISE_SEED),
    ):
        rows = pairing.get(key, [])
        record[f"{key}_pairing_conditions"] = [row["condition"] for row in rows]
        if strict and len(rows) != len(expected_conditions):
            failures.append(f"prediction_cache_manifest.json {key} pairing rows are incomplete")
        for row in rows:
            if row["split"] != expected_split:
                failures.append(f"noise pairing row for {key} reports split {row['split']!r}")
            if not row.get("digest_sha256"):
                failures.append(f"noise pairing row for {key}/{row.get('condition')} has no digest")
            if "stage2_position" not in str(row.get("rng_rule", "")):
                failures.append(
                    f"noise pairing row for {key}/{row.get('condition')} does not document the "
                    f"stage-2 RNG rule"
                )
            if strict and row["condition"] in expected_conditions:
                rederived = recompute_pairing_digest(
                    index_path, expected_split, row["condition"], noise_seed
                )
                if rederived["digest_sha256"] != row["digest_sha256"]:
                    failures.append(
                        f"noise pairing digest for {key}/{row['condition']} does not reproduce "
                        f"from the frozen index"
                    )
                if [probe["rng_seed"] for probe in row["probe_positions"]] != rederived["rng_seeds"]:
                    failures.append(
                        f"noise pairing RNG seeds for {key}/{row['condition']} do not reproduce"
                    )


def check_frontend_check(root: Path, failures: list[str], record: dict[str, Any]) -> None:
    path = root / "frontend_check.json"
    if not path.is_file():
        failures.append("frontend_check.json missing")
        return
    payload = read_json(path)
    for n_fft in NFFT_GRID:
        entry = payload.get(str(n_fft))
        if entry is None:
            failures.append(f"frontend_check.json lacks n_fft={n_fft}")
            continue
        record[f"nfft{n_fft}"] = {
            "output_shape": entry["output_shape"],
            "hop_length": entry["hop_length"],
            "finite": entry["finite"],
        }
        if entry["output_shape"] != [2, 1, 256, 256]:
            failures.append(f"frontend n_fft={n_fft} does not emit 1x256x256 per sample")
        if entry["hop_length"] != n_fft // 2:
            failures.append(f"frontend n_fft={n_fft} does not use hop = n_fft // 2")
        if not entry["finite"]:
            failures.append(f"frontend n_fft={n_fft} produced non-finite features")


def check_figures(
    root: Path, failures: list[str], record: dict[str, Any], strict: bool
) -> None:
    evidence_path = root / "figure_evidence.json"
    if not evidence_path.is_file():
        failures.append("figure_evidence.json missing")
        return
    evidence = read_json(evidence_path)
    if evidence.get("per_window_png_written"):
        failures.append("per-window PNGs were written, which the task forbids")
    record["per_window_png_written"] = evidence.get("per_window_png_written")
    record["rule"] = evidence.get("rule")
    classes = evidence.get("windows", {})
    if strict and set(classes) != set(CODE_TO_LABEL):
        failures.append("figure_evidence.json does not cover both machine classes")
    for code, entry in classes.items():
        if code not in CODE_TO_LABEL:
            failures.append(f"figure_evidence.json has an unexpected class {code}")
            continue
        if int(entry["class_label"]) != CODE_TO_LABEL[code]:
            failures.append(f"figure_evidence.json class {code} carries the wrong label")
        figure_path = root / entry["figure"]
        if not figure_path.is_file():
            failures.append(f"missing figure {entry['figure']}")
            continue
        try:
            width, height = read_png_dimensions(figure_path)
        except ValueError as error:
            failures.append(str(error))
            continue
        if width < 200 or height < 200:
            failures.append(f"{figure_path.name}: implausible PNG size {width}x{height}")
        if sha256_file(figure_path) != entry["figure_sha256"]:
            failures.append(f"{figure_path.name}: recorded hash does not match the file")
        record.setdefault("figures", {})[entry["figure"]] = {
            "width": width,
            "height": height,
            "sha256_match": True,
        }
        # a real decode proves the PNG is readable, not just well-formed
        try:
            import matplotlib.image as mpimg

            image = mpimg.imread(figure_path)
            record["figures"][entry["figure"]]["decoded_shape"] = list(np.shape(image))
        except Exception as error:  # pragma: no cover - depends on the environment
            failures.append(f"{figure_path.name} could not be decoded: {error!r}")
        details = entry.get("n_fft_detail", {})
        for n_fft in NFFT_GRID:
            detail = details.get(str(n_fft))
            if detail is None:
                failures.append(f"figure_evidence.json lacks n_fft={n_fft} for class {code}")
                continue
            if detail["model_input_shape"] != [256, 256]:
                failures.append(f"figure input for n_fft={n_fft} is not 256x256")
            if not detail["model_input_finite"]:
                failures.append(f"figure input for n_fft={n_fft} is not finite")


def check_manifest(
    root: Path, failures: list[str], record: dict[str, Any], strict: bool
) -> None:
    path = root / "experiment_manifest.json"
    if not path.is_file():
        failures.append("experiment_manifest.json missing")
        return
    manifest = read_json(path)
    record["runs_completed"] = manifest.get("runs_completed")
    record["runs_expected"] = manifest.get("runs_expected")
    if strict and manifest.get("runs_completed") != manifest.get("runs_expected"):
        failures.append("the manifest reports incomplete runs")
    design = manifest.get("design", {})
    if strict:
        if design.get("heldout_windows") != HELDOUT_WINDOWS:
            failures.append("the manifest does not declare 1260 held-out windows")
        if design.get("test_windows") != TEST_WINDOWS:
            failures.append("the manifest does not declare 300 old-test windows")
        if design.get("expected_heldout_rows") != EXPECTED_HELDOUT_ROWS:
            failures.append("the manifest does not declare 30,240 held-out rows")
        if design.get("expected_test_rows") != EXPECTED_TEST_ROWS:
            failures.append("the manifest does not declare 1,800 old-test rows")
        if design.get("hop_rule") != "hop_length = n_fft // 2 (exactly 50%)":
            failures.append("the manifest does not freeze hop = n_fft // 2")
    scope = manifest.get("scope", {})
    if not scope.get("not_a_cascade") or scope.get("three_class_performance_claimed"):
        failures.append("the manifest does not disclaim a cascade three-class reading")
    published = manifest.get("published_files", {})
    mismatched = []
    for name, entry in published.items():
        target = root / name
        if not target.is_file() or sha256_file(target) != entry.get("sha256"):
            mismatched.append(name)
    record["published_files"] = len(published)
    record["published_mismatches"] = mismatched
    if mismatched:
        failures.append(f"published files changed after the manifest: {mismatched}")
    selection_digest = sha256_file(root / "validation_nfft_selection.json")
    if manifest.get("selection", {}).get("sha256") not in (None, selection_digest):
        failures.append("the manifest records a different configuration-choice hash")


def check_report(root: Path, failures: list[str], record: dict[str, Any]) -> None:
    path = root / "report.md"
    if not path.is_file():
        failures.append("report.md missing")
        return
    text = path.read_text(encoding="utf-8")
    record["bytes"] = path.stat().st_size
    required = {
        "oracle": "oracle",
        "background_excluded": "背景",
        "not_cascade": "级联",
        "no_generalisation_claim": "跨会话",
        "awgn_caveat": "AWGN",
    }
    missing = [name for name, token in required.items() if token not in text]
    record["missing_statements"] = missing
    if missing:
        failures.append(f"report.md is missing required statements: {missing}")
    for token in ("30,240", "1,800"):
        if token not in text:
            failures.append(f"report.md does not mention {token}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def verify_outputs(
    root: Path | None = None,
    fixture: bool = False,
    nfft_values: Sequence[int] = NFFT_GRID,
    seeds: Sequence[int] = TRAINING_SEEDS,
    heldout_windows: int = HELDOUT_WINDOWS,
    test_windows: int = TEST_WINDOWS,
    check_frozen: bool = True,
    heldout_index: Path = HELDOUT_INDEX,
    old_index: Path = OLD_INDEX,
) -> dict[str, Any]:
    """Recompute everything and return {passed, failures, checks}."""
    root = Path(root) if root is not None else OUTPUT_ROOT
    strict = not fixture and check_frozen
    failures: list[str] = []
    checks: dict[str, Any] = {"root": str(root), "fixture_mode": fixture, "strict": strict}

    if check_frozen:
        check_frozen_inputs(failures, checks.setdefault("frozen_inputs", {}))
    else:
        checks["frozen_inputs"] = {"skipped": True}

    runs = check_runs(root, failures, checks.setdefault("runs", {}), nfft_values, seeds, strict)
    selection = check_selection(root, failures, checks.setdefault("selection", {}), runs, strict)
    check_generation_record(
        root, failures, checks.setdefault("generation_record", {}), checks["selection"].get("sha256")
    )
    selection_by_model = {
        (int(row["n_fft"]), int(row["training_seed"])): row
        for row in selection.get("per_model", [])
    }

    heldout_records = check_prediction_table(
        root / "heldout_window_predictions.csv",
        "heldout",
        heldout_index,
        failures,
        checks.setdefault("heldout_predictions", {}),
        heldout_windows,
        CONDITIONS,
        len(nfft_values) * len(seeds) * len(CONDITIONS) * heldout_windows,
        nfft_values,
        seeds,
        selection_by_model,
        strict,
    )
    check_prediction_table(
        root / "test_clean_window_predictions.csv",
        "test",
        old_index,
        failures,
        checks.setdefault("test_predictions", {}),
        test_windows,
        TEST_CONDITIONS,
        len(nfft_values) * len(seeds) * len(TEST_CONDITIONS) * test_windows,
        nfft_values,
        seeds,
        selection_by_model,
        strict,
    )

    metrics = check_metric_tables(
        root,
        failures,
        checks.setdefault("metrics", {}),
        heldout_records,
        "heldout",
        CONDITIONS,
        strict,
    )
    test_record = check_metric_tables(
        root,
        failures,
        checks.setdefault("test_metrics", {}),
        [
            {
                "n_fft": int(row["n_fft"]),
                "training_seed": int(row["training_seed"]),
                "condition": row["condition"],
                "filename": row["filename"],
                "label": int(row["label"]),
                "prediction": int(row["prediction"]),
            }
            for row in read_csv_rows(root / "test_clean_window_predictions.csv")
        ]
        if (root / "test_clean_window_predictions.csv").is_file()
        else [],
        "test",
        TEST_CONDITIONS,
        strict,
        table_name="test_clean_model_condition_metrics.csv",
        per_file_required=fixture,
    )
    if not metrics and heldout_records:
        failures.append("no held-out metric group could be recomputed")
    checks["recomputed_groups"] = {"heldout": len(metrics), "test": len(test_record)}

    check_cache_manifest(
        root,
        failures,
        checks.setdefault("cache", {}),
        nfft_values,
        {"heldout": heldout_windows, "test": test_windows},
        strict,
    )
    check_figures(root, failures, checks.setdefault("figures", {}), strict)
    check_frontend_check(root, failures, checks.setdefault("frontends", {}))
    check_manifest(root, failures, checks.setdefault("manifest", {}), strict)
    check_report(root, failures, checks.setdefault("report", {}))

    return {
        "task": "STAGE2-BASE independent verification",
        "root": str(root),
        "fixture_mode": fixture,
        "passed": not failures,
        "failures": failures,
        "checks": checks,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--report", type=Path, help="where to write verification.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.output_dir or OUTPUT_ROOT
    report = verify_outputs(root, fixture=args.fixture)
    target = args.report or (root / "verification.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for failure in report["failures"]:
        print(f"FAIL: {failure}")
    print(f"passed={report['passed']} failures={len(report['failures'])} -> {target}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
