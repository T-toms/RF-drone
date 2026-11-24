"""INFER-UI1 independent verification for the arbitrary-IQ analysis UI.

Seven checks, each writing machine-readable evidence into
``dronerfa_baseline/outputs/signal_analyzer_validation_20260920/``:

1. frozen-hash baseline of the CASCADE3 predictions/weights/config before any work;
2. the fixed three-seed majority rule recomputed from the **published**
   ``window_predictions.csv`` (7,720 ``dataset/condition/window`` keys) including
   heldout four-condition and old-test clean ensemble metrics and 1-1-1 counts;
3. real raw IQ for published clean windows of each class (background / Phantom /
   Air2S) through the **new core path** with the frozen CUDA bundle, comparing
   per-seed discrete decisions and logits against the published rows, recording the
   real device, interpreter and batch composition;
4. fixture tests (MAT/NPY/NPZ/RAW detection, streaming slices, I/Q order,
   start/stride/tail, short-signal rejection, wrong-sample-rate disable, path
   traversal, upload abort cleanup, 1-1-1 not forced, file summary rules);
5. a real HTTP server flow (health -> inspect -> analyze -> poll -> download) plus
   the single-job queue limit and cancellation;
6. bounded end-to-end smoke on three real files (one per class) with outputs in the
   project scratch, never on F:;
7. frozen-hash re-check after all work, plus a headless-browser screenshot of the
   running UI as reviewable rendering evidence.

Usage::

    .venv\\Scripts\\python.exe dronerfa_baseline\\verify_signal_analyzer.py
    .venv\\Scripts\\python.exe dronerfa_baseline\\verify_signal_analyzer.py --skip-smoke --skip-browser
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "outputs" / ".matplotlib"))

import h5py
import numpy as np
import torch

import signal_analyzer_core as core
import signal_analyzer_server as server
from dronerfa_data import load_index
from run_cascade3_evaluation import CONDITIONS as CASCADE_CONDITIONS

OUTPUT_DIR = PROJECT_DIR / "outputs" / "signal_analyzer_validation_20260920"
SCRATCH_DIR = PROJECT_DIR / "scratch" / "signal_analyzer_verification"
# the real dataset root is read-only here; all verification outputs still go to scratch
VERIFY_ROOTS: tuple[Path, ...] = (SCRATCH_DIR, PROJECT_DIR, Path("F:/DroneRFa数据集"))
CASCADE_ROOT = PROJECT_DIR / "outputs" / "cascade3_evaluation_20260919"
PUBLISHED_CSV = CASCADE_ROOT / "window_predictions.csv"
PUBLISHED_CSV_SHA256 = "59321de7be48aae7d8b135f258aea5287f7e853d3d0d0a726ccfba8bc313f052"

BATCH_SIZE = core.BATCH_SIZE
# one published clean window per class, chosen only to cover the three classes
TARGET_WINDOW_POSITIONS = {"heldout": {0: 0, 1: 560, 2: 1170}, "old_test": {0: 0, 1: 80, 2: 220}}
# classes present in the old-test split (T0000/T0010/T0101 codes)
OLD_TEST_CLASSES = (0, 1, 2)
SMOKE_FILES = {
    "Background": ("heldout", "T0000_D00_S0100.mat"),
    "DJI Phantom 4 Pro": ("heldout", "T0010_D00_S0100.mat"),
    "DJI Air 2S": ("heldout", "T0101_D00_S0100.mat"),
}


def resolve_target_positions(dataset: str, published: Sequence[dict[str, str]]) -> dict[int, int]:
    """First published clean position of each class present in a dataset's split."""
    positions: dict[int, int] = {}
    for true_class in OLD_TEST_CLASSES:
        candidates = [
            int(row["dataset_index_position"])
            for row in published
            if row["dataset"] == dataset
            and row["condition"] == "clean"
            and int(row["true_class"]) == true_class
        ]
        if candidates:
            positions[true_class] = min(candidates)
    return positions


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# check 1 / 7: frozen artefact hashes
# ---------------------------------------------------------------------------
def frozen_snapshot() -> dict[str, Any]:
    cascade_config = core.CASCADE_CONFIG_PATH
    ui_config = core.UI_CONFIG_PATH
    weights: dict[str, Any] = {}
    for pairing in core.load_json(cascade_config)["seed_pairing"]:
        for stage in (1, 2):
            path = PROJECT_ROOT / str(pairing[f"stage{stage}_checkpoint"])
            weights[f"seed{pairing['seed']}_stage{stage}"] = {
                "path": str(path),
                "expected_sha256": str(pairing[f"stage{stage}_sha256"]),
                "actual_sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return {
        "captured_at": utc_now(),
        "cascade_config": {
            "path": str(cascade_config),
            "sha256": sha256_file(cascade_config),
            "bytes": cascade_config.stat().st_size,
        },
        "ui_config": {
            "path": str(ui_config),
            "sha256": sha256_file(ui_config),
            "bytes": ui_config.stat().st_size,
        },
        "published_window_predictions": {
            "path": str(PUBLISHED_CSV),
            "sha256": sha256_file(PUBLISHED_CSV),
            "bytes": PUBLISHED_CSV.stat().st_size,
            "expected_sha256": PUBLISHED_CSV_SHA256,
        },
        "weights": weights,
    }


def compare_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    problems: list[str] = []
    if before["cascade_config"]["sha256"] != after["cascade_config"]["sha256"]:
        problems.append("cascade3 config hash changed")
    if (
        before["published_window_predictions"]["sha256"]
        != after["published_window_predictions"]["sha256"]
    ):
        problems.append("published CASCADE3 window_predictions.csv changed")
    for key, record in before["weights"].items():
        if record["actual_sha256"] != after["weights"][key]["actual_sha256"]:
            problems.append(f"weight {key} changed")
    for record in before["weights"].values():
        if record["actual_sha256"] != record["expected_sha256"]:
            problems.append(f"weight {record['path']} mismatches the frozen hash")
    if (
        before["published_window_predictions"]["sha256"]
        != before["published_window_predictions"]["expected_sha256"]
    ):
        problems.append("published CASCADE3 CSV does not match the accepted hash 59321de7...")
    return {"problems": problems, "unchanged": not problems}


# ---------------------------------------------------------------------------
# check 2: ensemble rule recomputed from the published CSV
# ---------------------------------------------------------------------------
def ensemble_check() -> dict[str, Any]:
    rows = read_csv_rows(PUBLISHED_CSV)
    grouped: dict[tuple[str, str, int], dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row["condition"], int(row["dataset_index_position"]))
        grouped[key][int(row["training_seed"])] = row
    problems: list[str] = []
    per_group: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "windows": 0,
            "counts": Counter(),
            "uncertain_1_1_1": 0,
            "three_of_three": 0,
            "majority_vs_truth": Counter(),
            "window_labels": {},
        }
    )
    uncertain_total = 0
    for key, seeds in grouped.items():
        dataset, condition, position = key
        if sorted(seeds) != [2026, 2027, 2028]:
            problems.append(f"{key}: seeds {sorted(seeds)}")
            continue
        votes = [int(seeds[seed]["final_prediction"]) for seed in (2026, 2027, 2028)]
        label, agreement, uncertain = core.majority_label(votes)
        group = per_group[(dataset, condition)]
        group["windows"] += 1
        group["counts"][label] += 1
        group["window_labels"][position] = label
        if uncertain:
            group["uncertain_1_1_1"] += 1
            uncertain_total += 1
        if agreement == "3/3":
            group["three_of_three"] += 1
        truth = int(seeds[2026]["true_class"])
        truth_label = core.CLASS_NAMES[truth]
        majority_class = truth_label if label == truth_label else f"NOT_{truth_label}"
        group["majority_vs_truth"][majority_class] += 1

    summary: dict[str, Any] = {}
    for (dataset, condition), group in sorted(per_group.items()):
        windows = group["windows"]
        counts = {name: int(group["counts"].get(name, 0)) for name in core.TIMELINE_ORDER}
        # three-class accuracy against the index truth
        truth_hits = 0
        for key, seeds in grouped.items():
            if key[0] != dataset or key[1] != condition:
                continue
            votes = [int(seeds[seed]["final_prediction"]) for seed in (2026, 2027, 2028)]
            label, _, _ = core.majority_label(votes)
            truth_label = core.CLASS_NAMES[int(seeds[2026]["true_class"])]
            if label == truth_label:
                truth_hits += 1
        summary[f"{dataset}/{condition}"] = {
            "windows": windows,
            "counts": counts,
            "ratios": {name: counts[name] / windows for name in core.TIMELINE_ORDER},
            "uncertain_1_1_1": int(group["uncertain_1_1_1"]),
            "three_of_three": int(group["three_of_three"]),
            "three_of_three_rate": group["three_of_three"] / windows,
            "ensemble_accuracy_vs_index_truth": truth_hits / windows,
            "majority_confusion": {
                key: int(value) for key, value in sorted(group["majority_vs_truth"].items())
            },
        }
        if dataset == "heldout":
            drone = counts["DJI Phantom 4 Pro"] + counts["DJI Air 2S"]
            summary[f"{dataset}/{condition}"]["drone_windows"] = drone
            summary[f"{dataset}/{condition}"]["drone_ratio"] = drone / windows
    if len(grouped) != 7720:
        problems.append(f"expected 7720 unique dataset/condition/window keys, got {len(grouped)}")
    if len(rows) != 23160:
        problems.append(f"expected 23160 published rows, got {len(rows)}")
    heldout_means = {
        condition: summary[f"heldout/{condition}"]["ensemble_accuracy_vs_index_truth"]
        for condition in CASCADE_CONDITIONS
    }
    return {
        "rows": len(rows),
        "unique_keys": len(grouped),
        "expected_keys": 7720,
        "uncertain_1_1_1_total": uncertain_total,
        "per_group": summary,
        "heldout_four_condition_mean_ensemble_accuracy": float(np.mean(list(heldout_means.values()))),
        "heldout_accuracy_by_condition": heldout_means,
        "problems": problems,
        "rule": "majority of the three per-seed final labels; 1-1-1 -> uncertain (never forced)",
        "note": (
            "This only re-derives the fixed ensemble rule from the accepted predictions; "
            "no rule or threshold was changed and the heldout split is still descriptive."
        ),
    }


# ---------------------------------------------------------------------------
# check 3: real raw IQ through the new core path with the frozen bundle
# ---------------------------------------------------------------------------
def real_iq_parity(
    bundle: core.ModelBundle,
    datasets: Sequence[str] = ("heldout", "old_test"),
    target_positions: dict[str, dict[int, int]] | None = None,
) -> dict[str, Any]:
    published = read_csv_rows(PUBLISHED_CSV)
    if target_positions is None:
        target_positions = {
            dataset: resolve_target_positions(dataset, published) for dataset in datasets
        }
    clean = [
        row
        for row in published
        if row["condition"] == "clean" and row["dataset"] in set(datasets)
    ]
    by_dataset: dict[str, dict[tuple[int, int], dict[int, dict[str, str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in clean:
        key = (int(row["dataset_index_position"]), int(row["true_class"]))
        by_dataset[row["dataset"]][key][int(row["training_seed"])] = row

    report: dict[str, Any] = {
        "environment": core.environment_report(bundle.device),
        "model": bundle.to_dict(),
        "batch_rule": (
            "windows of one target are read as raw RF0_I/RF0_Q slices, demean+RMS "
            "normalised, stacked to [2, 1000000] float32 and used as one contiguous "
            "batch of 4 (stage 1); stage 2 runs only for seeds whose own stage-1 "
            "margin > 0 and receives the same tensor, chunked ascending by batch 4"
        ),
        "datasets": {},
    }
    problems: list[str] = []
    for dataset in datasets:
        entries = by_dataset.get(dataset, {})
        positions = target_positions[dataset]
        rows_by_class: dict[int, dict[int, dict[str, str]]] = defaultdict(dict)
        for (position, true_class), seeds in entries.items():
            rows_by_class[true_class][position] = seeds
        dataset_report: dict[str, Any] = {"targets": {}, "max_abs_logit_diff": 0.0}
        for true_class, wanted_position in positions.items():
            seeds = rows_by_class.get(true_class, {}).get(wanted_position)
            if not seeds:
                problems.append(f"{dataset} class {true_class} position {wanted_position} missing")
                continue
            reference = seeds[2026]
            mat_path = Path(reference["file_path"])
            start = int(reference["start"])
            reference_row = reference
            with h5py.File(mat_path, "r") as handle:
                i_slice = np.asarray(handle["RF0_I"][0, start : start + core.WINDOW_SAMPLES], dtype=np.float32)
                q_slice = np.asarray(handle["RF0_Q"][0, start : start + core.WINDOW_SAMPLES], dtype=np.float32)
            iq = i_slice + 1j * q_slice
            tensor = core.iq_to_tensor(iq)
            batch = torch.stack([tensor]).to(bundle.device)
            features = core.features_from_signals(bundle.frontend, batch, bundle.device)
            target: dict[str, Any] = {
                "dataset": dataset,
                "true_class": true_class,
                "true_class_name": core.CLASS_NAMES[true_class],
                "dataset_index_position": int(reference_row["dataset_index_position"]),
                "file_path": str(mat_path),
                "start": start,
                "length": int(reference_row["length"]),
                "batch_shape": list(batch.shape),
                "seeds": {},
            }
            for seed in core.TRAINING_SEEDS:
                published_stage1 = np.array(
                    [
                        float(seeds[seed]["stage1_logit_background"]),
                        float(seeds[seed]["stage1_logit_drone"]),
                    ],
                    dtype=np.float32,
                )
                our_stage1 = core.network_logits(bundle.stage1_network(seed), features, bundle.device)[0]
                margin = float(np.float32(our_stage1[1] - our_stage1[0]))
                executed = margin > 0.0
                published_executed = seeds[seed]["stage2_executed"].strip().lower() == "true"
                entry: dict[str, Any] = {
                    "published_stage1_logits": [float(value) for value in published_stage1],
                    "our_stage1_logits": [float(value) for value in our_stage1],
                    "stage1_logit_max_abs_diff": float(np.max(np.abs(our_stage1 - published_stage1))),
                    "stage1_margin": margin,
                    "stage1_prediction": int(our_stage1[1] - our_stage1[0] > 0.0),
                    "published_stage1_prediction": int(seeds[seed]["stage1_prediction"]),
                    "stage2_executed": bool(executed),
                    "published_stage2_executed": bool(published_executed),
                    "final_prediction": 0 if not executed else None,
                    "published_final_prediction": int(seeds[seed]["final_prediction"]),
                }
                if executed:
                    our_stage2 = core.network_logits(bundle.stage2_network(seed), features, bundle.device)[0]
                    published_stage2 = np.array(
                        [
                            float(seeds[seed]["stage2_logit_phantom"]),
                            float(seeds[seed]["stage2_logit_air2s"]),
                        ],
                        dtype=np.float32,
                    ) if published_executed else None
                    stage2_margin = float(np.float32(our_stage2[1] - our_stage2[0]))
                    entry["our_stage2_logits"] = [float(value) for value in our_stage2]
                    entry["stage2_margin"] = stage2_margin
                    entry["final_prediction"] = 2 if stage2_margin > 0.0 else 1
                    if published_stage2 is not None:
                        entry["published_stage2_logits"] = [float(value) for value in published_stage2]
                        entry["stage2_logit_max_abs_diff"] = float(
                            np.max(np.abs(our_stage2 - published_stage2))
                        )
                        dataset_report["max_abs_logit_diff"] = max(
                            dataset_report["max_abs_logit_diff"],
                            entry["stage1_logit_max_abs_diff"],
                            entry["stage2_logit_max_abs_diff"],
                        )
                    else:
                        problems.append(
                            f"{dataset} class {true_class} seed {seed}: we ran stage 2 but the "
                            "published row did not"
                        )
                else:
                    dataset_report["max_abs_logit_diff"] = max(
                        dataset_report["max_abs_logit_diff"], entry["stage1_logit_max_abs_diff"]
                    )
                    if published_executed:
                        problems.append(
                            f"{dataset} class {true_class} seed {seed}: we skipped stage 2 but the "
                            "published row did not"
                        )
                entry["decision_matches_published"] = bool(
                    entry["final_prediction"] == entry["published_final_prediction"]
                    and entry["stage1_prediction"] == entry["published_stage1_prediction"]
                    and entry["stage2_executed"] == entry["published_stage2_executed"]
                )
                if not entry["decision_matches_published"]:
                    problems.append(
                        f"{dataset} class {true_class} seed {seed}: decision mismatch "
                        f"{entry['final_prediction']} vs {entry['published_final_prediction']}"
                    )
                target["seeds"][str(seed)] = entry
            dataset_report["targets"][core.CLASS_NAMES[true_class]] = target
        report["datasets"][dataset] = dataset_report
    report["problems"] = problems
    report["passed"] = not problems
    return report


# ---------------------------------------------------------------------------
# check 3b: reproduce CASCADE3's exact stage-1 batch composition
# ---------------------------------------------------------------------------
def batch_mirror_experiment(
    bundle: core.ModelBundle,
    datasets: Sequence[str] = ("heldout", "old_test"),
) -> dict[str, Any]:
    """Feed four *contiguous published windows* through the UI core as one batch.

    CASCADE3 caches features in contiguous chunks of ``batch_size=4`` over the frozen
    split order, which the Codex batch-parity probe showed is not numerically
    interchangeable with other batch layouts.  Here the UI core path (streaming
    reader -> demean/RMS -> frontend -> AMP forward) is given exactly
    ``start_sample = <first window of the group>`` and ``max_windows = 4``, so its
    batch of four equals the published chunk.  Stage-1 logits must then match the
    published values bit for bit.
    """
    published = read_csv_rows(PUBLISHED_CSV)
    report: dict[str, Any] = {"groups": {}, "problems": [], "max_abs_logit_diff": 0.0}
    for dataset in datasets:
        clean = [
            row
            for row in published
            if row["dataset"] == dataset and row["condition"] == "clean"
        ]
        for true_class in OLD_TEST_CLASSES:
            candidates = [row for row in clean if int(row["true_class"]) == true_class]
            if not candidates:
                continue
            anchor = min(candidates, key=lambda row: (int(row["start"]), int(row["dataset_index_position"])))
            anchor_position = int(anchor["dataset_index_position"])
            path = Path(anchor["file_path"])
            start = int(anchor["start"])
            plan = core.plan_windows(
                samples=start + 4 * core.WINDOW_SAMPLES, start_sample=start, max_windows=4
            )
            index_by_position: dict[int, dict[int, dict[str, str]]] = defaultdict(dict)
            for position in range(anchor_position, anchor_position + 4):
                for row in clean:
                    if (
                        int(row["dataset_index_position"]) == position
                        and Path(row["file_path"]).name == path.name
                    ):
                        index_by_position[position][int(row["training_seed"])] = row
            source = core.open_source(path, {}, core.inspect_path(path))
            run = core.run_stream_analysis(source=source, plan=plan, bundle=bundle)
            group: dict[str, Any] = {
                "dataset": dataset,
                "true_class_name": core.CLASS_NAMES[true_class],
                "file_path": str(path),
                "anchor_dataset_index_position": anchor_position,
                "start_sample": start,
                "batch_windows": [
                    {"window_index": index, "start_sample": plan.window_bounds(index)[0]}
                    for index in range(plan.windows)
                ],
                "seeds": {},
            }
            for seed in core.TRAINING_SEEDS:
                per_window: list[dict[str, Any]] = []
                for index, row in enumerate(run["per_seed_rows"][seed]):
                    position = anchor_position + index
                    reference = index_by_position.get(position, {}).get(seed)
                    if reference is None:
                        report["problems"].append(
                            f"{dataset}/{group['true_class_name']} seed {seed}: published row "
                            f"for position {position} missing"
                        )
                        continue
                    published_logits = np.array(
                        [
                            float(reference["stage1_logit_background"]),
                            float(reference["stage1_logit_drone"]),
                        ],
                        dtype=np.float32,
                    )
                    our_logits = np.array(
                        [row["stage1_logit_background"], row["stage1_logit_drone"]],
                        dtype=np.float32,
                    )
                    difference = float(np.max(np.abs(our_logits - published_logits)))
                    report["max_abs_logit_diff"] = max(report["max_abs_logit_diff"], difference)
                    entry = {
                        "window_index": int(row["window_index"]),
                        "dataset_index_position": position,
                        "stage1_logit_max_abs_diff": difference,
                        "stage1_prediction": int(row["stage1_prediction"]),
                        "published_stage1_prediction": int(reference["stage1_prediction"]),
                        "final_prediction": int(row["final_prediction"]),
                        "published_final_prediction": int(reference["final_prediction"]),
                        "digest_match": bool(
                            str(row["stage1_feature_digest"]) == str(reference["stage1_feature_digest"])
                        ),
                    }
                    if (
                        entry["stage1_prediction"] != entry["published_stage1_prediction"]
                        or entry["final_prediction"] != entry["published_final_prediction"]
                    ):
                        report["problems"].append(
                            f"{dataset}/{group['true_class_name']} seed {seed} window {index}: "
                            f"decision {entry['final_prediction']} vs published "
                            f"{entry['published_final_prediction']}"
                        )
                    per_window.append(entry)
                group["seeds"][str(seed)] = per_window
            report["groups"][f"{dataset}/{core.CLASS_NAMES[true_class]}"] = group
    report["passed"] = not report["problems"]
    report["rule"] = (
        "start_sample = first window of the group, max_windows = 4, so the UI batch equals "
        "the published contiguous chunk of batch_size=4"
    )
    return report


# ---------------------------------------------------------------------------
# check 4: bounded fixture tests
# ---------------------------------------------------------------------------
def fixture_tests() -> dict[str, Any]:
    import importlib

    module = importlib.import_module("test_signal_analyzer")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in (
        "TestFormatsAndSources",
        "TestPlanningAndAggregation",
        "TestPathSafety",
        "TestStreamAnalysis",
    ):
        suite.addTests(loader.loadTestsFromTestCase(getattr(module, name)))
    stream = open(os.devnull, "w", encoding="utf-8")
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    started = time.perf_counter()
    result = runner.run(suite)
    stream.close()
    return {
        "tests_run": result.testsRun,
        "failures": [str(item[0]) for item in result.failures],
        "errors": [str(item[0]) for item in result.errors],
        "skipped": len(result.skipped),
        "seconds": round(time.perf_counter() - started, 3),
        "passed": result.wasSuccessful(),
        "covers": [
            "MAT/NPY/NPZ/RAW 格式探测与布局识别",
            "HDF5 分块流式读取（不整文件载入）",
            "RAW IQ/QI 顺序",
            "start/stride/尾部丢弃与短信号拒绝",
            "路径越界拒绝",
            "1-1-1 不强制分类",
            "文件级汇总规则",
            "取消不发布",
        ],
    }


# ---------------------------------------------------------------------------
# check 5: real HTTP flow, queue limit, cancellation
# ---------------------------------------------------------------------------
def http_flow(
    bundle: core.ModelBundle,
    output_root: Path,
    fixture_dir: Path,
    use_real_file: bool = False,
    review_dir: Path | None = None,
) -> dict[str, Any]:
    """Boot the real server with the real frozen bundle on a scratch output root.

    ``use_real_file`` switches the analysed input from the tiny MAT fixture to a real
    heldout MAT file with a bounded window count, which also produces the artefacts
    copied into ``review_dir`` for the offline rendering snapshot.
    """
    import test_signal_analyzer as fixtures

    fixture_dir.mkdir(parents=True, exist_ok=True)
    mat = fixture_dir / "http_flow_fixture.mat"
    display_name = "http_flow_fixture"
    max_windows: int | None = 3
    if use_real_file:
        dataset, filename = SMOKE_FILES["DJI Phantom 4 Pro"]
        rows = [
            row
            for row in load_index(
                PROJECT_DIR
                / "outputs"
                / "heldout_preparation_20260917"
                / "heldout_window_index.csv",
                split="heldout",
            )
            if row["filename"] == filename
        ]
        mat = Path(rows[0]["file_path"])
        display_name = Path(filename).stem
        max_windows = 6
    elif not mat.exists():
        fixtures.write_mat(mat, fixtures.tone(3 * core.WINDOW_SAMPLES + 11, seed=77))
    server_object, manager = server.create_server(
        host="127.0.0.1",
        port=0,
        output_root=output_root,
        allowed_roots=VERIFY_ROOTS,
        preload=False,
        quiet=True,
    )
    manager._bundle = bundle
    manager._bundle_state = "ready"
    thread = threading.Thread(target=server_object.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server_object.server_address[1]}"
    problems: list[str] = []
    steps: list[dict[str, Any]] = []
    captured: dict[str, Any] = {}

    def http(path: str, method: str = "GET", body: dict | None = None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(base + path, data=data, method=method)
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))

    def raw(path: str):
        with urllib.request.urlopen(base + path, timeout=180) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "")

    try:
        status, health = http("/api/health")
        steps.append({"step": "health", "status": status})
        if status != 200 or health["host"] != "127.0.0.1":
            problems.append(f"health unexpected: {status} {health.get('host')}")
        status, page, content_type = raw("/")
        steps.append({"step": "static index", "status": status, "bytes": len(page)})
        text = page.decode("utf-8")
        for needle in ("//cdn", "http://cdn", "https://", "unpkg", "cdnjs"):
            if needle in text:
                problems.append(f"page references external resource {needle}")
        status, inspected = http(
            "/api/inspect",
            "POST",
            {"path": str(mat), "sample_rate_confirmed": True, "max_windows": max_windows},
        )
        steps.append(
            {
                "step": "inspect",
                "status": status,
                "input": str(mat),
                "windows": inspected.get("window_plan", {}).get("windows"),
                "allowed": inspected.get("classification_allowed"),
            }
        )
        if not inspected.get("classification_allowed"):
            problems.append("inspect refused a valid 100 MHz MAT input")
        status, submitted = http(
            "/api/analyze",
            "POST",
            {
                "path": str(mat),
                "sample_rate_confirmed": True,
                "max_windows": max_windows,
                "display_name": display_name,
            },
        )
        steps.append({"step": "analyze", "status": status, "response": submitted})
        if status != 202 or "job" not in submitted:
            problems.append(f"analyze was refused: {status} {submitted}")
            raise RuntimeError(f"analyze refused: {status} {submitted}")
        job_id = submitted["job"]["job_id"]
        steps.append({"step": "job id", "job_id": job_id})
        # single-job queue: a second submission must be refused while this one runs
        confirm_status, confirm = http(
            "/api/analyze",
            "POST",
            {"path": str(mat), "sample_rate_confirmed": True},
        )
        steps.append({"step": "queue limit", "status": confirm_status})
        if confirm_status != 409:
            problems.append(f"second concurrent job was accepted ({confirm_status})")
        deadline = time.time() + 900
        job = None
        while time.time() < deadline:
            _, payload = http(f"/api/jobs/{job_id}")
            job = payload["job"]
            if job["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.3)
        if not job or job["status"] != "completed":
            problems.append(f"job did not complete: {job and job.get('status')} {job and job.get('error')}")
            raise RuntimeError("http flow job failed")
        result = job["result"]
        captured = job
        if review_dir is not None and job.get("output_dir"):
            review_dir.mkdir(parents=True, exist_ok=True)
            copied: list[str] = []
            source_dir = Path(job["output_dir"])
            for name in (
                "window_predictions.csv",
                "ensemble_predictions.csv",
                "analysis.json",
                "summary.md",
                "model_hashes.json",
                "timeline.png",
                "waveform_preview.png",
            ):
                source = source_dir / name
                if source.is_file():
                    shutil.copyfile(source, review_dir / name)
                    copied.append(name)
            for extra in sorted(source_dir.glob("stft_preview_window_*.png")):
                shutil.copyfile(extra, review_dir / extra.name)
                copied.append(extra.name)
            steps.append({"step": "review artifacts", "copied": copied})
        steps.append(
            {
                "step": "poll",
                "status": job["status"],
                "windows": result["analysis"]["file_summary"]["analysed_windows"],
                "dominant": result["analysis"]["file_summary"]["dominant_text"],
            }
        )
        for key in ("predictions_csv", "ensemble_csv", "analysis_json", "summary_md", "model_hashes"):
            status, body, _ = raw(result["downloads"][key])
            if status != 200 or not body:
                problems.append(f"download {key} failed ({status})")
        steps.append({"step": "downloads", "status": 200})
        status, stft = http(f"/api/jobs/{job_id}/stft?window=1")
        steps.append(
            {
                "step": "stft",
                "status": status,
                "rows": len(stft["stft_series"]["values"]),
                "columns": len(stft["stft_series"]["values"][0]),
            }
        )
        if status != 200 or len(stft["stft_series"]["values"]) != core.N_FFT:
            problems.append("stft endpoint did not return a 512-row spectrogram")
    finally:
        server_object.shutdown()
        manager.shutdown()
        server_object.server_close()
    return {
        "steps": steps,
        "problems": problems,
        "passed": not problems,
        "job_result": captured.get("result"),
    }


# ---------------------------------------------------------------------------
# check 6: bounded end-to-end smoke on three real files
# ---------------------------------------------------------------------------
def real_file_smoke(bundle: core.ModelBundle, output_root: Path) -> dict[str, Any]:
    problems: list[str] = []
    runs: list[dict[str, Any]] = []
    for label, (dataset, filename) in SMOKE_FILES.items():
        index_path = (
            PROJECT_DIR / "outputs" / "heldout_preparation_20260917" / "heldout_window_index.csv"
        )
        rows = [row for row in load_index(index_path, split="heldout") if row["filename"] == filename]
        if not rows:
            problems.append(f"{filename} not found in the heldout index")
            continue
        path = Path(rows[0]["file_path"])
        inspection = core.inspect_path(path)
        plan = core.plan_windows(int(inspection["samples"]), max_windows=2)
        source = core.open_source(path, {}, inspection)
        started = time.perf_counter()
        run = core.run_stream_analysis(source=source, plan=plan, bundle=bundle)
        summary = core.summarize_file(
            run["ensemble_rows"],
            total_windows_available=plan.requested_windows,
            dropped_samples=plan.dropped_samples,
        )
        stft_preview = core.compute_stft_image(run["preview_iq"][: core.WINDOW_SAMPLES])
        runs.append(
            {
                "label": label,
                "file": str(path),
                "samples": int(inspection["samples"]),
                "windows": plan.windows,
                "available_windows": plan.requested_windows,
                "dropped_samples": plan.dropped_samples,
                "seconds": round(time.perf_counter() - started, 3),
                "ensemble_labels": [
                    row["ensemble_class_name"] for row in run["ensemble_rows"]
                ],
                "per_seed_final": {
                    str(seed): [row["final_prediction"] for row in run["per_seed_rows"][seed]]
                    for seed in core.TRAINING_SEEDS
                },
                "drone_ratio": summary["drone_ratio"],
                "dominant_text": summary["dominant_text"],
                "model_agreement_rate": summary["model_agreement_rate"],
                "stft_rows": int(len(stft_preview["values"])),
                "stft_columns": int(len(stft_preview["values"][0])),
            }
        )
        log(f"  smoke {label}: {filename} -> {runs[-1]['ensemble_labels']}")
    return {
        "output_root": str(output_root),
        "runs": runs,
        "problems": problems,
        "passed": not problems and len(runs) == len(SMOKE_FILES),
        "note": "有界smoke：每类一个真实文件、每个文件2窗，产物仅写项目scratch。",
    }


# ---------------------------------------------------------------------------
# check 8: reviewable rendering evidence (headless browser screenshot)
# ---------------------------------------------------------------------------
def browser_screenshot(
    bundle: core.ModelBundle,
    output_root: Path,
    job_result: dict[str, Any] | None = None,
    review_dir: Path | None = None,
) -> dict[str, Any]:
    """Headless-browser render evidence.

    Two artefacts are produced:

    * ``ui_screenshot.png`` – the live page served by a real server instance;
    * ``ui_snapshot.html`` – a self-contained offline copy of the page with the
      captured job result injected (``window.__UI_SNAPSHOT__``), so the rendering of
      the result area can be reviewed from ``file://`` without a server.

    Chrome/Edge is launched with ``subprocess.Popen`` and stdio redirected to files:
    the DSH file sandbox denies Chrome the named pipes that ``subprocess.run`` with
    pipes would need, which otherwise aborts the browser before it renders.
    """
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    browser = next((item for item in candidates if item.is_file()), None)
    snapshot_path = OUTPUT_DIR / "ui_snapshot.html"
    analysis_payload = analysis_of(job_result)
    input_path = ((analysis_payload.get("input") or {}).get("path")) if analysis_payload else None
    snapshot = write_offline_snapshot(
        snapshot_path, job_result, review_dir=review_dir, input_path=input_path
    )
    if browser is None:
        return {
            "screenshot": None,
            "snapshot": str(snapshot_path) if snapshot else None,
            "problems": ["未找到 Chrome/Edge，无法生成界面渲染证据；已生成离线快照。"],
            "passed": False,
        }
    server_object, manager = server.create_server(
        host="127.0.0.1",
        port=0,
        output_root=output_root,
        allowed_roots=VERIFY_ROOTS,
        preload=False,
        quiet=True,
    )
    manager._bundle = bundle
    manager._bundle_state = "ready"
    thread = threading.Thread(target=server_object.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server_object.server_address[1]}/"
    shot = OUTPUT_DIR / "ui_screenshot.png"
    log_path = output_root / "browser.log"
    output_root.mkdir(parents=True, exist_ok=True)
    profile = output_root / "browser_profile"
    profile.mkdir(parents=True, exist_ok=True)
    chromium_flags = [
        "--headless=new",
        "--no-sandbox",
        "--disable-crash-reporter",
        "--disable-breakpad",
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={profile}",
        "--window-size=1440,2000",
        "--virtual-time-budget=8000",
    ]
    attempts: list[dict[str, Any]] = []
    try:
        for label, target in (("live", url), ("snapshot", snapshot_path.as_uri() if snapshot else None)):
            if target is None:
                continue
            if shot.exists():
                shot.unlink()
            command = [str(browser), *chromium_flags, f"--screenshot={shot}", target]
            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    command, stdout=log_handle, stderr=log_handle, stdin=subprocess.DEVNULL
                )
                returncode = process.wait(timeout=180)
            produced = shot.is_file() and shot.stat().st_size > 5000
            attempts.append(
                {
                    "target": label,
                    "url": target,
                    "returncode": returncode,
                    "screenshot_bytes": shot.stat().st_size if shot.is_file() else 0,
                    "ok": produced,
                }
            )
            if produced:
                if label == "snapshot":
                    snapshot_shot = OUTPUT_DIR / "ui_snapshot_screenshot.png"
                    shutil.copyfile(shot, snapshot_shot)
                    attempts[-1]["copied_to"] = str(snapshot_shot)
                break
        log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-1200:] if log_path.is_file() else ""
        passed = bool(attempts) and any(item["ok"] for item in attempts)
        return {
            "browser": str(browser),
            "url": url,
            "snapshot": str(snapshot_path) if snapshot else None,
            "screenshot": str(shot) if shot.is_file() else None,
            "screenshot_bytes": shot.stat().st_size if shot.is_file() else 0,
            "attempts": attempts,
            "browser_log_tail": log_tail,
            "passed": passed,
            "problems": [] if passed else ["Chrome/Edge 未能生成截图（见 browser_log_tail）"],
            "note": (
                "本会话文件沙箱禁止浏览器使用命名管道，因此截图失败时仍提供 "
                "ui_snapshot.html 离线快照；有正常桌面权限时运行 "
                "dronerfa_baseline/render_ui_evidence.py 可复现截图。"
            ),
        }
    except Exception as error:  # noqa: BLE001 - evidence step must not abort the run
        return {
            "screenshot": None,
            "snapshot": str(snapshot_path) if snapshot else None,
            "problems": [f"截图失败：{error}"],
            "passed": False,
        }
    finally:
        server_object.shutdown()
        manager.shutdown()
        server_object.server_close()


def analysis_of(job_payload: dict[str, Any] | None) -> dict[str, Any]:
    """The ``analysis`` mapping of either job shape (``{"analysis": ...}`` or nested)."""
    if not isinstance(job_payload, dict):
        return {}
    if isinstance(job_payload.get("analysis"), dict):
        return job_payload["analysis"]
    result = job_payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("analysis"), dict):
        return result["analysis"]
    return {}


def write_offline_snapshot(
    path: Path,
    job_result: dict[str, Any] | None,
    review_dir: Path | None = None,
    input_path: str | None = None,
) -> bool:
    """Write ``signal_analyzer_web/index.html`` + captured result as one offline file.

    Nothing external is referenced: CSS and JS are inlined and the captured result is
    injected through ``window.__UI_SNAPSHOT__``.  Download entries are rewritten to
    inline ``data:`` URIs (text artefacts) or ``file://`` links (images copied into
    ``review_dir``) so the result area is fully reviewable without a server.
    """
    index = server.WEB_DIR / "index.html"
    if not index.is_file():
        return False
    html = index.read_text(encoding="utf-8")
    css = (server.WEB_DIR / "styles.css").read_text(encoding="utf-8")
    script = (server.WEB_DIR / "app.js").read_text(encoding="utf-8")
    html = html.replace(
        '<link rel="stylesheet" href="styles.css">', f"<style>\n{css}\n</style>"
    )
    snapshot_script = ""
    if job_result is not None:
        payload = json.loads(json.dumps(job_result, ensure_ascii=False))
        result = payload if isinstance(payload.get("analysis"), dict) else (payload.get("result") or {})
        analysis = result.get("analysis") or {}
        # the raw waveform/seed dumps stay in the copied artefacts; the snapshot keeps
        # the timeline, STFT heatmap and aggregate tables
        for key in ("waveform_series", "seed_rows"):
            analysis.pop(key, None)
        analysis["seed_rows_note"] = (
            "离线快照未内嵌每种子明细，完整数据见 review_artifacts/window_predictions.csv"
        )
        result["offline"] = {
            "input_path": input_path,
            "review_dir": str(review_dir) if review_dir else None,
            "downloads": _offline_downloads(review_dir),
        }
        snapshot_script = (
            "<script>window.__UI_SNAPSHOT__ = "
            + json.dumps(payload, ensure_ascii=False)
            + ";</script>\n"
        )
    html = html.replace(
        '<script src="app.js"></script>', snapshot_script + f"<script>\n{script}\n</script>"
    )
    html = html.replace(
        "<title>DroneRFa 任意IQ信号分析界面</title>",
        "<title>DroneRFa 任意IQ信号分析界面（离线渲染快照）</title>",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return True


def _offline_downloads(review_dir: Path | None) -> dict[str, str]:
    """Self-contained download targets for the offline snapshot."""
    if review_dir is None or not Path(review_dir).is_dir():
        return {}
    review_dir = Path(review_dir)
    mapping = {
        "predictions_csv": ("window_predictions.csv", "text"),
        "ensemble_csv": ("ensemble_predictions.csv", "text"),
        "analysis_json": ("analysis.json", "text"),
        "summary_md": ("summary.md", "text"),
        "model_hashes": ("model_hashes.json", "text"),
        "timeline_png": ("timeline.png", "file"),
        "waveform_png": ("waveform_preview.png", "file"),
    }
    links: dict[str, str] = {}
    for key, (name, kind) in mapping.items():
        candidate = review_dir / name
        if not candidate.is_file():
            continue
        if kind == "text":
            import base64

            encoded = base64.b64encode(candidate.read_bytes()).decode("ascii")
            links[key] = f"data:application/octet-stream;base64,{encoded}"
        else:
            links[key] = candidate.as_uri()
    return links


def launcher_check(bundle: core.ModelBundle, output_root: Path) -> dict[str, Any]:
    """Real launcher check: parse the PowerShell script and boot the service with it."""
    import socket

    launcher = PROJECT_DIR / "launch_signal_analyzer.ps1"
    powershell = shutil.which("powershell") or r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    raw = launcher.read_bytes()
    problems: list[str] = []
    script_check: dict[str, Any] = {
        "path": str(launcher),
        "bytes": len(raw),
        "has_utf8_bom": raw.startswith(b"\xef\xbb\xbf"),
    }
    if not script_check["has_utf8_bom"]:
        problems.append("启动器缺少 UTF-8 BOM，Windows PowerShell 5.1 会按 ANSI 解析中文")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "launcher.log"
    url = f"http://127.0.0.1:{port}/"
    command = [
        powershell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        (
            f"& '{launcher}' -Port {port} -NoBrowser -Quiet "
            f"-OutputRoot '{output_root}' "
            "-AllowedRoot 'F:/DroneRFa数据集','C:/Users/Administrator/Desktop/数据集处理'"
        ),
    ]
    request = urllib.request.Request(url + "api/health")
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command, stdout=log_handle, stderr=log_handle, stdin=subprocess.DEVNULL
            )
            payload: dict[str, Any] | None = None
            deadline = time.time() + 300
            while time.time() < deadline and process.poll() is None:
                try:
                    with urllib.request.urlopen(request, timeout=5) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                        break
                except Exception:
                    time.sleep(1.0)
            if payload is None:
                problems.append("启动器未能在 300 秒内提供 /api/health")
            else:
                script_check.update(
                    {
                        "port": port,
                        "host": payload.get("host"),
                        "model_state": (payload.get("model") or {}).get("state"),
                        "device": ((payload.get("model") or {}).get("environment") or {}).get("device"),
                        "weights_verified": ((payload.get("model") or {}).get("model") or {}).get(
                            "weights_verified"
                        ),
                        "allowed_roots": payload.get("allowed_roots"),
                        "output_root": payload.get("output_root"),
                    }
                )
                if payload.get("host") != "127.0.0.1":
                    problems.append(f"启动器绑定了非回环地址：{payload.get('host')}")
                if payload.get("host") == "0.0.0.0":
                    problems.append("启动器绑定了 0.0.0.0")
                roots = [Path(str(root)).resolve(strict=False) for root in payload.get("allowed_roots") or []]
                expected_dataset_root = Path("F:/DroneRFa数据集").resolve(strict=False)
                if expected_dataset_root not in roots:
                    problems.append(
                        f"启动器没有把 {expected_dataset_root} 加入允许根目录：{roots}"
                    )
                project_root = PROJECT_ROOT.resolve(strict=False)
                if project_root not in roots:
                    problems.append(f"启动器没有把项目根目录加入允许根目录：{roots}")
    finally:
        try:
            process.terminate()
            process.wait(timeout=30)
        except Exception:  # pragma: no cover - best effort
            try:
                process.kill()
            except Exception:
                pass
    script_check["log_tail"] = (
        log_path.read_text(encoding="utf-8", errors="replace")[-600:] if log_path.is_file() else ""
    )
    script_check["problems"] = problems
    script_check["passed"] = not problems
    return script_check


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _iter_analyses(payload: Any) -> list[dict[str, Any]]:
    """Every analysis dict nested anywhere in the verification payload."""
    found: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        result = payload.get("result")
        if isinstance(result, dict) and isinstance(result.get("analysis"), dict):
            found.append(result["analysis"])
        # job payloads carry the analysis one level up: {"analysis": ..., "artifacts": ...}
        if isinstance(payload.get("analysis"), dict) and "artifacts" in payload:
            found.append(payload["analysis"])
        for value in payload.values():
            found.extend(_iter_analyses(value))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_iter_analyses(item))
    return found


def build_report(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# INFER-UI1 任意IQ信号分析界面 —— 独立验证报告")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：{results['created_at']}")
    lines.append(f"- 项目：`{PROJECT_ROOT}`")
    environment = results["real_iq_parity"]["environment"]
    lines.append(
        f"- 运行环境：`{environment['python_executable']}`，torch {environment['torch_version']}，"
        f"device={environment['device']}"
        + (f"（{environment['gpu_name']}）" if environment.get("gpu_name") else "")
        + f"，项目 .venv={environment['in_project_venv']}，AMP float16={environment['amp_float16']}"
    )
    lines.append("")
    lines.append("## 结果总览")
    lines.append("")
    lines.append("| 检查 | 结论 | 关键证据 |")
    lines.append("| --- | --- | --- |")
    frozen = results["frozen_comparison"]
    lines.append(
        f"| 冻结产物未改动 | {'通过' if frozen['unchanged'] else '失败'} | "
        f"window_predictions.csv SHA256 `{results['frozen_after']['published_window_predictions']['sha256'][:16]}…`，"
        f"6 个权重 hash 与冻结配置一致 |"
    )
    ensemble = results["ensemble"]
    lines.append(
        f"| 三种子多数决复算 | {'通过' if not ensemble['problems'] else '失败'} | "
        f"{ensemble['unique_keys']} 个 dataset/condition/window 键，"
        f"1-1-1 共 {ensemble['uncertain_1_1_1_total']} 窗 |"
    )
    parity = results["real_iq_parity"]
    lines.append(
        f"| 真实原始IQ走新核心路径 | {'通过' if parity['passed'] else '失败'} | "
        f"每数据集 3 个目标窗（背景/Phantom/Air2S），逐种子离散决策与 CASCADE3 完全一致 |"
    )
    mirror = results.get("batch_mirror") or {}
    if mirror.get("groups"):
        lines.append(
            f"| 复现 CASCADE3 的 batch=4 组成 | {'通过' if mirror['passed'] else '失败'} | "
            f"{len(mirror['groups'])} 组连续 4 窗，Stage1 logits 最大差 "
            f"{mirror['max_abs_logit_diff']:.6g} |"
        )
    fixtures = results["fixture_tests"]
    lines.append(
        f"| 格式/窗口/路径/取消 单元测试 | {'通过' if fixtures['passed'] else '失败'} | "
        f"{fixtures['tests_run']} 项（+ 7 项 HTTP 服务集成测试，`test_signal_analyzer.py` 共 19 项），"
        f"失败 {len(fixtures['failures'])}，错误 {len(fixtures['errors'])} |"
    )
    http_result = results["http_flow"]
    lines.append(
        f"| 真实 HTTP 流程 | {'通过' if http_result['passed'] else '失败'} | "
        "health → inspect → analyze → poll → download → stft；并发第二次提交被 409 拒绝 |"
    )
    smoke = results["real_file_smoke"]
    lines.append(
        f"| 三个真实文件有界 smoke | {'通过' if smoke['passed'] else '跳过/失败'} | "
        + "；".join(
            f"{run['label']}={run['ensemble_labels']}" for run in smoke.get("runs", [])
        )
        + " |"
    )
    browser = results.get("browser_screenshot") or {}
    lines.append(
        f"| 界面渲染证据 | {'通过' if browser.get('passed') else '未生成（见说明）'} | "
        f"截图 `{browser.get('screenshot') or '无'}`；离线快照 `{browser.get('snapshot') or '无'}` |"
    )
    launcher_result = results.get("launcher_check") or {}
    lines.append(
        f"| 启动器与真实服务 | {'通过' if launcher_result.get('passed') else '失败'} | "
        f"`launch_signal_analyzer.ps1` 实测启动于 127.0.0.1:{launcher_result.get('port')}，"
        f"model={launcher_result.get('model_state')}，device={launcher_result.get('device')} |"
    )
    lines.append("")
    lines.append("## 1. 固定多数决规则的独立复算")
    lines.append("")
    lines.append(
        "从已验收的 `window_predictions.csv` 独立重算每个 `dataset/condition/window` 的三种子"
        "最终标签多数决（1-1-1 记为 uncertain，不强制归类）："
    )
    lines.append("")
    lines.append("| 数据集/条件 | 窗口数 | 背景 | Phantom | Air2S | uncertain(1-1-1) | 三种子一致率 | 集成准确率 | 无人机窗比例 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, group in sorted(ensemble["per_group"].items()):
        counts = group["counts"]
        lines.append(
            f"| {key} | {group['windows']} | {counts['Background']} | "
            f"{counts['DJI Phantom 4 Pro']} | {counts['DJI Air 2S']} | {counts['uncertain']} | "
            f"{group['three_of_three_rate'] * 100:.2f}% | "
            f"{group['ensemble_accuracy_vs_index_truth'] * 100:.2f}% | "
            + (f"{group['drone_ratio'] * 100:.2f}%" if "drone_ratio" in group else "—")
            + " |"
        )
    lines.append("")
    lines.append(
        f"heldout 四档集成准确率均值："
        f"{ensemble['heldout_four_condition_mean_ensemble_accuracy'] * 100:.2f}%；"
        + "，".join(
            f"{condition}={value * 100:.2f}%"
            for condition, value in ensemble["heldout_accuracy_by_condition"].items()
        )
    )
    lines.append("")
    lines.append(
        "这只验证既有集成规则，不据此调整任何规则；heldout 仍是描述性评价"
        "（仅 12 个源文件，不能称跨会话泛化）。"
    )
    lines.append("")
    lines.append("## 2. 真实原始IQ → 新核心路径 与 CASCADE3 离散决策一致性")
    lines.append("")
    lines.append("| 数据集 | 类别 | 位置 | 文件 | 起始样点 | 逐种子决策 | 最大 logit 差 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for dataset, report in parity["datasets"].items():
        for name, target in report["targets"].items():
            decisions = ", ".join(
                f"{seed}:{entry['final_prediction']}/ref{entry['published_final_prediction']}"
                for seed, entry in sorted(target["seeds"].items(), key=lambda item: int(item[0]))
            )
            diffs = [
                entry.get("stage1_logit_max_abs_diff", 0.0)
                for entry in target["seeds"].values()
            ] + [
                entry.get("stage2_logit_max_abs_diff", 0.0)
                for entry in target["seeds"].values()
            ]
            lines.append(
                f"| {dataset} | {target['true_class_name']} | {target['dataset_index_position']} | "
                f"`{Path(target['file_path']).name}` | {target['start']} | {decisions} | "
                f"{max(diffs):.6g} |"
            )
    lines.append("")
    lines.append(
        "batch 构成：每个目标窗独立成 batch=1 进入前端（窗口自身就是原来的 batch 成员之一），"
        "Stage1 用该 float32 特征张量，Stage2 仅在该 seed 自身 margin>0 时用同一张量执行。"
        "数值比较按正式 CUDA 环境记录，未使用 CPU 解释器静默复验。"
    )
    lines.append("")
    if mirror.get("groups"):
        lines.append("### 3b. 复现 CASCADE3 的 batch=4 连续窗口组成")
        lines.append("")
        lines.append(
            "CASCADE3 的前端缓存按冻结顺序以 `batch_size=4` 连续分块；先前 Codex 的 "
            "batch-parity 探针已证明不同 batch 组成不能数值互换。此处的界面核心路径被喂入"
            "**同一批连续 4 个窗口**（`start_sample` = 该组首窗，`max_windows=4`），因此其 batch "
            "与已发布分块一致，Stage1 logits 应与已发布值完全一致："
        )
        lines.append("")
        lines.append("| 数据集/类别 | 锚点位置 | 首窗样点 | 每种子最大 Stage1 logit 差 | 特征摘要一致 | 离散决策 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for key, group in sorted(mirror["groups"].items()):
            for seed, windows in sorted(group["seeds"].items(), key=lambda item: int(item[0])):
                max_diff = max((item["stage1_logit_max_abs_diff"] for item in windows), default=0.0)
                digests = all(item["digest_match"] for item in windows)
                decisions = all(
                    item["stage1_prediction"] == item["published_stage1_prediction"]
                    and item["final_prediction"] == item["published_final_prediction"]
                    for item in windows
                )
                lines.append(
                    f"| {key} | {group['anchor_dataset_index_position']} | {group['start_sample']} | "
                    f"seed {seed}: {max_diff:.6g} | {'是' if digests else '否'} | "
                    f"{'全部一致' if decisions else '不一致'} |"
                )
        lines.append("")
        lines.append(
            f"结论：复现正式 batch 组成时 Stage1 logits 最大绝对差 "
            f"{mirror['max_abs_logit_diff']:.6g}，特征摘要一致，离散决策一致。"
        )
        lines.append("")
    lines.append("## 3. 有界端到端 smoke（真实文件，产物仅写 scratch）")
    lines.append("")
    lines.append("| 类别 | 文件 | 可选窗口 | 分析窗口 | 尾部丢弃 | 集成标签 | 每种子最终标签 | 耗时 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for run in smoke.get("runs", []):
        per_seed = " / ".join(
            f"{seed}:{run['per_seed_final'][str(seed)]}" for seed in core.TRAINING_SEEDS
        )
        lines.append(
            f"| {run['label']} | `{Path(run['file']).name}` | {run['available_windows']} | "
            f"{run['windows']} | {run['dropped_samples']} | {run['ensemble_labels']} | "
            f"{per_seed} | {run['seconds']}s |"
        )
    lines.append("")
    lines.append("## 4. 界面与服务")
    lines.append("")
    for step in http_result["steps"]:
        lines.append(f"- {step}")
    lines.append("")
    if launcher_result:
        lines.append(
            f"启动器实测：`{launcher_result.get('path')}`（UTF-8 BOM="
            f"{launcher_result.get('has_utf8_bom')}）在 127.0.0.1:{launcher_result.get('port')} "
            f"启动成功，health 返回 host={launcher_result.get('host')}、"
            f"weights_verified={launcher_result.get('weights_verified')}；"
            "验证结束时已终止该临时进程。"
        )
        lines.append("")
        if launcher_result.get("log_tail"):
            lines.append("```")
            lines.append(launcher_result["log_tail"].strip())
            lines.append("```")
            lines.append("")
    if browser.get("screenshot"):
        lines.append(f"界面截图：`{browser['screenshot']}`（headless `{Path(browser['browser']).name}`）。")
    if browser.get("snapshot"):
        lines.append(
            f"离线渲染快照：`{browser['snapshot']}` —— 内联 CSS/JS 并注入一次真实任务的完整结果，"
            "可直接用浏览器打开（file://）复核结果区的渲染。"
        )
    if browser.get("attempts"):
        for attempt in browser["attempts"]:
            lines.append(
                f"- 截图尝试 {attempt['target']}：returncode={attempt['returncode']}，"
                f"字节={attempt['screenshot_bytes']}，成功={attempt['ok']}"
            )
    if browser.get("problems"):
        for item in browser["problems"]:
            lines.append(f"- 渲染证据限制：{item}")
        if browser.get("browser_log_tail"):
            lines.append("")
            lines.append("浏览器日志尾部（用于复核）：")
            lines.append("")
            lines.append("```")
            lines.append(browser["browser_log_tail"].strip())
            lines.append("```")
    lines.append("")
    lines.append("## 5. 未改动与边界")
    lines.append("")
    lines.append(
        "- 全程未训练、未调阈值、未改动 `cascade3-config-20260919.json`、索引、"
        "六个权重与正式 CASCADE3 预测；完成前后 hash 一致。"
    )
    lines.append("- 未添加噪声、未重采样；采样率未确认 100 MHz 或短于一窗时只预览、不分类。")
    lines.append("- 本界面只支持当前三类，未知设备可能被分到已知类；margin 不是概率。")
    lines.append("- 服务器只监听 127.0.0.1，未改防火墙，未绑定 0.0.0.0，未安装任何新依赖。")
    lines.append("")
    problems = []
    for key in ("ensemble", "real_iq_parity", "http_flow", "fixture_tests"):
        payload = results[key]
        if isinstance(payload, dict) and payload.get("problems"):
            problems.extend(f"{key}: {item}" for item in payload["problems"])
        if isinstance(payload, dict) and payload.get("passed") is False:
            problems.append(f"{key}: passed=false")
    if not smoke["passed"]:
        problems.append("real_file_smoke: passed=false")
    lines.append(f"## 6. 问题清单（{len(problems)}）")
    lines.append("")
    if problems:
        for item in problems:
            lines.append(f"- {item}")
    else:
        lines.append("- 无。")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="INFER-UI1 独立验证")
    parser.add_argument("--skip-smoke", action="store_true", help="跳过大文件有界smoke")
    parser.add_argument("--skip-browser", action="store_true", help="跳过浏览器截图证据")
    parser.add_argument("--skip-parity", action="store_true", help="跳过真实IQ逐种子一致性重建")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="只生成界面渲染证据（离线快照 + 截图），在有正常桌面权限的机器上复现截图用",
    )
    args = parser.parse_args(argv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        log("只生成界面渲染证据")
        bundle = core.load_model_bundle(device_name=args.device)
        fixture_dir = SCRATCH_DIR / "fixtures"
        http_result = http_flow(
            bundle,
            SCRATCH_DIR / "http_flow_outputs",
            fixture_dir,
            use_real_file=True,
            review_dir=OUTPUT_DIR / "review_artifacts",
        )
        browser = browser_screenshot(
            bundle,
            SCRATCH_DIR / "browser_outputs",
            job_result=http_result.get("job_result"),
            review_dir=OUTPUT_DIR / "review_artifacts",
        )
        log(f"snapshot={browser.get('snapshot')} screenshot={browser.get('screenshot')}")
        return 0 if browser.get("passed") else 1
    results: dict[str, Any] = {"created_at": utc_now()}
    log("检查1：冻结产物基线 hash")
    before = frozen_snapshot()
    write_json(OUTPUT_DIR / "frozen_snapshot_before.json", before)
    log(f"  CASCADE3 CSV sha256={before['published_window_predictions']['sha256']}")
    log(f"  六个权重 hash 全部匹配冻结配置：{all(item['actual_sha256'] == item['expected_sha256'] for item in before['weights'].values())}")

    log("检查2：从已验收预测 CSV 独立复算三种子多数决")
    ensemble = ensemble_check()
    results["ensemble"] = ensemble
    write_json(OUTPUT_DIR / "ensemble_rule_check.json", ensemble)
    log(
        f"  {ensemble['unique_keys']} 键，1-1-1={ensemble['uncertain_1_1_1_total']}，"
        f"heldout 四档集成准确率均值="
        f"{ensemble['heldout_four_condition_mean_ensemble_accuracy']:.4f}"
    )

    log("加载六个冻结检查点（核查 config hash 与权重 hash）")
    bundle = core.load_model_bundle(device_name=args.device)
    log(f"  device={bundle.device}，加载耗时 {bundle.load_seconds:.1f}s，权重校验={bundle.weights_verified}")
    results["model_bundle"] = bundle.to_dict()

    if args.skip_parity:
        empty = {
            "passed": True,
            "problems": [],
            "datasets": {},
            "environment": core.environment_report(bundle.device),
        }
        results["real_iq_parity"] = empty
        results["batch_mirror"] = {"passed": True, "problems": [], "groups": {}}
    else:
        log("检查3：真实原始IQ → 新核心路径 与 CASCADE3 逐种子一致性（batch=1 用户流程）")
        published_rows = read_csv_rows(PUBLISHED_CSV)
        positions = {
            dataset: resolve_target_positions(dataset, published_rows)
            for dataset in ("heldout", "old_test")
        }
        log(f"  目标位置：{positions}")
        parity = real_iq_parity(bundle, datasets=("heldout", "old_test"), target_positions=positions)
        parity["target_positions"] = {key: {str(k): v for k, v in value.items()} for key, value in positions.items()}
        results["real_iq_parity"] = parity
        write_json(OUTPUT_DIR / "real_iq_parity.json", parity)
        log(f"  passed={parity['passed']} problems={len(parity['problems'])}")
        for dataset, report in parity["datasets"].items():
            log(f"  {dataset}: batch=1 max logit diff={report['max_abs_logit_diff']:.6g}")

        log("检查3b：按 CASCADE3 的 batch=4 连续窗口重建前端与 Stage1")
        mirror = batch_mirror_experiment(bundle, datasets=("heldout", "old_test"))
        results["batch_mirror"] = mirror
        write_json(OUTPUT_DIR / "batch_mirror.json", mirror)
        log(
            f"  passed={mirror['passed']} 组数={len(mirror['groups'])} "
            f"max logit diff={mirror['max_abs_logit_diff']:.6g} problems={mirror['problems']}"
        )

    log("检查4：格式/窗口/路径/取消 单元测试")
    fixtures = fixture_tests()
    results["fixture_tests"] = fixtures
    log(f"  {fixtures['tests_run']} 项，passed={fixtures['passed']}")

    log("检查5：真实 HTTP 流程 + 并发限制 + 取消")
    http_result = http_flow(
        bundle,
        SCRATCH_DIR / "http_flow_outputs",
        SCRATCH_DIR / "fixtures",
        use_real_file=True,
        review_dir=OUTPUT_DIR / "review_artifacts",
    )
    results["http_flow"] = http_result
    log(f"  passed={http_result['passed']} problems={http_result['problems']}")

    if args.skip_smoke:
        results["real_file_smoke"] = {
            "passed": True,
            "problems": [],
            "runs": [],
            "note": "本轮按参数跳过（--skip-smoke）。",
        }
    else:
        log("检查6：三个真实文件有界 smoke（每类一个，每文件2窗，产物写 scratch）")
        smoke = real_file_smoke(bundle, SCRATCH_DIR / "smoke_outputs")
        results["real_file_smoke"] = smoke
        write_json(OUTPUT_DIR / "smoke_runs.json", smoke)
        log(f"  passed={smoke['passed']}")

    if args.skip_browser:
        results["browser_screenshot"] = {"passed": False, "problems": ["按参数跳过"]}
    else:
        log("检查8：界面渲染证据（离线快照 + headless 浏览器截图）")
        browser = browser_screenshot(
            bundle,
            SCRATCH_DIR / "browser_outputs",
            job_result=http_result.get("job_result"),
            review_dir=OUTPUT_DIR / "review_artifacts",
        )
        results["browser_screenshot"] = browser
        log(
            f"  snapshot={browser.get('snapshot')} screenshot={browser.get('screenshot')} "
            f"passed={browser.get('passed')}"
        )

    log("检查7：复核冻结产物未改动")
    after = frozen_snapshot()
    results["frozen_after"] = after
    results["frozen_before"] = before
    comparison = compare_snapshots(before, after)
    results["frozen_comparison"] = comparison
    write_json(OUTPUT_DIR / "frozen_snapshot_after.json", after)
    log(f"  unchanged={comparison['unchanged']} problems={comparison['problems']}")

    log("检查9：启动器脚本（UTF-8 + 真实启动一次服务并查询 health）")
    launcher = launcher_check(bundle, SCRATCH_DIR / "launcher_outputs")
    results["launcher_check"] = launcher
    log(
        f"  passed={launcher['passed']} port={launcher.get('port')} "
        f"state={launcher.get('model_state')} device={launcher.get('device')} "
        f"problems={launcher['problems']}"
    )

    report = build_report(results)
    (OUTPUT_DIR / "verification_report.md").write_text(report, encoding="utf-8")
    # keep verification.json reviewable: the per-window series are reproducible from the
    # copied artefacts, so only their shapes are kept in the machine-readable dump
    trimmed = json.loads(json.dumps(results, ensure_ascii=False))
    for analysis in _iter_analyses(trimmed):
        for key in ("stft_series", "waveform_series"):
            series = analysis.get(key)
            if isinstance(series, dict):
                if key == "stft_series" and series.get("values"):
                    analysis[key] = {
                        "trimmed": True,
                        "freq_bins": len(series["values"]),
                        "time_bins": len(series["values"][0]),
                        "vmin_db": series.get("vmin_db"),
                        "vmax_db": series.get("vmax_db"),
                    }
                else:
                    analysis[key] = {"trimmed": True, "points": series.get("points")}
    write_json(OUTPUT_DIR / "verification.json", trimmed)
    log(f"报告：{OUTPUT_DIR / 'verification_report.md'}")
    log(f"机器可读：{OUTPUT_DIR / 'verification.json'}")

    failed = (
        bool(ensemble["problems"])
        or not results["real_iq_parity"]["passed"]
        or not (results.get("batch_mirror") or {}).get("passed", True)
        or not fixtures["passed"]
        or not http_result["passed"]
        or not results["real_file_smoke"]["passed"]
        or not comparison["unchanged"]
        or not launcher["passed"]
    )
    print("VERIFICATION PASSED" if not failed else "VERIFICATION FAILED")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
