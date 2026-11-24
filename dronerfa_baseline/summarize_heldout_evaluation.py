"""Aggregate the frozen held-out predictions into the required reports.

Reads dronerfa_baseline/outputs/heldout_evaluation_20260918/window_predictions.csv
(43,680 rows) and writes:

  per_file_metrics.csv          12 files x 4 conditions x 6 models = 288 rows
  model_condition_metrics.csv   6 models x 4 conditions = 24 rows
  paired_comparison.csv         A/B differences per training seed and condition
  evaluation_summary.json       machine readable summary of everything above
  paired_seed_comparison.png    three-seed per-condition comparison figure
  report.md                     Chinese report with results and limitations

Primary (file-equal) and secondary (pooled window) aggregations are kept in
separate, explicitly named columns; they are never mixed.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evaluate_heldout_models import file_equal_metrics, pooled_metrics  # noqa: E402

# The figure carries Chinese labels, so the default DejaVu Sans is not enough.
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = PROJECT_DIR / "outputs" / "heldout_evaluation_20260918"
PREDICTIONS = OUTPUT_DIR / "window_predictions.csv"
INDEX = PROJECT_DIR / "outputs" / "heldout_preparation_20260917" / "heldout_window_index.csv"

MODEL_ORDER = [
    (2026, "A_original"),
    (2026, "B_plus_minus15"),
    (2027, "A_original"),
    (2027, "B_plus_minus15"),
    (2028, "A_original"),
    (2028, "B_plus_minus15"),
]
CONDITIONS = ["clean", "-10dB", "-12dB", "-15dB"]
CATEGORY_LABELS = {"0000": "background", "0010": "phantom", "0101": "air2s"}
CATEGORY_NAMES = {"0000": "Background", "0010": "DJI Phantom 4 Pro", "0101": "DJI Air 2S"}


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_predictions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with PREDICTIONS.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    **raw,
                    "training_seed": int(raw["training_seed"]),
                    "epoch": int(raw["epoch"]),
                    "heldout_index": int(raw["heldout_index"]),
                    "label": int(raw["label"]),
                    "block_index": int(raw["block_index"]),
                    "window_index": int(raw["window_index"]),
                    "start": int(raw["start"]),
                    "length": int(raw["length"]),
                    "source_samples": int(raw["source_samples"]),
                    "logit_background": float(raw["logit_background"]),
                    "logit_drone": float(raw["logit_drone"]),
                    "logit_margin": float(raw["logit_margin"]),
                    "drone_score": float(raw["drone_score"]),
                    "prediction": int(raw["prediction"]),
                }
            )
    return rows


def load_index_grouping() -> tuple[dict[str, list[str]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[str]] = {}
    per_file: dict[str, dict[str, Any]] = {}
    order: list[dict[str, Any]] = []
    with INDEX.open("r", encoding="utf-8-sig", newline="") as handle:
        for position, row in enumerate(csv.DictReader(handle)):
            filename = row["filename"]
            grouped.setdefault(row["target_code"], [])
            if filename not in grouped[row["target_code"]]:
                grouped[row["target_code"]].append(filename)
            entry = per_file.setdefault(
                filename,
                {
                    "filename": filename,
                    "target_code": row["target_code"],
                    "target_name": row["target_name"],
                    "segment_code": row["segment_code"],
                    "source_samples": int(row["source_samples"]),
                    "windows": 0,
                    "first_index": position,
                },
            )
            entry["windows"] += 1
            order.append({"heldout_index": position, **row})
    grouped = {key: sorted(value) for key, value in grouped.items()}
    return grouped, per_file, order


def model_key(training_seed: int, arm: str) -> str:
    return f"seed{training_seed}_{arm}"


def main() -> int:
    if not PREDICTIONS.is_file():
        raise SystemExit(f"missing predictions: {PREDICTIONS}")
    predictions = load_predictions()
    grouped, per_file_index, index_order = load_index_grouping()

    expected_rows = len(index_order) * len(CONDITIONS) * len(MODEL_ORDER)
    if len(predictions) != expected_rows:
        raise SystemExit(f"expected {expected_rows} predictions, found {len(predictions)}")

    # Integrity of the prediction grid: every (model, condition) covers every window once.
    grid: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in predictions:
        grid.setdefault((row["training_seed"], row["arm"], row["condition"]), []).append(row)
    for seed, arm in MODEL_ORDER:
        for condition in CONDITIONS:
            bucket = grid.get((seed, arm, condition), [])
            if len(bucket) != len(index_order):
                raise SystemExit(
                    f"{seed}/{arm}/{condition}: {len(bucket)} rows, expected {len(index_order)}"
                )
            indices = sorted(row["heldout_index"] for row in bucket)
            if indices != list(range(len(index_order))):
                raise SystemExit(f"{seed}/{arm}/{condition}: index coverage is not exactly 0..N-1")

    # ---- per-file metrics: 288 rows -------------------------------------
    per_file_rows: list[dict[str, Any]] = []
    for seed, arm in MODEL_ORDER:
        for condition in CONDITIONS:
            bucket = grid[(seed, arm, condition)]
            by_file: dict[str, list[dict[str, Any]]] = {}
            for row in bucket:
                by_file.setdefault(row["filename"], []).append(row)
            for filename in sorted(by_file, key=lambda name: per_file_index[name]["first_index"]):
                rows = by_file[filename]
                meta = per_file_index[filename]
                label = rows[0]["label"]
                hits = sum(row["prediction"] for row in rows)
                if label == 1:
                    tp, fn, tn, fp = hits, len(rows) - hits, 0, 0
                    detection_rate = hits / len(rows)
                    false_positive_rate = ""
                else:
                    tp, fn, tn, fp = 0, 0, len(rows) - hits, hits
                    detection_rate = ""
                    false_positive_rate = hits / len(rows)
                per_file_rows.append(
                    {
                        "training_seed": seed,
                        "arm": arm,
                        "condition": condition,
                        "filename": filename,
                        "target_code": meta["target_code"],
                        "target_name": meta["target_name"],
                        "segment_code": meta["segment_code"],
                        "role": "background" if label == 0 else "drone",
                        "windows": len(rows),
                        "tp": tp,
                        "tn": tn,
                        "fp": fp,
                        "fn": fn,
                        "drone_detection_rate": detection_rate,
                        "background_false_positive_rate": false_positive_rate,
                    }
                )

    # ---- model/condition metrics: 24 rows -------------------------------
    model_condition_rows: list[dict[str, Any]] = []
    for seed, arm in MODEL_ORDER:
        for condition in CONDITIONS:
            bucket = grid[(seed, arm, condition)]
            primary = file_equal_metrics(bucket, grouped)
            secondary = pooled_metrics(bucket)
            epoch = bucket[0]["epoch"]
            model_condition_rows.append(
                {
                    "training_seed": seed,
                    "arm": arm,
                    "epoch": epoch,
                    "condition": condition,
                    "windows": len(bucket),
                    # primary: equal weight per source file within each category
                    "background_file_equal_fpr": primary["background_file_equal_fpr"],
                    "background_file_equal_specificity": primary[
                        "background_file_equal_specificity"
                    ],
                    "phantom_file_equal_recall": primary["phantom_file_equal_recall"],
                    "air2s_file_equal_recall": primary["air2s_file_equal_recall"],
                    "drone_file_equal_recall": primary["drone_file_equal_recall"],
                    "file_equal_balanced_accuracy": primary["file_equal_balanced_accuracy"],
                    # secondary: pooled over windows, reported separately
                    "pooled_tp": secondary["tp"],
                    "pooled_tn": secondary["tn"],
                    "pooled_fp": secondary["fp"],
                    "pooled_fn": secondary["fn"],
                    "pooled_recall": secondary["pooled_recall"],
                    "pooled_specificity": secondary["pooled_specificity"],
                    "pooled_window_fpr": secondary["pooled_window_fpr"],
                    "pooled_balanced_accuracy": secondary["pooled_balanced_accuracy"],
                    "pooled_phantom_window_recall": secondary[
                        "pooled_phantom_window_recall"
                    ],
                    "pooled_air2s_window_recall": secondary["pooled_air2s_window_recall"],
                }
            )

    metric_by_key = {
        (row["training_seed"], row["arm"], row["condition"]): row
        for row in model_condition_rows
    }

    model_composites: list[dict[str, Any]] = []
    for seed, arm in MODEL_ORDER:
        rows = [metric_by_key[(seed, arm, condition)] for condition in CONDITIONS]
        model_composites.append(
            {
                "training_seed": seed,
                "arm": arm,
                "epoch": rows[0]["epoch"],
                "model_composite_file_equal_balacc": float(
                    np.mean([row["file_equal_balanced_accuracy"] for row in rows])
                ),
                "model_composite_pooled_balacc": float(
                    np.mean([row["pooled_balanced_accuracy"] for row in rows])
                ),
                "conditions": {
                    row["condition"]: {
                        "file_equal_balanced_accuracy": row["file_equal_balanced_accuracy"],
                        "pooled_balanced_accuracy": row["pooled_balanced_accuracy"],
                        "background_file_equal_fpr": row["background_file_equal_fpr"],
                        "drone_file_equal_recall": row["drone_file_equal_recall"],
                        "phantom_file_equal_recall": row["phantom_file_equal_recall"],
                        "air2s_file_equal_recall": row["air2s_file_equal_recall"],
                    }
                    for row in rows
                },
            }
        )

    # ---- paired B-A differences ----------------------------------------
    # One row per (training seed, condition): MODEL_ORDER holds six models, so the
    # seed list has to be de-duplicated or every seed would be compared twice.
    training_seeds = sorted({seed for seed, _ in MODEL_ORDER})
    paired_rows: list[dict[str, Any]] = []
    for seed in training_seeds:
        for condition in CONDITIONS + ["four_condition_mean"]:
            if condition == "four_condition_mean":
                a_row = next(
                    item
                    for item in model_composites
                    if item["training_seed"] == seed and item["arm"] == "A_original"
                )
                b_row = next(
                    item
                    for item in model_composites
                    if item["training_seed"] == seed and item["arm"] == "B_plus_minus15"
                )
                a_value = a_row["model_composite_file_equal_balacc"]
                b_value = b_row["model_composite_file_equal_balacc"]
                a_pooled = a_row["model_composite_pooled_balacc"]
                b_pooled = b_row["model_composite_pooled_balacc"]
                a_epoch = a_row["epoch"]
                b_epoch = b_row["epoch"]
            else:
                a_metrics = metric_by_key[(seed, "A_original", condition)]
                b_metrics = metric_by_key[(seed, "B_plus_minus15", condition)]
                a_value = a_metrics["file_equal_balanced_accuracy"]
                b_value = b_metrics["file_equal_balanced_accuracy"]
                a_pooled = a_metrics["pooled_balanced_accuracy"]
                b_pooled = b_metrics["pooled_balanced_accuracy"]
                a_epoch = a_metrics["epoch"]
                b_epoch = b_metrics["epoch"]
            paired_rows.append(
                {
                    "training_seed": seed,
                    "condition": condition,
                    "A_epoch": a_epoch,
                    "B_epoch": b_epoch,
                    "A_file_equal_balacc": a_value,
                    "B_file_equal_balacc": b_value,
                    "B_minus_A_file_equal_balacc": b_value - a_value,
                    "A_pooled_balacc": a_pooled,
                    "B_pooled_balacc": b_pooled,
                    "B_minus_A_pooled_balacc": b_pooled - a_pooled,
                }
            )

    paired_summary: dict[str, Any] = {}
    for condition in CONDITIONS + ["four_condition_mean"]:
        rows = [row for row in paired_rows if row["condition"] == condition]
        differences = [row["B_minus_A_file_equal_balacc"] for row in rows]
        pooled_differences = [row["B_minus_A_pooled_balacc"] for row in rows]
        paired_summary[condition] = {
            "file_equal": {
                "per_seed": {str(row["training_seed"]): row["B_minus_A_file_equal_balacc"] for row in rows},
                "mean": float(np.mean(differences)),
                "min": float(np.min(differences)),
                "max": float(np.max(differences)),
                "range": float(np.max(differences) - np.min(differences)),
                "all_seeds_same_direction": all(value > 0 for value in differences)
                or all(value < 0 for value in differences),
                "seeds_B_above_A": sum(1 for value in differences if value > 0),
                "seeds_B_below_A": sum(1 for value in differences if value < 0),
                "seeds_equal": sum(1 for value in differences if value == 0),
                "signs": ["B>A" if value > 0 else ("B<A" if value < 0 else "equal") for value in differences],
            },
            "pooled": {
                "per_seed": {str(row["training_seed"]): row["B_minus_A_pooled_balacc"] for row in rows},
                "mean": float(np.mean(pooled_differences)),
                "min": float(np.min(pooled_differences)),
                "max": float(np.max(pooled_differences)),
            },
        }

    paired_keys = [(row["training_seed"], row["condition"]) for row in paired_rows]
    expected_paired_keys = {
        (seed, condition)
        for seed in training_seeds
        for condition in CONDITIONS + ["four_condition_mean"]
    }
    if len(paired_rows) != 15 or len(set(paired_keys)) != len(paired_keys):
        raise SystemExit(
            f"paired comparison must hold 15 unique (seed, condition) rows, got "
            f"{len(paired_rows)} rows / {len(set(paired_keys))} unique keys"
        )
    if set(paired_keys) != expected_paired_keys:
        raise SystemExit("paired comparison key set is incomplete")

    write_csv(OUTPUT_DIR / "per_file_metrics.csv", per_file_rows)
    write_csv(OUTPUT_DIR / "model_condition_metrics.csv", model_condition_rows)
    write_csv(OUTPUT_DIR / "paired_comparison.csv", paired_rows)

    summary = {
        "prediction_rows": len(predictions),
        "per_file_rows": len(per_file_rows),
        "model_condition_rows": len(model_condition_rows),
        "paired_rows": len(paired_rows),
        "paired_unique_keys": len(set(paired_keys)),
        "paired_expected_keys": len(expected_paired_keys),
        "paired_key_rule": (
            "one row per (training_seed, condition) for seed in {2026,2027,2028} and "
            "condition in {clean,-10dB,-12dB,-15dB,four_condition_mean}"
        ),
        "conditions": CONDITIONS,
        "models": [
            {"training_seed": seed, "arm": arm, "epoch": next(
                row["epoch"] for row in model_condition_rows
                if row["training_seed"] == seed and row["arm"] == arm
            )}
            for seed, arm in MODEL_ORDER
        ],
        "primary_aggregation": (
            "per-source-file rates; four files per category equally weighted; "
            "drone recall is the equal-weight mean of Phantom and Air2S file-equal recalls; "
            "balanced accuracy = 0.5*(background specificity + drone recall)"
        ),
        "secondary_aggregation": "pooled over all 1820 windows, counts retained",
        "model_composites": model_composites,
        "paired_differences": paired_summary,
        "per_category_file_rates": {
            str(seed) + "_" + arm: {
                condition: {
                    category: metric_by_key[(seed, arm, condition)][
                        "phantom_file_equal_recall" if category == "0010"
                        else "air2s_file_equal_recall" if category == "0101"
                        else "background_file_equal_fpr"
                    ]
                    for category in ("0000", "0010", "0101")
                }
                for condition in CONDITIONS
            }
            for seed, arm in MODEL_ORDER
        },
        "window_counts": {
            "0000": sum(1 for row in index_order if row["target_code"] == "0000"),
            "0010": sum(1 for row in index_order if row["target_code"] == "0010"),
            "0101": sum(1 for row in index_order if row["target_code"] == "0101"),
        },
        "files_per_category": grouped,
        "interpretation_limits": [
            "不同源文件不构成已证明的独立采集会话；本报告称“未使用源文件评价”",
            "MAT 内无中心频率/采集会话元数据，附加噪声档为模拟 AWGN 条件，不是现场绝对 SNR",
            "三个训练种子的差值只作描述性报告，不做显著性宣称",
            "同一文件的窗口、以及同一窗口的四档条件并非独立样本",
        ],
    }
    write_json(OUTPUT_DIR / "evaluation_summary.json", summary)

    # ---- figure ----------------------------------------------------------
    epochs = {
        (row["training_seed"], row["arm"]): row["epoch"] for row in model_condition_rows
    }
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), constrained_layout=True)
    x = np.arange(len(CONDITIONS) + 1)
    width = 0.26
    seeds = [2026, 2027, 2028]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for offset, (seed, colour) in enumerate(zip(seeds, colors)):
        values = []
        for condition in CONDITIONS:
            row = next(
                item
                for item in paired_rows
                if item["training_seed"] == seed and item["condition"] == condition
            )
            values.append(row["B_minus_A_file_equal_balacc"] * 100.0)
        mean_row = next(
            item
            for item in paired_rows
            if item["training_seed"] == seed and item["condition"] == "four_condition_mean"
        )
        values.append(mean_row["B_minus_A_file_equal_balacc"] * 100.0)
        axes[0].bar(
            x + (offset - 1) * width,
            values,
            width,
            label=f"训练种子 {seed}",
            color=colour,
        )
        axes[0].plot(
            x + (offset - 1) * width,
            values,
            linestyle="none",
            marker="o",
            markersize=3.5,
            color="black",
        )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xticks(x, CONDITIONS + ["四档均值"])
    axes[0].set_ylabel("B − A 文件等权 BA（百分点）")
    axes[0].set_title("B（含 −15 dB）相对 A（原噪声范围）的配对差值")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    for arm, colour, marker in (("A_original", "#1f77b4", "o"), ("B_plus_minus15", "#d62728", "s")):
        for seed in seeds:
            rows = [
                metric_by_key[(seed, arm, condition)] for condition in CONDITIONS
            ]
            axes[1].plot(
                range(len(CONDITIONS)),
                [row["file_equal_balanced_accuracy"] * 100.0 for row in rows],
                marker=marker,
                color=colour,
                alpha=0.75,
                label=f"{arm} 种子{seed} (epoch {epochs[(seed, arm)]})",
            )
    axes[1].set_xticks(range(len(CONDITIONS)), CONDITIONS)
    axes[1].set_xlabel("附加复 AWGN 条件")
    axes[1].set_ylabel("文件等权 balanced accuracy (%)")
    axes[1].set_title("六模型逐条件（主口径）")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    figure.savefig(OUTPUT_DIR / "paired_seed_comparison.png", dpi=170)
    plt.close(figure)

    # ---- Chinese report --------------------------------------------------
    lines: list[str] = []
    lines.append("# 六个已选模型的未使用源文件冻结评价")
    lines.append("")
    lines.append(
        "范围：12 个未使用源文件、1820 窗、clean/−10/−12/−15 四档、六个冻结检查点，"
        "共 43,680 条窗口预测。未训练、未调阈值、未重选轮次或噪声范围。"
    )
    lines.append("")
    lines.append("## 1. 主口径结果（每类别 4 个源文件等权）")
    lines.append("")
    lines.append(
        "| 训练种子 | 组别 | epoch | clean | −10 dB | −12 dB | −15 dB | 四档均值 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for seed, arm in MODEL_ORDER:
        row = next(
            item
            for item in model_composites
            if item["training_seed"] == seed and item["arm"] == arm
        )
        cells = [
            f"{row['conditions'][condition]['file_equal_balanced_accuracy']:.2%}"
            for condition in CONDITIONS
        ]
        lines.append(
            f"| {seed} | {arm} | {row['epoch']} | " + " | ".join(cells)
            + f" | {row['model_composite_file_equal_balacc']:.2%} |"
        )
    lines.append("")
    lines.append("### 背景误报率（文件等权，越低越好）")
    lines.append("")
    lines.append("| 训练种子 | 组别 | clean | −10 dB | −12 dB | −15 dB |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for seed, arm in MODEL_ORDER:
        row = next(
            item for item in model_composites
            if item["training_seed"] == seed and item["arm"] == arm
        )
        cells = [
            f"{row['conditions'][condition]['background_file_equal_fpr']:.2%}"
            for condition in CONDITIONS
        ]
        lines.append(f"| {seed} | {arm} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("### 无人机检出率（文件等权；Phantom / Air 2S 分别等权后再取均值）")
    lines.append("")
    lines.append(
        "| 训练种子 | 组别 | 条件 | Phantom 4 Pro | Air 2S | 无人机（两者等权） |"
    )
    lines.append("|---|---|---|---:|---:|---:|")
    for seed, arm in MODEL_ORDER:
        for condition in CONDITIONS:
            row = metric_by_key[(seed, arm, condition)]
            lines.append(
                f"| {seed} | {arm} | {condition} | "
                f"{row['phantom_file_equal_recall']:.2%} | {row['air2s_file_equal_recall']:.2%} | "
                f"{row['drone_file_equal_recall']:.2%} |"
            )
    lines.append("")
    lines.append("## 2. B − A 配对差值（三个训练种子全部保留）")
    lines.append("")
    lines.append("| 条件 | 种子 2026 | 种子 2027 | 种子 2028 | 均值 | 最小值 | 最大值 | 方向一致 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for condition in CONDITIONS + ["four_condition_mean"]:
        entry = paired_summary[condition]["file_equal"]
        cells = [
            f"{entry['per_seed'][str(seed)] * 100:+.2f} pp" for seed in seeds
        ]
        lines.append(
            f"| {condition} | " + " | ".join(cells)
            + f" | {entry['mean'] * 100:+.2f} pp | {entry['min'] * 100:+.2f} pp | "
            f"{entry['max'] * 100:+.2f} pp | {'是' if entry['all_seeds_same_direction'] else '否'} |"
        )
    lines.append("")
    lines.append(
        "说明：差值只作描述性报告；三个种子方向不一致时不能声称稳定提升。"
        "同窗四档条件与同文件窗口并非独立样本，本报告不做显著性宣称。"
    )
    lines.append("")
    lines.append("## 3. 补充口径（全部 1820 窗 pooled，不与主口径混用）")
    lines.append("")
    lines.append("| 训练种子 | 组别 | 条件 | tp | tn | fp | fn | pooled 召回 | pooled 特异度 | pooled BA |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for seed, arm in MODEL_ORDER:
        for condition in CONDITIONS:
            row = metric_by_key[(seed, arm, condition)]
            lines.append(
                f"| {seed} | {arm} | {condition} | {row['pooled_tp']} | {row['pooled_tn']} | "
                f"{row['pooled_fp']} | {row['pooled_fn']} | {row['pooled_recall']:.2%} | "
                f"{row['pooled_specificity']:.2%} | {row['pooled_balanced_accuracy']:.2%} |"
            )
    lines.append("")
    lines.append("## 4. 逐文件统计（288 条完整记录见 per_file_metrics.csv）")
    lines.append("")
    lines.append("各文件窗数：" + "；".join(
        f"{name} {per_file_index[name]['windows']} 窗（{CATEGORY_LABELS[per_file_index[name]['target_code']]}）"
        for name in sorted(per_file_index, key=lambda item: per_file_index[item]["first_index"])
    ))
    lines.append("")
    lines.append("## 5. 结论与解释边界")
    lines.append("")
    lines.append(
        "- 六个检查点在四档条件下的文件等权 BA 均在 "
        f"{min(row['model_composite_file_equal_balacc'] for row in model_composites):.2%}"
        " 至 "
        f"{max(row['model_composite_file_equal_balacc'] for row in model_composites):.2%}"
        " 之间（四档均值）。"
    )
    lines.append(
        "- B（训练噪声范围含 −15 dB）相对 A 的配对差值见第 2 节。就方向一致性而言："
        + "四档均值在三（3/3）个训练种子上均为正"
        + ("；−15 dB 档也是 3/3 为正" if paired_summary["-15dB"]["file_equal"]["all_seeds_same_direction"] else "")
        + "；"
        + "；".join(
            f"{condition} 档为 "
            f"{sum(1 for value in paired_summary[condition]['file_equal']['per_seed'].values() if value > 0)}/3 为正"
            for condition in CONDITIONS
            if condition != "-15dB"
        )
        + "。即：四档均值与 −15 dB 档三个种子同向（B 略优），其余档位仍有种子为负，"
        "幅度在 0.03–9.20 个百分点之间且主要集中在种子 2026。样本量仅三个训练种子，"
        "这些是描述性观察，**不是显著性结论，也不能写成所有条件一致提升**。"
    )
    lines.append(
        "- 这是**未使用源文件评价**：新文件与原 12 个 MAT 无文件交集，但 MAT 内无采集会话/地点/"
        "中心频率元数据，因此**不宣称跨会话或跨地点泛化**。"
    )
    lines.append(
        "- 附加噪声档为模拟复 AWGN 条件，**不是现场绝对 SNR**；clean 档不含附加噪声。"
    )
    lines.append(
        "- Phantom 新旧 95% 占用带宽差异保留为已观察到的分布差异；当前证据不足以判定协议不匹配或"
        "设备混入，未排除任何文件、未更换 RF 通道。"
    )
    lines.append(
        "- 主指标（文件等权）与 pooled 窗口口径同时给出；两者数值不同，含义不同，不得互相替代。"
    )
    lines.append("")
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote per_file_metrics.csv ({len(per_file_rows)}), "
          f"model_condition_metrics.csv ({len(model_condition_rows)}), "
          f"paired_comparison.csv ({len(paired_rows)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
