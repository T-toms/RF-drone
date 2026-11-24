"""Audit and summarize paired training-seed differences without pooling windows."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_noise_ab import resolve_validation_seed, validate_reference_cache, write_csv, write_json


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
            "minimum": float(values.min()), "maximum": float(values.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    out = args.output_dir
    replication = read_json(out / "replication_manifest.json")
    assert replication["status"] == "completed"
    reference = Path(replication["reference_run"])
    directories = [reference] + [out / f"seed_{s}" for s in replication["new_training_seeds"]]
    reference_manifest = read_json(reference / "experiment_manifest.json")
    reference_cache = read_json(reference / "validation_cache_manifest.json")
    rows, all_selected, initial_hashes, checked_rows, raw_checked = [], [], [], 0, 0
    expected_codes = {"clean", "-10dB", "-12dB", "-15dB"}
    for directory in directories:
        manifest = read_json(directory / "experiment_manifest.json")
        assert manifest["status"] == "completed"
        arguments = manifest["arguments"]
        seed = arguments["seed"]
        assert arguments["epochs"] == 5 and arguments["batch_size"] == 4
        assert arguments["max_train_samples"] is None and arguments["max_val_samples"] is None
        assert resolve_validation_seed(seed, arguments.get("validation_seed")) == 12026
        assert manifest["arms"] == reference_manifest["arms"]
        for name in ["train_resnet18.py", "dronerfa_data.py", "model.py", "window_index.csv"]:
            assert manifest["source_sha256"][name] == reference_manifest["source_sha256"][name]
        validate_reference_cache(read_json(directory / "validation_cache_manifest.json"), reference_cache)
        pairing = read_json(directory / "pairing_verification.json")
        assert pairing["same_initial_weights"] and pairing["same_epoch_orders"]
        independent = read_json(directory / "independent_verification.json")
        assert independent["counts_and_selection_match"] and independent["same_ordered_validation_windows"]
        assert independent["verified_prediction_rows"] == 18000
        checked_rows += independent["verified_prediction_rows"]
        raw = read_json(directory / "raw_input_verification.json")
        assert raw["source_hashes_match"] and raw["original_checkpoint_and_diagnosis_unchanged"]
        assert all(r["prediction_mismatches"] == 0 and r["max_margin_difference"] <= 0.01 for r in raw["results"])
        raw_checked += sum(r["checked_predictions"] for r in raw["results"])
        for arm in manifest["arms"]:
            config = read_json(directory / arm / "run_config.json")
            assert config["train_samples"] == 920 and config["class_counts"] == {"0": 300, "1": 620}
            assert config["optimizer"] == {"type": "Adam", "learning_rate": 0.001, "weight_decay": 0.0}
        initial_hashes.append(read_json(directory / "A_original/run_config.json")["initial_model_sha256"])
        selected = read_csv(directory / "selected_model_comparison.csv")
        all_selected.extend({"training_seed": seed, **r} for r in selected)
        for policy in ["clean_only", "multi_noise"]:
            result = {"training_seed": seed, "selection_policy": policy}
            for arm, short in [("A_original", "A"), ("B_plus_minus15", "B")]:
                rr = [r for r in selected if r["arm"] == arm and r["selection_policy"] == policy]
                assert len(rr) == 4 and {r["condition"] for r in rr} == expected_codes
                assert all(int(r["air2s_samples"]) == 150 and int(r["phantom_samples"]) == 140
                           and int(r["tn"]) + int(r["fp"]) == 160 for r in rr)
                noisy = next(r for r in rr if r["condition"] == "-15dB")
                clean = next(r for r in rr if r["condition"] == "clean")
                result[f"{short}_epoch"] = int(noisy["epoch"])
                result[f"{short}_mean_ba"] = float(np.mean([float(r["balanced_accuracy"]) for r in rr]))
                for key in ["air2s_detected", "phantom_detected", "fp"]:
                    result[f"{short}_minus15_{key}"] = int(noisy[key])
                result[f"{short}_clean_fp"] = int(clean["fp"])
            for key in ["mean_ba", "minus15_air2s_detected", "minus15_phantom_detected", "minus15_fp", "clean_fp"]:
                result[f"delta_{key}"] = result[f"B_{key}"] - result[f"A_{key}"]
            rows.append(result)
    assert len(set(initial_hashes)) == len(directories), "Training initializations must differ across seeds"
    write_csv(out / "paired_seed_results.csv", rows)
    write_csv(out / "all_selected_condition_metrics.csv", all_selected)
    primary = [r for r in rows if r["selection_policy"] == "multi_noise"]
    delta_stats = {key: describe([r[key] for r in primary]) for key in primary[0] if key.startswith("delta_")}
    verification = {
        "training_seeds": [r["training_seed"] for r in primary], "validation_seed": 12026,
        "same_validation_feature_hashes": True, "same_training_implementation_and_index": True,
        "paired_initial_weights_and_orders": True, "different_initializations_across_seeds": True,
        "total_verified_prediction_rows": checked_rows, "total_raw_input_checks": raw_checked,
        "descriptive_paired_deltas": delta_stats,
        "new_replications_positive_ba_deltas": sum(r["delta_mean_ba"] > 0 for r in primary if r["training_seed"] != 2026),
    }
    write_json(out / "cross_seed_verification.json", verification)

    x = np.arange(len(primary))
    seeds = [str(r["training_seed"]) for r in primary]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for short, label, color in [("A", "A: original noise range", "#3978a8"), ("B", "B: add -15 dB", "#d67b29")]:
        axes[0, 0].plot(x, [100*r[f"{short}_mean_ba"] for r in primary], "o-", color=color, label=label)
        axes[1, 0].plot(x, [r[f"{short}_minus15_air2s_detected"] for r in primary], "o-", color=color, label=label)
        axes[1, 1].plot(x, [r[f"{short}_minus15_fp"] for r in primary], "o-", color=color, label=f"{short}: -15 dB")
        axes[1, 1].plot(x, [r[f"{short}_clean_fp"] for r in primary], "x--", color=color, label=f"{short}: clean")
    deltas = [100*r["delta_mean_ba"] for r in primary]
    axes[0, 1].bar(x, deltas, color=["#35906b" if d >= 0 else "#bd514d" for d in deltas], width=0.55)
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    for i, value in enumerate(deltas):
        axes[0, 1].annotate(f"{value:+.2f}", (i, value), xytext=(0, 5 if value >= 0 else -12), textcoords="offset points", ha="center")
    axes[0, 1].margins(y=0.2)
    axes[0, 0].set(title="Selected mean balanced accuracy", ylabel="Percent")
    axes[0, 1].set(title="Paired B - A difference", ylabel="Percentage points")
    axes[1, 0].set(title="Air 2S detected at -15 dB", ylabel="Windows out of 150", ylim=(-3, 155))
    axes[1, 1].set(title="Background false positives", ylabel="Windows out of 160", ylim=(0, None))
    for ax in axes.flat:
        ax.set(xticks=x, xticklabels=seeds, xlabel="Training seed (validation noise fixed)")
        ax.grid(alpha=0.2, axis="y")
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Training-seed replication on the same validation recordings")
    fig.savefig(out / "paired_seed_comparison.png", dpi=160)
    plt.close(fig)

    lines = ["# 固定验证噪声的三训练种子配对复现", "",
             "保留 seed=2026 的原试验，新增训练种子 2027、2028。每个种子下 A/B 各训练 5 轮，均采用预先指定的四档平均平衡准确率选模型，同分选最早轮次。",
             "验证噪声固定为 12026，四档原始窗口索引和 float32 频谱哈希与首轮完全相同。训练初始化、训练噪声抽样与样本顺序随训练种子变化；每个种子内的 A/B 初始权重和顺序保持配对。", "",
             "## 主要结果：每个种子的配对差值", "",
             "| 训练种子 | A 轮次 | B 轮次 | A 平均平衡准确率 | B 平均平衡准确率 | B−A（百分点） |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in primary:
        lines.append(f"| {r['training_seed']} | {r['A_epoch']} | {r['B_epoch']} | {r['A_mean_ba']:.2%} | {r['B_mean_ba']:.2%} | {100*r['delta_mean_ba']:+.2f} |")
    stats = delta_stats["delta_mean_ba"]
    new_positive = verification["new_replications_positive_ba_deltas"]
    lines += ["", f"新增两次复现中 {new_positive}/2 次 B 的综合分数高于 A。三个种子的配对差值均值为 {100*stats['mean']:+.2f} 个百分点，样本标准差 {100*stats['sample_sd']:.2f}，范围 {100*stats['minimum']:+.2f} 至 {100*stats['maximum']:+.2f}。",
              "这里的标准差描述训练随机性的波动，不是总体性能置信区间。2026 是先前探索结果，2027、2028 是本轮预先指定的复现。", "",
              "## 分项结果与误报代价", "",
              "| 种子 | −15 Air 检出 A→B（/150） | −15 Phantom 检出 A→B（/140） | −15 背景误报 A→B（/160） | clean 背景误报 A→B（/160） |",
              "|---|---:|---:|---:|---:|"]
    for r in primary:
        lines.append(f"| {r['training_seed']} | {r['A_minus15_air2s_detected']}→{r['B_minus15_air2s_detected']} | "
                     f"{r['A_minus15_phantom_detected']}→{r['B_minus15_phantom_detected']} | "
                     f"{r['A_minus15_fp']}→{r['B_minus15_fp']} | {r['A_clean_fp']}→{r['B_clean_fp']} |")
    lines += ["", "## 当前判断与下一步", "",
              "**本轮没有复现出跨三个训练种子方向一致的综合提升，不能将首轮 +2.94 个百分点称为稳定增益。**"
              if any(r["delta_mean_ba"] <= 0 for r in primary) else
              "三个种子的综合差值方向一致，但样本量与重复验证限制仍在，尚不能证明跨录制泛化。",
              "增加 −15 训练档位可以改变强噪声下的检出与误报取舍，但两个新增种子的改善来源和方向不同。",
              "当前证据适合保留 A/B 两套候选，冻结本轮规则，在尚未使用的源文件上评价全部种子的所选模型。不要只挑表现最好的种子作为结果。",
              "训练过程部分轮次出现从大量误报转为大量漏检的波动；它说明当前短轮数训练对种子和轮次敏感，尚不能把原因归结为某一种机制。",
              "已准备 [后续留出数据方案](后续留出数据方案.md) 和 12 个候选文件清单；尚未下载或评价。不同文件不等同于独立采集会话，跨会话结论仍需采集元数据。"]
    lines += ["", "## 检查与解释边界", "",
              f"- 三个种子共 {checked_rows:,} 条逐窗口预测的统计复核通过，所选检查点累计 {raw_checked} 条原始 IQ 抽查通过。",
              "- 两个新增种子的验证频谱与首轮完全一致，网络、前端、优化器和数据索引保持一致。",
              "- 不能把三轮验证窗口合并当成更多独立采集样本；三个种子仍反复使用同一批录制。",
              "- 选模和报告使用同一验证集，存在选择偏乐观；训练种子复现不能替代新录制验证。",
              "- A/B 比较的是整个等概率训练噪声分布的变化，不能仅归因于某个 −15 样本的独立作用。",
              "- 本轮不按结果挑选最好随机种子作为总体结论，也不把三个种子当成显著性证明。",
              "- 本地 DroneRFa 目录现有 12 个 MAT 文件均已用于当前 train/val/test 划分，未找到尚未使用的同项目源文件。新的独立录制验证仍需补充数据。", "",
              "![三个种子的配对比较](paired_seed_comparison.png)", "",
              "[逐种子结果表](paired_seed_results.csv) · [全部所选模型的逐档指标](all_selected_condition_metrics.csv) · [跨种子核查](cross_seed_verification.json)"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
