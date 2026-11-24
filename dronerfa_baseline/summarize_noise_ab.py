"""Independently audit per-window pilot outputs and render comparison plots."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metrics(records):
    y = np.asarray([int(r["label"]) for r in records])
    pred = np.asarray([int(r["prediction"]) for r in records])
    margin = np.asarray([float(r["logit_margin"]) for r in records])
    assert np.array_equal(pred, margin > 0)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    result = dict(tp=tp, tn=tn, fp=fp, fn=fn,
                  balanced_accuracy=(tp / (tp + fn) + tn / (tn + fp)) / 2)
    for code, name in [("0010", "phantom"), ("0101", "air2s")]:
        rr = [r for r in records if r["target_code"] == code]
        result[f"{name}_detected"] = sum(int(r["prediction"]) for r in rr)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    out = args.output_dir
    manifest = json.loads((out / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    comparison = read_csv(out / "selected_model_comparison.csv")
    conditions = ["clean", "-10dB", "-12dB", "-15dB"]
    histories, verified, reference_keys = {}, 0, {}
    selected = {}
    for arm in manifest["arms"]:
        directory = out / arm
        history = read_csv(directory / "training_history.csv")
        recorded_metrics = read_csv(directory / "validation_metrics.csv")
        histories[arm] = history
        selected[arm] = json.loads((directory / "selection.json").read_text())
        for policy, field in [("clean_only", "clean_balanced_accuracy"),
                              ("multi_noise", "mean_balanced_accuracy")]:
            expected = int(max(history, key=lambda row: float(row[field]))["epoch"])
            assert selected[arm][policy] == expected
            assert (directory / f"epoch_{expected:02d}.pt").stat().st_size > 0
        for row in history:
            epoch = int(row["epoch"])
            records = read_csv(directory / f"validation_scores_epoch_{epoch:02d}.csv")
            condition_metrics = []
            for condition in conditions:
                subset = [r for r in records if r["condition"] == condition]
                keys = [(r["file_path"], int(r["start"]), r["label"], r["target_code"]) for r in subset]
                assert len(keys) == len(set(keys))
                if condition not in reference_keys:
                    reference_keys[condition] = keys
                assert keys == reference_keys[condition]
                actual = metrics(subset)
                saved = next(r for r in recorded_metrics if int(r["epoch"]) == epoch and r["condition"] == condition)
                for key, value in actual.items():
                    assert math.isclose(float(saved[key]), value, abs_tol=1e-12), (arm, epoch, condition, key)
                for chosen in comparison:
                    if chosen["arm"] == arm and int(chosen["epoch"]) == epoch and chosen["condition"] == condition:
                        for key, value in actual.items():
                            assert math.isclose(float(chosen[key]), value, abs_tol=1e-12)
                condition_metrics.append(actual["balanced_accuracy"])
                verified += len(subset)
            assert math.isclose(float(row["mean_balanced_accuracy"]), np.mean(condition_metrics), abs_tol=1e-12)
            assert math.isclose(float(row["clean_balanced_accuracy"]), condition_metrics[0], abs_tol=1e-12)
    assert all(keys == reference_keys["clean"] for keys in reference_keys.values())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for arm, history in histories.items():
        epochs = [int(r["epoch"]) for r in history]
        label = "A: original noise range" if arm.startswith("A") else "B: add -15 dB"
        axes[0].plot(epochs, [float(r["mean_balanced_accuracy"]) for r in history], marker="o", label=label)
        axes[1].plot(epochs, [float(r["clean_balanced_accuracy"]) for r in history], marker="o", label=label)
    for ax, title, field in zip(axes,
            ["Mean over four validation conditions", "Clean validation only"],
            ["mean_balanced_accuracy", "clean_balanced_accuracy"]):
        lower = max(0.0, min(float(r[field]) for history in histories.values() for r in history) - 0.02)
        ax.set(title=title, xlabel="Epoch", ylabel="Balanced accuracy", ylim=(lower, 1.005),
               xticks=epochs)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(out / "epoch_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    styles = {"A_original": "#3978a8", "B_plus_minus15": "#d67b29"}
    for arm in histories:
        for policy in ["clean_only", "multi_noise"]:
            rr = [next(r for r in comparison if r["arm"] == arm and r["selection_policy"] == policy
                       and r["condition"] == condition) for condition in conditions]
            label = f"{arm[0]} / {policy} / epoch {rr[0]['epoch']}"
            for ax, field, title in zip(axes.flat,
                    ["air2s_recall", "phantom_recall", "fpr", "balanced_accuracy"],
                    ["Air 2S recall", "Phantom 4 Pro recall", "Background false-positive rate", "Balanced accuracy"]):
                ax.plot(range(4), [100 * float(r[field]) for r in rr],
                        marker="o" if policy == "multi_noise" else "x",
                        linestyle="-" if policy == "multi_noise" else "--",
                        color=styles[arm], label=label, alpha=0.85)
                ax.set(title=title, ylabel="Percent", xticks=range(4),
                       xticklabels=["clean", "-10", "-12", "-15"], xlabel="Added AWGN (dB)")
                ax.grid(alpha=0.25)
                if field != "fpr":
                    ax.set_ylim(-2, 103)
                else:
                    ax.set_ylim(-0.2, max(100 * float(r["fpr"]) for r in comparison) * 1.1 + 0.1)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Selected checkpoints on reused validation recordings (exploratory)")
    fig.savefig(out / "selected_model_comparison.png", dpi=160)
    plt.close(fig)

    details = ["", "## 配对比较与复核", "",
               f"独立从逐窗口分数重算并核对了 {verified:,} 条预测、每档混淆矩阵和选模轮次，均一致。", "",
               "| 附加噪声 | B−A：Air 检出数 | B−A：Phantom 检出数 | B−A：背景误报数 | B−A：平衡准确率（百分点） |",
               "|---|---:|---:|---:|---:|"]
    deltas = []
    for condition in conditions:
        a, b = [next(r for r in comparison if r["arm"] == arm and r["selection_policy"] == "multi_noise"
                     and r["condition"] == condition) for arm in manifest["arms"]]
        delta = {"condition": condition, **{key: float(b[key]) - float(a[key])
                  for key in ["air2s_detected", "phantom_detected", "fp", "balanced_accuracy"]}}
        deltas.append(delta)
        details.append(f"| {condition} | {delta['air2s_detected']:+.0f} | {delta['phantom_detected']:+.0f} | "
                       f"{delta['fp']:+.0f} | {100*delta['balanced_accuracy']:+.2f} |")
    means = {}
    for arm in manifest["arms"]:
        means[arm] = np.mean([float(r["balanced_accuracy"]) for r in comparison
                             if r["arm"] == arm and r["selection_policy"] == "multi_noise"])
    details += ["", f"多噪声选模后的四档平均平衡准确率：A={means['A_original']:.2%}，B={means['B_plus_minus15']:.2%}。",
                "B−A 反映本轮整个训练噪声分布变更的结果，不能隔离成仅 −15 档样本的独立因果效应。", "",
                "![各轮验证曲线](epoch_comparison.png)", "", "![所选模型的分项比较](selected_model_comparison.png)"]
    report = out / "report.md"
    original = report.read_text(encoding="utf-8").split("\n## 配对比较与复核")[0].rstrip()
    report.write_text(original + "\n" + "\n".join(details) + "\n", encoding="utf-8")
    verification = {"verified_prediction_rows": verified, "counts_and_selection_match": True,
                    "same_ordered_validation_windows": True, "selected_epochs": selected,
                    "mean_balanced_accuracy": means, "multi_noise_B_minus_A": deltas}
    (out / "independent_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
