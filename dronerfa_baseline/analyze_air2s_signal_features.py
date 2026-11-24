"""Compare signal features for stable and missed DJI Air 2S windows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from dronerfa_data import DroneRFaWindowDataset
from model import ResNet18, SpectrogramFrontEnd


ROOT = Path(__file__).resolve().parent
DIAGNOSIS = ROOT / "outputs" / "validation_diagnosis_20260917_161707"
WINDOW_ANALYSIS = DIAGNOSIS / "air2s_window_analysis_20260917"
CONDITIONS = {"clean": None, "-12dB": -12.0, "-15dB": -15.0}
GROUP_ORDER = (
    "detected_to_detected",
    "detected_to_missed",
    "missed_to_missed",
)
METRICS = (
    "spectral_peak_prominence_db",
    "spectral_top10_fraction",
    "temporal_power_cv",
    "image_frequency_gradient",
    "image_time_gradient",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce model inputs and compare Air 2S window features."
    )
    parser.add_argument("--diagnosis-dir", type=Path, default=DIAGNOSIS)
    parser.add_argument("--window-analysis-dir", type=Path, default=WINDOW_ANALYSIS)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_representatives(
    transitions: list[dict[str, str]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in transitions:
        grouped[row["transition_minus12_to_minus15"]].append(row)

    output: list[dict[str, object]] = []
    for group in GROUP_ORDER:
        rows = grouped[group]
        if not rows:
            raise ValueError(f"Missing transition group: {group}")
        margins = np.asarray([float(row["margin_minus12db"]) for row in rows])
        median = float(np.median(margins))
        selected = min(
            rows,
            key=lambda row: (
                abs(float(row["margin_minus12db"]) - median),
                int(row["start_sample"]),
            ),
        )
        output.append(
            {
                "transition_group": group,
                "selection_rule": "closest_to_group_median_minus12db_margin",
                "filename": selected["filename"],
                "file_path": selected["file_path"],
                "start_sample": int(selected["start_sample"]),
                "start_time_s": float(selected["start_time_s"]),
                "group_median_margin_minus12db": median,
                "selected_margin_clean": float(selected["margin_clean"]),
                "selected_margin_minus12db": float(selected["margin_minus12db"]),
                "selected_margin_minus15db": float(selected["margin_minus15db"]),
            }
        )
    return output


def compute_frontend_and_metrics(
    frontend: SpectrogramFrontEnd, signals: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    complex_iq = torch.complex(signals[:, 0], signals[:, 1])
    spectra = torch.stft(
        complex_iq,
        n_fft=frontend.n_fft,
        hop_length=frontend.hop_length,
        win_length=frontend.n_fft,
        window=frontend.window,
        center=False,
        normalized=True,
        onesided=False,
        return_complex=True,
    )
    power = spectra.abs().square().clamp_min(1e-12)
    frequency_power = power.mean(dim=2)
    frequency_median = torch.quantile(frequency_power, 0.5, dim=1)
    frequency_peak = frequency_power.max(dim=1).values
    prominence_db = 10.0 * torch.log10(frequency_peak / frequency_median.clamp_min(1e-12))
    top_count = max(1, int(round(frequency_power.shape[1] * 0.10)))
    top_fraction = torch.topk(frequency_power, top_count, dim=1).values.sum(dim=1) / frequency_power.sum(dim=1)

    temporal_power = power.mean(dim=1)
    temporal_cv = temporal_power.std(dim=1) / temporal_power.mean(dim=1).clamp_min(1e-12)

    log_power = 10.0 * torch.log10(power + 1e-10)
    log_power = torch.fft.fftshift(log_power, dim=1)
    mean = log_power.mean(dim=(1, 2), keepdim=True)
    std = log_power.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
    features = F.interpolate(
        ((log_power - mean) / std).unsqueeze(1),
        size=(frontend.image_size, frontend.image_size),
        mode="bilinear",
        align_corners=False,
    )
    metrics = {
        "spectral_peak_prominence_db": prominence_db,
        "spectral_top10_fraction": top_fraction,
        "temporal_power_cv": temporal_cv,
        "image_frequency_gradient": torch.diff(features, dim=2).abs().mean(dim=(1, 2, 3)),
        "image_time_gradient": torch.diff(features, dim=3).abs().mean(dim=(1, 2, 3)),
    }
    return features, metrics


def summarize_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["transition_group"]))].append(row)

    output: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for group in GROUP_ORDER:
            selected = grouped[(condition, group)]
            record: dict[str, object] = {
                "condition": condition,
                "transition_group": group,
                "windows": len(selected),
            }
            for metric in METRICS:
                values = np.asarray([float(row[metric]) for row in selected])
                record[f"{metric}_median"] = float(np.median(values))
                record[f"{metric}_q25"] = float(np.quantile(values, 0.25))
                record[f"{metric}_q75"] = float(np.quantile(values, 0.75))
            output.append(record)
    return output


def pairwise_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def discrimination_summary(
    transitions: list[dict[str, str]], metric_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []

    margin_predictors = {
        "clean_margin": "margin_clean",
        "minus10db_margin": "margin_minus10db",
        "minus12db_margin": "margin_minus12db",
    }
    for name, column in margin_predictors.items():
        stable = np.asarray(
            [
                float(row[column])
                for row in transitions
                if row["transition_minus12_to_minus15"] == "detected_to_detected"
            ]
        )
        newly_missed = np.asarray(
            [
                float(row[column])
                for row in transitions
                if row["transition_minus12_to_minus15"] == "detected_to_missed"
            ]
        )
        output.append(discrimination_record(name, "saved_model_margin", stable, newly_missed))

    for condition in ("clean", "-12dB"):
        for metric in METRICS:
            stable = np.asarray(
                [
                    float(row[metric])
                    for row in metric_rows
                    if row["condition"] == condition
                    and row["transition_group"] == "detected_to_detected"
                ]
            )
            newly_missed = np.asarray(
                [
                    float(row[metric])
                    for row in metric_rows
                    if row["condition"] == condition
                    and row["transition_group"] == "detected_to_missed"
                ]
            )
            output.append(
                discrimination_record(
                    f"{condition}_{metric}", "descriptive_signal_feature", stable, newly_missed
                )
            )
    return output


def discrimination_record(
    predictor: str,
    predictor_type: str,
    stable: np.ndarray,
    newly_missed: np.ndarray,
) -> dict[str, object]:
    raw_auc = pairwise_auc(stable, newly_missed)
    direction = "higher_in_stable" if raw_auc >= 0.5 else "lower_in_stable"
    return {
        "predictor": predictor,
        "predictor_type": predictor_type,
        "stable_windows": len(stable),
        "newly_missed_windows": len(newly_missed),
        "stable_median": float(np.median(stable)),
        "newly_missed_median": float(np.median(newly_missed)),
        "raw_auc_higher_predicts_stable": raw_auc,
        "data_oriented_direction": direction,
        "data_oriented_separation_auc": max(raw_auc, 1.0 - raw_auc),
    }


def plot_representatives(
    path: Path,
    representatives: list[dict[str, object]],
    images: dict[tuple[str, str], np.ndarray],
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
    labels = {
        "detected_to_detected": "Detected at -12 and -15 dB",
        "detected_to_missed": "Detected at -12, missed at -15 dB",
        "missed_to_missed": "Missed at -12 and -15 dB",
    }
    margins_by_group = {str(row["transition_group"]): row for row in representatives}
    image_handle = None
    for row_index, group in enumerate(GROUP_ORDER):
        selected = margins_by_group[group]
        for column_index, condition in enumerate(CONDITIONS):
            axis = axes[row_index, column_index]
            image_handle = axis.imshow(
                images[(group, condition)],
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=-2.5,
                vmax=2.5,
            )
            margin_key = {
                "clean": "selected_margin_clean",
                "-12dB": "selected_margin_minus12db",
                "-15dB": "selected_margin_minus15db",
            }[condition]
            axis.set_title(f"{condition}; margin={float(selected[margin_key]):.2f}")
            if column_index == 0:
                axis.set_ylabel(
                    f"{labels[group]}\nNormalized frequency bin",
                    fontsize=9,
                )
            if row_index == 2:
                axis.set_xlabel("Normalized time bin")
    fig.suptitle(
        "Representative DJI Air 2S model inputs\n"
        "Each row uses the window closest to its group's median -12 dB margin",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.13, right=0.84, bottom=0.08, top=0.90, wspace=0.12, hspace=0.25)
    if image_handle is not None:
        color_axis = fig.add_axes([0.88, 0.18, 0.018, 0.62])
        fig.colorbar(image_handle, cax=color_axis, label="Normalized log-power")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    representatives: list[dict[str, object]],
    summary: list[dict[str, object]],
    discriminators: list[dict[str, object]],
    verification: dict[str, object],
) -> None:
    by_key = {
        (str(row["condition"]), str(row["transition_group"])): row
        for row in summary
    }
    label = {
        "detected_to_detected": "持续检出",
        "detected_to_missed": "新增漏检",
        "missed_to_missed": "持续漏检",
    }
    by_predictor = {str(row["predictor"]): row for row in discriminators}
    lines = [
        "# Air 2S 代表窗口与信号特征对比",
        "",
        "本分析复现原诊断的验证集索引、噪声 seed 和前端处理。模型分数复算用于核对输入，未训练或修改模型。",
        "",
        "## 复现检查",
        "",
        f"- 复算窗口：{verification['rows_checked']} 个；预测不一致：{verification['prediction_mismatches']} 个。",
        f"- 保存 margin 与复算 margin 的最大绝对差：{float(verification['max_margin_abs_difference']):.6f}。",
        "- 时频图是模型实际接收的 256×256、逐窗标准化 log-power 输入；所有图使用同一颜色范围。",
        "",
        "## 代表窗口选择",
        "",
        "每组选择 −12 dB margin 最接近该组中位数的窗口，避免人为挑选最显眼的例子。",
        "",
        "| 分组 | 时间 | clean margin | −12 dB margin | −15 dB margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in representatives:
        lines.append(
            f"| {label[str(row['transition_group'])]} | {float(row['start_time_s']):.2f} s | "
            f"{float(row['selected_margin_clean']):.3f} | "
            f"{float(row['selected_margin_minus12db']):.3f} | "
            f"{float(row['selected_margin_minus15db']):.3f} |"
        )

    lines += [
        "",
        "## 全部窗口的描述性特征",
        "",
        "下面比较全部 150 个窗口，而非只比较三张代表图。`频谱峰值突出度`越高，表示平均频谱中的最强频率结构相对中位水平越突出；这些量是描述性代理，不等同于真实目标功率或现场 SNR。",
        "",
        "| 条件 | 分组 | 窗口 | 频谱峰值突出度中位数 | 频谱 Top 10% 能量占比中位数 |",
        "|---|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        for group in GROUP_ORDER:
            row = by_key[(condition, group)]
            lines.append(
                f"| {condition} | {label[group]} | {row['windows']} | "
                f"{float(row['spectral_peak_prominence_db_median']):.3f} dB | "
                f"{float(row['spectral_top10_fraction_median']):.3f} |"
            )
    lines += [
        "",
        "## 哪些单变量能区分持续检出与新增漏检",
        "",
        "这里仅比较 −12 dB 时仍被检出的 137 个窗口：49 个到 −15 dB 仍检出，88 个变成漏检。AUC 的方向依据本验证数据选择，因此只用于描述，不是独立测试性能。",
        "",
        "| 指标 | 持续检出中位数 | 新增漏检中位数 | 单变量区分 AUC |",
        "|---|---:|---:|---:|",
    ]
    for predictor in (
        "clean_margin",
        "minus10db_margin",
        "minus12db_margin",
        "-12dB_spectral_peak_prominence_db",
        "-12dB_spectral_top10_fraction",
        "-12dB_temporal_power_cv",
    ):
        row = by_predictor[predictor]
        lines.append(
            f"| {predictor} | {float(row['stable_median']):.4f} | "
            f"{float(row['newly_missed_median']):.4f} | "
            f"{float(row['data_oriented_separation_auc']):.4f} |"
        )
    lines += [
        "",
        "clean margin 对后续是否漏检的区分有限；差异从 −10 dB 开始扩大，到 −12 dB 已很明显。简单的频谱峰值突出度和频谱能量集中度在两组之间接近，不能解释主要差异。时间功率波动有一定关联，值得后续结合原始突发结构检查，但不能据此认定为原因。",
    ]
    lines += [
        "",
        "## 限制",
        "",
        "三组来自同一份 Air 2S 验证录制，且分组本身由模型输出定义。这些特征可以帮助提出下一步检查对象，不能单独证明某个物理特征导致模型漏检，也不能外推到所有 Air 2S 录制。",
        "",
        "![代表窗口时频输入](air2s_representative_spectrograms.png)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    diagnosis_dir = args.diagnosis_dir.resolve()
    analysis_dir = args.window_analysis_dir.resolve()
    config = json.loads((diagnosis_dir / "run_config.json").read_text(encoding="utf-8"))
    transitions = read_csv(analysis_dir / "air2s_window_transitions.csv")
    representatives = select_representatives(transitions)
    transition_by_key = {
        (row["file_path"], int(row["start_sample"])): row["transition_minus12_to_minus15"]
        for row in transitions
    }
    representative_by_key = {
        (str(row["file_path"]), int(row["start_sample"])): str(row["transition_group"])
        for row in representatives
    }

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the saved mixed-precision margins")
    device = torch.device("cuda")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(
            config["checkpoint"], map_location="cpu", weights_only=True
        )
    old = checkpoint["run_config"]["arguments"]
    frontend = SpectrogramFrontEnd(
        n_fft=old["n_fft"],
        hop_length=old["hop_length"],
        image_size=old["image_size"],
    ).to(device).eval()
    model = ResNet18(input_channels=1, num_classes=2).to(device).eval()
    frontend.load_state_dict(checkpoint["frontend_state"])
    model.load_state_dict(checkpoint["model_state"])

    saved_scores = {
        (row["condition"], row["file_path"], int(row["start"])): row
        for row in read_csv(diagnosis_dir / "validation_scores.csv")
        if row["target_name"] == "DJI Air 2S"
    }
    metric_rows: list[dict[str, object]] = []
    representative_images: dict[tuple[str, str], np.ndarray] = {}
    differences: list[float] = []
    prediction_mismatches = 0

    for condition, snr in CONDITIONS.items():
        dataset = DroneRFaWindowDataset(
            Path(config["index"]),
            split="val",
            fixed_snr=snr,
            apply_fixed_snr=snr is not None,
            seed=int(config["noise_seed"]),
        )
        air_indices = [
            index
            for index, row in enumerate(dataset.rows)
            if row["target_name"] == "DJI Air 2S"
        ]
        loader = DataLoader(
            Subset(dataset, air_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        cursor = 0
        for signals, _ in loader:
            signals = signals.to(device)
            features, metrics = compute_frontend_and_metrics(frontend, signals.float())
            with torch.amp.autocast("cuda", enabled=bool(old["amp"]), dtype=torch.float16):
                logits = model(features)
            margins = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()
            predictions = (margins > 0).astype(int)
            batch_indices = air_indices[cursor : cursor + len(signals)]
            for local_index, dataset_index in enumerate(batch_indices):
                source = dataset.rows[dataset_index]
                key = (source["file_path"], int(source["start"]))
                transition = transition_by_key[key]
                saved = saved_scores[(condition, key[0], key[1])]
                difference = abs(float(margins[local_index]) - float(saved["logit_margin"]))
                differences.append(difference)
                prediction_mismatches += int(
                    int(predictions[local_index]) != int(saved["default_prediction"])
                )
                record: dict[str, object] = {
                    "condition": condition,
                    "transition_group": transition,
                    "filename": source["filename"],
                    "file_path": source["file_path"],
                    "start_sample": int(source["start"]),
                    "start_time_s": int(source["start"]) / 100_000_000,
                    "saved_margin": float(saved["logit_margin"]),
                    "recomputed_margin": float(margins[local_index]),
                    "margin_abs_difference": difference,
                }
                for name, values in metrics.items():
                    record[name] = float(values[local_index].cpu())
                metric_rows.append(record)
                representative_group = representative_by_key.get(key)
                if representative_group:
                    representative_images[(representative_group, condition)] = (
                        features[local_index, 0].float().cpu().numpy()
                    )
            cursor += len(signals)
        dataset.close()
        if cursor != 150:
            raise AssertionError(f"Expected 150 Air 2S rows at {condition}, got {cursor}")

    summary = summarize_metrics(metric_rows)
    discriminators = discrimination_summary(transitions, metric_rows)
    verification: dict[str, object] = {
        "passed": False,
        "conditions": list(CONDITIONS),
        "rows_checked": len(metric_rows),
        "expected_rows": 450,
        "prediction_mismatches": prediction_mismatches,
        "max_margin_abs_difference": max(differences),
        "mean_margin_abs_difference": float(np.mean(differences)),
        "noise_seed": int(config["noise_seed"]),
        "full_validation_index_used": True,
        "checkpoint_sha256": digest(Path(config["checkpoint"])),
        "checkpoint_hash_matches_diagnosis": digest(Path(config["checkpoint"]))
        == config["checkpoint_sha256"],
    }
    verification["passed"] = bool(
        verification["rows_checked"] == verification["expected_rows"]
        and prediction_mismatches == 0
        and float(verification["max_margin_abs_difference"]) <= 0.02
        and verification["checkpoint_hash_matches_diagnosis"]
        and len(representative_images) == 9
    )
    if not verification["passed"]:
        raise AssertionError(json.dumps(verification, indent=2))

    write_csv(analysis_dir / "air2s_signal_feature_metrics.csv", metric_rows)
    write_csv(analysis_dir / "air2s_signal_feature_group_summary.csv", summary)
    write_csv(analysis_dir / "air2s_feature_discrimination.csv", discriminators)
    write_csv(analysis_dir / "air2s_representative_windows.csv", representatives)
    plot_representatives(
        analysis_dir / "air2s_representative_spectrograms.png",
        representatives,
        representative_images,
    )
    write_report(
        analysis_dir / "signal_feature_report.md",
        representatives,
        summary,
        discriminators,
        verification,
    )
    (analysis_dir / "signal_feature_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, indent=2))
    print(f"OUTPUT {analysis_dir}")


if __name__ == "__main__":
    main()
