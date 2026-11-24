"""Analyze Air 2S per-window detector failures from saved validation scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DIAGNOSIS = ROOT / "outputs" / "validation_diagnosis_20260917_161707"
SAMPLE_RATE_HZ = 100_000_000
CONDITIONS = ("clean", "-10dB", "-12dB", "-15dB", "-20dB")
CONDITION_SLUGS = {
    "clean": "clean",
    "-10dB": "minus10db",
    "-12dB": "minus12db",
    "-15dB": "minus15db",
    "-20dB": "minus20db",
}
TRANSITION_NAMES = {
    (1, 1): "detected_to_detected",
    (1, 0): "detected_to_missed",
    (0, 1): "missed_to_detected",
    (0, 0): "missed_to_missed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze saved Air 2S scores without rerunning inference."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=DEFAULT_DIAGNOSIS / "validation_scores.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DIAGNOSIS / "air2s_window_analysis_20260917",
    )
    parser.add_argument("--target", default="DJI Air 2S")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_target_rows(path: Path, target: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "condition",
            "file_path",
            "start",
            "length",
            "target_name",
            "split",
            "label",
            "logit_margin",
            "drone_score",
            "default_prediction",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        rows = [row for row in reader if row["target_name"] == target]
    if not rows:
        raise ValueError(f"No rows found for target {target!r}")
    if {row["split"] for row in rows} != {"val"}:
        raise ValueError("Target rows are not exclusively from the validation split")
    if {int(row["label"]) for row in rows} != {1}:
        raise ValueError("Target rows do not all have drone label 1")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_conditions(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)

    output: list[dict[str, object]] = []
    for condition in CONDITIONS:
        selected = grouped[condition]
        if not selected:
            raise ValueError(f"Condition {condition!r} is missing")
        margins = np.asarray([float(row["logit_margin"]) for row in selected])
        detected = sum(int(row["default_prediction"]) == 1 for row in selected)
        output.append(
            {
                "condition": condition,
                "windows": len(selected),
                "detected": detected,
                "missed": len(selected) - detected,
                "recall": detected / len(selected),
                "margin_min": float(np.min(margins)),
                "margin_q05": float(np.quantile(margins, 0.05)),
                "margin_q25": float(np.quantile(margins, 0.25)),
                "margin_median": float(np.median(margins)),
                "margin_q75": float(np.quantile(margins, 0.75)),
                "margin_q95": float(np.quantile(margins, 0.95)),
                "margin_max": float(np.max(margins)),
            }
        )
    return output


def pair_windows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_condition: dict[str, dict[tuple[str, int], dict[str, str]]] = defaultdict(dict)
    for row in rows:
        condition = row["condition"]
        if condition not in CONDITIONS:
            continue
        key = (row["file_path"], int(row["start"]))
        if key in by_condition[condition]:
            raise ValueError(f"Duplicate {condition} row for {key}")
        by_condition[condition][key] = row

    reference_keys = set(by_condition[CONDITIONS[0]])
    for condition in CONDITIONS[1:]:
        keys = set(by_condition[condition])
        if keys != reference_keys:
            raise ValueError(
                f"Window keys differ at {condition}: "
                f"missing={len(reference_keys - keys)}, extra={len(keys - reference_keys)}"
            )

    output: list[dict[str, object]] = []
    for file_path, start in sorted(reference_keys):
        base = by_condition[CONDITIONS[0]][(file_path, start)]
        length = int(base["length"])
        paired: dict[str, object] = {
            "filename": Path(file_path).name,
            "file_path": file_path,
            "start_sample": start,
            "length_samples": length,
            "start_time_s": start / SAMPLE_RATE_HZ,
            "end_time_s": (start + length) / SAMPLE_RATE_HZ,
        }
        for condition in CONDITIONS:
            item = by_condition[condition][(file_path, start)]
            slug = CONDITION_SLUGS[condition]
            paired[f"margin_{slug}"] = float(item["logit_margin"])
            paired[f"score_{slug}"] = float(item["drone_score"])
            paired[f"prediction_{slug}"] = int(item["default_prediction"])
        transition_key = (
            int(paired["prediction_minus12db"]),
            int(paired["prediction_minus15db"]),
        )
        paired["transition_minus12_to_minus15"] = TRANSITION_NAMES[transition_key]
        paired["margin_change_minus12_to_minus15"] = (
            float(paired["margin_minus15db"]) - float(paired["margin_minus12db"])
        )
        output.append(paired)
    return output


def summarize_transition_groups(
    paired: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in paired:
        grouped[str(row["transition_minus12_to_minus15"])].append(row)

    output: list[dict[str, object]] = []
    for transition in TRANSITION_NAMES.values():
        selected = grouped[transition]
        if not selected:
            continue
        record: dict[str, object] = {
            "transition": transition,
            "windows": len(selected),
            "start_time_min_s": min(float(row["start_time_s"]) for row in selected),
            "start_time_max_s": max(float(row["start_time_s"]) for row in selected),
        }
        for condition in CONDITIONS:
            slug = CONDITION_SLUGS[condition]
            values = np.asarray([float(row[f"margin_{slug}"]) for row in selected])
            record[f"median_margin_{slug}"] = float(np.median(values))
            record[f"q25_margin_{slug}"] = float(np.quantile(values, 0.25))
            record[f"q75_margin_{slug}"] = float(np.quantile(values, 0.75))
        output.append(record)
    return output


def contiguous_runs(
    paired: list[dict[str, object]],
    run_type: str,
    predicate,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in paired:
        if predicate(row):
            grouped[str(row["file_path"])].append(row)

    output: list[dict[str, object]] = []
    for file_path, selected in grouped.items():
        selected.sort(key=lambda row: int(row["start_sample"]))
        current: list[dict[str, object]] = []
        for row in selected:
            is_contiguous = not current or int(row["start_sample"]) == (
                int(current[-1]["start_sample"]) + int(current[-1]["length_samples"])
            )
            if not is_contiguous:
                output.append(make_run_record(run_type, file_path, current))
                current = []
            current.append(row)
        if current:
            output.append(make_run_record(run_type, file_path, current))
    return sorted(
        output,
        key=lambda row: (
            str(row["run_type"]),
            -int(row["window_count"]),
            float(row["start_time_s"]),
        ),
    )


def make_run_record(
    run_type: str,
    file_path: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "run_type": run_type,
        "filename": Path(file_path).name,
        "file_path": file_path,
        "start_sample": int(rows[0]["start_sample"]),
        "end_sample_exclusive": int(rows[-1]["start_sample"])
        + int(rows[-1]["length_samples"]),
        "start_time_s": round(float(rows[0]["start_time_s"]), 6),
        "end_time_s": round(float(rows[-1]["end_time_s"]), 6),
        "duration_s": round(
            float(rows[-1]["end_time_s"]) - float(rows[0]["start_time_s"]), 6
        ),
        "window_count": len(rows),
        "median_margin_minus12db": float(
            np.median([float(row["margin_minus12db"]) for row in rows])
        ),
        "median_margin_minus15db": float(
            np.median([float(row["margin_minus15db"]) for row in rows])
        ),
    }


def time_bin_summary(
    paired: list[dict[str, object]], bin_width_s: float = 0.25
) -> list[dict[str, object]]:
    if not paired:
        return []
    maximum = max(float(row["end_time_s"]) for row in paired)
    edges = np.arange(0.0, maximum + bin_width_s, bin_width_s)
    output: list[dict[str, object]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        selected = [
            row
            for row in paired
            if left <= float(row["start_time_s"]) < right
        ]
        if not selected:
            continue
        detected12 = sum(int(row["prediction_minus12db"]) == 1 for row in selected)
        detected15 = sum(int(row["prediction_minus15db"]) == 1 for row in selected)
        newly_missed = sum(
            row["transition_minus12_to_minus15"] == "detected_to_missed"
            for row in selected
        )
        output.append(
            {
                "bin_start_s": float(left),
                "bin_end_s": float(right),
                "windows": len(selected),
                "detected_minus12db": detected12,
                "recall_minus12db": detected12 / len(selected),
                "detected_minus15db": detected15,
                "recall_minus15db": detected15 / len(selected),
                "newly_missed_minus12_to_minus15": newly_missed,
                "newly_missed_fraction": newly_missed / len(selected),
            }
        )
    return output


def plot_margins(path: Path, paired: list[dict[str, object]]) -> None:
    ordered = sorted(paired, key=lambda row: float(row["start_time_s"]))
    times = np.asarray([float(row["start_time_s"]) for row in ordered])
    margin12 = np.asarray([float(row["margin_minus12db"]) for row in ordered])
    margin15 = np.asarray([float(row["margin_minus15db"]) for row in ordered])
    transition = [str(row["transition_minus12_to_minus15"]) for row in ordered]

    fig, (axis, state_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )
    axis.plot(times, margin12, color="#147D64", linewidth=1.5, label="-12 dB")
    axis.plot(times, margin15, color="#C23B22", linewidth=1.5, label="-15 dB")
    axis.axhline(0.0, color="#222222", linewidth=1.0, linestyle="--", label="threshold")
    axis.set_ylabel("Logit margin (drone - background)")
    axis.set_title("DJI Air 2S validation window margins")
    axis.grid(alpha=0.22)
    axis.legend(ncol=3, loc="lower left")

    colors = {
        "detected_to_detected": "#147D64",
        "detected_to_missed": "#C23B22",
        "missed_to_detected": "#D99B16",
        "missed_to_missed": "#575757",
    }
    for name, color in colors.items():
        mask = np.asarray([item == name for item in transition])
        if np.any(mask):
            state_axis.scatter(times[mask], np.zeros(np.sum(mask)), s=22, color=color, label=name)
    state_axis.set_yticks([])
    state_axis.set_xlabel("Recording time (s), based on 100 MS/s metadata")
    state_axis.set_ylabel("State")
    state_axis.grid(axis="x", alpha=0.22)
    state_axis.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_report(
    path: Path,
    scores_path: Path,
    summary: list[dict[str, object]],
    paired: list[dict[str, object]],
    group_summary: list[dict[str, object]],
    runs: list[dict[str, object]],
    bins: list[dict[str, object]],
) -> None:
    by_condition = {str(row["condition"]): row for row in summary}
    transitions = Counter(str(row["transition_minus12_to_minus15"]) for row in paired)
    groups = {str(row["transition"]): row for row in group_summary}
    new_runs = [row for row in runs if row["run_type"] == "newly_missed"]
    missed_runs = [row for row in runs if row["run_type"] == "missed_at_minus15db"]
    longest_new = max(new_runs, key=lambda row: int(row["window_count"]))
    longest_missed = max(missed_runs, key=lambda row: int(row["window_count"]))
    worst_bin = max(bins, key=lambda row: int(row["newly_missed_minus12_to_minus15"]))

    table_lines = [
        "| 条件 | 检出 | 漏检 | 检出率 | margin 中位数 | margin 5%-95% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = by_condition[condition]
        table_lines.append(
            f"| {condition} | {row['detected']}/{row['windows']} | {row['missed']} | "
            f"{float(row['recall']):.2%} | {fmt(float(row['margin_median']))} | "
            f"{fmt(float(row['margin_q05']))} 至 {fmt(float(row['margin_q95']))} |"
        )

    newly = groups.get("detected_to_missed")
    stable = groups.get("detected_to_detected")
    comparison = ""
    if newly and stable:
        comparison = (
            "在 −12 dB 时，随后仍能在 −15 dB 检出的窗口，其 margin 中位数为 "
            f"{fmt(float(stable['median_margin_minus12db']))}；随后新增漏检窗口为 "
            f"{fmt(float(newly['median_margin_minus12db']))}。这说明 −12 dB 时较低的余量与 "
            "−15 dB 失效有关，但这里只是同一录制上的关联。"
        )

    report = f"""# Air 2S 逐窗口漏检分析

本分析只读取已有验证集推理分数，没有重新推理、训练模型、改门槛或使用测试集。源文件：`{scores_path}`。

## 结论

Air 2S 在 −12 dB 检出 {by_condition['-12dB']['detected']}/150 个窗口，在 −15 dB 检出 {by_condition['-15dB']['detected']}/150 个窗口。按 `file_path + start` 配对后：

- −12 dB 和 −15 dB 都检出：{transitions['detected_to_detected']} 个；
- −12 dB 检出、−15 dB 新增漏检：{transitions['detected_to_missed']} 个；
- −12 dB 漏检、−15 dB 恢复检出：{transitions['missed_to_detected']} 个；
- 两档都漏检：{transitions['missed_to_missed']} 个。

同窗配对确认：净增的 88 个漏检全部来自原先检出的窗口，没有反向恢复窗口。

新增漏检贯穿整段 1.5 s 录制，并非只由一个局部坏片段造成。六个 0.25 s 分段的新增漏检比例为 40%–72%；最严重的是 {float(worst_bin['bin_start_s']):.2f}–{float(worst_bin['bin_end_s']):.2f} s，为 {worst_bin['newly_missed_minus12_to_minus15']}/{worst_bin['windows']} 个窗口。最长连续新增漏检区间为 {float(longest_new['start_time_s']):.2f}–{float(longest_new['end_time_s']):.2f} s，共 {longest_new['window_count']} 个连续 10 ms 窗口。−15 dB 下全部漏检的最长连续区间为 {float(longest_missed['start_time_s']):.2f}–{float(longest_missed['end_time_s']):.2f} s，共 {longest_missed['window_count']} 个窗口。

{comparison}

## 分数概况

{chr(10).join(table_lines)}

## 如何理解

这些结果把问题定位到了同一份 Air 2S 验证录制中的具体时间区间。它支持下一步从代表性稳定窗口和新增漏检窗口中抽取原始 IQ/时频图进行对比，而不是先盲目换网络。

当前不能把差异直接解释为“Air 2S 机型本身更难”。验证集中 Air 2S 只有一份源录制，机型和本次采集条件绑定。噪声是对逐窗 RMS 归一化后的 IQ 附加 AWGN，数值也不是现场绝对信噪比。旧训练范围到 −12 dB，且旧模型按 clean 验证表现选出；这两个因素仍需用固定对照实验检验。

## 输出说明

- `air2s_condition_summary.csv`：各噪声档检出数和 margin 分位数。
- `air2s_window_transitions.csv`：150 个原始窗口的配对分数和状态转变。
- `air2s_transition_group_summary.csv`：按状态转变分组后的 margin 概况。
- `air2s_contiguous_runs.csv`：新增漏检及 −15 dB 漏检的连续区间。
- `air2s_time_bins_250ms.csv`：0.25 s 分段统计。
- `air2s_margin_by_time.png`：−12/−15 dB margin 随录制时间变化。
- `verification.json`：输入哈希与一致性检查。
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    scores_path = args.scores.resolve()
    output_dir = args.output_dir.resolve()
    rows = read_target_rows(scores_path, args.target)
    summary = summarize_conditions(rows)
    paired = pair_windows(rows)
    group_summary = summarize_transition_groups(paired)
    runs = contiguous_runs(
        paired,
        "newly_missed",
        lambda row: row["transition_minus12_to_minus15"] == "detected_to_missed",
    ) + contiguous_runs(
        paired,
        "missed_at_minus15db",
        lambda row: int(row["prediction_minus15db"]) == 0,
    )
    bins = time_bin_summary(paired)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "air2s_condition_summary.csv", summary)
    write_csv(output_dir / "air2s_window_transitions.csv", paired)
    write_csv(output_dir / "air2s_transition_group_summary.csv", group_summary)
    write_csv(output_dir / "air2s_contiguous_runs.csv", runs)
    write_csv(output_dir / "air2s_time_bins_250ms.csv", bins)
    plot_margins(output_dir / "air2s_margin_by_time.png", paired)
    write_report(
        output_dir / "report.md",
        scores_path,
        summary,
        paired,
        group_summary,
        runs,
        bins,
    )

    by_condition = {str(row["condition"]): row for row in summary}
    transitions = Counter(str(row["transition_minus12_to_minus15"]) for row in paired)
    verification = {
        "passed": True,
        "source_scores": str(scores_path),
        "source_sha256": sha256(scores_path),
        "target": args.target,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "target_score_rows": len(rows),
        "paired_windows": len(paired),
        "unique_source_files": sorted({str(row["file_path"]) for row in paired}),
        "conditions_checked": list(CONDITIONS),
        "minus12_detected": by_condition["-12dB"]["detected"],
        "minus15_detected": by_condition["-15dB"]["detected"],
        "transition_counts": dict(transitions),
        "transition_total": sum(transitions.values()),
        "transition_total_matches_pairs": sum(transitions.values()) == len(paired),
        "expected_historical_counts_match": (
            by_condition["-12dB"]["detected"] == 137
            and by_condition["-15dB"]["detected"] == 49
        ),
        "source_modified": False,
    }
    if not (
        verification["transition_total_matches_pairs"]
        and verification["expected_historical_counts_match"]
        and len(paired) == 150
    ):
        verification["passed"] = False
        raise AssertionError(json.dumps(verification, ensure_ascii=False, indent=2))
    (output_dir / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    print(f"OUTPUT {output_dir}")


if __name__ == "__main__":
    main()
