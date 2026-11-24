from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.font_manager import FontProperties


ROOT = Path(r"C:\Users\Administrator\Desktop\数据集处理")
OUT = ROOT / "reports" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
FONT_BOLD = FontProperties(fname=str(FONT_PATH), weight="bold") if FONT_PATH.exists() else None


def text(ax, x, y, value, *, size=11, bold=False, ha="center", va="center"):
    ax.text(
        x,
        y,
        value,
        fontsize=size,
        fontproperties=FONT_BOLD if bold else FONT,
        ha=ha,
        va=va,
        color="black",
    )


def box(ax, xy, width, height, label, *, fill="white", size=10.5):
    rect = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.3,
        edgecolor="black",
        facecolor=fill,
    )
    ax.add_patch(rect)
    text(ax, xy[0] + width / 2, xy[1] + height / 2, label, size=size, bold=True)


def arrow(ax, start, end, label=None, label_offset=(0, 0.05)):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.4),
    )
    if label:
        text(
            ax,
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1],
            label,
            size=9.5,
        )


def build_flow():
    fig, ax = plt.subplots(figsize=(12.6, 3.2), dpi=180)
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    box(ax, (0.25, 1.75), 2.0, 0.75, "原始 I/Q 信号\n100 万采样点")
    box(ax, (2.75, 1.75), 2.15, 0.75, "STFT 时频图\nn_fft=512")
    box(ax, (5.4, 1.75), 2.35, 0.75, "阶段 1  ResNet18\n背景 / 无人机")
    box(ax, (9.4, 2.05), 2.25, 0.65, "输出：背景", fill="#f2f2f2")
    box(ax, (8.0, 0.5), 2.35, 0.75, "阶段 2  ResNet18\nPhantom 4 Pro / Air 2S")
    box(ax, (10.75, 0.5), 1.55, 0.75, "输出机型", fill="#f2f2f2")

    arrow(ax, (2.25, 2.125), (2.75, 2.125))
    arrow(ax, (4.9, 2.125), (5.4, 2.125))
    arrow(ax, (7.75, 2.23), (9.4, 2.37), "背景", (0, 0.1))
    arrow(ax, (6.65, 1.75), (8.0, 1.25), "无人机", (0.05, 0.03))
    arrow(ax, (10.35, 0.875), (10.75, 0.875))

    text(ax, 6.3, 3.0, "无人机检测与机型识别流程", size=13, bold=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "progress_logic.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(ax, ymin, ymax):
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="black", labelsize=8.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT)


def build_results():
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.35), dpi=180, gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]})

    nfft = [512, 1024, 2048, 4096]
    detection = [99.04, 98.07, 97.40, 97.98]
    ax = axes[0]
    ax.plot(range(4), detection, color="black", marker="o", linewidth=1.5, markersize=5)
    ax.set_xticks(range(4), [str(x) for x in nfft])
    style_axis(ax, 97.0, 99.4)
    ax.set_title("检测阶段：不同 n_fft", fontproperties=FONT_BOLD, fontsize=11)
    ax.set_xlabel("n_fft", fontproperties=FONT, fontsize=9)
    ax.set_ylabel("平衡准确率（%）", fontproperties=FONT, fontsize=9)
    for idx, value in enumerate(detection):
        ax.text(idx, value + 0.08, f"{value:.2f}", ha="center", va="bottom", fontsize=8.5, fontproperties=FONT)

    snr_labels = ["正常", "−10", "−12", "−15"]
    cascade = [99.75, 99.98, 99.63, 95.14]
    ax = axes[1]
    ax.plot(range(4), cascade, color="black", marker="s", linewidth=1.5, markersize=5)
    ax.set_xticks(range(4), snr_labels)
    style_axis(ax, 94.5, 100.3)
    ax.set_title("完整三分类：不同 SNR", fontproperties=FONT_BOLD, fontsize=11)
    ax.set_xlabel("SNR（dB）", fontproperties=FONT, fontsize=9)
    ax.set_ylabel("平衡准确率（%）", fontproperties=FONT, fontsize=9)
    for idx, value in enumerate(cascade):
        ax.text(idx, value + 0.15, f"{value:.2f}", ha="center", va="bottom", fontsize=8.5, fontproperties=FONT)

    ax = axes[2]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    text(ax, 0.5, 0.91, "其他关键结果", size=11, bold=True)
    labels = [
        ("能量门限", "98.30%"),
        ("PSD+SVM", "100.00%"),
        ("阶段 2 机型识别", "99.78% ± 0.09 个百分点"),
    ]
    y_values = [0.68, 0.45, 0.22]
    for (label, value), y in zip(labels, y_values):
        ax.add_patch(patches.Rectangle((0.08, y - 0.09), 0.84, 0.18, facecolor="#f2f2f2", edgecolor="black", linewidth=1.0))
        text(ax, 0.13, y + 0.025, label, size=9.5, bold=True, ha="left")
        text(ax, 0.87, y - 0.035, value, size=10.5, bold=True, ha="right")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "progress_results.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_confusion():
    source = ROOT / "dronerfa_baseline" / "outputs" / "cascade3_evaluation_20260919" / "confusion_matrices.csv"
    class_names = ["Background", "DJI Phantom 4 Pro", "DJI Air 2S"]
    class_index = {name: idx for idx, name in enumerate(class_names)}
    conditions = ["clean", "-15"]
    matrices = {condition: np.zeros((3, 3), dtype=float) for condition in conditions}

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            condition = row["condition"]
            if row["dataset"] != "heldout" or condition not in matrices:
                continue
            true_name = row["true_class_name"]
            pred_name = row["predicted_class_name"]
            if true_name in class_index and pred_name in class_index:
                matrices[condition][class_index[true_name], class_index[pred_name]] += int(row["count"])

    fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.55), dpi=180)
    titles = ["正常条件", "−15 dB AWGN"]
    for ax, condition, title in zip(axes, conditions, titles):
        matrix = matrices[condition]
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(normalized, cmap="Greys", vmin=0, vmax=1)
        ax.set_title(title, fontproperties=FONT_BOLD, fontsize=11)
        ax.set_xticks(range(3), class_names, rotation=20, ha="right")
        ax.set_yticks(range(3), class_names)
        ax.set_xlabel("预测类别", fontproperties=FONT, fontsize=9)
        ax.set_ylabel("真实类别", fontproperties=FONT, fontsize=9)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(FONT)
            label.set_fontsize(8.5)
        for i in range(3):
            for j in range(3):
                value = normalized[i, j] * 100
                color = "white" if normalized[i, j] > 0.5 else "black"
                ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=color, fontsize=9)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.9)

    fig.suptitle("完整三分类混淆矩阵（3 次重复实验汇总）", fontproperties=FONT_BOLD, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=2.5)
    fig.savefig(OUT / "progress_confusion.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    build_results()
    build_confusion()
    print(OUT / "progress_results.png")
    print(OUT / "progress_confusion.png")
