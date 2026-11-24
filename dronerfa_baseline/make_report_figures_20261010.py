#!/usr/bin/env python3
"""Report figures and full confusion-matrix tables for the 2026-10-10 supervisor report.

Reads only frozen result files (no IQ, no model). Writes PNGs and one markdown
file of count tables into reports/汇报图表_20261010/.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "reports/汇报图表_20261010"
CASCADE = HERE / "outputs/cascade12_development_20260927"
STAGE1_VAL = HERE / "outputs/stage1_12drones_20260926/seed_2026/validation_scores_epoch_03.csv"
STAGE2_VAL = HERE / "outputs/stage2_12drones_20260927/seed_2027/validation_scores_epoch_04.csv"
MINUS15 = HERE / "outputs/stage2_minus15_20260930/validation_comparison"
PROBE = HERE / "outputs/stage2_frequency_probe_20261010/probe_summary.json"

CODES = ("T0001", "T0010", "T0011", "T0100", "T0101", "T0110", "T0111",
         "T1000", "T1001", "T1010", "T1011", "T1100")
NAME = {"T0000": "背景", "T0001": "Phantom 3", "T0010": "Phantom 4 Pro", "T0011": "Matrice 200",
        "T0100": "Matrice 100", "T0101": "Air 2S", "T0110": "Mini 3 Pro", "T0111": "Inspire 2",
        "T1000": "Mavic Pro", "T1001": "Mini 2", "T1010": "Mavic 3（室内）",
        "T1011": "Matrice 300（室内）", "T1100": "Phantom 4 Pro RTK（室内）"}
COND = {"clean": "不加噪声", "-10dB": "信噪比 −10 dB", "-12dB": "信噪比 −12 dB", "-15dB": "信噪比 −15 dB"}
CASCADE_CLASSES = ("T0000",) + CODES

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES = ["#2a78d6", "#eb6834"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False,
                     "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#c3c2b7",
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def label(code: str) -> str:
    return f"{code} {NAME[code]}"


def plot_confusion(cm: np.ndarray, rows: list[str], cols: list[str], title: str, path: Path, note: str = "") -> None:
    """Row-normalised colour (= recall of the true class), each cell shows count and row %."""
    totals = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, totals, out=np.zeros_like(cm, dtype=float), where=totals > 0)
    h = 0.42 * len(rows) + 2.2
    w = 0.62 * len(cols) + 3.6
    fig, ax = plt.subplots(figsize=(w, h))
    ax.imshow(pct, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if cm[i, j] == 0:
                continue
            dark = pct[i, j] > 0.55
            p = pct[i, j] * 100
            ax.text(j, i, f"{cm[i, j]}\n{p:.1f}%" if p < 1 else f"{cm[i, j]}\n{p:.0f}%", ha="center", va="center", fontsize=7,
                    color=SURFACE if dark else INK, linespacing=1.0)
    ax.set_xticks(range(len(cols)), [label(c) for c in cols], rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(rows)), [f"{label(r)}（{int(totals[i, 0])}）" for i, r in enumerate(rows)], fontsize=7.5)
    ax.set_xlabel("模型判断为")
    ax.set_ylabel("真实机型（括号内为片段数）")
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=11, loc="left", color=INK)
    if note:
        fig.text(0.01, 0.005, note, fontsize=7.5, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def md_table(cm: np.ndarray, rows: list[str], cols: list[str]) -> str:
    head = "| 真实 \\ 判断 | " + " | ".join(cols) + " | 合计 | 正确率 |"
    sep = "|" + "---|" * (len(cols) + 3)
    lines = [head, sep]
    for i, r in enumerate(rows):
        total = int(cm[i].sum())
        j = cols.index(r) if r in cols else None
        acc = f"{cm[i, j] / total * 100:.1f}%" if total and j is not None else "—"
        cells = [f"**{v}**" if c == r else (str(v) if v else "·") for c, v in zip(cols, cm[i])]
        lines.append(f"| {r} {NAME[r]} | " + " | ".join(cells) + f" | {total} | {acc} |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# 汇报用完整结果表（2026-10-10）",
          "",
          "表中每一行是真实机型，每一列是模型判断的机型；对角线（加粗）是判对的片段数，“·”表示 0。",
          "“片段”= 一段 10 毫秒的信号（100 万个采样点），模型每次看一个片段给出一个判断。",
          "",
          "机型代号：" + "；".join(f"{c} = {NAME[c]}" for c in CASCADE_CLASSES),
          ""]
    figs = []

    # 1. Two-step system on test files
    cms = json.loads((CASCADE / "confusion_matrices.json").read_text(encoding="utf-8"))
    for role, role_cn, n in (("development_test", "测试集（新 10 种机型，20 个文件，每种 2 个文件）", 1),
                             ("development_test_exposed", "补充测试集（背景 + Phantom 4 Pro + Air 2S，6 个文件）", 2)):
        md += [f"## {n}. 两步识别系统在{role_cn}上的结果", "",
               "第 0 列“T0000 背景”= 第一步判断为“没有无人机”（漏检）。", ""]
        for cond, cond_cn in COND.items():
            cm = np.array(cms[f"{role}/{cond}"])
            keep = [i for i in range(13) if cm[i].sum() > 0]
            rows = [CASCADE_CLASSES[i] for i in keep]
            cols = list(CASCADE_CLASSES)
            sub = cm[keep]
            correct = sum(sub[k, keep[k]] for k in range(len(keep)))
            fn = f"{n:02d}_两步识别_{'测试集' if n == 1 else '补充测试集'}_{cond}.png"
            plot_confusion(sub, rows, cols, f"两步识别 · {role_cn.split('（')[0]} · {cond_cn}", OUT / fn,
                           note=f"颜色深浅 = 该真实机型的片段中被判成这一列的比例；整体判对 {correct}/{int(sub.sum())} 片段 = {correct / sub.sum() * 100:.1f}%")
            figs.append(fn)
            md += [f"### {cond_cn}（整体判对 {correct}/{int(sub.sum())} = {correct / sub.sum() * 100:.1f}%）", "",
                   f"图：`{fn}`", "", md_table(sub, rows, cols), ""]

    # 2. Step 2 alone on validation files (model selection data)
    rows_v = list(csv.DictReader(STAGE2_VAL.open(encoding="utf-8-sig", newline="")))
    md += ["## 3. 第二步（判断是哪种机型）单独在验证集上的结果", "",
           "验证集：12 种机型每种 1 个文件；这里把真实的无人机片段全部直接交给第二步，不经过第一步。",
           "这批数据被用来挑选训练轮次，所以结果偏乐观，仅作参考。", ""]
    for cond, cond_cn in COND.items():
        cm = np.zeros((12, 12), dtype=int)
        for r in rows_v:
            if r["condition"] == cond:
                cm[int(r["label"]), int(r["prediction"])] += 1
        acc = np.trace(cm) / cm.sum()
        fn = f"03_第二步_验证集_{cond}.png"
        plot_confusion(cm, list(CODES), list(CODES), f"第二步单独 · 验证集 · {cond_cn}", OUT / fn,
                       note=f"整体判对 {np.trace(cm)}/{cm.sum()} 片段 = {acc * 100:.1f}%")
        figs.append(fn)
        md += [f"### {cond_cn}（整体判对 {np.trace(cm)}/{cm.sum()} = {acc * 100:.1f}%）", "",
               f"图：`{fn}`", "", md_table(cm, list(CODES), list(CODES)), ""]

    # 3. Step 1 (drone present?) on validation and supplementary test
    s1 = list(csv.DictReader(STAGE1_VAL.open(encoding="utf-8-sig", newline="")))
    preds = list(csv.DictReader((CASCADE / "window_predictions.csv").open(encoding="utf-8-sig", newline="")))
    md += ["## 4. 第一步（有没有无人机）的结果", "",
           "| 数据 | 条件 | 背景判为背景 | 背景误判为无人机（误报） | 无人机判为无人机 | 无人机漏判为背景（漏检） | 误报率 | 漏检率 |",
           "|---|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2))
    for k, (src, src_cn) in enumerate(((s1, "验证集"), ([p for p in preds if p["role"] == "development_test_exposed"], "补充测试集"))):
        for j, (cond, cond_cn) in enumerate(COND.items()):
            cm = np.zeros((2, 2), dtype=int)
            for r in src:
                if r["condition"] != cond:
                    continue
                truth = int(r["label"]) if "label" in r else int(r["target_code"] != "T0000")
                pred = int(r["prediction"]) if "label" in r else int(r["stage1_prediction"])
                cm[truth, pred] += 1
            fpr, fnr = cm[0, 1] / cm[0].sum(), cm[1, 0] / cm[1].sum()
            md.append(f"| {src_cn} | {cond_cn} | {cm[0,0]} | {cm[0,1]} | {cm[1,1]} | {cm[1,0]} | {fpr*100:.1f}% | {fnr*100:.1f}% |")
            ax = axes[k, j]
            pct = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(pct, cmap=CMAP, vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{cm[a, b]}\n{pct[a, b]*100:.1f}%", ha="center", va="center", fontsize=8,
                            color=SURFACE if pct[a, b] > 0.55 else INK)
            ax.set_xticks([0, 1], ["判为背景", "判为无人机"], fontsize=8)
            ax.set_yticks([0, 1], ["真实背景", "真实无人机"] if j == 0 else ["", ""], fontsize=8)
            ax.set_title(f"{src_cn} · {cond_cn}", fontsize=9, loc="left")
            for s in ax.spines.values():
                s.set_visible(False)
    fig.tight_layout()
    fn = "04_第一步_有无无人机.png"
    fig.savefig(OUT / fn, dpi=150)
    plt.close(fig)
    figs.append(fn)
    md += ["", f"图：`{fn}`", ""]

    # 4. Recall per model per condition on the main test set
    pcm = list(csv.DictReader((CASCADE / "per_class_metrics.csv").open(encoding="utf-8-sig", newline="")))
    md += ["## 5. 测试集上每种机型的识别正确率（两步识别，按文件平均）", "",
           "| 机型 | " + " | ".join(COND.values()) + " |", "|---|" + "---|" * 4]
    by = defaultdict(dict)
    for r in pcm:
        by[(r["role"], r["target_code"])][r["condition"]] = float(r["file_equal_recall"])
    for (role, code), v in sorted(by.items()):
        if role == "development_test":
            md.append(f"| {code} {NAME[code]} | " + " | ".join(f"{v[c]*100:.1f}%" for c in COND) + " |")
    md.append("")

    # 5. -15 dB training experiment
    pc = list(csv.DictReader((MINUS15 / "per_condition.csv").open(encoding="utf-8-sig", newline="")))
    pcc = list(csv.DictReader((MINUS15 / "per_class_condition.csv").open(encoding="utf-8-sig", newline="")))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(pc))
    for k, (key, name) in enumerate((("baseline", "原模型（训练噪声到 −12 dB）"), ("minus15", "训练加入 −15 dB 噪声"))):
        vals = [float(r[key]) * 100 for r in pc]
        ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=SERIES[k], label=name, edgecolor=SURFACE, linewidth=2)
        for xi, v in zip(x, vals):
            ax.text(xi + (k - 0.5) * 0.36, v + 1, f"{v:.1f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [COND[r["condition"]] for r in pc])
    ax.set_ylim(0, 108)
    ax.set_ylabel("验证集正确率（%，按机型和文件平均）")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title("对比实验：训练时加入更强的噪声", fontsize=11, loc="left")
    fig.tight_layout()
    fn = "05_对比实验_训练加入负15dB噪声.png"
    fig.savefig(OUT / fn, dpi=150)
    plt.close(fig)
    figs.append(fn)
    md += ["## 6. 对比实验：训练时加入 −15 dB 噪声（2026-09-30 至 10-01）", "",
           "只改一处：训练时加的噪声从最强 −12 dB 扩展到 −15 dB，其他全部不变；在验证集上比较。", "",
           f"图：`{fn}`", "",
           "| 条件 | 原模型 | 加入 −15 dB 训练 | 变化 |", "|---|---|---|---|"]
    md += [f"| {COND[r['condition']]} | {float(r['baseline'])*100:.1f}% | {float(r['minus15'])*100:.1f}% | {float(r['delta'])*100:+.1f} |" for r in pc]
    md += ["", "每种机型：", "", "| 机型 | 条件 | 原模型 | 加入 −15 dB | 变化 |", "|---|---|---|---|---|"]
    md += [f"| {r['target_code']} {NAME[r['target_code']]} | {COND[r['condition']]} | {float(r['baseline_file_equal_recall'])*100:.1f}% | {float(r['minus15_file_equal_recall'])*100:.1f}% | {float(r['delta'])*100:+.1f} |"
           for r in sorted(pcc, key=lambda r: (r["target_code"], list(COND).index(r["condition"])))]
    md += ["", "结论：只有 −15 dB 明显变好，其余三个条件都变差，总体基本持平，没有替换原模型。", ""]

    # 6. Shift / mask probe
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    groups = [("T1011_S0101.mat", "Matrice 300 验证文件"),
              ("T1011_S0110.mat_w0-69", "Matrice 300 测试文件\n前 70 段（换信道前）"),
              ("T1011_S0110.mat_w70-159", "Matrice 300 测试文件\n后 90 段（换信道后）"),
              ("T1010_S0101.mat", "Mavic 3 验证文件"),
              ("T1100_S0101.mat", "Phantom 4 Pro RTK 验证文件")]
    edits = [("none", "不改动"),
             ("move_down", "把 2440–2470 MHz\n的信号挪到 2400–2430"),
             ("move_down_ctl", "把 2440–2470 MHz\n的信号挪到 2420–2450"),
             ("move_up", "把 2398–2423 MHz\n的信号挪到 2445–2470"),
             ("mask_low", "遮掉 2400–2425 MHz\n（疑似 WiFi）"),
             ("mask_ctl", "遮掉 2455–2480 MHz\n（对照）")]
    grid = np.array([[probe[g][e]["correct"] for e, _ in edits] for g, _ in groups])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]*100:.0f}%", ha="center", va="center", fontsize=9,
                    color=SURFACE if grid[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(edits)), [e for _, e in edits], fontsize=7.5)
    ax.set_yticks(range(len(groups)), [g for _, g in groups], fontsize=8)
    ax.set_xticks(np.arange(-.5, len(edits)), minor=True)
    ax.set_yticks(np.arange(-.5, len(groups)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("验证实验：在时频图上挪动或遮掉一段信号后，原模型（第二步）判对的比例（不加噪声）", fontsize=10, loc="left")
    fig.tight_layout()
    fn = "06_验证实验_挪动和遮挡信号.png"
    fig.savefig(OUT / fn, dpi=150)
    plt.close(fig)
    figs.append(fn)
    md += ["## 7. 验证实验：在时频图上挪动或遮掉一段信号（2026-10-10）", "",
           "不重新训练，直接修改输入给原模型的时频图，看判断怎么变。不改动时与原来的判断 100% 一致。", "",
           f"图：`{fn}`", "",
           "| 数据（片段数） | " + " | ".join(e.replace("\n", "") for _, e in edits) + " |", "|---|" + "---|" * len(edits)]
    for g, gn in groups:
        md.append(f"| {gn.replace(chr(10), '')}（{probe[g]['windows']}） | " +
                  " | ".join(f"{probe[g][e]['correct']*100:.0f}%（多判为 {probe[g][e]['top_predictions'][0][0]}）" for e, _ in edits) + " |")
    md.append("")

    (OUT / "完整结果表_20261010.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(figs))
    print("written", OUT)


if __name__ == "__main__":
    main()
