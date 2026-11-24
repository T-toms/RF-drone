# -*- coding: utf-8 -*-
"""Step 2: per-recording floor SHAPE (level removed), one thin line per recording, 3 panels per band."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

O = Path(__file__).resolve().parent
rs = list(csv.DictReader(open(O / "floor_per_recording.csv", encoding="utf-8")))
Z = np.load(O / "floor_curves.npz")
CAMP_COL = {"2026-07-09_urban": "#d62728", "2026-07-10_lab": "#2ca02c", "2026-07-11_home": "#ff7f0e",
            "2026-07-16_lab": "#9467bd", "noise_adrv": "black"}


def shape(gid, key="p10"):
    f, m, c = Z[f"{gid}|f"], Z[f"{gid}|mask"], Z[f"{gid}|{key}"]
    c = c - np.median(c[m])
    c = np.where(m, c, np.nan)
    return f, c


for band in ["2.4GHz", "5.8GHz"]:
    fig, axs = plt.subplots(3, 1, figsize=(13, 11), sharex=True, sharey=True)
    panels = [("data", "drone", "data/ 无人机录音（9月实验室采集，接收机 A）"),
              ("v3", "drone", "datasets_v3 无人机录音（7月，ADRV9009 接收机 B，全部 07-09 城区 25 dB）"),
              ("v3", "background", "datasets_v3 背景录音（接收机 B，按场景着色；黑线=接收机自噪声 noise_adrv）")]
    for ax, (ds, lab, title) in zip(axs, panels):
        sel = [r for r in rs if r["band"] == band and r["dataset"] == ds and r["label"] == lab]
        allc = []
        for r in sel:
            f, c = shape(r["group_id"])
            col = "#1f77b4" if ds == "data" else ("#d62728" if lab == "drone" else CAMP_COL.get(r["campaign"], "gray"))
            ax.plot(f, c, lw=0.6, alpha=0.5 if ds == "data" else 0.8, color=col)
            allc.append(c)
        ax.plot(f, np.nanmedian(np.array(allc), 0), lw=2, color="k" if ds == "data" else "#333333", ls="--", label="中位数")
        if lab == "background":
            for camp, col in CAMP_COL.items():
                ax.plot([], [], color=col, label=camp)
        ax.set_title(f"{title}   n={len(sel)} 条录音", fontsize=11)
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right", ncol=3)
        ax.set_ylabel("底噪形状 (dB)\n每条录音减去自身中位数")
    axs[-1].set_xlabel("绝对频率 (MHz)")
    axs[0].set_ylim(-4, 8 if band == "2.4GHz" else 4)
    fig.suptitle(f"{band}：每条录音的底噪形状（每个频点取全录音 1 ms 小格的第 10 百分位）", fontsize=13)
    fig.tight_layout()
    fig.savefig(O / f"fig_explore_shape_{band}.png", dpi=110)
    plt.close(fig)
print("ok")
