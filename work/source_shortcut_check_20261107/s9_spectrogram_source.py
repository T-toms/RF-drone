# -*- coding: utf-8 -*-
"""Step 9: spectrograms showing HOW data/ differs from v3 (the 'recognise the source' path).
Columns: data/ drone (receiver A) | v3 drone (receiver B) | v3 background (receiver B)
Row 1: ordinary spectrogram, absolute dB, one shared colour scale (what the eye normally sees)
Row 2: same cells, colour = dB above this recording's own floor, clipped to -2..+3 dB (floor texture made visible)
Row 3: time-collapsed floor shape of this recording (10th percentile per bin), other two columns in grey
Source: Task F cells (1 ms x 120 kHz for 2.4G / 97.7 kHz for 5.8G), i.e. the cropped, notched IQ that training uses.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
O = Path(__file__).resolve().parent
T = Path(r"C:/Users/Administrator/Desktop/数据集处理/work/crop_slice_20261107")
FS = {"2.4GHz": 61440000, "5.8GHz": 100000000}
FC = {"2.4GHz": 2430.0, "5.8GHz": 5800.0}
W = {"2.4GHz": 25.0, "5.8GHz": 45.0}
COL = ["#2a78d6", "#eb6834", "#1baf7a"]
NCELL = 300

blocks = list(csv.DictReader(open(T / "blocks.csv", encoding="utf-8-sig")))
recs = {r["group_id"]: r for r in csv.DictReader(open(O / "floor_per_recording.csv", encoding="utf-8"))}
Z = np.load(O / "floor_curves.npz")


def pick(band, ds, label, prefer):
    ids = [g for g, r in recs.items() if r["band"] == band and r["dataset"] == ds and r["label"] == label]
    for p in prefer:
        for g in ids:
            if p in g:
                return g
    return ids[0]


CHOICE = {
    "2.4GHz": [pick("2.4GHz", "data", "drone", ["001_2.4GHZ_10M_combination"]),
               pick("2.4GHz", "v3", "drone", ["025429_mini5pro"]),
               pick("2.4GHz", "v3", "background", ["041641_ambient"])],
    "5.8GHz": [pick("5.8GHz", "data", "drone", ["007_5.8GHz_20M"]),
               pick("5.8GHz", "v3", "drone", ["035819_mini5pro"]),
               pick("5.8GHz", "v3", "background", ["034952_ambient"])],
}
TITLE = ["data/ 无人机（接收机 A，9月实验室）", "v3 无人机（接收机 B，7月城区）", "v3 背景·无无人机（接收机 B，7月城区）"]


def cells(gid):
    b = [x for x in blocks if x["group_id"] == gid][0]
    return np.load(T / "cells" / Path(b["block_file"]).stem / "mean.npy")[:NCELL]


for band in ["2.4GHz", "5.8GHz"]:
    gids = CHOICE[band]
    nfft = 512 if band == "2.4GHz" else 1024
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / FS[band])) / 1e6 + FC[band]
    inb = np.abs(f - FC[band]) <= W[band]
    fb = f[inb]
    lv = [float(recs[g]["floor_level_db"]) for g in gids]
    vmin, vmax = min(lv) - 3, max(lv) + 30
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 1.25, 0.8], width_ratios=[1, 1, 1, 0.04], hspace=0.42, wspace=0.12)
    shapes = []
    for g in gids:
        m = Z[f"{g}|mask"]; c = Z[f"{g}|p10"]
        shapes.append(np.where(m, c - np.median(c[m]), np.nan)[inb])
    for j, g in enumerate(gids):
        M = cells(g)[:, inb]
        ext = [fb[0], fb[-1], 0, len(M)]
        a1 = fig.add_subplot(gs[0, j])
        im1 = a1.imshow(M, aspect="auto", origin="lower", cmap="Blues", vmin=vmin, vmax=vmax, extent=ext, interpolation="nearest")
        a1.set_title(f"{TITLE[j]}\n{recs[g]['model'] if recs[g]['model'] != 'none' else ''}  底噪电平 {lv[j]:.1f} dB", fontsize=10)
        a2 = fig.add_subplot(gs[1, j], sharex=a1)
        im2 = a2.imshow(M - lv[j], aspect="auto", origin="lower", cmap="RdBu_r", vmin=-3, vmax=3, extent=ext, interpolation="nearest")
        a3 = fig.add_subplot(gs[2, j], sharex=a1)
        for k in range(3):
            if k != j:
                a3.plot(fb, shapes[k], color="#b5b4ae", lw=0.8)
        a3.plot(fb, shapes[j], color=COL[j], lw=1.6)
        a3.axhline(0, color="#52514e", lw=0.5)
        a3.set_ylim(*((-2.5, 4.5) if band == "2.4GHz" else (-1.2, 2.5)))
        a3.grid(alpha=0.25); a3.set_xlabel("频率 (MHz)")
        if band == "2.4GHz":
            for ax in (a1, a2, a3):
                for wf in (2412, 2437):
                    ax.axvline(wf, color="#52514e", lw=0.6, ls="--")
        if j == 0:
            a1.set_ylabel("① 普通时频图\n时间 (ms)")
            a2.set_ylabel("② 只看底噪（放大）\n时间 (ms)")
            a3.set_ylabel("③ 底噪形状 (dB)\n彩色=本列，灰=另两列")
        else:
            for ax in (a1, a2):
                ax.set_yticklabels([])
    c1 = fig.colorbar(im1, cax=fig.add_subplot(gs[0, 3])); c1.set_label("功率 (dB，三列同一标尺)")
    c2 = fig.colorbar(im2, cax=fig.add_subplot(gs[1, 3])); c2.set_label("比本录音底噪高/低 (dB)")
    if band == "2.4GHz":
        sup = ("2.4 GHz（Task F 裁剪后 2405–2455 MHz，0.3 s）：data/ 的底噪是干净的，v3 的底噪里铺满了城区环境信号 —— 有没有无人机都一样\n"
               "① data/ 除无人机外底色均匀浅白；v3 两列到处是低电平块状信号，底噪电平高 2.5 dB　"
               "② 放大后 v3 两列几乎整片深红，data/ 几乎全白　③ v3 两列在 WiFi1/6 (虚线) 处都有 3–4 dB 鼓包，data/ 是平线")
    else:
        sup = ("5.8 GHz（Task F 裁剪后 5755–5845 MHz，0.3 s）：data/ 与 v3 的区别 = 接收机自己的底噪形状（差别 <1 dB，① 普通时频图里肉眼看不出）\n"
               "② 放大后 data/ 左侧 5755–5770 偏蓝、往右逐渐变红（倾斜）；v3 两列从左到右颜色均匀　"
               "③ v3 无人机与 v3 背景两条线几乎重合（两端下垂 ~0.6 dB），data/ 是一条斜线、右端不下垂")
    fig.suptitle(sup, fontsize=12, y=0.995)
    fig.savefig(O / f"图6_{band}_时频图_来源差异.png", dpi=115, bbox_inches="tight")
    plt.close(fig)
    print(band, gids, ["%.1f" % x for x in lv])
