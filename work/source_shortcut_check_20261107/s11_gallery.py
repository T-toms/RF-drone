# -*- coding: utf-8 -*-
"""Step 11: gallery of NORMAL spectrograms, many recordings side by side, one colour scale.
Row 1 data/ drones (receiver A) : 7 different models, standby + flight mixed
Row 2 v3 drones (receiver B)    : one recording per model
Row 3 v3 background (receiver B): spread over urban / lab / home (/ receiver self-noise)
Each panel = first 200 ms of the recording, colour = dB above the recording's own floor (0..40),
so absolute gain/normalisation plays no role. Picks are seeded-random within each model/campaign.
Plus a per-window figure (all keep=True 10 ms training windows): strongest signal vs. filled fraction.
"""
import csv, collections
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
NC, NCELL = 7, 200
rng = np.random.default_rng(2026)
blocks = list(csv.DictReader(open(T / "blocks.csv", encoding="utf-8-sig")))
recs = list(csv.DictReader(open(O / "floor_per_recording.csv", encoding="utf-8")))
lvl = {r["group_id"]: float(r["floor_level_db"]) for r in recs}
CAMP = {"2026-07-09_urban": "城区", "2026-07-10_lab": "实验室", "2026-07-11_home": "家", "2026-07-16_lab": "实验室(07-16)", "noise_adrv": "实验室(noise_adrv)"}
STATE = {"standby": "待机", "flight": "飞行", "unknown": "idle"}


def one_per(sel, key, n):
    by = collections.defaultdict(list)
    for r in sel:
        by[key(r)].append(r)
    keys = sorted(by)
    rng.shuffle(keys)
    out = [by[k][rng.integers(len(by[k]))] for k in keys]
    return out[:n]


def bg_pick(sel, n):
    by = collections.defaultdict(list)
    for r in sel:
        by[r["campaign"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    out, i = [], 0
    while len(out) < n and any(by.values()):
        for c in sorted(by):
            if by[c] and len(out) < n:
                out.append(by[c].pop())
    return out


def cells(gid):
    bl = [b for b in blocks if b["group_id"] == gid]
    return np.concatenate([np.load(T / "cells" / Path(b["block_file"]).stem / "mean.npy") for b in bl])[:NCELL]


for band in ["2.4GHz", "5.8GHz"]:
    sel = [r for r in recs if r["band"] == band]
    dd = [r for r in sel if r["dataset"] == "data"]
    # data/: alternate standby / flight so both states appear
    dpick = one_per(dd, lambda r: r["model"], 14)
    st = [r for r in dpick if r["state"] == "standby"][:4] + [r for r in dpick if r["state"] == "flight"][:3]
    dpick = (st + [r for r in dpick if r not in st])[:NC]
    vpick = one_per([r for r in sel if r["dataset"] == "v3" and r["label"] == "drone"], lambda r: r["model"], NC)
    bpick = bg_pick([r for r in sel if r["label"] == "background"], NC)
    rows = [("data/ 无人机\n接收机 A·9月实验室", dpick), ("v3 无人机\n接收机 B·7月城区", vpick), ("v3 背景(无无人机)\n接收机 B", bpick)]
    nfft = 512 if band == "2.4GHz" else 1024
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / FS[band])) / 1e6 + FC[band]
    inb = np.abs(f - FC[band]) <= W[band]
    fig, axs = plt.subplots(3, NC, figsize=(2.55 * NC, 12.2), sharey=True)
    for i, (rname, picks) in enumerate(rows):
        for j in range(NC):
            ax = axs[i, j]
            if j >= len(picks):
                ax.axis("off"); continue
            r = picks[j]
            M = cells(r["group_id"])[:, inb] - lvl[r["group_id"]]
            im = ax.imshow(M, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=40,
                           extent=[f[inb][0], f[inb][-1], 0, len(M)], interpolation="nearest")
            if r["label"] == "background":
                t = f"{CAMP.get(r['campaign'], r['campaign'])}"
            elif r["dataset"] == "data":
                t = f"{r['model']}·{STATE.get(r['state'], r['state'])}"
            else:
                t = r["model"]
            ax.set_title(t, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.set_xticks([f[inb][0] + 5, FC[band], f[inb][-1] - 5])
            ax.set_xticklabels([f"{f[inb][0] + 5:.0f}", f"{FC[band]:.0f}", f"{f[inb][-1] - 5:.0f}"])
            if j == 0:
                ax.set_ylabel(rname + "\n时间 (ms)", fontsize=10)
                ax.add_patch(plt.Rectangle((f[inb][0], 0), 2.5, 10, fill=False, ec="#eb6834", lw=1.5))
                ax.text(f[inb][0] + 3, 3, "10 ms=1个样本", fontsize=7, color="#eb6834", va="bottom")
    for ax in axs[-1]:
        ax.set_xlabel("频率 (MHz)", fontsize=8)
    cb = fig.colorbar(im, ax=axs, shrink=0.6, pad=0.01); cb.set_label("高出本录音底噪 (dB)")
    if band == "2.4GHz":
        sup = ("图7  2.4 GHz 普通时频图（每格 200 ms，Task F 裁剪后 2405–2455 MHz，所有格同一色标）\n"
               "第1行 data/ 无人机：除无人机信号外是干净的白底　第2行 v3 无人机 与 第3行 城区/家 背景：底下都铺着一层浅蓝杂信号　"
               "第3行 实验室背景：有很多强突发")
    else:
        sup = ("图7  5.8 GHz 普通时频图（每格 200 ms，Task F 裁剪后 5755–5845 MHz，所有格同一色标）\n"
               "第1行 data/ 无人机：信号强、占的频宽大　第2行 v3 无人机：弱得多、稀疏，mini3pro / evo2pro_v3 几乎看不见　"
               "第3行 v3 背景：城区几乎空白，实验室/家有突发")
    fig.suptitle(sup, fontsize=12)
    fig.savefig(O / f"图7_{band}_普通时频图对照.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(band, [[p["group_id"] for p in r[1]] for r in rows])

# ---------- per-window figure ----------
wm = list(csv.DictReader(open(O / "window_metrics_L10.csv", encoding="utf-8")))
GR = [(("data", "drone"), "data/ 无人机（接收机 A）", "#2a78d6"), (("v3", "drone"), "v3 无人机（接收机 B）", "#eb6834"),
      (("v3", "background"), "v3 背景（接收机 B）", "#1baf7a")]
fig, axs = plt.subplots(2, 2, figsize=(14, 9.5))
for k, band in enumerate(["2.4GHz", "5.8GHz"]):
    for (key, name, col) in GR:
        s = [x for x in wm if x["band"] == band and (x["dataset"], x["label"]) == key]
        p = np.array([float(x["peak_db"]) for x in s]); o = np.array([float(x["occ"]) for x in s]) * 100
        bins = np.linspace(-5, 60, 66)
        axs[0, k].hist(p, bins=bins, density=True, histtype="step", lw=2, color=col,
                       label=f"{name}  n={len(s)}  中位 {np.median(p):.0f} dB")
        axs[1, k].hist(o, bins=np.linspace(0, 80, 41), density=True, histtype="step", lw=2, color=col,
                       label=f"{name}  中位 {np.median(o):.0f}%")
    axs[0, k].set_title(f"{band}：每个 10 ms 样本里最强信号高出底噪多少", fontsize=11, loc="left")
    axs[0, k].set_xlabel("最强信号（99 百分位）高出底噪 (dB)")
    axs[1, k].set_title(f"{band}：每个 10 ms 样本的时频图有多少面积被信号占满", fontsize=11, loc="left")
    axs[1, k].set_xlabel("高出底噪 6 dB 的格子占比 (%)")
    for a in axs[:, k]:
        a.legend(fontsize=9, frameon=False); a.grid(alpha=0.25); a.set_ylabel("密度")
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.suptitle("图8  全部训练样本（Task F 10 ms、keep=True）的统计：同样是“无人机”，data/ 与 v3 的样本长得很不一样", fontsize=12)
fig.tight_layout(); fig.savefig(O / "图8_10ms样本统计.png", dpi=120); plt.close(fig)
print("ok")
