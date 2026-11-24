# -*- coding: utf-8 -*-
"""Step 13: SAME drone model, SAME video-bandwidth setting, data/ (receiver A) vs v3 (receiver B).
DJI Mini 5 Pro: data/ 007 standby 10/20/40M  <->  v3 mini5pro idle bw10/20/40 (raw JSON declared.bw), both bands.
Autel EVO II Pro V3 (2.4G only): data/ 011 standby 10M <-> v3 evo2pro_v3 idle (bw not recorded).
Rows: (1) data/ spectrogram 200 ms  (2) v3 spectrogram 200 ms  -- one colour scale, dB above own floor
      (3) time-averaged spectrum over the whole recording (dB above own floor)
      (4) floor shape (per-bin 10th percentile minus median)
Per-recording numbers = medians over that recording's keep=True 10 ms training windows (window_metrics_L10.csv).
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
C_DATA, C_V3 = "#2a78d6", "#eb6834"
NCELL = 200
blocks = list(csv.DictReader(open(T / "blocks.csv", encoding="utf-8-sig")))
lvl = {r["group_id"]: float(r["floor_level_db"]) for r in csv.DictReader(open(O / "floor_per_recording.csv", encoding="utf-8"))}
wm = list(csv.DictReader(open(O / "window_metrics_L10.csv", encoding="utf-8")))
Z = np.load(O / "floor_curves.npz")

PAIRS = {
    "2.4GHz": [("Mini 5 Pro · 图传 10M", "data:007_2.4GHZ_10M_combination-signal__2026-09-20-17-10-12", "v3:20260709-025325_mini5pro"),
               ("Mini 5 Pro · 图传 20M", "data:007_2.4GHz_20M-signal__2026-09-20-20-27-45", "v3:20260709-025429_mini5pro"),
               ("Mini 5 Pro · 图传 40M", "data:007_2.4GHz_40M-signal__2026-09-20-20-28-25", "v3:20260709-025530_mini5pro"),
               ("EVO II Pro V3 · data 10M / v3 未记录", "data:011_2.4GHz_10M-signal__2026-09-20-18-54-38", "v3:2026-07-09_urban_20260709-033510_evo2pro_v3")],
    "5.8GHz": [("Mini 5 Pro · 图传 10M", "data:007_5.8GHz_10M-signal__2026-09-20-20-34-40", "v3:20260709-035600_mini5pro"),
               ("Mini 5 Pro · 图传 20M", "data:007_5.8GHz_20M-signal__2026-09-20-20-34-08", "v3:20260709-035703_mini5pro"),
               ("Mini 5 Pro · 图传 40M", "data:007_5.8GHz_40M-signal__2026-09-20-20-33-38", "v3:20260709-035819_mini5pro")],
}


def cells(gid):
    bl = [b for b in blocks if b["group_id"] == gid]
    return np.concatenate([np.load(T / "cells" / Path(b["block_file"]).stem / "mean.npy") for b in bl])


def stats(gid):
    s = [x for x in wm if x["group_id"] == gid]
    return np.median([float(x["peak_db"]) for x in s]), np.median([float(x["occ"]) for x in s]) * 100, len(s)


rows_out = []
for band, pairs in PAIRS.items():
    nfft = 512 if band == "2.4GHz" else 1024
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / FS[band])) / 1e6 + FC[band]
    inb = np.abs(f - FC[band]) <= W[band]; fb = f[inb]
    nc = len(pairs)
    fig, axs = plt.subplots(4, nc, figsize=(4.6 * nc, 16), gridspec_kw=dict(height_ratios=[1.3, 1.3, 0.8, 0.8], hspace=0.42, wspace=0.18))
    for j, (name, gd, gv) in enumerate(pairs):
        pk_d, oc_d, n_d = stats(gd); pk_v, oc_v, n_v = stats(gv)
        rows_out.append((band, name, pk_d, oc_d, pk_v, oc_v))
        for i, (g, who, pk, oc, n) in enumerate([(gd, "data/ 接收机A", pk_d, oc_d, n_d), (gv, "v3 接收机B", pk_v, oc_v, n_v)]):
            M = cells(g)
            ax = axs[i, j]
            im = ax.imshow(M[:NCELL, inb] - lvl[g], aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=40,
                           extent=[fb[0], fb[-1], 0, NCELL], interpolation="nearest")
            head = f"{name}\n" if i == 0 else ""
            ax.set_title(f"{head}{who}｜最强 {pk:.0f} dB｜占满 {oc:.0f}%", fontsize=9.5)
            if j == 0:
                ax.set_ylabel(("① data/（9月实验室）" if i == 0 else "② v3（7月城区）") + "\n时间 (ms)")
        # (3) time-averaged spectrum
        ax = axs[2, j]
        for g, col, lab in [(gd, C_DATA, "data/"), (gv, C_V3, "v3")]:
            M = cells(g)[:, inb]
            avg = 10 * np.log10(np.mean(10 ** (M / 10), axis=0)) - lvl[g]
            ax.plot(fb, avg, color=col, lw=1.3, label=lab)
        ax.set_ylim(-2, 35); ax.grid(alpha=0.25); ax.legend(fontsize=8, frameon=False, loc="upper right")
        if j == 0:
            ax.set_ylabel("③ 全录音平均频谱\n高出底噪 (dB)")
        # (4) floor shape
        ax = axs[3, j]
        for g, col, lab in [(gd, C_DATA, "data/"), (gv, C_V3, "v3")]:
            m = Z[f"{g}|mask"]; c = Z[f"{g}|p10"]
            ax.plot(fb, np.where(m, c - np.median(c[m]), np.nan)[inb], color=col, lw=1.3, label=lab)
        ax.set_ylim(*((-2.5, 4.5) if band == "2.4GHz" else (-1.3, 2.5))); ax.grid(alpha=0.25)
        ax.axhline(0, color="#52514e", lw=0.5); ax.set_xlabel("频率 (MHz)"); ax.legend(fontsize=8, frameon=False, loc="upper right")
        if band == "2.4GHz":
            for a in (axs[0, j], axs[1, j], ax):
                for wf in (2412, 2437):
                    a.axvline(wf, color="#52514e", lw=0.6, ls="--")
        if j == 0:
            ax.set_ylabel("④ 底噪形状 (dB)")
    cb = fig.colorbar(im, ax=axs[:2, :], shrink=0.7, pad=0.01); cb.set_label("高出本录音底噪 (dB)")
    fig.suptitle(f"图9  {band}：同一型号、同一图传带宽设置，data/ 与 v3 对比（①② 每格 200 ms，同一色标；data/ 待机、v3 idle）\n"
                 "“最强 / 占满”= 该录音全部 10 ms 训练样本的中位数：最强信号高出底噪多少 dB / 高出底噪 6 dB 的格子占比",
                 fontsize=12, y=0.95)
    fig.savefig(O / f"图9_{band}_同型号对比.png", dpi=110, bbox_inches="tight"); plt.close(fig)
for r in rows_out:
    print("%s | %-34s | data peak %.0f dB occ %.0f%% | v3 peak %.0f dB occ %.0f%%" % r)
