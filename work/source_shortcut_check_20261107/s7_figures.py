# -*- coding: utf-8 -*-
"""Step 7: report figures (Chinese) for the teacher.
fig1 data composition | fig2 spectrogram comparison 2.4G | fig3 2.4G floor (WiFi = environment)
fig4 5.8G floor (receiver fingerprint) | fig5 shortcut test per test recording
"""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
O = Path(__file__).resolve().parent
T = Path(r"C:/Users/Administrator/Desktop/数据集处理/work/crop_slice_20261107")
C_DATA, C_V3D, C_BG = "#2a78d6", "#eb6834", "#1baf7a"
INK2 = "#52514e"
FS = {"2.4GHz": 61440000, "5.8GHz": 100000000}
FC = {"2.4GHz": 2430.0, "5.8GHz": 5800.0}

recs = list(csv.DictReader(open(O / "floor_per_recording.csv", encoding="utf-8")))
Z = np.load(O / "floor_curves.npz")


def shape(gid):
    f, m, c = Z[f"{gid}|f"], Z[f"{gid}|mask"], Z[f"{gid}|p10"]
    return f, np.where(m, c - np.median(c[m]), np.nan)


def med_shape(sel):
    f = None; cs = []
    for r in sel:
        f, c = shape(r["group_id"]); cs.append(c)
    return f, np.nanmedian(np.array(cs), 0), np.array(cs)


# ---------------- fig1: composition ----------------
cnt = [r for r in csv.DictReader(open(T / "counts.csv", encoding="utf-8-sig")) if r["L"] == "10"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, band in zip(axs, ["2.4GHz", "5.8GHz"]):
    cats = [("data", "drone", "data/ 无人机\n(接收机 A, 9月实验室)", C_DATA),
            ("v3", "drone", "v3 无人机\n(接收机 B, 7月城区)", C_V3D),
            ("v3", "background", "v3 背景\n(接收机 B, 城区/实验室/家)", C_BG)]
    vals = []
    for ds, lab, name, col in cats:
        s = {sp: sum(float(r["seconds"]) for r in cnt if r["band"] == band and r["dataset"] == ds and r["label"] == lab and r["split"] == sp) for sp in ["train", "val", "test"]}
        vals.append((name, col, s))
    y = np.arange(len(vals))[::-1]
    for yi, (name, col, s) in zip(y, vals):
        tot = sum(s.values())
        ax.barh(yi, tot, color=col, height=0.55)
        ax.text(tot + 1.5, yi, f"{tot:.1f} 秒  (训练 {s['train']:.1f} / 验证 {s['val']:.1f} / 测试 {s['test']:.1f})", va="center", fontsize=9, color="#0b0b0b")
    ax.set_yticks(y); ax.set_yticklabels([v[0] for v in vals], fontsize=9)
    ax.set_xlim(0, 175); ax.set_xlabel("录音总时长 (秒)，Task F 切出的全部 10 ms 片段")
    dr = sum(sum(v[2].values()) for v in vals[:2]); bg = sum(vals[2][2].values())
    ax.set_title(f"{band}：背景 : 无人机 = 1 : {dr / bg:.1f}；背景 100% 来自接收机 B", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
fig.suptitle("图1  数据构成：data/ 只有无人机，背景只来自另一台接收机（datasets_v3）", fontsize=13)
fig.tight_layout(); fig.savefig(O / "图1_数据构成.png", dpi=130); plt.close(fig)

# ---------------- fig2: spectrogram comparison 2.4G ----------------
blocks = list(csv.DictReader(open(T / "blocks.csv", encoding="utf-8-sig")))
def cells_of(gid, n=500):
    bl = [b for b in blocks if b["group_id"] == gid]
    assert bl, gid
    arrs = [np.load(T / "cells" / Path(b["block_file"]).stem / "mean.npy") for b in bl]
    bounds = np.cumsum([len(a) for a in arrs])[:-1]
    M = np.concatenate(arrs)[:n]
    return M, [b for b in bounds if b < n]
panels = [("data:001_2.4GHZ_10M_combination-signal__2026-09-20-15-49-42", "data/ 无人机 001 mini4 pro 待机\n接收机 A，9月实验室"),
          ("v3:20260709-025429_mini5pro", "v3 无人机 mini5pro\n接收机 B，07-09 城区"),
          ("v3:20260709-041641_ambient", "v3 背景（无无人机）\n接收机 B，07-09 城区"),
          ("v3:20260710-085302_ambient", "v3 背景（无无人机）\n接收机 B，07-10 实验室")]
fig, axs = plt.subplots(1, 4, figsize=(17, 6.6), sharey=True)
f = np.fft.fftshift(np.fft.fftfreq(512, 1 / FS["2.4GHz"])) / 1e6 + FC["2.4GHz"]
for ax, (gid, title) in zip(axs, panels):
    M, bounds = cells_of(gid)
    lvl = float([r for r in recs if r["group_id"] == gid][0]["floor_level_db"])
    im = ax.imshow(np.clip(M - lvl, 0, 25), aspect="auto", origin="lower", cmap="Blues",
                   extent=[f[0], f[-1], 0, len(M)], vmin=0, vmax=25, interpolation="nearest")
    for b in bounds:
        ax.axhline(b, color="#eb6834", lw=0.8, ls=":")
    for wf in (2412, 2437):
        ax.axvline(wf, color=INK2, lw=0.6, ls="--")
    ax.set_xlim(2405, 2455); ax.set_title(title, fontsize=9)
    ax.set_xlabel("频率 (MHz)")
axs[0].set_ylabel("时间 (ms)；橙色点线 = v3 文件拼接处（各段不连续）")
cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01); cb.set_label("高出本录音底噪 (dB)")
fig.suptitle("图2  2.4 GHz 长时频谱（每列 0.5 s，每行 1 ms）：城区两列在 WiFi1/6 附近始终有一层浅蓝的低电平能量；\n"
             "data/ 列没有；实验室列有很多突发信号，但不是一直存在。黑色竖虚线 = WiFi 1 (2412 MHz) / WiFi 6 (2437 MHz)", fontsize=12, y=1.04)
fig.savefig(O / "图2_2.4G时频谱对比.png", dpi=120, bbox_inches="tight"); plt.close(fig)

# ---------------- fig3: 2.4G floor ----------------
band = "2.4GHz"
fig, axs = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, sharey=True)
sel_data = [r for r in recs if r["band"] == band and r["dataset"] == "data"]
sel_clean = [r for r in recs if r["band"] == band and r["label"] == "background" and r["campaign"] in ("2026-07-10_lab", "2026-07-16_lab", "noise_adrv")]
sel_wifi_bg = [r for r in recs if r["band"] == band and r["label"] == "background" and r["campaign"] in ("2026-07-09_urban", "2026-07-11_home")]
sel_v3d = [r for r in recs if r["band"] == band and r["dataset"] == "v3" and r["label"] == "drone"]
for ax, groups, ttl in [
    (axs[0], [(sel_data, C_DATA, "-", "data/ 无人机录音 (接收机 A)"), (sel_clean, C_BG, "-", "v3 背景：实验室 07-10 / 07-16 / noise_adrv (接收机 B)")],
     "(a) 没有持续 WiFi 时：两台接收机的 2.4G 底噪大部分频段相差 <0.4 dB（绿线的几个尖峰是实验室里的持续窄信号；\n"
     "     2405–2407 下垂是 fc=2450 录音在 −45 MHz 处的滤波边缘，与图4 的 5.8G 边缘下垂同一机理）"),
    (axs[1], [(sel_v3d, C_V3D, "-", "v3 无人机录音 (全部 07-09 城区)"), (sel_wifi_bg, C_BG, "--", "v3 背景：07-09 城区 + 07-11 家")],
     "(b) 城区/家里：同样的两个鼓包 (中间有凹口) = WiFi 1、6 信道，是环境信号，不是接收机")]:
    for sel, col, ls, name in groups:
        fq, md, allc = med_shape(sel)
        lo, hi = np.nanpercentile(allc, 25, 0), np.nanpercentile(allc, 75, 0)
        ax.fill_between(fq, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(fq, md, color=col, lw=2, ls=ls, label=f"{name}  中位数，n={len(sel)} 条录音（阴影=25–75%）")
    for wf in (2412, 2437):
        ax.axvline(wf, color=INK2, lw=0.6, ls=":")
    ax.axhline(0, color=INK2, lw=0.5); ax.grid(alpha=0.25)
    ax.set_title(ttl, fontsize=11, loc="left"); ax.legend(fontsize=9, loc="upper right", frameon=False)
    ax.set_ylabel("底噪形状 (dB)")
axs[1].set_xlabel("频率 (MHz)；虚线 = WiFi 1 (2412) / 6 (2437) 信道中心")
axs[0].set_ylim(-3, 5.5)
fig.suptitle("图3  2.4 GHz 每条录音的底噪形状（每个频点取全录音 1 ms 小格功率的第 10 百分位，再减去自身中位数）", fontsize=12)
fig.tight_layout(); fig.savefig(O / "图3_2.4G底噪_WiFi是环境.png", dpi=130); plt.close(fig)

# ---------------- fig4: 5.8G floor ----------------
band = "5.8GHz"
fig, axs = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw=dict(width_ratios=[3, 1, 1]))
groups = [([r for r in recs if r["band"] == band and r["dataset"] == "data"], C_DATA, "-", "data/ 无人机 (接收机 A)"),
          ([r for r in recs if r["band"] == band and r["dataset"] == "v3" and r["label"] == "drone"], C_V3D, "-", "v3 无人机 (接收机 B)"),
          ([r for r in recs if r["band"] == band and r["label"] == "background"], C_BG, "--", "v3 背景 (接收机 B，城区/实验室/家)")]
for ax, xl in zip(axs, [(5755, 5845), (5755, 5762), (5835, 5845)]):
    for sel, col, ls, name in groups:
        fq, md, allc = med_shape(sel)
        lo, hi = np.nanpercentile(allc, 25, 0), np.nanpercentile(allc, 75, 0)
        ax.fill_between(fq, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(fq, md, color=col, lw=2, ls=ls, label=f"{name}，n={len(sel)}")
    ax.set_xlim(*xl); ax.axhline(0, color=INK2, lw=0.5); ax.grid(alpha=0.25); ax.set_xlabel("频率 (MHz)")
axs[0].set_ylim(-1.3, 3.2); axs[1].set_ylim(-1.3, 0.6); axs[2].set_ylim(-1.3, 0.6)
axs[0].set_ylabel("底噪形状 (dB)"); axs[0].legend(fontsize=9, frameon=False, loc="upper left")
axs[0].set_title("全频段：v3 无人机与 v3 背景重合，data/ 是另一种形状", fontsize=10, loc="left")
axs[1].set_title("左端放大：v3 最边 3 MHz 下垂约 0.7 dB", fontsize=10, loc="left")
axs[2].set_title("右端放大：v3 下垂约 0.6 dB，data/ 不下垂", fontsize=10, loc="left")
fig.suptitle("图4  5.8 GHz 底噪形状 = 接收机指纹：同一台接收机的有/无无人机录音完全一样，两台接收机之间不一样", fontsize=12)
fig.tight_layout(); fig.savefig(O / "图4_5.8G底噪_接收机指纹.png", dpi=130); plt.close(fig)

# ---------------- fig5: shortcut test ----------------
st = json.loads((O / "shortcut_test_result.json").read_text(encoding="utf-8"))
ctl = json.loads((O / "shortcut_controls.json").read_text(encoding="utf-8"))
feat_desc = {"5.8GHz": "只看频段两端 2 MHz 的底噪下垂量 + 右段底噪坡度", "2.4GHz": "只看 WiFi 1/6 信道底噪鼓包高度 + 左端底噪"}
fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
rng = np.random.default_rng(0)
for ax, band in zip(axs, ["2.4GHz", "5.8GHz"]):
    per = ctl[band]["per_test_group"]
    cats = [("v3 背景", lambda q: q["label"] == "background", C_BG),
            ("data/ 无人机", lambda q: q["label"] == "drone" and q["group"].startswith("data"), C_DATA),
            ("v3 无人机", lambda q: q["label"] == "drone" and q["group"].startswith("v3"), C_V3D)]
    for i, (name, fn, col) in enumerate(cats):
        ys = np.array([q["predicted_drone"] for q in per if fn(q)]) * 100
        xs = i + rng.uniform(-0.12, 0.12, len(ys))
        ax.scatter(xs, ys, s=60, color=col, edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        ax.text(i + 0.22, np.median(ys), f"{len(ys)} 条测试录音\n中位 {np.median(ys):.0f}%", va="center", fontsize=9, color="#0b0b0b")
    r = st[band]
    ax.set_xticks(range(3)); ax.set_xticklabels([c[0] for c in cats], fontsize=10)
    ax.set_xlim(-0.5, 2.9); ax.set_ylim(-5, 105); ax.grid(axis="y", alpha=0.25)
    ax.set_title(f"{band}：{feat_desc[band]}\n背景误报 {r['bg_false_alarm']*100:.1f}%  |  data/ 无人机检出 {r['recall_data_drone']*100:.1f}%  |  v3 无人机检出 {r['recall_v3_drone']*100:.1f}%",
                 fontsize=10, loc="left")
axs[0].set_ylabel("该录音的 10 ms 窗口中被判为“无人机”的比例 (%)")
fig.suptitle("图5  捷径测试：一个完全看不到无人机信号的简单分类器（逻辑回归，Task F 原划分，训练集训练、测试集评估）", fontsize=12)
fig.tight_layout(); fig.savefig(O / "图5_捷径测试.png", dpi=130); plt.close(fig)
print("figures ok")
