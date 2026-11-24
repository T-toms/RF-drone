"""Coordinator figures for the B2b summary report (reads worker CSVs, writes PNGs into a new report dir)."""
import csv, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
SRC = Path('C:/Users/Administrator/Desktop/数据集处理/work/fusionB_B2_20261112')
OUT = Path('C:/Users/Administrator/Desktop/数据集处理/work/汇报_一阶段时频图加IQ特征_20261117')
OUT.mkdir(exist_ok=True)
INK, INK2, SURF, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#e4e3df'
BLUE, ORANGE = '#2a78d6', '#eb6834'

# ---------- 1. architecture diagram ----------
fig, ax = plt.subplots(figsize=(14, 6.4), facecolor=SURF)
ax.set_xlim(0, 14); ax.set_ylim(0, 6.4); ax.axis('off')
def box(x, y, w, h, text, fc='#ffffff', ec=INK2, fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, weight='bold' if bold else 'normal')
def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color=INK2, lw=1.4))
# input
box(0.2, 2.6, 1.7, 1.2, '一片 20 ms\nIQ 数据', fc='#f0efec', bold=True)
# image branch (top)
box(2.5, 4.3, 2.0, 1.2, '时频图\n短时傅里叶变换\n256×256', fc='#e8f1fc')
box(5.0, 4.3, 2.3, 1.2, 'ResNet18 卷积部分\n(与基线 R 相同)', fc='#e8f1fc')
box(7.8, 4.3, 1.9, 1.2, '全局平均池化\n→ 512 维\n→ LayerNorm', fc='#e8f1fc')
# feature branch (bottom)
box(2.5, 0.9, 2.0, 1.2, '10 个 IQ 手工特征\n(F5 只用 5 个)', fc='#fdeee7')
box(5.0, 0.9, 2.3, 1.2, '缺失值填补\nz-score 标准化\n截断到 ±10', fc='#fdeee7')
box(7.8, 0.9, 1.9, 1.2, '全连接 10→64\nReLU\n→ LayerNorm', fc='#fdeee7')
# fusion
box(10.2, 2.6, 1.6, 1.2, '拼接\n512+64=576', fc='#f0efec', bold=True)
box(12.2, 2.6, 1.6, 1.2, 'Dropout 0.3\n全连接 576→2\n有 / 无无人机', fc='#f0efec')
arrow(1.9, 3.4, 2.5, 4.9); arrow(1.9, 3.0, 2.5, 1.5)
arrow(4.5, 4.9, 5.0, 4.9); arrow(7.3, 4.9, 7.8, 4.9); arrow(9.7, 4.9, 10.2, 3.5)
arrow(4.5, 1.5, 5.0, 1.5); arrow(7.3, 1.5, 7.8, 1.5); arrow(9.7, 1.5, 10.2, 2.9)
arrow(11.8, 3.2, 12.2, 3.2)
ax.text(2.5, 5.75, '时频图支路（看“图像”）', fontsize=11, color=BLUE, weight='bold')
ax.text(2.5, 0.45, '手工特征支路（看“统计量”）', fontsize=11, color=ORANGE, weight='bold')
ax.text(10.2, 2.25, '两路一起端到端训练，\n用同一个交叉熵损失', fontsize=9.5, color=INK2, va='top')
ax.text(7.0, 6.25, '图 A　融合网络 F 的结构（中间层拼接）', ha='center', fontsize=13, color=INK, weight='bold')
fig.savefig(OUT / '图A_融合网络结构.png', dpi=150, bbox_inches='tight', facecolor=SURF); plt.close(fig)

# ---------- 2. paired differences vs Z, per seed ----------
rows = list(csv.DictReader(open(SRC / 'metrics_all.csv', encoding='utf-8')))
def rate(b, s, sub, g, seed):
    for x in rows:
        if (x['band'], x['set'], x['subset'], x['group'], x['seed'], x['mode'], x['dependence']) == (b, s, sub, g, seed, 'argmax', 'original') and x['category'] in ('main', 'receiver'):
            return float(x['rate']) * 100
items = [('2.4GHz', 'test', 'v3_test_FA', '2.4G v3 背景虚警\n(越低越好)'),
         ('2.4GHz', 'test', 'B_nonG_recall', '2.4G v3 无人机召回\n(越高越好)'),
         ('2.4GHz', 'test', 'nonG_miss', '2.4G 非G 漏检\n(越低越好)'),
         ('2.4GHz', 'C1', 'C1_FA', '2.4G C1 虚警\n(越低越好)'),
         ('5.8GHz', 'test', 'v3_test_FA', '5.8G v3 背景虚警\n(越低越好)'),
         ('5.8GHz', 'test', 'B_nonG_recall', '5.8G v3 无人机召回\n(非G，越高越好)'),
         ('5.8GHz', 'test', 'nonG_miss', '5.8G 非G 漏检\n(越低越好)')]
seeds = ['2026', '2027', '2028']
fig, axes = plt.subplots(1, len(items), figsize=(16, 4.6), facecolor=SURF, sharey=False)
for ax, (b, s, sub, title) in zip(axes, items):
    for j, (g, c, mk) in enumerate([('F', BLUE, 'o'), ('F5', ORANGE, 's')]):
        d = [rate(b, s, sub, g, sd) - rate(b, s, sub, 'Z', sd) for sd in seeds]
        xs = [j + (k - 1) * 0.12 for k in range(3)]
        ax.scatter(xs, d, s=46, color=c, marker=mk, edgecolor=SURF, linewidth=1.5, zorder=3, label=f'{g} − Z（每点一个种子）')
        ax.hlines(st.mean(d), j - 0.25, j + 0.25, color=c, lw=2, zorder=2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['F−Z', 'F5−Z'], color=INK)
    ax.set_xlim(-0.6, 1.6); ax.set_title(title, fontsize=10, color=INK)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_facecolor(SURF)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)
axes[0].set_ylabel('与 Z 的差（百分点）', color=INK)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.07), fontsize=10)
fig.suptitle('图 B　加特征（F、F5）相对“特征置 0”（Z）的配对差：三个种子各一个点，横线为均值', y=1.14, fontsize=12.5, color=INK, weight='bold')
fig.tight_layout()
fig.savefig(OUT / '图B_加特征的配对差_逐种子.png', dpi=150, bbox_inches='tight', facecolor=SURF); plt.close(fig)
print('ok', OUT)
