import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parent
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
rows = list(csv.DictReader((D / 'ctrl_whitenoise_results.csv').open(encoding='utf-8')))
XS = [-10, -5, 0, 5, 10, 15, 20]
BLUE, ORANGE, INK2 = '#2a78d6', '#eb6834', '#52514e'

panels = [
    ('2.4GHz', 'data_drone', 'True', '2.4 GHz · 实验室(data/)无人机，训练见过的型号'),
    ('2.4GHz', 'v3_drone', 'True', '2.4 GHz · 城区(v3)无人机'),
    ('5.8GHz', 'data_drone', 'True', '5.8 GHz · 实验室(data/)无人机'),
    ('5.8GHz', 'v3_drone', 'True', '5.8 GHz · 城区(v3)无人机，训练见过的型号'),
]


def vals(b, g, sv, bt, x):
    v = [float(r['det']) * 100 for r in rows if r['band'] == b and r['group'] == g and r['seen'] == sv and r['bgtype'] == bt and r['x_db'] == str(x)]
    return np.mean(v), np.std(v, ddof=1), [round(z) for z in v]


fig, axs = plt.subplots(2, 2, figsize=(12, 8.6), sharey=True)
fig.patch.set_facecolor('#fcfcfb')
for ax, (b, g, sv, title) in zip(axs.flat, panels):
    ax.set_facecolor('#fcfcfb')
    n = [r['n'] for r in rows if r['band'] == b and r['group'] == g and r['seen'] == sv][0]
    cm, cs, _ = vals(b, g, sv, 'clean', 'inf')
    ax.axhline(cm, color=INK2, ls='--', lw=1.2)
    ax.text(20.4, cm, f'原样本 {cm:.0f}%', va='center', ha='left', fontsize=9, color=INK2)
    for bt, col, lab in [('wn', BLUE, '叠加到白噪声'), ('v3bg', ORANGE, '叠加到城区背景(v3)')]:
        m = [vals(b, g, sv, bt, x) for x in XS]
        mu = np.array([z[0] for z in m]); sd = np.array([z[1] for z in m])
        ax.fill_between(XS, mu - sd, mu + sd, color=col, alpha=.15, lw=0)
        ax.plot(XS, mu, color=col, lw=2, marker='o', ms=6, label=lab)
    ax.set_title(f'{title}（N={n}）', fontsize=10.5, loc='left')
    ax.set_xticks(XS); ax.set_ylim(0, 105); ax.grid(axis='y', color='#e6e5e0', lw=.8)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    ax.set_xlabel('无人机样本功率 / 背景功率 (dB)', fontsize=9.5, color=INK2)
    if ax in axs[:, 0]: ax.set_ylabel('判为“有无人机”的比例 (%)', fontsize=9.5, color=INK2)
axs[0, 0].legend(loc='lower right', frameon=False, fontsize=9.5)
fig.suptitle('同一批无人机样本、同样的功率比，只换“垫底”的背景：白噪声 vs 城区背景（3 个种子均值，阴影=±1 标准差）', fontsize=11.5, x=.02, ha='left')
fig.text(.02, .005, '纯白噪声单独送入：2.4 GHz 判为无人机 = 种子2026 100% / 2027 0% / 2028 0%；5.8 GHz 三个种子均为 0%（N=80/60）。', fontsize=9, color=INK2)
fig.tight_layout(rect=(0, .03, 1, .95))
fig.savefig(D / '图10_背景替换对照.png', dpi=150, facecolor=fig.get_facecolor())
print('ok')
