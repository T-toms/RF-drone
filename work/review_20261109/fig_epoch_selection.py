# 画当前 2.4G 基线（任务 M）三次训练每一轮的验证集误报与漏检，标出被选中的那一轮（2026-11-11，支线）
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
D = Path(__file__).resolve().parents[1] / 'side_M_notchfill_20261110'
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for ax, s in zip(axs, [2026, 2027, 2028]):
    r = list(csv.DictReader(open(D / f'train_log_Lb_seed{s}.csv', encoding='utf-8-sig')))
    e = [int(x['epoch']) for x in r]
    fa = [100 * float(x['val_fa_v3']) for x in r]
    ms = [100 * float(x['val_miss']) for x in r]
    sc = [float(x['val_fa_v3']) + float(x['val_fa_bgA']) + float(x['val_miss']) for x in r]
    vl = [float(x['val_loss']) for x in r]
    best = min(zip(sc, vl, e))[2]
    ax.plot(e, fa, 'o-', color='#c0392b', ms=4, label='城区背景误报（验证集 80 段，1 段 = 1.25%）')
    ax.plot(e, ms, 's-', color='#2471a3', ms=4, label='无人机漏检（验证集约 1000 段）')
    ax.axvline(best, color='green', lw=2, alpha=.6)
    ax.text(best + .3, 8, f'被选中：第 {best} 轮', color='green', fontsize=11)
    ax.set_ylim(-1, 15); ax.set_ylabel('%')
    ax.set_title(f'第 {[2026, 2027, 2028].index(s) + 1} 次训练（种子 {s}）')
    for x, y in zip(e, fa):
        if y > 15:
            ax.annotate(f'{y:.0f}%', (x, 14.5), ha='center', color='#c0392b', fontsize=9)
    ax.legend(fontsize=9, loc='upper right')
axs[-1].set_xlabel('训练轮数')
fig.suptitle('2.4G 当前基线：每一轮训练后在验证集上的表现，绿线 = 规则挑中的那一轮\n（规则：误报 + 漏检之和最小；超出 15% 的点在顶部标出数值）', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .95])
out = Path(__file__).resolve().parent / '图18_选模型规则_每轮验证集表现_2.4GHz.png'
fig.savefig(out, dpi=110); print(out)
