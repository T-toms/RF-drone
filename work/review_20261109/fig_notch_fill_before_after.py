# 给用户/老师看的陷波修补前后时频图（2026-11-10，支线）
# 每行一个样本：全幅修补前 | 全幅修补后 | 放大 2436–2454 MHz 修补前 | 放大修补后 | 按时间平均的逐行亮度（修补前后两条线）
# 修补前 = common.spec（任务 H/J/L 的原流程）；修补后 = 任务 M 交付的 spec_fill（从 side_M 目录读取函数，只读）。
# 不在图上画陷波位置的白线（会盖住凹坑），改为在右侧用红色小三角标出。
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'work/train_stage1_L20_20261108'))
sys.path.insert(0, str(ROOT / 'work/side_M_notchfill_20261110'))
from common import spec as spec_old, rows, plt   # 只读：不调用 model()，不写 CACHE
from fill import spec_fill, rowf, seg              # 任务 M 交付的修补函数
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
NOTCH = [2426.88, 2440.0, 2450.0]
H = Path('E:/drone_lab_spec_L20_20261108')
te = rows(H / '2.4GHz_test_spec_meta.csv'); tr = rows(H / '2.4GHz_train_spec_meta.csv')
bg = [r for r in rows('E:/drone_lab_spec_bgA_L20_20261109/bgA_meta.csv') if r['keep'] == 'True' and r['split'] == 'test']
first = lambda rs, **kw: next(r for r in rs if all(r[k] == v for k, v in kw.items()))

ex = []
ex.append(('实验室接收机的纯背景（只有噪声）', seg(bg[100])))
ex.append(('实验室无人机 avata2（信号跨过 2440、2450）', seg(first(te, dataset='data', model='avata2'))))
ex.append(('实验室无人机 mini4 pro（信号在 2450 附近）', seg(first(te, dataset='data', model='mini4 pro'))))
ex.append(('城区背景（有无线网）', seg(first(te, dataset='v3', label='background'))))
ex.append(('城区无人机 mini3pro', seg(first(te, dataset='v3', model='mini3pro'))))
# 叠加训练样本：实验室无人机 + 城区背景，取功率比最接近 0 dB 的一条配对
P = rows(ROOT / 'work/side_L_wifimix_20261110/mix_train_drone.pairs.csv')
p = min(P, key=lambda r: abs(float(r['X'])))
d = seg(first(tr, sample_id=p['sample_id'])); b = seg(first(tr, sample_id=p['background_sample_id']))
ex.append((f"叠加样本：实验室无人机 + 城区背景（{float(p['X']):+.1f} dB）", b + float(p['g']) * d))


def show(ax, z, title, zoom=False):
    ax.imshow(z, origin='lower', aspect='auto', extent=[0, 20, 2405, 2455], cmap='viridis', vmin=-2, vmax=3, interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('时间（毫秒）', fontsize=8); ax.set_ylabel('频率（MHz）', fontsize=8)
    if zoom:
        ax.set_ylim(2436, 2454)
    for n in NOTCH:
        if not zoom or 2436 <= n <= 2454:
            ax.annotate('', xy=(20, n), xytext=(21.2, n), xycoords='data', annotation_clip=False,
                        arrowprops=dict(arrowstyle='-|>', color='red', lw=1.2))


fig, axs = plt.subplots(len(ex), 5, figsize=(26, 4.3 * len(ex)), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 0.7]))
for row, (t, z) in zip(axs, ex):
    a = spec_old(z, '2.4GHz').astype(np.float32); b2 = spec_fill(z).astype(np.float32)
    show(row[0], a, f'{t}\n修补前'); show(row[1], b2, '修补后')
    show(row[2], a, '修补前（放大 2436–2454 MHz）', True); show(row[3], b2, '修补后（放大）', True)
    ax = row[4]
    ax.plot(a.mean(1), rowf, color='#c0392b', lw=1, label='修补前')
    ax.plot(b2.mean(1), rowf, color='#2471a3', lw=1, label='修补后')
    for n in NOTCH:
        ax.axhline(n, color='gray', ls=':', lw=.8)
    ax.set_ylim(2405, 2455); ax.set_xlabel('该行按时间平均的亮度（标准单位）', fontsize=8); ax.set_title('逐行平均亮度', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
fig.suptitle('2.4 GHz 陷波凹坑修补前后（红色三角 = 三处陷波：2426.88 / 2440 / 2450 MHz；颜色范围统一 −2 到 3）', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = Path(__file__).resolve().parent / '图15b_陷波修补前后_时频图对比_2.4GHz.png'
fig.savefig(out, dpi=90); print(out)
