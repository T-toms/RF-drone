# Flip 测试录音为什么没有图传大块：看原始录音全频段（截频前，fc 2400、122.88 MS/s，覆盖 2338.6–2461.4 MHz）的时频图（2026-11-11，支线）
# 只读 G 盘 data/ 原始 .mat；每个录音取一个 50 ms 文件。
import sys
from pathlib import Path
import numpy as np
from scipy import io, signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
R = Path('G:/drone/data')
recs = [('air 3s 训练：待机 10M combination', '006_2.4GHZ_10M_combination-signal', '2026-09-20-16-44-27'),
        ('air 3s 训练：飞行 40M', 'F006_2.4GHz_40M-signal', '2026-09-20-20-25-48'),
        ('air 3s 测试：飞行 10M', 'F006_2.4GHz_10M-signal', '2026-09-20-20-24-56'),
        ('atom se 训练：待机 10M', '013_2.4GHz_10M-signal', '2026-09-20-19-02-44'),
        ('atom se 测试：飞行 10M', 'F013_2.4GHz_10M-signal', '2026-09-20-19-03-17')]
fs, fc = 122.88e6, 2400.
fig, axs = plt.subplots(len(recs), 1, figsize=(14, 3.6 * len(recs)))
for ax, (t, d, stamp) in zip(axs, recs):
    f = sorted(p for p in R.rglob(f'{d}/*{stamp}*epoch-0_seg05.mat'))
    if not f:
        f = sorted(p for p in R.rglob(f'{d}/*{stamp}*.mat'))
    x = io.loadmat(f[0], variable_names=['data'])['data'].ravel().astype(np.complex64)
    fr, tt, S = signal.spectrogram(x, fs=fs, window='hann', nperseg=1024, noverlap=0, return_onesided=False, mode='psd')
    S = np.fft.fftshift(S, axes=0); fr = np.fft.fftshift(fr) / 1e6 + fc
    db = 10 * np.log10(S + 1e-30); v = np.percentile(db, 99.5)
    ax.pcolormesh(tt * 1e3, fr, db, vmin=v - 40, vmax=v, shading='auto', cmap='viridis')
    for y in (2405, 2455):
        ax.axhline(y, color='red', ls='--', lw=1)
    ax.set(title=f'{t}   {f[0].name}', ylabel='频率（MHz）', xlabel='时间（毫秒）')
    p = db.mean(1)
    lo, hi = (fr >= 2405) & (fr <= 2455), (fr > 2455)
    print(f'{t}: 截取带内平均 {p[lo].mean():.1f} dB，2455 MHz 以上平均 {p[hi].mean():.1f} dB，最强 1 MHz 段中心 {fr[np.argmax(np.convolve(p, np.ones(8) / 8, "same"))]:.1f} MHz')
fig.suptitle('air 3s 与 atom se 2.4G 原始录音全频段（红虚线 = 我们截取的 2405–2455 MHz）', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .97])
out = Path(__file__).resolve().parent / '图19d_air3s与atomse原始录音全频段_2.4GHz.png'
fig.savefig(out, dpi=85); print(out)

