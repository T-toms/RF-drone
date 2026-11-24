# 画 2.4G 城区接收机两次录音的全频段时频图（看信号是不是真在 2455 以上）：mavic2 测试录音、evo2pro_v3 训练录音（2026-11-11，支线）
import json
from pathlib import Path
import numpy as np
from scipy import io, signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
R = Path('G:/drone')
P = Path(__file__).resolve().parents[1]
plan = json.loads((P / 'crop_slice_20261107/plan.json').read_text(encoding='utf-8'))
want = [('mavic2 测试录音', '040617_mavic2'), ('mavic2 训练录音', '040756_mavic2'), ('EVO II Pro V3 训练录音', '033602_evo2pro_v3'), ('城区背景 07-09', '041641_ambient')]
fig, axs = plt.subplots(len(want), 1, figsize=(14, 3.4 * len(want)))
for ax, (t, k) in zip(axs, want):
    b = next(x for x in plan if x['dataset'] == 'v3' and x['band'] == '2.4GHz' and k in x['group_id'])
    p = b['parts'][0]
    x = io.loadmat(R / p['path'], variable_names=['IQ'])['IQ'].ravel()[p['start']:p['end']][:5_000_000]
    fr, tt, S = signal.spectrogram(x, fs=b['fs_in'], window='hann', nperseg=1024, noverlap=0, return_onesided=False, mode='psd')
    S = np.fft.fftshift(S, axes=0); fr = np.fft.fftshift(fr) / 1e6 + b['fc_in']
    db = 10 * np.log10(S + 1e-30); v = np.percentile(db, 99.5)
    ax.pcolormesh(tt * 1e3, fr, db, vmin=v - 35, vmax=v, shading='auto', cmap='viridis')
    for y in (2405, 2455):
        ax.axhline(y, color='red', ls='--', lw=1)
    ax.axhline(2483.5, color='white', ls=':', lw=1)
    ax.set(title=f'{t}（{b["group_id"][3:]}，中心 {b["fc_in"]:.0f} MHz）', ylabel='频率（MHz）', xlabel='时间（毫秒）')
fig.suptitle('2.4G 城区接收机原始录音全频段（红虚线 = 截取的 2405–2455 MHz；白点线 = ISM 上限 2483.5 MHz）', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .97])
out = Path(__file__).resolve().parent / '图19e_城区录音全频段_2.4GHz.png'
fig.savefig(out, dpi=85); print(out)
