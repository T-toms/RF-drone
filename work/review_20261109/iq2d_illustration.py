# What does a 2D-reshaped IQ input look like to a 2D CNN, compared with the spectrogram input?
# One 2.4G L=20 ms data/ drone test sample (mini5 pro) and one v3 urban background test sample.
# Rows of the reshaped matrix = 1024 consecutive samples (16.7 us at 61.44 MS/s).
import sys, csv, numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'C:/Users/Administrator/Desktop/数据集处理/work/train_stage1_L20_20261108')
from common import spec
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
FS, W = 61.44e6, 1024
rows = list(csv.DictReader(open('C:/Users/Administrator/Desktop/数据集处理/work/crop_slice_20261107/index_L20.csv', encoding='utf-8-sig')))
pick = {
    '实验室无人机 data/ mini5 pro': next(r for r in rows if r['band'] == '2.4GHz' and r['dataset'] == 'data' and r['model'] == 'mini5 pro' and r['split'] == 'test' and r['keep'] == 'True'),
    '城区背景 v3 07-09（有WiFi）': next(r for r in rows if r['band'] == '2.4GHz' and r['label'] == 'background' and r['group_id'] == 'v3:20260709-041641_ambient'),
}
fig, axs = plt.subplots(2, 5, figsize=(26, 10))
for ax, (name, r) in zip(axs, pick.items()):
    z = np.array(np.load(r['block_file'], mmap_mode='r')[int(r['start']):int(r['start']) + int(r['n_samples'])])
    H = len(z) // W; M = z[:H * W].reshape(H, W)
    s = spec(z, '2.4GHz').astype(np.float32)
    ax[0].imshow(s, origin='lower', aspect='auto', extent=[0, 20, 2405, 2455], cmap='viridis', vmin=-2, vmax=3)
    ax[0].set(title=f'{name}\n(a) 现在的输入：时频图 256×256', xlabel='时间 (ms)', ylabel='频率 (MHz)')
    lim = np.percentile(np.abs(M.real), 99.5)
    ax[1].imshow(M.real, origin='upper', aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim, extent=[0, W, H, 0])
    ax[1].set(title=f'(b) IQ 二维排列：I 通道 {H}×{W}\n每行 = 1024 个连续采样 (16.7 µs)', xlabel='行内采样序号', ylabel='行号（往下 = 时间往后）')
    ax[2].imshow(M.real[:24, :24], origin='upper', cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
    ax[2].set(title='(c) 左上角 24×24 放大：\n第一层 7×7 卷积核看到的就是这种小块', xlabel='行内 1 格 = 16 ns', ylabel='上下 1 格 = 16.7 µs')
    env = 20 * np.log10(np.abs(M) + 1e-12)
    ax[3].imshow(env, origin='upper', aspect='auto', cmap='magma', extent=[0, W, H, 0], vmin=np.percentile(env, 5), vmax=np.percentile(env, 99.5))
    ax[3].set(title='(d) 幅度 |I+jQ|（dB）同样排列：\n只看得到"什么时候有能量"，看不到频率', xlabel='行内采样序号', ylabel='行号')
    F = np.fft.fftshift(np.fft.fft(M, axis=1), axes=1); P = 20 * np.log10(np.abs(F) + 1e-9)
    f = 2430 + np.fft.fftshift(np.fft.fftfreq(W, 1 / FS)) / 1e6
    ax[4].imshow(P.T, origin='lower', aspect='auto', cmap='viridis', extent=[0, 20, f[0], f[-1]], vmin=np.percentile(P, 30), vmax=np.percentile(P, 99.8))
    ax[4].set_ylim(2405, 2455)
    ax[4].set(title='(e) 对 (b) 的每一行做 FFT 后：\n频率结构才出现（= 时频图）', xlabel='时间 (ms)', ylabel='频率 (MHz)')
    def c(a, b): return abs(np.vdot(a, b)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
    flat = z[:H * W]
    print(f'{name}: |corr| 左右相邻(1个采样) = {c(flat[:-1], flat[1:]):.3f}; 上下相邻(1024个采样) = {c(flat[:-W], flat[W:]):.3f}; 相隔7个采样 = {c(flat[:-7], flat[7:]):.3f}')
fig.suptitle('图14  IQ 二维排列送进二维卷积网络时，网络实际"看到"的是什么（2.4 GHz，20 ms 片段）', fontsize=16)
fig.tight_layout(); fig.savefig('C:/Users/Administrator/Desktop/数据集处理/work/review_20261109/图14_IQ二维排列示意.png', dpi=110)
print('saved')
