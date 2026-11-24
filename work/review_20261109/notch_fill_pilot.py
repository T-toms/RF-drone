# 陷波凹坑修补试算（2026-11-10，支线）
# 做法：在 STFT 原始频点上（60 kHz/点，池化前），把三处陷波的受影响频点逐帧换掉，再按原流程池化、取 dB、逐片标准化。
#   受影响频点 = 纯噪声平均谱里比两侧底噪低 0.3 dB 以上的点（notch_hole_profile.py 量得：2426.88 四点、2440 六点、2450 六点）。
#   方法 M（镜像复制）：坑左半边的点逐帧复制坑左侧外面对称位置的点，右半边同理。保留噪声纹理与逐帧电平，不引入随机数。
#   方法 I（线性插值）：逐帧在坑两侧最近的正常点之间按功率线性插值（对照用）。
# 只改时频图计算，不动 IQ；叠加、加噪都在 IQ 上做，之后同样走这个函数，所以对所有输入一致。
# 检验：纯噪声（bgA、陷波白噪声）上，修补后陷波所在行的均值/标准差是否与邻行一致；有信号的片段上画修补前后对比图。
import csv, sys
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'work/train_stage1_L20_20261108'))
from common import spec as spec_old, image, plt  # 只读
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
FS, FC = 61440000, 2430.
freq = np.fft.fftshift(np.fft.fftfreq(1024, 1 / FS)) / 1e6 + FC
keep = (freq >= FC - 25) & (freq <= FC + 25)
fk = freq[keep]
# 受影响频点（量得的结果，见 notch_hole_profile.py 输出）
HOLES = {2426.88: (2426.76, 2427.00), 2440.0: (2439.84, 2440.20), 2450.0: (2449.80, 2450.16)}
hole_idx = [np.where((fk >= lo - 1e-6) & (fk <= hi + 1e-6))[0] for lo, hi in HOLES.values()]


def fill_bins(p, method):
    p = p.clone()
    for idx in hole_idx:
        a, b = int(idx[0]), int(idx[-1])          # 坑内首末频点
        n = b - a + 1
        if method == 'M':
            half = n // 2
            for k in range(half):                 # 左半：a+k <- a-1-k
                p[a + k] = p[a - 1 - k]
            for k in range(n - half):             # 右半：b-k <- b+1+k
                p[b - k] = p[b + 1 + k]
        else:
            L, R = p[a - 1], p[b + 1]
            for k in range(n):
                w = (k + 1) / (n + 1)
                p[a + k] = (1 - w) * L + w * R
    return p


def spec_new(iq, method='M'):
    x = torch.as_tensor(iq, device='cuda', dtype=torch.complex64)
    s = torch.stft(x, 1024, 512, window=torch.hann_window(1024, device='cuda'), center=False, onesided=False, return_complex=True)
    p = torch.fft.fftshift(s.abs().square(), dim=-2)
    p = p[torch.as_tensor(keep, device='cuda')]
    p = fill_bins(p, method)
    p = F.adaptive_avg_pool2d(p[None, None], (256, 256))[0, 0]
    db = 10 * torch.log10(p + p.median().clamp_min(1e-30) * 1e-3)
    return ((db - db.mean()) / db.std().clamp_min(1e-8)).cpu().numpy().astype(np.float16)


edges = np.floor(np.arange(256) * len(fk) / 256).astype(int), np.ceil((np.arange(256) + 1) * len(fk) / 256).astype(int)
rowf = np.array([fk[a:b].mean() for a, b in zip(*edges)])
rows = lambda p: list(csv.DictReader(open(p, encoding='utf-8-sig')))


def seg(r):
    return np.array(np.load(r['block_file'], mmap_mode='r')[int(r['start']):int(r['start']) + int(r['n_samples'])])


def row_stats(imgs, n):
    """陷波所在行（±0.25 MHz）与两侧 0.5–1.5 MHz 行比较：行均值差、行内标准差比。"""
    X = np.asarray(imgs, dtype=np.float32)
    rm, rs = X.mean(2), X.std(2)                   # N, 256
    near = np.abs(rowf - n) <= 0.25
    ref = (np.abs(rowf - n) >= 0.5) & (np.abs(rowf - n) <= 1.5)
    dmean = (rm[:, near].min(1) - rm[:, ref].mean(1))
    rstd = rs[:, near].min(1) / rs[:, ref].mean(1)
    return float(np.median(dmean)), float(np.median(rstd))


if __name__ == '__main__':
    bg = [r for r in rows('E:/drone_lab_spec_bgA_L20_20261109/bgA_meta.csv') if r['keep'] == 'True' and r['split'] == 'test']
    rng = np.random.default_rng(20261110)
    bgs = [seg(bg[i]) for i in sorted(rng.choice(len(bg), 40, replace=False))]
    nn = np.load(OUT / 'notched_noise_2.4GHz.npy', mmap_mode='r')
    nns = [np.array(nn[i]) for i in range(40)]
    wn = [(rng.standard_normal(1228800) + 1j * rng.standard_normal(1228800)).astype(np.complex64) for _ in range(20)]
    print('行统计：陷波行均值减邻行均值（标准化单位，负=凹坑）；陷波行行内标准差/邻行（<1=更平滑）')
    for name, data in [('bgA 纯噪声', bgs), ('陷波白噪声', nns), ('未陷波白噪声(参照)', wn)]:
        for tag, fn in [('原流程', spec_old.__call__ if False else (lambda z: spec_old(z, '2.4GHz'))), ('镜像M', lambda z: spec_new(z, 'M')), ('插值I', lambda z: spec_new(z, 'I'))]:
            imgs = [fn(z) for z in data]
            st = [row_stats(imgs, n) for n in HOLES]
            print(f'  {name:14s} {tag:5s} ' + '  '.join(f'{n}: 均值差{d:+.2f} 标准差比{s:.2f}' for n, (d, s) in zip(HOLES, st)))
    # 对比图：bgA、陷波白噪声、data 无人机（avata2 信号跨 2440）、v3 城区背景
    H = 'E:/drone_lab_spec_L20_20261108/'
    tr = rows(H + '2.4GHz_test_spec_meta.csv')
    pick = [('bgA 纯噪声', bgs[0]), ('陷波白噪声', nns[0]),
            ('data avata2', seg(next(r for r in tr if r['model'] == 'avata2'))),
            ('v3 城区背景', seg(next(r for r in tr if r['label'] == 'background')))]
    fig, axs = plt.subplots(len(pick), 2, figsize=(13, 4.2 * len(pick)))
    for ax, (t, z) in zip(axs, pick):
        image(ax[0], spec_old(z, '2.4GHz').astype(np.float32), '2.4GHz', f'{t}：原流程（白色虚线=陷波位置）')
        image(ax[1], spec_new(z, 'M').astype(np.float32), '2.4GHz', f'{t}：修补后（镜像复制）')
    fig.tight_layout(); fig.savefig(OUT / 'notch_fill_pilot.png', dpi=110); plt.close(fig)
    # 放大图：2435–2455 MHz，便于看清三条线
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (t, z) in zip(axs, pick[:2]):
        for a, img, tt in [(ax[0], spec_old(z, '2.4GHz'), '原流程'), (ax[1], spec_new(z, 'M'), '修补后')]:
            image(a, img.astype(np.float32), '2.4GHz', f'{t}：{tt}（放大 2420–2455 MHz）'); a.set_ylim(2420, 2455)
    fig.tight_layout(); fig.savefig(OUT / 'notch_fill_pilot_zoom.png', dpi=110); plt.close(fig)
    print(OUT / 'notch_fill_pilot.png')
