import sys
sys.dont_write_bytecode=True
import csv
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
SAMPLE_FS, CENTER = 61440000, 2430.
freq = np.fft.fftshift(np.fft.fftfreq(1024, 1 / SAMPLE_FS)) / 1e6 + CENTER
keep = (freq >= CENTER - 25) & (freq <= CENTER + 25)
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


def spec_new(iq, method='M', dtype=np.float16):
    x = torch.as_tensor(iq, device='cuda', dtype=torch.complex64)
    s = torch.stft(x, 1024, 512, window=torch.hann_window(1024, device='cuda'), center=False, onesided=False, return_complex=True)
    p = torch.fft.fftshift(s.abs().square(), dim=-2)
    p = p[torch.as_tensor(keep, device='cuda')]
    p = fill_bins(p, method) if hole_idx else p
    p = F.adaptive_avg_pool2d(p[None, None], (256, 256))[0, 0]
    db = 10 * torch.log10(p + p.median().clamp_min(1e-30) * 1e-3)
    return ((db - db.mean()) / db.std().clamp_min(1e-8)).cpu().numpy().astype(dtype)


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


spec_fill=spec_new
