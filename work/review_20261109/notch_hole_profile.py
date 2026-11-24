# 量陷波凹坑的实际宽度和深度（2026-11-10，支线）
# 在 STFT 原始频点（60 kHz/点，未池化）上，对 bgA（纯噪声）和 data 无人机片段求时间平均功率谱，
# 看 2426.88 / 2440 / 2450 三处凹坑相对两侧 0.5–1.5 MHz 底噪低多少、跨几个频点，
# 以及池化成 256 行后影响几行。只读缓存与 IQ，不写任何共用目录。
import csv, sys
from pathlib import Path
import numpy as np, torch
FS, FC = 61440000, 2430.
NOTCH = [2426.88, 2440.0, 2450.0]
rows = lambda p: list(csv.DictReader(open(p, encoding='utf-8-sig')))


def psd_bins(iq):
    x = torch.as_tensor(iq, device='cuda', dtype=torch.complex64)
    s = torch.stft(x, 1024, 512, window=torch.hann_window(1024, device='cuda'), center=False, onesided=False, return_complex=True)
    p = torch.fft.fftshift(s.abs().square(), dim=-2)
    return p.mean(1).cpu().numpy(), p.cpu().numpy()


freq = np.fft.fftshift(np.fft.fftfreq(1024, 1 / FS)) / 1e6 + FC
keep = (freq >= FC - 25) & (freq <= FC + 25)
fk = freq[keep]
edges = np.floor(np.arange(256) * len(fk) / 256).astype(int), np.ceil((np.arange(256) + 1) * len(fk) / 256).astype(int)
rowf = np.array([fk[a:b].mean() for a, b in zip(*edges)])

bg = [r for r in rows('E:/drone_lab_spec_bgA_L20_20261109/bgA_meta.csv') if r['keep'] == 'True']
rng = np.random.default_rng(0)
pick = [bg[i] for i in sorted(rng.choice(len(bg), 60, replace=False))]
acc = 0
for r in pick:
    z = np.load(r['block_file'], mmap_mode='r')[int(r['start']):int(r['start']) + int(r['n_samples'])]
    acc = acc + psd_bins(np.array(z))[0]
mean_db = 10 * np.log10(acc / len(pick))[keep]
print('bins kept', keep.sum(), 'bin width kHz', FS / 1024 / 1e3, 'pooled row width kHz', 50e3 / 256)
for n in NOTCH:
    ref = (np.abs(fk - n) >= 0.5) & (np.abs(fk - n) <= 1.5)
    base = np.median(mean_db[ref])
    near = np.abs(fk - n) <= 0.6
    d = mean_db[near] - base
    print(f'notch {n}: depth min {d.min():+.1f} dB; bins <-1 dB: {int((d < -1).sum())} '
          f'({fk[near][d < -1].min():.3f}–{fk[near][d < -1].max():.3f} MHz); bins <-0.3 dB: {int((d < -0.3).sum())}')
    print('   per-bin dB rel. floor:', ' '.join(f'{v:+.1f}' for v in d))
    # pooled rows: average power per row then compare
    pw = 10 ** (mean_db / 10)
    rp = np.array([10 * np.log10(pw[a:b].mean()) for a, b in zip(*edges)])
    rref = (np.abs(rowf - n) >= 0.5) & (np.abs(rowf - n) <= 1.5)
    rd = rp - np.median(rp[rref])
    sel = np.abs(rowf - n) <= 0.6
    print('   pooled rows dB rel. floor:', ' '.join(f'{f:.2f}:{v:+.1f}' for f, v in zip(rowf[sel], rd[sel])))
