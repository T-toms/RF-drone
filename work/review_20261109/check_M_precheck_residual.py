# 任务 M 预检 3.1 残差来源核查（2026-11-10，支线）
# 假设：残差来自 (1) 输出存成 float16 的舍入；(2) dB 里的 "+ 中位数×1e-3" 地板项——修补改变了功率中位数，
# 地板项随之变，这不是整体平移缩放。做法：在 float32 下分别算 原流程 / 修补 / 修补但地板项沿用原中位数，比残差。
import sys, csv
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'work/review_20261109'))
from notch_fill_pilot import keep, fill_bins, rowf, HOLES  # 只用函数，不运行主程序
K = torch.as_tensor(keep, device='cuda')


def img(iq, fill, med_from=None):
    x = torch.as_tensor(iq, device='cuda', dtype=torch.complex64)
    s = torch.stft(x, 1024, 512, window=torch.hann_window(1024, device='cuda'), center=False, onesided=False, return_complex=True)
    p = torch.fft.fftshift(s.abs().square(), dim=-2)[K]
    if fill:
        p = fill_bins(p, 'M')
    p = F.adaptive_avg_pool2d(p[None, None], (256, 256))[0, 0]
    med = p.median() if med_from is None else med_from
    db = 10 * torch.log10(p + med.clamp_min(1e-30) * 1e-3)
    return ((db - db.mean()) / db.std().clamp_min(1e-8)), p.median()


bad = np.zeros(256, bool)
for n in HOLES:
    bad |= np.abs(rowf - n) <= 0.25
rs = list(csv.DictReader(open('E:/drone_lab_spec_L20_20261108/2.4GHz_test_spec_meta.csv', encoding='utf-8-sig')))
ids = ['L20_00007151', 'L20_00009388'] + [r['sample_id'] for r in rs[::150]][:8]
by = {r['sample_id']: r for r in rs}


def resid(a, b):
    a = a[~bad].ravel().double().cpu().numpy(); b = b[~bad].ravel().double().cpu().numpy()
    A = np.vstack([a, np.ones_like(a)]).T
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(np.abs(A @ c - b).max())


print('sample, label | float32 残差 | float32 地板沿用原中位数 | float16 残差')
for sid in ids:
    r = by[sid]
    z = np.array(np.load(r['block_file'], mmap_mode='r')[int(r['start']):int(r['start']) + int(r['n_samples'])])
    o, m0 = img(z, False)
    n1, _ = img(z, True)
    n2, _ = img(z, True, med_from=m0)
    r16 = resid(o.half().float(), n1.half().float())
    print(f"{sid} {r['dataset']}/{r['label']:10s} | {resid(o, n1):.5f} | {resid(o, n2):.6f} | {r16:.5f}")
