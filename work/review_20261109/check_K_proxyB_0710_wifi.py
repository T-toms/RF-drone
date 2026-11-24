# 核对任务 K 的 proxyB 底噪形状是否混入 07-10 下午会话的 WiFi（2026-11-10，支线）
# 方法与 work/side_5p8_controls_20261109/estimate.py 相同（块内 cell 的 p10、占用率<50% 掩码、跨块中位数），
# 只是把 07-10 train 块拆成上午 08:4x 与下午 11:4x 两组分别估计；各块先减去自身底噪 floor，便于比较形状。
# 运行：.venv/Scripts/python.exe -I work/review_20261109/check_K_proxyB_0710_wifi.py
from pathlib import Path
import csv
import numpy as np
P = Path(__file__).resolve().parents[2]
rows = lambda p: list(csv.DictReader(open(p, encoding='utf-8-sig')))
bs = rows(P / 'work/crop_slice_20261107/blocks.csv')
sp = {r['group_id']: r['split'] for r in rows(P / 'work/crop_slice_20261107/group_split.csv')}
FS, FC, n = 61440000, 2430., 512
freq = FC + np.fft.fftshift(np.fft.fftfreq(n, 1 / FS)) / 1e6


def est(sel):
    p10, occ = [], []
    for b in sel:
        c = np.load(P / 'work/crop_slice_20261107/cells' / Path(b['block_file']).stem / 'mean.npy')
        med = [np.median(c[:, (freq >= lo) & (freq < lo + 1)]) for lo in np.arange(FC - 25, FC + 25)]
        fl = float(min(med))
        p10.append(np.percentile(c, 10, axis=0) - fl)
        occ.append((c > fl + 3).mean(0))
    a = np.array(p10); v = np.array(occ) < .5
    return np.ma.median(np.ma.array(a, mask=~v), axis=0).filled(np.nan), v.sum(0), np.median(a, axis=0)


B = [b for b in bs if b['band'] == '2.4GHz' and sp[b['group_id']] == 'train' and b['dataset'] == 'v3'
     and b['label'] == 'background' and '20260710' in b['group_id']]
am = [b for b in B if any(t in b['group_id'] for t in ['-0848', '-0849', '-0853', '-0854'])]
pm = [b for b in B if any(t in b['group_id'] for t in ['-1143', '-1144', '-1147'])]
A = [b for b in bs if b['band'] == '2.4GHz' and sp[b['group_id']] == 'train' and b['dataset'] == 'data']
win = lambda lo, hi: (freq >= lo) & (freq <= hi)
ref = win(2451, 2454)
print('blocks: B all', len(B), 'B 08:4x', len(am), 'B 11:4x', len(pm), 'A', len(A))
for name, sel in [('B 08:4x', am), ('B 11:4x', pm), ('A data', A)]:
    s, c, raw = est(sel)
    out = []
    for lab, lo, hi in [('ch1 2408-16', 2408, 2416), ('ch6 2433-41', 2433, 2441), ('2445-49', 2445, 2449), ('2418-24', 2418, 2424)]:
        out.append(f'{lab}: masked {np.nanmean(s[win(lo, hi)]) - np.nanmean(s[ref]):+.2f} dB / raw p10 {np.mean(raw[win(lo, hi)]) - np.mean(raw[ref]):+.2f} dB')
    print(name, '| ch1 valid blocks min', c[win(2408, 2416)].min(), '|', '; '.join(out))
