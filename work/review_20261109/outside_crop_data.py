# 实验室接收机（data/）每次录音：无人机信号有多少落在固定截取频段之外（2026-11-11，支线）
# 数据：任务 A 的 1 ms 全频段缓存（截频前，fc 2400/5800，122.88 MS/s，1024 点 = 120 kHz/点，2000 个 1 ms 单元）。只读。
# 做法：每个单元每个频点，若 cell_max > 该录音底噪 + 10 dB 记为“有信号”，累加其平均功率超出底噪的部分（线性）。
#   去掉：录音两端各 1.5 MHz（滤波器滚降和折叠镜像）、已知稳定单音 ±0.3 MHz（fc、2426.88、2457.6、5740）。
#   区域：截取带内；截取带外但在 ISM 频段内；截取带外且在 ISM 外。另报“截取带上沿以上到录音上限”的能量，作为信号被录音上限截断的迹象。
import csv, json, collections
from pathlib import Path
import numpy as np
C = Path('E:/drone_lab_cache_20261031')
P = Path(__file__).resolve().parents[1]
split = {r['group_id']: r['split'] for r in csv.DictReader(open(P / 'crop_slice_20261107/group_split.csv', encoding='utf-8-sig'))}
CROP = {'2.4GHz': (2405, 2455), '5.8GHz': (5755, 5845)}
ISM = {'2.4GHz': (2400, 2483.5), '5.8GHz': (5725, 5850)}
TONES = {'2.4GHz': [2400.0, 2426.88, 2457.6], '5.8GHz': [5800.0, 5740.0]}
rows = []
for d in sorted(p for p in C.iterdir() if p.is_dir() and (p / 'meta.json').exists()):
    m = json.loads((d / 'meta.json').read_text(encoding='utf-8'))
    b = m['band']; fc = m['fc_hz'] / 1e6; fs = m['fs_assumed_hz'] / 1e6
    f = fc + np.fft.fftshift(np.fft.fftfreq(1024, 1 / fs))
    mx = np.load(d / 'cell_max_db.npy').astype(np.float32); mn = np.load(d / 'cell_mean_db.npy').astype(np.float32)
    v = np.load(d / 'cell_valid.npy'); mx, mn = mx[v], mn[v]
    fl = float(m['floor_db'])
    good = (f > f.min() + 1.5) & (f < f.max() - 1.5)
    for t in TONES[b]:
        good &= np.abs(f - t) > 0.3
    occ = mx > fl + 10
    ex = np.where(occ, np.maximum(10 ** (mn / 10) - 10 ** (fl / 10), 0), 0).sum(0) * good
    lo, hi = CROP[b]; ilo, ihi = ISM[b]
    inc = (f >= lo) & (f <= hi)
    out_ism = ~inc & (f >= ilo) & (f <= ihi)
    out_non = ~inc & ~((f >= ilo) & (f <= ihi))
    top = (f > hi) & (f < f.max() - 1.5)
    tot = ex.sum() + 1e-30
    sm = np.convolve(ex, np.ones(17) / 17, 'same')  # 约 2 MHz 平滑，找最强信号中心
    gid = 'data:' + d.name
    rows.append(dict(recording=d.name, model=m['model'], band=b, state=m['state'], bw=m['bw_label'], split=split.get(gid, '?'),
                     rec_range=f'{f.min():.1f}-{f.max():.1f}', in_crop=ex[inc].sum() / tot, out_ism=ex[out_ism].sum() / tot,
                     out_nonism=ex[out_non].sum() / tot, above_crop=ex[top].sum() / tot, peak_mhz=float(f[np.argmax(sm)])))
out = Path(__file__).resolve().parent / 'outside_crop_data.csv'
with open(out, 'w', encoding='utf-8', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('录音数', len(rows))
for b in ['2.4GHz', '5.8GHz']:
    print(f'\n== {b}：每次录音，截取带外（ISM 内）能量占比 | 截取带上沿以上占比 | 最强信号中心')
    for r in sorted((r for r in rows if r['band'] == b), key=lambda r: (r['model'], r['state'], r['bw'])):
        flag = ' <<<' if r['out_ism'] + r['out_nonism'] > 0.2 else ''
        print(f"  {r['model']:13s} {r['state']:8s} {r['bw']:4s} {r['split']:5s} 带内 {r['in_crop']*100:5.1f}%  带外ISM内 {r['out_ism']*100:5.1f}%  ISM外 {r['out_nonism']*100:4.1f}%  上沿以上 {r['above_crop']*100:5.1f}%  峰 {r['peak_mhz']:.1f}{flag}")
    by = collections.defaultdict(list)
    for r in rows:
        if r['band'] == b:
            by[r['model']].append(r['out_ism'] + r['out_nonism'])
    print(f'  -- 按机型：带外占比的 最大 / 中位（录音数）')
    for k, v in sorted(by.items()):
        print(f'     {k:13s} 最大 {max(v)*100:5.1f}%  中位 {np.median(v)*100:5.1f}%  ({len(v)})')
