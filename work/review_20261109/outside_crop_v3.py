# 城区接收机（v3）每次无人机录音：信号有多少落在固定截取频段之外（2026-11-11，支线）
# v3 是 100 MS/s、中心 2440/2450（覆盖约 2390–2500）或 5800（5750–5850），能看到整个 2.4G ISM 频段。
# 城区有无线网，所以不用功率，用“占用率超出城区背景的部分”：
#   每个 1 ms × 97.66 kHz 单元，最大功率 > 该块底噪 + 10 dB 记为占用；
#   无人机块每个频点的占用率，减去同中心频率的 07-09 城区背景块在该频点占用率的 90 百分位，负值记 0。
# 去掉两端各 1.5 MHz、中心频点 ±0.3 MHz。只读 G 盘 datasets_v3 原始 .mat（经任务 F 的 plan.json 定位）。
import csv, json, collections, sys
from pathlib import Path
import numpy as np
from scipy import io
R = Path('G:/drone')
P = Path(__file__).resolve().parents[1]
plan = json.loads((P / 'crop_slice_20261107/plan.json').read_text(encoding='utf-8'))
split = {r['group_id']: r['split'] for r in csv.DictReader(open(P / 'crop_slice_20261107/group_split.csv', encoding='utf-8-sig'))}
CROP = {'2.4GHz': (2405, 2455), '5.8GHz': (5755, 5845)}
ISM = {'2.4GHz': (2400, 2483.5), '5.8GHz': (5725, 5850)}
N = 1024


def occupancy(b):
    xs = [io.loadmat(R / p['path'], variable_names=['IQ'])['IQ'].ravel()[p['start']:p['end']].astype(np.complex64) for p in b['parts']]
    x = np.concatenate(xs)
    fs = b['fs_in']; per = int(round(fs / 1000)) // N  # 每个 1 ms 单元的帧数
    nf = (len(x) // N); fr = x[:nf * N].reshape(nf, N) * np.hanning(N).astype(np.float32)
    p = np.abs(np.fft.fftshift(np.fft.fft(fr, axis=1), axes=1)) ** 2
    nc = nf // per; p = p[:nc * per].reshape(nc, per, N)
    mean_db = 10 * np.log10(p.mean(1) + 1e-30); max_db = 10 * np.log10(p.max(1) + 1e-30)
    floor = float(np.median(np.percentile(mean_db, 10, axis=0)))
    return (max_db > floor + 10).mean(0)


bands = {}
for b in plan:
    if b['dataset'] != 'v3':
        continue
    key = (b['band'], b['fc_in'])
    if b['label'] == 'background' and '20260709' not in b['group_id']:
        continue  # 只用 07-09 城区背景做参照（城区无人机也是 07-09 录的）
    bands.setdefault(key, {'bg': [], 'drone': []})[('bg' if b['label'] == 'background' else 'drone')].append(b)
rows = []
for (band, fc), d in sorted(bands.items()):
    if not d['drone']:
        continue
    f = fc + np.fft.fftshift(np.fft.fftfreq(N, 1 / (d['drone'][0]['fs_in'] / 1e6)))
    good = (f > f.min() + 1.5) & (f < f.max() - 1.5) & (np.abs(f - fc) > 0.3)
    bg = np.array([occupancy(b) for b in d['bg']]) if d['bg'] else np.zeros((1, N))
    ref = np.percentile(bg, 90, axis=0)
    print(f'{band} fc={fc}: 无人机块 {len(d["drone"])}，城区背景参照块 {len(d["bg"])}', flush=True)
    lo, hi = CROP[band]; ilo, ihi = ISM[band]
    inc = (f >= lo) & (f <= hi); out_ism = ~inc & (f >= ilo) & (f <= ihi); out_non = ~inc & ~((f >= ilo) & (f <= ihi))
    for b in d['drone']:
        o = np.maximum(occupancy(b) - ref, 0) * good
        tot = o.sum() + 1e-12
        sm = np.convolve(o, np.ones(21) / 21, 'same')
        rows.append(dict(band=band, fc=fc, model=b['model'], group=b['group_id'], split=split.get(b['group_id'], '?'), block=Path(b['block_file']).stem,
                         rec_range=f'{f.min():.1f}-{f.max():.1f}', in_crop=o[inc].sum() / tot, out_ism=o[out_ism].sum() / tot,
                         out_nonism=o[out_non].sum() / tot, excess_total=float(o.sum()), peak_mhz=float(f[np.argmax(sm)])))
out = Path(__file__).resolve().parent / 'outside_crop_v3.csv'
with open(out, 'w', encoding='utf-8', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for band in ['2.4GHz', '5.8GHz']:
    print(f'\n== {band}：按录音组汇总（组内各块平均）')
    g = collections.defaultdict(list)
    for r in rows:
        if r['band'] == band:
            g[(r['model'], r['group'], r['split'], r['fc'])].append(r)
    by = collections.defaultdict(list)
    for (m, gid, sp, fc), rs in sorted(g.items()):
        oi = np.mean([r['out_ism'] for r in rs]); on = np.mean([r['out_nonism'] for r in rs]); ic = np.mean([r['in_crop'] for r in rs])
        by[m].append(oi + on)
        flag = ' <<<' if oi + on > 0.2 else ''
        print(f'  {m:14s} {sp:5s} fc={fc:.0f} 带内 {ic*100:5.1f}%  带外ISM内 {oi*100:5.1f}%  ISM外 {on*100:4.1f}%  峰 {np.median([r["peak_mhz"] for r in rs]):.1f}  {gid[-28:]}{flag}')
    print('  -- 按机型：带外占比 最大 / 中位（录音组数）')
    for k, v in sorted(by.items()):
        print(f'     {k:14s} 最大 {max(v)*100:5.1f}%  中位 {np.median(v)*100:5.1f}%  ({len(v)})')
