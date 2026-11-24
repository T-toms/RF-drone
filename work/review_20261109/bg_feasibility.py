# Feasibility: can receiver-A (data/) background be cut from existing data/ recordings?
# (1) time gaps: 1 ms cells with no emission inside the crop band
# (2) frequency gaps: sub-bands of the 122.88 MHz capture never occupied by any drone
# Input: Task A cache (cell_max_db 2000x1024 float16, 1 ms x 120 kHz, full capture band).
import json, glob, os, sys, numpy as np
CACHE = 'E:/drone_lab_cache_20261031'
OUT = os.path.dirname(os.path.abspath(__file__))
CROP = {'2.4GHz': (2405, 2455), '5.8GHz': (5755, 5845)}
EXCL = {'2.4GHz': [2400.0, 2426.88, 2440.0, 2450.0, 2457.6], '5.8GHz': [5800.0, 5795.5, 5740.0]}
T_DB = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0   # threshold above recording floor

rows = []
occ_union = {}
for d in sorted(glob.glob(os.path.join(CACHE, '*'))):
    mp = os.path.join(d, 'meta.json')
    if not os.path.exists(mp):
        continue
    m = json.load(open(mp, encoding='utf-8'))
    band, fc = m['band'], m['fc_hz'] / 1e6
    cmax = np.load(os.path.join(d, 'cell_mean_db.npy')).astype(np.float32)  # MEAN over the 1 ms cell (max of noise sits ~+5 dB above floor)
    valid = np.load(os.path.join(d, 'cell_valid.npy')) & ~np.isnan(cmax).any(1)
    cmax = np.nan_to_num(cmax, nan=999.0)
    f = fc + (np.arange(1024) - 512) * 0.12
    floor = m['floor_db']
    lo, hi = CROP[band]
    sel = (f >= lo) & (f <= hi)
    for e in EXCL[band]:
        sel &= np.abs(f - e) > 0.25
    hot = cmax > floor + T_DB                     # (cells, bins)
    occ = hot[valid].mean(0)
    occ_union.setdefault(band, []).append(occ)
    busy = hot[:, sel].any(1) | ~valid            # cell busy if any crop-band bin hot (invalid counts busy)
    # runs of quiet cells
    runs, r = [], 0
    for b in busy:
        if b:
            if r: runs.append(r)
            r = 0
        else:
            r += 1
    if r: runs.append(r)
    runs = np.array(runs or [0])
    rows.append(dict(rec=os.path.basename(d), band=band, model=m['model'], state=m['state'], bw=m['bw_label'],
                     duty=m.get('duty'), quiet_frac=float((~busy).mean()), longest_quiet_ms=int(runs.max()),
                     quiet_ms_in_runs_ge10=int(runs[runs >= 10].sum()), quiet_ms_in_runs_ge20=int(runs[runs >= 20].sum()),
                     n_runs_ge20=int((runs >= 20).sum())))

import csv
with open(os.path.join(OUT, f'bg_time_gaps_mean_T{T_DB:g}.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
for band in CROP:
    rb = [r for r in rows if r['band'] == band]
    print(f'== {band} T={T_DB} dB  recordings={len(rb)}')
    print('  recordings with any quiet run >=20 ms:', sum(r['n_runs_ge20'] > 0 for r in rb),
          ' total quiet ms in runs>=20:', sum(r['quiet_ms_in_runs_ge20'] for r in rb),
          ' in runs>=10:', sum(r['quiet_ms_in_runs_ge10'] for r in rb))
    for r in sorted(rb, key=lambda r: -r['longest_quiet_ms'])[:12]:
        print('   ', r['rec'], r['state'], 'longest', r['longest_quiet_ms'], 'ms  ge20', r['quiet_ms_in_runs_ge20'], 'quiet_frac %.2f' % r['quiet_frac'])
    O = np.stack(occ_union[band])                 # recordings x bins
    fc = 2400 if band == '2.4GHz' else 5800
    f = fc + (np.arange(1024) - 512) * 0.12
    mx = O.max(0)
    np.save(os.path.join(OUT, f'bg_occ_max_mean_{band}_T{T_DB:g}.npy'), np.stack([f, mx, O.mean(0)]))
    # contiguous frequency spans where max occupancy over ALL recordings < 1%
    free = mx < 0.01
    spans, s = [], None
    for i, v in enumerate(free):
        if v and s is None: s = i
        if (not v or i == len(free) - 1) and s is not None:
            e = i if not v else i + 1
            spans.append((f[s], f[e - 1] + 0.12)); s = None
    spans = sorted(spans, key=lambda x: -(x[1] - x[0]))[:6]
    print('  widest spans with occupancy <1% in every recording (MHz):', [(round(a, 2), round(b, 2), round(b - a, 2)) for a, b in spans])

