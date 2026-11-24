"""Task A: read-only MAT overview and raw-power feature cache (2026-10-31)."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import time
import warnings
from collections import defaultdict
import numpy as np
from scipy.io import loadmat
from scipy.signal import correlate
from scipy.stats import percentileofscore
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FS = 122_880_000
NFFT = 4096
SEG_SAMPLES = 6_144_000
CELL_SAMPLES = 122_880
THRESH_DB = 10
ROOT = Path('G:/drone')
PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'work/lab_long_overview_20261031'
CACHE = Path('E:/drone_lab_cache_20261031')
MANIFEST = PROJECT / 'work/audit_drone_20261023/data_preprocessing_manifest.csv'
FN = (4 * np.arange(1024) + 1.5 - 2048) / 4096
MASK = np.abs(FN) <= .45
MASK[511:514] = False
MODELS = dict(zip([f'{i:03}' for i in range(1, 15)],
    ['mini4 pro', 'Flip', 'mini 4k', 'neo', 'neo2', 'air 3s', 'mini5 pro',
     'mavic 4', 'avata2', 'evo lite 6k', 'evo ii pro v3', 'avata', 'atom se', 'atom']))
SHAPES = {'cell_mean_db': ((2000, 1024), 'float16'),
          'cell_max_db': ((2000, 1024), 'float16'),
          'cell_min_db': ((2000, 1024), 'float16'),
          'cell_valid': ((2000,), 'bool'), 'frame_power_db': ((60000,), 'float32'),
          'psd_mean': ((4096,), 'float64'), 'psd_max': ((4096,), 'float64'),
          'occ_any': ((1024,), 'float32'), 'occ_sus': ((1024,), 'float32')}


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean(x) for x in v]
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    return v


def write_json(p, data):
    p.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_json(p):
    return json.loads(p.read_text(encoding='utf-8'))


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-20))


def inventory():
    if not ROOT.is_dir():
        raise RuntimeError('G:/drone unavailable; ask coordinator, do not substitute inputs')
    n, newest = 0, 0
    for p in ROOT.rglob('*'):
        if p.is_file():
            n += 1
            newest = max(newest, p.stat().st_mtime_ns)
    return dict(file_count=n, latest_mtime_ns=newest)


def groups():
    result = defaultdict(dict)
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r['duplicate'] != 'False':
                continue
            ordinal = int(r['epoch']) * 20 + int(r['seg'])
            assert 0 <= ordinal < 40
            key = (r['folder'], r['session'])
            assert ordinal not in result[key], (key, ordinal)
            result[key][ordinal] = r
    if len(result) != 96:
        raise RuntimeError(f'Expected 96 recordings, got {len(result)}; ask coordinator')
    return dict(sorted(result.items()))


def bounds(r):
    if r['analysis_eligible'] != 'True':
        return 0, 0
    return int(r['start_sample']), int(r['end_sample_exclusive'])


def valid_cells(r):
    lo, hi = bounds(r)
    starts = np.arange(50) * CELL_SAMPLES
    return (starts >= lo) & (starts + CELL_SAMPLES <= hi)


def zoom_selection(rows):
    selected = []
    for target in (5, 20, 35):
        available = [i for i, r in rows.items()
                     if i // 20 == target // 20 and r['analysis_eligible'] == 'True']
        if not available:
            raise RuntimeError('No eligible zoom segment in epoch; ask coordinator')
        selected.append(min(available, key=lambda i: (abs(i-target), i)))
    return selected


def load(r):
    x = loadmat(Path(ROOT, r['path']), variable_names=['data'])['data'].ravel().astype(np.complex64, copy=False)
    assert x.size == SEG_SAMPLES, (r['path'], x.size)
    return x


def spectrum(x, nfft):
    w = np.hanning(nfft).astype(np.float32)
    frames = x.reshape(-1, nfft)
    return np.abs(np.fft.fftshift(np.fft.fft(frames * w, axis=1), axes=1)) ** 2 / np.sum(w*w)


def segment(x, lo, hi):
    p = spectrum(x, NFFT)
    folded = p.reshape(-1, 1024, 4).mean(axis=-1).reshape(50, 30, 1024)
    starts = np.arange(1500)*NFFT
    fv = (starts >= lo) & (starts + NFFT <= hi)
    cv = fv.reshape(50, 30).all(axis=1)
    cells = {}
    for name, v in [('mean', folded.mean(axis=1)), ('max', folded.max(axis=1)), ('min', folded.min(axis=1))]:
        value = db(v).astype(np.float16)
        value[~cv] = np.nan
        cells['cell_'+name+'_db'] = value
    power = db(np.mean(np.abs(x.reshape(1500, NFFT))**2, axis=1)).astype(np.float32)
    power[~fv] = np.nan
    return cells, cv, power, p[fv].sum(axis=0, dtype=np.float64), p[fv].max(axis=0) if fv.any() else np.zeros(NFFT), int(fv.sum())


def period(values, low, high):
    v = np.array(values, dtype=np.float64)
    if not np.isfinite(v).any():
        return None
    v[~np.isfinite(v)] = np.nanmedian(v)
    v -= v.mean()
    ac = correlate(v, v, mode='full', method='fft')[len(v)-1:]
    if ac[0] <= 0:
        return None
    ac /= ac[0]
    high = min(high, len(ac)-2)
    minima = [l for l in range(low, high+1) if ac[l] <= ac[l-1] and ac[l] < ac[l+1]]
    if not minima:
        return None
    peaks = [l for l in range(max(low, minima[0]), high+1) if ac[l] > ac[l-1] and ac[l] >= ac[l+1]]
    if not peaks:
        return None
    lag = max(peaks, key=lambda l: ac[l])
    return int(lag) if ac[lag] >= .3 else None


def boundary(a, mean, active, valid):
    blocks = []
    for k in range(20):
        sl = slice(k*100, (k+1)*100)
        ok = valid[sl]
        blocks.append(None if ok.sum() < 50 else (
            float(np.median(a[sl][ok])),
            float(np.median(mean[sl][ok][:, MASK].astype(np.float32).mean(axis=1))),
            active[sl][ok][:, MASK].mean(axis=0)))
    def diff(i, j):
        x, y = blocks[i], blocks[j]
        if x is None or y is None:
            return None
        nx, ny = np.linalg.norm(x[2]), np.linalg.norm(y[2])
        cosine = 0. if nx == ny == 0 else (1. if nx*ny == 0 else float(1-np.dot(x[2], y[2])/(nx*ny)))
        return np.array([abs(x[1]-y[1]), abs(x[0]-y[0]), cosine])
    within = [d for i in range(19) if i != 9 and (d := diff(i, i+1)) is not None]
    edge = diff(9, 10)
    ranks = [None]*3 if edge is None or not within else [float(percentileofscore(np.array(within)[:,k], edge[k], kind='rank')) for k in range(3)]
    return dict(boundary_power_delta_db=None if edge is None else edge[0],
                boundary_active_delta=None if edge is None else edge[1],
                boundary_cosine_distance=None if edge is None else edge[2],
                power_percentile=ranks[0], active_percentile=ranks[1], cosine_percentile=ranks[2],
                epoch_boundary_flag=any(x is not None and x > 95 for x in ranks),
                within_pair_count=len(within), within_pair_metrics=[v.tolist() for v in within])


def features(arr, fc):
    mean, maximum, valid = arr['cell_mean_db'], arr['cell_max_db'], arr['cell_valid']
    assert valid.any(), 'No valid cells; ask coordinator'
    floor = float(np.median(np.percentile(mean[valid][:, MASK].astype(np.float32), 10, axis=0)))
    active = (maximum.astype(np.float32) > floor+THRESH_DB) & MASK
    sustained = (mean.astype(np.float32) > floor+THRESH_DB) & MASK
    any_occ, sus_occ = active[valid].mean(axis=0).astype(np.float32), sustained[valid].mean(axis=0).astype(np.float32)
    arr['occ_any'], arr['occ_sus'] = any_occ, sus_occ
    a = active.sum(axis=1).astype(float)
    a[~valid] = np.nan
    occupied = np.flatnonzero((any_occ >= .05) & MASK)
    ranges = []
    for k in occupied:
        if ranges and k-ranges[-1][1]-1 <= 2:
            ranges[-1][1] = int(k)
        else:
            ranges.append([int(k), int(k)])
    bands = []
    for lo, hi in ranges:
        left = fc/1e6 + (4*lo-2048)/4096*FS/1e6
        right = fc/1e6 + (4*(hi+1)-2048)/4096*FS/1e6
        # Merged gaps are included; excluded mask bins never contribute statistics.
        ix = np.arange(lo, hi+1)[MASK[lo:hi+1]]
        bands.append(dict(lo_MHz=left, hi_MHz=right, width_MHz=right-left,
                          mean_occ_any=float(any_occ[ix].mean()), mean_occ_sus=float(sus_occ[ix].mean())))
    fine = period(arr['frame_power_db'], 3, 900)
    f = dict(floor_db=floor, duty=float(np.mean(a[valid] >= 5)), n_subbands=len(bands),
             subbands_mhz='; '.join(f"{b['lo_MHz']:.3f}-{b['hi_MHz']:.3f}" for b in bands),
             period_coarse_ms=period(a, 2, 200), period_fine_frames=fine,
             period_fine_ms=None if fine is None else fine*NFFT/FS*1000,
             occupied_bands=bands)
    f.update(boundary(a, mean, active, valid))
    return f, a


def title(meta):
    return (f"{meta['code']} {meta['model']} | {meta['state']} | {meta['band']} | BW {meta['bw_label']}"
            f"{' combination' if meta['combination'] else ''} | session {meta['session']} | "
            f"fc {meta['fc_hz']/1e6:g} MHz, fs assumed 122.88 MS/s")


def plots(meta, arr, a, zooms, out):
    fc = meta['fc_hz']/1e6
    floor = meta['floor_db']
    extent_f = (fc-FS/2e6, fc+FS/2e6)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#bdbdbd')
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), layout='constrained')
    f = fc+(np.arange(NFFT)-2048)/NFFT*FS/1e6
    for key in ('psd_mean', 'psd_max'):
        axes[0].plot(f, db(arr[key])-floor, label=key)
    axes[0].axhline(THRESH_DB, color='r', ls='--', label='floor + 10 dB')
    mask4096 = np.repeat(MASK, 4)
    axes[0].fill_between(f, 0, 1, where=~mask4096, transform=axes[0].get_xaxis_transform(), color='gray', alpha=.3)
    axes[0].set(xlabel='Frequency (MHz)', ylabel='Power above floor (dB)')
    axes[0].legend()
    for ax, key in zip(axes[1:3], ('cell_max_db', 'cell_mean_db')):
        im = ax.imshow(arr[key].astype(np.float32).T-floor, origin='lower', aspect='auto',
                       extent=(0, 2000, *extent_f), vmin=-5, vmax=40, cmap=cmap, interpolation='nearest')
        ax.axvline(1000, color='white', ls='--')
        ax.set(xlabel='Time (ms)', ylabel='Frequency (MHz)', title=key)
        top = ax.secondary_xaxis('top')
        top.set_xticks(np.arange(0, 2001, 50), labels=['']*41)
        fig.colorbar(im, ax=ax, label='dB above floor')
    axes[3].plot(np.arange(2000)+.5, a, lw=.6)
    axes[3].set(xlabel='Time (ms)', ylabel='Active bin count', xlim=(0, 2000))
    twin = axes[3].twinx()
    fp = arr['frame_power_db'].reshape(2000, 30)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        envelope = np.nanmax(fp, axis=1)
    twin.plot(np.arange(2000)+.5, envelope-np.nanmedian(envelope), color='orange', alpha=.7, lw=.5)
    twin.set_ylabel('Frame envelope relative to median (dB)')
    fig.suptitle(title(meta), fontsize=11)
    stem = f"{meta['folder']}__{meta['session']}"
    fig.savefig(out/(stem+'_overview.png'), dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), layout='constrained')
    for ax, ordinal in zip(axes, meta['zoom_ordinals']):
        z = zooms[ordinal]
        top = float(np.nanpercentile(z, 99.5))
        start = ordinal*SEG_SAMPLES/FS*1000
        end = start+SEG_SAMPLES/FS*1000
        im = ax.imshow(z.T, origin='lower', aspect='auto', extent=(start, end, *extent_f),
                       vmin=top-45, vmax=top, cmap=cmap, interpolation='nearest')
        for t in np.arange(start, end+.001, 10):
            ax.axvline(t, color='white', ls='--', lw=.6)
        ax.set(xlabel='Time in recording (ms)', ylabel='Frequency (MHz)', title=f'Ordinal {ordinal}: 1024-point Hann, hop 1024')
        fig.colorbar(im, ax=ax, label='Raw power (dB)')
    fig.suptitle(title(meta), fontsize=11)
    fig.savefig(out/(stem+'_zoom.png'), dpi=110)
    plt.close(fig)


def process(key, rows, cache=CACHE, out=OUT, loader=load):
    folder, session = key
    dest = cache/(folder+'__'+session)
    if (dest/'meta.json').exists():
        return read_json(dest/'meta.json'), True
    dest.mkdir(parents=True, exist_ok=True)
    code = folder.removeprefix('F')[:3]
    fc = 2400e6 if '2.4' in folder else 5800e6
    bandwidth = re.search(r'_(\d+)M?(?:_combination)?-signal$', folder)
    if bandwidth is None:
        raise RuntimeError(f'Cannot parse bandwidth: {folder}; ask coordinator')
    meta = dict(folder=folder, session=session, code=code, model=MODELS[code],
                state='flight' if folder.startswith('F') else 'standby', band='2.4GHz' if fc == 2400e6 else '5.8GHz',
                fc_hz=fc, bw_label=bandwidth.group(1)+'M', combination='_combination' in folder,
                fs_assumed_hz=FS, n_cells=2000, missing_ordinals=sorted(set(range(40))-rows.keys()),
                ineligible_ordinals=[i for i,r in rows.items() if r['analysis_eligible'] != 'True'],
                zoom_ordinals=zoom_selection(rows), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                sources=[dict(path=r['path'], ordinal=i, valid_start=bounds(r)[0], valid_end=bounds(r)[1]) for i,r in sorted(rows.items())])
    arr = {name: np.full(shape, False if dtype == 'bool' else np.nan, dtype=dtype)
           for name,(shape,dtype) in SHAPES.items() if not name.startswith('occ_')}
    psum, pmax, count, zooms = np.zeros(NFFT), np.zeros(NFFT), 0, {}
    for i,r in sorted(rows.items()):
        if r['analysis_eligible'] != 'True':
            continue
        x = loader(r)
        lo,hi = bounds(r)
        cells, cv, fp, total, maximum, n = segment(x, lo, hi)
        for name,v in cells.items():
            arr[name][i*50:(i+1)*50] = v
        arr['cell_valid'][i*50:(i+1)*50] = cv
        arr['frame_power_db'][i*1500:(i+1)*1500] = fp
        psum += total
        np.maximum(pmax, maximum, out=pmax)
        count += n
        if i in meta['zoom_ordinals']:
            z = db(spectrum(x, 1024)).astype(np.float32)
            starts = np.arange(6000)*1024
            z[(starts < lo) | (starts+1024 > hi)] = np.nan
            zooms[i] = z
    assert count > 0
    arr['psd_mean'], arr['psd_max'] = psum/count, pmax
    feat, a = features(arr, fc)
    meta.update(feat)
    meta['n_valid_cells'] = int(arr['cell_valid'].sum())
    meta['n_valid_frames'] = count
    for name,v in arr.items():
        np.save(dest/(name+'.npy'), v)
    plots(meta, arr, a, zooms, out)
    write_json(dest/'meta.json', meta)  # completion marker, written after all arrays and plots
    return meta, False


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summaries(metas):
    keys = ['folder','session','code','model','state','band','bw_label','combination','n_valid_cells',
            'floor_db','duty','n_subbands','subbands_mhz','period_coarse_ms','period_fine_frames','period_fine_ms',
            'epoch_boundary_flag','power_percentile','active_percentile','cosine_percentile']
    csv_write(OUT/'recording_summary.csv', [{k:m[k] for k in keys} for m in metas])
    edge_keys = ['folder','session','boundary_power_delta_db','boundary_active_delta','boundary_cosine_distance',
                 'power_percentile','active_percentile','cosine_percentile','epoch_boundary_flag','within_pair_count']
    csv_write(OUT/'epoch_boundary.csv', [{k:m[k] for k in edge_keys} for m in metas])
    write_json(OUT/'occupied_bands.json', {m['folder']+'__'+m['session']:m['occupied_bands'] for m in metas})
    for band in ('2.4GHz', '5.8GHz'):
        ms = sorted([m for m in metas if m['band'] == band], key=lambda m:(m['code'],m['state'],int(m['bw_label'][:-1]),m['session']))
        fc = ms[0]['fc_hz']/1e6
        data = np.stack([np.load(CACHE/(m['folder']+'__'+m['session'])/'occ_any.npy') for m in ms])
        fig,ax = plt.subplots(figsize=(16, max(8,len(ms)*.22)), layout='constrained')
        im = ax.imshow(data, origin='lower', aspect='auto', interpolation='nearest', vmin=0, vmax=1,
                       extent=(fc-FS/2e6, fc+FS/2e6, -.5,len(ms)-.5))
        ax.set_yticks(np.arange(len(ms)), [m['code']+' '+m['state']+' '+m['bw_label'] for m in ms], fontsize=7)
        ax.set(xlabel='Frequency (MHz)', title=band+' occupancy | fs assumed 122.88 MS/s')
        fig.colorbar(im,ax=ax,label='occ_any')
        fig.savefig(OUT/('band_occupancy_'+band+'.png'),dpi=110)
        plt.close(fig)


def verification(gs, metas):
    results = {}
    errors, counts, invalid, zeros, prefixes = [], [], [], [], []
    arrays = {}
    for m in metas:
        key = (m['folder'],m['session'])
        stem = '__'.join(key)
        d = CACHE/stem
        ar = {name:np.load(d/(name+'.npy'), mmap_mode='r') for name in SHAPES}
        arrays[key] = ar
        for name,(shape,dtype) in SHAPES.items():
            if ar[name].shape != shape or ar[name].dtype != np.dtype(dtype):
                errors.append([stem,name,str(ar[name].shape),str(ar[name].dtype)])
        expected = sum(sum(s >= bounds(r)[0] and s+CELL_SAMPLES <= bounds(r)[1]
                           for s in range(0,SEG_SAMPLES,CELL_SAMPLES)) for r in gs[key].values())
        if expected != m['n_valid_cells'] or expected != int(ar['cell_valid'].sum()):
            counts.append([stem,expected,m['n_valid_cells']])
        for ordinal in range(40):
            sl = slice(ordinal*50,(ordinal+1)*50)
            r = gs[key].get(ordinal)
            if r is None:
                ok = not ar['cell_valid'][sl].any() and all(np.isnan(ar[n][sl]).all() for n in ('cell_mean_db','cell_max_db','cell_min_db'))
                invalid.append(dict(recording=stem,ordinal=ordinal,pass_check=bool(ok)))
            elif r['all_zero'] == 'True':
                zeros.append(dict(recording=stem,ordinal=ordinal,pass_check=bool(not ar['cell_valid'][sl].any() and all(np.isnan(ar[n][sl]).all() for n in ('cell_mean_db','cell_max_db','cell_min_db')))))
            elif int(r['removed_zero_prefix_samples'] or 0) > 0:
                starts = np.arange(50)*CELL_SAMPLES
                before = starts < int(r['start_sample'])
                prefixes.append(dict(recording=stem,ordinal=ordinal,pass_check=bool(not ar['cell_valid'][sl][before].any() and all(np.isnan(ar[n][sl][before]).all() for n in ('cell_mean_db','cell_max_db','cell_min_db')))))
    results['1_shapes_dtypes'] = dict(status='pass' if len(metas)==96 and not errors else 'fail',recordings=len(metas),errors=errors)
    results['2_manifest_validity'] = dict(status='pass' if not counts else 'fail',mismatches=counts)
    rng = np.random.default_rng(20261031)
    sampled = []
    keys = list(gs)
    for idx in rng.choice(len(keys),3,replace=False):
        key = keys[int(idx)]
        ar = arrays[key]
        for cell in rng.choice(np.flatnonzero(ar['cell_valid']),2,replace=False):
            ordinal, local = divmod(int(cell),50)
            x = load(gs[key][ordinal])[local*CELL_SAMPLES:(local+1)*CELL_SAMPLES]
            p = spectrum(x,NFFT).reshape(30,1024,4).mean(axis=-1)
            diff = {}
            for n,v in [('mean',p.mean(axis=0)),('max',p.max(axis=0)),('min',p.min(axis=0))]:
                diff[n] = float(np.max(np.abs(db(v)-ar['cell_'+n+'_db'][cell].astype(np.float32))))
            sampled.append(dict(recording='__'.join(key),cell=int(cell),errors_db=diff))
    results['3_raw_recompute'] = dict(status='pass' if max(v for s in sampled for v in s['errors_db'].values()) < .1 else 'fail',samples=sampled)
    results['4_missing'] = dict(status='pass' if len(invalid)==4 and all(x['pass_check'] for x in invalid) else 'fail',segments=invalid)
    results['5_zero_prefix'] = dict(status='pass' if len(zeros)==15 and len(prefixes)==25 and all(x['pass_check'] for x in zeros+prefixes) else 'fail',zeros=zeros,prefixes=prefixes)
    floors = {}
    for band in ('2.4GHz','5.8GHz'):
        ms = [m for m in metas if m['band']==band]
        vals = [m['floor_db'] for m in ms]
        med = float(np.median(vals))
        floors[band] = dict(min=min(vals),median=med,max=max(vals),
                           over_6db_from_band_median=[m['folder']+'__'+m['session'] for m in ms if abs(m['floor_db']-med)>6])
    results['6_floor_ranges'] = dict(status='pass',bands=floors)
    energy = {}
    for key,ar in arrays.items():
        if '2.4' in key[0]:
            p = ar['psd_mean']
            energy['__'.join(key)] = float(p[2049:].sum()/p.sum())
    results['7_positive_energy'] = dict(status='pass',ratios=energy,below_08=[k for k,v in energy.items() if v<.8])
    pngs = [OUT/('__'.join(k)+suffix+'.png') for k in gs for suffix in ('_overview','_zoom')]
    pngs += [OUT/('band_occupancy_'+b+'.png') for b in ('2.4GHz','5.8GHz')]
    sizes = {p.name:p.stat().st_size if p.exists() else 0 for p in pngs}
    results['8_pngs'] = dict(status='pass' if all(n>50*1024 for n in sizes.values()) else 'fail',count=len(sizes),sizes_bytes=sizes)
    before = read_json(OUT/'source_inventory_before.json')
    after = inventory()
    write_json(OUT/'source_inventory_after.json', after)
    results['9_source_unchanged'] = dict(status='pass' if before==after else 'fail',before=before,after=after)
    write_json(OUT/'verification.json',results)
    return results


def config():
    return dict(fs_assumed_hz=FS,nfft=NFFT,window='np.hanning float32',hop=NFFT,
                segment_samples=SEG_SAMPLES,cell_samples=CELL_SAMPLES,frames_per_cell=30,
                cells_per_recording=2000,fold='mean of adjacent 4 fftshift power bins',
                power='abs(fftshift(fft(x*hann)))**2 / sum(hann**2)',db_floor=1e-20,
                normalization='none; raw digital power',threshold_db=THRESH_DB,
                mask=MASK.tolist(),mask_rule='abs(f_norm)<=0.45; exclude 511,512,513',
                freq_norm_centers=FN.tolist(),freq_mhz_centers={b:(f+FN*FS/1e6).tolist() for b,f in [('2.4GHz',2400),('5.8GHz',5800)]},
                floor_rule='median of per-bin valid-cell 10th percentiles within mask',
                active_rule='cell_max_db > floor+10',sustained_rule='cell_mean_db > floor+10',
                occupancy_threshold=.05,merge_gap_bins=2,duty_active_bins=5,
                coarse_lags=[2,200],fine_lags=[3,900],autocorrelation_threshold=.3,
                autocorrelation_normalization='divide by lag-zero; NaN fill recording median then demean',
                autocorrelation_peak_rule='first local minimum at/after low, then strongest local maximum through high; no extrema => null; peak >=0.3',
                epoch_block_cells=100,epoch_min_valid_cells=50,epoch_flag_percentile=95,
                percentile_rule='scipy.stats.percentileofscore(kind=rank); ties use average rank',
                zoom_targets=[5,20,35],zoom_nfft=1024,zoom_hop=1024,zoom_top_percentile=99.5,zoom_span_db=45,
                overview_limits_db=[-5,40],random_seed=20261031,
                floor_outlier_rule='absolute difference from same-band median > 6 dB')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', nargs='+', help='Exact folder names; optional folder__session selectors')
    args = ap.parse_args()
    gs = groups()
    chosen = {k:r for k,r in gs.items() if not args.only or k[0] in args.only or '__'.join(k) in args.only}
    if args.only and (not chosen or any(not any(s in (k[0],'__'.join(k)) for k in gs) for s in args.only)):
        raise RuntimeError('Unknown --only selector')
    before = inventory()  # never create outputs before verifying original input availability
    OUT.mkdir(parents=True,exist_ok=True)
    CACHE.mkdir(parents=True,exist_ok=True)
    if not (OUT/'source_inventory_before.json').exists():
        write_json(OUT/'source_inventory_before.json',before)
    write_json(OUT/'config.json',config())
    started = time.time()
    for key,rows in chosen.items():
        tick = time.time()
        m,skipped = process(key,rows)
        line = f"{list(gs).index(key)+1}/96 {key[0]} {key[1]} {time.time()-tick:.1f}s {'resumed' if skipped else 'completed'}"
        print(line,flush=True)
        with (OUT/'run.log').open('a',encoding='utf-8') as f:
            f.write(line+'\n')
    if args.only:
        print('Smoke processing done; inspect all six PNGs before full run.',flush=True)
        return
    metas = [read_json(CACHE/('__'.join(k))/'meta.json') for k in gs]
    summaries(metas)
    result = verification(gs,metas)
    flags = [m['folder']+'__'+m['session'] for m in metas if m['epoch_boundary_flag']]
    lines = ['# Task A results',f'Completed {len(metas)}/96 recordings; this invocation elapsed {time.time()-started:.1f} s (per-record timing: run.log).',
             'Raw power retained; fs assumed 122.88 MS/s; no screening, training or split performed.',
             'Verification: '+', '.join(k+': '+v['status'] for k,v in result.items()),
             'Floor ranges (dB): '+json.dumps(result['6_floor_ranges']['bands'],ensure_ascii=False),
             'Epoch boundary flags: '+('; '.join(flags) or 'none'),
             'Positive-frequency energy <0.8: '+('; '.join(result['7_positive_energy']['below_08']) or 'none'),
             'Known gaps: F009_2.4GHZ_60M ordinals 4/15/25; 003_2.4GHz_20M ordinal 10.',
             '15 all-zero and 25 zero-prefix source segments independently checked; invalid cells kept as NaN.',
             'Detailed numeric evidence: verification.json; recording_summary.csv; epoch_boundary.csv; occupied_bands.json.',
             'Visual smoke inspection is performed by the operator before the full run.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if any(v['status']!='pass' for v in result.values()):
        raise RuntimeError('Verification failed; inspect verification.json')


if __name__ == '__main__':
    main()
