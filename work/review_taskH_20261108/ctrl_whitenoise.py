"""Claude review of Task H: separate 'masking' from 'v3-environment cue'.

For each test drone sample d (data/ or v3), and the same v3 test background b_j that
Task H's mix_test.py paired it with, build:
  mix_v3  = b_j + g*d          (Task H's mix, recomputed here from its cache)
  mix_wn  = n_j + g*d          n_j = complex white Gaussian noise with power == mean|b_j|^2
plus noise-only n_j (FA control) and clean d (X = inf).
Same g as Task H (X = 10log10(g^2 P_d / P_b)). Report detection (argmax, 3 seeds) split
by whether the drone model appears in that band's TRAIN split.
"""
import sys
from pathlib import Path
TASKH = Path(__file__).resolve().parents[1] / 'train_stage1_L20_20261108'
sys.path.insert(0, str(TASKH))
from common import *  # noqa: E402,F401  (np, torch, spec, meta, model, predict, CACHE, BANDS, SEEDS, rows, writecsv)

OUTD = Path(__file__).resolve().parent
XS = [-10, -5, 0, 5, 10, 15, 20]
split_rows = rows(TASKH.parents[0] / 'crop_slice_20261107' / 'group_split.csv')


def train_models(b):
    return {r['model'] for r in split_rows if r['band'] == b and r['split'] == 'train' and r['label'] == 'drone'}


def main():
    res = []
    rng = np.random.default_rng(20261108)
    for b in BANDS:
        seen = train_models(b)
        rs = meta(b, 'test')
        bg = [r for r in rs if r['dataset'] == 'v3' and r['label'] == 'background']
        pw = []
        old = None
        for r in bg:
            if r['block_file'] != old:
                blk = np.load(r['block_file'], mmap_mode='r'); old = r['block_file']
            lo = int(r['start']); z = np.array(blk[lo:lo + int(r['n_samples'])])
            pw.append(float(np.mean(np.abs(z) ** 2, dtype=np.float64)))
        n_samp = int(bg[0]['n_samples'])
        models = [model(b, s)[0] for s in SEEDS]

        def noise(j):
            return ((rng.standard_normal(n_samp) + 1j * rng.standard_normal(n_samp)) * np.sqrt(pw[j] / 2)).astype(np.complex64)

        # noise-only FA control (one noise realisation per background power)
        nz = np.stack([spec(noise(j), b) for j in range(len(bg))])
        for s, m in zip(SEEDS, models):
            p = predict(m, torch.from_numpy(nz).unsqueeze(1))
            res.append(dict(band=b, group='noise_only', seen='na', bgtype='wn', x_db='none', seed=s, n=len(bg), det=float((p.argmax(1) == 1).mean())))

        clean_all = np.load(cachepath(b, 'test'), mmap_mode='r')
        for ds in ['data', 'v3']:
            idx = [i for i, r in enumerate(rs) if r['dataset'] == ds and r['label'] == 'drone']
            src = [rs[i] for i in idx]
            is_seen = np.array([r['model'] in seen for r in src])
            wn = np.empty((len(src), len(XS), 256, 256), np.float16)
            old = None
            for i, r in enumerate(src):
                if r['block_file'] != old:
                    blk = np.load(r['block_file'], mmap_mode='r'); old = r['block_file']
                lo = int(r['start']); d = np.array(blk[lo:lo + int(r['n_samples'])])
                pd = float(np.mean(np.abs(d) ** 2, dtype=np.float64))
                j = i % len(bg)  # same pairing as Task H mix_test.py
                n = noise(j)
                for k, x in enumerate(XS):
                    g = float(np.sqrt(pw[j] / max(pd, 1e-30) * 10 ** (x / 10)))
                    wn[i, k] = spec(n + g * d, b)
                if i % 200 == 0:
                    print(b, ds, i, len(src), flush=True)
            v3mix = np.load(CACHE / f'{b}_mix_{ds}_drone.npy', mmap_mode='r')
            assert v3mix.shape[0] == len(src), (v3mix.shape, len(src))
            clean = np.ascontiguousarray(clean_all[idx])
            for s, m in zip(SEEDS, models):
                pc = predict(m, torch.from_numpy(clean).unsqueeze(1)).argmax(1) == 1
                for sv in [True, False]:
                    msk = is_seen == sv
                    if msk.sum():
                        res.append(dict(band=b, group=ds + '_drone', seen=sv, bgtype='clean', x_db='inf', seed=s, n=int(msk.sum()), det=float(pc[msk].mean())))
                for k, x in enumerate(XS):
                    for bt, arr in [('wn', wn[:, k]), ('v3bg', np.ascontiguousarray(v3mix[:, k]))]:
                        pr = predict(m, torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(1)).argmax(1) == 1
                        for sv in [True, False]:
                            msk = is_seen == sv
                            if msk.sum():
                                res.append(dict(band=b, group=ds + '_drone', seen=sv, bgtype=bt, x_db=x, seed=s, n=int(msk.sum()), det=float(pr[msk].mean())))
            del wn, v3mix, clean
        del models
        torch.cuda.empty_cache()
    writecsv(OUTD / 'ctrl_whitenoise_results.csv', res)
    # summary
    agg = collections.defaultdict(list)
    for r in res:
        agg[(r['band'], r['group'], str(r['seen']), r['bgtype'], str(r['x_db']), r['n'])].append(r['det'])
    out = [dict(band=k[0], group=k[1], seen=k[2], bgtype=k[3], x_db=k[4], n=k[5], det_mean=float(np.mean(v)), det_std=float(np.std(v, ddof=1))) for k, v in agg.items()]
    writecsv(OUTD / 'ctrl_whitenoise_summary.csv', out)
    for r in out:
        print(r)


if __name__ == '__main__':
    main()
