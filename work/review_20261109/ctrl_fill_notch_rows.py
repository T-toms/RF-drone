# Falsification test of "notch dips in an otherwise empty band => model says drone".
# Replace the spectrogram rows around the 3 notch frequencies by linear interpolation of neighbours,
# re-standardize, and re-run the round-1 models. If the claim is right, FA on bgA / notched noise drops sharply.
# Also report drone recall and v3-bg FA on the same edit (to see that the edit does not simply push everything to bg).
import sys, csv, numpy as np, torch
sys.path.insert(0, 'C:/Users/Administrator/Desktop/数据集处理/work/train_stage1_L20_20261108')
from common import spec, model, predict, meta, cachepath, SEEDS
R = 'C:/Users/Administrator/Desktop/数据集处理/work/review_20261109/'
A = 'E:/drone_lab_spec_bgA_L20_20261109/'
# row frequencies: Task H keeps STFT bins in 2405..2455 then adaptive-avg-pools to 256 rows
FS = 61440000; fr = np.fft.fftshift(np.fft.fftfreq(1024, 1 / FS)) / 1e6 + 2430; fr = fr[(fr >= 2405) & (fr <= 2455)]
edges = np.floor(np.arange(256) * len(fr) / 256).astype(int), np.ceil((np.arange(256) + 1) * len(fr) / 256).astype(int)
rowf = np.array([fr[a:b].mean() for a, b in zip(*edges)])
bad = np.zeros(256, bool)
for n in [2426.88, 2440.0, 2450.0]:
    bad |= np.abs(rowf - n) <= 0.45
good = np.where(~bad)[0]
print('rows filled:', int(bad.sum()), [round(float(x), 2) for x in rowf[bad]])

def fill(X):
    X = X.astype(np.float32).copy()
    for i in range(len(X)):
        for t in range(256):
            pass
    # vectorized: interpolate along freq for every column
    Y = X.transpose(0, 2, 1)                      # N, time, freq
    Y[:, :, bad] = np.stack([np.stack([np.interp(np.where(bad)[0], good, col[good]) for col in img]) for img in Y])
    X = Y.transpose(0, 2, 1)
    m = X.mean((1, 2), keepdims=True); s = X.std((1, 2), keepdims=True)
    return ((X - m) / np.maximum(s, 1e-8)).astype(np.float16)

rng = np.random.default_rng(1)
def pick(a, ids, n=200):
    ids = list(ids)
    if len(ids) > n: ids = sorted(rng.choice(ids, n, replace=False))
    return np.ascontiguousarray(a[ids])
tm = meta('2.4GHz', 'test'); ta = np.load(cachepath('2.4GHz', 'test'), mmap_mode='r')
bm = list(csv.DictReader(open(A + 'bgA_meta.csv', encoding='utf-8-sig'))); ba = np.load(A + 'bgA_spec.npy', mmap_mode='r')
nn_ = np.load(R + 'notched_noise_2.4GHz.npy', mmap_mode='r')
sets = {
    'bgA test (receiver A mirror)': pick(ba, [i for i, r in enumerate(bm) if r['split'] == 'test' and r['keep'] == 'True']),
    'white noise + notches': np.stack([spec(np.array(z), '2.4GHz') for z in nn_]),
    'v3 bg test': pick(ta, [i for i, r in enumerate(tm) if r['label'] == 'background']),
    'data drone test': pick(ta, [i for i, r in enumerate(tm) if r['label'] == 'drone' and r['dataset'] == 'data']),
    'v3 drone test': pick(ta, [i for i, r in enumerate(tm) if r['label'] == 'drone' and r['dataset'] == 'v3']),
}
out = []
for s in SEEDS:
    m, c = model('2.4GHz', s)
    for k, v in sets.items():
        for tag, x in [('original', v), ('notch rows filled', fill(v))]:
            p = predict(m, torch.from_numpy(x).unsqueeze(1))[:, 1]
            out.append((k, tag, s, len(x), float((p > .5).mean())))
for k in sets:
    for tag in ['original', 'notch rows filled']:
        v = [r[4] for r in out if r[0] == k and r[1] == tag]
        print(f'{k:30s} {tag:18s} N={[r[3] for r in out if r[0]==k][0]:3d}  drone-rate per seed {[round(x,3) for x in v]}')
with open(R + 'ctrl_fill_notch_results.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['input', 'edit', 'seed', 'n', 'drone_rate_argmax']); w.writerows(out)
