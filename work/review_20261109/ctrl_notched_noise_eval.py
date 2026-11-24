# Evaluate round-1 2.4G models (3 seeds) on:
#  (1) plain white noise (no notch)         -- repeats the Task-H review control
#  (2) white noise + Task F FIR + notches   -- "empty band with notch dips"
#  (3) receiver-B noise_adrv_2450 (TRAIN split; seen in training, weak evidence)
#  (4) bgA test (receiver A mirror), and 013 real-quiet reference: numbers taken from Task I for comparison
import sys, csv, numpy as np, torch
from pathlib import Path
sys.path.insert(0, 'C:/Users/Administrator/Desktop/数据集处理/work/train_stage1_L20_20261108')
from common import spec, model, predict, meta, cachepath, SEEDS
R = Path('C:/Users/Administrator/Desktop/数据集处理/work/review_20261109')
nn_ = np.load(R / 'notched_noise_2.4GHz.npy', mmap_mode='r')
rng = np.random.default_rng(7)
wn = [((rng.standard_normal(nn_.shape[1]) + 1j * rng.standard_normal(nn_.shape[1])) / np.sqrt(2)).astype(np.complex64) for _ in range(nn_.shape[0])]
X = {'white noise (no notch)': np.stack([spec(z, '2.4GHz') for z in wn]),
     'white noise + FIR + notches': np.stack([spec(np.array(z), '2.4GHz') for z in nn_])}
tr = meta('2.4GHz', 'train'); ta = np.load(cachepath('2.4GHz', 'train'), mmap_mode='r')
ix = [i for i, r in enumerate(tr) if 'noise_adrv' in r['group_id']]
X['receiver-B noise_adrv_2450 (train)'] = np.ascontiguousarray(ta[ix])
for k, v in X.items():
    print(f'{k:38s} N={len(v):3d} burst-occ(>+3) median={np.median((v.astype(np.float32) > 3).mean((1, 2))):.4f} row-mean range={np.median(np.ptp(v.astype(np.float32).mean(2), 1)):.2f}')
rows = []
for s in SEEDS:
    m, c = model('2.4GHz', s)
    for k, v in X.items():
        p = predict(m, torch.from_numpy(v.astype(np.float16)).unsqueeze(1))[:, 1]
        rows.append((k, s, float((p > .5).mean()), float((p >= c['thr_fa1']).mean()), float(np.median(p))))
for r in rows:
    print(f'{r[0]:38s} seed {r[1]}  drone-rate argmax {r[2]:.3f}  thr_fa1 {r[3]:.3f}  median p_drone {r[4]:.3f}')
with open(R / 'ctrl_notched_noise_results.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['input', 'seed', 'drone_rate_argmax', 'drone_rate_thr', 'median_p']); w.writerows(rows)
