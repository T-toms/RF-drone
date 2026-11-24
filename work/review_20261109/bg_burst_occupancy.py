# Hypothesis check for Task K: the 2.4G round-1 model calls burst-free input "drone" because every 2.4G training
# background has bursts; the 5.8G model does not, because 5.8G training backgrounds include burst-free samples.
# Measure, per band and split, the fraction of background samples whose standardized spectrogram has
# almost no burst pixels (> +3), and the same stat for drone samples.
import csv, numpy as np, collections
H = 'E:/drone_lab_spec_L20_20261108/'
for band in ['2.4GHz', '5.8GHz']:
    for split in ['train', 'test']:
        m = list(csv.DictReader(open(H + f'{band}_{split}_spec_meta.csv', encoding='utf-8-sig')))
        a = np.load(H + f'{band}_{split}_spec.npy', mmap_mode='r')
        for lab in ['background', 'drone']:
            ids = [i for i, r in enumerate(m) if r['label'] == lab]
            if len(ids) > 600:
                ids = sorted(np.random.default_rng(0).choice(ids, 600, replace=False))
            occ = np.array([(a[i].astype(np.float32) > 3).mean() for i in ids])
            print(f'{band} {split:5s} {lab:10s} N={len(ids):4d}  burst-pixel frac median={np.median(occ)*100:.3f}%  '
                  f'samples with <0.05% burst pixels: {np.mean(occ < 5e-4)*100:5.1f}%')
        if split == 'train':
            bg = [(i, r['group_id']) for i, r in enumerate(m) if r['label'] == 'background']
            per = collections.defaultdict(list)
            for i, g in bg:
                per[g].append((a[i].astype(np.float32) > 3).mean())
            print('   per train bg session: ' + '; '.join(f'{g.split(":")[1][:15]} {np.median(v)*100:.3f}%' for g, v in sorted(per.items())))
