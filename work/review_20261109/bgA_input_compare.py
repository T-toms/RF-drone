# Why does the round-1 2.4G model call receiver-A background "drone" (100%) but receiver-B lab background "background" (0%)?
# Compare model inputs (standardized 256x256 dB spectrograms) across groups:
#   v3 test bg by session (07-09 urban, 07-10 lab), bgA test (mirror), and data drone test.
# Stats per sample: burst occupancy (pixels > +3 in standardized units), time-averaged frequency profile.
import csv, numpy as np, collections, json
H = 'E:/drone_lab_spec_L20_20261108/'
A = 'E:/drone_lab_spec_bgA_L20_20261109/'
def rows(p): return list(csv.DictReader(open(p, encoding='utf-8-sig')))
tm = rows(H + '2.4GHz_test_spec_meta.csv'); ta = np.load(H + '2.4GHz_test_spec.npy', mmap_mode='r')
bm = rows(A + 'bgA_meta.csv'); ba = np.load(A + 'bgA_spec.npy', mmap_mode='r')
groups = collections.defaultdict(list)
for i, r in enumerate(tm):
    if r['label'] == 'background':
        groups['v3bg ' + r['group_id'].split(':')[1][:8]].append(('t', i))
    elif r['dataset'] == 'data':
        groups['data drone'].append(('t', i))
    else:
        groups['v3 drone'].append(('t', i))
for i, r in enumerate(bm):
    if r['split'] == 'test' and r['keep'] == 'True':
        groups['bgA test'].append(('b', i))
rng = np.random.default_rng(0)
prof = {}
print('group | N | burst-occ(>+3) median | p90 | pixel std of per-row mean (freq structure) | row-mean range')
for g, ids in sorted(groups.items()):
    if len(ids) > 300: ids = [ids[k] for k in rng.choice(len(ids), 300, replace=False)]
    X = np.stack([(ta if s == 't' else ba)[i].astype(np.float32) for s, i in ids])
    occ = (X > 3).mean((1, 2))
    rowmean = X.mean(2)                # (N, 256 freq)
    prof[g] = rowmean.mean(0).tolist()
    print(f'{g:18s} | {len(ids):4d} | {np.median(occ):.4f} | {np.percentile(occ,90):.4f} | {np.median(rowmean.std(1)):.3f} | {np.median(np.ptp(rowmean,1)):.2f}')
json.dump(prof, open('work/review_20261109/bgA_input_profiles.json', 'w', encoding='utf-8'))
# profile difference between bgA and v3 lab bg (07-10) outside notches
f = np.linspace(2405, 2455, 256)
d = np.array(prof['bgA test']) - np.array(prof['v3bg 20260710'])
print('bgA - v3lab mean freq profile (standardized units), every 16 rows:')
print([(round(float(f[k]), 1), round(float(d[k]), 2)) for k in range(0, 256, 16)])
