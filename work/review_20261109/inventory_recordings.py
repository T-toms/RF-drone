# Full inventory: independent recordings (= Task F group_id) per source x band x model x state x bw,
# with seconds and split, plus raw/ data/ directory scan for codes with no data at all.
import csv, re, os, collections, glob
P = 'C:/Users/Administrator/Desktop/数据集处理/work/crop_slice_20261107/'
gs = list(csv.DictReader(open(P + 'group_split.csv', encoding='utf-8-sig')))
blocks = list(csv.DictReader(open(P + 'blocks.csv', encoding='utf-8-sig')))
meta = {}
for b in blocks:
    meta.setdefault(b['group_id'], (b['state'], b['bw_label']))
MODELS = ['mini4 pro', 'Flip', 'mini 4k', 'neo', 'neo2', 'air 3s', 'mini5 pro', 'mavic 4', 'avata2', 'evo lite 6k', 'evo ii pro v3', 'avata', 'atom se', 'atom']
# 1) per (dataset, band, model): number of groups, split of each, seconds
agg = collections.defaultdict(list)
for r in gs:
    if r['label'] != 'drone': continue
    st, bw = meta.get(r['group_id'], ('?', '?'))
    agg[(r['dataset'], r['band'], r['model'])].append((r['split'], st, bw, float(r['n_seconds'])))
print('dataset band model | groups | train/val/test groups | states | seconds')
for k in sorted(agg):
    v = agg[k]; c = collections.Counter(s for s, *_ in v)
    flag = '  <-- only 1 group' if len(v) == 1 else ('  <-- no train' if c['train'] == 0 else ('  <-- no val' if c['val'] == 0 else ''))
    print(f'{k[0]:4s} {k[1]:6s} {k[2]:14s} | {len(v):2d} | {c["train"]}/{c["val"]}/{c["test"]} | ' +
          ','.join(sorted({f"{st}:{bw}" for _, st, bw, _ in v})) + f' | {sum(x[3] for x in v):.2f}{flag}')
# 2) per (dataset, band, model, state): groups -- does each state exist in train?
print('\nmodel x state with that state absent from train:')
st_agg = collections.defaultdict(collections.Counter)
for k, v in agg.items():
    for s, st, bw, _ in v:
        st_agg[(k[0], k[1], k[2], st)][s] += 1
for k in sorted(st_agg):
    if st_agg[k]['train'] == 0:
        print('  ', k, dict(st_agg[k]))
# 3) data/ codes with no directory at all
d = 'G:/drone/data'
codes = collections.Counter()
for p in glob.glob(d + '/*/*/*'):
    m = re.match(r'^F?(\d{3})_(\d\.\dGHz)', os.path.basename(p), re.I)
    if m: codes[(int(m.group(1)), m.group(2).upper())] += 1
print('\ndata/ codes present per band:', sorted(codes))
for i, name in enumerate(MODELS, 1):
    bands = [b for (c, b) in codes if c == i]
    if not bands: print(f'  code {i:03d} {name}: NO DIRECTORY IN data/')
    elif len(bands) == 1: print(f'  code {i:03d} {name}: only {bands[0]}')
