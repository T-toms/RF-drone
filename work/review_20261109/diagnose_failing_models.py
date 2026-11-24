# 查 2.4G 未达标机型：train/val/test 各是哪次录音、什么状态/带宽，test 漏检集中在哪里（2026-11-11，支线）
import csv, collections
from pathlib import Path
P = Path(__file__).resolve().parents[1]
rd = lambda p: list(csv.DictReader(open(p, encoding='utf-8-sig')))
idx = rd(P / 'crop_slice_20261107/index_L20.csv')
MODELS = ['Flip', 'air 3s', 'atom se', 'mavic2', 'evo2pro_v3', 'mini3pro']
print('== L20 2.4G 各机型的同源组：split / 状态 / 带宽 / keep 片数')
g = collections.defaultdict(lambda: collections.Counter())
meta = {}
for r in idx:
    if r['band'] == '2.4GHz' and r['model'] in MODELS and r['keep'] == 'True':
        g[r['group_id']][r['split']] += 1
        meta[r['group_id']] = (r['dataset'], r['model'], r['state'], r['bw_label'])
for gid in sorted(g, key=lambda k: (meta[k][1], meta[k][0])):
    ds, m, st, bw = meta[gid]
    print(f'  {m:11s} {ds:4s} {st:8s} {bw:7s} {dict(g[gid])}  {gid[-45:]}')
print('\n== 任务 W 逐文件漏检（2.4G，argmax，三种子均值），只列这些机型')
f = P / 'main_W_window_20261110/miss_by_file.csv'
rs = rd(f)
print('  列名:', list(rs[0].keys())[:14])
for r in rs:
    if r.get('band') == '2.4GHz' and r.get('model') in MODELS and r.get('mode', r.get('decision')) == 'argmax':
        keys = [k for k in r if k.endswith('mean')] or [k for k in r if 'miss' in k]
        print('  ', r.get('L', r.get('window')), r['model'], r.get('dataset'), r.get('state'), r.get('bw_label'), r.get('n', ''), {k: r[k][:6] for k in keys})
