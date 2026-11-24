"""Coordinator-side summary of B2b metrics_all.csv per pre-registered rules (read-only on worker outputs)."""
import csv, statistics as st, sys
from pathlib import Path
SRC = Path('C:/Users/Administrator/Desktop/数据集处理/work/fusionB_B2_20261112/metrics_all.csv')
rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
mode = sys.argv[1] if len(sys.argv) > 1 else 'argmax'
G = ['R', 'Z', 'F', 'F5']; S = ['2026', '2027', '2028']
idx = {}
for x in rows:
    if x['dependence'] != 'original' or x['mode'] != mode: continue
    if x['category'] not in ('main', 'receiver', 'v3_model_recall', 'G_recall'): continue
    idx[(x['band'], x['set'], x['subset'], x['group'], x['seed'])] = x
def get(b, s, sub, g, seed):
    x = idx.get((b, s, sub, g, seed)); return None if x is None else (float(x['rate']), x['k'], x['n'])
items = {
 '2.4GHz': [('test','v3_test_FA'),('test','v3_drone_recall'),('test','nonG_miss'),('test','data_nonG_recall'),('C1','C1_FA'),('bgA_test','bgA_test_FA'),('notched_noise','notched_noise_FA'),('white_noise','white_noise_FA'),('quiet013','quiet013_FA')],
 '5.8GHz': [('test','v3_test_FA'),('test','v3_drone_recall'),('test','nonG_miss'),('test','data_nonG_recall'),('notched_noise','notched_noise_FA'),('white_noise','white_noise_FA')],
}
for b, its in items.items():
    print(f'\n## {b} ({mode}) mean±sd % ; per seed')
    for s, sub in its:
        line = [f'{sub:18s}']
        for g in G:
            v = [get(b, s, sub, g, d) for d in S]
            if any(z is None for z in v): line.append(f'{g}: n/a'); continue
            r = [z[0]*100 for z in v]
            line.append(f'{g}: {st.mean(r):6.2f}±{st.stdev(r):5.2f} [' + ','.join(f'{z[1]}/{z[2]}' for z in v) + ']')
        print(' | '.join(line))
    print(f'-- paired diffs vs Z (pp), per seed 2026/2027/2028')
    for s, sub in its[:5]:
        for g in ['F', 'F5']:
            d = []
            for seed in S:
                a, z = get(b, s, sub, g, seed), get(b, s, sub, 'Z', seed)
                d.append(None if a is None or z is None else (a[0]-z[0])*100)
            if None in d: print(f'{sub} {g}-Z n/a'); continue
            sign = 'same-sign' if all(v > 0 for v in d) or all(v < 0 for v in d) else 'mixed'
            print(f'{sub:18s} {g}-Z: mean {st.mean(d):+6.2f}  seeds {[round(v,2) for v in d]}  {sign}')
