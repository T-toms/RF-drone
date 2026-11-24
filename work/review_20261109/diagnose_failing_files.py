# 任务 W 逐文件漏检：未达标机型的 test 录音在各窗长、各种子的漏检 k/n（2026-11-11，支线）
import csv, collections
from pathlib import Path
P = Path(__file__).resolve().parents[1]
rs = list(csv.DictReader(open(P / 'main_W_window_20261110/miss_by_file.csv', encoding='utf-8-sig')))
print(collections.Counter(r['category'] for r in rs))
print('L 取值', sorted({r['L'] for r in rs}))
keys = {'Flip 待机10M': '002_2.4GHz_10M-signal__2026-09-20-19-24-45', 'air3s 飞行10M': 'F006_2.4GHz_10M',
        'atomse 飞行10M': 'F013_2.4GHz_10M', 'mavic2(v3)': '040617_mavic2', 'evo2pro_v3(v3)': '033510_evo2pro', 'mini3pro(v3)': '023828_mini3pro'}
d = collections.defaultdict(dict)
for r in rs:
    if r['band'] != '2.4GHz' or r['mode'] != 'argmax':
        continue
    for name, k in keys.items():
        if k in r['group']:
            d[(name, r['L'])][r['seed']] = f"{r['k']}/{r['n']}"
for name in keys:
    parts = [f"L{L}:" + ','.join(d[(name, L)].get(s, '-') for s in ['2026', '2027', '2028']) for L in ['10', '20', '30', '40', '50t'] if (name, L) in d]
    print(f'{name:16s}', '  '.join(parts))
