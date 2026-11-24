# 审核任务 L：v3 test 背景误报是哪些片段；v3 按型号与同型号召回（2026-11-10，支线）
import csv, collections, sys
from pathlib import Path
D = Path(__file__).resolve().parents[1] / 'side_L_wifimix_20261110'
rd = lambda p: list(csv.DictReader(open(D / p, encoding='utf-8-sig')))
c = collections.defaultdict(list)
for v in ['J', 'La', 'Lb']:
    for s in [2026, 2027, 2028]:
        f = f'J_test_predictions_2.4GHz_seed{s}.csv' if v == 'J' else f'test_predictions_2.4GHz_{v}_seed{s}.csv'
        for r in rd(f):
            if r['label'] == 'background' and r['pred_argmax'] == '1':
                c[(r['group_id'], r['sample_id'])].append(f"{v}{s}:{float(r['p_drone']):.2f}")
print('v3 test background false alarms (argmax):')
for k, v in sorted(c.items()):
    print('  ', k, v)
for name in ['v3_model_recall_paired.csv', 'same_model_recall_paired.csv', 'G_recall_paired.csv']:
    print('---', name)
    rs = rd(name)
    if not rs:
        continue
    keys = [k for k in rs[0] if not any(t in k for t in ('cp95', '_k', '_rate', '_std', '_n'))] + [k for k in rs[0] if k.endswith('2026_n')]
    for r in rs:
        if r.get('mode', 'argmax') == 'argmax':
            print('  ', {k: (f'{float(r[k]):.3f}' if k.endswith('_mean') else r[k]) for k in keys})
