# 看当前 2.4G 基线（任务 M）每个种子每个 epoch 的 val 指标，说明“选模型”是怎么选出来的（2026-11-11，支线）
import csv, glob
from pathlib import Path
D = Path(__file__).resolve().parents[1] / 'side_M_notchfill_20261110'
for f in sorted(D.glob('train_log_*seed*.csv')):
    if 'smoke' in f.name:
        continue
    r = list(csv.DictReader(open(f, encoding='utf-8-sig')))
    score = [(float(x['val_fa_v3']) + float(x['val_fa_bgA']) + float(x['val_miss']), float(x['val_loss']), int(x['epoch'])) for x in r]
    best = min(score)[2]
    print(f.name, 'epochs', len(r), 'best', best)
    for x in r:
        e = int(x['epoch'])
        s = float(x['val_fa_v3']) + float(x['val_fa_bgA']) + float(x['val_miss'])
        mark = ' <-- 选中' if e == best else ''
        print(f'  ep{e:2d} v3背景误报={float(x["val_fa_v3"]):.4f} bgA误报={float(x["val_fa_bgA"]):.4f} 漏检={float(x["val_miss"]):.4f} 三项和={s:.4f} val_loss={float(x["val_loss"]):.4f}{mark}')
