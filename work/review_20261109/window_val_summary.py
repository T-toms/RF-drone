# 任务 W：只看验证集（被选中那一轮）的误报与漏检，按窗长汇总；用于“按验证集定窗长”，不看测试集（2026-11-11，支线）
import csv
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1] / 'main_W_window_20261110'
SRC = {('2.4GHz', '20'): (Path(__file__).resolve().parents[1] / 'side_M_notchfill_20261110', 'train_log_Lb_seed{}.csv'),
       ('5.8GHz', '20'): (Path(__file__).resolve().parents[1] / 'train_stage1_L20_20261108', 'train_log_5.8GHz_seed{}.csv')}
for b in ['2.4GHz', '5.8GHz']:
    print(b)
    for L in ['10', '20', '30', '40', '50t']:
        d, pat = SRC.get((b, L), (W / f'L{L}', 'train_log_Lb_seed{}.csv' if b == '2.4GHz' else 'train_log_5.8GHz_seed{}.csv'))
        rows = []
        for s in [2026, 2027, 2028]:
            p = d / pat.format(s)
            if not p.exists():
                rows = None; break
            r = list(csv.DictReader(open(p, encoding='utf-8-sig')))
            fa_key = 'val_fa_v3' if 'val_fa_v3' in r[0] else 'val_fa'
            sc = [(float(x[fa_key]) + float(x.get('val_fa_bgA', 0) or 0) + float(x['val_miss']), float(x['val_loss']), x) for x in r]
            best = min(sc, key=lambda t: (t[0], t[1]))[2]
            last = r[-1]
            rows.append((float(best[fa_key]), float(best['val_miss']), float(last[fa_key]), float(last['val_miss'])))
        if rows is None:
            print(f'  L{L}: 缺日志 {d / pat}'); continue
        a = np.array(rows) * 100
        print(f'  L{L:>3}: 选中轮 val误报 {a[:,0].mean():.2f}% 漏检 {a[:,1].mean():.2f}% (三项和 {a[:,0].mean()+a[:,1].mean():.2f}) | 最后一轮 误报 {a[:,2].mean():.2f}% 漏检 {a[:,3].mean():.2f}%')
