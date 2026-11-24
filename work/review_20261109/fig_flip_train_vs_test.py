# Flip（2.4G）训练录音 vs 测试录音的时频图对比：看测试录音为什么整段被漏（2026-11-11，支线）
# 读任务 M 的修补后缓存（L20），只读。每行一次录音，取 3 段；标题写任务 M 三个种子对该段的 p_drone（只 test 有）。
import csv, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'work/train_stage1_L20_20261108'))
from common import rows, image, plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
C = Path('E:/side_M_notchfill_20261110/H')
OLD = Path('E:/drone_lab_spec_L20_20261108')
M = ROOT / 'work/side_M_notchfill_20261110'
pred = {}
for s in [2026, 2027, 2028]:
    for r in rows(M / f'M_test_predictions_2.4GHz_seed{s}.csv') if (M / f'M_test_predictions_2.4GHz_seed{s}.csv').exists() else rows(next(M.glob(f'*test_predictions*M*seed{s}.csv'))):
        pred.setdefault(r['sample_id'], []).append(float(r['p_drone']))
groups = [('训练：待机 10M（combination 录音 16:04）', 'train', '002_2.4GHZ_10M_combination-signal__2026-09-20-16-04-54'),
          ('训练：待机 20M', 'train', '002_2.4GHz_20M-signal__2026-09-20-19-24-05'),
          ('训练：飞行 20M', 'train', 'F002_2.4GHz_20M-signal__2026-09-20-19-44-16'),
          ('测试：待机 10M（19:24）', 'test', '002_2.4GHz_10M-signal__2026-09-20-19-24-45')]
fig, axs = plt.subplots(len(groups), 3, figsize=(16, 3.8 * len(groups)))
for row, (title, split, key) in zip(axs, groups):
    meta = rows(C / f'2.4GHz_{split}_spec_meta.csv')
    a = np.load(C / f'2.4GHz_{split}_spec.npy', mmap_mode='r')
    ids = [i for i, r in enumerate(meta) if key in r['group_id'] and r['keep'] == 'True']
    pick = [ids[0], ids[len(ids) // 2], ids[-1]] if len(ids) >= 3 else ids
    for ax, i in zip(row, pick):
        sid = meta[i]['sample_id']
        p = pred.get(sid)
        extra = '' if p is None else '  p(无人机)=' + '/'.join(f'{v:.2f}' for v in p)
        image(ax, np.asarray(a[i], dtype=np.float32), '2.4GHz', f'{title}\n{sid}{extra}')
fig.suptitle('Flip 2.4G：训练录音与测试录音（修补后时频图，L=20 ms）；测试行标题为 3 次训练模型给出的无人机概率', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .97])
out = Path(__file__).resolve().parent / '图19_Flip训练与测试录音对比_2.4GHz.png'
fig.savefig(out, dpi=90); print(out)
