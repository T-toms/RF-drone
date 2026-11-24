# Flip 测试录音里被漏掉的段长什么样（任务 M，L20），以及被漏段与检出段的“小跳频块”数量对比（2026-11-11，支线）
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'work/train_stage1_L20_20261108'))
from common import rows, image, plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
C = Path('E:/side_M_notchfill_20261110/H')
M = ROOT / 'work/side_M_notchfill_20261110'
meta = rows(C / '2.4GHz_test_spec_meta.csv'); a = np.load(C / '2.4GHz_test_spec.npy', mmap_mode='r')
key = '002_2.4GHz_10M-signal__2026-09-20-19-24-45'
ids = [i for i, r in enumerate(meta) if key in r['group_id'] and r['keep'] == 'True']
P = {}
for s in [2026, 2027, 2028]:
    for r in rows(M / f'M_test_predictions_2.4GHz_seed{s}.csv'):
        P.setdefault(r['sample_id'], []).append(float(r['p_drone']))
sid = [meta[i]['sample_id'] for i in ids]
pm = np.array([np.mean(P[s]) for s in sid]); nmiss = np.array([sum(v < .5 for v in P[s]) for s in sid])
# 粗数“亮块”面积：像素 > 2.5 标准单位的比例
bright = np.array([(np.asarray(a[i], dtype=np.float32) > 2.5).mean() for i in ids])
print('Flip 测试段数', len(ids), ' 至少一个种子漏检的段', int((nmiss > 0).sum()), ' 三个种子都漏的段', int((nmiss == 3).sum()))
print('亮块面积占比  检出段中位 %.4f  被漏段中位 %.4f' % (np.median(bright[nmiss == 0]), np.median(bright[nmiss > 0])))
order = np.argsort(pm)
pick = list(order[:4]) + list(order[-2:])
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
for ax, k in zip(axs.flat, pick):
    i = ids[k]
    tag = '被漏' if nmiss[k] > 0 else '检出'
    image(ax, np.asarray(a[i], dtype=np.float32), '2.4GHz', f'{tag}：{sid[k]}  p=' + '/'.join(f'{v:.2f}' for v in P[sid[k]]) + f'  亮块 {bright[k]*100:.2f}%')
fig.suptitle('Flip 测试录音（待机 10M，19:24）：模型给出无人机概率最低的 4 段（被漏）与最高的 2 段（检出）', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .95])
out = Path(__file__).resolve().parent / '图19b_Flip测试录音被漏段_2.4GHz.png'
fig.savefig(out, dpi=90); print(out)
