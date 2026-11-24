# 画出任务 L 中反复误报的 v3 test 背景片段，以及同会话的未误报片段作对比（2026-11-10，支线）
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'work/train_stage1_L20_20261108'))
from common import rows, image, plt, np  # 只读：不调用 model()，不写 CACHE
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
CACHE = Path('E:/drone_lab_spec_L20_20261108')
r = rows(CACHE / '2.4GHz_test_spec_meta.csv')
a = np.load(CACHE / '2.4GHz_test_spec.npy', mmap_mode='r')
idx = {x['sample_id']: i for i, x in enumerate(r)}
picks = [('L20_00009902', '5/6 次 L 训练误报'), ('L20_00009607', '3/6 次 L 训练误报'),
         ('L20_00009949', 'Lb2026 误报'), ('L20_00009940', 'J2026 误报')]
same = [x['sample_id'] for x in r if x['group_id'] == r[idx['L20_00009902']]['group_id'] and x['sample_id'] != 'L20_00009902'][:2]
picks += [(s, '同会话未误报') for s in same]
fig, axs = plt.subplots(2, 3, figsize=(18, 9))
for ax, (s, t) in zip(axs.flat, picks):
    image(ax, np.asarray(a[idx[s]], dtype=np.float32), '2.4GHz', f"{s} {r[idx[s]]['group_id'][3:]}\n{t}")
fig.tight_layout()
out = Path(__file__).resolve().parent / 'review_L_fa_slices.png'
fig.savefig(out, dpi=110)
print(out)
