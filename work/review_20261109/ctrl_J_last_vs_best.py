# Are the residual failures of Task J (seed2026 notched noise 100%, seed2027 quiet013 15/19) tied to the
# very early "best" checkpoints (epochs 2/10/4)? Compare best.pt vs last.pt (epoch 40) on the same inputs.
import sys, csv, numpy as np, torch
sys.path.insert(0, 'C:/Users/Administrator/Desktop/数据集处理/dronerfa_baseline')
sys.path.insert(0, 'C:/Users/Administrator/Desktop/数据集处理/work/train_stage1_L20_20261108')
from common import spec, predict, meta, cachepath, SEEDS
from model import ResNet18
D = 'E:/drone_lab_stage1_L20_bgA_20261109/'
A = 'E:/drone_lab_spec_bgA_L20_20261109/'
R = 'C:/Users/Administrator/Desktop/数据集处理/work/review_20261109/'
def load(p):
    ck = torch.load(p, map_location='cpu', weights_only=False)
    m = ResNet18(1, 2).cuda().to(memory_format=torch.channels_last); m.load_state_dict(ck['model']); m.eval(); return m, ck
tm = meta('2.4GHz', 'test'); ta = np.load(cachepath('2.4GHz', 'test'), mmap_mode='r')
bm = list(csv.DictReader(open(A + 'bgA_meta.csv', encoding='utf-8-sig'))); ba = np.load(A + 'bgA_spec.npy', mmap_mode='r')
G = {'evo lite 6k', 'evo ii pro v3'}
sets = {
    'v3 bg test': np.ascontiguousarray(ta[[i for i, r in enumerate(tm) if r['label'] == 'background']]),
    'bgA test': np.ascontiguousarray(ba[[i for i, r in enumerate(bm) if r['split'] == 'test' and r['keep'] == 'True']]),
    'notched noise': np.stack([spec(np.array(z), '2.4GHz') for z in np.load(R + 'notched_noise_2.4GHz.npy', mmap_mode='r')]),
    'data drone nonG': np.ascontiguousarray(ta[[i for i, r in enumerate(tm) if r['label'] == 'drone' and r['dataset'] == 'data' and r['model'] not in G]]),
    'v3 drone': np.ascontiguousarray(ta[[i for i, r in enumerate(tm) if r['label'] == 'drone' and r['dataset'] == 'v3']]),
}
q = list(csv.DictReader(open('C:/Users/Administrator/Desktop/数据集处理/work/train_stage1_L20_bgA_20261109/quiet013_metadata.csv', encoding='utf-8-sig')))
print('quiet013 meta columns:', list(q[0].keys()))
for s in SEEDS:
    for tag in ['best', 'last']:
        m, ck = load(D + f'2.4GHz_seed{s}_{tag}.pt')
        out = []
        for k, v in sets.items():
            p = predict(m, torch.from_numpy(v).unsqueeze(1))[:, 1]
            out.append(f'{k} {(p > .5).mean():.3f}')
        print(f'seed {s} {tag:4s} (epoch {ck["epoch"]:2d}) drone-rate argmax: ' + ' | '.join(out))
