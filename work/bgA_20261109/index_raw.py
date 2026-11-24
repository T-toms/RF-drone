from pathlib import Path
import json,csv,numpy as np
O=Path(__file__).resolve().parent;P=O.parents[1]
keys=next(csv.reader((P/'work/crop_slice_20261107/index_L20.csv').open(encoding='utf-8-sig')));rs=[]
for r in csv.DictReader((O/'raw_033706_predictions.csv').open(encoding='utf-8-sig')):
 if r['seed']!='2026':continue
 v={k:False for k in keys};v.update(block_file=r['path'],dataset='raw_recovered',band='5.8GHz',label='drone',model='evo2pro_v3',state='unknown',bw_label='unknown',group_id=r['group_id'],sample_id=f'raw033706_L20_{len(rs):08d}',start=int(r['start']),n_samples=int(r['n_samples']),split='test',keep=True);assert v['start']+v['n_samples']<=len(np.load(v['block_file'],mmap_mode='r'));rs.append(v)
assert len(rs)==25
with (O/'index_raw_033706_L20.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)
print('RAW_INDEX_OK',len(rs),len({r['block_file'] for r in rs}))
