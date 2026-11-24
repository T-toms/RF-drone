from pathlib import Path
import json,csv,numpy as np
O=Path(__file__).resolve().parent;P=O.parents[1];out=[]
for band in ['2.4GHz','5.8GHz']:
 for seed in [2026,2027,2028]:
  rs=list(csv.DictReader((P/f'work/train_stage1_L20_20261108/test_predictions_{band}_seed{seed}.csv').open(encoding='utf-8-sig')))
  def g(r):
   if r['dataset']=='data' and band=='2.4GHz':
    if '010_' in r['group_id']:return 'data_010_evo_lite_6k'
    if '011_' in r['group_id']:return 'data_011_evo_ii_pro_v3'
   if r['dataset']=='v3' and band=='5.8GHz':
    if '033809' in r['group_id']:return 'v3_033809_evo2pro_v3'
    if '034659' in r['group_id']:return 'v3_034659_fpv'
   return ''
  for group in ['all_drone','nonG_drone']+sorted({g(r) for r in rs if g(r)}):
   rr=[r for r in rs if r['label']=='drone' and (group=='all_drone' or (group=='nonG_drone' and not g(r)) or g(r)==group)]
   for rule in ['argmax','thr']:
    n=len(rr);k=sum(r['pred_'+rule]=='0' for r in rr);out.append(dict(band=band,seed=seed,group=group,rule=rule,N=n,miss=k,miss_rate=k/n if n else None))
(O/'generalization_existing.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
for band in ['2.4GHz','5.8GHz']:
 for group in sorted({r['group'] for r in out if r['band']==band}):
  for rule in ['argmax','thr']:
   rr=[r for r in out if r['band']==band and r['group']==group and r['rule']==rule];print(band,group,rule,rr[0]['N'],[r['miss'] for r in rr],np.mean([r['miss_rate'] for r in rr]),np.std([r['miss_rate'] for r in rr],ddof=1))
