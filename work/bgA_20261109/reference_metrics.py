from pathlib import Path
import csv,json,numpy as np
from scipy.stats import beta
O=Path(__file__).resolve().parent;P=O.parents[1]
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def metric(rr,pred,invert=False):
 n=len(rr);k=sum((int(r[pred])==0 if invert else int(r[pred])==1) for r in rr);return dict(N=n,k=k,rate=k/n if n else None,cp95_upper=float(beta.ppf(.95,k+1,n-k)) if k<n else 1.)
out=[]
for band in ['2.4GHz','5.8GHz']:
 for seed in [2026,2027,2028]:
  rr=[r for r in rows(P/f'work/train_stage1_L20_20261108/test_predictions_{band}_seed{seed}.csv') if r['dataset']=='v3' and r['label']=='background']
  for rule in ['argmax','thr']:out.append(dict(source='v3_test_background',band=band,seed=seed,rule=rule,**metric(rr,'pred_'+rule)))
for name,label in [('quiet013_predictions.csv','quiet013_background'),('raw_033706_predictions.csv','raw033706_miss')]:
 rr=rows(O/name)
 for seed in [2026,2027,2028]:
  for rule in ['argmax','thr']:out.append(dict(source=label,seed=seed,rule=rule,**metric([r for r in rr if int(r['seed'])==seed],'pred_'+rule,label.endswith('miss'))))
(O/'reference_metrics.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(out)
