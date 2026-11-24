from pathlib import Path
import sys,json,importlib.util,csv
sys.dont_write_bytecode=True
import numpy as np, torch
O=Path(__file__).resolve().parent;P=O.parents[1]
s=importlib.util.spec_from_file_location('common',P/'work/train_stage1_L20_20261108/common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
def specs(requests,band):
 out=[]
 for r in requests:
  z=np.load(r['path'],mmap_mode='r');out.append(c.spec(np.array(z[r['start']:r['start']+r['n_samples']]),band))
 return torch.from_numpy(np.array(out)).unsqueeze(1)
def main():
 pairs=json.loads((O/'raw_pairs.json').read_text(encoding='utf-8'));control=json.loads((O/'raw_control.json').read_text(encoding='utf-8'));req=[]
 for r in pairs:
  req.extend([dict(path=r['raw'],start=r['raw_start'],n_samples=r['n_samples']),dict(path=r['original'],start=r['original_start'],n_samples=r['n_samples'])])
 x=specs(req,'5.8GHz');agreement=[]
 for seed in c.SEEDS:
  m,ck=c.model('5.8GHz',seed);pr=c.predict(m,x)[:,1];agreement.append(dict(seed=seed,N=len(pairs),argmax=float(((pr[::2]>.5)==(pr[1::2]>.5)).mean()),threshold=float(((pr[::2]>=ck['thr_fa1'])==(pr[1::2]>=ck['thr_fa1'])).mean()),thr_fa1=float(ck['thr_fa1'])))
 passed=all(r['spectrum_pass'] for r in control) and all(r['argmax']==1 and r['threshold']==1 for r in agreement);results=dict(control_agreement=agreement,control_pass=passed)
 if passed:
  plans=json.loads((O/'raw_plan.json').read_text(encoding='utf-8'));rr=[]
  for p in plans['20260709-033706_evo2pro_v3']:
   z=np.load(p,mmap_mode='r')
   for j in range(0,len(z)-2000000+1,2000000):rr.append(dict(path=p,start=j,n_samples=2000000,group_id='raw:20260709-033706_evo2pro_v3',split='test',keep=True))
  xx=specs(rr,'5.8GHz');out=[]
  for seed in c.SEEDS:
   m,ck=c.model('5.8GHz',seed);pr=c.predict(m,xx)[:,1]
   for r,pd in zip(rr,pr):out.append(dict(**r,seed=seed,p_drone=float(pd),pred_argmax=int(pd>.5),pred_thr=int(pd>=ck['thr_fa1'])))
  c.writecsv(O/'raw_033706_predictions.csv',out);results['N_033706']=len(rr)
 (O/'raw_inference.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(results,flush=True)
if __name__=='__main__':main()
