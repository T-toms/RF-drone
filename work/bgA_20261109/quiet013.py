from pathlib import Path
import sys,json,csv,importlib.util
sys.dont_write_bytecode=True
import numpy as np,torch
O=Path(__file__).resolve().parent;P=O.parents[1]
s=importlib.util.spec_from_file_location('common',P/'work/train_stage1_L20_20261108/common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
bs=c.load(P/'work/crop_slice_20261107/plan.json');floors={r['group_id']:r['floor_db'] for r in c.load(P/'work/crop_slice_20261107/recording_features.json')};freq=2430+(np.arange(512)-256)*.12;sel=(freq>=2405)&(freq<=2455)
for r in c.rows(P/'work/crop_slice_20261107/notch_table.csv'):
 if r['band']=='2.4GHz':sel &=abs(freq-float(r['freq_mhz']))>.3
req=[];runs=[]
for b in bs:
 if b['dataset']!='data' or b['band']!='2.4GHz' or not (b['group_id'].startswith('data:013_') or b['group_id'].startswith('data:F013_')):continue
 mean=np.load(P/'work/crop_slice_20261107/cells'/Path(b['block_file']).stem/'mean.npy');quiet=np.all(mean[:,sel]<floors[b['group_id']]+3,axis=1);e=np.diff(np.r_[False,quiet,False].astype(int))
 for a,z in zip(np.flatnonzero(e==1),np.flatnonzero(e==-1)):
  if z-a<20:continue
  runs.append(dict(block=b['block_file'],start_ms=int(a),end_ms=int(z)))
  for j in range(a,z-19,20):req.append(dict(block_file=b['block_file'],start=int(j)*61440,n_samples=1228800,group_id=b['group_id']))
c.dump(O/'quiet013_runs.json',runs);out=[]
if req:
 specs=[]
 for r in req:
  z=np.load(r['block_file'],mmap_mode='r');specs.append(c.spec(np.array(z[r['start']:r['start']+r['n_samples']]),'2.4GHz'))
 x=torch.from_numpy(np.array(specs)).unsqueeze(1)
 for seed in c.SEEDS:
  m,ck=c.model('2.4GHz',seed);pr=c.predict(m,x)[:,1]
  for r,pd in zip(req,pr):out.append(dict(**r,seed=seed,p_drone=float(pd),pred_argmax=int(pd>.5),pred_thr=int(pd>=ck['thr_fa1'])))
c.writecsv(O/'quiet013_predictions.csv',out);print('QUIET013_N',len(req),runs,flush=True)

