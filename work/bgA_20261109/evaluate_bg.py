from pathlib import Path
import sys,json,csv,importlib.util,collections
sys.dont_write_bytecode=True
import numpy as np,torch
O=Path(__file__).resolve().parent;P=O.parents[1];E=Path('E:/drone_lab_spec_bgA_L20_20261109');E.mkdir(exist_ok=True)
s=importlib.util.spec_from_file_location('common',P/'work/train_stage1_L20_20261108/common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
def prepare():
 rr=c.rows(O/'index_bgA_L20.csv');cache=E/'bgA_spec.npy';meta=E/'bgA_meta.csv'
 if not cache.exists():
  out=np.lib.format.open_memmap(cache.with_suffix('.partial.npy'),mode='w+',dtype=np.float16,shape=(len(rr),256,256));last=None
  for i,r in enumerate(rr):
   if r['block_file']!=last:z=np.load(r['block_file'],mmap_mode='r');last=r['block_file']
   j=int(r['start']);out[i]=c.spec(np.array(z[j:j+int(r['n_samples'])]),'2.4GHz')
   if i%200==0:print('SPEC',i,len(rr),flush=True)
  out.flush();del out;cache.with_suffix('.partial.npy').replace(cache);c.writecsv(meta,rr)
 assert np.load(cache,mmap_mode='r').shape==(len(rr),256,256);assert json.loads((E/'paired_ids.json').read_text(encoding='utf-8'))==[r['paired_sample_id'] for r in rr];c.writecsv(meta,rr);return rr,np.load(cache,mmap_mode='r')
def main():
 rr,a=prepare();source={r['sample_id']:r for r in c.rows(P/'work/crop_slice_20261107/index_L20.csv')};out=[];pairout=[]
 for seed in c.SEEDS:
  m,ck=c.model('2.4GHz',seed);pr=c.predict(m,torch.from_numpy(np.array(a)).unsqueeze(1))[:,1]
  for r,v in zip(rr,pr):
   src=source[r['paired_sample_id']];out.append(dict(**r,source_model=src['model'],source_state=src['state'],source_bw_label=src['bw_label'],seed=seed,p_drone=float(v),pred_argmax=int(v>.5),pred_thr=int(v>=ck['thr_fa1'])))
  # Reuse predictions for retained original samples in all splits, infer only unavailable paired rows.
  known={}
  for split in ['train','val','test']:
   mr=c.meta('2.4GHz',split);ar=np.load(c.cachepath('2.4GHz',split),mmap_mode='r');prob=c.predict(m,torch.from_numpy(np.array(ar)).unsqueeze(1))[:,1]
   known.update({r['sample_id']:float(v) for r,v in zip(mr,prob)})
  missing=[source[r['paired_sample_id']] for r in rr if r['paired_sample_id'] not in known];specs=[];last=None;mp=E/'paired_missing_spec.npy'
  if mp.exists():
   assert json.loads((E/'paired_missing_ids.json').read_text(encoding='utf-8'))==[r['sample_id'] for r in missing];specs=np.load(mp)
  else:
   for r in missing:
    if r['block_file']!=last:z=np.load(r['block_file'],mmap_mode='r');last=r['block_file']
    j=int(r['start']);specs.append(c.spec(np.array(z[j:j+int(r['n_samples'])]),'2.4GHz'))
   np.save(mp,np.array(specs));(E/'paired_missing_ids.json').write_text(json.dumps([r['sample_id'] for r in missing]),encoding='utf-8')
  if missing:
   pp=c.predict(m,torch.from_numpy(np.array(specs)).unsqueeze(1))[:,1];known.update({r['sample_id']:float(v) for r,v in zip(missing,pp)})
  for r in rr:
   src=source[r['paired_sample_id']];v=known[src['sample_id']];pairout.append(dict(**src,paired_bgA_id=r['sample_id'],bgA_keep=r['keep'],seed=seed,p_drone=v,pred_argmax=int(v>.5),pred_thr=int(v>=ck['thr_fa1'])))
  c.writecsv(O/'bgA_predictions.csv',out);c.writecsv(O/'paired_drone_predictions.csv',pairout);print('INFERENCE_SEED_DONE',seed,flush=True)
 print('FULL_INFERENCE_DONE',flush=True)
if __name__=='__main__':main()

