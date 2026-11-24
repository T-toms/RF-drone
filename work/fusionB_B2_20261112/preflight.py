from base import *
sys.path.insert(0,str(ROOT/'work/train_stage1_L20_20261108'))
from common import spec
INDEX=rows(ROOT/'work/crop_slice_20261107/index_L20.csv')
BG=rows(ROOT/'work/bgA_20261109/index_bgA_L20.csv')
def rebuild():
 data={}
 for b in ['2.4GHz','5.8GHz']:
  for split in ['train','val','test','rejected_test']:
   rr=[dict(r) for r in INDEX if r['band']==b and r['split']==('test' if split=='rejected_test' else split) and (r['keep']!='True' and r['label']=='drone' if split=='rejected_test' else r['keep']=='True')]
   rr.sort(key=lambda r:(r['block_file'],int(r['start'])));data[f'{b}_{split}_spec_meta']=rr
 data['bgA_meta']=[dict(r) for r in BG if r['keep']=='True']
 return data
if __name__=='__main__':
 start=time.time();dump(OUT/'protected_before.json',hashes());data=rebuild();checks={'alignment':{},'features':[],'spec':[],'mix':[]}
 for name,rs in data.items():
  old=rows(B1/(name+'_features_meta.csv'))
  original_indices=np.arange(len(old))
  if name=='bgA_meta':
   assert [r['sample_id'] for r in BG]==[r['sample_id'] for r in old]
   original_indices=np.array([i for i,r in enumerate(old) if r['keep']=='True']);old=[old[i] for i in original_indices]
  assert [r['sample_id'] for r in rs]==[r['sample_id'] for r in old],('alignment',name,len(rs),len(old))
  checks['alignment'][name]=len(rs);writecsv(OUT/(name+'.csv'),rs)
  ff=np.load(B1/(name+'_features.npy'),mmap_mode='r')[original_indices];rng=np.random.default_rng(20261112)
  for i in rng.choice(len(rs),3,replace=False):
   r=rs[i];z=iq(r);f=feature(z,r['band']);delta=float(np.nanmax(abs(f-ff[i])));assert np.array_equal(np.isnan(f),np.isnan(ff[i])) and delta<=1e-9,('feature',name,i,delta)
   checks['features'].append(dict(name=name,row=int(i),sample_id=r['sample_id'],max_abs=delta))
   if len(checks['spec'])<3:
    eq=np.array_equal(spec(z,r['band']),spec(z,r['band']));assert eq;checks['spec'].append(eq)
 lookup={r['sample_id']:r for r in INDEX+BG}
 for name in ['side_L_wifimix_20261110/mix_train_drone.pairs.csv','side_L_wifimix_20261110/mix_train_bg.pairs.csv','train_stage1_L20_20261108/mix_pairs.csv']:
  for r in rows(ROOT/'work'/name)[:3]:
   d=iq(lookup[r['sample_id']]);b=iq(lookup[r['background_sample_id']]);pd=float(np.mean(abs(d)**2,dtype=np.float64));pb=float(np.mean(abs(b)**2,dtype=np.float64));actual=10*np.log10(float(r['g'])**2*pd/pb);expected=float(r.get('actual_power_ratio_db',r.get('achieved_ratio_db')));delta=abs(actual-expected);assert delta<=1e-6,('mix',name,delta);checks['mix'].append(dict(name=name,delta_db=delta))
 tr=data['2.4GHz_train_spec_meta'];counts=[sum(r['dataset']==ds and r['label']==lab for r in tr) for ds,lab in [('data','drone'),('v3','drone'),('v3','background')]]+[sum(r['split']==s for r in data['bgA_meta']) for s in ['train','val','test']]
 assert counts==[2832,152,234,2861,990,1287],counts
 for name,n in [('drone',2832),('bg',2861)]:assert len(rows(ROOT/f'work/side_L_wifimix_20261110/mix_train_{name}.pairs.csv'))==n
 checks['counts_2p4']=counts;checks['seconds']=time.time()-start;checks['passed']=True;dump(OUT/'preflight.json',checks);print(json.dumps(checks),flush=True)
