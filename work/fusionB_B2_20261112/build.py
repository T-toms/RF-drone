from base import *
from preflight import rebuild,INDEX,BG,spec
from concurrent.futures import ThreadPoolExecutor
NAMES=['shape_factor','log_kurt_a','papr999_db','c42n','log_gamma_max','sigma_f','spec_flatness','occ_frac','duty','log1p_n_bursts']
COLS=[json.loads((B1/'feature_names.json').read_text()).index(n) for n in NAMES]
def size():return sum(p.stat().st_size for d in [CACHE,OUT] for p in d.rglob('*') if p.is_file())
def build(name,rs,getiq,features=None):
 if (CACHE/(name+'.done.json')).exists():return
 assert size()<20*1024**3,'disk budget 20GiB'
 a=np.lib.format.open_memmap(CACHE/(name+'_spec.npy'),mode='w+',dtype='float16',shape=(len(rs),256,256));f=np.empty((len(rs),10));pending=[];start=time.time()
 with ThreadPoolExecutor(max_workers=4) as pool:
  for i in sorted(range(len(rs)),key=lambda i:(rs[i].get('block_file',''),int(rs[i].get('start',0)))):
   r=rs[i];z=getiq(r);a[i]=spec(z,r['band'])
   if features is None:pending.append((i,pool.submit(feature,z,r['band'])))
   else:f[i]=features[i]
   if len(pending)>=8:
    for j,fu in pending:f[j]=fu.result()[COLS]
    pending=[]
   if i%200==0:print('CACHE',name,i,len(rs),round(time.time()-start,1),flush=True)
  for j,fu in pending:f[j]=fu.result()[COLS]
 a.flush();np.save(CACHE/(name+'_features.npy'),f);writecsv(CACHE/(name+'_meta.csv'),rs);dump(CACHE/(name+'.done.json'),dict(n=len(rs),seconds=time.time()-start,bytes=size()))
def main():
 assert json.loads((OUT/'preflight.json').read_text())['passed'];CACHE.mkdir(exist_ok=True);start=time.time();data=rebuild()
 for old,rs in data.items():
  ff=np.load(B1/(old+'_features.npy'),mmap_mode='r')
  if old=='bgA_meta':ff=ff[[i for i,r in enumerate(BG) if r['keep']=='True']]
  name=old.replace('_spec_meta','').replace('bgA_meta','bgA');build(name,rs,iq,ff[:,COLS])
 lookup={r['sample_id']:r for r in INDEX+BG}
 def mix(r):return iq(lookup[r['background_sample_id']])+float(r['g'])*iq(lookup[r['sample_id']])
 for kind in ['drone','bg']:
  rs=[dict(lookup[r['sample_id']],**r) for r in rows(ROOT/f'work/side_L_wifimix_20261110/mix_train_{kind}.pairs.csv')];build('2.4GHz_mix_train_'+kind,rs,mix)
 pairs=rows(ROOT/'work/train_stage1_L20_20261108/mix_pairs.csv')
 for b in ['2.4GHz','5.8GHz']:
  rs=[dict(lookup[r['sample_id']],**r) for r in pairs if r['band']==b];build(b+'_mix_test',rs,mix)
 rs=rows(ROOT/'work/side_L_wifimix_20261110/C1.pairs.csv');build('2.4GHz_C1',rs,mix)
 for b in ['2.4GHz','5.8GHz']:
  if b=='2.4GHz':
   rng=np.random.default_rng(20261109)
   def white(r):return (rng.standard_normal(1228800)+1j*rng.standard_normal(1228800)).astype(np.complex64)
   no=np.load(ROOT/'work/review_20261109/notched_noise_2.4GHz.npy',mmap_mode='r')
   def notched(r):return np.array(no[int(r['i'])])
  else:
   sys.path.insert(0,str(ROOT/'work/side_5p8_controls_20261109'));from synthesis import generate
   def white(r):return generate(b,'a',int(r['i']))
   def notched(r):return generate(b,'b',int(r['i']))
  rs=[dict(band=b,label='background',dataset='control',sample_id=f'{b}_noise_{i}',i=i) for i in range(80)]
  build(b+'_white_noise',rs,white);build(b+'_notched_noise',rs,notched)
 plan={r['block_file']:r for r in json.loads((ROOT/'work/crop_slice_20261107/plan.json').read_text(encoding='utf-8'))};runs=json.loads((ROOT/'work/bgA_20261109/quiet013_runs.json').read_text(encoding='utf-8'));qr=[]
 for run in runs:
  for j in range(run['start_ms'],run['end_ms']-19,20):qr.append(dict(plan[run['block']],block_file=run['block'],start=str(j*61440),n_samples='1228800',sample_id=f'quiet013_{len(qr)}',band='2.4GHz',label='background'))
 assert len(qr)==19;build('2.4GHz_quiet013',qr,iq);dump(OUT/'cache_done.json',dict(seconds=time.time()-start,bytes=size()))
if __name__=='__main__':main()
