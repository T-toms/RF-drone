import os,sys,time,json,csv
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';sys.dont_write_bytecode=True
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent;P=O.parents[2];sys.path.insert(0,str(P/'work/fusionB_20261110'));sys.path.insert(0,str(O))
from extract import feature,NAMES,config
from generate import dump
FS=61440000;L=1228800;F=2430+(np.arange(1024)-512)*.06;MASK=config('2.4GHz')[2];WIN=np.hanning(1024)
FB=np.floor(F-2405).astype(int);BINS=[np.flatnonzero(MASK&(FB==i)) for i in range(50)]
CACHE=Path('E:/fusionB_features_20261110')
def getz(r):return np.array(np.load(r['block_file'],mmap_mode='r')[int(r['start']):int(r['start'])+int(r['n_samples'])])
def spectrum(z):
 z=np.asarray(z,dtype=np.complex128);z=z-z.mean();z=z/np.sqrt(np.mean(abs(z)**2));fr=z[:len(z)//1024*1024].reshape(-1,1024);return np.fft.fftshift(abs(np.fft.fft(fr*WIN,axis=1))**2,axes=1)
def cells(z):
 ps=spectrum(z);ts=np.floor((np.arange(len(ps))+.5)*1024/(FS*.001)).astype(int)
 q=np.stack([ps[:,ix].mean(1) for ix in BINS],axis=1)
 return np.stack([q[ts==j].mean(0) for j in range(20)])
def quiet(z):
 c=cells(z);exc=float(10*np.log10(c.max()/np.median(c)));return exc<=3,exc

def sliced(bs,dataset):
 out=[]
 for b in bs:
  n=len(np.load(b['block_file'],mmap_mode='r'))
  for st in range(0,n-L+1,L):out.append(dict(block_file=b['block_file'],dataset=dataset,band='2.4GHz',label='background',group_id=b['group_id'],nn=b.get('nn',str(b['k'])),sample_id=f'bgB_L20_{len(out):08d}' if dataset=='dataB_mirror' else f'B2440_{len(out):08d}',start=st,n_samples=L,keep=True))
 return out

def evaluate(rr,features=False):
 t=time.time();out=[];vals=[]
 def work(r):
  z=getz(r);q,ex=quiet(z);return dict(r,quiet=q,max_excess_db=ex),feature(z,'2.4GHz') if features else None
 with ThreadPoolExecutor(max_workers=4) as pool:
  for i,(r,v) in enumerate(pool.map(work,rr)):
   out.append(r)
   if features:vals.append(v)
   if i%250==0:print('EVAL',i,len(rr),round(time.time()-t,1),flush=True)
 return out,np.array(vals)
if __name__=='__main__':
 t=time.time();rng=np.random.default_rng(20261110);checks=[]
 for i in range(20):
  z=(rng.standard_normal(L)+1j*rng.standard_normal(L))/np.sqrt(2);q,ex=quiet(z);checks.append(dict(index=i,quiet=q,max_excess_db=ex))
 dump('white_check.json',checks)
 ba=pd.read_csv(CACHE/'bgA_meta_features_meta.csv',encoding='utf-8');ba=ba[ba.keep.astype(str)=='True'];assert len(ba)==5138,len(ba)
 xx=np.load(CACHE/'bgA_meta_features.npy');af=xx[ba.row_index.to_numpy(int)];np.save(O/'bgA_features.npy',af)
 differences=[]
 for i in rng.choice(len(ba),20,replace=False):differences.append(float(np.nanmax(abs(feature(getz(ba.iloc[i].to_dict()),'2.4GHz')-af[i]))))
 dump('feature_cache_check.json',dict(n=20,max_abs=max(differences)));assert max(differences)<=1e-9
 ar,_=evaluate(ba.to_dict('records'));pd.DataFrame(ar).to_csv(O/'bgA_index.csv',index=False,encoding='utf-8')
 bs=json.loads((O/'plan_bgB.json').read_text(encoding='utf-8'));br,bf=evaluate(sliced(bs,'dataB_mirror'),True);pd.DataFrame(br).to_csv(O/'bgB_index.csv',index=False,encoding='utf-8');np.save(O/'bgB_features.npy',bf)
 nq=sum(r['quiet'] for r in br);print('BGB_QUIET',nq,len(br),flush=True)
 if nq<10:raise RuntimeError('ASK_REQUIRED: bgB quiet < 10')
 plan=json.loads((P/'work/crop_slice_20261107/plan.json').read_text(encoding='utf-8'));ref=[b for b in plan if b['group_id']=='v3:20260715-075302_ambient@2440.0'];cr,cf=evaluate(sliced(ref,'v3'),True);pd.DataFrame(cr).to_csv(O/'B2440_index.csv',index=False,encoding='utf-8');np.save(O/'B2440_features.npy',cf)
 runs=json.loads((P/'work/bgA_20261109/quiet013_runs.json').read_text(encoding='utf-8'));rr=[]
 for run in runs:
  for j in range(run['start_ms'],run['end_ms']-19,20):rr.append(dict(block_file=run['block'],start=j*61440,n_samples=L,sample_id=f'A013_{len(rr):08d}',nn='013',group_id='A013'))
 assert len(rr)==19,len(rr)
 dr,df=evaluate(rr,True);pd.DataFrame(dr).to_csv(O/'A013_index.csv',index=False,encoding='utf-8');np.save(O/'A013_features.npy',df)
 dump('screen_runtime.json',dict(seconds=time.time()-t));print('SCREEN_DONE',flush=True)