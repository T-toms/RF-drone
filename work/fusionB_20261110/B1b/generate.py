import os,sys,time,json,csv,hashlib,copy
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE';sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;P=O.parents[2];sys.path.insert(0,str(O))
from process_copy import process
from chain_copy import chain,FS,R,io
E=Path('E:/fusionB_bgB_20261110');E.mkdir(exist_ok=True)
def dump(n,v): (O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def snapshot():
 d={}
 for name in ['crop_slice_20261107','bgA_20261109','fusionB_20261110']:
  for p in sorted((P/'work'/name).rglob('*')):
   if not p.is_file() or O in p.parents:continue
   h=hashlib.sha256()
   with p.open('rb') as f:
    for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
   d[str(p.relative_to(P))]=h.hexdigest()
 return d
if __name__=='__main__':
 t=time.time();dump('hash_before.json',snapshot());print('SNAPSHOT',flush=True)
 plan=json.loads((P/'work/crop_slice_20261107/plan.json').read_text(encoding='utf-8'))
 b=next(b for b in plan if b['group_id']=='v3:20260715-075302_ambient@2440.0')
 dest=process(b,E/'reproduction.npy');x=np.load(dest,mmap_mode='r');y=np.load(b['block_file'],mmap_mode='r');err=float(np.max(abs(x-y)));rel=float(np.max(abs(x-y)/np.maximum(abs(y),1e-30)))
 checks=dict(n_copy=len(x),n_original=len(y),max_abs=err,max_point_relative=rel);dump('implementation.json',checks);assert len(x)==len(y) and rel<1e-5,checks
 rng=np.random.default_rng(20261110);fs=100000000;n=1000000;mapping=[]
 for a in [2345,2370,2395]:
  raw=np.exp(2j*np.pi*(a-2367.0588)*1e6*np.arange(n)/fs)+.01*(rng.standard_normal(n)+1j*rng.standard_normal(n))
  z=chain(np.conj(raw),fs,2432.9412,'2.4GHz')[15000:-15000];freq=2430+np.fft.fftfreq(len(z),1/FS['2.4GHz'])[abs(np.fft.fft(z)).argmax()]/1e6
  mapping.append(dict(input_mhz=a,expected=4800-a,measured=float(freq),error=float(abs(freq-(4800-a)))))
 dump('mapping.json',mapping);assert max(r['error'] for r in mapping)<=.01,mapping
 print('CHECKS_PASS',checks,mapping,flush=True)
 excluded=list(csv.DictReader((P/'work/crop_slice_20261107/excluded_files.csv').open(encoding='utf-8-sig')));blocks=[]
 for nn in ['000032','000114','000196','000278','000360']:
  rr=sorted([r for r in excluded if '_2367.06M_' in r['path'] and f'_{nn}_' in r['path']],key=lambda r:r['path']);assert len(rr)==2
  parts=[]
  for r in rr:
   size=next(int(np.prod(shape)) for name,shape,dtype in io.whosmat(R/r['path']) if name=='IQ');parts.append(dict(path=r['path'],start=0,end=size))
  b=dict(dataset='v3',band='2.4GHz',label='background',model='none',state='unknown',bw_label='unknown',group_id='v3:20260715-075302_ambient@2367.06',parts=parts,fs_in=100000000,fc_in=2432.9412,n_in=sum(p['end'] for p in parts),source_start=0,k=len(blocks)+1,nn=nn)
  dest=E/(Path(parts[0]['path']).name.replace('_s0001_new.mat','')+'__bgB.npy');b['block_file']=str(dest);process(b,dest,True);blocks.append(b);dump('plan_bgB.json',blocks)
 dump('generation_runtime.json',dict(seconds=time.time()-t));print('GENERATION_DONE',flush=True)