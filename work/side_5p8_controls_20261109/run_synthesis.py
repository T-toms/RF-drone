from pathlib import Path
import sys,os,json,time,subprocess,concurrent.futures
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from synthesis import *
TORCH=P/'.venv/Scripts/python.exe'
def task(args):
 band,kind,i,cor=args
 return generate(band,kind,i,cor)
def estimate_task(args):
 band,kind,i,cor=args
 return np.percentile(cells(generate(band,kind,i,cor),band),10,axis=0)
def masks(band):
 f=shapes[band+'_A_freq']; pb=abs(f-FC[band])<=(25 if band=='2.4GHz' else 45); good=np.ones(len(f),bool)
 for r in notches:
  if r['band']==band:good &= abs(f-float(r['freq_mhz']))>.3
 return pb,good
def metrics(band,re,kind):
 pb,good=masks(band); d=re-shapes[band+'_'+kind]
 return dict(max_passband_db=float(abs(d[pb&good]).max()),max_fullgrid_db=float(abs(d[good]).max()),max_outside_db=float(abs(d[(~pb)&good]).max()))
if __name__=='__main__':
 start=time.time(); os.environ['PYTHONDONTWRITEBYTECODE']='1'; cal=[]; corrections={}
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for band in FS:
   for kind in ['A','B']:
    key=band+'_'+kind; cor=np.zeros_like(shapes[key]); target=shapes[key];pb,good=masks(band)
    for iteration in range(4):
     ar=list(pool.map(estimate_task,[(band,kind,i+50000,cor) for i in range(40)])); re=np.median(ar,axis=0); mt=metrics(band,re,kind)
     cal.append(dict(key=key,iteration=iteration,**mt));print('calibration',cal[-1],flush=True)
     if mt['max_passband_db']<=.3 or iteration==3:break
     delta=target-re;cor[pb&good]+=delta[pb&good]
     # Fill masked notch bins only; preserve target notch-free input spectrum.
     cor[pb&~good]=np.interp(shapes[key+'_freq'][pb&~good],shapes[key+'_freq'][pb&good],cor[pb&good])
    corrections[key]=cor
  np.savez(O/'corrections.npz',**corrections);(O/'calibration.json').write_text(json.dumps(cal,indent=2),encoding='utf-8')
  final=[]
  for band in FS:
   for kind in ['a','b','A','B']:
    cor=corrections.get(band+'_'+kind);proc=subprocess.Popen([str(TORCH),'-I','-B',str(O/'torch_stage.py'),'stream',band,kind],stdin=subprocess.PIPE)
    estimates=[]
    for first in range(0,200,8):
     for x in pool.map(task,[(band,kind,i,cor) for i in range(first,first+8)]):
      if kind in ['A','B']:estimates.append(np.percentile(cells(x,band),10,axis=0))
      proc.stdin.write(x.tobytes())
     if first%40==0:print('generated',band,kind,first+8,flush=True)
    proc.stdin.close();rc=proc.wait();assert rc==0,rc
    if estimates:
     re=np.median(estimates,axis=0);final.append(dict(key=band+'_'+kind,**metrics(band,re,kind)));np.save(O/f'reestimated_{band}_{kind}.npy',re)
     print('FINAL',final[-1],flush=True)
    (O/'final_validation.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
 (O/'synthesis_runtime.json').write_text(json.dumps(dict(seconds=time.time()-start,workers=4,seed_formula='2026110900 + band_index*100000 + kind_index*1000 + segment_index; calibration index=50000..50039; final=0..199')),encoding='utf-8')
