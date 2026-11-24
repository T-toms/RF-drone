from pathlib import Path
import sys,json
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parent))
import shape_final as sh
import numpy as np
O=sh.O;E=sh.E

def shape(b,s):
 stem=Path(b['block_file']).stem;path=E/'2.4GHz'/f'{stem}__bgA_s{s}.npy';cache=O/'shape_cache';cache.mkdir(exist_ok=True);cp=cache/(path.stem+'_cells.npy')
 if cp.exists():bg=np.load(cp)
 else:
  z=np.load(path,mmap_mode='r');nc=len(z)//61440;bg=np.empty((nc,512),np.float32);w=np.hanning(512).astype(np.float32)
  for j in range(nc):
   x=np.asarray(z[j*61440:(j+1)*61440])[:120*512].reshape(120,512);p=(abs(np.fft.fftshift(np.fft.fft(x*w,axis=1),axes=1))**2/np.sum(w**2)).mean(0);bg[j]=10*np.log10(np.maximum(p,1e-30))
  np.save(cp,bg)
 c=sh.cells(b);fl=sh.floors[stem];rs=[]
 for lo,ix in zip(range(2405,2455),sh.sub):
  frac=float(np.all(c[:,ix]<fl+3,axis=1).mean());dp=float(np.percentile(c[:,ix],10));bp=float(np.percentile(bg[:,ix],10));dm=float(np.median(c[:,ix]));bm=float(np.median(bg[:,ix]));rs.append(dict(lo_mhz=lo,quiet_fraction=frac,drone_p10=dp,bg_p10=bp,p10_diff=bp-dp,drone_median=dm,bg_median=bm,median_diff=bm-dm,status='judged' if frac>=.2 else 'not_judged'))
 vals=[abs(r['p10_diff']) for r in rs if r['status']=='judged'];skip=len(vals)<5;r=dict(block=stem,s=s,floor_noise=fl,n_judged=len(vals),skip=skip,max_abs_p10=max(vals,default=None),pass_shape=skip or max(vals,default=0)<=.5,subbands=rs);return r
if __name__=='__main__':
 rs=[]
 for b in json.loads((O/'pilot_plan.json').read_text(encoding='utf-8')):
  for s in [0,2]:
   if not (E/'2.4GHz'/f'{Path(b["block_file"]).stem}__bgA_s{s}.npy').exists():continue
   r=shape(b,s);rs.append(r);(O/'pilot_shape_p10.json').write_text(json.dumps(rs,indent=2),encoding='utf-8');print({k:v for k,v in r.items() if k!='subbands'},flush=True)
