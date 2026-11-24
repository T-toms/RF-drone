from pathlib import Path
import sys,json,time,csv
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parent))
import pilot_metrics as m
import numpy as np
O=m.O;P=m.P;E=m.E

def shape(b,path):
 c=np.load(P/'work/crop_slice_20261107/cells'/Path(b['block_file']).stem/'mean.npy');f=2430+(np.arange(512)-256)*.12;mask=(f>=2405)&(f<=2455)
 for r in csv.DictReader((P/'work/crop_slice_20261107/notch_table.csv').open(encoding='utf-8-sig')):
  if r['band']=='2.4GHz':mask &=abs(f-float(r['freq_mhz']))>.3
 features=json.loads((P/'work/crop_slice_20261107/recording_features.json').read_text(encoding='utf-8'));floor=next(v['floor_db'] for v in features if v['group_id']==b['group_id']);z=np.load(path,mmap_mode='r');cs=61440;win=np.hanning(512).astype(np.float32);avg=np.zeros(512,np.float64);nc=len(z)//cs
 for j in range(nc):
  x=np.asarray(z[j*cs:(j+1)*cs])[:120*512].reshape(120,512);avg+=(abs(np.fft.fftshift(np.fft.fft(x*win,axis=1),axes=1))**2/np.sum(win**2)).mean(0)
 avg/=nc;rs=[]
 for lo in range(2405,2455):
  ix=mask&(f>=lo)&(f<lo+1);quiet=np.all(c[:,ix]<floor+3,axis=1);n=int(quiet.sum());v=None
  if ix.any() and n>=50:
   dr=(10**(c[quiet][:,ix].astype(np.float64)/10)).mean();v=float(10*np.log10(avg[ix].mean()/dr))
  rs.append(dict(lo_mhz=lo,n_quiet=n,delta_db=v,status='judged' if v is not None else 'not_judged'))
 vals=[abs(r['delta_db']) for r in rs if r['delta_db'] is not None];return dict(shape_max_db=max(vals,default=None),shape_pass=bool(max(vals,default=0)<=.5),shape_judged=len(vals),shape_subbands=rs)

def main():
 mode=sys.argv[1] if len(sys.argv)>1 else 'pilot';shift=int(sys.argv[2]) if len(sys.argv)>2 else None
 plan=json.loads((O/('pilot_plan.json' if mode=='pilot' else 'full_plan.json')).read_text(encoding='utf-8'));dest=O/f'{mode}_revised_final.json';rs=json.loads(dest.read_text(encoding='utf-8')) if dest.exists() else []
 for b in plan:
  shifts=[0,2] if shift is None else [shift];pending=[s for s in shifts if not any(r['block']==Path(b['block_file']).stem and r['s']==s for r in rs)]
  if not pending:continue
  dp=m.stft(b['block_file']);dp=10*np.log10(np.maximum(dp,1e-30));dp-=np.median(dp,axis=1)[:,None]
  for s in pending:
   path=E/'2.4GHz'/f'{Path(b["block_file"]).stem}__bgA_s{s}.npy'
   while not path.exists():time.sleep(3)
   bp=m.stft(path);bp/=np.maximum(np.median(bp,axis=1)[:,None],1e-30);allmask=np.ones(len(m.freq),bool);d20,n20,nq=m.delta(dp,bp,20,allmask);d30,n30,_=m.delta(dp,bp,30,allmask);sub=[]
   for lo in range(2405,2455):
    val,nh,nl=m.delta(dp,bp,20,(m.freq>=lo)&(m.freq<lo+1));sub.append(dict(lo_mhz=lo,delta20=val,n_hot=nh,n_quiet=nl,status='judged' if nh>=2000 else 'not_judged'))
   offsets=np.random.default_rng(20261109).integers(int(.2*dp.shape[1]),int(.8*dp.shape[1]),size=5) if dp.shape[1]*512/61440000<2 else np.random.default_rng(20261109).integers(int(61440000/512),dp.shape[1],size=5);null=[]
   for offset in offsets:
    sh=sl=0.;nh=nl=0
    for j in range(0,len(m.freq),16):
     dd=np.roll(dp[j:j+16],int(offset),axis=1);bb=bp[j:j+16];hot=dd>20;quiet=dd<3;sh+=bb[hot].sum(dtype=np.float64);sl+=bb[quiet].sum(dtype=np.float64);nh+=int(hot.sum());nl+=int(quiet.sum())
    null.append(float(10*np.log10((sh/nh)/(sl/nl))) if nh and nl else None)
   del bp
   vals=[r['delta20'] for r in sub if r['status']=='judged' and r['delta20'] is not None];nv=[v for v in null if v is not None];mu=float(np.mean(nv)) if nv else None;sd=float(np.std(nv,ddof=1)) if len(nv)>1 else None;img=bool(d20 is not None and not(d20>.3 and (mu is None or d20>mu+3*sd)) and max(vals,default=-999)<=1);ss={};r=dict(block=Path(b['block_file']).stem,s=s,delta20=d20,delta30=d30,n_hot20=n20,n_hot30=n30,n_quiet=nq,subband_judged_max=max(vals,default=None),subbands=sub,null_shifts_samples=(offsets*512).tolist(),null_delta=null,null_mean=mu,null_std=sd,image_pass=img,**ss);r['pass']=img;rs.append(r);dest.write_text(json.dumps(rs,ensure_ascii=False,indent=2),encoding='utf-8');print({k:v for k,v in r.items() if k not in ['subbands','shape_subbands']},flush=True)
  del dp
 print('REVISED_DONE',mode,flush=True)
if __name__=='__main__':main()


