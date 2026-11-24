from pathlib import Path
import sys,json,csv,time
sys.dont_write_bytecode=True
import numpy as np
import torch
O=Path(__file__).resolve().parent;P=O.parents[1];E=Path('E:/drone_lab_bgA_20261109');torch.set_num_threads(4)
F=61440000;freq=np.fft.fftshift(np.fft.fftfreq(1024,1/F))/1e6+2430;sel=(freq>=2405)&(freq<=2455);freq=freq[sel];win=torch.hann_window(1024,device='cuda')
def stft(path):
 z=np.load(path,mmap_mode='r');nf=1+(len(z)-1024)//512;out=np.empty((sel.sum(),nf),np.float32)
 for j in range(0,nf,5000):
  end=min(nf,j+5000);x=torch.tensor(np.array(z[j*512:(end-1)*512+1024]),device='cuda');p=torch.fft.fftshift(torch.stft(x,1024,512,window=win,center=False,onesided=False,return_complex=True).abs().square(),dim=0);out[:,j:end]=p[torch.tensor(sel,device='cuda')].cpu().numpy()
 return out

def delta(d,b,threshold,mask):
 hot=(d[mask]>threshold);quiet=(d[mask]<3);bp=b[mask];nh=int(hot.sum());nq=int(quiet.sum());return (float(10*np.log10(bp[hot].mean(dtype=np.float64)/bp[quiet].mean(dtype=np.float64))) if nh and nq else None),nh,nq

def shape(b,path):
 stem=Path(b['block_file']).stem;c=np.load(P/'work/crop_slice_20261107/cells'/stem/'mean.npy');f=2430+(np.arange(512)-256)*.12;mask=(f>=2405)&(f<=2455)
 for r in csv.DictReader((P/'work/crop_slice_20261107/notch_table.csv').open(encoding='utf-8-sig')):
  if r['band']=='2.4GHz':mask &=abs(f-float(r['freq_mhz']))>.3
 features=json.loads((P/'work/crop_slice_20261107/recording_features.json').read_text(encoding='utf-8'));floor=next(v['floor_db'] for v in features if v['group_id']==b['group_id']);quiet=np.all(c[:,mask]<floor+3,axis=1);n=int(quiet.sum())
 if n<50:return dict(quiet_cells=n,shape_max_db=None,shape_pass=True,shape_status='skip_lt50')
 z=np.load(path,mmap_mode='r');cs=61440;nf=cs//512;win=np.hanning(512).astype(np.float32);avg=np.zeros(512,np.float64)
 for j in range(len(z)//cs):
  x=np.asarray(z[j*cs:(j+1)*cs]);p=abs(np.fft.fftshift(np.fft.fft(x[:nf*512].reshape(nf,512)*win,axis=1),axes=1))**2/np.sum(win**2);avg+=p.mean(0)
 avg/=len(z)//cs;dr=(10**(c[quiet].astype(np.float64)/10)).mean(0);diff=[]
 for lo in range(2405,2455):
  ix=mask&(f>=lo)&(f<lo+1)
  if ix.any():diff.append(float(10*np.log10(avg[ix].mean()/dr[ix].mean())))
 return dict(quiet_cells=n,shape_max_db=max(abs(np.array(diff))),shape_pass=bool(max(abs(np.array(diff)))<=.5),shape_status='tested',shape_diffs=diff)

def main():
 plan=json.loads((O/'pilot_plan.json').read_text(encoding='utf-8'));rs=[];sub=[]
 for b in plan:
  dp=stft(b['block_file']);dp=10*np.log10(np.maximum(dp,1e-30));dp-=np.median(dp,axis=1)[:,None]
  for s in [0,2]:
   path=E/'2.4GHz'/f'{Path(b["block_file"]).stem}__bgA_s{s}.npy'
   while not path.exists():time.sleep(3)
   bp=stft(path);bp/=np.maximum(np.median(bp,axis=1)[:,None],1e-30)
   allmask=np.ones(len(freq),bool);d20,n20,nq=delta(dp,bp,20,allmask);d30,n30,_=delta(dp,bp,30,allmask);values=[]
   for lo in range(2405,2455):
    val,nh,nl=delta(dp,bp,20,(freq>=lo)&(freq<lo+1));sub.append(dict(block=Path(b['block_file']).stem,s=s,lo_mhz=lo,delta20=val,n_hot=nh,n_quiet=nl));values.append(val)
   vals=[v for v in values if v is not None];ss=shape(b,path);r=dict(block=Path(b['block_file']).stem,s=s,delta20=d20,delta30=d30,n_hot20=n20,n_hot30=n30,n_quiet=nq,subband_max=max(vals,default=None),**ss);r['image_pass']=bool(d20 is not None and d20<=.3 and max(vals,default=999)<=1);r['pass']=r['image_pass'] and r['shape_pass'];rs.append(r);(O/'pilot_metrics.json').write_text(json.dumps(rs,ensure_ascii=False,indent=2),encoding='utf-8');(O/'pilot_subbands.json').write_text(json.dumps(sub,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(r,ensure_ascii=False),flush=True);del bp
  del dp
 print('PILOT_EVALUATION_DONE',flush=True)
if __name__=='__main__':main()
