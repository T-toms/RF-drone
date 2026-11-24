from pathlib import Path
import sys,os,json,hashlib,csv
sys.dont_write_bytecode=True;os.environ['KMP_DUPLICATE_LIB_OK']='TRUE';sys.path.insert(0,str(Path(__file__).resolve().parent))
import pipeline as c
from chain_copy import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;E=Path('E:/drone_lab_bgA_20261109');rawroot=Path('G:/drone/raw/2026-07-09_urban')
class RawSource:
 def __init__(self,b):self.a=np.memmap(R/b['parts'][0]['path'],dtype='<i2',mode='r').reshape(-1,2)
 def get(self,s,e):return self.a[s:e,0].astype(np.float32)+1j*self.a[s:e,1].astype(np.float32)
def meanps(z):
 nfft=10000;total=np.zeros(nfft);count=0;w=np.hanning(nfft)
 for j in range(0,len(z)-nfft+1,nfft*100):
  a=np.asarray(z[j:j+nfft*100]);a=a[:len(a)//nfft*nfft].reshape(-1,nfft);total+=(abs(np.fft.fftshift(np.fft.fft(a*w,axis=1),axes=1))**2).sum(0);count+=len(a)
 return total/count

def main():
 c.Source=RawSource;hashes=[];plans={};control=[]
 for rec in ['20260709-033809_evo2pro_v3','20260709-033706_evo2pro_v3']:
  plans[rec]=[]
  for p in sorted((rawroot/rec).glob('*.bin')):
   m=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();assert h==m['capture']['sha256'];hashes.append(dict(path=str(p),sha256=h,match=True));b=dict(parts=[dict(path=str(p.relative_to(R)),start=0,end=30000000)],fs_in=250000000,fc_in=5800.,band='5.8GHz',n_in=30000000);dest=E/'raw'/rec/(p.stem+'.npy');c.process(b,dest);plans[rec].append(str(dest))
 (O/'raw_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
 bs=json.loads((P/'work/crop_slice_20261107/plan.json').read_text(encoding='utf-8'));bs=[b for b in bs if '033809' in b['group_id']]
 pairs=[]
 for b,raw in zip(bs,plans['20260709-033809_evo2pro_v3']):
  x=np.load(raw,mmap_mode='r');y=np.load(b['block_file'],mmap_mode='r');template=np.asarray(y[100000:200000]);corr=signal.correlate(x,template,mode='valid',method='fft');lag=int(abs(corr).argmax())-100000;lo=max(0,-lag);hi=min(len(y),len(x)-lag);xx=x[lo+lag:hi+lag];yy=y[lo:hi];px=meanps(xx);py=meanps(yy);f=5800+np.fft.fftshift(np.fft.fftfreq(10000,1/1e8))/1e6;diff=[]
  for mhz in range(5755,5845):
   mask=(f>=mhz)&(f<mhz+1)&(abs(f-5800)>.3);diff.append(float(10*np.log10(px[mask].mean()/py[mask].mean())))
  r=dict(raw=raw,original=b['block_file'],lag_samples=lag,n_aligned=len(xx),max_1mhz_abs_db=float(max(abs(np.array(diff)))),mean_1mhz_db=float(np.mean(diff)),spectrum_pass=bool(max(abs(np.array(diff)))<=.5),diffs=diff);control.append(r)
  for j in range(lo,hi-2000000+1,2000000):pairs.append(dict(raw=raw,raw_start=j+lag,original=b['block_file'],original_start=j,n_samples=2000000))
 (O/'raw_control.json').write_text(json.dumps(control,indent=2),encoding='utf-8');(O/'raw_pairs.json').write_text(json.dumps(pairs,indent=2),encoding='utf-8');(O/'raw_plan.json').write_text(json.dumps(plans,indent=2),encoding='utf-8')
 print('RAW_CONTROL',control,flush=True)
 fig,axs=plt.subplots(5,1,figsize=(13,13));mini=[]
 for ax,p in zip(axs,sorted((rawroot/'20260709-024545_mini3pro').glob('*.bin'))):
  a=np.memmap(p,dtype='<i2',mode='r').reshape(-1,2);nfft=4096;nf=len(a)//nfft;z=a[:nf*nfft,0].astype(np.float32)+1j*a[:nf*nfft,1].astype(np.float32);pow=abs(np.fft.fftshift(np.fft.fft(z.reshape(nf,nfft)*np.hanning(nfft),axis=1),axes=1))**2;f=5800+np.fft.fftshift(np.fft.fftfreq(nfft,1/250e6))/1e6;sel=(f>=5755)&(f<=5845);pow=pow[:,sel];db=10*np.log10(np.maximum(pow,1e-30));ax.imshow(db.T,origin='lower',aspect='auto',extent=[0,len(a)/250000,5755,5845],vmin=np.median(db)-5,vmax=np.median(db)+25);ax.set(title=p.name,ylabel='MHz');mini.append(dict(file=p.name,median_db=float(np.median(db)),p99_excess_db=float(np.percentile(db,99)-np.median(db)),fraction_gt20=float((db>np.median(db,axis=0)+20).mean())))
 axs[-1].set_xlabel('Time (ms)');fig.tight_layout();fig.savefig(O/'mini3pro_024545_long.png',dpi=120);plt.close(fig);(O/'mini3pro_024545.json').write_text(json.dumps(mini,indent=2),encoding='utf-8')
if __name__=='__main__':main()

