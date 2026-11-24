"""Independent numeric evidence for the smoke gate and final pipeline."""
import importlib.util,sys,itertools
from pathlib import Path
spec=importlib.util.spec_from_file_location('crop',Path(__file__).with_name('crop.py'));c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
from scipy import signal
import numpy as np
O=c.O
def welch(x,fs,n=131072):
 f,p=signal.welch(x,fs=fs,nperseg=min(n,len(x)),noverlap=0,return_onesided=False);return np.fft.fftshift(f),np.fft.fftshift(p)
def filter_tests():
 results=[]
 for fs,fc,b in [(122880000,2400.,'2.4GHz'),(100000000,2440.,'2.4GHz'),(100000000,2450.,'2.4GHz'),(122880000,5800.,'5.8GHz'),(100000000,5800.,'5.8GHz')]:
  for off in [-c.WIDTH[b]+1,-5.,5.,c.WIDTH[b]-1,c.WIDTH[b]+3]:
   n=round(fs*.006);freq=(c.FC[b]+off-fc)*1e6;x=np.exp(2j*np.pi*freq*np.arange(n)/fs).astype(np.complex64);z=c.chain(x,fs,fc,b);z=z[round(c.FS[b]*.001):-round(c.FS[b]*.001)];amp=float(20*np.log10(np.sqrt(np.mean(abs(z)**2))+1e-30));df=float(abs(np.angle(np.vdot(z[:-1],z[1:]))*c.FS[b]/(2*np.pi)-off*1e6));stop=off>c.WIDTH[b];ok=amp<=-60 if stop else abs(amp)<=.1 and df<1000;results.append(dict(fs_in=fs,fc_in=fc,band=b,offset_mhz=off,amplitude_db=amp,frequency_error_hz=df,status='pass' if ok else 'fail'))
 return dict(status='pass' if all(x['status']=='pass' for x in results) else 'fail',tests=results)
def recompute_and_notch(bs,full=False):
 rng=np.random.default_rng(20261107);recalc=[];tests=[]
 # Every smoke source, or 3 randomly selected blocks per full source category.
 groups={}
 for b in bs:
  key=(b['dataset'],b['band'],b['label'],b['fc_in']);groups.setdefault(key,[]).append(b)
 selected=[]
 for key,bb in groups.items():
  for j in rng.choice(len(bb),min(3,len(bb)),replace=False):selected.extend([bb[j]] if full else [bb[j]]*3)
 for b in selected:
  fo=c.FS[b['band']];fs=b['fs_in'];up,down=c.ratio(fs,b['band']);trim=round(fo*.0002);z=np.load(b['block_file'],mmap_mode='r');ns=round(fo*.01);start=int(rng.integers(0,max(1,len(z)-ns)));outlo=start+trim;outend=outlo+ns;a=max(0,(outlo*down//up-round(fs*.003))//down*down);e=min(b['n_in'],int(np.ceil(outend*down/up))+round(fs*.003));x=c.Source(b).get(a,e)
  y=c.chain(x,fs,b['fc_in'],b['band'],a);before=c.chain(x,fs,b['fc_in'],b['band'],a,False);s=outlo-a*up//down;y=y[s:s+ns];before=before[s:s+ns];stored=np.asarray(z[start:start+ns]);err=float(np.max(abs(y-stored))/max(float(np.max(abs(stored))),1e-30));recalc.append(dict(block=b['block_file'],start=start,max_relative_error=err,status='pass' if err<1e-4 else 'fail'))
  f,ps=welch(y,fo);_,pb=welch(before,fo);f=f/1e6+c.FC[b['band']]
  for r in c.NOTCHES:
   if r['band']!=b['band']:continue
   df=abs(f-r['freq_mhz']);k=int(df.argmin());local=(df<=1)&(df>=r['half_width_khz']/1000*2);res=float(10*np.log10(max(ps[k],1e-30)/max(float(np.median(ps[local])),1e-30)));outside=(df>=.2)&(df<=2)
   for nr in c.NOTCHES:
    if nr['band']==b['band']:outside &= abs(f-nr['freq_mhz'])>=.2
   delta=abs(10*np.log10(np.maximum(ps[outside],1e-30)/np.maximum(pb[outside],1e-30)));d=float(np.max(delta));tests.append(dict(block=b['block_file'],freq_mhz=r['freq_mhz'],residual_db=res,max_outside_delta_db=d,status='pass' if res<=3 and d<=.1 else 'fail'))
 return dict(status='pass' if all(r['status']=='pass' for r in recalc) else 'fail',samples=recalc),dict(status='pass' if all(r['status']=='pass' for r in tests) else 'fail',samples=tests)
def floor_shapes(bs,prefix=''):
 curves={};pairs=[];sources={}
 for b in bs:
  key=f"{b['dataset']}_{b['label']}_{b['fc_in']}_{b['band']}";fo=c.FS[b['band']];z=np.load(b['block_file'],mmap_mode='r');cs=round(fo*.001);nc=len(z)//cs;powers=np.array([np.mean(abs(z[j*cs:(j+1)*cs])**2) for j in range(nc)]);sources.setdefault((b['band'],key),[]).extend((float(power),b['block_file'],j) for j,power in enumerate(powers))
 for (band,key),records in sources.items():
  records.sort();selected=records[:max(1,int(np.ceil(len(records)*.1)))];ps=None;current=None
  for _,path,j in selected:
   if current!=path:z=np.load(path,mmap_mode='r');current=path
   fo=c.FS[band];cs=round(fo*.001);f,p=welch(z[j*cs:(j+1)*cs],fo,8192);ps=p if ps is None else ps+p
  curves[band,key]=[(ps,len(selected),f/1e6+c.FC[band])]
  print('FLOOR_SOURCE',key,len(selected),flush=True)
 for band in c.FS:
  fig,ax=c.plt.subplots(figsize=(12,5));norm=[]
  for (b,key),data in curves.items():
   if b!=band:continue
   f=data[0][2];p=sum(v[0] for v in data)/sum(v[1] for v in data);mask=abs(f-c.FC[band])<=c.WIDTH[band]
   for r in c.NOTCHES:
    if r['band']==band:mask &= abs(f-r['freq_mhz'])>max(r['half_width_khz']/1000,.03)
   db=10*np.log10(np.maximum(p,1e-30));db-=np.median(db[mask]);ax.plot(f[mask],db[mask],label=key);norm.append((key,f[mask],db[mask]))
  for a,b in itertools.combinations(norm,2):
   d=abs(a[2]-b[2]);i=int(d.argmax());pairs.append(dict(band=band,source_a=a[0],source_b=b[0],max_difference_db=float(d[i]),frequency_mhz=float(a[1][i])))
  ax.set(xlabel='Absolute frequency (MHz)',ylabel='Normalized floor PSD (dB)',title='Lowest-power 10% of 1 ms cells');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(O/(prefix+'floor_shape_'+band+'.png'),dpi=120);c.plt.close(fig)
 return dict(status='pass' if all(p['max_difference_db']<=1 for p in pairs) else 'fail',pairs=pairs,note='Differences above 1 dB reported without correction; globally lowest-power 10% of 1ms cells per source.')
def main():
 mode=sys.argv[1];bs=c.read(O/'plan.json');bs=c.smoke_select(bs) if mode=='smoke' else bs;result=dict(v3_contiguity=dict(status='pass' if all(x['contiguous'] for x in c.read(O/'contiguity.json')) else 'fail',cases=c.read(O/'contiguity.json')),filter_chain=c.read(O/'filter_tests.json') if (O/'filter_tests.json').exists() else filter_tests());result['recompute'],result['notch']=recompute_and_notch(bs,mode=='full');result['floor_shape']=floor_shapes(bs,'smoke_' if mode=='smoke' else '');c.dump(O/(mode+'_verification.json'),result);print({k:v['status'] for k,v in result.items()},flush=True)
if __name__=='__main__':main()
