from pathlib import Path
import csv,json,importlib.util,time,sys
import numpy as np
from scipy import signal,fft
P=Path('C:/Users/Administrator/Desktop/数据集处理'); O=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('lab_preproc',P/'work/prep_lines_20261104/lab_preproc.py'); lab=importlib.util.module_from_spec(sp);sp.loader.exec_module(lab)
FS={'2.4GHz':61440000,'5.8GHz':100000000}; FC={'2.4GHz':2430.,'5.8GHz':5800.}; shapes=np.load(O/'shapes.npz'); filters=np.load(P/'work/crop_slice_20261107/filters.npz')
with (P/'work/crop_slice_20261107/notch_table.csv').open(encoding='utf-8-sig') as f:notches=list(csv.DictReader(f))
def cells(x,band):
 n=512 if band=='2.4GHz' else 1024; cs=FS[band]//1000; nf=cs//n; w=np.hanning(n).astype(np.float32)
 a=x[:len(x)//cs*cs].reshape(-1,cs)[:,:nf*n].reshape(-1,nf,n)
 p=abs(fft.fftshift(fft.fft(a*w,axis=-1),axes=-1))**2/np.sum(w*w)
 return 10*np.log10(np.maximum(p.mean(1),1e-30))
def generate(band,kind,index,correction=None):
 N=FS[band]//50; pad=20000; L=N+2*pad
 seed=2026110900+(0 if band=='2.4GHz' else 100000)+{'a':0,'b':1000,'A':2000,'B':3000}[kind]+index
 rng=np.random.default_rng(seed); x=((rng.standard_normal(L)+1j*rng.standard_normal(L))/np.sqrt(2)).astype(np.complex64)
 if kind in ['A','B']:
  key=band+'_'+kind; f=shapes[key+'_freq']; target=shapes[key]
  if correction is not None:
   target=target+correction
   passband=abs(f-FC[band])<=(25 if band=='2.4GHz' else 45)
   # Coordinator-approved correction: constant edge extension before terminal FIR.
   target=np.interp(f,f[passband],target[passband])
  s=np.interp(np.fft.fftfreq(L,1/FS[band])/1e6+FC[band],f,target)
  x=fft.ifft(fft.fft(x)*np.sqrt(10**(s/10))).astype(np.complex64)
 if kind!='a':
  x=signal.fftconvolve(x,filters[band],mode='same')
  for r in notches:
   if r['band']==band:x=lab.notch(x,FS[band],(float(r['freq_mhz'])-FC[band])*1e6,float(r['half_width_khz'])*500.)
 return x[pad:pad+N].astype(np.complex64)
if __name__=='__main__':
 out=[]
 for band in FS:
  for receiver in ['A','B']:
   key=band+'_'+receiver; t=time.time(); cc=[cells(generate(band,receiver,i),band) for i in range(10)]
   # Same block-wise p10 followed by cross-block median; each independent segment is a block.
   re=np.median([np.percentile(c,10,axis=0) for c in cc],axis=0)
   f=shapes[key+'_freq']; good=np.ones(len(f),bool)
   for r in notches:
    if r['band']==band:good &= abs(f-float(r['freq_mhz']))>.3
   diff=re-shapes[key]; passband=good&(abs(f-FC[band])<=(25 if band=='2.4GHz' else 45))
   out.append(dict(key=key,max_all_db=float(abs(diff[good]).max()),max_passband_db=float(abs(diff[passband]).max()),median_passband_db=float(np.median(diff[passband])),worst_freq_mhz=float(f[np.where(good)[0][np.argmax(abs(diff[good]))]])))
   print(out[-1],time.time()-t,flush=True)
 (O/'pilot_validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
