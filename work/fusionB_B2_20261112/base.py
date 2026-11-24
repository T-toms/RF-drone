import sys,os
sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import ast,csv,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
B1=Path('E:/fusionB_features_20261110');CACHE=Path('E:/fusionB_B2_20261112')
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def writecsv(p,rs):
 if not rs:return
 with Path(p).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
def hashes():
 result={}
 for d in [ROOT/'work'/n for n in ['crop_slice_20261107','train_stage1_L20_20261108','bgA_20261109','side_L_wifimix_20261110','fusionB_20261110','review_20261109']]+[B1]:
  for p in sorted(d.rglob('*')):
   if p.is_file():
    h=hashlib.sha256()
    with p.open('rb') as f:
     for z in iter(lambda:f.read(8*1024*1024),b''):h.update(z)
    result[str(p)]=h.hexdigest()
 return result
def iq(r):
 a=np.load(r['block_file'],mmap_mode='r');lo=int(r['start']);z=np.array(a[lo:lo+int(r['n_samples'])]);assert len(z)==int(r['n_samples']);return z
NOTCH=rows(ROOT/'work/crop_slice_20261107/notch_table.csv')

def config(band):
    fs,fc,lo,hi=(61.44e6,2430,2405,2455) if band=='2.4GHz' else (100e6,5800,5755,5845)
    freq=fc+(np.arange(1024)-512)*fs/1024/1e6
    mask=(freq>=lo)&(freq<=hi)
    for r in NOTCH:
        if r['band']==band:mask &= abs(freq-float(r['freq_mhz']))>0.3
    return fs,int(round(fs*.0001)),mask

def feature(z,band):
    # Production: bounded PSD batches; all arithmetic complex128/float64.
    z=np.asarray(z,dtype=np.complex128);fs,L,mask=config(band);n=len(z)
    with np.errstate(all='ignore'):
        m=z.mean();raw=np.abs(z)**2;w=z-m;p0=np.mean(np.abs(w)**2);x=w/np.sqrt(p0)
        a=np.abs(x);p=a*a;ma=a.mean();v=(a-ma)**2;s2=np.mean(x*x)
        ps=np.zeros(1024);win=np.hanning(1024);frames=x[:n//1024*1024].reshape(-1,1024)
        for k in range(0,len(frames),128):ps+=np.sum(np.abs(np.fft.fft(frames[k:k+128]*win,axis=1))**2,axis=0)
        ps=np.fft.fftshift(ps/len(frames))[mask];db=10*np.log10(ps)
        fp=p[:n//L*L].reshape(-1,L).mean(1);on=fp>np.percentile(fp,10)*10**.6
        strong=(a[1:]>1)&(a[:-1]>1);inst=np.angle(x[1:]*x[:-1].conj())[strong]/(2*np.pi)
        fraw=raw[:n//L*L].reshape(-1,L).mean(1);LL=10*L
        db1=10*np.log10(raw[:n//LL*LL].reshape(-1,LL).mean(1));c42=np.mean(p*p)-abs(s2)**2-2
        re=w.real;im=w.imag
        result=[1/ma,np.log(np.mean(v*v)/np.mean(v)**2),10*np.log10(np.percentile(p,99.9)),np.sign(c42)*np.log1p(abs(c42)),np.log1p(abs(np.mean((x*x)**2)-3*s2*s2)),np.log1p(abs(np.mean(x*x*p)-3*s2)),abs(s2),np.log(np.max(np.abs(np.fft.rfft(a/ma-1)[1:])**2)/n),np.std(inst) if len(inst)>=1000 else np.nan,np.exp(np.mean(np.log(ps)))/np.mean(ps),np.mean(db>np.median(db)+10),on.mean(),np.log1p(np.sum(on[1:]&~on[:-1])),10*np.log10(raw.mean()),10*np.log10(np.percentile(fraw,10)),np.percentile(db1,90)-np.percentile(db1,10),abs(m)/np.sqrt(raw.mean()),10*np.log10(np.mean(re*re)/np.mean(im*im)),np.mean(re*im)/np.sqrt(np.mean(re*re)*np.mean(im*im))]
    result=np.array(result,dtype=np.float64);result[~np.isfinite(result)]=np.nan
    return result
