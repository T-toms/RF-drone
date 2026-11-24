import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import csv,json,time,hashlib,sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path('C:/Users/Administrator/Desktop/数据集处理')
OUT=ROOT/'work/fusionB_20261110'
CACHE=Path('E:/fusionB_features_20261110')
NAMES=['shape_factor','log_kurt_a','papr999_db','c42n','c40n_abs','c41n_abs','c20n_abs','log_gamma_max','sigma_f','spec_flatness','occ_frac','duty','log1p_n_bursts','ctl_abs_power_db','ctl_floor_db','ctl_dyn_range_db','ctl_dc_rel','ctl_iq_power_ratio_db','ctl_iq_corr']
PROTECTED=['crop_slice_20261107','train_stage1_L20_20261108','bgA_20261109','train_stage1_L20_bgA_20261109','side_5p8_controls_20261109']
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def writecsv(p,rr):
    if not rr:return
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        fields=list(dict.fromkeys(k for r in rr for k in r))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
def snapshot():
    result={}
    for d in PROTECTED:
        for p in sorted((ROOT/'work'/d).rglob('*')):
            if p.is_file():
                h=hashlib.sha256()
                with p.open('rb') as f:
                    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
                result[str(p.relative_to(ROOT))]=h.hexdigest()
    return result
def inputs():
    pp=sorted(Path('E:/drone_lab_spec_L20_20261108').glob('*_spec_meta.csv'))
    pp.append(Path('E:/drone_lab_spec_bgA_L20_20261109/bgA_meta.csv'))
    return {p.stem:rows(p) for p in pp}
NOTCH=rows(ROOT/'work/crop_slice_20261107/notch_table.csv')
NOTCH.append(dict(band='2.4GHz',freq_mhz=2432.9412))
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
