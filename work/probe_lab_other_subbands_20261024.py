"""Check additional spectral components; do not assume the strongest is DroneID."""
from pathlib import Path
from fractions import Fraction
import json
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d,binary_closing
from scipy.signal import resample_poly,fftconvolve

O=Path(__file__).parent/'parameter_investigation_20261024'
templates=[]
for var in ['frequency_zc','time_zc_truncated']:
    n=np.arange(601);seq=np.exp(-1j*np.pi*600*n*(n+1)/601)
    f=np.zeros(1024,dtype=np.complex128)
    if var=='frequency_zc':f[212:813]=seq;f[512]=0
    else:
        a=np.fft.fft(seq,n=1024);a[301:724]=0;a[0]=0;f=np.fft.fftshift(a)
    t=np.fft.ifft(np.fft.ifftshift(f));t=(t/np.linalg.norm(t)).astype(np.complex64)
    for conj in [False,True]:templates.append((var,conj,np.conj(t) if conj else t))
rows=[json.loads(s) for s in (O/'measurements.jsonl').read_text(encoding='utf8').splitlines()]
selected=[r for r in rows if r['folder'] in ['007_2.4GHZ_10M_combination-signal','009_2.4GHZ_10M-signal','007_5.8GHz_10M-signal']]
out=O/'other_subband_signatures.jsonl'
done={json.loads(s)['path'] for s in out.read_text(encoding='utf8').splitlines()} if out.exists() else set()
for r in selected:
    if r['path'] in done:continue
    spec=np.load(O/r['spectra_file']);db=10*np.log10(uniform_filter1d(10**(spec['mean']/10),size=17)+1e-30)
    mask=binary_closing(db>max(np.percentile(db,20)+12,np.percentile(db,99)-20),structure=np.ones(7))
    edge=np.diff(np.r_[False,mask,False].astype(np.int8));bands=[]
    for lo,hi in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)):
        if .025<(hi-lo)/8192<.2:bands.append(dict(low=lo/8192-.5,high=hi/8192-.5))
    x=loadmat(r['path'],variable_names=['data'])['data'].ravel();n=np.arange(len(x));results=[]
    for b in bands:
        center=(b['low']+b['high'])/2;mixed=(x*np.exp(-2j*np.pi*center*n)).astype(np.complex64)
        for fs in [61440000,100000000,122880000,125000000,150000000,153600000,200000000,245760000,250000000]:
            ratio=Fraction(15360000,fs);z=resample_poly(mixed,ratio.numerator,ratio.denominator)
            power=np.r_[0.,np.cumsum(np.abs(z).astype(np.float64)**2)];en=power[1024:]-power[:-1024];denom=np.sqrt(np.maximum(en,1e-30));scores=[]
            for var,conj,t in templates:
                c=np.abs(fftconvolve(z,np.conj(t[::-1]),mode='valid'))/denom;c[en<np.median(en)*.1]=0;k=int(np.argmax(c))
                scores.append(dict(variant=var,conjugated=conj,maximum=float(c[k]),offset=k))
            best=max(scores,key=lambda q:q['maximum']);results.append(dict(band=b,center=center,fs_hz=fs,scores=scores))
            print(r['folder'],round(center,5),fs/1e6,round(best['maximum'],3),flush=True)
    with out.open('a',encoding='utf8') as f:f.write(json.dumps(dict(path=r['path'],folder=r['folder'],bands=bands,results=results),ensure_ascii=False)+'\n')
print('Additional subband probe complete',flush=True)
