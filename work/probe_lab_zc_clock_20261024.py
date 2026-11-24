"""Test candidate clocks against published DroneID synchronization structures.

This is signature matching, not packet decoding. A weak/absent match is not proof
that a protocol is absent. No acquisition metadata are changed.
"""
from pathlib import Path
from fractions import Fraction
import json
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly,fftconvolve,find_peaks

W=Path(__file__).parent;O=W/'parameter_investigation_20261024';TARGET=15360000
def template(root,variant):
    n=np.arange(601);zc=np.exp(-1j*np.pi*root*n*(n+1)/601)
    f=np.zeros(1024,dtype=np.complex128)
    if variant=='frequency_zc':f[212:813]=zc;f[512]=0
    else:
        a=np.fft.fft(zc,n=1024);a[301:724]=0;a[0]=0
        f=np.fft.fftshift(a)
    t=np.fft.ifft(np.fft.ifftshift(f));return (t/np.linalg.norm(t)).astype(np.complex64)
TEMPLATES={(root,var,conj):np.conj(t) if conj else t for root in [600,147] for var in ['frequency_zc','time_zc_truncated'] for t in [template(root,var)] for conj in [False,True]}
def score(z):
    power=np.r_[0.,np.cumsum(np.abs(z).astype(np.float64)**2)];en=power[1024:]-power[:-1024]
    denom=np.sqrt(np.maximum(en,1e-30));out=[]
    for (root,var,conj),t in TEMPLATES.items():
        c=np.abs(fftconvolve(z,np.conj(t[::-1]),mode='valid'))/denom
        c[en<np.median(en)*.1]=0
        peaks,_=find_peaks(c,distance=500);top=sorted(peaks,key=lambda k:c[k],reverse=True)[:8]
        out.append(dict(root=root,variant=var,conjugated=conj,maximum=float(c.max()),peaks=[dict(offset=int(k),score=float(c[k])) for k in top]))
    return out

# Sanity check the signature matcher with a known clock and both signatures.
z=np.zeros(10000,dtype=np.complex64);z[1000:2024]=TEMPLATES[(600,'frequency_zc',False)];z[3192:4216]=TEMPLATES[(147,'frequency_zc',False)]
rng=np.random.default_rng(20261024);z+=.001*(rng.normal(size=z.size)+1j*rng.normal(size=z.size))
up=resample_poly(z,8,1);back=resample_poly(up,1,8)
sanity=score(back);assert min(max(x['maximum'] for x in sanity if x['root']==root) for root in [600,147])>.98
(O/'zc_synthetic_sanity.json').write_text(json.dumps(sanity,indent=2),encoding='utf8')

rows=[json.loads(s) for s in (O/'measurements.jsonl').read_text(encoding='utf8').splitlines()]
selected=[r for r in rows if r['folder'] in ['007_2.4GHZ_10M_combination-signal','009_2.4GHZ_10M-signal','20260709-022336_mavic3classic']]
out=O/'zc_clock_candidates.jsonl';done={json.loads(s)['path'] for s in out.read_text(encoding='utf8').splitlines()} if out.exists() else set()
for r in selected:
    if r['path'] in done:continue
    p=Path(r['path'])
    if r['kind']=='data':x=loadmat(p,variable_names=['data'])['data'].ravel()
    else:
        v=np.fromfile(p,dtype='<i2',count=2*6144000).reshape(-1,2);x=v[:,0].astype(np.float32)+1j*v[:,1].astype(np.float32)
    m=next(m for m in r['metrics'] if m['profile']=='active' and m['drop_db']==10)
    center=(m['low_cycles_per_sample']+m['high_cycles_per_sample'])/2
    n=np.arange(x.size);mixed=(x*np.exp(-2j*np.pi*center*n)).astype(np.complex64);del x,n
    hypotheses=[]
    for fs in [61440000,100000000,122880000,125000000,150000000,153600000,200000000,245760000,250000000]:
        ratio=Fraction(TARGET,fs);z=resample_poly(mixed,ratio.numerator,ratio.denominator)
        scores=score(z);hypotheses.append(dict(assumed_fs_hz=fs,signatures=scores))
        print(r['folder'],fs/1e6,' '.join(f'{root}:{max(t["maximum"] for t in scores if t["root"]==root):.3f}' for root in [600,147]),flush=True)
    result=dict(path=r['path'],kind=r['kind'],folder=r['folder'],normalized_frequency_shift=center,hypotheses=hypotheses)
    with out.open('a',encoding='utf8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
print('ZC probe complete',flush=True)
