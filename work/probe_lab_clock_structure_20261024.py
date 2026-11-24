"""Probe normalized autocorrelation lags without assuming a sample clock."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.io import loadmat
from scipy.fft import fft,ifft,next_fast_len
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(__file__).parent;O=W/'parameter_investigation_20261024';B=262144;MAXLAG=20000
rows=[json.loads(s) for s in (O/'measurements.jsonl').read_text(encoding='utf8').splitlines()]
selected=[r for r in rows if r['nominal_bw_hz']==10000000 and 'error' not in r]
results=[]
for i,r in enumerate(selected):
    p=Path(r['path']);tag=hashlib.sha256(r['path'].encode()).hexdigest()[:12]
    if r['kind']=='data':x=loadmat(p,variable_names=['data'])['data'].ravel()
    else:x=np.memmap(p,mode='r',dtype='<i2').reshape(-1,2)
    curves=[]
    for start in np.linspace(r['start'],len(x)-B,16,dtype=int):
        z=x[start:start+B] if r['kind']=='data' else x[start:start+B,0].astype(np.float32)+1j*x[start:start+B,1].astype(np.float32)
        z=z-np.mean(z);f=fft(z,n=next_fast_len(2*B));c=ifft(f*np.conj(f))[:MAXLAG+1]
        curves.append(np.abs(c)/max(c[0].real,1e-30)*B/(B-np.arange(MAXLAG+1)))
    a=np.mean(curves,axis=0);peaks,_=find_peaks(a[100:],distance=30,prominence=.0005);peaks+=100
    peaks=sorted(peaks,key=lambda k:a[k],reverse=True)[:15]
    result=dict(path=r['path'],kind=r['kind'],folder=r['folder'],peaks=[dict(lag=int(k),correlation=float(a[k])) for k in peaks],curve=tag+'_acf.npz')
    np.savez_compressed(O/result['curve'],acf=a.astype(np.float32));results.append(result)
    print(i+1,len(selected),r['folder'],[(k,round(float(a[k]),4)) for k in peaks[:5]],flush=True)
(O/'clock_structure.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
fig,axs=plt.subplots(2,1,figsize=(12,7))
for r in results:
    a=np.load(O/r['curve'])['acf'];ax=axs[0 if r['kind']=='data' else 1];ax.plot(np.arange(len(a))[100:],a[100:],label=r['folder'],lw=.8)
for ax in axs:ax.set_ylim(0,.12);ax.set_xlim(100,20000);ax.grid(alpha=.25);ax.legend(fontsize=6);ax.set_ylabel('Mean normalized |ACF|')
axs[0].set_title('data: no sampling rate assumed');axs[1].set_title('raw controls: known 250 MS/s');axs[1].set_xlabel('Lag in original samples');fig.tight_layout();fig.savefig(O/'clock_structure.png',dpi=120)
