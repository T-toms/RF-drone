"""Conditional sample-rate estimates from labelled bandwidth, with raw controls.

Outputs are hypotheses. Nominal bandwidth is NOT assumed to equal occupied
bandwidth without validation. Original IQ files are always read-only.
"""
from pathlib import Path
import csv,json,re,argparse,hashlib,struct,collections
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d,binary_closing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W=Path(__file__).parent; A=W/'audit_drone_20261023'; O=W/'parameter_investigation_20261024';O.mkdir(exist_ok=True)
ROOT=Path('G:/drone');NFFT=8192;NFRAMES=256
def js(name,v): (O/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def loadjson(name):return json.loads((A/name).read_text(encoding='utf8'))
def bw(folder):return float(re.search(r'_(\d+)(?:M)?(?:_combination)?-signal$',folder).group(1))*1e6
def measure(p,kind,start=0):
    if kind=='data':x=loadmat(p,variable_names=['data'])['data'].reshape(-1);n=x.size
    else:x=np.memmap(p,mode='r',dtype='<i2').reshape(-1,2);n=x.shape[0]
    starts=np.unique(np.linspace(start,n-NFFT,min(NFRAMES,(n-start)//NFFT),dtype=np.int64))
    win=np.hanning(NFFT).astype(np.float32);powers=[]
    for i in starts:
        z=x[i:i+NFFT] if kind=='data' else x[i:i+NFFT,0].astype(np.float32)+1j*x[i:i+NFFT,1].astype(np.float32)
        powers.append(np.abs(np.fft.fftshift(np.fft.fft(z*win)))**2)
    ps=np.asarray(powers);energy=ps.sum(axis=1);active=ps[energy>=np.quantile(energy,.75)]
    profiles={'mean':ps.mean(axis=0),'active':active.mean(axis=0)};metrics=[]
    for profile,v in profiles.items():
        smooth=uniform_filter1d(v,size=17,mode='wrap');db=10*np.log10(smooth+1e-24)
        noise=float(np.percentile(db,20));peak=float(np.percentile(db,99))
        for drop in [6,10,15,20]:
            threshold=max(peak-drop,noise+6)
            mask=binary_closing(db>=threshold,structure=np.ones(7),border_value=0)
            edges=np.diff(np.r_[False,mask,False].astype(np.int8));ss=np.flatnonzero(edges==1);ee=np.flatnonzero(edges==-1)
            # Broadest component is a candidate, not an identified target.
            if not len(ss):continue
            j=int(np.argmax(ee-ss));lo,hi=int(ss[j]),int(ee[j]);frac=(hi-lo)/NFFT
            metrics.append(dict(profile=profile,drop_db=drop,threshold_db=threshold,dynamic_db=peak-noise,
                width_fraction_fs=frac,low_cycles_per_sample=lo/NFFT-.5,high_cycles_per_sample=hi/NFFT-.5,
                component_count=len(ss),edge_touching=lo<=8 or hi>=NFFT-8))
    # Frequency bins are kept in normalized units. Overview is for inspection.
    return metrics,dict(mean=10*np.log10(profiles['mean']+1e-24).astype(np.float32),active=10*np.log10(profiles['active']+1e-24).astype(np.float32),
        overview=(10*np.log10(ps.reshape(len(ps),1024,8).mean(axis=2)+1e-24)).astype(np.float32),frame_starts=starts,
        energy_db=10*np.log10(energy+1e-24).astype(np.float32))

def select(all_data=False):
    rows=[r for r in csv.DictReader((A/'data_preprocessing_manifest.csv').open(encoding='utf-8-sig')) if r['analysis_eligible']=='True']
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['folder'],r['session'])].append(r)
    selected=[]
    for (folder,session),rs in sorted(groups.items()):
        if not all_data and not (folder.startswith(('009_','F009_')) or folder in ['002_2.4GHz_10M-signal','002_2.4GHZ_10M_combination-signal','007_2.4GHZ_10M_combination-signal','007_2.4GHz_20M-signal','007_2.4GHz_40M-signal','007_5.8GHz_10M-signal','007_5.8GHz_20M-signal','007_5.8GHz_40M-signal']):continue
        # Same ordinal in each recording, with full numeric data when possible.
        r=min(rs,key=lambda r:(int(r['start_sample'])!=0,abs(int(r['epoch'])*20+int(r['seg'])-5)))
        selected.append(dict(kind='data',path=str(ROOT/r['path']),folder=folder,session=session,nominal_bw_hz=bw(folder),known_fs_hz=None,start=int(r['start_sample'])))
    rawgood={r['path'] for r in csv.DictReader((A/'raw_preprocessing_manifest.csv').open(encoding='utf-8-sig')) if r['analysis_eligible']=='True'}
    seen=set()
    for x in loadjson('raw_json_complete.json'):
        m=x['metadata'];p=str((ROOT/x['path']).with_suffix('.bin'));sid=m['session']['id']
        if p not in rawgood or m['declared']['class']!='drone' or sid in seen:continue
        if m['declared']['model'][0] not in ['mini5pro','matrice4','mavic3classic']:continue
        b=m['declared']['bw'][0]
        if not b:continue
        seen.add(sid)
        selected.append(dict(kind='raw',path=p,folder=sid,session=sid,nominal_bw_hz=b,known_fs_hz=m['capture']['sample_rate_hz'],start=0))
    return selected

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--all-data',action='store_true');args=parser.parse_args()
    selection=select(args.all_data);js('selection.json',selection)
    out=O/'measurements.jsonl';done={json.loads(s)['path'] for s in out.read_text(encoding='utf8').splitlines()} if out.exists() else set()
    for k,r in enumerate(selection):
        if r['path'] in done:continue
        try:
            metrics,spec=measure(Path(r['path']),r['kind'],r['start'])
            tag=hashlib.sha256(r['path'].encode()).hexdigest()[:12];np.savez_compressed(O/(tag+'.npz'),**spec)
            for m in metrics:
                m['conditional_fs_hz']=r['nominal_bw_hz']/m['width_fraction_fs']
                if r['known_fs_hz']:m['measured_bw_hz']=r['known_fs_hz']*m['width_fraction_fs']
            result={**r,'metrics':metrics,'spectra_file':tag+'.npz'}
            with Path(r['path']).open('rb') as f:result['header_hex']=f.read(128).hex() if r['kind']=='data' else None
        except Exception as e:result={**r,'error':repr(e)}
        with out.open('a',encoding='utf8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
        print(f'{k+1}/{len(selection)} {r["kind"]} {r["folder"]}',flush=True)
    results=[json.loads(s) for s in out.read_text(encoding='utf8').splitlines()];plot(results)
    js('measurement_status.json',dict(selected=len(selection),completed=len(results),errors=[r for r in results if 'error' in r],interpretation='conditional estimates; no fs/fc metadata filled'))

def plot(results):
    rows=[r for r in results if 'error' not in r and r['kind']=='data' and r['folder'].startswith(('009_','F009_'))]
    fig,axes=plt.subplots(len(rows),1,figsize=(12,2.0*len(rows)),squeeze=False)
    for ax,r in zip(axes[:,0],rows):
        s=np.load(O/r['spectra_file']);f=np.arange(NFFT)/NFFT-.5
        ax.plot(f,s['mean']-np.max(s['mean']),label='mean',lw=.8);ax.plot(f,s['active']-np.max(s['active']),label='active-quarter',lw=.8)
        ax.set_title(r['folder']+' / '+r['session'],fontsize=9);ax.set_ylim(-60,5);ax.set_xlim(-.5,.5);ax.grid(alpha=.25)
        ax.set_ylabel('Relative dB');ax.legend(fontsize=7,loc='lower left')
    axes[-1,0].set_xlabel('Normalized frequency (cycles/sample)');fig.tight_layout();fig.savefig(O/'avata2_bandwidth_profiles.png',dpi=110);plt.close(fig)

if __name__=='__main__':main()
