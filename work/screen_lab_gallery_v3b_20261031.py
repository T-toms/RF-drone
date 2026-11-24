"""Task D gallery: sole Task D reader of original MAT; stop immediately on I/O failure."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import stft
from screen_lab_windows_v3b_20261031 import OUT,CACHE,A,read,dump,SEED
import matplotlib.pyplot as plt

def plan():
    with (OUT/'recording_retention.csv').open(encoding='utf-8') as f: ret=[r for r in csv.DictReader(f) if r['L']=='10' and r['ref']=='recording' and r['no_characteristic']=='False']
    worst=sorted(ret,key=lambda r:(-int(r['n_discard']),r['recording']))[:6]; selected=[r['recording'] for r in worst]
    best=sorted(ret,key=lambda r:(-float(r['retention']),r['recording']))
    selected += [r['recording'] for r in best if r['recording'] not in selected][:2]
    with (OUT/'window_index.csv').open(encoding='utf-8') as f: rows=[r for r in csv.DictReader(f) if r['L']=='10']
    rng=np.random.default_rng(SEED); plans=[]
    def sample(pool,n): return [pool[int(i)] for i in rng.choice(len(pool),min(n,len(pool)),replace=False)]
    for rec in selected:
        rr=[r for r in rows if r['recording']==rec]; keep=[r for r in rr if r['discard_recording']=='False']; discard=[r for r in rr if r['discard_recording']=='True']
        chosen=[]
        covered=set()
        for r in sorted(discard,key=lambda r:len(r['reasons_recording'].split(';')),reverse=True):
            reasons=set(r['reasons_recording'].split(';'))
            if reasons-covered and len(chosen)<3:
                chosen.append(r); covered |= reasons
        chosen+=sample([r for r in discard if r not in chosen],3-len(chosen))
        plans.append(dict(recording=rec,n_keep=len(keep),n_discard=len(discard),windows=sample(keep,3)+chosen))
    return plans

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--plan',action='store_true'); args=ap.parse_args(); plans=plan(); dump(OUT/'gallery_selection.json',plans)
    if args.plan:
        print(json.dumps([{k:v for k,v in p.items() if k!='windows'} for p in plans],indent=2)); return
    root=Path('G:/drone')
    if not root.is_dir(): raise OSError('G:/drone unavailable; stop and ask coordinator')
    for p in plans:
        m=read(CACHE/p['recording']/'meta.json'); fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained'); windows=p['windows']; keeps=[r for r in windows if r['discard_recording']=='False']; discards=[r for r in windows if r['discard_recording']=='True']
        for ax,r in zip(axes.flat,keeps+[None]*(3-len(keeps))+discards+[None]*(3-len(discards))):
            if r is None:
                ax.text(.5,.5,'no discarded windows',ha='center'); ax.set_axis_off(); continue
            parts=[]
            for src in json.loads(r['sources']):
                # Read MAT once for this selected window, retaining only its requested sample interval.
                raw=loadmat(root/src['path'],variable_names=['data'])['data'].ravel(); parts.append(raw[src['s']:src['e']].astype(np.complex64)); del raw
            x=np.concatenate(parts); assert len(x)==1228800
            f,t,z=stft(x,fs=122880000,window=np.hanning(1024),nperseg=1024,noverlap=768,nfft=1024,return_onesided=False,boundary=None,padded=False)
            power=10*np.log10(np.maximum(abs(np.fft.fftshift(z,axes=0))**2,1e-20)); freq=np.fft.fftshift(f)/1e6+m['fc_hz']/1e6; times=t*1000+int(r['start_cell']); top=np.percentile(power,99.5)
            im=ax.imshow(power,origin='lower',aspect='auto',extent=(times[0],times[-1],freq[0],freq[-1]),vmin=top-45,vmax=top,cmap='viridis',interpolation='nearest'); fig.colorbar(im,ax=ax,label='STFT power (dB)')
            from matplotlib.patches import Rectangle
            for band in read(OUT/'environment_bands.json')[m['band']]:
                ax.add_patch(Rectangle((times[0],band['lo_MHz']),times[-1]-times[0],band['hi_MHz']-band['lo_MHz'],fill=False,edgecolor='cyan',linestyle='--'))
            reasons=r['reasons_recording'].split(';') if r['reasons_recording'] else []
            import textwrap
            ax.set(title='\n'.join(textwrap.wrap(('discard' if r['discard_recording']=='True' else 'keep')+' | '+(', '.join(reasons) or 'no trigger'),55)),xlabel='Time in recording (ms)',ylabel='Frequency (MHz)')
        fig.suptitle(p['recording']+' | 1024 Hann, hop 256; seed 20261031'); fig.savefig(OUT/('gallery_'+p['recording']+'.png'),dpi=110); plt.close(fig); print(p['recording'],flush=True)
if __name__=='__main__': main()
